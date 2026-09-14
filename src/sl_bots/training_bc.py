"""带逐维歧义掩码的行为克隆训练入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from typing import Any

from .contracts import DataPurpose, ensure_purpose
from .lineage import DatasetManifestV1, ProductionLineageError
from .model import MirageActor


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required for YAML training configuration") from error
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError("training configuration must be a mapping")
    return value


def load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    return dict(_load_yaml(config))


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for BC training") from error
    return torch, F


@dataclass(frozen=True)
class TrainingRunManifestV1:
    name: str
    purpose: DataPurpose
    dataset_manifest: DatasetManifestV1
    steps: int
    loss_history: tuple[float, ...]
    checkpoint_path: str
    config_sha256: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def assert_exportable(self, target_purpose: DataPurpose | str = DataPurpose.PRODUCTION) -> None:
        target = ensure_purpose(target_purpose)
        self.dataset_manifest.assert_exportable(target_purpose=target)
        if self.purpose != target:
            from .lineage import ProductionLineageError

            raise ProductionLineageError(
                f"training run purpose {self.purpose!r} cannot be exported as {target!r}"
            )


def _column(table: Any, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _load_dataset(dataset_manifest: DatasetManifestV1, config: Mapping[str, Any]) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for BC training") from error
    dataset_path = config.get("dataset_path") or dataset_manifest.metadata.get("path")
    if not dataset_path:
        raise ValueError("dataset_path is required in config or dataset manifest")
    table = pq.read_table(dataset_path)
    metadata = table.schema.metadata or {}
    encoded_purpose = metadata.get(b"purpose")
    if encoded_purpose is None:
        raise ValueError("dataset Parquet metadata must declare purpose")
    try:
        actual_purpose = encoded_purpose.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dataset Parquet purpose metadata must be UTF-8") from error
    expected_purpose = dataset_manifest.effective_purpose().value
    if actual_purpose != expected_purpose:
        raise ProductionLineageError(
            f"dataset purpose {actual_purpose!r} does not match manifest purpose {expected_purpose!r}"
        )
    required = (
        "observation",
        "target_tick",
        "forward",
        "side",
        "up",
        "yaw_delta_deg",
        "pitch_delta_deg",
        "buttons",
        "weapon_select",
        "buy_action",
        "loss_mask",
    )
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(missing)}")
    result = {name: _column(table, name) for name in required}
    if "sequence_id" in table.column_names:
        result["sequence_id"] = _column(table, "sequence_id")
    else:
        result["sequence_id"] = ["sequence-0"] * len(result["observation"])
    if "duration_s" in table.column_names:
        result["duration_s"] = _column(table, "duration_s")
    else:
        result["duration_s"] = [1.0 / 128.0] * len(result["observation"])
    if not result["observation"]:
        raise ValueError("dataset must contain at least one sequence row")
    return result


def _button_targets(values: Sequence[int], torch: Any) -> Any:
    targets = []
    for value in values:
        targets.append([(int(value) >> bit) & 1 for bit in range(32)])
    return torch.tensor(targets, dtype=torch.float32)


def _masked_mean(value: Any, mask: Any, torch: Any) -> Any:
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator


def _detach_recurrent_state(state: Any) -> Any:
    return type(state)(
        tactical=state.tactical.detach(),
        action=state.action.detach(),
        tick=state.tick.detach(),
    )


def _sequence_row_groups(sequence_ids: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, value in enumerate(sequence_ids):
        sequence_id = str(value)
        if not sequence_id:
            raise ValueError("sequence_id values must be non-empty")
        if sequence_id not in groups:
            groups[sequence_id] = []
            order.append(sequence_id)
        groups[sequence_id].append(index)
    return tuple(tuple(groups[sequence_id]) for sequence_id in order)


def train_bc(
    config: Mapping[str, Any] | str | Path,
    dataset_manifest: DatasetManifestV1,
    *,
    output_dir: str | Path | None = None,
) -> TrainingRunManifestV1:
    """执行短 BC 训练并保存 checkpoint；训练血缘继承输入数据 purpose。"""

    loaded = load_config(config)
    torch, F = _torch_modules()
    dataset = _load_dataset(dataset_manifest, loaded)
    seed = int(loaded.get("seed", 7))
    torch.manual_seed(seed)
    model = MirageActor(
        weapon_count=int(loaded.get("weapon_count", 16)),
        buy_count=int(loaded.get("buy_action_count", 32)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(loaded.get("learning_rate", 1e-3)))
    max_steps = max(1, int(loaded.get("max_steps", 1)))
    batch_size = max(1, min(10, int(loaded.get("batch_size", 10))))
    rows = len(dataset["observation"])
    observations = dataset["observation"]
    movement = torch.tensor(
        list(zip(dataset["forward"], dataset["side"], dataset["up"])), dtype=torch.float32
    )
    mouse = torch.tensor(
        list(zip(dataset["yaw_delta_deg"], dataset["pitch_delta_deg"])), dtype=torch.float32
    )
    buttons = _button_targets(dataset["buttons"], torch)
    weapon_count = int(loaded.get("weapon_count", 16))
    buy_count = int(loaded.get("buy_action_count", 32))
    weapons = torch.tensor(dataset["weapon_select"], dtype=torch.long).clamp(0, weapon_count - 1)
    buys = torch.tensor(dataset["buy_action"], dtype=torch.long).clamp(0, buy_count - 1)
    loss_masks = torch.tensor(dataset["loss_mask"], dtype=torch.long)
    sequence_ids = tuple(str(value) for value in dataset["sequence_id"])
    sequence_groups = _sequence_row_groups(sequence_ids)
    loss_history: list[float] = []
    model.train()
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float32)
        sequence_rows = 0
        for sequence_indices in sequence_groups:
            recurrent_state = model.initial_state(1)
            for sequence_position, index in enumerate(sequence_indices, start=1):
                try:
                    duration_s = float(dataset["duration_s"][index])
                except (TypeError, ValueError) as error:
                    raise ValueError("dataset duration_s values must be numeric") from error
                if not math.isfinite(duration_s) or duration_s <= 0.0:
                    duration_s = 1.0 / 128.0
                duration = torch.tensor(
                    [duration_s],
                    dtype=torch.float32,
                    device=next(model.parameters()).device,
                )
                output = model([observations[index]], recurrent_state, delta_time_s=duration)
                mask = loss_masks[index : index + 1]
                movement_target = movement[index : index + 1]
                mouse_target = mouse[index : index + 1]
                button_target = buttons[index : index + 1]
                weapon_target = weapons[index : index + 1]
                buy_target = buys[index : index + 1]
                move_mask = torch.stack(
                    [((mask & (1 << bit)) != 0).float() for bit in range(3)], dim=1
                )
                mouse_mask = torch.stack(
                    [((mask & (1 << bit)) != 0).float() for bit in (3, 4)], dim=1
                )
                button_mask = ((mask & (1 << 5)) != 0).float()
                weapon_mask = ((mask & (1 << 6)) != 0).float()
                buy_mask = ((mask & (1 << 7)) != 0).float()
                move_loss = _masked_mean((output.movement_mean - movement_target) ** 2, move_mask, torch)
                mouse_loss = _masked_mean((output.mouse_loc - mouse_target) ** 2, mouse_mask, torch)
                button_loss = _masked_mean(
                    F.binary_cross_entropy_with_logits(
                        output.button_logits,
                        button_target.to(output.button_logits.device),
                        reduction="none",
                    ).mean(dim=1),
                    button_mask.to(output.button_logits.device),
                    torch,
                )
                weapon_loss = _masked_mean(
                    F.cross_entropy(
                        output.weapon_logits,
                        weapon_target.to(output.weapon_logits.device),
                        reduction="none",
                    ),
                    weapon_mask.to(output.weapon_logits.device),
                    torch,
                )
                buy_loss = _masked_mean(
                    F.cross_entropy(
                        output.buy_logits,
                        buy_target.to(output.buy_logits.device),
                        reduction="none",
                    ),
                    buy_mask.to(output.buy_logits.device),
                    torch,
                )
                total_loss = total_loss + move_loss + mouse_loss + button_loss + weapon_loss + buy_loss
                sequence_rows += 1
                recurrent_state = output.recurrent_state
                if sequence_position % batch_size == 0:
                    recurrent_state = _detach_recurrent_state(recurrent_state)
        total_loss = total_loss / max(1, sequence_rows)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(float(total_loss.detach().cpu()))
    if output_dir is None:
        output_dir = loaded.get("output_dir")
    if output_dir is None:
        raise ValueError("output_dir is required either as an argument or in config")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_bytes = json.dumps(loaded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    checkpoint = output_path / "bc-checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": loaded,
            "dataset_manifest": dataset_manifest,
            "loss_history": loss_history,
        },
        checkpoint,
    )
    purpose = dataset_manifest.effective_purpose()
    return TrainingRunManifestV1(
        name=output_path.name,
        purpose=purpose,
        dataset_manifest=dataset_manifest,
        steps=max_steps,
        loss_history=tuple(loss_history),
        checkpoint_path=str(checkpoint),
        config_sha256=config_sha256,
        metadata={
            "map_name": "de_mirage",
            "training_method": "behavior_cloning",
            "optimizer_state": str(checkpoint),
            "source_sha256": dataset_manifest.source_sha256 or "",
        },
    )
