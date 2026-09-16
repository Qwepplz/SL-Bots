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
from .training_dataset import RecurrentMiniBatchV1, iter_recurrent_minibatches, sequence_paths
from .training_device import TrainingDeviceReportV1, resolve_training_device


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


def _sequence_paths_for_manifest(
    dataset_manifest: DatasetManifestV1,
    config: Mapping[str, Any],
) -> tuple[Path, ...]:
    configured_path = config.get("dataset_path")
    if configured_path:
        return (Path(str(configured_path)).resolve(),)
    train_paths = sequence_paths(dataset_manifest, "train")
    if train_paths:
        return train_paths
    manifest_path = dataset_manifest.metadata.get("path")
    if manifest_path:
        return (Path(manifest_path).resolve(),)
    raise ValueError("dataset manifest does not declare training sequence paths")


def _validation_paths_for_manifest(dataset_manifest: DatasetManifestV1) -> tuple[Path, ...]:
    return sequence_paths(dataset_manifest, "validation")


def _validate_sequence_purpose(paths: Sequence[Path], expected: DataPurpose) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for BC training") from error
    for path in paths:
        parquet_file = pq.ParquetFile(str(path))
        metadata = parquet_file.schema_arrow.metadata or {}
        value = metadata.get(b"purpose")
        if value is None:
            raise ValueError(f"sequence shard {path} must declare purpose")
        try:
            actual = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"sequence shard {path} purpose must be UTF-8") from error
        if actual != expected.value:
            raise ProductionLineageError(
                f"dataset purpose {actual!r} does not match manifest purpose {expected.value!r}"
            )


def _load_dataset(
    dataset_manifest: DatasetManifestV1,
    config: Mapping[str, Any],
) -> tuple[Path, ...]:
    """解析并校验 shard 路径；数据行仍由流式迭代器按窗口读取。"""
    if not isinstance(dataset_manifest, DatasetManifestV1):
        raise TypeError("dataset_manifest must be a DatasetManifestV1")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    purpose = dataset_manifest.effective_purpose()
    paths = _sequence_paths_for_manifest(dataset_manifest, config)
    _validate_sequence_purpose(paths, purpose)
    return paths


def _reset_state(state: Any, keep: Any, torch: Any) -> Any:
    keep = keep.to(dtype=torch.bool)
    zeros_tactical = torch.zeros_like(state.tactical)
    zeros_action = torch.zeros_like(state.action)
    zeros_tick = torch.zeros_like(state.tick)
    return type(state)(
        tactical=torch.where(keep.unsqueeze(1), state.tactical, zeros_tactical),
        action=torch.where(keep.unsqueeze(1), state.action, zeros_action),
        tick=torch.where(keep, state.tick, zeros_tick),
    )


def _batch_loss(
    model: Any,
    batch: RecurrentMiniBatchV1,
    torch: Any,
    F: Any,
    *,
    training: bool,
) -> tuple[Any, dict[str, Any]]:
    state = model.initial_state(batch.batch_size, device=batch.observations.device)
    totals = {
        "movement": torch.zeros((), device=batch.observations.device),
        "mouse": torch.zeros((), device=batch.observations.device),
        "buttons": torch.zeros((), device=batch.observations.device),
        "weapon": torch.zeros((), device=batch.observations.device),
        "buy": torch.zeros((), device=batch.observations.device),
    }
    total = torch.zeros((), device=batch.observations.device)
    for time_index in range(batch.observations.shape[0]):
        state = _reset_state(state, batch.hidden_state_mask[time_index], torch)
        output = model(
            batch.observations[time_index],
            state,
            delta_time_s=batch.duration_s[time_index],
        )
        mask = batch.loss_masks[time_index]
        target = batch.actions[time_index]
        movement_mask = torch.stack(
            [((mask & (1 << bit)) != 0).to(torch.float32) for bit in range(3)], dim=1
        )
        mouse_mask = torch.stack(
            [((mask & (1 << bit)) != 0).to(torch.float32) for bit in (3, 4)], dim=1
        )
        button_mask = ((mask & (1 << 5)) != 0).to(torch.float32)
        weapon_mask = ((mask & (1 << 6)) != 0).to(torch.float32)
        buy_mask = ((mask & (1 << 7)) != 0).to(torch.float32)
        movement_loss = ((output.movement_mean - target[:, :3]) ** 2 * movement_mask).sum()
        movement_loss = movement_loss / movement_mask.sum().clamp_min(1.0)
        mouse_loss = ((output.mouse_loc - target[:, 3:5]) ** 2 * mouse_mask).sum()
        mouse_loss = mouse_loss / mouse_mask.sum().clamp_min(1.0)
        button_values = target[:, 5].round().to(torch.int64).unsqueeze(1)
        button_target = ((button_values >> torch.arange(32, device=target.device)) & 1).to(torch.float32)
        button_value = F.binary_cross_entropy_with_logits(
            output.button_logits,
            button_target,
            reduction="none",
        ).mean(dim=1)
        button_value = (button_value * button_mask).sum() / button_mask.sum().clamp_min(1.0)
        weapon_target = target[:, 6].round().to(torch.int64).clamp(0, model.weapon_count - 1)
        weapon_value = F.cross_entropy(output.weapon_logits, weapon_target, reduction="none")
        weapon_value = (weapon_value * weapon_mask).sum() / weapon_mask.sum().clamp_min(1.0)
        buy_target = target[:, 7].round().to(torch.int64).clamp(0, model.buy_count - 1)
        buy_value = F.cross_entropy(output.buy_logits, buy_target, reduction="none")
        buy_value = (buy_value * buy_mask).sum() / buy_mask.sum().clamp_min(1.0)
        current = movement_loss + mouse_loss + button_value + weapon_value + buy_value
        total = total + current
        totals["movement"] = totals["movement"] + movement_loss
        totals["mouse"] = totals["mouse"] + mouse_loss
        totals["buttons"] = totals["buttons"] + button_value
        totals["weapon"] = totals["weapon"] + weapon_value
        totals["buy"] = totals["buy"] + buy_value
        state = output.recurrent_state
    divisor = max(1, int(batch.observations.shape[0]))
    return total / divisor, {name: value / divisor for name, value in totals.items()}


def train_bc(
    config: Mapping[str, Any] | str | Path,
    dataset_manifest: DatasetManifestV1,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> TrainingRunManifestV1:
    loaded = load_config(config)
    torch, F = _torch_modules()
    purpose = dataset_manifest.effective_purpose()
    requested_device = str(device or loaded.get("device") or ("cuda" if purpose is DataPurpose.PRODUCTION else "cpu"))
    resolved_device = resolve_training_device(requested_device)
    if str(loaded.get("dtype", "float32")) != "float32":
        raise ValueError("BC training requires float32 tensors")
    if bool(loaded.get("amp", False)):
        raise ValueError("BC training requires AMP to be disabled")
    seed = int(loaded.get("seed", 7))
    sequence_length = int(loaded.get("sequence_length", 128))
    batch_sequences = int(loaded.get("batch_sequences", loaded.get("batch_size", 32)))
    if sequence_length <= 0 or batch_sequences <= 0:
        raise ValueError("sequence_length and batch_sequences must be positive")
    train_paths = _load_dataset(dataset_manifest, loaded)
    validation_paths = _validation_paths_for_manifest(dataset_manifest)
    if validation_paths:
        _validate_sequence_purpose(validation_paths, purpose)
    torch.manual_seed(seed)
    model = MirageActor(
        weapon_count=int(loaded.get("weapon_count", 16)),
        buy_count=int(loaded.get("buy_action_count", 32)),
        max_batch_size=batch_sequences,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(loaded.get("learning_rate", 1e-3))
    )
    epochs = max(1, int(loaded.get("max_steps", loaded.get("max_epochs", 1))))
    if output_dir is None:
        output_dir = loaded.get("output_dir")
    if output_dir is None:
        raise ValueError("output_dir is required either as an argument or in config")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path / "bc-checkpoint.pt"
    checkpoint_interval = max(1, int(loaded.get("checkpoint_interval_updates", 500)))
    loaded["device"] = requested_device
    loaded["sequence_length"] = sequence_length
    loaded["batch_sequences"] = batch_sequences
    loss_history: list[float] = []
    validation_loss_history: list[float] = []
    head_wise_loss = {name: 0.0 for name in ("movement", "mouse", "buttons", "weapon", "buy")}
    first_loss = 0.0
    last_gradient_norm = 0.0
    parameter_updated = False
    updates = 0
    scanner_cursor: dict[str, Any] = {"epoch": 0, "path_index": 0, "sequence_length": sequence_length}

    def _device_report() -> TrainingDeviceReportV1:
        return TrainingDeviceReportV1(
            requested=requested_device,
            device_type=resolved_device.type,
            device_index=resolved_device.index,
            device_name=(
                str(torch.cuda.get_device_name(resolved_device))
                if resolved_device.type == "cuda"
                else "CPU"
            ),
            torch_version=str(torch.__version__),
            hip_version=str(getattr(torch.version, "hip", None) or "unavailable"),
            smoke_loss=first_loss,
            gradient_norm=last_gradient_norm,
            parameter_updated=parameter_updated,
        )

    def _save_checkpoint(epoch_value: int) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": loaded,
                "dataset_manifest": dataset_manifest,
                "loss_history": loss_history,
                "validation_loss_history": validation_loss_history,
                "head_wise_loss": head_wise_loss,
                "epoch": epoch_value,
                "scanner_cursor": scanner_cursor,
                "training_device_report": _device_report(),
            },
            checkpoint,
        )

    model.train()
    for epoch in range(epochs):
        for path_index, batch in enumerate(
            iter_recurrent_minibatches(train_paths, sequence_length, batch_sequences, seed + epoch)
        ):
            batch = batch.to(resolved_device)
            before = tuple(parameter.detach().clone() for parameter in model.parameters())
            optimizer.zero_grad(set_to_none=True)
            total_loss, components = _batch_loss(model, batch, torch, F, training=True)
            if not bool(torch.isfinite(total_loss)):
                raise RuntimeError("BC training produced a non-finite loss")
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(loaded.get("gradient_clip_norm", 1.0)), error_if_nonfinite=True
            )
            optimizer.step()
            parameter_updated = parameter_updated or any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
            updates += 1
            scanner_cursor = {"epoch": epoch, "path_index": path_index + 1, "sequence_length": sequence_length}
            first_loss = first_loss or float(total_loss.detach().cpu())
            last_gradient_norm = float(gradient_norm.detach().cpu())
            loss_history.append(float(total_loss.detach().cpu()))
            head_wise_loss = {
                name: float(value.detach().cpu()) for name, value in components.items()
            }
            validation_interval = max(1, int(loaded.get("validation_interval_updates", 250)))
            if validation_paths and updates % validation_interval == 0:
                model.eval()
                values: list[float] = []
                with torch.no_grad():
                    for validation_batch in iter_recurrent_minibatches(
                        validation_paths,
                        sequence_length,
                        batch_sequences,
                        seed + 100000 + updates,
                    ):
                        value, _ = _batch_loss(
                            model,
                            validation_batch.to(resolved_device),
                            torch,
                            F,
                            training=False,
                        )
                        values.append(float(value.detach().cpu()))
                if values:
                    validation_loss_history.append(sum(values) / len(values))
                model.train()
            if updates > 0 and updates % checkpoint_interval == 0:
                _save_checkpoint(epoch + 1)
    config_bytes = json.dumps(loaded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    _save_checkpoint(epochs)
    device_report = _device_report()
    return TrainingRunManifestV1(
        name=output_path.name,
        purpose=purpose,
        dataset_manifest=dataset_manifest,
        steps=updates,
        loss_history=tuple(loss_history),
        checkpoint_path=str(checkpoint),
        config_sha256=config_sha256,
        metadata={
            "map_name": "de_mirage",
            "training_method": "behavior_cloning",
            "optimizer_state": str(checkpoint),
            "source_sha256": dataset_manifest.source_sha256 or "",
            "device_type": resolved_device.type,
            "device_index": "" if resolved_device.index is None else str(resolved_device.index),
            "device_name": device_report.device_name,
            "torch_version": str(torch.__version__),
            "hip_version": device_report.hip_version,
            "sequence_length": str(sequence_length),
            "batch_sequences": str(batch_sequences),
            "parameter_updated": str(parameter_updated).lower(),
            "validation_loss": "" if not validation_loss_history else str(validation_loss_history[-1]),
        },
    )
