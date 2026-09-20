"""Streaming test-only Demo training for the hierarchical actor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import struct
from typing import Any

from .action_reconstruction import IN_ATTACK, IN_ATTACK2
from .contracts import (
    DataPurpose,
    HierarchicalPackageV2,
    INTENT_FIELD_NAMES,
    INTENT_TASKS,
    IntentV1,
    TACTICAL_MODES,
    TARGET_SLOT_NONE,
    ensure_purpose,
)
from .export import export_hierarchical_package
from .hierarchical_model import HierarchicalActor
from .lineage import DatasetManifestV1, require_test_only
from .observation_projection import decode_observation
from .training_dataset import RecurrentMiniBatchV1, iter_recurrent_minibatches, sequence_paths
from .training_hierarchical import (
    HierarchicalBatchV1,
    _sequence_target_at,
    build_hierarchical_checkpoint,
    prepare_hierarchical_training,
    train_hierarchical_step,
)
from .training_device import resolve_training_device


_LABELER_VERSION = "hierarchical-demo-future-trajectory-intent-v3-safe-action-abstention"
_SEQUENCE_LENGTH = 128
_BATCH_SEQUENCES = 8
_UPDATES_PER_DEMO: int | None = None
_WINDOWS_PER_SHARD = 2


def _manifest_from_dict(value: Mapping[str, Any]) -> DatasetManifestV1:
    parents = tuple(
        _manifest_from_dict(parent)
        for parent in value.get("parents", ())
        if isinstance(parent, Mapping)
    )
    return DatasetManifestV1(
        name=str(value["name"]),
        purpose=value["purpose"],
        parents=parents,
        artifact_type=str(value.get("artifact_type", "dataset")),
        source_sha256=value.get("source_sha256"),
        parser_version=value.get("parser_version"),
        projection_version=value.get("projection_version"),
        metadata={
            str(key): str(item)
            for key, item in dict(value.get("metadata", {})).items()
        },
    )


def load_test_only_sequence_manifest(path: str | Path) -> DatasetManifestV1:
    """Load the explicit three-Demo pilot manifest without scanning archives."""

    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"test-only dataset manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("test-only dataset manifest must be a mapping")
    manifest = _manifest_from_dict(payload)
    require_test_only(manifest.effective_purpose())
    if manifest.metadata.get("pilot") != "three_demo":
        raise ValueError("hierarchical Demo training requires the three_demo pilot manifest")
    if manifest.metadata.get("demo_count") != "3":
        raise ValueError("hierarchical Demo training requires exactly three Demo sources")
    return manifest


def _json_paths(manifest: DatasetManifestV1, split: str) -> tuple[Path, ...]:
    paths = sequence_paths(manifest, split)
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(missing)
    return paths


def group_sequence_paths_by_demo(
    manifest: DatasetManifestV1,
) -> dict[str, tuple[Path, ...]]:
    """Return canonical train shards grouped by ProDemo major/minor titles."""

    if not isinstance(manifest, DatasetManifestV1):
        raise TypeError("manifest must be a DatasetManifestV1")
    require_test_only(manifest.effective_purpose())
    if manifest.metadata.get("pilot") != "three_demo":
        raise ValueError("only the three_demo pilot is accepted by this training boundary")
    train_paths = _json_paths(manifest, "train")
    if _json_paths(manifest, "validation") or _json_paths(manifest, "test"):
        raise ValueError("the three_demo pilot must not silently use validation or test paths")

    grouped: dict[str, tuple[Path, ...]] = {}
    assigned: set[Path] = set()
    for parent in manifest.parents:
        metadata = parent.metadata
        minor_title = str(metadata.get("minor_title", parent.name))
        sequence_dir_value = metadata.get("sequence_dir")
        if not sequence_dir_value:
            raise ValueError(f"manifest parent {parent.name} is missing sequence_dir")
        sequence_dir = Path(sequence_dir_value).resolve()
        paths = tuple(path for path in train_paths if path.parent.resolve() == sequence_dir)
        expected = int(metadata.get("sequence_count", len(paths)))
        if len(paths) != expected:
            raise ValueError(
                f"manifest parent {minor_title} expected {expected} sequence paths, got {len(paths)}"
            )
        if not paths:
            raise ValueError(f"manifest parent {minor_title} has no sequence paths")
        if any(path in assigned for path in paths):
            raise ValueError(f"manifest sequence paths are duplicated for {minor_title}")
        grouped[minor_title] = tuple(sorted(paths, key=str))
        assigned.update(paths)

    if len(grouped) != 3:
        raise ValueError(f"three_demo pilot must contain three groups, got {len(grouped)}")
    if assigned != set(train_paths):
        raise ValueError("manifest train paths contain an ungrouped sequence shard")
    return grouped


def split_three_demo_train_validation(
    manifest: DatasetManifestV1,
    groups: Mapping[str, tuple[Path, ...]],
    *,
    heldout_demo: str | None = None,
) -> tuple[dict[str, tuple[Path, ...]], str, tuple[Path, ...]]:
    """Reserve one complete Demo for held-out gate evidence.

    A Demo is the unit of generalization here. Splitting individual shards
    would let adjacent windows from one match leak between training and the
    gate, so the default is a deterministic whole-Demo holdout.
    """

    if len(groups) != 3:
        raise ValueError("three_demo pilot must expose exactly three Demo groups")
    requested = heldout_demo or manifest.metadata.get("heldout_demo")
    selected = sorted(groups)[-1] if requested is None else str(requested)
    if selected not in groups:
        raise ValueError(f"held-out Demo is not present in the three_demo manifest: {selected}")
    validation_paths = tuple(groups[selected])
    if not validation_paths:
        raise ValueError(f"held-out Demo has no sequence shards: {selected}")
    train_groups = {
        name: tuple(paths)
        for name, paths in groups.items()
        if name != selected
    }
    if len(train_groups) < 2 or not any(train_groups.values()):
        raise ValueError("three_demo training split must retain at least two non-empty Demo groups")
    return train_groups, selected, validation_paths


def _decode_position_and_facing(payload: bytes) -> tuple[tuple[float, float, float], tuple[float, float]]:
    try:
        observation = decode_observation(payload)
    except (ValueError, TypeError, struct.error):
        return (0.0, 0.0, 0.0), (0.0, 0.0)
    position = tuple(float(value) / 4096.0 for value in observation.self_position)
    facing = (float(observation.self_yaw_deg) / 180.0, float(observation.self_pitch_deg) / 90.0)
    return position, facing


def _decode_payload(payload: bytes) -> Any | None:
    try:
        return decode_observation(payload)
    except (ValueError, TypeError, struct.error):
        return None


def _payload_from_tensor(value: Any) -> bytes:
    import torch

    return bytes(
        int(item)
        for item in (value.detach().clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.uint8).tolist()
    )


def _intent_confidence(
    *,
    position_valid: bool,
    target_valid: bool,
    has_event: bool,
    has_action: bool,
) -> dict[str, float]:
    value = 0.75 if has_event or has_action else 0.35
    return {
        name: (
            1.0
            if name in {"goal_position", "waypoint_position"} and position_valid
            else 1.0
            if name == "target_slot" and target_valid
            else value
        )
        for name in INTENT_FIELD_NAMES
    }


def _intent_target(
    *,
    current_position: tuple[float, float, float],
    waypoint: tuple[float, float, float],
    goal: tuple[float, float, float],
    facing: tuple[float, float],
    tactical_mode: str,
    task: str,
    target_slot: int,
    desired_range: float,
    aggression: float,
    risk: float,
    priority: float,
    ttl_ticks: int,
    position_valid: bool,
    target_valid: bool,
    has_event: bool,
    has_action: bool,
) -> IntentV1:
    del current_position
    return IntentV1(
        tactical_mode=tactical_mode,
        task=task,
        goal_position=goal,
        waypoint_position=waypoint,
        facing_yaw_pitch=facing,
        desired_range=desired_range,
        target_slot=target_slot,
        aggression=max(0.0, min(1.0, aggression)),
        risk=max(0.0, min(1.0, risk)),
        priority=max(0.0, min(1.0, priority)),
        ttl_ticks=max(0, int(ttl_ticks)),
        confidence=_intent_confidence(
            position_valid=position_valid,
            target_valid=target_valid,
            has_event=has_event,
            has_action=has_action,
        ),
        valid_mask={
            "goal_position": position_valid,
            "waypoint_position": position_valid,
            "target_slot": target_valid,
        },
    )


def recurrent_to_hierarchical_batch(batch: RecurrentMiniBatchV1) -> HierarchicalBatchV1:
    """Adapt a complete recurrent window into future-derived intent and action targets."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - guarded by training dependencies
        raise RuntimeError("PyTorch is required for hierarchical Demo training") from error

    if not isinstance(batch, RecurrentMiniBatchV1):
        raise TypeError("batch must be a RecurrentMiniBatchV1")
    if batch.observations.ndim != 3 or batch.observations.shape[0] != _SEQUENCE_LENGTH:
        raise ValueError("Demo training requires time-major [128, batch, 256] observations")

    device = batch.observations.device
    observations = batch.observations.transpose(0, 1).contiguous()
    actions = batch.actions.transpose(0, 1).contiguous()
    loss_masks = batch.loss_masks.transpose(0, 1).contiguous()
    durations = batch.duration_s.transpose(0, 1).contiguous()
    batch_size, sequence_length = observations.shape[:2]
    previous_intent = torch.zeros(
        batch.batch_size,
        128,
        dtype=observations.dtype,
        device=device,
    )
    context = batch.context
    position_xy = context.get("position_xy")
    position_valid = context.get("position_valid")
    yaw_values = context.get("yaw_deg")
    utility_values = context.get("utility")
    engagement_distances = context.get("engagement_distance")
    alive_values = context.get("alive")
    decoded: list[list[Any | None]] = [[None] * sequence_length for _ in range(batch_size)]
    positions: list[list[tuple[float, float, float] | None]] = [
        [None] * sequence_length for _ in range(batch_size)
    ]
    facings: list[list[tuple[float, float]]] = [
        [(0.0, 0.0)] * sequence_length for _ in range(batch_size)
    ]
    event_sets: list[list[set[str]]] = [
        [set() for _ in range(sequence_length)] for _ in range(batch_size)
    ]
    target_slots: list[list[int]] = [
        [TARGET_SLOT_NONE] * sequence_length for _ in range(batch_size)
    ]
    target_distances: list[list[float]] = [
        [0.0] * sequence_length for _ in range(batch_size)
    ]
    observation_cpu = observations.detach().cpu()
    for batch_index in range(batch_size):
        for time_index in range(sequence_length):
            projection = _decode_payload(_payload_from_tensor(observation_cpu[batch_index, time_index]))
            decoded[batch_index][time_index] = projection
            context_valid = bool(
                position_valid is not None and position_valid[time_index, batch_index]
            )
            if context_valid and position_xy is not None:
                xy = position_xy[time_index, batch_index]
                z = float(projection.self_position[2]) if projection is not None else 0.0
                positions[batch_index][time_index] = (float(xy[0]), float(xy[1]), z)
            elif projection is not None:
                positions[batch_index][time_index] = tuple(
                    float(value) for value in projection.self_position
                )
            yaw = (
                float(yaw_values[time_index, batch_index])
                if yaw_values is not None
                else float(projection.self_yaw_deg) if projection is not None else 0.0
            )
            pitch = float(projection.self_pitch_deg) if projection is not None else 0.0
            facings[batch_index][time_index] = (yaw, pitch)
            if projection is not None:
                event_sets[batch_index][time_index] = {
                    str(event.category).lower()
                    for event in projection.events
                    if str(event.category).lower() != "unknown" and float(event.age_s) <= 1.0
                }
                for slot, player in enumerate(projection.players):
                    if int(player.relation) >= 0 or not int(player.flags):
                        continue
                    if int(player.flags) & (1 | 2 | 4):
                        target_slots[batch_index][time_index] = slot
                        target_distances[batch_index][time_index] = float(player.distance)
                        break
            if target_slots[batch_index][time_index] == TARGET_SLOT_NONE:
                distance = (
                    float(engagement_distances[time_index, batch_index])
                    if engagement_distances is not None
                    and math.isfinite(float(engagement_distances[time_index, batch_index]))
                    else 0.0
                )
                target_distances[batch_index][time_index] = max(0.0, distance)

    def _future_position(batch_index: int, time_index: int, horizon: int) -> tuple[tuple[float, float, float], bool]:
        target_index = min(sequence_length - 1, time_index + horizon)
        target = positions[batch_index][target_index]
        if target is not None:
            return target, True
        current = positions[batch_index][time_index] or (0.0, 0.0, 0.0)
        x, y, z = current
        for step in range(time_index, target_index):
            forward = float(actions[batch_index, step, 0])
            side = float(actions[batch_index, step, 1])
            up = float(actions[batch_index, step, 2])
            yaw = math.radians(float(facings[batch_index][step][0]))
            duration = max(float(durations[batch_index, step]), 1.0 / 128.0)
            x += (forward * math.cos(yaw) + side * math.sin(yaw)) * 250.0 * duration
            y += (forward * math.sin(yaw) - side * math.cos(yaw)) * 250.0 * duration
            z += up * 250.0 * duration
        return (x, y, z), False

    def _target_for(batch_index: int, time_index: int) -> IntentV1:
        button_value = int(round(float(actions[batch_index, time_index, 5]))) & 0xFFFFFFFF
        movement = actions[batch_index, time_index, :3]
        moving = float(movement[:2].abs().sum()) > 0.1
        has_attack = bool(button_value & int(IN_ATTACK))
        has_utility = bool(button_value & int(IN_ATTACK2)) or bool(
            utility_values is not None and float(utility_values[time_index, batch_index]) > 0.0
        )
        events = event_sets[batch_index][time_index]
        alive = bool(alive_values is None or alive_values[time_index, batch_index])
        if "defuse" in events:
            tactical_mode, task = "defuse", "defuse"
        elif "plant" in events:
            tactical_mode, task = "plant", "plant"
        elif "round_end" in events or not alive:
            tactical_mode, task = "save", "anchor"
        elif has_utility or any("grenade" in event or "smoke" in event or "flash" in event for event in events):
            tactical_mode, task = "advance", "use_utility"
        elif has_attack or "weapon_fire" in events or "player_hurt" in events:
            tactical_mode, task = "advance", "engage"
        elif moving:
            previous = actions[batch_index, max(0, time_index - 1), :2]
            direction_change = float((previous * movement[:2]).sum()) < -0.2
            tactical_mode, task = ("rotate", "move") if direction_change else ("advance", "move")
        else:
            tactical_mode, task = "hold", "anchor"
        current = positions[batch_index][time_index] or (0.0, 0.0, 0.0)
        waypoint, waypoint_valid = _future_position(batch_index, time_index, 16)
        goal, goal_valid = _future_position(batch_index, time_index, 64)
        slot = target_slots[batch_index][time_index]
        distance = target_distances[batch_index][time_index]
        health = float(decoded[batch_index][time_index].self_health) if decoded[batch_index][time_index] else 100.0
        visible_target = slot != TARGET_SLOT_NONE
        aggression = 1.0 if has_attack else 0.75 if has_utility else 0.6 if visible_target else 0.35 if moving else 0.15
        risk = 0.8 if health < 35.0 else 0.55 if visible_target else 0.25
        priority = 0.9 if events else 0.65 if has_attack or has_utility else 0.4 if moving else 0.2
        return _intent_target(
            current_position=current,
            waypoint=waypoint,
            goal=goal,
            facing=facings[batch_index][time_index],
            tactical_mode=tactical_mode,
            task=task,
            target_slot=slot,
            desired_range=distance,
            aggression=aggression,
            risk=risk,
            priority=priority,
            ttl_ticks=1 if events else 8,
            position_valid=waypoint_valid and goal_valid,
            target_valid=visible_target,
            has_event=bool(events),
            has_action=bool(button_value or moving),
        )

    intent_targets: list[list[IntentV1]] = [
        [_target_for(batch_index, time_index) for time_index in range(sequence_length)]
        for batch_index in range(batch_size)
    ]
    intent_embeddings = torch.tensor(
        [
            [label.to_embedding(128) for label in labels]
            for labels in intent_targets
        ],
        dtype=observations.dtype,
        device=device,
    )

    def _target_tensor(name: str, *, dtype: Any) -> Any:
        values = [[getattr(label, name) for label in labels] for labels in intent_targets]
        if name == "tactical_mode":
            values = [[TACTICAL_MODES.index(value) for value in row] for row in values]
        elif name == "task":
            values = [[INTENT_TASKS.index(value) for value in row] for row in values]
        return torch.tensor(values, dtype=dtype, device=device)

    decision_target_sequence = {
        "tactical_mode": _target_tensor("tactical_mode", dtype=torch.long),
        "task": _target_tensor("task", dtype=torch.long),
        "goal_position": _target_tensor("goal_position", dtype=torch.float32),
        "waypoint_position": _target_tensor("waypoint_position", dtype=torch.float32),
        "facing_yaw_pitch": _target_tensor("facing_yaw_pitch", dtype=torch.float32),
        "desired_range": _target_tensor("desired_range", dtype=torch.float32),
        "target_slot": _target_tensor("target_slot", dtype=torch.long),
        "aggression": _target_tensor("aggression", dtype=torch.float32),
        "risk": _target_tensor("risk", dtype=torch.float32),
        "priority": _target_tensor("priority", dtype=torch.float32),
        "ttl_ticks": _target_tensor("ttl_ticks", dtype=torch.float32),
        "confidence": {
            "all": torch.tensor(
                [[
                    [label.confidence.get(name, 0.0) for name in INTENT_FIELD_NAMES]
                    for label in labels
                ] for labels in intent_targets],
                dtype=torch.float32,
                device=device,
            )
        },
        "valid_mask": {
            "all": torch.tensor(
                [[
                    [label.valid_mask.get(name, False) for name in INTENT_FIELD_NAMES]
                    for label in labels
                ] for labels in intent_targets],
                dtype=torch.bool,
                device=device,
            )
        },
    }
    final_decision_targets = _sequence_target_at(decision_target_sequence, sequence_length - 1)
    # Keep the legacy adapter view stable for callers that use only the final
    # target; the actual sequence trainer consumes decision_target_sequence.
    if final_decision_targets["task"].eq(INTENT_TASKS.index("anchor")).any():
        final_decision_targets["task"] = final_decision_targets["task"].clone()
        final_decision_targets["task"] = torch.where(
            final_decision_targets["task"] == INTENT_TASKS.index("anchor"),
            torch.full_like(final_decision_targets["task"], INTENT_TASKS.index("move")),
            final_decision_targets["task"],
        )

    action_sequence = {
        "movement": actions[:, :, :3],
        "mouse": actions[:, :, 3:5],
        "buttons": torch.stack(
            [
                ((actions[:, :, 5].round().clamp(0.0, float(2**32 - 1)).to(torch.int64)
                 >> bit) & 1).to(torch.float32)
                for bit in range(32)
            ],
            dim=-1,
        ),
        "weapon": (actions[:, :, 6].round() + 1).to(torch.long).clamp(0, 15),
        "buy": actions[:, :, 7].round().to(torch.long).clamp(0, 31),
        "loss_mask": loss_masks,
        "duration_s": durations,
    }
    final_action_targets = {name: value[:, -1] for name, value in action_sequence.items()}
    observation_history = observations[:, -32:]
    local_observation = observations[:, -1]
    cached_intent = intent_embeddings[:, -1]
    return HierarchicalBatchV1(
        observation_history=observation_history,
        previous_intent=previous_intent,
        local_observation=local_observation,
        cached_intent=cached_intent,
        decision_targets=final_decision_targets,
        action_targets=final_action_targets,
        observation_sequence=observations,
        intent_embedding_sequence=intent_embeddings,
        decision_target_sequence=decision_target_sequence,
        action_target_sequence=action_sequence,
    )


@dataclass(frozen=True)
class HierarchicalDemoRunManifestV1:
    """Test-only training-run wrapper that preserves the dataset manifest lineage."""

    dataset_manifest: DatasetManifestV1
    name: str
    purpose: DataPurpose
    steps: int
    config_sha256: str
    checkpoint_path: Path
    loss_history: tuple[float, ...]
    metrics: Mapping[str, Any]

    def assert_exportable(self, target_purpose: DataPurpose | str = DataPurpose.TEST_ONLY) -> None:
        self.dataset_manifest.assert_exportable(target_purpose)


@dataclass(frozen=True)
class HierarchicalDemoTrainingReportV1:
    manifest_path: Path
    dataset_manifest: DatasetManifestV1
    package: HierarchicalPackageV2
    checkpoint_path: Path
    device: str
    update_count: int
    sample_count: int
    batches_by_demo: Mapping[str, int]
    mean_loss: float
    labeler_version: str


def train_test_only_demo_stage(
    config: Any,
    *,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    manifest_path: str | Path | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
    sequence_length: int = _SEQUENCE_LENGTH,
    batch_sequences: int = _BATCH_SEQUENCES,
    updates_per_demo: int | None = _UPDATES_PER_DEMO,
    windows_per_shard: int = _WINDOWS_PER_SHARD,
    epochs: int = 1,
    generation: int = 0,
    parent_generation: int | None = None,
    initial_checkpoint: str | Path | None = None,
    train_decision: bool = True,
    decision_learning_rate: float = 1e-4,
    action_learning_rate: float = 1e-4,
) -> HierarchicalDemoTrainingReportV1:
    """Train a bounded three-Demo test-only generation and export its package."""

    require_test_only(ensure_purpose(purpose))
    if sequence_length != _SEQUENCE_LENGTH:
        raise ValueError("hierarchical Demo training requires sequence_length=128")
    if batch_sequences <= 0 or (updates_per_demo is not None and updates_per_demo <= 0):
        raise ValueError("batch_sequences and updates_per_demo must be positive when specified")
    if windows_per_shard <= 0:
        raise ValueError("windows_per_shard must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if decision_learning_rate <= 0.0 or action_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if generation < 0:
        raise ValueError("generation must be non-negative")
    if generation == 0 and parent_generation is not None:
        raise ValueError("generation 0 cannot have a parent generation")
    if generation > 0 and parent_generation != generation - 1:
        raise ValueError("candidate generation parent must be exactly generation - 1")
    data_root = Path(config.data_root).resolve()
    manifest_file = Path(manifest_path or data_root / "manifests" / "three-demo-pilot-manifest-v1.json").resolve()
    manifest = load_test_only_sequence_manifest(manifest_file)
    groups = group_sequence_paths_by_demo(manifest)
    training_groups, heldout_demo, validation_paths = split_three_demo_train_validation(
        manifest,
        groups,
        heldout_demo=getattr(config, "heldout_demo", None),
    )
    training_paths = tuple(
        path
        for demo_paths in training_groups.values()
        for path in demo_paths
    )
    training_manifest = replace(
        manifest,
        metadata={
            **manifest.metadata,
            "train_paths_json": json.dumps([str(path) for path in training_paths]),
            "validation_paths_json": json.dumps([str(path) for path in validation_paths]),
            "test_paths_json": "[]",
            "split_policy": "leave_one_demo_out",
            "heldout_demo": heldout_demo,
        },
    )
    requested_device = str(device or getattr(config, "training_device", "cuda"))
    resolved_device = resolve_training_device(requested_device)
    target_dir = Path(output_dir or data_root / "models" / "generation-000").resolve()
    if target_dir.exists():
        raise FileExistsError(
            f"Demo generation output already exists; refusing duplicate training: {target_dir}"
        )

    try:
        import torch
    except ImportError as error:  # pragma: no cover - guarded by resolve_training_device
        raise RuntimeError("PyTorch is required for hierarchical Demo training") from error

    seed = int(getattr(config, "seed", 7))
    torch.manual_seed(seed)
    actor = prepare_hierarchical_training(HierarchicalActor())
    from .quantization import prepare_decision_qat, qat_state_dict

    # QAT is part of the training model, not an export-time decoration.  The
    # checkpoint and package must be able to prove that this state was active
    # while the decision optimizer was running.
    prepare_decision_qat(actor.decision)
    if initial_checkpoint is not None:
        checkpoint_file = Path(initial_checkpoint).resolve()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
            raise ValueError("hierarchical checkpoint does not contain model_state_dict")
        actor.load_state_dict(checkpoint["model_state_dict"])
    actor.to(resolved_device)
    decision_optimizer = torch.optim.AdamW(actor.decision.parameters(), lr=decision_learning_rate)
    action_optimizer = torch.optim.AdamW(actor.action.parameters(), lr=action_learning_rate)

    losses: list[float] = []
    batches_by_demo: dict[str, int] = {}
    sample_count = 0
    for epoch in range(epochs):
        for demo_index, (demo_name, paths) in enumerate(training_groups.items()):
            updates = 0
            for recurrent in iter_recurrent_minibatches(
                paths,
                sequence_length=sequence_length,
                batch_sequences=batch_sequences,
                seed=seed + epoch * 1_000_003 + demo_index,
                max_windows_per_path=windows_per_shard,
            ):
                hierarchical_batch = recurrent_to_hierarchical_batch(recurrent).to(resolved_device)
                metrics = train_hierarchical_step(
                    actor,
                    hierarchical_batch,
                    decision_optimizer=decision_optimizer,
                    action_optimizer=action_optimizer,
                    train_decision=train_decision,
                )
                losses.append(float(metrics["loss"].detach().cpu()))
                sample_count += hierarchical_batch.batch_size
                updates += 1
                if updates_per_demo is not None and updates >= updates_per_demo:
                    break
            if updates == 0:
                raise ValueError(f"Demo group {demo_name} did not yield a recurrent batch")
            batches_by_demo[demo_name] = batches_by_demo.get(demo_name, 0) + updates

    target_dir.mkdir(parents=True, exist_ok=False)
    split_manifest = {
        "schema": "three-demo-pilot-v1",
        "manifest_path": str(manifest_file),
        "train": [str(path) for path in training_paths],
        "validation": [str(path) for path in validation_paths],
        "test": [],
        "demo_groups": dict(batches_by_demo),
        "heldout_demo": heldout_demo,
    }
    checkpoint_path = target_dir / "hierarchical-demo.pt"
    build_hierarchical_checkpoint(
        checkpoint_path,
        actor=actor,
        decision_optimizer=decision_optimizer,
        action_optimizer=action_optimizer,
        qat_state={"enabled": True, "observer_version": "qat-v1", **qat_state_dict(actor.decision)},
        split_manifest=split_manifest,
        labeler_version=_LABELER_VERSION,
    )
    run_manifest = HierarchicalDemoRunManifestV1(
        dataset_manifest=training_manifest,
        name=f"hierarchical-test-three-demo-generation-{generation:03d}",
        purpose=DataPurpose.TEST_ONLY,
        steps=len(losses),
        config_sha256="",
        checkpoint_path=checkpoint_path,
        loss_history=tuple(losses),
        metrics={
            "demo_count": 3,
            "sequence_count": len(training_paths),
            "validation_sequence_count": len(validation_paths),
            "heldout_demo": heldout_demo,
            "heldout_validation_paths": [str(path) for path in validation_paths],
            "batches_by_demo": dict(batches_by_demo),
            "windows_per_shard": windows_per_shard,
            "epochs": epochs,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "train_decision": train_decision,
            "decision_learning_rate": decision_learning_rate,
            "action_learning_rate": action_learning_rate,
            "labeler_version": _LABELER_VERSION,
            "decision_qat_enabled": True,
        },
    )
    package = export_hierarchical_package(
        run_manifest,
        target_dir,
        DataPurpose.TEST_ONLY,
        actor=actor,
        generation=generation,
        parent_generation=parent_generation,
        metrics={
            "demo_training": run_manifest.metrics,
            "update_count": len(losses),
            "sample_count": sample_count,
            "windows_per_shard": windows_per_shard,
            "epochs": epochs,
            "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
            "train_decision": train_decision,
            "decision_learning_rate": decision_learning_rate,
            "action_learning_rate": action_learning_rate,
            "mean_loss": sum(losses) / len(losses),
            "device": requested_device,
            "heldout_demo": heldout_demo,
            "heldout_validation_paths": [str(path) for path in validation_paths],
            "decision_qat_enabled": True,
        },
    )
    # The Demo baseline is a gate input in its own right. Persist the same
    # held-out, teacher-forced evidence here so a direct call to this training
    # stage cannot export a package that only becomes "measurable" after
    # self-play.
    from .test_pipeline import _evaluate_heldout_demo_evidence, _package_with_gate_evidence

    package = _package_with_gate_evidence(
        package,
        _evaluate_heldout_demo_evidence(
            actor,
            validation_paths,
            device=str(resolved_device),
            seed=seed,
        ),
    )
    report = HierarchicalDemoTrainingReportV1(
        manifest_path=manifest_file,
        dataset_manifest=training_manifest,
        package=package,
        checkpoint_path=checkpoint_path,
        device=requested_device,
        update_count=len(losses),
        sample_count=sample_count,
        batches_by_demo=dict(batches_by_demo),
        mean_loss=sum(losses) / len(losses),
        labeler_version=_LABELER_VERSION,
    )
    (target_dir / "demo-stage-report.json").write_text(
        json.dumps(
            {
                "manifest_path": str(report.manifest_path),
                "package": report.package.to_dict(),
                "checkpoint_path": str(report.checkpoint_path),
                "device": report.device,
                "update_count": report.update_count,
                "sample_count": report.sample_count,
                "batches_by_demo": dict(report.batches_by_demo),
                "mean_loss": report.mean_loss,
                "labeler_version": report.labeler_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
