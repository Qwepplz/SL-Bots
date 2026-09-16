from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Mapping

from .contracts import BotActionV1, DataPurpose, Phase, ensure_purpose
from .lineage import DatasetManifestV1, ProductionLineageError
from .training_device import resolve_training_device


_HUMAN_TARGET_NAMES = frozenset(
    {
        "max_angular_velocity_deg_s",
        "max_angular_acceleration_deg_s2",
        "stop_go_ratio",
        "economy_choice_count",
        "utility_event_rate",
    }
)

_ACTION_CONTINUOUS_FEATURES = 5
_ACTION_BUTTON_FEATURES = 32
_ACTION_WEAPON_FEATURES = 17
_ACTION_BUY_FEATURES = 32
_ACTION_MASK_FEATURES = 8
ACTION_FEATURE_SIZE = (
    _ACTION_CONTINUOUS_FEATURES
    + _ACTION_BUTTON_FEATURES
    + _ACTION_WEAPON_FEATURES
    + _ACTION_BUY_FEATURES
    + _ACTION_MASK_FEATURES
)


class PhaseMixError(ValueError):
    pass


@dataclass(frozen=True)
class WindowSampleV1:
    phase: Phase
    observations: tuple[bytes, ...]
    actions: tuple[BotActionV1, ...]
    source: str
    episode_id: str
    delta_time_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase(self.phase))
        observations = tuple(bytes(value) for value in self.observations)
        actions = tuple(self.actions)
        if len(observations) != len(actions):
            raise ValueError("window observations and actions must have equal lengths")
        if any(len(value) != 256 for value in observations):
            raise ValueError("window observations must be 256 bytes")
        if any(not isinstance(action, BotActionV1) for action in actions):
            raise TypeError("window actions must contain BotActionV1 values")
        if self.source not in {"demo", "selfplay"}:
            raise ValueError("window source must be demo or selfplay")
        if not self.episode_id:
            raise ValueError("window episode_id cannot be empty")
        durations = (
            tuple(float(value) for value in self.delta_time_s)
            if self.delta_time_s
            else tuple(1.0 / 128.0 for _ in observations)
        )
        if len(durations) != len(observations):
            raise ValueError("window delta_time_s must align with observations")
        if any(not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("window delta_time_s values must be finite and positive")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "delta_time_s", durations)

    @property
    def window_length(self) -> int:
        return len(self.observations)


def _validate_window_phases(windows: Sequence[WindowSampleV1], phase: Phase) -> tuple[WindowSampleV1, ...]:
    normalized = tuple(windows)
    if any(not isinstance(window, WindowSampleV1) for window in normalized):
        raise TypeError("windows must contain WindowSampleV1 values")
    if any(window.phase is not phase for window in normalized):
        raise PhaseMixError(f"window batch contains a phase other than {phase.name.lower()}")
    return normalized


class _WindowStream:
    def __init__(self, source: Iterable[WindowSampleV1]) -> None:
        self.source = source
        self.restartable = isinstance(source, Sequence) or iter(source) is not source
        self.iterator = iter(source)

    def next(self) -> WindowSampleV1:
        try:
            return next(self.iterator)
        except StopIteration as error:
            if not self.restartable:
                raise ValueError("GAIL window stream ended before the requested updates") from error
            self.iterator = iter(self.source)
            try:
                return next(self.iterator)
            except StopIteration as empty_error:
                raise ValueError("GAIL window stream cannot be empty") from empty_error


def _stream_batch(
    stream: _WindowStream,
    count: int,
    *,
    phase: Phase,
    window_length: int,
) -> list[WindowSampleV1]:
    selected: list[WindowSampleV1] = []
    for _ in range(count):
        window = stream.next()
        if not isinstance(window, WindowSampleV1):
            raise TypeError("GAIL streams must contain WindowSampleV1 values")
        if window.phase is not phase:
            raise PhaseMixError(f"window batch contains a phase other than {phase.name.lower()}")
        if window.window_length != window_length:
            raise ValueError(
                f"GAIL windows must contain exactly {window_length} ticks; got {window.window_length}"
            )
        selected.append(window)
    return selected


def _gail_paths(paths: Iterable[str | Path] | str | Path) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        values = (Path(paths),)
    else:
        values = tuple(Path(value) for value in paths)
    if not values:
        raise ValueError("GAIL window paths cannot be empty")
    return tuple(value.resolve() for value in values)


def _row_value(batch: Any, name: str, index: int) -> Any:
    column_index = batch.schema.get_field_index(name)
    value = batch.column(column_index)[index]
    return value.as_py() if hasattr(value, "as_py") else value


def _parquet_phase(parquet_file: Any, phase: Phase) -> None:
    metadata = parquet_file.schema_arrow.metadata or {}
    encoded_phase = metadata.get(b"phase")
    if encoded_phase is not None and encoded_phase.decode("utf-8").lower() != phase.name.lower():
        raise PhaseMixError(f"GAIL shard phase does not match {phase.name.lower()}")


def _action_from_sequence_row(batch: Any, index: int) -> BotActionV1:
    return BotActionV1(
        target_tick=int(_row_value(batch, "target_tick", index)),
        forward=float(_row_value(batch, "forward", index)),
        side=float(_row_value(batch, "side", index)),
        up=float(_row_value(batch, "up", index)),
        yaw_delta_deg=float(_row_value(batch, "yaw_delta_deg", index)),
        pitch_delta_deg=float(_row_value(batch, "pitch_delta_deg", index)),
        buttons=int(_row_value(batch, "buttons", index)),
        weapon_select=int(_row_value(batch, "weapon_select", index)),
        buy_action=int(_row_value(batch, "buy_action", index)),
        action_valid_mask=int(_row_value(batch, "loss_mask", index)),
    )


def _iter_sequence_windows(
    parquet_file: Any,
    *,
    source: str,
    phase: Phase,
    window_length: int,
    required: Sequence[str],
) -> Iterator[WindowSampleV1]:
    current_sequence: str | None = None
    observations: list[bytes] = []
    actions: list[BotActionV1] = []
    durations: list[float] = []
    for batch in parquet_file.iter_batches(batch_size=window_length, columns=list(required)):
        for index in range(batch.num_rows):
            sequence_id = str(_row_value(batch, "sequence_id", index))
            if not sequence_id:
                raise ValueError("GAIL sequence_id values must be non-empty")
            if current_sequence != sequence_id:
                observations = []
                actions = []
                durations = []
            current_sequence = sequence_id
            observation = bytes(_row_value(batch, "observation", index))
            if len(observation) != 256:
                raise ValueError("GAIL observations must be 256 bytes")
            observations.append(observation)
            actions.append(_action_from_sequence_row(batch, index))
            duration_s = float(_row_value(batch, "duration_s", index))
            durations.append(duration_s if duration_s > 0.0 else 1.0 / 128.0)
            if len(observations) == window_length:
                yield WindowSampleV1(
                    phase=phase,
                    observations=tuple(observations),
                    actions=tuple(actions),
                    source=source,
                    episode_id=sequence_id,
                    delta_time_s=tuple(durations),
                )
                observations = []
                actions = []
                durations = []


def _iter_trajectory_windows(
    parquet_file: Any,
    *,
    source: str,
    phase: Phase,
    window_length: int,
    required: Sequence[str],
) -> Iterator[WindowSampleV1]:
    buffers: dict[tuple[str, str], tuple[list[bytes], list[BotActionV1], list[float]]] = {}
    last_ticks: dict[tuple[str, str], int] = {}
    for batch in parquet_file.iter_batches(batch_size=max(window_length, 256), columns=list(required)):
        for index in range(batch.num_rows):
            row_phase = Phase[str(_row_value(batch, "phase", index)).upper()]
            if row_phase is not phase:
                raise PhaseMixError(f"GAIL trajectory phase does not match {phase.name.lower()}")
            episode_id = str(_row_value(batch, "episode_id", index))
            bot_id = str(_row_value(batch, "bot_id", index))
            if not episode_id or not bot_id:
                raise ValueError("GAIL trajectory episode_id and bot_id cannot be empty")
            key = (episode_id, bot_id)
            hidden_state_mask = bool(_row_value(batch, "hidden_state_mask", index))
            if not hidden_state_mask:
                buffers[key] = ([], [], [])
                last_ticks.pop(key, None)
            if key not in buffers:
                raise ValueError("GAIL trajectory starts without a hidden-state boundary")
            server_tick = int(_row_value(batch, "server_tick", index))
            previous_tick = last_ticks.get(key)
            if previous_tick is not None and server_tick <= previous_tick:
                raise ValueError("GAIL trajectory ticks must be strictly monotonic per bot")
            observation = bytes(_row_value(batch, "observation", index))
            if len(observation) != 256:
                raise ValueError("GAIL observations must be 256 bytes")
            action = BotActionV1.from_bytes(bytes(_row_value(batch, "action", index)))
            observations, actions, durations = buffers[key]
            observations.append(observation)
            actions.append(action)
            previous_tick = last_ticks.get(key)
            durations.append(
                1.0 / 128.0
                if previous_tick is None
                else max(1, server_tick - previous_tick) / 128.0
            )
            last_ticks[key] = server_tick
            if len(observations) == window_length:
                yield WindowSampleV1(
                    phase=phase,
                    observations=tuple(observations),
                    actions=tuple(actions),
                    source=source,
                    episode_id=f"{episode_id}/{bot_id}",
                    delta_time_s=tuple(durations),
                )
                buffers[key] = ([], [], [])
            if bool(_row_value(batch, "done", index)):
                buffers.pop(key, None)
                last_ticks.pop(key, None)


def iter_gail_windows(
    paths: Iterable[str | Path] | str | Path,
    *,
    source: str,
    phase: Phase | int = Phase.LIVE,
    window_length: int = 128,
) -> Iterator[WindowSampleV1]:
    if source not in {"demo", "selfplay"}:
        raise ValueError("window source must be demo or selfplay")
    if not isinstance(window_length, int) or isinstance(window_length, bool) or window_length <= 0:
        raise ValueError("window_length must be positive")
    phase = Phase(phase)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for GAIL window streaming") from error
    required = (
        "observation",
        "sequence_id",
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
        "duration_s",
    )
    trajectory_required = (
        "episode_id",
        "phase",
        "server_tick",
        "bot_id",
        "observation",
        "action",
        "done",
        "hidden_state_mask",
    )
    for path in _gail_paths(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet_file = pq.ParquetFile(str(path))
        missing = [name for name in required if name not in parquet_file.schema.names]
        if not missing:
            _parquet_phase(parquet_file, phase)
            yield from _iter_sequence_windows(
                parquet_file,
                source=source,
                phase=phase,
                window_length=window_length,
                required=required,
            )
            continue
        trajectory_missing = [
            name for name in trajectory_required if name not in parquet_file.schema.names
        ]
        if trajectory_missing:
            raise ValueError(f"GAIL shard is missing columns: {', '.join(missing)}")
        _parquet_phase(parquet_file, phase)
        yield from _iter_trajectory_windows(
            parquet_file,
            source=source,
            phase=phase,
            window_length=window_length,
            required=trajectory_required,
        )


class GailWindowPathStream:
    def __init__(
        self,
        paths: Iterable[str | Path] | str | Path,
        *,
        source: str,
        phase: Phase | int = Phase.LIVE,
        window_length: int = 128,
    ) -> None:
        self.paths = _gail_paths(paths)
        self.source = source
        self.phase = Phase(phase)
        self.window_length = window_length

    def __iter__(self) -> Iterator[WindowSampleV1]:
        return iter_gail_windows(
            self.paths,
            source=self.source,
            phase=self.phase,
            window_length=self.window_length,
        )


def _window_tensors(
    windows: Sequence[WindowSampleV1],
    *,
    device: Any,
) -> tuple[Any, Any, Any]:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for discriminator training")
    observations = torch.tensor(
        [[[byte for byte in observation] for observation in window.observations] for window in windows],
        dtype=torch.float32,
        device=device,
    ) / 255.0
    actions = torch.tensor(
        [
            [_action_feature_vector(action) for action in window.actions]
            for window in windows
        ],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor(
        [1.0 if window.source == "demo" else 0.0 for window in windows],
        dtype=torch.float32,
        device=device,
    )
    return observations, actions, labels


def _action_feature_vector(action: BotActionV1) -> list[float]:
    features = [
        float(action.forward),
        float(action.side),
        float(action.up),
        float(action.yaw_delta_deg),
        float(action.pitch_delta_deg),
    ]
    features.extend(float((int(action.buttons) >> bit) & 1) for bit in range(_ACTION_BUTTON_FEATURES))
    weapon_index = min(_ACTION_WEAPON_FEATURES - 1, max(0, int(action.weapon_select) + 1))
    features.extend(float(index == weapon_index) for index in range(_ACTION_WEAPON_FEATURES))
    buy_index = min(_ACTION_BUY_FEATURES - 1, max(0, int(action.buy_action)))
    features.extend(float(index == buy_index) for index in range(_ACTION_BUY_FEATURES))
    features.extend(
        float(bool(int(action.action_valid_mask) & (1 << bit)))
        for bit in range(_ACTION_MASK_FEATURES)
    )
    return features


def _lineage_manifests(manifests: Iterable[DatasetManifestV1]) -> tuple[DatasetManifestV1, ...]:
    result: list[DatasetManifestV1] = []
    visited: set[int] = set()

    def visit(manifest: DatasetManifestV1) -> None:
        if id(manifest) in visited:
            return
        visited.add(id(manifest))
        result.append(manifest)
        for parent in manifest.parents:
            visit(parent)

    for manifest in manifests:
        if not isinstance(manifest, DatasetManifestV1):
            raise TypeError("GAIL parents must contain DatasetManifestV1 values")
        visit(manifest)
    return tuple(result)


def _validate_gail_parents(
    parent_manifests: Sequence[DatasetManifestV1],
    purpose: DataPurpose,
) -> tuple[DatasetManifestV1, ...]:
    parents = tuple(parent_manifests)
    lineage = _lineage_manifests(parents)
    if purpose is not DataPurpose.PRODUCTION:
        return parents
    for manifest in parents:
        if manifest.effective_purpose() is DataPurpose.TEST_ONLY:
            raise ProductionLineageError(
                f"GAIL parent {manifest.name} has a test_only ancestor"
            )
    rulesets = {
        str(manifest.metadata.get("ruleset", "")).lower()
        for manifest in lineage
    }
    if "mr15" not in rulesets or "mr12" not in rulesets:
        raise ProductionLineageError(
            "production GAIL requires both an MR15 expert corpus and an MR12 bootstrap parent"
        )
    return parents


def _gail_dataset_manifest(
    *,
    name: str,
    phase: Phase,
    purpose: DataPurpose,
    parent_manifests: Sequence[DatasetManifestV1],
    window_length: int,
    batch_size: int,
    demo_ratio: float,
    steps: int,
    seed: int,
    device_type: str,
    device_name: str,
    hip_version: str,
) -> DatasetManifestV1:
    parents = _validate_gail_parents(parent_manifests, purpose)
    return DatasetManifestV1(
        name=f"gail:{name}",
        purpose=purpose,
        parents=parents,
        artifact_type="gail_windows",
        metadata={
            "phase": phase.name.lower(),
            "expert_ruleset": "mr15",
            "learner_ruleset": "mr12",
            "window_length": str(window_length),
            "batch_size": str(batch_size),
            "demo_ratio": str(demo_ratio),
            "max_updates": str(steps),
            "validation_interval_updates": "100",
            "early_stopping_patience": "5",
            "seed": str(seed),
            "device_type": device_type,
            "device_name": device_name,
            "hip_version": hip_version,
        },
    )


def sample_rehearsal_windows(
    demo_windows: Sequence[WindowSampleV1],
    selfplay_windows: Sequence[WindowSampleV1],
    *,
    phase: Phase | int,
    batch_size: int,
    demo_ratio: float,
    seed: int = 7,
) -> tuple[WindowSampleV1, ...]:
    phase = Phase(phase)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be between 0 and 1")
    demos = _validate_window_phases(demo_windows, phase)
    learners = _validate_window_phases(selfplay_windows, phase)
    demo_count = min(batch_size, int(batch_size * demo_ratio + 0.5))
    learner_count = batch_size - demo_count
    if demo_count and not demos:
        raise ValueError("demo rehearsal requires at least one demo window")
    if learner_count and not learners:
        raise ValueError("self-play sampling requires at least one self-play window")
    rng = random.Random(seed)
    selected = [demos[rng.randrange(len(demos))] for _ in range(demo_count)]
    selected.extend(learners[rng.randrange(len(learners))] for _ in range(learner_count))
    rng.shuffle(selected)
    return tuple(selected)


def discriminator_bce_with_logits(logits: Sequence[float], labels: Sequence[float]) -> float:
    if len(logits) != len(labels) or not logits:
        raise ValueError("logits and labels must be non-empty and equal length")
    total = 0.0
    for logit, label in zip(logits, labels):
        value = float(logit)
        target = float(label)
        if target not in (0.0, 1.0):
            raise ValueError("discriminator labels must be 0 or 1")
        total += max(value, 0.0) - value * target + math.log1p(math.exp(-abs(value)))
    return total / len(logits)


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if nn is not None:

    class WindowDiscriminator(nn.Module):
        action_feature_size = ACTION_FEATURE_SIZE

        def __init__(self, hidden_size: int = 128) -> None:
            super().__init__()
            self.temporal = nn.GRU(
                256 + ACTION_FEATURE_SIZE,
                hidden_size,
                batch_first=True,
            )
            self.network = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, observations: Any, actions: Any) -> Any:
            if observations.ndim != 3 or observations.shape[-1] != 256:
                raise ValueError("discriminator observations must have shape [batch, time, 256]")
            if actions.ndim != 3 or actions.shape[-1] not in {5, ACTION_FEATURE_SIZE}:
                raise ValueError(
                    "discriminator actions must have shape [batch, time, 5] or "
                    f"[batch, time, {ACTION_FEATURE_SIZE}]"
                )
            action_features = actions.float()
            if action_features.shape[-1] == 5:
                action_features = F.pad(
                    action_features,
                    (0, ACTION_FEATURE_SIZE - _ACTION_CONTINUOUS_FEATURES),
                )
            sequence = torch.cat([observations.float(), action_features], dim=-1)
            _, state = self.temporal(sequence)
            return self.network(state[-1]).squeeze(-1)

else:

    class WindowDiscriminator:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch 2.9 is required for WindowDiscriminator")


@dataclass(frozen=True)
class DiscriminatorRunManifestV1:
    name: str
    purpose: DataPurpose
    dataset_manifest: DatasetManifestV1
    steps: int
    loss_history: tuple[float, ...]
    checkpoint_path: str
    config_sha256: str
    device_type: str = "cpu"
    device_index: int | None = None
    device_name: str = ""
    torch_version: str = ""
    hip_version: str = ""
    validation_history: tuple[float, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def assert_exportable(self, target_purpose: DataPurpose | str = DataPurpose.PRODUCTION) -> None:
        target = ensure_purpose(target_purpose)
        self.dataset_manifest.assert_exportable(target)
        if target is DataPurpose.PRODUCTION and self.purpose is DataPurpose.TEST_ONLY:
            raise ProductionLineageError("test-only discriminator cannot be exported as production")


def run_discriminator_updates(
    discriminator: Any,
    demo_windows: Iterable[WindowSampleV1],
    selfplay_windows: Iterable[WindowSampleV1],
    steps: int,
    batch_size: int,
    demo_ratio: float,
    *,
    phase: Phase | int = Phase.LIVE,
    device: str = "cuda",
    seed: int = 7,
    learning_rate: float = 0.001,
    window_length: int = 128,
    validation_demo_windows: Iterable[WindowSampleV1] | None = None,
    validation_selfplay_windows: Iterable[WindowSampleV1] | None = None,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    parent_manifests: Sequence[DatasetManifestV1] = (),
    dataset_manifest: DatasetManifestV1 | None = None,
    output_dir: str | Path | None = None,
    name: str = "discriminator",
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> DiscriminatorRunManifestV1:
    if torch is None or F is None:
        raise RuntimeError("PyTorch 2.9 is required for discriminator training")
    if not isinstance(discriminator, nn.Module):
        raise TypeError("discriminator must be a torch.nn.Module")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 2000:
        raise ValueError("steps must be between 1 and 2000")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= demo_ratio <= 1.0:
        raise ValueError("demo_ratio must be between 0 and 1")
    if not isinstance(window_length, int) or isinstance(window_length, bool) or window_length <= 0:
        raise ValueError("window_length must be positive")
    if learning_rate <= 0.0 or not math.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be finite and positive")
    phase = Phase(phase)
    run_purpose = ensure_purpose(purpose)
    resolved = resolve_training_device(str(device))
    device_index = None
    device_name = "CPU"
    hip_version = "unavailable"
    if resolved.type == "cuda":
        device_index = int(resolved.index if resolved.index is not None else torch.cuda.current_device())
        resolved = torch.device("cuda", device_index)
        device_name = str(torch.cuda.get_device_name(device_index))
        hip_version = str(getattr(torch.version, "hip", None) or "unavailable")
    discriminator.to(resolved)
    discriminator.train()
    for parameter in discriminator.parameters():
        if parameter.device != resolved:
            raise RuntimeError("GAIL discriminator parameters are not on the requested device")
    dataset = dataset_manifest or _gail_dataset_manifest(
        name=name,
        phase=phase,
        purpose=run_purpose,
        parent_manifests=parent_manifests,
        window_length=window_length,
        batch_size=batch_size,
        demo_ratio=demo_ratio,
        steps=steps,
        seed=seed,
        device_type=resolved.type,
        device_name=device_name,
        hip_version=hip_version,
    )
    if not isinstance(dataset, DatasetManifestV1):
        raise TypeError("dataset_manifest must be DatasetManifestV1")
    if run_purpose is DataPurpose.PRODUCTION:
        _validate_gail_parents(dataset.parents, run_purpose)
        dataset.assert_exportable(DataPurpose.PRODUCTION)
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate)
    if optimizer_state_dict is not None:
        if not isinstance(optimizer_state_dict, Mapping):
            raise TypeError("optimizer_state_dict must be a mapping")
        optimizer.load_state_dict(dict(optimizer_state_dict))
        for state in optimizer.state.values():
            for key, value in tuple(state.items()):
                if hasattr(value, "to"):
                    state[key] = value.to(resolved)
    demo_stream = _WindowStream(demo_windows)
    selfplay_stream = _WindowStream(selfplay_windows)
    validation_demo_stream = (
        _WindowStream(validation_demo_windows) if validation_demo_windows is not None else None
    )
    validation_selfplay_stream = (
        _WindowStream(validation_selfplay_windows) if validation_selfplay_windows is not None else None
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    loss_history: list[float] = []
    validation_history: list[float] = []
    best_validation = math.inf
    stale_updates = 0
    demo_count = min(batch_size, int(batch_size * demo_ratio + 0.5))
    learner_count = batch_size - demo_count
    for update in range(1, steps + 1):
        windows = _stream_batch(
            demo_stream,
            demo_count,
            phase=phase,
            window_length=window_length,
        )
        windows.extend(
            _stream_batch(
                selfplay_stream,
                learner_count,
                phase=phase,
                window_length=window_length,
            )
        )
        rng.shuffle(windows)
        observations, actions, labels = _window_tensors(windows, device=resolved)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            discriminator(observations, actions),
            labels,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("GAIL discriminator produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        if (
            update % 100 == 0
            and validation_demo_stream is not None
            and validation_selfplay_stream is not None
        ):
            validation_windows = _stream_batch(
                validation_demo_stream,
                demo_count,
                phase=phase,
                window_length=window_length,
            )
            validation_windows.extend(
                _stream_batch(
                    validation_selfplay_stream,
                    learner_count,
                    phase=phase,
                    window_length=window_length,
                )
            )
            validation_observations, validation_actions, validation_labels = _window_tensors(
                validation_windows,
                device=resolved,
            )
            discriminator.eval()
            with torch.no_grad():
                validation_loss = F.binary_cross_entropy_with_logits(
                    discriminator(validation_observations, validation_actions),
                    validation_labels,
                )
            discriminator.train()
            validation_value = float(validation_loss.detach().cpu())
            validation_history.append(validation_value)
            if validation_value < best_validation:
                best_validation = validation_value
                stale_updates = 0
            else:
                stale_updates += 1
                if stale_updates >= 5:
                    break
    metadata = dict(dataset.metadata)
    metadata.update(
        {
            "updates_completed": str(len(loss_history)),
            "gradient_clip_norm": "1.0",
            "checkpoint_cpu_loadable": "true",
        }
    )
    checkpoint_path = ""
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "gail-checkpoint.pt"
        payload = {
            "model_state_dict": {key: value.detach().cpu() for key, value in discriminator.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {
                "hidden_size": int(
                    getattr(
                        getattr(discriminator, "network", (None,))[0],
                        "out_features",
                        128,
                    )
                )
            },
            "phase": phase.name.lower(),
            "loss_history": loss_history,
            "validation_history": validation_history,
            "dataset_manifest": _manifest_payload(dataset),
            "device": {
                "type": resolved.type,
                "index": device_index,
                "name": device_name,
                "torch_version": str(torch.__version__),
                "hip_version": hip_version,
            },
        }
        _atomic_torch_save(checkpoint, payload)
        checkpoint_path = str(checkpoint)
    config_sha256 = hashlib.sha256(
        (
            f"{phase.name.lower()}|{steps}|{batch_size}|{demo_ratio}|{seed}|"
            f"{window_length}|{learning_rate}|{resolved}"
        ).encode()
    ).hexdigest()
    return DiscriminatorRunManifestV1(
        name=name,
        purpose=run_purpose,
        dataset_manifest=dataset,
        steps=len(loss_history),
        loss_history=tuple(loss_history),
        checkpoint_path=checkpoint_path,
        config_sha256=config_sha256,
        device_type=resolved.type,
        device_index=device_index,
        device_name=device_name,
        torch_version=str(torch.__version__),
        hip_version=hip_version,
        validation_history=tuple(validation_history),
        metadata=metadata,
    )


def _manifest_payload(manifest: DatasetManifestV1) -> dict[str, Any]:
    return {
        "schema": "dataset-manifest-v1",
        "name": manifest.name,
        "purpose": manifest.purpose.value,
        "artifact_type": manifest.artifact_type,
        "source_sha256": manifest.source_sha256,
        "parser_version": manifest.parser_version,
        "projection_version": manifest.projection_version,
        "parents": [_manifest_payload(parent) for parent in manifest.parents],
        "metadata": {str(key): str(value) for key, value in manifest.metadata.items()},
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def train_discriminator(
    demo_windows: Iterable[WindowSampleV1],
    selfplay_windows: Iterable[WindowSampleV1],
    *,
    phase: Phase | int,
    output_dir: str | Path,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    steps: int = 2000,
    batch_size: int = 64,
    demo_ratio: float = 0.5,
    seed: int = 7,
    parent_manifests: Sequence[DatasetManifestV1] = (),
    device: str = "cuda",
    learning_rate: float = 0.001,
    window_length: int = 128,
    validation_demo_windows: Iterable[WindowSampleV1] | None = None,
    validation_selfplay_windows: Iterable[WindowSampleV1] | None = None,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> DiscriminatorRunManifestV1:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for discriminator training")
    phase = Phase(phase)
    run_purpose = ensure_purpose(purpose)
    return run_discriminator_updates(
        WindowDiscriminator(),
        demo_windows,
        selfplay_windows,
        steps,
        batch_size,
        demo_ratio,
        phase=phase,
        device=device,
        seed=seed,
        learning_rate=learning_rate,
        window_length=window_length,
        validation_demo_windows=validation_demo_windows,
        validation_selfplay_windows=validation_selfplay_windows,
        purpose=run_purpose,
        parent_manifests=parent_manifests,
        output_dir=output_dir,
        name=Path(output_dir).name,
        optimizer_state_dict=optimizer_state_dict,
    )


@dataclass(frozen=True)
class HumanLikeMetricsV1:
    max_angular_velocity_deg_s: float
    max_angular_acceleration_deg_s2: float
    stop_go_ratio: float
    space_occupancy: float
    engagement_distance: float
    utility_event_rate: float
    economy_choice_count: int
    survival_time_s: float

    @property
    def max_angular_acceleration_deg_s(self) -> float:
        return self.max_angular_acceleration_deg_s2


class _HumanMetricsAccumulator:
    """计算单个或多个回合的指标，只保留固定大小的状态。"""

    _CELL_BITMAP_SIZE = 1 << 15

    def __init__(self) -> None:
        self.count = 0
        self._first_time: float | None = None
        self._last_time: float | None = None
        self._elapsed_s = 0.0
        self._episode_survival_s = 0.0
        self._previous_yaw: float | None = None
        self._previous_velocity: float | None = None
        self._previous_movement: bool | None = None
        self._max_velocity = 0.0
        self._max_acceleration = 0.0
        self._stop_go_changes = 0
        self._distance_sum = 0.0
        self._distance_count = 0
        self._utility_events = 0
        self._economy_choices = 0
        self._cells = bytearray(self._CELL_BITMAP_SIZE)
        self._occupied_cells = 0

    def begin_episode(self) -> None:
        self._finish_episode()
        self._first_time = None
        self._last_time = None
        self._previous_yaw = None
        self._previous_velocity = None
        self._previous_movement = None

    def _finish_episode(self) -> None:
        if self._first_time is not None and self._last_time is not None:
            self._episode_survival_s += max(0.0, self._last_time - self._first_time)

    @staticmethod
    def _wrapped_delta(value: float) -> float:
        return (value + 180.0) % 360.0 - 180.0

    def add(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("human-like records must be mappings")
        explicit_time = record.get("time_s")
        if explicit_time is None:
            time_s = self._last_time if self._last_time is not None else 0.0
        else:
            time_s = float(explicit_time)
        if not math.isfinite(time_s):
            raise ValueError("human-like time_s values must be finite")
        duration_value = record.get("delta_time_s", record.get("duration_s"))
        if duration_value is None:
            duration_s = (
                max(time_s - self._last_time, 1e-6)
                if self._last_time is not None
                else 1.0 / 128.0
            )
        else:
            duration_s = float(duration_value)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("human-like duration values must be finite and positive")
        if self._last_time is not None:
            delta_time = max(time_s - self._last_time, duration_s, 1e-6)
            try:
                yaw = float(record.get("yaw_deg"))
            except (TypeError, ValueError):
                yaw = self._previous_yaw + float(record.get("yaw_delta_deg", 0.0))
            delta_yaw = self._wrapped_delta(yaw - float(self._previous_yaw or 0.0))
            velocity = delta_yaw / delta_time
            self._max_velocity = max(self._max_velocity, abs(velocity))
            if self._previous_velocity is not None:
                self._max_acceleration = max(
                    self._max_acceleration,
                    abs(velocity - self._previous_velocity) / delta_time,
                )
            self._previous_velocity = velocity
        else:
            try:
                yaw = float(record.get("yaw_deg"))
            except (TypeError, ValueError):
                yaw = float(record.get("yaw_delta_deg", 0.0))
            self._first_time = time_s
        movement = abs(float(record.get("forward", 0.0))) > 0.1
        if self._previous_movement is not None:
            self._stop_go_changes += int(movement != self._previous_movement)
        self._previous_movement = movement
        position = record.get("position")
        if (
            bool(record.get("position_valid", position is not None))
            and position is not None
            and len(position) >= 2
        ):
            cell = (round(float(position[0]) / 128.0), round(float(position[1]) / 128.0))
            slot = hash(cell) % (self._CELL_BITMAP_SIZE * 8)
            byte_index, bit_index = divmod(slot, 8)
            if not self._cells[byte_index] & (1 << bit_index):
                self._cells[byte_index] |= 1 << bit_index
                self._occupied_cells += 1
        distance = record.get("engagement_distance")
        if distance is not None:
            self._distance_sum += float(distance)
            self._distance_count += 1
        if record.get("utility") not in (None, "", 0, False):
            self._utility_events += 1
        if float(record.get("buy_action", 0)) > 0.0:
            self._economy_choices += 1
        self.count += 1
        self._elapsed_s += duration_s
        self._last_time = time_s
        self._previous_yaw = yaw

    def finish(self) -> HumanLikeMetricsV1:
        self._finish_episode()
        survival = self._episode_survival_s
        duration = max(survival, self._elapsed_s, 1e-6)
        return HumanLikeMetricsV1(
            max_angular_velocity_deg_s=self._max_velocity,
            max_angular_acceleration_deg_s2=self._max_acceleration,
            stop_go_ratio=self._stop_go_changes / max(1, self.count - 1),
            space_occupancy=self._occupied_cells / max(1, self.count),
            engagement_distance=(
                self._distance_sum / self._distance_count
                if self._distance_count
                else 0.0
            ),
            utility_event_rate=self._utility_events / duration,
            economy_choice_count=self._economy_choices,
            survival_time_s=survival,
        )


class _HumanMetricQuantiles:
    _CAPACITY = 2048
    _FIELDS = (
        "max_angular_velocity_deg_s",
        "max_angular_acceleration_deg_s2",
        "stop_go_ratio",
        "space_occupancy",
        "engagement_distance",
        "utility_event_rate",
        "economy_choice_count",
        "survival_time_s",
    )

    def __init__(self) -> None:
        self._seen = 0
        self._reservoirs = {name: [] for name in self._FIELDS}
        self._random = random.Random(7)

    @property
    def sample_count(self) -> int:
        return self._seen

    @property
    def retained_count(self) -> int:
        return min(self._seen, self._CAPACITY)

    def add(self, metrics: HumanLikeMetricsV1) -> None:
        values = asdict(metrics)
        self._seen += 1
        for name in self._FIELDS:
            reservoir = self._reservoirs[name]
            value = float(values[name])
            if len(reservoir) < self._CAPACITY:
                reservoir.append(value)
                continue
            slot = self._random.randrange(self._seen)
            if slot < self._CAPACITY:
                reservoir[slot] = value

    def quantiles(self) -> dict[str, dict[str, float]]:
        return {
            name: _metric_quantiles(values)
            for name, values in self._reservoirs.items()
        }


def _stream_human_metrics(
    records: Iterable[Mapping[str, Any]],
) -> tuple[HumanLikeMetricsV1, int, _HumanMetricQuantiles]:
    aggregate = _HumanMetricsAccumulator()
    group = _HumanMetricsAccumulator()
    current_episode: str | None = None
    group_quantiles = _HumanMetricQuantiles()
    count = 0
    for record in records:
        episode_id = str(record.get("episode_id", "all"))
        if current_episode is None:
            current_episode = episode_id
        elif episode_id != current_episode:
            group_quantiles.add(group.finish())
            group = _HumanMetricsAccumulator()
            aggregate.begin_episode()
            current_episode = episode_id
        aggregate.add(record)
        group.add(record)
        count += 1
    if current_episode is not None:
        group_quantiles.add(group.finish())
    return aggregate.finish(), count, group_quantiles


def compute_human_like_metrics(records: Iterable[Mapping[str, Any]]) -> HumanLikeMetricsV1:
    metrics, _, _ = _stream_human_metrics(records)
    return metrics


@dataclass(frozen=True)
class BootstrapMatchV1:
    instance_id: str
    match_id: str
    policy_generation: int
    ruleset: str
    terminal: bool
    duration_s: float
    purpose: DataPurpose | str = DataPurpose.PRODUCTION

    def __post_init__(self) -> None:
        instance_id = str(self.instance_id)
        match_id = str(self.match_id)
        ruleset = str(self.ruleset).lower()
        if not instance_id or not match_id:
            raise ValueError("bootstrap match instance_id and match_id cannot be empty")
        if ruleset != "mr12":
            raise ValueError("bootstrap matches must use the mr12 ruleset")
        if self.policy_generation != 0:
            raise ValueError("bootstrap matches must use frozen policy generation 0")
        if not isinstance(self.terminal, bool) or not self.terminal:
            raise ValueError("bootstrap matches must terminate normally")
        duration = float(self.duration_s)
        if not math.isfinite(duration) or not 0.0 <= duration <= 3600.0:
            raise ValueError("bootstrap match duration must be between 0 and 3600 seconds")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "match_id", match_id)
        object.__setattr__(self, "ruleset", ruleset)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "purpose", ensure_purpose(self.purpose))


def build_bootstrap_manifest(
    matches: Sequence[BootstrapMatchV1],
    selected_server_count: int,
    *,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    parent_manifests: Sequence[DatasetManifestV1] = (),
    name: str = "mr12-bootstrap",
) -> DatasetManifestV1:
    requested = ensure_purpose(purpose)
    normalized = tuple(matches)
    if not 1 <= selected_server_count <= 4:
        raise ValueError("selected_server_count must be between 1 and 4")
    if len(normalized) != selected_server_count:
        raise ValueError("bootstrap match count must equal selected_server_count")
    if any(not isinstance(match, BootstrapMatchV1) for match in normalized):
        raise TypeError("matches must contain BootstrapMatchV1 values")
    if len({match.instance_id for match in normalized}) != len(normalized):
        raise ValueError("bootstrap instance IDs must be unique")
    if len({match.match_id for match in normalized}) != len(normalized):
        raise ValueError("bootstrap match IDs must be unique")
    parents = tuple(parent_manifests)
    if any(not isinstance(parent, DatasetManifestV1) for parent in parents):
        raise TypeError("bootstrap parents must contain DatasetManifestV1 values")
    if requested is DataPurpose.PRODUCTION:
        if any(match.purpose is DataPurpose.TEST_ONLY for match in normalized):
            raise ProductionLineageError("test-only bootstrap match cannot enter production")
        for parent in parents:
            parent.assert_exportable(DataPurpose.PRODUCTION)
    canonical = "|".join(
        f"{match.instance_id}:{match.match_id}:{match.policy_generation}:{match.duration_s:.9f}"
        for match in normalized
    )
    source_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return DatasetManifestV1(
        name=name,
        purpose=requested,
        parents=parents,
        artifact_type="mr12_bootstrap",
        source_sha256=source_sha256,
        metadata={
            "ruleset": "mr12",
            "bootstrap": "true",
            "selected_server_count": str(selected_server_count),
            "instance_ids": ",".join(match.instance_id for match in normalized),
            "match_ids": ",".join(match.match_id for match in normalized),
            "policy_generation": "0",
            "match_duration_limit_s": "3600",
            "max_match_duration_s": f"{max(match.duration_s for match in normalized):.6f}",
        },
    )


def _validate_corpus_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("corpus_sha256 must be a 64-character hexadecimal digest")
    return normalized


def _validate_human_manifest(
    manifest: DatasetManifestV1,
    *,
    split: str,
    purpose: DataPurpose,
) -> None:
    if not isinstance(manifest, DatasetManifestV1):
        raise TypeError("human-like target manifests must be DatasetManifestV1 values")
    ruleset = str(manifest.metadata.get("ruleset", "")).lower()
    if ruleset != "mr15":
        raise ProductionLineageError("human-like targets must be derived from an MR15 manifest")
    declared_split = str(manifest.metadata.get("split", "")).lower()
    if declared_split and declared_split != split:
        raise ValueError(f"human-like target manifest split must be {split}")
    if manifest.effective_purpose() is not purpose:
        raise ProductionLineageError("human-like target manifest purpose does not match the target purpose")
    if purpose is DataPurpose.PRODUCTION:
        manifest.assert_exportable(DataPurpose.PRODUCTION)


def _metric_quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(float(value) for value in values)

    def quantile(level: float) -> float:
        position = (len(ordered) - 1) * level
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return {f"p{int(level * 100)}": quantile(level) for level in (0.50, 0.95, 0.99)}


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def compute_human_like_targets(
    train_records: Iterable[Mapping[str, Any]],
    validation_records: Iterable[Mapping[str, Any]],
    *,
    corpus_sha256: str,
    train_manifest: DatasetManifestV1 | None = None,
    validation_manifest: DatasetManifestV1 | None = None,
    purpose: DataPurpose | str | None = None,
    output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    corpus_sha256 = _validate_corpus_sha256(corpus_sha256)
    manifests = tuple(
        manifest
        for manifest in (train_manifest, validation_manifest)
        if manifest is not None
    )
    if purpose is None:
        requested = (
            DataPurpose.PRODUCTION
            if manifests and all(manifest.effective_purpose() is DataPurpose.PRODUCTION for manifest in manifests)
            else DataPurpose.TEST_ONLY
        )
    else:
        requested = ensure_purpose(purpose)
    if train_manifest is not None:
        _validate_human_manifest(train_manifest, split="train", purpose=requested)
    if validation_manifest is not None:
        _validate_human_manifest(validation_manifest, split="validation", purpose=requested)
    train_metrics, train_count, train_group_quantiles = _stream_human_metrics(train_records)
    validation_metrics, validation_count, validation_group_quantiles = _stream_human_metrics(
        validation_records
    )
    if not train_count and not validation_count:
        raise ValueError("human-like target records cannot both be empty")
    unique_parents: list[DatasetManifestV1] = []
    seen_parent_ids: set[int] = set()
    for manifest in manifests:
        if id(manifest) not in seen_parent_ids:
            unique_parents.append(manifest)
            seen_parent_ids.add(id(manifest))
    target_manifest = DatasetManifestV1(
        name="human-like-targets",
        purpose=requested,
        parents=tuple(unique_parents),
        artifact_type="human_like_targets",
        source_sha256=corpus_sha256,
        metadata={
            "ruleset": "mr15",
            "corpus_sha256": corpus_sha256,
            "train_sample_count": str(train_count),
            "validation_sample_count": str(validation_count),
        },
    )
    if requested is DataPurpose.PRODUCTION:
        target_manifest.assert_exportable(DataPurpose.PRODUCTION)

    split_payload: dict[str, Any] = {}
    quantiles: dict[str, Any] = {}
    for split, metrics, group_quantiles in (
        ("train", train_metrics, train_group_quantiles),
        ("validation", validation_metrics, validation_group_quantiles),
    ):
        split_payload[split] = asdict(metrics)
        quantiles[split] = group_quantiles.quantiles()
    train_metrics = split_payload["train"]
    targets = {
        name: value
        for name, value in train_metrics.items()
        if name in _HUMAN_TARGET_NAMES
    }
    payload: dict[str, Any] = {
        "schema": "human-like-targets-v1",
        "purpose": requested.value,
        "dataset_manifest": _manifest_payload(target_manifest),
        "corpus_sha256": corpus_sha256,
        "sample_counts": {"train": train_count, "validation": validation_count},
        "metrics": split_payload,
        "quantiles": quantiles,
        "targets": targets,
    }
    if output_path is not None:
        _atomic_json_write(Path(output_path), payload)
    return payload
