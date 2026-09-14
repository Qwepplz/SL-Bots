from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BotActionV1, DataPurpose, Phase, ensure_purpose
from .lineage import DatasetManifestV1, ProductionLineageError


class PhaseMixError(ValueError):
    pass


@dataclass(frozen=True)
class WindowSampleV1:
    phase: Phase
    observations: tuple[bytes, ...]
    actions: tuple[BotActionV1, ...]
    source: str
    episode_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase(self.phase))
        observations = tuple(bytes(value) for value in self.observations)
        actions = tuple(self.actions)
        if len(observations) != len(actions):
            raise ValueError("window observations and actions must have equal lengths")
        if any(len(value) != 256 for value in observations):
            raise ValueError("window observations must be 256 bytes")
        if self.source not in {"demo", "selfplay"}:
            raise ValueError("window source must be demo or selfplay")
        if not self.episode_id:
            raise ValueError("window episode_id cannot be empty")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)


def _validate_window_phases(windows: Sequence[WindowSampleV1], phase: Phase) -> tuple[WindowSampleV1, ...]:
    normalized = tuple(windows)
    if any(not isinstance(window, WindowSampleV1) for window in normalized):
        raise TypeError("windows must contain WindowSampleV1 values")
    if any(window.phase is not phase for window in normalized):
        raise PhaseMixError(f"window batch contains a phase other than {phase.name.lower()}")
    return normalized


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
        def __init__(self, hidden_size: int = 128) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(261, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, observations: Any, actions: Any) -> Any:
            if observations.ndim != 3 or observations.shape[-1] != 256:
                raise ValueError("discriminator observations must have shape [batch, time, 256]")
            if actions.ndim != 3 or actions.shape[-1] != 5:
                raise ValueError("discriminator actions must have shape [batch, time, 5]")
            features = torch.cat([observations.float().mean(dim=1), actions.float().mean(dim=1)], dim=-1)
            return self.network(features).squeeze(-1)

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

    def assert_exportable(self, target_purpose: DataPurpose | str = DataPurpose.PRODUCTION) -> None:
        target = ensure_purpose(target_purpose)
        self.dataset_manifest.assert_exportable(target)
        if target is DataPurpose.PRODUCTION and self.purpose is DataPurpose.TEST_ONLY:
            raise ProductionLineageError("test-only discriminator cannot be exported as production")


def train_discriminator(
    demo_windows: Sequence[WindowSampleV1],
    selfplay_windows: Sequence[WindowSampleV1],
    *,
    phase: Phase | int,
    output_dir: str | Path,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    steps: int = 1,
    batch_size: int = 10,
    demo_ratio: float = 0.5,
    seed: int = 7,
    parent_manifests: Sequence[DatasetManifestV1] = (),
) -> DiscriminatorRunManifestV1:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for discriminator training")
    if steps <= 0:
        raise ValueError("steps must be positive")
    phase = Phase(phase)
    run_purpose = ensure_purpose(purpose)
    windows = sample_rehearsal_windows(
        demo_windows,
        selfplay_windows,
        phase=phase,
        batch_size=batch_size,
        demo_ratio=demo_ratio,
        seed=seed,
    )
    model = WindowDiscriminator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    observations = torch.tensor(
        [[[byte for byte in observation] for observation in window.observations] for window in windows],
        dtype=torch.float32,
    ) / 255.0
    actions = torch.tensor(
        [
            [
                [action.forward, action.side, action.up, action.yaw_delta_deg, action.pitch_delta_deg]
                for action in window.actions
            ]
            for window in windows
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1.0 if window.source == "demo" else 0.0 for window in windows])
    loss_history: list[float] = []
    torch.manual_seed(seed)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(observations, actions), labels)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "discriminator.pt"
    dataset = DatasetManifestV1(
        name=f"gail:{output.name}",
        purpose=run_purpose,
        parents=tuple(parent_manifests),
        artifact_type="gail_windows",
        metadata={"phase": phase.name.lower(), "demo_ratio": str(demo_ratio)},
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"hidden_size": 128},
            "phase": phase.name.lower(),
            "loss_history": loss_history,
            "dataset_manifest": dataset,
        },
        checkpoint,
    )
    config_sha256 = hashlib.sha256(
        f"{phase.name.lower()}|{steps}|{batch_size}|{demo_ratio}|{seed}".encode()
    ).hexdigest()
    return DiscriminatorRunManifestV1(
        name=output.name,
        purpose=run_purpose,
        dataset_manifest=dataset,
        steps=steps,
        loss_history=tuple(loss_history),
        checkpoint_path=str(checkpoint),
        config_sha256=config_sha256,
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


def compute_human_like_metrics(records: Sequence[Mapping[str, Any]]) -> HumanLikeMetricsV1:
    if not records:
        return HumanLikeMetricsV1(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    times = [float(record.get("time_s", index / 128.0)) for index, record in enumerate(records)]
    yaws = [float(record.get("yaw_deg", 0.0)) for record in records]
    angular_velocities: list[float] = []
    for index in range(1, len(records)):
        delta_t = max(times[index] - times[index - 1], 1e-6)
        delta_yaw = (yaws[index] - yaws[index - 1] + 180.0) % 360.0 - 180.0
        angular_velocities.append(delta_yaw / delta_t)
    angular_accelerations = [
        (angular_velocities[index] - angular_velocities[index - 1]) /
        max(times[index + 1] - times[index], 1e-6)
        for index in range(1, len(angular_velocities))
    ]
    movement = [abs(float(record.get("forward", 0.0))) > 0.1 for record in records]
    stop_go_changes = sum(left != right for left, right in zip(movement, movement[1:]))
    stop_go_ratio = stop_go_changes / max(1, len(movement) - 1)
    cells = set()
    for record in records:
        position = record.get("position")
        if record.get("position_valid", position is not None) and position is not None and len(position) >= 2:
            cells.add((round(float(position[0]) / 128.0), round(float(position[1]) / 128.0)))
    space_occupancy = len(cells) / max(1, len(records))
    distances = [
        float(record["engagement_distance"])
        for record in records
        if record.get("engagement_distance") is not None
    ]
    duration = max(times[-1] - times[0], 1e-6)
    utility_events = sum(record.get("utility") not in (None, "", 0) for record in records)
    economy_choices = sum(float(record.get("buy_action", 0)) > 0 for record in records)
    return HumanLikeMetricsV1(
        max_angular_velocity_deg_s=max((abs(value) for value in angular_velocities), default=0.0),
        max_angular_acceleration_deg_s2=max((abs(value) for value in angular_accelerations), default=0.0),
        stop_go_ratio=stop_go_ratio,
        space_occupancy=space_occupancy,
        engagement_distance=sum(distances) / len(distances) if distances else 0.0,
        utility_event_rate=utility_events / duration,
        economy_choice_count=economy_choices,
        survival_time_s=max(times) - min(times),
    )
