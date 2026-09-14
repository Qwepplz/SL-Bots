from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BotActionV1, DataPurpose, Phase, ensure_purpose
from .lineage import DatasetManifestV1, ProductionLineageError
from .selfplay import CriticSnapshotV1, TransitionBatchV1


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(values) != len(rewards) + 1 or len(dones) != len(rewards):
        raise ValueError("values must contain one bootstrap value more than rewards")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between 0 and 1")
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[index] else 1.0
        delta = float(rewards[index]) + gamma * float(values[index + 1]) * continuation - float(values[index])
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    returns = tuple(advantages[index] + float(values[index]) for index in range(len(advantages)))
    return tuple(advantages), returns


def ppo_clip_objective(
    new_log_prob: Any,
    old_log_prob: Any,
    advantages: Any,
    *,
    clip_ratio: float = 0.2,
) -> tuple[Any, Any]:
    """计算 PPO-Clip actor loss 及被裁剪样本比例。"""
    if not 0.0 < clip_ratio < 1.0:
        raise ValueError("clip_ratio must be between 0 and 1")
    if getattr(new_log_prob, "shape", None) != getattr(old_log_prob, "shape", None):
        raise ValueError("new_log_prob and old_log_prob must have equal shapes")
    if getattr(new_log_prob, "shape", None) != getattr(advantages, "shape", None):
        raise ValueError("advantages must align with log probabilities")
    ratio = (new_log_prob - old_log_prob).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    objective = torch.minimum(unclipped, clipped)
    clip_fraction = ((ratio < 1.0 - clip_ratio) | (ratio > 1.0 + clip_ratio)).float().mean()
    return -objective.mean(), clip_fraction


def truncate_rollout(
    transitions: Sequence[TransitionBatchV1],
    *,
    horizon: int,
) -> tuple[tuple[TransitionBatchV1, ...], ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    normalized = tuple(transitions)
    if not normalized:
        return ()
    phase = normalized[0].phase
    if any(transition.phase is not phase for transition in normalized):
        raise ValueError("a recurrent rollout cannot mix Get5 phases")
    return tuple(
        normalized[start : start + horizon]
        for start in range(0, len(normalized), horizon)
    )


@dataclass(frozen=True)
class MappoBatchV1:
    phase: Phase
    bot_ids: tuple[str, ...]
    observations: tuple[bytes, ...]
    actions: tuple[BotActionV1, ...]
    rewards: tuple[float, ...]
    dones: tuple[bool, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]
    hidden_state_mask: tuple[bool, ...]
    critic_snapshots: tuple[CriticSnapshotV1, ...]

    def __post_init__(self) -> None:
        phase = Phase(self.phase)
        object.__setattr__(self, "phase", phase)
        observations = tuple(bytes(value) for value in self.observations)
        actions = tuple(self.actions)
        rewards = tuple(float(value) for value in self.rewards)
        dones = tuple(bool(value) for value in self.dones)
        advantages = tuple(float(value) for value in self.advantages)
        returns = tuple(float(value) for value in self.returns)
        hidden_state_mask = tuple(bool(value) for value in self.hidden_state_mask)
        size = len(observations)
        if any(len(value) != 256 for value in observations):
            raise ValueError("Mappo observations must be 256 bytes")
        if not all(len(values) == size for values in (actions, rewards, dones, advantages, returns, hidden_state_mask)):
            raise ValueError("Mappo batch fields must have equal lengths")
        if len(self.bot_ids) != size:
            raise ValueError("Mappo bot_ids must align with flattened observations")
        object.__setattr__(self, "bot_ids", tuple(self.bot_ids))
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "dones", dones)
        object.__setattr__(self, "advantages", advantages)
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "hidden_state_mask", hidden_state_mask)
        object.__setattr__(self, "critic_snapshots", tuple(self.critic_snapshots))


def _snapshot_value(snapshot: CriticSnapshotV1 | None, bot_id: str) -> float:
    if snapshot is None:
        return 0.0
    value = snapshot.values.get(bot_id, snapshot.values.get("value", 0.0))
    if isinstance(value, Mapping):
        value = value.get("value", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def prepare_mappo_batch(
    transitions: Sequence[TransitionBatchV1],
    *,
    critic_snapshots: Sequence[CriticSnapshotV1] = (),
    values: Mapping[str, Sequence[float]] | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> MappoBatchV1:
    normalized = tuple(transitions)
    if not normalized:
        raise ValueError("transitions cannot be empty")
    phase = normalized[0].phase
    if any(transition.phase is not phase for transition in normalized):
        raise ValueError("Mappo batch cannot mix phases")
    snapshots = tuple(critic_snapshots)
    first_ids = normalized[0].bot_ids
    if any(transition.bot_ids != first_ids for transition in normalized):
        raise ValueError("Mappo recurrent batch requires stable bot ids")
    advantages_by_bot: dict[str, tuple[float, ...]] = {}
    returns_by_bot: dict[str, tuple[float, ...]] = {}
    for bot_index, bot_id in enumerate(first_ids):
        bot_rewards = [transition.rewards[bot_index] for transition in normalized]
        bot_dones = [transition.dones[bot_index] for transition in normalized]
        if values is not None:
            bot_values = list(values.get(bot_id, ()))
            if len(bot_values) not in {len(normalized), len(normalized) + 1}:
                raise ValueError("critic values must have one value per step plus an optional bootstrap")
            if len(bot_values) == len(normalized):
                bot_values.append(0.0)
        else:
            bot_values = [
                _snapshot_value(snapshots[index] if index < len(snapshots) else None, bot_id)
                for index in range(len(normalized) + 1)
            ]
        advantage, returns = compute_gae(
            bot_rewards,
            bot_values,
            bot_dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages_by_bot[bot_id] = advantage
        returns_by_bot[bot_id] = returns
    flat_bot_ids: list[str] = []
    flat_observations: list[bytes] = []
    flat_actions: list[BotActionV1] = []
    flat_rewards: list[float] = []
    flat_dones: list[bool] = []
    flat_advantages: list[float] = []
    flat_returns: list[float] = []
    flat_masks: list[bool] = []
    for time_index, transition in enumerate(normalized):
        for bot_index, bot_id in enumerate(transition.bot_ids):
            flat_bot_ids.append(bot_id)
            flat_observations.append(transition.observations[bot_index])
            flat_actions.append(transition.actions[bot_index])
            flat_rewards.append(transition.rewards[bot_index])
            flat_dones.append(transition.dones[bot_index])
            flat_advantages.append(advantages_by_bot[bot_id][time_index])
            flat_returns.append(returns_by_bot[bot_id][time_index])
            flat_masks.append(transition.hidden_state_mask[bot_index])
    return MappoBatchV1(
        phase=phase,
        bot_ids=tuple(flat_bot_ids),
        observations=tuple(flat_observations),
        actions=tuple(flat_actions),
        rewards=tuple(flat_rewards),
        dones=tuple(flat_dones),
        advantages=tuple(flat_advantages),
        returns=tuple(flat_returns),
        hidden_state_mask=tuple(flat_masks),
        critic_snapshots=snapshots,
    )


try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:

    class CentralValueCritic(nn.Module):
        def __init__(self, feature_size: int = 64, hidden_size: int = 128) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, features: Any) -> Any:
            return self.network(features.float()).squeeze(-1)

else:

    class CentralValueCritic:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch 2.9 is required for CentralValueCritic")


_ACTION_MASK_BITS = (1, 2, 4, 8, 16, 32, 64, 128)


def _action_tensors(actions: Sequence[BotActionV1], device: Any) -> tuple[Any, Any, Any, Any, Any]:
    continuous = torch.tensor(
        [
            [
                action.forward,
                action.side,
                action.up,
                action.yaw_delta_deg,
                action.pitch_delta_deg,
            ]
            for action in actions
        ],
        dtype=torch.float32,
        device=device,
    )
    button_bits = torch.tensor(
        [
            [(int(action.buttons) >> bit) & 1 for bit in range(32)]
            for action in actions
        ],
        dtype=torch.float32,
        device=device,
    )
    weapon_indices = torch.tensor(
        [max(0, int(action.weapon_select)) for action in actions],
        dtype=torch.long,
        device=device,
    )
    buy_indices = torch.tensor(
        [max(0, int(action.buy_action)) for action in actions],
        dtype=torch.long,
        device=device,
    )
    masks = torch.tensor(
        [[1.0 if int(action.action_valid_mask) & bit else 0.0 for bit in _ACTION_MASK_BITS] for action in actions],
        dtype=torch.float32,
        device=device,
    )
    return continuous, button_bits, weapon_indices, buy_indices, masks


def _mouse_log_prob(output: Any, mouse: Any) -> Any:
    component_offsets = torch.tensor((-1.0, 0.0, 1.0), dtype=mouse.dtype, device=mouse.device).reshape(1, 1, 3)
    component_loc = output.mouse_loc.unsqueeze(-1) + component_offsets * output.mouse_scale.unsqueeze(-1)
    component_scale = output.mouse_scale.unsqueeze(-1)
    standardized = (mouse.unsqueeze(-1) - component_loc) / component_scale
    component_log_prob = (
        -standardized
        - 2.0 * torch.nn.functional.softplus(-standardized)
        - component_scale.log()
    )
    mixture_log_prob = torch.log_softmax(output.mouse_mix_logits, dim=-1)
    return torch.logsumexp(mixture_log_prob + component_log_prob, dim=-1)


def _policy_log_prob(output: Any, actions: Sequence[BotActionV1]) -> Any:
    continuous, button_bits, weapon_indices, buy_indices, masks = _action_tensors(
        actions,
        output.action_vector.device,
    )
    movement_value = ((continuous[:, :3] + 1.0) / 2.0).clamp(1e-5, 1.0 - 1e-5)
    movement_distribution = torch.distributions.Beta(output.movement_alpha, output.movement_beta)
    movement_log_prob = movement_distribution.log_prob(movement_value) - math.log(2.0)
    mouse_log_prob = _mouse_log_prob(output, continuous[:, 3:5])
    button_log_prob = torch.distributions.Bernoulli(logits=output.button_logits).log_prob(button_bits).sum(dim=-1)
    weapon_indices = weapon_indices.clamp_max(output.weapon_logits.shape[-1] - 1)
    buy_indices = buy_indices.clamp_max(output.buy_logits.shape[-1] - 1)
    weapon_log_prob = torch.distributions.Categorical(logits=output.weapon_logits).log_prob(weapon_indices)
    buy_log_prob = torch.distributions.Categorical(logits=output.buy_logits).log_prob(buy_indices)
    return (
        (movement_log_prob * masks[:, :3]).sum(dim=-1)
        + (mouse_log_prob * masks[:, 3:5]).sum(dim=-1)
        + button_log_prob * masks[:, 5]
        + weapon_log_prob * masks[:, 6]
        + buy_log_prob * masks[:, 7]
    )


def _policy_entropy(output: Any) -> Any:
    movement_entropy = torch.distributions.Beta(output.movement_alpha, output.movement_beta).entropy().sum(dim=-1)
    movement_entropy = movement_entropy - 3.0 * math.log(2.0)
    mouse_entropy = (output.mouse_scale.log() + 2.0).sum(dim=-1)
    mix_prob = torch.softmax(output.mouse_mix_logits, dim=-1)
    mix_entropy = -(mix_prob * torch.log_softmax(output.mouse_mix_logits, dim=-1)).sum(dim=-1).sum(dim=-1)
    button_entropy = torch.distributions.Bernoulli(logits=output.button_logits).entropy().sum(dim=-1)
    weapon_entropy = torch.distributions.Categorical(logits=output.weapon_logits).entropy()
    buy_entropy = torch.distributions.Categorical(logits=output.buy_logits).entropy()
    return movement_entropy + mouse_entropy + mix_entropy + button_entropy + weapon_entropy + buy_entropy


def _bernoulli_kl(current_logits: Any, reference_logits: Any) -> Any:
    current = torch.sigmoid(current_logits).clamp(1e-6, 1.0 - 1e-6)
    reference = torch.sigmoid(reference_logits).clamp(1e-6, 1.0 - 1e-6)
    return (
        current * (current.log() - reference.log())
        + (1.0 - current) * ((1.0 - current).log() - (1.0 - reference).log())
    ).sum(dim=-1)


def _policy_kl(current: Any, reference: Any) -> Any:
    movement = torch.distributions.kl_divergence(
        torch.distributions.Beta(current.movement_alpha, current.movement_beta),
        torch.distributions.Beta(reference.movement_alpha, reference.movement_beta),
    ).sum(dim=-1)
    mouse = torch.distributions.kl_divergence(
        torch.distributions.Normal(current.mouse_loc, current.mouse_scale),
        torch.distributions.Normal(reference.mouse_loc, reference.mouse_scale),
    ).sum(dim=-1)
    current_mix = torch.log_softmax(current.mouse_mix_logits, dim=-1)
    reference_mix = torch.log_softmax(reference.mouse_mix_logits, dim=-1)
    mix_prob = current_mix.exp()
    mouse = mouse + (mix_prob * (current_mix - reference_mix)).sum(dim=-1).sum(dim=-1)
    return (
        movement
        + mouse
        + _bernoulli_kl(current.button_logits, reference.button_logits)
        + torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=current.weapon_logits),
            torch.distributions.Categorical(logits=reference.weapon_logits),
        )
        + torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=current.buy_logits),
            torch.distributions.Categorical(logits=reference.buy_logits),
        )
    )


def _reset_recurrent_state(state: Any, keep: Any) -> Any:
    keep = keep.to(state.tactical.device)
    return type(state)(
        tactical=torch.where(keep.unsqueeze(1), state.tactical, torch.zeros_like(state.tactical)),
        action=torch.where(keep.unsqueeze(1), state.action, torch.zeros_like(state.action)),
        tick=torch.where(keep, state.tick, torch.zeros_like(state.tick)),
    )


def _unroll_actor(actor: Any, observations: Any, hidden_state_mask: Any) -> list[Any]:
    time_steps, bot_count, _ = observations.shape
    state = actor.initial_state(bot_count, device=observations.device)
    outputs: list[Any] = []
    for time_index in range(time_steps):
        state = _reset_recurrent_state(state, hidden_state_mask[time_index])
        output = actor(observations[time_index], state)
        outputs.append(output)
        state = output.recurrent_state
    return outputs


def _append_numeric(value: Any, output: list[float]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _append_numeric(value[key], output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _append_numeric(item, output)
    elif isinstance(value, bool):
        output.append(float(value))
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        output.append(float(value))


def _critic_features(
    snapshots: Sequence[CriticSnapshotV1],
    bot_ids: Sequence[str],
    time_steps: int,
    bot_count: int,
    device: Any,
) -> Any:
    features = torch.zeros(time_steps, bot_count, 64, dtype=torch.float32, device=device)
    for time_index in range(time_steps):
        snapshot = snapshots[time_index] if time_index < len(snapshots) else None
        numeric: list[float] = []
        if snapshot is not None:
            _append_numeric(snapshot.values, numeric)
        for bot_index in range(bot_count):
            bot_id = bot_ids[time_index * bot_count + bot_index]
            features[time_index, bot_index, 0] = _snapshot_value(snapshot, bot_id)
            if numeric:
                values = torch.tensor(numeric[:63], dtype=torch.float32, device=device)
                features[time_index, bot_index, 1 : 1 + len(values)] = values
    return features


def _discriminator_reward(discriminator: Any, observations: Any, predicted_actions: Any) -> Any:
    logits = discriminator(observations.permute(1, 0, 2), predicted_actions.permute(1, 0, 2))
    logits = logits.reshape(-1)
    if logits.shape[0] != observations.shape[1]:
        raise ValueError("GAIL discriminator must return one logit per bot")
    return -torch.nn.functional.logsigmoid(-logits)


def _human_target(targets: Mapping[str, float] | None, name: str) -> float | None:
    if targets is None or name not in targets:
        return None
    value = float(targets[name])
    if not math.isfinite(value):
        raise ValueError(f"human_like_targets[{name!r}] must be finite")
    return value


def _human_like_penalty(predicted_actions: Any, outputs: Sequence[Any], targets: Mapping[str, float] | None) -> Any:
    penalty = predicted_actions.sum() * 0.0
    if not targets:
        return penalty
    delta_time = 1.0 / 128.0
    max_velocity = _human_target(targets, "max_angular_velocity_deg_s")
    if max_velocity is not None:
        velocity = predicted_actions[:, :, 3].abs() / delta_time
        penalty = penalty + torch.relu(velocity - max_velocity).mean()
    stop_go_target = _human_target(targets, "stop_go_ratio")
    if stop_go_target is not None and predicted_actions.shape[0] > 1:
        movement = torch.sigmoid((predicted_actions[:, :, 0].abs() - 0.1) * 8.0)
        stop_go = (movement[1:] - movement[:-1]).abs().mean()
        penalty = penalty + torch.relu(stop_go - stop_go_target)
    acceleration_target = _human_target(targets, "max_angular_acceleration_deg_s2")
    if acceleration_target is not None and predicted_actions.shape[0] > 2:
        velocity = predicted_actions[:, :, 3] / delta_time
        acceleration = (velocity[1:] - velocity[:-1]) / delta_time
        penalty = penalty + torch.relu(acceleration.abs() - acceleration_target).mean()
    economy_target = _human_target(targets, "economy_choice_count")
    if economy_target is not None:
        buy_probability = torch.stack([torch.softmax(output.buy_logits, dim=-1)[:, 1:].sum(dim=-1) for output in outputs])
        penalty = penalty + torch.relu(buy_probability.sum() - economy_target)
    utility_target = _human_target(targets, "utility_event_rate")
    if utility_target is not None:
        button_probability = torch.stack([torch.sigmoid(output.button_logits)[:, 4:8].mean(dim=-1) for output in outputs])
        penalty = penalty + torch.relu(button_probability.mean() / delta_time - utility_target)
    return penalty


def _human_like_records(
    predicted_actions: Any,
    outputs: Sequence[Any],
    observations: Sequence[bytes] | None = None,
) -> list[Mapping[str, Any]]:
    from .observation_projection import PLAYER_AUDIBLE, PLAYER_DIRECT, PLAYER_RADAR, decode_observation

    records: list[Mapping[str, Any]] = []
    yaw = torch.zeros(predicted_actions.shape[1], dtype=predicted_actions.dtype, device=predicted_actions.device)
    for time_index, output in enumerate(outputs):
        yaw = yaw + predicted_actions[time_index, :, 3]
        buy = torch.argmax(output.buy_logits, dim=-1)
        decoded = None
        if observations is not None and time_index < len(observations):
            try:
                decoded = decode_observation(observations[time_index])
            except (TypeError, ValueError, struct.error):
                decoded = None
        engagement_distance = None
        utility = None
        position = None
        if decoded is not None:
            position = decoded.self_position[:2]
            visible_enemies = [
                player.distance
                for player in decoded.players
                if player.relation == -1
                and player.distance > 0.0
                and player.flags & (PLAYER_DIRECT | PLAYER_RADAR | PLAYER_AUDIBLE)
            ]
            if visible_enemies:
                engagement_distance = min(visible_enemies)
            utility_events = {
                "grenade_projectile_throw",
                "smoke_start",
                "flash_explode",
                "fire_grenade_start",
                "he_explode",
                "decoy_start",
            }
            if any(event.category in utility_events for event in decoded.events):
                utility = True
        for bot_index in range(predicted_actions.shape[1]):
            record: dict[str, Any] = {
                "time_s": time_index / 128.0,
                "yaw_deg": float(yaw[bot_index].detach().cpu()),
                "forward": float(predicted_actions[time_index, bot_index, 0].detach().cpu()),
                "buy_action": int(buy[bot_index].detach().cpu()),
                "position_valid": position is not None,
            }
            if position is not None:
                record["position"] = position
            if engagement_distance is not None:
                record["engagement_distance"] = engagement_distance
            if utility is not None:
                record["utility"] = utility
            records.append(record)
    return records


def _human_like_metric_values(
    predicted_actions: Any,
    outputs: Sequence[Any],
    observations: Sequence[bytes] | None = None,
) -> dict[str, float]:
    from .training_gail import compute_human_like_metrics

    metrics = []
    for bot_index in range(predicted_actions.shape[1]):
        bot_actions = predicted_actions[:, bot_index : bot_index + 1, :]
        bot_outputs = [
            type(output)(
                movement_alpha=output.movement_alpha[bot_index : bot_index + 1],
                movement_beta=output.movement_beta[bot_index : bot_index + 1],
                mouse_loc=output.mouse_loc[bot_index : bot_index + 1],
                mouse_scale=output.mouse_scale[bot_index : bot_index + 1],
                mouse_mix_logits=output.mouse_mix_logits[bot_index : bot_index + 1],
                button_logits=output.button_logits[bot_index : bot_index + 1],
                weapon_logits=output.weapon_logits[bot_index : bot_index + 1],
                buy_logits=output.buy_logits[bot_index : bot_index + 1],
                recurrent_state=output.recurrent_state,
                condition_embedding=output.condition_embedding[bot_index : bot_index + 1],
                action_vector=output.action_vector[bot_index : bot_index + 1],
            )
            for output in outputs
        ]
        bot_observations = None
        if observations is not None:
            bot_observations = [observations[time_index * predicted_actions.shape[1] + bot_index] for time_index in range(predicted_actions.shape[0])]
        metrics.append(compute_human_like_metrics(_human_like_records(bot_actions, bot_outputs, bot_observations)))
    if not metrics:
        return {}
    fields = (
        "max_angular_velocity_deg_s",
        "max_angular_acceleration_deg_s2",
        "stop_go_ratio",
        "space_occupancy",
        "engagement_distance",
        "utility_event_rate",
        "economy_choice_count",
        "survival_time_s",
    )
    return {
        f"human_{name}": sum(float(getattr(metric, name)) for metric in metrics) / len(metrics)
        for name in fields
    }


@dataclass(frozen=True)
class MappoRunManifestV1:
    name: str
    purpose: DataPurpose
    dataset_manifest: DatasetManifestV1
    steps: int
    loss_history: tuple[float, ...]
    checkpoint_path: str
    config_sha256: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose", ensure_purpose(self.purpose))
        object.__setattr__(self, "metrics", {str(key): float(value) for key, value in self.metrics.items()})

    def assert_exportable(self, target_purpose: DataPurpose | str = DataPurpose.PRODUCTION) -> None:
        target = ensure_purpose(target_purpose)
        self.dataset_manifest.assert_exportable(target)
        if target is DataPurpose.PRODUCTION and self.purpose is DataPurpose.TEST_ONLY:
            raise ProductionLineageError("test-only MAPPO run cannot be exported as production")


def _flatten_rollouts(
    rollouts: Sequence[TransitionBatchV1] | Sequence[Sequence[TransitionBatchV1]],
) -> tuple[TransitionBatchV1, ...]:
    if not rollouts:
        return ()
    first = rollouts[0]
    if isinstance(first, TransitionBatchV1):
        return tuple(rollouts)  # type: ignore[arg-type]
    flattened: list[TransitionBatchV1] = []
    for rollout in rollouts:  # type: ignore[assignment]
        flattened.extend(rollout)
    return tuple(flattened)


def train_mappo(
    actor: Any,
    rollouts: Sequence[TransitionBatchV1] | Sequence[Sequence[TransitionBatchV1]],
    *,
    output_dir: str | Path,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    steps: int = 1,
    parent_manifests: Sequence[DatasetManifestV1] = (),
    seed: int = 7,
    critic_snapshots: Sequence[CriticSnapshotV1] = (),
    discriminator: Any | None = None,
    reference_actor: Any | None = None,
    human_like_targets: Mapping[str, float] | None = None,
    gail_coefficient: float = 0.1,
    kl_coefficient: float = 0.01,
    constraint_coefficient: float = 0.1,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.0,
    critic: Any | None = None,
    critic_state_dict: Mapping[str, Any] | None = None,
    behavior_actor: Any | None = None,
) -> MappoRunManifestV1:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for MAPPO training")
    if actor is None or not hasattr(actor, "parameters"):
        raise TypeError("actor must be a PyTorch module")
    if behavior_actor is not None and not hasattr(behavior_actor, "parameters"):
        raise TypeError("behavior_actor must be a PyTorch module")
    if steps <= 0:
        raise ValueError("steps must be positive")
    for name, coefficient in (
        ("gail_coefficient", gail_coefficient),
        ("kl_coefficient", kl_coefficient),
        ("constraint_coefficient", constraint_coefficient),
        ("value_coefficient", value_coefficient),
        ("entropy_coefficient", entropy_coefficient),
    ):
        if coefficient < 0.0 or not math.isfinite(float(coefficient)):
            raise ValueError(f"{name} must be a finite non-negative number")
    if not 0.0 < clip_ratio < 1.0:
        raise ValueError("clip_ratio must be between 0 and 1")
    transitions = _flatten_rollouts(rollouts)
    if not transitions:
        raise ValueError("rollouts cannot be empty")
    torch.manual_seed(seed)
    bot_count = len(transitions[0].bot_ids)
    if not 1 <= bot_count <= 10:
        raise ValueError("MAPPO rollouts must contain between 1 and 10 bot slots")
    time_steps = len(transitions)
    device = next(actor.parameters()).device
    old_policy = behavior_actor if behavior_actor is not None else actor
    old_policy = old_policy.to(device)
    old_policy.eval()
    if critic is None:
        critic = CentralValueCritic(feature_size=64, hidden_size=128)
    elif not hasattr(critic, "parameters") or not hasattr(critic, "forward"):
        raise TypeError("critic must be a PyTorch module")
    critic = critic.to(device)
    if critic_state_dict is not None:
        if not isinstance(critic_state_dict, Mapping):
            raise TypeError("critic_state_dict must be a mapping")
        critic.load_state_dict(dict(critic_state_dict))
    flat_bot_ids = tuple(
        bot_id
        for transition in transitions
        for bot_id in transition.bot_ids
    )
    critic_features = _critic_features(
        critic_snapshots,
        flat_bot_ids,
        time_steps,
        bot_count,
        device,
    )
    with torch.no_grad():
        initial_values = critic(
            critic_features.reshape(time_steps * bot_count, 64)
        ).reshape(time_steps, bot_count).detach()
        bootstrap_values = critic(critic_features[-1]).reshape(bot_count).detach()
    critic_value_sequences = {
        bot_id: [
            float(initial_values[time_index, bot_index].cpu())
            for time_index in range(time_steps)
        ] + [float(bootstrap_values[bot_index].cpu())]
        for bot_index, bot_id in enumerate(transitions[0].bot_ids)
    }
    prepared = prepare_mappo_batch(
        transitions,
        critic_snapshots=critic_snapshots,
        values=critic_value_sequences,
    )
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    observations = torch.tensor(
        [[byte for byte in value] for value in prepared.observations],
        dtype=torch.float32,
        device=device,
    ).reshape(time_steps, bot_count, 256) / 255.0
    advantages = torch.tensor(prepared.advantages, dtype=torch.float32, device=device).reshape(time_steps, bot_count)
    returns = torch.tensor(prepared.returns, dtype=torch.float32, device=device).reshape(time_steps, bot_count)
    hidden_mask = torch.tensor(prepared.hidden_state_mask, dtype=torch.bool, device=device).reshape(time_steps, bot_count)
    action_mask = torch.tensor(
        [action.action_valid_mask != 0 for action in prepared.actions],
        dtype=torch.bool,
        device=device,
    ).reshape(time_steps, bot_count)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    with torch.no_grad():
        old_outputs = _unroll_actor(old_policy, observations, hidden_mask)
        old_log_probs = torch.stack(
            [_policy_log_prob(output, prepared.actions[time_index * bot_count : (time_index + 1) * bot_count]) for time_index, output in enumerate(old_outputs)]
        ).detach()
        old_values = initial_values
    if reference_actor is not None and hasattr(reference_actor, "to"):
        reference_actor = reference_actor.to(device)
        reference_actor.eval()
    loss_history: list[float] = []
    component_history: dict[str, list[float]] = {
        "ppo_actor_loss": [],
        "critic_loss": [],
        "clip_fraction": [],
        "gail_reward": [],
        "kl_penalty": [],
        "human_constraint": [],
        "entropy": [],
        "old_value_mean": [],
    }
    last_outputs: list[Any] = []
    last_predicted_actions: Any | None = None
    active_samples = (hidden_mask & action_mask).reshape(-1)
    value_samples = hidden_mask.reshape(-1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        outputs = _unroll_actor(actor, observations, hidden_mask)
        new_log_probs = torch.stack(
            [_policy_log_prob(output, prepared.actions[time_index * bot_count : (time_index + 1) * bot_count]) for time_index, output in enumerate(outputs)]
        )
        predicted_actions = torch.stack(
            [torch.cat([output.movement_mean, output.mouse_loc], dim=-1) for output in outputs]
        )
        last_outputs = outputs
        last_predicted_actions = predicted_actions
        gail_advantages = torch.zeros_like(advantages)
        gail_reward = predicted_actions.sum() * 0.0
        if discriminator is not None:
            gail_reward = _discriminator_reward(discriminator, observations, predicted_actions).detach()
            gail_advantages = gail_reward.unsqueeze(0).expand_as(advantages)
        flat_advantages = advantages.reshape(-1)
        selected = active_samples
        if bool(selected.any()):
            selected_advantages = flat_advantages[selected]
            selected_advantages = (selected_advantages - selected_advantages.mean()) / selected_advantages.std(unbiased=False).clamp_min(1e-6)
            if discriminator is not None:
                selected_advantages = selected_advantages + gail_coefficient * gail_advantages.reshape(-1)[selected]
            actor_loss, clip_fraction = ppo_clip_objective(
                new_log_probs.reshape(-1)[selected],
                old_log_probs.reshape(-1)[selected],
                selected_advantages,
                clip_ratio=clip_ratio,
            )
        else:
            actor_loss = predicted_actions.sum() * 0.0
            clip_fraction = predicted_actions.sum() * 0.0
        current_values = critic(critic_features.reshape(time_steps * bot_count, 64)).reshape(time_steps, bot_count)
        if bool(value_samples.any()):
            current_value_samples = current_values.reshape(-1)[value_samples]
            old_value_samples = old_values.reshape(-1)[value_samples]
            return_samples = returns.reshape(-1)[value_samples]
            clipped_values = old_value_samples + (current_value_samples - old_value_samples).clamp(
                -clip_ratio,
                clip_ratio,
            )
            critic_loss = 0.5 * torch.maximum(
                (current_value_samples - return_samples) ** 2,
                (clipped_values - return_samples) ** 2,
            ).mean()
        else:
            critic_loss = current_values.sum() * 0.0
        kl_penalty = predicted_actions.sum() * 0.0
        if reference_actor is not None:
            with torch.no_grad():
                reference_outputs = _unroll_actor(reference_actor, observations, hidden_mask)
            kl_values = torch.stack(
                [_policy_kl(output, reference_outputs[index]) for index, output in enumerate(outputs)]
            )
            kl_penalty = kl_values[hidden_mask].mean() if bool(hidden_mask.any()) else kl_values.mean() * 0.0
        entropy = torch.stack([_policy_entropy(output) for output in outputs]).mean()
        constraint_loss = _human_like_penalty(predicted_actions, outputs, human_like_targets)
        total_loss = (
            actor_loss
            + value_coefficient * critic_loss
            + kl_coefficient * kl_penalty
            + constraint_coefficient * constraint_loss
            - entropy_coefficient * entropy
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        optimizer.step()
        critic_optimizer.step()
        loss_history.append(float(total_loss.detach().cpu()))
        component_history["ppo_actor_loss"].append(float(actor_loss.detach().cpu()))
        component_history["critic_loss"].append(float(critic_loss.detach().cpu()))
        component_history["clip_fraction"].append(float(clip_fraction.detach().cpu()))
        component_history["gail_reward"].append(float(gail_reward.mean().detach().cpu()))
        component_history["kl_penalty"].append(float(kl_penalty.detach().cpu()))
        component_history["human_constraint"].append(float(constraint_loss.detach().cpu()))
        component_history["entropy"].append(float(entropy.detach().cpu()))
        component_history["old_value_mean"].append(float(old_values.mean().detach().cpu()))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = output_root / "mappo-checkpoint.pt"
    metrics = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in component_history.items()
    }
    if last_predicted_actions is not None:
        metrics.update(
            _human_like_metric_values(
                last_predicted_actions.detach(),
                last_outputs,
                prepared.observations,
            )
        )
    config_text = (
        f"{steps}|{seed}|{prepared.phase.name.lower()}|{clip_ratio}|{gail_coefficient}|"
        f"{kl_coefficient}|{constraint_coefficient}|{value_coefficient}|{entropy_coefficient}|"
        f"{discriminator is not None}|{reference_actor is not None}|{bool(human_like_targets)}"
        f"|{behavior_actor is not None}"
    )
    config_sha256 = hashlib.sha256(config_text.encode()).hexdigest()
    dataset = DatasetManifestV1(
        name=f"mappo:{output_root.name}",
        purpose=ensure_purpose(purpose),
        parents=tuple(parent_manifests),
        artifact_type="mappo_rollout",
        metadata={
            "phase": prepared.phase.name.lower(),
            "ctde_actor_observations": "isolated",
            "ppo_clip_ratio": str(clip_ratio),
            "critic": "central_value_critic",
            "gail_enabled": str(discriminator is not None).lower(),
            "kl_enabled": str(reference_actor is not None).lower(),
            "human_constraints_enabled": str(bool(human_like_targets)).lower(),
            "behavior_policy_frozen": str(behavior_actor is not None).lower(),
            "metrics": str(metrics),
        },
    )
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "critic_config": {"feature_size": 64, "hidden_size": 128},
            "loss_history": loss_history,
            "metrics": metrics,
            "dataset_manifest": dataset,
        },
        checkpoint,
    )
    return MappoRunManifestV1(
        name=output_root.name,
        purpose=dataset.effective_purpose(),
        dataset_manifest=dataset,
        steps=steps,
        loss_history=tuple(loss_history),
        checkpoint_path=str(checkpoint),
        config_sha256=config_sha256,
        metrics=metrics,
    )
