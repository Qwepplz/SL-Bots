from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import BotActionV1, DataPurpose, Phase, ensure_purpose
from .lineage import DatasetManifestV1, ProductionLineageError
from .selfplay import CriticSnapshotV1, RolloutEnvelopeV1, TransitionBatchV1
from .training_device import resolve_training_device


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


@dataclass(frozen=True)
class RecurrentRolloutV1:
    instance_id: str
    match_id: str
    policy_generation: int
    transitions: tuple[TransitionBatchV1, ...]
    critic_snapshots: tuple[CriticSnapshotV1, ...]
    shard_sha256: str = ""
    critic_sha256: str = ""
    dataset_manifest: DatasetManifestV1 | None = None
    critic_manifest: DatasetManifestV1 | None = None
    terminal: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        instance_id = str(self.instance_id)
        match_id = str(self.match_id)
        if not instance_id or not match_id:
            raise ValueError("recurrent rollout instance_id and match_id are required")
        if not isinstance(self.policy_generation, int) or isinstance(self.policy_generation, bool):
            raise TypeError("recurrent rollout policy_generation must be an integer")
        if self.policy_generation < 0:
            raise ValueError("recurrent rollout policy_generation must be non-negative")
        transitions = tuple(self.transitions)
        snapshots = tuple(self.critic_snapshots)
        if any(not isinstance(item, TransitionBatchV1) for item in transitions):
            raise TypeError("recurrent rollout transitions must contain TransitionBatchV1 values")
        if any(not isinstance(item, CriticSnapshotV1) for item in snapshots):
            raise TypeError("recurrent rollout critic_snapshots must contain CriticSnapshotV1 values")
        if self.dataset_manifest is not None and not isinstance(self.dataset_manifest, DatasetManifestV1):
            raise TypeError("recurrent rollout dataset_manifest must be DatasetManifestV1")
        if self.critic_manifest is not None and not isinstance(self.critic_manifest, DatasetManifestV1):
            raise TypeError("recurrent rollout critic_manifest must be DatasetManifestV1")
        terminal = bool(self.terminal)
        truncated = bool(self.truncated)
        if terminal and truncated:
            raise ValueError("recurrent rollout terminal and truncated cannot both be true")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "match_id", match_id)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "critic_snapshots", snapshots)
        object.__setattr__(self, "terminal", terminal)
        object.__setattr__(self, "truncated", truncated)
        self.assert_recurrent_boundaries()

    @property
    def phase(self) -> Phase:
        if not self.transitions:
            raise ValueError("recurrent rollout cannot expose a phase without transitions")
        return self.transitions[0].phase

    @property
    def hidden_state_mask(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(transition.hidden_state_mask for transition in self.transitions)

    def assert_recurrent_boundaries(self) -> None:
        if not self.transitions:
            raise ValueError("recurrent rollout transitions cannot be empty")
        phase = self.transitions[0].phase
        bot_ids = self.transitions[0].bot_ids
        epoch = self.transitions[0].epoch
        if any(transition.phase is not phase for transition in self.transitions):
            raise ValueError("recurrent rollout cannot mix Get5 phases")
        if any(transition.bot_ids != bot_ids for transition in self.transitions):
            raise ValueError("recurrent rollout bot ids must remain stable")
        if any(transition.epoch != epoch for transition in self.transitions):
            raise ValueError("recurrent rollout cannot cross epochs")
        if any(
            current.server_tick <= previous.server_tick
            for previous, current in zip(self.transitions, self.transitions[1:])
        ):
            raise ValueError("recurrent rollout server ticks must be strictly increasing")
        if any(self.transitions[0].hidden_state_mask):
            raise ValueError("recurrent rollout first hidden state mask must reset")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_from_payload(value: Mapping[str, Any]) -> DatasetManifestV1:
    parents = tuple(
        _manifest_from_payload(parent)
        for parent in value.get("parents", ())
        if isinstance(parent, Mapping)
    )
    return DatasetManifestV1(
        name=str(value["name"]),
        purpose=ensure_purpose(value["purpose"]),
        parents=parents,
        artifact_type=str(value.get("artifact_type", "dataset")),
        source_sha256=value.get("source_sha256"),
        parser_version=value.get("parser_version"),
        projection_version=value.get("projection_version"),
        metadata={str(key): str(item) for key, item in value.get("metadata", {}).items()},
    )


def _read_manifest_sidecar(
    path: Path,
    *,
    artifact_type: str,
    artifact_sha256: str,
    expected: Mapping[str, str],
) -> DatasetManifestV1:
    if not path.is_file():
        raise ValueError(f"rollout manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"rollout manifest is not valid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("rollout manifest must be a JSON object")
    manifest = _manifest_from_payload(payload)
    if manifest.artifact_type != artifact_type:
        raise ValueError(f"rollout manifest artifact type must be {artifact_type}")
    metadata = dict(manifest.metadata)
    for key, expected_value in expected.items():
        actual = metadata.get(key, payload.get(key))
        if actual is not None and str(actual) != str(expected_value):
            raise ValueError(f"rollout manifest {key} does not match the envelope")
    declared_hash = metadata.get("shard_sha256") or metadata.get("sha256") or payload.get("shard_sha256") or payload.get("sha256")
    if declared_hash is not None and str(declared_hash).lower() != artifact_sha256:
        raise ValueError(f"rollout manifest hash does not match {path}")
    return manifest


def _metadata_phase(value: Any) -> Phase:
    if isinstance(value, str):
        try:
            return Phase[value.upper()]
        except KeyError as error:
            raise ValueError(f"unknown rollout phase: {value}") from error
    return Phase(int(value))


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid rollout boolean metadata: {value}")


def load_rollout_envelope(envelope: RolloutEnvelopeV1) -> RecurrentRolloutV1:
    if not isinstance(envelope, RolloutEnvelopeV1):
        raise TypeError("envelope must be RolloutEnvelopeV1")
    shard_path = Path(envelope.shard_path)
    critic_path = Path(envelope.critic_path)
    if not shard_path.is_file() or not critic_path.is_file():
        raise FileNotFoundError(shard_path if not shard_path.is_file() else critic_path)
    shard_sha256 = _sha256_file(shard_path)
    critic_sha256 = _sha256_file(critic_path)
    expected = {
        "instance_id": envelope.instance_id,
        "match_id": envelope.match_id,
        "policy_generation": str(envelope.policy_generation),
        "phase": envelope.phase.name.lower(),
        "terminal": str(envelope.terminal).lower(),
        "truncated": str(envelope.truncated).lower(),
    }
    dataset_manifest = _read_manifest_sidecar(
        shard_path.with_name(f"{shard_path.stem}.manifest.json"),
        artifact_type="trajectory",
        artifact_sha256=shard_sha256,
        expected=expected,
    )
    critic_manifest = _read_manifest_sidecar(
        critic_path.with_name(f"{critic_path.stem}.manifest.json"),
        artifact_type="critic_trajectory",
        artifact_sha256=critic_sha256,
        expected=expected,
    )
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for rollout loading") from error

    trajectory_file = pq.ParquetFile(str(shard_path))
    required = (
        "phase",
        "epoch",
        "server_tick",
        "bot_id",
        "observation",
        "action",
        "reward",
        "done",
        "state_fault",
        "hidden_state_mask",
    )
    missing = [name for name in required if name not in trajectory_file.schema.names]
    if missing:
        raise ValueError(f"rollout shard is missing columns: {', '.join(missing)}")
    table_metadata = trajectory_file.schema_arrow.metadata or {}
    for key, expected_value in expected.items():
        encoded = table_metadata.get(key.encode("utf-8"))
        if encoded is not None and encoded.decode("utf-8") != str(expected_value):
            raise ValueError(f"rollout shard {key} does not match the envelope")

    def row_value(batch: Any, name: str, index: int) -> Any:
        value = batch.column(batch.schema.get_field_index(name))[index]
        return value.as_py() if hasattr(value, "as_py") else value

    decision_columns = (
        "decision_action_tactical",
        "decision_action_task",
        "decision_action_target",
        "decision_log_prob",
    )
    present_decision_columns = tuple(
        name for name in decision_columns if name in trajectory_file.schema.names
    )
    if present_decision_columns and len(present_decision_columns) != len(decision_columns):
        missing_decision = [name for name in decision_columns if name not in present_decision_columns]
        raise ValueError(
            "rollout shard has incomplete decision behavior columns: "
            + ", ".join(missing_decision)
        )
    columns = list(required) + list(present_decision_columns)

    rows_by_tick: dict[int, list[dict[str, Any]]] = {}
    tick_order: list[int] = []
    for batch in trajectory_file.iter_batches(batch_size=4096, columns=columns):
        for index in range(batch.num_rows):
            phase = _metadata_phase(row_value(batch, "phase", index))
            if phase is not envelope.phase:
                raise ValueError("rollout row phase does not match the envelope")
            server_tick = int(row_value(batch, "server_tick", index))
            if server_tick not in rows_by_tick:
                rows_by_tick[server_tick] = []
                tick_order.append(server_tick)
            row = {
                    "phase": phase,
                    "epoch": int(row_value(batch, "epoch", index)),
                    "server_tick": server_tick,
                    "bot_id": str(row_value(batch, "bot_id", index)),
                    "observation": bytes(row_value(batch, "observation", index)),
                    "action": BotActionV1.from_bytes(bytes(row_value(batch, "action", index))),
                    "reward": float(row_value(batch, "reward", index)),
                    "done": bool(row_value(batch, "done", index)),
                    "state_fault": bool(row_value(batch, "state_fault", index)),
                    "hidden_state_mask": bool(row_value(batch, "hidden_state_mask", index)),
                }
            if present_decision_columns:
                tactical = row_value(batch, "decision_action_tactical", index)
                task = row_value(batch, "decision_action_task", index)
                target = row_value(batch, "decision_action_target", index)
                log_prob = row_value(batch, "decision_log_prob", index)
                values = (tactical, task, target)
                if all(value is None for value in values):
                    if log_prob is not None:
                        raise ValueError("rollout decision log probability has no decision action")
                    row["decision_action"] = None
                    row["decision_log_prob"] = None
                elif any(value is None for value in values) or log_prob is None:
                    raise ValueError("rollout decision behavior row is incomplete")
                else:
                    row["decision_action"] = tuple(int(value) for value in values)
                    row["decision_log_prob"] = float(log_prob)
            else:
                row["decision_action"] = None
                row["decision_log_prob"] = None
            rows_by_tick[server_tick].append(row)
    transitions: list[TransitionBatchV1] = []
    for server_tick in tick_order:
        tick_rows = rows_by_tick[server_tick]
        bot_ids = tuple(row["bot_id"] for row in tick_rows)
        if len(set(bot_ids)) != len(bot_ids):
            raise ValueError("rollout shard contains duplicate bot rows for one server tick")
        transitions.append(
            TransitionBatchV1(
                epoch=tick_rows[0]["epoch"],
                phase=envelope.phase,
                server_tick=server_tick,
                bot_ids=bot_ids,
                observations=tuple(row["observation"] for row in tick_rows),
                actions=tuple(row["action"] for row in tick_rows),
                rewards=tuple(row["reward"] for row in tick_rows),
                dones=tuple(row["done"] for row in tick_rows),
                state_faults=tuple(row["state_fault"] for row in tick_rows),
                hidden_state_mask=tuple(row["hidden_state_mask"] for row in tick_rows),
                decision_actions=tuple(row["decision_action"] for row in tick_rows),
                decision_log_probs=tuple(row["decision_log_prob"] for row in tick_rows),
            )
        )

    critic_file = pq.ParquetFile(str(critic_path))
    critic_required = ("phase", "server_tick", "snapshot_json")
    missing = [name for name in critic_required if name not in critic_file.schema.names]
    if missing:
        raise ValueError(f"critic shard is missing columns: {', '.join(missing)}")
    snapshots: list[CriticSnapshotV1] = []
    for batch in critic_file.iter_batches(batch_size=4096, columns=list(critic_required)):
        for index in range(batch.num_rows):
            phase = _metadata_phase(row_value(batch, "phase", index))
            if phase is not envelope.phase:
                raise ValueError("critic row phase does not match the envelope")
            values = json.loads(str(row_value(batch, "snapshot_json", index)))
            if not isinstance(values, Mapping):
                raise ValueError("critic snapshot must be a JSON object")
            snapshots.append(CriticSnapshotV1(int(row_value(batch, "server_tick", index)), values))
    return RecurrentRolloutV1(
        instance_id=envelope.instance_id,
        match_id=envelope.match_id,
        policy_generation=envelope.policy_generation,
        transitions=tuple(transitions),
        critic_snapshots=tuple(snapshots),
        shard_sha256=shard_sha256,
        critic_sha256=critic_sha256,
        dataset_manifest=dataset_manifest,
        critic_manifest=critic_manifest,
        terminal=envelope.terminal,
        truncated=envelope.truncated,
    )


def prepare_recurrent_rollouts(
    envelopes: Sequence[RolloutEnvelopeV1 | RecurrentRolloutV1],
) -> tuple[RecurrentRolloutV1, ...]:
    normalized = tuple(envelopes)
    if any(
        not isinstance(envelope, (RolloutEnvelopeV1, RecurrentRolloutV1))
        for envelope in normalized
    ):
        raise TypeError("envelopes must contain RolloutEnvelopeV1 or RecurrentRolloutV1 values")
    if any(isinstance(envelope, RolloutEnvelopeV1) for envelope in normalized) and any(
        isinstance(envelope, RecurrentRolloutV1) for envelope in normalized
    ):
        raise TypeError("prepare_recurrent_rollouts cannot mix envelope and loaded rollout values")
    groups = (
        tuple(load_rollout_envelope(envelope) for envelope in normalized)
        if normalized and isinstance(normalized[0], RolloutEnvelopeV1)
        else tuple(normalized)
    )
    if not groups:
        return ()
    generations = {rollout.policy_generation for rollout in groups}
    if len(generations) != 1:
        raise ValueError("MAPPO update cannot mix policy generations")
    phases = {rollout.phase for rollout in groups}
    if len(phases) != 1:
        raise ValueError("MAPPO update cannot mix Get5 phases")
    identities = {
        (rollout.instance_id, rollout.match_id, rollout.shard_sha256 or "")
        for rollout in groups
    }
    if len(identities) != len(groups):
        raise ValueError("MAPPO update cannot merge duplicate instance and match envelopes")
    for rollout in groups:
        rollout.assert_recurrent_boundaries()
    return groups


def validate_hierarchical_rollouts(
    rollouts: Sequence[RecurrentRolloutV1],
) -> tuple[RecurrentRolloutV1, ...]:
    """Validate one complete, Demo-only recurrent sequence per server match.

    Actor tensors are deliberately sourced only from ``TransitionBatchV1.actor_observations``;
    critic snapshots remain a separate shard and are never merged into the actor input schema.
    """

    try:
        normalized = prepare_recurrent_rollouts(tuple(rollouts))
    except ValueError as error:
        if "policy generations" in str(error):
            raise ValueError("all hierarchical MAPPO rollouts must use the same Demo-only generation") from error
        raise
    if not normalized:
        raise ValueError("hierarchical MAPPO rollouts cannot be empty")
    identities = {(item.instance_id, item.match_id) for item in normalized}
    if len(identities) != len(normalized):
        raise ValueError("hierarchical MAPPO cannot merge duplicate instance and match sequences")
    generations = {item.policy_generation for item in normalized}
    if len(generations) != 1:
        raise ValueError("all hierarchical MAPPO rollouts must use the same Demo-only generation")
    for rollout in normalized:
        if not rollout.terminal or rollout.truncated:
            raise ValueError("hierarchical MAPPO requires complete terminal rollouts")
        manifest = rollout.dataset_manifest
        critic_manifest = rollout.critic_manifest
        if manifest is None or manifest.effective_purpose() is not DataPurpose.TEST_ONLY:
            raise ValueError("hierarchical MAPPO rollouts must be Demo-only test_only artifacts")
        if critic_manifest is None or critic_manifest.effective_purpose() is not DataPurpose.TEST_ONLY:
            raise ValueError("hierarchical MAPPO critic snapshots must be separate Demo-only shards")
        generation = str(rollout.policy_generation)
        for current_manifest in (manifest, critic_manifest):
            declared_generation = current_manifest.metadata.get("generation")
            if declared_generation is not None and declared_generation != generation:
                raise ValueError("hierarchical MAPPO rollout manifest generation does not match policy generation")
            source = current_manifest.metadata.get("source")
            if source is not None and source != "demo-only":
                raise ValueError("hierarchical MAPPO rollouts must use Demo-only source artifacts")
        for transition in rollout.transitions:
            if any(len(observation) != 256 for observation in transition.actor_observations):
                raise ValueError("hierarchical actor rollouts may contain only 256-byte local observations")
            if len(transition.decision_actions) != len(transition.bot_ids):
                raise ValueError("hierarchical MAPPO rollouts must carry decision behavior labels")
            if len(transition.decision_log_probs) != len(transition.bot_ids):
                raise ValueError("hierarchical MAPPO rollouts must carry decision behavior log probabilities")
            for action, log_prob in zip(
                transition.decision_actions,
                transition.decision_log_probs,
            ):
                if action is None and log_prob is None:
                    continue
                if action is None or log_prob is None or not math.isfinite(float(log_prob)):
                    raise ValueError("hierarchical MAPPO rollout decision behavior is incomplete")
        if not any(transition.has_decision_behavior for transition in rollout.transitions):
            raise ValueError("hierarchical MAPPO rollouts contain no sampled decision actions")
    return normalized


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


def _detach_recurrent_state(state: Any) -> Any:
    return type(state)(
        tactical=state.tactical.detach(),
        action=state.action.detach(),
        tick=state.tick.detach(),
    )


def _unroll_actor(
    actor: Any,
    observations: Any,
    hidden_state_mask: Any,
    *,
    tbptt_window: int = 128,
    delta_time_s: Any | None = None,
) -> list[Any]:
    if not isinstance(tbptt_window, int) or isinstance(tbptt_window, bool) or tbptt_window <= 0:
        raise ValueError("tbptt_window must be a positive integer")
    time_steps, bot_count, _ = observations.shape
    state = actor.initial_state(bot_count, device=observations.device)
    outputs: list[Any] = []
    for time_index in range(time_steps):
        state = _reset_recurrent_state(state, hidden_state_mask[time_index])
        if delta_time_s is None:
            output = actor(observations[time_index], state)
        else:
            output = actor(
                observations[time_index],
                state,
                delta_time_s=delta_time_s[time_index],
            )
        outputs.append(output)
        state = output.recurrent_state
        if (time_index + 1) % tbptt_window == 0:
            state = _detach_recurrent_state(state)
    return outputs


def _unroll_for_training(
    actor: Any,
    observations: Any,
    hidden_state_mask: Any,
    tbptt_window: int,
    delta_time_s: Any | None = None,
) -> list[Any]:
    if tbptt_window == 128:
        if delta_time_s is None:
            return _unroll_actor(
                actor,
                observations,
                hidden_state_mask,
            )
        return _unroll_actor(
            actor,
            observations,
            hidden_state_mask,
            delta_time_s=delta_time_s,
        )
    return _unroll_actor(
        actor,
        observations,
        hidden_state_mask,
        tbptt_window=tbptt_window,
        delta_time_s=delta_time_s,
    )


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


def _discriminator_action_features(predicted_actions: Any, outputs: Sequence[Any] | None = None) -> Any:
    if outputs is None:
        return predicted_actions
    from .training_gail import ACTION_FEATURE_SIZE

    discrete_features = []
    for output in outputs:
        weapon = torch.softmax(output.weapon_logits, dim=-1)
        if weapon.shape[-1] < 17:
            weapon = torch.nn.functional.pad(weapon, (0, 17 - weapon.shape[-1]))
        else:
            weapon = weapon[..., :17]
        buy = torch.softmax(output.buy_logits, dim=-1)
        if buy.shape[-1] < 32:
            buy = torch.nn.functional.pad(buy, (0, 32 - buy.shape[-1]))
        else:
            buy = buy[..., :32]
        masks = torch.ones(
            (*output.button_logits.shape[:-1], 8),
            dtype=output.button_logits.dtype,
            device=output.button_logits.device,
        )
        discrete_features.append(
            torch.cat(
                [torch.sigmoid(output.button_logits), weapon, buy, masks],
                dim=-1,
            )
        )
    features = torch.cat([predicted_actions, torch.stack(discrete_features)], dim=-1)
    if features.shape[-1] != ACTION_FEATURE_SIZE:
        raise ValueError("actor discriminator features do not match the GAIL action schema")
    return features


def _discriminator_reward(
    discriminator: Any,
    observations: Any,
    predicted_actions: Any,
    outputs: Sequence[Any] | None = None,
) -> Any:
    action_features = _discriminator_action_features(predicted_actions, outputs)
    logits = discriminator(observations.permute(1, 0, 2), action_features.permute(1, 0, 2))
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
    policy_generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "purpose", ensure_purpose(self.purpose))
        object.__setattr__(self, "metrics", {str(key): float(value) for key, value in self.metrics.items()})
        if not isinstance(self.policy_generation, int) or isinstance(self.policy_generation, bool) or self.policy_generation < 0:
            raise ValueError("policy_generation must be a non-negative integer")

    def assert_exportable(self, target_purpose: DataPurpose | str = DataPurpose.PRODUCTION) -> None:
        target = ensure_purpose(target_purpose)
        self.dataset_manifest.assert_exportable(target)
        if target is DataPurpose.PRODUCTION and self.purpose is DataPurpose.TEST_ONLY:
            raise ProductionLineageError("test-only MAPPO run cannot be exported as production")


def _coerce_rollout_groups(
    rollouts: Sequence[
        TransitionBatchV1
        | Sequence[TransitionBatchV1]
        | RecurrentRolloutV1
    ],
    critic_snapshots: Sequence[CriticSnapshotV1],
) -> tuple[tuple[tuple[TransitionBatchV1, ...], ...], tuple[tuple[CriticSnapshotV1, ...], ...], tuple[RecurrentRolloutV1, ...]]:
    normalized = tuple(rollouts)
    if not normalized:
        raise ValueError("rollouts cannot be empty")
    if all(isinstance(item, RecurrentRolloutV1) for item in normalized):
        recurrent = prepare_recurrent_rollouts(normalized)
        return (
            tuple(tuple(item.transitions) for item in recurrent),
            tuple(tuple(item.critic_snapshots) for item in recurrent),
            recurrent,
        )
    if any(isinstance(item, RecurrentRolloutV1) for item in normalized):
        raise TypeError("rollouts cannot mix RecurrentRolloutV1 with transition batches")
    if all(isinstance(item, TransitionBatchV1) for item in normalized):
        transitions = tuple(normalized)
        return ((transitions,), (tuple(critic_snapshots),), ())
    groups: list[tuple[TransitionBatchV1, ...]] = []
    for item in normalized:
        if isinstance(item, (str, bytes, bytearray)):
            raise TypeError("rollout groups must contain TransitionBatchV1 values")
        group = tuple(item)
        if any(not isinstance(transition, TransitionBatchV1) for transition in group):
            raise TypeError("rollout groups must contain TransitionBatchV1 values")
        if not group:
            raise ValueError("rollout groups cannot be empty")
        groups.append(group)
    return (tuple(groups), tuple(() for _ in groups), ())


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
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


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict, key=str):
        value = state_dict[name]
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(name).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(str(name).encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _validate_mappo_device(
    actor: Any,
    purpose: DataPurpose,
    requested_device: str | None,
) -> Any:
    if requested_device is None:
        if purpose is DataPurpose.PRODUCTION:
            requested_device = "cuda"
        else:
            requested_device = str(next(actor.parameters()).device)
    resolved = resolve_training_device(str(requested_device))
    if purpose is DataPurpose.PRODUCTION and resolved.type != "cuda":
        raise RuntimeError("production MAPPO requires GPU training")
    return resolved


def _validate_generation_numbers(number: int, parent_generation: int | None) -> None:
    if not isinstance(number, int) or isinstance(number, bool) or number < 0:
        raise ValueError("policy_generation must be a non-negative integer")
    if number == 0 and parent_generation is not None:
        raise ValueError("generation 0 cannot have a parent generation")
    if number > 0 and parent_generation != number - 1:
        raise ValueError("policy generation parent must be exactly number - 1")


def _validate_mappo_lineage(
    parent_manifests: Sequence[DatasetManifestV1],
    purpose: DataPurpose,
) -> None:
    if any(not isinstance(parent, DatasetManifestV1) for parent in parent_manifests):
        raise TypeError("MAPPO parents must contain DatasetManifestV1 values")
    if purpose is DataPurpose.PRODUCTION:
        for parent in parent_manifests:
            parent.assert_exportable(DataPurpose.PRODUCTION)


def _freeze_module(module: Any | None) -> None:
    if module is None or not hasattr(module, "parameters"):
        return
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _load_optimizer_state(optimizer: Any, state: Mapping[str, Any] | None, device: Any) -> None:
    if state is None:
        return
    if not isinstance(state, Mapping):
        raise TypeError("optimizer state must be a mapping")
    optimizer.load_state_dict(dict(state))
    for values in optimizer.state.values():
        for name, value in tuple(values.items()):
            if hasattr(value, "to"):
                values[name] = value.to(device)


def _demo_rehearsal_loss(
    actor: Any,
    windows: Sequence[Any],
    *,
    device: Any,
    tbptt_window: int,
) -> tuple[Any, int]:
    if not windows:
        return next(actor.parameters()).sum() * 0.0, 0
    from .training_gail import WindowSampleV1

    losses: list[Any] = []
    for window in windows:
        if not isinstance(window, WindowSampleV1):
            raise TypeError("demo_rehearsal_windows must contain WindowSampleV1 values")
        if window.source != "demo":
            continue
        time_steps = window.window_length
        observations = torch.tensor(
            [[byte for byte in value] for value in window.observations],
            dtype=torch.float32,
            device=device,
        ).reshape(time_steps, 1, 256) / 255.0
        durations = torch.tensor(
            window.delta_time_s,
            dtype=torch.float32,
            device=device,
        ).reshape(time_steps, 1)
        hidden_mask = torch.ones((time_steps, 1), dtype=torch.bool, device=device)
        hidden_mask[0] = False
        outputs = _unroll_for_training(
            actor,
            observations,
            hidden_mask,
            tbptt_window,
            delta_time_s=durations,
        )
        log_probs = torch.stack(
            [
                _policy_log_prob(output, (window.actions[index],))
                for index, output in enumerate(outputs)
            ]
        )
        losses.append(-log_probs.mean())
    if not losses:
        return next(actor.parameters()).sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def _train_mappo_multiple(
    actor: Any,
    groups: tuple[tuple[TransitionBatchV1, ...], ...],
    group_snapshots: tuple[tuple[CriticSnapshotV1, ...], ...],
    recurrent_groups: tuple[RecurrentRolloutV1, ...],
    *,
    output_dir: str | Path,
    purpose: DataPurpose | str,
    steps: int,
    parent_manifests: Sequence[DatasetManifestV1],
    seed: int,
    discriminator: Any | None,
    reference_actor: Any | None,
    human_like_targets: Mapping[str, float] | None,
    demo_rehearsal_windows: Sequence[Any],
    demo_rehearsal_coefficient: float,
    gail_coefficient: float,
    kl_coefficient: float,
    constraint_coefficient: float,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    critic: Any | None,
    critic_state_dict: Mapping[str, Any] | None,
    actor_optimizer_state_dict: Mapping[str, Any] | None,
    critic_optimizer_state_dict: Mapping[str, Any] | None,
    behavior_actor: Any | None,
    device: str | None,
    policy_generation: int,
    parent_generation: int | None,
    deadline_metadata: Mapping[str, Any] | None,
    tbptt_window: int,
    minibatch_sequences: int,
    epochs: int,
    gradient_clip_norm: float,
) -> MappoRunManifestV1:
    run_purpose = ensure_purpose(purpose)
    _validate_mappo_lineage(parent_manifests, run_purpose)
    _validate_generation_numbers(policy_generation, parent_generation)
    resolved = _validate_mappo_device(actor, run_purpose, device)
    actor.to(resolved)
    actor.train()
    if behavior_actor is not None:
        behavior_actor.to(resolved)
        behavior_actor.eval()
        _freeze_module(behavior_actor)
    old_policy = behavior_actor if behavior_actor is not None else actor
    behavior_sha256 = _state_dict_sha256(old_policy.state_dict())
    if discriminator is not None and hasattr(discriminator, "to"):
        discriminator.to(resolved)
        discriminator.eval()
        _freeze_module(discriminator)
    if reference_actor is not None and hasattr(reference_actor, "to"):
        reference_actor.to(resolved)
        reference_actor.eval()
        _freeze_module(reference_actor)
    if critic is None:
        critic = CentralValueCritic(feature_size=64, hidden_size=128)
    elif not hasattr(critic, "parameters") or not hasattr(critic, "forward"):
        raise TypeError("critic must be a PyTorch module")
    critic.to(resolved)
    critic.train()
    if critic_state_dict is not None:
        if not isinstance(critic_state_dict, Mapping):
            raise TypeError("critic_state_dict must be a mapping")
        critic.load_state_dict(dict(critic_state_dict))
    contexts: list[dict[str, Any]] = []
    for group_index, transitions in enumerate(groups):
        phase = transitions[0].phase
        bot_count = len(transitions[0].bot_ids)
        if not 1 <= bot_count <= 10:
            raise ValueError("MAPPO rollouts must contain between 1 and 10 bot slots")
        if any(transition.phase is not phase for transition in transitions):
            raise ValueError("MAPPO rollout group cannot mix phases")
        if any(transition.bot_ids != transitions[0].bot_ids for transition in transitions):
            raise ValueError("MAPPO rollout group bot ids must remain stable")
        time_steps = len(transitions)
        snapshots = group_snapshots[group_index]
        flat_bot_ids = tuple(
            bot_id
            for transition in transitions
            for bot_id in transition.bot_ids
        )
        critic_features = _critic_features(
            snapshots,
            flat_bot_ids,
            time_steps,
            bot_count,
            resolved,
        )
        with torch.no_grad():
            initial_values = critic(
                critic_features.reshape(time_steps * bot_count, 64)
            ).reshape(time_steps, bot_count).detach()
            bootstrap_values = critic(critic_features[-1]).reshape(bot_count).detach()
        value_sequences = {
            bot_id: [
                float(initial_values[time_index, bot_index].cpu())
                for time_index in range(time_steps)
            ] + [float(bootstrap_values[bot_index].cpu())]
            for bot_index, bot_id in enumerate(transitions[0].bot_ids)
        }
        prepared = prepare_mappo_batch(
            transitions,
            critic_snapshots=snapshots,
            values=value_sequences,
        )
        observations = torch.tensor(
            [[byte for byte in value] for value in prepared.observations],
            dtype=torch.float32,
            device=resolved,
        ).reshape(time_steps, bot_count, 256) / 255.0
        advantages = torch.tensor(prepared.advantages, dtype=torch.float32, device=resolved).reshape(time_steps, bot_count)
        returns = torch.tensor(prepared.returns, dtype=torch.float32, device=resolved).reshape(time_steps, bot_count)
        hidden_mask = torch.tensor(prepared.hidden_state_mask, dtype=torch.bool, device=resolved).reshape(time_steps, bot_count)
        action_mask = torch.tensor(
            [action.action_valid_mask != 0 for action in prepared.actions],
            dtype=torch.bool,
            device=resolved,
        ).reshape(time_steps, bot_count)
        actions_by_time = tuple(
            tuple(prepared.actions[time_index * bot_count : (time_index + 1) * bot_count])
            for time_index in range(time_steps)
        )
        with torch.no_grad():
            old_outputs = _unroll_for_training(old_policy, observations, hidden_mask, tbptt_window)
            old_log_probs = torch.stack(
                [
                    _policy_log_prob(old_outputs[time_index], actions_by_time[time_index])
                    for time_index in range(time_steps)
                ]
            ).detach()
        contexts.append(
            {
                "phase": phase,
                "transitions": transitions,
                "prepared": prepared,
                "observations": observations,
                "advantages": advantages,
                "returns": returns,
                "hidden_mask": hidden_mask,
                "action_mask": action_mask,
                "actions_by_time": actions_by_time,
                "critic_features": critic_features,
                "old_values": initial_values,
                "old_log_probs": old_log_probs,
                "time_steps": time_steps,
                "bot_count": bot_count,
            }
        )
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
    _load_optimizer_state(optimizer, actor_optimizer_state_dict, resolved)
    _load_optimizer_state(critic_optimizer, critic_optimizer_state_dict, resolved)
    torch.manual_seed(seed)
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
        "demo_rehearsal_loss": [],
    }
    last_context: dict[str, Any] | None = None
    last_outputs: list[Any] = []
    last_predicted_actions: Any | None = None
    for _ in range(epochs):
        for batch_start in range(0, len(contexts), minibatch_sequences):
            context_batch = contexts[batch_start : batch_start + minibatch_sequences]
            optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            sequence_losses: list[Any] = []
            aggregates = {name: 0.0 for name in component_history}
            for context in context_batch:
                outputs = _unroll_for_training(
                    actor,
                    context["observations"],
                    context["hidden_mask"],
                    tbptt_window,
                )
                new_log_probs = torch.stack(
                    [
                        _policy_log_prob(outputs[time_index], context["actions_by_time"][time_index])
                        for time_index in range(context["time_steps"])
                    ]
                )
                predicted_actions = torch.stack(
                    [torch.cat([output.movement_mean, output.mouse_loc], dim=-1) for output in outputs]
                )
                gail_reward = predicted_actions.sum() * 0.0
                gail_advantages = torch.zeros_like(context["advantages"])
                if discriminator is not None:
                    gail_reward = _discriminator_reward(
                        discriminator,
                        context["observations"],
                        predicted_actions,
                        outputs,
                    ).detach()
                    gail_advantages = gail_reward.unsqueeze(0).expand_as(context["advantages"])
                selected = (context["hidden_mask"] & context["action_mask"]).reshape(-1)
                if bool(selected.any()):
                    selected_advantages = context["advantages"].reshape(-1)[selected]
                    selected_advantages = (selected_advantages - selected_advantages.mean()) / selected_advantages.std(unbiased=False).clamp_min(1e-6)
                    if discriminator is not None:
                        selected_advantages = selected_advantages + gail_coefficient * gail_advantages.reshape(-1)[selected]
                    actor_loss, clip_fraction = ppo_clip_objective(
                        new_log_probs.reshape(-1)[selected],
                        context["old_log_probs"].reshape(-1)[selected],
                        selected_advantages,
                        clip_ratio=clip_ratio,
                    )
                else:
                    actor_loss = predicted_actions.sum() * 0.0
                    clip_fraction = predicted_actions.sum() * 0.0
                current_values = critic(
                    context["critic_features"].reshape(context["time_steps"] * context["bot_count"], 64)
                ).reshape(context["time_steps"], context["bot_count"])
                value_selected = context["hidden_mask"].reshape(-1)
                if bool(value_selected.any()):
                    current_value_samples = current_values.reshape(-1)[value_selected]
                    old_value_samples = context["old_values"].reshape(-1)[value_selected]
                    return_samples = context["returns"].reshape(-1)[value_selected]
                    clipped_values = old_value_samples + (current_value_samples - old_value_samples).clamp(-clip_ratio, clip_ratio)
                    critic_loss = 0.5 * torch.maximum(
                        (current_value_samples - return_samples) ** 2,
                        (clipped_values - return_samples) ** 2,
                    ).mean()
                else:
                    critic_loss = current_values.sum() * 0.0
                kl_penalty = predicted_actions.sum() * 0.0
                if reference_actor is not None:
                    with torch.no_grad():
                        reference_outputs = _unroll_for_training(
                            reference_actor,
                            context["observations"],
                            context["hidden_mask"],
                            tbptt_window,
                        )
                    kl_values = torch.stack(
                        [_policy_kl(outputs[index], reference_outputs[index]) for index in range(context["time_steps"])]
                    )
                    kl_penalty = kl_values[context["hidden_mask"]].mean() if bool(context["hidden_mask"].any()) else kl_values.mean() * 0.0
                entropy = torch.stack([_policy_entropy(output) for output in outputs]).mean()
                constraint_loss = _human_like_penalty(predicted_actions, outputs, human_like_targets)
                sequence_loss = (
                    actor_loss
                    + value_coefficient * critic_loss
                    + kl_coefficient * kl_penalty
                    + constraint_coefficient * constraint_loss
                    - entropy_coefficient * entropy
                )
                sequence_losses.append(sequence_loss)
                aggregates["ppo_actor_loss"] += float(actor_loss.detach().cpu())
                aggregates["critic_loss"] += float(critic_loss.detach().cpu())
                aggregates["clip_fraction"] += float(clip_fraction.detach().cpu())
                aggregates["gail_reward"] += float(gail_reward.mean().detach().cpu())
                aggregates["kl_penalty"] += float(kl_penalty.detach().cpu())
                aggregates["human_constraint"] += float(constraint_loss.detach().cpu())
                aggregates["entropy"] += float(entropy.detach().cpu())
                aggregates["old_value_mean"] += float(context["old_values"].mean().detach().cpu())
                last_context = context
                last_outputs = outputs
                last_predicted_actions = predicted_actions
            rehearsal_loss, _ = _demo_rehearsal_loss(
                actor,
                demo_rehearsal_windows,
                device=resolved,
                tbptt_window=tbptt_window,
            )
            total_loss = (
                torch.stack(sequence_losses).mean()
                + demo_rehearsal_coefficient * rehearsal_loss
            )
            if not bool(torch.isfinite(total_loss)):
                raise RuntimeError("MAPPO produced a non-finite loss")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            critic_optimizer.step()
            loss_history.append(float(total_loss.detach().cpu()))
            for name in component_history:
                if name == "demo_rehearsal_loss":
                    component_history[name].append(float(rehearsal_loss.detach().cpu()))
                else:
                    component_history[name].append(aggregates[name] / len(context_batch))
    output_root = Path(output_dir)
    metrics = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in component_history.items()
    }
    if last_context is not None and last_predicted_actions is not None:
        metrics.update(
            _human_like_metric_values(
                last_predicted_actions.detach(),
                last_outputs,
                last_context["prepared"].observations,
            )
        )
    metrics["rollout_sequence_count"] = float(len(contexts))
    metrics["policy_generation"] = float(policy_generation)
    metrics["demo_rehearsal_window_count"] = float(
        sum(1 for window in demo_rehearsal_windows if getattr(window, "source", None) == "demo")
    )
    phase = contexts[0]["phase"]
    inherited_parents = list(parent_manifests)
    for rollout in recurrent_groups:
        for manifest in (rollout.dataset_manifest, rollout.critic_manifest):
            if manifest is not None and manifest not in inherited_parents:
                inherited_parents.append(manifest)
    dataset = DatasetManifestV1(
        name=f"mappo:{output_root.name}",
        purpose=run_purpose,
        parents=tuple(inherited_parents),
        artifact_type="mappo_rollout",
        metadata={
            "phase": phase.name.lower(),
            "ruleset": "mr12",
            "ctde_actor_observations": "isolated",
            "rollout_sequence_count": str(len(contexts)),
            "optimizer_updates": str(len(loss_history)),
            "policy_generation": str(int(metrics["policy_generation"])),
            "tbptt_window": str(tbptt_window),
            "minibatch_sequences": str(minibatch_sequences),
            "epochs": str(epochs),
            "ppo_clip_ratio": str(clip_ratio),
            "value_coefficient": str(value_coefficient),
            "gail_coefficient": str(gail_coefficient),
            "kl_coefficient": str(kl_coefficient),
            "human_constraint_coefficient": str(constraint_coefficient),
            "gradient_clip_norm": str(gradient_clip_norm),
            "seed": str(seed),
            "device_type": resolved.type,
            "device_index": str(resolved.index) if resolved.index is not None else "",
            "gail_enabled": str(discriminator is not None).lower(),
            "kl_enabled": str(reference_actor is not None).lower(),
            "human_constraints_enabled": str(bool(human_like_targets)).lower(),
            "behavior_policy_frozen": str(behavior_actor is not None).lower(),
            "demo_rehearsal_window_count": str(int(metrics["demo_rehearsal_window_count"])),
            "demo_rehearsal_coefficient": str(demo_rehearsal_coefficient),
            "metrics": str(metrics),
            **({str(key): str(value) for key, value in (deadline_metadata or {}).items()}),
        },
    )
    config_sha256 = hashlib.sha256(
        f"{steps}|{seed}|{phase.name.lower()}|{clip_ratio}|{gail_coefficient}|{kl_coefficient}|{constraint_coefficient}|{value_coefficient}|{resolved}|{len(contexts)}|{tbptt_window}|{minibatch_sequences}|{epochs}|{gradient_clip_norm}".encode()
    ).hexdigest()
    checkpoint = output_root / "mappo-checkpoint.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "actor_state_dict": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
            "critic_state_dict": {key: value.detach().cpu() for key, value in critic.state_dict().items()},
            "actor_optimizer_state_dict": optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "critic_config": {"feature_size": 64, "hidden_size": 128},
            "loss_history": loss_history,
            "metrics": metrics,
            "dataset_manifest": dataset,
            "device": {"type": resolved.type, "index": resolved.index},
            "policy_generation": int(metrics["policy_generation"]),
            "parent_generation": parent_generation,
            "behavior_sha256": behavior_sha256,
            "deadline": dict(deadline_metadata or {}),
            "rollout_shards": [
                {"shard_sha256": item.shard_sha256, "critic_sha256": item.critic_sha256}
                for item in recurrent_groups
            ],
        },
    )
    return MappoRunManifestV1(
        name=output_root.name,
        purpose=dataset.effective_purpose(),
        dataset_manifest=dataset,
        steps=len(loss_history),
        loss_history=tuple(loss_history),
        checkpoint_path=str(checkpoint),
        config_sha256=config_sha256,
        metrics=metrics,
        policy_generation=policy_generation,
    )


def train_mappo(
    actor: Any,
    rollouts: Sequence[
        TransitionBatchV1
        | Sequence[TransitionBatchV1]
        | RecurrentRolloutV1
    ],
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
    demo_rehearsal_windows: Sequence[Any] = (),
    demo_rehearsal_coefficient: float = 0.01,
    gail_coefficient: float = 0.1,
    kl_coefficient: float = 0.01,
    constraint_coefficient: float = 0.1,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.0,
    tbptt_window: int = 128,
    minibatch_sequences: int = 32,
    epochs: int | None = None,
    gradient_clip_norm: float = 1.0,
    critic: Any | None = None,
    critic_state_dict: Mapping[str, Any] | None = None,
    actor_optimizer_state_dict: Mapping[str, Any] | None = None,
    critic_optimizer_state_dict: Mapping[str, Any] | None = None,
    behavior_actor: Any | None = None,
    device: str | None = None,
    policy_generation: int | None = None,
    parent_generation: int | None = None,
    deadline_metadata: Mapping[str, Any] | None = None,
) -> MappoRunManifestV1:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for MAPPO training")
    if actor is None or not hasattr(actor, "parameters"):
        raise TypeError("actor must be a PyTorch module")
    if behavior_actor is not None and not hasattr(behavior_actor, "parameters"):
        raise TypeError("behavior_actor must be a PyTorch module")
    if steps <= 0:
        raise ValueError("steps must be positive")
    effective_epochs = steps if epochs is None else epochs
    for name, value in (
        ("tbptt_window", tbptt_window),
        ("minibatch_sequences", minibatch_sequences),
        ("epochs", effective_epochs),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(float(gradient_clip_norm)) or gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be a positive finite number")
    for name, coefficient in (
        ("gail_coefficient", gail_coefficient),
        ("kl_coefficient", kl_coefficient),
        ("constraint_coefficient", constraint_coefficient),
        ("value_coefficient", value_coefficient),
        ("entropy_coefficient", entropy_coefficient),
        ("demo_rehearsal_coefficient", demo_rehearsal_coefficient),
    ):
        if coefficient < 0.0 or not math.isfinite(float(coefficient)):
            raise ValueError(f"{name} must be a finite non-negative number")
    if not 0.0 < clip_ratio < 1.0:
        raise ValueError("clip_ratio must be between 0 and 1")
    run_purpose = ensure_purpose(purpose)
    demo_rehearsal_windows = tuple(demo_rehearsal_windows or ())
    _validate_mappo_lineage(parent_manifests, run_purpose)
    groups, group_snapshots, recurrent_groups = _coerce_rollout_groups(rollouts, critic_snapshots)
    if recurrent_groups:
        embedded_generation = recurrent_groups[0].policy_generation
        if parent_generation is not None and parent_generation != embedded_generation:
            raise ValueError("MAPPO parent generation does not match rollout generation")
        if parent_generation is None:
            if policy_generation is None:
                policy_generation = embedded_generation
            elif policy_generation != embedded_generation:
                raise ValueError("MAPPO policy generation does not match rollout generation")
        else:
            expected_generation = embedded_generation + 1
            if policy_generation is None:
                policy_generation = expected_generation
            elif policy_generation != expected_generation:
                raise ValueError("MAPPO policy generation must be the next rollout generation")
    elif policy_generation is None:
        policy_generation = 0
    _validate_generation_numbers(policy_generation, parent_generation)
    if groups:
        return _train_mappo_multiple(
            actor,
            groups,
            group_snapshots,
            recurrent_groups,
            output_dir=output_dir,
            purpose=run_purpose,
            steps=steps,
            parent_manifests=parent_manifests,
            seed=seed,
            discriminator=discriminator,
            reference_actor=reference_actor,
            human_like_targets=human_like_targets,
            demo_rehearsal_windows=demo_rehearsal_windows,
            demo_rehearsal_coefficient=float(demo_rehearsal_coefficient),
            gail_coefficient=gail_coefficient,
            kl_coefficient=kl_coefficient,
            constraint_coefficient=constraint_coefficient,
            clip_ratio=clip_ratio,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            critic=critic,
            critic_state_dict=critic_state_dict,
            actor_optimizer_state_dict=actor_optimizer_state_dict,
            critic_optimizer_state_dict=critic_optimizer_state_dict,
            behavior_actor=behavior_actor,
            device=device,
            policy_generation=policy_generation,
            parent_generation=parent_generation,
            deadline_metadata=deadline_metadata,
            tbptt_window=tbptt_window,
            minibatch_sequences=minibatch_sequences,
            epochs=effective_epochs,
            gradient_clip_norm=gradient_clip_norm,
        )
