from __future__ import annotations

import json
import hashlib
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import BotActionV1, DataPurpose, MAX_BOTS, Phase, Side, ensure_purpose
from .get5_control import TrainingBoundaryV1
from .lineage import DatasetManifestV1
from .rewards import RewardLedger, RewardSignals, RewardWeights

OBSERVATION_BYTES = 256
TRAJECTORY_SCHEMA_VERSION = "trajectory-v1"
_EPISODE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SERVER_TICK_RATE = 128
_EVENT_WINDOW_TICKS = 8 * _SERVER_TICK_RATE
_EVENT_AGE_ROUNDING_TICKS = _SERVER_TICK_RATE // 2


class EpisodeNotRunning(RuntimeError):
    pass


class EpisodePaused(EpisodeNotRunning):
    pass


@dataclass(frozen=True)
class RosterProfile:
    bot_id: str
    team: int
    is_human: bool = False
    difficulty: str = "normal"
    personality: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.bot_id:
            raise ValueError("roster profile bot_id cannot be empty")
        if self.team not in (0, 2, 3):
            raise ValueError("roster profile team must be 0, 2, or 3")
        if not self.difficulty:
            raise ValueError("roster profile difficulty cannot be empty")


@dataclass(frozen=True)
class CriticSnapshotV1:
    server_tick: int
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.server_tick < 0:
            raise ValueError("critic snapshot server_tick must be non-negative")
        object.__setattr__(self, "values", dict(self.values))


@dataclass(frozen=True)
class TransitionBatchV1:
    epoch: int
    phase: Phase
    server_tick: int
    bot_ids: tuple[str, ...]
    observations: tuple[bytes, ...]
    actions: tuple[BotActionV1, ...]
    rewards: tuple[float, ...]
    dones: tuple[bool, ...]
    state_faults: tuple[bool, ...]
    hidden_state_mask: tuple[bool, ...]
    observation_metadata: Mapping[str, str] = field(default_factory=dict)
    # A decision action/log-probability pair is present only on decision refresh
    # ticks.  Keeping it in the rollout is required for a real PPO behavior
    # policy; reconstructing it from the post-hoc actor is not valid.
    decision_actions: tuple[tuple[int, int, int] | None, ...] = ()
    decision_log_probs: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.server_tick < 0:
            raise ValueError("transition epoch and server_tick must be non-negative")
        phase = Phase(self.phase)
        object.__setattr__(self, "phase", phase)
        bot_ids = tuple(self.bot_ids)
        observations = tuple(bytes(value) for value in self.observations)
        actions = tuple(self.actions)
        rewards = tuple(float(value) for value in self.rewards)
        dones = tuple(bool(value) for value in self.dones)
        state_faults = tuple(bool(value) for value in self.state_faults)
        hidden_state_mask = tuple(bool(value) for value in self.hidden_state_mask)
        decision_actions = (
            tuple(self.decision_actions)
            if self.decision_actions
            else tuple(None for _ in bot_ids)
        )
        decision_log_probs = (
            tuple(self.decision_log_probs)
            if self.decision_log_probs
            else tuple(None for _ in bot_ids)
        )
        size = len(bot_ids)
        if len(set(bot_ids)) != size:
            raise ValueError("transition bot_ids must be unique")
        if not all(len(value) == OBSERVATION_BYTES for value in observations):
            raise ValueError(f"every transition observation must be {OBSERVATION_BYTES} bytes")
        if len(observations) != size or len(actions) != size or len(rewards) != size:
            raise ValueError("transition fields must have equal bot counts")
        if len(dones) != size or len(state_faults) != size or len(hidden_state_mask) != size:
            raise ValueError("transition flags must have equal bot counts")
        if len(decision_actions) != size or len(decision_log_probs) != size:
            raise ValueError("decision behavior fields must have equal bot counts")
        for action, log_prob in zip(decision_actions, decision_log_probs):
            if action is None:
                if log_prob is not None:
                    raise ValueError("decision log probability requires a decision action")
                continue
            if (
                not isinstance(action, tuple)
                or len(action) != 3
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in action)
            ):
                raise ValueError("decision action must be a non-negative tactical/task/target tuple")
            if log_prob is None or not math.isfinite(float(log_prob)):
                raise ValueError("decision log probability must be finite")
        object.__setattr__(self, "bot_ids", bot_ids)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "dones", dones)
        object.__setattr__(self, "state_faults", state_faults)
        object.__setattr__(self, "hidden_state_mask", hidden_state_mask)
        object.__setattr__(self, "observation_metadata", dict(self.observation_metadata))
        object.__setattr__(self, "decision_actions", decision_actions)
        object.__setattr__(self, "decision_log_probs", decision_log_probs)

    @property
    def actor_observations(self) -> tuple[bytes, ...]:
        return self.observations

    @property
    def has_decision_behavior(self) -> bool:
        return any(action is not None for action in self.decision_actions)


@dataclass(frozen=True)
class TrajectoryManifestV1:
    episode_id: str
    phase: Phase
    path: Path
    critic_path: Path
    dataset: DatasetManifestV1
    critic_dataset: DatasetManifestV1
    transition_count: int

    def __post_init__(self) -> None:
        if not self.episode_id or self.transition_count < 0:
            raise ValueError("invalid trajectory manifest")
        object.__setattr__(self, "phase", Phase(self.phase))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "critic_path", Path(self.critic_path))


@dataclass(frozen=True)
class RolloutEnvelopeV1:
    instance_id: str
    match_id: str
    policy_generation: int
    phase: Phase
    shard_path: Path
    critic_path: Path
    terminal: bool
    truncated: bool

    def __post_init__(self) -> None:
        if not self.instance_id or not self.match_id:
            raise ValueError("rollout envelope instance_id and match_id are required")
        if self.policy_generation < 0:
            raise ValueError("rollout envelope policy_generation must be non-negative")
        if self.terminal and self.truncated:
            raise ValueError("rollout envelope terminal and truncated cannot both be true")
        object.__setattr__(self, "phase", Phase(self.phase))
        object.__setattr__(self, "shard_path", Path(self.shard_path))
        object.__setattr__(self, "critic_path", Path(self.critic_path))
        if self.shard_path.name.endswith(".tmp") or self.critic_path.name.endswith(".tmp"):
            raise ValueError("rollout envelope paths must be atomically completed files")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable wire representation used by match manifests."""

        return {
            "instance_id": self.instance_id,
            "match_id": self.match_id,
            "policy_generation": self.policy_generation,
            "phase": self.phase.name.lower(),
            "shard_path": str(self.shard_path),
            "critic_path": str(self.critic_path),
            "terminal": self.terminal,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class MatchCompletionV1:
    """Terminal marker carrying every non-truncated shard of one match."""

    instance_id: str
    match_id: str
    policy_generation: int
    envelopes: tuple[RolloutEnvelopeV1, ...] = ()
    terminal: bool = True

    def __post_init__(self) -> None:
        if not str(self.instance_id) or not str(self.match_id):
            raise ValueError("match completion requires instance_id and match_id")
        if not isinstance(self.policy_generation, int) or isinstance(self.policy_generation, bool):
            raise TypeError("match completion policy_generation must be an integer")
        if self.policy_generation < 0:
            raise ValueError("match completion policy_generation must be non-negative")
        if not isinstance(self.terminal, bool) or not self.terminal:
            raise ValueError("match completion marker must be terminal")
        envelopes = tuple(self.envelopes)
        if any(not isinstance(envelope, RolloutEnvelopeV1) for envelope in envelopes):
            raise TypeError("match completion envelopes must contain RolloutEnvelopeV1 values")
        for envelope in envelopes:
            if envelope.instance_id != str(self.instance_id) or envelope.match_id != str(self.match_id):
                raise ValueError("match completion envelope identity does not match the marker")
            if envelope.policy_generation != self.policy_generation:
                raise ValueError("match completion envelope generation does not match the marker")
            if envelope.truncated:
                raise ValueError("match completion cannot contain truncated envelopes")
        object.__setattr__(self, "instance_id", str(self.instance_id))
        object.__setattr__(self, "match_id", str(self.match_id))
        object.__setattr__(self, "envelopes", envelopes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "match_id": self.match_id,
            "policy_generation": self.policy_generation,
            "envelopes": [envelope.to_dict() for envelope in self.envelopes],
            "terminal": True,
        }


class SelfPlayAdapter(Protocol):
    def start_episode(self, phase: Phase, roster_profiles: tuple[RosterProfile, ...], epoch: int) -> None: ...

    def step(self, server_tick: int) -> Mapping[str, Any] | None: ...


_UTILITY_EVENT_NAMES = (
    "smoke_start_self",
    "flash_explode_self",
    "he_explode_self",
    "fire_grenade_start_self",
)
_REWARDED_EVENT_NAMES = (
    "kill_self",
    "round_end",
    "round_win_t",
    "round_win_ct",
    *_UTILITY_EVENT_NAMES,
)
_TERMINAL_EVENT_NAMES = frozenset(("round_end", "round_win_t", "round_win_ct"))


@dataclass
class _ObservedRewardEvent:
    category: str
    occurrence_tick: int
    last_tick: int
    last_age_s: float


@dataclass
class _ObservationRewardState:
    events: dict[str, list[_ObservedRewardEvent]] = field(default_factory=dict)
    totals: dict[str, dict[str, int]] = field(default_factory=dict)
    new_terminal: bool = False

    def reset(self) -> None:
        self.events.clear()
        self.totals.clear()
        self.new_terminal = False


def _same_observed_event(
    observed: _ObservedRewardEvent,
    category: str,
    age_s: float,
    server_tick: int,
) -> bool:
    if observed.category != category:
        return False
    elapsed_s = max(0, int(server_tick) - observed.last_tick) / _SERVER_TICK_RATE
    age_delta_s = float(age_s) - observed.last_age_s
    if age_delta_s < 0.0:
        return False
    return age_delta_s <= math.ceil(elapsed_s) + 1.0


def _advance_observation_reward_state(
    state: _ObservationRewardState,
    bot_id: str,
    events: Sequence[Any],
    server_tick: int,
) -> Mapping[str, int]:
    observed_events = state.events.setdefault(bot_id, [])
    totals = state.totals.setdefault(bot_id, {})
    current_tick = int(server_tick)
    cutoff_tick = current_tick - _EVENT_WINDOW_TICKS
    observed_events = [
        observed
        for observed in observed_events
        if observed.occurrence_tick >= cutoff_tick
    ]
    observed_events.sort(key=lambda observed: observed.occurrence_tick)
    matched: set[int] = set()
    # The bridge writes the newest event first; match the suffix chronologically
    # so each observed event keeps its own occurrence record.
    for event in reversed(events):
        category = str(getattr(event, "category", "unknown"))
        if category not in _REWARDED_EVENT_NAMES:
            continue
        age_s = max(0.0, float(getattr(event, "age_s", 0.0)))
        match_index = next(
            (
                index
                for index, observed in enumerate(observed_events)
                if index not in matched
                and _same_observed_event(observed, category, age_s, current_tick)
            ),
            None,
        )
        if match_index is not None:
            matched.add(match_index)
            observed = observed_events[match_index]
            observed.last_tick = current_tick
            observed.last_age_s = age_s
            continue
        observed_index = len(observed_events)
        observed_events.append(
            _ObservedRewardEvent(
                category=category,
                occurrence_tick=current_tick
                - max(
                    0,
                    int(round(age_s * _SERVER_TICK_RATE)) - _EVENT_AGE_ROUNDING_TICKS,
                ),
                last_tick=current_tick,
                last_age_s=age_s,
            )
        )
        matched.add(observed_index)
        totals[category] = totals.get(category, 0) + 1
        if category in _TERMINAL_EVENT_NAMES:
            state.new_terminal = True
    state.events[bot_id] = [
        observed
        for observed in observed_events
        if observed.occurrence_tick >= cutoff_tick
    ]
    return totals


def _observation_reward_inputs(
    observation_batch: Any,
    action_batch: Any,
    server_tick: int,
    bot_ids: Sequence[str],
    *,
    event_state: _ObservationRewardState | None = None,
) -> Mapping[str, RewardSignals]:
    del action_batch
    from .observation_projection import decode_observation

    state = event_state if event_state is not None else _ObservationRewardState()
    state.new_terminal = False
    rewards: dict[str, RewardSignals] = {}
    for bot_id, payload in zip(bot_ids, observation_batch.observations):
        try:
            observation = decode_observation(payload)
        except (TypeError, ValueError):
            rewards[bot_id] = RewardSignals()
            continue
        totals = _advance_observation_reward_state(state, bot_id, observation.events, server_tick)
        winning_event = "round_win_t" if int(observation.side) == int(Side.T) else "round_win_ct"
        losing_event = "round_win_ct" if winning_event == "round_win_t" else "round_win_t"
        rewards[bot_id] = RewardSignals(
            kills=totals.get("kill_self", 0),
            objectives=totals.get("round_end", 0),
            utility_success=sum(totals.get(name, 0) for name in _UTILITY_EVENT_NAMES),
            team_won=totals.get(winning_event, 0),
            team_lost=totals.get(losing_event, 0),
        )
    return rewards


def _observation_critic_snapshot(
    observation_batch: Any,
    action_batch: Any,
    server_tick: int,
    bot_ids: Sequence[str],
) -> CriticSnapshotV1:
    from .observation_projection import decode_observation

    values: dict[str, Any] = {
        "phase": int(observation_batch.observations[0][5]),
        "bot_count": int(observation_batch.bot_count),
    }
    for bot_id, payload in zip(bot_ids, observation_batch.observations):
        try:
            observation = decode_observation(payload)
        except (TypeError, ValueError):
            continue
        values[bot_id] = {
            "health": observation.self_health,
            "armor": observation.self_armor,
            "money": observation.self_money,
            "alive": bool(observation.self_flags & 1),
            "visible_enemy_count": sum(
                player.relation == -1 and player.flags != 0 for player in observation.players
            ),
        }
    return CriticSnapshotV1(server_tick=int(action_batch.server_tick or server_tick), values=values)


def _observation_done(
    observation_batch: Any,
    *,
    event_state: _ObservationRewardState | None = None,
) -> bool:
    from .observation_projection import decode_observation

    if event_state is not None:
        done = event_state.new_terminal
        event_state.new_terminal = False
        return done
    for payload in observation_batch.observations:
        try:
            observation = decode_observation(payload)
        except (TypeError, ValueError):
            continue
        if any(
            event.category in _TERMINAL_EVENT_NAMES and event.age_s <= 0.0
            for event in observation.events
        ):
            return True
    return False


class SharedMemorySelfPlayAdapter:
    """把共享内存中的观察/动作批次接入自对弈控制器。"""

    def __init__(
        self,
        *,
        transport: Any,
        runtime: Any | None = None,
        reward_provider: Callable[[Any, Any, int], Mapping[str, Any] | None] | None = None,
        critic_snapshot_provider: Callable[[Any, Any, int], CriticSnapshotV1 | Mapping[str, Any] | None] | None = None,
        done_provider: Callable[[Any, Any, int], bool | Mapping[str, bool]] | None = None,
    ) -> None:
        self.transport = transport
        self.runtime = runtime
        self.reward_provider = reward_provider
        self.critic_snapshot_provider = critic_snapshot_provider
        self.done_provider = done_provider
        self._phase: Phase | None = None
        self._epoch: int | None = None
        self._bot_ids: tuple[str, ...] = ()
        self._event_state = _ObservationRewardState()

    def start_episode(
        self,
        phase: Phase,
        roster_profiles: tuple[RosterProfile, ...],
        epoch: int,
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        bot_profiles = tuple(profile for profile in roster_profiles if not profile.is_human)
        if not bot_profiles:
            raise ValueError("shared-memory self-play requires at least one bot")
        current_epoch = int(self.transport.refresh_epoch())
        if current_epoch != epoch:
            raise ValueError(f"self-play epoch mismatch: expected {current_epoch}, got {epoch}")
        self._phase = Phase(phase)
        self._epoch = int(epoch)
        self._bot_ids = tuple(profile.bot_id for profile in bot_profiles)
        self._event_state.reset()

    def step(self, server_tick: int) -> Mapping[str, Any] | None:
        if self._phase is None or self._epoch is None:
            raise EpisodeNotRunning("self-play adapter has not started an episode")
        observation_batch = self.transport.try_read_observation()
        if observation_batch is None:
            return None
        if observation_batch.bot_count != len(self._bot_ids):
            raise ValueError(
                f"self-play observation bot count mismatch: expected {len(self._bot_ids)}, "
                f"got {observation_batch.bot_count}"
            )
        action_batch = (
            self.runtime.process_observation_batch(observation_batch)
            if self.runtime is not None
            else self.transport.try_read_action()
        )
        if action_batch is None:
            return None
        if action_batch.bot_count != len(self._bot_ids):
            raise ValueError("self-play action batch does not match the active roster")
        observation_server_tick = int(observation_batch.server_tick)
        try:
            observed_phase = Phase(observation_batch.observations[0][5])
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("self-play observation does not contain a valid Get5 phase") from error
        reward_inputs = (
            self.reward_provider(observation_batch, action_batch, observation_server_tick)
            if self.reward_provider is not None
            else _observation_reward_inputs(
                observation_batch,
                action_batch,
                observation_server_tick,
                self._bot_ids,
                event_state=self._event_state,
            )
        )
        critic_snapshot = (
            self.critic_snapshot_provider(observation_batch, action_batch, observation_server_tick)
            if self.critic_snapshot_provider is not None
            else _observation_critic_snapshot(observation_batch, action_batch, observation_server_tick, self._bot_ids)
        )
        done = (
            self.done_provider(observation_batch, action_batch, observation_server_tick)
            if self.done_provider is not None
            else _observation_done(observation_batch, event_state=self._event_state)
        )
        return {
            "epoch": action_batch.epoch,
            "phase": observed_phase,
            "server_tick": action_batch.server_tick,
            "observations": dict(zip(self._bot_ids, observation_batch.observations)),
            "actions": dict(zip(self._bot_ids, action_batch.actions)),
            "decision_actions": dict(zip(self._bot_ids, action_batch.decision_actions)),
            "decision_log_probs": dict(zip(self._bot_ids, action_batch.decision_log_probs)),
            "fallback_bots": {
                bot_id
                for bot_id, action in zip(self._bot_ids, action_batch.actions)
                if action.action_valid_mask == 0
            },
            "reward_inputs": reward_inputs,
            "critic_snapshot": critic_snapshot,
            "done": done,
        }


class SelfPlayController:
    def __init__(
        self,
        *,
        data_root: str | Path,
        purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
        source_manifest: DatasetManifestV1 | None = None,
        adapter: SelfPlayAdapter | None = None,
        reward_weights: RewardWeights = RewardWeights(),
        instance_id: str = "",
        match_id: str = "",
        policy_generation: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.purpose = ensure_purpose(purpose)
        if source_manifest is not None and not isinstance(source_manifest, DatasetManifestV1):
            raise TypeError("source_manifest must be DatasetManifestV1")
        self.source_manifest = source_manifest
        self.adapter = adapter
        self.instance_id = str(instance_id)
        self.match_id = str(match_id)
        if policy_generation < 0:
            raise ValueError("policy_generation must be non-negative")
        self.policy_generation = int(policy_generation)
        self.reward_ledger = RewardLedger(reward_weights)
        self._status = "idle"
        self._match_terminal = False
        self._phase: Phase | None = None
        self._epoch = 0
        self._episode_id = ""
        self._profiles: tuple[RosterProfile, ...] = ()
        self._bot_profiles: tuple[RosterProfile, ...] = ()
        self._batches: list[TransitionBatchV1] = []
        self._critic_snapshots: list[CriticSnapshotV1] = []
        self._completed_manifests: list[TrajectoryManifestV1] = []
        self._completed_envelopes: list[RolloutEnvelopeV1] = []
        self._pending_envelopes: list[RolloutEnvelopeV1] = []
        self._rollout_shard_index = 0
        self._next_hidden_reset = True
        self._boundaries: list[str] = []
        self._pause_reason = "technical"
        self._training_allowed = True

    @property
    def status(self) -> str:
        return self._status

    @property
    def phase(self) -> Phase:
        if self._phase is None:
            raise EpisodeNotRunning("no episode has been started")
        return self._phase

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def bot_ids(self) -> tuple[str, ...]:
        return tuple(profile.bot_id for profile in self._bot_profiles)

    @property
    def human_seat_count(self) -> int:
        explicit_humans = sum(profile.is_human for profile in self._profiles)
        empty_seats = max(0, 10 - len(self._profiles))
        return explicit_humans + empty_seats

    @property
    def boundaries(self) -> tuple[str, ...]:
        return tuple(self._boundaries)

    @property
    def critic_snapshots(self) -> tuple[CriticSnapshotV1, ...]:
        return tuple(self._critic_snapshots)

    @property
    def transitions(self) -> tuple[TransitionBatchV1, ...]:
        segments = [self._load_completed(envelope).transitions for envelope in self._completed_envelopes]
        if self._batches:
            segments.append(tuple(self._batches))
        return tuple(batch for segment in segments for batch in segment)

    @property
    def rollout_segments(self) -> tuple[tuple[TransitionBatchV1, ...], ...]:
        segments = [self._load_completed(envelope).transitions for envelope in self._completed_envelopes]
        if self._batches:
            segments.append(tuple(self._batches))
        return tuple(segments)

    @property
    def critic_snapshot_segments(self) -> tuple[tuple[CriticSnapshotV1, ...], ...]:
        segments = [self._load_completed(envelope).critic_snapshots for envelope in self._completed_envelopes]
        if self._critic_snapshots:
            segments.append(tuple(self._critic_snapshots))
        return tuple(segments)

    @property
    def completed_manifests(self) -> tuple[TrajectoryManifestV1, ...]:
        return tuple(self._completed_manifests)

    @property
    def completed_envelopes(self) -> tuple[RolloutEnvelopeV1, ...]:
        """All atomically written, non-truncated shards for the active match."""

        return tuple(self._completed_envelopes)

    @property
    def match_terminal(self) -> bool:
        """Whether Get5 has reached a terminal map/series boundary."""

        return self._match_terminal

    @staticmethod
    def _load_completed(envelope: RolloutEnvelopeV1) -> Any:
        from .training_mappo import load_rollout_envelope

        return load_rollout_envelope(envelope)

    def start_episode(
        self,
        phase: Phase | int,
        roster_profiles: Sequence[RosterProfile | Mapping[str, Any]],
        *,
        episode_id: str | None = None,
        epoch: int = 0,
    ) -> None:
        if self._status in {"running", "paused"}:
            raise RuntimeError("an episode is already active")
        normalized = tuple(self._normalize_profile(value) for value in roster_profiles)
        if not normalized or len(normalized) > 10:
            raise ValueError("roster must contain between 1 and 10 total seats")
        bot_profiles = tuple(profile for profile in normalized if not profile.is_human)
        if not 1 <= len(bot_profiles) <= MAX_BOTS:
            raise ValueError("roster must contain between 1 and 10 controlled bots")
        if len({profile.bot_id for profile in normalized}) != len(normalized):
            raise ValueError("roster profile ids must be unique")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._phase = Phase(phase)
        self._epoch = int(epoch)
        self._match_terminal = False
        self._episode_id = self._safe_episode_id(episode_id or f"{self._phase.name.lower()}-{self._epoch}")
        self._profiles = normalized
        self._bot_profiles = bot_profiles
        self._batches.clear()
        self._critic_snapshots.clear()
        self._rollout_shard_index = 0
        self._next_hidden_reset = True
        self._boundaries.clear()
        self._pause_reason = "technical"
        self._training_allowed = True
        self.reward_ledger.reset()
        self._status = "running"
        if self.adapter is not None:
            self.adapter.start_episode(self._phase, normalized, self._epoch)

    def apply_boundaries(self, boundaries: Sequence[TrainingBoundaryV1] = ()) -> None:
        for boundary in boundaries:
            self._apply_training_boundary(boundary)
            if self._status == "finished":
                return

    def accept(
        self,
        observation: Any,
        action: Any,
        boundaries: Sequence[TrainingBoundaryV1] = (),
    ) -> None:
        self.apply_boundaries(boundaries)
        if self._status == "finished":
            return
        if self._status == "paused" or not self._training_allowed:
            return
        if self._phase is None:
            raise EpisodeNotRunning("no episode has been started")
        observation_epoch = int(observation.epoch)
        action_epoch = int(action.epoch)
        if observation_epoch != action_epoch or observation_epoch != self._epoch:
            raise ValueError(
                f"self-play batch epoch mismatch: controller={self._epoch}, "
                f"observation={observation_epoch}, action={action_epoch}"
            )
        if int(action.server_tick) != int(observation.server_tick):
            raise ValueError("observation and action server ticks must match")
        try:
            observed_phase = Phase(int(observation.observations[0][5]))
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("self-play observation does not contain a valid Get5 phase") from error
        if observed_phase is not self.phase:
            raise ValueError(
                f"self-play phase mismatch: controller={self.phase.name.lower()}, "
                f"observation={observed_phase.name.lower()}"
            )
        fallback_bots = {
            bot_id
            for bot_id, bot_action in zip(self.bot_ids, action.actions)
            if bot_action.action_valid_mask == 0
        }
        self.step(
            action.actions,
            observations=observation.observations,
            server_tick=int(observation.server_tick),
            fallback_bots=fallback_bots,
        )
        if self._next_hidden_reset and self._batches:
            self._reset_last_hidden_mask()
            self._next_hidden_reset = False

    def drain_completed(self, rollout_horizon: int = 1024) -> tuple[RolloutEnvelopeV1, ...]:
        if not isinstance(rollout_horizon, int) or isinstance(rollout_horizon, bool) or rollout_horizon < 1:
            raise ValueError("rollout_horizon must be a positive integer")
        while len(self._batches) >= rollout_horizon:
            self._flush_rollout_segment()
        return self._drain_pending_envelopes()

    def flush_pending(self, *, truncated: bool = False) -> tuple[RolloutEnvelopeV1, ...]:
        if not isinstance(truncated, bool):
            raise TypeError("truncated must be a boolean")
        if self._batches:
            self._flush_rollout_segment(truncated=truncated)
        return self._drain_pending_envelopes()

    def _apply_training_boundary(self, boundary: TrainingBoundaryV1) -> None:
        if not isinstance(boundary, TrainingBoundaryV1):
            raise TypeError("boundaries must contain TrainingBoundaryV1 values")
        if boundary.epoch < 0 or boundary.server_tick < 0:
            raise ValueError("training boundary epoch and server_tick must be non-negative")
        kind = str(boundary.kind)
        self._boundaries.append(kind)
        if boundary.discard_uncommitted:
            self._discard_current_rollout()
        if kind in {"pause", "resume", "fallback"} and boundary.epoch != self._epoch:
            raise ValueError(
                f"training boundary epoch mismatch: controller={self._epoch}, boundary={boundary.epoch}"
            )
        if kind == "pause":
            if self._batches:
                self._flush_rollout_segment(boundary=True)
            self._status = "paused"
            self._training_allowed = False
            self._next_hidden_reset = True
            return
        if kind == "resume":
            if self._status == "paused":
                self._status = "running"
            self._training_allowed = bool(boundary.training_allowed)
            self._next_hidden_reset = True
            return
        if kind == "backup_restore":
            self._epoch = int(boundary.epoch)
            self.reward_ledger.reset()
            self._status = "running"
            self._training_allowed = bool(boundary.training_allowed)
            self._next_hidden_reset = True
            return
        if kind == "fallback":
            self._status = "running"
            self._training_allowed = False
            self._next_hidden_reset = True
            return
        if boundary.epoch != self._epoch:
            if kind in {"round_start", "map_end", "series_end"}:
                if self._batches:
                    self._flush_rollout_segment(
                        terminal=kind in {"map_end", "series_end"},
                        boundary=True,
                    )
                self._epoch = int(boundary.epoch)
                self.reward_ledger.reset()
            elif not self._batches and kind in {"rules_validated", "going_live", "live"}:
                self._epoch = int(boundary.epoch)
                self.reward_ledger.reset()
            else:
                raise ValueError(
                    f"training boundary epoch mismatch: controller={self._epoch}, boundary={boundary.epoch}"
                )
        if kind in {"map_end", "series_end"} or boundary.terminal:
            if self._batches:
                self._flush_rollout_segment(terminal=True, boundary=True)
            self._match_terminal = True
            self._status = "finished"
            self._training_allowed = False
            self._next_hidden_reset = True
            return
        if kind in {"round_end", "halftime", "overtime_start", "round_start"} or boundary.reset_hidden:
            if self._batches:
                self._flush_rollout_segment(boundary=True)
            self._next_hidden_reset = True
        self._training_allowed = bool(boundary.training_allowed)

    def _drain_pending_envelopes(self) -> tuple[RolloutEnvelopeV1, ...]:
        envelopes = tuple(self._pending_envelopes)
        self._pending_envelopes.clear()
        return envelopes

    def _flush_rollout_segment(
        self,
        *,
        terminal: bool = False,
        truncated: bool = False,
        boundary: bool = False,
    ) -> TrajectoryManifestV1 | None:
        if terminal and truncated:
            raise ValueError("rollout segment terminal and truncated cannot both be true")
        if not self._batches:
            return None
        if self._phase is None:
            raise EpisodeNotRunning("no episode has been started")
        if boundary:
            self._close_segment_at_boundary()
        phase_name = self._phase.name.lower()
        phase_root = self.data_root / "trajectories" / phase_name
        critic_root = self.data_root / "critics" / phase_name
        shard_id = f"{self._episode_id}.shard-{self._rollout_shard_index:06d}"
        path = phase_root / f"{shard_id}.parquet"
        critic_path = critic_root / f"{shard_id}.parquet"
        trajectory_manifest_path = path.with_name(f"{path.stem}.manifest.json")
        critic_manifest_path = critic_path.with_name(f"{critic_path.stem}.manifest.json")
        metadata = {
            "schema": TRAJECTORY_SCHEMA_VERSION,
            "phase": phase_name,
            "episode_id": self._episode_id,
            "instance_id": self.instance_id or "default",
            "match_id": self.match_id or self._episode_id,
            "policy_generation": str(self.policy_generation),
            "shard_index": str(self._rollout_shard_index),
            "terminal": str(bool(terminal)).lower(),
            "truncated": str(bool(truncated)).lower(),
            "actor_critic_separate": "true",
        }
        parents = () if self.source_manifest is None else (self.source_manifest,)
        created: list[Path] = []
        try:
            self._atomic_write(
                path,
                lambda temporary: self._write_trajectory(temporary, metadata=metadata),
            )
            created.append(path)
            self._atomic_write(
                critic_path,
                lambda temporary: self._write_critic(temporary, metadata=metadata),
            )
            created.append(critic_path)
            shard_sha256 = self._sha256_file(path)
            critic_sha256 = self._sha256_file(critic_path)
            dataset = DatasetManifestV1(
                name=f"trajectory:{shard_id}",
                purpose=self.purpose,
                parents=parents,
                artifact_type="trajectory",
                metadata={**metadata, "shard_sha256": shard_sha256},
            )
            critic_dataset = DatasetManifestV1(
                name=f"critic:{shard_id}",
                purpose=self.purpose,
                parents=parents,
                artifact_type="critic_trajectory",
                metadata={**metadata, "critic_sha256": critic_sha256, "actor_observations": "excluded"},
            )
            manifest = TrajectoryManifestV1(
                episode_id=shard_id,
                phase=self._phase,
                path=path,
                critic_path=critic_path,
                dataset=dataset,
                critic_dataset=critic_dataset,
                transition_count=sum(len(batch.bot_ids) for batch in self._batches),
            )
            self._atomic_write_json(trajectory_manifest_path, self._manifest_payload(dataset))
            created.append(trajectory_manifest_path)
            self._atomic_write_json(critic_manifest_path, self._manifest_payload(critic_dataset))
            created.append(critic_manifest_path)
        except Exception:
            for created_path in created:
                try:
                    created_path.unlink()
                except FileNotFoundError:
                    pass
            self._remove_empty_directories(phase_root)
            self._remove_empty_directories(critic_root)
            raise
        envelope = RolloutEnvelopeV1(
            instance_id=self.instance_id or "default",
            match_id=self.match_id or self._episode_id,
            policy_generation=self.policy_generation,
            phase=self._phase,
            shard_path=path,
            critic_path=critic_path,
            terminal=bool(terminal),
            truncated=bool(truncated),
        )
        self._completed_manifests.append(manifest)
        self._completed_envelopes.append(envelope)
        self._pending_envelopes.append(envelope)
        self._batches.clear()
        self._critic_snapshots.clear()
        self._rollout_shard_index += 1
        self._next_hidden_reset = True
        if terminal:
            self._match_terminal = True
            self._status = "finished"
            self._training_allowed = False
        return manifest

    def _discard_current_rollout(self) -> None:
        self._batches.clear()
        self._critic_snapshots.clear()
        self.reward_ledger.reset()
        self._next_hidden_reset = True

    def _reset_last_hidden_mask(self) -> None:
        if not self._batches:
            return
        last = self._batches[-1]
        self._batches[-1] = replace(
            last,
            hidden_state_mask=tuple(False for _ in last.hidden_state_mask),
        )

    @staticmethod
    def _manifest_payload(manifest: DatasetManifestV1) -> dict[str, Any]:
        return {
            "schema": "dataset-manifest-v1",
            "name": manifest.name,
            "purpose": manifest.purpose.value,
            "artifact_type": manifest.artifact_type,
            "source_sha256": manifest.source_sha256,
            "parser_version": manifest.parser_version,
            "projection_version": manifest.projection_version,
            "parents": [SelfPlayController._manifest_payload(parent) for parent in manifest.parents],
            "metadata": {str(key): str(value) for key, value in manifest.metadata.items()},
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite completed rollout artifact: {path}")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            writer(temporary)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            SelfPlayController._fsync_directory(path.parent)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        def write_json(temporary: Path) -> None:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")

        SelfPlayController._atomic_write(path, write_json)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
            return
        try:
            descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _remove_empty_directories(self, directory: Path) -> None:
        current = Path(directory)
        while current != self.data_root and current != current.parent:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def step(
        self,
        actions: Mapping[str, BotActionV1 | None] | Sequence[BotActionV1 | None] | None,
        *,
        observations: Mapping[str, bytes] | Sequence[bytes] | None = None,
        server_tick: int | None = None,
        reward_inputs: Mapping[str, RewardSignals | Mapping[str, float]] | None = None,
        done: bool | Mapping[str, bool] = False,
        fallback_bots: set[str] | frozenset[str] | None = None,
        critic_snapshot: CriticSnapshotV1 | Mapping[str, Any] | None = None,
        decision_actions: Mapping[str, tuple[int, int, int] | None]
        | Sequence[tuple[int, int, int] | None]
        | None = None,
        decision_log_probs: Mapping[str, float | None] | Sequence[float | None] | None = None,
        delta_time_s: float = 1.0 / 128.0,
    ) -> TransitionBatchV1 | None:
        self._require_running()
        if server_tick is None:
            server_tick = self._batches[-1].server_tick + 1 if self._batches else 0
        if server_tick < 0:
            raise ValueError("server_tick must be non-negative")
        adapter_data = self.adapter.step(server_tick) if self.adapter is not None else None
        if adapter_data is not None:
            adapter_epoch = int(adapter_data.get("epoch", self._epoch))
            adapter_phase = Phase(adapter_data.get("phase", self.phase))
            if adapter_epoch != self._epoch or adapter_phase is not self.phase:
                self._restart_segment(adapter_phase, adapter_epoch)
            server_tick = int(adapter_data.get("server_tick", server_tick))
            if observations is None:
                observations = adapter_data.get("observations")
            if actions is None:
                actions = adapter_data.get("actions")
            if fallback_bots is None:
                fallback_bots = set(adapter_data.get("fallback_bots", ()))
            if reward_inputs is None:
                reward_inputs = adapter_data.get("reward_inputs")
            if critic_snapshot is None:
                critic_snapshot = adapter_data.get("critic_snapshot")
            if decision_actions is None:
                decision_actions = adapter_data.get("decision_actions")
            if decision_log_probs is None:
                decision_log_probs = adapter_data.get("decision_log_probs")
            if done is False and "done" in adapter_data:
                done = adapter_data["done"]
        if actions is None:
            if self.adapter is not None and adapter_data is None:
                return None
            raise ValueError("actions are required when the adapter did not provide an action batch")
        bot_ids = self.bot_ids
        action_values = self._align_values(actions, bot_ids, "actions")
        observation_values = self._align_observations(observations, bot_ids)
        decision_action_values = self._align_values(
            decision_actions if decision_actions is not None else tuple(None for _ in bot_ids),
            bot_ids,
            "decision_actions",
        )
        decision_log_prob_values = self._align_values(
            decision_log_probs if decision_log_probs is not None else tuple(None for _ in bot_ids),
            bot_ids,
            "decision_log_probs",
        )
        fallback = set(fallback_bots or ())
        unknown_fallback = fallback - set(bot_ids)
        if unknown_fallback:
            raise ValueError(f"fallback_bots contains unknown ids: {sorted(unknown_fallback)}")
        normalized_actions: list[BotActionV1] = []
        state_faults: list[bool] = []
        for bot_id, action in zip(bot_ids, action_values):
            if bot_id in fallback or action is None:
                normalized_actions.append(self._neutral_action(server_tick + 1))
                state_faults.append(True)
            elif not isinstance(action, BotActionV1):
                raise TypeError("actions must contain BotActionV1 or None")
            else:
                normalized_actions.append(action)
                state_faults.append(False)
        rewards = tuple(
            self.reward_ledger.update(
                bot_id,
                (reward_inputs or {}).get(bot_id),
                delta_time_s=delta_time_s,
                done=self._done_for(done, bot_id),
            )
            for bot_id in bot_ids
        )
        dones = tuple(self._done_for(done, bot_id) for bot_id in bot_ids)
        batch = TransitionBatchV1(
            epoch=self._epoch,
            phase=self.phase,
            server_tick=int(server_tick),
            bot_ids=bot_ids,
            observations=tuple(observation_values),
            actions=tuple(normalized_actions),
            rewards=rewards,
            dones=dones,
            state_faults=tuple(state_faults),
            hidden_state_mask=tuple(not fault for fault in state_faults),
            observation_metadata={"schema": TRAJECTORY_SCHEMA_VERSION, "phase": self.phase.name.lower()},
            decision_actions=tuple(decision_action_values),
            decision_log_probs=tuple(decision_log_prob_values),
        )
        self._batches.append(batch)
        if self._next_hidden_reset:
            self._reset_last_hidden_mask()
            self._next_hidden_reset = False
        if critic_snapshot is not None:
            snapshot = (
                critic_snapshot
                if isinstance(critic_snapshot, CriticSnapshotV1)
                else CriticSnapshotV1(int(server_tick), critic_snapshot)
            )
            self._critic_snapshots.append(snapshot)
        return batch

    def _restart_segment(self, phase: Phase, epoch: int) -> None:
        if self._batches:
            self._close_segment_at_boundary()
            self.finish_episode()
        else:
            self._status = "finished"
        segment_number = len(self._completed_manifests) + 1
        episode_id = f"{self._episode_id}-{Phase(phase).name.lower()}-{epoch}-{segment_number}"
        self.start_episode(
            phase,
            self._profiles,
            episode_id=episode_id,
            epoch=epoch,
        )

    def _close_segment_at_boundary(self) -> None:
        if not self._batches:
            return
        last = self._batches[-1]
        if all(last.dones):
            return
        self._batches[-1] = TransitionBatchV1(
            epoch=last.epoch,
            phase=last.phase,
            server_tick=last.server_tick,
            bot_ids=last.bot_ids,
            observations=last.observations,
            actions=last.actions,
            rewards=last.rewards,
            dones=tuple(True for _ in last.dones),
            state_faults=last.state_faults,
            hidden_state_mask=last.hidden_state_mask,
            observation_metadata=last.observation_metadata,
            decision_actions=last.decision_actions,
            decision_log_probs=last.decision_log_probs,
        )

    def pause(self, reason: str = "technical") -> None:
        self._require_running()
        if not reason:
            raise ValueError("pause reason cannot be empty")
        self._pause_reason = reason
        self._status = "paused"
        self._boundaries.append(f"{reason}:pause")

    def resume(self) -> None:
        if self._status != "paused":
            raise RuntimeError("episode is not paused")
        self._status = "running"
        self._boundaries.append(f"{self._pause_reason}:resume")

    def reload_round(self, *, epoch: int | None = None) -> None:
        self._require_running()
        if epoch is not None:
            if epoch < 0:
                raise ValueError("epoch must be non-negative")
            self._epoch = int(epoch)
        else:
            self._epoch += 1
        self.reward_ledger.reset()
        self._boundaries.append("reload")

    def record_boundary(self, name: str) -> None:
        self._require_running()
        if name not in {"halftime", "overtime"}:
            raise ValueError("boundary must be halftime or overtime")
        self._boundaries.append(name)

    def finish_episode(self) -> TrajectoryManifestV1:
        self._require_running()
        if self._phase is None:
            raise EpisodeNotRunning("no episode has been started")
        phase_name = self._phase.name.lower()
        phase_root = self.data_root / "trajectories" / phase_name
        critic_root = self.data_root / "critics" / phase_name
        phase_root.mkdir(parents=True, exist_ok=True)
        critic_root.mkdir(parents=True, exist_ok=True)
        path = phase_root / f"{self._episode_id}.parquet"
        critic_path = critic_root / f"{self._episode_id}.parquet"
        self._write_trajectory(path)
        self._write_critic(critic_path)
        parents = () if self.source_manifest is None else (self.source_manifest,)
        metadata = {
            "schema": TRAJECTORY_SCHEMA_VERSION,
            "phase": phase_name,
            "episode_id": self._episode_id,
            "bot_count": str(len(self._bot_profiles)),
            "human_seat_count": str(self.human_seat_count),
            "boundaries": ",".join(self._boundaries),
            "actor_critic_separate": "true",
        }
        shard_sha256 = self._sha256_file(path)
        critic_sha256 = self._sha256_file(critic_path)
        metadata["shard_sha256"] = shard_sha256
        critic_metadata = {
            key: value for key, value in metadata.items() if key != "shard_sha256"
        }
        critic_metadata.update({"critic_sha256": critic_sha256, "actor_observations": "excluded"})
        dataset = DatasetManifestV1(
            name=f"trajectory:{self._episode_id}",
            purpose=self.purpose,
            parents=parents,
            artifact_type="trajectory",
            metadata=metadata,
        )
        critic_dataset = DatasetManifestV1(
            name=f"critic:{self._episode_id}",
            purpose=self.purpose,
            parents=parents,
            artifact_type="critic_trajectory",
            metadata=critic_metadata,
        )
        self._atomic_write_json(path.with_name(f"{path.stem}.manifest.json"), self._manifest_payload(dataset))
        self._atomic_write_json(critic_path.with_name(f"{critic_path.stem}.manifest.json"), self._manifest_payload(critic_dataset))
        manifest = TrajectoryManifestV1(
            episode_id=self._episode_id,
            phase=self._phase,
            path=path,
            critic_path=critic_path,
            dataset=dataset,
            critic_dataset=critic_dataset,
            transition_count=sum(len(batch.bot_ids) for batch in self._batches),
        )
        envelope = RolloutEnvelopeV1(
            instance_id=self.instance_id or "default",
            match_id=self.match_id or self._episode_id,
            policy_generation=self.policy_generation,
            phase=self._phase,
            shard_path=path,
            critic_path=critic_path,
            terminal=True,
            truncated=False,
        )
        self._completed_manifests.append(manifest)
        self._completed_envelopes.append(envelope)
        self._pending_envelopes.append(envelope)
        self._batches.clear()
        self._critic_snapshots.clear()
        self._status = "finished"
        return manifest

    def _require_running(self) -> None:
        if self._status == "paused":
            raise EpisodePaused("episode is paused")
        if self._status != "running":
            raise EpisodeNotRunning("episode is not running")

    @staticmethod
    def _normalize_profile(value: RosterProfile | Mapping[str, Any]) -> RosterProfile:
        if isinstance(value, RosterProfile):
            return value
        if isinstance(value, Mapping):
            return RosterProfile(
                bot_id=str(value["bot_id"]),
                team=int(value.get("team", 0)),
                is_human=bool(value.get("is_human", False)),
                difficulty=str(value.get("difficulty", "normal")),
                personality=tuple(float(item) for item in value.get("personality", ())),
            )
        raise TypeError("roster_profiles must contain RosterProfile or mapping values")

    @staticmethod
    def _safe_episode_id(value: str) -> str:
        normalized = _EPISODE_ID_RE.sub("-", value).strip("-.")
        if not normalized:
            raise ValueError("episode_id cannot be empty")
        return normalized

    @staticmethod
    def _neutral_action(target_tick: int) -> BotActionV1:
        return BotActionV1(
            target_tick=target_tick,
            forward=0.0,
            side=0.0,
            up=0.0,
            yaw_delta_deg=0.0,
            pitch_delta_deg=0.0,
            buttons=0,
            weapon_select=-1,
            buy_action=0,
            action_valid_mask=0xFF,
        )

    @staticmethod
    def _align_values(values: Mapping[str, Any] | Sequence[Any], ids: tuple[str, ...], name: str) -> tuple[Any, ...]:
        if isinstance(values, Mapping):
            unknown = set(values) - set(ids)
            if unknown:
                raise ValueError(f"{name} contains unknown ids: {sorted(unknown)}")
            return tuple(values.get(bot_id) for bot_id in ids)
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a mapping or sequence")
        values = tuple(values)
        if len(values) != len(ids):
            raise ValueError(f"{name} must contain one value per bot")
        return values

    def _align_observations(
        self,
        values: Mapping[str, bytes] | Sequence[bytes] | None,
        ids: tuple[str, ...],
    ) -> tuple[bytes, ...]:
        if values is None:
            values = {bot_id: bytes(OBSERVATION_BYTES) for bot_id in ids}
        aligned = self._align_values(values, ids, "observations")
        return tuple(bytes(value) for value in aligned)

    @staticmethod
    def _done_for(done: bool | Mapping[str, bool], bot_id: str) -> bool:
        if isinstance(done, Mapping):
            return bool(done.get(bot_id, False))
        return bool(done)

    def _write_trajectory(self, path: Path, *, metadata: Mapping[str, str] | None = None) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            (batch, index)
            for batch in self._batches
            for index in range(len(batch.bot_ids))
        ]
        table_metadata = {
            b"schema": TRAJECTORY_SCHEMA_VERSION.encode(),
            b"actor_observation_excludes_critic": b"true",
        }
        if metadata:
            table_metadata.update(
                {str(key).encode(): str(value).encode() for key, value in metadata.items()}
            )
        table = pa.table(
            {
                "episode_id": [self._episode_id for _, _ in rows],
                "phase": [batch.phase.name.lower() for batch, _ in rows],
                "epoch": [batch.epoch for batch, _ in rows],
                "server_tick": [batch.server_tick for batch, _ in rows],
                "bot_id": [batch.bot_ids[index] for batch, index in rows],
                "observation": [batch.observations[index] for batch, index in rows],
                "action": [batch.actions[index].to_bytes() for batch, index in rows],
                "reward": [batch.rewards[index] for batch, index in rows],
                "done": [batch.dones[index] for batch, index in rows],
                "state_fault": [batch.state_faults[index] for batch, index in rows],
                "hidden_state_mask": [batch.hidden_state_mask[index] for batch, index in rows],
                "decision_action_tactical": [
                    None if batch.decision_actions[index] is None else batch.decision_actions[index][0]
                    for batch, index in rows
                ],
                "decision_action_task": [
                    None if batch.decision_actions[index] is None else batch.decision_actions[index][1]
                    for batch, index in rows
                ],
                "decision_action_target": [
                    None if batch.decision_actions[index] is None else batch.decision_actions[index][2]
                    for batch, index in rows
                ],
                "decision_log_prob": [
                    batch.decision_log_probs[index] for batch, index in rows
                ],
            },
            metadata=table_metadata,
        )
        pq.write_table(table, path, compression="zstd")

    def _write_critic(self, path: Path, *, metadata: Mapping[str, str] | None = None) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table_metadata = {b"schema": b"critic-trajectory-v1", b"actor_observations": b"excluded"}
        if metadata:
            table_metadata.update(
                {str(key).encode(): str(value).encode() for key, value in metadata.items()}
            )
        table = pa.table(
            {
                "episode_id": [self._episode_id for snapshot in self._critic_snapshots],
                "phase": [self.phase.name.lower() for snapshot in self._critic_snapshots],
                "server_tick": [snapshot.server_tick for snapshot in self._critic_snapshots],
                "snapshot_json": [
                    json.dumps(snapshot.values, sort_keys=True, separators=(",", ":"))
                    for snapshot in self._critic_snapshots
                ],
            },
            metadata=table_metadata,
        )
        pq.write_table(table, path, compression="zstd")
