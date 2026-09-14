from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


@dataclass(frozen=True)
class RewardWeights:
    kill: float = 1.0
    assist: float = 0.25
    death: float = -1.0
    damage_dealt: float = 0.01
    damage_taken: float = -0.005
    objective: float = 1.0
    utility_damage: float = 0.02
    utility_success: float = 0.25
    economy: float = 0.001
    team_win: float = 2.0
    team_loss: float = -2.0
    teammate_loss: float = -0.15
    survival_per_second: float = 0.01


@dataclass(frozen=True)
class RewardSignals:
    kills: float = 0.0
    assists: float = 0.0
    deaths: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    objectives: float = 0.0
    utility_damage: float = 0.0
    utility_success: float = 0.0
    economy_delta: float = 0.0
    team_won: float = 0.0
    team_lost: float = 0.0
    teammates_lost: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> "RewardSignals":
        if values is None:
            return cls()
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown reward signal(s): {sorted(unknown)}")
        return cls(**{name: float(value) for name, value in values.items()})

    def delta(self, previous: "RewardSignals | None") -> "RewardSignals":
        if previous is None:
            return self
        return RewardSignals(
            **{
                field.name: getattr(self, field.name) - getattr(previous, field.name)
                for field in fields(self)
            }
        )


def compute_reward(
    current: RewardSignals | Mapping[str, float] | None = None,
    previous: RewardSignals | Mapping[str, float] | None = None,
    *,
    delta_time_s: float = 1.0 / 128.0,
    weights: RewardWeights = RewardWeights(),
) -> float:
    if delta_time_s < 0.0:
        raise ValueError("delta_time_s must be non-negative")
    current_signals = (
        current if isinstance(current, RewardSignals) else RewardSignals.from_mapping(current)
    )
    previous_signals = (
        None
        if previous is None
        else previous
        if isinstance(previous, RewardSignals)
        else RewardSignals.from_mapping(previous)
    )
    signal = current_signals.delta(previous_signals)
    return (
        signal.kills * weights.kill
        + signal.assists * weights.assist
        + signal.deaths * weights.death
        + signal.damage_dealt * weights.damage_dealt
        + signal.damage_taken * weights.damage_taken
        + signal.objectives * weights.objective
        + signal.utility_damage * weights.utility_damage
        + signal.utility_success * weights.utility_success
        + signal.economy_delta * weights.economy
        + signal.team_won * weights.team_win
        + signal.team_lost * weights.team_loss
        + signal.teammates_lost * weights.teammate_loss
        + delta_time_s * weights.survival_per_second
    )


class RewardLedger:
    def __init__(self, weights: RewardWeights = RewardWeights()) -> None:
        self.weights = weights
        self._previous: dict[str, RewardSignals] = {}

    def update(
        self,
        bot_id: str,
        signals: RewardSignals | Mapping[str, float] | None,
        *,
        delta_time_s: float = 1.0 / 128.0,
        done: bool = False,
    ) -> float:
        current = signals if isinstance(signals, RewardSignals) else RewardSignals.from_mapping(signals)
        reward = compute_reward(
            current,
            self._previous.get(bot_id),
            delta_time_s=delta_time_s,
            weights=self.weights,
        )
        self._previous[bot_id] = current
        return reward

    def reset(self) -> None:
        self._previous.clear()
