"""Structured hierarchical intents, offline pseudo-labels and action guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import struct
from typing import Any

from .action_reconstruction import (
    ACTION_MASK_BUY,
    IN_ATTACK,
    IN_ATTACK2,
    ActionLabelV1,
)
from .contracts import (
    BotActionV1,
    BotObservationV1,
    DemoEventV1,
    DecisionOutputV1,
    INTENT_FIELD_NAMES,
    INTENT_TASKS,
    IntentLabelV1,
    IntentV1,
    TACTICAL_MODES,
    TARGET_SLOT_NONE,
)


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _clamp(value: Any, lower: float, upper: float, fallback: float = 0.0) -> float:
    return max(lower, min(upper, _finite(value, fallback)))


def _vector(value: Any, size: int) -> tuple[float, ...] | None:
    if isinstance(value, Mapping):
        value = tuple(value.get(axis) for axis in ("x", "y", "z")[:size])
    elif not isinstance(value, Sequence) and hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) < size:
        return None
    return tuple(_finite(item) for item in value[:size])


def _position(observation: Any) -> tuple[float, float, float] | None:
    for name in ("self_position", "position"):
        value = getattr(observation, name, None)
        vector = _vector(value, 3)
        if vector is not None:
            return vector  # type: ignore[return-value]
    if isinstance(observation, Mapping):
        for name in ("self_position", "position"):
            vector = _vector(observation.get(name), 3)
            if vector is not None:
                return vector  # type: ignore[return-value]
    if isinstance(observation, BotObservationV1):
        try:
            from .observation_projection import decode_observation

            return decode_observation(observation).self_position
        except (ValueError, TypeError, struct.error):
            return None
    return None


def _event_type(event: Any) -> str:
    if isinstance(event, DemoEventV1):
        return event.event_type.lower()
    if isinstance(event, Mapping):
        return str(event.get("event_type", event.get("type", ""))).lower()
    return str(getattr(event, "event_type", getattr(event, "type", ""))).lower()


def _event_payload(event: Any) -> Mapping[str, object]:
    if isinstance(event, DemoEventV1):
        return event.payload
    if isinstance(event, Mapping):
        return event
    payload = getattr(event, "payload", {})
    return payload if isinstance(payload, Mapping) else {}


def _events_at(events: Sequence[Any], index: int, tick: int) -> tuple[Any, ...]:
    if index < len(events):
        candidate = events[index]
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray, Mapping)):
            return tuple(candidate)
    selected = []
    for event in events:
        event_tick = getattr(event, "tick", None)
        if isinstance(event, Mapping):
            event_tick = event.get("tick", event.get("server_tick"))
        if event_tick is None or int(event_tick) == tick:
            selected.append(event)
    return tuple(selected)


def canonicalize_intent(output: DecisionOutputV1 | IntentV1 | Mapping[str, Any]) -> IntentV1:
    """Convert an untrusted decision output into the bounded IntentV1 contract."""

    def get(name: str, default: Any) -> Any:
        if isinstance(output, Mapping):
            return output.get(name, default)
        return getattr(output, name, default)

    modes = str(get("tactical_mode", "hold"))
    tasks = str(get("task", "move"))
    goal = _vector(get("goal_position", (0.0, 0.0, 0.0)), 3) or (0.0, 0.0, 0.0)
    waypoint = _vector(get("waypoint_position", (0.0, 0.0, 0.0)), 3) or (0.0, 0.0, 0.0)
    facing = _vector(get("facing_yaw_pitch", (0.0, 0.0)), 2) or (0.0, 0.0)
    yaw = ((facing[0] + 180.0) % 360.0) - 180.0
    if yaw == -180.0 and facing[0] > 0.0:
        yaw = 180.0
    target_slot = int(get("target_slot", TARGET_SLOT_NONE))
    if not 0 <= target_slot <= TARGET_SLOT_NONE:
        target_slot = TARGET_SLOT_NONE
    confidence = get("confidence", {})
    valid_mask = get("valid_mask", {})
    return IntentV1(
        tactical_mode=modes if modes in TACTICAL_MODES else "hold",
        task=tasks if tasks in INTENT_TASKS else "move",
        goal_position=goal,
        waypoint_position=waypoint,
        facing_yaw_pitch=(yaw, _clamp(facing[1], -89.0, 89.0)),
        desired_range=_clamp(get("desired_range", 0.0), 0.0, 4096.0),
        target_slot=target_slot,
        aggression=_clamp(get("aggression", 0.0), 0.0, 1.0),
        risk=_clamp(get("risk", 0.0), 0.0, 1.0),
        priority=_clamp(get("priority", 0.0), 0.0, 1.0),
        ttl_ticks=max(0, int(get("ttl_ticks", 0))),
        confidence=confidence if isinstance(confidence, Mapping) else {},
        valid_mask=valid_mask if isinstance(valid_mask, Mapping) else {},
    )


def _action_value(action: Any, name: str, default: Any = 0) -> Any:
    return getattr(action, name, default)


_THROWABLE_WEAPON_INDICES = frozenset({11, 12, 13, 14, 15})


def _is_throwable_weapon(value: Any) -> bool:
    if isinstance(value, int) and value in _THROWABLE_WEAPON_INDICES:
        return True
    normalized = str(value or "").strip().lower()
    return "grenade" in normalized or "flashbang" in normalized


def guard_action(
    intent: IntentV1,
    action: BotActionV1,
    *,
    observation: BotObservationV1 | None = None,
) -> BotActionV1:
    """Mask macro actions that the current intent does not authorize."""

    if not isinstance(intent, IntentV1):
        raise TypeError("intent must be IntentV1")
    if not isinstance(action, BotActionV1):
        raise TypeError("action must be BotActionV1")
    buttons = action.buttons
    weapon_select = action.weapon_select
    buy_action = action.buy_action
    valid_mask = intent.valid_mask
    if buy_action and not bool(valid_mask.get("buy_action", False)):
        buy_action = 0
        action_mask = action.action_valid_mask & ~ACTION_MASK_BUY
    else:
        action_mask = action.action_valid_mask
    utility_allowed = bool(valid_mask.get("utility_action", intent.task == "use_utility"))
    if buttons & IN_ATTACK2 and not utility_allowed:
        buttons &= ~IN_ATTACK2
    selected_throwable = _is_throwable_weapon(weapon_select)
    current_throwable = False
    if observation is not None:
        if not isinstance(observation, BotObservationV1):
            raise TypeError("observation must be BotObservationV1 when provided")
        try:
            from .observation_projection import decode_observation

            current_throwable = _is_throwable_weapon(
                decode_observation(observation).weapon_name
            )
        except (TypeError, ValueError, struct.error):
            current_throwable = False
    if not utility_allowed and (selected_throwable or current_throwable):
        # Both primary and secondary attack can throw a grenade in CS:GO.  A
        # non-utility intent must not select a throwable and then fire it.
        buttons &= ~(IN_ATTACK | IN_ATTACK2)
        if selected_throwable:
            weapon_select = -1
    if weapon_select >= 0 and not bool(valid_mask.get("weapon_select", True)):
        weapon_select = -1
    return BotActionV1(
        target_tick=action.target_tick,
        forward=action.forward,
        side=action.side,
        up=action.up,
        yaw_delta_deg=action.yaw_delta_deg,
        pitch_delta_deg=action.pitch_delta_deg,
        buttons=buttons,
        weapon_select=weapon_select,
        buy_action=buy_action,
        action_valid_mask=action_mask,
    )


def guard_friendly_fire(observation: BotObservationV1, action: BotActionV1) -> BotActionV1:
    """Remove a direct-fire command when its resulting aim is on a teammate."""

    if not isinstance(observation, BotObservationV1):
        raise TypeError("observation must be BotObservationV1")
    if not isinstance(action, BotActionV1):
        raise TypeError("action must be BotActionV1")
    if not action.buttons & IN_ATTACK:
        return action

    payload = observation.to_bytes()
    for slot in range(9):
        offset = 44 + slot * 12
        relation, flags, bearing_cdeg, elevation_deg = struct.unpack_from(
            "<bBhb", payload, offset + 2
        )
        if relation != 1 or not (flags & (1 << 0)) or not (flags & (1 << 3)):
            continue
        bearing = (float(bearing_cdeg) / 100.0 - float(action.yaw_delta_deg) + 180.0) % 360.0 - 180.0
        elevation = float(elevation_deg) - float(action.pitch_delta_deg)
        if abs(bearing) <= 10.0 and abs(elevation) <= 10.0:
            return BotActionV1(
                target_tick=action.target_tick,
                forward=action.forward,
                side=action.side,
                up=action.up,
                yaw_delta_deg=action.yaw_delta_deg,
                pitch_delta_deg=action.pitch_delta_deg,
                buttons=action.buttons & ~IN_ATTACK,
                weapon_select=action.weapon_select,
                buy_action=action.buy_action,
                action_valid_mask=action.action_valid_mask,
            )
    return action


@dataclass(frozen=True)
class IntentStatisticsV1:
    """Frozen train-only normalization and class weights for intent labels."""

    fitted_split: str
    position_min: tuple[float, float, float]
    position_max: tuple[float, float, float]
    category_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fitted_split != "train":
            raise ValueError("intent statistics must be fitted from the train split")
        if len(self.position_min) != 3 or len(self.position_max) != 3:
            raise ValueError("position statistics must contain three values")
        object.__setattr__(
            self,
            "category_weights",
            {key: dict(value) for key, value in self.category_weights.items()},
        )

    def normalize_position(self, position: Sequence[float]) -> tuple[float, float, float]:
        if len(position) != 3:
            raise ValueError("position must contain three values")
        normalized = []
        for value, lower, upper in zip(position, self.position_min, self.position_max):
            span = upper - lower
            normalized.append(0.0 if span <= 0.0 else (float(value) - lower) / span)
        return tuple(normalized)  # type: ignore[return-value]

    def assert_can_apply(self, split: str) -> None:
        if str(split) not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")


def fit_intent_statistics(
    labels: Sequence[IntentLabelV1],
    *,
    split: str,
) -> IntentStatisticsV1:
    if str(split) != "train":
        raise ValueError("intent statistics can only be fitted from the train split")
    if not labels:
        raise ValueError("cannot fit intent statistics from an empty split")
    counts: dict[str, dict[str, int]] = {"tactical_mode": {}, "task": {}}
    positions = []
    for label in labels:
        counts["tactical_mode"][label.tactical_mode] = counts["tactical_mode"].get(label.tactical_mode, 0) + 1
        counts["task"][label.task] = counts["task"].get(label.task, 0) + 1
        if label.valid_mask.get("goal_position", True):
            positions.extend((label.goal_position, label.waypoint_position))
    if not positions:
        positions = [(0.0, 0.0, 0.0)]
    weights = {
        name: {
            category: len(values) / (len(categories) * count)
            for category, count in categories.items()
        }
        for name, categories in counts.items()
        for values in (labels,)
    }
    return IntentStatisticsV1(
        fitted_split="train",
        position_min=tuple(min(position[index] for position in positions) for index in range(3)),
        position_max=tuple(max(position[index] for position in positions) for index in range(3)),
        category_weights=weights,
    )


def _mode_and_task(events: Sequence[Any], action: Any) -> tuple[str, str]:
    event_types = {_event_type(event) for event in events}
    if any("defuse" in name for name in event_types):
        return "defuse", "defuse"
    if any("plant" in name for name in event_types):
        return "plant", "plant"
    if any("bomb" in name for name in event_types):
        return "defend_bomb", "support"
    buttons = int(_action_value(action, "buttons", 0))
    if buttons & IN_ATTACK:
        return "advance", "engage"
    if abs(float(_action_value(action, "forward", 0.0))) + abs(float(_action_value(action, "side", 0.0))) > 0.1:
        return "advance", "move"
    return "hold", "anchor"


def label_intent_sequence(
    observations: Sequence[BotObservationV1],
    actions: Sequence[ActionLabelV1],
    events: Sequence[DemoEventV1],
    *,
    tickrate: int = 128,
) -> tuple[IntentLabelV1, ...]:
    """Create future-derived labels without modifying or enriching observations."""

    if tickrate <= 0:
        raise ValueError("tickrate must be positive")
    if len(actions) not in {0, len(observations)}:
        raise ValueError("observations and actions must have equal lengths")
    positions = tuple(_position(observation) for observation in observations)
    labels: list[IntentLabelV1] = []
    for index, observation in enumerate(observations):
        current_tick = int(getattr(observation, "server_tick", index))
        waypoint_index = min(len(observations) - 1, index + tickrate)
        goal_index = min(len(observations) - 1, index + tickrate * 5)
        waypoint = positions[waypoint_index]
        goal = positions[goal_index]
        current_events = _events_at(events, index, current_tick)
        action = actions[index] if actions else None
        mode, task = _mode_and_task(current_events, action)
        valid_goal = goal is not None
        valid_waypoint = waypoint is not None
        target_slot = TARGET_SLOT_NONE
        for event in current_events:
            payload = _event_payload(event)
            try:
                candidate = int(payload.get("target_slot", TARGET_SLOT_NONE))
            except (TypeError, ValueError):
                candidate = TARGET_SLOT_NONE
            if 0 <= candidate <= TARGET_SLOT_NONE:
                target_slot = candidate
                break
        labels.append(
            IntentLabelV1(
                target_tick=current_tick,
                tactical_mode=mode,
                task=task,
                goal_position=goal or (0.0, 0.0, 0.0),
                waypoint_position=waypoint or (0.0, 0.0, 0.0),
                facing_yaw_pitch=(0.0, 0.0),
                desired_range=0.0,
                target_slot=target_slot,
                aggression=1.0 if task == "engage" else 0.25,
                risk=0.75 if mode in {"retake", "defuse"} else 0.25,
                priority=1.0 if current_events else 0.5,
                ttl_ticks=8,
                confidence={
                    "goal_position": 1.0 if valid_goal else 0.0,
                    "waypoint_position": 1.0 if valid_waypoint else 0.0,
                    "tactical_mode": 0.75 if current_events or action is not None else 0.25,
                    "task": 0.75 if current_events or action is not None else 0.25,
                    "target_slot": 1.0 if target_slot != TARGET_SLOT_NONE else 0.0,
                },
                valid_mask={
                    "goal_position": valid_goal,
                    "waypoint_position": valid_waypoint,
                    "target_slot": target_slot != TARGET_SLOT_NONE,
                },
            )
        )
    return tuple(labels)
