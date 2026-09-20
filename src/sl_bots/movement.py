"""Versioned, discrete movement labels and the fixed v3 action boundary."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

from sl_bots.action_reconstruction import (
    IN_BACK,
    IN_DUCK,
    IN_FORWARD,
    IN_JUMP,
    IN_MOVELEFT,
    IN_MOVERIGHT,
    IN_SPEED,
)
from sl_bots.contracts import BotActionV1


MOVEMENT_HORIZONS_S = (0.0, 0.125, 0.250)
MOVEMENT_DIRECTION_COUNT = 16
MOVEMENT_CLASS_COUNT = 17
STANCE_NAMES = ("run", "walk", "crouch")
STANCE_RUN = 0
STANCE_WALK = 1
STANCE_CROUCH = 2
JUMP_NO = 0
JUMP_YES = 1
IDLE_CLASS = 0
PLAN_HORIZON_DIM = 25
PLAN_EMBEDDING_SIZE = 75
MOVEMENT_BUTTON_MASK = (
    IN_JUMP
    | IN_DUCK
    | IN_SPEED
    | IN_FORWARD
    | IN_BACK
    | IN_MOVELEFT
    | IN_MOVERIGHT
)
_SECTOR_RADIANS = 2.0 * math.pi / MOVEMENT_DIRECTION_COUNT
_STANCE_AMPLITUDE = (1.0, 0.52, 0.34)


def _tuple_of(value: Iterable[object], *, size: int, name: str) -> tuple[object, ...]:
    result = tuple(value)
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} horizons")
    return result


def _class_tuple(value: Sequence[object], *, upper: int, name: str) -> tuple[int, ...]:
    result = _tuple_of(value, size=len(MOVEMENT_HORIZONS_S), name=name)
    normalized: list[int] = []
    for item in result:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < upper:
            raise ValueError(f"{name} values must be integers in [0, {upper - 1}]")
        normalized.append(item)
    return tuple(normalized)


def _bool_tuple(value: Sequence[object], *, name: str) -> tuple[bool, ...]:
    result = _tuple_of(value, size=len(MOVEMENT_HORIZONS_S), name=name)
    if any(not isinstance(item, bool) for item in result):
        raise ValueError(f"{name} values must be bool")
    return tuple(result)


@dataclass(frozen=True)
class MovementLabelV1:
    move: tuple[int, int, int]
    stance: tuple[int, int, int]
    jump: tuple[int, int, int]
    valid: tuple[bool, bool, bool]
    stance_valid: tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        move = _class_tuple(self.move, upper=MOVEMENT_CLASS_COUNT, name="move")
        stance = _class_tuple(self.stance, upper=len(STANCE_NAMES), name="stance")
        jump = _class_tuple(self.jump, upper=2, name="jump")
        valid = _bool_tuple(self.valid, name="valid")
        stance_valid = _bool_tuple(self.stance_valid, name="stance_valid")
        for index, is_valid in enumerate(valid):
            if stance_valid[index] and (not is_valid or move[index] == IDLE_CLASS):
                raise ValueError("stance_valid is only allowed for valid moving horizons")
        object.__setattr__(self, "move", move)
        object.__setattr__(self, "stance", stance)
        object.__setattr__(self, "jump", jump)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "stance_valid", stance_valid)


@dataclass(frozen=True)
class MovementPlanV1:
    move: tuple[int, int, int]
    stance: tuple[int, int, int]
    jump: tuple[int, int, int]
    plan_embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "move", _class_tuple(self.move, upper=MOVEMENT_CLASS_COUNT, name="move"))
        object.__setattr__(self, "stance", _class_tuple(self.stance, upper=len(STANCE_NAMES), name="stance"))
        object.__setattr__(self, "jump", _class_tuple(self.jump, upper=2, name="jump"))
        embedding = tuple(float(item) for item in self.plan_embedding)
        if len(embedding) != PLAN_EMBEDDING_SIZE:
            raise ValueError(f"plan_embedding must contain exactly {PLAN_EMBEDDING_SIZE} values")
        if any(not math.isfinite(item) for item in embedding):
            raise ValueError("plan_embedding must contain finite values")
        object.__setattr__(self, "plan_embedding", embedding)


@dataclass(frozen=True)
class DecodedMovementV1:
    forward: float
    side: float
    up: float
    buttons: int
    horizon_index: int
    fallback: bool = False

    def __post_init__(self) -> None:
        for name in ("forward", "side", "up"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
            object.__setattr__(self, name, value)
        if isinstance(self.buttons, bool) or not isinstance(self.buttons, int) or not 0 <= self.buttons <= 0xFFFFFFFF:
            raise ValueError("buttons must fit uint32")
        if self.horizon_index not in (-1, 0, 1, 2):
            raise ValueError("horizon_index must be -1 or a movement horizon")
        object.__setattr__(self, "fallback", bool(self.fallback))


def encode_direction_class(forward: float, side: float) -> int:
    forward = float(forward)
    side = float(side)
    if not math.isfinite(forward) or not math.isfinite(side):
        raise ValueError("movement axes must be finite")
    if math.hypot(forward, side) < 0.1:
        return IDLE_CLASS
    angle = math.atan2(side, forward)
    sector = math.floor((angle + _SECTOR_RADIANS / 2.0) / _SECTOR_RADIANS) % MOVEMENT_DIRECTION_COUNT
    return int(sector + 1)


def decode_direction_class(direction: int) -> tuple[float, float]:
    if isinstance(direction, bool) or not isinstance(direction, int) or not 0 <= direction < MOVEMENT_CLASS_COUNT:
        raise ValueError("direction class must be an integer in [0, 16]")
    if direction == IDLE_CLASS:
        return (0.0, 0.0)
    angle = (direction - 1) * _SECTOR_RADIANS
    forward = math.cos(angle)
    side = math.sin(angle)
    return (0.0 if abs(forward) < 1e-12 else forward, 0.0 if abs(side) < 1e-12 else side)


def encode_stance_class(buttons: int) -> int:
    if isinstance(buttons, bool) or not isinstance(buttons, int) or not 0 <= buttons <= 0xFFFFFFFF:
        raise ValueError("buttons must fit uint32")
    if buttons & IN_DUCK:
        return STANCE_CROUCH
    if buttons & IN_SPEED:
        return STANCE_WALK
    return STANCE_RUN


def encode_jump_class(*, up: float, buttons: int) -> int:
    if not math.isfinite(float(up)):
        raise ValueError("up must be finite")
    return JUMP_YES if buttons & IN_JUMP or float(up) > 0.1 else JUMP_NO


def label_from_action(*, forward: float, side: float, up: float, buttons: int) -> MovementLabelV1:
    for name, value in (("forward", forward), ("side", side), ("up", up)):
        if not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [-1, 1]")
    if isinstance(buttons, bool) or not isinstance(buttons, int) or not 0 <= buttons <= 0xFFFFFFFF:
        raise ValueError("buttons must fit uint32")
    move = encode_direction_class(forward, side)
    stance = encode_stance_class(buttons)
    jump = encode_jump_class(up=up, buttons=buttons)
    moving = move != IDLE_CLASS
    return MovementLabelV1(
        move=(move, move, move),
        stance=(stance, stance, stance),
        jump=(jump, jump, jump),
        valid=(True, True, True),
        stance_valid=(moving, moving, moving),
    )


encode_movement_label = label_from_action


def movement_plan_from_label(label: MovementLabelV1) -> MovementPlanV1:
    if not isinstance(label, MovementLabelV1):
        raise TypeError("label must be MovementLabelV1")
    embedding: list[float] = []
    for move, stance, jump, valid in zip(label.move, label.stance, label.jump, label.valid, strict=True):
        embedding.extend(1.0 if index == move else 0.0 for index in range(MOVEMENT_CLASS_COUNT) if valid)
        if not valid:
            embedding.extend(0.0 for _ in range(MOVEMENT_CLASS_COUNT))
        embedding.extend(1.0 if index == stance else 0.0 for index in range(len(STANCE_NAMES)) if valid)
        if not valid:
            embedding.extend(0.0 for _ in range(len(STANCE_NAMES)))
        embedding.extend(1.0 if index == jump else 0.0 for index in range(2) if valid)
        if not valid:
            embedding.extend(0.0 for _ in range(2))
        embedding.extend((1.0 if move != IDLE_CLASS and valid else 0.0, 1.0 if valid else 0.0, 1.0 if valid else 0.0))
    return MovementPlanV1(label.move, label.stance, label.jump, tuple(embedding))


label_to_plan = movement_plan_from_label


def _horizon_for_age(age_s: float) -> int | None:
    age_s = float(age_s)
    if not math.isfinite(age_s) or age_s < 0.0:
        raise ValueError("age_s must be finite and non-negative")
    if age_s < MOVEMENT_HORIZONS_S[1]:
        return 0
    if age_s < MOVEMENT_HORIZONS_S[2]:
        return 1
    if age_s < 0.375:
        return 2
    return None


def _direction_buttons(forward: float, side: float) -> int:
    buttons = 0
    if forward > 0.1:
        buttons |= IN_FORWARD
    elif forward < -0.1:
        buttons |= IN_BACK
    if side > 0.1:
        buttons |= IN_MOVELEFT
    elif side < -0.1:
        buttons |= IN_MOVERIGHT
    return buttons


def decode_movement_plan(plan: MovementPlanV1, *, age_s: float) -> DecodedMovementV1:
    if not isinstance(plan, MovementPlanV1):
        raise TypeError("plan must be MovementPlanV1")
    horizon = _horizon_for_age(age_s)
    if horizon is None:
        return DecodedMovementV1(0.0, 0.0, 0.0, 0, -1, fallback=True)
    move = plan.move[horizon]
    if move == IDLE_CLASS:
        forward, side = 0.0, 0.0
    else:
        forward, side = decode_direction_class(move)
    buttons = _direction_buttons(forward, side)
    if move != IDLE_CLASS:
        stance = plan.stance[horizon]
        amplitude = _STANCE_AMPLITUDE[stance]
        forward *= amplitude
        side *= amplitude
        if stance == STANCE_WALK:
            buttons |= IN_SPEED
        elif stance == STANCE_CROUCH:
            buttons |= IN_DUCK
    up = 0.0
    if plan.jump[horizon] == JUMP_YES:
        buttons |= IN_JUMP
        up = 1.0
    return DecodedMovementV1(forward, side, up, buttons, horizon)


def merge_movement_and_reaction(movement: DecodedMovementV1, reaction: BotActionV1) -> BotActionV1:
    if not isinstance(movement, DecodedMovementV1):
        raise TypeError("movement must be DecodedMovementV1")
    if not isinstance(reaction, BotActionV1):
        raise TypeError("reaction must be BotActionV1")
    buttons = (reaction.buttons & ~MOVEMENT_BUTTON_MASK) | (movement.buttons & MOVEMENT_BUTTON_MASK)
    return BotActionV1(
        target_tick=reaction.target_tick,
        forward=movement.forward,
        side=movement.side,
        up=movement.up,
        yaw_delta_deg=reaction.yaw_delta_deg,
        pitch_delta_deg=reaction.pitch_delta_deg,
        buttons=buttons,
        weapon_select=reaction.weapon_select,
        buy_action=reaction.buy_action,
        action_valid_mask=reaction.action_valid_mask,
    )
