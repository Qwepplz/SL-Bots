"""Physical-time movement labels and complete-Demo split helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random

from .action_reconstruction import (
    ACTION_MASK_BUTTONS,
    ACTION_MASK_FORWARD,
    ACTION_MASK_SIDE,
    ACTION_MASK_UP,
    ActionLabelV1,
)
from .movement import MovementLabelV1, label_from_action


@dataclass(frozen=True)
class MovementFrameV1:
    observation: bytes
    forward: float
    side: float
    up: float
    buttons: int
    duration_s: float
    sequence_id: str
    demo_id: str
    round_id: str
    player_id: str
    target_tick: int = 0
    loss_mask: int = 0

    def __post_init__(self) -> None:
        payload = bytes(self.observation)
        if len(payload) != 256:
            raise ValueError("movement frame observations must contain 256 bytes")
        object.__setattr__(self, "observation", payload)
        for name in ("forward", "side", "up", "duration_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name != "duration_s" and not -1.0 <= value <= 1.0) or (
                name == "duration_s" and value < 0.0
            ):
                raise ValueError(f"{name} must be finite and in its valid range")
            object.__setattr__(self, name, value)
        if isinstance(self.buttons, bool) or not isinstance(self.buttons, int) or not 0 <= self.buttons <= 0xFFFFFFFF:
            raise ValueError("buttons must fit uint32")
        for name in ("sequence_id", "demo_id", "round_id", "player_id"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if isinstance(self.target_tick, bool) or not isinstance(self.target_tick, int) or self.target_tick < 0:
            raise ValueError("target_tick must be a non-negative integer")
        if isinstance(self.loss_mask, bool) or not isinstance(self.loss_mask, int) or not 0 <= self.loss_mask <= 0xFF:
            raise ValueError("loss_mask must be an eight-bit action mask")

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "MovementFrameV1":
        return cls(
            observation=bytes(row["observation"]),
            forward=float(row.get("forward", 0.0)),
            side=float(row.get("side", 0.0)),
            up=float(row.get("up", 0.0)),
            buttons=int(row.get("buttons", 0) or 0),
            duration_s=float(row.get("duration_s", 0.0) or 0.0),
            sequence_id=str(row.get("sequence_id", "")),
            demo_id=str(row.get("demo_id", "demo")),
            round_id=str(row.get("round_id", row.get("round", "round-0"))),
            player_id=str(row.get("player_id", row.get("entity_id", "player"))),
            target_tick=int(row.get("target_tick", 0) or 0),
            loss_mask=int(row.get("loss_mask", 0) or 0),
        )


def _same_segment(anchor: MovementFrameV1, candidate: MovementFrameV1) -> bool:
    return (
        anchor.demo_id == candidate.demo_id
        and anchor.sequence_id == candidate.sequence_id
        and anchor.round_id == candidate.round_id
        and anchor.player_id == candidate.player_id
    )


def _frame_label(frame: MovementFrameV1) -> MovementLabelV1:
    return label_from_action(
        forward=frame.forward,
        side=frame.side,
        up=frame.up,
        buttons=frame.buttons,
    )


def _movement_label_valid(frame: MovementFrameV1) -> bool:
    required = ACTION_MASK_FORWARD | ACTION_MASK_SIDE | ACTION_MASK_UP
    return (frame.loss_mask & required) == required


def _future_frame(
    frames: Sequence[MovementFrameV1],
    index: int,
    horizon_s: float,
) -> MovementFrameV1 | None:
    anchor = frames[index]
    if horizon_s == 0.0:
        return anchor
    elapsed = 0.0
    for cursor in range(index, len(frames) - 1):
        current = frames[cursor]
        candidate = frames[cursor + 1]
        if not _same_segment(anchor, current) or not _same_segment(anchor, candidate):
            return None
        elapsed += current.duration_s
        if elapsed + 1e-9 >= horizon_s:
            return candidate
    return None


def movement_label_at(frames: Sequence[MovementFrameV1], index: int) -> MovementLabelV1:
    """Build the factorized label by walking cumulative physical time."""

    if not frames:
        raise ValueError("frames cannot be empty")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(frames):
        raise IndexError("frame index is out of range")
    current = _frame_label(frames[index])
    moves: list[int] = []
    stances: list[int] = []
    jumps: list[int] = []
    valid: list[bool] = []
    stance_valid: list[bool] = []
    for horizon_index, horizon_s in enumerate((0.0, 0.125, 0.250)):
        target = _future_frame(frames, index, horizon_s)
        if target is None:
            moves.append(current.move[0])
            stances.append(current.stance[0])
            jumps.append(current.jump[0])
            valid.append(False)
            stance_valid.append(False)
            continue
        target_label = _frame_label(target)
        moves.append(target_label.move[0])
        stances.append(target_label.stance[0])
        jumps.append(target_label.jump[0])
        is_valid = _movement_label_valid(target)
        valid.append(is_valid)
        stance_valid.append(is_valid and bool(target.loss_mask & ACTION_MASK_BUTTONS) and target_label.stance_valid[0])
    return MovementLabelV1(tuple(moves), tuple(stances), tuple(jumps), tuple(valid), tuple(stance_valid))


def movement_labels_for_sequence(frames: Sequence[MovementFrameV1]) -> tuple[MovementLabelV1, ...]:
    return tuple(movement_label_at(frames, index) for index in range(len(frames)))


build_movement_labels = movement_labels_for_sequence


def frame_from_action_label(
    observation: bytes,
    action: ActionLabelV1,
    *,
    demo_id: str,
    round_id: str,
    player_id: str,
) -> MovementFrameV1:
    return MovementFrameV1(
        observation=observation,
        forward=action.forward,
        side=action.side,
        up=action.up,
        buttons=action.buttons,
        duration_s=action.duration_s,
        sequence_id=action.sequence_id,
        demo_id=demo_id,
        round_id=round_id,
        player_id=player_id,
        target_tick=action.target_tick,
        loss_mask=action.loss_mask,
    )


def split_demo_ids(
    demo_sha256s: Sequence[str],
    *,
    seed: int,
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """Split complete Demo identities without splitting a match or sequence."""

    values = tuple(str(value) for value in demo_sha256s)
    if not values or any(not value for value in values):
        raise ValueError("Demo identities cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("Demo identities must be unique")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (train_count, validation_count, test_count) if value is not None):
        raise ValueError("split counts must be non-negative integers")
    if train_count is None and validation_count is None and test_count is None:
        if len(values) == 3:
            train_count, validation_count, test_count = 2, 1, 0
        elif len(values) == 10:
            train_count, validation_count, test_count = 8, 1, 1
        else:
            raise ValueError("automatic split supports exactly 3 or 10 complete Demos")
    train_count = 0 if train_count is None else train_count
    validation_count = 0 if validation_count is None else validation_count
    test_count = 0 if test_count is None else test_count
    if train_count + validation_count + test_count != len(values):
        raise ValueError("split counts must cover every complete Demo")
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    train_end = train_count
    validation_end = train_end + validation_count
    return {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(shuffled[train_end:validation_end]),
        "test": tuple(shuffled[validation_end:]),
    }
