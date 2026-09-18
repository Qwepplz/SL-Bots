"""Versioned wire and training contracts shared by SL-Bots components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from collections.abc import Mapping
import math
from pathlib import Path
import struct
import zlib
from typing import Any


PROTOCOL_MAGIC = b"SLBOTS1\x00"
SCHEMA_VERSION = 1
MAX_BOTS = 10
HEADER_STRUCT = struct.Struct("<8sHHIIHHQI")
HEADER_SIZE = HEADER_STRUCT.size
OBSERVATION_RECORD_SIZE = 256
ACTION_RECORD_STRUCT = struct.Struct("<ifffffIhhI")
ACTION_RECORD_SIZE = ACTION_RECORD_STRUCT.size
CONTROL_MAGIC = b"SLBCTL1\x00"
CONTROL_SCHEMA_VERSION = 1
CONTROL_EVENT_SIZE = 64


class ProtocolError(ValueError):
    """Raised when a serialized Protocol V1 packet is invalid."""


class Phase(IntEnum):
    WARMUP = 0
    KNIFE = 1
    LIVE = 2


class Side(IntEnum):
    T = 2
    CT = 3


class DataPurpose(str, Enum):
    TEST_ONLY = "test_only"
    PRODUCTION = "production"


HIERARCHICAL_STATE_SCHEMA = {
    "decision_memory": "float32[batch,32,512]",
    "action_hidden": "float32[batch,384]",
    "cached_intent": "float32[batch,128]",
    "last_decision_tick": "int64[batch]",
}


@dataclass(frozen=True)
class HierarchicalPackageV2:
    """Atomic manifest binding the decision and action ONNX artifacts."""

    generation: int
    decision_path: Path
    action_path: Path
    purpose: DataPurpose
    decision_sha256: str
    action_sha256: str
    state_schema: Mapping[str, str]
    parent_generation: int | None = None
    decision_parameter_count: int = 0
    action_parameter_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generation, int) or isinstance(self.generation, bool) or self.generation < 0:
            raise ValueError("package generation must be a non-negative integer")
        if self.generation == 0 and self.parent_generation is not None:
            raise ValueError("generation 0 cannot have a parent generation")
        if self.generation > 0 and self.parent_generation != self.generation - 1:
            raise ValueError("package parent must be exactly generation - 1")
        object.__setattr__(self, "decision_path", Path(self.decision_path).resolve())
        object.__setattr__(self, "action_path", Path(self.action_path).resolve())
        object.__setattr__(self, "purpose", ensure_purpose(self.purpose))
        for name in ("decision_sha256", "action_sha256"):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be a 64-character hexadecimal digest")
            object.__setattr__(self, name, digest)
        state_schema = dict(self.state_schema)
        if state_schema != HIERARCHICAL_STATE_SCHEMA:
            raise ValueError("state schema does not match hierarchical actor contract")
        object.__setattr__(self, "state_schema", state_schema)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.decision_parameter_count < 0 or self.action_parameter_count < 0:
            raise ValueError("package parameter counts must be non-negative")

    def assert_loadable(self) -> None:
        if self.purpose is not DataPurpose.TEST_ONLY:
            raise ValueError("hierarchical package must remain test_only")
        metadata_purpose = self.metadata.get("purpose")
        lineage = self.metadata.get("training_lineage")
        lineage_purpose = lineage.get("purpose") if isinstance(lineage, Mapping) else None
        if metadata_purpose != DataPurpose.TEST_ONLY.value or lineage_purpose != DataPurpose.TEST_ONLY.value:
            raise ValueError("hierarchical package lineage must be test_only")
        for path in (self.decision_path, self.action_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size <= 0:
                raise ValueError(f"hierarchical package artifact is empty: {path}")
        import hashlib

        for path, expected, name in (
            (self.decision_path, self.decision_sha256, "decision"),
            (self.action_path, self.action_sha256, "action"),
        ):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{name} artifact hash does not match package manifest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hierarchical-package-v2",
            "generation": self.generation,
            "parent_generation": self.parent_generation,
            "decision_path": str(self.decision_path),
            "action_path": str(self.action_path),
            "purpose": self.purpose.value,
            "decision_sha256": self.decision_sha256,
            "action_sha256": self.action_sha256,
            "decision_parameter_count": self.decision_parameter_count,
            "action_parameter_count": self.action_parameter_count,
            "state_schema": dict(self.state_schema),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HierarchicalPackageV2":
        if payload.get("schema") != "hierarchical-package-v2":
            raise ValueError("package manifest schema must be hierarchical-package-v2")
        return cls(
            generation=int(payload["generation"]),
            parent_generation=(
                None if payload.get("parent_generation") is None else int(payload["parent_generation"])
            ),
            decision_path=Path(payload["decision_path"]),
            action_path=Path(payload["action_path"]),
            purpose=payload["purpose"],
            decision_sha256=str(payload["decision_sha256"]),
            action_sha256=str(payload["action_sha256"]),
            decision_parameter_count=int(payload.get("decision_parameter_count", 0)),
            action_parameter_count=int(payload.get("action_parameter_count", 0)),
            state_schema=payload["state_schema"],
            metadata=payload.get("metadata", {}),
        )


TACTICAL_MODES = (
    "hold",
    "advance",
    "rotate",
    "retake",
    "save",
    "plant",
    "defend_bomb",
    "defuse",
)
INTENT_TASKS = (
    "move",
    "engage",
    "anchor",
    "clear",
    "use_utility",
    "plant",
    "defuse",
    "support",
)
TARGET_SLOT_NONE = 9
INTENT_FIELD_NAMES = (
    "tactical_mode",
    "task",
    "goal_position",
    "waypoint_position",
    "facing_yaw_pitch",
    "desired_range",
    "target_slot",
    "aggression",
    "risk",
    "priority",
    "ttl_ticks",
)


def _normalize_intent_mapping(
    values: Mapping[str, object] | None,
    *,
    default: object,
) -> dict[str, object]:
    result = {name: default for name in INTENT_FIELD_NAMES}
    if values:
        result.update({str(key): value for key, value in values.items()})
    return result


@dataclass(frozen=True)
class IntentV1:
    """Structured macro intent shared by the decision and action layers."""

    tactical_mode: str = "hold"
    task: str = "move"
    goal_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    waypoint_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing_yaw_pitch: tuple[float, float] = (0.0, 0.0)
    desired_range: float = 0.0
    target_slot: int = TARGET_SLOT_NONE
    aggression: float = 0.0
    risk: float = 0.0
    priority: float = 0.0
    ttl_ticks: int = 0
    confidence: Mapping[str, float] = field(default_factory=dict)
    valid_mask: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_position", tuple(float(v) for v in self.goal_position))
        object.__setattr__(self, "waypoint_position", tuple(float(v) for v in self.waypoint_position))
        object.__setattr__(self, "facing_yaw_pitch", tuple(float(v) for v in self.facing_yaw_pitch))
        if len(self.goal_position) != 3 or len(self.waypoint_position) != 3:
            raise ValueError("intent positions must contain three values")
        if len(self.facing_yaw_pitch) != 2:
            raise ValueError("intent facing_yaw_pitch must contain two values")
        if not isinstance(self.target_slot, int) or isinstance(self.target_slot, bool):
            raise TypeError("target_slot must be an integer")
        if not isinstance(self.ttl_ticks, int) or isinstance(self.ttl_ticks, bool):
            raise TypeError("ttl_ticks must be an integer")
        confidence = _normalize_intent_mapping(self.confidence, default=1.0)
        valid_mask = _normalize_intent_mapping(self.valid_mask, default=True)
        object.__setattr__(
            self,
            "confidence",
            {key: max(0.0, min(1.0, float(value))) for key, value in confidence.items()},
        )
        object.__setattr__(
            self,
            "valid_mask",
            {key: bool(value) for key, value in valid_mask.items()},
        )

    def to_embedding(self, size: int = 128) -> tuple[float, ...]:
        if size < 71:
            raise ValueError("intent embedding size must be at least 71")
        values: list[float] = []
        values.extend(float(index == TACTICAL_MODES.index(self.tactical_mode)) for index in range(len(TACTICAL_MODES)) if self.tactical_mode in TACTICAL_MODES)
        if self.tactical_mode not in TACTICAL_MODES:
            values.extend(0.0 for _ in TACTICAL_MODES)
        values.extend(float(index == INTENT_TASKS.index(self.task)) for index in range(len(INTENT_TASKS)) if self.task in INTENT_TASKS)
        if self.task not in INTENT_TASKS:
            values.extend(0.0 for _ in INTENT_TASKS)
        values.extend(self.goal_position)
        values.extend(self.waypoint_position)
        values.extend(self.facing_yaw_pitch)
        values.append(float(self.desired_range))
        values.extend(float(index == self.target_slot) for index in range(TARGET_SLOT_NONE + 1))
        values.extend((float(self.aggression), float(self.risk), float(self.priority)))
        values.append(float(self.ttl_ticks))
        values.extend(float(self.confidence.get(name, 0.0)) for name in INTENT_FIELD_NAMES[:16])
        values.extend(float(self.valid_mask.get(name, False)) for name in INTENT_FIELD_NAMES[:16])
        return tuple(values[:size] + [0.0] * max(0, size - len(values)))


@dataclass(frozen=True)
class DecisionOutputV1(IntentV1):
    """Untrusted decision-head output before canonicalization."""

    def __post_init__(self) -> None:
        # Decision heads carry tensors during training; canonicalize_intent applies
        # scalar bounds only at the runtime boundary.
        if not isinstance(self.confidence, Mapping):
            object.__setattr__(self, "confidence", {})
        if not isinstance(self.valid_mask, Mapping):
            object.__setattr__(self, "valid_mask", {})

    @property
    def tactical_mode_logits(self) -> object:
        return self.tactical_mode

    @property
    def task_logits(self) -> object:
        return self.task

    @property
    def goal_position_tensor(self) -> object:
        return self.goal_position

    @property
    def waypoint_position_tensor(self) -> object:
        return self.waypoint_position


@dataclass(frozen=True)
class IntentLabelV1(IntentV1):
    """Offline future-derived intent target with its source tick."""

    target_tick: int = 0


@dataclass(frozen=True)
class DemoEventV1:
    tick: int
    event_type: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tick, int) or self.tick < 0:
            raise ValueError("event tick must be a non-negative integer")
        event_type = str(self.event_type)
        if not event_type:
            raise ValueError("event_type cannot be empty")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "payload", dict(self.payload))


def _require_uint(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must fit in uint{bits}")


def _require_int(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if not -(1 << (bits - 1)) <= value < 1 << (bits - 1):
        raise ValueError(f"{name} must fit in int{bits}")


def _float32(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("action float fields must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("action float fields must be finite")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("action float fields must fit in float32") from error


@dataclass(frozen=True)
class BotObservationV1:
    """Fixed-size observation record carried by the IPC batch."""

    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        payload = bytes(self.payload)
        if len(payload) != OBSERVATION_RECORD_SIZE:
            raise ValueError(
                f"observation payload must be {OBSERVATION_RECORD_SIZE} bytes"
            )
        object.__setattr__(self, "payload", payload)

    def to_bytes(self) -> bytes:
        return self.payload


@dataclass(frozen=True)
class BotActionV1:
    target_tick: int
    forward: float
    side: float
    up: float
    yaw_delta_deg: float
    pitch_delta_deg: float
    buttons: int
    weapon_select: int
    buy_action: int
    action_valid_mask: int

    def __post_init__(self) -> None:
        _require_int("target_tick", self.target_tick, 32)
        for name in ("forward", "side", "up"):
            value = _float32(getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
            object.__setattr__(self, name, value)
        for name in ("yaw_delta_deg", "pitch_delta_deg"):
            object.__setattr__(self, name, _float32(getattr(self, name)))
        _require_uint("buttons", self.buttons, 32)
        _require_int("weapon_select", self.weapon_select, 16)
        _require_int("buy_action", self.buy_action, 16)
        _require_uint("action_valid_mask", self.action_valid_mask, 32)

    def to_bytes(self) -> bytes:
        return ACTION_RECORD_STRUCT.pack(
            self.target_tick,
            self.forward,
            self.side,
            self.up,
            self.yaw_delta_deg,
            self.pitch_delta_deg,
            self.buttons,
            self.weapon_select,
            self.buy_action,
            self.action_valid_mask,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "BotActionV1":
        if len(payload) != ACTION_RECORD_SIZE:
            raise ProtocolError(f"action payload must be {ACTION_RECORD_SIZE} bytes")
        return cls(*ACTION_RECORD_STRUCT.unpack(payload))


def _validate_batch_header(epoch: int, server_tick: int, write_sequence: int) -> None:
    _require_uint("epoch", epoch, 32)
    _require_uint("server_tick", server_tick, 32)
    _require_uint("write_sequence", write_sequence, 64)


def _pack_packet(
    epoch: int,
    server_tick: int,
    write_sequence: int,
    records: tuple[bytes, ...],
    record_size: int,
) -> bytes:
    _validate_batch_header(epoch, server_tick, write_sequence)
    if len(records) > MAX_BOTS:
        raise ValueError(f"bot count cannot exceed {MAX_BOTS}")
    if any(len(record) != record_size for record in records):
        raise ValueError(f"every record must be {record_size} bytes")
    packet_size = HEADER_SIZE + MAX_BOTS * record_size
    if packet_size > 0xFFFF:
        raise ValueError("packet size exceeds Protocol V1 header capacity")
    payload = b"".join(records) + b"\x00" * ((MAX_BOTS - len(records)) * record_size)
    header_without_crc = HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        SCHEMA_VERSION,
        packet_size,
        epoch,
        server_tick,
        len(records),
        0,
        write_sequence,
        0,
    )
    checksum = zlib.crc32(header_without_crc + payload) & 0xFFFFFFFF
    header = HEADER_STRUCT.pack(
        PROTOCOL_MAGIC,
        SCHEMA_VERSION,
        packet_size,
        epoch,
        server_tick,
        len(records),
        0,
        write_sequence,
        checksum,
    )
    return header + payload


def _unpack_packet(data: bytes, record_size: int) -> tuple[int, int, int, int, bytes]:
    data = bytes(data)
    if len(data) < HEADER_SIZE:
        raise ProtocolError("packet is shorter than the Protocol V1 header")
    (
        magic,
        schema_version,
        packet_size,
        epoch,
        server_tick,
        bot_count,
        reserved,
        write_sequence,
        checksum,
    ) = HEADER_STRUCT.unpack_from(data)
    if magic != PROTOCOL_MAGIC:
        raise ProtocolError("invalid magic")
    if schema_version != SCHEMA_VERSION:
        raise ProtocolError(f"unsupported schema_version: {schema_version}")
    expected_size = HEADER_SIZE + MAX_BOTS * record_size
    if packet_size != expected_size:
        raise ProtocolError(f"invalid record_size: {packet_size}")
    if len(data) != packet_size:
        raise ProtocolError("packet length does not match record_size")
    if bot_count > MAX_BOTS:
        raise ProtocolError(f"bot count cannot exceed {MAX_BOTS}")
    if reserved != 0:
        raise ProtocolError("reserved header bits must be zero")
    payload = data[HEADER_SIZE:]
    header_without_crc = HEADER_STRUCT.pack(
        magic,
        schema_version,
        packet_size,
        epoch,
        server_tick,
        bot_count,
        reserved,
        write_sequence,
        0,
    )
    expected_checksum = zlib.crc32(header_without_crc + payload) & 0xFFFFFFFF
    if checksum != expected_checksum:
        raise ProtocolError("CRC32 validation failed")
    return epoch, server_tick, write_sequence, bot_count, payload


@dataclass(frozen=True)
class ObservationBatchV1:
    epoch: int
    server_tick: int
    observations: tuple[bytes, ...]
    write_sequence: int = 0

    def __post_init__(self) -> None:
        _validate_batch_header(self.epoch, self.server_tick, self.write_sequence)
        normalized = tuple(
            observation.to_bytes()
            if isinstance(observation, BotObservationV1)
            else bytes(observation)
            for observation in self.observations
        )
        if len(normalized) > MAX_BOTS:
            raise ValueError(f"bot count cannot exceed {MAX_BOTS}")
        if any(len(observation) != OBSERVATION_RECORD_SIZE for observation in normalized):
            raise ValueError(
                f"every observation must be {OBSERVATION_RECORD_SIZE} bytes"
            )
        object.__setattr__(self, "observations", normalized)

    @property
    def bot_count(self) -> int:
        return len(self.observations)

    @property
    def packet_size(self) -> int:
        return HEADER_SIZE + MAX_BOTS * OBSERVATION_RECORD_SIZE

    def pack(self) -> bytes:
        return _pack_packet(
            self.epoch,
            self.server_tick,
            self.write_sequence,
            self.observations,
            OBSERVATION_RECORD_SIZE,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ObservationBatchV1":
        epoch, server_tick, write_sequence, bot_count, payload = _unpack_packet(
            data, OBSERVATION_RECORD_SIZE
        )
        observations = tuple(
            payload[offset : offset + OBSERVATION_RECORD_SIZE]
            for offset in range(0, bot_count * OBSERVATION_RECORD_SIZE, OBSERVATION_RECORD_SIZE)
        )
        return cls(epoch, server_tick, observations, write_sequence)


@dataclass(frozen=True)
class ActionBatchV1:
    epoch: int
    server_tick: int
    actions: tuple[BotActionV1, ...]
    write_sequence: int = 0
    # These fields stay outside the fixed IPC packet.  They carry the decision
    # behavior that produced the action to the in-process self-play recorder.
    # ``None`` means that this tick reused the cached intent rather than
    # refreshing the decision policy.
    decision_actions: tuple[tuple[int, int, int] | None, ...] = ()
    decision_log_probs: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        _validate_batch_header(self.epoch, self.server_tick, self.write_sequence)
        actions = tuple(self.actions)
        if len(actions) > MAX_BOTS:
            raise ValueError(f"bot count cannot exceed {MAX_BOTS}")
        if any(not isinstance(action, BotActionV1) for action in actions):
            raise TypeError("actions must contain BotActionV1 values")
        decision_actions = (
            tuple(self.decision_actions)
            if self.decision_actions
            else tuple(None for _ in actions)
        )
        decision_log_probs = (
            tuple(self.decision_log_probs)
            if self.decision_log_probs
            else tuple(None for _ in actions)
        )
        if len(decision_actions) != len(actions) or len(decision_log_probs) != len(actions):
            raise ValueError("decision behavior fields must align with actions")
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
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "decision_actions", decision_actions)
        object.__setattr__(self, "decision_log_probs", decision_log_probs)

    @property
    def bot_count(self) -> int:
        return len(self.actions)

    @property
    def packet_size(self) -> int:
        return HEADER_SIZE + MAX_BOTS * ACTION_RECORD_SIZE

    def pack(self) -> bytes:
        return _pack_packet(
            self.epoch,
            self.server_tick,
            self.write_sequence,
            tuple(action.to_bytes() for action in self.actions),
            ACTION_RECORD_SIZE,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ActionBatchV1":
        epoch, server_tick, write_sequence, bot_count, payload = _unpack_packet(
            data, ACTION_RECORD_SIZE
        )
        actions = tuple(
            BotActionV1.from_bytes(
                payload[offset : offset + ACTION_RECORD_SIZE]
            )
            for offset in range(0, bot_count * ACTION_RECORD_SIZE, ACTION_RECORD_SIZE)
        )
        return cls(epoch, server_tick, actions, write_sequence)


def ensure_purpose(value: DataPurpose | str) -> DataPurpose:
    if isinstance(value, DataPurpose):
        return value
    try:
        return DataPurpose(value)
    except ValueError as error:
        raise ValueError(f"unknown data purpose: {value}") from error
