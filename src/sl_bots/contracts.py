"""Versioned wire and training contracts shared by SL-Bots components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
import struct
import zlib


PROTOCOL_MAGIC = b"SLBOTS1\x00"
SCHEMA_VERSION = 1
MAX_BOTS = 10
HEADER_STRUCT = struct.Struct("<8sHHIIHHQI")
HEADER_SIZE = HEADER_STRUCT.size
OBSERVATION_RECORD_SIZE = 256
ACTION_RECORD_STRUCT = struct.Struct("<ifffffIhhI")
ACTION_RECORD_SIZE = ACTION_RECORD_STRUCT.size


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

    def __post_init__(self) -> None:
        _validate_batch_header(self.epoch, self.server_tick, self.write_sequence)
        actions = tuple(self.actions)
        if len(actions) > MAX_BOTS:
            raise ValueError(f"bot count cannot exceed {MAX_BOTS}")
        if any(not isinstance(action, BotActionV1) for action in actions):
            raise TypeError("actions must contain BotActionV1 values")
        object.__setattr__(self, "actions", actions)

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
