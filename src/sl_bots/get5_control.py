from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Any
import zlib

from .contracts import (
    CONTROL_EVENT_SIZE,
    CONTROL_MAGIC,
    CONTROL_SCHEMA_VERSION,
    ProtocolError,
)


CONTROL_STRUCT = struct.Struct("<8sHHQIihhhhhIqQI2x")
CONTROL_CRC_OFFSET = 58


class Get5ControlEventType(IntEnum):
    RULES_VALIDATED = 1
    GOING_LIVE = 2
    LIVE = 3
    ROUND_START = 4
    ROUND_END = 5
    PAUSE = 6
    RESUME = 7
    HALFTIME = 8
    OVERTIME_START = 9
    BACKUP_RESTORE = 10
    MAP_END = 11
    SERIES_END = 12
    LATENCY_SAMPLE = 13
    FALLBACK = 14


@dataclass(frozen=True)
class Get5ControlEventV1:
    event_sequence: int
    event_type: Get5ControlEventType
    epoch: int
    server_tick: int
    get5_state: int = 0
    map_number: int = 0
    round_number: int = 0
    team1_score: int = 0
    team2_score: int = 0
    policy_generation: int = 0
    value_us: int = 0
    match_id_hash: int = 0

    def pack(self) -> bytes:
        event_type = Get5ControlEventType(self.event_type)
        payload = CONTROL_STRUCT.pack(
            CONTROL_MAGIC,
            CONTROL_SCHEMA_VERSION,
            int(event_type),
            self.event_sequence,
            self.epoch,
            self.server_tick,
            self.get5_state,
            self.map_number,
            self.round_number,
            self.team1_score,
            self.team2_score,
            self.policy_generation,
            self.value_us,
            self.match_id_hash,
            0,
        )
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        return payload[:CONTROL_CRC_OFFSET] + struct.pack("<I", checksum) + payload[CONTROL_CRC_OFFSET + 4 :]

    @classmethod
    def unpack(cls, payload: bytes) -> "Get5ControlEventV1":
        payload = bytes(payload)
        if len(payload) != CONTROL_EVENT_SIZE:
            raise ProtocolError(f"control event must be {CONTROL_STRUCT.size} bytes")
        values = CONTROL_STRUCT.unpack(payload)
        (
            magic,
            schema_version,
            event_type,
            event_sequence,
            epoch,
            server_tick,
            get5_state,
            map_number,
            round_number,
            team1_score,
            team2_score,
            policy_generation,
            value_us,
            match_id_hash,
            checksum,
        ) = values
        if magic != CONTROL_MAGIC:
            raise ProtocolError("invalid control magic")
        if schema_version != CONTROL_SCHEMA_VERSION:
            raise ProtocolError(f"unsupported control schema: {schema_version}")
        try:
            event_type = Get5ControlEventType(event_type)
        except ValueError as exc:
            raise ProtocolError(f"unknown control event type: {event_type}") from exc
        expected = bytearray(payload)
        struct.pack_into("<I", expected, CONTROL_CRC_OFFSET, 0)
        if checksum != (zlib.crc32(expected) & 0xFFFFFFFF):
            raise ProtocolError("control event crc validation failed")
        return cls(
            event_sequence=event_sequence,
            event_type=event_type,
            epoch=epoch,
            server_tick=server_tick,
            get5_state=get5_state,
            map_number=map_number,
            round_number=round_number,
            team1_score=team1_score,
            team2_score=team2_score,
            policy_generation=policy_generation,
            value_us=value_us,
            match_id_hash=match_id_hash,
        )


def hash_match_id(match_id: str) -> int:
    value = match_id.encode("utf-8")
    low = 0x811C9DC5
    high = 0x01000193
    for byte in value:
        low = ((low ^ byte) * 0x01000193) & 0xFFFFFFFF
        high = ((high ^ byte) * 0x01000193) & 0xFFFFFFFF
    return low | (high << 32)


@dataclass(frozen=True)
class TrainingBoundaryV1:
    kind: str
    epoch: int
    server_tick: int
    terminal: bool = False
    reset_hidden: bool = False
    discard_uncommitted: bool = False
    training_allowed: bool = True


class Get5ControlState:
    def __init__(
        self,
        *,
        expected_match_id: str | None = None,
        expected_match_id_hash: int | None = None,
    ) -> None:
        if expected_match_id is not None and expected_match_id_hash is not None:
            raise ValueError("provide expected_match_id or expected_match_id_hash, not both")
        if expected_match_id is not None:
            expected_match_id_hash = hash_match_id(str(expected_match_id))
        if expected_match_id_hash is not None:
            if (
                isinstance(expected_match_id_hash, bool)
                or not isinstance(expected_match_id_hash, int)
                or not 0 <= expected_match_id_hash <= 0xFFFFFFFFFFFFFFFF
            ):
                raise ValueError("expected_match_id_hash must be a uint64")
        self.expected_match_id_hash = expected_match_id_hash
        self.epoch = 0
        self.last_event_sequence = 0
        self._boundaries: list[TrainingBoundaryV1] = []
        self.paused = False
        self.map_number = 0
        self.round_number = 0
        self.team1_score = 0
        self.team2_score = 0
        self.policy_generation = 0
        self.last_latency_us: int | None = None
        self.last_latency_tick: int | None = None

    def apply(self, event: Get5ControlEventV1) -> tuple[TrainingBoundaryV1, ...]:
        event = event if isinstance(event, Get5ControlEventV1) else Get5ControlEventV1.unpack(event)
        if (
            self.expected_match_id_hash is not None
            and int(event.match_id_hash) != int(self.expected_match_id_hash)
        ):
            raise ProtocolError("control event match ID hash does not match the active match")
        if event.event_sequence <= self.last_event_sequence:
            raise ProtocolError("control event sequence must increase")
        if self.epoch and event.epoch != self.epoch:
            allowed_epoch_events = {
                Get5ControlEventType.BACKUP_RESTORE,
                Get5ControlEventType.ROUND_START,
                Get5ControlEventType.MAP_END,
                Get5ControlEventType.SERIES_END,
            }
            if event.epoch <= self.epoch or event.event_type not in allowed_epoch_events:
                raise ProtocolError("control event epoch changed without backup restore")
        if not self.epoch and event.epoch < 1:
            raise ProtocolError("control event epoch must start at one")

        self.last_event_sequence = event.event_sequence
        self.epoch = event.epoch
        self.map_number = event.map_number
        self.round_number = event.round_number
        self.team1_score = event.team1_score
        self.team2_score = event.team2_score
        self.policy_generation = int(event.policy_generation)
        event_type = Get5ControlEventType(event.event_type)

        if event_type == Get5ControlEventType.PAUSE:
            self.paused = True
            return (self._boundary(event, "pause", training_allowed=False),)
        if event_type == Get5ControlEventType.RESUME:
            self.paused = False
            return (self._boundary(event, "resume"),)
        if event_type == Get5ControlEventType.BACKUP_RESTORE:
            self.paused = False
            return (
                self._boundary(
                    event,
                    "backup_restore",
                    reset_hidden=True,
                    discard_uncommitted=True,
                ),
            )
        if event_type == Get5ControlEventType.MAP_END:
            self.paused = False
            return (self._boundary(event, "map_end", terminal=True, reset_hidden=True),)
        if event_type == Get5ControlEventType.SERIES_END:
            self.paused = False
            return (self._boundary(event, "series_end", terminal=True, reset_hidden=True),)
        if event_type == Get5ControlEventType.HALFTIME:
            return (self._boundary(event, "halftime", reset_hidden=True),)
        if event_type == Get5ControlEventType.OVERTIME_START:
            return (self._boundary(event, "overtime_start", reset_hidden=True),)
        if event_type == Get5ControlEventType.ROUND_END:
            if event.round_number == 12:
                return (self._boundary(event, "halftime", reset_hidden=True),)
            if event.round_number == 24 and event.team1_score == event.team2_score:
                return (self._boundary(event, "overtime_start", reset_hidden=True),)
            return (self._boundary(event, "round_end", reset_hidden=True),)
        if event_type == Get5ControlEventType.ROUND_START:
            return (self._boundary(event, "round_start"),)
        if event_type == Get5ControlEventType.RULES_VALIDATED:
            return (self._boundary(event, "rules_validated"),)
        if event_type == Get5ControlEventType.GOING_LIVE:
            return (self._boundary(event, "going_live"),)
        if event_type == Get5ControlEventType.LIVE:
            return (self._boundary(event, "live"),)
        if event_type == Get5ControlEventType.LATENCY_SAMPLE:
            self.last_latency_us = int(event.value_us)
            self.last_latency_tick = int(event.server_tick)
            return ()
        if event_type == Get5ControlEventType.FALLBACK:
            return (
                self._boundary(
                    event,
                    "fallback",
                    reset_hidden=True,
                    discard_uncommitted=True,
                    training_allowed=False,
                ),
            )
        raise ProtocolError(f"unsupported control event type: {event_type}")

    @property
    def boundaries(self) -> tuple[TrainingBoundaryV1, ...]:
        return tuple(self._boundaries)

    def consume_available(self, transport: Any) -> tuple[TrainingBoundaryV1, ...]:
        consumed: list[TrainingBoundaryV1] = []
        while True:
            event = transport.try_read_control()
            if event is None:
                break
            boundaries = self.apply(event)
            consumed.extend(boundaries)
            self._boundaries.extend(boundaries)
        return tuple(consumed)

    @staticmethod
    def _boundary(
        event: Get5ControlEventV1,
        kind: str,
        *,
        terminal: bool = False,
        reset_hidden: bool = False,
        discard_uncommitted: bool = False,
        training_allowed: bool = True,
    ) -> TrainingBoundaryV1:
        return TrainingBoundaryV1(
            kind=kind,
            epoch=event.epoch,
            server_tick=event.server_tick,
            terminal=terminal,
            reset_hidden=reset_hidden,
            discard_uncommitted=discard_uncommitted,
            training_allowed=training_allowed,
        )


__all__ = [
    "CONTROL_CRC_OFFSET",
    "CONTROL_EVENT_SIZE",
    "CONTROL_MAGIC",
    "CONTROL_SCHEMA_VERSION",
    "CONTROL_STRUCT",
    "Get5ControlEventType",
    "Get5ControlEventV1",
    "Get5ControlState",
    "ProtocolError",
    "TrainingBoundaryV1",
    "hash_match_id",
]
