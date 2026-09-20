"""真人等价的局部观察投影与 256 字节量化编码。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import BotObservationV1, Phase, Side


OBSERVATION_MAGIC = b"OBS1"
OBSERVATION_SCHEMA_VERSION = 1
PLAYER_SLOT_COUNT = 9
SOUND_SLOT_COUNT = 16
EVENT_SLOT_COUNT = 16
RAY_COUNT = 40

PLAYER_DIRECT = 1 << 0
PLAYER_RADAR = 1 << 1
PLAYER_AUDIBLE = 1 << 2
PLAYER_ALIVE = 1 << 3
PLAYER_KNOWN = 1 << 4
PLAYER_PERIPHERAL = 1 << 5

_SELF_STRUCT = struct.Struct("<HBBHhhhhhbbBB")
_PLAYER_STRUCT = struct.Struct("<hbBhbHBBB")
assert _SELF_STRUCT.size == 20
assert _PLAYER_STRUCT.size == 12

_MAP_IDS = {"de_mirage": 1}
_MAP_NAMES = {value: key for key, value in _MAP_IDS.items()}
_SOUND_IDS = {
    "unknown": 0,
    "footstep": 1,
    "weapon_fire": 2,
    "grenade": 3,
    "flash": 4,
    "smoke": 5,
    "reload": 6,
    "jump": 7,
}
_SOUND_NAMES = {value: key for key, value in _SOUND_IDS.items()}
_EVENT_IDS = {
    "unknown": 0,
    "weapon_fire": 1,
    "weapon_reload": 2,
    "player_hurt": 3,
    "player_flashed": 4,
    "kill": 5,
    "grenade_projectile_throw": 6,
    "grenade_projectile_bounce": 7,
    "grenade_projectile_destroy": 8,
    "round_start": 9,
    "round_end": 10,
    "round_freezetime_end": 11,
    "team_side_switch": 12,
    "smoke_start": 13,
    "smoke_expired": 14,
    "flash_explode": 15,
    "fire_grenade_start": 16,
    "fire_grenade_expired": 17,
    "he_explode": 18,
    "decoy_start": 19,
    "decoy_expired": 20,
    "footstep": 21,
    "player_jump": 22,
    "round_win_t": 23,
    "round_win_ct": 24,
    "kill_self": 25,
    "smoke_start_self": 26,
    "flash_explode_self": 27,
    "he_explode_self": 28,
    "fire_grenade_start_self": 29,
}
_EVENT_NAMES = {value: key for key, value in _EVENT_IDS.items()}
_SELF_EVENT_NAMES = {
    "kill": "kill_self",
    "smoke_start": "smoke_start_self",
    "flash_explode": "flash_explode_self",
    "he_explode": "he_explode_self",
    "fire_grenade_start": "fire_grenade_start_self",
}
_WEAPON_IDS = {
    "": 0,
    "Glock-18": 1,
    "USP-S": 2,
    "P2000": 3,
    "AK-47": 4,
    "M4A4": 5,
    "M4A1-S": 6,
    "AWP": 7,
    "SSG 08": 8,
    "Deagle": 9,
    "Knife": 10,
    "HE Grenade": 11,
    "Flashbang": 12,
    "Smoke Grenade": 13,
    "C4": 14,
}
_WEAPON_NAMES = {value: key for key, value in _WEAPON_IDS.items()}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _as_vector(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping):
        try:
            return (float(value["x"]), float(value["y"]), float(value["z"]))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 3:
            try:
                return (float(value[0]), float(value[1]), float(value[2]))
            except (TypeError, ValueError):
                return None
    return None


def _mapping_value(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _identity_values(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        values = [value.get("entity_id"), value.get("steam_id32"), value.get("id")]
    else:
        values = [value]
    return {str(item) for item in values if item is not None}


def _matches_observer(value: Any, observer_id: int | str) -> bool:
    return bool(_identity_values(value) & {str(observer_id)})


def _channel(value: Any, observer_id: int | str) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == str(observer_id):
                return bool(item)
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_matches_observer(item, observer_id) for item in value)
    return bool(value)


def _player_id(player: Mapping[str, Any]) -> int | str | None:
    return _mapping_value(player, "entity_id", "steam_id32", "id")


def _find_observer(players: Sequence[Mapping[str, Any]], observer_id: int | str) -> Mapping[str, Any]:
    for player in players:
        if _matches_observer(_player_id(player), observer_id):
            return player
    raise KeyError(f"observer {observer_id!r} is not present in the tick")


def _event_type(event: Mapping[str, Any]) -> str:
    value = event.get("type", "unknown")
    return str(value).lower()


def _source_id(event: Mapping[str, Any]) -> int | str | None:
    event_type = _event_type(event)
    if event_type in {"kill", "player_death", "player_hurt", "player_flashed"}:
        names = ("source_id", "attacker_id", "shooter_id", "player_id")
    elif event_type in {
        "smoke_start",
        "flash_explode",
        "he_explode",
        "fire_grenade_start",
        "grenade_projectile_throw",
    }:
        names = ("source_id", "thrower_id", "owner_id", "shooter_id", "player_id")
    else:
        names = ("source_id", "player_id", "shooter_id", "attacker_id", "thrower_id")
    for name in names:
        if name in event:
            return event[name]
    if event_type in {"kill", "player_death", "player_hurt", "player_flashed"}:
        mapping_names = ("source", "attacker", "shooter", "player")
    elif event_type in {
        "smoke_start",
        "flash_explode",
        "he_explode",
        "fire_grenade_start",
        "grenade_projectile_throw",
    }:
        mapping_names = ("source", "thrower", "shooter", "player", "owner")
    else:
        mapping_names = ("source", "player", "shooter", "attacker", "thrower")
    for name in mapping_names:
        value = event.get(name)
        if isinstance(value, Mapping):
            return _player_id(value)
    return None


def _position_from_event(event: Mapping[str, Any]) -> tuple[float, float, float] | None:
    return _as_vector(_mapping_value(event, "position", "source_position"))


def _geometry(
    observer_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    dx = target_position[0] - observer_position[0]
    dy = target_position[1] - observer_position[1]
    dz = target_position[2] - observer_position[2]
    horizontal = math.hypot(dx, dy)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    bearing = math.degrees(math.atan2(dy, dx)) if horizontal else 0.0
    elevation = math.degrees(math.atan2(dz, horizontal)) if distance else 0.0
    return bearing, elevation, distance


def _stable_noise(memory: "ObservationMemory", observer_id: int | str, target_id: Any, tick: int, channel: str) -> float:
    if memory.noise_scale <= 0.0:
        return 0.0
    seed = f"{observer_id}:{target_id}:{tick}:{channel}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(seed).digest()[:4], "little") / 2**32
    return (value * 2.0 - 1.0) * memory.noise_scale


def _distance_to_segment(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float:
    vector = tuple(end[index] - start[index] for index in range(3))
    denominator = sum(component * component for component in vector)
    if denominator <= 1e-9:
        return math.dist(point, start)
    fraction = sum((point[index] - start[index]) * vector[index] for index in range(3)) / denominator
    fraction = _clamp(fraction, 0.0, 1.0)
    closest = tuple(start[index] + fraction * vector[index] for index in range(3))
    return math.dist(point, closest)


@dataclass(frozen=True)
class ProjectedPlayer:
    entity_id: int = 0
    relation: int = 0
    flags: int = 0
    bearing_deg: float = 0.0
    elevation_deg: float = 0.0
    distance: float = 0.0
    age_s: float = 0.0
    confidence: float = 0.0
    visible_ratio: float = 0.0


@dataclass(frozen=True)
class ProjectedSound:
    category: str = "unknown"
    bearing_deg: float = 0.0
    distance: float = 0.0
    age_s: float = 0.0
    occluded: bool = False
    masked: bool = False
    loudness: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class ProjectedEvent:
    category: str = "unknown"
    age_s: float = 0.0
    known_participants: bool = False


@dataclass(frozen=True)
class ProjectedObservation:
    map_name: str
    phase: int
    side: int
    self_entity_id: int
    self_health: int
    self_armor: int
    self_money: int
    self_position: tuple[float, float, float]
    self_yaw_deg: float
    self_pitch_deg: float
    self_velocity: tuple[float, float, float]
    self_flags: int
    flash_recovery: float
    smoke_occlusion: float
    action_cooldown: float
    weapon_name: str
    difficulty: tuple[float, ...]
    personality: tuple[float, ...]
    current_area: int
    target_area: int
    stage_flags: int
    players: tuple[ProjectedPlayer, ...]
    sounds: tuple[ProjectedSound, ...]
    events: tuple[ProjectedEvent, ...]
    rays: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.map_name != "de_mirage":
            raise ValueError("only de_mirage is supported")
        if len(self.players) != PLAYER_SLOT_COUNT:
            raise ValueError(f"players must contain {PLAYER_SLOT_COUNT} slots")
        if len(self.sounds) != SOUND_SLOT_COUNT:
            raise ValueError(f"sounds must contain {SOUND_SLOT_COUNT} slots")
        if len(self.events) != EVENT_SLOT_COUNT:
            raise ValueError(f"events must contain {EVENT_SLOT_COUNT} slots")
        if len(self.rays) != RAY_COUNT:
            raise ValueError(f"rays must contain {RAY_COUNT} values")
        if len(self.difficulty) != 4 or len(self.personality) != 4:
            raise ValueError("difficulty and personality must contain four values")


@dataclass
class _SmokeState:
    position: tuple[float, float, float]
    started_at: float
    duration_s: float
    radius: float
    entity_id: str


@dataclass
class ObservationMemory:
    """逐 Bot 短期记忆；不同 observer_id 的状态永不共用。"""

    phase: Phase = Phase.LIVE
    side: Side | None = None
    map_name: str = "de_mirage"
    difficulty: tuple[float, ...] = (0.75, 0.75, 0.75, 0.75)
    personality: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    noise_scale: float = 0.02
    hearing_radius: float = 1800.0
    peripheral_delay_s: float = 0.0
    radar_delay_s: float = 0.0
    _last_time: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _sounds: dict[str, list[ProjectedSound]] = field(default_factory=dict, init=False, repr=False)
    _events: dict[str, list[ProjectedEvent]] = field(default_factory=dict, init=False, repr=False)
    _smokes: dict[str, _SmokeState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.map_name = str(self.map_name)
        if self.map_name != "de_mirage":
            raise ValueError("only de_mirage is supported")
        self.difficulty = _condition_vector(self.difficulty)
        self.personality = _condition_vector(self.personality)
        if self.noise_scale < 0.0 or self.hearing_radius <= 0.0:
            raise ValueError("noise_scale must be non-negative and hearing_radius must be positive")

    def reset_stage(self, phase: Phase | int) -> None:
        self.phase = Phase(phase)
        self._sounds.clear()
        self._events.clear()
        self._smokes.clear()
        self._last_time.clear()


def _condition_vector(values: Sequence[float]) -> tuple[float, float, float, float]:
    normalized = tuple(_clamp(float(value), 0.0, 1.0) for value in values)
    if len(normalized) != 4:
        raise ValueError("condition vectors must contain four values")
    return normalized  # type: ignore[return-value]


def _phase_from_tick(raw_tick: Any, fallback: Phase) -> Phase:
    value = getattr(raw_tick, "phase", None)
    if value is None:
        return Phase(fallback)
    if isinstance(value, str):
        names = {"warmup": Phase.WARMUP, "knife": Phase.KNIFE, "kniferound": Phase.KNIFE, "live": Phase.LIVE}
        return names.get(value.lower(), Phase(fallback))
    return Phase(value)


def _side_from_observer(observer: Mapping[str, Any], memory: ObservationMemory) -> int:
    if memory.side is not None:
        return int(memory.side)
    team = observer.get("team", 0)
    try:
        return int(team) if int(team) in (int(Side.T), int(Side.CT)) else 0
    except (TypeError, ValueError):
        return 0


def _weapon_name(observer: Mapping[str, Any]) -> str:
    value = observer.get("active_weapon", "")
    if isinstance(value, Mapping):
        value = value.get("name", "")
    return str(value)


def _weapon_id(name: str, observer: Mapping[str, Any]) -> int:
    if isinstance(observer.get("weapon_id"), int):
        return _clamp_int(observer["weapon_id"], 0, 255)
    return _WEAPON_IDS.get(name, 0)


def _visibility_fraction(player: Mapping[str, Any], observer_id: int | str) -> float:
    value = _mapping_value(player, "visible_ratio_to", "visible_fraction_to", default=None)
    if isinstance(value, Mapping):
        value = value.get(str(observer_id), value.get(observer_id, 0.0))
    if value is None:
        value = player.get("visible_ratio", 1.0)
    try:
        return _clamp(float(value), 0.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


def _radar_age(player: Mapping[str, Any], observer_id: int | str) -> float:
    value = player.get("radar_age_s", 0.0)
    if isinstance(value, Mapping):
        value = value.get(str(observer_id), value.get(observer_id, 0.0))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _is_direct(player: Mapping[str, Any], observer_id: int | str) -> bool:
    for name in ("directly_visible_to", "direct_visible_to", "visible_to"):
        if name in player and _channel(player[name], observer_id):
            return True
    return _channel(player.get("directly_visible", False), observer_id)


def _is_radar(player: Mapping[str, Any], observer_id: int | str) -> bool:
    for name in ("radar_visible_to", "radar_spotted_to"):
        if name in player and _channel(player[name], observer_id):
            return True
    return _channel(player.get("radar_visible", False), observer_id)


def _event_is_audible(
    event: Mapping[str, Any],
    observer_id: int | str,
    observer_position: tuple[float, float, float],
    memory: ObservationMemory,
) -> tuple[bool, float, float, bool, bool, float]:
    if "audible_to" in event and not _channel(event["audible_to"], observer_id):
        return False, 0.0, 0.0, False, False, 0.0
    position = _position_from_event(event)
    distance_value = event.get("distance")
    if position is not None:
        bearing, _, distance = _geometry(observer_position, position)
    else:
        try:
            distance = max(0.0, float(distance_value))
        except (TypeError, ValueError):
            distance = 0.0
        try:
            bearing = float(event.get("bearing_deg", event.get("bearing", 0.0)))
        except (TypeError, ValueError):
            bearing = 0.0
    if event.get("audible") is False:
        return False, bearing, distance, False, False, 0.0
    if distance > memory.hearing_radius and "audible_to" not in event:
        return False, bearing, distance, False, False, 0.0
    try:
        loudness = _clamp(float(event.get("loudness", 1.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        loudness = 0.0
    occluded = bool(event.get("occluded", False))
    masked = bool(event.get("masked", event.get("masking", False)))
    try:
        confidence = _clamp(float(event.get("confidence", 1.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence *= (1.0 - 0.45 * float(occluded)) * (1.0 - 0.35 * float(masked)) * loudness
    return confidence > 0.0, bearing, distance, occluded, masked, confidence


def _sound_category(event_type: str) -> str:
    if event_type in _SOUND_IDS:
        return event_type
    if "grenade" in event_type or "smoke" in event_type or "flash" in event_type:
        return "grenade"
    if event_type in {"weapon_fire", "weapon_reload", "footstep", "jump"}:
        return event_type
    return "unknown"


def _update_smokes(memory: ObservationMemory, events: Sequence[Mapping[str, Any]], now: float) -> None:
    for event in events:
        kind = _event_type(event)
        if kind == "smoke_start":
            position = _position_from_event(event)
            if position is None:
                continue
            entity_id = str(event.get("entity_id", f"{position}:{now}"))
            try:
                duration = max(0.1, float(event.get("duration_s", 18.0)))
                radius = max(1.0, float(event.get("radius", 144.0)))
            except (TypeError, ValueError):
                duration, radius = 18.0, 144.0
            memory._smokes[entity_id] = _SmokeState(position, now, duration, radius, entity_id)
        elif kind == "smoke_expired":
            entity_id = event.get("entity_id")
            if entity_id is not None:
                memory._smokes.pop(str(entity_id), None)
    expired = [key for key, smoke in memory._smokes.items() if now - smoke.started_at > smoke.duration_s]
    for key in expired:
        memory._smokes.pop(key, None)


def _smoke_line_occlusion(
    observer_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    memory: ObservationMemory,
) -> float:
    coverage = 0.0
    for smoke in memory._smokes.values():
        distance = _distance_to_segment(smoke.position, observer_position, target_position)
        if distance < smoke.radius:
            coverage = max(coverage, _clamp(1.0 - distance / smoke.radius, 0.0, 1.0))
    return coverage


def _encode_u8(value: float) -> int:
    return _clamp_int(round(_clamp(value, 0.0, 1.0) * 255.0), 0, 255)


def _decode_u8(value: int) -> float:
    return value / 255.0


def _encode_bearing(value: float) -> int:
    return _clamp_int(round(_clamp(value, -327.67, 327.67) * 100.0), -32768, 32767)


def _decode_bearing(value: int) -> float:
    return value / 100.0


def _encode_observation(projected: ProjectedObservation) -> BotObservationV1:
    payload = bytearray(256)
    struct.pack_into(
        "<4sBBBB",
        payload,
        0,
        OBSERVATION_MAGIC,
        OBSERVATION_SCHEMA_VERSION,
        int(projected.phase),
        int(projected.side),
        _MAP_IDS[projected.map_name],
    )
    self_flags = _clamp_int(projected.self_flags, 0, 255)
    effects = (_encode_u8(projected.flash_recovery) & 0x3F) | (_encode_u8(projected.smoke_occlusion) & 0xC0)
    if projected.smoke_occlusion < 1.0:
        effects = (_encode_u8(projected.flash_recovery) & 0x3F) | (_clamp_int(round(projected.smoke_occlusion * 3.0), 0, 3) << 6)
    position = tuple(_clamp_int(round(value), -32768, 32767) for value in projected.self_position)
    velocity = tuple(_clamp_int(round(_clamp(value / 16.0, -128.0, 127.0)), -128, 127) for value in projected.self_velocity[:2])
    struct.pack_into(
        _SELF_STRUCT.format,
        payload,
        8,
        _clamp_int(projected.self_entity_id, 0, 65535),
        _clamp_int(projected.self_health, 0, 255),
        _clamp_int(projected.self_armor, 0, 255),
        _clamp_int(projected.self_money, 0, 65535),
        *position,
        _clamp_int(round(_clamp(projected.self_yaw_deg, -180.0, 180.0) * 10.0), -32768, 32767),
        _clamp_int(round(_clamp(projected.self_pitch_deg, -90.0, 90.0) * 10.0), -32768, 32767),
        *velocity,
        self_flags,
        effects,
    )
    conditions = [
        *(_encode_u8(value) for value in projected.difficulty),
        *(_encode_u8(value) for value in projected.personality),
        _clamp_int(projected.current_area, 0, 255),
        _clamp_int(projected.target_area, 0, 255),
        _encode_u8(projected.flash_recovery),
        _encode_u8(projected.smoke_occlusion),
        _encode_u8(projected.action_cooldown),
        _WEAPON_IDS.get(projected.weapon_name, 0),
        _clamp_int(projected.stage_flags, 0, 255),
        0,
    ]
    payload[28:44] = bytes(conditions)
    for index, player in enumerate(projected.players):
        offset = 44 + index * _PLAYER_STRUCT.size
        struct.pack_into(
            _PLAYER_STRUCT.format,
            payload,
            offset,
            _clamp_int(player.entity_id, -32768, 32767),
            _clamp_int(player.relation, -128, 127),
            _clamp_int(player.flags, 0, 255),
            _encode_bearing(player.bearing_deg),
            _clamp_int(round(_clamp(player.elevation_deg, -128.0, 127.0)), -128, 127),
            _clamp_int(round(_clamp(player.distance, 0.0, 65535.0)), 0, 65535),
            _clamp_int(round(_clamp(player.age_s, 0.0, 25.5) * 10.0), 0, 255),
            _encode_u8(player.confidence),
            _encode_u8(player.visible_ratio),
        )
    for index, sound in enumerate(projected.sounds):
        offset = 152 + index * 3
        category_id = _SOUND_IDS.get(sound.category, 0) & 0x1F
        if sound.occluded:
            category_id |= 1 << 5
        if sound.masked:
            category_id |= 1 << 6
        if sound.confidence > 0.0:
            category_id |= 1 << 7
        bearing = _clamp_int(round(_clamp(sound.bearing_deg / 2.0, -128.0, 127.0)), -128, 127)
        distance = _clamp_int(round(_clamp(sound.distance / 256.0, 0.0, 15.0)), 0, 15)
        age = _clamp_int(round(_clamp(sound.age_s / 0.25, 0.0, 15.0)), 0, 15)
        payload[offset : offset + 3] = bytes((category_id, bearing & 0xFF, (distance << 4) | age))
    for index, event in enumerate(projected.events):
        event_id = _EVENT_IDS.get(event.category, 0) & 0x1F
        age = _clamp_int(round(_clamp(event.age_s / 1.0, 0.0, 7.0)), 0, 7)
        payload[200 + index] = event_id | (age << 5)
    for index, ray in enumerate(projected.rays):
        payload[216 + index] = _clamp_int(round(_clamp(ray / 4096.0, 0.0, 1.0) * 63.0), 0, 63)
    return BotObservationV1(bytes(payload))


def decode_observation(observation: BotObservationV1 | bytes) -> ProjectedObservation:
    payload = observation.to_bytes() if isinstance(observation, BotObservationV1) else bytes(observation)
    if len(payload) != 256:
        raise ValueError("observation payload must be 256 bytes")
    magic, version, phase, side, map_id = struct.unpack_from("<4sBBBB", payload, 0)
    if magic != OBSERVATION_MAGIC or version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("unsupported observation payload")
    if map_id not in _MAP_NAMES:
        raise ValueError(f"unknown map id {map_id}")
    values = _SELF_STRUCT.unpack_from(payload, 8)
    entity_id, health, armor, money, px, py, pz, yaw, pitch, vx, vy, flags, effects = values
    condition_values = tuple(payload[28:44])
    difficulty = tuple(_decode_u8(value) for value in condition_values[:4])
    personality = tuple(_decode_u8(value) for value in condition_values[4:8])
    players = []
    for index in range(PLAYER_SLOT_COUNT):
        values = _PLAYER_STRUCT.unpack_from(payload, 44 + index * _PLAYER_STRUCT.size)
        player_entity, relation, player_flags, bearing, elevation, distance, age, confidence, visible = values
        players.append(
            ProjectedPlayer(
                entity_id=player_entity,
                relation=relation,
                flags=player_flags,
                bearing_deg=_decode_bearing(bearing),
                elevation_deg=float(elevation),
                distance=float(distance),
                age_s=age / 10.0,
                confidence=_decode_u8(confidence),
                visible_ratio=_decode_u8(visible),
            )
        )
    sounds = []
    for index in range(SOUND_SLOT_COUNT):
        category_flags, bearing, packed = payload[152 + index * 3 : 155 + index * 3]
        category_id = category_flags & 0x1F
        sounds.append(
            ProjectedSound(
                category=_SOUND_NAMES.get(category_id, "unknown"),
                bearing_deg=float(struct.unpack("<b", bytes((bearing,)))[0] * 2),
                distance=float((packed >> 4) * 256),
                age_s=float(packed & 0x0F) * 0.25,
                occluded=bool(category_flags & (1 << 5)),
                masked=bool(category_flags & (1 << 6)),
                confidence=1.0 if category_flags & (1 << 7) else 0.0,
            )
        )
    events_values = []
    for index in range(EVENT_SLOT_COUNT):
        packed = payload[200 + index]
        events_values.append(
            ProjectedEvent(
                category=_EVENT_NAMES.get(packed & 0x1F, "unknown"),
                age_s=float(packed >> 5),
                known_participants=False,
            )
        )
    rays = tuple(float(value & 0x3F) / 63.0 * 4096.0 for value in payload[216:256])
    return ProjectedObservation(
        map_name=_MAP_NAMES[map_id],
        phase=phase,
        side=side,
        self_entity_id=entity_id,
        self_health=health,
        self_armor=armor,
        self_money=money,
        self_position=(float(px), float(py), float(pz)),
        self_yaw_deg=yaw / 10.0,
        self_pitch_deg=pitch / 10.0,
        self_velocity=(float(vx * 16), float(vy * 16), 0.0),
        self_flags=flags,
        flash_recovery=_decode_u8(condition_values[10]),
        smoke_occlusion=_decode_u8(condition_values[11]),
        action_cooldown=_decode_u8(condition_values[12]),
        weapon_name=_WEAPON_NAMES.get(condition_values[13], ""),
        difficulty=difficulty,
        personality=personality,
        current_area=condition_values[8],
        target_area=condition_values[9],
        stage_flags=condition_values[14],
        players=tuple(players),
        sounds=tuple(sounds),
        events=tuple(events_values),
        rays=rays,
    )


def _event_slot(event: Mapping[str, Any], now: float, observer_id: int | str) -> ProjectedEvent:
    kind = _event_type(event)
    source_id = _source_id(event)
    known = source_id is not None and str(source_id) == str(observer_id)
    if known:
        kind = _SELF_EVENT_NAMES.get(kind, kind)
    return ProjectedEvent(category=kind if kind in _EVENT_IDS else "unknown", age_s=now, known_participants=known)


def _ray_values(observer: Mapping[str, Any]) -> tuple[float, ...]:
    raw = observer.get("rays", ())
    values: list[float] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for ray in raw[:RAY_COUNT]:
            if isinstance(ray, Mapping):
                value = ray.get("distance", ray.get("range", 4096.0))
            else:
                value = ray
            try:
                values.append(max(0.0, float(value)))
            except (TypeError, ValueError):
                values.append(4096.0)
    values.extend([4096.0] * (RAY_COUNT - len(values)))
    return tuple(values[:RAY_COUNT])


def project_observation(
    raw_tick: Any,
    observer_id: int | str,
    memory: ObservationMemory,
) -> BotObservationV1:
    if not isinstance(memory, ObservationMemory):
        raise TypeError("memory must be ObservationMemory")
    players = tuple(dict(player) for player in getattr(raw_tick, "players", ()))
    events = tuple(dict(event) for event in getattr(raw_tick, "events", ()))
    observer = _find_observer(players, observer_id)
    now = float(getattr(raw_tick, "demo_time_s"))
    tick_number = int(getattr(raw_tick, "server_tick"))
    observer_key = str(observer_id)
    previous_time = memory._last_time.get(observer_key)
    if previous_time is not None and now < previous_time:
        raise ValueError("observation times must be monotonic per observer")
    memory._last_time[observer_key] = now
    _update_smokes(memory, events, now)
    observer_position = _as_vector(observer.get("position")) or (0.0, 0.0, 0.0)
    # Demo snapshots record the same eye origin used by GetClientEyePosition.
    observer_eye_position = _as_vector(observer.get("position_eyes")) or observer_position
    observer_pitch = float(observer.get("view_pitch_deg", 0.0))
    current_sounds: list[ProjectedSound] = []
    audible_sources: dict[
        str, tuple[float, float, float, bool, bool, float, tuple[float, float, float] | None]
    ] = {}
    for event in events:
        audible, bearing, distance, occluded, masked, confidence = _event_is_audible(
            event, observer_id, observer_position, memory
        )
        if not audible:
            continue
        try:
            observer_yaw = float(observer.get("view_yaw_deg", 0.0))
        except (TypeError, ValueError):
            observer_yaw = 0.0
        bearing = (bearing - observer_yaw + 180.0) % 360.0 - 180.0
        category = _sound_category(_event_type(event))
        current_sounds.append(
            ProjectedSound(
                category=category,
                bearing_deg=bearing + _stable_noise(memory, observer_id, _source_id(event), tick_number, "sound"),
                distance=distance,
                age_s=now,
                occluded=occluded,
                masked=masked,
                loudness=_clamp(float(event.get("loudness", 1.0)), 0.0, 1.0),
                confidence=confidence,
            )
        )
        source_id = _source_id(event)
        if source_id is not None:
            audible_sources[str(source_id)] = (
                bearing,
                0.0,
                distance,
                occluded,
                masked,
                confidence,
                _position_from_event(event),
            )
    history = memory._sounds.setdefault(observer_key, [])
    history.extend(current_sounds)
    memory._sounds[observer_key] = [sound for sound in history if now - sound.age_s <= 4.0][-SOUND_SLOT_COUNT:]
    current_events = [_event_slot(event, now, observer_id) for event in events if _event_type(event) in _EVENT_IDS]
    event_history = memory._events.setdefault(observer_key, [])
    event_history.extend(current_events)
    memory._events[observer_key] = [event for event in event_history if now - event.age_s <= 8.0][-EVENT_SLOT_COUNT:]

    phase = _phase_from_tick(raw_tick, memory.phase)
    memory.phase = phase
    side = _side_from_observer(observer, memory)
    projected_players: list[ProjectedPlayer] = []
    for player in sorted(players, key=lambda item: str(_player_id(item))):
        player_id = _player_id(player)
        if player_id is None or _matches_observer(player_id, observer_id):
            continue
        relation = 1 if player.get("team") == observer.get("team") else -1
        alive = bool(player.get("is_alive", player.get("health", 0) > 0))
        direct = _is_direct(player, observer_id)
        radar = _is_radar(player, observer_id)
        audible_data = audible_sources.get(str(player_id))
        audible = audible_data is not None
        flags = 0
        if direct:
            flags |= PLAYER_DIRECT
        if radar:
            flags |= PLAYER_RADAR
        if audible:
            flags |= PLAYER_AUDIBLE
        if alive:
            flags |= PLAYER_ALIVE
        teammate = relation == 1
        disclosed = teammate or direct or radar or audible
        if not disclosed:
            projected_players.append(ProjectedPlayer(relation=relation))
            continue
        if alive:
            flags |= PLAYER_KNOWN if teammate or direct or bool(player.get("identity_known", False)) else 0
        if bool(player.get("peripheral", False)) or float(player.get("visibility_delay_s", 0.0)) > 0.0:
            flags |= PLAYER_PERIPHERAL
        sound_only = audible and not direct and not radar and not teammate
        target_position = _as_vector(player.get("position"))
        if sound_only and audible_data is not None:
            target_position = audible_data[6]
        if target_position is None:
            if sound_only and audible_data is not None:
                bearing, elevation, distance, occluded, masked, confidence = audible_data[:6]
                noise = _stable_noise(memory, observer_id, player_id, tick_number, "player")
                bearing += noise * 0.5
                distance *= 1.0 + noise * 0.01
                projected_players.append(
                    ProjectedPlayer(
                        entity_id=0,
                        relation=relation,
                        flags=flags,
                        bearing_deg=bearing,
                        elevation_deg=elevation,
                        distance=max(0.0, distance),
                        age_s=0.0,
                        confidence=confidence,
                        visible_ratio=confidence,
                    )
                )
                continue
            projected_players.append(ProjectedPlayer(entity_id=0, relation=relation, flags=flags))
            continue
        bearing, elevation, distance = _geometry(observer_eye_position, target_position)
        # The wire field is relative Source pitch (positive down), not world elevation.
        elevation = math.fmod((-elevation) % 360.0 - observer_pitch, 360.0)
        if elevation > 180.0:
            elevation -= 360.0
        elif elevation < -180.0:
            elevation += 360.0
        try:
            observer_yaw = float(observer.get("view_yaw_deg", 0.0))
        except (TypeError, ValueError):
            observer_yaw = 0.0
        bearing = (bearing - observer_yaw + 180.0) % 360.0 - 180.0
        noise = _stable_noise(memory, observer_id, player_id, tick_number, "player")
        bearing += noise * 0.5
        distance *= 1.0 + noise * 0.01
        smoke = _smoke_line_occlusion(observer_position, target_position, memory)
        if "smoke_occlusion" in player:
            try:
                smoke = max(smoke, _clamp(float(player["smoke_occlusion"]), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
        visible_ratio = _visibility_fraction(player, observer_id) if direct else 0.0
        if direct:
            visible_ratio *= max(0.0, 1.0 - smoke)
        age = _radar_age(player, observer_id) + (memory.radar_delay_s if radar else 0.0)
        if audible:
            _, _, _, occluded, masked, confidence, _ = audible_data
        elif direct:
            delay = max(0.0, float(player.get("visibility_delay_s", memory.peripheral_delay_s)))
            confidence = _clamp(math.exp(-delay / 0.35), 0.0, 1.0) * max(visible_ratio, 0.1)
            age = delay
        elif radar:
            confidence = _clamp(0.85 * math.exp(-age / 1.0), 0.0, 1.0)
        else:
            confidence = 1.0
        identity_known = teammate or direct or bool(player.get("identity_known", False))
        disclosed_id = int(player_id) if identity_known and str(player_id).lstrip("-").isdigit() else 0
        projected_players.append(
            ProjectedPlayer(
                entity_id=disclosed_id,
                relation=relation,
                flags=flags,
                bearing_deg=bearing,
                elevation_deg=elevation,
                distance=max(0.0, distance),
                age_s=max(0.0, age),
                confidence=confidence,
                visible_ratio=visible_ratio if direct else (confidence if audible else 0.0),
            )
        )
    projected_players = projected_players[:PLAYER_SLOT_COUNT]
    projected_players.extend(ProjectedPlayer() for _ in range(PLAYER_SLOT_COUNT - len(projected_players)))

    sound_slots = [
        ProjectedSound(
            category=sound.category,
            bearing_deg=sound.bearing_deg,
            distance=sound.distance,
            age_s=max(0.0, now - sound.age_s),
            occluded=sound.occluded,
            masked=sound.masked,
            loudness=sound.loudness,
            confidence=sound.confidence,
        )
        for sound in memory._sounds.get(observer_key, ())
    ][-SOUND_SLOT_COUNT:]
    sound_slots.extend(ProjectedSound() for _ in range(SOUND_SLOT_COUNT - len(sound_slots)))
    event_slots = [
        ProjectedEvent(
            category=event.category,
            age_s=max(0.0, now - event.age_s),
            known_participants=event.known_participants,
        )
        for event in memory._events.get(observer_key, ())
    ][-EVENT_SLOT_COUNT:]
    event_slots.extend(ProjectedEvent() for _ in range(EVENT_SLOT_COUNT - len(event_slots)))
    velocity = _as_vector(observer.get("velocity")) or (0.0, 0.0, 0.0)
    try:
        flash_remaining = max(0.0, float(observer.get("flash_remaining_s", 0.0)))
        flash_duration = max(flash_remaining, float(observer.get("flash_duration_s", 0.0)))
    except (TypeError, ValueError):
        flash_remaining, flash_duration = 0.0, 0.0
    flash_recovery = 1.0 if flash_duration <= 0.0 else _clamp(1.0 - flash_remaining / flash_duration, 0.0, 1.0)
    self_smoke = 0.0
    for smoke in memory._smokes.values():
        self_smoke = max(self_smoke, _clamp(1.0 - math.dist(smoke.position, observer_position) / smoke.radius, 0.0, 1.0))
    if "smoke_occlusion" in observer:
        try:
            self_smoke = max(self_smoke, _clamp(float(observer["smoke_occlusion"]), 0.0, 1.0))
        except (TypeError, ValueError):
            pass
    self_flags = 0
    for bit, name in (
        (1 << 0, "is_alive"),
        (1 << 1, "is_ducking"),
        (1 << 2, "is_walking"),
        (1 << 3, "is_scoped"),
        (1 << 4, "is_airborne"),
        (1 << 5, "is_blinded"),
        (1 << 6, "is_defusing"),
        (1 << 7, "is_planting"),
    ):
        if bool(observer.get(name, False)):
            self_flags |= bit
    raw_stage_flags = observer.get("stage_flags", 0)
    try:
        stage_flags = int(raw_stage_flags)
    except (TypeError, ValueError):
        stage_flags = 0
    projected = ProjectedObservation(
        map_name=memory.map_name,
        phase=int(phase),
        side=side,
        self_entity_id=int(observer.get("entity_id", 0)),
        self_health=int(observer.get("health", 0)),
        self_armor=int(observer.get("armor", 0)),
        self_money=int(observer.get("money", 0)),
        self_position=observer_position,
        self_yaw_deg=float(observer.get("view_yaw_deg", 0.0)),
        self_pitch_deg=float(observer.get("view_pitch_deg", 0.0)),
        self_velocity=velocity,
        self_flags=self_flags,
        flash_recovery=flash_recovery,
        smoke_occlusion=self_smoke,
        action_cooldown=_clamp(float(observer.get("action_cooldown", 0.0)), 0.0, 1.0),
        weapon_name=_weapon_name(observer),
        difficulty=memory.difficulty,
        personality=memory.personality,
        current_area=int(observer.get("area_id", 0)),
        target_area=int(observer.get("target_area", 0)),
        stage_flags=stage_flags,
        players=tuple(projected_players),
        sounds=tuple(sound_slots),
        events=tuple(event_slots),
        rays=_ray_values(observer),
    )
    return _encode_observation(projected)
