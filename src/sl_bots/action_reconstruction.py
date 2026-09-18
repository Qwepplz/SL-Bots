"""从相邻原始 Tick 重建可通过 UserCmd 表达的动作标签。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import BotActionV1, OBSERVATION_RECORD_SIZE
from .lineage import DatasetManifestV1


ACTION_MASK_FORWARD = 1 << 0
ACTION_MASK_SIDE = 1 << 1
ACTION_MASK_UP = 1 << 2
ACTION_MASK_YAW = 1 << 3
ACTION_MASK_PITCH = 1 << 4
ACTION_MASK_BUTTONS = 1 << 5
ACTION_MASK_WEAPON = 1 << 6
ACTION_MASK_BUY = 1 << 7

IN_ATTACK = 1 << 0
IN_JUMP = 1 << 1
IN_DUCK = 1 << 2
IN_FORWARD = 1 << 3
IN_BACK = 1 << 4
IN_USE = 1 << 5
IN_ATTACK2 = 1 << 11
IN_RELOAD = 1 << 13
IN_SPEED = 1 << 17
IN_MOVELEFT = 1 << 9
IN_MOVERIGHT = 1 << 10


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _vector(player: Mapping[str, Any], name: str) -> tuple[float, float, float] | None:
    value = player.get(name)
    if isinstance(value, Mapping):
        try:
            return float(value["x"]), float(value["y"]), float(value["z"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    return None


def _player_id(player: Mapping[str, Any]) -> Any:
    return player.get("entity_id", player.get("steam_id32", player.get("id")))


def _same_id(left: Any, right: Any) -> bool:
    return left is not None and right is not None and str(left) == str(right)


def _select_player(tick: Any, observer_id: Any | None) -> Mapping[str, Any]:
    players = tuple(getattr(tick, "players", ()))
    if not players:
        raise ValueError("tick has no player state")
    if observer_id is not None:
        for player in players:
            if _same_id(_player_id(player), observer_id):
                return player
        raise KeyError(f"observer {observer_id!r} is not present in the tick")
    for player in players:
        if player.get("is_observer") or player.get("controlled"):
            return player
    return sorted(players, key=lambda player: str(_player_id(player)))[0]


def _delta_time(tick: Any, next_tick: Any) -> float:
    value = float(next_tick.demo_time_s) - float(tick.demo_time_s)
    if value > 0.0:
        return value
    value = float(getattr(next_tick, "delta_time_s", 0.0))
    if value > 0.0:
        return value
    tick_delta = int(next_tick.server_tick) - int(tick.server_tick)
    if tick_delta > 0:
        return tick_delta / 128.0
    raise ValueError("adjacent ticks must have positive duration")


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _events_between(tick: Any, next_tick: Any) -> tuple[Mapping[str, Any], ...]:
    values = list(getattr(tick, "events", ())) + list(getattr(next_tick, "events", ()))
    return tuple(dict(value) for value in values)


def _event_belongs(event: Mapping[str, Any], player_id: Any) -> bool:
    for name in ("source_id", "player_id", "shooter_id", "attacker_id", "thrower_id", "victim_id"):
        if name in event:
            return _same_id(event[name], player_id)
    for name in ("source", "player", "shooter", "attacker", "thrower", "victim"):
        value = event.get(name)
        if isinstance(value, Mapping) and _player_id(value) is not None:
            return _same_id(_player_id(value), player_id)
    return False


def _button_label(player: Mapping[str, Any], events: Sequence[Mapping[str, Any]], player_id: Any) -> tuple[int, bool]:
    if "buttons" in player:
        try:
            return int(player["buttons"]) & 0xFFFFFFFF, True
        except (TypeError, ValueError):
            return 0, False
    buttons = 0
    evidence = False
    state_bits = (
        ("is_ducking", IN_DUCK),
        ("is_walking", IN_SPEED),
    )
    for name, bit in state_bits:
        if name in player:
            evidence = True
            if player.get(name) and bit:
                buttons |= bit
    for event in events:
        if not _event_belongs(event, player_id):
            continue
        kind = str(event.get("type", "")).lower()
        if kind in {"weapon_fire", "player_fire"}:
            buttons |= IN_ATTACK
            evidence = True
        elif kind in {"weapon_fire_alt", "attack2"}:
            buttons |= IN_ATTACK2
            evidence = True
        elif kind in {"player_jump", "jump"}:
            buttons |= IN_JUMP
            evidence = True
        elif kind in {"weapon_reload", "reload"}:
            buttons |= IN_RELOAD
            evidence = True
        elif kind == "use":
            buttons |= IN_USE
            evidence = True
    return buttons, evidence


def _local_movement(tick_player: Mapping[str, Any], next_player: Mapping[str, Any], dt: float) -> tuple[float, float, float, int]:
    velocity = _vector(next_player, "velocity") or _vector(tick_player, "velocity")
    mask = 0
    if velocity is None:
        position = _vector(tick_player, "position")
        next_position = _vector(next_player, "position")
        if position is not None and next_position is not None:
            velocity = tuple((next_position[index] - position[index]) / dt for index in range(3))
    if velocity is None:
        return 0.0, 0.0, 0.0, mask
    try:
        yaw = math.radians(float(tick_player.get("view_yaw_deg", next_player.get("view_yaw_deg", 0.0))))
    except (TypeError, ValueError):
        yaw = 0.0
    forward_speed = velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw)
    side_speed = -velocity[0] * math.sin(yaw) + velocity[1] * math.cos(yaw)
    forward = _clamp(forward_speed / 250.0, -1.0, 1.0)
    side = _clamp(side_speed / 250.0, -1.0, 1.0)
    up = _clamp(velocity[2] / 250.0, -1.0, 1.0)
    mask = ACTION_MASK_FORWARD | ACTION_MASK_SIDE | ACTION_MASK_UP
    return forward, side, up, mask


def _angle_delta(tick_player: Mapping[str, Any], next_player: Mapping[str, Any], name: str) -> tuple[float, float, bool]:
    if name not in tick_player or name not in next_player:
        return 0.0, 0.0, False
    try:
        delta = _wrap_degrees(float(next_player[name]) - float(tick_player[name]))
    except (TypeError, ValueError):
        return 0.0, 0.0, False
    return delta, delta, True


def _weapon_label(tick_player: Mapping[str, Any], next_player: Mapping[str, Any]) -> tuple[int, bool]:
    if "weapon_select" in next_player:
        try:
            return int(next_player["weapon_select"]), True
        except (TypeError, ValueError):
            return -1, False
    if tick_player.get("active_weapon") == next_player.get("active_weapon"):
        return -1, "active_weapon" in tick_player and "active_weapon" in next_player
    if "weapon_id" in next_player:
        try:
            return int(next_player["weapon_id"]), True
        except (TypeError, ValueError):
            pass
    return -1, False


def _buy_label(events: Sequence[Mapping[str, Any]], next_player: Mapping[str, Any]) -> tuple[int, bool]:
    if "buy_action" in next_player:
        try:
            return int(next_player["buy_action"]), True
        except (TypeError, ValueError):
            return 0, False
    for event in events:
        if str(event.get("type", "")).lower() in {"buy", "item_pickup", "purchase"}:
            value = event.get("buy_action", event.get("action", 0))
            try:
                return int(value), True
            except (TypeError, ValueError):
                return 0, False
    return 0, False


@dataclass(frozen=True)
class ActionLabelV1:
    target_tick: int
    forward: float = 0.0
    side: float = 0.0
    up: float = 0.0
    yaw_delta_deg: float = 0.0
    pitch_delta_deg: float = 0.0
    buttons: int = 0
    weapon_select: int = -1
    buy_action: int = 0
    loss_mask: int = 0
    duration_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    pitch_rate_deg_s: float = 0.0
    frame_delta_ticks: int = 0
    sequence_id: str = "sequence-0"

    def __post_init__(self) -> None:
        if not isinstance(self.target_tick, int) or self.target_tick < 0:
            raise ValueError("target_tick must be a non-negative integer")
        for name in ("forward", "side", "up"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
            object.__setattr__(self, name, value)
        for name in ("yaw_delta_deg", "pitch_delta_deg", "duration_s", "yaw_rate_deg_s", "pitch_rate_deg_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 and name == "duration_s":
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0 <= int(self.buttons) <= 0xFFFFFFFF:
            raise ValueError("buttons must fit uint32")
        object.__setattr__(self, "buttons", int(self.buttons))
        sequence_id = str(self.sequence_id)
        if not sequence_id:
            raise ValueError("sequence_id cannot be empty")
        object.__setattr__(self, "sequence_id", sequence_id)

    def to_action(self) -> BotActionV1:
        return BotActionV1(
            target_tick=self.target_tick,
            forward=self.forward,
            side=self.side,
            up=self.up,
            yaw_delta_deg=self.yaw_delta_deg,
            pitch_delta_deg=self.pitch_delta_deg,
            buttons=self.buttons,
            weapon_select=self.weapon_select,
            buy_action=self.buy_action,
            action_valid_mask=self.loss_mask,
        )


def reconstruct_action(
    prev_tick: Any,
    tick: Any,
    next_tick: Any,
    *,
    observer_id: Any | None = None,
) -> ActionLabelV1:
    """重建 Tick N 到 Tick N+1 的 UserCmd 标签，无法唯一确定的维度不进 loss。"""

    del prev_tick
    tick_player = _select_player(tick, observer_id)
    next_player = _select_player(next_tick, observer_id)
    dt = _delta_time(tick, next_tick)
    player_id = _player_id(tick_player)
    forward, side, up, movement_mask = _local_movement(tick_player, next_player, dt)
    yaw_delta, _, yaw_valid = _angle_delta(tick_player, next_player, "view_yaw_deg")
    pitch_delta, _, pitch_valid = _angle_delta(tick_player, next_player, "view_pitch_deg")
    events = _events_between(tick, next_tick)
    buttons, buttons_valid = _button_label(next_player, events, player_id)
    weapon_select, weapon_valid = _weapon_label(tick_player, next_player)
    buy_action, buy_valid = _buy_label(events, next_player)
    loss_mask = movement_mask
    if yaw_valid:
        loss_mask |= ACTION_MASK_YAW
    if pitch_valid:
        loss_mask |= ACTION_MASK_PITCH
    if buttons_valid:
        loss_mask |= ACTION_MASK_BUTTONS
    if weapon_valid:
        loss_mask |= ACTION_MASK_WEAPON
    if buy_valid:
        loss_mask |= ACTION_MASK_BUY
    try:
        tick_delta = int(next_tick.server_tick) - int(tick.server_tick)
    except (TypeError, ValueError):
        tick_delta = 0
    return ActionLabelV1(
        target_tick=int(next_tick.server_tick),
        forward=forward,
        side=side,
        up=up,
        yaw_delta_deg=yaw_delta,
        pitch_delta_deg=pitch_delta,
        buttons=buttons,
        weapon_select=weapon_select,
        buy_action=buy_action,
        loss_mask=loss_mask,
        duration_s=dt,
        yaw_rate_deg_s=abs(yaw_delta) / dt if yaw_valid else 0.0,
        pitch_rate_deg_s=abs(pitch_delta) / dt if pitch_valid else 0.0,
        frame_delta_ticks=max(0, tick_delta),
    )


def write_sequence_shard(
    output: str | Path,
    observations: Sequence[bytes | Any],
    actions: Sequence[ActionLabelV1],
    *,
    source_manifest: DatasetManifestV1,
    phase: str,
    sequence_ids: Sequence[str] | None = None,
    human_records: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> DatasetManifestV1:
    """写入单阶段 Parquet shard，并把源 Demo 与投影版本写入 schema 元数据。"""

    if len(observations) != len(actions):
        raise ValueError("observations and actions must have the same length")
    phase = str(phase).lower()
    if phase not in {"warmup", "knife", "live"}:
        raise ValueError("phase must be warmup, knife or live")
    normalized_observations = []
    for observation in observations:
        value = observation.to_bytes() if hasattr(observation, "to_bytes") else bytes(observation)
        if len(value) != OBSERVATION_RECORD_SIZE:
            raise ValueError(f"every observation must be {OBSERVATION_RECORD_SIZE} bytes")
        normalized_observations.append(value)
    for action in actions:
        if not isinstance(action, ActionLabelV1):
            raise TypeError("actions must contain ActionLabelV1 values")
    if sequence_ids is None:
        normalized_sequence_ids = [action.sequence_id for action in actions]
    else:
        if len(sequence_ids) != len(actions):
            raise ValueError("sequence_ids and actions must have the same length")
        normalized_sequence_ids = [str(value) for value in sequence_ids]
        if any(not value for value in normalized_sequence_ids):
            raise ValueError("sequence_ids cannot contain empty values")
    if human_records is None:
        normalized_human_records: tuple[Mapping[str, Any], ...] = tuple({} for _ in actions)
    else:
        if len(human_records) != len(actions):
            raise ValueError("human_records and actions must have the same length")
        normalized_human_records = tuple(human_records)
        if any(not isinstance(record, Mapping) for record in normalized_human_records):
            raise TypeError("human_records must contain mappings")
    absolute_yaws: list[float] = []
    accumulated_yaw = 0.0
    for action, record in zip(actions, normalized_human_records):
        if record.get("yaw_deg") is None:
            accumulated_yaw += float(action.yaw_delta_deg)
            absolute_yaws.append(accumulated_yaw)
        else:
            absolute_yaws.append(float(record["yaw_deg"]))

    def human_number(record: Mapping[str, Any], name: str, default: float = 0.0) -> float:
        value = record.get(name, default)
        if value in (None, ""):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"human record {name} must be numeric") from error

    utility_values = []
    position_x = []
    position_y = []
    position_valid = []
    engagement_distances = []
    alive_values = []
    for record in normalized_human_records:
        utility = record.get("utility", 0)
        utility_values.append(
            int(utility not in (None, "", 0, False))
            if not isinstance(utility, (int, float))
            else int(utility)
        )
        position = record.get("position")
        valid = bool(record.get("position_valid", position is not None))
        if position is not None and len(position) >= 2:
            position_x.append(human_number({"value": position[0]}, "value"))
            position_y.append(human_number({"value": position[1]}, "value"))
        else:
            position_x.append(None)
            position_y.append(None)
            valid = False
        position_valid.append(valid)
        distance = record.get("engagement_distance")
        engagement_distances.append(
            None if distance in (None, "") else human_number({"value": distance}, "value")
        )
        alive_values.append(
            None if record.get("alive") is None else bool(record.get("alive"))
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for sequence Parquet shards") from error
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_metadata = {
        "schema": "SequenceDatasetV1",
        "purpose": source_manifest.effective_purpose().value,
        "phase": phase,
        "source_sha256": source_manifest.source_sha256 or "",
        "parser_version": source_manifest.parser_version or "",
        "projection_version": "ObservationProjectionV1",
        "lineage_parent": source_manifest.name,
    }
    if metadata:
        schema_metadata.update({str(key): str(value) for key, value in metadata.items()})
    table = pa.table(
        {
            "observation": pa.array(normalized_observations, type=pa.binary(OBSERVATION_RECORD_SIZE)),
            "sequence_id": pa.array(normalized_sequence_ids, type=pa.string()),
            "target_tick": pa.array([action.target_tick for action in actions], type=pa.int32()),
            "forward": pa.array([action.forward for action in actions], type=pa.float32()),
            "side": pa.array([action.side for action in actions], type=pa.float32()),
            "up": pa.array([action.up for action in actions], type=pa.float32()),
            "yaw_delta_deg": pa.array([action.yaw_delta_deg for action in actions], type=pa.float32()),
            "pitch_delta_deg": pa.array([action.pitch_delta_deg for action in actions], type=pa.float32()),
            "buttons": pa.array([action.buttons for action in actions], type=pa.uint32()),
            "weapon_select": pa.array([action.weapon_select for action in actions], type=pa.int16()),
            "buy_action": pa.array([action.buy_action for action in actions], type=pa.int16()),
            "loss_mask": pa.array([action.loss_mask for action in actions], type=pa.uint32()),
            "duration_s": pa.array([action.duration_s for action in actions], type=pa.float32()),
            "yaw_deg": pa.array(absolute_yaws, type=pa.float32()),
            "utility": pa.array(utility_values, type=pa.int32()),
            "position_x": pa.array(position_x, type=pa.float32()),
            "position_y": pa.array(position_y, type=pa.float32()),
            "position_valid": pa.array(position_valid, type=pa.bool_()),
            "engagement_distance": pa.array(engagement_distances, type=pa.float32()),
            "alive": pa.array(alive_values, type=pa.bool_()),
        }
    ).replace_schema_metadata({key.encode("utf-8"): value.encode("utf-8") for key, value in schema_metadata.items()})
    pq.write_table(table, output_path, compression="zstd", use_dictionary=False)
    return DatasetManifestV1(
        name=output_path.stem,
        purpose=source_manifest.effective_purpose(),
        parents=(source_manifest,),
        artifact_type="sequence_parquet",
        source_sha256=source_manifest.source_sha256,
        parser_version=source_manifest.parser_version,
        projection_version="ObservationProjectionV1",
        metadata={
            "path": str(output_path),
            "phase": phase,
            "source_sha256": source_manifest.source_sha256 or "",
            "parser_version": source_manifest.parser_version or "",
            "projection_version": "ObservationProjectionV1",
            **({str(key): str(value) for key, value in metadata.items()} if metadata else {}),
        },
    )
