"""有界 Demo 序列索引和循环 TBPTT minibatch。"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from hashlib import sha256
import json
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .action_reconstruction import IN_SPEED, ActionLabelV1, reconstruct_action, write_sequence_shard
from .contracts import DataPurpose, Phase, ensure_purpose
from .demo_pipeline import (
    DemoCorpusEntryV1,
    RawTickV1,
    discover_demo_headers,
    ingest_demo,
    read_raw_ticks,
)
from .lineage import DatasetManifestV1, ProductionLineageError
from .observation_projection import ObservationMemory, project_observation


_ACTION_WIDTH = 8
_REQUIRED_COLUMNS = (
    "observation",
    "sequence_id",
    "target_tick",
    "forward",
    "side",
    "up",
    "yaw_delta_deg",
    "pitch_delta_deg",
    "buttons",
    "weapon_select",
    "buy_action",
    "loss_mask",
    "duration_s",
)
_OPTIONAL_COLUMNS = (
    "yaw_deg",
    "utility",
    "position_x",
    "position_y",
    "position_valid",
    "engagement_distance",
    "alive",
)


@dataclass(frozen=True)
class RecurrentMiniBatchV1:
    observations: Any
    actions: Any
    loss_masks: Any
    duration_s: Any
    hidden_state_mask: Any
    sequence_ids: tuple[tuple[str, ...], ...]
    target_ticks: Any | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    burn_in_mask: Any | None = None
    detach_mask: Any | None = None
    sequence_start_mask: Any | None = None

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[1])

    def to(self, device: Any) -> "RecurrentMiniBatchV1":
        return replace(
            self,
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            loss_masks=self.loss_masks.to(device),
            duration_s=self.duration_s.to(device),
            hidden_state_mask=self.hidden_state_mask.to(device),
            target_ticks=None if self.target_ticks is None else self.target_ticks.to(device),
            burn_in_mask=None if self.burn_in_mask is None else self.burn_in_mask.to(device),
            detach_mask=None if self.detach_mask is None else self.detach_mask.to(device),
            sequence_start_mask=None if self.sequence_start_mask is None else self.sequence_start_mask.to(device),
            context={
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.context.items()
            },
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for recurrent dataset batches") from error
    return torch


def _arrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for recurrent dataset batches") from error
    return pq


def _row_value(batch: Any, name: str, index: int, default: Any = None) -> Any:
    if name not in batch.schema.names:
        return default
    return batch.column(batch.schema.get_field_index(name))[index].as_py()


def _iter_rows(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    pq = _arrow_parquet()
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet_file = pq.ParquetFile(str(path))
    missing = [name for name in _REQUIRED_COLUMNS if name not in parquet_file.schema.names]
    if missing:
        raise ValueError(f"sequence shard is missing columns: {', '.join(missing)}")
    available_columns = tuple(
        name for name in (*_REQUIRED_COLUMNS, *_OPTIONAL_COLUMNS)
        if name in parquet_file.schema.names
    )
    for batch in parquet_file.iter_batches(batch_size=max(1, batch_size), columns=list(available_columns)):
        for index in range(batch.num_rows):
            yield {
                name: _row_value(batch, name, index)
                for name in available_columns
            }


def _normalize_observation(value: Any) -> bytes:
    if hasattr(value, "as_py"):
        value = value.as_py()
    value = bytes(value)
    if len(value) != 256:
        raise ValueError("sequence observations must contain 256-byte payloads")
    return value


def _normalize_window(rows: Sequence[Mapping[str, Any]], sequence_length: int) -> RecurrentMiniBatchV1:
    torch = _torch()
    if not rows:
        raise ValueError("recurrent windows cannot be empty")
    batch_size = len(rows)
    observations = torch.zeros((sequence_length, batch_size, 256), dtype=torch.float32)
    actions = torch.zeros((sequence_length, batch_size, _ACTION_WIDTH), dtype=torch.float32)
    loss_masks = torch.zeros((sequence_length, batch_size), dtype=torch.int64)
    target_ticks = torch.zeros((sequence_length, batch_size), dtype=torch.int64)
    duration_s = torch.full((sequence_length, batch_size), 1.0 / 128.0, dtype=torch.float32)
    hidden_state_mask = torch.zeros((sequence_length, batch_size), dtype=torch.bool)
    context: dict[str, Any] = {
        "yaw_deg": torch.zeros((sequence_length, batch_size), dtype=torch.float32),
        "utility": torch.zeros((sequence_length, batch_size), dtype=torch.float32),
        "position_xy": torch.zeros((sequence_length, batch_size, 2), dtype=torch.float32),
        "position_valid": torch.zeros((sequence_length, batch_size), dtype=torch.bool),
        "engagement_distance": torch.full(
            (sequence_length, batch_size), float("nan"), dtype=torch.float32
        ),
        "alive": torch.ones((sequence_length, batch_size), dtype=torch.bool),
    }
    sequence_ids: list[str] = []
    for batch_index, row_group in enumerate(rows):
        sequence_id = str(row_group[0]["sequence_id"])
        if not sequence_id:
            raise ValueError("sequence_id values must be non-empty")
        sequence_ids.append(sequence_id)
        if any(str(row["sequence_id"]) != sequence_id for row in row_group):
            raise ValueError("a recurrent window cannot cross sequence IDs")
        for time_index, row in enumerate(row_group[:sequence_length]):
            observations[time_index, batch_index] = torch.tensor(
                list(_normalize_observation(row["observation"])), dtype=torch.float32
            ) / 255.0
            actions[time_index, batch_index] = torch.tensor(
                [
                    float(row["forward"]),
                    float(row["side"]),
                    float(row["up"]),
                    float(row["yaw_delta_deg"]),
                    float(row["pitch_delta_deg"]),
                    float(row["buttons"]),
                    float(row["weapon_select"]),
                    float(row["buy_action"]),
                ],
                dtype=torch.float32,
            )
            loss_masks[time_index, batch_index] = int(row["loss_mask"])
            target_ticks[time_index, batch_index] = int(row.get("target_tick", 0) or 0)
            duration_s[time_index, batch_index] = max(float(row["duration_s"]), 1.0 / 128.0)
            hidden_state_mask[time_index, batch_index] = time_index > 0
            context["yaw_deg"][time_index, batch_index] = float(row.get("yaw_deg") or 0.0)
            context["utility"][time_index, batch_index] = float(row.get("utility") or 0.0)
            position_x = row.get("position_x")
            position_y = row.get("position_y")
            if position_x is not None and position_y is not None:
                context["position_xy"][time_index, batch_index] = torch.tensor(
                    [float(position_x), float(position_y)], dtype=torch.float32
                )
            context["position_valid"][time_index, batch_index] = bool(
                row.get("position_valid", position_x is not None and position_y is not None)
            )
            distance = row.get("engagement_distance")
            if distance is not None:
                context["engagement_distance"][time_index, batch_index] = float(distance)
            alive = row.get("alive")
            if alive is not None:
                context["alive"][time_index, batch_index] = bool(alive)
    # Older sequence shards were generated before the walking UserCmd bit was
    # represented by the action labeler.  The observation contract still
    # carries self_flags.is_walking at byte 26, so recover IN_SPEED at load
    # time without rewriting or duplicating the canonical Parquet corpus.
    self_flags = torch.round(observations[:, :, 26] * 255.0).to(torch.int64)
    walking = (self_flags & (1 << 2)) != 0
    buttons = actions[:, :, 5].to(torch.int64)
    actions[:, :, 5] = (buttons | walking.to(torch.int64) * IN_SPEED).to(torch.float32)
    time_major_ids = tuple(tuple(sequence_ids[index] for index in range(batch_size)) for _ in range(sequence_length))
    return RecurrentMiniBatchV1(
        observations=observations,
        actions=actions,
        loss_masks=loss_masks,
        duration_s=duration_s,
        hidden_state_mask=hidden_state_mask,
        sequence_ids=time_major_ids,
        target_ticks=target_ticks,
        context=context,
    )


def _path_windows(path: Path, sequence_length: int) -> Iterator[list[dict[str, Any]]]:
    current: list[dict[str, Any]] = []
    current_sequence: str | None = None
    for row in _iter_rows(path, sequence_length):
        sequence_id = str(row["sequence_id"])
        if current_sequence is None:
            current_sequence = sequence_id
        if sequence_id != current_sequence:
            if current:
                yield current
            current = []
            current_sequence = sequence_id
        current.append(row)
        if len(current) == sequence_length:
            yield current
            current = []
    if current:
        yield current


def _window_has_action_signal(window: Sequence[Mapping[str, Any]]) -> bool:
    """Identify windows that contain a non-neutral discrete action."""

    for row in window:
        try:
            if int(row.get("buttons", 0) or 0) != 0:
                return True
            if int(row.get("weapon_select", -1) or -1) != -1:
                return True
            if int(row.get("buy_action", 0) or 0) != 0:
                return True
            if float(row.get("utility", 0.0) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _bounded_path_windows(
    path: Path,
    sequence_length: int,
    *,
    limit: int | None,
    rng: random.Random,
) -> Iterator[list[dict[str, Any]]]:
    if limit is None:
        yield from _path_windows(path, sequence_length)
        return
    if limit <= 0:
        raise ValueError("max_windows_per_path must be positive when specified")
    priority_reservoir: list[list[dict[str, Any]]] = []
    normal_reservoir: list[list[dict[str, Any]]] = []
    priority_seen = 0
    normal_seen = 0
    for window in _path_windows(path, sequence_length):
        if _window_has_action_signal(window):
            priority_seen += 1
            if len(priority_reservoir) < limit:
                priority_reservoir.append(window)
            else:
                replacement = rng.randrange(priority_seen)
                if replacement < limit:
                    priority_reservoir[replacement] = window
            continue
        normal_seen += 1
        if len(normal_reservoir) < limit:
            normal_reservoir.append(window)
        else:
            replacement = rng.randrange(normal_seen)
            if replacement < limit:
                normal_reservoir[replacement] = window
    reservoir = priority_reservoir[:limit]
    if len(reservoir) < limit:
        reservoir.extend(normal_reservoir[: limit - len(reservoir)])
    rng.shuffle(reservoir)
    yield from reservoir


def iter_recurrent_minibatches(
    paths: Sequence[Path],
    sequence_length: int,
    batch_sequences: int,
    seed: int,
    *,
    max_windows_per_path: int | None = None,
) -> Iterator[RecurrentMiniBatchV1]:
    if sequence_length <= 0 or batch_sequences <= 0:
        raise ValueError("sequence_length and batch_sequences must be positive")
    if max_windows_per_path is not None and max_windows_per_path <= 0:
        raise ValueError("max_windows_per_path must be positive when specified")
    ordered_paths = [Path(path).resolve() for path in paths]
    random.Random(seed).shuffle(ordered_paths)
    pending: list[list[dict[str, Any]]] = []
    for path_index, path in enumerate(ordered_paths):
        path_rng = random.Random(seed + path_index * 1_000_003)
        for window in _bounded_path_windows(
            path,
            sequence_length,
            limit=max_windows_per_path,
            rng=path_rng,
        ):
            pending.append(window)
            if len(pending) == batch_sequences:
                yield _normalize_window(pending, sequence_length)
                pending = []
    if pending:
        yield _normalize_window(pending, sequence_length)


def _continuous_path_windows(path: Path, sequence_length: int, *, random_window: bool, rng: random.Random) -> Iterator[list[dict[str, Any]]]:
    rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in _iter_rows(path, sequence_length):
        rows_by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    for rows in rows_by_sequence.values():
        if random_window and len(rows) > sequence_length:
            starts = list(range(0, len(rows) - sequence_length + 1, max(1, sequence_length // 2)))
            if starts[-1] != len(rows) - sequence_length:
                starts.append(len(rows) - sequence_length)
        else:
            starts = list(range(0, len(rows), sequence_length))
        previous_start: int | None = None
        for start in starts:
            window = [dict(row) for row in rows[start : start + sequence_length]]
            if window:
                window[0]["_window_continuation"] = (
                    previous_start is not None and start == previous_start + sequence_length
                )
                window[0]["_window_burn_in"] = bool(random_window and start > 0)
                previous_start = start
            yield window


def iter_continuous_recurrent_minibatches(
    paths: Sequence[Path],
    sequence_length: int,
    batch_sequences: int,
    seed: int,
    *,
    burn_in_ticks: int = 32,
    random_windows: bool = False,
) -> Iterator[RecurrentMiniBatchV1]:
    """Yield v3 windows with continuation/detach and burn-in metadata.

    A continuation window keeps the incoming hidden state at its first row and
    marks that row for detach.  Only a true sequence start clears the hidden
    state.  Burn-in rows are retained for state evolution but have zero loss.

    Windows for one sequence are yielded in temporal order and are never put in
    the same batch.  This makes the caller's batch-consumption order sufficient
    for committing the preceding window's final hidden state before the next
    continuation window requests it.
    """

    if sequence_length <= 0 or batch_sequences <= 0:
        raise ValueError("sequence_length and batch_sequences must be positive")
    if burn_in_ticks < 0 or burn_in_ticks >= sequence_length:
        raise ValueError("burn_in_ticks must be in [0, sequence_length)")
    pending: list[list[dict[str, Any]]] = []
    pending_continuation: list[bool] = []
    for path_index, path in enumerate(tuple(Path(value).resolve() for value in paths)):
        rng = random.Random(seed + path_index * 1_000_003)
        for window in _continuous_path_windows(path, sequence_length, random_window=random_windows, rng=rng):
            if not window:
                continue
            sequence_id = str(window[0]["sequence_id"])
            pending_sequence_ids = {str(value[0]["sequence_id"]) for value in pending if value}
            if pending and sequence_id in pending_sequence_ids:
                yield _normalize_continuous_windows(pending, pending_continuation, sequence_length, burn_in_ticks)
                pending = []
                pending_continuation = []
            pending.append(window)
            pending_continuation.append(bool(window[0].get("_window_continuation", False)))
            if len(pending) == batch_sequences:
                yield _normalize_continuous_windows(pending, pending_continuation, sequence_length, burn_in_ticks)
                pending = []
                pending_continuation = []
    if pending:
        yield _normalize_continuous_windows(pending, pending_continuation, sequence_length, burn_in_ticks)


def _normalize_continuous_windows(
    windows: Sequence[Sequence[Mapping[str, Any]]],
    continuations: Sequence[bool],
    sequence_length: int,
    burn_in_ticks: int,
) -> RecurrentMiniBatchV1:
    batch = _normalize_window(windows, sequence_length)
    torch = _torch()
    hidden_state_mask = torch.zeros((sequence_length, len(windows)), dtype=torch.bool)
    burn_in_mask = torch.zeros((sequence_length, len(windows)), dtype=torch.bool)
    detach_mask = torch.zeros((sequence_length, len(windows)), dtype=torch.bool)
    sequence_start_mask = torch.zeros((sequence_length, len(windows)), dtype=torch.bool)
    for batch_index, (window, continuation) in enumerate(zip(windows, continuations, strict=True)):
        for time_index in range(min(sequence_length, len(window))):
            hidden_state_mask[time_index, batch_index] = continuation or time_index > 0
        if not continuation and window:
            sequence_start_mask[0, batch_index] = True
        if continuation:
            detach_mask[0, batch_index] = True
        if continuation or bool(window[0].get("_window_burn_in", False)):
            burn_in_mask[: min(burn_in_ticks, len(window)), batch_index] = True
    loss_masks = batch.loss_masks.clone()
    loss_masks[burn_in_mask] = 0
    context = dict(batch.context)
    context["burn_in_mask"] = burn_in_mask
    context["detach_mask"] = detach_mask
    context["sequence_start_mask"] = sequence_start_mask
    return replace(
        batch,
        loss_masks=loss_masks,
        hidden_state_mask=hidden_state_mask,
        burn_in_mask=burn_in_mask,
        detach_mask=detach_mask,
        sequence_start_mask=sequence_start_mask,
        context=context,
    )


def sequence_paths(manifest: DatasetManifestV1, split: str) -> tuple[Path, ...]:
    if not isinstance(manifest, DatasetManifestV1):
        raise TypeError("manifest must be DatasetManifestV1")
    split = str(split)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    encoded = manifest.metadata.get(f"{split}_paths_json", "[]")
    try:
        values = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest {split} paths are not valid JSON") from error
    if not isinstance(values, list):
        raise ValueError(f"manifest {split} paths must be a JSON list")
    return tuple(Path(str(value)).resolve() for value in values)


def _player_id(player: Mapping[str, Any]) -> str | None:
    for name in ("entity_id", "steam_id32", "id"):
        value = player.get(name)
        if value is not None:
            return str(value)
    return None


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _sequence_groups(ticks: Sequence[RawTickV1]) -> dict[tuple[str, str], list[RawTickV1]]:
    groups: dict[tuple[str, str], list[RawTickV1]] = {}
    current_round: str | None = None
    for tick in ticks:
        event_round: str | None = None
        round_start_without_number = False
        for event in tick.events:
            for name in ("round_number", "round", "round_id"):
                if name in event:
                    event_round = str(event[name])
                    break
            if event_round is None and str(event.get("type", "")) == "round_start":
                round_start_without_number = True
        if event_round is not None:
            current_round = event_round
        elif round_start_without_number:
            try:
                current_round = str(int(current_round or "0") + 1)
            except ValueError:
                current_round = "1"
        for player in tick.players:
            player_id = _player_id(player)
            if player_id is None:
                continue
            round_number = None
            for name in ("round_number", "round", "round_id"):
                if name in player:
                    round_number = str(player[name])
                    current_round = round_number
                    break
            key = (round_number or event_round or current_round or "0", player_id)
            groups.setdefault(key, []).append(tick)
    return groups


def _event_belongs_to(event: Mapping[str, Any], player_id: str) -> bool:
    for name in ("source_id", "player_id", "shooter_id", "attacker_id", "thrower_id"):
        if name in event and str(event[name]) == player_id:
            return True
    for name in ("source", "player", "shooter", "attacker", "thrower"):
        value = event.get(name)
        if isinstance(value, Mapping) and _player_id(value) == player_id:
            return True
    return False


def _human_record(tick: RawTickV1, next_tick: RawTickV1, player_id: str, duration_s: float) -> dict[str, Any]:
    player = next(
        (value for value in tick.players if _player_id(value) == player_id),
        {},
    )
    position = player.get("position")
    if isinstance(position, Mapping):
        position = (position.get("x"), position.get("y"), position.get("z"))
    if not isinstance(position, Sequence) or isinstance(position, (str, bytes, bytearray)) or len(position) < 2:
        position = None
    else:
        try:
            position = tuple(float(value) for value in position[:3])
        except (TypeError, ValueError):
            position = None
    utility_types = {
        "grenade_projectile_throw",
        "smoke_start",
        "flash_explode",
        "fire_grenade_start",
        "he_explode",
        "decoy_start",
    }
    utility = 0
    engagement_distance: float | None = None
    for event in (*tick.events, *next_tick.events):
        event_type = str(event.get("type", "")).lower()
        if _event_belongs_to(event, player_id) and (
            event_type in utility_types or "grenade" in event_type or event.get("utility")
        ):
            utility += 1
        if _event_belongs_to(event, player_id) and event_type == "kill":
            try:
                engagement_distance = float(event["distance"])
            except (KeyError, TypeError, ValueError):
                pass
    alive = player.get("is_alive", player.get("alive"))
    if alive is None and player.get("health") is not None:
        alive = float(player["health"]) > 0.0
    try:
        yaw_deg = float(player.get("view_yaw_deg", 0.0))
    except (TypeError, ValueError):
        yaw_deg = 0.0
    return {
        "yaw_deg": yaw_deg,
        "duration_s": duration_s,
        "utility": utility,
        "position": position,
        "position_valid": position is not None,
        "engagement_distance": engagement_distance,
        "alive": None if alive is None else bool(alive),
    }


def _sequence_manifest(
    raw_manifest: DatasetManifestV1,
    entry: DemoCorpusEntryV1,
) -> DatasetManifestV1:
    return DatasetManifestV1(
        name=f"{entry.sha256}-mr15",
        purpose=raw_manifest.effective_purpose(),
        parents=(raw_manifest,),
        artifact_type="mr15_demo_sequence_source",
        source_sha256=entry.sha256,
        parser_version=raw_manifest.parser_version,
        projection_version=raw_manifest.projection_version,
        metadata={
            "map_name": "de_mirage",
            "ruleset": "mr15",
            "regulation_max_rounds": "30",
            "demo_sha256": entry.sha256,
            "source_demo_folder": entry.path.parent.name,
        },
    )


def _convert_demo(
    entry: DemoCorpusEntryV1,
    raw_manifest: DatasetManifestV1,
    sequence_root: Path,
    split: str,
) -> tuple[Path, ...]:
    raw_path = raw_manifest.metadata.get("raw_arrow_path") or raw_manifest.metadata.get("path")
    if not raw_path:
        raise ValueError(f"raw manifest {raw_manifest.name} does not declare an Arrow path")
    ticks = read_raw_ticks(raw_path)
    source_manifest = _sequence_manifest(raw_manifest, entry)
    paths: list[Path] = []
    for (round_number, player_id), group in sorted(_sequence_groups(ticks).items()):
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda tick: tick.server_tick)
        memory = ObservationMemory(phase=Phase.LIVE)
        observations = []
        actions: list[ActionLabelV1] = []
        human_records: list[Mapping[str, Any]] = []
        sequence_id = f"{entry.sha256}/de_mirage/{round_number}/{player_id}"
        for index in range(len(group) - 1):
            current = group[index]
            following = group[index + 1]
            observations.append(project_observation(current, player_id, memory).to_bytes())
            action = reconstruct_action(
                group[index - 1] if index else current,
                current,
                following,
                observer_id=player_id,
            )
            actions.append(
                ActionLabelV1(
                    target_tick=action.target_tick,
                    forward=action.forward,
                    side=action.side,
                    up=action.up,
                    yaw_delta_deg=action.yaw_delta_deg,
                    pitch_delta_deg=action.pitch_delta_deg,
                    buttons=action.buttons,
                    weapon_select=action.weapon_select,
                    buy_action=action.buy_action,
                    loss_mask=action.loss_mask,
                    duration_s=action.duration_s,
                    yaw_rate_deg_s=action.yaw_rate_deg_s,
                    pitch_rate_deg_s=action.pitch_rate_deg_s,
                    frame_delta_ticks=action.frame_delta_ticks,
                    sequence_id=sequence_id,
                )
            )
            human_records.append(
                _human_record(current, following, player_id, action.duration_s)
            )
        output = sequence_root / split / entry.path.parent.name / (
            f"{_safe_filename(entry.sha256[:16])}-r{_safe_filename(round_number)}-p{_safe_filename(player_id)}.parquet"
        )
        overtime = any(
            bool(player.get("overtime", False))
            for tick in group
            for player in tick.players
            if _player_id(player) == player_id
        ) or round_number.isdigit() and int(round_number) > 30
        write_sequence_shard(
            output,
            observations,
            actions,
            source_manifest=source_manifest,
            phase="live",
            human_records=human_records,
            metadata={
                "ruleset": "mr15",
                "regulation_max_rounds": "30",
                "demo_sha256": entry.sha256,
                "source_demo_folder": entry.path.parent.name,
                "overtime": str(overtime).lower(),
            },
        )
        paths.append(output.resolve())
    return tuple(paths)


def ingest_demo_corpus(
    input_root: Path,
    data_root: Path,
    extractor_path: Path,
    purpose: DataPurpose,
    expected_count: int = 10,
    *,
    allow_header_only: bool = False,
) -> DatasetManifestV1:
    requested_purpose = ensure_purpose(purpose)
    input_root = Path(input_root).resolve()
    root = Path(data_root).resolve()
    try:
        root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError("data root must be outside the read-only demo root")
    if requested_purpose is DataPurpose.TEST_ONLY and expected_count != 10:
        raise ValueError("test-only corpus requires exactly 10 Mirage demos")
    entries = tuple(sorted(discover_demo_headers(input_root), key=lambda item: item.sha256))
    mirage = tuple(entry for entry in entries if entry.header.map_name == "de_mirage")
    if len(mirage) != expected_count:
        raise ValueError(f"expected {expected_count} Mirage demos, found {len(mirage)}")
    hashes = [entry.sha256 for entry in mirage]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Mirage Demo SHA-256 values must be unique")
    sequence_root = root / "sequences"
    split_entries = {
        "train": mirage[:8],
        "validation": mirage[8:9],
        "test": mirage[9:],
    }
    parents: list[DatasetManifestV1] = []
    split_paths: dict[str, tuple[Path, ...]] = {}
    seen_parent_names: set[str] = set()
    for split, split_values in split_entries.items():
        paths: list[Path] = []
        for entry in split_values:
            raw_manifest = ingest_demo(
                entry.path,
                root,
                requested_purpose,
                extractor_path=extractor_path,
                allow_header_only=allow_header_only,
            )
            if raw_manifest.effective_purpose() is not requested_purpose:
                raise ProductionLineageError(
                    f"Demo {entry.path} purpose does not match corpus purpose"
                )
            parent = _sequence_manifest(raw_manifest, entry)
            if parent.name not in seen_parent_names:
                parents.append(parent)
                seen_parent_names.add(parent.name)
            paths.extend(_convert_demo(entry, raw_manifest, sequence_root, split))
        split_paths[split] = tuple(paths)
    metadata: dict[str, str] = {
        "map_name": "de_mirage",
        "ruleset": "mr15",
        "regulation_max_rounds": "30",
        "expected_demo_count": str(expected_count),
        "splits_json": json.dumps(
            {
                split: [entry.sha256 for entry in split_values]
                for split, split_values in split_entries.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source_demo_folders_json": json.dumps(
            {
                split: [
                    {
                        "sha256": entry.sha256,
                        "source_demo_folder": entry.path.parent.name,
                    }
                    for entry in split_values
                ]
                for split, split_values in split_entries.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "input_root": str(input_root),
        "corpus_sha256": sha256("".join(hashes).encode("ascii")).hexdigest(),
    }
    for split, paths in split_paths.items():
        metadata[f"{split}_paths_json"] = json.dumps([str(path) for path in paths], separators=(",", ":"))
    return DatasetManifestV1(
        name="prodemo-mr15-corpus",
        purpose=requested_purpose,
        parents=tuple(parents),
        artifact_type="mr15_sequence_corpus",
        source_sha256=metadata["corpus_sha256"],
        parser_version="demoinfocs-golang/v3.3.0",
        projection_version="ObservationProjectionV1",
        metadata=metadata,
    )
