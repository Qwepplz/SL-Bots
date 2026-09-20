"""Source 1 GOTV ingestion and native-tick Arrow IPC persistence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from collections.abc import Iterable, Mapping
import struct
from typing import Any

from .contracts import DataPurpose, ensure_purpose
from .lineage import DatasetManifestV1, ProductionLineageError


DEMO_HEADER_SIZE = 1072
DEMO_FILE_STAMP = b"HL2DEMO"
PARSER_VERSION = "demoinfocs-golang/v3.3.0"
RAW_SCHEMA_VERSION = "RawTickV1"


class DemoHeaderError(ValueError):
    """Raised when a file does not contain a valid Source 1 demo header."""


class DemoExtractorUnavailable(RuntimeError):
    """Raised when full Demo extraction was requested without demoextract."""


@dataclass(frozen=True)
class DemoHeaderV1:
    file_stamp: str
    demo_protocol: int
    network_protocol: int
    server_name: str
    client_name: str
    map_name: str
    game_directory: str
    playback_time_s: float
    playback_ticks: int
    playback_frames: int
    signon_length: int


@dataclass(frozen=True)
class DemoCorpusEntryV1:
    path: Path
    header: DemoHeaderV1
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if len(self.sha256) != 64:
            raise ValueError("demo sha256 must be a 64-character hex digest")
        object.__setattr__(self, "path", path)

    @property
    def absolute_path(self) -> Path:
        return self.path

    @property
    def source_sha256(self) -> str:
        return self.sha256


@dataclass(frozen=True)
class RawTickV1:
    server_tick: int
    demo_time_s: float
    delta_time_s: float
    players: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.server_tick, int):
            raise TypeError("server_tick must be an integer")
        if self.server_tick < 0:
            raise ValueError("server_tick must be non-negative")
        if self.demo_time_s < 0.0 or self.delta_time_s < 0.0:
            raise ValueError("demo time and delta time must be non-negative")
        object.__setattr__(self, "players", tuple(dict(player) for player in self.players))
        object.__setattr__(self, "events", tuple(dict(event) for event in self.events))


def _read_c_string(header: bytes, offset: int, length: int) -> str:
    value = header[offset : offset + length].split(b"\x00", 1)[0]
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DemoHeaderError("demo header contains invalid UTF-8") from error


def read_demo_header(path: str | Path) -> DemoHeaderV1:
    source = Path(path)
    with source.open("rb") as stream:
        header = stream.read(DEMO_HEADER_SIZE)
    if len(header) != DEMO_HEADER_SIZE:
        raise DemoHeaderError("demo header is truncated")
    if not header[:8].startswith(DEMO_FILE_STAMP):
        raise DemoHeaderError("expected HL2DEMO in the first 8 bytes")
    demo_protocol, network_protocol = struct.unpack_from("<ii", header, 8)
    playback_time_s, playback_ticks, playback_frames, signon_length = struct.unpack_from(
        "<fiii", header, 1056
    )
    return DemoHeaderV1(
        file_stamp="HL2DEMO",
        demo_protocol=demo_protocol,
        network_protocol=network_protocol,
        server_name=_read_c_string(header, 16, 260),
        client_name=_read_c_string(header, 276, 260),
        map_name=_read_c_string(header, 536, 260),
        game_directory=_read_c_string(header, 796, 260),
        playback_time_s=playback_time_s,
        playback_ticks=playback_ticks,
        playback_frames=playback_frames,
        signon_length=signon_length,
    )


def _validate_ticks(ticks: tuple[RawTickV1, ...]) -> None:
    previous_tick = None
    previous_time = None
    for tick in ticks:
        if previous_tick is not None and tick.server_tick <= previous_tick:
            raise ValueError("server ticks must be strictly monotonic")
        if previous_time is not None and tick.demo_time_s < previous_time:
            raise ValueError("demo times must be monotonic")
        previous_tick = tick.server_tick
        previous_time = tick.demo_time_s


def _json_records(records: Iterable[Mapping[str, Any]]) -> str:
    return json.dumps(
        list(records),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _arrow():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("pyarrow is required for RawTickV1 Arrow IPC") from error
    return pa


def write_raw_ticks(
    output: str | Path,
    ticks: Iterable[RawTickV1],
    *,
    map_name: str,
    purpose: DataPurpose | str,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    purpose = ensure_purpose(purpose)
    if map_name != "de_mirage":
        raise ValueError("only de_mirage is supported")
    normalized_ticks = tuple(ticks)
    _validate_ticks(normalized_ticks)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pa = _arrow()
    schema_metadata = {
        b"schema": RAW_SCHEMA_VERSION.encode("ascii"),
        b"purpose": purpose.value.encode("ascii"),
        b"map_name": map_name.encode("ascii"),
    }
    if metadata:
        schema_metadata.update({
            key.encode("utf-8"): value.encode("utf-8")
            for key, value in metadata.items()
        })
    schema = pa.schema(
        [
            pa.field("server_tick", pa.int32()),
            pa.field("demo_time_s", pa.float64()),
            pa.field("delta_time_s", pa.float32()),
            pa.field("players_json", pa.string()),
            pa.field("events_json", pa.string()),
        ],
        metadata=schema_metadata,
    )
    table = pa.Table.from_arrays(
        [
            pa.array([tick.server_tick for tick in normalized_ticks], type=pa.int32()),
            pa.array([tick.demo_time_s for tick in normalized_ticks], type=pa.float64()),
            pa.array([tick.delta_time_s for tick in normalized_ticks], type=pa.float32()),
            pa.array([_json_records(tick.players) for tick in normalized_ticks], type=pa.string()),
            pa.array([_json_records(tick.events) for tick in normalized_ticks], type=pa.string()),
        ],
        schema=schema,
    )
    with pa.OSFile(str(output_path), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
    return output_path


def read_raw_ticks(path: str | Path) -> tuple[RawTickV1, ...]:
    pa = _arrow()
    source = pa.memory_map(str(Path(path)), "r")
    try:
        table = pa.ipc.open_file(source).read_all()
    finally:
        source.close()
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    return tuple(
        RawTickV1(
            server_tick=server_tick,
            demo_time_s=demo_time_s,
            delta_time_s=delta_time_s,
            players=tuple(json.loads(players)),
            events=tuple(json.loads(events)),
        )
        for server_tick, demo_time_s, delta_time_s, players, events in zip(
            columns["server_tick"],
            columns["demo_time_s"],
            columns["delta_time_s"],
            columns["players_json"],
            columns["events_json"],
        )
    )


def _under_demo_root(path: Path) -> bool:
    demo_root = Path("E:/Demo").resolve()
    return _under_root(path, demo_root)


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _default_extractor() -> Path | None:
    root = Path(__file__).resolve().parents[2] / "tools" / "demoextract"
    for name in ("demoextract.exe", "demoextract"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_demo_headers(input_root: str | Path) -> tuple[DemoCorpusEntryV1, ...]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    entries: list[DemoCorpusEntryV1] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() == ".dem"),
        key=lambda candidate: str(candidate).lower(),
    ):
        header = read_demo_header(path)
        entries.append(DemoCorpusEntryV1(path=path, header=header, sha256=_sha256(path)))
    return tuple(entries)


def ingest_demo(
    path: str | Path,
    data_root: str | Path,
    purpose: DataPurpose | str,
    *,
    extractor_path: str | Path | None = None,
    allow_header_only: bool = False,
    demo_root: str | Path | None = None,
) -> DatasetManifestV1:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    requested_purpose = ensure_purpose(purpose)
    resolved_demo_root = Path(demo_root).resolve() if demo_root is not None else None
    if resolved_demo_root is not None and not _under_root(source, resolved_demo_root):
        raise ValueError("source demo must be located under demo_root")
    is_test_demo = _under_demo_root(source) or (
        resolved_demo_root is not None and _under_root(source, resolved_demo_root)
    )
    if is_test_demo and requested_purpose is DataPurpose.PRODUCTION:
        raise ProductionLineageError("E:/Demo inputs are always test_only")
    effective_purpose = DataPurpose.TEST_ONLY if is_test_demo else requested_purpose
    header = read_demo_header(source)
    if header.map_name != "de_mirage":
        raise ValueError("only de_mirage is supported")
    root = Path(data_root).resolve()
    source_demo_folder = source.parent.name
    output = root / "raw" / source_demo_folder / f"{source.stem}.arrow"
    extractor = Path(extractor_path).resolve() if extractor_path else _default_extractor()
    if extractor is not None and extractor.is_file():
        completed = subprocess.run(
            [
                str(extractor),
                "--input",
                str(source),
                "--output",
                str(output),
                "--purpose",
                effective_purpose.value,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "demoextract failed")
        if not output.is_file():
            raise RuntimeError("demoextract did not create the requested Arrow file")
        read_raw_ticks(output)
        extractor_mode = "demoinfocs-golang"
    elif allow_header_only:
        write_raw_ticks(
            output,
            (RawTickV1(server_tick=0, demo_time_s=0.0, delta_time_s=0.0),),
            map_name=header.map_name,
            purpose=effective_purpose,
            metadata={"extractor_mode": "header_only_test"},
        )
        extractor_mode = "header_only_test"
    else:
        raise DemoExtractorUnavailable(
            "full extraction requires tools/demoextract/demoextract(.exe)"
        )
    return DatasetManifestV1(
        name=source.stem,
        purpose=effective_purpose,
        artifact_type="raw_tick_arrow",
        source_sha256=_sha256(source),
        parser_version=PARSER_VERSION,
        projection_version=RAW_SCHEMA_VERSION,
        metadata={
            "map_name": header.map_name,
            "raw_arrow_path": str(output),
            "source_demo_folder": source_demo_folder,
            "extractor_mode": extractor_mode,
            "demo_protocol": str(header.demo_protocol),
            "network_protocol": str(header.network_protocol),
        },
    )
