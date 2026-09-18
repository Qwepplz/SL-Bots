from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .contracts import DataPurpose, ensure_purpose
from .lineage import DatasetManifestV1
from .server_farm import (
    CompleteMatchWaveV1,
    build_server_specs,
    collect_single_match_wave,
)
from .training_gail import BootstrapMatchV1, build_bootstrap_manifest


STATE_SCHEMA = "training-run-state-v1"
DEFAULT_DURATION_SECONDS = 7200.0
CALIBRATION_QUICK_SECONDS = 60.0
CALIBRATION_SOAK_SECONDS = 600.0
CALIBRATION_P99_LIMIT_MS = 7.8125
CALIBRATION_MEMORY_LIMIT_GIB = 4.0
CALIBRATION_MIN_TICKRATE_HZ = 120.0
CALIBRATION_MAX_TICKRATE_HZ = 136.0
ALL_SERVERS_UNAVAILABLE_SECONDS = 30.0


class TrainingOrchestratorError(RuntimeError):
    pass


class InstanceUnavailableError(RuntimeError):
    def __init__(self, instance_id: str, reason: str = "instance unavailable") -> None:
        self.instance_id = str(instance_id)
        self.reason = str(reason)
        super().__init__(f"server instance {self.instance_id} unavailable: {self.reason}")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, DataPurpose):
        return value.value
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    root = path.resolve()
    if root.is_file():
        return _sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(_sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def _config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_value(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    try:
        parameters = tuple(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return callback(*args)
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return callback(*args)
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return callback(*args[: len(positional)])


def _manifest_payload(manifest: DatasetManifestV1) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "purpose": manifest.purpose.value,
        "artifact_type": manifest.artifact_type,
        "source_sha256": manifest.source_sha256,
        "parser_version": manifest.parser_version,
        "projection_version": manifest.projection_version,
        "metadata": dict(manifest.metadata),
        "parents": [_manifest_payload(parent) for parent in manifest.parents],
    }


def _pick(value: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return default


@dataclass(frozen=True)
class CalibrationMetricsV1:
    ticks: int
    rules_validated: bool
    p99_ms: float
    fallback_32_tick: int
    ipc_collisions: int
    invalid_generation: int
    available_memory_gib: float
    gpu_errors: int
    tickrate_hz: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "CalibrationMetricsV1") -> "CalibrationMetricsV1":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("calibration metrics must be a mapping")
        servers = value.get("servers")
        if isinstance(servers, Sequence) and not isinstance(servers, (str, bytes)) and servers:
            normalized = tuple(cls.from_mapping(item) for item in servers)
            return cls(
                ticks=min(item.ticks for item in normalized),
                rules_validated=all(item.rules_validated for item in normalized),
                p99_ms=max(item.p99_ms for item in normalized),
                fallback_32_tick=sum(item.fallback_32_tick for item in normalized),
                ipc_collisions=sum(item.ipc_collisions for item in normalized),
                invalid_generation=sum(item.invalid_generation for item in normalized),
                available_memory_gib=min(item.available_memory_gib for item in normalized),
                gpu_errors=sum(item.gpu_errors for item in normalized),
                tickrate_hz=min(item.tickrate_hz for item in normalized),
            )
        try:
            ticks = int(_pick(value, ("ticks", "tickrate", "tick_rate"), 0))
            rules = _pick(
                value,
                ("rules_validated", "ruleset_validated", "get5_rules_validated"),
                False,
            )
            if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes)):
                rules = all(bool(item) for item in rules)
            elif isinstance(rules, str):
                rules = rules.strip().lower() in {"true", "1", "yes", "validated"}
            p99_ms = float(_pick(value, ("p99_ms", "latency_p99_ms"), math.inf))
            fallback = int(
                _pick(
                    value,
                    ("fallback_32_tick", "fallback_32_tick_count", "zero_32_tick_fallback"),
                    1,
                )
            )
            collisions = int(_pick(value, ("ipc_collisions", "ipc_collision_count"), 1))
            invalid_generation = int(
                _pick(value, ("invalid_generation", "invalid_generation_count"), 1)
            )
            memory = float(
                _pick(value, ("available_memory_gib", "ram_available_gib", "ram_gib"), 0.0)
            )
            gpu_errors = int(_pick(value, ("gpu_errors", "gpu_error_count"), 1))
            tickrate_hz = float(
                _pick(value, ("tickrate_hz", "observed_tickrate_hz", "tick_rate_hz"), 0.0)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("calibration metrics contain invalid values") from error
        if not math.isfinite(p99_ms) or not math.isfinite(memory) or not math.isfinite(tickrate_hz):
            raise ValueError("calibration latency, memory and tickrate must be finite")
        return cls(
            ticks=ticks,
            rules_validated=bool(rules),
            p99_ms=p99_ms,
            fallback_32_tick=fallback,
            ipc_collisions=collisions,
            invalid_generation=invalid_generation,
            available_memory_gib=memory,
            gpu_errors=gpu_errors,
            tickrate_hz=tickrate_hz,
        )

    @property
    def passed(self) -> bool:
        return self.passes()

    def passes(
        self,
        *,
        p99_limit_ms: float = CALIBRATION_P99_LIMIT_MS,
        min_memory_gib: float = CALIBRATION_MEMORY_LIMIT_GIB,
        min_tickrate_hz: float = CALIBRATION_MIN_TICKRATE_HZ,
        max_tickrate_hz: float = CALIBRATION_MAX_TICKRATE_HZ,
    ) -> bool:
        return (
            self.ticks == 128
            and min_tickrate_hz <= self.tickrate_hz <= max_tickrate_hz
            and self.rules_validated
            and self.p99_ms < p99_limit_ms
            and self.fallback_32_tick == 0
            and self.ipc_collisions == 0
            and self.invalid_generation == 0
            and self.available_memory_gib > min_memory_gib
            and self.gpu_errors == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "rules_validated": self.rules_validated,
            "p99_ms": self.p99_ms,
            "fallback_32_tick": self.fallback_32_tick,
            "ipc_collisions": self.ipc_collisions,
            "invalid_generation": self.invalid_generation,
            "available_memory_gib": self.available_memory_gib,
            "gpu_errors": self.gpu_errors,
            "tickrate_hz": self.tickrate_hz,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class CalibrationAttemptV1:
    server_count: int
    stage: str
    duration_s: float
    metrics: CalibrationMetricsV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_count": self.server_count,
            "stage": self.stage,
            "duration_s": self.duration_s,
            "metrics": self.metrics.to_dict(),
        }


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _valid_sha256(value: Any) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("rollout shard_sha256 must be a 64-character hexadecimal digest")
    return normalized


class TrainingOrchestrator:
    def __init__(
        self,
        config: Mapping[str, Any] | str | Path,
        demo_root: str | Path,
        data_root: str | Path,
        server_root: str | Path,
        max_servers: int = 4,
        purpose: DataPurpose | str = DataPurpose.PRODUCTION,
        run_id: str = "gpu-training",
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        calibration_probe: Callable[[int, str, float], Mapping[str, Any] | CalibrationMetricsV1] | None = None,
        bootstrap_runner: Callable[[int], Sequence[BootstrapMatchV1]] | None = None,
        wave_collector: Callable[[float], Sequence[Any] | Mapping[str, Any] | None] | None = None,
        wave_trainer: Callable[[Sequence[Any], int], Any] | None = None,
        generation_publisher: Callable[..., Any] | None = None,
        match_loader: Callable[[int], Any] | None = None,
        stop_sampling: Callable[[float], Any] | None = None,
        truncation_flusher: Callable[[float], Any] | None = None,
        instance_restarter: Callable[[str], Any] | None = None,
        worker_stop: Callable[[], Any] | None = None,
        runtime_stop: Callable[[], Any] | None = None,
        server_stop: Callable[[], Any] | None = None,
        single_wave_collector: Callable[..., Any] | None = None,
        single_wave_trainer: Callable[..., Any] | None = None,
        candidate_exporter: Callable[..., Any] | None = None,
        farm: Any | None = None,
        parent_manifests: Sequence[DatasetManifestV1] = (),
        get5_template_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        if isinstance(config, Mapping):
            self.config = dict(_canonical_value(config))
        else:
            self.config = self._load_config(config)
        configured_state_schema = str(self.config.get("state_schema", STATE_SCHEMA))
        if configured_state_schema != STATE_SCHEMA:
            raise ValueError(
                f"unsupported training state schema: {configured_state_schema}"
            )
        if not isinstance(max_servers, int) or isinstance(max_servers, bool) or not 1 <= max_servers <= 4:
            raise ValueError("max_servers must be between 1 and 4")
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_seconds must be a positive finite number")
        self.demo_root = Path(demo_root).resolve(strict=False)
        self.data_root = Path(data_root).resolve(strict=False)
        self.server_root = Path(server_root).resolve(strict=False)
        self.max_servers = max_servers
        self.purpose = ensure_purpose(purpose)
        self.run_id = str(run_id)
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        self.duration_seconds = duration
        self.calibration_quick_seconds = self._positive_config_float(
            "calibration_quick_seconds",
            CALIBRATION_QUICK_SECONDS,
        )
        self.calibration_soak_seconds = self._positive_config_float(
            "calibration_soak_seconds",
            CALIBRATION_SOAK_SECONDS,
        )
        self.calibration_p99_limit_ms = self._positive_config_float(
            "calibration_p99_limit_ms",
            CALIBRATION_P99_LIMIT_MS,
        )
        self.calibration_min_memory_gib = self._positive_config_float(
            "calibration_min_memory_gib",
            CALIBRATION_MEMORY_LIMIT_GIB,
        )
        self.calibration_min_tickrate_hz = self._positive_config_float(
            "calibration_min_tickrate_hz",
            CALIBRATION_MIN_TICKRATE_HZ,
        )
        self.calibration_max_tickrate_hz = self._positive_config_float(
            "calibration_max_tickrate_hz",
            CALIBRATION_MAX_TICKRATE_HZ,
        )
        if self.calibration_min_tickrate_hz > self.calibration_max_tickrate_hz:
            raise ValueError("calibration tickrate bounds are invalid")
        self.all_servers_unavailable_seconds = self._positive_config_float(
            "all_servers_unavailable_seconds",
            ALL_SERVERS_UNAVAILABLE_SECONDS,
        )
        self._clock = time.perf_counter if clock is None else clock
        self._wall_clock = time.time if wall_clock is None else wall_clock
        self._sleep = time.sleep if sleep is None else sleep
        self.calibration_probe = calibration_probe or self._missing_calibration_probe
        self.bootstrap_runner = bootstrap_runner
        self.wave_collector = wave_collector
        self.wave_trainer = wave_trainer
        self.generation_publisher = generation_publisher
        self.match_loader = match_loader
        self.stop_sampling_callback = stop_sampling
        self.truncation_flusher = truncation_flusher
        self.instance_restarter = instance_restarter
        self.worker_stop = worker_stop
        self.runtime_stop = runtime_stop
        self.server_stop = server_stop
        self.single_wave_collector = single_wave_collector
        self.single_wave_trainer = single_wave_trainer
        self.candidate_exporter = candidate_exporter
        self.farm = farm
        self.parent_manifests = tuple(parent_manifests)
        if any(not isinstance(parent, DatasetManifestV1) for parent in self.parent_manifests):
            raise TypeError("parent_manifests must contain DatasetManifestV1 values")
        repository_root = Path(__file__).resolve().parents[2]
        self.get5_template_path = (
            repository_root / "config" / "get5" / "selfplay_mr12.template.json"
            if get5_template_path is None
            else Path(get5_template_path).resolve(strict=False)
        )
        self.state_path = self.data_root / "state" / "run-state.json"
        self._config_sha256 = _config_sha256(self.config)
        self._demo_corpus_sha256: str | None = None
        self._get5_template_sha256: str | None = None
        self._resume_requested = bool(resume)
        self._resume_loaded = False
        self._bootstrap_manifest: DatasetManifestV1 | None = None
        self._deadline: float | None = None
        self._started_at: float | None = None
        self._run_duration: float | None = None
        self._sample_accepting = False
        self._sampling_stopped = False
        self._truncations_flushed = False
        self._timed_complete = False
        self._stop_requested = False
        self._incomplete = False
        self._incomplete_reason = ""
        self._unavailable_since: dict[str, float] = {}
        self._pending_wave_by_instance: dict[str, Any] = {}
        self._pending_completed_segments_by_instance: dict[str, tuple[Any, ...]] = {}
        self._pending_segments_by_match: dict[tuple[str, str], list[Any]] = {}
        self._instance_ids: tuple[str, ...] = ()
        self._single_wave_executed = False
        self._single_wave_result: CompleteMatchWaveV1 | None = None
        self._finalized = False
        self._state: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "purpose": self.purpose.value,
            "phase": "created",
            "deadline": None,
            "deadline_epoch_seconds": None,
            "duration_seconds": self.duration_seconds,
            "selected_concurrency": None,
            "generation": 0,
            "last_generation": 0,
            "consumed_shard_sha256": [],
            "config_sha256": self._config_sha256,
            "demo_corpus_sha256": None,
            "get5_template_sha256": None,
            "calibration_attempts": [],
            "bootstrap_manifest": None,
            "sample_accepting": False,
            "effective_duration_seconds": 0.0,
            "incomplete": False,
            "incomplete_reason": "",
            "unavailable_instances": {},
            "single_wave_updates": 0,
            "candidate_exports": 0,
        }

    def _positive_config_float(self, name: str, default: float) -> float:
        value = self.config.get(name, default)
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a positive finite number") from error
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
        return normalized

    @staticmethod
    def _load_config(config: str | Path) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required for training orchestration") from error
        with Path(config).open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, Mapping):
            raise ValueError("training configuration must be a mapping")
        return dict(_canonical_value(value))

    @property
    def generation(self) -> int:
        return int(self._state["generation"])

    @property
    def selected_concurrency(self) -> int | None:
        value = self._state.get("selected_concurrency")
        return None if value is None else int(value)

    @property
    def bootstrap_ready(self) -> bool:
        return self._bootstrap_manifest is not None

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(json.dumps(_canonical_value(self._state)))

    def _missing_calibration_probe(self, count: int, stage: str, duration_s: float) -> Mapping[str, Any]:
        raise TrainingOrchestratorError(
            "calibration_probe is required to provide real server telemetry"
        )

    def _compute_input_hashes(self) -> None:
        if not self.demo_root.exists():
            raise FileNotFoundError(self.demo_root)
        if not self.get5_template_path.is_file():
            raise FileNotFoundError(self.get5_template_path)
        self._demo_corpus_sha256 = _sha256_tree(self.demo_root)
        self._get5_template_sha256 = _sha256_file(self.get5_template_path)
        self._state["demo_corpus_sha256"] = self._demo_corpus_sha256
        self._state["get5_template_sha256"] = self._get5_template_sha256

    def _load_and_validate_resume_state(self) -> None:
        if self._resume_loaded:
            return
        if not self.state_path.is_file():
            raise FileNotFoundError(f"resume state is missing: {self.state_path}")
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("resume state is not valid JSON") from error
        if not isinstance(payload, Mapping) or payload.get("schema") != STATE_SCHEMA:
            raise ValueError("resume state schema is invalid")
        if str(payload.get("config_sha256")) != self._config_sha256:
            raise ValueError("resume config hash does not match")
        if str(payload.get("demo_corpus_sha256")) != self._demo_corpus_sha256:
            raise ValueError("resume demo corpus hash does not match")
        if str(payload.get("get5_template_sha256")) != self._get5_template_sha256:
            raise ValueError("resume Get5 template hash does not match")
        selected = payload.get("selected_concurrency")
        if selected is not None and (not isinstance(selected, int) or not 1 <= selected <= self.max_servers):
            raise ValueError("resume selected concurrency is invalid")
        generation = payload.get("generation", 0)
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("resume generation is invalid")
        last_generation = payload.get("last_generation", generation)
        if last_generation != generation:
            raise ValueError("resume last generation does not match generation")
        consumed = payload.get("consumed_shard_sha256", [])
        if not isinstance(consumed, list):
            raise ValueError("resume consumed shard hashes must be a list")
        normalized = [_valid_sha256(item) for item in consumed]
        if len(set(normalized)) != len(normalized):
            raise ValueError("resume state contains duplicate consumed shard hashes")
        saved_duration = payload.get("duration_seconds")
        try:
            saved_duration = float(saved_duration)
        except (TypeError, ValueError) as error:
            raise ValueError("resume duration_seconds is invalid") from error
        if not math.isfinite(saved_duration) or saved_duration <= 0.0:
            raise ValueError("resume duration_seconds is invalid")
        saved_deadline_epoch = payload.get("deadline_epoch_seconds")
        if saved_deadline_epoch is not None:
            try:
                saved_deadline_epoch = float(saved_deadline_epoch)
            except (TypeError, ValueError) as error:
                raise ValueError("resume deadline_epoch_seconds is invalid") from error
            if not math.isfinite(saved_deadline_epoch):
                raise ValueError("resume deadline_epoch_seconds is invalid")
        saved_effective_duration = payload.get("effective_duration_seconds", 0.0)
        try:
            saved_effective_duration = float(saved_effective_duration)
        except (TypeError, ValueError) as error:
            raise ValueError("resume effective_duration_seconds is invalid") from error
        if (
            not math.isfinite(saved_effective_duration)
            or saved_effective_duration < 0.0
            or saved_effective_duration > saved_duration
        ):
            raise ValueError("resume effective_duration_seconds is invalid")
        pointer = self.data_root / "models" / "current-generation.json"
        if pointer.is_file():
            from .export import load_policy_generation

            published = load_policy_generation(pointer)
            if published.number != generation:
                raise ValueError("resume policy pointer does not match last generation")
        self._state.update(dict(payload))
        self._state["consumed_shard_sha256"] = normalized
        self._state["generation"] = generation
        self._state["last_generation"] = generation
        self._state["selected_concurrency"] = selected
        self._state["incomplete"] = False
        self._state["incomplete_reason"] = ""
        bootstrap = self._state.get("bootstrap_manifest")
        if isinstance(bootstrap, Mapping):
            self._bootstrap_manifest = self._manifest_from_payload(bootstrap)
        self._resume_loaded = True

    @staticmethod
    def _manifest_from_payload(value: Mapping[str, Any]) -> DatasetManifestV1:
        parents = tuple(
            TrainingOrchestrator._manifest_from_payload(item)
            for item in value.get("parents", ())
            if isinstance(item, Mapping)
        )
        return DatasetManifestV1(
            name=str(value["name"]),
            purpose=ensure_purpose(value["purpose"]),
            parents=parents,
            artifact_type=str(value.get("artifact_type", "dataset")),
            source_sha256=value.get("source_sha256"),
            parser_version=value.get("parser_version"),
            projection_version=value.get("projection_version"),
            metadata={str(key): str(item) for key, item in value.get("metadata", {}).items()},
        )

    def persist_state(self, deadline: float | None = None) -> Path:
        if deadline is not None:
            self._state["deadline"] = float(deadline)
        self._state["selected_concurrency"] = self.selected_concurrency
        self._state["generation"] = self.generation
        self._state["last_generation"] = self.generation
        self._state["consumed_shard_sha256"] = list(self._state.get("consumed_shard_sha256", []))
        _atomic_write_json(self.state_path, self._state)
        return self.state_path

    def _mark_incomplete(self, reason: str) -> None:
        self._incomplete = True
        self._incomplete_reason = str(reason)
        self._state["incomplete"] = True
        self._state["incomplete_reason"] = self._incomplete_reason
        self._state["phase"] = "incomplete"

    def _calibrate(self) -> int:
        attempts: list[dict[str, Any]] = list(self._state.get("calibration_attempts", []))
        highest_quick_pass = 0
        for count in range(1, self.max_servers + 1):
            quick_raw = _invoke(
                self.calibration_probe,
                count,
                "quick",
                self.calibration_quick_seconds,
            )
            quick = CalibrationMetricsV1.from_mapping(quick_raw)
            quick_attempt = CalibrationAttemptV1(count, "quick", self.calibration_quick_seconds, quick)
            attempts.append(quick_attempt.to_dict())
            self._state["calibration_attempts"] = attempts
            self.persist_state()
            if not quick.passes(
                p99_limit_ms=self.calibration_p99_limit_ms,
                min_memory_gib=self.calibration_min_memory_gib,
                min_tickrate_hz=self.calibration_min_tickrate_hz,
                max_tickrate_hz=self.calibration_max_tickrate_hz,
            ):
                break
            highest_quick_pass = count
        selected = 0
        for count in range(highest_quick_pass, 0, -1):
            soak_raw = _invoke(
                self.calibration_probe,
                count,
                "soak",
                self.calibration_soak_seconds,
            )
            soak = CalibrationMetricsV1.from_mapping(soak_raw)
            soak_attempt = CalibrationAttemptV1(count, "soak", self.calibration_soak_seconds, soak)
            attempts.append(soak_attempt.to_dict())
            self._state["calibration_attempts"] = attempts
            self.persist_state()
            if soak.passes(
                p99_limit_ms=self.calibration_p99_limit_ms,
                min_memory_gib=self.calibration_min_memory_gib,
                min_tickrate_hz=self.calibration_min_tickrate_hz,
                max_tickrate_hz=self.calibration_max_tickrate_hz,
            ):
                selected = count
                break
        if selected < 1:
            self._mark_incomplete("no server concurrency candidate passed calibration")
            self.persist_state()
            raise TrainingOrchestratorError("calibration rejected every server concurrency")
        self._state["selected_concurrency"] = selected
        self._instance_ids = tuple(f"{index:02d}" for index in range(1, selected + 1))
        return selected

    def run_preflight(self) -> dict[str, Any]:
        self._compute_input_hashes()
        if self._resume_requested:
            self._load_and_validate_resume_state()
        if self.purpose is DataPurpose.PRODUCTION:
            from .training_device import resolve_training_device

            requested_device = str(self.config.get("device", "cuda"))
            device = resolve_training_device(requested_device)
            self._state["training_device"] = {
                "type": device.type,
                "index": device.index,
            }
        build_server_specs(
            self.max_servers,
            self.run_id,
            self.server_root,
            self.data_root,
        )
        self._state["phase"] = "preflight"
        self._state["purpose"] = self.purpose.value
        self._state["config_sha256"] = self._config_sha256
        self.persist_state()
        if self.selected_concurrency is None:
            self._calibrate()
        else:
            self._instance_ids = tuple(
                f"{index:02d}" for index in range(1, self.selected_concurrency + 1)
            )
        self._state["phase"] = "preflight_complete"
        self.persist_state()
        return self.state

    def run_bootstrap(self) -> dict[str, Any]:
        if self.selected_concurrency is None:
            self.run_preflight()
        if self.selected_concurrency is None:
            raise TrainingOrchestratorError("preflight did not select server concurrency")
        if self._bootstrap_manifest is not None:
            return self.state
        if self.bootstrap_runner is None:
            raise TrainingOrchestratorError("bootstrap_runner is required to validate MR12 servers")
        self._state["phase"] = "bootstrap"
        self.persist_state()
        raw_matches = _invoke(self.bootstrap_runner, self.selected_concurrency)
        if isinstance(raw_matches, Mapping):
            raw_matches = raw_matches.get("matches", ())
        matches = tuple(raw_matches or ())
        if any(not isinstance(match, BootstrapMatchV1) for match in matches):
            raise TypeError("bootstrap_runner must return BootstrapMatchV1 values")
        self._bootstrap_manifest = build_bootstrap_manifest(
            matches,
            self.selected_concurrency,
            purpose=self.purpose,
            parent_manifests=self.parent_manifests,
        )
        self._state["bootstrap_manifest"] = _manifest_payload(self._bootstrap_manifest)
        self._state["phase"] = "bootstrap_complete"
        self.persist_state()
        return self.state

    def _normalize_sample(self, sample: Any) -> Any:
        try:
            from .selfplay import RolloutEnvelopeV1
            from .training_mappo import load_rollout_envelope
        except ImportError:
            return sample
        if isinstance(sample, RolloutEnvelopeV1):
            return load_rollout_envelope(sample)
        return sample

    def _sample_hash(self, sample: Any) -> str:
        declared = _field(sample, "shard_sha256")
        if declared:
            return _valid_sha256(declared)
        path = _field(sample, "shard_path")
        if path is not None and Path(path).is_file():
            return _sha256_file(Path(path))
        raise ValueError("completed rollout must declare shard_sha256 or shard_path")

    def _sample_identity(self, sample: Any) -> tuple[str, str]:
        instance_id = str(_field(sample, "instance_id", ""))
        match_id = str(_field(sample, "match_id", ""))
        if not instance_id or not match_id:
            raise ValueError("completed rollout must declare instance_id and match_id")
        return instance_id, match_id

    def _register_unavailable(self, instance_id: str, reason: str = "") -> None:
        instance_id = str(instance_id)
        if not instance_id:
            instance_id = "unknown"
        self._pending_wave_by_instance.pop(instance_id, None)
        self._pending_completed_segments_by_instance.pop(instance_id, None)
        for identity in tuple(self._pending_segments_by_match):
            if identity[0] == instance_id:
                self._pending_segments_by_match.pop(identity, None)
        now = float(self._clock())
        self._unavailable_since.setdefault(instance_id, now)
        unavailable = dict(self._state.get("unavailable_instances", {}))
        unavailable[instance_id] = {
            "since": self._unavailable_since[instance_id],
            "reason": str(reason),
        }
        self._state["unavailable_instances"] = unavailable
        if self.instance_restarter is not None:
            _invoke(self.instance_restarter, instance_id)
        expected = set(self._instance_ids)
        if expected and expected.issubset(self._unavailable_since):
            elapsed = now - max(self._unavailable_since[item] for item in expected)
            if elapsed >= self.all_servers_unavailable_seconds:
                self._stop_requested = True
                self._mark_incomplete("all selected server instances unavailable for more than 30 seconds")

    def mark_instance_unavailable(self, instance_id: str, reason: str = "") -> dict[str, Any]:
        self._register_unavailable(instance_id, reason)
        self.persist_state(self._deadline)
        return self.state

    def mark_instance_available(self, instance_id: str) -> dict[str, Any]:
        instance_id = str(instance_id)
        self._unavailable_since.pop(instance_id, None)
        unavailable = dict(self._state.get("unavailable_instances", {}))
        unavailable.pop(instance_id, None)
        self._state["unavailable_instances"] = unavailable
        self.persist_state(self._deadline)
        return self.state

    def collect_generation_wave(self, deadline: float) -> tuple[Any, ...]:
        if self.wave_collector is None:
            raise TrainingOrchestratorError("wave_collector is required for timed self-play")
        try:
            raw_wave = _invoke(self.wave_collector, deadline)
        except InstanceUnavailableError as error:
            self._register_unavailable(error.instance_id, error.reason)
            return ()
        unavailable: Sequence[Any] = ()
        if isinstance(raw_wave, Mapping):
            unavailable = raw_wave.get("unavailable_instances", ())
            raw_wave = raw_wave.get("rollouts", raw_wave.get("completed", ()))
        for item in unavailable or ():
            if isinstance(item, Mapping):
                self._register_unavailable(str(item.get("instance_id", "")), str(item.get("reason", "")))
            else:
                self._register_unavailable(str(item))
        incoming: list[tuple[Any, tuple[Any, ...]]] = []
        identities: set[tuple[str, str]] = set()
        for item in tuple(raw_wave or ()):
            sample = self._normalize_sample(item)
            shard_hash = self._sample_hash(sample)
            if shard_hash in set(self._state.get("consumed_shard_sha256", [])):
                continue
            terminal_field = _field(sample, "terminal")
            if terminal_field is None:
                if self.purpose is DataPurpose.PRODUCTION:
                    raise ValueError("production rollout must declare terminal state")
                terminal = True
            else:
                terminal = bool(terminal_field)
            truncated = bool(_field(sample, "truncated", False))
            instance_id, match_id = self._sample_identity(sample)
            if self._instance_ids and instance_id not in self._instance_ids:
                raise ValueError("completed rollout belongs to an unselected server instance")
            generation = _field(sample, "policy_generation")
            if generation is not None and int(generation) != self.generation:
                self._mark_incomplete("rollout policy generation does not match active generation")
                self.persist_state(deadline)
                raise ValueError("rollout policy generation does not match active generation")
            identity = (instance_id, match_id)
            if identity in identities:
                raise ValueError("a generation wave contains duplicate instance and match")
            if not terminal:
                if truncated:
                    self._state.setdefault("incomplete_shards", []).append(
                        shard_hash
                    )
                    self._pending_segments_by_match.pop(identity, None)
                else:
                    pending = self._pending_segments_by_match.setdefault(identity, [])
                    if all(self._sample_hash(existing) != shard_hash for existing in pending):
                        pending.append(sample)
                continue
            identities.add(identity)
            existing = self._pending_wave_by_instance.get(instance_id)
            if existing is not None:
                existing_identity = self._sample_identity(existing)
                if existing_identity == identity and self._sample_hash(existing) == shard_hash:
                    continue
                raise ValueError("a generation wave contains multiple matches for one instance")
            segments = tuple([*self._pending_segments_by_match.pop(identity, []), sample])
            incoming.append((sample, segments))
            self.mark_instance_available(instance_id)
        for sample, segments in incoming:
            instance_id, _ = self._sample_identity(sample)
            self._pending_wave_by_instance[instance_id] = sample
            self._pending_completed_segments_by_instance[instance_id] = segments
        self._check_unavailable_timeout()
        self.persist_state(deadline)
        expected = set(self._instance_ids) - set(self._unavailable_since)
        if not expected or not expected.issubset(self._pending_wave_by_instance):
            return ()
        ordered = tuple(
            segment
            for instance_id in self._instance_ids
            if instance_id in expected
            for segment in self._pending_completed_segments_by_instance[instance_id]
        )
        for instance_id in expected:
            self._pending_wave_by_instance.pop(instance_id, None)
            self._pending_completed_segments_by_instance.pop(instance_id, None)
        return ordered

    def _check_unavailable_timeout(self) -> None:
        expected = set(self._instance_ids)
        if not expected or not expected.issubset(self._unavailable_since):
            return
        now = float(self._clock())
        elapsed = now - max(self._unavailable_since[item] for item in expected)
        if elapsed >= self.all_servers_unavailable_seconds:
            self._stop_requested = True
            self._mark_incomplete("all selected server instances unavailable for more than 30 seconds")

    def _publish_generation(self, result: Any, next_generation: int) -> None:
        if result is None:
            if self.purpose is DataPurpose.PRODUCTION and self.generation_publisher is None:
                raise TrainingOrchestratorError(
                    "production wave training must return or publish a policy generation"
                )
            if self.generation_publisher is not None:
                _invoke(self.generation_publisher, result, next_generation)
            return
        try:
            from .export import PolicyGenerationV1, publish_policy_generation
        except ImportError:
            PolicyGenerationV1 = ()
            publish_policy_generation = None
        if PolicyGenerationV1 and isinstance(result, PolicyGenerationV1):
            if result.number != next_generation:
                raise ValueError("published policy generation is not the next generation")
            if self.generation_publisher is None:
                publish_policy_generation(
                    result,
                    self.data_root / "models" / "current-generation.json",
                )
            else:
                _invoke(self.generation_publisher, result, next_generation)
            return
        declared = _field(result, "policy_generation")
        if declared is None and isinstance(result, Mapping):
            declared = result.get("number")
        if declared is not None and int(declared) != next_generation:
            raise ValueError("training result policy generation is not the next generation")
        if self.generation_publisher is None:
            if self.purpose is DataPurpose.PRODUCTION:
                raise TrainingOrchestratorError(
                    "production wave training needs a generation publisher"
                )
            return
        _invoke(self.generation_publisher, result, next_generation)

    def train_and_publish_generation(self, wave: Sequence[Any]) -> Any:
        consumed = set(self._state.get("consumed_shard_sha256", []))
        unseen: list[Any] = []
        wave_hashes: list[str] = []
        for item in wave:
            sample = self._normalize_sample(item)
            shard_hash = self._sample_hash(sample)
            if shard_hash in consumed or shard_hash in wave_hashes:
                continue
            generation = _field(sample, "policy_generation")
            if generation is not None and int(generation) != self.generation:
                raise ValueError("training wave contains an invalid policy generation")
            self._sample_identity(sample)
            wave_hashes.append(shard_hash)
            unseen.append(sample)
        if not unseen:
            self.persist_state(self._deadline)
            return None
        if self.wave_trainer is None:
            raise TrainingOrchestratorError("wave_trainer is required for MAPPO training")
        active_generation = self.generation
        next_generation = active_generation + 1
        result = _invoke(self.wave_trainer, tuple(unseen), active_generation)
        self._publish_generation(result, next_generation)
        self._state["consumed_shard_sha256"] = [*consumed, *wave_hashes]
        self._state["last_wave"] = {
            "policy_generation": active_generation,
            "next_policy_generation": next_generation,
            "shard_sha256": wave_hashes,
        }
        self._state["generation"] = next_generation
        self._state["last_generation"] = next_generation
        self.persist_state(self._deadline)
        if self.match_loader is not None:
            try:
                _invoke(self.match_loader, next_generation)
            except Exception as error:
                self._mark_incomplete(f"next generation match loading failed: {error}")
                self.persist_state(self._deadline)
                raise
        return result

    @property
    def single_wave_result(self) -> CompleteMatchWaveV1 | None:
        return self._single_wave_result

    def run_single_match_wave(
        self,
        specs: Sequence[Any],
        package: Any,
        *,
        host_timescale: int,
        ruleset: str = "mr12",
        match_runner: Callable[..., Any] | None = None,
        restart_instance: Callable[..., Any] | None = None,
        watchdog_seconds: float = 120.0,
        wall_clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        wave_collector: Callable[..., Any] | None = None,
        mappo_update: Callable[..., Any] | None = None,
        candidate_exporter: Callable[..., Any] | None = None,
    ) -> Any:
        """Run the independent test-only primitive: one wave, one update, one export.

        This method is intentionally separate from ``run_timed_selfplay``.  It has no
        wall-clock training deadline and never publishes a next generation.  A failed
        collection returns an aborted wave without exposing any other server's shard.
        """

        if self.purpose is not DataPurpose.TEST_ONLY:
            raise ValueError("single match wave is restricted to test_only")
        if self._single_wave_executed:
            raise RuntimeError("single match wave already executed")
        collector = wave_collector or self.single_wave_collector or collect_single_match_wave
        trainer = mappo_update or self.single_wave_trainer
        exporter = candidate_exporter or self.candidate_exporter
        if trainer is None or exporter is None:
            raise TrainingOrchestratorError(
                "single match wave requires exactly one MAPPO updater and candidate exporter"
            )
        self._single_wave_executed = True
        if collector is collect_single_match_wave:
            result = collector(
                specs,
                package,
                host_timescale=host_timescale,
                ruleset=ruleset,
                match_runner=match_runner,
                restart_instance=restart_instance,
                watchdog_seconds=watchdog_seconds,
                wall_clock=wall_clock,
                sleep=sleep,
            )
        else:
            result = _invoke(collector, specs, package, host_timescale)
        if not isinstance(result, CompleteMatchWaveV1):
            raise TypeError("single wave collector must return CompleteMatchWaveV1")
        self._single_wave_result = result
        self._state["single_wave_status"] = result.status
        self._state["single_wave_attempts"] = dict(result.attempts)
        self._state["single_wave_manifests"] = [manifest.to_dict() for manifest in result.manifests]
        if not result.complete:
            self._state["phase"] = "single_wave_aborted"
            self._state["single_wave_abort_reason"] = result.abort_reason
            self.persist_state(self._deadline)
            return None
        update_result = _invoke(trainer, result.shards)
        self._state["single_wave_updates"] = 1
        self._state["single_wave_update_payload_count"] = len(result.shards)
        exported = _invoke(exporter, update_result, result)
        self._state["candidate_exports"] = 1
        self._state["phase"] = "single_wave_candidate_exported"
        self.persist_state(self._deadline)
        return exported

    def stop_accepting_samples(self, cutoff: float | None = None) -> None:
        if self._sampling_stopped:
            return
        cutoff_value = float(self._clock() if cutoff is None else cutoff)
        if self.stop_sampling_callback is not None:
            _invoke(self.stop_sampling_callback, cutoff_value)
        self._sampling_stopped = True
        self._sample_accepting = False
        self._state["sample_accepting"] = False
        self._state["sampling_cutoff"] = cutoff_value

    def flush_deadline_truncations(self, deadline: float | None = None) -> int:
        if self._truncations_flushed:
            return int(self._state.get("flushed_truncated_trajectories", 0))
        cutoff = float(self._clock() if deadline is None else deadline)
        result: Any = 0
        if self.truncation_flusher is not None:
            result = _invoke(self.truncation_flusher, cutoff)
        if isinstance(result, Mapping):
            result = result.get("count", result.get("truncated_trajectories", 0))
        try:
            count = max(0, int(result or 0))
        except (TypeError, ValueError) as error:
            raise ValueError("truncation flusher must return an integer count") from error
        self._state["flushed_truncated_trajectories"] = count
        self._state["truncation_cutoff"] = cutoff
        self._truncations_flushed = True
        self.persist_state(cutoff)
        return count

    def run_timed_selfplay(self, duration_s: float | None = None) -> dict[str, Any]:
        if self.selected_concurrency is None:
            self.run_preflight()
        if self._bootstrap_manifest is None:
            self.run_bootstrap()
        if self._resume_loaded:
            saved_duration = float(self._state["duration_seconds"])
            if duration_s is not None and float(duration_s) != saved_duration:
                raise ValueError("resume duration_s must match the saved run duration")
            run_duration = saved_duration
        else:
            run_duration = self.duration_seconds if duration_s is None else float(duration_s)
        if not math.isfinite(run_duration) or run_duration <= 0.0:
            raise ValueError("duration_s must be a positive finite number")
        if self.purpose is DataPurpose.PRODUCTION and run_duration != DEFAULT_DURATION_SECONDS:
            raise ValueError("production timed self-play duration must be exactly 7200 seconds")
        start = float(self._clock())
        elapsed_before_resume = 0.0
        if self._resume_loaded:
            saved_deadline_epoch = self._state.get("deadline_epoch_seconds")
            if saved_deadline_epoch is None:
                if float(self._state.get("effective_duration_seconds", 0.0)) != 0.0:
                    raise ValueError(
                        "resume state without a timed deadline cannot have elapsed duration"
                    )
                remaining = run_duration
                self._state["deadline_epoch_seconds"] = float(self._wall_clock()) + run_duration
            else:
                remaining = max(0.0, float(saved_deadline_epoch) - float(self._wall_clock()))
                elapsed_before_resume = min(
                    run_duration,
                    max(
                        float(self._state.get("effective_duration_seconds", 0.0)),
                        run_duration - remaining,
                    ),
                )
            deadline = start + remaining
            self._started_at = start - elapsed_before_resume
        else:
            deadline = start + run_duration
            self._started_at = start
        self._deadline = deadline
        self._run_duration = run_duration
        self._sample_accepting = True
        self._sampling_stopped = False
        self._truncations_flushed = False
        self._timed_complete = False
        self._stop_requested = False
        self._state["phase"] = "timed_selfplay"
        self._state["deadline"] = deadline
        if not self._resume_loaded:
            self._state["deadline_epoch_seconds"] = float(self._wall_clock()) + run_duration
        self._state["duration_seconds"] = run_duration
        self._state["sample_accepting"] = True
        self._state["effective_duration_seconds"] = elapsed_before_resume
        self.persist_state(deadline)
        try:
            if self.match_loader is not None:
                _invoke(self.match_loader, self.generation)
            while not self._stop_requested and float(self._clock()) < deadline:
                wave = self.collect_generation_wave(deadline)
                if wave:
                    self.train_and_publish_generation(wave)
                elapsed = min(
                    run_duration,
                    elapsed_before_resume + max(0.0, float(self._clock()) - start),
                )
                self._state["effective_duration_seconds"] = elapsed
                self.persist_state(deadline)
                if not wave and not self._stop_requested:
                    remaining = deadline - float(self._clock())
                    if remaining > 0.0:
                        self._sleep(min(0.25, remaining))
            cutoff = min(deadline, float(self._clock()))
            self.stop_accepting_samples(cutoff)
            self.flush_deadline_truncations(cutoff)
            self._state["effective_duration_seconds"] = min(
                run_duration,
                elapsed_before_resume + max(0.0, cutoff - start),
            )
            if self._stop_requested:
                self._mark_incomplete("timed self-play stopped before the deadline")
            else:
                self._timed_complete = cutoff >= deadline
                self._state["phase"] = "deadline_reached" if self._timed_complete else "incomplete"
                if not self._timed_complete:
                    self._mark_incomplete("timed self-play stopped before the deadline")
            self.persist_state(deadline)
        except KeyboardInterrupt:
            cutoff = min(deadline, float(self._clock()))
            self.stop_accepting_samples(cutoff)
            self.flush_deadline_truncations(cutoff)
            self._mark_incomplete("interrupted by user")
            self.persist_state(deadline)
            raise
        except Exception as error:
            cutoff = min(deadline, float(self._clock()))
            self.stop_accepting_samples(cutoff)
            self.flush_deadline_truncations(cutoff)
            self._mark_incomplete(str(error))
            self.persist_state(deadline)
            raise
        return self.state

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            return self.state
        if self._sample_accepting:
            cutoff = float(self._clock())
            self.stop_accepting_samples(cutoff)
            self.flush_deadline_truncations(cutoff)
            self._mark_incomplete("finalized before timed self-play reached its deadline")
        try:
            if self.worker_stop is not None:
                _invoke(self.worker_stop)
            if self.runtime_stop is not None:
                _invoke(self.runtime_stop)
            if self.server_stop is not None:
                _invoke(self.server_stop)
            elif self.farm is not None:
                self.farm.stop()
        except Exception as error:
            self._mark_incomplete(f"orderly shutdown failed: {error}")
        if not self._incomplete:
            if not self._timed_complete:
                self._mark_incomplete("timed self-play did not reach its deadline")
            else:
                self._state["phase"] = "completed"
        if self._incomplete:
            self._state["phase"] = "incomplete"
        self._state["sample_accepting"] = False
        self._finalized = True
        self.persist_state(self._deadline)
        return self.state


__all__ = [
    "ALL_SERVERS_UNAVAILABLE_SECONDS",
    "CALIBRATION_MEMORY_LIMIT_GIB",
    "CALIBRATION_P99_LIMIT_MS",
    "CALIBRATION_QUICK_SECONDS",
    "CALIBRATION_SOAK_SECONDS",
    "CalibrationAttemptV1",
    "CalibrationMetricsV1",
    "CompleteMatchWaveV1",
    "InstanceUnavailableError",
    "STATE_SCHEMA",
    "TrainingOrchestrator",
    "TrainingOrchestratorError",
    "collect_single_match_wave",
]
