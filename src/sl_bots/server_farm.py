from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .rcon import (
    RconClientV1,
    RconCredentialsV1,
    RconReadinessV1,
    wait_for_rcon_readiness,
    write_rcon_run_config,
)
from .server_launcher import build_srcds_command, launch_srcds, resolve_srcds_executable


_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_INSTANCE_PATTERN = re.compile(r"[0-9]{2}\Z")
_STATUS_BOT_COUNT_PATTERN = re.compile(r"players\s*:\s*\d+\s+humans,\s*(\d+)\s+bots", re.IGNORECASE)
_GET5_WARMUP_ADVANCE_TIMEOUT_S = 20.0
_GET5_WARMUP_POLL_INTERVAL_S = 0.1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GET5_TEMPLATE = _REPOSITORY_ROOT / "config" / "get5" / "selfplay_mr12.template.json"


def _probe_resource_evidence_is_valid(
    resource_evidence: bool,
    resource_error: bool,
    cpu_percent: float | None,
    ram_mb: float | None,
    gpu_percent: float | None,
) -> bool:
    """Require measured, finite host-resource values before a probe is stable."""

    if not resource_evidence or resource_error:
        return False
    for value in (cpu_percent, ram_mb, gpu_percent):
        try:
            if value is None or not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _complete_match_shard_items(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Path, Mapping)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class ServerInstanceSpecV1:
    instance_id: str
    game_port: int
    client_port: int
    tv_port: int
    steam_port: int
    rcon_port: int
    ipc_name: str
    get5_config_path: Path
    log_dir: Path

    @property
    def ports(self) -> tuple[int, int, int, int]:
        return self.game_port, self.client_port, self.tv_port, self.steam_port

    @property
    def rcon_endpoint(self) -> tuple[str, int]:
        return "127.0.0.1", self.rcon_port


@dataclass(frozen=True)
class CompleteMatchManifestV1:
    """The only match artifact that a single self-play wave may accept."""

    instance_id: str
    match_id: str
    package_sha256: str
    policy_generation: int
    host_timescale: int
    ruleset: str
    shard: Any = None
    terminal: bool = True
    attempt: int = 1

    def __post_init__(self) -> None:
        if not str(self.instance_id) or not str(self.match_id):
            raise ValueError("complete match manifest requires instance_id and match_id")
        digest = str(self.package_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("complete match package_sha256 must be a 64-character digest")
        if not isinstance(self.policy_generation, int) or isinstance(self.policy_generation, bool):
            raise TypeError("complete match policy_generation must be an integer")
        if self.policy_generation < 0:
            raise ValueError("complete match policy_generation must be non-negative")
        if not isinstance(self.host_timescale, int) or isinstance(self.host_timescale, bool):
            raise TypeError("complete match host_timescale must be an integer")
        if self.host_timescale < 1:
            raise ValueError("complete match host_timescale must be positive")
        if not str(self.ruleset):
            raise ValueError("complete match ruleset cannot be empty")
        if self.shard is None:
            raise ValueError("complete match manifest requires a rollout shard")
        if isinstance(self.shard, (str, Path)) and not str(self.shard).strip():
            raise ValueError("complete match rollout shard cannot be empty")
        shard_items = _complete_match_shard_items(self.shard)
        if not shard_items:
            raise ValueError("complete match manifest rollout shard sequence cannot be empty")
        marker_envelopes = getattr(self.shard, "envelopes", None)
        if marker_envelopes is None and isinstance(self.shard, Mapping):
            marker_envelopes = self.shard.get("envelopes")
        if marker_envelopes is not None:
            if not isinstance(marker_envelopes, Sequence) or not marker_envelopes:
                raise ValueError("complete match terminal marker must contain rollout envelopes")
            marker_terminal = getattr(self.shard, "terminal", None)
            if marker_terminal is None and isinstance(self.shard, Mapping):
                marker_terminal = self.shard.get("terminal", False)
            if not bool(marker_terminal):
                raise ValueError("complete match terminal marker must be terminal")
        for item in shard_items:
            if isinstance(item, Mapping):
                if "envelopes" in item:
                    continue
                shard_path = item.get("shard_path", item.get("path", item.get("shard")))
                if shard_path is None or not str(shard_path).strip():
                    raise ValueError("complete match manifest rollout shard mapping is invalid")
            if hasattr(item, "truncated") and bool(getattr(item, "truncated")):
                raise ValueError("complete match rollout shard cannot be truncated")
        if len(shard_items) == 1 and hasattr(shard_items[0], "terminal") and not bool(
            getattr(shard_items[0], "terminal")
        ):
            raise ValueError("complete match rollout shard must be terminal")
        if not bool(self.terminal):
            raise ValueError("complete match manifest must be terminal")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("complete match attempt must be a positive integer")
        object.__setattr__(self, "instance_id", str(self.instance_id))
        object.__setattr__(self, "match_id", str(self.match_id))
        object.__setattr__(self, "package_sha256", digest)
        object.__setattr__(self, "ruleset", str(self.ruleset))
        object.__setattr__(self, "terminal", True)

    def to_dict(self) -> dict[str, Any]:
        shard = self.shard
        if isinstance(shard, Sequence) and not isinstance(shard, (str, bytes, bytearray, Path, Mapping)):
            shard = [item.to_dict() if hasattr(item, "to_dict") else item for item in shard]
        elif hasattr(shard, "to_dict"):
            shard = shard.to_dict()
        return {
            "instance_id": self.instance_id,
            "match_id": self.match_id,
            "package_sha256": self.package_sha256,
            "policy_generation": self.policy_generation,
            "host_timescale": self.host_timescale,
            "ruleset": self.ruleset,
            "shard": shard,
            "terminal": self.terminal,
            "attempt": self.attempt,
        }


@dataclass(frozen=True)
class CompleteMatchWaveV1:
    """Result of one server group playing exactly one match each."""

    status: str
    package_sha256: str
    policy_generation: int
    host_timescale: int
    ruleset: str
    manifests: tuple[CompleteMatchManifestV1, ...]
    attempts: Mapping[str, int]
    discarded_instances: tuple[str, ...] = ()
    abort_reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"completed", "aborted"}:
            raise ValueError("single match wave status must be completed or aborted")
        digest = str(self.package_sha256).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("single match wave package_sha256 must be a 64-character digest")
        manifests = tuple(self.manifests)
        if any(not isinstance(manifest, CompleteMatchManifestV1) for manifest in manifests):
            raise TypeError("single match wave manifests must be CompleteMatchManifestV1 values")
        if len({manifest.instance_id for manifest in manifests}) != len(manifests):
            raise ValueError("single match wave cannot contain duplicate instances")
        if any(
            manifest.package_sha256 != digest
            or manifest.policy_generation != self.policy_generation
            or manifest.host_timescale != self.host_timescale
            or manifest.ruleset != self.ruleset
            for manifest in manifests
        ):
            raise ValueError("single match wave manifest context does not match the wave")
        if self.status == "aborted" and manifests:
            raise ValueError("aborted single match wave cannot expose completed manifests")
        if self.status == "completed" and not manifests:
            raise ValueError("completed single match wave must expose rollout manifests")
        normalized_attempts = {str(instance_id): int(attempt) for instance_id, attempt in self.attempts.items()}
        if any(attempt < 1 for attempt in normalized_attempts.values()):
            raise ValueError("single match wave attempts must be positive")
        object.__setattr__(self, "package_sha256", digest)
        object.__setattr__(self, "manifests", manifests)
        object.__setattr__(self, "attempts", normalized_attempts)
        object.__setattr__(self, "discarded_instances", tuple(str(item) for item in self.discarded_instances))

    @property
    def complete(self) -> bool:
        return self.status == "completed"

    @property
    def shards(self) -> tuple[Any, ...]:
        shards = tuple(manifest.shard for manifest in self.manifests)
        if not shards or any(shard is None for shard in shards):
            raise ValueError("completed single match wave must expose one rollout shard per instance")
        return shards

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "package_sha256": self.package_sha256,
            "policy_generation": self.policy_generation,
            "host_timescale": self.host_timescale,
            "ruleset": self.ruleset,
            "manifests": [manifest.to_dict() for manifest in self.manifests],
            "attempts": dict(self.attempts),
            "discarded_instances": list(self.discarded_instances),
            "abort_reason": self.abort_reason,
        }


@dataclass(frozen=True)
class CapacityProbeSampleV1:
    requested_instances: int
    host_timescale: int = 1
    stable_window_s: float = 0.0
    process_alive: bool = False
    rcon_ready: bool = False
    map_progress: bool = False
    tick_backlog: bool = False
    ipc_conflict: bool = False
    inference_timeout: bool = False
    action_loss: bool = False
    memory_error: bool = False
    gpu_error: bool = False
    cpu_percent: float | None = None
    ram_mb: float | None = None
    gpu_percent: float | None = None
    resource_evidence: bool = False
    resource_error: bool = False
    exit_code: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.requested_instances, int) or not 1 <= self.requested_instances <= 32:
            raise ValueError("capacity probe instance count must be between 1 and 32")
        if self.host_timescale != 1:
            raise ValueError("capacity probes must run at host_timescale 1")
        if self.stable_window_s < 0:
            raise ValueError("stable_window_s cannot be negative")

    @property
    def stable(self) -> bool:
        return (
            self.process_alive
            and self.rcon_ready
            and self.map_progress
            and not self.tick_backlog
            and not self.ipc_conflict
            and not self.inference_timeout
            and not self.action_loss
            and not self.memory_error
            and not self.gpu_error
            and _probe_resource_evidence_is_valid(
                self.resource_evidence,
                self.resource_error,
                self.cpu_percent,
                self.ram_mb,
                self.gpu_percent,
            )
            and self.exit_code in (None, 0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_instances": self.requested_instances,
            "host_timescale": self.host_timescale,
            "stable_window_s": self.stable_window_s,
            "process_alive": self.process_alive,
            "rcon_ready": self.rcon_ready,
            "map_progress": self.map_progress,
            "tick_backlog": self.tick_backlog,
            "ipc_conflict": self.ipc_conflict,
            "inference_timeout": self.inference_timeout,
            "action_loss": self.action_loss,
            "memory_error": self.memory_error,
            "gpu_error": self.gpu_error,
            "cpu_percent": self.cpu_percent,
            "ram_mb": self.ram_mb,
            "gpu_percent": self.gpu_percent,
            "resource_evidence": self.resource_evidence,
            "resource_error": self.resource_error,
            "exit_code": self.exit_code,
            "stable": self.stable,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CapacityProbeReportV1:
    selected_instances: int
    samples: tuple[CapacityProbeSampleV1, ...]
    selection_reason: str
    host_timescale: int = 1
    probe_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_identity", dict(self.probe_identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_instances": self.selected_instances,
            "host_timescale": self.host_timescale,
            "selection_reason": self.selection_reason,
            "probe_identity": dict(self.probe_identity),
            "probe_fingerprint": self.probe_identity.get("fingerprint", ""),
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class TimescaleProbeSampleV1:
    timescale: int
    supported: bool
    readback_ok: bool
    tick_speedup: float
    action_p99_ms: float
    deadline_misses: int
    process_alive: bool = True
    rcon_ready: bool = True
    map_progress: bool = True
    ipc_conflict: bool = False
    inference_timeout: bool = False
    action_loss: bool = False
    memory_error: bool = False
    gpu_error: bool = False
    cpu_percent: float | None = None
    ram_mb: float | None = None
    gpu_percent: float | None = None
    resource_evidence: bool = False
    resource_error: bool = False
    detail: str = ""
    stable_window_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.timescale, int) or self.timescale < 1:
            raise ValueError("timescale must be a positive integer")
        if self.action_p99_ms < 0 or self.tick_speedup < 0 or self.deadline_misses < 0:
            raise ValueError("timescale probe metrics cannot be negative")
        if self.stable_window_s < 0:
            raise ValueError("stable_window_s cannot be negative")

    @property
    def stable(self) -> bool:
        deadline_ms = 7.8125 / self.timescale
        return (
            self.supported
            and self.readback_ok
            and self.process_alive
            and self.rcon_ready
            and self.map_progress
            and self.tick_speedup >= float(self.timescale) * 0.9
            and self.action_p99_ms < deadline_ms
            and self.deadline_misses == 0
            and not self.ipc_conflict
            and not self.inference_timeout
            and not self.action_loss
            and not self.memory_error
            and not self.gpu_error
            and _probe_resource_evidence_is_valid(
                self.resource_evidence,
                self.resource_error,
                self.cpu_percent,
                self.ram_mb,
                self.gpu_percent,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timescale": self.timescale,
            "supported": self.supported,
            "readback_ok": self.readback_ok,
            "tick_speedup": self.tick_speedup,
            "action_p99_ms": self.action_p99_ms,
            "deadline_misses": self.deadline_misses,
            "process_alive": self.process_alive,
            "rcon_ready": self.rcon_ready,
            "map_progress": self.map_progress,
            "ipc_conflict": self.ipc_conflict,
            "inference_timeout": self.inference_timeout,
            "action_loss": self.action_loss,
            "memory_error": self.memory_error,
            "gpu_error": self.gpu_error,
            "cpu_percent": self.cpu_percent,
            "ram_mb": self.ram_mb,
            "gpu_percent": self.gpu_percent,
            "resource_evidence": self.resource_evidence,
            "resource_error": self.resource_error,
            "stable": self.stable,
            "detail": self.detail,
            "stable_window_s": self.stable_window_s,
        }


@dataclass(frozen=True)
class TimescaleProbeReportV1:
    selected_timescale: int
    supported: bool
    samples: tuple[TimescaleProbeSampleV1, ...]
    selection_reason: str
    probe_identity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_identity", dict(self.probe_identity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_timescale": self.selected_timescale,
            "supported": self.supported,
            "selection_reason": self.selection_reason,
            "probe_identity": dict(self.probe_identity),
            "probe_fingerprint": self.probe_identity.get("fingerprint", ""),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def _validate_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a single safe path component")
    return value


def server_get5_path(server_root: Path, run_id: str, instance_id: str) -> Path:
    root = Path(server_root).resolve(strict=False)
    safe_run_id = _validate_component(run_id, "run_id")
    if not isinstance(instance_id, str) or not _INSTANCE_PATTERN.fullmatch(instance_id):
        raise ValueError("instance_id must be a two-digit value")
    path = root / "csgo" / "cfg" / "get5" / "slbots" / safe_run_id / f"instance-{instance_id}.json"
    resolved = path.resolve(strict=False)
    if root not in resolved.parents:
        raise ValueError("Get5 config path escaped server root")
    return resolved


def build_server_specs(
    count: int,
    run_id: str,
    server_root: Path,
    output_root: Path,
) -> tuple[ServerInstanceSpecV1, ...]:
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 32:
        raise ValueError("server count must be between 1 and 32")
    safe_run_id = _validate_component(run_id, "run_id")
    server_root = Path(server_root)
    output_root = Path(output_root)
    return tuple(
        ServerInstanceSpecV1(
            instance_id=f"{index:02d}",
            game_port=27100 + index,
            client_port=27200 + index,
            tv_port=27300 + index,
            steam_port=27400 + index,
            rcon_port=27100 + index,
            ipc_name=f"SLBots_de_mirage_{index:02d}",
            get5_config_path=server_get5_path(server_root, safe_run_id, f"{index:02d}"),
            log_dir=(output_root / "logs" / f"instance-{index:02d}").resolve(strict=False),
        )
        for index in range(1, count + 1)
    )


def _replace_match_id(value: Any, match_id: str) -> Any:
    if isinstance(value, str):
        return value.replace("{MATCH_ID}", match_id).replace("{MATCHID}", match_id)
    if isinstance(value, list):
        return [_replace_match_id(item, match_id) for item in value]
    if isinstance(value, dict):
        return {key: _replace_match_id(item, match_id) for key, item in value.items()}
    return value


def render_get5_config(
    spec: ServerInstanceSpecV1,
    match_id: str,
    *,
    generation: int,
    template_path: Path | None = None,
) -> dict[str, Any]:
    if not match_id or not isinstance(match_id, str):
        raise ValueError("match_id is required")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    source = _GET5_TEMPLATE if template_path is None else Path(template_path)
    data = _replace_match_id(json.loads(source.read_text(encoding="utf-8")), match_id)
    if not isinstance(data, dict):
        raise ValueError("Get5 template root must be an object")
    data["matchid"] = match_id
    cvars = data.setdefault("cvars", {})
    if not isinstance(cvars, dict):
        raise ValueError("Get5 template cvars must be an object")
    cvars["sl_bots_instance_id"] = spec.instance_id
    cvars["sl_bots_policy_generation"] = str(generation)
    cvars["get5_server_id"] = f"slbots-{spec.instance_id}"
    return data


class SingleMatchWaveError(RuntimeError):
    """A match failed before reaching a terminal Get5 manifest."""


def _field_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _invoke_compatible(callback: Callable[..., Any], *args: Any) -> Any:
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


def _package_sha256(package: Any) -> str:
    metadata = _field_value(package, "metadata", {})
    if isinstance(metadata, Mapping):
        declared = metadata.get("package_sha256") or metadata.get("sha256")
        if declared:
            digest = str(declared).lower()
            if len(digest) == 64 and not any(character not in "0123456789abcdef" for character in digest):
                return digest
    declared = _field_value(package, "package_sha256")
    if declared:
        digest = str(declared).lower()
        if len(digest) == 64 and not any(character not in "0123456789abcdef" for character in digest):
            return digest
    payload = {
        "generation": _field_value(package, "generation", _field_value(package, "policy_generation", 0)),
        "decision_sha256": str(_field_value(package, "decision_sha256", "")),
        "action_sha256": str(_field_value(package, "action_sha256", "")),
        "metadata": metadata if isinstance(metadata, Mapping) else {},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _package_context(package: Any, host_timescale: int, ruleset: str) -> dict[str, Any]:
    purpose = _field_value(package, "purpose")
    if hasattr(purpose, "value"):
        purpose = purpose.value
    if purpose is not None and str(purpose) != "test_only":
        raise ValueError("single self-play wave accepts only test_only packages")
    generation = _field_value(package, "generation", _field_value(package, "policy_generation", 0))
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("single self-play package generation must be a non-negative integer")
    return {
        "package_sha256": _package_sha256(package),
        "policy_generation": generation,
        "host_timescale": host_timescale,
        "ruleset": ruleset,
    }


def _match_progress_key(event: Any) -> tuple[Any, ...] | None:
    values = tuple(
        _field_value(event, name)
        for name in (
            "server_tick",
            "tick",
            "get5_progress",
            "get5_state",
            "round",
            "phase",
            "score",
        )
    )
    if any(value is not None for value in values):
        return values
    if bool(_field_value(event, "terminal", False)):
        return ("terminal",)
    return None


def _open_match_session(
    runner: Callable[..., Any],
    spec: ServerInstanceSpecV1,
    attempt: int,
    context: Mapping[str, Any],
) -> Any:
    session = _invoke_compatible(runner, spec, attempt, context)
    if hasattr(session, "poll") and callable(session.poll):
        return session
    if isinstance(session, Mapping) and "events" in session:
        return iter(session["events"] or ())
    if isinstance(session, (str, bytes, bytearray)):
        return iter((session,))
    if isinstance(session, Mapping) or isinstance(session, CompleteMatchManifestV1):
        return iter((session,))
    try:
        return iter(session)
    except TypeError:
        return iter((session,))


def _next_match_event(session: Any) -> tuple[bool, Any]:
    if hasattr(session, "poll") and callable(session.poll):
        event = session.poll()
        if event is None:
            process = getattr(session, "process", None)
            is_alive = getattr(process, "is_alive", None)
            if callable(is_alive) and bool(is_alive()):
                # Pollable sessions may use None for an empty-but-live queue.
                # Only an exhausted iterator or a dead process is an ended
                # stream; the watchdog handles a live session with no progress.
                return True, {"status": "waiting"}
        return event is not None, event
    try:
        return True, next(session)
    except StopIteration:
        return False, None


def _close_match_session(session: Any) -> None:
    close = getattr(session, "close", None)
    if callable(close):
        close()


def _is_match_failure(event: Any) -> tuple[bool, str]:
    status = str(_field_value(event, "status", "")).strip().lower()
    if status in {"crashed", "crash", "failed", "failure", "stalled", "timeout", "aborted"}:
        return True, str(_field_value(event, "reason", status))
    if bool(_field_value(event, "crashed", False)) or bool(_field_value(event, "stalled", False)):
        return True, str(_field_value(event, "reason", "match failed"))
    if bool(_field_value(event, "error", False)):
        return True, str(_field_value(event, "reason", "match failed"))
    return False, ""


def _build_complete_manifest(
    spec: ServerInstanceSpecV1,
    event: Any,
    context: Mapping[str, Any],
    attempt: int,
) -> CompleteMatchManifestV1:
    nested = _field_value(event, "manifest")
    source = nested if nested is not None else event
    package_sha = str(_field_value(source, "package_sha256", context["package_sha256"])).lower()
    generation = int(_field_value(source, "policy_generation", context["policy_generation"]))
    timescale = int(_field_value(source, "host_timescale", context["host_timescale"]))
    ruleset = str(_field_value(source, "ruleset", context["ruleset"]))
    if package_sha != context["package_sha256"]:
        raise SingleMatchWaveError("completed match package hash does not match the wave package")
    if generation != context["policy_generation"]:
        raise SingleMatchWaveError("completed match generation does not match the wave package")
    if timescale != context["host_timescale"]:
        raise SingleMatchWaveError("completed match timescale does not match the wave")
    if ruleset != context["ruleset"]:
        raise SingleMatchWaveError("completed match ruleset does not match the wave")
    match_id = str(
        _field_value(
            source,
            "match_id",
            _field_value(event, "match_id", f"single-wave-{spec.instance_id}"),
        )
    )
    shard = _field_value(source, "shard", _field_value(source, "rollout"))
    if shard is None:
        raise SingleMatchWaveError(
            f"terminal match {match_id} did not provide a verifiable rollout shard"
        )
    return CompleteMatchManifestV1(
        instance_id=str(_field_value(source, "instance_id", spec.instance_id)),
        match_id=match_id,
        package_sha256=package_sha,
        policy_generation=generation,
        host_timescale=timescale,
        ruleset=ruleset,
        shard=shard,
        terminal=True,
        attempt=attempt,
    )


def collect_single_match_wave(
    specs: Sequence[ServerInstanceSpecV1],
    package: Any,
    *,
    host_timescale: int,
    ruleset: str = "mr12",
    match_runner: Callable[..., Any] | None = None,
    restart_instance: Callable[..., Any] | None = None,
    watchdog_seconds: float = 120.0,
    wall_clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> CompleteMatchWaveV1:
    """Collect one complete Get5 match per instance with one retry at most.

    ``match_runner`` is deliberately injected at this boundary.  A runner returns either
    an iterator of progress events, a pollable session, or one terminal event.  The
    collector has no duration deadline: only a missing server/Get5 progress watchdog
    can fail an active match.
    """

    selected = tuple(specs)
    if not selected:
        raise ValueError("single match wave requires at least one server specification")
    if len({spec.instance_id for spec in selected}) != len(selected):
        raise ValueError("single match wave instance IDs must be unique")
    if not isinstance(host_timescale, int) or isinstance(host_timescale, bool) or host_timescale < 1:
        raise ValueError("host_timescale must be a positive integer")
    if not isinstance(ruleset, str) or not ruleset:
        raise ValueError("ruleset cannot be empty")
    if watchdog_seconds <= 0:
        raise ValueError("watchdog_seconds must be positive")
    runner = match_runner or _field_value(package, "match_runner")
    if runner is None:
        raise RuntimeError("single match wave requires an injected match_runner")
    now = time.monotonic if wall_clock is None else wall_clock
    pause = time.sleep if sleep is None else sleep
    context = _package_context(package, host_timescale, ruleset)
    attempts: dict[str, int] = {spec.instance_id: 0 for spec in selected}
    sessions: dict[str, Any] = {}
    last_progress: dict[str, float] = {}
    last_progress_key: dict[str, tuple[Any, ...] | None] = {}
    manifests: dict[str, CompleteMatchManifestV1] = {}

    def open_attempt(spec: ServerInstanceSpecV1) -> None:
        instance_id = spec.instance_id
        attempts[instance_id] += 1
        try:
            sessions[instance_id] = _open_match_session(runner, spec, attempts[instance_id], context)
        except Exception as error:
            sessions.pop(instance_id, None)
            raise SingleMatchWaveError(str(error)) from error
        last_progress[instance_id] = float(now())
        last_progress_key[instance_id] = None

    def abort(reason: str) -> CompleteMatchWaveV1:
        for session in tuple(sessions.values()):
            try:
                _close_match_session(session)
            except Exception:
                pass
        return CompleteMatchWaveV1(
            status="aborted",
            package_sha256=context["package_sha256"],
            policy_generation=context["policy_generation"],
            host_timescale=host_timescale,
            ruleset=ruleset,
            manifests=(),
            attempts=attempts,
            discarded_instances=tuple(spec.instance_id for spec in selected),
            abort_reason=str(reason),
        )

    for spec in selected:
        try:
            open_attempt(spec)
        except SingleMatchWaveError as error:
            if attempts[spec.instance_id] >= 2:
                return abort(str(error))
            if restart_instance is not None:
                try:
                    _invoke_compatible(restart_instance, spec, attempts[spec.instance_id], str(error))
                except Exception as restart_error:
                    return abort(f"restart of {spec.instance_id} failed: {restart_error}")
            try:
                open_attempt(spec)
            except SingleMatchWaveError as retry_error:
                return abort(str(retry_error))

    while sessions:
        made_progress = False
        for spec in selected:
            instance_id = spec.instance_id
            session = sessions.get(instance_id)
            if session is None:
                continue
            try:
                emitted, event = _next_match_event(session)
            except Exception as error:
                emitted, event = True, {"status": "crashed", "reason": str(error)}
            if not emitted:
                reason = "match stream ended before Get5 terminal"
                failure = True
            else:
                failure, reason = _is_match_failure(event)
            if not failure and emitted:
                progress_key = _match_progress_key(event)
                if progress_key is not None and progress_key != last_progress_key[instance_id]:
                    last_progress_key[instance_id] = progress_key
                    last_progress[instance_id] = float(now())
                    made_progress = True
                terminal = bool(_field_value(event, "terminal", False))
                status = str(_field_value(event, "status", "")).lower()
                terminal = terminal or status in {"terminal", "complete", "completed"}
                if terminal:
                    try:
                        manifests[instance_id] = _build_complete_manifest(
                            spec,
                            event,
                            context,
                            attempts[instance_id],
                        )
                    except Exception as error:
                        failure, reason = True, str(error)
                    else:
                        _close_match_session(session)
                        sessions.pop(instance_id, None)
                        made_progress = True
                        continue
            if not failure and float(now()) - last_progress[instance_id] >= watchdog_seconds:
                failure = True
                reason = f"no server tick/Get5 progress for {watchdog_seconds:g} seconds"
            if not failure:
                continue
            try:
                _close_match_session(session)
            except Exception:
                pass
            sessions.pop(instance_id, None)
            if attempts[instance_id] >= 2:
                return abort(f"instance {instance_id} failed twice: {reason}")
            if restart_instance is not None:
                try:
                    _invoke_compatible(restart_instance, spec, attempts[instance_id], reason)
                except Exception as restart_error:
                    return abort(f"restart of {instance_id} failed: {restart_error}")
            try:
                open_attempt(spec)
            except SingleMatchWaveError as retry_error:
                return abort(f"instance {instance_id} retry failed: {retry_error}")
            made_progress = True
        if sessions and not made_progress:
            pause(min(0.1, watchdog_seconds / 10.0))

    ordered = tuple(manifests[spec.instance_id] for spec in selected)
    if len(ordered) != len(selected):
        return abort("single match wave ended without one terminal manifest per instance")
    return CompleteMatchWaveV1(
        status="completed",
        package_sha256=context["package_sha256"],
        policy_generation=context["policy_generation"],
        host_timescale=host_timescale,
        ruleset=ruleset,
        manifests=ordered,
        attempts=attempts,
    )


class DedicatedServerFarm:
    def __init__(
        self,
        server_root: Path,
        output_root: Path,
        run_id: str,
        *,
        max_servers: int = 32,
        process_factory: Callable[..., Any] | None = None,
        rcon_client_factory: Callable[..., Any] | None = None,
        readiness_timeout_s: float = 30.0,
    ) -> None:
        if not isinstance(max_servers, int) or isinstance(max_servers, bool) or not 1 <= max_servers <= 32:
            raise ValueError("max_servers must be between 1 and 32")
        self.server_root = Path(server_root).resolve(strict=False)
        self.output_root = Path(output_root).resolve(strict=False)
        self.run_id = _validate_component(run_id, "run_id")
        self.max_servers = max_servers
        self._process_factory = subprocess.Popen if process_factory is None else process_factory
        self._rcon_client_factory = RconClientV1 if rcon_client_factory is None else rcon_client_factory
        if readiness_timeout_s <= 0:
            raise ValueError("readiness_timeout_s must be positive")
        self.readiness_timeout_s = float(readiness_timeout_s)
        self._processes: dict[str, Any] = {}
        self._rcon_clients: dict[str, Any] = {}
        self._rcon_credentials: dict[str, RconCredentialsV1] = {}
        self._readiness: dict[str, RconReadinessV1] = {}
        self._log_handles: dict[str, tuple[Any, Any]] = {}
        self._specs: tuple[ServerInstanceSpecV1, ...] = ()
        self._match_ids: dict[str, str] = {}

    @property
    def processes(self) -> dict[str, Any]:
        return dict(self._processes)

    @property
    def match_ids(self) -> dict[str, str]:
        return dict(self._match_ids)

    @property
    def readiness(self) -> dict[str, RconReadinessV1]:
        return dict(self._readiness)

    def start(
        self,
        specs: Sequence[ServerInstanceSpecV1],
        *,
        dry_run: bool = False,
    ) -> tuple[ServerInstanceSpecV1, ...]:
        selected = tuple(specs)
        if not 1 <= len(selected) <= self.max_servers:
            raise ValueError(f"server count must be between 1 and {self.max_servers}")
        self._validate_specs(selected)
        if dry_run:
            return selected
        resolve_srcds_executable(self.server_root)
        if self._processes:
            raise RuntimeError("server farm is already running")
        self._specs = selected
        try:
            for spec in selected:
                self._start_instance(spec)
        except Exception:
            self.stop(timeout_s=0)
            raise
        return selected

    def load_get5_match(
        self,
        generation: int,
        *,
        instance_id: str | None = None,
        template_path: Path | None = None,
        attempt: int = 1,
        force_start: bool = True,
    ) -> tuple[Path, ...]:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError("match attempt must be a positive integer")
        specs = self._active_specs(instance_id)
        paths: list[Path] = []
        for spec in specs:
            match_id = (
                f"slbots-{self.run_id}-g{generation:03d}-a{attempt:02d}-i{spec.instance_id}"
            )
            config = render_get5_config(
                spec,
                match_id,
                generation=generation,
                template_path=template_path,
            )
            _atomic_write_json(spec.get5_config_path, config)
            relative = spec.get5_config_path.relative_to(self.server_root / "csgo")
            console_path = relative.as_posix()
            process = self._processes[spec.instance_id]
            if process.poll() is not None:
                raise RuntimeError(f"server instance {spec.instance_id} exited before match load")
            self._send_rcon(spec, f"get5_loadmatch {console_path}")
            self._send_rcon(spec, "bot_quota 10")
            self._send_rcon(spec, "bot_quota_mode fill")
            self._send_rcon(spec, "bot_join_after_player 0")
            self._send_rcon(spec, "bot_kick")
            for _ in range(5):
                self._send_rcon(spec, "bot_add_t")
                self._send_rcon(spec, "bot_add_ct")
            # CS:GO processes bot joins asynchronously.  Do not start Get5 while
            # only a subset of the intended roster has entered the server; that
            # produces a plausible-looking warmup/knife state but no complete
            # match.  The status query is an admission check, not a match timeout.
            bot_deadline = time.monotonic() + 5.0
            next_side = "t"
            while time.monotonic() < bot_deadline:
                status = self._send_rcon(spec, "status")
                bot_count = _status_bot_count(status)
                if bot_count is None or bot_count >= 10:
                    break
                self._send_rcon(spec, f"bot_add_{next_side}")
                next_side = "ct" if next_side == "t" else "t"
                time.sleep(0.1)
            self._match_ids[spec.instance_id] = match_id
            paths.append(spec.get5_config_path)
        if force_start:
            self.start_get5_match(instance_id=instance_id)
        return tuple(paths)

    def start_get5_match(self, *, instance_id: str | None = None) -> None:
        """Enter the live Get5 match after every model worker is ready."""

        specs = self._active_specs(instance_id)
        for spec in specs:
            # Get5 counts fake clients while evaluating its ready system.  The
            # command is deliberately separated from load_get5_match so the
            # bridge cannot enter LIVE before the model process has initialized.
            self._send_rcon(spec, "get5_forcestart")
            self._advance_get5_warmup(spec)

    def set_policy_generation(self, generation: int, *, instance_id: str | None = None) -> None:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        specs = self._active_specs(instance_id)
        for spec in specs:
            process = self._processes[spec.instance_id]
            if process.poll() is not None:
                raise RuntimeError(f"server instance {spec.instance_id} exited before policy update")
            self._send_rcon(spec, f"sl_bots_policy_generation {generation}")

    def send_rcon_command(self, command: str, *, instance_id: str | None = None) -> dict[str, str]:
        """Send an administrative command without exposing a process stdin pipe."""

        if not isinstance(command, str) or not command.strip():
            raise ValueError("RCON command cannot be empty")
        return {
            spec.instance_id: self._send_rcon(spec, command)
            for spec in self._active_specs(instance_id)
        }

    def set_host_timescale(self, timescale: int, *, instance_id: str | None = None) -> dict[str, str]:
        if not isinstance(timescale, int) or isinstance(timescale, bool) or timescale < 1:
            raise ValueError("host_timescale must be a positive integer")
        return self.send_rcon_command(f"host_timescale {timescale}", instance_id=instance_id)

    def restart_instance(self, instance_id: str, *, timeout_s: float = 10.0) -> None:
        spec = self._spec_for_instance(instance_id)
        self._active_specs(spec.instance_id)
        self.stop_instance(spec.instance_id, timeout_s=timeout_s)
        self._start_instance(spec)

    def stop_instance(self, instance_id: str, *, timeout_s: float = 10.0) -> None:
        spec = self._spec_for_instance(instance_id)
        process = self._processes.pop(spec.instance_id, None)
        if process is None:
            raise RuntimeError(f"server instance {spec.instance_id} is not running")
        try:
            self._quit_rcon(spec)
            self._stop_process(process, timeout_s=timeout_s)
        finally:
            client = self._rcon_clients.pop(spec.instance_id, None)
            if client is not None and hasattr(client, "close"):
                client.close()
            self._rcon_credentials.pop(spec.instance_id, None)
            self._readiness.pop(spec.instance_id, None)
            handles = self._log_handles.pop(spec.instance_id, ())
            for handle in handles:
                handle.close()
            self._match_ids.pop(spec.instance_id, None)

    def stop(self, *, timeout_s: float = 10.0) -> None:
        processes = tuple(self._processes.values())
        for spec in self._specs:
            if spec.instance_id in self._processes:
                self._quit_rcon(spec)
        deadline = time.perf_counter() + max(0.0, timeout_s)
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.perf_counter())
            try:
                process.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, TimeoutError):
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, TimeoutError):
                    process.kill()
                    process.wait(timeout=2.0)
        for stdout, stderr in self._log_handles.values():
            stdout.close()
            stderr.close()
        self._log_handles.clear()
        for client in self._rcon_clients.values():
            if hasattr(client, "close"):
                client.close()
        self._rcon_clients.clear()
        self._rcon_credentials.clear()
        self._readiness.clear()
        self._processes.clear()
        self._specs = ()
        self._match_ids.clear()

    def orderly_shutdown(
        self,
        *,
        stop_sampling: Callable[[], Any] | None = None,
        flush_manifest: Callable[[], Any] | None = None,
        runtime_stop: Callable[[], Any] | None = None,
        worker_stop: Callable[[], Any] | None = None,
        timeout_s: float = 10.0,
    ) -> tuple[str, ...]:
        """Close one wave in a deterministic order and only touch tracked processes."""

        order: list[str] = []
        if stop_sampling is not None:
            stop_sampling()
            order.append("stop_sampling")
        if flush_manifest is not None:
            flush_manifest()
            order.append("flush_manifest")
        if runtime_stop is not None:
            runtime_stop()
            order.append("runtime_stop")
        if worker_stop is not None:
            worker_stop()
            order.append("worker_stop")
        self.stop(timeout_s=timeout_s)
        order.extend(("rcon_quit", "launcher_wait"))
        return tuple(order)

    def _start_instance(self, spec: ServerInstanceSpecV1) -> None:
        if spec.instance_id in self._processes:
            raise RuntimeError(f"server instance {spec.instance_id} is already running")
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (spec.log_dir / "server.stdout.log").open("ab")
        stderr = (spec.log_dir / "server.stderr.log").open("ab")
        executable = resolve_srcds_executable(self.server_root)
        # Source's command-line parser treats a value beginning with ``-`` as
        # another option.  Use hex-only credentials so every generated
        # password is unambiguously consumed by ``+rcon_password``.
        password = secrets.token_hex(24)
        credentials = RconCredentialsV1("127.0.0.1", spec.rcon_port, password)
        write_rcon_run_config(
            self.output_root / "run-config" / f"instance-{spec.instance_id}.json",
            credentials,
        )
        command = build_srcds_command(
            executable,
            game_port=spec.game_port,
            client_port=spec.client_port,
            tv_port=spec.tv_port,
            steam_port=spec.steam_port,
            instance_id=spec.instance_id,
            rcon_password=password,
        )
        try:
            if self._process_factory is subprocess.Popen:
                process = launch_srcds(
                    self.server_root,
                    command[1:],
                    stdout=stdout,
                    stderr=stderr,
                )
            else:
                process = self._process_factory(
                    command,
                    cwd=str(executable.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    bufsize=1,
                )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        self._log_handles[spec.instance_id] = (stdout, stderr)
        self._processes[spec.instance_id] = process
        self._rcon_credentials[spec.instance_id] = credentials
        client = self._rcon_client_factory(
            credentials.host,
            credentials.port,
            credentials.password,
            timeout_s=2.0,
        )
        self._rcon_clients[spec.instance_id] = client
        try:
            self._readiness[spec.instance_id] = wait_for_rcon_readiness(
                process,
                client,
                expected_map="de_mirage",
                required_plugins=("Get5", "sl_bots"),
                timeout_s=self.readiness_timeout_s,
            )
            # Source executes +cvar arguments before SourceMod creates plugin
            # ConVars, so +sl_bots_instance_id is reported as an unknown
            # command on a real srcds.  Set the instance after readiness and
            # reload the bridge so its transport is opened under the unique
            # per-instance mapping instead of the shared map-only fallback.
            self._send_rcon(spec, f"sl_bots_instance_id {spec.instance_id}")
            self._send_rcon(spec, "sm plugins reload sl_bots_bridge")
        except Exception:
            self._rcon_clients.pop(spec.instance_id, None)
            self._rcon_credentials.pop(spec.instance_id, None)
            self._processes.pop(spec.instance_id, None)
            if hasattr(client, "close"):
                client.close()
            self._stop_process(process, timeout_s=2.0)
            handles = self._log_handles.pop(spec.instance_id, ())
            for handle in handles:
                handle.close()
            raise

    @staticmethod
    def _stop_process(process: Any, *, timeout_s: float) -> None:
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=max(0.0, timeout_s))
        except (subprocess.TimeoutExpired, TimeoutError):
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, TimeoutError):
                process.kill()
                process.wait(timeout=2.0)

    def _spec_for_instance(self, instance_id: str) -> ServerInstanceSpecV1:
        normalized = str(instance_id)
        for spec in self._specs:
            if spec.instance_id == normalized:
                return spec
        raise ValueError(f"unknown server instance: {instance_id}")

    def _active_specs(self, instance_id: str | None) -> tuple[ServerInstanceSpecV1, ...]:
        expected = {spec.instance_id for spec in self._specs}
        if not self._specs or set(self._processes) != expected:
            raise RuntimeError("server farm must be started before operating on a match")
        if instance_id is None:
            return self._specs
        spec = self._spec_for_instance(instance_id)
        if spec.instance_id not in self._processes:
            raise RuntimeError(f"server instance {spec.instance_id} is not running")
        return (spec,)

    def _validate_specs(self, specs: Sequence[ServerInstanceSpecV1]) -> None:
        ids = [spec.instance_id for spec in specs]
        if len(set(ids)) != len(ids):
            raise ValueError("instance IDs must be unique")
        ports = [port for spec in specs for port in spec.ports]
        if len(set(ports)) != len(ports):
            raise ValueError("server ports must be unique")
        if any(spec.rcon_port != spec.game_port for spec in specs):
            raise ValueError("Source RCON must use the instance game port")
        ipc_names = [spec.ipc_name for spec in specs]
        if len(set(ipc_names)) != len(ipc_names):
            raise ValueError("IPC names must be unique")
        for spec in specs:
            if not _INSTANCE_PATTERN.fullmatch(spec.instance_id):
                raise ValueError("instance_id must be a two-digit value")

    def _send_rcon(self, spec: ServerInstanceSpecV1, command: str) -> str:
        client = self._rcon_clients.get(spec.instance_id)
        if client is None:
            raise RuntimeError(f"RCON client is not ready for server instance {spec.instance_id}")
        return str(client.execute(command))

    def _advance_get5_warmup(self, spec: ServerInstanceSpecV1) -> None:
        """Finish the CS:GO warmup countdown once Get5 has entered going_live.

        The target CS:GO dedicated server can leave Get5 in ``going_live`` even
        with ``mp_warmup_pausetimer 0`` when the server has no human clients.
        Get5 transitions on the next ``round_prestart`` event; ending warmup by
        RCON is the supported engine command that produces that event.  A fake
        RCON client that does not expose JSON status is treated as an offline
        test double and returns immediately.
        """

        deadline = time.monotonic() + _GET5_WARMUP_ADVANCE_TIMEOUT_S
        warmup_end_sent = False
        while True:
            state = _get5_gamestate(self._send_rcon(spec, "get5_status"))
            if state is None:
                return
            if state == "live":
                return
            if state == "going_live":
                if not warmup_end_sent:
                    self._send_rcon(spec, "mp_warmup_end")
                    warmup_end_sent = True
            elif state not in {"warmup", "knife", "waiting_for_knife_round_decision"}:
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(_GET5_WARMUP_POLL_INTERVAL_S)

    def _quit_rcon(self, spec: ServerInstanceSpecV1) -> None:
        client = self._rcon_clients.get(spec.instance_id)
        if client is None:
            return
        try:
            client.execute("quit")
        except Exception:
            # The process wait/termination path remains authoritative if RCON is already gone.
            return


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _status_bot_count(status: str) -> int | None:
    match = _STATUS_BOT_COUNT_PATTERN.search(str(status))
    return None if match is None else int(match.group(1))


def _get5_gamestate(status: str) -> str | None:
    try:
        payload = json.loads(str(status))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    state = payload.get("gamestate")
    if not isinstance(state, str) or not state.strip():
        return None
    return state.strip().lower()


def _write_probe_report(path: str | Path | None, report: Any) -> None:
    if path is not None:
        _atomic_write_json(Path(path), report.to_dict())


def _capacity_sample(value: CapacityProbeSampleV1 | Mapping[str, Any], requested: int) -> CapacityProbeSampleV1:
    if isinstance(value, CapacityProbeSampleV1):
        sample = value
    elif isinstance(value, Mapping):
        sample = CapacityProbeSampleV1(requested_instances=requested, **dict(value))
    else:
        raise TypeError("capacity probe runner must return CapacityProbeSampleV1 or a mapping")
    if sample.requested_instances != requested:
        raise ValueError("capacity probe sample count does not match the requested candidate")
    return sample


def probe_capacity(
    max_instances: int = 32,
    *,
    probe_runner: Callable[[int], CapacityProbeSampleV1 | Mapping[str, Any]] | None = None,
    stable_window_s: float = 20.0,
    report_path: str | Path | None = None,
    probe_identity: Mapping[str, Any] | None = None,
) -> CapacityProbeReportV1:
    """Probe 1x capacity exponentially, then binary-search the first failure."""

    if not isinstance(max_instances, int) or isinstance(max_instances, bool) or not 1 <= max_instances <= 32:
        raise ValueError("max_instances must be between 1 and 32")
    if stable_window_s < 0:
        raise ValueError("stable_window_s cannot be negative")
    if probe_runner is None:
        raise RuntimeError("probe_capacity requires an injected probe_runner for this offline boundary")
    candidates: list[int] = []
    candidate = 1
    while candidate < max_instances:
        candidates.append(candidate)
        candidate *= 2
    candidates.append(max_instances)
    samples: list[CapacityProbeSampleV1] = []
    by_count: dict[int, CapacityProbeSampleV1] = {}

    def run(count: int) -> CapacityProbeSampleV1:
        if count not in by_count:
            sample = _capacity_sample(probe_runner(count), count)
            by_count[count] = sample
            samples.append(sample)
        return by_count[count]

    def is_stable(sample: CapacityProbeSampleV1) -> bool:
        return sample.stable and (
            stable_window_s <= 0.0 or sample.stable_window_s >= stable_window_s
        )

    last_stable = 0
    first_failed: int | None = None
    for count in candidates:
        sample = run(count)
        if is_stable(sample):
            last_stable = count
            continue
        first_failed = count
        break
    if first_failed is not None:
        low = last_stable + 1
        high = first_failed - 1
        while low <= high:
            middle = (low + high) // 2
            if is_stable(run(middle)):
                last_stable = middle
                low = middle + 1
            else:
                high = middle - 1
        reason = f"first unstable candidate {first_failed}; binary search selected {last_stable}"
    else:
        reason = f"all exponential candidates stable through {last_stable}"
    report = CapacityProbeReportV1(
        last_stable,
        tuple(samples),
        reason,
        probe_identity=dict(probe_identity or {}),
    )
    _write_probe_report(report_path, report)
    return report

def _timescale_sample(value: TimescaleProbeSampleV1 | Mapping[str, Any], timescale: int) -> TimescaleProbeSampleV1:
    if isinstance(value, TimescaleProbeSampleV1):
        sample = value
    elif isinstance(value, Mapping):
        sample = TimescaleProbeSampleV1(timescale=timescale, **dict(value))
    else:
        raise TypeError("timescale probe runner must return TimescaleProbeSampleV1 or a mapping")
    if sample.timescale != timescale:
        raise ValueError("timescale probe sample does not match the requested candidate")
    return sample


def probe_timescale(
    specs: Sequence[ServerInstanceSpecV1],
    candidates: Sequence[int] = (1, 2, 4, 8),
    *,
    probe_runner: Callable[[int, tuple[ServerInstanceSpecV1, ...]], TimescaleProbeSampleV1 | Mapping[str, Any]] | None = None,
    stable_window_s: float = 20.0,
    report_path: str | Path | None = None,
    probe_identity: Mapping[str, Any] | None = None,
) -> TimescaleProbeReportV1:
    """Probe one timescale for the whole server group and select one shared value."""

    selected_specs = tuple(specs)
    if not selected_specs:
        raise ValueError("timescale probe requires at least one server specification")
    normalized = tuple(int(value) for value in candidates)
    if not normalized or normalized[0] != 1 or any(value < 1 for value in normalized):
        raise ValueError("timescale candidates must start at 1 and be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("timescale candidates must be unique")
    if stable_window_s < 0:
        raise ValueError("stable_window_s cannot be negative")
    if probe_runner is None:
        raise RuntimeError("probe_timescale requires an injected probe_runner for this offline boundary")
    samples: list[TimescaleProbeSampleV1] = []
    selected = 1

    def is_stable(sample: TimescaleProbeSampleV1) -> bool:
        return sample.stable and (
            stable_window_s <= 0.0 or sample.stable_window_s >= stable_window_s
        )

    for timescale in normalized:
        sample = _timescale_sample(probe_runner(timescale, selected_specs), timescale)
        samples.append(sample)
        if timescale == 1 and (not sample.supported or not sample.readback_ok):
            # A real CS:GO server can reject this cheat-protected cvar in
            # multiplayer.  Once the 1x readback proves the control is not
            # available, higher candidates cannot be meaningful and would
            # only create redundant server runs.
            break
        if is_stable(sample):
            selected = timescale
    baseline = samples[0]
    supported = baseline.supported and baseline.readback_ok
    if not supported:
        reason = "host_timescale is unsupported or readback failed; selected 1x"
        selected = 1
    else:
        reason = f"selected highest stable shared timescale {selected}x"
    report = TimescaleProbeReportV1(
        selected,
        supported,
        tuple(samples),
        reason,
        probe_identity=dict(probe_identity or {}),
    )
    _write_probe_report(report_path, report)
    return report


__all__ = [
    "CapacityProbeReportV1",
    "CapacityProbeSampleV1",
    "CompleteMatchManifestV1",
    "CompleteMatchWaveV1",
    "DedicatedServerFarm",
    "ServerInstanceSpecV1",
    "SingleMatchWaveError",
    "TimescaleProbeReportV1",
    "TimescaleProbeSampleV1",
    "build_server_specs",
    "collect_single_match_wave",
    "probe_capacity",
    "probe_timescale",
    "render_get5_config",
    "server_get5_path",
]
