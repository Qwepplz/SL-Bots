"""Test-only hierarchical training boundary and configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import secrets
import shutil
import subprocess
import struct
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Literal

import yaml

from .contracts import DataPurpose, Phase, ensure_purpose
from .lineage import DatasetManifestV1, require_test_only
from .selfplay import MatchCompletionV1, RolloutEnvelopeV1, RosterProfile, SelfPlayController
from .selfplay_worker import SelfPlayWorker
from .server_farm import (
    CompleteMatchWaveV1,
    _probe_resource_evidence_is_valid,
    collect_single_match_wave,
)


_ALLOWED_TIMESCALES = (1, 2, 4, 8)
_FORBIDDEN_DURATION_VALUES = frozenset({600, 7200})
_TEST_ROSTER_SIZE = 10
_DEFAULT_PROBE_MAX_AGE_S = 24.0 * 60.0 * 60.0


def _run_hierarchical_selfplay_process(
    spec: Any,
    generation: int,
    match_id: str,
    data_root: Path,
    purpose: DataPurpose,
    source_manifest: DatasetManifestV1 | None,
    decision_model_path: Path,
    action_model_path: Path,
    epoch: int,
    profiles: tuple[RosterProfile, ...],
    rollout_queue: Any,
    metrics_queue: Any,
    error_queue: Any,
    ready_event: Any,
    stop_event: Any,
    rollout_horizon: int,
    wait_timeout_ms: int,
    ipc_capacity: int,
    ipc_timeout_s: float,
    max_bots: int,
) -> None:
    """Run one test-only hierarchical worker in a spawn-safe child process."""

    transport: Any | None = None
    worker_started = False
    try:
        # Keep the shared-memory open/retry boundary identical to the production
        # worker, but construct the dual-session hierarchical runtime here so
        # the formal test-pipeline entry point is not dependent on an injected
        # fake runner or the legacy single-model actor.
        from .training_pipeline import _open_child_transport
        from .runtime import HierarchicalRuntimeService, HierarchicalSharedMemoryRuntimeV1
        from .get5_control import Get5ControlState

        transport = _open_child_transport(
            spec,
            capacity=int(ipc_capacity),
            epoch=int(epoch),
            timeout_s=float(ipc_timeout_s),
        )
        runtime = HierarchicalRuntimeService(
            decision_model_path=decision_model_path,
            action_model_path=action_model_path,
            max_bots=int(max_bots),
            # Self-play must be a categorical behavior policy if its log
            # probabilities are later consumed by MAPPO.
            stochastic_decisions=True,
            stochastic_actions=True,
        )
        shared_runtime = HierarchicalSharedMemoryRuntimeV1(
            transport=transport,
            runtime=runtime,
            max_bots=int(max_bots),
        )
        controller = SelfPlayController(
            data_root=data_root,
            purpose=purpose,
            source_manifest=source_manifest,
            adapter=None,
            instance_id=str(spec.instance_id),
            match_id=str(match_id),
            policy_generation=int(generation),
        )
        controller.start_episode(
            Phase.LIVE,
            profiles,
            episode_id=f"{match_id}-live",
            epoch=int(epoch),
        )
        worker = SelfPlayWorker(
            transport=transport,
            runtime=shared_runtime,
            controller=controller,
            control_state=Get5ControlState(expected_match_id=str(match_id)),
            rollout_horizon=int(rollout_horizon),
            wait_timeout_ms=int(wait_timeout_ms),
        )
        ready_event.set()
        worker_started = True
        worker.run(stop_event, rollout_queue, metrics_queue)
    except BaseException as error:
        try:
            error_queue.put({"type": type(error).__name__, "message": str(error)})
        except Exception:
            pass
        raise
    finally:
        if transport is not None and not worker_started:
            transport.close()


@dataclass
class _HierarchicalMatchSession:
    """Parent-side non-blocking view of one spawned hierarchical worker."""

    spec: Any
    match_id: str
    package_context: Mapping[str, Any]
    process: Any
    stop_event: Any
    rollout_queue: Any
    metrics_queue: Any
    error_queue: Any
    ready_event: Any | None = None
    envelopes: list[RolloutEnvelopeV1] = field(default_factory=list)
    completion: MatchCompletionV1 | None = None
    latest_metric: Mapping[str, Any] | None = None
    metrics_seen: int = 0
    metric_backlog_events: int = 0
    metric_samples_with_action: int = 0
    action_ack_count: int = 0
    action_ack_missing_count: int = 0
    error_box: list[str] = field(default_factory=list)
    terminal_reported: bool = False
    closed: bool = False

    def poll(self) -> Mapping[str, Any] | None:
        if self.closed:
            return None
        self._drain_queues()
        if self.error_box:
            return {
                "status": "crashed",
                "reason": self.error_box[-1],
                "match_id": self.match_id,
            }
        if self.completion is not None and not self.terminal_reported:
            self.terminal_reported = True
            return {
                "status": "completed",
                "terminal": True,
                "instance_id": self.spec.instance_id,
                "match_id": self.match_id,
                "package_sha256": self.package_context["package_sha256"],
                "policy_generation": self.package_context["policy_generation"],
                "host_timescale": self.package_context["host_timescale"],
                "ruleset": self.package_context["ruleset"],
                # Keep the terminal marker attached to the segment list. A
                # lone non-terminal envelope is not itself a complete match;
                # the marker is the proof that no more segments can arrive.
                "shard": self.completion,
            }
        process_exitcode = getattr(self.process, "exitcode", None)
        is_alive = getattr(self.process, "is_alive", lambda: False)()
        if not is_alive and process_exitcode not in (None, 0) and not self.terminal_reported:
            return {
                "status": "crashed",
                "reason": f"hierarchical worker exited with code {process_exitcode}",
                "match_id": self.match_id,
            }
        if self.latest_metric is not None:
            metric = self.latest_metric
            self.latest_metric = None
            live_event = dict(metric)
            live_event.update(
                {
                    "status": "live",
                    "server_tick": int(metric.get("server_tick", 0)),
                    "match_id": self.match_id,
                    "instance_id": self.spec.instance_id,
                    "metrics_seen": self.metrics_seen,
                    "metric_backlog_events": self.metric_backlog_events,
                    "source_action_sample_count": self.metric_samples_with_action,
                    "source_action_ack_count": self.action_ack_count,
                    "source_action_ack_ratio": (
                        self.action_ack_count / self.metric_samples_with_action
                        if self.metric_samples_with_action
                        else 0.0
                    ),
                }
            )
            return live_event
        if not is_alive and not self.terminal_reported:
            return {
                "status": "crashed",
                "reason": "hierarchical worker exited before terminal rollout",
                "match_id": self.match_id,
            }
        # An alive worker with an empty IPC queue is a normal poll result, not
        # an ended match stream.  The farm watchdog owns the decision about
        # whether this waiting state lasts too long.
        return {
            "status": "waiting",
            "match_id": self.match_id,
            "instance_id": self.spec.instance_id,
        }

    def _drain_queues(self) -> None:
        # A parent poll may intentionally run slower than the 128 Hz worker.
        # Only a genuinely large pending queue is evidence of transport
        # pressure; multiple metrics coalesced by one normal poll are not.
        try:
            pending_metrics = int(self.metrics_queue.qsize())
        except (AttributeError, NotImplementedError, OSError, ValueError):
            pending_metrics = 0
        if pending_metrics > 32:
            self.metric_backlog_events += pending_metrics
        while True:
            try:
                error = self.error_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(error, Mapping):
                self.error_box.append(
                    f"{error.get('type', 'ChildProcessError')}: {error.get('message', '')}"
                )
            else:
                self.error_box.append(str(error))
        while True:
            try:
                envelope = self.rollout_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(envelope, MatchCompletionV1):
                self.completion = envelope
                continue
            if not isinstance(envelope, RolloutEnvelopeV1):
                self.error_box.append("hierarchical worker emitted a non-envelope rollout")
                continue
            self.envelopes.append(envelope)
        while True:
            try:
                metric = self.metrics_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(metric, Mapping):
                self.metrics_seen += 1
                self.metric_samples_with_action += 1
                if bool(metric.get("source_action_apply_ack", False)):
                    self.action_ack_count += 1
                else:
                    self.action_ack_missing_count += 1
                self.latest_metric = dict(metric)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        is_alive = getattr(self.process, "is_alive", lambda: False)
        if is_alive():
            self.process.join(timeout=5.0)
        if is_alive():
            terminate = getattr(self.process, "terminate", None)
            if callable(terminate):
                terminate()
            self.process.join(timeout=2.0)
        for channel in (self.rollout_queue, self.metrics_queue, self.error_queue):
            close = getattr(channel, "close", None)
            if callable(close):
                close()
            cancel = getattr(channel, "cancel_join_thread", None)
            if callable(cancel):
                cancel()


class _HierarchicalMatchRunner:
    """Real farm/worker match runner used by the default test-only CLI."""

    def __init__(
        self,
        config: "HierarchicalTestConfigV1",
        specs: Sequence[Any],
        package: Any,
        *,
        data_root: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        from .server_farm import DedicatedServerFarm

        self.config = config
        self.specs = tuple(specs)
        self.package = package
        self.data_root = Path(config.data_root if data_root is None else data_root).resolve(
            strict=False
        )
        self.run_id = str(run_id or f"test-pipeline-{secrets.token_hex(8)}")
        self.farm = DedicatedServerFarm(
            config.server_root,
            self.data_root,
            self.run_id,
            max_servers=len(self.specs),
        )
        self._context: dict[str, Any] | None = None
        self._source_manifest = self._load_source_manifest()
        self._sessions: dict[str, _HierarchicalMatchSession] = {}
        self._started_match_ids: set[str] = set()
        self._started = False

    def __call__(self, spec: Any, attempt: int, context: Mapping[str, Any]) -> Any:
        if not self._started:
            self._start_wave(context)
        elif self._context != dict(context):
            raise RuntimeError("hierarchical match runner context changed within one wave")
        instance_id = str(spec.instance_id)
        if int(attempt) > 1:
            old = self._sessions.pop(instance_id, None)
            if old is not None:
                old.close()
            self._spawn_session(spec, context)
            pending = [
                (pending_id, pending_session)
                for pending_id, pending_session in self._sessions.items()
                if pending_id not in self._started_match_ids
            ]
            for _pending_id, pending_session in pending:
                self._wait_for_worker_ready(pending_session)
            for pending_id, _pending_session in pending:
                self.farm.start_get5_match(instance_id=pending_id)
                self._started_match_ids.add(pending_id)
        session = self._sessions.get(instance_id)
        if session is None:
            self._spawn_session(spec, context)
            session = self._sessions[instance_id]
        return session

    def restart_instance(self, spec: Any, attempt: int, reason: str = "") -> None:
        del reason
        instance_id = str(spec.instance_id)
        old = self._sessions.pop(instance_id, None)
        if old is not None:
            old.close()
        self.farm.restart_instance(instance_id)
        context = self._context
        if context is None:
            raise RuntimeError("cannot restart a match before the runner starts")
        self.farm.set_policy_generation(int(context["policy_generation"]), instance_id=instance_id)
        self.farm.set_host_timescale(int(context["host_timescale"]), instance_id=instance_id)
        self.farm.load_get5_match(
            int(context["policy_generation"]),
            instance_id=instance_id,
            attempt=max(1, int(attempt) + 1),
            force_start=False,
        )
        self._started_match_ids.discard(instance_id)

    def close(self) -> None:
        for session in tuple(self._sessions.values()):
            session.close()
        self._sessions.clear()
        self._started_match_ids.clear()
        if self._started:
            self.farm.stop(timeout_s=5.0)
        self._started = False

    def _start_wave(self, context: Mapping[str, Any]) -> None:
        if not self.specs:
            raise ValueError("hierarchical match runner requires at least one server specification")
        if str(context.get("ruleset", "")).lower() != "mr12":
            raise ValueError("hierarchical self-play runner only supports Get5 MR12")
        self._context = dict(context)
        generation = int(context["policy_generation"])
        timescale = int(context["host_timescale"])
        self.farm.start(self.specs, dry_run=False)
        self.farm.set_policy_generation(generation)
        if timescale > 1:
            self.farm.send_rcon_command("sv_cheats 1")
        self.farm.set_host_timescale(timescale)
        self.farm.load_get5_match(generation, attempt=1, force_start=False)
        self._started = True
        for spec in self.specs:
            self._spawn_session(spec, context)
        for session in tuple(self._sessions.values()):
            self._wait_for_worker_ready(session)
        self.farm.start_get5_match()
        self._started_match_ids.update(str(spec.instance_id) for spec in self.specs)

    def _spawn_session(self, spec: Any, context: Mapping[str, Any]) -> None:
        epoch = self._shared_memory_epoch(spec)
        generation = int(context["policy_generation"])
        match_id = str(self.farm.match_ids.get(spec.instance_id, f"single-wave-{spec.instance_id}"))
        mp_context = mp.get_context("spawn")
        stop_event = mp_context.Event()
        rollout_queue = mp_context.Queue()
        metrics_queue = mp_context.Queue()
        error_queue = mp_context.Queue()
        ready_event = mp_context.Event()
        profiles = tuple(
            RosterProfile(bot_id=f"bot-{index:02d}", team=2 if index % 2 == 0 else 3)
            for index in range(_TEST_ROSTER_SIZE)
        )
        process = mp_context.Process(
            target=_run_hierarchical_selfplay_process,
            name=f"sl-bots-hierarchical-selfplay-{spec.instance_id}",
            args=(
                spec,
                generation,
                match_id,
                self.data_root,
                DataPurpose.TEST_ONLY,
                self._source_manifest,
                Path(self.package.decision_path),
                Path(self.package.action_path),
                epoch,
                profiles,
                rollout_queue,
                metrics_queue,
                error_queue,
                ready_event,
                stop_event,
                1024,
                100,
                64,
                60.0,
                _TEST_ROSTER_SIZE,
            ),
        )
        session = _HierarchicalMatchSession(
            spec=spec,
            match_id=match_id,
            package_context=context,
            process=process,
            stop_event=stop_event,
            rollout_queue=rollout_queue,
            metrics_queue=metrics_queue,
            error_queue=error_queue,
            ready_event=ready_event,
        )
        self._sessions[str(spec.instance_id)] = session
        process.start()

    def _wait_for_worker_ready(self, session: _HierarchicalMatchSession) -> None:
        """Require runtime/model initialization before Get5 is forced live."""

        ready_event = session.ready_event
        if ready_event is None:
            raise RuntimeError(f"hierarchical worker {session.match_id} has no ready handshake")
        deadline = time.monotonic() + 60.0
        while True:
            session._drain_queues()
            if session.error_box:
                raise RuntimeError(
                    f"hierarchical worker {session.match_id} failed before ready: "
                    f"{session.error_box[-1]}"
                )
            is_alive = getattr(session.process, "is_alive", lambda: False)
            if bool(ready_event.is_set()):
                return
            if not is_alive():
                raise RuntimeError(
                    f"hierarchical worker {session.match_id} exited before ready"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"hierarchical worker {session.match_id} did not become ready"
                )
            time.sleep(0.01)

    def _shared_memory_epoch(self, spec: Any) -> int:
        from .runtime import Win32SharedMemoryTransportV1

        last_error: BaseException | None = None
        for epoch in range(1, 65):
            transport = None
            try:
                transport = Win32SharedMemoryTransportV1(
                    spec.ipc_name,
                    capacity=64,
                    epoch=epoch,
                    open_existing=True,
                )
                return int(transport.refresh_epoch())
            except (OSError, RuntimeError, ValueError) as error:
                last_error = error
            finally:
                if transport is not None:
                    transport.close()
        raise RuntimeError(
            f"unable to read shared-memory epoch for hierarchical worker {spec.instance_id}"
        ) from last_error

    def _load_source_manifest(self) -> DatasetManifestV1 | None:
        manifest_path = self.config.data_root / "manifests" / "three-demo-pilot-manifest-v1.json"
        if manifest_path.is_file():
            from .hierarchical_demo_training import load_test_only_sequence_manifest

            return load_test_only_sequence_manifest(manifest_path)
        return DatasetManifestV1(
            name="test-only-hierarchical-selfplay",
            purpose=DataPurpose.TEST_ONLY,
            artifact_type="trajectory",
        )


_PROBE_CPU_LIMIT_PERCENT = 95.0
_PROBE_RAM_LIMIT_FRACTION = 0.90
_PROBE_GPU_LIMIT_PERCENT = 95.0
_PROBE_RESOURCE_SAMPLE_INTERVAL_S = 0.25


@dataclass(frozen=True)
class _ProbeResourceSnapshot:
    cpu_percent: float | None
    ram_mb: float | None
    gpu_percent: float | None
    resource_evidence: bool
    resource_error: bool
    ipc_conflict: bool
    memory_error: bool
    gpu_error: bool
    detail: str


class _ProbeResourceMonitor:
    """Collect fail-closed host-resource evidence for a live probe."""

    def __init__(self) -> None:
        self._psutil: Any | None = None
        self._process_ids: set[int] = set()
        self._processes: dict[int, Any] = {}
        self._gpu_reader: Callable[[], float] | None = None
        self._gpu_detail = ""
        self._cpu_max: float | None = None
        self._ram_max_mb: float | None = None
        self._gpu_max: float | None = None
        self._samples = 0
        self._resource_evidence = False
        self._resource_error = False
        self._ipc_conflict = False
        self._memory_error = False
        self._gpu_error = False
        self._details: list[str] = []

    def start(self, runner: Any) -> None:
        try:
            import psutil
        except ImportError as error:
            self._resource_error = True
            self._details.append(f"psutil unavailable: {error}")
            return
        self._psutil = psutil
        self._gpu_reader = self._build_gpu_reader()
        self._refresh_processes(runner)
        if not self._processes:
            self._resource_error = True
            self._details.append("probe has no measurable server or worker process")
            return
        for process in tuple(self._processes.values()):
            try:
                process.cpu_percent(None)
            except (OSError, self._psutil.Error) as error:
                self._resource_error = True
                self._details.append(f"failed to prime process telemetry: {error}")

    def sample(self, runner: Any) -> None:
        self._refresh_processes(runner)
        self._update_ipc_state(runner)
        if self._psutil is None or self._gpu_reader is None:
            return
        if not self._processes:
            self._resource_error = True
            self._details.append("probe process telemetry disappeared")
            return
        try:
            cpu_total = sum(float(process.cpu_percent(None)) for process in self._processes.values())
            cpu_count = max(1, int(self._psutil.cpu_count() or 1))
            cpu_percent = (cpu_total / cpu_count)
            ram_mb = sum(
                float(process.memory_info().rss) for process in self._processes.values()
            ) / (1024.0 * 1024.0)
            total_memory_mb = float(self._psutil.virtual_memory().total) / (1024.0 * 1024.0)
        except (OSError, TypeError, ValueError, self._psutil.Error) as error:
            self._resource_error = True
            self._details.append(f"failed to collect process telemetry: {error}")
            return
        try:
            gpu_percent = float(self._gpu_reader())
        except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
            self._resource_error = True
            self._details.append(f"failed to collect GPU telemetry: {error}")
            return
        self._samples += 1
        self._cpu_max = cpu_percent if self._cpu_max is None else max(self._cpu_max, cpu_percent)
        self._ram_max_mb = ram_mb if self._ram_max_mb is None else max(self._ram_max_mb, ram_mb)
        self._gpu_max = gpu_percent if self._gpu_max is None else max(self._gpu_max, gpu_percent)
        self._resource_evidence = True
        if cpu_percent >= _PROBE_CPU_LIMIT_PERCENT:
            self._resource_error = True
            self._details.append(f"CPU load exceeded {_PROBE_CPU_LIMIT_PERCENT:.1f}%: {cpu_percent:.2f}%")
        if ram_mb >= total_memory_mb * _PROBE_RAM_LIMIT_FRACTION:
            self._memory_error = True
            self._details.append(
                f"RAM load exceeded {_PROBE_RAM_LIMIT_FRACTION:.0%}: "
                f"{ram_mb:.1f}MiB/{total_memory_mb:.1f}MiB"
            )
        if gpu_percent >= _PROBE_GPU_LIMIT_PERCENT:
            self._gpu_error = True
            self._details.append(f"GPU load exceeded {_PROBE_GPU_LIMIT_PERCENT:.1f}%: {gpu_percent:.2f}%")

    def snapshot(self) -> _ProbeResourceSnapshot:
        return _ProbeResourceSnapshot(
            cpu_percent=self._cpu_max,
            ram_mb=self._ram_max_mb,
            gpu_percent=self._gpu_max,
            resource_evidence=self._resource_evidence and self._samples > 0,
            resource_error=self._resource_error,
            ipc_conflict=self._ipc_conflict,
            memory_error=self._memory_error,
            gpu_error=self._gpu_error,
            detail="; ".join(dict.fromkeys([self._gpu_detail, *self._details])).strip("; "),
        )

    def _refresh_processes(self, runner: Any) -> None:
        current_ids: set[int] = set()
        farm = getattr(runner, "farm", None)
        for process in getattr(farm, "processes", {}).values():
            pid = getattr(process, "pid", None)
            if isinstance(pid, int) and pid > 0:
                current_ids.add(pid)
        for session in getattr(runner, "_sessions", {}).values():
            pid = getattr(getattr(session, "process", None), "pid", None)
            if isinstance(pid, int) and pid > 0:
                current_ids.add(pid)
        self._process_ids.update(current_ids)
        if self._psutil is None:
            return
        refreshed: dict[int, Any] = {}
        for pid in sorted(self._process_ids):
            try:
                process = self._psutil.Process(pid)
                refreshed[pid] = process
                for child in process.children(recursive=True):
                    refreshed[int(child.pid)] = child
            except (OSError, self._psutil.Error):
                # Process liveness is reported by the probe itself. Missing
                # telemetry is nevertheless a resource-evidence failure.
                self._resource_error = True
        self._processes = refreshed

    def _update_ipc_state(self, runner: Any) -> None:
        markers = (
            "ipc",
            "shared memory",
            "file mapping",
            "already exists",
            "access is denied",
        )
        for session in getattr(runner, "_sessions", {}).values():
            for message in getattr(session, "error_box", ()):
                text = str(message).lower()
                if any(marker in text for marker in markers):
                    self._ipc_conflict = True

    def _build_gpu_reader(self) -> Callable[[], float] | None:
        for command in ("nvidia-smi", "rocm-smi"):
            executable = shutil.which(command)
            if executable is None:
                continue

            def read_command(executable: str = executable) -> float:
                result = subprocess.run(
                    [executable, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
                    if executable.lower().endswith("nvidia-smi.exe") or executable.lower().endswith("nvidia-smi")
                    else [executable, "--showuse", "--csv"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or f"{executable} exited {result.returncode}")
                values: list[float] = []
                for token in result.stdout.replace(",", " ").replace("%", " ").split():
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                    if 0.0 <= value <= 100.0:
                        values.append(value)
                if not values:
                    raise RuntimeError(f"{executable} returned no GPU utilization")
                return max(values)

            self._gpu_detail = f"GPU telemetry via {command}"
            return read_command

        try:
            import torch

            if not bool(torch.cuda.is_available()):
                self._gpu_detail = "no CUDA/ROCm accelerator reported"
                return lambda: 0.0
        except (ImportError, RuntimeError) as error:
            self._gpu_detail = f"unable to determine accelerator availability: {error}"
            self._resource_error = True
            return None

        def read_torch() -> float:
            try:
                value = float(torch.cuda.utilization())
                self._gpu_detail = "GPU telemetry via torch.cuda.utilization"
                return value
            except (
                AttributeError,
                ImportError,
                ModuleNotFoundError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                # ROCm builds can expose a working device while the optional
                # amd-smi Python binding is unavailable.  Device memory
                # pressure is still a real, locally measured safety signal;
                # report it explicitly instead of treating the GPU as unknown.
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    total = float(total_bytes)
                    if total <= 0.0:
                        raise RuntimeError("GPU memory capacity is zero")
                    self._gpu_detail = (
                        "GPU telemetry via torch.cuda.mem_get_info "
                        f"(memory-pressure fallback; utilization unavailable: {error})"
                    )
                    return max(0.0, min(100.0, (1.0 - float(free_bytes) / total) * 100.0))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as memory_error:
                    raise RuntimeError(
                        f"GPU utilization and memory telemetry unavailable: {error}; {memory_error}"
                    ) from memory_error

        return read_torch


@dataclass(frozen=True)
class _HierarchicalWorkerProbeResult:
    """Full-load evidence collected from one worker process per server."""

    worker_count: int
    process_alive: bool
    rcon_ready: bool
    readback_ok: bool
    map_progress: bool
    tick_backlog: bool
    inference_timeout: bool
    action_loss: bool
    action_p99_ms: float
    deadline_misses: int
    tick_speedup: float
    stable_window_s: float
    tick_counts: Mapping[str, int]
    metric_counts: Mapping[str, int]
    resource_evidence: bool
    resource_error: bool
    ipc_conflict: bool
    memory_error: bool
    gpu_error: bool
    cpu_percent: float | None
    ram_mb: float | None
    gpu_percent: float | None
    detail: str


def _run_hierarchical_worker_probe(
    config: "HierarchicalTestConfigV1",
    baseline: Any,
    specs: Sequence[Any],
    *,
    host_timescale: int,
    stable_window_s: float,
) -> _HierarchicalWorkerProbeResult:
    """Run the exact N-server/N-worker/IPC load used by the formal wave."""

    from .server_farm import _package_context

    selected = tuple(specs)
    if not selected:
        raise ValueError("full-load probe requires at least one server specification")
    if stable_window_s <= 0.0:
        raise ValueError("full-load probe requires a positive stable window")
    context = _package_context(baseline, int(host_timescale), "mr12")
    probe_root = config.data_root / "probes"
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_data_root = Path(
        tempfile.mkdtemp(prefix="hierarchical-live-", dir=str(probe_root))
    )
    runner = _HierarchicalMatchRunner(
        config,
        selected,
        baseline,
        data_root=probe_data_root,
    )
    last_tick: dict[str, int] = {}
    first_tick: dict[str, int] = {}
    tick_counts = {str(spec.instance_id): 0 for spec in selected}
    metric_counts = {str(spec.instance_id): 0 for spec in selected}
    errors: list[str] = []
    tick_backlog = False
    inference_timeout = False
    action_loss = False
    action_p99_ms = 0.0
    deadline_misses = 0
    resource_monitor = _ProbeResourceMonitor()
    try:
        # This starts one srcds, one shared-memory runtime process, and one
        # worker session for every selected server before any measurement.
        runner._start_wave(context)
        readiness = runner.farm.readiness
        rcon_ready = all(bool(item.ready) for item in readiness.values())
        statuses = runner.farm.send_rcon_command("status")
        map_progress = all(bool(str(value).strip()) for value in statuses.values())
        readback = runner.farm.send_rcon_command("host_timescale")
        readback_ok = all(str(host_timescale) in str(value) for value in readback.values())
        resource_monitor.start(runner)
        measurement_started = time.monotonic()
        last_tick_at = {str(spec.instance_id): measurement_started for spec in selected}
        deadline_ms = 7.8125 / max(1, int(host_timescale))
        last_resource_sample_at = measurement_started - _PROBE_RESOURCE_SAMPLE_INTERVAL_S
        while time.monotonic() - measurement_started < float(stable_window_s):
            now = time.monotonic()
            for instance_id, session in tuple(runner._sessions.items()):
                event = session.poll()
                if not isinstance(event, Mapping):
                    continue
                status = str(event.get("status", "")).lower()
                if status in {"crashed", "failed", "aborted"}:
                    errors.append(f"{instance_id}: {event.get('reason', status)}")
                    action_loss = True
                    continue
                if status == "completed":
                    errors.append(f"{instance_id}: worker ended during stable probe")
                    action_loss = True
                    continue
                if status != "live":
                    if now - last_tick_at[instance_id] > max(1.0, 2.0 / max(1, host_timescale)):
                        tick_backlog = True
                    continue
                metric_counts[instance_id] += 1
                server_tick = int(event.get("server_tick", 0))
                if server_tick <= last_tick.get(instance_id, -1):
                    tick_backlog = True
                else:
                    first_tick.setdefault(instance_id, server_tick)
                    last_tick[instance_id] = server_tick
                    tick_counts[instance_id] += 1
                    last_tick_at[instance_id] = now
                # Session metric coalescing is normal at a 10 ms parent poll
                # interval.  Only a queue-level backlog (measured by the
                # session before draining) is a capacity failure.
                if int(event.get("metric_backlog_events", 0)) > 0:
                    tick_backlog = True
                action_p99_ms = max(
                    action_p99_ms,
                    float(event.get("runtime_action_p99_ms", 0.0)),
                )
                deadline_misses = max(
                    deadline_misses,
                    int(event.get("runtime_total_tick_deadline_miss", 0)),
                )
                if bool(event.get("runtime_process_failed", False)):
                    action_loss = True
                if int(event.get("fallback_count", 0)) > 0:
                    action_loss = True
            if any(
                now - last_tick_at[instance_id] > max(1.0, 2.0 / max(1, host_timescale))
                for instance_id in last_tick_at
            ):
                tick_backlog = True
            if action_p99_ms >= deadline_ms or deadline_misses > 0:
                inference_timeout = True
            if now - last_resource_sample_at >= _PROBE_RESOURCE_SAMPLE_INTERVAL_S:
                resource_monitor.sample(runner)
                last_resource_sample_at = now
            time.sleep(0.01)
        resource_monitor.sample(runner)
        resource_snapshot = resource_monitor.snapshot()
        process_alive = all(
            getattr(process, "poll", lambda: 1)() is None
            for process in runner.farm.processes.values()
        ) and all(
            bool(getattr(session.process, "is_alive", lambda: False)())
            for session in runner._sessions.values()
        )
        if any(count <= 0 for count in metric_counts.values()):
            action_loss = True
        action_ack_counts = {
            instance_id: int(session.action_ack_count)
            for instance_id, session in runner._sessions.items()
        }
        action_sample_counts = {
            instance_id: int(session.metric_samples_with_action)
            for instance_id, session in runner._sessions.items()
        }
        if any(
            action_sample_counts.get(instance_id, 0) <= 0
            or action_ack_counts.get(instance_id, 0)
            / max(1, action_sample_counts.get(instance_id, 0))
            < 0.95
            for instance_id in tick_counts
        ):
            action_loss = True
        elapsed = max(0.0, time.monotonic() - measurement_started)
        speedups = [
            ((last_tick[instance_id] - first_tick[instance_id]) / max(elapsed, 1e-6)) / 128.0
            for instance_id in tick_counts
            if instance_id in last_tick and instance_id in first_tick
        ]
        tick_speedup = min(speedups) if speedups else 0.0
        detail = (
            f"full-load workers={len(selected)} ticks={dict(tick_counts)} "
            f"metrics={dict(metric_counts)} action_p99_ms={action_p99_ms:.4f} "
            f"deadline_misses={deadline_misses} action_ack={action_ack_counts}/"
            f"{action_sample_counts} resources="
            f"cpu={resource_snapshot.cpu_percent} ram_mb={resource_snapshot.ram_mb} "
            f"gpu={resource_snapshot.gpu_percent} resource_error={resource_snapshot.resource_error} "
            f"ipc_conflict={resource_snapshot.ipc_conflict} errors={errors}"
        )
        if resource_snapshot.detail:
            detail = f"{detail} resource_detail={resource_snapshot.detail}"
        return _HierarchicalWorkerProbeResult(
            worker_count=len(selected),
            process_alive=process_alive,
            rcon_ready=rcon_ready,
            readback_ok=readback_ok,
            map_progress=map_progress,
            tick_backlog=tick_backlog,
            inference_timeout=inference_timeout,
            action_loss=action_loss,
            action_p99_ms=action_p99_ms,
            deadline_misses=deadline_misses,
            tick_speedup=tick_speedup,
            stable_window_s=elapsed,
            tick_counts=tick_counts,
            metric_counts=metric_counts,
            resource_evidence=resource_snapshot.resource_evidence,
            resource_error=resource_snapshot.resource_error,
            ipc_conflict=resource_snapshot.ipc_conflict,
            memory_error=resource_snapshot.memory_error,
            gpu_error=resource_snapshot.gpu_error,
            cpu_percent=resource_snapshot.cpu_percent,
            ram_mb=resource_snapshot.ram_mb,
            gpu_percent=resource_snapshot.gpu_percent,
            detail=detail,
        )
    finally:
        runner.close()
        shutil.rmtree(probe_data_root, ignore_errors=True)



def _reject_duration_config(value: object, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if "duration" in key_text or key_text in {
                "selfplay_seconds",
                "self_play_seconds",
                "training_seconds",
            }:
                raise ValueError(
                    f"test-only configuration cannot define self-play duration: {path}.{key}"
                )
            _reject_duration_config(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_duration_config(child, f"{path}[{index}]")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in _FORBIDDEN_DURATION_VALUES:
            raise ValueError(
                f"test-only configuration cannot contain fixed duration value {value}"
            )


def _value(mapping: Mapping[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _invoke_pipeline_callback(callback: Callable[..., Any], *args: Any) -> Any:
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


@dataclass(frozen=True)
class HierarchicalTestConfigV1:
    demo_root: Path
    data_root: Path
    server_root: Path
    map_name: Literal["de_mirage"] = "de_mirage"
    max_server_probe: int = 32
    timescale_candidates: tuple[int, ...] = _ALLOWED_TIMESCALES
    probe_stable_window_s: float = 20.0
    probe_max_age_s: float = _DEFAULT_PROBE_MAX_AGE_S
    seed: int = 7
    training_device: str = "cuda"
    demo_sequence_length: int = 128
    demo_batch_sequences: int = 8
    demo_updates_per_demo: int | None = None
    heldout_demo: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "demo_root", Path(self.demo_root))
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "server_root", Path(self.server_root))
        if self.map_name != "de_mirage":
            raise ValueError("hierarchical test pipeline only supports de_mirage")
        if not isinstance(self.max_server_probe, int) or isinstance(
            self.max_server_probe, bool
        ):
            raise TypeError("max_server_probe must be an integer")
        if not 1 <= self.max_server_probe <= 32:
            raise ValueError("max_server_probe must be between 1 and 32")
        candidates = tuple(int(value) for value in self.timescale_candidates)
        if not candidates or any(value not in _ALLOWED_TIMESCALES for value in candidates):
            raise ValueError("timescale_candidates must use only 1, 2, 4 and 8")
        if candidates[0] != 1:
            raise ValueError("timescale_candidates must start at 1x")
        object.__setattr__(self, "timescale_candidates", candidates)
        if not math.isfinite(float(self.probe_stable_window_s)) or self.probe_stable_window_s < 0.0:
            raise ValueError("probe_stable_window_s must be finite and non-negative")
        if not math.isfinite(float(self.probe_max_age_s)) or self.probe_max_age_s <= 0.0:
            raise ValueError("probe_max_age_s must be finite and positive")
        if self.heldout_demo is not None and not str(self.heldout_demo).strip():
            raise ValueError("heldout_demo must be non-empty when specified")
        if self.heldout_demo is not None:
            object.__setattr__(self, "heldout_demo", str(self.heldout_demo))

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        demo_root: str | Path | None = None,
        data_root: str | Path | None = None,
        server_root: str | Path | None = None,
    ) -> "HierarchicalTestConfigV1":
        config_path = Path(path)
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except OSError as error:
            raise FileNotFoundError(config_path) from error
        if not isinstance(payload, Mapping):
            raise ValueError("hierarchical test config must be a mapping")
        _reject_duration_config(payload)
        purpose = ensure_purpose(str(payload.get("purpose", DataPurpose.TEST_ONLY.value)))
        require_test_only(purpose)

        server = payload.get("server", {})
        if not isinstance(server, Mapping):
            raise ValueError("server config must be a mapping")
        timescale = payload.get("timescale", {})
        if not isinstance(timescale, Mapping):
            raise ValueError("timescale config must be a mapping")
        training = payload.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("training config must be a mapping")
        probe = payload.get("probe", {})
        if not isinstance(probe, Mapping):
            raise ValueError("probe config must be a mapping")
        raw_candidates = _value(
            payload,
            "timescale_candidates",
            default=_value(timescale, "candidates", default=_ALLOWED_TIMESCALES),
        )
        if isinstance(raw_candidates, (str, bytes)):
            raise ValueError("timescale_candidates must be a sequence")

        resolved_demo_root = demo_root or payload.get("demo_root")
        resolved_data_root = data_root or payload.get("data_root")
        resolved_server_root = server_root or payload.get("server_root")
        if resolved_demo_root is None or resolved_data_root is None or resolved_server_root is None:
            raise ValueError("demo_root, data_root and server_root are required")
        return cls(
            demo_root=resolved_demo_root,
            data_root=resolved_data_root,
            server_root=resolved_server_root,
            map_name=str(payload.get("map_name", payload.get("map", "de_mirage"))),
            max_server_probe=int(
                _value(
                    payload,
                    "max_server_probe",
                    default=_value(server, "max_server_probe", default=32),
                )
            ),
            timescale_candidates=tuple(int(value) for value in raw_candidates),
            probe_stable_window_s=float(
                _value(
                    payload,
                    "probe_stable_window_s",
                    default=_value(probe, "stable_window_seconds", default=20.0),
                )
            ),
            probe_max_age_s=float(
                _value(
                    payload,
                    "probe_max_age_s",
                    default=_value(probe, "max_age_seconds", default=_DEFAULT_PROBE_MAX_AGE_S),
                )
            ),
            seed=int(payload.get("seed", 7)),
            training_device=str(
                _value(
                    payload,
                    "training_device",
                    "device",
                    default=_value(training, "device", default="cuda"),
                )
            ),
            demo_sequence_length=int(
                _value(training, "sequence_length", default=128)
            ),
            demo_batch_sequences=int(
                _value(training, "batch_sequences", default=8)
            ),
            demo_updates_per_demo=(
                None
                if _value(training, "updates_per_demo", default=None) is None
                else int(_value(training, "updates_per_demo", default=None))
            ),
            heldout_demo=(
                None
                if _value(payload, "heldout_demo", default=_value(training, "heldout_demo", default=None)) is None
                else str(_value(payload, "heldout_demo", default=_value(training, "heldout_demo", default=None)))
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "purpose": DataPurpose.TEST_ONLY.value,
            "demo_root": str(self.demo_root),
            "data_root": str(self.data_root),
            "server_root": str(self.server_root),
            "map_name": self.map_name,
            "max_server_probe": self.max_server_probe,
            "timescale_candidates": list(self.timescale_candidates),
            "probe_stable_window_s": self.probe_stable_window_s,
            "probe_max_age_s": self.probe_max_age_s,
            "seed": self.seed,
            "training_device": self.training_device,
            "training": {
                "sequence_length": self.demo_sequence_length,
                "batch_sequences": self.demo_batch_sequences,
                "updates_per_demo": self.demo_updates_per_demo,
                "heldout_demo": self.heldout_demo,
            },
        }


def _probe_identity(
    config: HierarchicalTestConfigV1,
    baseline: Any,
    *,
    probed_at_unix_s: float | None = None,
) -> dict[str, Any]:
    """Bind persisted probe selections to the exact model/config/host."""

    from .server_farm import _package_sha256

    executable = (Path(config.server_root) / "srcds.exe").resolve(strict=False)
    executable_sha256 = ""
    executable_size = 0
    executable_mtime_ns = 0
    if executable.is_file():
        digest = hashlib.sha256()
        with executable.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        executable_sha256 = digest.hexdigest()
        stat = executable.stat()
        executable_size = int(stat.st_size)
        executable_mtime_ns = int(stat.st_mtime_ns)
    hardware_payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    hardware_fingerprint = hashlib.sha256(
        json.dumps(hardware_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config_sha256 = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    core = {
        "package_sha256": _package_sha256(baseline),
        "decision_sha256": str(getattr(baseline, "decision_sha256", "")),
        "action_sha256": str(getattr(baseline, "action_sha256", "")),
        "policy_generation": int(getattr(baseline, "generation", 0)),
        "config_sha256": config_sha256,
        "server_executable_sha256": executable_sha256,
        "server_executable_size": executable_size,
        "server_executable_mtime_ns": executable_mtime_ns,
        "hardware_fingerprint": hardware_fingerprint,
    }
    fingerprint = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        **core,
        "fingerprint": fingerprint,
        "probed_at_unix_s": float(time.time() if probed_at_unix_s is None else probed_at_unix_s),
    }


def _probe_identity_matches(
    stored: Any,
    expected: Mapping[str, Any],
    *,
    max_age_s: float,
) -> bool:
    if not isinstance(stored, Mapping):
        return False
    for name in (
        "fingerprint",
        "package_sha256",
        "decision_sha256",
        "action_sha256",
        "config_sha256",
        "server_executable_sha256",
        "hardware_fingerprint",
    ):
        if str(stored.get(name, "")) != str(expected.get(name, "")):
            return False
    try:
        probed_at = float(stored["probed_at_unix_s"])
    except (KeyError, TypeError, ValueError):
        return False
    now = time.time()
    return math.isfinite(probed_at) and -300.0 <= now - probed_at <= float(max_age_s)


def _run_live_test_probes(
    config: HierarchicalTestConfigV1,
    baseline: Any,
) -> tuple[Any, Any]:
    """Run real server/RCON probes and persist the bounded selections.

    The probe deliberately treats an unmeasured model deadline, timescale
    speedup, or host-resource sample as unstable. This makes the default
    conservative while still exercising the same server, RCON, shared-memory,
    worker and map-progress boundary that the wave will use. A caller may
    provide a stricter injected probe runner for a hardware-specific latency
    measurement.
    """

    from .server_farm import (
        CapacityProbeSampleV1,
        TimescaleProbeSampleV1,
        build_server_specs,
        probe_capacity,
        probe_timescale,
    )

    probe_root = config.data_root / "probes"
    run_id = "test-pipeline-probe"
    probe_identity = _probe_identity(config, baseline)

    def capacity_runner(count: int) -> CapacityProbeSampleV1:
        specs = build_server_specs(count, run_id, config.server_root, config.data_root)
        try:
            result = _run_hierarchical_worker_probe(
                config,
                baseline,
                specs,
                host_timescale=1,
                stable_window_s=config.probe_stable_window_s,
            )
            return CapacityProbeSampleV1(
                requested_instances=count,
                process_alive=result.process_alive,
                rcon_ready=result.rcon_ready,
                map_progress=result.map_progress,
                tick_backlog=result.tick_backlog,
                stable_window_s=result.stable_window_s,
                inference_timeout=result.inference_timeout,
                action_loss=result.action_loss,
                ipc_conflict=result.ipc_conflict,
                memory_error=result.memory_error,
                gpu_error=result.gpu_error,
                cpu_percent=result.cpu_percent,
                ram_mb=result.ram_mb,
                gpu_percent=result.gpu_percent,
                resource_evidence=result.resource_evidence,
                resource_error=result.resource_error,
                detail=result.detail,
            )
        except Exception as error:
            error_text = str(error).lower()
            return CapacityProbeSampleV1(
                requested_instances=count,
                ipc_conflict=any(
                    marker in error_text
                    for marker in ("ipc", "shared memory", "file mapping", "already exists")
                ),
                resource_error=True,
                detail=f"live probe failed: {type(error).__name__}: {error}",
            )

    capacity = probe_capacity(
        config.max_server_probe,
        probe_runner=capacity_runner,
        stable_window_s=config.probe_stable_window_s,
        report_path=probe_root / "capacity.json",
        probe_identity=probe_identity,
    )
    selected_specs = build_server_specs(
        capacity.selected_instances,
        run_id,
        config.server_root,
        config.data_root,
    )

    def timescale_runner(timescale: int, specs: tuple[Any, ...]) -> TimescaleProbeSampleV1:
        try:
            result = _run_hierarchical_worker_probe(
                config,
                baseline,
                specs,
                host_timescale=timescale,
                stable_window_s=config.probe_stable_window_s,
            )
            return TimescaleProbeSampleV1(
                timescale=timescale,
                supported=result.readback_ok,
                readback_ok=result.readback_ok,
                tick_speedup=result.tick_speedup,
                action_p99_ms=result.action_p99_ms,
                deadline_misses=result.deadline_misses,
                process_alive=result.process_alive,
                rcon_ready=result.rcon_ready,
                map_progress=result.map_progress,
                stable_window_s=result.stable_window_s,
                action_loss=result.action_loss,
                inference_timeout=result.inference_timeout,
                ipc_conflict=result.ipc_conflict,
                memory_error=result.memory_error,
                gpu_error=result.gpu_error,
                cpu_percent=result.cpu_percent,
                ram_mb=result.ram_mb,
                gpu_percent=result.gpu_percent,
                resource_evidence=result.resource_evidence,
                resource_error=result.resource_error,
                detail=result.detail,
            )
        except Exception as error:
            error_text = str(error).lower()
            return TimescaleProbeSampleV1(
                timescale=timescale,
                supported=False,
                readback_ok=False,
                tick_speedup=0.0,
                action_p99_ms=1_000_000_000.0,
                deadline_misses=1,
                ipc_conflict=any(
                    marker in error_text
                    for marker in ("ipc", "shared memory", "file mapping", "already exists")
                ),
                resource_error=True,
                detail=f"live probe failed: {type(error).__name__}: {error}",
            )

    timescale = probe_timescale(
        selected_specs,
        config.timescale_candidates,
        probe_runner=timescale_runner,
        stable_window_s=config.probe_stable_window_s,
        report_path=probe_root / "timescale.json",
        probe_identity=probe_identity,
    )
    return capacity, timescale


class TestTrainingPipeline:
    """Entry boundary for the independent, test-only training pipeline."""

    __test__ = False

    def __init__(
        self,
        config: HierarchicalTestConfigV1,
        *,
        purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
        demo_stage_runner: Callable[..., Any] | None = None,
        selected_specs: Sequence[Any] = (),
        selected_timescale: int = 1,
        validation_manifest: Any | None = None,
        wave_collector: Callable[..., Any] | None = None,
        mappo_update: Callable[..., Any] | None = None,
        candidate_exporter: Callable[..., Any] | None = None,
        probe_stage_runner: Callable[..., Any] | None = None,
        gate_evidence_builder: Callable[..., Any] | None = None,
        finalizer: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(config, HierarchicalTestConfigV1):
            raise TypeError("config must be HierarchicalTestConfigV1")
        self.config = config
        self.purpose = require_test_only(purpose)
        if not isinstance(selected_timescale, int) or isinstance(selected_timescale, bool):
            raise TypeError("selected_timescale must be an integer")
        if selected_timescale < 1:
            raise ValueError("selected_timescale must be positive")
        self.demo_stage_runner = demo_stage_runner
        self.wave_collector = wave_collector
        self.mappo_update = mappo_update
        self.candidate_exporter = candidate_exporter
        self.probe_stage_runner = probe_stage_runner
        self.gate_evidence_builder = gate_evidence_builder
        self.finalizer = finalizer
        self.last_wave: CompleteMatchWaveV1 | None = None
        self.active_package: Any | None = None
        self.candidate_package: Any | None = None
        self.selected_specs: tuple[Any, ...] = tuple(selected_specs)
        self.selected_timescale = selected_timescale
        self.validation_manifest = validation_manifest
        self.demo_stage_report: Any | None = None
        self.update_count = 0
        self.candidate_export_count = 0
        self._single_wave_executed = False

    def run_probe_stage(self, baseline: Any) -> tuple[Any, Any]:
        """Bind capacity/timescale selected by persisted or live probes."""

        if self.probe_stage_runner is not None:
            result = _invoke_pipeline_callback(self.probe_stage_runner, self.config, baseline)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("probe stage must return (capacity_report, timescale_report)")
            capacity, timescale = result
            selected_instances = int(getattr(capacity, "selected_instances"))
            selected_timescale = int(getattr(timescale, "selected_timescale"))
        else:
            persisted = _load_persisted_probe_selection(self.config, baseline)
            if persisted is None:
                capacity, timescale = _run_live_test_probes(self.config, baseline)
                selected_instances = int(capacity.selected_instances)
                selected_timescale = int(timescale.selected_timescale)
            else:
                capacity = SimpleNamespace(selected_instances=persisted[0])
                timescale = SimpleNamespace(selected_timescale=persisted[1])
                selected_instances, selected_timescale = persisted
        if not 1 <= selected_instances <= self.config.max_server_probe:
            raise ValueError("probe stage selected an invalid server count")
        if selected_timescale not in self.config.timescale_candidates:
            raise ValueError("probe stage selected an invalid host timescale")
        from .server_farm import build_server_specs

        self.selected_specs = build_server_specs(
            selected_instances,
            "test-pipeline",
            self.config.server_root,
            self.config.data_root,
        )
        self.selected_timescale = selected_timescale
        return capacity, timescale

    def finalize(self) -> None:
        """Stop default farm/workers without changing model or dataset artifacts."""

        if self.finalizer is not None:
            self.finalizer()

    def child_manifest(
        self,
        name: str,
        *,
        parents: Sequence[DatasetManifestV1] = (),
        artifact_type: str = "test_only_artifact",
        metadata: Mapping[str, str] | None = None,
    ) -> DatasetManifestV1:
        manifest = DatasetManifestV1(
            name=name,
            purpose=self.purpose,
            parents=tuple(parents),
            artifact_type=artifact_type,
            metadata=dict(metadata or {}),
        )
        require_test_only(manifest.effective_purpose())
        return manifest

    def run_demo_stage(self) -> object:
        if self.demo_stage_runner is None:
            generation_manifest = (
                self.config.data_root
                / "models"
                / "generation-000"
                / "package-v2.json"
            )
            if generation_manifest.is_file():
                # The three-Demo pilot is deliberately idempotent.  Re-running the
                # entry point must not create a second training wave or overwrite a
                # verified generation just because the previous run reached a later
                # stage and was interrupted.
                from types import SimpleNamespace

                from .export import load_hierarchical_package

                baseline = load_hierarchical_package(generation_manifest)
                if baseline.generation != 0:
                    raise ValueError(
                        "test-only Demo baseline must be generation-000: "
                        f"{generation_manifest}"
                    )
                self.demo_stage_report = SimpleNamespace(
                    package=baseline,
                    reused=True,
                    package_manifest=generation_manifest,
                )
            else:
                from .hierarchical_demo_training import train_test_only_demo_stage

                self.demo_stage_report = train_test_only_demo_stage(
                    self.config,
                    purpose=self.purpose,
                    sequence_length=self.config.demo_sequence_length,
                    batch_sequences=self.config.demo_batch_sequences,
                    updates_per_demo=self.config.demo_updates_per_demo,
                )
                baseline = self.demo_stage_report.package
        else:
            baseline = _invoke_pipeline_callback(
                self.demo_stage_runner,
                self.config,
                self.purpose,
            )
        if self.gate_evidence_builder is not None:
            if _package_has_gate_evidence(baseline):
                package_metrics = getattr(baseline, "metadata", {}).get("metrics", {})
                evidence = {
                    "validation_quantiles": dict(package_metrics["validation_quantiles"]),
                }
            else:
                evidence = _invoke_pipeline_callback(self.gate_evidence_builder, baseline, None)
            if not isinstance(evidence, Mapping):
                raise TypeError("held-out human gate evidence builder must return a mapping")
            quantiles = evidence.get("validation_quantiles")
            if isinstance(self.validation_manifest, Mapping) and isinstance(quantiles, Mapping):
                self.validation_manifest = {
                    **dict(self.validation_manifest),
                    "quantiles": dict(quantiles),
                }
            elif isinstance(quantiles, Mapping) and self.validation_manifest is not None:
                try:
                    setattr(self.validation_manifest, "quantiles", dict(quantiles))
                except (AttributeError, TypeError):
                    pass
            baseline = _package_with_gate_evidence(baseline, evidence)
        self.active_package = baseline
        return baseline

    def run_single_selfplay_wave(
        self,
        baseline: object,
        *,
        specs: Sequence[Any] = (),
        host_timescale: int = 1,
        ruleset: str = "mr12",
        match_runner: Callable[..., Any] | None = None,
        restart_instance: Callable[..., Any] | None = None,
        watchdog_seconds: float = 120.0,
        wall_clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        wave_collector: Callable[..., Any] | None = None,
        mappo_update: Callable[..., Any] | None = None,
        candidate_exporter: Callable[..., Any] | None = None,
    ) -> object:
        """Run exactly one complete test-only wave and leave active on the baseline.

        A successful result is a pending candidate returned by ``candidate_exporter``;
        an aborted wave returns the Demo-only baseline and never calls either training
        or export callback.
        """

        if self._single_wave_executed:
            raise RuntimeError("single match wave already executed")
        self._single_wave_executed = True
        self.active_package = baseline
        collector = wave_collector or self.wave_collector or collect_single_match_wave
        updater = mappo_update or self.mappo_update
        exporter = candidate_exporter or self.candidate_exporter
        if updater is None or exporter is None:
            raise RuntimeError("single self-play wave requires one MAPPO updater and one candidate exporter")
        selected_specs = tuple(specs) or self.selected_specs
        if collector is collect_single_match_wave:
            if not selected_specs:
                raise ValueError("single self-play wave requires selected server specifications")
            result = collector(
                selected_specs,
                baseline,
                host_timescale=host_timescale,
                ruleset=ruleset,
                match_runner=match_runner,
                restart_instance=restart_instance,
                watchdog_seconds=watchdog_seconds,
                wall_clock=wall_clock,
                sleep=sleep,
            )
        else:
            result = _invoke_pipeline_callback(collector, selected_specs, baseline, host_timescale)
        if not isinstance(result, CompleteMatchWaveV1):
            raise TypeError("single self-play collector must return CompleteMatchWaveV1")
        self.last_wave = result
        if not result.complete:
            return baseline
        update_result = _invoke_pipeline_callback(updater, result.shards)
        self.update_count = 1
        candidate = _invoke_pipeline_callback(exporter, update_result, result)
        self.candidate_export_count = 1
        self.candidate_package = candidate
        return candidate

    def run_human_gate(
        self,
        baseline: object,
        candidate: object,
        validation_manifest: object,
        *,
        pending_pointer: str | Path | None = None,
        active_pointer: str | Path | None = None,
    ) -> object:
        from .human_gate import evaluate_human_gate

        return evaluate_human_gate(
            baseline,
            candidate,
            validation_manifest,
            pending_pointer=pending_pointer,
            active_pointer=active_pointer,
        )

    def select_pointers(
        self,
        baseline: object,
        candidate: object,
        gate: object,
    ) -> dict[str, str | None]:
        """Keep the Demo-only active pointer and publish only an accepted pending package."""

        active_pointer = self.config.data_root / "models" / "active.json"
        pending_pointer = self.config.data_root / "models" / "pending.json"
        from .contracts import HierarchicalPackageV2
        from .export import publish_hierarchical_package

        if isinstance(baseline, HierarchicalPackageV2):
            publish_hierarchical_package(baseline, active_pointer, slot="active")
        accepted = bool(getattr(gate, "accepted", False))
        if accepted:
            if not isinstance(candidate, HierarchicalPackageV2):
                raise TypeError("accepted test candidate must be a HierarchicalPackageV2")
            publish_hierarchical_package(candidate, pending_pointer, slot="pending")
        return {
            "active": str(active_pointer) if baseline is not None else None,
            "pending": str(pending_pointer) if accepted else None,
            "status": "accepted-test-only" if accepted else "rejected-test-only",
        }


def _load_test_validation_manifest(data_root: Path) -> Any:
    """Load explicit validation evidence without inventing a passing gate."""

    candidates = (
        data_root / "manifests" / "human-validation-v1.json",
        data_root / "manifests" / "validation-manifest-v1.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"test validation manifest is not valid JSON: {path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"test validation manifest must be a mapping: {path}")
        return dict(payload)
    # Missing evidence is deliberately represented, not silently replaced with
    # permissive quantiles.  human_gate.evaluate_human_gate will reject it.
    return SimpleNamespace(purpose=DataPurpose.TEST_ONLY, quantiles={})


def _load_persisted_probe_selection(
    config: HierarchicalTestConfigV1,
    baseline: Any | None = None,
) -> tuple[int, int] | None:
    """Read reports only when they are bound to the current baseline and host."""

    capacity_path = config.data_root / "probes" / "capacity.json"
    timescale_path = config.data_root / "probes" / "timescale.json"
    if not capacity_path.is_file() or not timescale_path.is_file():
        return None
    if baseline is None:
        return None
    payloads: list[Mapping[str, Any]] = []
    for path in (capacity_path, timescale_path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"probe report is not valid JSON: {path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"probe report must be a mapping: {path}")
        payloads.append(payload)
    expected_identity = _probe_identity(config, baseline)
    if any(
        not _probe_identity_matches(
            payload.get("probe_identity"),
            expected_identity,
            max_age_s=config.probe_max_age_s,
        )
        for payload in payloads
    ):
        # An old or unbound report must cause a fresh full-load probe rather
        # than silently selecting stale capacity/timescale values.
        return None
    capacity_samples = payloads[0].get("samples")
    timescale_samples = payloads[1].get("samples")
    if not isinstance(capacity_samples, list) or not isinstance(timescale_samples, list):
        raise ValueError("probe reports must contain sample lists")
    def sample_value(sample: Mapping[str, Any], key: str, default: int = -1) -> int:
        try:
            return int(sample.get(key, default))
        except (TypeError, ValueError):
            return default

    def sample_float(sample: Mapping[str, Any], key: str) -> float | None:
        try:
            value = float(sample[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def sample_resource_valid(sample: object) -> bool:
        if not isinstance(sample, Mapping):
            return False
        return _probe_resource_evidence_is_valid(
            bool(sample.get("resource_evidence", False)),
            bool(sample.get("resource_error", True)),
            sample_float(sample, "cpu_percent"),
            sample_float(sample, "ram_mb"),
            sample_float(sample, "gpu_percent"),
        )

    capacity_evidence = next(
        (
            sample
            for sample in capacity_samples
            if isinstance(sample, Mapping)
            and sample_value(sample, "requested_instances")
            == sample_value(payloads[0], "selected_instances")
        ),
        None,
    )
    if (
        not isinstance(capacity_evidence, Mapping)
        or not bool(capacity_evidence.get("stable", False))
        or float(capacity_evidence.get("stable_window_s", 0.0))
        < float(config.probe_stable_window_s)
        or not sample_resource_valid(capacity_evidence)
    ):
        raise ValueError("capacity probe report lacks a stable sample for its selected instance count")
    timescale_evidence = next(
        (
            sample
            for sample in timescale_samples
            if isinstance(sample, Mapping)
            and sample_value(sample, "timescale")
            == sample_value(payloads[1], "selected_timescale")
        ),
        None,
    )
    if (
        not isinstance(timescale_evidence, Mapping)
        or not bool(timescale_evidence.get("stable", False))
        or float(timescale_evidence.get("stable_window_s", 0.0))
        < float(config.probe_stable_window_s)
        or not sample_resource_valid(timescale_evidence)
    ):
        raise ValueError("timescale probe report lacks a stable sample for its selected timescale")
    try:
        selected_instances = int(payloads[0]["selected_instances"])
        selected_timescale = int(payloads[1]["selected_timescale"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("probe reports must declare selected_instances and selected_timescale") from error
    if not 1 <= selected_instances <= config.max_server_probe:
        raise ValueError("persisted capacity selection is outside the configured probe bound")
    if selected_timescale not in config.timescale_candidates:
        raise ValueError("persisted timescale selection is not one of the configured candidates")
    return selected_instances, selected_timescale


def _fallback_hierarchical_checkpoint(package: Any, data_root: Path) -> Path:
    metadata = getattr(package, "metadata", {})
    training_run = metadata.get("training_run", {}) if isinstance(metadata, Mapping) else {}
    declared = training_run.get("checkpoint_path") if isinstance(training_run, Mapping) else None
    candidates: list[Path] = []
    if declared:
        candidates.append(Path(str(declared)))
    generation = int(getattr(package, "generation", 0))
    candidates.append(data_root / "models" / f"generation-{generation:03d}" / "hierarchical-demo.pt")
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (data_root / candidate)
        if resolved.is_file():
            return resolved.resolve()
    raise FileNotFoundError(
        "baseline hierarchical checkpoint is missing; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_hierarchical_actor(package: Any, data_root: Path) -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("PyTorch is required for the test-only MAPPO updater") from error
    from .hierarchical_model import HierarchicalActor
    from .quantization import prepare_action_fp32, prepare_decision_qat

    checkpoint_path = _fallback_hierarchical_checkpoint(package, data_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model_state_dict"), Mapping
    ):
        raise ValueError(f"hierarchical checkpoint lacks model_state_dict: {checkpoint_path}")
    actor = HierarchicalActor()
    prepare_decision_qat(actor.decision)
    prepare_action_fp32(actor.action)
    actor.load_state_dict(checkpoint["model_state_dict"])
    return actor


def _validation_sequence_paths(source: Any) -> tuple[Path, ...]:
    """Resolve explicit held-out sequence shards from the validation manifest."""

    raw: Any = None
    if isinstance(source, DatasetManifestV1):
        from .training_dataset import sequence_paths

        raw = sequence_paths(source, "validation") or sequence_paths(source, "test")
    elif isinstance(source, Mapping):
        for key in ("sequence_paths", "validation_paths", "paths"):
            if key in source:
                raw = source[key]
                break
        if raw is None and isinstance(source.get("metadata"), Mapping):
            metadata = source["metadata"]
            for key in ("sequence_paths_json", "validation_paths_json", "paths_json"):
                if key in metadata:
                    raw = metadata[key]
                    break
    else:
        for key in ("sequence_paths", "validation_paths", "paths"):
            value = getattr(source, key, None)
            if value is not None:
                raw = value
                break
    if raw is None:
        return ()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("held-out validation sequence paths are not valid JSON") from error
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ValueError("held-out validation sequence paths must be a sequence")
    paths = tuple(Path(str(value)).resolve() for value in raw)
    if not paths:
        return ()
    if len(set(paths)) != len(paths):
        raise ValueError("held-out validation sequence paths must be unique")
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    return paths


def _package_heldout_validation_paths(package: Any) -> tuple[Path, ...]:
    """Read the split recorded by the Demo training run, never infer it from rollout data."""

    metadata = getattr(package, "metadata", {})
    metrics = metadata.get("metrics", {}) if isinstance(metadata, Mapping) else {}
    raw: Any = metrics.get("heldout_validation_paths") if isinstance(metrics, Mapping) else None
    if raw is None and isinstance(metrics, Mapping):
        demo_training = metrics.get("demo_training")
        if isinstance(demo_training, Mapping):
            raw = demo_training.get("heldout_validation_paths")
    if raw is None:
        return ()
    return _validation_sequence_paths({"sequence_paths": raw})


def _package_has_gate_evidence(package: Any) -> bool:
    from .human_gate import REQUIRED_HUMAN_METRICS

    metadata = getattr(package, "metadata", {})
    metrics = metadata.get("metrics", {}) if isinstance(metadata, Mapping) else {}
    human_metrics = metrics.get("human_metrics") if isinstance(metrics, Mapping) else None
    quantiles = metrics.get("validation_quantiles") if isinstance(metrics, Mapping) else None
    return (
        isinstance(metrics, Mapping)
        and "intent_nll" in metrics
        and isinstance(human_metrics, Mapping)
        and set(human_metrics) == set(REQUIRED_HUMAN_METRICS)
        and isinstance(quantiles, Mapping)
        and set(quantiles) == set(REQUIRED_HUMAN_METRICS)
    )


def _package_with_gate_evidence(package: Any, evidence: Mapping[str, Any]) -> Any:
    """Attach immutable held-out evidence to a package and its manifest file."""

    if not hasattr(package, "metadata") or not hasattr(package, "to_dict"):
        raise TypeError("human gate evidence requires a hierarchical package")
    metadata = dict(getattr(package, "metadata", {}))
    metrics = dict(metadata.get("metrics", {})) if isinstance(metadata.get("metrics"), Mapping) else {}
    metrics.update(dict(evidence))
    metadata["metrics"] = metrics
    enriched = replace(package, metadata=metadata)
    manifest_path = Path(enriched.decision_path).with_name("package-v2.json")
    manifest_path.write_text(
        json.dumps(enriched.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return enriched


def _evaluate_heldout_demo_evidence(
    actor: Any,
    paths: Sequence[Path],
    *,
    device: str,
    seed: int = 7,
    max_paths: int = 64,
) -> dict[str, Any]:
    """Teacher-force one actor on held-out Demo shards for both gate metrics."""

    if not paths:
        raise ValueError("human gate evidence requires explicit held-out Demo sequence paths")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("PyTorch is required for held-out human gate evidence") from error

    from .action_reconstruction import IN_ATTACK, IN_ATTACK2
    from .hierarchical_demo_training import recurrent_to_hierarchical_batch
    from .observation_projection import decode_observation
    from .human_gate import REQUIRED_HUMAN_METRICS
    from .training_dataset import iter_recurrent_minibatches
    from .training_gail import compute_human_like_metrics
    from .training_hierarchical import (
        _build_predicted_intent_sequence,
        _sequence_target_at,
    )
    from .contracts import BotObservationV1, INTENT_FIELD_NAMES
    from .intent import guard_action, guard_friendly_fire
    from .runtime import HierarchicalRuntimeService

    resolved_device = torch.device(device)
    actor = actor.to(resolved_device)
    actor.eval()
    selected_paths = tuple(Path(path).resolve() for path in paths[:max_paths])
    if not selected_paths:
        raise ValueError("held-out Demo validation path selection is empty")
    model_records_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    human_records_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    decision_loss_sum = 0.0
    decision_weight_sum = 0.0
    window_count = 0
    throwable_weapon_names = {
        "he grenade",
        "flashbang",
        "smoke grenade",
        "molotov",
        "incendiary grenade",
        "decoy grenade",
    }
    throwable_weapon_indices = {11, 12, 13, 14, 15}

    def is_throwable_weapon(name: Any, weapon_index: int | None = None) -> bool:
        normalized = str(name or "").strip().lower()
        return (
            normalized in throwable_weapon_names
            or "grenade" in normalized
            or "flashbang" in normalized
            or (weapon_index is not None and weapon_index in throwable_weapon_indices)
        )

    def add_record(
        groups: dict[str, list[Mapping[str, Any]]],
        episode_id: str,
        record: Mapping[str, Any],
    ) -> None:
        groups.setdefault(episode_id, []).append(record)

    def scalar(value: Any, time_index: int, bot_index: int, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            item = value[time_index, bot_index]
            return float(item.detach().cpu()) if hasattr(item, "detach") else float(item)
        except (IndexError, TypeError, ValueError):
            return default

    def payload(value: Any) -> bytes:
        values = value.detach().cpu().clamp(0.0, 1.0) if hasattr(value, "detach") else value
        return bytes(int(round(float(item) * 255.0)) for item in values.tolist())

    def observation_context(
        recurrent: Any,
        time_index: int,
        bot_index: int,
        *,
        episode_id: str,
        time_s: float,
    ) -> dict[str, Any]:
        context = recurrent.context
        observation = None
        try:
            observation = decode_observation(payload(recurrent.observations[time_index, bot_index]))
        except (TypeError, ValueError, struct.error):
            pass
        position_valid = bool(scalar(context.get("position_valid"), time_index, bot_index, 0.0))
        position_xy = context.get("position_xy")
        position: tuple[float, float] | None = None
        if position_valid and position_xy is not None:
            try:
                position = (
                    float(position_xy[time_index, bot_index, 0]),
                    float(position_xy[time_index, bot_index, 1]),
                )
            except (IndexError, TypeError, ValueError):
                position = None
        if position is None and observation is not None:
            position = tuple(float(value) for value in observation.self_position[:2])
            position_valid = True
        alive_value = context.get("alive")
        alive = bool(scalar(alive_value, time_index, bot_index, 1.0))
        if alive_value is None and observation is not None:
            alive = bool(int(observation.self_flags) & 1)
        distance = scalar(context.get("engagement_distance"), time_index, bot_index, 0.0)
        if distance <= 0.0 and observation is not None:
            distances = [
                float(player.distance)
                for player in observation.players
                if int(player.relation) == -1
                and float(player.distance) > 0.0
                and int(player.flags) & (PLAYER_DIRECT | PLAYER_RADAR | PLAYER_AUDIBLE)
            ]
            if distances:
                distance = min(distances)
        return {
            "episode_id": episode_id,
            "time_s": time_s,
            "position": position,
            "position_valid": position_valid,
            "engagement_distance": distance if distance > 0.0 else None,
            "alive": alive,
            "utility": scalar(context.get("utility"), time_index, bot_index, 0.0),
        }

    def percentile(values: Sequence[float], fraction: float) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        ratio = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * ratio

    with torch.no_grad():
        for recurrent in iter_recurrent_minibatches(
            selected_paths,
            sequence_length=128,
            batch_sequences=1,
            seed=int(seed),
            max_windows_per_path=1,
        ):
            batch = recurrent_to_hierarchical_batch(recurrent).to(resolved_device)
            predicted_intents, decision_outputs = _build_predicted_intent_sequence(actor, batch)
            target_sequence = batch.decision_target_sequence or {}
            for decision_index, output in enumerate(decision_outputs):
                target_index = min(127, decision_index * 8)
                targets = _sequence_target_at(target_sequence, target_index)
                categorical = (
                    (output.tactical_mode_logits, targets["tactical_mode"]),
                    (output.task_logits, targets["task"]),
                    (output.target_slot, targets["target_slot"]),
                )
                continuous = (
                    ("goal_position", output.goal_position_tensor, targets["goal_position"], 4096.0),
                    ("waypoint_position", output.waypoint_position_tensor, targets["waypoint_position"], 4096.0),
                    ("facing_yaw_pitch", output.facing_yaw_pitch, targets["facing_yaw_pitch"], 180.0),
                    ("desired_range", output.desired_range, targets["desired_range"], 4096.0),
                    ("aggression", output.aggression, targets["aggression"], 1.0),
                    ("risk", output.risk, targets["risk"], 1.0),
                    ("priority", output.priority, targets["priority"], 1.0),
                    ("ttl_ticks", output.ttl_ticks, targets["ttl_ticks"], 64.0),
                )
                confidence = targets["confidence"]["all"].to(torch.float32)
                valid = targets["valid_mask"]["all"].to(torch.float32)
                for field_index, (logits, target) in enumerate(categorical):
                    weight = confidence[:, field_index] * valid[:, field_index]
                    values = -F.log_softmax(logits, dim=-1).gather(
                        -1, target.to(torch.long).unsqueeze(-1)
                    ).squeeze(-1)
                    decision_loss_sum += float((values * weight).sum().cpu())
                    decision_weight_sum += float(weight.sum().cpu())
                for field_index, (_name, prediction, target, scale) in enumerate(continuous, start=3):
                    weight = confidence[:, field_index] * valid[:, field_index]
                    values = ((prediction - target.to(prediction.dtype)) / scale).square()
                    if values.ndim > 1:
                        values = values.mean(dim=-1)
                    decision_loss_sum += float((0.5 * values * weight).sum().cpu())
                    decision_weight_sum += float(weight.sum().cpu())

            hidden = torch.zeros(
                batch.batch_size,
                384,
                dtype=batch.observation_sequence.dtype,
                device=resolved_device,
            )
            action_outputs: list[Any] = []
            for time_index in range(128):
                action_output = actor.action(
                    batch.observation_sequence[:, time_index],
                    predicted_intents[:, time_index],
                    hidden,
                )
                action_outputs.append(action_output)
                hidden = action_output.recurrent_state.action
            yaw_by_bot = [0.0] * batch.batch_size
            human_yaw_by_bot = [0.0] * batch.batch_size
            time_by_bot = [0.0] * batch.batch_size
            for time_index, action_output in enumerate(action_outputs):
                duration = max(
                    float(recurrent.duration_s[time_index, 0]),
                    1.0 / 128.0,
                )
                for bot_index in range(batch.batch_size):
                    episode_id = str(recurrent.sequence_ids[time_index][bot_index])
                    common = observation_context(
                        recurrent,
                        time_index,
                        bot_index,
                        episode_id=episode_id,
                        time_s=time_by_bot[bot_index],
                    )
                    human_action = recurrent.actions[time_index, bot_index]
                    decoded_observation = None
                    try:
                        decoded_observation = decode_observation(
                            payload(recurrent.observations[time_index, bot_index])
                        )
                    except (TypeError, ValueError, struct.error):
                        pass
                    human_yaw_value = recurrent.context.get("yaw_deg")
                    if human_yaw_value is not None:
                        human_yaw = scalar(human_yaw_value, time_index, bot_index)
                    else:
                        human_yaw_by_bot[bot_index] += float(human_action[3])
                        human_yaw = human_yaw_by_bot[bot_index]
                    human_record = {
                        **common,
                        "forward": float(human_action[0]),
                        "yaw_deg": human_yaw,
                        "buy_action": int(round(float(human_action[7]))),
                        "fire": bool(
                            int(round(float(human_action[5]))) & int(IN_ATTACK)
                        ) and not is_throwable_weapon(
                            getattr(decoded_observation, "weapon_name", "")
                        ),
                        "utility": bool(
                            decoded_observation is not None
                            and any(
                                str(event.category).lower()
                                in {
                                    "grenade_projectile_throw",
                                    "smoke_start_self",
                                    "flash_explode_self",
                                    "fire_grenade_start_self",
                                    "he_explode_self",
                                    "decoy_start",
                                }
                                for event in decoded_observation.events
                            )
                        ),
                    }
                    add_record(human_records_by_episode, episode_id, human_record)
                    decision_output = decision_outputs[
                        min(time_index // 8, len(decision_outputs) - 1)
                    ]
                    decision_labels = (
                        int(decision_output.tactical_mode_logits[bot_index].argmax().cpu()),
                        int(decision_output.task_logits[bot_index].argmax().cpu()),
                        int(decision_output.target_slot[bot_index].argmax().cpu()),
                    )
                    observation_wire = BotObservationV1(
                        payload(recurrent.observations[time_index, bot_index])
                    )
                    runtime_intent = HierarchicalRuntimeService._decision_to_intent(
                        {
                            "tactical_mode_logits": decision_output.tactical_mode_logits,
                            "task_logits": decision_output.task_logits,
                            "goal_position": decision_output.goal_position_tensor,
                            "waypoint_position": decision_output.waypoint_position_tensor,
                            "facing_yaw_pitch": decision_output.facing_yaw_pitch,
                            "desired_range": decision_output.desired_range,
                            "target_slot_logits": decision_output.target_slot,
                            "aggression": decision_output.aggression,
                            "risk": decision_output.risk,
                            "priority": decision_output.priority,
                            "ttl_ticks": decision_output.ttl_ticks,
                        },
                        bot_index,
                        labels=decision_labels,
                        buy_allowed=HierarchicalRuntimeService._buy_allowed(observation_wire),
                    )
                    raw_action = action_output.to_actions([int(time_index) + 1])[bot_index]
                    guarded_action = guard_action(
                        runtime_intent,
                        raw_action,
                        observation=observation_wire,
                    )
                    guarded_action = guard_friendly_fire(observation_wire, guarded_action)
                    yaw_by_bot[bot_index] += float(guarded_action.yaw_delta_deg)
                    model_throwable = is_throwable_weapon(
                        getattr(decoded_observation, "weapon_name", ""),
                        guarded_action.weapon_select,
                    )
                    model_record = {
                        "episode_id": episode_id,
                        "time_s": time_by_bot[bot_index],
                        "delta_time_s": duration,
                        "forward": float(guarded_action.forward),
                        "yaw_deg": yaw_by_bot[bot_index],
                        "buy_action": int(guarded_action.buy_action),
                        # Candidate gate metrics must come from candidate
                        # outputs.  Do not copy Demo position, distance, or
                        # alive state into this record.
                        "fire": bool(guarded_action.buttons & int(IN_ATTACK)) and not model_throwable,
                        "utility": bool(
                            guarded_action.buttons & int(IN_ATTACK | IN_ATTACK2)
                        ) and model_throwable,
                    }
                    add_record(model_records_by_episode, episode_id, model_record)
                    time_by_bot[bot_index] += duration
            window_count += 1

    if not model_records_by_episode or decision_weight_sum <= 0.0:
        raise ValueError("held-out Demo evaluation produced no weighted evidence")
    human_metrics_by_episode = [
        compute_human_like_metrics(records)
        for records in human_records_by_episode.values()
        if records
    ]
    if not human_metrics_by_episode:
        raise ValueError("held-out Demo evaluation produced no human reference metrics")
    from dataclasses import asdict

    from .human_gate import REQUIRED_HUMAN_METRICS

    metric_names = tuple(REQUIRED_HUMAN_METRICS)
    quantiles = {
        name: {
            "p05": percentile([float(asdict(item)[name]) for item in human_metrics_by_episode], 0.05),
            "p95": percentile([float(asdict(item)[name]) for item in human_metrics_by_episode], 0.95),
        }
        for name in metric_names
    }
    model_records = [
        record
        for episode_id in model_records_by_episode
        for record in model_records_by_episode[episode_id]
    ]
    model_metric_values = asdict(compute_human_like_metrics(model_records))
    model_metrics = {
        name: float(model_metric_values[name])
        for name in metric_names
    }
    return {
        "intent_nll": decision_loss_sum / decision_weight_sum,
        "human_metrics": model_metrics,
        "validation_quantiles": quantiles,
        "evidence_source": "held-out-demo-teacher-forced",
        "evidence_record_count": len(model_records),
        "decision_sample_count": int(decision_weight_sum),
        "validation_window_count": window_count,
        "validation_path_count": len(selected_paths),
    }


def _flatten_rollout_shards(shards: Sequence[Any]) -> tuple[Any, ...]:
    """Flatten a wave's shard values while preserving one terminal match identity."""

    from .training_mappo import RecurrentRolloutV1

    flattened: list[Any] = []
    for shard in shards:
        if isinstance(shard, MatchCompletionV1):
            flattened.extend(_flatten_rollout_shards(shard.envelopes))
        elif isinstance(shard, (RolloutEnvelopeV1, RecurrentRolloutV1)):
            flattened.append(shard)
        elif isinstance(shard, Sequence) and not isinstance(shard, (str, bytes, bytearray)):
            flattened.extend(_flatten_rollout_shards(tuple(shard)))
        else:
            flattened.append(shard)
    return tuple(flattened)


def _merge_complete_match_rollouts(shards: Sequence[Any]) -> tuple[Any, ...]:
    """Load every segment of each completed match into one terminal rollout."""

    from .training_mappo import RecurrentRolloutV1, load_rollout_envelope

    grouped: dict[tuple[str, str, int], list[RecurrentRolloutV1]] = {}
    order: list[tuple[str, str, int]] = []
    for shard in _flatten_rollout_shards(shards):
        rollout = load_rollout_envelope(shard) if isinstance(shard, RolloutEnvelopeV1) else shard
        if not isinstance(rollout, RecurrentRolloutV1):
            raise TypeError("complete match shards must be rollout envelopes or recurrent rollouts")
        if rollout.truncated:
            raise ValueError("complete match cannot include truncated rollout segments")
        key = (rollout.instance_id, rollout.match_id, rollout.policy_generation)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(rollout)

    merged: list[RecurrentRolloutV1] = []
    from .lineage import DatasetManifestV1

    for key in order:
        segments = grouped[key]
        first = segments[0]
        phase = first.phase
        bot_ids = first.transitions[0].bot_ids
        epoch = first.transitions[0].epoch
        transitions = []
        critic_snapshots = []
        rollout_parents = []
        critic_parents = []
        hashes: list[str] = []
        critic_hashes: list[str] = []
        for segment in segments:
            if segment.phase is not phase:
                raise ValueError("complete match rollout segments cannot mix Get5 phases")
            if any(transition.bot_ids != bot_ids for transition in segment.transitions):
                raise ValueError("complete match rollout bot ids must remain stable")
            segment_epoch = segment.transitions[0].epoch
            if any(transition.epoch != segment_epoch for transition in segment.transitions):
                raise ValueError("each complete match rollout segment must use one IPC epoch")
            if segment_epoch != epoch and transitions and not all(transitions[-1].dones):
                raise ValueError("complete match IPC epoch changes must follow a terminal segment boundary")
            # Get5 advances the IPC epoch at round/side boundaries. The
            # merged artifact is a logical match sequence, so normalize that
            # transport-local field after validating every segment's own
            # epoch. Boundary dones and hidden-state resets preserve the
            # recurrent semantics across the normalization.
            transitions.extend(
                replace(transition, epoch=epoch)
                for transition in segment.transitions
            )
            critic_snapshots.extend(segment.critic_snapshots)
            if segment.dataset_manifest is not None:
                if segment.dataset_manifest not in rollout_parents:
                    rollout_parents.append(segment.dataset_manifest)
            if segment.critic_manifest is not None:
                if segment.critic_manifest not in critic_parents:
                    critic_parents.append(segment.critic_manifest)
            hashes.append(segment.shard_sha256)
            critic_hashes.append(segment.critic_sha256)
        if not transitions:
            raise ValueError("complete match rollout contains no transitions")
        if any(
            current.server_tick <= previous.server_tick
            for previous, current in zip(transitions, transitions[1:])
        ):
            raise ValueError("complete match rollout server ticks must be strictly increasing")
        merged.append(
            RecurrentRolloutV1(
                instance_id=key[0],
                match_id=key[1],
                policy_generation=key[2],
                transitions=tuple(transitions),
                critic_snapshots=tuple(critic_snapshots),
                shard_sha256=hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest(),
                critic_sha256=hashlib.sha256("|".join(critic_hashes).encode("utf-8")).hexdigest(),
                dataset_manifest=DatasetManifestV1(
                    name=f"complete-match:{key[0]}:{key[1]}",
                    purpose=DataPurpose.TEST_ONLY,
                    parents=tuple(rollout_parents),
                    artifact_type="trajectory",
                    metadata={"complete_match": "true", "segment_count": str(len(segments))},
                ),
                critic_manifest=DatasetManifestV1(
                    name=f"complete-match-critic:{key[0]}:{key[1]}",
                    purpose=DataPurpose.TEST_ONLY,
                    parents=tuple(critic_parents),
                    artifact_type="critic_trajectory",
                    metadata={"complete_match": "true", "segment_count": str(len(segments))},
                ),
                terminal=True,
                truncated=False,
            )
        )
    return tuple(merged)


def _single_wave_dataset_manifest(
    wave: CompleteMatchWaveV1,
    *,
    expected_rollout_count: int | None = None,
) -> DatasetManifestV1:
    """Build candidate lineage from the actual terminal wave rollouts."""

    if not isinstance(wave, CompleteMatchWaveV1) or not wave.complete:
        raise ValueError("candidate lineage requires a completed single-match wave")
    rollouts = _merge_complete_match_rollouts(wave.shards)
    if expected_rollout_count is not None and len(rollouts) != int(expected_rollout_count):
        raise ValueError(
            "candidate lineage rollout count does not match the single-wave update "
            f"({len(rollouts)} != {expected_rollout_count})"
        )
    from .training_mappo import RecurrentRolloutV1

    if not rollouts or any(not isinstance(rollout, RecurrentRolloutV1) for rollout in rollouts):
        raise ValueError("candidate lineage requires complete recurrent rollout manifests")
    parents: list[DatasetManifestV1] = []
    for rollout in rollouts:
        for manifest in (rollout.dataset_manifest, rollout.critic_manifest):
            if not isinstance(manifest, DatasetManifestV1):
                raise ValueError("candidate lineage is missing a rollout or critic manifest")
            if all(manifest != existing for existing in parents):
                parents.append(manifest)

    def shard_lineage_payload(value: Any) -> Any:
        """Keep wave metadata JSON-safe without embedding rollout objects."""

        if isinstance(value, RecurrentRolloutV1):
            return {
                "type": "RecurrentRolloutV1",
                "instance_id": value.instance_id,
                "match_id": value.match_id,
                "policy_generation": value.policy_generation,
                "shard_sha256": value.shard_sha256,
                "critic_sha256": value.critic_sha256,
                "transition_count": len(value.transitions),
                "terminal": value.terminal,
                "truncated": value.truncated,
            }
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return shard_lineage_payload(value.to_dict())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): shard_lineage_payload(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [shard_lineage_payload(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": type(value).__name__, "repr": repr(value)}

    match_payload = [
        {
            "instance_id": manifest.instance_id,
            "match_id": manifest.match_id,
            "package_sha256": manifest.package_sha256,
            "policy_generation": manifest.policy_generation,
            "host_timescale": manifest.host_timescale,
            "ruleset": manifest.ruleset,
            "terminal": manifest.terminal,
            "attempt": manifest.attempt,
            "shard": shard_lineage_payload(manifest.shard),
        }
        for manifest in wave.manifests
    ]
    rollout_payload = [
        {
            "instance_id": rollout.instance_id,
            "match_id": rollout.match_id,
            "policy_generation": rollout.policy_generation,
            "shard_sha256": rollout.shard_sha256,
            "critic_sha256": rollout.critic_sha256,
            "transition_count": len(rollout.transitions),
            "terminal": rollout.terminal,
            "truncated": rollout.truncated,
        }
        for rollout in rollouts
    ]
    metadata = {
        "wave_package_sha256": wave.package_sha256,
        "wave_policy_generation": str(wave.policy_generation),
        "wave_host_timescale": str(wave.host_timescale),
        "wave_ruleset": wave.ruleset,
        "wave_manifest_json": json.dumps(match_payload, ensure_ascii=False, sort_keys=True),
        "rollout_manifest_json": json.dumps(rollout_payload, ensure_ascii=False, sort_keys=True),
        "rollout_count": str(len(rollouts)),
    }
    lineage_digest = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DatasetManifestV1(
        name=f"single-wave:{wave.package_sha256[:12]}:{wave.policy_generation}",
        purpose=DataPurpose.TEST_ONLY,
        parents=tuple(parents),
        artifact_type="single_wave_mappo_rollout",
        source_sha256=lineage_digest,
        metadata=metadata,
    )


def _rollout_human_gate_evidence(shards: Sequence[Any]) -> dict[str, Any]:
    """Derive candidate gate evidence from the actual terminal rollout rows."""

    from dataclasses import asdict

    from .action_reconstruction import IN_ATTACK, IN_ATTACK2
    from .human_gate import REQUIRED_HUMAN_METRICS
    from .observation_projection import decode_observation
    from .training_gail import compute_human_like_metrics
    from .training_mappo import RecurrentRolloutV1, load_rollout_envelope

    records_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    episode_order: list[str] = []
    log_probs: list[float] = []

    def append_record(episode_id: str, record: Mapping[str, Any]) -> None:
        if episode_id not in records_by_episode:
            records_by_episode[episode_id] = []
            episode_order.append(episode_id)
        records_by_episode[episode_id].append(record)

    throwable_names = {
        "he grenade",
        "flashbang",
        "smoke grenade",
        "molotov",
        "incendiary grenade",
        "decoy grenade",
    }

    for rollout in _merge_complete_match_rollouts(shards):
        if not isinstance(rollout, RecurrentRolloutV1):
            raise TypeError("candidate gate evidence requires recurrent rollout shards")
        if not rollout.terminal or rollout.truncated:
            raise ValueError("candidate gate evidence requires complete terminal rollouts")
        if not rollout.transitions:
            raise ValueError("candidate terminal rollout has no transitions")
        first_tick = int(rollout.transitions[0].server_tick)
        previous_tick: int | None = None
        yaw_by_bot = [0.0] * len(rollout.transitions[0].bot_ids)
        for transition in rollout.transitions:
            tick = int(transition.server_tick)
            delta_time_s = (
                1.0 / 128.0
                if previous_tick is None
                else max(1, tick - previous_tick) / 128.0
            )
            time_s = max(0, tick - first_tick) / 128.0
            for bot_index, bot_id in enumerate(transition.bot_ids):
                action = transition.actions[bot_index]
                yaw_by_bot[bot_index] += float(action.yaw_delta_deg)
                record: dict[str, Any] = {
                    "episode_id": f"{rollout.match_id}:{bot_id}",
                    "time_s": time_s,
                    "delta_time_s": delta_time_s,
                    "forward": float(action.forward),
                    "yaw_deg": yaw_by_bot[bot_index],
                    "buy_action": int(action.buy_action),
                }
                try:
                    projection = decode_observation(transition.observations[bot_index])
                except (TypeError, ValueError, struct.error):
                    projection = None
                weapon_name = str(getattr(projection, "weapon_name", "")).strip().lower()
                throwable = weapon_name in throwable_names or "grenade" in weapon_name
                record["fire"] = bool(int(action.buttons) & int(IN_ATTACK)) and not throwable
                record["utility"] = bool(
                    int(action.buttons) & int(IN_ATTACK | IN_ATTACK2)
                ) and throwable
                append_record(f"{rollout.match_id}:{bot_id}", record)
                stored_log_prob = transition.decision_log_probs[bot_index]
                if stored_log_prob is not None:
                    value = float(stored_log_prob)
                    if not math.isfinite(value):
                        raise ValueError("candidate rollout contains a non-finite decision log probability")
                    log_probs.append(value)
            previous_tick = tick
    records = [
        record
        for episode_id in episode_order
        for record in records_by_episode[episode_id]
    ]
    if not records:
        raise ValueError("candidate rollout evidence contains no action records")
    if not log_probs:
        raise ValueError("candidate rollout evidence contains no decision log probabilities")
    all_metrics = asdict(compute_human_like_metrics(records))
    metrics = {name: float(all_metrics[name]) for name in REQUIRED_HUMAN_METRICS}
    return {
        "intent_nll": -sum(log_probs) / len(log_probs),
        "human_metrics": metrics,
        "evidence_source": "complete-test-only-rollout",
        "evidence_record_count": len(records),
        "decision_sample_count": len(log_probs),
    }


def build_test_pipeline(
    config: HierarchicalTestConfigV1,
    *,
    purpose: DataPurpose | str = DataPurpose.TEST_ONLY,
    selected_specs: Sequence[Any] | None = None,
    selected_timescale: int | None = None,
    validation_manifest: Any | None = None,
    wave_collector: Callable[..., Any] | None = None,
    match_runner: Callable[..., Any] | None = None,
    mappo_update: Callable[..., Any] | None = None,
    candidate_exporter: Callable[..., Any] | None = None,
    probe_stage_runner: Callable[..., Any] | None = None,
) -> TestTrainingPipeline:
    """Construct the fully bound test-only pipeline used by the CLI.

    The collector still accepts an injected match runner because server-event
    transport is an integration boundary.  All artifacts after collection,
    including the real one-wave MAPPO update and candidate export, are bound
    here rather than left as ``None`` defaults at the public CLI entry point.
    """

    from .server_farm import collect_single_match_wave

    if not isinstance(config, HierarchicalTestConfigV1):
        raise TypeError("config must be HierarchicalTestConfigV1")
    normalized_selected_specs = None if selected_specs is None else tuple(selected_specs)
    resolved_specs = (
        tuple(normalized_selected_specs)
        if normalized_selected_specs is not None
        else ()
    )
    if normalized_selected_specs is not None and not resolved_specs:
        raise ValueError("test-only pipeline factory must select at least one server specification")
    resolved_timescale = (
        int(selected_timescale)
        if selected_timescale is not None
        else 1
    )
    resolved_validation = (
        validation_manifest
        if validation_manifest is not None
        else _load_test_validation_manifest(config.data_root)
    )
    holder: dict[str, Any] = {}

    def build_gate_evidence(package: Any, actor: Any | None = None) -> Mapping[str, Any]:
        validation_paths = _validation_sequence_paths(resolved_validation)
        if not validation_paths:
            validation_paths = _package_heldout_validation_paths(package)
        if not validation_paths:
            raise RuntimeError(
                "test-only human gate requires an explicit held-out Demo validation split"
            )
        resolved_actor = actor
        if resolved_actor is None:
            resolved_actor = _load_hierarchical_actor(package, config.data_root)
        return _evaluate_heldout_demo_evidence(
            resolved_actor,
            validation_paths,
            device=config.training_device,
            seed=config.seed,
        )

    if wave_collector is None:
        def collect_wave(specs: Sequence[Any], baseline: Any, host_timescale: int) -> Any:
            resolved_runner = match_runner
            restart_instance = None
            if resolved_runner is None:
                resolved_runner = holder.get("match_runner")
                if resolved_runner is None:
                    resolved_runner = _HierarchicalMatchRunner(config, specs, baseline)
                    holder["match_runner"] = resolved_runner
                restart_instance = getattr(resolved_runner, "restart_instance", None)
            return collect_single_match_wave(
                specs,
                baseline,
                host_timescale=host_timescale,
                ruleset="mr12",
                match_runner=resolved_runner,
                restart_instance=restart_instance,
            )

        resolved_collector: Callable[..., Any] = collect_wave
    else:
        resolved_collector = wave_collector

    if mappo_update is None:
        def update_wave(shards: Sequence[Any]) -> Any:
            from .training_hierarchical import train_hierarchical_single_wave

            pipeline = holder.get("pipeline")
            baseline = getattr(pipeline, "active_package", None)
            if baseline is None:
                raise RuntimeError("test-only MAPPO updater was called before Demo baseline binding")
            rollouts = _merge_complete_match_rollouts(shards)
            generation = int(getattr(baseline, "generation", 0))
            actor = _load_hierarchical_actor(baseline, config.data_root)
            output_dir = config.data_root / "training" / "mappo" / f"single-wave-g{generation:03d}"
            report = train_hierarchical_single_wave(
                actor,
                tuple(rollouts),
                output_dir=output_dir,
                device=config.training_device,
                demo_action_path=getattr(baseline, "action_path", None),
                demo_action_sha256=getattr(baseline, "action_sha256", None),
            )
            holder["actor"] = actor
            holder["report"] = report
            return report

        resolved_updater: Callable[..., Any] = update_wave
    else:
        resolved_updater = mappo_update

    if candidate_exporter is None:
        def export_candidate(update_result: Any, wave: CompleteMatchWaveV1) -> Any:
            from .hierarchical_demo_training import HierarchicalDemoRunManifestV1
            from .export import export_hierarchical_package

            pipeline = holder.get("pipeline")
            baseline = getattr(pipeline, "active_package", None)
            actor = holder.get("actor")
            if baseline is None or actor is None:
                raise RuntimeError("candidate exporter was called before the MAPPO actor was bound")
            report = update_result
            generation = int(getattr(baseline, "generation", 0)) + 1
            dataset_manifest = _single_wave_dataset_manifest(
                wave,
                expected_rollout_count=int(getattr(report, "rollout_count", 0)),
            )
            gate_evidence = _invoke_pipeline_callback(
                build_gate_evidence,
                baseline,
                actor,
            )
            validation_quantiles = getattr(
                getattr(pipeline, "validation_manifest", None),
                "quantiles",
                None,
            )
            if isinstance(getattr(pipeline, "validation_manifest", None), Mapping):
                validation_quantiles = pipeline.validation_manifest.get("quantiles")
            run_metrics = {
                "decision_qat_enabled": bool(report.decision_qat_enabled),
                "single_wave": True,
                "rollout_count": int(report.rollout_count),
                **gate_evidence,
            }
            if isinstance(validation_quantiles, Mapping):
                run_metrics["validation_quantiles"] = dict(validation_quantiles)
            run_manifest = HierarchicalDemoRunManifestV1(
                dataset_manifest=dataset_manifest,
                name=f"hierarchical-test-single-wave-generation-{generation:03d}",
                purpose=DataPurpose.TEST_ONLY,
                steps=1,
                config_sha256="",
                checkpoint_path=Path(report.checkpoint_path),
                loss_history=tuple(float(value) for value in report.losses.values()),
                metrics=run_metrics,
            )
            package = export_hierarchical_package(
                run_manifest,
                config.data_root / "models" / f"generation-{generation:03d}",
                DataPurpose.TEST_ONLY,
                actor=actor,
                generation=generation,
                parent_generation=int(getattr(baseline, "generation", 0)),
                metrics=run_manifest.metrics,
            )
            if package.action_sha256 != baseline.action_sha256:
                raise RuntimeError("single-wave candidate changed the Demo-only action artifact")
            return package

        resolved_exporter: Callable[..., Any] = export_candidate
    else:
        resolved_exporter = candidate_exporter

    def finalize_runner() -> None:
        runner = holder.get("match_runner")
        close = getattr(runner, "close", None)
        if callable(close):
            close()

    pipeline = TestTrainingPipeline(
        config,
        purpose=purpose,
        selected_specs=resolved_specs,
        selected_timescale=resolved_timescale,
        validation_manifest=resolved_validation,
        wave_collector=resolved_collector,
        mappo_update=resolved_updater,
        candidate_exporter=resolved_exporter,
        probe_stage_runner=probe_stage_runner,
        gate_evidence_builder=build_gate_evidence,
        finalizer=finalize_runner,
    )
    holder["pipeline"] = pipeline
    return pipeline
