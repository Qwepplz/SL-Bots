from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import itertools
import math
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from .contracts import DataPurpose, Phase, ensure_purpose
from .export import (
    PolicyGenerationV1,
    _load_actor,
    load_policy_generation,
    publish_policy_generation,
    export_actor,
)
from .selfplay import RolloutEnvelopeV1, RosterProfile, SelfPlayController
from .selfplay_worker import SelfPlayWorker
from .server_farm import DedicatedServerFarm, ServerInstanceSpecV1, build_server_specs
from .training_bc import TrainingRunManifestV1, load_config, train_bc
from .training_dataset import ingest_demo_corpus, sequence_paths
from .training_gail import (
    BootstrapMatchV1,
    DiscriminatorRunManifestV1,
    GailWindowPathStream,
    WindowSampleV1,
    WindowDiscriminator,
    build_bootstrap_manifest,
    compute_human_like_targets,
    sample_rehearsal_windows,
    train_discriminator,
)
from .training_mappo import RecurrentRolloutV1, load_rollout_envelope, train_mappo
from .runtime import RuntimeService, Win32SharedMemoryTransportV1
from .get5_control import Get5ControlState


def _percentile(values: Sequence[float], level: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * level
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for production training") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            import os

            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _manifest_from_payload(value: Mapping[str, Any]) -> Any:
    from .lineage import DatasetManifestV1

    parents = tuple(
        _manifest_from_payload(item)
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


def _parquet_value(batch: Any, name: str, index: int) -> Any:
    value = batch.column(batch.schema.get_field_index(name))[index]
    return value.as_py() if hasattr(value, "as_py") else value


def _sequence_records(paths: Sequence[Path]) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for human-like target extraction") from error
    for path in paths:
        parquet_file = pq.ParquetFile(str(path))
        required = ("forward", "yaw_delta_deg", "buy_action", "duration_s", "utility")
        missing = [name for name in required if name not in parquet_file.schema.names]
        if missing:
            raise ValueError(f"sequence shard is missing columns: {', '.join(missing)}")
        optional = tuple(
            name
            for name in (
                "sequence_id",
                "position_x",
                "position_y",
                "position_valid",
                "engagement_distance",
                "alive",
            )
            if name in parquet_file.schema.names
        )
        columns = [*required, *optional]
        elapsed_s = 0.0
        absolute_yaw_deg = 0.0
        previous_sequence: str | None = None
        for batch in parquet_file.iter_batches(batch_size=4096, columns=columns):
            for index in range(batch.num_rows):
                sequence_id = (
                    str(_parquet_value(batch, "sequence_id", index))
                    if "sequence_id" in optional
                    else str(path)
                )
                if not sequence_id:
                    raise ValueError("human-like sequence_id values must be non-empty")
                if sequence_id != previous_sequence:
                    elapsed_s = 0.0
                    absolute_yaw_deg = 0.0
                    previous_sequence = sequence_id
                duration_s = float(_parquet_value(batch, "duration_s", index))
                if not math.isfinite(duration_s) or duration_s <= 0.0:
                    raise ValueError("human-like duration_s values must be finite and positive")
                absolute_yaw_deg += float(_parquet_value(batch, "yaw_delta_deg", index))
                record: dict[str, Any] = {
                    "episode_id": f"{path}:{sequence_id}",
                    "time_s": elapsed_s,
                    "delta_time_s": duration_s,
                    "forward": float(_parquet_value(batch, "forward", index)),
                    "yaw_deg": absolute_yaw_deg,
                    "buy_action": int(_parquet_value(batch, "buy_action", index)),
                    "utility": _parquet_value(batch, "utility", index),
                }
                if "position_x" in optional and "position_y" in optional:
                    position_valid = (
                        bool(_parquet_value(batch, "position_valid", index))
                        if "position_valid" in optional
                        else True
                    )
                    record["position_valid"] = position_valid
                    record["position"] = (
                        float(_parquet_value(batch, "position_x", index)),
                        float(_parquet_value(batch, "position_y", index)),
                    )
                if "engagement_distance" in optional:
                    record["engagement_distance"] = float(
                        _parquet_value(batch, "engagement_distance", index)
                    )
                if "alive" in optional:
                    record["alive"] = bool(_parquet_value(batch, "alive", index))
                yield record
                elapsed_s += duration_s


def _open_child_transport(
    spec: ServerInstanceSpecV1,
    *,
    capacity: int,
    epoch: int,
    timeout_s: float,
) -> Win32SharedMemoryTransportV1:
    deadline = time.monotonic() + float(timeout_s)
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        transport: Win32SharedMemoryTransportV1 | None = None
        try:
            transport = Win32SharedMemoryTransportV1(
                spec.ipc_name,
                capacity=capacity,
                epoch=int(epoch),
                open_existing=True,
            )
            actual_epoch = int(transport.refresh_epoch())
            if actual_epoch != int(epoch):
                raise ValueError(
                    f"shared-memory epoch changed while starting {spec.instance_id}: "
                    f"expected {epoch}, got {actual_epoch}"
                )
            return transport
        except (OSError, RuntimeError, ValueError) as error:
            last_error = error
            if transport is not None:
                transport.close()
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(
        f"unable to open SourcePawn shared-memory transport for {spec.instance_id}"
    ) from last_error


def _run_selfplay_process(
    spec: ServerInstanceSpecV1,
    generation: int,
    match_id: str,
    data_root: Path,
    purpose: DataPurpose,
    source_manifest: Any,
    model_path: Path,
    epoch: int,
    profiles: tuple[RosterProfile, ...],
    rollout_queue: Any,
    metrics_queue: Any,
    error_queue: Any,
    stop_event: Any,
    rollout_horizon: int,
    wait_timeout_ms: int,
    ipc_capacity: int,
    ipc_timeout_s: float,
    max_bots: int,
) -> None:
    transport: Win32SharedMemoryTransportV1 | None = None
    worker_started = False
    try:
        transport = _open_child_transport(
            spec,
            capacity=ipc_capacity,
            epoch=epoch,
            timeout_s=ipc_timeout_s,
        )
        control_state = Get5ControlState(expected_match_id=match_id)
        runtime = RuntimeService(
            transport=transport,
            model_path=model_path,
            max_bots=int(max_bots),
            policy_generation=int(generation),
        )
        controller = SelfPlayController(
            data_root=data_root,
            purpose=purpose,
            source_manifest=source_manifest,
            adapter=None,
            instance_id=spec.instance_id,
            match_id=match_id,
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
            runtime=runtime,
            controller=controller,
            control_state=control_state,
            rollout_horizon=int(rollout_horizon),
            wait_timeout_ms=int(wait_timeout_ms),
        )
        worker_started = True
        worker.run(stop_event, rollout_queue, metrics_queue)
    except BaseException as error:
        try:
            error_queue.put(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
        except Exception:
            pass
        raise
    finally:
        if transport is not None and not worker_started:
            transport.close()


@dataclass
class _LiveSession:
    spec: ServerInstanceSpecV1
    transport: Any
    stop_event: Any
    rollout_queue: Any
    metrics_queue: Any
    process: Any
    error_queue: Any
    error_box: list[str] = field(default_factory=list)
    envelopes: list[RolloutEnvelopeV1] = field(default_factory=list)
    terminal_seen: bool = False
    ticks: int = 0
    ack_count: int = 0
    latencies_us: list[float] = field(default_factory=list)
    fallback_count: int = 0
    runtime_fallback_count: int = 0
    invalid_generation: int = 0
    process_failed: bool = False
    rules_validated: bool = False
    started_ns: int = 0
    first_server_tick: int | None = None
    last_server_tick: int | None = None
    first_tick_ns: int | None = None
    last_tick_ns: int | None = None
    tickrate_hz: float = 0.0


class ProductionTrainingPipeline:
    def __init__(
        self,
        *,
        config: Mapping[str, Any] | str | Path,
        demo_root: str | Path,
        data_root: str | Path,
        server_root: str | Path,
        max_servers: int | None = None,
        purpose: DataPurpose | str = DataPurpose.PRODUCTION,
        run_id: str = "gpu-training",
        duration_seconds: float | None = None,
        extractor_path: str | Path | None = None,
        farm: DedicatedServerFarm | None = None,
        transport_factory: Callable[[ServerInstanceSpecV1], Any] | None = None,
    ) -> None:
        self.config = load_config(config)
        self.demo_root = Path(demo_root).resolve(strict=False)
        self.data_root = Path(data_root).resolve(strict=False)
        self.server_root = Path(server_root).resolve(strict=False)
        configured_max_servers = self._config_int("max_servers", 4)
        resolved_max_servers = configured_max_servers if max_servers is None else int(max_servers)
        if not 1 <= resolved_max_servers <= 4:
            raise ValueError("max_servers must be between 1 and 4")
        self.max_servers = resolved_max_servers
        self.purpose = ensure_purpose(purpose)
        if self.purpose is not DataPurpose.PRODUCTION:
            raise ValueError("ProductionTrainingPipeline requires production purpose")
        self.run_id = str(run_id)
        configured_duration = (
            self.config.get("selfplay_duration_seconds", 7200.0)
            if duration_seconds is None
            else duration_seconds
        )
        self.duration_seconds = float(configured_duration)
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be a positive finite number")
        self.max_bots = self._config_int("max_bots", 10)
        if self.max_bots != 10:
            raise ValueError("production Get5 MR12 training requires exactly 10 bots per server")
        self.extractor_path = None if extractor_path is None else Path(extractor_path).resolve()
        self.specs = build_server_specs(
            self.max_servers,
            self.run_id,
            self.server_root,
            self.data_root,
        )
        self.farm = farm or DedicatedServerFarm(
            self.server_root,
            self.data_root,
            self.run_id,
            max_servers=self.max_servers,
        )
        self.transport_factory = transport_factory
        self.corpus_manifest: Any | None = None
        self.bc_run: Any | None = None
        self.gail_run: Any | None = None
        self.discriminator: Any | None = None
        self.gail_optimizer_state_dict: Mapping[str, Any] | None = None
        self.human_targets: Mapping[str, float] = {}
        self.human_targets_manifest: Any | None = None
        self.rehearsal_actor: Any | None = None
        self.critic: Any | None = None
        self.actor_optimizer_state_dict: Mapping[str, Any] | None = None
        self.critic_optimizer_state_dict: Mapping[str, Any] | None = None
        self.generations: dict[int, PolicyGenerationV1] = {}
        self.actor: Any | None = None
        self.selected_count: int | None = None
        self.active_generation = 0
        self._farm_started = False
        self._sessions: dict[str, _LiveSession] = {}
        self._last_match_envelopes: tuple[RolloutEnvelopeV1, ...] = ()
        self._last_match_durations: dict[str, float] = {}
        self._truncated_count = 0

    @property
    def pointer_path(self) -> Path:
        return self.data_root / "models" / "current-generation.json"

    def _config_int(self, name: str, default: int) -> int:
        value = int(self.config.get(name, default))
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def _config_float(self, name: str, default: float) -> float:
        value = float(self.config.get(name, default))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def prepare(self, *, resume: bool = False) -> None:
        if resume:
            self._prepare_resume()
            return
        self.generations.clear()
        self.active_generation = 0
        self.corpus_manifest = ingest_demo_corpus(
            self.demo_root,
            self.data_root,
            self.extractor_path,
            self.purpose,
            expected_count=self._config_int("demo_expected_count", 10),
            allow_header_only=False,
        )
        self.bc_run = train_bc(
            self.config,
            self.corpus_manifest,
            output_dir=self.data_root / "training" / "bc",
            device=str(self.config.get("device", "cuda")),
        )
        self.actor = _load_actor(self.bc_run, None)
        self.rehearsal_actor = copy.deepcopy(self.actor)
        self._write_initial_generation()
        train_paths = sequence_paths(self.corpus_manifest, "train")
        validation_paths = sequence_paths(self.corpus_manifest, "validation")
        train_records = _sequence_records(train_paths)
        validation_records = _sequence_records(validation_paths)
        target_payload = compute_human_like_targets(
            train_records,
            validation_records,
            corpus_sha256=str(self.corpus_manifest.source_sha256),
            train_manifest=self.corpus_manifest,
            validation_manifest=self.corpus_manifest,
            purpose=self.purpose,
            output_path=self.data_root / "training" / "human-like-targets.json",
        )
        self.human_targets = {
            str(key): float(value)
            for key, value in target_payload["targets"].items()
        }
        self.human_targets_manifest = _manifest_from_payload(target_payload["dataset_manifest"])
        self.active_generation = 0

    def training_device_probe(self) -> Any:
        from .training_device import run_training_device_smoke

        return run_training_device_smoke(str(self.config.get("device", "cuda")))

    def _load_checkpoint(self, path: Path) -> Mapping[str, Any]:
        import torch

        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"training checkpoint must be a mapping: {path}")
        return payload

    @staticmethod
    def _checkpoint_manifest(payload: Mapping[str, Any], *, path: Path) -> Any:
        value = payload.get("dataset_manifest")
        if hasattr(value, "effective_purpose"):
            return value
        if isinstance(value, Mapping):
            return _manifest_from_payload(value)
        raise ValueError(f"checkpoint dataset_manifest is missing: {path}")

    def _prepare_resume(self) -> None:
        """Restore every training artifact without rebuilding the Demo or BC run."""
        if not self.pointer_path.is_file():
            raise FileNotFoundError(f"resume policy pointer is missing: {self.pointer_path}")
        current = load_policy_generation(self.pointer_path)
        self.generations[current.number] = current
        self.active_generation = current.number

        bc_path = self.data_root / "training" / "bc" / "bc-checkpoint.pt"
        bc_checkpoint = self._load_checkpoint(bc_path)
        self.corpus_manifest = self._checkpoint_manifest(bc_checkpoint, path=bc_path)
        bc_config = bc_checkpoint.get("config", {})
        if not isinstance(bc_config, Mapping):
            raise ValueError("BC checkpoint config must be a mapping")
        self.bc_run = TrainingRunManifestV1(
            name="bc",
            purpose=self.corpus_manifest.effective_purpose(),
            dataset_manifest=self.corpus_manifest,
            steps=len(tuple(bc_checkpoint.get("loss_history", ()))),
            loss_history=tuple(float(value) for value in bc_checkpoint.get("loss_history", ())),
            checkpoint_path=str(bc_path),
            config_sha256=hashlib.sha256(
                json.dumps(dict(bc_config), sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            metadata={"restored": "true", "checkpoint": str(bc_path)},
        )
        self.rehearsal_actor = _load_actor(self.bc_run, None)
        self.actor = _load_actor(
            type("Run", (), {"checkpoint_path": str(current.actor_checkpoint)})(),
            None,
        )
        self._load_generation_training_state(current)

        target_path = self.data_root / "training" / "human-like-targets.json"
        try:
            target_payload = json.loads(target_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"human-like target artifact is not valid JSON: {target_path}") from error
        if not isinstance(target_payload, Mapping) or target_payload.get("schema") != "human-like-targets-v1":
            raise ValueError("human-like target artifact schema is invalid")
        if str(target_payload.get("corpus_sha256")) != str(self.corpus_manifest.source_sha256):
            raise ValueError("human-like target corpus hash does not match the BC corpus")
        target_manifest_payload = target_payload.get("dataset_manifest")
        if not isinstance(target_manifest_payload, Mapping):
            raise ValueError("human-like target dataset manifest is missing")
        self.human_targets_manifest = _manifest_from_payload(target_manifest_payload)
        if self.human_targets_manifest.source_sha256 != self.corpus_manifest.source_sha256:
            raise ValueError("human-like target manifest source does not match the BC corpus")
        targets = target_payload.get("targets", {})
        if not isinstance(targets, Mapping):
            raise ValueError("human-like target values must be a mapping")
        self.human_targets = {str(key): float(value) for key, value in targets.items()}

        gail_path = self.data_root / "training" / "gail" / "gail-checkpoint.pt"
        gail_checkpoint = self._load_checkpoint(gail_path)
        gail_manifest_payload = gail_checkpoint.get("dataset_manifest")
        if not isinstance(gail_manifest_payload, Mapping):
            raise ValueError("GAIL checkpoint dataset manifest is missing")
        gail_manifest = _manifest_from_payload(gail_manifest_payload)
        gail_loss = tuple(float(value) for value in gail_checkpoint.get("loss_history", ()))
        gail_device = gail_checkpoint.get("device", {})
        if not isinstance(gail_device, Mapping):
            gail_device = {}
        self.gail_run = DiscriminatorRunManifestV1(
            name="gail",
            purpose=gail_manifest.effective_purpose(),
            dataset_manifest=gail_manifest,
            steps=len(gail_loss),
            loss_history=gail_loss,
            checkpoint_path=str(gail_path),
            config_sha256=hashlib.sha256(
                json.dumps(
                    {
                        "phase": gail_checkpoint.get("phase"),
                        "steps": len(gail_loss),
                        "device": dict(gail_device),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            device_type=str(gail_device.get("type", "cpu")),
            device_index=(
                None
                if gail_device.get("index") is None
                else int(gail_device["index"])
            ),
            device_name=str(gail_device.get("name", "")),
            torch_version=str(gail_device.get("torch_version", "")),
            hip_version=str(gail_device.get("hip_version", "")),
            validation_history=tuple(
                float(value) for value in gail_checkpoint.get("validation_history", ())
            ),
            metadata={"restored": "true", "checkpoint": str(gail_path)},
        )
        gail_optimizer_state = gail_checkpoint.get("optimizer_state_dict")
        if not isinstance(gail_optimizer_state, Mapping):
            raise ValueError("GAIL checkpoint optimizer state is missing")
        self.gail_optimizer_state_dict = dict(gail_optimizer_state)
        model_config = gail_checkpoint.get("model_config", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("GAIL checkpoint model_config must be a mapping")
        self.discriminator = WindowDiscriminator(
            hidden_size=int(model_config.get("hidden_size", 128))
        )
        self.discriminator.load_state_dict(gail_checkpoint["model_state_dict"])
        self.discriminator.eval()

    def _load_generation_training_state(self, generation: PolicyGenerationV1) -> None:
        from .training_mappo import CentralValueCritic

        critic_checkpoint = self._load_checkpoint(generation.critic_checkpoint)
        optimizer_checkpoint = self._load_checkpoint(generation.optimizer_checkpoint)
        critic_config = critic_checkpoint.get("critic_config", {})
        if not isinstance(critic_config, Mapping):
            raise ValueError("critic checkpoint config must be a mapping")
        self.critic = CentralValueCritic(
            feature_size=int(critic_config.get("feature_size", 64)),
            hidden_size=int(critic_config.get("hidden_size", 128)),
        )
        critic_state = critic_checkpoint.get("critic_state_dict")
        if not isinstance(critic_state, Mapping):
            raise ValueError("critic checkpoint state is missing")
        self.critic.load_state_dict(dict(critic_state))
        actor_optimizer_state = optimizer_checkpoint.get("actor_optimizer_state_dict")
        critic_optimizer_state = optimizer_checkpoint.get("critic_optimizer_state_dict")
        behavior_sha256 = optimizer_checkpoint.get("behavior_sha256")
        if not isinstance(actor_optimizer_state, Mapping) or not isinstance(critic_optimizer_state, Mapping):
            raise ValueError("generation optimizer checkpoint is missing optimizer state")
        if not isinstance(behavior_sha256, str) or len(behavior_sha256) != 64:
            raise ValueError("generation optimizer checkpoint is missing behavior_sha256")
        if "policy_generation" in optimizer_checkpoint and int(optimizer_checkpoint["policy_generation"]) != generation.number:
            raise ValueError("generation optimizer checkpoint number does not match policy pointer")
        if "policy_generation" in critic_checkpoint and int(critic_checkpoint["policy_generation"]) != generation.number:
            raise ValueError("critic checkpoint number does not match policy pointer")
        self.actor_optimizer_state_dict = dict(actor_optimizer_state)
        self.critic_optimizer_state_dict = dict(critic_optimizer_state)

    def _write_initial_generation(self) -> PolicyGenerationV1:
        if self.actor is None or self.bc_run is None:
            raise RuntimeError("production pipeline must prepare BC before exporting generation 0")
        import torch
        from .training_mappo import CentralValueCritic

        output_root = self.data_root / "models" / "generation-000"
        output_root.mkdir(parents=True, exist_ok=True)
        critic_path = output_root / "critic.pt"
        optimizer_path = output_root / "optimizer.pt"
        checkpoint = torch.load(
            self.bc_run.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        critic = CentralValueCritic(feature_size=64, hidden_size=128)
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
        self.critic = critic
        self.actor_optimizer_state_dict = dict(checkpoint.get("optimizer_state_dict", {}))
        self.critic_optimizer_state_dict = critic_optimizer.state_dict()
        _atomic_torch_save(
            critic_path,
            {
                "policy_generation": 0,
                "critic_state_dict": critic.state_dict(),
                "critic_config": {"feature_size": 64, "hidden_size": 128},
                "dataset_manifest": self.corpus_manifest,
            },
        )
        _atomic_torch_save(
            optimizer_path,
            {
                "policy_generation": 0,
                "actor_optimizer_state_dict": self.actor_optimizer_state_dict,
                "critic_optimizer_state_dict": self.critic_optimizer_state_dict,
                "behavior_sha256": _state_dict_hash(self.actor.state_dict()),
            },
        )
        onnx_path = output_root / "actor.onnx"
        export = export_actor(
            self.bc_run,
            onnx_path,
            self.purpose,
            actor=self.actor,
        )
        generation = PolicyGenerationV1(
            number=0,
            actor_checkpoint=Path(self.bc_run.checkpoint_path),
            critic_checkpoint=critic_path,
            optimizer_checkpoint=optimizer_path,
            onnx_path=onnx_path,
            behavior_sha256=export.sha256,
            parent_generation=None,
        )
        self.generations[0] = generation
        publish_policy_generation(generation, self.pointer_path)
        return generation

    def calibration_probe(self, count: int, stage: str, duration_s: float) -> Mapping[str, Any]:
        if self.generations.get(0) is None:
            raise RuntimeError("generation 0 actor package is unavailable for calibration")
        try:
            import psutil
        except ImportError as error:
            raise RuntimeError("psutil is required for real calibration memory telemetry") from error
        self.selected_count = int(count)
        self._stop_workers()
        self._stop_farm()
        self._start_farm(count)
        self.farm.set_policy_generation(0)
        self.farm.load_get5_match(0)
        self._start_sessions(0)
        deadline = time.monotonic() + float(duration_s)
        minimum_available_memory_gib = float(psutil.virtual_memory().available) / (1024.0 ** 3)
        try:
            while time.monotonic() < deadline:
                self._drain_all()
                minimum_available_memory_gib = min(
                    minimum_available_memory_gib,
                    float(psutil.virtual_memory().available) / (1024.0 ** 3),
                )
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            self._drain_all()
            minimum_available_memory_gib = min(
                minimum_available_memory_gib,
                float(psutil.virtual_memory().available) / (1024.0 ** 3),
            )
            return self._calibration_metrics(
                stage,
                available_memory_gib=minimum_available_memory_gib,
            )
        finally:
            self._stop_workers()
            self._stop_farm()

    def _calibration_metrics(
        self,
        stage: str,
        *,
        available_memory_gib: float | None = None,
    ) -> Mapping[str, Any]:
        if available_memory_gib is None:
            try:
                import psutil
            except ImportError as error:
                raise RuntimeError("psutil is required for real calibration memory telemetry") from error
            available_memory_gib = float(psutil.virtual_memory().available) / (1024.0 ** 3)
        if not math.isfinite(float(available_memory_gib)):
            raise RuntimeError("calibration available memory telemetry is not finite")
        servers: list[Mapping[str, Any]] = []
        for session in self._sessions.values():
            error_text = "" if not session.error_box else str(session.error_box[-1])
            lowered = error_text.lower()
            servers.append(
                {
                    "instance_id": session.spec.instance_id,
                    "ticks": session.ticks,
                    "tickrate_hz": session.tickrate_hz,
                    "rules_validated": session.rules_validated,
                    "p99_ms": _percentile(session.latencies_us, 0.99) / 1000.0,
                    "fallback_32_tick": session.fallback_count + session.runtime_fallback_count,
                    "source_fallback_count": session.fallback_count,
                    "runtime_fallback_count": session.runtime_fallback_count,
                    "ipc_collisions": int("ipc" in lowered or "mapping" in lowered),
                    "invalid_generation": session.invalid_generation,
                    "available_memory_gib": available_memory_gib,
                    "gpu_errors": int(session.process_failed and not ("ipc" in lowered or "mapping" in lowered)),
                    "stage": stage,
                }
            )
        if len(servers) != int(self.selected_count or 0):
            raise RuntimeError("calibration did not retain telemetry for every selected server")
        return {"servers": servers}

    def bootstrap_runner(self, count: int) -> tuple[BootstrapMatchV1, ...]:
        self.selected_count = int(count)
        try:
            self._load_match(0)
            self._wait_for_terminal_matches(
                self._config_float("bootstrap_timeout_seconds", 3600.0)
            )
            envelopes = self._last_match_envelopes
            matches = tuple(
                BootstrapMatchV1(
                    instance_id=spec.instance_id,
                    match_id=self._match_id(0, spec.instance_id),
                    policy_generation=0,
                    ruleset="mr12",
                    terminal=True,
                    duration_s=min(
                        3600.0,
                        max(0.0, self._session_duration(spec.instance_id)),
                    ),
                    purpose=self.purpose,
                )
                for spec in self.specs[:count]
            )
            if len(envelopes) < count:
                raise RuntimeError("bootstrap did not produce a terminal rollout for every server")
            self._train_gail(envelopes, matches)
            return matches
        except Exception:
            self._stop_workers()
            self._stop_farm()
            raise
        finally:
            self._stop_workers()

    def _train_gail(
        self,
        envelopes: Sequence[RolloutEnvelopeV1],
        matches: Sequence[BootstrapMatchV1],
    ) -> None:
        if self.corpus_manifest is None:
            raise RuntimeError("Demo corpus must be prepared before GAIL")
        bootstrap_manifest = build_bootstrap_manifest(
            matches,
            int(self.selected_count or 0),
            purpose=self.purpose,
            parent_manifests=(self.corpus_manifest,),
        )
        demo_paths = sequence_paths(self.corpus_manifest, "train")
        validation_demo_paths = sequence_paths(self.corpus_manifest, "validation")
        selfplay_paths = tuple(
            envelope.shard_path for envelope in envelopes if not envelope.truncated
        )
        if not demo_paths or not validation_demo_paths or not selfplay_paths:
            raise RuntimeError("production GAIL requires MR15 Demo and MR12 bootstrap windows")
        gail_output = self.data_root / "training" / "gail"
        window_length = self._config_int("gail_window_length", 128)
        self.gail_run = train_discriminator(
            GailWindowPathStream(
                demo_paths,
                source="demo",
                phase=Phase.LIVE,
                window_length=window_length,
            ),
            GailWindowPathStream(
                selfplay_paths,
                source="selfplay",
                phase=Phase.LIVE,
                window_length=window_length,
            ),
            phase=Phase.LIVE,
            output_dir=gail_output,
            purpose=self.purpose,
            steps=self._config_int("gail_max_updates", 2000),
            batch_size=self._config_int("gail_batch_size", 64),
            demo_ratio=float(self.config.get("gail_demo_ratio", 0.5)),
            seed=int(self.config.get("seed", 7)),
            parent_manifests=(self.corpus_manifest, bootstrap_manifest),
            device=str(self.config.get("device", "cuda")),
            learning_rate=float(self.config.get("learning_rate", 0.001)),
            window_length=window_length,
            validation_demo_windows=GailWindowPathStream(
                validation_demo_paths,
                source="demo",
                phase=Phase.LIVE,
                window_length=window_length,
            ),
            validation_selfplay_windows=GailWindowPathStream(
                selfplay_paths,
                source="selfplay",
                phase=Phase.LIVE,
                window_length=window_length,
            ),
            optimizer_state_dict=self.gail_optimizer_state_dict,
        )
        import torch

        checkpoint = torch.load(self.gail_run.checkpoint_path, map_location="cpu", weights_only=False)
        gail_optimizer_state = checkpoint.get("optimizer_state_dict")
        if not isinstance(gail_optimizer_state, Mapping):
            raise ValueError("GAIL checkpoint optimizer state is missing")
        self.gail_optimizer_state_dict = dict(gail_optimizer_state)
        config = checkpoint.get("model_config", {})
        self.discriminator = WindowDiscriminator(hidden_size=int(config.get("hidden_size", 128)))
        self.discriminator.load_state_dict(checkpoint["model_state_dict"])
        self.discriminator.eval()

    def _load_match(self, generation: int) -> None:
        if self.selected_count is None:
            raise RuntimeError("server concurrency has not been selected")
        if generation not in self.generations:
            if generation == 0:
                self._write_initial_generation()
            else:
                raise FileNotFoundError(f"policy generation {generation} is unavailable")
        self.active_generation = int(generation)
        self._stop_workers()
        self._start_farm(self.selected_count)
        self.farm.set_policy_generation(generation)
        self.farm.load_get5_match(generation)
        self._start_sessions(generation)

    def match_loader(self, generation: int) -> None:
        self._load_match(int(generation))

    def wave_collector(self, deadline: float) -> Mapping[str, Any]:
        if not self._sessions:
            raise RuntimeError("wave collector has no live self-play sessions")
        while time.monotonic() < float(deadline):
            self._drain_all()
            unavailable = self._unavailable_sessions()
            if unavailable:
                return {
                    "rollouts": self._all_session_envelopes(),
                    "unavailable_instances": unavailable,
                }
            if self._sessions and all(session.terminal_seen for session in self._sessions.values()):
                result = self._all_session_envelopes()
                self._last_match_envelopes = result
                self._stop_workers()
                return {"rollouts": result}
            time.sleep(min(0.05, max(0.0, float(deadline) - time.monotonic())))
        self._drain_all()
        return {"rollouts": self._all_session_envelopes()}

    def _rehearsal_windows(self, rollouts: Sequence[Any]) -> tuple[WindowSampleV1, ...]:
        if self.corpus_manifest is None:
            raise RuntimeError("Demo corpus is unavailable for MAPPO rehearsal")
        window_length = self._config_int("mappo_rehearsal_window_length", 128)
        batch_size = self._config_int("mappo_rehearsal_batch_size", 32)
        demo_limit = max(batch_size, self._config_int("mappo_rehearsal_demo_pool", batch_size))
        selfplay_limit = max(
            batch_size,
            self._config_int("mappo_rehearsal_selfplay_pool", batch_size),
        )
        demo_paths = sequence_paths(self.corpus_manifest, "train")
        demo_windows = tuple(
            itertools.islice(
                GailWindowPathStream(
                    demo_paths,
                    source="demo",
                    phase=Phase.LIVE,
                    window_length=window_length,
                ),
                demo_limit,
            )
        )
        selfplay_pool: list[WindowSampleV1] = []
        for item in rollouts:
            rollout = load_rollout_envelope(item) if isinstance(item, RolloutEnvelopeV1) else item
            if not isinstance(rollout, RecurrentRolloutV1):
                raise TypeError("MAPPO rehearsal rollouts must be loaded recurrent rollouts")
            for bot_index, bot_id in enumerate(rollout.transitions[0].bot_ids):
                observations: list[bytes] = []
                actions: list[Any] = []
                durations: list[float] = []
                previous_tick: int | None = None
                for transition in rollout.transitions:
                    observations.append(transition.observations[bot_index])
                    actions.append(transition.actions[bot_index])
                    durations.append(
                        1.0 / 128.0
                        if previous_tick is None
                        else max(1, transition.server_tick - previous_tick) / 128.0
                    )
                    previous_tick = transition.server_tick
                for start in range(0, len(observations) - window_length + 1, window_length):
                    if len(selfplay_pool) >= selfplay_limit:
                        break
                    selfplay_pool.append(
                        WindowSampleV1(
                            phase=Phase.LIVE,
                            observations=tuple(observations[start : start + window_length]),
                            actions=tuple(actions[start : start + window_length]),
                            source="selfplay",
                            episode_id=f"{rollout.match_id}/{bot_id}/{start}",
                            delta_time_s=tuple(durations[start : start + window_length]),
                        )
                    )
                if len(selfplay_pool) >= selfplay_limit:
                    break
            if len(selfplay_pool) >= selfplay_limit:
                break
        if not demo_windows:
            raise RuntimeError("production MAPPO rehearsal requires complete MR15 Demo windows")
        return sample_rehearsal_windows(
            demo_windows,
            tuple(selfplay_pool),
            phase=Phase.LIVE,
            batch_size=batch_size,
            demo_ratio=float(self.config.get("mappo_rehearsal_demo_ratio", 0.5)),
            seed=int(self.config.get("seed", 7)),
        )

    def wave_trainer(self, rollouts: Sequence[Any], generation: int) -> PolicyGenerationV1:
        if self.actor is None:
            raise RuntimeError("actor is unavailable for MAPPO")
        if self.discriminator is None or self.gail_run is None:
            raise RuntimeError("GAIL discriminator is unavailable for MAPPO")
        import torch

        behavior_actor = copy.deepcopy(self.actor)
        if self.rehearsal_actor is None:
            raise RuntimeError("BC rehearsal actor is unavailable for MAPPO")
        reference_actor = copy.deepcopy(self.rehearsal_actor)
        rehearsal_windows = self._rehearsal_windows(rollouts)
        output_root = self.data_root / "training" / "mappo" / f"g{generation + 1:03d}"
        run = train_mappo(
            self.actor,
            tuple(rollouts),
            output_dir=output_root,
            purpose=self.purpose,
            steps=1,
            parent_manifests=tuple(
                parent
                for parent in (
                    self.corpus_manifest,
                    self.gail_run.dataset_manifest,
                    self.human_targets_manifest,
                )
                if parent is not None
            ),
            seed=int(self.config.get("seed", 7)),
            discriminator=self.discriminator,
            reference_actor=reference_actor,
            behavior_actor=behavior_actor,
            critic=self.critic,
            actor_optimizer_state_dict=self.actor_optimizer_state_dict,
            critic_optimizer_state_dict=self.critic_optimizer_state_dict,
            human_like_targets=self.human_targets,
            demo_rehearsal_windows=rehearsal_windows,
            demo_rehearsal_coefficient=float(
                self.config.get("mappo_demo_rehearsal_coefficient", 0.01)
            ),
            gail_coefficient=float(self.config.get("mappo_gail_coefficient", 0.1)),
            kl_coefficient=float(self.config.get("mappo_kl_coefficient", 0.01)),
            constraint_coefficient=float(
                self.config.get("mappo_human_constraint_coefficient", 0.1)
            ),
            clip_ratio=float(self.config.get("mappo_clip_ratio", 0.2)),
            value_coefficient=float(self.config.get("mappo_value_coefficient", 0.5)),
            entropy_coefficient=float(self.config.get("mappo_entropy_coefficient", 0.0)),
            tbptt_window=self._config_int("mappo_tbptt_window", 128),
            minibatch_sequences=self._config_int("mappo_minibatch_sequences", 32),
            epochs=self._config_int("mappo_epochs", 4),
            gradient_clip_norm=float(self.config.get("mappo_gradient_clip_norm", 1.0)),
            device=str(self.config.get("device", "cuda")),
            policy_generation=generation + 1,
            parent_generation=generation,
            deadline_metadata={
                "timed_selfplay_duration_seconds": str(self.duration_seconds),
            },
        )
        checkpoint = torch.load(run.checkpoint_path, map_location="cpu", weights_only=False)
        generation_root = self.data_root / "models" / f"generation-{generation + 1:03d}"
        critic_path = generation_root / "critic.pt"
        optimizer_path = generation_root / "optimizer.pt"
        _atomic_torch_save(
            critic_path,
            {
                "policy_generation": generation + 1,
                "critic_state_dict": checkpoint["critic_state_dict"],
                "critic_config": checkpoint.get(
                    "critic_config",
                    {"feature_size": 64, "hidden_size": 128},
                ),
                "dataset_manifest": run.dataset_manifest,
            },
        )
        _atomic_torch_save(
            optimizer_path,
            {
                "policy_generation": generation + 1,
                "actor_optimizer_state_dict": checkpoint["actor_optimizer_state_dict"],
                "critic_optimizer_state_dict": checkpoint["critic_optimizer_state_dict"],
                "behavior_sha256": checkpoint["behavior_sha256"],
            },
        )
        actor_cpu = copy.deepcopy(self.actor).to("cpu")
        export = export_actor(
            run,
            generation_root / "actor.onnx",
            self.purpose,
            actor=actor_cpu,
            metrics=run.metrics,
        )
        result = PolicyGenerationV1(
            number=generation + 1,
            actor_checkpoint=Path(run.checkpoint_path),
            critic_checkpoint=critic_path,
            optimizer_checkpoint=optimizer_path,
            onnx_path=export.model_path,
            behavior_sha256=export.sha256,
            parent_generation=generation,
        )
        self.generations[generation + 1] = result
        self.actor_optimizer_state_dict = checkpoint["actor_optimizer_state_dict"]
        self.critic_optimizer_state_dict = checkpoint["critic_optimizer_state_dict"]
        return result

    def generation_publisher(self, generation: Any, next_generation: int) -> None:
        if not isinstance(generation, PolicyGenerationV1):
            raise TypeError("production publisher requires PolicyGenerationV1")
        if generation.number != int(next_generation):
            raise ValueError("published generation number does not match orchestrator")
        publish_policy_generation(generation, self.pointer_path)
        self.generations[generation.number] = generation

    def stop_sampling(self, cutoff: float) -> None:
        del cutoff
        self._stop_workers()

    def truncation_flusher(self, deadline: float) -> int:
        del deadline
        self._stop_workers()
        return self._truncated_count

    def instance_restarter(self, instance_id: str) -> None:
        normalized = str(instance_id)
        selected_ids = {
            spec.instance_id for spec in self.specs[: int(self.selected_count or 0)]
        }
        if normalized not in selected_ids:
            raise ValueError(f"unknown server instance: {instance_id}")
        session = self._sessions.pop(normalized, None)
        if session is not None:
            self._stop_session(session)
        if not self._farm_started:
            if self.selected_count is None:
                raise RuntimeError("server concurrency has not been selected")
            self._start_farm(self.selected_count)
        else:
            self.farm.restart_instance(normalized)
        self.farm.set_policy_generation(self.active_generation, instance_id=normalized)
        self.farm.load_get5_match(self.active_generation, instance_id=normalized)
        self._start_sessions(self.active_generation, instance_ids=(normalized,))

    def worker_stop(self) -> None:
        self._stop_workers()

    def runtime_stop(self) -> None:
        self._stop_workers()

    def server_stop(self) -> None:
        self._stop_workers()
        self._stop_farm()

    def _start_farm(self, count: int) -> None:
        count = int(count)
        if self._farm_started and len(self.farm.processes) == count:
            return
        if self._farm_started:
            self._stop_farm()
        self.farm.start(self.specs[:count], dry_run=False)
        self._farm_started = True

    def _stop_farm(self) -> None:
        if not self._farm_started:
            return
        self.farm.stop()
        self._farm_started = False

    def _start_sessions(
        self,
        generation: int,
        *,
        instance_ids: Sequence[str] | None = None,
    ) -> None:
        generation_info = self.generations.get(generation)
        if generation_info is None:
            raise FileNotFoundError(f"policy generation {generation} is unavailable")
        if self.corpus_manifest is None:
            raise RuntimeError("production self-play requires a prepared Demo corpus")
        selected_specs = self.specs[: int(self.selected_count or 0)]
        if instance_ids is not None:
            requested_ids = {str(instance_id) for instance_id in instance_ids}
            selected_specs = tuple(
                spec for spec in selected_specs if spec.instance_id in requested_ids
            )
            if {spec.instance_id for spec in selected_specs} != requested_ids:
                raise ValueError("instance_ids must refer to selected server instances")
        else:
            self._truncated_count = 0
        profiles = tuple(
            RosterProfile(
                bot_id=f"bot-{index:02d}",
                team=2 if index % 2 == 0 else 3,
            )
            for index in range(self.max_bots)
        )
        mp_context = mp.get_context("spawn")
        for spec in selected_specs:
            transport = self._new_transport(spec)
            epoch = int(transport.refresh_epoch())
            stop_event = mp_context.Event()
            rollout_queue = mp_context.Queue()
            metrics_queue = mp_context.Queue()
            error_queue = mp_context.Queue()
            process = mp_context.Process(
                target=_run_selfplay_process,
                name=f"sl-bots-selfplay-{spec.instance_id}",
                args=(
                    spec,
                    int(generation),
                    self._match_id(generation, spec.instance_id),
                    self.data_root,
                    self.purpose,
                    self.corpus_manifest,
                    Path(generation_info.onnx_path),
                    epoch,
                    profiles,
                    rollout_queue,
                    metrics_queue,
                    error_queue,
                    stop_event,
                    self._config_int("rollout_horizon", 1024),
                    self._config_int("selfplay_wait_timeout_ms", 100),
                    self._config_int("ipc_capacity", 64),
                    self._config_float("ipc_connect_timeout_s", 60.0),
                    self.max_bots,
                ),
            )
            session = _LiveSession(
                spec=spec,
                transport=transport,
                stop_event=stop_event,
                rollout_queue=rollout_queue,
                metrics_queue=metrics_queue,
                process=process,
                error_queue=error_queue,
                started_ns=time.perf_counter_ns(),
            )
            self._sessions[spec.instance_id] = session
            process.start()

    def _new_transport(self, spec: ServerInstanceSpecV1) -> Any:
        if self.transport_factory is not None:
            return self.transport_factory(spec)
        last_error: BaseException | None = None
        capacity = self._config_int("ipc_capacity", 64)
        deadline = time.monotonic() + self._config_float("ipc_connect_timeout_s", 60.0)
        while time.monotonic() < deadline:
            for epoch in range(1, 65):
                try:
                    return Win32SharedMemoryTransportV1(
                        spec.ipc_name,
                        capacity=capacity,
                        epoch=epoch,
                        open_existing=True,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    last_error = error
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        raise RuntimeError(
            f"unable to open SourcePawn shared-memory transport for {spec.instance_id}"
        ) from last_error

    def _drain_all(self) -> None:
        for session in tuple(self._sessions.values()):
            self._drain_session(session)

    def _drain_session(self, session: _LiveSession) -> None:
        while True:
            try:
                error = session.error_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(error, Mapping):
                session.error_box.append(
                    f"{error.get('type', 'ChildProcessError')}: "
                    f"{error.get('message', '')}"
                )
            else:
                session.error_box.append(str(error))
            session.process_failed = True
        while True:
            try:
                envelope = session.rollout_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(envelope, RolloutEnvelopeV1):
                raise TypeError("self-play worker emitted a non-envelope rollout")
            session.envelopes.append(envelope)
            if envelope.terminal:
                session.terminal_seen = True
            if envelope.truncated:
                self._truncated_count += 1
        while True:
            try:
                metric = session.metrics_queue.get_nowait()
            except queue.Empty:
                break
            self._record_tick_telemetry(session, metric)
            if bool(metric.get("source_action_apply_ack")):
                session.ack_count += 1
                latency = metric.get("latency_us")
                if latency is not None and math.isfinite(float(latency)):
                    session.latencies_us.append(float(latency))
            session.fallback_count += int(metric.get("source_fallback_count", 0))
            session.runtime_fallback_count += int(metric.get("fallback_count", 0))
            session.rules_validated = session.rules_validated or bool(
                metric.get("rules_validated")
            )
            if (
                metric.get("source_policy_generation") is not None
                and int(metric["source_policy_generation"]) != int(metric.get("policy_generation", -1))
            ):
                session.invalid_generation += 1
            session.process_failed = session.process_failed or bool(metric.get("runtime_process_failed"))
        exitcode = getattr(session.process, "exitcode", None)
        if exitcode not in (None, 0):
            session.process_failed = True

    @staticmethod
    def _record_tick_telemetry(session: _LiveSession, metric: Mapping[str, Any]) -> None:
        try:
            server_tick = int(metric["server_tick"])
            tick_ns = int(metric["read_completed_ns"])
        except (KeyError, TypeError, ValueError, OverflowError):
            session.process_failed = True
            return
        if session.last_server_tick is not None and server_tick <= session.last_server_tick:
            return
        if session.first_server_tick is None:
            session.first_server_tick = server_tick
            session.first_tick_ns = tick_ns
            session.ticks = 1
        else:
            session.ticks = min(server_tick - session.first_server_tick + 1, 128)
        session.last_server_tick = server_tick
        session.last_tick_ns = tick_ns
        if session.first_tick_ns is not None and tick_ns > session.first_tick_ns:
            elapsed_s = (tick_ns - session.first_tick_ns) / 1_000_000_000.0
            session.tickrate_hz = (server_tick - session.first_server_tick) / elapsed_s

    def _unavailable_sessions(self) -> tuple[Mapping[str, str], ...]:
        unavailable: list[Mapping[str, str]] = []
        for session in self._sessions.values():
            if session.process.is_alive() or session.terminal_seen:
                continue
            reason = str(session.error_box[-1]) if session.error_box else "worker exited before terminal match"
            unavailable.append({"instance_id": session.spec.instance_id, "reason": reason})
        return tuple(unavailable)

    def _all_session_envelopes(self) -> tuple[RolloutEnvelopeV1, ...]:
        return tuple(
            envelope
            for instance_id in sorted(self._sessions)
            for envelope in self._sessions[instance_id].envelopes
        )

    def _wait_for_terminal_matches(self, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self._drain_all()
            unavailable = self._unavailable_sessions()
            if unavailable:
                raise RuntimeError(f"bootstrap server failed: {unavailable[0]['reason']}")
            if self._sessions and all(session.terminal_seen for session in self._sessions.values()):
                self._last_match_envelopes = self._all_session_envelopes()
                self._stop_workers()
                return
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        self._drain_all()
        raise TimeoutError("bootstrap matches did not reach terminal Get5 state before the deadline")

    def _stop_workers(self) -> None:
        sessions = tuple(self._sessions.values())
        for session in sessions:
            self._stop_session(session)
        self._sessions.clear()

    def _stop_session(self, session: _LiveSession) -> None:
        stopped_ns = time.perf_counter_ns()
        session.stop_event.set()
        if session.process.is_alive():
            session.process.join(timeout=5.0)
        if session.process.is_alive():
            abort = getattr(session.transport, "abort", None)
            if callable(abort):
                abort()
            session.process.join(timeout=2.0)
        if session.process.is_alive():
            terminate = getattr(session.process, "terminate", None)
            if callable(terminate):
                terminate()
            session.process.join(timeout=2.0)
        started_ns = session.started_ns or stopped_ns
        self._last_match_durations[session.spec.instance_id] = max(
            0.0,
            (stopped_ns - started_ns) / 1_000_000_000.0,
        )
        self._drain_session(session)
        close = getattr(session.transport, "close", None)
        if callable(close):
            close()

    def _session_duration(self, instance_id: str) -> float:
        session = self._sessions.get(str(instance_id))
        if session is not None:
            return max(
                0.0,
                (time.perf_counter_ns() - session.started_ns) / 1_000_000_000.0,
            )
        return float(self._last_match_durations.get(str(instance_id), 0.0))

    def _match_id(self, generation: int, instance_id: str) -> str:
        return f"slbots-{self.run_id}-g{int(generation):03d}-i{instance_id}"


def _state_dict_hash(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict, key=str):
        value = state_dict[name]
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(name).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(str(name).encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


__all__ = ["ProductionTrainingPipeline"]
