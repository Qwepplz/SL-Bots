"""Command-line entry point for the SL-Bots toolchain."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from . import __version__
from .contracts import DataPurpose, Phase, ensure_purpose
from .demo_pipeline import ingest_demo
from .lineage import DatasetManifestV1, ProductionLineageError


Command = tuple[str, ...]

_COMMANDS: tuple[Command, ...] = (
    ("demo", "ingest"),
    ("train", "bc"),
    ("train", "selfplay"),
    ("export",),
    ("runtime", "serve"),
)
_HUMAN_TARGET_NAMES = frozenset(
    {
        "max_angular_velocity_deg_s",
        "max_angular_acceleration_deg_s2",
        "stop_go_ratio",
        "economy_choice_count",
        "utility_event_rate",
    }
)


def _add_data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="根数据目录；所有数据、检查点和运行产物必须位于其下。",
    )


def _add_leaf(
    parent: argparse._SubParsersAction[argparse.ArgumentParser],
    command: Command,
) -> argparse.ArgumentParser:
    parser = parent.add_parser(" ".join(command[1:]) if len(command) > 1 else command[0])
    _add_data_root(parser)
    parser.set_defaults(command_path=command)
    if command == ("demo", "ingest"):
        parser.add_argument("--input", type=Path, help="GOTV Demo 文件")
        parser.add_argument(
            "--purpose",
            choices=[purpose.value for purpose in DataPurpose],
            default=DataPurpose.TEST_ONLY.value,
        )
        parser.add_argument("--extractor", type=Path)
        parser.add_argument("--allow-header-only", action="store_true")
        parser.add_argument("--manifest-output", type=Path)
    elif command == ("train", "bc"):
        parser.add_argument("--config", type=Path)
        parser.add_argument("--dataset-path", type=Path)
        parser.add_argument("--manifest", type=Path)
        parser.add_argument(
            "--purpose",
            choices=[purpose.value for purpose in DataPurpose],
            default=DataPurpose.TEST_ONLY.value,
        )
        parser.add_argument("--output-dir", type=Path)
    elif command == ("train", "selfplay"):
        parser.add_argument("--model", type=Path)
        parser.add_argument("--checkpoint", type=Path)
        parser.add_argument("--training-manifest", type=Path)
        parser.add_argument("--gail-checkpoint", type=Path)
        parser.add_argument("--reference-checkpoint", type=Path)
        parser.add_argument("--human-like-targets", type=Path)
        parser.add_argument("--gail-coefficient", type=float, default=0.1)
        parser.add_argument("--kl-coefficient", type=float, default=0.01)
        parser.add_argument("--constraint-coefficient", type=float, default=0.1)
        parser.add_argument("--ipc-name", default="SLBots_de_mirage")
        parser.add_argument("--epoch", type=int, default=1)
        parser.add_argument("--capacity", type=int, default=64)
        parser.add_argument("--max-bots", type=int, default=10)
        parser.add_argument("--bot-count", type=int, default=10)
        parser.add_argument("--phase", choices=["warmup", "knife", "live"], default="live")
        parser.add_argument("--ticks", type=int, default=1)
        parser.add_argument("--mappo-steps", type=int, default=1)
        parser.add_argument("--episode-id")
        parser.add_argument(
            "--purpose",
            choices=[purpose.value for purpose in DataPurpose],
            default=DataPurpose.TEST_ONLY.value,
        )
    elif command == ("export",):
        parser.add_argument("--checkpoint", type=Path)
        parser.add_argument("--manifest", type=Path)
        parser.add_argument(
            "--purpose",
            choices=[purpose.value for purpose in DataPurpose],
            default=DataPurpose.TEST_ONLY.value,
        )
        parser.add_argument("--output", type=Path)
    elif command == ("runtime", "serve"):
        parser.add_argument("--model", type=Path)
        parser.add_argument("--ipc-name", default="SLBots_de_mirage")
        parser.add_argument("--epoch", type=int, default=1)
        parser.add_argument("--capacity", type=int, default=64)
        parser.add_argument("--max-bots", type=int, default=10)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sl-bots")
    parser.add_argument(
        "--version",
        action="version",
        version=f"sl-bots {__version__}",
    )
    top = parser.add_subparsers(dest="group", required=True)

    demo = top.add_parser("demo", help="GOTV Demo 数据管线")
    demo_sub = demo.add_subparsers(dest="operation", required=True)
    _add_leaf(demo_sub, ("demo", "ingest"))

    train = top.add_parser("train", help="训练管线")
    train_sub = train.add_subparsers(dest="algorithm", required=True)
    _add_leaf(train_sub, ("train", "bc"))
    _add_leaf(train_sub, ("train", "selfplay"))

    _add_leaf(top, ("export",))

    runtime = top.add_parser("runtime", help="在线推理运行时")
    runtime_sub = runtime.add_subparsers(dest="operation", required=True)
    _add_leaf(runtime_sub, ("runtime", "serve"))

    return parser


def _manifest_to_dict(manifest: DatasetManifestV1) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "purpose": manifest.purpose.value,
        "parents": [_manifest_to_dict(parent) for parent in manifest.parents],
        "artifact_type": manifest.artifact_type,
        "source_sha256": manifest.source_sha256,
        "parser_version": manifest.parser_version,
        "projection_version": manifest.projection_version,
        "metadata": dict(manifest.metadata),
    }


def _manifest_from_dict(value: Mapping[str, Any]) -> DatasetManifestV1:
    parents = tuple(_manifest_from_dict(parent) for parent in value.get("parents", ()))
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _load_dataset_manifest(args: argparse.Namespace) -> DatasetManifestV1:
    if args.manifest is not None:
        return _manifest_from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
    if args.dataset_path is None:
        raise ValueError("--dataset-path or --manifest is required")
    return DatasetManifestV1(
        name=args.dataset_path.stem,
        purpose=ensure_purpose(args.purpose),
        artifact_type="action_label_parquet",
        metadata={"path": str(args.dataset_path)},
    )


def _load_training_manifest(path: Path) -> Any:
    from .training_bc import TrainingRunManifestV1

    value = json.loads(path.read_text(encoding="utf-8"))
    return TrainingRunManifestV1(
        name=str(value["name"]),
        purpose=ensure_purpose(value["purpose"]),
        dataset_manifest=_manifest_from_dict(value["dataset_manifest"]),
        steps=int(value["steps"]),
        loss_history=tuple(float(item) for item in value.get("loss_history", ())),
        checkpoint_path=str(value["checkpoint_path"]),
        config_sha256=str(value["config_sha256"]),
        metadata={str(key): str(item) for key, item in value.get("metadata", {}).items()},
    )


def _load_checkpoint_run(path: Path) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for self-play training") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("self-play checkpoint must contain a mapping payload")
    dataset_manifest = checkpoint.get("dataset_manifest")
    if not isinstance(dataset_manifest, DatasetManifestV1):
        raise ValueError("self-play checkpoint must contain an embedded DatasetManifestV1")
    return SimpleNamespace(
        name=path.stem,
        purpose=dataset_manifest.effective_purpose(),
        dataset_manifest=dataset_manifest,
        checkpoint_path=str(path),
        steps=0,
        loss_history=tuple(),
        config_sha256="0" * 64,
    )


def _load_gail_discriminator(
    path: Path,
    expected_manifest: DatasetManifestV1,
    target_purpose: DataPurpose,
) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for self-play constraints") from error
    from .training_gail import WindowDiscriminator

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("GAIL checkpoint must contain a mapping payload")
    dataset_manifest = checkpoint.get("dataset_manifest")
    if not isinstance(dataset_manifest, DatasetManifestV1):
        raise ProductionLineageError("GAIL checkpoint must contain an embedded DatasetManifestV1")
    if dataset_manifest.effective_purpose() is not expected_manifest.effective_purpose():
        raise ProductionLineageError(
            "GAIL checkpoint lineage does not match the self-play actor lineage"
        )
    dataset_manifest.assert_exportable(target_purpose)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("GAIL checkpoint does not contain model_state_dict")
    config = checkpoint.get("model_config", {})
    if not isinstance(config, Mapping):
        raise ValueError("GAIL checkpoint model_config must be a mapping")
    discriminator = WindowDiscriminator(hidden_size=int(config.get("hidden_size", 128)))
    discriminator.load_state_dict(state_dict)
    discriminator.eval()
    return discriminator


def _load_human_like_targets(
    path: Path,
    expected_manifest: DatasetManifestV1,
    target_purpose: DataPurpose,
) -> Mapping[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("human-like targets must be valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("human-like targets must be a JSON object")
    source_manifest = payload.get("dataset_manifest")
    if source_manifest is not None:
        if isinstance(source_manifest, Mapping):
            source_manifest = _manifest_from_dict(source_manifest)
        if not isinstance(source_manifest, DatasetManifestV1):
            raise ProductionLineageError("human-like target dataset_manifest is invalid")
        if source_manifest.effective_purpose() is not expected_manifest.effective_purpose():
            raise ProductionLineageError(
                "human-like target lineage does not match the self-play actor lineage"
            )
        source_manifest.assert_exportable(target_purpose)
    elif "purpose" in payload:
        declared_purpose = ensure_purpose(payload["purpose"])
        if declared_purpose is not expected_manifest.effective_purpose():
            raise ProductionLineageError(
                "human-like target purpose does not match the self-play actor lineage"
            )
        if target_purpose is DataPurpose.PRODUCTION and declared_purpose is DataPurpose.TEST_ONLY:
            raise ProductionLineageError("test-only human-like targets cannot drive production training")
    values = payload.get("targets", payload.get("metrics", payload))
    if not isinstance(values, Mapping) or not values:
        raise ValueError("human-like targets must contain a non-empty targets object")
    targets: dict[str, float] = {}
    for name, value in values.items():
        if name in {"dataset_manifest", "purpose", "targets"}:
            continue
        normalized_name = str(name)
        if normalized_name.startswith("human_"):
            normalized_name = normalized_name[6:]
        if normalized_name not in _HUMAN_TARGET_NAMES:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"human-like target {name!r} must be numeric") from error
        if not math.isfinite(numeric):
            raise ValueError(f"human-like target {name!r} must be finite")
        targets[normalized_name] = numeric
    if not targets:
        raise ValueError("human-like targets must contain at least one numeric target")
    return targets


def _load_selfplay_constraints(
    args: argparse.Namespace,
    run_manifest: Any,
    actor: Any,
) -> tuple[Any, Any, Mapping[str, float]]:
    if args.gail_checkpoint is None:
        raise ValueError("--gail-checkpoint is required for formal self-play training")
    if args.human_like_targets is None:
        raise ValueError("--human-like-targets is required for formal self-play training")
    source_manifest = run_manifest.dataset_manifest
    target_purpose = ensure_purpose(args.purpose)
    discriminator = _load_gail_discriminator(
        args.gail_checkpoint,
        source_manifest,
        target_purpose,
    )
    if args.reference_checkpoint is None:
        reference_actor = copy.deepcopy(actor)
    else:
        reference_run = _load_checkpoint_run(args.reference_checkpoint)
        reference_manifest = _checkpoint_dataset_manifest(reference_run)
        if reference_manifest is None:
            raise ProductionLineageError(
                "reference checkpoint must contain an embedded DatasetManifestV1"
            )
        if reference_manifest.effective_purpose() is not source_manifest.effective_purpose():
            raise ProductionLineageError(
                "reference checkpoint lineage does not match the self-play actor lineage"
            )
        reference_manifest.assert_exportable(target_purpose)
        reference_actor = _load_actor(reference_run, None)
    if hasattr(reference_actor, "eval"):
        reference_actor.eval()
    targets = _load_human_like_targets(
        args.human_like_targets,
        source_manifest,
        target_purpose,
    )
    return discriminator, reference_actor, targets


def _run_demo_ingest(args: argparse.Namespace) -> int:
    if args.input is None:
        raise ValueError("--input is required for demo ingest")
    manifest = ingest_demo(
        args.input,
        args.data_root,
        ensure_purpose(args.purpose),
        extractor_path=args.extractor,
        allow_header_only=args.allow_header_only,
    )
    payload = _manifest_to_dict(manifest)
    if args.manifest_output is not None:
        _write_json(args.manifest_output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _run_bc(args: argparse.Namespace) -> int:
    from .training_bc import load_config, train_bc

    dataset_manifest = _load_dataset_manifest(args)
    config: dict[str, Any] = load_config(args.config) if args.config is not None else {}
    if args.dataset_path is not None:
        config.setdefault("dataset_path", str(args.dataset_path))
    output_dir = args.output_dir or args.data_root / "training" / "bc"
    config.setdefault("output_dir", str(output_dir))
    run = train_bc(config, dataset_manifest, output_dir=output_dir)
    print(json.dumps({
        "name": run.name,
        "purpose": run.purpose.value,
        "checkpoint_path": run.checkpoint_path,
        "steps": run.steps,
        "loss_history": list(run.loss_history),
        "config_sha256": run.config_sha256,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _run_selfplay(args: argparse.Namespace) -> int:
    from .runtime import RuntimeService, Win32SharedMemoryTransportV1
    from .selfplay import RosterProfile, SelfPlayController, SharedMemorySelfPlayAdapter
    from .export import (
        _checkpoint_dataset_manifest,
        _load_actor,
        validate_model_checkpoint_binding,
    )
    from .training_mappo import CentralValueCritic, train_mappo

    if args.epoch < 0 or args.ticks < 0:
        raise ValueError("--epoch and --ticks must be non-negative")
    if args.mappo_steps <= 0:
        raise ValueError("--mappo-steps must be positive")
    if args.model is None:
        raise ValueError("--model is required for self-play inference")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.training_manifest is not None:
        run_manifest = _load_training_manifest(args.training_manifest)
    elif args.checkpoint is not None:
        run_manifest = _load_checkpoint_run(args.checkpoint)
    else:
        raise ValueError("--checkpoint or --training-manifest is required for self-play training")
    requested_purpose = ensure_purpose(args.purpose)
    source_manifest = _checkpoint_dataset_manifest(run_manifest)
    if source_manifest is None:
        raise ValueError("self-play checkpoint must contain an embedded DatasetManifestV1")
    if source_manifest != run_manifest.dataset_manifest:
        raise ValueError("training manifest does not match the checkpoint dataset manifest")
    validate_model_checkpoint_binding(args.model, run_manifest)
    source_manifest.assert_exportable(requested_purpose)
    training_purpose = source_manifest.effective_purpose()
    if requested_purpose is DataPurpose.PRODUCTION:
        training_purpose = DataPurpose.PRODUCTION
    if not 1 <= args.bot_count <= args.max_bots <= 10:
        raise ValueError("bot counts must satisfy 1 <= bot-count <= max-bots <= 10")
    phase = {"warmup": Phase.WARMUP, "knife": Phase.KNIFE, "live": Phase.LIVE}[args.phase]
    profiles = tuple(
        RosterProfile(bot_id=f"bot-{index}", team=2 if index % 2 == 0 else 3)
        for index in range(args.bot_count)
    )
    actor = _load_actor(run_manifest, None)
    behavior_actor = copy.deepcopy(actor)
    if hasattr(behavior_actor, "eval"):
        behavior_actor.eval()
    discriminator, reference_actor, human_like_targets = _load_selfplay_constraints(
        args,
        run_manifest,
        actor,
    )
    with Win32SharedMemoryTransportV1(args.ipc_name, capacity=args.capacity, epoch=args.epoch) as transport:
        runtime = (
            RuntimeService(transport=transport, model_path=args.model, max_bots=args.max_bots)
            if args.model is not None
            else None
        )
        adapter = SharedMemorySelfPlayAdapter(transport=transport, runtime=runtime)
        controller = SelfPlayController(
            data_root=args.data_root,
            purpose=training_purpose,
            source_manifest=source_manifest,
            adapter=adapter,
        )
        controller.start_episode(phase, profiles, episode_id=args.episode_id, epoch=args.epoch)
        for _ in range(args.ticks):
            controller.step(None)
            time.sleep(0.001)
        if not controller.transitions:
            raise RuntimeError("self-play produced no transitions; verify the server and model are running")
        manifest = controller.finish_episode()
        critic = CentralValueCritic()
        mappo_runs = []
        for index, segment in enumerate(controller.rollout_segments):
            if not segment:
                continue
            segment_manifest = controller.completed_manifests[index]
            snapshots = controller.critic_snapshot_segments[index]
            run = train_mappo(
                actor,
                segment,
                output_dir=args.data_root / "training" / "mappo" / segment_manifest.episode_id,
                purpose=training_purpose,
                steps=args.mappo_steps,
                parent_manifests=(segment_manifest.dataset, segment_manifest.critic_dataset),
                critic_snapshots=snapshots,
                critic=critic,
                behavior_actor=behavior_actor,
                discriminator=discriminator,
                reference_actor=reference_actor,
                human_like_targets=human_like_targets,
                gail_coefficient=args.gail_coefficient,
                kl_coefficient=args.kl_coefficient,
                constraint_coefficient=args.constraint_coefficient,
            )
            mappo_runs.append(run)
    print(json.dumps({
        "episode_id": manifest.episode_id,
        "path": str(manifest.path),
        "critic_path": str(manifest.critic_path),
        "transition_count": manifest.transition_count,
        "purpose": training_purpose.value,
        "mappo_runs": [
            {
                "checkpoint_path": run.checkpoint_path,
                "steps": run.steps,
                "loss_history": list(run.loss_history),
            }
            for run in mappo_runs
        ],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _run_export(args: argparse.Namespace) -> int:
    from .export import export_actor
    from .training_bc import TrainingRunManifestV1

    purpose = ensure_purpose(args.purpose)
    if args.manifest is not None:
        run = _load_training_manifest(args.manifest)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint or --manifest is required")
        dataset = DatasetManifestV1(
            name=args.checkpoint.stem,
            purpose=purpose,
            artifact_type="training_checkpoint",
            metadata={"checkpoint": str(args.checkpoint)},
        )
        run = TrainingRunManifestV1(
            name=args.checkpoint.stem,
            purpose=purpose,
            dataset_manifest=dataset,
            steps=0,
            loss_history=(),
            checkpoint_path=str(args.checkpoint),
            config_sha256="0" * 64,
        )
    output = args.output or args.data_root / "exports" / "actor.onnx"
    manifest = export_actor(run, output, purpose)
    print(json.dumps({
        "model_path": str(manifest.model_path),
        "metadata_path": str(manifest.metadata_path),
        "purpose": manifest.purpose.value,
        "sha256": manifest.sha256,
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _run_runtime(args: argparse.Namespace) -> int:
    from .runtime import RuntimeService, Win32SharedMemoryTransportV1

    if args.model is None:
        raise ValueError("--model is required for runtime serve")
    with Win32SharedMemoryTransportV1(args.ipc_name, capacity=args.capacity, epoch=args.epoch) as transport:
        RuntimeService(
            transport=transport,
            model_path=args.model,
            max_bots=args.max_bots,
        ).run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        command = tuple(args.command_path)
        if command == ("demo", "ingest"):
            return _run_demo_ingest(args)
        if command == ("train", "bc"):
            return _run_bc(args)
        if command == ("train", "selfplay"):
            return _run_selfplay(args)
        if command == ("export",):
            return _run_export(args)
        if command == ("runtime", "serve"):
            return _run_runtime(args)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    raise RuntimeError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
