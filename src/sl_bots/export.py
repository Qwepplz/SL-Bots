from __future__ import annotations

import hashlib
import json
import os
import subprocess
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    DataPurpose,
    HierarchicalPackageV2,
    HierarchicalPackageV3,
    MOVEMENT_STATE_SCHEMA_V3,
    ensure_purpose,
)
from .lineage import DatasetManifestV1, ProductionLineageError
from .model import RecurrentStateV1, MirageActor


@dataclass(frozen=True)
class ExportManifestV1:
    model_path: Path
    metadata_path: Path
    sha256: str
    purpose: DataPurpose
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        object.__setattr__(self, "metadata_path", Path(self.metadata_path))
        object.__setattr__(self, "purpose", ensure_purpose(self.purpose))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PolicyGenerationV1:
    number: int
    actor_checkpoint: Path
    critic_checkpoint: Path
    optimizer_checkpoint: Path
    onnx_path: Path
    behavior_sha256: str
    parent_generation: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.number, int) or isinstance(self.number, bool) or self.number < 0:
            raise ValueError("policy generation number must be a non-negative integer")
        if self.number == 0 and self.parent_generation is not None:
            raise ValueError("generation 0 cannot have a parent generation")
        if self.number > 0 and self.parent_generation != self.number - 1:
            raise ValueError("policy generation parent must be exactly number - 1")
        for name in ("actor_checkpoint", "critic_checkpoint", "optimizer_checkpoint", "onnx_path"):
            path = Path(getattr(self, name)).resolve()
            if not path.name:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, path)
        behavior_sha256 = str(self.behavior_sha256).lower()
        if len(behavior_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in behavior_sha256
        ):
            raise ValueError("behavior_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "behavior_sha256", behavior_sha256)

    def assert_publishable(self) -> None:
        for name in ("actor_checkpoint", "critic_checkpoint", "optimizer_checkpoint", "onnx_path"):
            path = getattr(self, name)
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size <= 0:
                raise ValueError(f"policy generation artifact is empty: {path}")
        actual = _sha256_file(self.onnx_path)
        if actual != self.behavior_sha256:
            raise ValueError("policy generation behavior hash does not match the ONNX artifact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-generation-v1",
            "number": self.number,
            "actor_checkpoint": str(self.actor_checkpoint),
            "critic_checkpoint": str(self.critic_checkpoint),
            "optimizer_checkpoint": str(self.optimizer_checkpoint),
            "onnx_path": str(self.onnx_path),
            "behavior_sha256": self.behavior_sha256,
            "parent_generation": self.parent_generation,
        }


def publish_policy_generation(
    generation: PolicyGenerationV1,
    pointer_path: str | Path,
) -> Path:
    if not isinstance(generation, PolicyGenerationV1):
        raise TypeError("generation must be PolicyGenerationV1")
    generation.assert_publishable()
    destination = Path(pointer_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(generation.to_dict(), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(str(destination.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def load_policy_generation(pointer_path: str | Path) -> PolicyGenerationV1:
    path = Path(pointer_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("policy generation pointer is not valid JSON") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != "policy-generation-v1":
        raise ValueError("policy generation pointer schema is invalid")
    generation = PolicyGenerationV1(
        number=int(payload["number"]),
        actor_checkpoint=Path(str(payload["actor_checkpoint"])),
        critic_checkpoint=Path(str(payload["critic_checkpoint"])),
        optimizer_checkpoint=Path(str(payload["optimizer_checkpoint"])),
        onnx_path=Path(str(payload["onnx_path"])),
        behavior_sha256=str(payload["behavior_sha256"]),
        parent_generation=(
            None if payload.get("parent_generation") is None else int(payload["parent_generation"])
        ),
    )
    generation.assert_publishable()
    return generation


def _manifest_to_dict(manifest: DatasetManifestV1) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "purpose": manifest.purpose.value,
        "artifact_type": manifest.artifact_type,
        "source_sha256": manifest.source_sha256,
        "parser_version": manifest.parser_version,
        "projection_version": manifest.projection_version,
        "metadata": dict(manifest.metadata),
        "parents": [_manifest_to_dict(parent) for parent in manifest.parents],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict, key=str):
        value = state_dict[name]
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.numpy().tobytes()
            descriptor = f"{tensor.dtype}|{tuple(tensor.shape)}".encode("utf-8")
        else:
            raw = repr(value).encode("utf-8")
            descriptor = type(value).__qualname__.encode("utf-8")
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(descriptor)
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_checkpoint_payload(run_manifest: Any) -> Mapping[str, Any]:
    checkpoint_path = Path(run_manifest.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for checkpoint lineage validation") from error
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ProductionLineageError("checkpoint must contain a mapping payload")
    return checkpoint


def _git_metadata(repo_root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {"git_head": run("rev-parse", "HEAD"), "git_status": run("status", "--short")}


def _load_actor(run_manifest: Any, actor: Any | None) -> Any:
    if actor is not None:
        return actor
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for ONNX export") from error
    checkpoint_path = Path(run_manifest.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = MirageActor(
        weapon_count=int(config.get("weapon_count", 16)),
        buy_count=int(config.get("buy_action_count", 32)),
    )
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("actor_state_dict")
    if state_dict is None:
        raise ValueError("checkpoint does not contain model_state_dict or actor_state_dict")
    model.load_state_dict(state_dict)
    return model


def _checkpoint_dataset_manifest(run_manifest: Any) -> DatasetManifestV1 | None:
    checkpoint = _load_checkpoint_payload(run_manifest)
    manifest = checkpoint.get("dataset_manifest")
    if manifest is None:
        return None
    if not isinstance(manifest, DatasetManifestV1):
        raise ProductionLineageError("checkpoint dataset_manifest is not a DatasetManifestV1")
    return manifest


def _checkpoint_actor_state_sha256(run_manifest: Any) -> str:
    checkpoint = _load_checkpoint_payload(run_manifest)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("actor_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ProductionLineageError(
            "checkpoint must contain model_state_dict or actor_state_dict for actor binding"
        )
    return _state_dict_sha256(state_dict)


def validate_model_checkpoint_binding(
    model_path: str | Path,
    run_manifest: Any,
) -> Mapping[str, Any]:
    """验证 self-play 使用的 ONNX 包来自同一份训练 checkpoint。"""

    model = Path(model_path)
    metadata_path = model.with_suffix(".json")
    if not metadata_path.is_file():
        raise ProductionLineageError(
            f"actor package metadata is required for self-play: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionLineageError("actor package metadata is not valid JSON") from error
    if not isinstance(metadata, Mapping) or metadata.get("format") != "sl-bots-actor-package-v1":
        raise ProductionLineageError("actor package metadata format is invalid")

    model_sha256 = _sha256_file(model)
    if metadata.get("sha256") != model_sha256:
        raise ProductionLineageError("actor package model hash does not match the ONNX file")

    checkpoint_path = Path(run_manifest.checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise ProductionLineageError("actor package checkpoint hash does not match the checkpoint")

    checkpoint_manifest = _checkpoint_dataset_manifest(run_manifest)
    if checkpoint_manifest is None:
        raise ProductionLineageError("actor package binding requires checkpoint dataset lineage")
    if metadata.get("checkpoint_dataset_manifest") != _manifest_to_dict(checkpoint_manifest):
        raise ProductionLineageError("actor package dataset lineage does not match the checkpoint")

    actor_state_sha256 = _checkpoint_actor_state_sha256(run_manifest)
    if metadata.get("actor_state_sha256") != actor_state_sha256:
        raise ProductionLineageError("actor package actor weights do not match the checkpoint")
    _validate_exported_actor_outputs(model, run_manifest)
    return metadata


def _validate_exported_actor_outputs(model_path: Path, run_manifest: Any) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch, NumPy, and ONNX Runtime are required for actor/checkpoint binding"
        ) from error

    actor = _load_actor(run_manifest, None)
    actor.eval()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for observation in (
        torch.zeros(1, 256, dtype=torch.float32),
        torch.linspace(0.0, 1.0, steps=256, dtype=torch.float32).reshape(1, 256),
    ):
        state = actor.initial_state(1)
        with torch.no_grad():
            output = actor(observation, state)
        expected = (
            output.movement_alpha,
            output.movement_beta,
            output.action_vector[:, :5],
            output.mouse_loc,
            output.mouse_scale,
            output.mouse_mix_logits,
            output.button_logits,
            output.weapon_logits,
            output.buy_logits,
            output.recurrent_state.tactical,
            output.recurrent_state.action,
            output.recurrent_state.tick,
        )
        actual = session.run(
            None,
            {
                "observation": observation.numpy(),
                "tactical_state": state.tactical.numpy(),
                "action_state": state.action.numpy(),
                "tick": state.tick.numpy(),
            },
        )
        if len(actual) != len(expected) or any(
            tuple(value.shape) != tuple(expected_value.shape)
            or not np.allclose(
                value,
                expected_value.detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-5,
            )
            for value, expected_value in zip(actual, expected)
        ):
            raise ProductionLineageError(
                "actor package outputs do not match the checkpoint actor weights"
            )


def _export_onnx(actor: Any, output_path: Path) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for ONNX export") from error

    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, observation: Any, tactical: Any, action: Any, tick: Any) -> Any:
            state = RecurrentStateV1(tactical=tactical, action=action, tick=tick)
            output = self.model(observation, state)
            return (
                output.movement_alpha,
                output.movement_beta,
                output.action_vector[:, :5],
                output.mouse_loc,
                output.mouse_scale,
                output.mouse_mix_logits,
                output.button_logits,
                output.weapon_logits,
                output.buy_logits,
                output.recurrent_state.tactical,
                output.recurrent_state.action,
                output.recurrent_state.tick,
            )

    actor.eval()
    wrapper = OnnxWrapper(actor)
    dummy_observation = torch.zeros(1, 256, dtype=torch.float32)
    dummy_tactical = torch.zeros(1, actor.tactical_hidden_size, dtype=torch.float32)
    dummy_action = torch.zeros(1, actor.action_hidden_size, dtype=torch.float32)
    dummy_tick = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_observation, dummy_tactical, dummy_action, dummy_tick),
        output_path,
        input_names=["observation", "tactical_state", "action_state", "tick"],
        output_names=[
            "movement_alpha",
            "movement_beta",
            "action_vector",
            "mouse_loc",
            "mouse_scale",
            "mouse_mix_logits",
            "button_logits",
            "weapon_logits",
            "buy_logits",
            "next_tactical_state",
            "next_action_state",
            "next_tick",
        ],
        dynamic_axes={
            "movement_alpha": {0: "batch"},
            "movement_beta": {0: "batch"},
            "observation": {0: "batch"},
            "tactical_state": {0: "batch"},
            "action_state": {0: "batch"},
            "tick": {0: "batch"},
            "action_vector": {0: "batch"},
            "mouse_loc": {0: "batch"},
            "mouse_scale": {0: "batch"},
            "mouse_mix_logits": {0: "batch"},
            "button_logits": {0: "batch"},
            "weapon_logits": {0: "batch"},
            "buy_logits": {0: "batch"},
            "next_tactical_state": {0: "batch"},
            "next_action_state": {0: "batch"},
            "next_tick": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def compare_runtime_outputs(
    torch_output: Sequence[float],
    onnx_output: Sequence[float],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    if len(torch_output) != len(onnx_output):
        return False
    return all(
        abs(float(left) - float(right)) <= atol + rtol * abs(float(left))
        for left, right in zip(torch_output, onnx_output)
    )


def export_actor(
    run_manifest: Any,
    output_path: str | Path,
    purpose: DataPurpose | str,
    *,
    actor: Any | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> ExportManifestV1:
    target = ensure_purpose(purpose)
    if not hasattr(run_manifest, "assert_exportable"):
        raise TypeError("run_manifest must expose assert_exportable")
    run_manifest.assert_exportable(target)
    if target is DataPurpose.PRODUCTION and getattr(run_manifest, "purpose", None) is not DataPurpose.PRODUCTION:
        raise ProductionLineageError("only production lineage can create a production actor package")
    destination = Path(output_path)
    if destination.suffix.lower() != ".onnx":
        raise ValueError("output_path must have an .onnx suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest: DatasetManifestV1 | None = None
    checkpoint_path = Path(getattr(run_manifest, "checkpoint_path", ""))
    if actor is None:
        checkpoint_manifest = _checkpoint_dataset_manifest(run_manifest)
        if checkpoint_manifest is None and target is DataPurpose.PRODUCTION:
            raise ProductionLineageError("production export requires checkpoint dataset lineage")
        if checkpoint_manifest is not None:
            run_purpose = ensure_purpose(getattr(run_manifest, "purpose", target))
            checkpoint_purpose = checkpoint_manifest.effective_purpose()
            if checkpoint_purpose is not run_purpose:
                raise ProductionLineageError(
                    "checkpoint dataset purpose does not match training run purpose"
                )
            checkpoint_manifest.assert_exportable(target)
    model = _load_actor(run_manifest, actor)
    checkpoint_binding: dict[str, Any] = {}
    if checkpoint_path.is_file():
        if checkpoint_manifest is None:
            checkpoint_manifest = _checkpoint_dataset_manifest(run_manifest)
        checkpoint_state_sha256 = _checkpoint_actor_state_sha256(run_manifest)
        model_state_sha256 = _state_dict_sha256(model.state_dict())
        if checkpoint_state_sha256 != model_state_sha256:
            raise ProductionLineageError(
                "export actor weights do not match the training checkpoint"
            )
        if checkpoint_manifest is None:
            raise ProductionLineageError("checkpoint dataset lineage is required for actor binding")
        checkpoint_binding = {
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "actor_state_sha256": checkpoint_state_sha256,
            "checkpoint_dataset_manifest": _manifest_to_dict(checkpoint_manifest),
        }
    _export_onnx(model, destination)
    payload = destination.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    repo_root = Path(__file__).resolve().parents[2]
    training_dataset = getattr(run_manifest, "dataset_manifest", None)
    metadata: dict[str, Any] = {
        "format": "sl-bots-actor-package-v1",
        "model_architecture": "MirageActor",
        "map": "de_mirage",
        "observation_schema": "BotObservationV1/256",
        "action_schema": "BotActionV1/36",
        "onnx_outputs": [
            "movement_alpha",
            "movement_beta",
            "action_vector",
            "mouse_loc",
            "mouse_scale",
            "mouse_mix_logits",
            "button_logits",
            "weapon_logits",
            "buy_logits",
            "next_tactical_state",
            "next_action_state",
            "next_tick",
        ],
        "recurrent_state_schema": {
            "tactical": "float32[256]",
            "action": "float32[128]",
            "tick": "int64[1]",
        },
        "purpose": target.value,
        "training_run": {
            "name": getattr(run_manifest, "name", ""),
            "purpose": getattr(getattr(run_manifest, "purpose", None), "value", None),
            "steps": getattr(run_manifest, "steps", None),
            "loss_history": list(getattr(run_manifest, "loss_history", ())),
            "metrics": dict(getattr(run_manifest, "metrics", {})),
            "config_sha256": getattr(run_manifest, "config_sha256", ""),
            "checkpoint_path": getattr(run_manifest, "checkpoint_path", ""),
        },
        "training_lineage": _manifest_to_dict(training_dataset) if isinstance(training_dataset, DatasetManifestV1) else {},
        "metrics": dict(metrics or {}),
        "sha256": sha256,
        **checkpoint_binding,
        **_git_metadata(repo_root),
    }
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return ExportManifestV1(destination, metadata_path, sha256, target, metadata)


def export_hierarchical_package(
    run_manifest: Any,
    output_dir: str | Path,
    purpose: DataPurpose | str,
    *,
    actor: Any,
    generation: int = 0,
    parent_generation: int | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> HierarchicalPackageV2:
    """Export and atomically describe the test-only decision/action pair."""

    target = ensure_purpose(purpose)
    if target is not DataPurpose.TEST_ONLY:
        raise ProductionLineageError("hierarchical package export is test_only only")
    if not hasattr(run_manifest, "assert_exportable"):
        raise TypeError("run_manifest must expose assert_exportable")
    run_manifest.assert_exportable(target)
    if ensure_purpose(getattr(run_manifest, "purpose", target)) is not DataPurpose.TEST_ONLY:
        raise ProductionLineageError("hierarchical package lineage must be test_only")
    if not hasattr(actor, "decision") or not hasattr(actor, "action"):
        raise TypeError("actor must expose decision and action branches")

    from .quantization import (
        contains_fake_quant,
        export_action_fp32,
        export_decision_int8,
        prepare_action_fp32,
        validate_action_fp32_graph,
        validate_decision_int8_graph,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    decision = copy.deepcopy(actor.decision).cpu()
    action = copy.deepcopy(actor.action).cpu()
    if not contains_fake_quant(decision):
        raise ValueError(
            "hierarchical package export requires a decision model trained with QAT; "
            "export-time fake-quant preparation is not accepted"
        )
    run_metrics = getattr(run_manifest, "metrics", {})
    export_metrics = metrics if isinstance(metrics, Mapping) else {}
    qat_proven = isinstance(run_metrics, Mapping) and bool(
        run_metrics.get("decision_qat_enabled", False)
    )
    qat_proven = qat_proven or bool(export_metrics.get("decision_qat_enabled", False))
    checkpoint_path = Path(getattr(run_manifest, "checkpoint_path", ""))
    if checkpoint_path.is_file():
        try:
            import torch
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError("PyTorch is required to verify decision QAT provenance") from error
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, Mapping):
            qat_state = checkpoint.get("qat_state", checkpoint.get("decision_qat_state"))
            qat_proven = qat_proven or (
                isinstance(qat_state, Mapping) and bool(qat_state.get("enabled", False))
            )
    if not qat_proven:
        raise ValueError(
            "hierarchical package export is missing decision QAT training provenance"
        )
    prepare_action_fp32(action)
    decision_path = destination / "decision.int8.onnx"
    action_path = destination / "action.fp32.onnx"
    export_decision_int8(decision, decision_path)
    export_action_fp32(action, action_path)
    decision_report = validate_decision_int8_graph(decision_path)
    action_report = validate_action_fp32_graph(action_path)
    training_dataset = getattr(run_manifest, "dataset_manifest", None)
    training_lineage = (
        _manifest_to_dict(training_dataset)
        if isinstance(training_dataset, DatasetManifestV1)
        else {"purpose": DataPurpose.TEST_ONLY.value}
    )
    package_metrics = _json_safe({
        "decision_graph": decision_report,
        "action_graph": action_report,
        **dict(metrics or {}),
    })
    package = HierarchicalPackageV2(
        generation=generation,
        parent_generation=parent_generation,
        decision_path=decision_path,
        action_path=action_path,
        purpose=target,
        decision_sha256=_sha256_file(decision_path),
        action_sha256=_sha256_file(action_path),
        state_schema={
            "decision_memory": "float32[batch,32,512]",
            "action_hidden": "float32[batch,384]",
            "cached_intent": "float32[batch,128]",
            "last_decision_tick": "int64[batch]",
        },
        decision_parameter_count=sum(parameter.numel() for parameter in decision.parameters()),
        action_parameter_count=sum(parameter.numel() for parameter in action.parameters()),
        metadata={
            "purpose": target.value,
            "training_lineage": training_lineage,
            "training_run": {
                "name": getattr(run_manifest, "name", ""),
                "steps": getattr(run_manifest, "steps", None),
                "config_sha256": getattr(run_manifest, "config_sha256", ""),
                "checkpoint_path": str(getattr(run_manifest, "checkpoint_path", "")),
            },
            "decision_qat_proven": True,
            "metrics": package_metrics,
        },
    )
    package.assert_loadable()
    (destination / "package-v2.json").write_text(
        json.dumps(package.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (destination / "metrics.json").write_text(
        json.dumps(package_metrics, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return package


def export_hierarchical_package_v3(
    run_manifest: Any,
    output_dir: str | Path,
    purpose: DataPurpose | str,
    *,
    movement_model: Any,
    action_model: Any,
    generation: int = 0,
    parent_generation: int | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> HierarchicalPackageV3:
    """Export a test-only v3 movement/reaction pair after the offline gate."""

    target = ensure_purpose(purpose)
    if target is not DataPurpose.TEST_ONLY:
        raise ProductionLineageError("movement v3 package export is test_only only")
    if not hasattr(run_manifest, "assert_exportable"):
        raise TypeError("run_manifest must expose assert_exportable")
    run_manifest.assert_exportable(target)
    if ensure_purpose(getattr(run_manifest, "purpose", target)) is not DataPurpose.TEST_ONLY:
        raise ProductionLineageError("movement v3 package lineage must be test_only")

    run_metrics = getattr(run_manifest, "metrics", {})
    export_metrics = dict(metrics or {})
    offline_gate_passed = bool(
        (run_metrics.get("offline_gate_passed", False) if isinstance(run_metrics, Mapping) else False)
        or export_metrics.get("offline_gate_passed", False)
    )
    if not offline_gate_passed:
        raise ValueError("movement v3 package export requires offline gate evidence")

    from .quantization import (
        contains_fake_quant,
        export_movement_int8,
        export_reactive_action_fp32,
        prepare_action_fp32,
        validate_movement_int8_graph,
        validate_reactive_action_fp32_graph,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    movement = copy.deepcopy(movement_model).cpu()
    action = copy.deepcopy(action_model).cpu()
    if not contains_fake_quant(movement):
        raise ValueError(
            "movement v3 package export requires a movement model trained with QAT; "
            "export-time fake-quant preparation is not accepted"
        )
    prepare_action_fp32(action)
    movement_path = destination / "movement.int8.onnx"
    action_path = destination / "action.fp32.onnx"
    export_movement_int8(movement, movement_path)
    export_reactive_action_fp32(action, action_path)
    movement_report = validate_movement_int8_graph(movement_path)
    action_report = validate_reactive_action_fp32_graph(action_path)
    training_dataset = getattr(run_manifest, "dataset_manifest", None)
    training_lineage = (
        _manifest_to_dict(training_dataset)
        if isinstance(training_dataset, DatasetManifestV1)
        else {"purpose": DataPurpose.TEST_ONLY.value}
    )
    package_metrics = _json_safe(
        {
            "movement_graph": movement_report,
            "action_graph": action_report,
            "offline_gate_passed": True,
            **export_metrics,
        }
    )
    package = HierarchicalPackageV3(
        generation=generation,
        parent_generation=parent_generation,
        movement_path=movement_path,
        action_path=action_path,
        purpose=target,
        movement_sha256=_sha256_file(movement_path),
        action_sha256=_sha256_file(action_path),
        state_schema=MOVEMENT_STATE_SCHEMA_V3,
        movement_parameter_count=sum(parameter.numel() for parameter in movement.parameters()),
        action_parameter_count=sum(parameter.numel() for parameter in action.parameters()),
        metadata={
            "purpose": target.value,
            "training_lineage": training_lineage,
            "training_run": {
                "name": getattr(run_manifest, "name", ""),
                "steps": getattr(run_manifest, "steps", None),
                "config_sha256": getattr(run_manifest, "config_sha256", ""),
                "checkpoint_path": str(getattr(run_manifest, "checkpoint_path", "")),
            },
            "movement_qat_proven": True,
            "metrics": package_metrics,
        },
    )
    package.assert_loadable()
    (destination / "package-v3.json").write_text(
        json.dumps(package.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (destination / "metrics.json").write_text(
        json.dumps(package_metrics, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return package


def load_hierarchical_package(manifest_path: str | Path) -> HierarchicalPackageV2:
    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("hierarchical package manifest must be a JSON object")
    payload = dict(payload)
    for key in ("decision_path", "action_path"):
        path = Path(payload[key])
        if not path.is_absolute():
            payload[key] = str((manifest.parent / path).resolve())
    package = HierarchicalPackageV2.from_dict(payload)
    package.assert_loadable()
    return package


def load_hierarchical_package_v3(manifest_path: str | Path) -> HierarchicalPackageV3:
    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("movement v3 package manifest must be a JSON object")
    payload = dict(payload)
    for key in ("movement_path", "action_path"):
        path = Path(payload[key])
        if not path.is_absolute():
            payload[key] = str((manifest.parent / path).resolve())
    package = HierarchicalPackageV3.from_dict(payload)
    package.assert_loadable()
    return package


def publish_hierarchical_package(
    package: HierarchicalPackageV2,
    pointer_path: str | Path,
    *,
    slot: str,
) -> Path:
    """Write an atomic active/pending pointer only after full package validation."""

    if slot not in {"active", "pending"}:
        raise ValueError("package pointer slot must be active or pending")
    package.assert_loadable()
    pointer = Path(pointer_path)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    payload = package.to_dict()
    payload["pointer_slot"] = slot
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temporary, pointer)
    return pointer


def publish_hierarchical_package_v3(
    package: HierarchicalPackageV3,
    pointer_path: str | Path,
    *,
    slot: str,
) -> Path:
    """Publish v3 only to pending, leaving the v2 active pointer untouched."""

    if slot != "pending":
        raise ValueError("movement v3 package can only be published to pending")
    if not isinstance(package, HierarchicalPackageV3):
        raise TypeError("movement v3 publisher requires HierarchicalPackageV3")
    package.assert_loadable()
    pointer = Path(pointer_path)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    payload = package.to_dict()
    payload["pointer_slot"] = slot
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, pointer)
    return pointer
