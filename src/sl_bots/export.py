from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import DataPurpose, ensure_purpose
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
