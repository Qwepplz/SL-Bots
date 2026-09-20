"""Quantization-aware training helpers for the test-only decision model."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from typing import Any


try:
    import torch
    from torch import Tensor, nn
    from torch.ao.quantization import FakeQuantize
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised by the minimal runtime boundary
    torch = None
    Tensor = Any
    nn = None
    FakeQuantize = None
    MovingAverageMinMaxObserver = None
    F = None


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for decision QAT")
    return torch


def _fake_quant(*, dtype: Any, quant_min: int, quant_max: int) -> Any:
    torch_module = _require_torch()
    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=quant_min,
        quant_max=quant_max,
        dtype=dtype,
        qscheme=torch_module.per_tensor_symmetric,
    )


if nn is not None:

    class QATLinear(nn.Module):
        """A float-master Linear with explicit weight and activation fake-quant."""

        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            self.linear = linear
            self.weight_fake_quant = _fake_quant(
                dtype=torch.qint8,
                quant_min=-128,
                quant_max=127,
            )
            self.activation_fake_quant = _fake_quant(
                dtype=torch.quint8,
                quant_min=0,
                quant_max=255,
            )

        @classmethod
        def from_linear(cls, linear: nn.Linear) -> "QATLinear":
            if not isinstance(linear, nn.Linear):
                raise TypeError("QATLinear.from_linear requires nn.Linear")
            return cls(linear)

        @property
        def in_features(self) -> int:
            return self.linear.in_features

        @property
        def out_features(self) -> int:
            return self.linear.out_features

        @property
        def weight(self) -> Tensor:
            """Expose the float master weight for PyTorch attention modules."""

            return self.linear.weight

        @property
        def bias(self) -> Tensor | None:
            """Expose the float master bias for PyTorch attention modules."""

            return self.linear.bias

        def forward(self, input_value: Tensor) -> Tensor:
            weight = self.weight_fake_quant(self.linear.weight)
            output = F.linear(input_value, weight, self.linear.bias)
            return self.activation_fake_quant(output)


else:

    class QATLinear:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch 2.9 is required for decision QAT")


def _replace_linear_children(module: Any) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, QATLinear):
            continue
        if nn is not None and isinstance(child, nn.Linear):
            setattr(module, name, QATLinear.from_linear(child))
            continue
        _replace_linear_children(child)


def prepare_decision_qat(model: Any) -> Any:
    """Attach fake-quant modules to every decision-layer Linear in-place."""

    _require_torch()
    _replace_linear_children(model)
    setattr(model, "_decision_qat_prepared", True)
    return model


def contains_fake_quant(model: Any) -> bool:
    """Return whether a module tree contains active QAT fake-quant observers."""

    if nn is None:
        return False
    return any(isinstance(module, FakeQuantize) for module in model.modules())


def prepare_action_fp32(model: Any) -> Any:
    """Keep an action model in ordinary FP32 and reject accidental QAT."""

    _require_torch()
    if contains_fake_quant(model):
        raise ValueError("action policy must not contain observers or fake-quant modules")
    model.float()
    for parameter in model.parameters():
        if parameter.dtype is not torch.float32:
            raise ValueError("action policy parameters must remain FP32")
    return model


def freeze_qat_observers(model: Any) -> Any:
    """Freeze calibration statistics before tracing the QAT-eval graph."""

    _require_torch()
    if nn is not None:
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
    return model


def qat_state_dict(model: Any) -> dict[str, Any]:
    """Return only observer/fake-quant state for a reproducible checkpoint."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "fake_quant" in name or "activation_post_process" in name
    }


DECISION_INPUT_NAMES = ("observation_history", "previous_intent", "memory")
DECISION_OUTPUT_NAMES = (
    "tactical_mode_logits",
    "task_logits",
    "goal_position",
    "waypoint_position",
    "facing_yaw_pitch",
    "desired_range",
    "target_slot_logits",
    "aggression",
    "risk",
    "priority",
    "ttl_ticks",
    "next_decision_memory",
)
ACTION_INPUT_NAMES = ("local_observation", "cached_intent", "action_hidden")
ACTION_OUTPUT_NAMES = (
    "movement_alpha",
    "movement_beta",
    "action_vector",
    "mouse_loc",
    "mouse_scale",
    "mouse_mix_logits",
    "button_logits",
    "weapon_logits",
    "buy_logits",
    "next_action_hidden",
)
MOVEMENT_INPUT_NAMES = ("observation",)
MOVEMENT_OUTPUT_NAMES = ("move_logits", "stance_logits", "jump_logits", "plan_embedding")
REACTIVE_ACTION_INPUT_NAMES = ("local_observation", "movement_plan", "action_hidden")
REACTIVE_ACTION_OUTPUT_NAMES = (
    "mouse_loc",
    "mouse_scale",
    "mouse_mix_logits",
    "button_logits",
    "weapon_logits",
    "buy_logits",
    "next_hidden",
    "plan_condition",
)


def prepare_movement_qat(model: Any) -> Any:
    """Attach the same float-master fake-quant boundary used by v3 movement."""

    _require_torch()
    _replace_linear_children(model)
    setattr(model, "_movement_qat_prepared", True)
    return model


def _export_onnx_model(model: Any, path: Path, *, kind: str) -> None:
    torch_module = _require_torch()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("ONNX export requires a CPU model copy")
    if kind == "decision":

        class DecisionWrapper(torch_module.nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, observation_history: Any, previous_intent: Any, memory: Any) -> Any:
                output = self.wrapped(observation_history, previous_intent, memory)
                return (
                    output.tactical_mode_logits,
                    output.task_logits,
                    output.goal_position_tensor,
                    output.waypoint_position_tensor,
                    output.facing_yaw_pitch,
                    output.desired_range,
                    output.target_slot,
                    output.aggression,
                    output.risk,
                    output.priority,
                    output.ttl_ticks,
                    output.decision_memory,
                )

        wrapper = DecisionWrapper(model)
        arguments = (
            torch_module.zeros(1, 32, 256, dtype=torch_module.float32),
            torch_module.zeros(1, 128, dtype=torch_module.float32),
            torch_module.zeros(1, 32, 512, dtype=torch_module.float32),
        )
        input_names = DECISION_INPUT_NAMES
        output_names = DECISION_OUTPUT_NAMES
        dynamic_axes = {
            name: {0: "batch"}
            for name in (*input_names, *output_names)
        }
    elif kind == "action":

        class ActionWrapper(torch_module.nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, local_observation: Any, cached_intent: Any, action_hidden: Any) -> Any:
                output = self.wrapped(local_observation, cached_intent, action_hidden)
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
                    output.recurrent_state.action,
                )

        wrapper = ActionWrapper(model)
        arguments = (
            torch_module.zeros(1, 256, dtype=torch_module.float32),
            torch_module.zeros(1, 128, dtype=torch_module.float32),
            torch_module.zeros(1, 384, dtype=torch_module.float32),
        )
        input_names = ACTION_INPUT_NAMES
        output_names = ACTION_OUTPUT_NAMES
        dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    elif kind == "movement":

        class MovementWrapper(torch_module.nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, observation: Any) -> Any:
                output = self.wrapped(observation)
                return (
                    output.move_logits,
                    output.stance_logits,
                    output.jump_logits,
                    output.plan_embedding,
                )

        wrapper = MovementWrapper(model)
        arguments = (torch_module.zeros(1, 256, dtype=torch_module.float32),)
        input_names = MOVEMENT_INPUT_NAMES
        output_names = MOVEMENT_OUTPUT_NAMES
        dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    elif kind == "reactive_action":

        class ReactiveActionWrapper(torch_module.nn.Module):
            def __init__(self, wrapped: Any) -> None:
                super().__init__()
                self.wrapped = wrapped

            def forward(self, local_observation: Any, movement_plan: Any, action_hidden: Any) -> Any:
                output = self.wrapped(local_observation, movement_plan, action_hidden)
                return (
                    output.mouse_loc,
                    output.mouse_scale,
                    output.mouse_mix_logits,
                    output.button_logits,
                    output.weapon_logits,
                    output.buy_logits,
                    output.next_hidden,
                    output.plan_condition,
                )

        wrapper = ReactiveActionWrapper(model)
        arguments = (
            torch_module.zeros(1, 256, dtype=torch_module.float32),
            torch_module.zeros(1, 75, dtype=torch_module.float32),
            torch_module.zeros(1, 384, dtype=torch_module.float32),
        )
        input_names = REACTIVE_ACTION_INPUT_NAMES
        output_names = REACTIVE_ACTION_OUTPUT_NAMES
        dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    else:
        raise ValueError(f"unsupported ONNX model kind: {kind}")
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.onnx.export(
        wrapper,
        arguments,
        path,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )


def export_decision_int8(model: Any, output_path: str | Path) -> Path:
    """Export QAT-eval decision weights, then quantize dense weights to INT8."""

    _require_torch()
    if not contains_fake_quant(model):
        raise ValueError("decision INT8 export requires a QAT-prepared decision model")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnxruntime is required for INT8 decision export") from error
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu()
    freeze_qat_observers(export_model)
    with tempfile.TemporaryDirectory(prefix="sl-bots-decision-", dir=destination.parent) as directory:
        float_path = Path(directory) / "decision.float.onnx"
        _replace_qat_for_export(export_model)
        _export_onnx_model(export_model, float_path, kind="decision")
        _fold_initializer_transposes(float_path)
        quantize_dynamic(
            str(float_path),
            str(destination),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    return destination


def export_action_fp32(model: Any, output_path: str | Path) -> Path:
    """Export the unquantized action branch with FP32 initializers."""

    _require_torch()
    prepare_action_fp32(model)
    destination = Path(output_path)
    _export_onnx_model(model, destination, kind="action")
    return destination


def export_movement_int8(model: Any, output_path: str | Path) -> Path:
    """Export the QAT-prepared movement branch and dynamically quantize its weights."""

    _require_torch()
    if not contains_fake_quant(model):
        raise ValueError("movement INT8 export requires a QAT-prepared movement model")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnxruntime is required for INT8 movement export") from error
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu()
    freeze_qat_observers(export_model)
    with tempfile.TemporaryDirectory(prefix="sl-bots-movement-", dir=destination.parent) as directory:
        float_path = Path(directory) / "movement.float.onnx"
        _replace_qat_for_export(export_model)
        _export_onnx_model(export_model, float_path, kind="movement")
        _fold_initializer_transposes(float_path)
        quantize_dynamic(
            str(float_path),
            str(destination),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    return destination


def export_reactive_action_fp32(model: Any, output_path: str | Path) -> Path:
    """Export the v3 reaction branch without any quantization observers."""

    _require_torch()
    prepare_action_fp32(model)
    destination = Path(output_path)
    _export_onnx_model(model, destination, kind="reactive_action")
    return destination


def _load_onnx(path: str | Path) -> Any:
    try:
        import onnx
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnx is required for graph validation") from error
    return onnx.load(str(path))


def _replace_qat_for_export(model: Any) -> Any:
    """Fold fake-quant wrappers to plain Linear modules before ONNX quantization.

    ORT's dynamic quantizer only recognizes ordinary Linear/MatMul patterns.  A
    QATLinear wrapper is correct during training but hides most dense weights from
    that pattern matcher, which would otherwise produce a graph with one INT8
    head and many silently retained FP32 weights.
    """

    _require_torch()
    for name, child in list(model.named_children()):
        if isinstance(child, QATLinear):
            linear = torch.nn.Linear(
                child.linear.in_features,
                child.linear.out_features,
                bias=child.linear.bias is not None,
            ).to(device=child.linear.weight.device, dtype=child.linear.weight.dtype)
            with torch.no_grad():
                # The FP32 master weight is the trained QAT parameter.  Applying
                # an uncalibrated observer here can collapse a fresh/eval model to
                # zeros (the default observer scale is intentionally conservative);
                # the ONNX dynamic quantizer below performs the deployment INT8
                # conversion, while the source model and checkpoint still prove
                # that QAT was active during training.
                linear.weight.copy_(child.linear.weight)
                if child.linear.bias is not None and linear.bias is not None:
                    linear.bias.copy_(child.linear.bias)
            setattr(model, name, linear)
            continue
        _replace_qat_for_export(child)
    return model


def _fold_initializer_transposes(path: Path) -> None:
    """Make transposed Linear initializers visible to ORT's weight quantizer."""

    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnx is required for decision graph preparation") from error

    model = onnx.load(str(path))
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    aliases: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            source = node.input[0]
            while source in aliases:
                source = aliases[source]
            if source in initializer_names:
                aliases[node.output[0]] = source

    def resolve_alias(value: str) -> str:
        seen: set[str] = set()
        while value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    replacements: dict[str, str] = {}
    folded: list[Any] = []
    retained_nodes: list[Any] = []
    for node in graph.node:
        if node.op_type != "Transpose" or not node.input:
            retained_nodes.append(node)
            continue
        source_name = resolve_alias(node.input[0])
        if source_name not in initializer_names:
            retained_nodes.append(node)
            continue
        initializer = next(item for item in graph.initializer if item.name == source_name)
        array = numpy_helper.to_array(initializer)
        if array.ndim != 2:
            retained_nodes.append(node)
            continue
        permutation = tuple(
            int(value)
            for attribute in node.attribute
            if attribute.name == "perm"
            for value in attribute.ints
        ) or (1, 0)
        folded_name = f"{node.output[0]}__folded_weight"
        folded.append(numpy_helper.from_array(array.transpose(permutation), name=folded_name))
        replacements[node.output[0]] = folded_name
    for node in retained_nodes:
        for index, value in enumerate(node.input):
            value = resolve_alias(value)
            node.input[index] = replacements.get(value, value)
    del graph.node[:]
    graph.node.extend(retained_nodes)
    graph.initializer.extend(folded)
    onnx.save(model, str(path))


def _data_type_name(data_type: int) -> str:
    import onnx

    return onnx.TensorProto.DataType.Name(data_type)


def _graph_report(path: str | Path) -> dict[str, Any]:
    graph = _load_onnx(path).graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    initializer_types = {_data_type_name(initializer.data_type) for initializer in graph.initializer}
    int8_initializer_names = {
        initializer.name
        for initializer in graph.initializer
        if _data_type_name(initializer.data_type) in {"INT8", "UINT8"}
    }
    integer_ops = {
        "MatMulInteger",
        "QLinearMatMul",
        "QLinearConv",
        "ConvInteger",
        "GemmInteger",
        "DynamicQuantizeLinear",
        "QuantizeLinear",
        "DequantizeLinear",
    }
    integer_nodes = [node for node in graph.node if node.op_type in integer_ops]
    weight_nodes = []
    quantized_weight_nodes = 0
    unquantized_weight_nodes: list[str] = []
    for node in graph.node:
        if node.op_type not in {"MatMul", "Gemm", "MatMulInteger", "GemmInteger", "QLinearMatMul"}:
            continue
        candidates = tuple(node.input[1:])
        initializer_candidate = next(
            (name for name in candidates if name in initializer_names),
            None,
        )
        if initializer_candidate is None:
            continue
        weight_nodes.append(node.name or initializer_candidate)
        if any(name in int8_initializer_names for name in candidates):
            quantized_weight_nodes += 1
        else:
            unquantized_weight_nodes.append(node.name or initializer_candidate)
    return {
        "initializer_dtypes": initializer_types,
        "int8_initializer_names": int8_initializer_names,
        "integer_ops": tuple(node.op_type for node in integer_nodes),
        "quantized_weight_nodes": quantized_weight_nodes,
        "supported_weight_nodes": len(weight_nodes),
        "unquantized_weight_nodes": tuple(unquantized_weight_nodes),
        "has_qdq_or_integer_node": bool(integer_nodes or int8_initializer_names),
    }


def validate_decision_int8_graph(path: str | Path) -> dict[str, Any]:
    report = _graph_report(path)
    if report["supported_weight_nodes"] <= 0:
        raise ValueError("decision graph has no supported dense weight node")
    if report["unquantized_weight_nodes"]:
        raise ValueError(
            "decision graph has unquantized supported weights: "
            + ", ".join(report["unquantized_weight_nodes"])
        )
    if report["quantized_weight_nodes"] != report["supported_weight_nodes"]:
        raise ValueError("decision graph does not quantize every supported dense weight")
    if not report["has_qdq_or_integer_node"]:
        raise ValueError("decision graph has no Q/DQ or integer quantization node")
    return report


def validate_action_fp32_graph(path: str | Path) -> dict[str, Any]:
    report = _graph_report(path)
    if any(data_type in {"INT8", "UINT8"} for data_type in report["initializer_dtypes"]):
        raise ValueError("action graph contains INT8/UINT8 initializers")
    if report["has_qdq_or_integer_node"]:
        raise ValueError("action graph contains Q/DQ or integer quantization nodes")
    return report


def validate_movement_int8_graph(path: str | Path) -> dict[str, Any]:
    """Require real INT8/QDQ evidence for every supported movement weight."""

    report = _graph_report(path)
    if report["supported_weight_nodes"] <= 0:
        raise ValueError("movement graph has no supported dense weight node")
    if report["unquantized_weight_nodes"]:
        raise ValueError(
            "movement graph has unquantized supported weights: "
            + ", ".join(report["unquantized_weight_nodes"])
        )
    if report["quantized_weight_nodes"] != report["supported_weight_nodes"]:
        raise ValueError("movement graph does not quantize every supported dense weight")
    if not report["has_qdq_or_integer_node"]:
        raise ValueError("movement graph has no Q/DQ or integer quantization node")
    return report


def validate_reactive_action_fp32_graph(path: str | Path) -> dict[str, Any]:
    """Alias with a v3-specific name to make the movement/action boundary explicit."""

    return validate_action_fp32_graph(path)


def _compare_outputs(expected: Any, actual: Any) -> dict[str, float]:
    import numpy as np

    expected_array = expected.detach().cpu().numpy()
    actual_array = np.asarray(actual)
    difference = np.abs(expected_array - actual_array)
    return {
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
    }


def compare_decision_onnx(
    model: Any,
    path: str | Path,
    observation_history: Any,
    previous_intent: Any,
    memory: Any,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnxruntime is required for output comparison") from error
    model.eval()
    with torch.no_grad():
        output = model(observation_history, previous_intent, memory)
        expected = (
            output.tactical_mode_logits,
            output.task_logits,
            output.goal_position_tensor,
            output.waypoint_position_tensor,
            output.facing_yaw_pitch,
            output.desired_range,
            output.target_slot,
            output.aggression,
            output.risk,
            output.priority,
            output.ttl_ticks,
            output.decision_memory,
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(DECISION_OUTPUT_NAMES),
        {
            "observation_history": observation_history.detach().cpu().numpy(),
            "previous_intent": previous_intent.detach().cpu().numpy(),
            "memory": memory.detach().cpu().numpy(),
        },
    )
    errors = {
        name: _compare_outputs(expected_value, actual_value)
        for name, expected_value, actual_value in zip(DECISION_OUTPUT_NAMES, expected, actual)
    }
    head_errors = [errors[name]["max_abs_error"] for name in DECISION_OUTPUT_NAMES[:-1]]
    return {
        "max_abs_error": max(error["max_abs_error"] for error in errors.values()),
        "mean_abs_error": sum(error["mean_abs_error"] for error in errors.values()) / len(errors),
        "head_max_abs_error": max(head_errors),
        "memory_max_abs_error": errors["next_decision_memory"]["max_abs_error"],
    }


def compare_action_onnx(
    model: Any,
    path: str | Path,
    local_observation: Any,
    cached_intent: Any,
    action_hidden: Any,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnxruntime is required for output comparison") from error
    model.eval()
    with torch.no_grad():
        output = model(local_observation, cached_intent, action_hidden)
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
            output.recurrent_state.action,
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(ACTION_OUTPUT_NAMES),
        {
            "local_observation": local_observation.detach().cpu().numpy(),
            "cached_intent": cached_intent.detach().cpu().numpy(),
            "action_hidden": action_hidden.detach().cpu().numpy(),
        },
    )
    errors = [_compare_outputs(expected_value, actual_value) for expected_value, actual_value in zip(expected, actual)]
    return {
        "max_abs_error": max(error["max_abs_error"] for error in errors),
        "mean_abs_error": sum(error["mean_abs_error"] for error in errors) / len(errors),
    }


def compare_movement_onnx(model: Any, path: str | Path, observation: Any) -> dict[str, Any]:
    """Compare Torch and INT8 movement outputs, including classification argmaxes."""

    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnxruntime is required for output comparison") from error
    model.eval()
    with torch.no_grad():
        output = model(observation)
        expected = (
            output.move_logits,
            output.stance_logits,
            output.jump_logits,
            output.plan_embedding,
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(MOVEMENT_OUTPUT_NAMES),
        {"observation": observation.detach().cpu().numpy()},
    )
    errors = [_compare_outputs(expected_value, actual_value) for expected_value, actual_value in zip(expected, actual)]
    argmax_equal = all(
        torch.equal(
            expected_value.detach().cpu().argmax(dim=-1),
            torch.from_numpy(actual_value).argmax(dim=-1),
        )
        for expected_value, actual_value in zip(expected[:3], actual[:3])
    )
    return {
        "max_abs_error": max(error["max_abs_error"] for error in errors),
        "mean_abs_error": sum(error["mean_abs_error"] for error in errors) / len(errors),
        "argmax_equal": bool(argmax_equal),
    }


def compare_reactive_action_onnx(
    model: Any,
    path: str | Path,
    local_observation: Any,
    movement_plan: Any,
    action_hidden: Any,
) -> dict[str, float]:
    """Compare Torch and FP32 ONNX reaction outputs."""

    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("onnxruntime is required for output comparison") from error
    model.eval()
    with torch.no_grad():
        output = model(local_observation, movement_plan, action_hidden)
        expected = (
            output.mouse_loc,
            output.mouse_scale,
            output.mouse_mix_logits,
            output.button_logits,
            output.weapon_logits,
            output.buy_logits,
            output.next_hidden,
            output.plan_condition,
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(REACTIVE_ACTION_OUTPUT_NAMES),
        {
            "local_observation": local_observation.detach().cpu().numpy(),
            "movement_plan": movement_plan.detach().cpu().numpy(),
            "action_hidden": action_hidden.detach().cpu().numpy(),
        },
    )
    errors = [_compare_outputs(expected_value, actual_value) for expected_value, actual_value in zip(expected, actual)]
    return {
        "max_abs_error": max(error["max_abs_error"] for error in errors),
        "mean_abs_error": sum(error["mean_abs_error"] for error in errors) / len(errors),
    }
