"""训练设备契约与最小真实参数更新探测。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class TrainingDeviceReportV1:
    requested: str
    device_type: str
    device_index: int | None
    device_name: str
    torch_version: str
    hip_version: str
    smoke_loss: float
    gradient_norm: float
    parameter_updated: bool


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch 2.9 is required for training") from error
    return torch


def resolve_training_device(requested: str) -> Any:
    torch = _torch()
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("training device must be a non-empty string")
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid training device: {requested!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU training requires an available HIP device")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"GPU training device index is unavailable: {index}")
        device_name = str(torch.cuda.get_device_name(index))
        if "RX 9070 XT" not in device_name:
            raise RuntimeError(
                f"GPU training requires Radeon RX 9070 XT, detected {device_name}"
            )
    elif device.type != "cpu":
        raise ValueError("training device must be cuda or cpu")
    return device


def run_training_device_smoke(requested: str) -> TrainingDeviceReportV1:
    torch = _torch()
    device = resolve_training_device(requested)
    from .model import MirageActor

    torch.manual_seed(7)
    actor = MirageActor().to(device)
    actor.train()
    before = tuple(parameter.detach().clone() for parameter in actor.parameters())
    state = actor.initial_state(1, device=device)
    output = actor([bytes(256)], state)
    loss = output.movement_mean.square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("training device smoke produced a non-finite loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer = torch.optim.SGD(actor.parameters(), lr=1e-3)
    optimizer.step()
    parameter_updated = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, actor.parameters())
    )
    if not parameter_updated:
        raise RuntimeError("training device smoke did not update parameters")
    if not math.isfinite(float(gradient_norm.detach().cpu())):
        raise RuntimeError("training device smoke produced a non-finite gradient norm")
    if device.type == "cuda":
        device_index = torch.cuda.current_device() if device.index is None else device.index
        device_name = str(torch.cuda.get_device_name(device_index))
    else:
        device_index = None
        device_name = "CPU"
    return TrainingDeviceReportV1(
        requested=requested,
        device_type=device.type,
        device_index=device_index,
        device_name=device_name,
        torch_version=str(torch.__version__),
        hip_version=str(getattr(torch.version, "hip", None) or "unavailable"),
        smoke_loss=float(loss.detach().cpu()),
        gradient_norm=float(gradient_norm.detach().cpu()),
        parameter_updated=parameter_updated,
    )
