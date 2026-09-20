"""Two-stage, test-only training utilities for the v3 movement policy.

The module deliberately keeps dataset policy separate from the optimizers: class
weights are fitted once from the train split, movement is trained before reaction,
and the reaction stage consumes the movement model's predicted plan embedding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from .action_reconstruction import (
    ACTION_MASK_BUY,
    ACTION_MASK_BUTTONS,
    ACTION_MASK_PITCH,
    ACTION_MASK_WEAPON,
    ACTION_MASK_YAW,
)
from .movement import MOVEMENT_BUTTON_MASK, MovementLabelV1
from .movement_model import MovementOutputV1


_HEAD_SIZES = {"move": 17, "stance": 3, "jump": 2}
_HORIZON_COUNT = 3
_HORIZON_WEIGHTS = (1.0, 0.5, 0.25)


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return result if dtype is None else result.to(dtype=dtype)


def _labels_to_targets(labels: Any) -> dict[str, torch.Tensor]:
    if isinstance(labels, Mapping):
        required = ("move", "stance", "jump", "valid", "stance_valid")
        missing = [name for name in required if name not in labels]
        if missing:
            raise ValueError(f"movement targets missing fields: {', '.join(missing)}")
        return {
            "move": _as_tensor(labels["move"], dtype=torch.long),
            "stance": _as_tensor(labels["stance"], dtype=torch.long),
            "jump": _as_tensor(labels["jump"], dtype=torch.long),
            "valid": _as_tensor(labels["valid"], dtype=torch.bool),
            "stance_valid": _as_tensor(labels["stance_valid"], dtype=torch.bool),
        }
    if isinstance(labels, MovementLabelV1):
        labels = (labels,)
    if isinstance(labels, Sequence) and labels and all(
        isinstance(label, MovementLabelV1) for label in labels
    ):
        return {
            name: torch.tensor(
                [getattr(label, name) for label in labels],
                dtype=torch.bool if name.endswith("valid") else torch.long,
            )
            for name in ("move", "stance", "jump", "valid", "stance_valid")
        }
    raise TypeError("movement targets must be a mapping or MovementLabelV1 sequence")


@dataclass(frozen=True)
class MovementFrequencyWeightsV1:
    """Train-only inverse-frequency weights for the three movement heads."""

    move: torch.Tensor
    stance: torch.Tensor
    jump: torch.Tensor
    source_split: str = "train"

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError("movement frequency weights must be fitted from the train split")
        for name, expected in _HEAD_SIZES.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != expected:
                raise ValueError(f"{name} frequency weights must have shape [{expected}]")
            if value.dtype not in (torch.float16, torch.float32, torch.float64):
                raise TypeError(f"{name} frequency weights must be floating point")
            if not bool(torch.isfinite(value).all()) or bool((value <= 0.0).any()):
                raise ValueError(f"{name} frequency weights must be finite and positive")
        object.__setattr__(self, "move", self.move.detach().clone())
        object.__setattr__(self, "stance", self.stance.detach().clone())
        object.__setattr__(self, "jump", self.jump.detach().clone())

    def for_head(self, name: str, *, device: torch.device | str | None = None) -> torch.Tensor:
        if name not in _HEAD_SIZES:
            raise KeyError(name)
        value = getattr(self, name)
        return value if device is None else value.to(device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_split": self.source_split,
            "move": self.move.detach().cpu().tolist(),
            "stance": self.stance.detach().cpu().tolist(),
            "jump": self.jump.detach().cpu().tolist(),
        }


def _inverse_frequency(target: torch.Tensor, valid: torch.Tensor, class_count: int) -> torch.Tensor:
    target = target.reshape(-1).to(torch.long)
    valid = valid.reshape(-1).to(torch.bool)
    if target.numel() != valid.numel():
        raise ValueError("movement target and validity mask must have equal size")
    selected = target[valid]
    if selected.numel() == 0:
        return torch.ones(class_count, dtype=torch.float32)
    if bool((selected < 0).any()) or bool((selected >= class_count).any()):
        raise ValueError("movement target class is outside the configured head range")
    counts = torch.bincount(selected, minlength=class_count).to(torch.float32)
    present = counts > 0.0
    sample_count = counts.sum().clamp_min(1.0)
    raw = torch.sqrt(sample_count / (float(class_count) * (counts + 1.0)))
    weights = raw.clamp(min=0.5, max=4.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights


def fit_frequency_weights(targets: Any, *, split: str = "train") -> MovementFrequencyWeightsV1:
    """Fit global weights once; validation/test callers must pass the fitted object."""

    if str(split) != "train":
        raise ValueError("frequency weights may only be fitted from the train split")
    normalized = _labels_to_targets(targets)
    return MovementFrequencyWeightsV1(
        move=_inverse_frequency(normalized["move"], normalized["valid"], 17),
        stance=_inverse_frequency(normalized["stance"], normalized["stance_valid"], 3),
        jump=_inverse_frequency(normalized["jump"], normalized["valid"], 2),
    )


compute_frequency_weights = fit_frequency_weights
fit_global_frequency_weights = fit_frequency_weights


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    horizon_weight: Sequence[float] = _HORIZON_WEIGHTS,
) -> torch.Tensor:
    if logits.ndim != 3 or target.shape != logits.shape[:2] or mask.shape != target.shape:
        raise ValueError("movement logits, target and mask shapes do not align")
    per_sample = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.to(torch.long).reshape(-1),
        weight=weight.to(device=logits.device, dtype=logits.dtype),
        reduction="none",
    ).reshape_as(target)
    mask_value = mask.to(device=logits.device, dtype=per_sample.dtype)
    horizon_value = torch.as_tensor(horizon_weight, device=logits.device, dtype=per_sample.dtype)
    if horizon_value.shape != (logits.shape[1],):
        raise ValueError("horizon_weight must match the number of movement horizons")
    loss = per_sample.sum() * 0.0
    for horizon_index, horizon_scale in enumerate(horizon_value):
        horizon_mask = mask_value[:, horizon_index]
        horizon_loss = (per_sample[:, horizon_index] * horizon_mask).sum() / horizon_mask.sum().clamp_min(1.0)
        loss = loss + horizon_scale * horizon_loss
    return loss


def movement_loss(
    output: MovementOutputV1 | Any,
    targets: Any,
    frequency_weights: MovementFrequencyWeightsV1,
    *,
    return_components: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Compute masked move/stance/jump CE with fixed train-global weights."""

    normalized = _labels_to_targets(targets)
    valid = normalized["valid"].to(output.move_logits.device)
    stance_valid = (normalized["stance_valid"] & normalized["valid"]).to(output.stance_logits.device)
    components = {
        "move": _masked_cross_entropy(
            output.move_logits,
            normalized["move"].to(output.move_logits.device),
            valid,
            frequency_weights.for_head("move", device=output.move_logits.device),
        ),
        "stance": _masked_cross_entropy(
            output.stance_logits,
            normalized["stance"].to(output.stance_logits.device),
            stance_valid,
            frequency_weights.for_head("stance", device=output.stance_logits.device),
        ),
        "jump": _masked_cross_entropy(
            output.jump_logits,
            normalized["jump"].to(output.jump_logits.device),
            valid,
            frequency_weights.for_head("jump", device=output.jump_logits.device),
        ),
    }
    return components if return_components else sum(components.values())


def _reaction_button_targets(target: torch.Tensor, width: int) -> torch.Tensor:
    if target.ndim == 1:
        bits = torch.arange(width, device=target.device, dtype=torch.long)
        return ((target.to(torch.long).unsqueeze(-1) & (1 << bits)) != 0).to(torch.float32)
    if target.ndim == 2 and target.shape[-1] == width:
        return target.to(torch.float32)
    raise ValueError(f"reaction buttons must have shape [batch] or [batch,{width}]")


def _reaction_mouse_nll(output: Any, target: torch.Tensor) -> torch.Tensor:
    if target.shape != output.mouse_loc.shape or output.mouse_mix_logits.shape[:2] != target.shape:
        raise ValueError("reaction mouse target/output shapes do not align")
    loc = output.mouse_loc.unsqueeze(-1)
    scale = output.mouse_scale.clamp_min(1e-4).unsqueeze(-1)
    target = target.unsqueeze(-1)
    gaussian = -0.5 * ((target - loc) / scale).square() - scale.log() - 0.5 * math.log(2.0 * math.pi)
    mixture = torch.log_softmax(output.mouse_mix_logits, dim=-1) + gaussian
    return -torch.logsumexp(mixture, dim=-1)


def _loss_bit_mask(
    targets: Mapping[str, Any],
    *,
    bit: int,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    value = targets.get("loss_mask")
    if value is None:
        value = torch.full((batch,), 0xFF, dtype=torch.long)
    mask = _as_tensor(value, dtype=torch.long).to(device)
    if mask.shape != (batch,):
        raise ValueError("reaction loss_mask must have shape [batch]")
    return ((mask & int(bit)) != 0).to(torch.float32)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def reaction_loss(output: Any, targets: Mapping[str, Any]) -> torch.Tensor:
    """Reaction objective; movement axes and movement-owned buttons are excluded."""

    device = output.mouse_loc.device
    batch = output.mouse_loc.shape[0]
    yaw_mask = _loss_bit_mask(targets, bit=ACTION_MASK_YAW, batch=batch, device=device)
    pitch_mask = _loss_bit_mask(targets, bit=ACTION_MASK_PITCH, batch=batch, device=device)
    mouse_mask = torch.stack((yaw_mask, pitch_mask), dim=-1)
    mouse = _reaction_mouse_nll(output, _as_tensor(targets["mouse"], dtype=output.mouse_loc.dtype).to(device))
    mouse = _masked_mean(mouse, mouse_mask)

    button_target = _reaction_button_targets(_as_tensor(targets["buttons"]).to(device), output.button_logits.shape[-1])
    button_logits = output.button_logits
    movement_indices = [index for index in range(button_logits.shape[-1]) if MOVEMENT_BUTTON_MASK & (1 << index)]
    reaction_indices = [index for index in range(button_logits.shape[-1]) if index not in movement_indices]
    if not reaction_indices:
        button_value = button_logits.sum() * 0.0
    else:
        button_per = F.binary_cross_entropy_with_logits(
            button_logits[:, reaction_indices],
            button_target[:, reaction_indices].to(button_logits.dtype),
            reduction="none",
        ).mean(dim=-1)
        button_mask = _loss_bit_mask(targets, bit=ACTION_MASK_BUTTONS, batch=batch, device=device)
        button_value = _masked_mean(button_per, button_mask)

    def categorical(name: str, logits: torch.Tensor, bit: int) -> torch.Tensor:
        target = _as_tensor(targets[name], dtype=torch.long).to(device).reshape(batch)
        per = F.cross_entropy(logits, target, reduction="none")
        return _masked_mean(per, _loss_bit_mask(targets, bit=bit, batch=batch, device=device))

    return (
        mouse
        + button_value
        + categorical("weapon", output.weapon_logits, ACTION_MASK_WEAPON)
        + categorical("buy", output.buy_logits, ACTION_MASK_BUY)
    )


@dataclass
class TBPTTStateV1:
    """Per-sequence hidden-state store for continuous recurrent shards."""

    hidden_size: int
    _hidden: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError("TBPTT hidden_size must be a positive integer")

    @staticmethod
    def _sequence_ids(batch: Any) -> tuple[str, ...]:
        raw = getattr(batch, "sequence_ids", ())
        if not raw:
            raise ValueError("TBPTT batch must include sequence_ids")
        first = raw[0]
        sequence_ids = tuple(str(value) for value in first)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("TBPTT batch sequence_ids must be unique")
        return sequence_ids

    @staticmethod
    def _starts(batch: Any, count: int) -> tuple[bool, ...]:
        mask = getattr(batch, "sequence_start_mask", None)
        if mask is None:
            mask = getattr(batch, "hidden_state_mask", None)
        if mask is None:
            return tuple(True for _ in range(count))
        values = mask[0]
        return tuple(bool(value) for value in values)

    def hidden_for(self, batch: Any, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        sequence_ids = self._sequence_ids(batch)
        starts = self._starts(batch, len(sequence_ids))
        result = []
        for sequence_id, starts_here in zip(sequence_ids, starts):
            if starts_here or sequence_id not in self._hidden:
                result.append(torch.zeros(self.hidden_size, device=device, dtype=dtype))
            else:
                result.append(self._hidden[sequence_id].to(device=device, dtype=dtype))
        return torch.stack(result, dim=0)

    def commit(self, batch: Any, hidden_sequence: torch.Tensor) -> None:
        sequence_ids = self._sequence_ids(batch)
        if hidden_sequence.ndim == 2:
            final = hidden_sequence
        elif hidden_sequence.ndim == 3 and hidden_sequence.shape[1] == len(sequence_ids):
            final = hidden_sequence[-1]
        else:
            raise ValueError("TBPTT hidden_sequence must have shape [batch,H] or [time,batch,H]")
        if tuple(final.shape) != (len(sequence_ids), self.hidden_size):
            raise ValueError("TBPTT hidden_sequence shape does not match batch and hidden_size")
        for index, sequence_id in enumerate(sequence_ids):
            self._hidden[sequence_id] = final[index].detach().clone()

    def reset(self, sequence_ids: Sequence[str] | None = None) -> None:
        if sequence_ids is None:
            self._hidden.clear()
        else:
            for sequence_id in sequence_ids:
                self._hidden.pop(str(sequence_id), None)


def masked_tbptt_loss(per_step_loss: torch.Tensor, batch: Any) -> torch.Tensor:
    """Average loss after removing loader padding and burn-in steps."""

    if per_step_loss.ndim < 2:
        raise ValueError("per_step_loss must have time and batch dimensions")
    mask = getattr(batch, "loss_masks", None)
    if mask is None:
        mask = torch.ones(per_step_loss.shape[:2], device=per_step_loss.device)
    else:
        mask = mask.to(device=per_step_loss.device, dtype=per_step_loss.dtype)
    burn_in = getattr(batch, "burn_in_mask", None)
    if burn_in is not None:
        mask = mask * (~burn_in.to(device=per_step_loss.device, dtype=torch.bool)).to(per_step_loss.dtype)
    while mask.ndim < per_step_loss.ndim:
        mask = mask.unsqueeze(-1)
    return (per_step_loss * mask).sum() / mask.sum().clamp_min(1.0)


tbptt_hidden_for = TBPTTStateV1.hidden_for
commit_tbptt_hidden = TBPTTStateV1.commit


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MovementTrainerV3:
    """Minimal FP32 movement trainer with fixed train-global class weights."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        frequency_weights: MovementFrequencyWeightsV1,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(frequency_weights, MovementFrequencyWeightsV1):
            raise TypeError("MovementTrainerV3 requires MovementFrequencyWeightsV1")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.frequency_weights = frequency_weights
        self.device = torch.device(device)

    def train_batch(self, observations: torch.Tensor, targets: Any) -> float:
        self.model.train()
        observations = observations.to(self.device)
        normalized = {
            key: value.to(self.device) for key, value in _labels_to_targets(targets).items()
        }
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(observations)
        loss = movement_loss(output, normalized, self.frequency_weights)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def validation_loss(self, observations: torch.Tensor, targets: Any) -> float:
        self.model.eval()
        output = self.model(observations.to(self.device))
        loss = movement_loss(output, targets, self.frequency_weights)
        return float(loss.detach().cpu())

    @torch.no_grad()
    def accuracy(self, observations: torch.Tensor, targets: Any) -> dict[str, float]:
        self.model.eval()
        normalized = _labels_to_targets(targets)
        output = self.model(observations.to(self.device))
        result: dict[str, float] = {}
        for name, mask_name in (("move", "valid"), ("stance", "stance_valid"), ("jump", "valid")):
            prediction = getattr(output, f"{name}_logits").argmax(dim=-1).cpu()
            target = normalized[name]
            mask = normalized[mask_name]
            denominator = int(mask.sum())
            result[name] = 0.0 if denominator == 0 else float(((prediction == target) & mask).sum() / denominator)
        return result


class ReactionTrainerV3:
    """Reaction trainer that freezes movement and consumes predicted plans."""

    def __init__(
        self,
        movement_model: torch.nn.Module,
        reaction_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.movement_model = movement_model.to(device)
        self.reaction_model = reaction_model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        for parameter in self.movement_model.parameters():
            parameter.requires_grad_(False)
        self.movement_model.eval()

    def train_batch(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor,
        targets: Mapping[str, Any],
    ) -> float:
        self.reaction_model.train()
        observations = observations.to(self.device)
        hidden = hidden.to(self.device)
        with torch.no_grad():
            predicted_plan = self.movement_model(observations).plan_embedding
        output = self.reaction_model(observations, predicted_plan, hidden)
        self.optimizer.zero_grad(set_to_none=True)
        loss = reaction_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


@dataclass
class MovementEarlyStoppingV1:
    patience: int = 5
    min_delta: float = 0.0
    best_loss: float = math.inf
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        if not math.isfinite(float(self.min_delta)) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative")

    def update(self, validation_loss: float) -> bool:
        value = float(validation_loss)
        if not math.isfinite(value):
            raise ValueError("validation_loss must be finite")
        if value < self.best_loss - self.min_delta:
            self.best_loss = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped


def build_movement_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    split_manifest: Mapping[str, Any],
    labeler_version: str,
    frequency_weights: MovementFrequencyWeightsV1,
    seed: int,
) -> dict[str, Any]:
    """Persist a resumable FP32 checkpoint with reproducibility metadata."""

    if not isinstance(frequency_weights, MovementFrequencyWeightsV1):
        raise TypeError("frequency_weights must be MovementFrequencyWeightsV1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    payload: dict[str, Any] = {
        "schema": "movement-training-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "split_manifest": dict(split_manifest),
        "labeler_version": str(labeler_version),
        "frequency_weights": frequency_weights.to_dict(),
        "seed": seed,
        "parameter_sha256": _parameter_sha256(model),
        "parameter_counts": {"movement": sum(parameter.numel() for parameter in model.parameters())},
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return payload
