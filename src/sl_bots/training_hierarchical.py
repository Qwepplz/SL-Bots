"""Demo multi-task training for the test-only hierarchical actor."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_reconstruction import (
    ACTION_MASK_BUTTONS,
    ACTION_MASK_BUY,
    ACTION_MASK_FORWARD,
    ACTION_MASK_PITCH,
    ACTION_MASK_SIDE,
    ACTION_MASK_UP,
    ACTION_MASK_WEAPON,
    ACTION_MASK_YAW,
    IN_ATTACK,
    IN_DUCK,
    IN_SPEED,
)
from .hierarchical_model import (
    ACTION_HIDDEN_SIZE,
    DECISION_MEMORY_TOKENS,
    DECISION_MODEL_WIDTH,
    INTENT_EMBEDDING_SIZE,
    HierarchicalActor,
    decision_output_to_intent_embedding,
)


HIERARCHICAL_INPUT_SCHEMA = (
    "observation_history",
    "previous_intent",
    "local_observation",
    "cached_intent",
)
_DECISION_CATEGORICAL_FIELDS = ("tactical_mode", "task", "target_slot")
_DECISION_CONTINUOUS_FIELDS = (
    "goal_position",
    "waypoint_position",
    "facing_yaw_pitch",
    "desired_range",
    "aggression",
    "risk",
    "priority",
    "ttl_ticks",
)
_DECISION_FIELD_ORDER = (
    "tactical_mode",
    "task",
    "goal_position",
    "waypoint_position",
    "facing_yaw_pitch",
    "desired_range",
    "target_slot",
    "aggression",
    "risk",
    "priority",
    "ttl_ticks",
)


try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - minimal runtime import boundary
    torch = None
    Tensor = Any
    F = None


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for hierarchical training")
    return torch


def _move_value(value: Any, device: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    return value


@dataclass(frozen=True)
class HierarchicalBatchV1:
    """The only tensors that may be passed to the actor during Demo training."""

    observation_history: Any
    previous_intent: Any
    local_observation: Any
    cached_intent: Any
    decision_targets: Mapping[str, Any]
    action_targets: Mapping[str, Any]
    observation_sequence: Any | None = None
    intent_embedding_sequence: Any | None = None
    decision_target_sequence: Mapping[str, Any] | None = None
    action_target_sequence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_model_inputs(self.model_inputs())
        if not isinstance(self.decision_targets, Mapping):
            raise TypeError("decision_targets must be a mapping")
        if not isinstance(self.action_targets, Mapping):
            raise TypeError("action_targets must be a mapping")

    @property
    def batch_size(self) -> int:
        return int(self.local_observation.shape[0])

    def model_inputs(self) -> dict[str, Any]:
        return {
            "observation_history": self.observation_history,
            "previous_intent": self.previous_intent,
            "local_observation": self.local_observation,
            "cached_intent": self.cached_intent,
        }

    def to(self, device: Any) -> "HierarchicalBatchV1":
        return HierarchicalBatchV1(
            observation_history=self.observation_history.to(device),
            previous_intent=self.previous_intent.to(device),
            local_observation=self.local_observation.to(device),
            cached_intent=self.cached_intent.to(device),
            decision_targets=_move_value(self.decision_targets, device),
            action_targets=_move_value(self.action_targets, device),
            observation_sequence=(
                None if self.observation_sequence is None else self.observation_sequence.to(device)
            ),
            intent_embedding_sequence=(
                None
                if self.intent_embedding_sequence is None
                else self.intent_embedding_sequence.to(device)
            ),
            decision_target_sequence=(
                None
                if self.decision_target_sequence is None
                else _move_value(self.decision_target_sequence, device)
            ),
            action_target_sequence=(
                None
                if self.action_target_sequence is None
                else _move_value(self.action_target_sequence, device)
            ),
        )


def validate_model_inputs(inputs: Mapping[str, Any]) -> None:
    """Reject future labels or any field outside the actor input contract."""

    names = tuple(inputs)
    unexpected = sorted(set(names) - set(HIERARCHICAL_INPUT_SCHEMA))
    missing = sorted(set(HIERARCHICAL_INPUT_SCHEMA) - set(names))
    if unexpected:
        raise ValueError(f"model input schema contains unauthorized fields: {', '.join(unexpected)}")
    if missing:
        raise ValueError(f"model input schema is missing fields: {', '.join(missing)}")
    history = inputs["observation_history"]
    previous_intent = inputs["previous_intent"]
    local_observation = inputs["local_observation"]
    cached_intent = inputs["cached_intent"]
    if getattr(history, "ndim", None) != 3 or tuple(history.shape[1:]) != (32, 256):
        raise ValueError("observation_history must have shape [batch, 32, 256]")
    for name, value in (
        ("previous_intent", previous_intent),
        ("cached_intent", cached_intent),
    ):
        if getattr(value, "ndim", None) != 2 or value.shape[1] != INTENT_EMBEDDING_SIZE:
            raise ValueError(f"{name} must have shape [batch, 128]")
    if getattr(local_observation, "ndim", None) != 2 or local_observation.shape[1] != 256:
        raise ValueError("local_observation must have shape [batch, 256]")
    batch_size = history.shape[0]
    if any(value.shape[0] != batch_size for value in (previous_intent, local_observation, cached_intent)):
        raise ValueError("all hierarchical model inputs must share the batch dimension")
    if "observation_sequence" in inputs or "intent_embedding_sequence" in inputs:
        raise ValueError("sequence targets must remain outside the actor input contract")


def _target_weight(targets: Mapping[str, Any], field: str, batch_size: int, device: Any) -> Any:
    confidence = targets.get("confidence", {})
    valid_mask = targets.get("valid_mask", {})
    if isinstance(confidence, Mapping) and "all" in confidence:
        confidence_all = confidence["all"]
        index = _DECISION_FIELD_ORDER.index(field)
        confidence_value = confidence_all[:, index]
    elif isinstance(confidence, Mapping) and field in confidence:
        confidence_value = confidence[field]
    else:
        confidence_value = torch.ones(batch_size, device=device)
    if isinstance(valid_mask, Mapping) and "all" in valid_mask:
        mask_value = valid_mask["all"][:, _DECISION_FIELD_ORDER.index(field)]
    elif isinstance(valid_mask, Mapping) and field in valid_mask:
        mask_value = valid_mask[field]
    else:
        mask_value = torch.ones(batch_size, device=device)
    return confidence_value.to(torch.float32).reshape(batch_size) * mask_value.to(torch.float32).reshape(batch_size)


def _weighted_mean(value: Any, weight: Any) -> Any:
    if value.ndim > weight.ndim:
        value = value.reshape(value.shape[0], -1).mean(dim=1)
    weight = weight.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _balanced_categorical_cross_entropy(
    logits: Any,
    target: Any,
    sample_weight: Any | None = None,
) -> Any:
    """Keep rare tactical/task/target labels from collapsing to the mode."""

    class_count = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, class_count)
    flat_target = target.to(torch.long).reshape(-1)
    counts = torch.bincount(flat_target, minlength=class_count).to(flat_logits.dtype)
    present = counts > 0.0
    class_weight = torch.ones(class_count, dtype=flat_logits.dtype, device=flat_logits.device)
    mean_count = counts[present].mean().clamp_min(1.0)
    inverse_frequency = mean_count / counts.clamp_min(1.0)
    class_weight[present] = inverse_frequency[present].clamp(min=0.25, max=16.0)
    value = F.cross_entropy(flat_logits, flat_target, weight=class_weight, reduction="none")
    if sample_weight is None:
        return value.mean()
    return _weighted_mean(value, sample_weight.to(value.device, dtype=value.dtype).reshape(-1))


def _decision_loss(output: Any, targets: Mapping[str, Any]) -> Any:
    batch_size = output.tactical_mode_logits.shape[0]
    device = output.tactical_mode_logits.device
    total = torch.zeros((), device=device)
    logits_by_field = {
        "tactical_mode": output.tactical_mode_logits,
        "task": output.task_logits,
        "target_slot": output.target_slot,
    }
    for field, logits in logits_by_field.items():
        target = targets[field].to(torch.long).reshape(batch_size)
        value = _balanced_categorical_cross_entropy(
            logits,
            target,
            _target_weight(targets, field, batch_size, device),
        )
        total = total + value
    continuous_by_field = {
        "goal_position": (output.goal_position_tensor, 4096.0),
        "waypoint_position": (output.waypoint_position_tensor, 4096.0),
        "facing_yaw_pitch": (output.facing_yaw_pitch, 180.0),
        "desired_range": (output.desired_range, 4096.0),
        "aggression": (output.aggression, 1.0),
        "risk": (output.risk, 1.0),
        "priority": (output.priority, 1.0),
        "ttl_ticks": (output.ttl_ticks, 64.0),
    }
    for field, (prediction, scale) in continuous_by_field.items():
        target = targets[field].to(prediction.dtype)
        value = ((prediction - target) / float(scale)).square()
        total = total + _weighted_mean(value, _target_weight(targets, field, batch_size, device))
    return total / (len(logits_by_field) + len(continuous_by_field))


def _masked_mean(value: Any, mask: Any) -> Any:
    mask = mask.to(dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _action_mask(targets: Mapping[str, Any], bit: int, shape: Any, device: Any) -> Any:
    value = targets.get("loss_mask")
    if value is None:
        return torch.ones(shape, dtype=torch.float32, device=device)
    return ((value.to(device=device, dtype=torch.int64) & int(bit)) != 0).to(torch.float32)


def _button_positive_weights(buttons_target: Any) -> Any:
    """Balance sparse buttons while keeping crouch/walk priors conservative."""

    positive_count = buttons_target.sum(dim=tuple(range(buttons_target.ndim - 1)))
    sample_count = torch.tensor(
        buttons_target.numel() // buttons_target.shape[-1],
        dtype=buttons_target.dtype,
        device=buttons_target.device,
    )
    negative_count = sample_count - positive_count
    button_pos_weight = torch.where(
        positive_count > 0.0,
        (negative_count / positive_count.clamp_min(1.0)).clamp(min=1.0, max=128.0),
        torch.ones_like(positive_count),
    )
    button_weight_cap = torch.full_like(button_pos_weight, 8.0)
    button_weight_cap[IN_ATTACK.bit_length() - 1] = 64.0
    button_weight_cap[IN_DUCK.bit_length() - 1] = 2.0
    button_weight_cap[IN_SPEED.bit_length() - 1] = 4.0
    return torch.minimum(button_pos_weight, button_weight_cap)


def _action_loss(output: Any, targets: Mapping[str, Any]) -> Any:
    movement_mean = 2.0 * output.movement_alpha / (
        output.movement_alpha + output.movement_beta
    ) - 1.0
    movement_target = targets["movement"].to(movement_mean.dtype)
    mouse_target = targets["mouse"].to(output.mouse_loc.dtype)
    buttons_target = targets["buttons"].to(output.button_logits.dtype)
    movement_mask = torch.stack(
        [
            _action_mask(targets, ACTION_MASK_FORWARD, movement_mean.shape[:-1], movement_mean.device),
            _action_mask(targets, ACTION_MASK_SIDE, movement_mean.shape[:-1], movement_mean.device),
            _action_mask(targets, ACTION_MASK_UP, movement_mean.shape[:-1], movement_mean.device),
        ],
        dim=-1,
    )
    mouse_mask = torch.stack(
        [
            _action_mask(targets, ACTION_MASK_YAW, output.mouse_loc.shape[:-1], output.mouse_loc.device),
            _action_mask(targets, ACTION_MASK_PITCH, output.mouse_loc.shape[:-1], output.mouse_loc.device),
        ],
        dim=-1,
    )
    buttons_mask = _action_mask(
        targets, ACTION_MASK_BUTTONS, output.button_logits.shape[:-1], output.button_logits.device
    )
    # The runtime decoder is deterministic (logit >= 0).  A plain BCE head
    # therefore learns the rare fire prior as "never fire".  Keep fire
    # learnable while bounding crouch/walk so sparse labels cannot become a
    # random movement policy.
    button_pos_weight = _button_positive_weights(buttons_target)
    movement = _masked_mean((movement_mean - movement_target).square(), movement_mask)
    mouse = _masked_mean((output.mouse_loc - mouse_target).square(), mouse_mask)
    button_value = F.binary_cross_entropy_with_logits(
        output.button_logits,
        buttons_target,
        reduction="none",
        pos_weight=button_pos_weight,
    )
    buttons = _masked_mean(button_value, buttons_mask)

    weapon_logits = output.weapon_logits.reshape(-1, output.weapon_logits.shape[-1])
    weapon_target = targets["weapon"].to(torch.long).reshape(-1)
    weapon_value = F.cross_entropy(weapon_logits, weapon_target, reduction="none")
    weapon_mask = _action_mask(
        targets, ACTION_MASK_WEAPON, output.weapon_logits.shape[:-1], output.weapon_logits.device
    ).reshape(-1)
    weapon = _masked_mean(weapon_value, weapon_mask)

    buy_logits = output.buy_logits.reshape(-1, output.buy_logits.shape[-1])
    buy_target = targets["buy"].to(torch.long).reshape(-1)
    buy_value = F.cross_entropy(buy_logits, buy_target, reduction="none")
    # A missing buy label means "do not emit a buy command" at runtime.  It
    # must not leave the categorical head unconstrained, otherwise an
    # otherwise valid action is rewritten by the permission guard every tick.
    buy_mask = torch.ones(
        output.buy_logits.shape[:-1], dtype=torch.float32, device=output.buy_logits.device
    ).reshape(-1)
    buy = _masked_mean(buy_value, buy_mask)
    return movement + mouse + buttons + weapon + buy


def _sequence_target_at(targets: Mapping[str, Any], index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in targets.items():
        if isinstance(value, Mapping):
            result[name] = _sequence_target_at(value, index)
        elif torch is not None and isinstance(value, torch.Tensor) and value.ndim >= 2:
            result[name] = value[:, index]
        else:
            result[name] = value
    return result


def _stack_action_outputs(outputs: Sequence[Any]) -> Any:
    if not outputs:
        raise ValueError("cannot stack an empty action output sequence")

    class ActionSequenceOutput:
        pass

    result = ActionSequenceOutput()
    for name in (
        "movement_alpha",
        "movement_beta",
        "mouse_loc",
        "mouse_scale",
        "mouse_mix_logits",
        "button_logits",
        "weapon_logits",
        "buy_logits",
    ):
        setattr(result, name, torch.stack([getattr(item, name) for item in outputs], dim=1))
    return result


def _decision_history_at(observations: Any, index: int) -> Any:
    """Return a fixed 32-token history, padding the beginning of a window."""

    history_start = index - (DECISION_MEMORY_TOKENS - 1)
    if history_start >= 0:
        return observations[:, history_start : index + 1]
    padding = observations[:, :1].expand(-1, -history_start, -1)
    return torch.cat((padding, observations[:, : index + 1]), dim=1)


def _build_predicted_intent_sequence(actor: Any, batch: Any) -> tuple[Any, tuple[Any, ...]]:
    """Roll the decision branch as runtime does and build the action intents."""

    _require_torch()
    observations = batch.observation_sequence
    if observations is None or batch.intent_embedding_sequence is None:
        raise ValueError("sequence intent rollout requires observation and intent sequences")
    if observations.ndim != 3 or observations.shape[1] < 1:
        raise ValueError("observation_sequence must have shape [batch, time, features]")
    memory = torch.zeros(
        batch.batch_size,
        DECISION_MEMORY_TOKENS,
        DECISION_MODEL_WIDTH,
        dtype=observations.dtype,
        device=observations.device,
    )
    current_intent = batch.previous_intent
    intent_chunks: list[Any] = []
    decision_outputs: list[Any] = []
    for index in range(0, int(observations.shape[1]), 8):
        history = _decision_history_at(observations, index)
        decision = actor.decision(history, current_intent, memory)
        predicted_intent = decision_output_to_intent_embedding(decision)
        end = min(int(observations.shape[1]), index + 8)
        intent_chunks.append(predicted_intent.unsqueeze(1).expand(-1, end - index, -1))
        decision_outputs.append(decision)
        memory = getattr(decision, "decision_memory", memory).detach()
        # Refreshes are independent through the cached-intent boundary while
        # the current output graph remains available to the action loss.
        current_intent = predicted_intent.detach()
    return torch.cat(intent_chunks, dim=1), tuple(decision_outputs)


def train_hierarchical_step(
    actor: HierarchicalActor,
    batch: HierarchicalBatchV1,
    *,
    decision_optimizer: Any,
    action_optimizer: Any,
    gradient_clip_norm: float | None = 1.0,
    train_decision: bool = True,
) -> dict[str, Any]:
    """Run one Demo update while keeping the two optimizer states independent."""

    _require_torch()
    validate_model_inputs(batch.model_inputs())
    actor.train()
    if train_decision:
        decision_optimizer.zero_grad(set_to_none=True)
    action_optimizer.zero_grad(set_to_none=True)
    device = batch.local_observation.device
    predicted_intent_sequence = None
    decision_outputs: list[Any] = []
    decision_targets: list[Mapping[str, Any]] = []
    if batch.observation_sequence is not None and batch.intent_embedding_sequence is not None:
        if train_decision:
            predicted_intent_sequence, decision_outputs = _build_predicted_intent_sequence(actor, batch)
        else:
            with torch.no_grad():
                predicted_intent_sequence, decision_outputs = _build_predicted_intent_sequence(actor, batch)
        target_sequence = batch.decision_target_sequence or {}
        decision_targets = [
            _sequence_target_at(target_sequence, index)
            for index in range(0, int(batch.observation_sequence.shape[1]), 8)
        ]
    elif train_decision:
        memory = torch.zeros(batch.batch_size, DECISION_MEMORY_TOKENS, DECISION_MODEL_WIDTH, device=device)
        decision_outputs.append(actor.decision(batch.observation_history, batch.previous_intent, memory))
        decision_targets.append(batch.decision_targets)
    if train_decision:
        decision_loss = torch.stack(
            [_decision_loss(output, targets) for output, targets in zip(decision_outputs, decision_targets)]
        ).mean()
    else:
        decision_loss = torch.zeros((), device=device)

    hidden = torch.zeros(batch.batch_size, ACTION_HIDDEN_SIZE, device=device)
    action_outputs: list[Any] = []
    if batch.observation_sequence is not None and batch.intent_embedding_sequence is not None:
        if predicted_intent_sequence is None:
            raise RuntimeError("predicted intent sequence was not built")
        for index in range(int(batch.observation_sequence.shape[1])):
            action = actor.action(
                batch.observation_sequence[:, index],
                predicted_intent_sequence[:, index],
                hidden,
            )
            action_outputs.append(action)
            hidden = action.recurrent_state.action
        action = _stack_action_outputs(action_outputs)
        action_targets = batch.action_target_sequence or batch.action_targets
    else:
        action = actor.action(batch.local_observation, batch.cached_intent, hidden)
        action_targets = batch.action_targets
    action_loss = _action_loss(action, action_targets)
    total = decision_loss + action_loss
    if not bool(torch.isfinite(total)):
        raise RuntimeError("hierarchical training produced a non-finite loss")
    total.backward()
    if gradient_clip_norm is not None:
        if train_decision:
            torch.nn.utils.clip_grad_norm_(
                actor.decision.parameters(),
                max_norm=gradient_clip_norm,
                error_if_nonfinite=True,
            )
        torch.nn.utils.clip_grad_norm_(
            actor.action.parameters(),
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
    if train_decision:
        decision_optimizer.step()
    action_optimizer.step()
    return {
        "loss": total.detach(),
        "decision_loss": decision_loss.detach(),
        "action_loss": action_loss.detach(),
    }


def prepare_hierarchical_training(actor: HierarchicalActor) -> HierarchicalActor:
    """Prepare both branches once at the beginning of Demo training."""

    from .quantization import (
        contains_fake_quant,
        prepare_action_fp32,
        prepare_decision_qat,
    )

    if not contains_fake_quant(actor.decision):
        prepare_decision_qat(actor.decision)
    prepare_action_fp32(actor.action)
    return actor


@dataclass
class EarlyStoppingV1:
    """Validation-only early stopping; the held-out test split is never consulted."""

    patience: int = 3
    best_validation_loss: float = math.inf
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("early-stopping patience must be positive")

    def update(self, validation_loss: float) -> bool:
        if not math.isfinite(validation_loss):
            raise ValueError("validation loss must be finite")
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped


def weighted_validation_loss(decision_loss: Any, action_loss: Any) -> Any:
    """The validation objective used by early stopping."""

    return decision_loss + action_loss


def build_hierarchical_checkpoint(
    path: str | Path,
    *,
    actor: HierarchicalActor,
    decision_optimizer: Any,
    action_optimizer: Any,
    qat_state: Mapping[str, Any] | None,
    split_manifest: Mapping[str, Any],
    labeler_version: str,
) -> dict[str, Any]:
    """Save a resumable checkpoint without merging decision/action optimizer state."""

    _require_torch()
    if qat_state is None:
        from .quantization import qat_state_dict

        qat_state = qat_state_dict(actor.decision)
    payload = {
        "model_state_dict": actor.state_dict(),
        "decision_optimizer_state_dict": decision_optimizer.state_dict(),
        "action_optimizer_state_dict": action_optimizer.state_dict(),
        "qat_state": dict(qat_state or {}),
        "split_manifest": dict(split_manifest),
        "labeler_version": str(labeler_version),
        "parameter_counts": {
            "decision": sum(parameter.numel() for parameter in actor.decision.parameters()),
            "action": sum(parameter.numel() for parameter in actor.action.parameters()),
            "total": sum(parameter.numel() for parameter in actor.parameters()),
        },
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return payload


def run_hierarchical_training_smoke(requested: str) -> dict[str, Any]:
    """Run one real FP32-master update on the requested training device."""

    _require_torch()
    from .training_device import resolve_training_device

    device = resolve_training_device(requested)
    torch.manual_seed(17)
    actor = prepare_hierarchical_training(HierarchicalActor()).to(device)
    from .quantization import contains_fake_quant
    batch_size = 1
    batch = HierarchicalBatchV1(
        observation_history=torch.randn(batch_size, 32, 256, device=device),
        previous_intent=torch.randn(batch_size, 128, device=device),
        local_observation=torch.randn(batch_size, 256, device=device),
        cached_intent=torch.randn(batch_size, 128, device=device),
        decision_targets={
            "tactical_mode": torch.zeros(batch_size, dtype=torch.long, device=device),
            "task": torch.zeros(batch_size, dtype=torch.long, device=device),
            "goal_position": torch.zeros(batch_size, 3, device=device),
            "waypoint_position": torch.zeros(batch_size, 3, device=device),
            "facing_yaw_pitch": torch.zeros(batch_size, 2, device=device),
            "desired_range": torch.zeros(batch_size, device=device),
            "target_slot": torch.full((batch_size,), 9, dtype=torch.long, device=device),
            "aggression": torch.zeros(batch_size, device=device),
            "risk": torch.zeros(batch_size, device=device),
            "priority": torch.zeros(batch_size, device=device),
            "ttl_ticks": torch.ones(batch_size, device=device),
        },
        action_targets={
            "movement": torch.zeros(batch_size, 3, device=device),
            "mouse": torch.zeros(batch_size, 2, device=device),
            "buttons": torch.zeros(batch_size, 32, device=device),
            "weapon": torch.zeros(batch_size, dtype=torch.long, device=device),
            "buy": torch.zeros(batch_size, dtype=torch.long, device=device),
        },
    )
    decision_optimizer = torch.optim.SGD(actor.decision.parameters(), lr=1e-4)
    action_optimizer = torch.optim.SGD(actor.action.parameters(), lr=1e-4)
    decision_before = tuple(parameter.detach().clone() for parameter in actor.decision.parameters())
    action_before = tuple(parameter.detach().clone() for parameter in actor.action.parameters())
    metrics = train_hierarchical_step(
        actor,
        batch,
        decision_optimizer=decision_optimizer,
        action_optimizer=action_optimizer,
    )
    decision_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(decision_before, actor.decision.parameters())
    )
    action_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(action_before, actor.action.parameters())
    )
    return {
        "device_type": device.type,
        "device_index": device.index,
        "loss": float(metrics["loss"].cpu()),
        "loss_finite": bool(torch.isfinite(metrics["loss"])),
        "decision_parameter_updated": decision_updated,
        "action_parameter_updated": action_updated,
        "fp32_master_weights": all(parameter.dtype == torch.float32 for parameter in actor.parameters()),
        "decision_qat_enabled": contains_fake_quant(actor.decision),
    }


@dataclass(frozen=True)
class HierarchicalSingleWaveReportV1:
    """Evidence for the one permitted MAPPO update in the test-only wave."""

    rollout_count: int
    wave_updates: int
    policy_generation: int
    next_wave_started: bool
    losses: Mapping[str, float]
    optimizer_parameter_names: tuple[str, ...]
    optimizer_parameter_ids: tuple[int, ...]
    action_state_sha256_before: str
    action_state_sha256_after: str
    action_onnx_sha256: str
    decision_parameter_updated: bool
    critic_parameter_updated: bool
    decision_qat_enabled: bool
    competition_result_read: bool
    checkpoint_path: Path
    decision_onnx_path: Path
    action_onnx_path: Path
    decision_graph: Mapping[str, Any]
    action_graph: Mapping[str, Any]
    actor_observation_fields: tuple[str, ...]
    critic_fields: tuple[str, ...]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict, key=str):
        value = state_dict[name]
        digest.update(str(name).encode("utf-8"))
        if torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _hierarchical_rollout_tensors(
    rollout: Any,
    *,
    device: Any,
) -> Any:
    """Build time-aligned actor inputs without fabricating recurrent history."""

    torch_module = _require_torch()
    from .training_mappo import _critic_features

    transitions = rollout.transitions
    bot_count = len(transitions[0].bot_ids)
    observations = torch_module.tensor(
        [
            list(observation)
            for transition in transitions
            for observation in transition.actor_observations
        ],
        dtype=torch_module.float32,
        device=device,
    ).reshape(len(transitions), bot_count, 256) / 255.0
    time_steps = len(transitions)
    observation_history = torch_module.zeros(
        time_steps,
        bot_count,
        DECISION_MEMORY_TOKENS,
        256,
        dtype=torch_module.float32,
        device=device,
    )
    for time_index in range(time_steps):
        start = max(0, time_index - DECISION_MEMORY_TOKENS + 1)
        window = observations[start : time_index + 1]
        observation_history[time_index, :, -window.shape[0] :] = window.transpose(0, 1)
    rewards = torch_module.tensor(
        [transition.rewards for transition in transitions],
        dtype=torch_module.float32,
        device=device,
    )
    dones = torch_module.tensor(
        [transition.dones for transition in transitions],
        dtype=torch_module.bool,
        device=device,
    )
    hidden_state_mask = torch_module.tensor(
        [transition.hidden_state_mask for transition in transitions],
        dtype=torch_module.bool,
        device=device,
    )
    state_faults = torch_module.tensor(
        [transition.state_faults for transition in transitions],
        dtype=torch_module.bool,
        device=device,
    )
    bot_ids = tuple(
        bot_id
        for transition in transitions
        for bot_id in transition.bot_ids
    )
    critic_features = _critic_features(
        rollout.critic_snapshots,
        bot_ids,
        len(transitions),
        bot_count,
        device,
    ).reshape(-1, 64)
    return {
        "observation_history": observation_history,
        "rewards": rewards,
        "dones": dones,
        "hidden_state_mask": hidden_state_mask,
        "state_faults": state_faults,
        "critic_features": critic_features.reshape(time_steps, bot_count, 64),
    }


def _unroll_behavior_decision(
    decision_actor: Any,
    tensors: Mapping[str, Any],
    *,
    device: Any,
    decision_actions: Sequence[Sequence[tuple[int, int, int] | None]],
    refresh_mask: Any | None = None,
    refresh_interval: int = 8,
) -> tuple[tuple[Any | None, ...], Any, Any, Any]:
    """Replay the behavior policy state used by the hierarchical runtime.

    The rollout stores the actual categorical decision and behavior log-probability.
    Previous intent and transformer memory are reconstructed only as inputs to the
    current policy evaluation, using the frozen behavior model and real sliding
    observation history; they are never replaced with repeated current observations
    or zeroed state for every transition.
    """

    torch_module = _require_torch()
    history = tensors["observation_history"]
    time_steps, bot_count = history.shape[:2]
    if len(decision_actions) != time_steps:
        raise ValueError("decision actions must align with rollout time")
    if any(len(row) != bot_count for row in decision_actions):
        raise ValueError("decision actions must align with rollout bot count")
    previous_intent = torch_module.zeros(
        bot_count,
        INTENT_EMBEDDING_SIZE,
        dtype=torch_module.float32,
        device=device,
    )
    memory = torch_module.zeros(
        bot_count,
        DECISION_MEMORY_TOKENS,
        DECISION_MODEL_WIDTH,
        dtype=torch_module.float32,
        device=device,
    )
    if refresh_mask is None:
        refresh_mask = (
            torch_module.arange(time_steps, device=device) % refresh_interval == 0
        ).unsqueeze(1).expand(time_steps, bot_count)
    else:
        refresh_mask = refresh_mask.to(device=device, dtype=torch_module.bool)
        if tuple(refresh_mask.shape) != (time_steps, bot_count):
            raise ValueError("decision refresh mask must align with rollout time and bots")
    previous_sequence: list[Any] = []
    memory_sequence: list[Any] = []
    outputs: list[Any | None] = []
    with torch_module.no_grad():
        for time_index in range(time_steps):
            previous_sequence.append(previous_intent.detach().clone())
            memory_sequence.append(memory.detach().clone())
            current_refresh = refresh_mask[time_index]
            if not bool(current_refresh.any()):
                outputs.append(None)
                continue
            output = decision_actor(
                history[time_index],
                previous_intent,
                memory,
            )
            outputs.append(output)
            labels_by_head: list[list[int]] = [[], [], []]
            for bot_index, stored_action in enumerate(decision_actions[time_index]):
                if stored_action is None:
                    fallback = (
                        int(output.tactical_mode_logits[bot_index].argmax()),
                        int(output.task_logits[bot_index].argmax()),
                        int(output.target_slot[bot_index].argmax()),
                    )
                    action = fallback
                else:
                    action = tuple(int(value) for value in stored_action)
                    if len(action) != 3:
                        raise ValueError("a decision action must contain three categorical labels")
                for head_index in range(3):
                    labels_by_head[head_index].append(action[head_index])
            label_tensors = tuple(
                torch_module.tensor(values, dtype=torch_module.long, device=device)
                for values in labels_by_head
            )
            next_intent = decision_output_to_intent_embedding(
                output,
                labels=label_tensors,
            ).detach()
            next_memory = output.decision_memory.detach()
            previous_intent = torch_module.where(
                current_refresh.unsqueeze(-1),
                next_intent,
                previous_intent,
            )
            memory = torch_module.where(
                current_refresh[:, None, None],
                next_memory,
                memory,
            )
    return (
        tuple(outputs),
        torch_module.stack(previous_sequence),
        torch_module.stack(memory_sequence),
        refresh_mask,
    )


def _decision_log_prob(output: Any, labels: Any) -> Any:
    """Joint log probability of the three categorical decision heads."""

    torch_module = _require_torch()
    tactical, task, target = labels
    return (
        torch_module.log_softmax(output.tactical_mode_logits, dim=-1)
        .gather(-1, tactical.to(torch_module.long).unsqueeze(-1))
        .squeeze(-1)
        + torch_module.log_softmax(output.task_logits, dim=-1)
        .gather(-1, task.to(torch_module.long).unsqueeze(-1))
        .squeeze(-1)
        + torch_module.log_softmax(output.target_slot, dim=-1)
        .gather(-1, target.to(torch_module.long).unsqueeze(-1))
        .squeeze(-1)
    )


def train_hierarchical_single_wave(
    actor: HierarchicalActor,
    rollouts: Sequence[Any],
    *,
    output_dir: str | Path,
    critic: Any | None = None,
    device: str = "cpu",
    decision_learning_rate: float = 1e-5,
    critic_learning_rate: float = 1e-4,
    demo_action_path: str | Path | None = None,
    demo_action_sha256: str | None = None,
) -> HierarchicalSingleWaveReportV1:
    """Run exactly one joint decision/critic update with the action branch frozen.

    The function intentionally has no action optimizer and never consumes competition
    scores.  It is a test-only single-wave boundary: complete Demo-only rollouts are
    validated first, all sequences are evaluated independently, and a single optimizer
    step is applied after their losses are aggregated.
    """

    torch_module = _require_torch()
    from .quantization import (
        contains_fake_quant,
        export_action_fp32,
        export_decision_int8,
        qat_state_dict,
        validate_action_fp32_graph,
        validate_decision_int8_graph,
    )
    from .training_device import resolve_training_device
    from .training_mappo import (
        CentralValueCritic,
        _atomic_torch_save,
        _state_dict_sha256,
        compute_gae,
        ppo_clip_objective,
        validate_hierarchical_rollouts,
    )

    validated = validate_hierarchical_rollouts(tuple(rollouts))
    if not contains_fake_quant(actor.decision):
        raise ValueError("single-wave MAPPO requires a decision model QAT-prepared before rollout collection")
    resolved_device = resolve_training_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    actor.to(resolved_device)
    behavior_decision = copy.deepcopy(actor.decision).to(resolved_device).eval()
    if critic is None:
        critic = CentralValueCritic(feature_size=64, hidden_size=128)
    if not hasattr(critic, "parameters") or not hasattr(critic, "forward"):
        raise TypeError("critic must be a PyTorch module")
    critic.to(resolved_device)

    action_requires_grad = tuple(parameter.requires_grad for parameter in actor.action.parameters())
    for parameter in actor.action.parameters():
        parameter.requires_grad_(False)
    actor.train()
    actor.action.eval()
    actor.decision.train()
    critic.train()

    action_state_before = _state_dict_sha256(actor.action.state_dict())
    decision_before = tuple(parameter.detach().cpu().clone() for parameter in actor.decision.parameters())
    critic_before = tuple(parameter.detach().cpu().clone() for parameter in critic.parameters())
    decision_parameters = list(actor.decision.named_parameters())
    critic_parameters = list(critic.named_parameters())
    optimizer_parameters = [parameter for _, parameter in decision_parameters + critic_parameters]
    optimizer_parameter_names = tuple(
        [f"decision.{name}" for name, _ in decision_parameters]
        + [f"critic.{name}" for name, _ in critic_parameters]
    )
    optimizer_parameter_ids = tuple(id(parameter) for parameter in optimizer_parameters)
    decision_optimizer = torch_module.optim.AdamW(optimizer_parameters[: len(decision_parameters)], lr=decision_learning_rate)
    critic_optimizer = torch_module.optim.AdamW(optimizer_parameters[len(decision_parameters) :], lr=critic_learning_rate)

    ppo_losses: list[Any] = []
    value_losses: list[Any] = []
    demo_kl_losses: list[Any] = []
    human_constraint_losses: list[Any] = []
    behavior_action_count = 0
    for rollout in validated:
        tensors = _hierarchical_rollout_tensors(
            rollout,
            device=resolved_device,
        )
        decision_refresh_mask = torch_module.tensor(
            [
                [decision_action is not None for decision_action in transition.decision_actions]
                for transition in rollout.transitions
            ],
            dtype=torch_module.bool,
            device=resolved_device,
        )
        behavior_outputs, previous_intent, memory, refresh_mask = _unroll_behavior_decision(
            behavior_decision,
            tensors,
            device=resolved_device,
            decision_actions=[transition.decision_actions for transition in rollout.transitions],
            refresh_mask=decision_refresh_mask,
        )
        time_steps, bot_count = tensors["rewards"].shape
        with torch_module.no_grad():
            old_values = critic(
                tensors["critic_features"].reshape(time_steps * bot_count, 64)
            ).reshape(time_steps, bot_count)
        bootstrap = torch_module.zeros(bot_count, dtype=torch_module.float32, device=resolved_device)
        advantages_by_bot: list[tuple[float, ...]] = []
        returns_by_bot: list[tuple[float, ...]] = []
        for bot_index in range(bot_count):
            advantage, returns = compute_gae(
                tensors["rewards"][:, bot_index].detach().cpu().tolist(),
                [
                    *old_values[:, bot_index].detach().cpu().tolist(),
                    float(bootstrap[bot_index].cpu()),
                ],
                tensors["dones"][:, bot_index].detach().cpu().tolist(),
            )
            advantages_by_bot.append(advantage)
            returns_by_bot.append(returns)
        advantages = torch_module.tensor(
            advantages_by_bot,
            dtype=torch_module.float32,
            device=resolved_device,
        ).transpose(0, 1)
        returns = torch_module.tensor(
            returns_by_bot,
            dtype=torch_module.float32,
            device=resolved_device,
        ).transpose(0, 1)
        current_values = critic(
            tensors["critic_features"].reshape(time_steps * bot_count, 64)
        ).reshape(time_steps, bot_count)
        # The hidden-state mask marks recurrent resets, not valid value
        # targets. Every non-faulted transition contributes to the critic;
        # using only reset rows would reduce a complete match to one sample.
        value_mask = ~tensors["state_faults"]
        if not bool(value_mask.any()):
            raise ValueError("hierarchical MAPPO rollout contains no valid critic targets")
        value_loss = torch_module.nn.functional.mse_loss(
            current_values[value_mask],
            returns[value_mask],
        )

        old_log_probs: list[Any] = []
        new_log_probs: list[Any] = []
        kl_losses: list[Any] = []
        actor_advantages: list[Any] = []
        current_decision_outputs: list[Any] = []
        for time_index, behavior_output in enumerate(behavior_outputs):
            if behavior_output is None:
                continue
            labels = []
            stored_log_probs = []
            selected_indices = []
            for bot_index, decision_action in enumerate(rollout.transitions[time_index].decision_actions):
                if decision_action is None:
                    continue
                stored = rollout.transitions[time_index].decision_log_probs[bot_index]
                if stored is None:
                    raise ValueError("decision behavior log probability is missing for a sampled action")
                labels.append(decision_action)
                selected_indices.append(bot_index)
                stored_log_probs.append(float(stored))
            if not labels:
                continue
            label_tensor = torch_module.tensor(labels, dtype=torch_module.long, device=resolved_device).unbind(dim=1)
            selected = torch_module.tensor(selected_indices, dtype=torch_module.long, device=resolved_device)
            current_output = actor.decision(
                tensors["observation_history"][time_index].index_select(0, selected),
                previous_intent[time_index].index_select(0, selected),
                memory[time_index].index_select(0, selected),
            )
            # The stored labels/log-probs are the decisions actually emitted by
            # the behavior runtime.  They are not replaced with class 0 or a
            # detached re-evaluation of the current policy.
            old_log_probs.append(
                torch_module.tensor(stored_log_probs, dtype=torch_module.float32, device=resolved_device)
            )
            new_log_probs.append(_decision_log_prob(current_output, label_tensor))
            behavior_distributions = (
                torch_module.softmax(behavior_output.tactical_mode_logits.index_select(0, selected).detach(), dim=-1),
                torch_module.softmax(behavior_output.task_logits.index_select(0, selected).detach(), dim=-1),
                torch_module.softmax(behavior_output.target_slot.index_select(0, selected).detach(), dim=-1),
            )
            current_distributions = (
                torch_module.log_softmax(current_output.tactical_mode_logits, dim=-1),
                torch_module.log_softmax(current_output.task_logits, dim=-1),
                torch_module.log_softmax(current_output.target_slot, dim=-1),
            )
            kl_losses.append(
                sum(
                    torch_module.nn.functional.kl_div(
                        current_log_probs,
                        reference_probs,
                        reduction="batchmean",
                    )
                    for current_log_probs, reference_probs in zip(
                        current_distributions,
                        behavior_distributions,
                    )
                )
            )
            actor_advantages.append(advantages[time_index].index_select(0, selected))
            current_decision_outputs.append(current_output)
            behavior_action_count += len(labels)
        if not new_log_probs:
            raise ValueError("hierarchical MAPPO rollout contains no usable sampled decision actions")
        old_log_prob = torch_module.cat(old_log_probs)
        new_log_prob = torch_module.cat(new_log_probs)
        actor_advantage = torch_module.cat(actor_advantages)
        if actor_advantage.numel() > 1:
            actor_advantage = (actor_advantage - actor_advantage.mean()) / actor_advantage.std(unbiased=False).clamp_min(1e-6)
        ppo_loss, _ = ppo_clip_objective(new_log_prob, old_log_prob, actor_advantage)
        demo_kl_loss = torch_module.stack(kl_losses).mean()
        human_constraint_loss = (
            torch_module.cat([output.aggression for output in current_decision_outputs]).square().mean()
            + torch_module.cat([output.risk for output in current_decision_outputs]).square().mean()
            + torch_module.cat([output.priority for output in current_decision_outputs]).square().mean()
        ) * 1e-3
        ppo_losses.append(ppo_loss)
        value_losses.append(value_loss)
        demo_kl_losses.append(demo_kl_loss)
        human_constraint_losses.append(human_constraint_loss)

    ppo_loss = torch_module.stack(ppo_losses).mean()
    value_loss = torch_module.stack(value_losses).mean()
    demo_kl_loss = torch_module.stack(demo_kl_losses).mean()
    human_constraint_loss = torch_module.stack(human_constraint_losses).mean()
    total_loss = ppo_loss + value_loss + demo_kl_loss + human_constraint_loss
    if not bool(torch_module.isfinite(total_loss)):
        raise RuntimeError("single-wave MAPPO produced a non-finite loss")
    decision_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch_module.nn.utils.clip_grad_norm_(actor.decision.parameters(), 1.0, error_if_nonfinite=True)
    torch_module.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
    decision_optimizer.step()
    critic_optimizer.step()

    decision_updated = any(
        not torch_module.equal(before, after.detach().cpu())
        for before, after in zip(decision_before, actor.decision.parameters())
    )
    critic_updated = any(
        not torch_module.equal(before, after.detach().cpu())
        for before, after in zip(critic_before, critic.parameters())
    )
    action_state_after = _state_dict_sha256(actor.action.state_dict())
    if action_state_before != action_state_after:
        raise RuntimeError("single-wave MAPPO changed the frozen action branch")

    decision_onnx_path = output_path / "decision-int8.onnx"
    action_onnx_path = Path(demo_action_path) if demo_action_path is not None else output_path / "demo-only-action-fp32.onnx"
    actor.eval()
    export_decision_int8(actor.decision, decision_onnx_path)
    if demo_action_path is None:
        export_action_fp32(actor.action, action_onnx_path)
    elif not action_onnx_path.is_file():
        raise FileNotFoundError(f"Demo-only FP32 action artifact not found: {action_onnx_path}")
    action_onnx_sha256 = _file_sha256(action_onnx_path)
    if demo_action_sha256 is not None and action_onnx_sha256 != demo_action_sha256:
        raise ValueError("Demo-only FP32 action artifact hash does not match the supplied hash")
    decision_graph = validate_decision_int8_graph(decision_onnx_path)
    action_graph = validate_action_fp32_graph(action_onnx_path)

    checkpoint_path = output_path / "hierarchical-mappo-single-wave.pt"
    qat_state = {"enabled": True, **qat_state_dict(actor.decision)}
    checkpoint = {
        "model_state_dict": actor.state_dict(),
        "decision_state_dict": actor.decision.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "decision_optimizer_state_dict": decision_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "decision_qat_state": qat_state,
        "action_state_sha256": action_state_before,
        "action_onnx_sha256": action_onnx_sha256,
        "action_onnx_path": str(action_onnx_path),
        "decision_onnx_path": str(decision_onnx_path),
        "optimizer_parameter_names": optimizer_parameter_names,
        "optimizer_parameter_ids": optimizer_parameter_ids,
        "actor_observation_fields": ("observation",),
        "critic_fields": ("critic_snapshots",),
        "policy_generation": validated[0].policy_generation,
        "parent_generation": validated[0].policy_generation,
        "next_policy_generation": None,
        "wave_updates": 1,
        "next_wave_started": False,
        "competition_result_read": False,
        "losses": {
            "ppo_loss": float(ppo_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "demo_kl_loss": float(demo_kl_loss.detach().cpu()),
            "human_constraint_loss": float(human_constraint_loss.detach().cpu()),
        },
        "rollout_metadata": tuple(
            {
                "instance_id": rollout.instance_id,
                "match_id": rollout.match_id,
                "policy_generation": rollout.policy_generation,
                "transition_count": len(rollout.transitions),
                "terminal": rollout.terminal,
                "truncated": rollout.truncated,
            }
            for rollout in validated
        ),
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    for parameter, requires_grad in zip(actor.action.parameters(), action_requires_grad):
        parameter.requires_grad_(requires_grad)

    return HierarchicalSingleWaveReportV1(
        rollout_count=len(validated),
        wave_updates=1,
        policy_generation=validated[0].policy_generation,
        next_wave_started=False,
        losses=checkpoint["losses"],
        optimizer_parameter_names=optimizer_parameter_names,
        optimizer_parameter_ids=optimizer_parameter_ids,
        action_state_sha256_before=action_state_before,
        action_state_sha256_after=action_state_after,
        action_onnx_sha256=action_onnx_sha256,
        decision_parameter_updated=decision_updated,
        critic_parameter_updated=critic_updated,
        decision_qat_enabled=contains_fake_quant(actor.decision),
        competition_result_read=False,
        checkpoint_path=checkpoint_path,
        decision_onnx_path=decision_onnx_path,
        action_onnx_path=action_onnx_path,
        decision_graph=decision_graph,
        action_graph=action_graph,
        actor_observation_fields=("observation",),
        critic_fields=("critic_snapshots",),
    )
