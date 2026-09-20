"""Two-rate, local-observation hierarchical actor for the test-only model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import DecisionOutputV1
from .model import PolicyOutputV1, RecurrentStateV1


try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    Tensor = Any
    nn = None
    F = None


OBSERVATION_FEATURES = 256
INTENT_EMBEDDING_SIZE = 128
DECISION_MEMORY_TOKENS = 32
DECISION_MODEL_WIDTH = 512
ACTION_HIDDEN_SIZE = 384


@dataclass
class HierarchicalStateV1:
    decision_memory: Any
    action_hidden: Any
    cached_intent: Any
    last_decision_tick: Any


@dataclass
class HierarchicalOutputV1:
    decision: DecisionOutputV1
    action: PolicyOutputV1
    state: HierarchicalStateV1


def decision_refresh_due(
    server_tick: int,
    last_decision_tick: int,
    intent_ttl_ticks: int,
    *,
    target_invalid: bool = False,
    important_event: bool = False,
    reflection_request_tick: int | None = None,
) -> bool:
    """Pure scheduler for 16 Hz decisions and event-driven refreshes."""

    if server_tick < 0 or last_decision_tick < -1:
        raise ValueError("ticks must be non-negative except last_decision_tick=-1")
    if intent_ttl_ticks < 0:
        raise ValueError("intent_ttl_ticks must be non-negative")
    if last_decision_tick < 0 or server_tick == 0:
        return True
    if server_tick % 8 == 0:
        return True
    if target_invalid or important_event:
        return True
    if intent_ttl_ticks and server_tick - last_decision_tick >= intent_ttl_ticks:
        return True
    if reflection_request_tick is not None and server_tick >= reflection_request_tick + 1:
        return True
    return False


def _categorical_intent_condition(logits: Any, label: Any | None) -> Any:
    """Return a hard canonical category with a straight-through gradient."""

    if logits.ndim != 2:
        raise ValueError("intent categorical logits must have shape [batch, classes]")
    probabilities = torch.softmax(logits, dim=-1)
    if label is None:
        indices = logits.argmax(dim=-1)
    elif isinstance(label, Tensor):
        indices = label.to(device=logits.device, dtype=torch.long)
    else:
        indices = torch.as_tensor(label, device=logits.device, dtype=torch.long)
    if indices.ndim == 0 and logits.shape[0] == 1:
        indices = indices.reshape(1)
    if tuple(indices.shape) != (logits.shape[0],):
        raise ValueError("intent categorical labels must have shape [batch]")
    if label is not None:
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= logits.shape[-1])):
            raise ValueError("intent categorical labels are outside the decision head range")
    hard = F.one_hot(indices, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    return probabilities + (hard - probabilities).detach()


def _finite_intent_tensor(value: Any) -> Any:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def decision_output_to_intent_embedding(
    output: DecisionOutputV1,
    *,
    labels: tuple[Any, Any, Any] | None = None,
) -> Any:
    """Build the canonical IntentV1 condition used by the action head.

    The forward value is a one-hot/canonical embedding. During training the
    categorical labels use a straight-through estimator so the decision loss
    can still receive gradients through the softmax policy distribution. A
    supplied ``labels`` tuple is the actual sampled behavior decision and is
    therefore also used when reconstructing a rollout.
    """

    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for hierarchical intent embeddings")
    if labels is not None and len(labels) != 3:
        raise ValueError("intent labels must contain tactical, task and target classes")
    tactical = _categorical_intent_condition(
        output.tactical_mode_logits,
        None if labels is None else labels[0],
    )
    task = _categorical_intent_condition(
        output.task_logits,
        None if labels is None else labels[1],
    )
    target_slot = _categorical_intent_condition(
        output.target_slot,
        None if labels is None else labels[2],
    )
    goal = _finite_intent_tensor(output.goal_position_tensor)
    waypoint = _finite_intent_tensor(output.waypoint_position_tensor)
    raw_facing = _finite_intent_tensor(output.facing_yaw_pitch)
    raw_yaw = raw_facing[..., 0]
    wrapped_yaw = torch.remainder(raw_yaw + 180.0, 360.0) - 180.0
    yaw = torch.where(
        (raw_yaw >= -180.0) & (raw_yaw <= 180.0),
        raw_yaw,
        wrapped_yaw,
    )
    yaw = torch.where(
        (wrapped_yaw == -180.0) & (raw_yaw > 0.0),
        yaw.new_full((), 180.0),
        yaw,
    )
    facing = torch.stack((yaw, raw_facing[..., 1].clamp(-89.0, 89.0)), dim=-1)
    desired_range = _finite_intent_tensor(output.desired_range).clamp(0.0, 4096.0).unsqueeze(-1)
    scalars = torch.stack(
        (
            _finite_intent_tensor(output.aggression).clamp(0.0, 1.0),
            _finite_intent_tensor(output.risk).clamp(0.0, 1.0),
            _finite_intent_tensor(output.priority).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    ttl = torch.trunc(torch.clamp_min(_finite_intent_tensor(output.ttl_ticks), 0.0)).unsqueeze(-1)
    # Runtime ONNX exposes only the decision heads below; confidence and
    # valid-mask tensors are contract defaults rather than model outputs.
    confidence = output.tactical_mode_logits.new_ones(
        output.tactical_mode_logits.shape[0], 11
    )
    valid_mask = torch.ones_like(confidence)
    values = torch.cat(
        (
            tactical,
            task,
            goal,
            waypoint,
            facing,
            desired_range,
            target_slot,
            scalars,
            ttl,
            confidence,
            valid_mask,
        ),
        dim=-1,
    )
    if values.shape[-1] > INTENT_EMBEDDING_SIZE:
        raise ValueError("decision intent embedding exceeds the runtime contract")
    return F.pad(values, (0, INTENT_EMBEDDING_SIZE - values.shape[-1]))


def canonical_observation_features(observation: Any) -> Any:
    """Decode the fixed observation bytes into stable semantic float features.

    The IPC contract is a 256-byte packed record, not a 256-float vector.  A
    raw byte MLP can memorize serialization patterns but cannot reliably learn
    health, bearings, distance, flags, and event ages.  This pure-tensor
    decoder keeps the model input shape unchanged and is used by both training
    and ONNX runtime paths.
    """

    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for observation decoding")
    if getattr(observation, "shape", None) is None or observation.shape[-1] != OBSERVATION_FEATURES:
        raise ValueError(f"observation must end with {OBSERVATION_FEATURES} bytes")
    raw = observation.to(dtype=torch.float32).clamp(0.0, 1.0) * 255.0

    def u16(offset: int) -> Any:
        return raw[..., offset] + 256.0 * raw[..., offset + 1]

    def s8(offset: int) -> Any:
        value = raw[..., offset]
        return torch.where(value >= 128.0, value - 256.0, value)

    def s16(offset: int) -> Any:
        value = u16(offset)
        return torch.where(value >= 32768.0, value - 65536.0, value)

    header = raw[..., :8] / 255.0
    self_flags = raw[..., 26] / 255.0
    self_features = torch.stack(
        (
            u16(8) / 65535.0,
            raw[..., 10] / 100.0,
            raw[..., 11] / 100.0,
            u16(12) / 65535.0,
            s16(14) / 4096.0,
            s16(16) / 4096.0,
            s16(18) / 4096.0,
            s16(20) / 1800.0,
            s16(22) / 900.0,
            s8(24) / 128.0,
            s8(25) / 128.0,
            self_flags,
            raw[..., 27] / 255.0,
            raw[..., 8] / 255.0,
            raw[..., 9] / 255.0,
            raw[..., 12] / 255.0,
            raw[..., 13] / 255.0,
            raw[..., 14] / 255.0,
            raw[..., 15] / 255.0,
            raw[..., 16] / 255.0,
        ),
        dim=-1,
    )
    condition = raw[..., 28:44] / 255.0

    player_features = []
    for index in range(9):
        offset = 44 + index * 12
        flags = raw[..., offset + 3]
        player_features.append(
            torch.stack(
                (
                    u16(offset) / 65535.0,
                    s8(offset + 2),
                    flags / 255.0,
                    s16(offset + 4) / 18000.0,
                    s8(offset + 6) / 90.0,
                    u16(offset + 7) / 4096.0,
                    raw[..., offset + 9] / 255.0,
                    raw[..., offset + 10] / 255.0,
                    raw[..., offset + 11] / 255.0,
                    torch.remainder(torch.floor(flags), 2.0),
                    torch.remainder(torch.floor(flags / 2.0), 2.0),
                    torch.remainder(torch.floor(flags / 4.0), 2.0),
                ),
                dim=-1,
            )
        )
    players = torch.cat(player_features, dim=-1)

    sound_features = []
    for index in range(16):
        offset = 152 + index * 3
        sound_features.append(
            torch.stack(
                (
                    torch.remainder(torch.floor(raw[..., offset]), 32.0) / 31.0,
                    s8(offset + 1) / 128.0,
                    torch.floor(raw[..., offset + 2] / 16.0) / 15.0,
                ),
                dim=-1,
            )
        )
    sounds = torch.cat(sound_features, dim=-1)

    event_values = raw[..., 200:216]
    events = torch.remainder(torch.floor(event_values), 32.0) / 31.0
    rays = raw[..., 216:256] / 63.0
    features = torch.cat((header, self_features, condition, players, sounds, events, rays), dim=-1)
    if features.shape[-1] != OBSERVATION_FEATURES:
        raise RuntimeError("canonical observation feature size drifted")
    return features


if nn is not None:

    class DecisionTransformer(nn.Module):
        """Five-layer Transformer that emits structured intent heads."""

        def __init__(
            self,
            *,
            observation_features: int = OBSERVATION_FEATURES,
            intent_embedding_size: int = INTENT_EMBEDDING_SIZE,
            history_tokens: int = DECISION_MEMORY_TOKENS,
            model_width: int = DECISION_MODEL_WIDTH,
            nhead: int = 8,
            ffn_width: int = 2048,
            layers: int = 5,
        ) -> None:
            super().__init__()
            if history_tokens != DECISION_MEMORY_TOKENS or model_width != DECISION_MODEL_WIDTH:
                raise ValueError("the test-only decision model has a fixed state shape")
            self.history_tokens = history_tokens
            self.model_width = model_width
            self.input_projection = nn.Linear(observation_features + intent_embedding_size, model_width)
            self.position_embedding = nn.Parameter(torch.zeros(1, history_tokens, model_width))
            layer = nn.TransformerEncoderLayer(
                d_model=model_width,
                nhead=nhead,
                dim_feedforward=ffn_width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.output_norm = nn.LayerNorm(model_width)
            self.tactical_mode_head = nn.Linear(model_width, 8)
            self.task_head = nn.Linear(model_width, 8)
            self.goal_head = nn.Linear(model_width, 3)
            self.waypoint_head = nn.Linear(model_width, 3)
            self.facing_head = nn.Linear(model_width, 2)
            self.range_head = nn.Linear(model_width, 1)
            self.target_slot_head = nn.Linear(model_width, 10)
            self.scalar_head = nn.Linear(model_width, 4)
            self.ttl_head = nn.Linear(model_width, 1)
            self.confidence_head = nn.Linear(model_width, len((
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
            )))

        def forward(
            self,
            observation_history: Tensor,
            previous_intent: Tensor,
            memory: Tensor,
        ) -> DecisionOutputV1:
            observation_history = canonical_observation_features(observation_history)
            if observation_history.ndim == 2:
                observation_history = observation_history.unsqueeze(1)
            if previous_intent.ndim == 2:
                previous_intent = previous_intent.unsqueeze(1).expand(
                    -1, observation_history.shape[1], -1
                )
            if observation_history.ndim != 3 or previous_intent.ndim != 3:
                raise ValueError("decision inputs must be [batch, history, features]")
            if observation_history.shape[1] != self.history_tokens:
                raise ValueError(f"decision history must contain {self.history_tokens} tokens")
            if previous_intent.shape[:2] != observation_history.shape[:2]:
                raise ValueError("previous_intent history shape must match observations")
            expected_memory = (
                observation_history.shape[0],
                self.history_tokens,
                self.model_width,
            )
            if tuple(memory.shape) != expected_memory:
                raise ValueError(f"decision memory must have shape {expected_memory}")
            tokens = self.input_projection(torch.cat([observation_history, previous_intent], dim=-1))
            tokens = tokens + self.position_embedding + memory
            encoded = self.output_norm(self.encoder(tokens))
            hidden = encoded[:, -1]
            scalars = self.scalar_head(hidden)
            output = DecisionOutputV1(
                tactical_mode=self.tactical_mode_head(hidden),
                task=self.task_head(hidden),
                goal_position=self.goal_head(hidden),
                waypoint_position=self.waypoint_head(hidden),
                facing_yaw_pitch=self.facing_head(hidden),
                desired_range=self.range_head(hidden).squeeze(-1),
                target_slot=self.target_slot_head(hidden),
                aggression=scalars[:, 0],
                risk=scalars[:, 1],
                priority=scalars[:, 2],
                ttl_ticks=self.ttl_head(hidden).squeeze(-1),
                confidence={"all": torch.sigmoid(self.confidence_head(hidden))},
                valid_mask={"all": torch.ones_like(self.confidence_head(hidden), dtype=torch.bool)},
            )
            object.__setattr__(output, "decision_memory", encoded)
            return output


    class ActionPolicy(nn.Module):
        """128 Hz local reaction policy with a 384-unit FP32 GRU."""

        def __init__(
            self,
            *,
            observation_features: int = OBSERVATION_FEATURES,
            intent_embedding_size: int = INTENT_EMBEDDING_SIZE,
            hidden_size: int = ACTION_HIDDEN_SIZE,
            weapon_count: int = 16,
            buy_count: int = 32,
        ) -> None:
            super().__init__()
            if hidden_size != ACTION_HIDDEN_SIZE:
                raise ValueError("the test-only action model has a fixed hidden shape")
            self.hidden_size = hidden_size
            self.local_encoder = nn.Sequential(
                nn.Linear(observation_features, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            self.intent_encoder = nn.Sequential(
                nn.Linear(intent_embedding_size, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            )
            self.action_gru = nn.GRUCell(320, hidden_size)
            self.movement_head = nn.Linear(hidden_size, 6)
            self.mouse_loc_head = nn.Linear(hidden_size, 2)
            self.mouse_scale_head = nn.Linear(hidden_size, 2)
            self.mouse_mix_head = nn.Linear(hidden_size, 6)
            self.button_head = nn.Linear(hidden_size, 32)
            self.weapon_head = nn.Linear(hidden_size, weapon_count)
            self.buy_head = nn.Linear(hidden_size, buy_count)

        def forward(self, local_observation: Tensor, cached_intent: Tensor, hidden: Tensor) -> PolicyOutputV1:
            local_observation = canonical_observation_features(local_observation)
            if local_observation.ndim != 2 or local_observation.shape[1] != OBSERVATION_FEATURES:
                raise ValueError("local_observation must have shape [batch, 256]")
            if cached_intent.shape != (local_observation.shape[0], INTENT_EMBEDDING_SIZE):
                raise ValueError("cached_intent must have shape [batch, 128]")
            if hidden.shape != (local_observation.shape[0], self.hidden_size):
                raise ValueError("action hidden state shape does not match local observation")
            local = self.local_encoder(local_observation)
            intent = self.intent_encoder(cached_intent)
            action = self.action_gru(torch.cat([local, intent], dim=-1), hidden)
            movement = self.movement_head(action)
            alpha = F.softplus(movement[:, :3]) + 1.0
            beta = F.softplus(movement[:, 3:]) + 1.0
            mouse_loc = self.mouse_loc_head(action)
            mouse_scale = F.softplus(self.mouse_scale_head(action)) + 1e-3
            return PolicyOutputV1(
                movement_alpha=alpha,
                movement_beta=beta,
                mouse_loc=mouse_loc,
                mouse_scale=mouse_scale,
                mouse_mix_logits=self.mouse_mix_head(action).reshape(-1, 2, 3),
                button_logits=self.button_head(action),
                weapon_logits=self.weapon_head(action),
                buy_logits=self.buy_head(action),
                recurrent_state=RecurrentStateV1(
                    tactical=local.new_zeros((local.shape[0], 0)),
                    action=action,
                    tick=local.new_zeros((local.shape[0],), dtype=torch.long),
                ),
                condition_embedding=intent,
                action_vector=torch.cat([2.0 * alpha / (alpha + beta) - 1.0, mouse_loc], dim=1),
            )

        def forward_sequence(
            self,
            local_observation: Tensor,
            cached_intent: Tensor,
            hidden: Tensor,
        ) -> tuple[PolicyOutputV1, ...]:
            """Unroll the same recurrent action policy over a complete Demo window."""

            if local_observation.ndim != 3 or local_observation.shape[-1] != OBSERVATION_FEATURES:
                raise ValueError("local_observation sequence must have shape [batch, time, 256]")
            if cached_intent.shape[:2] != local_observation.shape[:2]:
                raise ValueError("cached_intent sequence must match local_observation time shape")
            outputs = []
            state = hidden
            for index in range(local_observation.shape[1]):
                output = self.forward(
                    local_observation[:, index],
                    cached_intent[:, index],
                    state,
                )
                outputs.append(output)
                state = output.recurrent_state.action
            return tuple(outputs)


    class HierarchicalActor(nn.Module):
        """Shared-weight actor with independent per-Bot hierarchical state."""

        def __init__(self) -> None:
            super().__init__()
            self.decision = DecisionTransformer()
            self.action = ActionPolicy()

        def initial_state(self, batch_size: int, *, device: Any | None = None) -> HierarchicalStateV1:
            if not isinstance(batch_size, int) or batch_size < 1:
                raise ValueError("batch_size must be positive")
            if device is None:
                device = next(self.parameters()).device
            return HierarchicalStateV1(
                decision_memory=torch.zeros(
                    batch_size, DECISION_MEMORY_TOKENS, DECISION_MODEL_WIDTH, device=device
                ),
                action_hidden=torch.zeros(batch_size, ACTION_HIDDEN_SIZE, device=device),
                cached_intent=torch.zeros(batch_size, INTENT_EMBEDDING_SIZE, device=device),
                last_decision_tick=torch.full((batch_size,), -1, dtype=torch.long, device=device),
            )

        def forward(
            self,
            observation_history: Tensor,
            local_observation: Tensor,
            state: HierarchicalStateV1,
        ) -> HierarchicalOutputV1:
            decision = self.decision(
                observation_history,
                state.cached_intent,
                state.decision_memory,
            )
            predicted_intent = decision_output_to_intent_embedding(decision)
            action = self.action(local_observation, predicted_intent, state.action_hidden)
            next_state = HierarchicalStateV1(
                decision_memory=getattr(decision, "decision_memory", state.decision_memory),
                action_hidden=action.recurrent_state.action,
                cached_intent=predicted_intent,
                last_decision_tick=state.last_decision_tick,
            )
            return HierarchicalOutputV1(decision=decision, action=action, state=next_state)


else:

    class DecisionTransformer:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch 2.9 is required for DecisionTransformer")


    class ActionPolicy:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch 2.9 is required for ActionPolicy")


    class HierarchicalActor:
        def __init__(self, **_: Any) -> None:
            raise RuntimeError("PyTorch 2.9 is required for HierarchicalActor")
