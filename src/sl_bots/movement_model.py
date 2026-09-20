"""Stateless v3 movement Transformer and 128 Hz reaction policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .hierarchical_model import canonical_observation_features
from .entity_tokens import TOKEN_COUNT, TOKEN_WIDTH, legal_entity_tokens


MOVEMENT_MODEL_WIDTH = 256
MOVEMENT_MODEL_LAYERS = 4
MOVEMENT_MODEL_HEADS = 1
MOVEMENT_MODEL_FFN_WIDTH = 2048
MOVEMENT_HORIZON_COUNT = 3
MOVEMENT_CLASS_COUNT = 17
STANCE_CLASS_COUNT = 3
JUMP_CLASS_COUNT = 2
MOVEMENT_PLAN_SIZE = 75
ACTION_HIDDEN_SIZE_V3 = 384


@dataclass
class MovementOutputV1:
    move_logits: Tensor
    stance_logits: Tensor
    jump_logits: Tensor
    plan_embedding: Tensor


@dataclass
class ReactiveActionOutputV3:
    mouse_loc: Tensor
    mouse_scale: Tensor
    mouse_mix_logits: Tensor
    button_logits: Tensor
    weapon_logits: Tensor
    buy_logits: Tensor
    next_hidden: Tensor
    plan_condition: Tensor


def _check_observation(observation: Tensor) -> None:
    if observation.ndim != 2 or observation.shape[-1] != 256:
        raise ValueError("observation must have shape [batch,256]")


class LegalEntityMovementTransformer(nn.Module):
    """A fixed-size, current-observation-only movement classifier."""

    def __init__(
        self,
        *,
        token_width: int = MOVEMENT_MODEL_WIDTH,
        layers: int = MOVEMENT_MODEL_LAYERS,
        heads: int = MOVEMENT_MODEL_HEADS,
        ffn_width: int = MOVEMENT_MODEL_FFN_WIDTH,
        horizons: int = MOVEMENT_HORIZON_COUNT,
    ) -> None:
        super().__init__()
        if (token_width, layers, heads, ffn_width, horizons) != (
            MOVEMENT_MODEL_WIDTH,
            MOVEMENT_MODEL_LAYERS,
            MOVEMENT_MODEL_HEADS,
            MOVEMENT_MODEL_FFN_WIDTH,
            MOVEMENT_HORIZON_COUNT,
        ):
            raise ValueError("v3 movement model uses a fixed 256x4x1/2048 contract")
        self.token_width = token_width
        self.horizons = horizons
        self.type_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(TOKEN_WIDTH, token_width),
                    nn.GELU(),
                    nn.Linear(token_width, token_width),
                    nn.GELU(),
                    nn.Linear(token_width, token_width),
                )
                for _ in range(6)
            ]
        )
        self.token_type_embedding = nn.Embedding(6, token_width)
        self.slot_embedding = nn.Embedding(TOKEN_COUNT, token_width)
        layer = nn.TransformerEncoderLayer(
            d_model=token_width,
            nhead=heads,
            dim_feedforward=ffn_width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(token_width)
        self.move_head = nn.Linear(token_width, horizons * MOVEMENT_CLASS_COUNT)
        self.stance_head = nn.Linear(token_width, horizons * STANCE_CLASS_COUNT)
        self.jump_head = nn.Linear(token_width, horizons * JUMP_CLASS_COUNT)

    def _project_tokens(self, features: Tensor, token_type: Tensor) -> Tensor:
        projected = torch.zeros(
            features.shape[0], features.shape[1], self.token_width,
            dtype=features.dtype,
            device=features.device,
        )
        for token_kind, projection in enumerate(self.type_projections):
            candidate = projection(features)
            mask = (token_type == token_kind).view(1, -1, 1)
            projected = torch.where(mask, candidate, projected)
        return projected

    def forward(self, observation: Tensor) -> MovementOutputV1:
        _check_observation(observation)
        tokens = legal_entity_tokens(observation)
        hidden = self._project_tokens(tokens.features, tokens.token_type)
        hidden = hidden + self.token_type_embedding(tokens.token_type).unsqueeze(0)
        hidden = hidden + self.slot_embedding.weight.unsqueeze(0)
        encoded = self.encoder(hidden, src_key_padding_mask=~tokens.valid_mask)
        self_hidden = self.output_norm(encoded[:, 0])
        move_logits = self.move_head(self_hidden).reshape(-1, self.horizons, MOVEMENT_CLASS_COUNT)
        stance_logits = self.stance_head(self_hidden).reshape(-1, self.horizons, STANCE_CLASS_COUNT)
        jump_logits = self.jump_head(self_hidden).reshape(-1, self.horizons, JUMP_CLASS_COUNT)
        move_probabilities = torch.softmax(move_logits, dim=-1)
        stance_probabilities = torch.softmax(stance_logits, dim=-1)
        jump_probabilities = torch.softmax(jump_logits, dim=-1)
        plan_parts = []
        for horizon in range(self.horizons):
            move = move_probabilities[:, horizon]
            stance = stance_probabilities[:, horizon]
            jump = jump_probabilities[:, horizon]
            moving_probability = 1.0 - move[:, 0:1]
            confidence = torch.maximum(
                torch.maximum(move.max(dim=-1, keepdim=True).values, stance.max(dim=-1, keepdim=True).values),
                jump.max(dim=-1, keepdim=True).values,
            )
            valid = torch.ones_like(confidence)
            plan_parts.append(torch.cat((move, stance, jump, moving_probability, confidence, valid), dim=-1))
        plan_embedding = torch.cat(plan_parts, dim=-1)
        return MovementOutputV1(move_logits, stance_logits, jump_logits, plan_embedding)


class ReactiveActionPolicyV3(nn.Module):
    """FP32 128 Hz reaction policy conditioned on a predicted movement plan."""

    def __init__(
        self,
        *,
        plan_size: int = MOVEMENT_PLAN_SIZE,
        hidden_size: int = ACTION_HIDDEN_SIZE_V3,
        weapon_count: int = 16,
        buy_count: int = 32,
    ) -> None:
        super().__init__()
        if plan_size != MOVEMENT_PLAN_SIZE or hidden_size != ACTION_HIDDEN_SIZE_V3:
            raise ValueError("v3 reaction policy uses plan [batch,75] and hidden [batch,384]")
        self.hidden_size = hidden_size
        self.local_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.plan_encoder = nn.Sequential(
            nn.Linear(plan_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.action_gru = nn.GRUCell(320, hidden_size)
        self.mouse_loc_head = nn.Linear(hidden_size, 2)
        self.mouse_scale_head = nn.Linear(hidden_size, 2)
        self.mouse_mix_head = nn.Linear(hidden_size, 6)
        self.button_head = nn.Linear(hidden_size, 32)
        self.weapon_head = nn.Linear(hidden_size, weapon_count)
        self.buy_head = nn.Linear(hidden_size, buy_count)

    def forward(self, local_observation: Tensor, movement_plan: Tensor, hidden: Tensor) -> ReactiveActionOutputV3:
        _check_observation(local_observation)
        if movement_plan.shape != (local_observation.shape[0], MOVEMENT_PLAN_SIZE):
            raise ValueError("movement_plan must have shape [batch,75]")
        if hidden.shape != (local_observation.shape[0], self.hidden_size):
            raise ValueError("hidden must have shape [batch,384]")
        local = self.local_encoder(canonical_observation_features(local_observation))
        plan_condition = self.plan_encoder(movement_plan)
        next_hidden = self.action_gru(torch.cat((local, plan_condition), dim=-1), hidden)
        return ReactiveActionOutputV3(
            mouse_loc=self.mouse_loc_head(next_hidden),
            mouse_scale=F.softplus(self.mouse_scale_head(next_hidden)) + 1e-3,
            mouse_mix_logits=self.mouse_mix_head(next_hidden).reshape(-1, 2, 3),
            button_logits=self.button_head(next_hidden),
            weapon_logits=self.weapon_head(next_hidden),
            buy_logits=self.buy_head(next_hidden),
            next_hidden=next_hidden,
            plan_condition=plan_condition,
        )

    def forward_sequence(self, local_observation: Tensor, movement_plan: Tensor, hidden: Tensor) -> tuple[ReactiveActionOutputV3, ...]:
        if local_observation.ndim != 3 or local_observation.shape[-1] != 256:
            raise ValueError("local_observation sequence must have shape [batch,time,256]")
        if movement_plan.shape[:2] != local_observation.shape[:2] or movement_plan.shape[-1] != MOVEMENT_PLAN_SIZE:
            raise ValueError("movement_plan sequence must have shape [batch,time,75]")
        outputs: list[ReactiveActionOutputV3] = []
        state = hidden
        for index in range(local_observation.shape[1]):
            output = self.forward(local_observation[:, index], movement_plan[:, index], state)
            outputs.append(output)
            state = output.next_hidden
        return tuple(outputs)


def migrate_v2_action_weights(
    target: ReactiveActionPolicyV3,
    source: nn.Module | None = None,
) -> dict[str, tuple[str, ...]]:
    """Copy shape-compatible v2 reaction weights and report every decision."""

    if not isinstance(target, ReactiveActionPolicyV3):
        raise TypeError("target must be ReactiveActionPolicyV3")
    if source is None:
        from .hierarchical_model import ActionPolicy

        source = ActionPolicy()
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied: list[str] = []
    reinitialized: list[str] = []
    with torch.no_grad():
        for name, value in target_state.items():
            if name.startswith("plan_encoder."):
                reinitialized.append(name)
                continue
            candidate = source_state.get(name)
            if candidate is not None and candidate.shape == value.shape:
                value.copy_(candidate)
                copied.append(name)
            else:
                reinitialized.append(name)
        target.load_state_dict(target_state)
    return {"copied": tuple(copied), "reinitialized": tuple(reinitialized)}


class MovementActorV3(nn.Module):
    """Convenience wrapper keeping movement and reaction state separate."""

    def __init__(self) -> None:
        super().__init__()
        self.movement = LegalEntityMovementTransformer()
        self.reaction = ReactiveActionPolicyV3()

    def forward(self, observation: Tensor, hidden: Tensor) -> tuple[MovementOutputV1, ReactiveActionOutputV3]:
        movement = self.movement(observation)
        reaction = self.reaction(observation, movement.plan_embedding, hidden)
        return movement, reaction
