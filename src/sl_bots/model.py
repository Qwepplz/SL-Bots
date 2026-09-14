"""共享 Mirage Actor：槽位注意力、双速率 GRU 与条件 FiLM。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from .contracts import BotActionV1, BotObservationV1, OBSERVATION_RECORD_SIZE


try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    Tensor = Any
    nn = None
    F = None


@dataclass
class RecurrentStateV1:
    tactical: Any
    action: Any
    tick: Any

    @property
    def tactical_hidden(self) -> Any:
        return self.tactical

    @property
    def action_hidden(self) -> Any:
        return self.action


@dataclass
class PolicyOutputV1:
    movement_alpha: Any
    movement_beta: Any
    mouse_loc: Any
    mouse_scale: Any
    mouse_mix_logits: Any
    button_logits: Any
    weapon_logits: Any
    buy_logits: Any
    recurrent_state: RecurrentStateV1
    condition_embedding: Any
    action_vector: Any

    @property
    def state(self) -> RecurrentStateV1:
        return self.recurrent_state

    @property
    def movement_mean(self) -> Any:
        return 2.0 * self.movement_alpha / (self.movement_alpha + self.movement_beta) - 1.0

    @property
    def forward(self) -> Any:
        return self.movement_mean[:, 0]

    @property
    def side(self) -> Any:
        return self.movement_mean[:, 1]

    @property
    def up(self) -> Any:
        return self.movement_mean[:, 2]

    def to_actions(self, target_ticks: Sequence[int] | Any) -> tuple[BotActionV1, ...]:
        if torch is None:
            raise RuntimeError("PyTorch 2.9 is required to convert policy output")
        if isinstance(target_ticks, Tensor):
            target_values = target_ticks.detach().cpu().tolist()
        else:
            target_values = list(target_ticks)
        movement = self.movement_mean.detach().cpu()
        mouse = self.mouse_loc.detach().cpu()
        buttons = (torch.sigmoid(self.button_logits.detach().cpu()) >= 0.5).to(torch.int64)
        weapon = self.weapon_logits.detach().cpu().argmax(dim=-1)
        buy = self.buy_logits.detach().cpu().argmax(dim=-1)
        return tuple(
            BotActionV1(
                target_tick=int(target_values[index]),
                forward=float(movement[index, 0]),
                side=float(movement[index, 1]),
                up=float(movement[index, 2]),
                yaw_delta_deg=float(mouse[index, 0].clamp(-180.0, 180.0)),
                pitch_delta_deg=float(mouse[index, 1].clamp(-90.0, 90.0)),
                buttons=sum(int(buttons[index, bit]) << bit for bit in range(buttons.shape[1])),
                weapon_select=-1 if int(weapon[index]) == 0 else int(weapon[index]),
                buy_action=int(buy[index]),
                action_valid_mask=0xFF,
            )
            for index in range(len(target_values))
        )


def _torch_required() -> None:
    if torch is None:
        raise RuntimeError("PyTorch 2.9 is required for MirageActor")


def _payload_batch(observation: Any, device: Any) -> Any:
    _torch_required()
    if isinstance(observation, BotObservationV1):
        values = [observation.to_bytes()]
    elif isinstance(observation, (bytes, bytearray, memoryview)):
        values = [bytes(observation)]
    elif isinstance(observation, Tensor):
        values = observation
    elif isinstance(observation, dict) and "payload" in observation:
        return _payload_batch(observation["payload"], device)
    elif isinstance(observation, Sequence):
        values = list(observation)
    else:
        raise TypeError("observation must be a payload, tensor or sequence of payloads")
    if isinstance(values, Tensor):
        tensor = values.to(device=device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != OBSERVATION_RECORD_SIZE:
            raise ValueError(f"observation tensor must have shape [batch, {OBSERVATION_RECORD_SIZE}]")
        return torch.where(tensor.detach().amax() <= 1.0, tensor, tensor / 255.0)
    normalized = []
    for value in values:
        if isinstance(value, BotObservationV1):
            value = value.to_bytes()
        value = bytes(value)
        if len(value) != OBSERVATION_RECORD_SIZE:
            raise ValueError(f"every observation must be {OBSERVATION_RECORD_SIZE} bytes")
        normalized.append(list(value))
    if not normalized:
        raise ValueError("observation batch cannot be empty")
    return torch.tensor(normalized, device=device, dtype=torch.float32) / 255.0


if nn is not None:

    class MirageActor(nn.Module):
        tactical_hidden_size = 256
        action_hidden_size = 128
        observation_size = OBSERVATION_RECORD_SIZE

        def __init__(self, *, condition_size: int = 8, weapon_count: int = 16, buy_count: int = 32) -> None:
            super().__init__()
            self.weapon_count = weapon_count
            self.buy_count = buy_count
            self.self_encoder = nn.Sequential(nn.Linear(20, 64), nn.LayerNorm(64), nn.GELU())
            self.player_encoder = nn.Sequential(nn.Linear(12, 64), nn.LayerNorm(64), nn.GELU())
            self.sound_encoder = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
            self.event_encoder = nn.Sequential(nn.Linear(1, 16), nn.LayerNorm(16), nn.GELU())
            self.ray_encoder = nn.Sequential(nn.Linear(40, 32), nn.LayerNorm(32), nn.GELU())
            self.player_attention = nn.MultiheadAttention(64, 4, batch_first=True)
            self.sound_attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.event_attention = nn.MultiheadAttention(16, 4, batch_first=True)
            self.condition_encoder = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU())
            self.phase_side_encoder = nn.Sequential(nn.Linear(2, 8), nn.LayerNorm(8), nn.GELU())
            self.fusion = nn.Sequential(
                nn.Linear(64 + 64 + 32 + 16 + 32 + 32 + 8, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            self.tactical_gru = nn.GRUCell(256, self.tactical_hidden_size)
            self.tactical_film = nn.Linear(32, self.tactical_hidden_size * 2)
            self.action_input = nn.Sequential(
                nn.Linear(256 + self.tactical_hidden_size + 32, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            self.action_gru = nn.GRUCell(256, self.action_hidden_size)
            self.action_film = nn.Linear(32, self.action_hidden_size * 2)
            self.movement_head = nn.Linear(self.action_hidden_size, 6)
            self.mouse_loc_head = nn.Linear(self.action_hidden_size, 2)
            self.mouse_scale_head = nn.Linear(self.action_hidden_size, 2)
            self.mouse_mix_head = nn.Linear(self.action_hidden_size, 6)
            self.button_head = nn.Linear(self.action_hidden_size, 32)
            self.weapon_head = nn.Linear(self.action_hidden_size, weapon_count)
            self.buy_head = nn.Linear(self.action_hidden_size, buy_count)

        def initial_state(self, batch_size: int, *, device: Any | None = None) -> RecurrentStateV1:
            _torch_required()
            if not isinstance(batch_size, int) or not 1 <= batch_size <= 10:
                raise ValueError("batch_size must be between 1 and 10")
            if device is None:
                device = next(self.parameters()).device
            return RecurrentStateV1(
                tactical=torch.zeros(batch_size, self.tactical_hidden_size, device=device),
                action=torch.zeros(batch_size, self.action_hidden_size, device=device),
                tick=torch.zeros(batch_size, dtype=torch.long, device=device),
            )

        def _encode(self, observation: Any) -> tuple[Any, Any]:
            values = _payload_batch(observation, next(self.parameters()).device)
            batch_size = values.shape[0]
            if batch_size > 10:
                raise ValueError("batch size cannot exceed 10 bot slots")
            self_features = self.self_encoder(values[:, 8:28])
            player_values = values[:, 44:152].reshape(batch_size, 9, 12)
            player_tokens = self.player_encoder(player_values)
            player_tokens, _ = self.player_attention(player_tokens, player_tokens, player_tokens)
            player_features = player_tokens.mean(dim=1)
            sound_values = values[:, 152:200].reshape(batch_size, 16, 3)
            sound_tokens = self.sound_encoder(sound_values)
            sound_tokens, _ = self.sound_attention(sound_tokens, sound_tokens, sound_tokens)
            sound_features = sound_tokens.mean(dim=1)
            event_values = values[:, 200:216].reshape(batch_size, 16, 1)
            event_tokens = self.event_encoder(event_values)
            event_tokens, _ = self.event_attention(event_tokens, event_tokens, event_tokens)
            event_features = event_tokens.mean(dim=1)
            ray_features = self.ray_encoder(values[:, 216:256])
            condition_values = values[:, 28:44]
            condition_embedding = self.condition_encoder(condition_values)
            phase_side = self.phase_side_encoder(values[:, 5:7] * 3.0)
            fused = self.fusion(
                torch.cat(
                    [
                        self_features,
                        player_features,
                        sound_features,
                        event_features,
                        ray_features,
                        condition_embedding,
                        phase_side,
                    ],
                    dim=1,
                )
            )
            return fused, condition_embedding

        @staticmethod
        def _film(hidden: Any, layer: Any, condition: Any) -> Any:
            gamma, beta = layer(condition).chunk(2, dim=1)
            return torch.tanh(hidden * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta)

        def forward(
            self,
            observation: Any,
            recurrent_state: RecurrentStateV1 | None = None,
            *,
            delta_time_s: Any | None = None,
        ) -> PolicyOutputV1:
            fused, condition_embedding = self._encode(observation)
            batch_size = fused.shape[0]
            if recurrent_state is None:
                recurrent_state = self.initial_state(batch_size, device=fused.device)
            if recurrent_state.tactical.shape != (batch_size, self.tactical_hidden_size):
                raise ValueError("tactical recurrent state shape does not match observation batch")
            if recurrent_state.action.shape != (batch_size, self.action_hidden_size):
                raise ValueError("action recurrent state shape does not match observation batch")
            if recurrent_state.tick.shape != (batch_size,):
                raise ValueError("recurrent tick shape does not match observation batch")
            if delta_time_s is None:
                delta_steps = torch.ones(batch_size, dtype=torch.long, device=fused.device)
            else:
                durations = torch.as_tensor(
                    delta_time_s,
                    dtype=torch.float32,
                    device=fused.device,
                ).reshape(-1)
                if durations.numel() == 1:
                    durations = durations.expand(batch_size)
                if durations.shape != (batch_size,):
                    raise ValueError("delta_time_s must contain one value per observation")
                if bool((~torch.isfinite(durations)).any()) or bool((durations <= 0.0).any()):
                    raise ValueError("delta_time_s values must be finite and positive")
                delta_steps = torch.clamp(torch.round(durations * 128.0), min=1.0, max=4096.0).to(torch.long)
            tactical_candidate = self.tactical_gru(fused, recurrent_state.tactical)
            tactical_update = (
                (recurrent_state.tick.remainder(4) == 0) | (delta_steps > 1)
            ).unsqueeze(1)
            tactical = torch.where(tactical_update, tactical_candidate, recurrent_state.tactical)
            tactical = self._film(tactical, self.tactical_film, condition_embedding)
            duration_scale = delta_steps.to(fused.dtype).unsqueeze(1)
            action_input = self.action_input(
                torch.cat([fused * duration_scale, tactical, condition_embedding], dim=1)
            )
            action_candidate = self.action_gru(action_input, recurrent_state.action)
            action = self._film(action_candidate, self.action_film, condition_embedding)
            movement_parameters = self.movement_head(action)
            movement_alpha = F.softplus(movement_parameters[:, :3]) + 1.0
            movement_beta = F.softplus(movement_parameters[:, 3:]) + 1.0
            mouse_loc = self.mouse_loc_head(action)
            mouse_scale = F.softplus(self.mouse_scale_head(action)) + 1e-3
            mouse_mix_logits = self.mouse_mix_head(action).reshape(batch_size, 2, 3)
            button_logits = self.button_head(action)
            weapon_logits = self.weapon_head(action)
            buy_logits = self.buy_head(action)
            next_state = RecurrentStateV1(
                tactical=tactical,
                action=action,
                tick=recurrent_state.tick + delta_steps,
            )
            action_vector = torch.cat(
                [
                    2.0 * movement_alpha / (movement_alpha + movement_beta) - 1.0,
                    mouse_loc,
                    torch.sigmoid(button_logits),
                ],
                dim=1,
            )
            return PolicyOutputV1(
                movement_alpha=movement_alpha,
                movement_beta=movement_beta,
                mouse_loc=mouse_loc,
                mouse_scale=mouse_scale,
                mouse_mix_logits=mouse_mix_logits,
                button_logits=button_logits,
                weapon_logits=weapon_logits,
                buy_logits=buy_logits,
                recurrent_state=next_state,
                condition_embedding=condition_embedding,
                action_vector=action_vector,
            )

else:

    class MirageActor:
        tactical_hidden_size = 256
        action_hidden_size = 128
        observation_size = OBSERVATION_RECORD_SIZE

        def __init__(self, **_: Any) -> None:
            _torch_required()
