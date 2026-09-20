"""Pure-tensor tokenization of the legal 256-byte observation payload."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


OBSERVATION_BYTES = 256
TOKEN_COUNT = 44
TOKEN_WIDTH = 32
PLAYER_TOKEN_COUNT = 9
SOUND_TOKEN_COUNT = 16
EVENT_TOKEN_COUNT = 16
SELF_TOKEN_INDEX = 0
PLAYER_TOKEN_START = 1
SOUND_TOKEN_START = 10
EVENT_TOKEN_START = 26
CONTEXT_TOKEN_INDEX = 42
GEOMETRY_TOKEN_INDEX = 43

PLAYER_DIRECT = 1 << 0
PLAYER_RADAR = 1 << 1
PLAYER_AUDIBLE = 1 << 2
PLAYER_ALIVE = 1 << 3
PLAYER_KNOWN = 1 << 4
PLAYER_PERIPHERAL = 1 << 5


@dataclass(frozen=True)
class LegalEntityTokensV1:
    features: torch.Tensor
    valid_mask: torch.Tensor
    token_type: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or tuple(self.features.shape[1:]) != (TOKEN_COUNT, TOKEN_WIDTH):
            raise ValueError("features must have shape [batch,44,32]")
        if self.features.dtype != torch.float32:
            raise ValueError("features must be float32")
        if self.valid_mask.dtype is not torch.bool or tuple(self.valid_mask.shape) != (
            self.features.shape[0],
            TOKEN_COUNT,
        ):
            raise ValueError("valid_mask must have shape [batch,44] and bool dtype")
        if self.token_type.dtype is not torch.int64 or tuple(self.token_type.shape) != (TOKEN_COUNT,):
            raise ValueError("token_type must have shape [44] and int64 dtype")


def _raw_bytes(observation: torch.Tensor) -> torch.Tensor:
    if not isinstance(observation, torch.Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_BYTES:
        raise ValueError(f"observation must have shape [batch,{OBSERVATION_BYTES}]")
    return observation.to(dtype=torch.float32).clamp(0.0, 1.0) * 255.0


def _u16(raw: torch.Tensor, offset: int) -> torch.Tensor:
    return raw[..., offset] + 256.0 * raw[..., offset + 1]


def _s8(raw: torch.Tensor, offset: int) -> torch.Tensor:
    value = raw[..., offset]
    return torch.where(value >= 128.0, value - 256.0, value)


def _s16(raw: torch.Tensor, offset: int) -> torch.Tensor:
    value = _u16(raw, offset)
    return torch.where(value >= 32768.0, value - 65536.0, value)


def _bit(value: torch.Tensor, bit: int) -> torch.Tensor:
    return torch.remainder(torch.floor(value / float(1 << bit)), 2.0)


def _pad(features: torch.Tensor, width: int = TOKEN_WIDTH) -> torch.Tensor:
    missing = width - features.shape[-1]
    if missing < 0:
        raise ValueError("token feature layout exceeds token width")
    if missing == 0:
        return features
    return torch.cat((features, torch.zeros((*features.shape[:-1], missing), dtype=features.dtype, device=features.device)), dim=-1)


def _self_token(raw: torch.Tensor) -> torch.Tensor:
    yaw = _s16(raw, 20) / 10.0 * math.pi / 180.0
    flags = raw[..., 26]
    position = torch.stack((_s16(raw, 14) / 4096.0, _s16(raw, 16) / 4096.0, _s16(raw, 18) / 4096.0), dim=-1)
    flag_bits = torch.stack(tuple(_bit(flags, bit) for bit in range(8)), dim=-1)
    core = torch.stack(
        (
            raw[..., 4] / 255.0,
            raw[..., 5] / 255.0,
            raw[..., 7] / 255.0,
            _u16(raw, 8) / 65535.0,
            raw[..., 10] / 100.0,
            raw[..., 11] / 100.0,
            _u16(raw, 12) / 65535.0,
            position[..., 0],
            position[..., 1],
            position[..., 2],
            torch.sin(yaw),
            torch.cos(yaw),
            _s16(raw, 22) / 900.0,
            _s8(raw, 24) / 128.0,
            _s8(raw, 25) / 128.0,
            raw[..., 27] / 255.0,
            raw[..., 36] / 255.0,
            raw[..., 37] / 255.0,
            raw[..., 38] / 255.0,
            raw[..., 41] / 255.0,
        ),
        dim=-1,
    )
    return _pad(torch.cat((core, flag_bits), dim=-1))


def _player_tokens(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    masks = []
    for index in range(PLAYER_TOKEN_COUNT):
        offset = 44 + index * 12
        relation_raw = _s8(raw, offset + 2)
        relation = torch.where(relation_raw < 0.0, -torch.ones_like(relation_raw), torch.where(relation_raw > 0.0, torch.ones_like(relation_raw), torch.zeros_like(relation_raw)))
        flags = raw[..., offset + 3]
        bearing = _s16(raw, offset + 4) / 100.0 * math.pi / 180.0
        direct = _bit(flags, 0)
        radar = _bit(flags, 1)
        audible = _bit(flags, 2)
        alive = _bit(flags, 3)
        known = _bit(flags, 4)
        peripheral = _bit(flags, 5)
        features = torch.stack(
            (
                relation,
                flags / 255.0,
                _s8(raw, offset + 6) / 90.0,
                torch.sin(bearing),
                torch.cos(bearing),
                _u16(raw, offset + 7) / 4096.0,
                raw[..., offset + 9] / 255.0,
                raw[..., offset + 10] / 255.0,
                direct,
                raw[..., offset + 11] / 255.0,
                radar,
                audible,
                alive,
                known,
                peripheral,
            ),
            dim=-1,
        )
        source = torch.maximum(torch.maximum(direct, radar), torch.maximum(audible, peripheral))
        valid = (known > 0.5) & ((source > 0.5) | (relation > 0.5))
        tokens.append(torch.where(valid.unsqueeze(-1), _pad(features), torch.zeros((raw.shape[0], TOKEN_WIDTH), dtype=raw.dtype, device=raw.device)))
        masks.append(valid)
    return torch.stack(tokens, dim=1), torch.stack(masks, dim=1)


def _sound_tokens(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = []
    masks = []
    for index in range(SOUND_TOKEN_COUNT):
        offset = 152 + index * 3
        category_flags = raw[..., offset]
        packed = raw[..., offset + 2]
        category = torch.remainder(torch.floor(category_flags), 32.0) / 31.0
        bearing = _s8(raw, offset + 1) * 2.0 * math.pi / 180.0
        features = torch.stack(
            (
                category,
                torch.sin(bearing),
                torch.cos(bearing),
                torch.floor(packed / 16.0) / 15.0,
                torch.remainder(torch.floor(packed), 16.0) / 15.0,
                _bit(category_flags, 5),
                _bit(category_flags, 6),
                _bit(category_flags, 7),
            ),
            dim=-1,
        )
        valid = torch.remainder(torch.floor(category_flags), 32.0) > 0.0
        tokens.append(torch.where(valid.unsqueeze(-1), _pad(features), torch.zeros((raw.shape[0], TOKEN_WIDTH), dtype=raw.dtype, device=raw.device)))
        masks.append(valid)
    return torch.stack(tokens, dim=1), torch.stack(masks, dim=1)


def _event_tokens(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = raw[..., 200:216]
    categories = torch.remainder(torch.floor(values), 32.0) / 31.0
    ages = torch.floor(values / 32.0) / 7.0
    features = torch.stack((categories, ages, (categories > 0.0).to(raw.dtype)), dim=-1)
    return _pad(features), torch.remainder(torch.floor(values), 32.0) > 0.0


def _context_token(raw: torch.Tensor) -> torch.Tensor:
    return _pad(torch.cat((raw[..., :8] / 255.0, raw[..., 28:44] / 255.0), dim=-1))


def _geometry_token(raw: torch.Tensor) -> torch.Tensor:
    rays = raw[..., 216:256] / 63.0
    compressed = torch.cat((rays[..., :24], rays[..., 24:].reshape(raw.shape[0], 8, 2).mean(dim=-1)), dim=-1)
    return _pad(compressed)


def legal_entity_tokens(observation: torch.Tensor) -> LegalEntityTokensV1:
    """Decode normalized observation bytes into fixed legal entity tokens."""

    raw = _raw_bytes(observation)
    self_token = _self_token(raw).unsqueeze(1)
    player_tokens, player_mask = _player_tokens(raw)
    sound_tokens, sound_mask = _sound_tokens(raw)
    event_tokens, event_mask = _event_tokens(raw)
    context_token = _context_token(raw).unsqueeze(1)
    geometry_token = _geometry_token(raw).unsqueeze(1)
    features = torch.cat((self_token, player_tokens, sound_tokens, event_tokens, context_token, geometry_token), dim=1)
    valid_mask = torch.cat(
        (
            torch.ones((raw.shape[0], 1), dtype=torch.bool, device=raw.device),
            player_mask,
            sound_mask,
            event_mask,
            torch.ones((raw.shape[0], 2), dtype=torch.bool, device=raw.device),
        ),
        dim=1,
    )
    token_type = torch.tensor(
        [0] + [1] * PLAYER_TOKEN_COUNT + [2] * SOUND_TOKEN_COUNT + [3] * EVENT_TOKEN_COUNT + [4, 5],
        dtype=torch.int64,
        device=raw.device,
    )
    return LegalEntityTokensV1(features, valid_mask, token_type)
