"""Dreamer actor, value model, and return utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn
from torch.distributions import Independent, Normal

from ball_rssm.models.networks import build_mlp, softplus_std


@dataclass
class ActorConfig:
    feature_dim: int
    action_dim: int
    hidden_dim: int = 128
    min_std: float = 0.05
    max_std: float = 2.0
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ActorConfig":
        valid_names = {field.name for field in fields(cls)}
        data = {key: value for key, value in state.items() if key in valid_names}
        if data.get("action_low") is not None:
            data["action_low"] = tuple(float(x) for x in data["action_low"])
        if data.get("action_high") is not None:
            data["action_high"] = tuple(float(x) for x in data["action_high"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriticConfig:
    feature_dim: int
    hidden_dim: int = 128

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "CriticConfig":
        valid_names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in state.items() if key in valid_names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BoundedActionDistribution:
    """Tanh-squashed diagonal Gaussian scaled to normalized action bounds."""

    def __init__(self, base_dist: Independent, action_low: torch.Tensor, action_high: torch.Tensor) -> None:
        self.base_dist = base_dist
        self.action_low = action_low
        self.action_high = action_high

    def rsample(self) -> torch.Tensor:
        # Dreamer trains the actor through imagined rollouts, so actions need a
        # reparameterized sample. PlaNet's CEM actions were optimizer samples.
        raw = self.base_dist.rsample()
        return self._scale(torch.tanh(raw))

    def sample_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base_dist.sample()
        action = self._scale(torch.tanh(raw))
        log_prob = self.base_dist.log_prob(raw)
        return action, log_prob

    def mode(self) -> torch.Tensor:
        raw_mean = self.base_dist.base_dist.mean
        return self._scale(torch.tanh(raw_mean))

    def entropy(self) -> torch.Tensor:
        return self.base_dist.entropy()

    def _scale(self, squashed: torch.Tensor) -> torch.Tensor:
        low = self.action_low.to(device=squashed.device, dtype=squashed.dtype)
        high = self.action_high.to(device=squashed.device, dtype=squashed.dtype)
        while low.ndim < squashed.ndim:
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        return low + 0.5 * (squashed + 1.0) * (high - low)


class ActionDecoder(nn.Module):
    """Continuous Dreamer action decoder over normalized actions."""

    def __init__(self, config: ActorConfig) -> None:
        super().__init__()
        if config.action_low is None or config.action_high is None:
            action_low = torch.full((config.action_dim,), -1.0)
            action_high = torch.full((config.action_dim,), 1.0)
        else:
            action_low = torch.tensor(config.action_low, dtype=torch.float32)
            action_high = torch.tensor(config.action_high, dtype=torch.float32)
        if action_low.shape != (config.action_dim,) or action_high.shape != (config.action_dim,):
            raise ValueError("action bounds must match action_dim")
        if torch.any(action_high <= action_low):
            raise ValueError("action_high must be greater than action_low")

        self.config = config
        self.action_size = config.action_dim
        self.action_model = build_mlp(config.feature_dim, config.hidden_dim, 2 * config.action_dim)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)

    def get_dist(self, features: torch.Tensor) -> BoundedActionDistribution:
        # DreamerV1 adds this learned policy head on RSSM features; PlaNet used
        # the world model at control time with an external action optimizer.
        flat = features.reshape(-1, features.shape[-1])
        mean_raw, std_raw = torch.chunk(self.action_model(flat), 2, dim=-1)
        mean = torch.tanh(mean_raw)
        std = softplus_std(std_raw, self.config.min_std).clamp(max=self.config.max_std)
        mean = mean.reshape(*features.shape[:-1], -1)
        std = std.reshape(*features.shape[:-1], -1)
        dist = Independent(Normal(mean, std), 1)
        return BoundedActionDistribution(dist, self.action_low, self.action_high)

    def forward(self, features: torch.Tensor, deter: bool = False) -> torch.Tensor | BoundedActionDistribution:
        dist = self.get_dist(features)
        return dist.mode() if deter else dist

    def sample(self, features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.get_dist(features)
        return dist.mode() if deterministic else dist.rsample()

    def add_exploration(self, action: torch.Tensor, action_noise: float = 0.3) -> torch.Tensor:
        if action_noise <= 0.0:
            return action
        noisy = Normal(action, action_noise).rsample()
        low = self.action_low.to(device=noisy.device, dtype=noisy.dtype)
        high = self.action_high.to(device=noisy.device, dtype=noisy.dtype)
        while low.ndim < noisy.ndim:
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        return torch.max(torch.min(noisy, high), low)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        legacy_prefix = prefix + "net."
        current_prefix = prefix + "action_model."
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                state_dict[current_prefix + key[len(legacy_prefix) :]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Actor(ActionDecoder):
    """Compatibility name for the Dreamer action decoder."""


class DenseDecoder(nn.Module):
    """Dense scalar decoder used as the Dreamer value model."""

    def __init__(self, config: CriticConfig) -> None:
        super().__init__()
        self.config = config
        self.value_model = build_mlp(config.feature_dim, config.hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # DreamerV1 learns a value bootstrap for imagined TD(lambda) returns.
        # PlaNet did not need this critic because it planned finite horizons.
        flat = features.reshape(-1, features.shape[-1])
        value = self.value_model(flat)
        return value.reshape(*features.shape[:-1], 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        legacy_prefix = prefix + "net."
        current_prefix = prefix + "value_model."
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                state_dict[current_prefix + key[len(legacy_prefix) :]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Critic(DenseDecoder):
    """Compatibility name for the Dreamer scalar value model."""


def compute_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    td_lam: float,
    last_value: torch.Tensor,
) -> torch.Tensor:
    """Compute time-major Dreamer TD(lambda) returns with reverse recursion."""

    if rewards.shape != values.shape or rewards.shape != discounts.shape:
        raise ValueError("rewards, values, and discounts must share shape [time, batch, 1]")
    if last_value.shape != rewards[-1].shape:
        raise ValueError("last_value must have shape [batch, 1]")
    if not 0.0 <= td_lam <= 1.0:
        raise ValueError("td_lam must be in [0, 1]")

    next_values = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
    reward_with_bootstrap = rewards + discounts * next_values * (1.0 - td_lam)

    returns: list[torch.Tensor] = []
    next_return = last_value
    for t in range(rewards.shape[0] - 1, -1, -1):
        next_return = reward_with_bootstrap[t] + discounts[t] * td_lam * next_return
        returns.append(next_return)
    # TODO: check if I need if t = H condition, then next_return = rewards + discounts * values[t] 
    return torch.flip(torch.stack(returns), dims=[0])


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    bootstrap: torch.Tensor,
    pcont: torch.Tensor,
    lambda_: float,
    stop_gradient: bool = True,
) -> torch.Tensor:
    """Compute Dreamer TD(lambda) returns along time dimension 1."""

    if reward.shape != value.shape or reward.shape != pcont.shape:
        raise ValueError("reward, value, and pcont must share shape [batch, time, 1]")
    if bootstrap.shape != reward[:, 0].shape:
        raise ValueError("bootstrap must have shape [batch, 1]")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    returns = compute_return(
        reward.transpose(0, 1),
        value.transpose(0, 1),
        pcont.transpose(0, 1),
        lambda_,
        bootstrap,
    ).transpose(0, 1)
    return returns.detach() if stop_gradient else returns


def discount_weights(pcont: torch.Tensor) -> torch.Tensor:
    """Return cumulative continuation weights for imagined losses."""

    if pcont.ndim != 3 or pcont.shape[-1] != 1:
        raise ValueError("pcont must have shape [batch, time, 1]")
    ones = torch.ones_like(pcont[:, :1])
    if pcont.shape[1] == 1:
        return ones
    return torch.cumprod(torch.cat([ones, pcont[:, :-1]], dim=1), dim=1)
