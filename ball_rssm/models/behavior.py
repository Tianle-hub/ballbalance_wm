"""Dreamer actor, value model, and return utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Iterator

import torch
from torch import nn
from torch.distributions import Independent, Normal

from ball_rssm.models.networks import build_mlp, softplus_std

if TYPE_CHECKING:
    from ball_rssm.models.rssm import RSSMState
    from ball_rssm.models.world_model import WorldModel


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
    value_head_dist: str = "normal"

    def __post_init__(self) -> None:
        self.value_head_dist = normalize_value_dist(self.value_head_dist)

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

    def rsample_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.base_dist.rsample()
        action = self._scale(torch.tanh(raw))
        log_prob = self.base_dist.log_prob(raw.detach())
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

    is_discrete_action = False

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
        return self.mean(features)

    def mean(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        value = self.value_model(flat)
        return value.reshape(*features.shape[:-1], 1)

    def get_dist(self, features: torch.Tensor) -> Independent:
        return Independent(Normal(self.mean(features), 1.0), 1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        legacy_prefix = prefix + "net."
        current_prefix = prefix + "value_model."
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                state_dict[current_prefix + key[len(legacy_prefix) :]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Critic(DenseDecoder):
    """Compatibility name for the Dreamer scalar value model."""


@contextmanager
def freeze_parameters(*modules: nn.Module) -> Iterator[None]:
    params = [param for module in modules for param in module.parameters()]
    previous = [param.requires_grad for param in params]
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in zip(params, previous):
            param.requires_grad_(requires_grad)


def normalize_value_dist(name: object) -> str:
    normalized = str(name).lower().strip()
    if normalized in {"normal", "gaussian", "fixed_normal", "fixed-std-normal"}:
        return "normal"
    if normalized in {"mse", "deterministic"}:
        return "mse"
    raise ValueError("value_head_dist must be one of: normal, mse")


def value_loss(pred: torch.Tensor, target: torch.Tensor, dist_name: str) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    dist_name = normalize_value_dist(dist_name)
    if dist_name == "mse":
        return (pred - target) ** 2
    return -Independent(Normal(pred, 1.0), 1).log_prob(target).unsqueeze(-1)


def dreamer_actor_loss_from_start(
    world_model: "WorldModel",
    critic: Critic,
    imagine: Any,
    start: "RSSMState",
    discount: float,
    lambda_: float,
    actor_entropy_scale: float,
    actor_gradient: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Imagine from posterior starts and compute the full Dreamer actor loss."""

    with freeze_parameters(world_model, critic):
        # V1 uses Dreamer dynamics backpropagation through imagined model
        # rollouts. V2 can switch to a score-function policy gradient.
        states, _, entropy, log_prob = imagine(
            start,
            deterministic=False,
            actor_gradient=actor_gradient,
        )
        features = world_model.features_from_sequence(states)
        reward = world_model.predict_reward_sequence(states)
        continuation = world_model.predict_continuation_sequence(states)
        value = critic(features)
        return dreamer_actor_loss(
            reward=reward,
            continuation=continuation,
            value=value,
            entropy=entropy,
            log_prob=log_prob,
            discount=discount,
            lambda_=lambda_,
            actor_entropy_scale=actor_entropy_scale,
            actor_gradient=actor_gradient,
        )


def dreamer_critic_loss_from_start(
    world_model: "WorldModel",
    critic: Critic,
    target_critic: Critic,
    imagine: Any,
    start: "RSSMState",
    discount: float,
    lambda_: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Imagine from posterior starts and compute the full Dreamer critic loss."""

    with torch.no_grad():
        # The critic learns the same imagined TD(lambda) returns that drive the
        # actor, using a slowly updated target critic for the bootstrap.
        states, _, _, _ = imagine(start, deterministic=False)
        features = world_model.features_from_sequence(states)
        reward = world_model.predict_reward_sequence(states)
        continuation = world_model.predict_continuation_sequence(states)
        target_value = target_critic(features)
        returns, weights = dreamer_critic_targets(
            reward=reward,
            continuation=continuation,
            target_value=target_value,
            discount=discount,
            lambda_=lambda_,
        )
        features = features[:, :-1].detach()

    pred = critic(features)
    return dreamer_critic_loss(
        pred=pred,
        returns=returns,
        weights=weights,
        value_head_dist=critic.config.value_head_dist,
    )


def dreamer_actor_loss(
    reward: torch.Tensor,
    continuation: torch.Tensor,
    value: torch.Tensor,
    entropy: torch.Tensor,
    log_prob: torch.Tensor,
    discount: float,
    lambda_: float,
    actor_entropy_scale: float,
    actor_gradient: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Dreamer actor loss from imagined rewards, values, and action stats."""

    pcont = discount * continuation
    returns = compute_return(
        reward[:, :-1].transpose(0, 1),
        value[:, :-1].transpose(0, 1),
        pcont[:, :-1].transpose(0, 1),
        lambda_,
        value[:, -1],
    ).transpose(0, 1)
    weights = discount_weights(pcont[:, :-1]).detach()
    entropy_bonus = entropy[:, :-1].mean()
    if actor_gradient in {"reinforce", "both"}:
        advantage = (returns - value[:, :-1]).detach()
        reinforce_objective = (weights * log_prob[:, :-1] * advantage).mean()
    else:
        reinforce_objective = torch.zeros((), device=returns.device, dtype=returns.dtype)
    if actor_gradient in {"dynamics", "both"}:
        dynamics_objective = (weights * returns).mean()
    else:
        dynamics_objective = torch.zeros((), device=returns.device, dtype=returns.dtype)

    objective = reinforce_objective + dynamics_objective
    imagined_return_objective = (weights * returns.detach()).mean()
    loss = -objective - actor_entropy_scale * entropy_bonus
    metrics = {
        "actor_loss": loss.detach(),
        "actor_objective": objective.detach(),
        "actor_imagined_return_objective": imagined_return_objective.detach(),
        "actor_dynamics_objective": dynamics_objective.detach(),
        "actor_reinforce_objective": reinforce_objective.detach(),
        "actor_entropy": entropy_bonus.detach(),
        "actor_gradient_reinforce": torch.as_tensor(float(actor_gradient == "reinforce"), device=returns.device),
        "actor_gradient_both": torch.as_tensor(float(actor_gradient == "both"), device=returns.device),
        "imagined_reward_mean": reward.detach().mean(),
        "imagined_continue_mean": continuation.detach().mean(),
    }
    return loss, metrics


def dreamer_critic_targets(
    reward: torch.Tensor,
    continuation: torch.Tensor,
    target_value: torch.Tensor,
    discount: float,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute TD(lambda) critic targets and discount weights from imagined rollouts."""

    pcont = discount * continuation
    returns = compute_return(
        reward[:, :-1].transpose(0, 1),
        target_value[:, :-1].transpose(0, 1),
        pcont[:, :-1].transpose(0, 1),
        lambda_,
        target_value[:, -1],
    ).transpose(0, 1)
    weights = discount_weights(pcont[:, :-1]).detach()
    return returns.detach(), weights


def dreamer_critic_loss(
    pred: torch.Tensor,
    returns: torch.Tensor,
    weights: torch.Tensor,
    value_head_dist: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Dreamer critic loss against imagined TD(lambda) targets."""

    per_step_loss = value_loss(pred, returns, value_head_dist)
    value_mse = (pred - returns) ** 2
    loss = (weights * per_step_loss).mean()
    metrics = {
        "critic_loss": loss.detach(),
        "critic_mse": (weights * value_mse).mean().detach(),
        "critic_value_mean": pred.detach().mean(),
        "critic_target_mean": returns.detach().mean(),
    }
    return loss, metrics


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
