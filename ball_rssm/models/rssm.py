"""Recurrent state-space models used by Dreamer V1 and V2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Independent, Normal, OneHotCategoricalStraightThrough

from ball_rssm.models.networks import build_mlp, softplus_std


@dataclass
class RSSMState:
    # h is deterministic memory; z is the sampled stochastic latent feature.
    h: torch.Tensor
    z: torch.Tensor
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    logits: torch.Tensor | None = None


class RSSM(nn.Module):
    """RSSM with deterministic GRU memory and continuous or discrete latents."""

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int = 128,
        stoch_dim: int = 16,
        hidden_dim: int = 128,
        min_std: float = 1e-4,
        discrete: bool = False,
        discrete_classes: int = 32,
    ) -> None:
        super().__init__()
        if discrete_classes < 2:
            raise ValueError("discrete_classes must be at least 2")
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std
        self.discrete = discrete
        self.discrete_classes = discrete_classes
        self.stoch_feature_dim = stoch_dim * discrete_classes if discrete else stoch_dim
        self._dist_output_dim = stoch_dim * discrete_classes if discrete else 2 * stoch_dim

        # The prior predicts the next stochastic state from recurrent memory alone.
        # The posterior corrects that prior with the encoded observation at the same step.
        self.gru = nn.GRUCell(self.stoch_feature_dim + action_dim, deter_dim)
        self.prior_net = build_mlp(deter_dim, hidden_dim, self._dist_output_dim)
        self.posterior_net = build_mlp(deter_dim + embed_dim, hidden_dim, self._dist_output_dim)

    def init_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        """Create the zero initial latent belief for a batch."""

        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_feature_dim, device=device)
        if self.discrete:
            logits = torch.zeros(batch_size, self.stoch_dim, self.discrete_classes, device=device)
            return RSSMState(h=h, z=z, logits=logits)

        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones(batch_size, self.stoch_dim, device=device)
        return RSSMState(h=h, z=z, mean=mean, std=std)

    def img_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[RSSMState, Independent]:
        """Predict the next prior state from previous latent state and action."""

        # Imagination step: advance latent dynamics with no observation correction.
        x = torch.cat([prev_state.z, action], dim=-1)
        h = self.gru(x, prev_state.h)
        state, dist = self._state_from_params(h, self.prior_net(h), deterministic=deterministic)
        return state, dist

    def obs_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState, Independent, Independent]:
        """Update one timestep with an observation embedding.

        Returns posterior state, prior state, prior distribution, and posterior
        distribution for KL training.
        """

        # Observation step: first build the action-conditioned prior, then infer z_t
        # from the prior memory and current observation embedding.
        prior_state, prior_dist = self.img_step(prev_state, action)
        posterior_state, posterior_dist = self._state_from_params(
            prior_state.h,
            self.posterior_net(torch.cat([prior_state.h, embed], dim=-1)),
            deterministic=False,
        )
        return posterior_state, prior_state, prior_dist, posterior_dist

    def observe(self, embed_seq: torch.Tensor, action_seq: torch.Tensor) -> dict[str, object]:
        """Infer posterior/prior latent sequences for a full observation window."""

        batch_size, obs_steps, _ = embed_seq.shape
        device = embed_seq.device
        prev = self.init_state(batch_size, device)
        zero_action = torch.zeros(batch_size, self.action_dim, device=device, dtype=embed_seq.dtype)

        post_states: list[RSSMState] = []
        prior_states: list[RSSMState] = []
        prior_dists: list[Independent] = []
        posterior_dists: list[Independent] = []
        for t in range(obs_steps):
            # obs_seq has T+1 entries while action_seq has T. At t=0 there is
            # no previous action, so a zero action anchors the initial posterior.
            action = zero_action if t == 0 else action_seq[:, t - 1]
            posterior, prior, prior_dist, posterior_dist = self.obs_step(prev, action, embed_seq[:, t])
            post_states.append(posterior)
            prior_states.append(prior)
            prior_dists.append(prior_dist)
            posterior_dists.append(posterior_dist)
            prev = posterior

        return {
            "posterior": stack_states(post_states),
            "prior": stack_states(prior_states),
            "prior_dists": prior_dists,
            "posterior_dists": posterior_dists,
            "last_state": post_states[-1],
        }

    def imagine(
        self,
        start_state: RSSMState,
        action_seq: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, object]:
        """Roll the prior forward from a start state under future actions."""

        prev = start_state
        states: list[RSSMState] = []
        dists: list[Independent] = []
        for t in range(action_seq.shape[1]):
            # Planning/evaluation uses only the prior after the context state.
            prev, dist = self.img_step(prev, action_seq[:, t], deterministic=deterministic)
            states.append(prev)
            dists.append(dist)
        return {"prior": stack_states(states), "prior_dists": dists, "last_state": prev}

    def get_dist(self, state: RSSMState | dict[str, torch.Tensor]) -> Independent:
        """Return the latent distribution represented by an RSSM state."""

        if isinstance(state, RSSMState):
            if state.logits is not None:
                return self._discrete_dist(state.logits)
            if state.mean is None or state.std is None:
                raise ValueError("continuous RSSM state requires mean and std")
            return self._normal_dist(state.mean, state.std)

        logits = state.get("logits")
        if logits is not None:
            return self._discrete_dist(logits)
        if "mean" not in state or "std" not in state:
            raise ValueError("continuous RSSM state dictionary requires mean and std")
        return self._normal_dist(state["mean"], state["std"])

    def detach_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Detach all RSSM state tensors, preserving the state dictionary shape."""

        return {key: value.detach() for key, value in state.items()}

    def _state_from_params(
        self,
        h: torch.Tensor,
        params: torch.Tensor,
        deterministic: bool,
    ) -> tuple[RSSMState, Independent]:
        if self.discrete:
            logits = params.reshape(*params.shape[:-1], self.stoch_dim, self.discrete_classes)
            dist = self._discrete_dist(logits)
            if deterministic:
                index = logits.argmax(dim=-1)
                z_unflat = F.one_hot(index, self.discrete_classes).to(dtype=params.dtype)
            else:
                z_unflat = dist.rsample()
            z = z_unflat.reshape(*params.shape[:-1], self.stoch_feature_dim)
            return RSSMState(h=h, z=z, logits=logits), dist

        mean, std = self._stats(params)
        dist = self._normal_dist(mean, std)
        z = mean if deterministic else dist.rsample()
        return RSSMState(h=h, z=z, mean=mean, std=std), dist

    def _stats(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(params, 2, dim=-1)
        return mean, softplus_std(raw_std, self.min_std)

    @staticmethod
    def _normal_dist(mean: torch.Tensor, std: torch.Tensor) -> Independent:
        return Independent(Normal(mean, std), 1)

    @staticmethod
    def _discrete_dist(logits: torch.Tensor) -> Independent:
        return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)


def repeat_state(state: RSSMState, repeats: int) -> RSSMState:
    """Tile one RSSM state across a candidate batch."""

    def repeat_optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.repeat(repeats, *([1] * (value.ndim - 1)))

    return RSSMState(
        h=state.h.repeat(repeats, 1),
        z=state.z.repeat(repeats, 1),
        mean=repeat_optional(state.mean),
        std=repeat_optional(state.std),
        logits=repeat_optional(state.logits),
    )


def stack_states(states: list[RSSMState]) -> dict[str, torch.Tensor]:
    """Stack per-step RSSMState objects into time-major dictionaries."""

    # Convert a Python list of per-step states into [batch, time, dim] tensors.
    stacked = {
        "h": torch.stack([state.h for state in states], dim=1),
        "z": torch.stack([state.z for state in states], dim=1),
    }
    for name in ("mean", "std", "logits"):
        values = [getattr(state, name) for state in states]
        if values[0] is None:
            continue
        if any(value is None for value in values):
            raise ValueError(f"mixed RSSM state field {name!r} cannot be stacked")
        stacked[name] = torch.stack(values, dim=1)
    return stacked
