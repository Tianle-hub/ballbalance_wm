"""Gaussian recurrent state-space model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Independent, Normal

from ball_rssm.models.networks import build_mlp, softplus_std


@dataclass
class RSSMState:
    # h is deterministic memory; z is the sampled stochastic latent and its Gaussian stats.
    h: torch.Tensor
    z: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int = 128,
        stoch_dim: int = 16,
        hidden_dim: int = 128,
        min_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        # The prior predicts the next stochastic state from recurrent memory alone.
        # The posterior corrects that prior with the encoded observation at the same step.
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior_net = build_mlp(deter_dim, hidden_dim, 2 * stoch_dim)
        self.posterior_net = build_mlp(deter_dim + embed_dim, hidden_dim, 2 * stoch_dim)

    def init_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones(batch_size, self.stoch_dim, device=device)
        return RSSMState(h=h, z=z, mean=mean, std=std)

    def img_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[RSSMState, Independent]:
        # Imagination step: advance latent dynamics with no observation correction.
        x = torch.cat([prev_state.z, action], dim=-1)
        h = self.gru(x, prev_state.h)
        mean, std = self._stats(self.prior_net(h))
        dist = self._dist(mean, std)
        z = mean if deterministic else dist.rsample()
        return RSSMState(h=h, z=z, mean=mean, std=std), dist

    def obs_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState, Independent, Independent]:
        # Observation step: first build the action-conditioned prior, then infer z_t
        # from the prior memory and current observation embedding.
        prior_state, prior_dist = self.img_step(prev_state, action)
        mean, std = self._stats(self.posterior_net(torch.cat([prior_state.h, embed], dim=-1)))
        posterior_dist = self._dist(mean, std)
        z = posterior_dist.rsample()
        posterior_state = RSSMState(h=prior_state.h, z=z, mean=mean, std=std)
        return posterior_state, prior_state, prior_dist, posterior_dist

    def observe(self, embed_seq: torch.Tensor, action_seq: torch.Tensor) -> dict[str, object]:
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
        prev = start_state
        states: list[RSSMState] = []
        dists: list[Independent] = []
        for t in range(action_seq.shape[1]):
            # Planning/evaluation uses only the prior after the context state.
            prev, dist = self.img_step(prev, action_seq[:, t], deterministic=deterministic)
            states.append(prev)
            dists.append(dist)
        return {"prior": stack_states(states), "prior_dists": dists, "last_state": prev}

    def _stats(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(params, 2, dim=-1)
        return mean, softplus_std(raw_std, self.min_std)

    @staticmethod
    def _dist(mean: torch.Tensor, std: torch.Tensor) -> Independent:
        return Independent(Normal(mean, std), 1)


def repeat_state(state: RSSMState, repeats: int) -> RSSMState:
    # CEM evaluates many action candidates from the same current belief state.
    return RSSMState(
        h=state.h.repeat(repeats, 1),
        z=state.z.repeat(repeats, 1),
        mean=state.mean.repeat(repeats, 1),
        std=state.std.repeat(repeats, 1),
    )


def stack_states(states: list[RSSMState]) -> dict[str, torch.Tensor]:
    # Convert a Python list of per-step states into [batch, time, dim] tensors.
    return {
        "h": torch.stack([state.h for state in states], dim=1),
        "z": torch.stack([state.z for state in states], dim=1),
        "mean": torch.stack([state.mean for state in states], dim=1),
        "std": torch.stack([state.std for state in states], dim=1),
    }
