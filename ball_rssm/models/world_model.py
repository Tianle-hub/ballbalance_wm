"""Low-dimensional RSSM world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence

from ball_rssm.models.networks import build_mlp
from ball_rssm.models.rssm import RSSM, RSSMState


@dataclass
class WorldModelConfig:
    obs_dim: int = 6
    action_dim: int = 2
    deter_dim: int = 128
    stoch_dim: int = 16
    embed_dim: int = 64
    hidden_dim: int = 128
    min_std: float = 1e-4
    beta_kl: float = 1.0
    free_nats: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = build_mlp(config.obs_dim, config.hidden_dim, config.embed_dim)
        self.rssm = RSSM(
            action_dim=config.action_dim,
            embed_dim=config.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            hidden_dim=config.hidden_dim,
            min_std=config.min_std,
        )
        self.decoder = build_mlp(config.deter_dim + config.stoch_dim, config.hidden_dim, config.obs_dim)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> dict[str, object]:
        batch_size, obs_steps, obs_dim = obs_seq.shape
        embed = self.encoder(obs_seq.reshape(batch_size * obs_steps, obs_dim)).reshape(batch_size, obs_steps, -1)
        rssm_out = self.rssm.observe(embed, action_seq)

        posterior = rssm_out["posterior"]
        prior = rssm_out["prior"]
        assert isinstance(posterior, dict)
        assert isinstance(prior, dict)

        recon = self.decode_features(posterior["h"], posterior["z"])
        prior_recon = self.decode_features(prior["h"], prior["z"])
        return {
            **rssm_out,
            "recon": recon,
            "prior_recon": prior_recon,
        }

    def loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
        done_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del reward_seq, done_seq
        out = self.forward(obs_seq, action_seq)
        recon = out["recon"]
        posterior = out["posterior"]
        prior = out["prior"]
        assert isinstance(recon, torch.Tensor)
        assert isinstance(posterior, dict)
        assert isinstance(prior, dict)

        recon_loss = torch.mean((recon[:, 1:] - obs_seq[:, 1:]) ** 2)
        posterior_dist = independent_normal(posterior["mean"], posterior["std"])
        prior_dist = independent_normal(prior["mean"], prior["std"])
        kl = kl_divergence(posterior_dist, prior_dist)
        raw_kl = kl[:, 1:].mean()
        kl_loss = torch.clamp(kl[:, 1:], min=self.config.free_nats).mean()
        total_loss = recon_loss + self.config.beta_kl * kl_loss

        metrics = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "raw_kl": raw_kl.detach(),
            "posterior_std_mean": posterior["std"].detach().mean(),
            "prior_std_mean": prior["std"].detach().mean(),
            "recon0_loss": torch.mean((recon[:, :1] - obs_seq[:, :1]) ** 2).detach(),
        }
        return total_loss, metrics

    def reconstruct(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        out = self.forward(obs_seq, action_seq)
        recon = out["recon"]
        assert isinstance(recon, torch.Tensor)
        return recon

    def open_loop_predict(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        context_len: int,
        horizon: int,
    ) -> torch.Tensor:
        if context_len < 0:
            raise ValueError("context_len must be non-negative")
        horizon = min(horizon, action_seq.shape[1] - context_len)
        if horizon <= 0:
            raise ValueError("No future actions available for open-loop prediction")

        context_obs = obs_seq[:, : context_len + 1]
        context_action = action_seq[:, :context_len]
        context_out = self.forward(context_obs, context_action)
        start_state = context_out["last_state"]
        assert isinstance(start_state, RSSMState)
        future_actions = action_seq[:, context_len : context_len + horizon]
        imagined = self.rssm.imagine(start_state, future_actions)
        prior = imagined["prior"]
        assert isinstance(prior, dict)
        return self.decode_features(prior["h"], prior["z"])

    def encode_obs(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_norm)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        return self.rssm.init_state(batch_size, device)

    def posterior_update(
        self,
        prev_state: RSSMState,
        prev_action_norm: torch.Tensor,
        obs_norm: torch.Tensor,
    ) -> RSSMState:
        embed = self.encode_obs(obs_norm)
        posterior, _, _, _ = self.rssm.obs_step(prev_state, prev_action_norm, embed)
        return posterior

    def imagine_rollout(
        self,
        start_state: RSSMState,
        action_seq_norm: torch.Tensor,
        deterministic: bool = True,
    ) -> dict[str, object]:
        return self.rssm.imagine(start_state, action_seq_norm, deterministic=deterministic)

    def decode_state_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decode_features(states["h"], states["z"])

    def decode_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feature = torch.cat([h, z], dim=-1)
        flat = feature.reshape(-1, feature.shape[-1])
        decoded = self.decoder(flat)
        return decoded.reshape(*feature.shape[:-1], -1)


def independent_normal(mean: torch.Tensor, std: torch.Tensor) -> Independent:
    return Independent(Normal(mean, std), 1)
