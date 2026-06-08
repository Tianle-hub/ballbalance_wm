"""Low-dimensional RSSM world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence

from ball_rssm.models.networks import build_mlp
from ball_rssm.models.rssm import RSSM, RSSMState


@dataclass
class WorldModelConfig:
    obs_dim: int = 6
    action_dim: int = 2
    reward_dim: int = 1
    deter_dim: int = 128
    stoch_dim: int = 16
    embed_dim: int = 64
    hidden_dim: int = 128
    min_std: float = 1e-4
    dreamer_version: str = "v1"
    discrete_classes: int = 32
    beta_kl: float = 1.0
    free_nats: float = 1.0
    kl_alpha: float = 0.8
    reward_loss_weight: float = 1.0
    continuation_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        self.dreamer_version = normalize_dreamer_version(self.dreamer_version)
        if self.discrete_classes < 2:
            raise ValueError("discrete_classes must be at least 2")
        if not 0.0 <= self.kl_alpha <= 1.0:
            raise ValueError("kl_alpha must be in [0, 1]")

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "WorldModelConfig":
        # Checkpoints may contain obsolete config keys from earlier reward modes.
        valid_names = {field.name for field in fields(cls)}
        data = {key: value for key, value in state.items() if key in valid_names}
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def is_v2(self) -> bool:
        return self.dreamer_version == "v2"

    @property
    def stoch_feature_dim(self) -> int:
        return self.stoch_dim * self.discrete_classes if self.is_v2 else self.stoch_dim

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_feature_dim


class WorldModel(nn.Module):
    """RSSM world model with observation, reward, and continuation heads."""

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
            discrete=config.is_v2,
            discrete_classes=config.discrete_classes,
        )
        feature_dim = config.feature_dim
        # Heads consume the RSSM feature [h_t, z_t]. Reward stays scalar; episode
        # termination is modeled separately through continuation probability.
        # Dreamer actor/value training uses reward and continuation inside
        # imagined rollouts instead of handing a frozen model to a CEM planner.
        self.decoder = build_mlp(feature_dim, config.hidden_dim, config.obs_dim)
        self.reward_model = build_mlp(feature_dim, config.hidden_dim, config.reward_dim)
        self.continuation_model = build_mlp(feature_dim, config.hidden_dim, 1)

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> dict[str, object]:
        """Encode observations, run RSSM inference, and decode all prediction heads."""

        batch_size, obs_steps, obs_dim = obs_seq.shape
        embed = self.encoder(obs_seq.reshape(batch_size * obs_steps, obs_dim)).reshape(batch_size, obs_steps, -1)
        rssm_out = self.rssm.observe(embed, action_seq)

        posterior = rssm_out["posterior"]
        prior = rssm_out["prior"]
        assert isinstance(posterior, dict)
        assert isinstance(prior, dict)

        # Posterior heads are used for supervised training; prior heads expose the
        # same predictions from imagined states for diagnostics and open-loop use.
        recon = self.decode_features(posterior["h"], posterior["z"])
        prior_recon = self.decode_features(prior["h"], prior["z"])
        reward_pred, continuation_logit = self.reward_outputs_from_features(posterior["h"], posterior["z"])
        prior_reward_pred, prior_continuation_logit = self.reward_outputs_from_features(prior["h"], prior["z"])
        return {
            **rssm_out,
            "recon": recon,
            "prior_recon": prior_recon,
            "reward_pred": reward_pred,
            "prior_reward_pred": prior_reward_pred,
            "continuation_logit": continuation_logit,
            "prior_continuation_logit": prior_continuation_logit,
        }

    def loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
        done_seq: torch.Tensor | None = None,
        terminated_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute RSSM training loss and detached logging metrics.

        The action/reward/done tensors have T steps, while obs_seq has T+1
        observations. Reward and continuation targets are aligned to posterior
        latents at indices 1..T.
        """

        out = self.forward(obs_seq, action_seq)
        recon = out["recon"]
        reward_pred = out["reward_pred"]
        continuation_logit = out["continuation_logit"]
        posterior = out["posterior"]
        prior = out["prior"]
        assert isinstance(recon, torch.Tensor)
        assert isinstance(reward_pred, torch.Tensor)
        assert isinstance(continuation_logit, torch.Tensor)
        assert isinstance(posterior, dict)
        assert isinstance(prior, dict)

        # Masks keep the first terminal transition but remove artificial padding
        # after an episode has ended.
        effective_mask, nonterminal_mask, terminal_mask, post_done_mask = transition_masks(
            action_seq=action_seq,
            done_seq=done_seq,
            terminated_seq=terminated_seq,
        )

        recon_err = torch.mean((recon[:, 1:] - obs_seq[:, 1:]) ** 2, dim=-1)
        recon_loss = masked_mean(recon_err, effective_mask)
        reward_loss = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
        continuation_loss = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
        reward_metrics: dict[str, torch.Tensor] = {}
        if reward_seq is not None:
            if reward_seq.shape[:2] != action_seq.shape[:2]:
                raise ValueError("reward_seq must have shape [batch, action_steps, reward_dim]")
            # reward_seq[:, t] is produced by action_seq[:, t] and obs_seq[:, t+1],
            # so it is predicted from the posterior latent at index t+1.
            reward_err = torch.mean((reward_pred[:, 1:] - reward_seq) ** 2, dim=-1)
            reward_loss = masked_mean(reward_err, effective_mask)

            reward_metrics = {
                "reward_loss_nonterminal": masked_mean(reward_err, nonterminal_mask),
                "reward_loss_terminal": masked_mean(reward_err, terminal_mask),
                "reward_loss_post_done_padding": masked_mean(reward_err, post_done_mask),
                "reward_effective_fraction": effective_mask.float().mean(),
                "reward_terminal_fraction": terminal_mask.float().mean(),
                "reward_post_done_padding_fraction": post_done_mask.float().mean(),
            }

        # Continuation is 1 for non-terminal effective transitions and 0 at true
        # terminations. Time-limit truncation is not treated as a failure if a
        # separate terminated flag is available.
        continuation_target = (~terminal_mask).unsqueeze(-1).to(dtype=obs_seq.dtype)
        continuation_bce = nn.functional.binary_cross_entropy_with_logits(
            continuation_logit[:, 1:],
            continuation_target,
            reduction="none",
        ).squeeze(-1)
        continuation_loss = masked_mean(continuation_bce, effective_mask)
        continuation_prob = torch.sigmoid(continuation_logit[:, 1:]).squeeze(-1)
        reward_metrics.update(
            {
                "continuation_prob_nonterminal": masked_mean(continuation_prob, nonterminal_mask),
                "continuation_prob_terminal": masked_mean(continuation_prob, terminal_mask),
            }
        )
        posterior_dist = self.rssm.get_dist(posterior)
        prior_dist = self.rssm.get_dist(prior)
        kl = kl_divergence(posterior_dist, prior_dist)
        raw_kl = masked_mean(kl[:, 1:], effective_mask)
        if self.config.is_v2:
            posterior_detached = self.rssm.detach_state(posterior)
            prior_detached = self.rssm.detach_state(prior)
            dynamics_kl = kl_divergence(self.rssm.get_dist(posterior_detached), prior_dist)
            representation_kl = kl_divergence(posterior_dist, self.rssm.get_dist(prior_detached))
            dynamics_kl_loss = masked_mean(dynamics_kl[:, 1:], effective_mask)
            representation_kl_loss = masked_mean(representation_kl[:, 1:], effective_mask)
            kl_loss = (
                self.config.kl_alpha * dynamics_kl_loss
                + (1.0 - self.config.kl_alpha) * representation_kl_loss
            )
        else:
            # Dreamer V1 applies free nats to the mean KL so small posterior-prior
            # mismatches do not dominate the reconstruction objective.
            dynamics_kl_loss = raw_kl
            representation_kl_loss = raw_kl
            kl_loss = torch.maximum(raw_kl, raw_kl.new_tensor(self.config.free_nats))
        total_loss = (
            recon_loss
            + self.config.beta_kl * kl_loss
            + self.config.reward_loss_weight * reward_loss
            + self.config.continuation_loss_weight * continuation_loss
        )

        metrics = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "continuation_loss": continuation_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "raw_kl": raw_kl.detach(),
            "dynamics_kl_loss": dynamics_kl_loss.detach(),
            "representation_kl_loss": representation_kl_loss.detach(),
            "recon0_loss": torch.mean((recon[:, :1] - obs_seq[:, :1]) ** 2).detach(),
        }
        if "std" in posterior and "std" in prior:
            metrics["posterior_std_mean"] = posterior["std"].detach().mean()
            metrics["prior_std_mean"] = prior["std"].detach().mean()
        if "logits" in posterior and "logits" in prior:
            metrics["posterior_entropy_mean"] = posterior_dist.entropy().detach().mean()
            metrics["prior_entropy_mean"] = prior_dist.entropy().detach().mean()
        metrics.update({key: value.detach() for key, value in reward_metrics.items()})
        return total_loss, metrics

    def reconstruct(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Return posterior reconstruction for an observed sequence."""

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
        """Predict future observations using posterior context then prior rollout."""

        if context_len < 0:
            raise ValueError("context_len must be non-negative")
        horizon = min(horizon, action_seq.shape[1] - context_len)
        if horizon <= 0:
            raise ValueError("No future actions available for open-loop prediction")

        # Warm up with posterior inference, then roll only the learned prior.
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

    def open_loop_predict_rewards(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        context_len: int,
        horizon: int,
    ) -> torch.Tensor:
        """Predict future rewards over an open-loop prior rollout."""

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
        return self.predict_reward_from_features(prior["h"], prior["z"])

    def open_loop_predict_continuation(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        context_len: int,
        horizon: int,
    ) -> torch.Tensor:
        """Predict future continuation probabilities over an open-loop prior rollout."""

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
        return self.predict_continuation_sequence(prior)

    def encode_obs(self, obs_norm: torch.Tensor) -> torch.Tensor:
        """Embed normalized low-dimensional observations."""

        return self.encoder(obs_norm)

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        """Create an initial RSSM state through the contained dynamics model."""

        return self.rssm.init_state(batch_size, device)

    def posterior_update(
        self,
        prev_state: RSSMState,
        prev_action_norm: torch.Tensor,
        obs_norm: torch.Tensor,
    ) -> RSSMState:
        """Assimilate one real observation into the current RSSM belief."""

        embed = self.encode_obs(obs_norm)
        posterior, _, _, _ = self.rssm.obs_step(prev_state, prev_action_norm, embed)
        return posterior

    def imagine_rollout(
        self,
        start_state: RSSMState,
        action_seq_norm: torch.Tensor,
        deterministic: bool = True,
    ) -> dict[str, object]:
        """Imagine a latent rollout from a current belief and normalized actions."""

        return self.rssm.imagine(start_state, action_seq_norm, deterministic=deterministic)

    def decode_state_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode RSSM state features into normalized observations."""

        return self.decode_features(states["h"], states["z"])

    def predict_reward_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict normalized rewards from RSSM state features."""

        return self.predict_reward_from_features(states["h"], states["z"])

    def predict_continuation_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict continuation probabilities from RSSM state features."""

        return self.predict_continuation_from_features(states["h"], states["z"])

    def decode_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode concatenated [h, z] features, preserving leading dimensions."""

        feature = self.features_from_tensors(h, z)
        flat = feature.reshape(-1, feature.shape[-1])
        decoded = self.decoder(flat)
        return decoded.reshape(*feature.shape[:-1], -1)

    def features_from_tensors(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate deterministic and stochastic RSSM tensors."""

        return torch.cat([h, z], dim=-1)

    def features_from_state(self, state: RSSMState) -> torch.Tensor:
        """Return Dreamer features `[h, z]` for one RSSM state."""

        return self.features_from_tensors(state.h, state.z)

    def features_from_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return Dreamer features for a stacked RSSM state sequence."""

        return self.features_from_tensors(states["h"], states["z"])

    def predict_reward_from_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict reward from deterministic and stochastic latent features."""

        reward, _ = self.reward_outputs_from_features(h, z)
        return reward

    def predict_continuation_from_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict probability that imagined rollout continues after each state."""

        feature = torch.cat([h, z], dim=-1)
        flat = feature.reshape(-1, feature.shape[-1])
        continuation_logit = self.continuation_model(flat).reshape(*feature.shape[:-1], -1)
        return torch.sigmoid(continuation_logit)

    def reward_outputs_from_features(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run reward and continuation heads on concatenated RSSM features."""

        feature = torch.cat([h, z], dim=-1)
        flat = feature.reshape(-1, feature.shape[-1])
        reward = self.reward_model(flat)
        reward = reward.reshape(*feature.shape[:-1], -1)
        continuation = self.continuation_model(flat)
        continuation = continuation.reshape(*feature.shape[:-1], -1)
        return reward, continuation


def independent_normal(mean: torch.Tensor, std: torch.Tensor) -> Independent:
    return Independent(Normal(mean, std), 1)


def normalize_dreamer_version(version: object) -> str:
    normalized = str(version).lower().replace("dreamer", "").strip()
    if normalized in {"1", "v1"}:
        return "v1"
    if normalized in {"2", "v2"}:
        return "v2"
    raise ValueError("dreamer_version must be one of: v1, v2")


def transition_masks(
    action_seq: torch.Tensor,
    done_seq: torch.Tensor | None,
    terminated_seq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build masks for effective, nonterminal, terminal, and padded transitions."""

    batch_size, action_steps = action_seq.shape[:2]
    device = action_seq.device
    if done_seq is None:
        effective = torch.ones(batch_size, action_steps, device=device, dtype=torch.bool)
        terminal = torch.zeros_like(effective)
        return effective, effective, terminal, ~effective

    done = done_seq.squeeze(-1).bool()
    # A transition is effective if the previous transition did not already end
    # the episode. This leaves the terminal transition itself trainable.
    effective = torch.ones_like(done, dtype=torch.bool)
    effective[:, 1:] = ~done[:, :-1]
    post_done = ~effective
    if terminated_seq is not None:
        terminal = effective & terminated_seq.squeeze(-1).bool()
    else:
        terminal = effective & done
    nonterminal = effective & ~terminal
    return effective, nonterminal, terminal, post_done


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked entries, returning zero when the mask is empty."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    denom = mask.to(dtype=values.dtype).sum()
    if denom <= 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return values.masked_select(mask).mean()
