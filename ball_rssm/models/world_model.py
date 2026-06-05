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
    reward_dim: int = 1
    deter_dim: int = 128
    stoch_dim: int = 16
    embed_dim: int = 64
    hidden_dim: int = 128
    min_std: float = 1e-4
    beta_kl: float = 1.0
    free_nats: float = 1.0
    reward_loss_weight: float = 1.0
    reward_prediction_mode: str = "continuation"
    reward_clip_min: float = -1.0
    fall_reward_threshold: float = -0.99
    fall_penalty_value: float = -30.0
    fall_loss_weight: float = 1.0
    fall_prediction_loss_weight: float = 1.0
    continuation_loss_weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class WorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        if config.reward_prediction_mode not in {"raw", "clip", "split_fall", "continuation"}:
            raise ValueError("reward_prediction_mode must be one of: raw, clip, split_fall, continuation")
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
        feature_dim = config.deter_dim + config.stoch_dim
        self.decoder = build_mlp(feature_dim, config.hidden_dim, config.obs_dim)
        self.reward_model = build_mlp(feature_dim, config.hidden_dim, config.reward_dim)
        if config.reward_prediction_mode == "continuation":
            self.continuation_model = build_mlp(feature_dim, config.hidden_dim, 1)
        if config.reward_prediction_mode == "split_fall":
            self.fall_model = build_mlp(feature_dim, config.hidden_dim, 1)

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
        reward_cont_pred, continuation_logit, fall_logit = self.reward_outputs_from_features(posterior["h"], posterior["z"])
        prior_reward_cont_pred, prior_continuation_logit, prior_fall_logit = self.reward_outputs_from_features(
            prior["h"], prior["z"]
        )
        reward_pred = self.combine_reward_outputs(reward_cont_pred, fall_logit)
        prior_reward_pred = self.combine_reward_outputs(prior_reward_cont_pred, prior_fall_logit)
        return {
            **rssm_out,
            "recon": recon,
            "prior_recon": prior_recon,
            "reward_pred": reward_pred,
            "prior_reward_pred": prior_reward_pred,
            "reward_cont_pred": reward_cont_pred,
            "prior_reward_cont_pred": prior_reward_cont_pred,
            "continuation_logit": continuation_logit,
            "prior_continuation_logit": prior_continuation_logit,
            "fall_logit": fall_logit,
            "prior_fall_logit": prior_fall_logit,
        }

    def loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
        done_seq: torch.Tensor | None = None,
        terminated_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self.forward(obs_seq, action_seq)
        recon = out["recon"]
        reward_pred = out["reward_pred"]
        reward_cont_pred = out["reward_cont_pred"]
        continuation_logit = out["continuation_logit"]
        fall_logit = out["fall_logit"]
        posterior = out["posterior"]
        prior = out["prior"]
        assert isinstance(recon, torch.Tensor)
        assert isinstance(reward_pred, torch.Tensor)
        assert isinstance(reward_cont_pred, torch.Tensor)
        assert continuation_logit is None or isinstance(continuation_logit, torch.Tensor)
        assert fall_logit is None or isinstance(fall_logit, torch.Tensor)
        assert isinstance(posterior, dict)
        assert isinstance(prior, dict)

        effective_mask, nonterminal_mask, fall_terminal_mask, post_done_mask = transition_masks(
            action_seq=action_seq,
            reward_seq=reward_seq,
            done_seq=done_seq,
            terminated_seq=terminated_seq,
            fall_reward_threshold=self.config.fall_reward_threshold,
        )

        recon_err = torch.mean((recon[:, 1:] - obs_seq[:, 1:]) ** 2, dim=-1)
        recon_loss = masked_mean(recon_err, effective_mask)
        reward_loss = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
        fall_prediction_loss = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
        continuation_loss = torch.zeros((), device=obs_seq.device, dtype=obs_seq.dtype)
        reward_metrics: dict[str, torch.Tensor] = {}
        if reward_seq is not None:
            if reward_seq.shape[:2] != action_seq.shape[:2]:
                raise ValueError("reward_seq must have shape [batch, action_steps, reward_dim]")
            reward_target = reward_seq
            if self.config.reward_prediction_mode in {"clip", "split_fall"}:
                reward_target = torch.clamp(reward_target, min=self.config.reward_clip_min)
            reward_err = torch.mean((reward_cont_pred[:, 1:] - reward_target) ** 2, dim=-1)
            reward_weights = torch.ones_like(reward_err)
            reward_weights = torch.where(
                fall_terminal_mask,
                torch.full_like(reward_weights, self.config.fall_loss_weight),
                reward_weights,
            )
            reward_loss = masked_weighted_mean(reward_err, effective_mask, reward_weights)

            raw_reward_err = torch.mean((reward_pred[:, 1:] - reward_seq) ** 2, dim=-1)
            reward_metrics = {
                "reward_loss_nonterminal": masked_mean(raw_reward_err, nonterminal_mask),
                "reward_loss_fall_terminal": masked_mean(raw_reward_err, fall_terminal_mask),
                "reward_loss_post_done_padding": masked_mean(raw_reward_err, post_done_mask),
                "reward_effective_fraction": effective_mask.float().mean(),
                "reward_fall_terminal_fraction": fall_terminal_mask.float().mean(),
                "reward_post_done_padding_fraction": post_done_mask.float().mean(),
            }

            if self.config.reward_prediction_mode == "split_fall":
                if fall_logit is None:
                    raise RuntimeError("split_fall mode requires fall_model")
                fall_target = fall_terminal_mask.unsqueeze(-1).to(dtype=obs_seq.dtype)
                fall_bce = nn.functional.binary_cross_entropy_with_logits(
                    fall_logit[:, 1:],
                    fall_target,
                    reduction="none",
                ).squeeze(-1)
                fall_weights = torch.where(
                    fall_terminal_mask,
                    torch.full_like(fall_bce, self.config.fall_loss_weight),
                    torch.ones_like(fall_bce),
                )
                fall_prediction_loss = masked_weighted_mean(fall_bce, effective_mask, fall_weights)
                fall_prob = torch.sigmoid(fall_logit[:, 1:]).squeeze(-1)
                reward_metrics.update(
                    {
                        "fall_prediction_loss": fall_prediction_loss.detach(),
                        "fall_prob_nonterminal": masked_mean(fall_prob, nonterminal_mask),
                        "fall_prob_fall_terminal": masked_mean(fall_prob, fall_terminal_mask),
                    }
                )
            if self.config.reward_prediction_mode == "continuation":
                if continuation_logit is None:
                    raise RuntimeError("continuation mode requires continuation_model")
                continuation_target = (~fall_terminal_mask).unsqueeze(-1).to(dtype=obs_seq.dtype)
                continuation_bce = nn.functional.binary_cross_entropy_with_logits(
                    continuation_logit[:, 1:],
                    continuation_target,
                    reduction="none",
                ).squeeze(-1)
                continuation_weights = torch.where(
                    fall_terminal_mask,
                    torch.full_like(continuation_bce, self.config.fall_loss_weight),
                    torch.ones_like(continuation_bce),
                )
                continuation_loss = masked_weighted_mean(continuation_bce, effective_mask, continuation_weights)
                continuation_prob = torch.sigmoid(continuation_logit[:, 1:]).squeeze(-1)
                reward_metrics.update(
                    {
                        "continuation_loss": continuation_loss.detach(),
                        "continuation_prob_nonterminal": masked_mean(continuation_prob, nonterminal_mask),
                        "continuation_prob_terminal": masked_mean(continuation_prob, fall_terminal_mask),
                    }
                )
        posterior_dist = independent_normal(posterior["mean"], posterior["std"])
        prior_dist = independent_normal(prior["mean"], prior["std"])
        kl = kl_divergence(posterior_dist, prior_dist)
        raw_kl = masked_mean(kl[:, 1:], effective_mask)
        kl_loss = masked_mean(torch.clamp(kl[:, 1:], min=self.config.free_nats), effective_mask)
        total_loss = (
            recon_loss
            + self.config.beta_kl * kl_loss
            + self.config.reward_loss_weight * reward_loss
            + self.config.fall_prediction_loss_weight * fall_prediction_loss
            + self.config.continuation_loss_weight * continuation_loss
        )

        metrics = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "continuation_loss": continuation_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "raw_kl": raw_kl.detach(),
            "posterior_std_mean": posterior["std"].detach().mean(),
            "prior_std_mean": prior["std"].detach().mean(),
            "recon0_loss": torch.mean((recon[:, :1] - obs_seq[:, :1]) ** 2).detach(),
        }
        metrics.update({key: value.detach() for key, value in reward_metrics.items()})
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

    def open_loop_predict_rewards(
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
        return self.predict_reward_from_features(prior["h"], prior["z"])

    def open_loop_predict_continuation(
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
        return self.predict_continuation_sequence(prior)

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

    def predict_reward_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.predict_reward_from_features(states["h"], states["z"])

    def predict_continuation_sequence(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.predict_continuation_from_features(states["h"], states["z"])

    def decode_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feature = torch.cat([h, z], dim=-1)
        flat = feature.reshape(-1, feature.shape[-1])
        decoded = self.decoder(flat)
        return decoded.reshape(*feature.shape[:-1], -1)

    def predict_reward_from_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        reward_cont, _, fall_logit = self.reward_outputs_from_features(h, z)
        return self.combine_reward_outputs(reward_cont, fall_logit)

    def predict_continuation_from_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feature = torch.cat([h, z], dim=-1)
        if not hasattr(self, "continuation_model"):
            return torch.ones(*feature.shape[:-1], 1, device=feature.device, dtype=feature.dtype)
        flat = feature.reshape(-1, feature.shape[-1])
        continuation_logit = self.continuation_model(flat).reshape(*feature.shape[:-1], -1)
        return torch.sigmoid(continuation_logit)

    def reward_outputs_from_features(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        feature = torch.cat([h, z], dim=-1)
        flat = feature.reshape(-1, feature.shape[-1])
        reward = self.reward_model(flat)
        reward = reward.reshape(*feature.shape[:-1], -1)
        continuation_logit = None
        if hasattr(self, "continuation_model"):
            continuation_flat = self.continuation_model(flat)
            continuation_logit = continuation_flat.reshape(*feature.shape[:-1], -1)
        fall_logit = None
        if hasattr(self, "fall_model"):
            fall_flat = self.fall_model(flat)
            fall_logit = fall_flat.reshape(*feature.shape[:-1], -1)
        return reward, continuation_logit, fall_logit

    def combine_reward_outputs(
        self,
        reward_cont: torch.Tensor,
        fall_logit: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.reward_prediction_mode != "split_fall" or fall_logit is None:
            return reward_cont
        fall_prob = torch.sigmoid(fall_logit)
        return reward_cont + fall_prob * self.config.fall_penalty_value


def independent_normal(mean: torch.Tensor, std: torch.Tensor) -> Independent:
    return Independent(Normal(mean, std), 1)


def transition_masks(
    action_seq: torch.Tensor,
    reward_seq: torch.Tensor | None,
    done_seq: torch.Tensor | None,
    terminated_seq: torch.Tensor | None,
    fall_reward_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, action_steps = action_seq.shape[:2]
    device = action_seq.device
    if done_seq is None:
        effective = torch.ones(batch_size, action_steps, device=device, dtype=torch.bool)
        fall_terminal = torch.zeros_like(effective)
        return effective, effective, fall_terminal, ~effective

    done = done_seq.squeeze(-1).bool()
    effective = torch.ones_like(done, dtype=torch.bool)
    effective[:, 1:] = ~done[:, :-1]
    post_done = ~effective
    nonterminal = effective & ~done
    if terminated_seq is not None:
        fall_terminal = effective & terminated_seq.squeeze(-1).bool()
    elif reward_seq is not None:
        fall_terminal = effective & done & (reward_seq.squeeze(-1) <= fall_reward_threshold)
    else:
        fall_terminal = torch.zeros_like(done, dtype=torch.bool)
    return effective, nonterminal, fall_terminal, post_done


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")
    denom = mask.to(dtype=values.dtype).sum()
    if denom <= 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return values.masked_select(mask).mean()


def masked_weighted_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape or values.shape != weights.shape:
        raise ValueError("values, mask, and weights must have the same shape")
    active_weights = weights * mask.to(dtype=weights.dtype)
    denom = active_weights.sum()
    if denom <= 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * active_weights).sum() / denom
