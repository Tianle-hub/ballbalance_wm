from __future__ import annotations

import torch

from ball_rssm.models import WorldModel, WorldModelConfig


def test_world_model_shapes_and_finite_loss() -> None:
    batch_size = 3
    seq_len = 7
    obs_seq = torch.randn(batch_size, seq_len + 1, 6)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1)

    model = WorldModel(
        WorldModelConfig(
            obs_dim=6,
            action_dim=2,
            deter_dim=32,
            stoch_dim=8,
            embed_dim=16,
            hidden_dim=32,
        )
    )
    out = model.forward(obs_seq, action_seq)

    assert out["recon"].shape == (batch_size, seq_len + 1, 6)
    assert out["prior_recon"].shape == (batch_size, seq_len + 1, 6)
    assert out["reward_pred"].shape == (batch_size, seq_len + 1, 1)
    assert out["prior_reward_pred"].shape == (batch_size, seq_len + 1, 1)
    assert out["prior"]["mean"].shape == (batch_size, seq_len + 1, 8)
    assert out["prior"]["std"].shape == (batch_size, seq_len + 1, 8)
    assert out["posterior"]["mean"].shape == (batch_size, seq_len + 1, 8)
    assert out["posterior"]["std"].shape == (batch_size, seq_len + 1, 8)

    loss, metrics = model.loss(obs_seq, action_seq, reward_seq)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["recon_loss"])
    assert torch.isfinite(metrics["reward_loss"])
    assert torch.isfinite(metrics["kl_loss"])

    pred = model.open_loop_predict(obs_seq, action_seq, context_len=2, horizon=4)
    assert pred.shape == (batch_size, 4, 6)
    reward_pred = model.open_loop_predict_rewards(obs_seq, action_seq, context_len=2, horizon=4)
    assert reward_pred.shape == (batch_size, 4, 1)
