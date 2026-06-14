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
    assert out["continuation_logit"].shape == (batch_size, seq_len + 1, 1)
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
    continuation_pred = model.open_loop_predict_continuation(obs_seq, action_seq, context_len=2, horizon=4)
    assert continuation_pred.shape == (batch_size, 4, 1)
    assert torch.all((continuation_pred >= 0.0) & (continuation_pred <= 1.0))


def test_dreamer_v2_discrete_world_model_shapes_and_balanced_kl() -> None:
    batch_size = 3
    seq_len = 6
    obs_seq = torch.randn(batch_size, seq_len + 1, 6)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1)

    model = WorldModel(
        WorldModelConfig(
            obs_dim=6,
            action_dim=2,
            deter_dim=32,
            stoch_dim=4,
            discrete_classes=8,
            embed_dim=16,
            hidden_dim=32,
            dreamer_version="v2",
        )
    )
    out = model.forward(obs_seq, action_seq)

    assert model.feature_dim == 32 + 4 * 8
    assert out["recon"].shape == (batch_size, seq_len + 1, 6)
    assert out["prior"]["z"].shape == (batch_size, seq_len + 1, 4 * 8)
    assert out["prior"]["logits"].shape == (batch_size, seq_len + 1, 4, 8)
    assert out["posterior"]["logits"].shape == (batch_size, seq_len + 1, 4, 8)

    loss, metrics = model.loss(obs_seq, action_seq, reward_seq)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])
    assert torch.isfinite(metrics["dynamics_kl_loss"])
    assert torch.isfinite(metrics["representation_kl_loss"])
    assert torch.isfinite(metrics["posterior_entropy_mean"])


def test_dreamer_v2_kl_free_avg_threshold() -> None:
    batch_size = 2
    seq_len = 4
    obs_seq = torch.randn(batch_size, seq_len + 1, 6)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1)

    model = WorldModel(
        WorldModelConfig(
            obs_dim=6,
            action_dim=2,
            deter_dim=16,
            stoch_dim=3,
            discrete_classes=5,
            embed_dim=8,
            hidden_dim=16,
            dreamer_version="v2",
            kl_free=10.0,
            kl_balance=0.8,
            kl_forward=False,
            kl_free_avg=True,
        )
    )

    loss, metrics = model.loss(obs_seq, action_seq, reward_seq)

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["kl_loss"], torch.tensor(10.0), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(metrics["dynamics_kl_loss"], torch.tensor(10.0), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(metrics["representation_kl_loss"], torch.tensor(10.0), rtol=0.0, atol=1e-6)


def test_dreamer_v2_forward_kl_and_free_per_step_are_finite() -> None:
    batch_size = 2
    seq_len = 4
    obs_seq = torch.randn(batch_size, seq_len + 1, 6)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1)

    model = WorldModel(
        WorldModelConfig(
            obs_dim=6,
            action_dim=2,
            deter_dim=16,
            stoch_dim=3,
            discrete_classes=5,
            embed_dim=8,
            hidden_dim=16,
            dreamer_version="v2",
            kl_free=0.1,
            kl_balance=0.7,
            kl_forward=True,
            kl_free_avg=False,
        )
    )

    loss, metrics = model.loss(obs_seq, action_seq, reward_seq)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])
    assert torch.isfinite(metrics["raw_kl"])
    assert metrics["kl_loss"].item() >= 0.1


def test_continuation_reward_model_shapes_and_finite_loss() -> None:
    batch_size = 3
    seq_len = 7
    obs_seq = torch.randn(batch_size, seq_len + 1, 6)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1).clamp(-1.0, 1.0)
    done_seq = torch.zeros(batch_size, seq_len, 1)
    terminated_seq = torch.zeros(batch_size, seq_len, 1)
    done_seq[0, 3:] = 1.0
    terminated_seq[0, 3] = 1.0
    reward_seq[0, 3] = -1.0

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

    assert out["reward_pred"].shape == (batch_size, seq_len + 1, 1)
    assert out["continuation_logit"].shape == (batch_size, seq_len + 1, 1)

    loss, metrics = model.loss(obs_seq, action_seq, reward_seq, done_seq, terminated_seq)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["reward_loss"])
    assert torch.isfinite(metrics["continuation_loss"])
    assert torch.isfinite(metrics["continuation_prob_nonterminal"])
    assert torch.isfinite(metrics["continuation_prob_terminal"])


def test_world_model_masks_is_first_reset_transitions() -> None:
    batch_size = 2
    seq_len = 5
    obs_seq = torch.randn(batch_size, seq_len + 1, 6)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1)
    done_seq = torch.zeros(batch_size, seq_len, 1)
    terminated_seq = torch.zeros(batch_size, seq_len, 1)
    is_first_seq = torch.zeros(batch_size, seq_len + 1, 1)
    is_first_seq[:, 0] = 1.0
    is_first_seq[0, 3] = 1.0

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
    loss, metrics = model.loss(obs_seq, action_seq, reward_seq, done_seq, terminated_seq, is_first_seq)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["reward_loss"])
    assert metrics["reward_effective_fraction"].item() < 1.0
    assert metrics["reward_post_done_padding_fraction"].item() > 0.0


def test_pixel_world_model_shapes_and_finite_loss() -> None:
    batch_size = 2
    seq_len = 3
    obs_shape = (3, 64, 64)
    obs_seq = torch.rand(batch_size, seq_len + 1, *obs_shape)
    action_seq = torch.randn(batch_size, seq_len, 2)
    reward_seq = torch.randn(batch_size, seq_len, 1)

    model = WorldModel(
        WorldModelConfig(
            obs_type="pixel",
            obs_shape=obs_shape,
            action_dim=2,
            deter_dim=24,
            stoch_dim=6,
            embed_dim=32,
            hidden_dim=32,
        )
    )
    out = model.forward(obs_seq, action_seq)

    assert model.config.obs_dim == 3 * 64 * 64
    assert out["recon"].shape == (batch_size, seq_len + 1, *obs_shape)
    assert out["prior_recon"].shape == (batch_size, seq_len + 1, *obs_shape)
    assert out["reward_pred"].shape == (batch_size, seq_len + 1, 1)
    assert out["continuation_logit"].shape == (batch_size, seq_len + 1, 1)
    assert out["prior"]["mean"].shape == (batch_size, seq_len + 1, 6)
    assert out["posterior"]["std"].shape == (batch_size, seq_len + 1, 6)

    loss, metrics = model.loss(obs_seq, action_seq, reward_seq)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["recon_loss"])
    assert torch.isfinite(metrics["reward_loss"])
    assert torch.isfinite(metrics["kl_loss"])

    pred = model.open_loop_predict(obs_seq, action_seq, context_len=1, horizon=2)
    assert pred.shape == (batch_size, 2, *obs_shape)


def test_world_model_config_round_trips_pixel_shape_from_dict() -> None:
    config = WorldModelConfig.from_dict(
        {
            "obs_type": "pixels",
            "obs_shape": [3, 64, 64],
            "action_dim": 4,
            "dreamer_version": "dreamer2",
            "stoch_dim": 4,
        }
    )

    assert config.obs_type == "pixel"
    assert config.obs_shape == (3, 64, 64)
    assert config.obs_dim == 3 * 64 * 64
    assert config.action_dim == 4
    assert config.is_v2
