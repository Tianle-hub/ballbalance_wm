from __future__ import annotations

import torch

from ball_rssm.buffer import Buffer
from ball_rssm.data.sequence_dataset import batch_to_device
from ball_rssm.models import Actor, ActorConfig, Critic, CriticConfig, Normalizer, WorldModel, WorldModelConfig
from ball_rssm.models.behavior import lambda_return
from ball_rssm.models.rssm import RSSMState
from ball_rssm.trainer import DreamerTrainConfig, Trainer, sample_state_batch


def test_actor_outputs_bounded_normalized_actions() -> None:
    actor = Actor(
        ActorConfig(
            feature_dim=12,
            action_dim=2,
            hidden_dim=16,
            action_low=(-0.5, -0.25),
            action_high=(0.5, 0.25),
        )
    )
    features = torch.randn(4, 3, 12)

    sample = actor.sample(features, deterministic=False)
    mode = actor.sample(features, deterministic=True)

    assert sample.shape == (4, 3, 2)
    assert mode.shape == (4, 3, 2)
    assert torch.all(sample[..., 0] <= 0.5 + 1e-6)
    assert torch.all(sample[..., 0] >= -0.5 - 1e-6)
    assert torch.all(sample[..., 1] <= 0.25 + 1e-6)
    assert torch.all(sample[..., 1] >= -0.25 - 1e-6)


def test_lambda_return_matches_discounted_return_when_lambda_one() -> None:
    reward = torch.ones(1, 3, 1)
    value = torch.zeros(1, 3, 1)
    bootstrap = torch.zeros(1, 1)
    pcont = torch.full((1, 3, 1), 0.9)

    returns = lambda_return(reward, value, bootstrap, pcont, lambda_=1.0)

    expected = torch.tensor([[[1.0 + 0.9 + 0.9**2], [1.0 + 0.9], [1.0]]])
    torch.testing.assert_close(returns, expected)


def test_sample_state_batch_caps_posterior_starts() -> None:
    state = RSSMState(
        h=torch.randn(10, 3),
        z=torch.randn(10, 2),
        mean=torch.randn(10, 2),
        std=torch.rand(10, 2) + 0.1,
    )

    sampled = sample_state_batch(state, max_states=4)
    unchanged = sample_state_batch(state, max_states=20)

    assert sampled.h.shape == (4, 3)
    assert sampled.z.shape == (4, 2)
    assert sampled.mean.shape == (4, 2)
    assert sampled.std.shape == (4, 2)
    assert unchanged is state


def test_trainer_updates_world_actor_and_critic_one_batch(tmp_path) -> None:
    buffer = Buffer.collect_data(num_episodes=3, max_episode_steps=8, seed=0, mode="mixed")
    dataset = buffer.sequence_dataset(seq_len=4, split="all")
    batch = batch_to_device(
        {key: torch.stack([dataset[i][key] for i in range(2)]) for key in ("obs", "action", "reward", "done", "terminated")},
        "cpu",
    )
    train_obs, train_action, train_reward = dataset.selected_arrays()
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward)

    world_config = WorldModelConfig(deter_dim=16, stoch_dim=4, embed_dim=8, hidden_dim=16)
    world_model = WorldModel(world_config)
    actor = Actor(
        ActorConfig(
            feature_dim=world_model.feature_dim,
            action_dim=world_config.action_dim,
            hidden_dim=16,
            action_low=(-1.0, -1.0),
            action_high=(1.0, 1.0),
        )
    )
    critic = Critic(CriticConfig(feature_dim=world_model.feature_dim, hidden_dim=16))
    target_critic = Critic(CriticConfig(feature_dim=world_model.feature_dim, hidden_dim=16))
    target_critic.load_state_dict(critic.state_dict())

    trainer = Trainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        buffer=buffer,
        world_optimizer=torch.optim.Adam(world_model.parameters(), lr=1e-3),
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
        normalizer=normalizer,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        config=DreamerTrainConfig(imagination_horizon=3, behavior_batch_size=3, grad_clip=10.0),
    )
    obs = normalizer.normalize_obs(batch["obs"])
    action = normalizer.normalize_action(batch["action"])
    reward = normalizer.normalize_reward(batch["reward"])

    metrics = trainer.train_batch(obs, action, reward, batch["done"], batch["terminated"])

    assert int(metrics["behavior_sample_count"].item()) == 3
    assert int(metrics["behavior_start_count"].item()) > int(metrics["behavior_sample_count"].item())
    for key in ("total_loss", "actor_loss", "critic_loss", "world_grad_norm", "actor_grad_norm", "critic_grad_norm"):
        assert key in metrics
        assert torch.isfinite(metrics[key])
