from __future__ import annotations

import numpy as np
import torch

from ball_rssm.data.sequence_dataset import SequenceDataset
from ball_rssm.buffer import Buffer
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from scripts.collect_dataset import InitialStateBounds, collect_dataset, save_dataset


def test_sequence_dataset_window_shapes(tmp_path) -> None:
    obs = np.random.randn(4, 11, 6).astype(np.float32)
    action = np.random.randn(4, 10, 2).astype(np.float32)
    reward = np.random.randn(4, 10, 1).astype(np.float32)
    done = np.zeros((4, 10, 1), dtype=np.float32)
    terminated = np.zeros((4, 10, 1), dtype=np.float32)
    truncated = np.zeros((4, 10, 1), dtype=np.float32)
    path = tmp_path / "dummy.npz"
    np.savez(path, obs=obs, action=action, reward=reward, done=done, terminated=terminated, truncated=truncated)

    dataset = SequenceDataset(path, seq_len=5, split="all")
    sample = dataset[0]

    assert sample["obs"].shape == (6, 6)
    assert sample["action"].shape == (5, 2)
    assert sample["reward"].shape == (5, 1)
    assert sample["done"].shape == (5, 1)
    assert sample["terminated"].shape == (5, 1)
    assert sample["truncated"].shape == (5, 1)
    assert sample["obs"].dtype.is_floating_point
    assert sample["action"].dtype.is_floating_point


def test_buffer_collect_save_load_and_sequence_dataset(tmp_path) -> None:
    path = tmp_path / "buffer_dataset.npz"
    buffer = Buffer.collect_data(num_episodes=3, max_episode_steps=12, seed=5, mode="mixed")
    buffer.save(path, mode="mixed")

    loaded = Buffer.load(path)
    dataset = loaded.sequence_dataset(seq_len=6, split="all")
    sample = dataset[0]

    assert loaded.obs_buffer.shape == (3, 13, 3, 64, 64)
    assert loaded.action_buffer.shape == (3, 12, 2)
    assert loaded.reward_buffer.shape == (3, 12, 1)
    assert sample["obs"].shape == (7, 3, 64, 64)
    assert sample["action"].shape == (6, 2)
    assert sample["terminated"].shape == (6, 1)
    assert sample["truncated"].shape == (6, 1)


def test_coverage_collection_can_feed_dreamer_world_model_training(tmp_path) -> None:
    collected = collect_dataset(
        num_episodes=6,
        max_episode_steps=20,
        seed=3,
        mode="coverage",
        initial_bounds=InitialStateBounds(pos=0.25, vel=0.20, angle=0.12),
        target_bound=0.15,
        action_noise_std=0.04,
    )
    path = tmp_path / "coverage.npz"
    save_dataset(collected, path, mode="coverage")

    dataset = SequenceDataset(path, seq_len=10, split="all")
    assert len(dataset) > 0

    train_obs, train_action, train_reward = dataset.selected_arrays()
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("obs", "action", "reward", "done", "terminated")
    }
    obs = normalizer.normalize_obs(batch["obs"])
    action = normalizer.normalize_action(batch["action"])
    reward = normalizer.normalize_reward(batch["reward"])

    model = WorldModel(
        WorldModelConfig(
            obs_shape=tuple(int(x) for x in train_obs.shape[2:]),
            action_dim=2,
            deter_dim=32,
            stoch_dim=8,
            embed_dim=16,
            hidden_dim=32,
        )
    )
    loss, metrics = model.loss(obs, action, reward, batch["done"], batch["terminated"])
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["recon_loss"])
    assert torch.isfinite(metrics["reward_loss"])
    assert torch.isfinite(metrics["reward_loss_nonterminal"])
    assert torch.isfinite(metrics["reward_loss_terminal"])
    assert torch.isfinite(metrics["reward_loss_post_done_padding"])
    assert torch.isfinite(metrics["continuation_loss"])
    assert torch.isfinite(metrics["kl_loss"])
    loss.backward()
