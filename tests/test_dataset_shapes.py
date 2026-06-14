from __future__ import annotations

import numpy as np
import torch

from ball_rssm.data.sequence_dataset import SequenceDataset
from ball_rssm.buffer import Buffer
from ball_rssm.envs.dm_control import preprocess_pixel_frame
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.stream_replay import StreamReplay
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


def test_pixel_sequence_dataset_and_normalizer_shapes(tmp_path) -> None:
    obs = np.random.rand(3, 7, 3, 64, 64).astype(np.float32)
    action = np.random.randn(3, 6, 2).astype(np.float32)
    reward = np.random.randn(3, 6, 1).astype(np.float32)
    path = tmp_path / "pixel_dummy.npz"
    np.savez(path, obs=obs, action=action, reward=reward)

    dataset = SequenceDataset(path, seq_len=4, split="all")
    sample = dataset[0]
    train_obs, train_action, train_reward = dataset.selected_arrays()
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward)
    obs_norm = normalizer.normalize_obs(sample["obs"].unsqueeze(0))

    assert sample["obs"].shape == (5, 3, 64, 64)
    assert sample["action"].shape == (4, 2)
    assert normalizer.obs_mean.shape == (3, 64, 64)
    assert obs_norm.shape == (1, 5, 3, 64, 64)


def test_pixel_preprocessing_matches_dreamer_centered_scale() -> None:
    frame = np.array(
        [
            [[0, 127, 255], [255, 127, 0]],
            [[64, 128, 192], [32, 160, 224]],
        ],
        dtype=np.uint8,
    )

    processed = preprocess_pixel_frame(frame)

    assert processed.shape == (3, 2, 2)
    np.testing.assert_allclose(processed[:, 0, 0], np.array([-0.5, 127.0 / 255.0 - 0.5, 0.5]))
    assert processed.dtype == np.float32
    assert processed.min() >= -0.5
    assert processed.max() <= 0.5


def test_normalizer_can_leave_pixel_observations_unscaled() -> None:
    obs = (np.random.rand(2, 5, 3, 8, 8).astype(np.float32) - 0.5)
    action = np.random.uniform(-1.0, 1.0, size=(2, 4, 2)).astype(np.float32)
    reward = np.random.randn(2, 4, 1).astype(np.float32)

    normalizer = Normalizer.from_arrays(obs, action, reward, normalize_obs=False, normalize_action=False)
    obs_t = torch.from_numpy(obs[:1])
    action_t = torch.from_numpy(action[:1])

    torch.testing.assert_close(normalizer.normalize_obs(obs_t), obs_t)
    torch.testing.assert_close(normalizer.denormalize_obs(obs_t), obs_t)
    torch.testing.assert_close(normalizer.normalize_action(action_t), action_t)
    assert normalizer.obs_mean.shape == (3, 8, 8)
    assert torch.all(normalizer.obs_mean == 0.0)
    assert torch.all(normalizer.obs_std == 1.0)


def test_buffer_collect_save_load_and_sequence_dataset(tmp_path) -> None:
    path = tmp_path / "buffer_dataset.npz"
    buffer = Buffer.collect_data(num_episodes=3, max_episode_steps=12, seed=5, mode="mixed")
    buffer.save(path, mode="mixed")

    loaded = Buffer.load(path)
    dataset = loaded.sequence_dataset(seq_len=6, split="all")
    sample = dataset[0]

    assert loaded.obs_buffer.shape == (3, 13, 6)
    assert loaded.action_buffer.shape == (3, 12, 2)
    assert loaded.reward_buffer.shape == (3, 12, 1)
    assert sample["obs"].shape == (7, 6)
    assert sample["action"].shape == (6, 2)
    assert sample["terminated"].shape == (6, 1)
    assert sample["truncated"].shape == (6, 1)


def test_stream_replay_saves_is_first_step_stream(tmp_path) -> None:
    replay = StreamReplay(
        capacity_steps=10,
        obs_shape=(3,),
        action_shape=(2,),
        max_episode_steps=4,
        action_low=-np.ones(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
    )
    replay.start_episode(np.array([0.0, 0.1, 0.2], dtype=np.float32))
    replay.add_transition(
        action=np.array([0.5, -0.5], dtype=np.float32),
        reward=1.0,
        terminated=False,
        truncated=False,
        done=False,
        next_obs=np.array([0.2, 0.3, 0.4], dtype=np.float32),
    )
    replay.start_episode(np.array([1.0, 1.1, 1.2], dtype=np.float32))
    replay.add_transition(
        action=np.array([-0.25, 0.25], dtype=np.float32),
        reward=0.5,
        terminated=True,
        truncated=False,
        done=True,
        next_obs=np.array([1.2, 1.3, 1.4], dtype=np.float32),
    )

    path = tmp_path / "stream_replay.npz"
    replay.save(path)
    loaded = StreamReplay.load(path)
    dataset = loaded.sequence_dataset(seq_len=2, split="all")
    sample = dataset[0]

    arrays = loaded.to_dataset()
    assert loaded.size == 2
    assert loaded.num_steps == 3
    assert arrays["obs"].shape == (1, 4, 3)
    assert arrays["action"].shape == (1, 3, 2)
    assert arrays["is_first"].shape == (1, 4, 1)
    assert arrays["is_first"][0, 0, 0]
    assert arrays["is_first"][0, 2, 0]
    assert sample["is_first"].shape == (3, 1)


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
            obs_dim=6,
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
