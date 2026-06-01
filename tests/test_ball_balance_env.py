from __future__ import annotations

import numpy as np

from ball_rssm.envs import BallBalanceEnv
from scripts.collect_dataset import collect_dataset


def test_reset_returns_obs_and_info() -> None:
    env = BallBalanceEnv()
    obs, info = env.reset(seed=0)
    try:
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_action_space_samples_are_valid() -> None:
    env = BallBalanceEnv()
    try:
        for _ in range(10):
            assert env.action_space.contains(env.action_space.sample())
    finally:
        env.close()


def test_step_api_and_observation_space() -> None:
    env = BallBalanceEnv()
    obs, _ = env.reset(seed=1)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    try:
        assert env.observation_space.contains(obs)
        assert env.observation_space.contains(next_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert info["state"].shape == (6,)
        assert info["acceleration"].shape == (2,)
        assert info["action_clipped"].shape == (2,)
    finally:
        env.close()


def test_deterministic_reset_with_same_seed() -> None:
    env = BallBalanceEnv()
    try:
        obs_a, _ = env.reset(seed=123)
        obs_b, _ = env.reset(seed=123)
        np.testing.assert_allclose(obs_a, obs_b)
    finally:
        env.close()


def test_zero_action_from_zero_state_stays_near_zero() -> None:
    env = BallBalanceEnv()
    try:
        env.reset(seed=0)
        env.state[:] = 0.0
        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            assert not terminated
            assert not truncated
            np.testing.assert_allclose(obs, np.zeros(6, dtype=np.float32), atol=1e-6)
    finally:
        env.close()


def test_large_initial_position_eventually_falls_or_truncates() -> None:
    env = BallBalanceEnv(config={"max_episode_steps": 200})
    try:
        env.reset(seed=0)
        env.state[:] = np.array([0.49, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)
        done = False
        for _ in range(env.config.max_episode_steps):
            _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            done = terminated or truncated
            if done:
                break
        assert done
    finally:
        env.close()


def test_dataset_collector_shapes() -> None:
    dataset = collect_dataset(num_episodes=3, max_episode_steps=12, seed=7, mode="mixed")
    assert dataset["obs"].shape == (3, 13, 6)
    assert dataset["action"].shape == (3, 12, 2)
    assert dataset["reward"].shape == (3, 12, 1)
    assert dataset["terminated"].shape == (3, 12, 1)
    assert dataset["truncated"].shape == (3, 12, 1)
    assert dataset["done"].shape == (3, 12, 1)
    assert dataset["obs"].dtype == np.float32
    assert dataset["action"].dtype == np.float32
