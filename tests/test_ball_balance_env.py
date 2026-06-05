from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
            obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            assert not terminated
            assert not truncated
            assert reward == 1.0
            np.testing.assert_allclose(obs, np.zeros(6, dtype=np.float32), atol=1e-6)
    finally:
        env.close()


def test_fall_reward_is_bounded_terminal_penalty() -> None:
    env = BallBalanceEnv()
    try:
        env.reset(seed=0)
        env.state[:] = np.array([0.49, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert terminated
        assert not truncated
        assert info["fallen"]
        assert reward == -1.0
    finally:
        env.close()


def test_positive_theta_y_command_increases_x() -> None:
    env = BallBalanceEnv()
    try:
        env.reset(seed=0)
        env.state[:] = 0.0
        action = np.array([0.0, 0.1], dtype=np.float32)

        obs, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert not truncated
        assert info["acceleration"][0] > 0.0
        assert obs[2] > 0.0
        assert obs[0] > 0.0

        previous_x = obs[0]
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated
        assert obs[0] > previous_x
    finally:
        env.close()


def test_positive_theta_x_command_decreases_y() -> None:
    env = BallBalanceEnv()
    try:
        env.reset(seed=0)
        env.state[:] = 0.0
        action = np.array([0.1, 0.0], dtype=np.float32)

        obs, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert not truncated
        assert info["acceleration"][1] < 0.0
        assert obs[3] < 0.0
        assert obs[1] < 0.0

        previous_y = obs[1]
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            assert not truncated
        assert obs[1] < previous_y
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


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
