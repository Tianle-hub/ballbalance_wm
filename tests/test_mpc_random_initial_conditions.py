from __future__ import annotations

import os

import numpy as np
import pytest

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import InitialConditionBounds, run_mpc_episode, sample_initial_state
from ball_rssm.planning.rssm_mpc import RSSMMPCController


def test_env_reset_options_state_sets_bounded_state() -> None:
    env = BallBalanceEnv()
    state = np.array([0.1, -0.1, 0.02, -0.03, 0.04, -0.02], dtype=np.float32)
    try:
        obs, _ = env.reset(seed=0, options={"state": state})
        np.testing.assert_allclose(obs, state, atol=1e-7)
    finally:
        env.close()


def test_rssm_mpc_smoke_action_from_checkpoint() -> None:
    checkpoint = os.environ.get("RSSM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set RSSM_CHECKPOINT to run RSSM MPC checkpoint smoke test")

    env = BallBalanceEnv()
    try:
        obs, _ = env.reset(seed=0, options={"state": sample_initial_state(np.random.default_rng(0), InitialConditionBounds(0.2, 0.08, 0.04))})
        controller = RSSMMPCController(
            checkpoint_path=checkpoint,
            action_space=env.action_space,
            horizon=5,
            num_candidates=32,
            num_elites=4,
            num_iterations=2,
            device="cpu",
            cost_mode="center",
            seed=0,
        )
        controller.reset(obs)
        action = controller.act(obs)
        assert env.action_space.contains(action.astype(env.action_space.dtype))
        assert np.isfinite(action).all()
    finally:
        env.close()


def test_rssm_mpc_random_initial_conditions_integration() -> None:
    checkpoint = os.environ.get("RSSM_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set RSSM_CHECKPOINT to run RSSM MPC integration test")

    env = BallBalanceEnv(config={"max_episode_steps": 100})
    rng = np.random.default_rng(3)
    non_fall_count = 0
    total = 3
    try:
        for idx in range(total):
            controller = RSSMMPCController(
                checkpoint_path=checkpoint,
                action_space=env.action_space,
                horizon=5,
                num_candidates=32,
                num_elites=4,
                num_iterations=2,
                device="cpu",
                cost_mode="center",
                seed=idx,
            )
            initial_state = sample_initial_state(rng, InitialConditionBounds(0.2, 0.08, 0.04))
            episode = run_mpc_episode(controller, env, initial_state, max_steps=100)
            obs = np.asarray(episode["obs"])
            action = np.asarray(episode["action"])
            assert np.isfinite(obs).all()
            assert np.isfinite(action).all()
            assert np.isfinite(np.linalg.norm(obs[-1, :2]))
            assert np.all(action <= env.action_space.high + 1e-6)
            assert np.all(action >= env.action_space.low - 1e-6)
            non_fall_count += int(not bool(episode["terminated"]))
    finally:
        env.close()
    assert non_fall_count >= 1
