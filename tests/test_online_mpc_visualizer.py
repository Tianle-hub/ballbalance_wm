from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.online_mpc_visualizer import (
    OnlineInitialBounds,
    sample_online_initial_state,
    sample_target,
    select_episode_task,
)


def test_sample_online_initial_state_respects_per_dimension_bounds() -> None:
    rng = np.random.default_rng(0)
    bounds = OnlineInitialBounds(x=0.1, y=0.2, vx=0.03, vy=0.04, theta_x=0.05, theta_y=0.06)

    states = np.asarray([sample_online_initial_state(rng, bounds) for _ in range(100)])

    assert states.shape == (100, 6)
    assert np.all(np.abs(states[:, 0]) <= bounds.x)
    assert np.all(np.abs(states[:, 1]) <= bounds.y)
    assert np.all(np.abs(states[:, 2]) <= bounds.vx)
    assert np.all(np.abs(states[:, 3]) <= bounds.vy)
    assert np.all(np.abs(states[:, 4]) <= bounds.theta_x)
    assert np.all(np.abs(states[:, 5]) <= bounds.theta_y)


def test_sample_target_respects_bound() -> None:
    rng = np.random.default_rng(1)
    targets = np.asarray([sample_target(rng, 0.15) for _ in range(100)])

    assert targets.shape == (100, 2)
    assert np.all(np.abs(targets) <= 0.15)


def test_select_episode_task_center_and_viapoint() -> None:
    rng = np.random.default_rng(2)

    center_task, center_target = select_episode_task(rng, "center", 0.15)
    viapoint_task, viapoint_target = select_episode_task(rng, "viapoint", 0.15)

    assert center_task == "center"
    assert center_target == (0.0, 0.0)
    assert viapoint_task == "viapoint"
    assert abs(viapoint_target[0]) <= 0.15
    assert abs(viapoint_target[1]) <= 0.15


def test_select_episode_task_random_samples_valid_task() -> None:
    rng = np.random.default_rng(3)

    for _ in range(20):
        task_name, target = select_episode_task(rng, "random", 0.15)
        assert task_name in {"center", "viapoint"}
        if task_name == "center":
            assert target == (0.0, 0.0)
        else:
            assert abs(target[0]) <= 0.15
            assert abs(target[1]) <= 0.15
