from __future__ import annotations

import numpy as np
import torch

from ball_rssm.evaluation import RandomICBounds, evaluate_imagined_trajectory_error, generate_random_ic_dataset
from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig


def test_random_ic_dataset_bounds_and_shapes() -> None:
    bounds = RandomICBounds(pos=0.2, vel=0.1, angle=0.05)
    dataset = generate_random_ic_dataset(num_episodes=5, horizon=8, context_len=3, seed=4, bounds=bounds, action_mode="mixed")

    assert dataset["obs"].shape == (5, 12, 6)
    assert dataset["action"].shape == (5, 11, 2)
    assert dataset["initial_state"].shape == (5, 6)
    assert np.all(np.abs(dataset["initial_state"][:, :2]) <= bounds.pos + 1e-6)
    assert np.all(np.abs(dataset["initial_state"][:, 2:4]) <= bounds.vel + 1e-6)
    assert np.all(np.abs(dataset["initial_state"][:, 4:6]) <= bounds.angle + 1e-6)
    assert np.isfinite(dataset["obs"]).all()
    assert np.isfinite(dataset["action"]).all()


def test_random_ic_imagined_error_metrics_are_finite() -> None:
    dataset = generate_random_ic_dataset(num_episodes=4, horizon=5, context_len=2, seed=6, bounds=RandomICBounds(0.15, 0.08, 0.04))
    model = WorldModel(WorldModelConfig(deter_dim=16, stoch_dim=4, embed_dim=8, hidden_dim=16))
    normalizer = Normalizer.from_arrays(dataset["obs"], dataset["action"])

    metrics = evaluate_imagined_trajectory_error(
        model=model,
        normalizer=normalizer,
        dataset=dataset,
        context_len=2,
        horizon=5,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert metrics["num_episodes"] == 4
    assert metrics["horizon"] == 5
    assert np.isfinite(metrics["obs_mse"])
    assert np.isfinite(metrics["position_l2_mean"])
    assert len(metrics["position_l2_curve"]) == 5
    assert len(metrics["velocity_l2_curve"]) == 5
