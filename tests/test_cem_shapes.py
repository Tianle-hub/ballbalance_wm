from __future__ import annotations

import torch

from ball_rssm.planning.cem import CEMGDPlanner, CEMPlanner


def test_cem_optimizes_quadratic_and_respects_shapes() -> None:
    horizon = 6
    action_dim = 2
    candidate_shapes = []

    planner = CEMPlanner(
        action_dim=action_dim,
        horizon=horizon,
        num_candidates=128,
        num_elites=16,
        num_iterations=4,
        action_low=-0.25,
        action_high=0.25,
        init_std=0.2,
        min_std=0.01,
        device="cpu",
        seed=0,
    )

    def cost_fn(actions: torch.Tensor) -> torch.Tensor:
        candidate_shapes.append(tuple(actions.shape))
        return (actions**2).sum(dim=(1, 2))

    result = planner.plan(cost_fn)
    assert result.best_action_sequence.shape == (horizon, action_dim)
    assert result.best_cost.item() >= 0.0
    assert torch.all(result.best_action_sequence <= 0.25 + 1e-6)
    assert torch.all(result.best_action_sequence >= -0.25 - 1e-6)
    assert all(shape == (128, horizon, action_dim) for shape in candidate_shapes)
    assert torch.linalg.norm(result.best_action_sequence) < 0.5


def test_cem_warm_start_shift_preserves_shape() -> None:
    planner = CEMPlanner(action_dim=2, horizon=5, num_candidates=16, num_elites=4, num_iterations=1, device="cpu")
    sequence = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 100.0
    planner.shift_mean(sequence)
    assert planner.mean.shape == (5, 2)
    torch.testing.assert_close(planner.mean[:-1], sequence[1:])
    torch.testing.assert_close(planner.mean[-1], sequence[-1])


def test_cem_gd_uses_differentiable_refinement() -> None:
    horizon = 4
    action_dim = 1
    grad_calls = 0
    planner = CEMGDPlanner(
        action_dim=action_dim,
        horizon=horizon,
        num_candidates=64,
        num_elites=8,
        num_iterations=2,
        action_low=-1.0,
        action_high=1.0,
        init_std=0.5,
        gd_num_sequences=2,
        gd_iterations=10,
        gd_lr=0.05,
        device="cpu",
        seed=1,
    )
    target = torch.full((horizon, action_dim), 0.3)

    def cost_fn(actions: torch.Tensor) -> torch.Tensor:
        nonlocal grad_calls
        if actions.requires_grad:
            grad_calls += 1
        return ((actions - target) ** 2).sum(dim=(1, 2))

    result = planner.plan(cost_fn)

    assert grad_calls > 0
    assert result.best_action_sequence.shape == (horizon, action_dim)
    assert result.best_cost.item() >= 0.0
    assert torch.all(result.best_action_sequence <= 1.0 + 1e-6)
    assert torch.all(result.best_action_sequence >= -1.0 - 1e-6)
    assert result.diagnostics["planner_type"] == "cem_gd"
    gd = result.diagnostics["gradient_descent"]
    assert isinstance(gd, dict)
    assert gd["cost_after"] <= gd["cost_before"]
