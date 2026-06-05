"""Cross-entropy planners for bounded action sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


CostFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class CEMResult:
    best_action_sequence: torch.Tensor
    best_cost: torch.Tensor
    diagnostics: dict[str, object]


class CEMPlanner:
    """Cross-entropy optimizer over finite-horizon continuous action sequences."""

    def __init__(
        self,
        action_dim: int,
        horizon: int = 25,
        num_candidates: int = 1024,
        num_elites: int = 100,
        num_iterations: int = 4,
        action_low: np.ndarray | torch.Tensor | float = -1.0,
        action_high: np.ndarray | torch.Tensor | float = 1.0,
        init_std: float | torch.Tensor = 0.15,
        min_std: float = 0.02,
        momentum: float = 0.1,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if not 0 < num_elites <= num_candidates:
            raise ValueError("num_elites must be in (0, num_candidates]")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")

        self.action_dim = action_dim
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.device = torch.device(device)
        self.min_std = min_std
        self.momentum = momentum
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

        self.action_low = _as_action_tensor(action_low, horizon, action_dim, self.device)
        self.action_high = _as_action_tensor(action_high, horizon, action_dim, self.device)
        self.init_std = _as_action_tensor(init_std, horizon, action_dim, self.device)
        self.reset_distribution()

    def reset_distribution(self) -> None:
        """Reset CEM sampling distribution to zero mean and initial std."""

        # Mean/std parameterize the sampling distribution over the whole action horizon.
        self.mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        self.std = self.init_std.clone().clamp_min(self.min_std)

    def shift_mean(self, previous_sequence: torch.Tensor) -> None:
        """Warm-start the next MPC step from the previous best sequence."""

        previous_sequence = torch.as_tensor(previous_sequence, device=self.device, dtype=torch.float32)
        if previous_sequence.shape != (self.horizon, self.action_dim):
            raise ValueError(f"expected previous_sequence shape {(self.horizon, self.action_dim)}")
        # Receding-horizon warm start: after executing action 0, reuse the rest
        # of the previous best plan as the next initial mean.
        self.mean[:-1] = previous_sequence[1:]
        self.mean[-1] = previous_sequence[-1]
        self.mean = torch.clamp(self.mean, self.action_low, self.action_high)

    def plan(self, cost_fn: CostFn) -> CEMResult:
        """Optimize action sequence candidates under a batched cost function."""

        best_sequence: torch.Tensor | None = None
        best_cost = torch.tensor(float("inf"), device=self.device)
        best_cost_per_iter: list[float] = []
        elite_cost_mean_per_iter: list[float] = []

        for _ in range(self.num_iterations):
            # Sample bounded action sequences around the current distribution,
            # evaluate them in batch, then refit to the lowest-cost elites.
            noise = torch.randn(
                self.num_candidates,
                self.horizon,
                self.action_dim,
                generator=self.generator,
                device=self.device,
            )
            candidates = self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise
            candidates = torch.clamp(candidates, self.action_low.unsqueeze(0), self.action_high.unsqueeze(0))
            costs = _evaluate_costs(cost_fn, candidates)

            elite_costs, elite_idx = torch.topk(costs, k=self.num_elites, largest=False)
            elites = candidates[elite_idx]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(self.min_std)

            # Momentum damps abrupt distribution jumps between CEM iterations.
            old_mean = self.mean
            old_std = self.std
            self.mean = self.momentum * old_mean + (1.0 - self.momentum) * elite_mean
            self.std = (self.momentum * old_std + (1.0 - self.momentum) * elite_std).clamp_min(self.min_std)
            self.mean = torch.clamp(self.mean, self.action_low, self.action_high)

            iter_best_cost = elite_costs[0]
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_sequence = candidates[elite_idx[0]].detach().clone()
            best_cost_per_iter.append(float(iter_best_cost.detach().cpu()))
            elite_cost_mean_per_iter.append(float(elite_costs.mean().detach().cpu()))

        if best_sequence is None:
            raise RuntimeError("CEM did not evaluate any candidates")

        diagnostics = {
            "planner_type": "cem",
            "best_cost_per_iteration": best_cost_per_iter,
            "elite_cost_mean_per_iteration": elite_cost_mean_per_iter,
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "best_cost": float(best_cost.detach().cpu()),
        }
        return CEMResult(best_action_sequence=best_sequence, best_cost=best_cost, diagnostics=diagnostics)


class CEMGDPlanner(CEMPlanner):
    """CEM planner with projected gradient refinement of top sampled sequences."""

    def __init__(
        self,
        *args: Any,
        gd_num_sequences: int = 3,
        gd_iterations: int = 15,
        gd_lr: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if gd_num_sequences <= 0:
            raise ValueError("gd_num_sequences must be positive")
        if gd_iterations < 0:
            raise ValueError("gd_iterations must be non-negative")
        if gd_lr <= 0.0:
            raise ValueError("gd_lr must be positive")
        self.gd_num_sequences = min(gd_num_sequences, self.num_elites)
        self.gd_iterations = gd_iterations
        self.gd_lr = gd_lr

    def plan(self, cost_fn: CostFn) -> CEMResult:
        """Run CEM, then refine the best sampled sequences with action gradients."""

        best_sequence: torch.Tensor | None = None
        best_cost = torch.tensor(float("inf"), device=self.device)
        best_cost_per_iter: list[float] = []
        elite_cost_mean_per_iter: list[float] = []
        top_sequences: list[torch.Tensor] = []
        top_costs: list[torch.Tensor] = []

        for _ in range(self.num_iterations):
            noise = torch.randn(
                self.num_candidates,
                self.horizon,
                self.action_dim,
                generator=self.generator,
                device=self.device,
            )
            candidates = self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise
            candidates = torch.clamp(candidates, self.action_low.unsqueeze(0), self.action_high.unsqueeze(0))
            costs = _evaluate_costs(cost_fn, candidates)

            elite_costs, elite_idx = torch.topk(costs, k=self.num_elites, largest=False)
            elites = candidates[elite_idx]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(self.min_std)

            old_mean = self.mean
            old_std = self.std
            self.mean = self.momentum * old_mean + (1.0 - self.momentum) * elite_mean
            self.std = (self.momentum * old_std + (1.0 - self.momentum) * elite_std).clamp_min(self.min_std)
            self.mean = torch.clamp(self.mean, self.action_low, self.action_high)

            iter_best_cost = elite_costs[0]
            iter_best_sequence = candidates[elite_idx[0]]
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_sequence = iter_best_sequence.detach().clone()
            best_cost_per_iter.append(float(iter_best_cost.detach().cpu()))
            elite_cost_mean_per_iter.append(float(elite_costs.mean().detach().cpu()))

            keep = min(self.gd_num_sequences, elite_idx.numel())
            top_sequences.extend(candidates[elite_idx[:keep]].detach().unbind(dim=0))
            top_costs.extend(elite_costs[:keep].detach().unbind(dim=0))

        if best_sequence is None:
            raise RuntimeError("CEM-GD did not evaluate any candidates")

        gd_diagnostics: dict[str, object] = {
            "enabled": self.gd_iterations > 0,
            "num_sequences": 0,
            "cost_before": None,
            "best_cost_per_iteration": [],
        }
        if self.gd_iterations > 0 and top_sequences:
            starts = self._select_gradient_starts(top_sequences, top_costs)
            refined, gd_diagnostics = self._gradient_refine(cost_fn, starts)
            refined_costs = _evaluate_costs(cost_fn, refined)
            refined_best_cost, refined_idx = torch.min(refined_costs, dim=0)
            if refined_best_cost < best_cost:
                best_cost = refined_best_cost.detach()
                best_sequence = refined[refined_idx].detach().clone()

        diagnostics = {
            "planner_type": "cem_gd",
            "best_cost_per_iteration": best_cost_per_iter,
            "elite_cost_mean_per_iteration": elite_cost_mean_per_iter,
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "best_cost": float(best_cost.detach().cpu()),
            "gradient_descent": gd_diagnostics,
        }
        return CEMResult(best_action_sequence=best_sequence, best_cost=best_cost, diagnostics=diagnostics)

    def _select_gradient_starts(self, top_sequences: list[torch.Tensor], top_costs: list[torch.Tensor]) -> torch.Tensor:
        costs = torch.stack(top_costs)
        keep = min(self.gd_num_sequences, costs.numel())
        _, idx = torch.topk(costs, k=keep, largest=False)
        return torch.stack([top_sequences[int(i)] for i in idx], dim=0).to(self.device)

    def _gradient_refine(self, cost_fn: CostFn, starts: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        actions = starts.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([actions], lr=self.gd_lr)

        initial_costs = _evaluate_costs(cost_fn, actions.detach())
        best_costs = initial_costs.detach().clone()
        best_actions = actions.detach().clone()
        best_cost_per_iter: list[float] = [float(best_costs.min().cpu())]

        for _ in range(self.gd_iterations):
            optimizer.zero_grad(set_to_none=True)
            costs = _evaluate_costs(cost_fn, actions)
            loss = costs.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                actions.clamp_(self.action_low.unsqueeze(0), self.action_high.unsqueeze(0))

            current_costs = _evaluate_costs(cost_fn, actions.detach())
            improved = current_costs.detach() < best_costs
            if improved.any():
                best_costs[improved] = current_costs.detach()[improved]
                best_actions[improved] = actions.detach()[improved]
            best_cost_per_iter.append(float(best_costs.min().cpu()))

        return best_actions, {
            "enabled": True,
            "num_sequences": int(starts.shape[0]),
            "iterations": self.gd_iterations,
            "lr": self.gd_lr,
            "cost_before": float(initial_costs.detach().min().cpu()),
            "best_cost_per_iteration": best_cost_per_iter,
            "cost_after": float(best_costs.min().cpu()),
        }


def _as_action_tensor(value: np.ndarray | torch.Tensor | float, horizon: int, action_dim: int, device: torch.device) -> torch.Tensor:
    """Broadcast scalar/per-action bounds to [horizon, action_dim]."""

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.expand(horizon, action_dim)
    elif tensor.shape == (action_dim,):
        tensor = tensor.unsqueeze(0).expand(horizon, action_dim)
    elif tensor.shape != (horizon, action_dim):
        raise ValueError(f"Expected scalar, {(action_dim,)}, or {(horizon, action_dim)}, got {tuple(tensor.shape)}")
    return tensor.clone()


def _evaluate_costs(cost_fn: CostFn, actions: torch.Tensor) -> torch.Tensor:
    costs = cost_fn(actions)
    expected_shape = (actions.shape[0],)
    if costs.shape != expected_shape:
        raise ValueError(f"cost_fn must return shape {expected_shape}, got {tuple(costs.shape)}")
    if not torch.isfinite(costs).all():
        raise ValueError("cost_fn returned NaN or Inf")
    return costs
