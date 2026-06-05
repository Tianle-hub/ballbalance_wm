"""Control costs computed on denormalized decoded observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CostWeights:
    w_pos: float = 20.0
    w_vel: float = 1.0
    w_angle: float = 0.1
    w_action: float = 0.01
    w_smooth: float = 0.1
    w_terminal: float = 50.0
    w_boundary: float = 1000.0
    board_size: float = 1.0
    w_via: float = 100.0
    w_running: float = 2.0
    w_reward: float = 1.0
    discount: float = 0.99


def center_stabilization_cost(
    pred_obs: torch.Tensor,
    actions: torch.Tensor,
    weights: CostWeights | None = None,
) -> torch.Tensor:
    """Cost for stabilizing at board center."""

    return point_stabilization_cost(pred_obs, actions, target_xy=None, weights=weights)


def point_stabilization_cost(
    pred_obs: torch.Tensor,
    actions: torch.Tensor,
    target_xy: torch.Tensor | tuple[float, float] | None,
    weights: CostWeights | None = None,
) -> torch.Tensor:
    """Quadratic stabilization cost toward a fixed xy target."""

    weights = weights or CostWeights()
    target = _target_tensor(target_xy, pred_obs)
    # State costs operate on denormalized observations: [x, y, vx, vy, theta_x, theta_y].
    pos_err = pred_obs[..., :2] - target
    cost = weights.w_pos * (pos_err**2).sum(dim=-1)
    cost = cost + weights.w_vel * (pred_obs[..., 2:4] ** 2).sum(dim=-1)
    cost = cost + weights.w_angle * (pred_obs[..., 4:6] ** 2).sum(dim=-1)
    cost = cost + weights.w_action * (actions**2).sum(dim=-1)
    cost = cost + _smoothness_cost(actions, weights)
    cost = cost + _boundary_cost(pred_obs, weights)
    total = cost.sum(dim=-1)
    # The terminal term biases finite-horizon plans toward ending near the target,
    # not merely passing through it early.
    terminal_err = (pred_obs[:, -1, :2] - target.squeeze(0).squeeze(0)) ** 2
    return total + weights.w_terminal * terminal_err.sum(dim=-1)


def timed_viapoint_cost(
    pred_obs: torch.Tensor,
    actions: torch.Tensor,
    target_xy: torch.Tensor | tuple[float, float],
    via_step: int,
    weights: CostWeights | None = None,
) -> torch.Tensor:
    """Cost for reaching a target at a specified planning timestep."""

    weights = weights or CostWeights()
    via_step = int(max(0, min(via_step, pred_obs.shape[1] - 1)))
    target = _target_tensor(target_xy, pred_obs)
    # Via-point mode still carries a small running target cost, but emphasizes
    # the requested time index.
    pos_err = pred_obs[..., :2] - target
    running = weights.w_running * (pos_err**2).sum(dim=-1).sum(dim=-1)
    via_err = (pred_obs[:, via_step, :2] - target.squeeze(0).squeeze(0)) ** 2
    cost = running + weights.w_via * via_err.sum(dim=-1)
    cost = cost + weights.w_vel * (pred_obs[..., 2:4] ** 2).sum(dim=-1).sum(dim=-1)
    cost = cost + weights.w_action * (actions**2).sum(dim=-1).sum(dim=-1)
    cost = cost + _smoothness_cost(actions, weights).sum(dim=-1)
    cost = cost + _boundary_cost(pred_obs, weights).sum(dim=-1)
    return cost


def target_trajectory_cost(
    pred_obs: torch.Tensor,
    actions: torch.Tensor,
    target_xy_seq: torch.Tensor,
    weights: CostWeights | None = None,
) -> torch.Tensor:
    """Track a full target xy trajectory over the planning horizon."""

    weights = weights or CostWeights()
    target_xy_seq = target_xy_seq.to(pred_obs.device, dtype=pred_obs.dtype)
    pos_err = pred_obs[..., :2] - target_xy_seq.unsqueeze(0)
    cost = weights.w_pos * (pos_err**2).sum(dim=-1)
    cost = cost + weights.w_vel * (pred_obs[..., 2:4] ** 2).sum(dim=-1)
    cost = cost + weights.w_action * (actions**2).sum(dim=-1)
    cost = cost + _smoothness_cost(actions, weights)
    cost = cost + _boundary_cost(pred_obs, weights)
    return cost.sum(dim=-1)


def learned_reward_cost(
    pred_reward: torch.Tensor,
    pred_obs: torch.Tensor,
    actions: torch.Tensor,
    pred_continue: torch.Tensor | None = None,
    weights: CostWeights | None = None,
) -> torch.Tensor:
    """Convert predicted environment reward into a minimization objective."""

    weights = weights or CostWeights()
    # Learned reward is maximized by minimizing its negative; smoothness and
    # boundary penalties keep the optimizer from exploiting model artifacts.
    reward_return = continuation_discounted_return(pred_reward, pred_continue, weights.discount)
    smoothness = _smoothness_cost(actions, weights).sum(dim=-1)
    boundary = _boundary_cost(pred_obs, weights).sum(dim=-1)
    return -weights.w_reward * reward_return + smoothness + boundary


def continuation_discounted_return(
    pred_reward: torch.Tensor,
    pred_continue: torch.Tensor | None,
    discount: float,
) -> torch.Tensor:
    """Sum rewards with discounting gated by predicted continuation."""

    reward = pred_reward.squeeze(-1)
    _, horizon = reward.shape
    if pred_continue is None:
        step_weights = torch.ones_like(reward)
    else:
        continuation = pred_continue.squeeze(-1).clamp(0.0, 1.0)
        step_weights = torch.ones_like(reward)
        if horizon > 1:
            # Weight reward at step k by prod_{i<k} gamma * c_i, so return stops
            # accumulating after predicted terminal states.
            step_weights[:, 1:] = torch.cumprod(discount * continuation[:, :-1], dim=1)
    return (step_weights * reward).sum(dim=-1)


def _target_tensor(target_xy: torch.Tensor | tuple[float, float] | None, pred_obs: torch.Tensor) -> torch.Tensor:
    """Convert target xy to a broadcastable [1, 1, 2] tensor."""

    if target_xy is None:
        target = torch.zeros(2, device=pred_obs.device, dtype=pred_obs.dtype)
    else:
        target = torch.as_tensor(target_xy, device=pred_obs.device, dtype=pred_obs.dtype)
    return target.reshape(1, 1, 2)


def _smoothness_cost(actions: torch.Tensor, weights: CostWeights) -> torch.Tensor:
    """Penalize action magnitude at the first step and changes afterward."""

    # Penalize both the first command magnitude and command-to-command changes.
    first = actions[:, :1]
    diffs = torch.cat([first, actions[:, 1:] - actions[:, :-1]], dim=1)
    return weights.w_smooth * (diffs**2).sum(dim=-1)


def _boundary_cost(pred_obs: torch.Tensor, weights: CostWeights) -> torch.Tensor:
    """Apply a large penalty to predicted states outside the board."""

    half = weights.board_size / 2.0
    outside = (pred_obs[..., 0].abs() > half) | (pred_obs[..., 1].abs() > half)
    return weights.w_boundary * outside.to(pred_obs.dtype)
