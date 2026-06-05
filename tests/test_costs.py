from __future__ import annotations

import torch

from ball_rssm.planning.costs import (
    center_stabilization_cost,
    continuation_discounted_return,
    learned_reward_cost,
    point_stabilization_cost,
    timed_viapoint_cost,
)


def test_center_cost_shape_and_position_ordering() -> None:
    actions = torch.zeros(2, 4, 2)
    pred_obs = torch.zeros(2, 4, 6)
    pred_obs[1, :, 0] = 0.3
    cost = center_stabilization_cost(pred_obs, actions)
    assert cost.shape == (2,)
    assert cost[0] < cost[1]


def test_center_cost_boundary_penalty() -> None:
    actions = torch.zeros(2, 4, 2)
    pred_obs = torch.zeros(2, 4, 6)
    pred_obs[1, :, 0] = 0.6
    cost = center_stabilization_cost(pred_obs, actions)
    assert cost[1] > cost[0] + 100.0


def test_point_cost_lower_near_target() -> None:
    actions = torch.zeros(2, 5, 2)
    pred_obs = torch.zeros(2, 5, 6)
    target = (0.2, -0.1)
    pred_obs[0, :, :2] = torch.tensor(target)
    pred_obs[1, :, :2] = torch.tensor([-0.2, 0.1])
    cost = point_stabilization_cost(pred_obs, actions, target)
    assert cost[0] < cost[1]


def test_timed_viapoint_cost_lower_at_via_step() -> None:
    actions = torch.zeros(2, 6, 2)
    pred_obs = torch.zeros(2, 6, 6)
    target = (0.15, 0.1)
    pred_obs[0, 3, :2] = torch.tensor(target)
    pred_obs[1, 3, :2] = torch.tensor([-0.15, -0.1])
    cost = timed_viapoint_cost(pred_obs, actions, target_xy=target, via_step=3)
    assert cost[0] < cost[1]


def test_learned_reward_cost_prefers_higher_reward() -> None:
    actions = torch.zeros(2, 4, 2)
    pred_obs = torch.zeros(2, 4, 6)
    pred_reward = torch.zeros(2, 4, 1)
    pred_reward[0, :, 0] = 1.0
    pred_reward[1, :, 0] = -1.0

    cost = learned_reward_cost(pred_reward, pred_obs, actions)

    assert cost.shape == (2,)
    assert cost[0] < cost[1]


def test_continuation_discounted_return_stops_after_terminal() -> None:
    pred_reward = torch.ones(2, 4, 1)
    pred_continue = torch.ones(2, 4, 1)
    pred_continue[1, 1:, 0] = 0.0

    returns = continuation_discounted_return(pred_reward, pred_continue, discount=1.0)

    assert returns.shape == (2,)
    assert returns[0] == 4.0
    assert returns[1] == 2.0
