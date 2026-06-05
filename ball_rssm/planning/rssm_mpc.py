"""Receding-horizon MPC controller using CEM over RSSM prior rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from gymnasium import spaces

from ball_rssm.models import Normalizer, WorldModel, WorldModelConfig
from ball_rssm.models.rssm import RSSMState, repeat_state
from ball_rssm.planning.cem import CEMGDPlanner, CEMPlanner
from ball_rssm.planning.costs import (
    CostWeights,
    center_stabilization_cost,
    learned_reward_cost,
    point_stabilization_cost,
    timed_viapoint_cost,
)
from ball_rssm.utils.checkpoint import load_checkpoint

CostMode = Literal["center", "point", "timed_viapoint"]
PlanningObjective = Literal["state_cost", "reward", "hybrid"]
PlannerType = Literal["cem", "cem_gd"]


@dataclass
class MPCDiagnostics:
    best_cost: float
    best_action_sequence: np.ndarray
    predicted_obs: np.ndarray
    predicted_reward: np.ndarray
    predicted_continue: np.ndarray
    cem: dict[str, object]


class RSSMMPCController:
    """Receding-horizon controller that plans with CEM over RSSM prior rollouts."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        action_space: spaces.Box,
        horizon: int = 25,
        num_candidates: int = 1024,
        num_elites: int = 100,
        num_iterations: int = 4,
        device: torch.device | str = "cpu",
        cost_mode: CostMode = "center",
        planning_objective: PlanningObjective = "state_cost",
        planner_type: str = "cem",
        gd_num_sequences: int = 3,
        gd_iterations: int = 15,
        gd_lr: float = 0.01,
        target_xy: tuple[float, float] | None = None,
        via_step: int | None = None,
        stochastic: bool = False,
        seed: int | None = None,
        weights: CostWeights | None = None,
    ) -> None:
        if not isinstance(action_space, spaces.Box):
            raise TypeError("action_space must be gymnasium.spaces.Box")
        self.device = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        # from_dict ignores obsolete config keys so continuation-mode checkpoints
        # remain loadable after reward-model cleanup.
        self.model = WorldModel(WorldModelConfig.from_dict(checkpoint["config"])).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(self.device)

        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.horizon = horizon
        self.cost_mode = cost_mode
        self.planning_objective = planning_objective
        self.planner_type = _normalize_planner_type(planner_type)
        self.target_xy = target_xy
        self.via_step = via_step if via_step is not None else max(0, horizon - 1)
        self.stochastic = stochastic
        self.weights = weights or CostWeights()
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.state: RSSMState | None = None
        self.needs_update = False
        self.last_diagnostics: MPCDiagnostics | None = None

        action_low = np.asarray(action_space.low, dtype=np.float32).reshape(self.action_dim)
        action_high = np.asarray(action_space.high, dtype=np.float32).reshape(self.action_dim)
        planner_cls = CEMGDPlanner if self.planner_type == "cem_gd" else CEMPlanner
        planner_kwargs: dict[str, object] = {}
        if self.planner_type == "cem_gd":
            planner_kwargs = {
                "gd_num_sequences": gd_num_sequences,
                "gd_iterations": gd_iterations,
                "gd_lr": gd_lr,
            }
        self.planner = planner_cls(
            action_dim=self.action_dim,
            horizon=horizon,
            num_candidates=num_candidates,
            num_elites=num_elites,
            num_iterations=num_iterations,
            action_low=action_low,
            action_high=action_high,
            device=self.device,
            seed=seed,
            **planner_kwargs,
        )

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        """Reset latent belief, CEM distribution, and optional initial observation."""

        self.state = self.model.initial_state(1, self.device)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.planner.reset_distribution()
        self.needs_update = False
        self.last_diagnostics = None
        if initial_obs is not None:
            # Initialize the belief with the first real observation before planning.
            self.state = self._posterior_update(initial_obs, self.prev_action)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Plan from the current observation and return the first MPC action."""

        if self.state is None:
            self.reset(initial_obs=obs)
        elif self.needs_update:
            # Incorporate the observation produced by the previously executed action.
            self.state = self._posterior_update(obs, self.prev_action)

        assert self.state is not None

        def cost_fn(candidate_actions_real: torch.Tensor) -> torch.Tensor:
            # CEM proposes real action units; the model path normalizes them internally.
            pred_obs, pred_reward, pred_continue = self._predict_for_candidates(candidate_actions_real)
            return self._cost(pred_obs, pred_reward, pred_continue, candidate_actions_real)

        result = self.planner.plan(cost_fn)
        best_sequence = result.best_action_sequence.detach()
        best_action = best_sequence[0].detach().cpu().numpy().astype(np.float32)
        best_action = np.clip(best_action, self.action_space.low, self.action_space.high).astype(np.float32)
        pred_obs_t, pred_reward_t, pred_continue_t = self._predict_for_candidates(best_sequence.unsqueeze(0))
        pred_obs = pred_obs_t[0].detach().cpu().numpy()
        pred_reward = pred_reward_t[0].detach().cpu().numpy()
        pred_continue = pred_continue_t[0].detach().cpu().numpy()

        # Execute only the first action, then shift the plan for the next MPC step.
        self.planner.shift_mean(best_sequence)
        self.prev_action = best_action.reshape(self.action_dim)
        self.needs_update = True
        self.last_diagnostics = MPCDiagnostics(
            best_cost=float(result.best_cost.detach().cpu()),
            best_action_sequence=best_sequence.detach().cpu().numpy(),
            predicted_obs=pred_obs,
            predicted_reward=pred_reward,
            predicted_continue=pred_continue,
            cem=result.diagnostics,
        )
        return best_action.reshape(self.action_space.shape)

    def _posterior_update(self, obs: np.ndarray, prev_action: np.ndarray) -> RSSMState:
        """Normalize one observation/action pair and update posterior belief."""

        assert self.state is not None
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device).reshape(1, -1)
        obs_norm = self.normalizer.normalize_obs(obs_t)
        action_norm = self.normalizer.normalize_action(action_t)
        with torch.no_grad():
            return self.model.posterior_update(self.state, action_norm, obs_norm)

    def _predict_obs_for_candidates(self, candidate_actions_real: torch.Tensor) -> torch.Tensor:
        """Decode imagined observations for candidate real-unit action sequences."""

        pred_obs, _, _ = self._predict_for_candidates(candidate_actions_real)
        return pred_obs

    def _predict_for_candidates(
        self,
        candidate_actions_real: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine candidate action rollouts and return denormalized predictions."""

        assert self.state is not None
        num_candidates = candidate_actions_real.shape[0]
        grad_context = torch.enable_grad() if candidate_actions_real.requires_grad else torch.no_grad()
        with grad_context:
            action_norm = self.normalizer.normalize_action(candidate_actions_real)
            start_state = repeat_state(self.state, num_candidates)
            # Candidate rollouts use prior imagination only; future observations
            # are unavailable during planning.
            imagined = self.model.imagine_rollout(start_state, action_norm, deterministic=not self.stochastic)
            prior = imagined["prior"]
            assert isinstance(prior, dict)
            obs_norm_pred = self.model.decode_state_sequence(prior)
            reward_norm_pred = self.model.predict_reward_sequence(prior)
            continue_pred = self.model.predict_continuation_sequence(prior)
            return (
                self.normalizer.denormalize_obs(obs_norm_pred),
                self.normalizer.denormalize_reward(reward_norm_pred),
                continue_pred,
            )

    def _cost(
        self,
        pred_obs: torch.Tensor,
        pred_reward: torch.Tensor,
        pred_continue: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the selected planning objective for candidate rollouts."""

        if self.planning_objective == "reward":
            # Learned-reward planning uses continuation to discount impossible
            # future rewards after predicted terminal states.
            return learned_reward_cost(pred_reward, pred_obs, actions, pred_continue, self.weights)

        if self.cost_mode == "center":
            state_cost = center_stabilization_cost(pred_obs, actions, self.weights)
        elif self.cost_mode == "point":
            if self.target_xy is None:
                raise ValueError("target_xy is required for point cost")
            state_cost = point_stabilization_cost(pred_obs, actions, self.target_xy, self.weights)
        elif self.cost_mode == "timed_viapoint":
            if self.target_xy is None:
                raise ValueError("target_xy is required for timed_viapoint cost")
            state_cost = timed_viapoint_cost(pred_obs, actions, self.target_xy, self.via_step, self.weights)
        else:
            raise ValueError(f"Unsupported cost_mode={self.cost_mode!r}")

        if self.planning_objective == "state_cost":
            return state_cost
        if self.planning_objective == "hybrid":
            return state_cost + learned_reward_cost(pred_reward, pred_obs, actions, pred_continue, self.weights)
        raise ValueError(f"Unsupported planning_objective={self.planning_objective!r}")

    def diagnostics_dict(self) -> dict[str, Any]:
        """Return diagnostics from the most recent `act` call."""

        if self.last_diagnostics is None:
            return {}
        return {
            "best_cost": self.last_diagnostics.best_cost,
            "best_action_sequence": self.last_diagnostics.best_action_sequence,
            "predicted_obs": self.last_diagnostics.predicted_obs,
            "predicted_reward": self.last_diagnostics.predicted_reward,
            "predicted_continue": self.last_diagnostics.predicted_continue,
            "cem": self.last_diagnostics.cem,
        }


def _normalize_planner_type(planner_type: str) -> PlannerType:
    normalized = planner_type.lower().replace("-", "_")
    if normalized == "cem":
        return "cem"
    if normalized in {"cem_gd", "cemgd"}:
        return "cem_gd"
    raise ValueError(f"Unsupported planner_type={planner_type!r}")
