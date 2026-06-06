"""Compare center-stabilization controllers from identical initial states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceConfig, BallBalanceEnv
from ball_rssm.planning.cem import CEMGDPlanner, CEMPlanner
from ball_rssm.planning.costs import CostWeights, center_stabilization_cost
from ball_rssm.planning.rollout import (
    InitialConditionBounds,
    aggregate_metrics,
    episode_metrics,
    pd_action,
    sample_initial_state,
    save_episode_npz,
)
from ball_rssm.planning.rssm_mpc import RSSMMPCController

MethodName = str


class Controller(Protocol):
    horizon: int

    def reset(self, initial_obs: np.ndarray | None = None) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...

    def diagnostics_dict(self) -> dict[str, Any]: ...


@dataclass
class PDController:
    action_low: np.ndarray
    action_high: np.ndarray
    kp: float = 0.8
    kd: float = 0.25
    horizon: int = 0

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        del initial_obs

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = pd_action(obs, target_xy=(0.0, 0.0), kp=self.kp, kd=self.kd)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def diagnostics_dict(self) -> dict[str, Any]:
        return {}


class LQRController:
    def __init__(
        self,
        config: BallBalanceConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        q_pos: float = 20.0,
        q_vel: float = 2.0,
        q_angle: float = 0.2,
        r_action: float = 0.05,
    ) -> None:
        self.action_low = action_low
        self.action_high = action_high
        self.horizon = 0
        q = np.diag([q_pos, q_pos, q_vel, q_vel, q_angle, q_angle]).astype(np.float64)
        r = np.diag([r_action, r_action]).astype(np.float64)
        a, b = linearized_dynamics_matrices(config)
        self.k = discrete_lqr_gain(a, b, q, r)

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        del initial_obs

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = -self.k @ np.asarray(obs, dtype=np.float64)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def diagnostics_dict(self) -> dict[str, Any]:
        return {}


class AnalyticMPCController:
    """Full-observable CEM-MPC using the environment equations as the prediction model."""

    def __init__(
        self,
        config: BallBalanceConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        num_candidates: int,
        num_elites: int,
        num_iterations: int,
        planner_type: str,
        gd_num_sequences: int,
        gd_iterations: int,
        gd_lr: float,
        device: torch.device | str,
        seed: int,
        weights: CostWeights | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
        self.horizon = horizon
        self.weights = weights or CostWeights(board_size=config.board_size)
        self.last_diagnostics: dict[str, Any] = {}

        planner_cls = CEMGDPlanner if normalize_planner_type(planner_type) == "cem_gd" else CEMPlanner
        planner_kwargs: dict[str, object] = {}
        if planner_cls is CEMGDPlanner:
            planner_kwargs = {
                "gd_num_sequences": gd_num_sequences,
                "gd_iterations": gd_iterations,
                "gd_lr": gd_lr,
            }
        self.planner = planner_cls(
            action_dim=2,
            horizon=horizon,
            num_candidates=num_candidates,
            num_elites=num_elites,
            num_iterations=num_iterations,
            action_low=action_low.astype(np.float32),
            action_high=action_high.astype(np.float32),
            device=self.device,
            seed=seed,
            **planner_kwargs,
        )

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        del initial_obs
        self.planner.reset_distribution()
        self.last_diagnostics = {}

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        def cost_fn(actions: torch.Tensor) -> torch.Tensor:
            pred_obs = rollout_true_dynamics(state, actions, self.config)
            return center_stabilization_cost(pred_obs, actions, self.weights)

        result = self.planner.plan(cost_fn)
        best_sequence = result.best_action_sequence.detach()
        self.planner.shift_mean(best_sequence)
        pred_obs = rollout_true_dynamics(state, best_sequence.unsqueeze(0), self.config)[0].detach().cpu().numpy()
        self.last_diagnostics = {
            "best_cost": float(result.best_cost.detach().cpu()),
            "best_action_sequence": best_sequence.detach().cpu().numpy(),
            "predicted_obs": pred_obs,
            "cem": result.diagnostics,
        }
        return best_sequence[0].detach().cpu().numpy().astype(np.float32)

    def diagnostics_dict(self) -> dict[str, Any]:
        return self.last_diagnostics


class MPPIMPCController:
    """Full-observable MPPI controller using the true differentiable dynamics."""

    def __init__(
        self,
        config: BallBalanceConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        num_candidates: int,
        num_iterations: int,
        temperature: float,
        noise_std: float,
        momentum: float,
        device: torch.device | str,
        seed: int,
        weights: CostWeights | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if temperature <= 0.0:
            raise ValueError("mppi temperature must be positive")
        if noise_std <= 0.0:
            raise ValueError("mppi noise std must be positive")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("mppi momentum must be in [0, 1]")

        self.config = config
        self.device = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_std = noise_std
        self.momentum = momentum
        self.weights = weights or CostWeights(board_size=config.board_size)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device).view(1, 1, 2)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device).view(1, 1, 2)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.mean = torch.zeros(self.horizon, 2, dtype=torch.float32, device=self.device)
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        del initial_obs
        self.mean.zero_()
        self.last_diagnostics = {}

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        best_sequence = self.mean.detach().clone()
        best_cost = torch.tensor(float("inf"), device=self.device)
        best_cost_per_iter: list[float] = []
        weighted_cost_per_iter: list[float] = []

        for _ in range(self.num_iterations):
            noise = torch.randn(
                self.num_candidates,
                self.horizon,
                2,
                generator=self.generator,
                device=self.device,
            ) * self.noise_std
            candidates = self.mean.unsqueeze(0) + noise
            candidates = torch.clamp(candidates, self.action_low, self.action_high)
            costs = true_dynamics_center_cost(state, candidates, self.config, self.weights)

            iter_best_cost, iter_best_idx = torch.min(costs, dim=0)
            if iter_best_cost < best_cost:
                best_cost = iter_best_cost.detach()
                best_sequence = candidates[iter_best_idx].detach().clone()

            weights = torch.softmax(-(costs - costs.min()) / self.temperature, dim=0)
            updated_mean = torch.sum(weights.view(-1, 1, 1) * candidates, dim=0)
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * updated_mean
            self.mean = torch.clamp(self.mean, self.action_low.squeeze(0), self.action_high.squeeze(0))
            best_cost_per_iter.append(float(iter_best_cost.detach().cpu()))
            weighted_cost_per_iter.append(float(torch.sum(weights * costs).detach().cpu()))

        self.shift_mean(best_sequence)
        pred_obs = rollout_true_dynamics(state, best_sequence.unsqueeze(0), self.config)[0].detach().cpu().numpy()
        self.last_diagnostics = {
            "best_cost": float(best_cost.detach().cpu()),
            "best_action_sequence": best_sequence.detach().cpu().numpy(),
            "predicted_obs": pred_obs,
            "mppi": {
                "planner_type": "mppi",
                "temperature": self.temperature,
                "noise_std": self.noise_std,
                "best_cost_per_iteration": best_cost_per_iter,
                "weighted_cost_per_iteration": weighted_cost_per_iter,
            },
        }
        return best_sequence[0].detach().cpu().numpy().astype(np.float32)

    def shift_mean(self, previous_sequence: torch.Tensor) -> None:
        self.mean[:-1] = previous_sequence[1:]
        self.mean[-1] = previous_sequence[-1]
        self.mean = torch.clamp(self.mean, self.action_low.squeeze(0), self.action_high.squeeze(0))

    def diagnostics_dict(self) -> dict[str, Any]:
        return self.last_diagnostics


class DirectGradientMPCController:
    """Full-observable direct-shooting gradient MPC over true dynamics.

    This is the optimization-based baseline closest to iLQR/DDP in this script:
    it optimizes the finite-horizon action sequence with gradients through the
    differentiable dynamics, then executes the first action.
    """

    def __init__(
        self,
        config: BallBalanceConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        iterations: int,
        lr: float,
        device: torch.device | str,
        weights: CostWeights | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if iterations <= 0:
            raise ValueError("gradient MPC iterations must be positive")
        if lr <= 0.0:
            raise ValueError("gradient MPC lr must be positive")

        self.config = config
        self.device = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
        self.horizon = horizon
        self.iterations = iterations
        self.lr = lr
        self.weights = weights or CostWeights(board_size=config.board_size)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device).view(1, 1, 2).expand(1, self.horizon, 2).clone()
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device).view(1, 1, 2).expand(1, self.horizon, 2).clone()
        self.mean = torch.zeros(1, self.horizon, 2, dtype=torch.float32, device=self.device)
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self, initial_obs: np.ndarray | None = None) -> None:
        del initial_obs
        self.mean.zero_()
        self.last_diagnostics = {}

    def act(self, obs: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = self.mean.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([actions], lr=self.lr)

        best_sequence = actions.detach()[0].clone()
        best_cost = true_dynamics_center_cost(state, actions.detach(), self.config, self.weights)[0].detach()
        best_cost_per_iter: list[float] = [float(best_cost.cpu())]

        for _ in range(self.iterations):
            optimizer.zero_grad(set_to_none=True)
            cost = true_dynamics_center_cost(state, actions, self.config, self.weights)[0]
            cost.backward()
            optimizer.step()
            with torch.no_grad():
                actions.clamp_(self.action_low, self.action_high)
                current_cost = true_dynamics_center_cost(state, actions, self.config, self.weights)[0].detach()
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sequence = actions.detach()[0].clone()
                best_cost_per_iter.append(float(best_cost.cpu()))

        self.shift_mean(best_sequence)
        pred_obs = rollout_true_dynamics(state, best_sequence.unsqueeze(0), self.config)[0].detach().cpu().numpy()
        self.last_diagnostics = {
            "best_cost": float(best_cost.cpu()),
            "best_action_sequence": best_sequence.detach().cpu().numpy(),
            "predicted_obs": pred_obs,
            "direct_gradient_mpc": {
                "planner_type": "direct_shooting_gradient",
                "iterations": self.iterations,
                "lr": self.lr,
                "best_cost_per_iteration": best_cost_per_iter,
            },
        }
        return best_sequence[0].detach().cpu().numpy().astype(np.float32)

    def shift_mean(self, previous_sequence: torch.Tensor) -> None:
        self.mean[0, :-1] = previous_sequence[1:]
        self.mean[0, -1] = previous_sequence[-1]
        self.mean = torch.clamp(self.mean, self.action_low, self.action_high)

    def diagnostics_dict(self) -> dict[str, Any]:
        return self.last_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--methods", default="pd,lqr,mpc_cem,mpc_mppi,mpc_gd,rssm_cem,rssm_cem_gd")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--planning-objective", choices=["state_cost", "reward", "hybrid"], default="reward")
    parser.add_argument("--gd-num-sequences", type=int, default=3)
    parser.add_argument("--gd-iterations", type=int, default=15)
    parser.add_argument("--gd-lr", type=float, default=0.01)
    parser.add_argument("--mppi-temperature", type=float, default=1.0)
    parser.add_argument("--mppi-noise-std", type=float, default=0.15)
    parser.add_argument("--mppi-momentum", type=float, default=0.0)
    parser.add_argument("--pd-kp", type=float, default=0.8)
    parser.add_argument("--pd-kd", type=float, default=0.25)
    parser.add_argument("--lqr-q-pos", type=float, default=20.0)
    parser.add_argument("--lqr-q-vel", type=float, default=2.0)
    parser.add_argument("--lqr-q-angle", type=float, default=0.2)
    parser.add_argument("--lqr-r-action", type=float, default=0.05)
    parser.add_argument("--pos-bound", type=float, default=0.25)
    parser.add_argument("--vel-bound", type=float, default=0.10)
    parser.add_argument("--angle-bound", type=float, default=0.05)
    parser.add_argument("--steady-window", type=int, default=50, help="Number of final observations used for steady-state error.")
    parser.add_argument("--settling-threshold", type=float, default=0.05, help="Distance band for settling/stabilization time.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--save-episodes", action="store_true")
    parser.add_argument("--no-summary-plots", dest="summary_plots", action="store_false", help="Disable aggregate comparison plots.")
    parser.set_defaults(summary_plots=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    if any(method.startswith("rssm_") for method in methods) and args.checkpoint is None:
        parser.error("--checkpoint is required when methods include rssm_cem or rssm_cem_gd")

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    bounds = InitialConditionBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    initial_states = np.stack([sample_initial_state(rng, bounds) for _ in range(args.num_episodes)], axis=0)
    np.savez_compressed(out_dir / "initial_states.npz", initial_states=initial_states)

    env = BallBalanceEnv(config={"max_episode_steps": args.max_steps})
    try:
        controllers = build_controllers(args, methods, env)
        summary: dict[str, Any] = {"config": vars(args), "methods": {}}
        for method_name, controller in controllers.items():
            metrics: list[dict[str, float | bool]] = []
            method_dir = out_dir / method_name
            if args.save_episodes:
                method_dir.mkdir(parents=True, exist_ok=True)

            for episode_idx, initial_state in enumerate(initial_states):
                episode = run_controller_episode(controller, env, initial_state, args.max_steps)
                episode_metric = episode_metrics(episode, (0.0, 0.0), final_threshold=0.05, last_window_threshold=0.07)
                episode_metric.update(
                    control_response_metrics(
                        episode,
                        target_xy=(0.0, 0.0),
                        dt=env.config.dt,
                        steady_window=args.steady_window,
                        settling_threshold=args.settling_threshold,
                    )
                )
                episode_metric.update(computation_budget_metrics(episode, dt=env.config.dt))
                metrics.append(episode_metric)
                if args.save_episodes:
                    save_episode_npz(episode, method_dir / f"episode_{episode_idx:03d}.npz")
                print(
                    f"method={method_name} episode={episode_idx:03d} "
                    f"final={episode_metric['final_distance']:.4f} "
                    f"steady={episode_metric['steady_state_error']:.4f} "
                    f"settling={episode_metric['settling_time_sec']:.2f}s "
                    f"success={episode_metric['success']} fell={episode_metric['fell']}"
                )

            summary["methods"][method_name] = {
                "episodes": metrics,
                "aggregate": aggregate_metrics(metrics),
            }
    finally:
        env.close()

    (out_dir / "comparison_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_aggregate_csv(summary, out_dir / "comparison_aggregate.csv")
    if args.summary_plots:
        plot_control_metric_summary(summary, out_dir / "control_metric_summary.png")
        plot_steady_error_vs_settling_time(summary, out_dir / "steady_error_vs_settling_time.png")
        plot_computation_time_summary(summary, out_dir / "computation_time_summary.png")
    print(json.dumps({name: data["aggregate"] for name, data in summary["methods"].items()}, indent=2))
    print(f"wrote comparison outputs to {out_dir}")


def build_controllers(args: argparse.Namespace, methods: list[MethodName], env: BallBalanceEnv) -> dict[str, Controller]:
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    controllers: dict[str, Controller] = {}
    for method in methods:
        if method == "pd":
            controllers[method] = PDController(action_low=action_low, action_high=action_high, kp=args.pd_kp, kd=args.pd_kd)
        elif method == "lqr":
            controllers[method] = LQRController(
                config=env.config,
                action_low=action_low,
                action_high=action_high,
                q_pos=args.lqr_q_pos,
                q_vel=args.lqr_q_vel,
                q_angle=args.lqr_q_angle,
                r_action=args.lqr_r_action,
            )
        elif method == "analytic_mpc":
            controllers[method] = build_true_dynamics_cem_controller(
                args=args,
                env=env,
                action_low=action_low,
                action_high=action_high,
                planner_type="cem",
                seed=args.seed + 17,
            )
        elif method in {"mpc_cem", "mpc_cem_gd"}:
            controllers[method] = build_true_dynamics_cem_controller(
                args=args,
                env=env,
                action_low=action_low,
                action_high=action_high,
                planner_type="cem_gd" if method == "mpc_cem_gd" else "cem",
                seed=args.seed + (17 if method == "mpc_cem" else 23),
            )
        elif method == "mpc_mppi":
            controllers[method] = MPPIMPCController(
                config=env.config,
                action_low=action_low,
                action_high=action_high,
                horizon=args.horizon,
                num_candidates=args.num_candidates,
                num_iterations=args.num_iterations,
                temperature=args.mppi_temperature,
                noise_std=args.mppi_noise_std,
                momentum=args.mppi_momentum,
                device=args.device,
                seed=args.seed + 37,
            )
        elif method == "mpc_gd":
            controllers[method] = DirectGradientMPCController(
                config=env.config,
                action_low=action_low,
                action_high=action_high,
                horizon=args.horizon,
                iterations=args.gd_iterations,
                lr=args.gd_lr,
                device=args.device,
            )
        elif method in {"rssm_cem", "rssm_cem_gd"}:
            assert args.checkpoint is not None
            planner_type = "cem_gd" if method == "rssm_cem_gd" else "cem"
            controllers[method] = RSSMMPCController(
                checkpoint_path=args.checkpoint,
                action_space=env.action_space,
                horizon=args.horizon,
                num_candidates=args.num_candidates,
                num_elites=args.num_elites,
                num_iterations=args.num_iterations,
                device=args.device,
                cost_mode="center",
                planning_objective=args.planning_objective,
                planner_type=planner_type,
                gd_num_sequences=args.gd_num_sequences,
                gd_iterations=args.gd_iterations,
                gd_lr=args.gd_lr,
                seed=args.seed + (101 if method == "rssm_cem" else 202),
            )
        else:
            raise ValueError(f"unsupported method: {method}")
    return controllers


def build_true_dynamics_cem_controller(
    args: argparse.Namespace,
    env: BallBalanceEnv,
    action_low: np.ndarray,
    action_high: np.ndarray,
    planner_type: str,
    seed: int,
) -> AnalyticMPCController:
    return AnalyticMPCController(
        config=env.config,
        action_low=action_low,
        action_high=action_high,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        planner_type=planner_type,
        gd_num_sequences=args.gd_num_sequences,
        gd_iterations=args.gd_iterations,
        gd_lr=args.gd_lr,
        device=args.device,
        seed=seed,
    )


def run_controller_episode(
    controller: Controller,
    env: BallBalanceEnv,
    initial_state: np.ndarray,
    max_steps: int,
) -> dict[str, np.ndarray | float | bool]:
    obs, _ = env.reset(options={"state": initial_state})
    controller.reset(obs)
    observations = [obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    costs: list[float] = []
    predicted_obs: list[np.ndarray] = []
    computation_times: list[float] = []
    terminated = False
    truncated = False

    for _ in range(max_steps):
        start = time.perf_counter()
        action = controller.act(obs)
        computation_times.append(time.perf_counter() - start)
        diagnostics = controller.diagnostics_dict()
        if "best_cost" in diagnostics:
            costs.append(float(diagnostics["best_cost"]))
        if "predicted_obs" in diagnostics:
            predicted_obs.append(np.asarray(diagnostics["predicted_obs"], dtype=np.float32))

        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        if terminated or truncated:
            break

    obs_arr = np.asarray(observations, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32)
    reward_arr = np.asarray(rewards, dtype=np.float32)
    pred_arr = (
        np.asarray(predicted_obs, dtype=np.float32)
        if predicted_obs
        else np.zeros((0, getattr(controller, "horizon", 0), 6), dtype=np.float32)
    )
    return {
        "obs": obs_arr,
        "action": action_arr,
        "reward": reward_arr,
        "cost": np.asarray(costs, dtype=np.float32),
        "predicted_obs": pred_arr,
        "distance": np.linalg.norm(obs_arr[:, :2], axis=-1).astype(np.float32),
        "computation_time": np.asarray(computation_times, dtype=np.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "total_reward": float(reward_arr.sum()) if reward_arr.size else 0.0,
    }


def control_response_metrics(
    episode: dict[str, np.ndarray | float | bool],
    target_xy: tuple[float, float],
    dt: float,
    steady_window: int,
    settling_threshold: float,
) -> dict[str, float | bool]:
    """Compute standard control-response metrics for center stabilization.

    Settling time is the first time after which the position error remains
    inside the threshold band. If an episode never settles, the metric is set to
    the episode duration and `settled` is false.
    """

    if steady_window <= 0:
        raise ValueError("steady_window must be positive")
    if settling_threshold <= 0.0:
        raise ValueError("settling_threshold must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    obs = np.asarray(episode["obs"], dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    distance = np.linalg.norm(obs[:, :2] - target, axis=-1)
    tail = distance[-min(steady_window, distance.shape[0]) :]

    settled = False
    settling_step = max(distance.shape[0] - 1, 0)
    within_band = distance <= settling_threshold
    if not bool(episode["terminated"]):
        for step in range(distance.shape[0]):
            if bool(np.all(within_band[step:])):
                settled = True
                settling_step = step
                break

    return {
        "steady_state_error": float(tail.mean()),
        "steady_state_rmse": float(np.sqrt(np.mean(tail**2))),
        "steady_state_max_error": float(tail.max()),
        "settled": bool(settled),
        "settling_step": float(settling_step),
        "settling_time_sec": float(settling_step * dt),
    }


def computation_budget_metrics(
    episode: dict[str, np.ndarray | float | bool],
    dt: float,
) -> dict[str, float]:
    """Compute episode-level timing cost relative to simulated control time."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    compute_time = np.asarray(episode["computation_time"], dtype=np.float32)
    if not compute_time.size:
        return {
            "simulated_duration_sec": 0.0,
            "compute_real_time_factor": 0.0,
        }
    simulated_duration = float(compute_time.shape[0] * dt)
    total_compute = float(compute_time.sum())
    return {
        "simulated_duration_sec": simulated_duration,
        "compute_real_time_factor": total_compute / simulated_duration if simulated_duration > 0.0 else 0.0,
    }


def rollout_true_dynamics(initial_state: torch.Tensor, actions: torch.Tensor, config: BallBalanceConfig) -> torch.Tensor:
    """Vectorized differentiable rollout of the environment's visible dynamics."""

    state = initial_state.to(actions.device, dtype=actions.dtype).unsqueeze(0).expand(actions.shape[0], -1)
    states: list[torch.Tensor] = []
    action_low = torch.full((2,), -config.max_angle, dtype=actions.dtype, device=actions.device)
    action_high = torch.full((2,), config.max_angle, dtype=actions.dtype, device=actions.device)
    for t in range(actions.shape[1]):
        action = torch.clamp(actions[:, t], action_low, action_high)
        theta_x_cmd = action[:, 0]
        theta_y_cmd = action[:, 1]
        x, y, vx, vy, theta_x, theta_y = state.unbind(dim=-1)

        theta_x = theta_x + (config.dt / config.angle_tau) * (theta_x_cmd - theta_x)
        theta_y = theta_y + (config.dt / config.angle_tau) * (theta_y_cmd - theta_y)
        theta_x = torch.clamp(theta_x, -config.max_angle, config.max_angle)
        theta_y = torch.clamp(theta_y, -config.max_angle, config.max_angle)

        ax = (5.0 / 7.0) * config.g * torch.sin(theta_y) - config.damping * vx
        ay = -(5.0 / 7.0) * config.g * torch.sin(theta_x) - config.damping * vy
        vx = vx + ax * config.dt
        vy = vy + ay * config.dt
        x = x + vx * config.dt
        y = y + vy * config.dt
        state = torch.stack([x, y, vx, vy, theta_x, theta_y], dim=-1)
        states.append(state)
    return torch.stack(states, dim=1)


def true_dynamics_center_cost(
    state: torch.Tensor,
    actions: torch.Tensor,
    config: BallBalanceConfig,
    weights: CostWeights,
) -> torch.Tensor:
    pred_obs = rollout_true_dynamics(state, actions, config)
    return center_stabilization_cost(pred_obs, actions, weights)


def linearized_dynamics_matrices(config: BallBalanceConfig) -> tuple[np.ndarray, np.ndarray]:
    dt = float(config.dt)
    alpha = dt / float(config.angle_tau)
    beta = 1.0 - alpha
    vel_decay = 1.0 - float(config.damping) * dt
    accel_gain = (5.0 / 7.0) * float(config.g) * dt

    a = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros((6, 2), dtype=np.float64)

    a[0, 0] = 1.0
    a[0, 2] = dt * vel_decay
    a[0, 5] = dt * accel_gain * beta
    b[0, 1] = dt * accel_gain * alpha

    a[1, 1] = 1.0
    a[1, 3] = dt * vel_decay
    a[1, 4] = -dt * accel_gain * beta
    b[1, 0] = -dt * accel_gain * alpha

    a[2, 2] = vel_decay
    a[2, 5] = accel_gain * beta
    b[2, 1] = accel_gain * alpha

    a[3, 3] = vel_decay
    a[3, 4] = -accel_gain * beta
    b[3, 0] = -accel_gain * alpha

    a[4, 4] = beta
    b[4, 0] = alpha

    a[5, 5] = beta
    b[5, 1] = alpha
    return a, b


def discrete_lqr_gain(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    iterations: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    p = q.copy()
    for _ in range(iterations):
        bt_p = b.T @ p
        next_p = a.T @ p @ a - a.T @ p @ b @ np.linalg.solve(r + bt_p @ b, bt_p @ a) + q
        if np.max(np.abs(next_p - p)) < tol:
            p = next_p
            break
        p = next_p
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


def parse_methods(raw: str) -> list[MethodName]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {
        "pd",
        "lqr",
        "analytic_mpc",
        "mpc_cem",
        "mpc_cem_gd",
        "mpc_mppi",
        "mpc_gd",
        "rssm_cem",
        "rssm_cem_gd",
    }
    unknown = set(methods) - valid
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(sorted(unknown))}")
    return methods


def normalize_planner_type(planner_type: str) -> str:
    normalized = planner_type.lower().replace("-", "_")
    if normalized in {"cem", "cem_gd"}:
        return normalized
    raise ValueError(f"unsupported planner type: {planner_type}")


def default_out_dir(checkpoint: str | None) -> Path:
    if checkpoint is None:
        return Path("runs") / "center_baseline_comparison"
    return Path(checkpoint).resolve().parents[1] / "center_baseline_comparison"


def write_aggregate_csv(summary: dict[str, Any], path: Path) -> None:
    rows = [
        {"method": method, **data["aggregate"]}
        for method, data in summary["methods"].items()
    ]
    if not rows:
        return
    fieldnames = ["method"] + [key for key in rows[0] if key != "method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_control_metric_summary(summary: dict[str, Any], path: Path) -> None:
    ensure_writable_matplotlib_cache()
    import matplotlib.pyplot as plt

    methods, aggregates = method_aggregates(summary)
    steady = [aggregate["steady_state_error"] for aggregate in aggregates]
    settling = [aggregate["settling_time_sec"] for aggregate in aggregates]
    settled_rate = [aggregate["settled_rate"] for aggregate in aggregates]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(methods))
    axes[0].bar(x, steady, color="tab:blue")
    axes[0].set_title("Steady-state error")
    axes[0].set_ylabel("mean final-window distance [m]")

    axes[1].bar(x, settling, color="tab:orange")
    axes[1].set_title("Settling time")
    axes[1].set_ylabel("time [s]")

    axes[2].bar(x, settled_rate, color="tab:green")
    axes[2].set_title("Settled episodes")
    axes[2].set_ylabel("rate")
    axes[2].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_steady_error_vs_settling_time(summary: dict[str, Any], path: Path) -> None:
    ensure_writable_matplotlib_cache()
    import matplotlib.pyplot as plt

    methods, aggregates = method_aggregates(summary)
    steady = np.asarray([aggregate["steady_state_error"] for aggregate in aggregates], dtype=np.float32)
    settling = np.asarray([aggregate["settling_time_sec"] for aggregate in aggregates], dtype=np.float32)
    settled_rate = np.asarray([aggregate["settled_rate"] for aggregate in aggregates], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 80.0 + 220.0 * settled_rate
    scatter = ax.scatter(settling, steady, s=sizes, c=settled_rate, cmap="viridis", vmin=0.0, vmax=1.0, edgecolor="black")
    for method, x, y in zip(methods, settling, steady):
        ax.annotate(method, (float(x), float(y)), xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("settling time [s]")
    ax.set_ylabel("steady-state error [m]")
    ax.set_title("Control response tradeoff")
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("settled rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_computation_time_summary(summary: dict[str, Any], path: Path) -> None:
    ensure_writable_matplotlib_cache()
    import matplotlib.pyplot as plt

    methods, aggregates = method_aggregates(summary)
    mean_ms = np.asarray([aggregate["mean_compute_time"] for aggregate in aggregates], dtype=np.float32) * 1000.0
    p95_ms = np.asarray([aggregate["p95_compute_time"] for aggregate in aggregates], dtype=np.float32) * 1000.0
    max_ms = np.asarray([aggregate["max_compute_time"] for aggregate in aggregates], dtype=np.float32) * 1000.0
    real_time = np.asarray([aggregate["compute_real_time_factor"] for aggregate in aggregates], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(methods))
    width = 0.25
    axes[0].bar(x - width, mean_ms, width=width, label="mean")
    axes[0].bar(x, p95_ms, width=width, label="p95")
    axes[0].bar(x + width, max_ms, width=width, label="max")
    axes[0].set_title("Per-step planning latency")
    axes[0].set_ylabel("milliseconds")
    axes[0].legend()

    axes[1].bar(x, real_time, color="tab:red")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, label="real time")
    axes[1].set_title("Compute / simulated time")
    axes[1].set_ylabel("real-time factor")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def method_aggregates(summary: dict[str, Any]) -> tuple[list[str], list[dict[str, float]]]:
    methods = list(summary["methods"].keys())
    aggregates = [summary["methods"][method]["aggregate"] for method in methods]
    return methods, aggregates


def ensure_writable_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        config_dir = Path.home() / ".config" / "matplotlib"
        if not is_path_or_parent_writable(config_dir):
            fallback = Path(tempfile.gettempdir()) / "ballbalance_matplotlib"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback)

    if "XDG_CACHE_HOME" not in os.environ:
        cache_dir = Path.home() / ".cache"
        if not is_path_or_parent_writable(cache_dir):
            fallback_cache = Path(tempfile.gettempdir()) / "ballbalance_cache"
            fallback_cache.mkdir(parents=True, exist_ok=True)
            os.environ["XDG_CACHE_HOME"] = str(fallback_cache)


def is_path_or_parent_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    return path.parent.exists() and os.access(path.parent, os.W_OK)


if __name__ == "__main__":
    main()
