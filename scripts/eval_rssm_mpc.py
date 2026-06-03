"""Evaluate RSSM MPC over many random initial conditions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import InitialConditionBounds, aggregate_metrics, episode_metrics, run_mpc_episode, sample_initial_state
from ball_rssm.planning.rssm_mpc import RSSMMPCController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["center", "point"], default="center")
    parser.add_argument("--target-x", type=float, default=0.15)
    parser.add_argument("--target-y", type=float, default=-0.10)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--planning-objective", choices=["state_cost", "reward", "hybrid"], default="state_cost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    target = (0.0, 0.0) if args.mode == "center" else (args.target_x, args.target_y)
    final_threshold = 0.05 if args.mode == "center" else 0.06
    last_threshold = 0.07 if args.mode == "center" else 0.08
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / f"eval_mpc_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    metrics = []
    env = BallBalanceEnv(config={"max_episode_steps": args.max_steps})
    try:
        controller = RSSMMPCController(
            checkpoint_path=args.checkpoint,
            action_space=env.action_space,
            horizon=args.horizon,
            num_candidates=args.num_candidates,
            num_elites=args.num_elites,
            num_iterations=args.num_iterations,
            device=args.device,
            cost_mode=args.mode,
            planning_objective=args.planning_objective,
            target_xy=None if args.mode == "center" else target,
            seed=args.seed,
        )
        for episode_idx in range(args.num_episodes):
            initial_state = sample_initial_state(rng, InitialConditionBounds())
            episode = run_mpc_episode(controller, env, initial_state, args.max_steps, target_xy=target)
            metrics.append(episode_metrics(episode, target, final_threshold=final_threshold, last_window_threshold=last_threshold))
            print(f"episode={episode_idx:03d} final_distance={metrics[-1]['final_distance']:.4f} fell={metrics[-1]['fell']}")
    finally:
        env.close()

    summary = {"episodes": metrics, "aggregate": aggregate_metrics(metrics), "config": vars(args)}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
