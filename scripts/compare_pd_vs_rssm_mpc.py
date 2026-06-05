"""Compare PD and RSSM MPC on identical random initial states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import (
    InitialConditionBounds,
    aggregate_metrics,
    episode_metrics,
    run_mpc_episode,
    run_pd_episode,
    sample_initial_state,
)
from ball_rssm.planning.rssm_mpc import RSSMMPCController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--planning-objective", choices=["state_cost", "reward", "hybrid"], default="state_cost")
    parser.add_argument("--planner-type", choices=["cem", "cem_gd"], default="cem")
    parser.add_argument("--gd-num-sequences", type=int, default=3)
    parser.add_argument("--gd-iterations", type=int, default=15)
    parser.add_argument("--gd-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "compare_pd_vs_mpc"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    initial_states = [sample_initial_state(rng, InitialConditionBounds()) for _ in range(args.num_episodes)]
    pd_metrics = []
    mpc_metrics = []

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
            cost_mode="center",
            planning_objective=args.planning_objective,
            planner_type=args.planner_type,
            gd_num_sequences=args.gd_num_sequences,
            gd_iterations=args.gd_iterations,
            gd_lr=args.gd_lr,
            seed=args.seed,
        )
        for episode_idx, initial_state in enumerate(initial_states):
            pd_episode = run_pd_episode(env, initial_state, args.max_steps, target_xy=(0.0, 0.0))
            pd_metrics.append(episode_metrics(pd_episode, (0.0, 0.0), final_threshold=0.05, last_window_threshold=0.07))

            mpc_episode = run_mpc_episode(controller, env, initial_state, args.max_steps, target_xy=(0.0, 0.0))
            mpc_metrics.append(episode_metrics(mpc_episode, (0.0, 0.0), final_threshold=0.05, last_window_threshold=0.07))
            print(
                f"episode={episode_idx:03d} "
                f"pd_final={pd_metrics[-1]['final_distance']:.4f} "
                f"mpc_final={mpc_metrics[-1]['final_distance']:.4f}"
            )
    finally:
        env.close()

    summary = {
        "pd": {"episodes": pd_metrics, "aggregate": aggregate_metrics(pd_metrics)},
        "rssm_mpc": {"episodes": mpc_metrics, "aggregate": aggregate_metrics(mpc_metrics)},
        "config": vars(args),
    }
    (out_dir / "comparison_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"pd": summary["pd"]["aggregate"], "rssm_mpc": summary["rssm_mpc"]["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
