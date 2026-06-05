"""Stress-test RSSM MPC from board positions that are rare or unseen in data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.planning.rollout import aggregate_metrics, episode_metrics, run_mpc_episode, save_episode_npz, save_mpc_plots
from ball_rssm.planning.rssm_mpc import RSSMMPCController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heatmap-data", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--planner-type", choices=["cem", "cem_gd", "both"], default="both")
    parser.add_argument("--planning-objective", choices=["state_cost", "reward", "hybrid"], default="reward")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-count", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--num-candidates", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--gd-num-sequences", type=int, default=3)
    parser.add_argument("--gd-iterations", type=int, default=15)
    parser.add_argument("--gd-lr", type=float, default=0.01)
    parser.add_argument("--vel-bound", type=float, default=0.02)
    parser.add_argument("--angle-bound", type=float, default=0.02)
    parser.add_argument("--position-margin", type=float, default=0.02)
    parser.add_argument("--jitter-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--save-episodes", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if args.max_count < 0:
        parser.error("--max-count must be non-negative")
    if args.vel_bound < 0.0 or args.angle_bound < 0.0:
        parser.error("--vel-bound and --angle-bound must be non-negative")
    if not 0.0 <= args.jitter_fraction <= 0.5:
        parser.error("--jitter-fraction must be in [0, 0.5]")

    rng = np.random.default_rng(args.seed)
    env = BallBalanceEnv(config={"max_episode_steps": args.max_steps})
    try:
        initial_states, bin_counts, bin_xy = sample_unvisited_initial_states(
            heatmap_data=Path(args.heatmap_data),
            num_states=args.num_episodes,
            max_count=args.max_count,
            vel_bound=args.vel_bound,
            angle_bound=args.angle_bound,
            position_margin=args.position_margin,
            jitter_fraction=args.jitter_fraction,
            board_half=env.config.board_size / 2.0,
            rng=rng,
        )

        out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "unvisited_mpc_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "unvisited_initial_states.npz",
            initial_states=initial_states,
            bin_counts=bin_counts,
            bin_xy=bin_xy,
        )

        planner_types = ["cem", "cem_gd"] if args.planner_type == "both" else [args.planner_type]
        summary: dict[str, object] = {
            "config": vars(args),
            "initial_state_source": {
                "heatmap_data": str(Path(args.heatmap_data)),
                "max_count": args.max_count,
            },
            "planners": {},
        }

        for planner_type in planner_types:
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
                planner_type=planner_type,
                gd_num_sequences=args.gd_num_sequences,
                gd_iterations=args.gd_iterations,
                gd_lr=args.gd_lr,
                seed=args.seed + (100 if planner_type == "cem" else 200),
            )

            planner_dir = out_dir / planner_type
            if args.save_episodes or args.save_plots:
                planner_dir.mkdir(parents=True, exist_ok=True)
            metrics = []
            for episode_idx, initial_state in enumerate(initial_states):
                episode = run_mpc_episode(controller, env, initial_state, args.max_steps, target_xy=(0.0, 0.0))
                metric = episode_metrics(episode, (0.0, 0.0), final_threshold=0.05, last_window_threshold=0.07)
                metric["initial_bin_count"] = float(bin_counts[episode_idx])
                metric["initial_x"] = float(initial_state[0])
                metric["initial_y"] = float(initial_state[1])
                metrics.append(metric)
                if args.save_episodes:
                    save_episode_npz(episode, planner_dir / f"episode_{episode_idx:03d}.npz")
                if args.save_plots:
                    save_mpc_plots(
                        episode,
                        planner_dir / f"episode_{episode_idx:03d}",
                        target_xy=(0.0, 0.0),
                        title_prefix=f"RSSM {planner_type} unvisited",
                    )
                print(
                    f"planner={planner_type} episode={episode_idx:03d} "
                    f"count={bin_counts[episode_idx]} final={metric['final_distance']:.4f} "
                    f"success={metric['success']} fell={metric['fell']}"
                )

            summary["planners"][planner_type] = {
                "episodes": metrics,
                "aggregate": aggregate_metrics(metrics),
            }
    finally:
        env.close()

    (out_dir / "unvisited_mpc_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value["aggregate"] for key, value in summary["planners"].items()}, indent=2))
    print(f"wrote unvisited-state MPC evaluation to {out_dir}")


def sample_unvisited_initial_states(
    heatmap_data: Path,
    num_states: int,
    max_count: int,
    vel_bound: float,
    angle_bound: float,
    position_margin: float,
    jitter_fraction: float,
    board_half: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(heatmap_data) as data:
        counts = data["counts"].astype(np.int64)
        x_edges = data["x_edges"].astype(np.float32)
        y_edges = data["y_edges"].astype(np.float32)

    rows, cols = np.where(counts <= max_count)
    if rows.size < num_states:
        order = np.argsort(counts.reshape(-1), kind="stable")
        rows, cols = np.unravel_index(order[:num_states], counts.shape)

    candidate_indices = np.arange(rows.size)
    replace = rows.size < num_states
    chosen = rng.choice(candidate_indices, size=num_states, replace=replace)
    chosen_rows = rows[chosen]
    chosen_cols = cols[chosen]
    bin_counts = counts[chosen_rows, chosen_cols]

    x_centers = 0.5 * (x_edges[chosen_cols] + x_edges[chosen_cols + 1])
    y_centers = 0.5 * (y_edges[chosen_rows] + y_edges[chosen_rows + 1])
    x_width = x_edges[chosen_cols + 1] - x_edges[chosen_cols]
    y_width = y_edges[chosen_rows + 1] - y_edges[chosen_rows]
    x = x_centers + rng.uniform(-jitter_fraction, jitter_fraction, size=num_states) * x_width
    y = y_centers + rng.uniform(-jitter_fraction, jitter_fraction, size=num_states) * y_width

    limit = max(0.0, board_half - position_margin)
    x = np.clip(x, -limit, limit)
    y = np.clip(y, -limit, limit)
    vx = rng.uniform(-vel_bound, vel_bound, size=num_states)
    vy = rng.uniform(-vel_bound, vel_bound, size=num_states)
    theta_x = rng.uniform(-angle_bound, angle_bound, size=num_states)
    theta_y = rng.uniform(-angle_bound, angle_bound, size=num_states)

    initial_states = np.stack([x, y, vx, vy, theta_x, theta_y], axis=-1).astype(np.float32)
    bin_xy = np.stack([x_centers, y_centers], axis=-1).astype(np.float32)
    return initial_states, bin_counts.astype(np.int64), bin_xy


if __name__ == "__main__":
    main()
