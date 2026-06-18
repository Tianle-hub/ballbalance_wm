"""Evaluate a trained Dreamer actor in the true ball balance environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.agent import DreamerAgent
from ball_rssm.envs import BallBalanceEnv
from ball_rssm.evaluation import (
    InitialConditionBounds,
    aggregate_metrics,
    episode_metrics,
    run_policy_episode,
    sample_initial_state,
    save_episode_npz,
    save_policy_plots,
)
from ball_rssm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos-bound", type=float, default=0.25)
    parser.add_argument("--vel-bound", type=float, default=0.20)
    parser.add_argument("--angle-bound", type=float, default=0.12)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the actor instead of using its mode.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--final-threshold", type=float, default=0.05)
    parser.add_argument("--last-window-threshold", type=float, default=0.07)
    parser.add_argument("--save-episodes", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.final_threshold <= 0.0:
        raise ValueError("final_threshold must be positive")
    if args.last_window_threshold <= 0.0:
        raise ValueError("last_window_threshold must be positive")

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parents[1] / "policy_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = BallBalanceEnv(render_mode="human" if args.render else None, config={"max_episode_steps": args.max_steps})
    bounds = InitialConditionBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    bounds.validate(env)
    agent = DreamerAgent(
        checkpoint_path=args.checkpoint,
        action_space=env.action_space,
        device=device,
        deterministic=not args.stochastic,
    )
    rng = np.random.default_rng(args.seed)
    metrics = []
    target = (0.0, 0.0)
    print(
        "success requirement: "
        f"target={target}, no termination/fall, final_distance < {args.final_threshold:.3f}, "
        f"last50_mean_distance < {args.last_window_threshold:.3f}"
    )

    try:
        for episode_idx in range(args.num_episodes):
            initial_state = sample_initial_state(rng, bounds)
            episode = run_policy_episode(agent, env, initial_state, max_steps=args.max_steps, render=args.render)
            episode_metric = episode_metrics(episode)
            episode_metric.update(
                success_metrics(
                    episode,
                    target_xy=target,
                    final_threshold=args.final_threshold,
                    last_window_threshold=args.last_window_threshold,
                )
            )
            metrics.append(episode_metric)
            if args.save_episodes:
                save_episode_npz(episode, out_dir / f"episode_{episode_idx:03d}.npz")
            if args.save_plots:
                save_policy_plots(episode, out_dir / f"episode_{episode_idx:03d}")
            print(
                f"episode={episode_idx:03d} success={episode_metric['success']} "
                f"terminated={episode_metric['terminated']} return={episode_metric['return']:.3f} "
                f"steps={episode_metric['steps']} final_distance={episode_metric['final_distance']:.4f} "
                f"last50_distance={episode_metric['last50_distance']:.4f}"
            )
    finally:
        env.close()

    aggregate = aggregate_metrics(metrics)
    success_count = sum(bool(metric["success"]) for metric in metrics)
    aggregate["success_rate"] = success_count / len(metrics) if metrics else 0.0
    summary = {
        "checkpoint": str(args.checkpoint),
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "bounds": {"pos": args.pos_bound, "vel": args.vel_bound, "angle": args.angle_bound},
        "success_requirement": {
            "target_x": target[0],
            "target_y": target[1],
            "final_threshold": args.final_threshold,
            "last_window_threshold": args.last_window_threshold,
            "requires_no_termination": True,
        },
        "episodes": metrics,
        "aggregate": aggregate,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"successes={success_count}/{len(metrics)} success_rate={aggregate['success_rate']:.3f}")
    print(f"wrote Dreamer policy evaluation to {out_dir}")


def success_metrics(
    episode: dict[str, object],
    target_xy: tuple[float, float],
    final_threshold: float,
    last_window_threshold: float,
) -> dict[str, float | bool]:
    obs = np.asarray(episode["obs"], dtype=np.float32)
    target = np.asarray(target_xy, dtype=np.float32)
    distance = np.linalg.norm(obs[:, :2] - target[None], axis=-1)
    tail = distance[-min(50, distance.shape[0]) :]
    terminated = bool(np.asarray(episode["terminated"]).any())
    success = (not terminated) and float(distance[-1]) < final_threshold and float(tail.mean()) < last_window_threshold
    return {
        "success": bool(success),
        "last50_distance": float(tail.mean()),
    }


if __name__ == "__main__":
    main()
