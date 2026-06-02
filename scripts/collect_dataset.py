"""Collect offline ball balance trajectories into an NPZ file."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv

PolicyMode = Literal["random_smooth", "pd", "mixed", "mpc_cover"]


@dataclass(frozen=True)
class InitialStateBounds:
    pos: float = 0.20
    vel: float = 0.05
    angle: float = 0.0

    def validate(self, env: BallBalanceEnv) -> None:
        if self.pos < 0.0 or self.vel < 0.0 or self.angle < 0.0:
            raise ValueError("initial-state bounds must be non-negative")
        if self.pos >= env.config.board_size / 2.0:
            raise ValueError("pos bound must be strictly inside the board")
        if self.angle > env.config.max_angle:
            raise ValueError("angle bound must be <= env.config.max_angle")


def pd_action(obs: np.ndarray, kp: float = 0.8, kd: float = 0.25, target: np.ndarray | None = None) -> np.ndarray:
    if target is None:
        target = np.zeros(2, dtype=np.float32)
    x, y, vx, vy, _, _ = obs
    err_x = x - target[0]
    err_y = y - target[1]
    theta_y_cmd = -kp * err_x - kd * vx
    theta_x_cmd = kp * err_y + kd * vy
    return np.array([theta_x_cmd, theta_y_cmd], dtype=np.float32)


def collect_dataset(
    num_episodes: int,
    max_episode_steps: int,
    seed: int,
    mode: PolicyMode,
    initial_bounds: InitialStateBounds = InitialStateBounds(),
    target_bound: float = 0.12,
    action_noise_std: float = 0.03,
) -> dict[str, np.ndarray]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if mode not in ("random_smooth", "pd", "mixed", "mpc_cover"):
        raise ValueError(f"Unsupported mode: {mode}")
    if target_bound < 0.0:
        raise ValueError("target_bound must be non-negative")
    if action_noise_std < 0.0:
        raise ValueError("action_noise_std must be non-negative")

    env = BallBalanceEnv(config={"max_episode_steps": max_episode_steps})
    initial_bounds.validate(env)
    rng = np.random.default_rng(seed)

    obs_data = np.zeros((num_episodes, max_episode_steps + 1, 6), dtype=np.float32)
    action_data = np.zeros((num_episodes, max_episode_steps, 2), dtype=np.float32)
    reward_data = np.zeros((num_episodes, max_episode_steps, 1), dtype=np.float32)
    terminated_data = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
    truncated_data = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)
    done_data = np.zeros((num_episodes, max_episode_steps, 1), dtype=bool)

    try:
        for episode in range(num_episodes):
            episode_mode = _choose_episode_mode(mode, rng)
            obs, _ = env.reset(seed=seed + episode, options={"state": _sample_initial_state(rng, initial_bounds)})
            obs_data[episode, 0] = obs

            action = np.zeros(2, dtype=np.float32)
            target = np.zeros(2, dtype=np.float32)
            noise_std = 0.0
            hold_action = np.zeros(2, dtype=np.float32)
            hold_steps = 0
            if episode_mode in ("pd_noisy", "pd_offset"):
                target = rng.uniform(-target_bound, target_bound, size=2).astype(np.float32)
                noise_std = action_noise_std if episode_mode == "pd_noisy" else 0.0

            done = False
            final_obs = obs.copy()

            for t in range(max_episode_steps):
                if done:
                    obs_data[episode, t + 1] = final_obs
                    action_data[episode, t] = 0.0
                    reward_data[episode, t, 0] = 0.0
                    terminated_data[episode, t, 0] = True
                    truncated_data[episode, t, 0] = False
                    done_data[episode, t, 0] = True
                    continue

                if episode_mode == "random_smooth":
                    random_action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                    action = 0.95 * action + 0.05 * random_action
                elif episode_mode == "random_uniform":
                    action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                elif episode_mode == "random_burst":
                    if hold_steps <= 0:
                        hold_action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                        hold_steps = int(rng.integers(5, 31))
                    action = 0.70 * action + 0.30 * hold_action
                    hold_steps -= 1
                else:
                    action = pd_action(obs, target=target)
                    if noise_std > 0.0:
                        action = action + rng.normal(0.0, noise_std, size=2).astype(np.float32)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                final_obs = next_obs.copy()

                obs_data[episode, t + 1] = next_obs
                action_data[episode, t] = np.clip(action, env.action_space.low, env.action_space.high)
                reward_data[episode, t, 0] = reward
                terminated_data[episode, t, 0] = terminated
                truncated_data[episode, t, 0] = truncated
                done_data[episode, t, 0] = done
                obs = next_obs
    finally:
        env.close()

    return {
        "obs": obs_data,
        "action": action_data,
        "reward": reward_data,
        "terminated": terminated_data,
        "truncated": truncated_data,
        "done": done_data,
        "initial_bounds": np.array([initial_bounds.pos, initial_bounds.vel, initial_bounds.angle], dtype=np.float32),
    }


def save_dataset(dataset: dict[str, np.ndarray], out: str | Path, **metadata: object) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset, **metadata)


def _choose_episode_mode(mode: PolicyMode, rng: np.random.Generator) -> str:
    if mode != "mixed":
        if mode != "mpc_cover":
            return mode
        value = rng.random()
        if value < 0.25:
            return "pd"
        if value < 0.45:
            return "pd_offset"
        if value < 0.65:
            return "pd_noisy"
        if value < 0.80:
            return "random_smooth"
        if value < 0.90:
            return "random_burst"
        return "random_uniform"
    value = rng.random()
    if value < 0.4:
        return "random_smooth"
    if value < 0.8:
        return "pd"
    return "pd_noisy"


def _sample_initial_state(rng: np.random.Generator, bounds: InitialStateBounds) -> np.ndarray:
    return np.array(
        [
            rng.uniform(-bounds.pos, bounds.pos),
            rng.uniform(-bounds.pos, bounds.pos),
            rng.uniform(-bounds.vel, bounds.vel),
            rng.uniform(-bounds.vel, bounds.vel),
            rng.uniform(-bounds.angle, bounds.angle),
            rng.uniform(-bounds.angle, bounds.angle),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--out", type=str, default="data/ball_balance_dataset.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["random_smooth", "pd", "mixed", "mpc_cover"], default="mixed")
    parser.add_argument("--pos-bound", type=float, default=0.20)
    parser.add_argument("--vel-bound", type=float, default=0.05)
    parser.add_argument("--angle-bound", type=float, default=0.0)
    parser.add_argument("--target-bound", type=float, default=0.12)
    parser.add_argument("--action-noise-std", type=float, default=0.03)
    args = parser.parse_args()

    initial_bounds = InitialStateBounds(pos=args.pos_bound, vel=args.vel_bound, angle=args.angle_bound)
    dataset = collect_dataset(
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        mode=args.mode,
        initial_bounds=initial_bounds,
        target_bound=args.target_bound,
        action_noise_std=args.action_noise_std,
    )
    save_dataset(
        dataset,
        args.out,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        mode=args.mode,
        pos_bound=args.pos_bound,
        vel_bound=args.vel_bound,
        angle_bound=args.angle_bound,
        target_bound=args.target_bound,
        action_noise_std=args.action_noise_std,
    )
    print(f"saved {args.num_episodes} episodes to {args.out}")


if __name__ == "__main__":
    main()
