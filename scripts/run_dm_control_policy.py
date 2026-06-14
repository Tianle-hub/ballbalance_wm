"""Evaluate a trained Dreamer policy on a DM-Control task with optional rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs.dm_control import DMControlConfig, DMControlEnv, NormalizeActionWrapper
from ball_rssm.models import Actor, ActorConfig, Normalizer, WorldModel, WorldModelConfig
from ball_rssm.models.rssm import RSSMState
from ball_rssm.utils.checkpoint import load_checkpoint


class DMControlPolicy:
    """Stateful Dreamer policy for DM-Control observations."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
    ) -> None:
        self.device = torch.device(device if str(device) != "cuda" or torch.cuda.is_available() else "cpu")
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        world_config = checkpoint.get("world_model_config", checkpoint.get("config"))
        actor_config = checkpoint.get("actor_config")
        if world_config is None or actor_config is None:
            raise KeyError("checkpoint must contain world_model_config and actor_config")

        self.world_model = WorldModel(WorldModelConfig.from_dict(world_config)).to(self.device)
        self.world_model.load_state_dict(checkpoint.get("world_model_state_dict", checkpoint["model_state_dict"]))
        self.world_model.eval()
        self.world_model.requires_grad_(False)

        self.actor = Actor(ActorConfig.from_dict(actor_config)).to(self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.eval()
        self.actor.requires_grad_(False)

        self.normalizer = Normalizer.load_state_dict(checkpoint["normalizer"]).to(self.device)
        self.action_low = action_low.reshape(-1).astype(np.float32)
        self.action_high = action_high.reshape(-1).astype(np.float32)
        self.action_shape = tuple(action_low.shape)
        self.action_dim = int(np.prod(action_low.shape))
        self.deterministic = deterministic
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.state: RSSMState | None = None
        self.needs_update = False

    def reset(self, initial_obs: np.ndarray) -> None:
        self.state = self.world_model.initial_state(1, self.device)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.needs_update = False
        self.state = self._posterior_update(initial_obs, self.prev_action, is_first=True)

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.reset(obs)
        elif self.needs_update:
            self.state = self._posterior_update(obs, self.prev_action)

        assert self.state is not None
        with torch.no_grad():
            features = self.world_model.features_from_state(self.state)
            action_tensor = self.actor.sample(features, deterministic=self.deterministic).reshape(-1)
        action = action_tensor.detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, self.action_low, self.action_high)
        self.prev_action = action
        self.needs_update = True
        return action.reshape(self.action_shape)

    def _posterior_update(self, obs: np.ndarray, prev_action: np.ndarray, is_first: bool = False) -> RSSMState:
        assert self.state is not None
        obs_shape = self.world_model.config.obs_shape
        if obs_shape is None:
            raise ValueError("world model config does not define obs_shape")
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, *obs_shape)
        action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device).reshape(1, -1)
        is_first_t = torch.as_tensor([is_first], dtype=torch.float32, device=self.device)
        return self.world_model.posterior_update(
            self.state,
            self.normalizer.normalize_action(action_t),
            self.normalizer.normalize_obs(obs_t),
            is_first_t,
        )


def infer_dm_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> DMControlConfig:
    train_args = checkpoint.get("train_args", {})
    dm_state = {}
    if isinstance(train_args, dict):
        maybe_dm = train_args.get("dm_control", {})
        if isinstance(maybe_dm, dict):
            dm_state.update(maybe_dm)
    domain = args.domain or str(dm_state.get("domain", "cartpole"))
    task = args.task or str(dm_state.get("task", "swingup"))
    obs_type = args.obs_type or str(dm_state.get("obs_type", "state"))
    return DMControlConfig(
        domain=domain,
        task=task,
        obs_type=obs_type,
        action_repeat=args.action_repeat or int(dm_state.get("action_repeat", 1)),
        height=args.height or int(dm_state.get("height", 64)),
        width=args.width or int(dm_state.get("width", 64)),
        camera_id=args.camera_id if args.camera_id is not None else int(dm_state.get("camera_id", 0)),
        mujoco_gl=args.mujoco_gl or dm_state.get("mujoco_gl"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--obs-type", choices=["state", "pixel"], default=None)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--frames-out", default=None)
    parser.add_argument("--gif-out", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    dm_config = infer_dm_config(checkpoint, args)
    env = NormalizeActionWrapper(DMControlEnv(dm_config, seed=args.seed))
    policy = DMControlPolicy(
        checkpoint_path=args.checkpoint,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
        deterministic=args.deterministic,
    )
    frames_dir = Path(args.frames_out) if args.frames_out is not None else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []

    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(args.num_episodes):
            obs = env.reset()
            policy.reset(obs)
            total_reward = 0.0
            length = 0
            for step in range(args.max_steps):
                if args.render or args.gif_out is not None or frames_dir is not None:
                    frame = env.render()
                    if args.gif_out is not None:
                        frames.append(frame)
                    if frames_dir is not None:
                        import imageio.v3 as iio

                        iio.imwrite(frames_dir / f"episode_{episode:03d}_step_{step:05d}.png", frame)

                action = policy.act(obs)
                obs, reward, _, _, done = env.step(action)
                total_reward += reward
                length = step + 1
                if done:
                    break
            returns.append(total_reward)
            lengths.append(length)
            print(f"episode={episode} return={total_reward:.3f} length={length}")
    finally:
        env.close()

    if args.gif_out is not None and frames:
        import imageio.v2 as imageio

        out_path = Path(args.gif_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames, fps=args.fps)
        print(f"saved gif to {out_path}")

    summary = {
        "domain": dm_config.domain,
        "task": dm_config.task,
        "obs_type": dm_config.obs_type,
        "episodes": args.num_episodes,
        "avg_return": float(np.mean(returns)) if returns else float("nan"),
        "avg_length": float(np.mean(lengths)) if lengths else float("nan"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
