"""Train Dreamer V1/V2 online on a DeepMind Control Suite task."""

from __future__ import annotations

import argparse
import json
import sys
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.buffer import InitialStateBounds
from ball_rssm.envs.dm_control import (
    DMControlConfig,
    DMControlDriver,
    DMControlEnv,
    NormalizeActionWrapper,
    collect_random_dm_control,
)
from ball_rssm.models import Actor, ActorConfig, Critic, CriticConfig, Normalizer, WorldModel, WorldModelConfig
from ball_rssm.models.rssm import RSSMState
from ball_rssm.stream_replay import StreamReplay
from ball_rssm.trainer import DreamerTrainConfig, Trainer
from ball_rssm.utils.checkpoint import load_checkpoint
from ball_rssm.utils.seed import set_seed
from scripts.train_dreamer import find_resume_checkpoint, normalized_action_bounds


class DMControlTrainer(Trainer):
    """Trainer with DM-Control actor collection instead of BallBalanceEnv collection."""

    def __init__(self, *args: Any, dm_config: DMControlConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dm_config = dm_config

    def collect_actor_episodes(
        self,
        num_episodes: int,
        max_episode_steps: int,
        seed: int,
        initial_bounds: InitialStateBounds = InitialStateBounds(),
        exploration_noise: float = 0.0,
        configured_exploration_noise: float | None = None,
    ) -> dict[str, float]:
        del initial_bounds
        if configured_exploration_noise is None:
            configured_exploration_noise = exploration_noise
        if num_episodes <= 0:
            return {
                "collect_avg_reward": float("nan"),
                "collect_max_reward": float("nan"),
                "collect_min_reward": float("nan"),
                "collect_avg_length": float("nan"),
                "collect_episodes": 0.0,
                "replay_episodes": float(self.buffer.size),
                "exploration_mode_policy_entropy": float(self.exploration_mode == "policy_entropy"),
                "exploration_noise": float(exploration_noise),
                "configured_exploration_noise": float(configured_exploration_noise),
            }
        if max_episode_steps != self.buffer.max_episode_steps:
            raise ValueError("collector max_episode_steps must match replay buffer")

        env = NormalizeActionWrapper(DMControlEnv(self.dm_config, seed=seed))
        driver = DMControlDriver(env, self.buffer, max_episode_steps=max_episode_steps)
        was_world_training = self.world_model.training
        was_actor_training = self.actor.training
        self.world_model.eval()
        self.actor.eval()
        state: RSSMState | None = None
        prev_action = np.zeros(env.action_shape, dtype=np.float32)

        def policy(obs: np.ndarray, is_first: bool) -> np.ndarray:
            nonlocal state, prev_action
            if state is None or is_first:
                state = self.world_model.initial_state(1, self.device)
                prev_action = np.zeros(env.action_shape, dtype=np.float32)
            assert state is not None
            state = self._posterior_update_np(state, obs, prev_action, is_first=is_first)
            with torch.no_grad():
                action_norm = self._sample_actor_action_norm(state, exploration_noise)
                action = self.normalizer.denormalize_action(action_norm).reshape(env.action_shape)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            action_np = np.clip(action_np, env.action_low, env.action_high)
            prev_action = action_np
            return action_np

        try:
            driver_metrics = driver.run(policy, num_episodes)
        finally:
            env.close()
            self.world_model.train(was_world_training)
            self.actor.train(was_actor_training)

        return {
            "collect_avg_reward": driver_metrics["avg_reward"],
            "collect_max_reward": driver_metrics["max_reward"],
            "collect_min_reward": driver_metrics["min_reward"],
            "collect_avg_length": driver_metrics["avg_length"],
            "collect_episodes": float(num_episodes),
            "replay_episodes": float(self.buffer.size),
            "exploration_mode_policy_entropy": float(self.exploration_mode == "policy_entropy"),
            "exploration_noise": float(exploration_noise),
            "configured_exploration_noise": float(configured_exploration_noise),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="cartpole")
    parser.add_argument("--task", default="swingup")
    parser.add_argument("--obs-type", choices=["state", "pixel"], default="state")
    parser.add_argument("--run-dir", default="runs/dmc_cartpole_swingup_v1")
    parser.add_argument("--dataset", default=None, help="Optional seed replay NPZ; otherwise random replay is collected.")
    parser.add_argument("--dreamer-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--online-iterations", type=int, default=100)
    parser.add_argument("--update-steps", type=int, default=100)
    parser.add_argument("--seed-episodes", type=int, default=20)
    parser.add_argument("--collect-episodes", type=int, default=5)
    parser.add_argument("--buffer-episodes", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--mujoco-gl", choices=["egl", "osmesa", "glfw"], default=None)
    parser.add_argument("--world-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--critic-lr", type=float, default=8e-5)
    parser.add_argument("--deter-dim", type=int, default=128)
    parser.add_argument("--stoch-dim", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--discrete-classes", type=int, default=32)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--beta-kl", type=float, default=1.0)
    parser.add_argument("--free-nats", type=float, default=1.0)
    parser.add_argument("--kl-alpha", type=float, default=0.8)
    parser.add_argument("--reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--continuation-loss-weight", type=float, default=1.0)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    parser.add_argument("--behavior-batch-size", type=int, default=4096)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--actor-gradient", choices=["auto", "dynamics", "reinforce", "both"], default="auto")
    parser.add_argument("--actor-entropy-scale", type=float, default=1e-3)
    parser.add_argument("--exploration-mode", choices=["auto", "noise", "policy_entropy"], default="auto")
    parser.add_argument("--exploration-noise", type=float, default=0.3)
    parser.add_argument("--exploration-decay", type=float, default=1.0)
    parser.add_argument("--min-exploration-noise", type=float, default=0.0)
    parser.add_argument("--target-tau", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    dm_config = DMControlConfig(
        domain=args.domain,
        task=args.task,
        obs_type=args.obs_type,
        action_repeat=args.action_repeat,
        height=args.height,
        width=args.width,
        camera_id=args.camera_id,
        mujoco_gl=args.mujoco_gl,
    )

    capacity_steps = max(args.buffer_episodes * (args.max_episode_steps + 1), 1)
    replay_resume_path = run_dir / "replay" / "latest.npz"
    if args.dataset is None and args.resume and replay_resume_path.exists():
        buffer = StreamReplay.load(replay_resume_path, capacity_steps=capacity_steps)
        print(
            f"resuming DM-Control replay from {replay_resume_path} "
            f"with {buffer.size} episodes and {buffer.num_steps} transitions"
        )
    elif args.dataset is not None:
        buffer = StreamReplay.load(args.dataset, capacity_steps=capacity_steps)
    else:
        arrays = collect_random_dm_control(
            config=dm_config,
            num_episodes=args.seed_episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        )
        buffer = StreamReplay.from_arrays(
            arrays,
            capacity_steps=capacity_steps,
            max_episode_steps=args.max_episode_steps,
            metadata={"action_normalization": "dm_control_normalized"},
        )

    if buffer.max_episode_steps != args.max_episode_steps:
        args.max_episode_steps = buffer.max_episode_steps
    if buffer.action_low is None:
        action_low = -np.ones(buffer.action_shape, dtype=np.float32)
    else:
        action_low = buffer.action_low.astype(np.float32)
    if buffer.action_high is None:
        action_high = np.ones(buffer.action_shape, dtype=np.float32)
    else:
        action_high = buffer.action_high.astype(np.float32)
    action_bounds = (action_low, action_high)

    print(
        f"DM-Control replay ready: episodes={buffer.size}, transitions={buffer.num_steps}, "
        f"obs_shape={buffer.obs_shape}, action_shape={buffer.action_shape}"
    )

    train_ds = buffer.sequence_dataset(args.seq_len, split="train", val_fraction=args.val_fraction, seed=args.seed)
    train_obs, train_action, train_reward = train_ds.selected_arrays()
    # DM-Control actions are already stored in the wrapper's [-1, 1] coordinate
    # system, matching Danijar's normalized action wrappers.
    normalizer = Normalizer.from_arrays(train_obs, train_action, train_reward, normalize_action=False)
    obs_shape = tuple(int(dim) for dim in train_obs.shape[2:])
    action_dim = int(train_action.shape[-1])
    world_obs_type = "pixel" if args.obs_type == "pixel" else "vector"

    resume_path = find_resume_checkpoint(run_dir, args.resume_from) if args.resume else None
    checkpoint: dict[str, Any] | None = load_checkpoint(resume_path, device) if resume_path is not None else None
    if checkpoint is None:
        world_config = WorldModelConfig(
            obs_dim=int(prod(obs_shape)),
            obs_shape=obs_shape,
            obs_type=world_obs_type,
            action_dim=action_dim,
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dreamer_version=args.dreamer_version,
            discrete_classes=args.discrete_classes,
            beta_kl=args.beta_kl,
            free_nats=args.free_nats,
            kl_alpha=args.kl_alpha,
            reward_loss_weight=args.reward_loss_weight,
            continuation_loss_weight=args.continuation_loss_weight,
        )
        action_low, action_high = normalized_action_bounds(normalizer, action_bounds=action_bounds)
        actor_config = ActorConfig(
            feature_dim=world_config.feature_dim,
            action_dim=world_config.action_dim,
            hidden_dim=args.actor_hidden_dim,
            action_low=action_low,
            action_high=action_high,
        )
        critic_config = CriticConfig(feature_dim=world_config.feature_dim, hidden_dim=args.critic_hidden_dim)
        start_epoch = 0
        best_val_loss = float("inf")
    else:
        if "actor_config" not in checkpoint:
            raise KeyError(f"{resume_path} is a world-model-only checkpoint; train Dreamer with --no-resume")
        world_config = WorldModelConfig.from_dict(checkpoint.get("world_model_config", checkpoint["config"]))
        actor_config = ActorConfig.from_dict(checkpoint["actor_config"])
        critic_config = CriticConfig.from_dict(checkpoint["critic_config"])
        normalizer = Normalizer.load_state_dict(checkpoint["normalizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resuming from {resume_path} at iteration {start_epoch}; target iterations={args.online_iterations}")

    normalizer = normalizer.to(device)
    world_model = WorldModel(world_config).to(device)
    actor = Actor(actor_config).to(device)
    critic = Critic(critic_config).to(device)
    target_critic = Critic(critic_config).to(device)
    target_critic.load_state_dict(critic.state_dict())

    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=args.world_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    if checkpoint is not None:
        world_model.load_state_dict(checkpoint.get("world_model_state_dict", checkpoint["model_state_dict"]))
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        target_critic.load_state_dict(checkpoint.get("target_critic_state_dict", checkpoint["critic_state_dict"]))
        if "world_optimizer_state_dict" in checkpoint:
            world_optimizer.load_state_dict(checkpoint["world_optimizer_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    dreamer_config = DreamerTrainConfig(
        imagination_horizon=args.imagination_horizon,
        behavior_batch_size=args.behavior_batch_size,
        discount=args.discount,
        lambda_=args.lambda_,
        actor_gradient=args.actor_gradient,
        actor_entropy_scale=args.actor_entropy_scale,
        exploration_mode=args.exploration_mode,
        exploration_noise=args.exploration_noise,
        exploration_decay=args.exploration_decay,
        min_exploration_noise=args.min_exploration_noise,
        grad_clip=args.grad_clip,
        target_tau=args.target_tau,
    )

    config_dict = {
        **vars(args),
        "dm_control": dm_config.to_dict(),
        "replay": {
            "type": "step_stream",
            "action_normalization": "dm_control_normalized",
            "capacity_steps": buffer.capacity_steps,
            "num_steps": buffer.num_steps,
        },
        "world_model": world_config.to_dict(),
        "actor": actor_config.to_dict(),
        "critic": critic_config.to_dict(),
        "dreamer": dreamer_config.to_dict(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    trainer = DMControlTrainer(
        world_model=world_model,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        buffer=buffer,
        world_optimizer=world_optimizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        normalizer=normalizer,
        device=device,
        run_dir=run_dir,
        config=dreamer_config,
        dm_config=dm_config,
    )
    trainer.train_online(
        iterations=args.online_iterations,
        update_steps=args.update_steps,
        collect_episodes=args.collect_episodes,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        seed=args.seed,
        train_args=config_dict,
        initial_bounds=InitialStateBounds(),
        start_iteration=start_epoch,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    main()
