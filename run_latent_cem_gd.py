import argparse
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from dreamer import Dreamer, make_env


def infer_algo(checkpoint, stoch_size, discrete_classes):
    weight = checkpoint['rssm']['fc_state_prior.weight']
    output_size = weight.shape[0]
    if output_size == stoch_size * discrete_classes:
        return 'Dreamerv2'
    if output_size == 2 * stoch_size:
        return 'Dreamerv1'
    raise ValueError(
        f"Cannot infer RSSM type from fc_state_prior output size {output_size}. "
        "Pass --algo explicitly and matching --stoch-size/--discrete-classes."
    )


def build_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Dreamer world model with latent CEM-GD planning."
    )
    parser.add_argument('--checkpoint-path', required=True, help='Dreamer checkpoint to load')
    parser.add_argument('--env', type=str, default='ball-balance')
    parser.add_argument('--algo', type=str, default='auto', choices=['auto', 'Dreamerv1', 'Dreamerv2'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no-gpu', action='store_true')

    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--time-limit', type=int, default=300)
    parser.add_argument('--action-repeat', type=int, default=2)

    parser.add_argument('--horizon', type=int, default=15)
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--num-iterations', type=int, default=5)
    parser.add_argument('--elite-ratio', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num-top', type=int, default=3)
    parser.add_argument('--resample-amount', type=int, default=20)
    parser.add_argument('--no-gd', dest='use_gd', action='store_false')
    parser.set_defaults(use_gd=True)
    parser.add_argument('--stochastic-rollout', dest='deterministic_rollout', action='store_false')
    parser.set_defaults(deterministic_rollout=True)
    parser.add_argument('--bootstrap-value', action='store_true')

    parser.add_argument('--cnn-activation-function', type=str, default='relu')
    parser.add_argument('--dense-activation-function', type=str, default='elu')
    parser.add_argument('--obs-embed-size', type=int, default=1024)
    parser.add_argument('--num-units', type=int, default=400)
    parser.add_argument('--deter-size', type=int, default=200)
    parser.add_argument('--stoch-size', type=int, default=30)
    parser.add_argument('--discrete-classes', type=int, default=32)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--action-noise', type=float, default=0.0)

    parser.add_argument('--use-disc-model', dest='use_disc_model', action='store_true', default=None)
    parser.add_argument('--no-use-disc-model', dest='use_disc_model', action='store_false')

    parser.add_argument('--save-rollout', type=str, default='')
    parser.add_argument('--save-gif', type=str, default='', help='Path or directory for rollout GIF output')
    parser.add_argument('--gif-fps', type=float, default=20.0, help='Frames per second for saved GIFs')

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    if args.algo == 'auto':
        args.algo = infer_algo(checkpoint, args.stoch_size, args.discrete_classes)
    if args.use_disc_model is None:
        args.use_disc_model = checkpoint.get('discount_model', None) is not None

    # Fields used by Dreamer construction but not by this evaluation script.
    args.restore = True
    args.buffer_size = 1000000
    args.train_seq_len = 50
    args.batch_size = 50
    args.model_learning_rate = 6e-4
    args.value_learning_rate = 8e-5
    args.actor_learning_rate = 8e-5
    args.grad_clip_norm = 100.0
    args.imagine_horizon = args.horizon
    args.exp_name = 'latent_cem_gd'
    args.max_episode_length = args.time_limit
    args.free_nats = 3.0
    args.td_lambda = 0.95
    args.kl_loss_coeff = 1.0
    args.kl_alpha = 0.8
    args.disc_loss_coeff = 10.0
    args.actor_grad = 'dynamics'
    args.actor_grad_mix = 0.1
    args.actor_ent = 1e-4

    return args


def shift_plan(plan):
    shifted = plan.detach().clone().roll(-1, dims=0)
    shifted[-1].zero_()
    return shifted


def resolve_episode_gif_path(save_gif, episode, episodes):
    if not save_gif:
        return ''

    root, ext = os.path.splitext(save_gif)
    if ext.lower() == '.gif':
        if episodes == 1:
            return save_gif
        return f"{root}_ep{episode:03d}{ext}"

    os.makedirs(save_gif, exist_ok=True)
    return os.path.join(save_gif, f"episode_{episode:03d}.gif")


def save_episode_gif(frames, path, fps):
    if not frames:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / fps)))
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    args = build_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    env = make_env(args)
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.shape[0]
    dreamer = Dreamer(args, obs_shape, action_size, device, restore=True)

    for module in dreamer.world_model_modules + dreamer.value_modules + dreamer.actor_modules:
        module.eval()

    optimizer = dreamer.build_cem_gd_optimizer(
        horizon=args.horizon,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        num_iterations=args.num_iterations,
        elite_ratio=args.elite_ratio,
        population_size=args.population_size,
        alpha=args.alpha,
        num_top=args.num_top,
        resample_amount=args.resample_amount,
    )

    episode_rewards = []
    episode_lengths = []
    action_times = []
    rollout_rows = []

    print(
        "Loaded planner setup: "
        f"algo={args.algo}, use_disc_model={args.use_disc_model}, "
        f"horizon={args.horizon}, population={args.population_size}, use_gd={args.use_gd}"
    )

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        prev_state = dreamer.rssm.init_state(1, device)
        prev_action = torch.zeros(1, action_size, device=device)
        x0 = torch.zeros(args.horizon, action_size, device=device)
        total_reward = 0.0
        steps = 0
        video_frames = []

        while not done:
            if args.save_gif:
                video_frames.append(obs['image'].transpose(1, 2, 0).copy())

            action_start = time.perf_counter()
            posterior, action, plan = dreamer.act_with_cem_gd_planner(
                obs,
                prev_state,
                prev_action,
                optimizer=optimizer,
                horizon=args.horizon,
                x0=x0,
                use_opt=args.use_gd,
                deterministic=args.deterministic_rollout,
                bootstrap_value=args.bootstrap_value,
                return_plan=True,
            )
            action_times.append(time.perf_counter() - action_start)
            action_np = action[0].detach().cpu().numpy()
            next_obs, reward, done, info = env.step(action_np)

            if args.save_rollout:
                state = obs.get('state')
                row = {
                    'episode': episode,
                    'step': steps,
                    'reward': float(reward),
                    'done': bool(done),
                    'action': action_np.copy(),
                    'state': None if state is None else np.asarray(state, dtype=np.float32).copy(),
                }
                rollout_rows.append(row)

            total_reward += float(reward)
            steps += 1
            obs = next_obs
            prev_state = posterior
            prev_action = action.detach()
            x0 = shift_plan(plan)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"episode={episode} reward={total_reward:.3f} length={steps}")

        gif_path = resolve_episode_gif_path(args.save_gif, episode, args.episodes)
        if gif_path:
            save_episode_gif(video_frames, gif_path, args.gif_fps)
            print(f"saved_gif={gif_path}")

    rewards = np.asarray(episode_rewards, dtype=np.float32)
    lengths = np.asarray(episode_lengths, dtype=np.int32)
    print(
        f"mean_reward={rewards.mean():.3f} std_reward={rewards.std():.3f} "
        f"max_reward={rewards.max():.3f} min_reward={rewards.min():.3f} "
        f"mean_length={lengths.mean():.1f}"
    )
    if action_times:
        action_times = np.asarray(action_times, dtype=np.float64)
        print(
            f"action_time_mean={action_times.mean():.4f}s "
            f"action_time_std={action_times.std():.4f}s "
            f"action_time_min={action_times.min():.4f}s "
            f"action_time_max={action_times.max():.4f}s"
        )

    if args.save_rollout:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_rollout)), exist_ok=True)
        np.savez_compressed(
            args.save_rollout,
            rewards=rewards,
            lengths=lengths,
            rows=np.asarray(rollout_rows, dtype=object),
        )
        print(f"saved_rollout={args.save_rollout}")


if __name__ == '__main__':
    main()
