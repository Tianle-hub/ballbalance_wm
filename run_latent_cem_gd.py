import argparse
import copy
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
    parser.add_argument('--timing-details', action='store_true', help='Print per-action planner timing breakdowns')
    parser.add_argument('--online', action='store_true', help='Show online MPC planning dashboard')
    parser.add_argument('--online-pause', type=float, default=0.001, help='Pause between online plot updates')
    parser.add_argument('--online-history', type=int, default=300, help='Maximum history length shown online')
    parser.add_argument('--save-online-gif', type=str, default='', help='Path for online dashboard GIF recording')
    parser.add_argument('--no-save-online-gif', action='store_true', help='Disable default online dashboard recording')
    parser.add_argument('--online-record-fps', type=float, default=10.0, help='Frames per second for online dashboard GIF')

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


def timing_value(records, key):
    return np.asarray([record[key] for record in records if key in record], dtype=np.float64)


def nested_timing_value(records, section, key):
    return np.asarray(
        [
            record.get(section, {}).get(key)
            for record in records
            if key in record.get(section, {})
        ],
        dtype=np.float64,
    )


def format_stats(prefix, values, unit='s'):
    if values.size == 0:
        return None
    return (
        f"{prefix}_mean={values.mean():.4f}{unit} "
        f"{prefix}_std={values.std():.4f}{unit} "
        f"{prefix}_min={values.min():.4f}{unit} "
        f"{prefix}_max={values.max():.4f}{unit}"
    )


def find_ball_config(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, '_env', None)
        config = getattr(inner, 'config', None)
        if config is not None:
            return config
        current = inner
    return None


def board_limits(board_size):
    half = board_size / 2.0
    return -half * 1.15, half * 1.15


def image_board_bounds(image_shape, board_size):
    height, width = image_shape[:2]
    margin = max(2, int(0.08 * min(width, height)))
    left, top = margin, margin
    right, bottom = width - margin - 1, height - margin - 1
    return left, top, right, bottom


def pixel_to_board(pixel_xy, image_shape, board_size):
    left, top, right, bottom = image_board_bounds(image_shape, board_size)
    px, py = pixel_xy
    half = board_size / 2.0
    x = (px - left) / max(right - left, 1) * board_size - half
    y = half - (py - top) / max(bottom - top, 1) * board_size
    return np.array([x, y], dtype=np.float32)


def board_to_pixel(xy, image_shape, board_size):
    left, top, right, bottom = image_board_bounds(image_shape, board_size)
    half = board_size / 2.0
    x, y = xy
    px = left + (x + half) / board_size * (right - left)
    py = bottom - (y + half) / board_size * (bottom - top)
    return np.array([px, py], dtype=np.float32)


def estimate_ball_pixel(frame):
    frame = np.asarray(frame, dtype=np.float32)
    red = frame[..., 0]
    green = frame[..., 1]
    blue = frame[..., 2]
    score = red - np.maximum(green, blue)
    score = np.maximum(score, 0.0)
    if float(score.sum()) < 1e-6:
        return None
    yy, xx = np.indices(score.shape)
    weight = score / score.sum()
    px = float((xx * weight).sum())
    py = float((yy * weight).sum())
    return np.array([px, py], dtype=np.float32)


def decode_obs_mean(policy, features):
    out_batch_shape = features.shape[:-1]
    out = policy.obs_decoder.dense(features)
    out = torch.reshape(out, [-1, 32 * policy.obs_decoder.depth, 1, 1])
    out = policy.obs_decoder.convtranspose(out)
    return torch.reshape(out, (*out_batch_shape, *policy.obs_decoder.output_shape))


def decoded_frame_from_mean(mean_chw):
    frame = mean_chw.detach().cpu().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    frame = np.clip(frame + 0.5, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def decode_imagined_ball_path(policy, start_state, plan, deterministic, board_size, image_shape):
    pixels = []
    board_points = []
    state = policy.rssm.detach_state(start_state)
    with torch.no_grad():
        for action in plan:
            action = action.to(policy.device).unsqueeze(0)
            if deterministic:
                state = policy._deterministic_imagine_step(state, action)
            else:
                state = policy.rssm.imagine_step(state, action)
            features = policy.rssm.get_feat(state)
            decoded = decode_obs_mean(policy, features)[0]
            frame = decoded_frame_from_mean(decoded)
            pixel = estimate_ball_pixel(frame)
            if pixel is None:
                pixels.append(np.array([np.nan, np.nan], dtype=np.float32))
                board_points.append(np.array([np.nan, np.nan], dtype=np.float32))
            else:
                pixels.append(pixel)
                board_points.append(pixel_to_board(pixel, image_shape, board_size))
    return np.asarray(pixels, dtype=np.float32), np.asarray(board_points, dtype=np.float32)


class OnlineMPCVisualizer:
    def __init__(self, board_size, max_angle, action_bound, history_len=300, pause=0.001):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.board_size = board_size
        self.max_angle = max_angle
        self.action_bound = action_bound
        self.history_len = history_len
        self.pause_seconds = pause

        self.plt.ion()
        self.fig = self.plt.figure(figsize=(14, 8))
        grid = self.fig.add_gridspec(2, 3)
        self.image_ax = self.fig.add_subplot(grid[0, 0])
        self.board_ax = self.fig.add_subplot(grid[1, 0])
        self.action_ax = self.fig.add_subplot(grid[0, 1:])
        self.pos_ax = self.fig.add_subplot(grid[1, 1])
        self.vel_ax = self.fig.add_subplot(grid[1, 2])

        self.image_artist = None
        (self.image_future_line,) = self.image_ax.plot([], [], "--", color="#17becf", linewidth=1.6)
        (self.image_future_dot,) = self.image_ax.plot([], [], "o", color="#17becf", markersize=3)

        half = board_size / 2.0
        low, high = board_limits(board_size)
        self.board_ax.set_xlim(low, high)
        self.board_ax.set_ylim(low, high)
        self.board_ax.set_aspect("equal", adjustable="box")
        self.board_ax.axhline(0.0, color="0.85", linewidth=1)
        self.board_ax.axvline(0.0, color="0.85", linewidth=1)
        self.board_ax.add_patch(
            self.plt.Rectangle((-half, -half), board_size, board_size, fill=False, edgecolor="black")
        )
        self.board_ax.set_title("board position")
        (self.actual_path_line,) = self.board_ax.plot([], [], color="black", linewidth=1.4, label="actual")
        (self.ball_dot,) = self.board_ax.plot([], [], "o", color="#d62728", markersize=7)
        (self.imagined_path_line,) = self.board_ax.plot([], [], "--", color="#17becf", linewidth=1.5, label="imagined")
        self.board_ax.legend(loc="upper right", fontsize=8)

        self.action_ax.set_title("executed action and current planned sequence")
        self.action_ax.set_ylim(-action_bound * 1.25, action_bound * 1.25)
        (self.action_x_line,) = self.action_ax.plot([], [], color="#1f77b4", label="exec ax")
        (self.action_y_line,) = self.action_ax.plot([], [], color="#ff7f0e", label="exec ay")
        (self.plan_x_line,) = self.action_ax.plot([], [], "--", color="#1f77b4", alpha=0.7, label="planned ax")
        (self.plan_y_line,) = self.action_ax.plot([], [], "--", color="#ff7f0e", alpha=0.7, label="planned ay")
        self.action_ax.legend(loc="upper right", fontsize=8)

        self.pos_ax.set_title("position / board angle")
        self.pos_ax.set_ylim(-max(half * 1.2, max_angle * 1.3), max(half * 1.2, max_angle * 1.3))
        (self.x_line,) = self.pos_ax.plot([], [], color="#1f77b4", label="x")
        (self.y_line,) = self.pos_ax.plot([], [], color="#ff7f0e", label="y")
        (self.theta_x_line,) = self.pos_ax.plot([], [], color="#9467bd", label="theta_x")
        (self.theta_y_line,) = self.pos_ax.plot([], [], color="#8c564b", label="theta_y")
        self.pos_ax.legend(loc="upper right", fontsize=8)

        self.vel_ax.set_title("velocity")
        self.vel_ax.set_ylim(-5.0, 5.0)
        (self.vx_line,) = self.vel_ax.plot([], [], color="#2ca02c", label="vx")
        (self.vy_line,) = self.vel_ax.plot([], [], color="#e377c2", label="vy")
        self.vel_ax.legend(loc="upper right", fontsize=8)

        for axis in (self.action_ax, self.pos_ax, self.vel_ax, self.board_ax):
            axis.grid(True, color="0.9", linewidth=0.8)

        self.image_ax.set_title("current render with decoded imagined path")
        self.image_ax.set_axis_off()
        self.fig.tight_layout()
        self.plt.show(block=False)

    def _window(self, values):
        if len(values) <= self.history_len:
            return np.asarray(values)
        return np.asarray(values[-self.history_len:])

    def update(self, episode, step, obs, plan, imagined_pixels, imagined_board):
        frame = obs['image'].transpose(1, 2, 0)
        if self.image_artist is None:
            self.image_artist = self.image_ax.imshow(frame)
        else:
            self.image_artist.set_data(frame)

        if imagined_pixels.size and np.isfinite(imagined_pixels).any():
            valid = np.isfinite(imagined_pixels).all(axis=1)
            self.image_future_line.set_data(imagined_pixels[valid, 0], imagined_pixels[valid, 1])
            self.image_future_dot.set_data(imagined_pixels[valid, 0], imagined_pixels[valid, 1])
        else:
            self.image_future_line.set_data([], [])
            self.image_future_dot.set_data([], [])

        states = self._window(episode["states"])
        actions = self._window(episode["actions"])
        state_offset = max(0, len(episode["states"]) - len(states))
        action_offset = max(0, len(episode["actions"]) - len(actions))

        if states.size:
            state_steps = np.arange(state_offset, state_offset + len(states))
            self.actual_path_line.set_data(states[:, 0], states[:, 1])
            self.ball_dot.set_data([states[-1, 0]], [states[-1, 1]])
            self.x_line.set_data(state_steps, states[:, 0])
            self.y_line.set_data(state_steps, states[:, 1])
            self.vx_line.set_data(state_steps, states[:, 2])
            self.vy_line.set_data(state_steps, states[:, 3])
            self.theta_x_line.set_data(state_steps, states[:, 4])
            self.theta_y_line.set_data(state_steps, states[:, 5])

        if imagined_board.size and np.isfinite(imagined_board).any():
            valid = np.isfinite(imagined_board).all(axis=1)
            self.imagined_path_line.set_data(imagined_board[valid, 0], imagined_board[valid, 1])
        else:
            self.imagined_path_line.set_data([], [])

        if actions.size:
            action_steps = np.arange(action_offset + 1, action_offset + 1 + len(actions))
            self.action_x_line.set_data(action_steps, actions[:, 0])
            self.action_y_line.set_data(action_steps, actions[:, 1])

        plan_np = plan.detach().cpu().numpy()
        future_steps = step + 1 + np.arange(plan_np.shape[0])
        self.plan_x_line.set_data(future_steps, plan_np[:, 0])
        self.plan_y_line.set_data(future_steps, plan_np[:, 1])

        right = max(step + plan_np.shape[0] + 1, self.history_len)
        left = max(0, right - self.history_len)
        for axis in (self.action_ax, self.pos_ax, self.vel_ax):
            axis.set_xlim(left, right)

        reward = float(np.sum(episode["rewards"])) if episode["rewards"] else 0.0
        self.fig.suptitle(f"episode={episode['index']} step={step} return={reward:.2f}")
        self.fig.canvas.draw_idle()
        self.plt.pause(self.pause_seconds)

    def capture_frame(self):
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])


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
    visualizer = None
    board_size = 1.0
    if args.online:
        config = find_ball_config(env)
        board_size = float(getattr(config, 'board_size', 1.0))
        max_angle = float(getattr(config, 'max_angle', 0.25))
        action_bound = float(
            max(
                np.max(np.abs(env.action_space.low)),
                np.max(np.abs(env.action_space.high)),
                1e-6,
            )
        )
        visualizer = OnlineMPCVisualizer(
            board_size,
            max_angle,
            action_bound,
            history_len=args.online_history,
            pause=args.online_pause,
        )
    online_gif_path = ''
    online_frames = []
    if args.online and not args.no_save_online_gif:
        online_gif_path = args.save_online_gif
        if not online_gif_path:
            online_gif_path = os.path.join(
                'outputs',
                time.strftime('latent_cem_gd_online_%Y%m%d_%H%M%S.gif'),
            )

    episode_rewards = []
    episode_lengths = []
    action_times = []
    optimizer_timings = []
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
        online_episode = {
            "index": episode,
            "states": [np.asarray(obs.get('state', np.zeros(6)), dtype=np.float32)],
            "actions": [],
            "rewards": [],
        }

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
            action_time = time.perf_counter() - action_start
            action_times.append(action_time)
            optimizer_timing = copy.deepcopy(getattr(optimizer, 'last_timing', {}))
            if optimizer_timing:
                optimizer_timings.append(optimizer_timing)
                if args.timing_details:
                    cem_time = optimizer_timing.get('cem_time', 0.0)
                    gd_time = optimizer_timing.get('gd_time', 0.0)
                    final_eval_time = optimizer_timing.get('final_eval_time', 0.0)
                    overhead = action_time - optimizer_timing.get('total_time', 0.0)
                    gd_stats = optimizer_timing.get('gd', {})
                    print(
                        f"timing episode={episode} step={steps} "
                        f"action={action_time:.4f}s cem={cem_time:.4f}s "
                        f"gd={gd_time:.4f}s final_eval={final_eval_time:.4f}s "
                        f"overhead={overhead:.4f}s "
                        f"gd_success={gd_stats.get('successful_steps', 0)} "
                        f"gd_unsuccess={gd_stats.get('unsuccessful_steps', 0)} "
                        f"gd_failures={gd_stats.get('line_search_failures', 0)}"
                    )
            action_np = action[0].detach().cpu().numpy()

            if visualizer is not None:
                online_episode["actions"].append(action_np.copy())
                frame_shape = obs['image'].transpose(1, 2, 0).shape
                imagined_pixels, imagined_board = decode_imagined_ball_path(
                    dreamer,
                    posterior,
                    plan,
                    args.deterministic_rollout,
                    board_size,
                    frame_shape,
                )
                current_state = np.asarray(obs.get('state', np.zeros(6)), dtype=np.float32)
                current_board = current_state[:2]
                current_pixel = board_to_pixel(current_board, frame_shape, board_size)
                imagined_board = np.vstack([current_board[None, :], imagined_board])
                imagined_pixels = np.vstack([current_pixel[None, :], imagined_pixels])
                visualizer.update(
                    online_episode,
                    steps,
                    obs,
                    plan,
                    imagined_pixels,
                    imagined_board,
                )
                if online_gif_path:
                    online_frames.append(visualizer.capture_frame())

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
            if visualizer is not None:
                online_episode["rewards"].append(float(reward))
                online_episode["states"].append(
                    np.asarray(obs.get('state', np.zeros(6)), dtype=np.float32)
                )
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
    if optimizer_timings:
        for prefix, values in [
            ('optimizer_total_time', timing_value(optimizer_timings, 'total_time')),
            ('cem_time', timing_value(optimizer_timings, 'cem_time')),
            ('gd_time', timing_value(optimizer_timings, 'gd_time')),
            ('final_eval_time', timing_value(optimizer_timings, 'final_eval_time')),
            ('cem_reward_eval_time', nested_timing_value(optimizer_timings, 'cem', 'reward_eval_time')),
            ('gd_objective_time', nested_timing_value(optimizer_timings, 'gd', 'objective_time')),
            ('gd_backward_time', nested_timing_value(optimizer_timings, 'gd', 'backward_time')),
            ('gd_adam_step_time', nested_timing_value(optimizer_timings, 'gd', 'adam_step_time')),
        ]:
            line = format_stats(prefix, values)
            if line:
                print(line)

        optimizer_total = timing_value(optimizer_timings, 'total_time')
        if action_times.size and optimizer_total.size == action_times.size:
            overhead = action_times - optimizer_total
            line = format_stats('planner_call_overhead_time', overhead)
            if line:
                print(line)

        gd_success = nested_timing_value(optimizer_timings, 'gd', 'successful_steps')
        gd_unsuccess = nested_timing_value(optimizer_timings, 'gd', 'unsuccessful_steps')
        gd_failures = nested_timing_value(optimizer_timings, 'gd', 'line_search_failures')
        if gd_success.size:
            print(
                f"gd_successful_steps_total={int(gd_success.sum())} "
                f"gd_unsuccessful_steps_total={int(gd_unsuccess.sum())} "
                f"gd_line_search_failures_total={int(gd_failures.sum())}"
            )
        cem_pop = nested_timing_value(optimizer_timings, 'cem', 'population_amount')
        cem_iters = nested_timing_value(optimizer_timings, 'cem', 'iterations')
        if cem_pop.size:
            print(
                f"cem_population_amount_first={int(cem_pop[0])} "
                f"cem_population_amount_last={int(cem_pop[-1])} "
                f"cem_iterations_first={int(cem_iters[0])} "
                f"cem_iterations_last={int(cem_iters[-1])}"
            )

    if online_gif_path and online_frames:
        save_episode_gif(online_frames, online_gif_path, args.online_record_fps)
        print(f"saved_online_gif={online_gif_path}")

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
