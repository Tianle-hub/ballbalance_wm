"""Analytical ball-board balancing environment."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class BallBalanceConfig:
    """Physical, actuator, and reward parameters for BallBalanceEnv."""

    dt: float = 0.02
    g: float = 9.81
    board_size: float = 1.0
    max_angle: float = 0.25
    max_episode_steps: int = 300
    damping: float = 0.05
    angle_tau: float = 0.08
    init_pos_range: float = 0.45
    init_vel_range: float = 0.2
    reward_pos_weight: float = 1.0
    # TODO: test following weight
    reward_vel_weight: float = 0.10
    reward_angle_weight: float = 2.0
    reward_action_weight: float = 0.05
    reward_min: float = -1.0
    reward_max: float = 1.0
    fall_penalty: float = 1.0

    @classmethod
    def from_config(cls, config: "BallBalanceConfig | dict[str, Any] | None") -> "BallBalanceConfig":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, dict):
            raise TypeError("config must be None, BallBalanceConfig, or a dict")

        valid_names = {field.name for field in fields(cls)}
        unknown = set(config) - valid_names
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown BallBalanceConfig field(s): {unknown_list}")
        return replace(cls(), **config)


class BallBalanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """A simple analytical simulator for a ball rolling on a tilting board."""

    metadata = {"render_modes": [None, "human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        config: BallBalanceConfig | dict[str, Any] | None = None,
        render_fps: float | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        if render_fps is not None and render_fps <= 0.0:
            raise ValueError("render_fps must be positive")

        self.config = BallBalanceConfig.from_config(config)
        if self.config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.config.angle_tau <= 0.0:
            raise ValueError("angle_tau must be positive")
        if self.config.board_size <= 0.0:
            raise ValueError("board_size must be positive")
        if self.config.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        self.render_mode = render_mode
        self.render_fps = float(render_fps if render_fps is not None else self.metadata["render_fps"])
        self.state = np.zeros(6, dtype=np.float64)
        self.step_count = 0

        max_angle = np.float32(self.config.max_angle)
        half_board = np.float32(self.config.board_size / 2.0)
        obs_low = np.array(
            [-1.2 * half_board, -1.2 * half_board, -5.0, -5.0, -max_angle, -max_angle],
            dtype=np.float32,
        )
        obs_high = np.array(
            [1.2 * half_board, 1.2 * half_board, 5.0, 5.0, max_angle, max_angle],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-max_angle,
            high=max_angle,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._fig = None
        self._ax = None
        self._trajectory_artist = None
        self._ball_artist = None
        self._board_artist = None
        self._trajectory_xy: list[np.ndarray] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        cfg = self.config
        self.step_count = 0
        if options is not None and "state" in options:
            self.set_state(np.asarray(options["state"], dtype=np.float64))
        else:
            x = self.np_random.uniform(-cfg.init_pos_range, cfg.init_pos_range)
            y = self.np_random.uniform(-cfg.init_pos_range, cfg.init_pos_range)
            vx = self.np_random.uniform(-cfg.init_vel_range, cfg.init_vel_range)
            vy = self.np_random.uniform(-cfg.init_vel_range, cfg.init_vel_range)
            self.state = np.array([x, y, vx, vy, 0.0, 0.0], dtype=np.float64)

        obs = self._get_obs()
        self._trajectory_xy = [self.state[:2].copy()]
        info = self._make_info(
            acceleration=np.zeros(2, dtype=np.float32),
            action_clipped=np.zeros(2, dtype=np.float32),
        )
        return obs, info

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (6,):
            raise ValueError("state must have shape (6,)")
        self.state = state.copy()
        self._trajectory_xy = [self.state[:2].copy()]

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        action_arr = np.asarray(action, dtype=np.float32)
        action_clipped = np.clip(action_arr, self.action_space.low, self.action_space.high).astype(np.float32)
        theta_x_cmd, theta_y_cmd = action_clipped.astype(np.float64)

        x, y, vx, vy, theta_x, theta_y = self.state

        theta_x += (cfg.dt / cfg.angle_tau) * (theta_x_cmd - theta_x)
        theta_y += (cfg.dt / cfg.angle_tau) * (theta_y_cmd - theta_y)
        theta_x = float(np.clip(theta_x, -cfg.max_angle, cfg.max_angle))
        theta_y = float(np.clip(theta_y, -cfg.max_angle, cfg.max_angle))

        ax = (5.0 / 7.0) * cfg.g * np.sin(theta_y) - cfg.damping * vx
        ay = -(5.0 / 7.0) * cfg.g * np.sin(theta_x) - cfg.damping * vy

        vx += ax * cfg.dt
        vy += ay * cfg.dt
        x += vx * cfg.dt
        y += vy * cfg.dt
        self.state = np.array([x, y, vx, vy, theta_x, theta_y], dtype=np.float64)
        self._trajectory_xy.append(self.state[:2].copy())

        self.step_count += 1
        fallen = abs(x) > cfg.board_size / 2.0 or abs(y) > cfg.board_size / 2.0
        terminated = bool(fallen)
        truncated = bool(self.step_count >= cfg.max_episode_steps)

        if fallen:
            reward = -abs(cfg.fall_penalty)
        else:
            max_dist = cfg.board_size / 2.0
            pos_cost = (x**2 + y**2) / max(max_dist**2, 1e-12)
            reward = (
                1.0
                - cfg.reward_pos_weight * pos_cost
                - cfg.reward_vel_weight * (vx**2 + vy**2)
                - cfg.reward_angle_weight * (theta_x**2 + theta_y**2)
                - cfg.reward_action_weight * (theta_x_cmd**2 + theta_y_cmd**2)
            )
            reward = float(np.clip(reward, cfg.reward_min, cfg.reward_max))

        obs = self._get_obs()
        acceleration = np.array([ax, ay], dtype=np.float32)
        info = self._make_info(acceleration=acceleration, action_clipped=action_clipped)
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        self._ensure_render_objects()
        assert self._fig is not None
        assert self._ax is not None
        assert self._ball_artist is not None
        assert self._trajectory_artist is not None

        x, y = self.state[:2]
        if not self._trajectory_xy:
            self._trajectory_xy.append(self.state[:2].copy())
        trajectory = np.asarray(self._trajectory_xy, dtype=np.float64)
        self._trajectory_artist.set_data(trajectory[:, 0], trajectory[:, 1])
        self._ball_artist.set_data([x], [y])
        self._ax.set_title(f"step={self.step_count}  x={x:+.3f}  y={y:+.3f}")
        self._fig.canvas.draw()

        if self.render_mode == "human":
            import matplotlib.pyplot as plt

            plt.pause(1.0 / self.render_fps)
            return None

        if self.render_mode == "rgb_array":
            rgba = np.asarray(self._fig.canvas.buffer_rgba())
            return np.ascontiguousarray(rgba[:, :, :3])

        return None

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
        self._fig = None
        self._ax = None
        self._trajectory_artist = None
        self._ball_artist = None
        self._board_artist = None

    def _get_obs(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def _make_info(self, acceleration: np.ndarray, action_clipped: np.ndarray) -> dict[str, Any]:
        fallen = abs(self.state[0]) > self.config.board_size / 2.0 or abs(self.state[1]) > self.config.board_size / 2.0
        return {
            "state": self.state.copy(),
            "step_count": self.step_count,
            "fallen": bool(fallen),
            "acceleration": acceleration.astype(np.float32, copy=True),
            "action_clipped": action_clipped.astype(np.float32, copy=True),
        }

    def _ensure_render_objects(self) -> None:
        if self._fig is not None:
            return

        if self.render_mode == "rgb_array":
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(5, 5), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
        else:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        half = self.config.board_size / 2.0
        board = ax.add_patch(_rectangle_patch((-half, -half), self.config.board_size, self.config.board_size))
        (trajectory,) = ax.plot([], [], color="black", linewidth=1.6, alpha=0.85, zorder=2)
        (ball,) = ax.plot([], [], "o", color="tab:red", markersize=12, zorder=3)
        ax.axhline(0.0, color="0.85", linewidth=1)
        ax.axvline(0.0, color="0.85", linewidth=1)
        ax.set_xlim(-half * 1.15, half * 1.15)
        ax.set_ylim(-half * 1.15, half * 1.15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, color="0.9", linewidth=0.8)

        self._fig = fig
        self._ax = ax
        self._board_artist = board
        self._trajectory_artist = trajectory
        self._ball_artist = ball


def _rectangle_patch(xy: tuple[float, float], width: float, height: float):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, width, height, fill=False, linewidth=2, edgecolor="black")
