"""Smoke-test a DeepMind Control Suite environment.

Examples:
    python scripts/run_dm_control_env.py --domain cartpole --task swingup --steps 100
    python scripts/run_dm_control_env.py --domain walker --task walk --render --frames-out out
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def sample_bounded_action(spec, rng: np.random.Generator) -> np.ndarray:
    """Sample a random action from a dm_env bounded action spec."""

    low = np.asarray(spec.minimum, dtype=np.float32)
    high = np.asarray(spec.maximum, dtype=np.float32)
    return rng.uniform(low, high).astype(spec.dtype)


def describe_observation(observation: dict[str, np.ndarray]) -> str:
    parts = []
    for name, value in observation.items():
        array = np.asarray(value)
        parts.append(f"{name}: shape={array.shape}, dtype={array.dtype}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="cartpole", help="DM-Control domain name, e.g. cartpole, walker, cheetah")
    parser.add_argument("--task", default="swingup", help="DM-Control task name, e.g. swingup, walk, run")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render RGB frames from the physics camera.")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--frames-out", default=None, help="Optional directory for rendered PNG frames.")
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        choices=["egl", "osmesa", "glfw"],
        help="Set MUJOCO_GL before importing dm_control. Try osmesa on CPU-only headless machines.",
    )
    args = parser.parse_args()

    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    from dm_control import suite

    rng = np.random.default_rng(args.seed)
    env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs={"random": args.seed})
    action_spec = env.action_spec()
    obs_spec = env.observation_spec()
    frames_dir = Path(args.frames_out) if args.frames_out is not None else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v3 as iio
    else:
        iio = None

    print(f"Loaded dm_control suite task: {args.domain}/{args.task}")
    print(f"Action spec: shape={action_spec.shape}, dtype={action_spec.dtype}")
    print(f"Action bounds: low={np.asarray(action_spec.minimum)}, high={np.asarray(action_spec.maximum)}")
    print("Observation spec:")
    for name, spec in obs_spec.items():
        print(f"  {name}: shape={spec.shape}, dtype={spec.dtype}")

    timestep = env.reset()
    print(f"Initial observation: {describe_observation(timestep.observation)}")

    total_reward = 0.0
    rendered = 0
    for step in range(args.steps):
        action = sample_bounded_action(action_spec, rng)
        timestep = env.step(action)
        total_reward += float(timestep.reward or 0.0)

        should_render = args.render and step % max(args.render_every, 1) == 0
        if should_render:
            frame = env.physics.render(
                height=args.height,
                width=args.width,
                camera_id=args.camera_id,
            )
            rendered += 1
            if frames_dir is not None:
                assert iio is not None
                iio.imwrite(frames_dir / f"frame_{step:05d}.png", frame)

        if timestep.last():
            timestep = env.reset()

    print(f"Completed {args.steps} random steps; total_reward={total_reward:.3f}; rendered_frames={rendered}")


if __name__ == "__main__":
    main()
