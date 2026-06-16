"""Smoke-test a DeepMind Control Suite environment.

Examples:
    python scripts/run_dm_control_env.py --domain cartpole --task swingup --steps 100
    python scripts/run_dm_control_env.py --domain walker --task walk --viewer --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs.dm_control import DMControlConfig, resolve_camera_fovy, resolve_camera_id


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


def configure_gl_backend(mujoco_gl: str | None) -> None:
    if mujoco_gl is None:
        return

    os.environ["MUJOCO_GL"] = mujoco_gl
    if mujoco_gl in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", mujoco_gl)


def import_suite():
    try:
        from dm_control import suite
    except AttributeError as exc:
        if "'NoneType' object has no attribute 'glGetError'" in str(exc):
            raise RuntimeError(
                "PyOpenGL could not initialize the selected offscreen renderer. "
                "On headless machines, prefer '--mujoco-gl egl'. "
                "The 'osmesa' backend also requires system OSMesa libraries."
            ) from exc
        raise
    return suite


def make_random_policy(action_spec, rng: np.random.Generator):
    def policy(_timestep):
        return sample_bounded_action(action_spec, rng)

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="cartpole", help="DM-Control domain name, e.g. cartpole, walker, cheetah")
    parser.add_argument("--task", default="swingup", help="DM-Control task name, e.g. swingup, walk, run")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Render RGB frames from the physics camera.")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--camera-fovy", type=float, default=None)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--frames-out", default=None, help="Optional directory for rendered PNG frames.")
    parser.add_argument("--viewer", action="store_true", help="Launch the live dm_control viewer.")
    parser.add_argument(
        "--viewer-policy",
        default="random",
        choices=["random", "default"],
        help="Policy used by the live viewer. 'default' uses dm_control's midpoint action.",
    )
    parser.add_argument("--viewer-width", type=int, default=1024)
    parser.add_argument("--viewer-height", type=int, default=768)
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        choices=["egl", "osmesa", "glfw"],
        help="Set MUJOCO_GL before importing dm_control. Use glfw for the live viewer.",
    )
    args = parser.parse_args()
    dm_config = DMControlConfig(
        domain=args.domain,
        task=args.task,
        obs_type="pixel" if args.render else "state",
        height=args.height,
        width=args.width,
        camera_id=args.camera_id,
        camera_fovy=args.camera_fovy,
        mujoco_gl=args.mujoco_gl,
    )
    camera_id = resolve_camera_id(dm_config)
    camera_fovy = resolve_camera_fovy(dm_config, camera_id)

    configure_gl_backend(args.mujoco_gl)

    try:
        suite = import_suite()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    rng = np.random.default_rng(args.seed)
    env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs={"random": args.seed})
    if camera_fovy is not None:
        env.physics.model.cam_fovy[camera_id] = float(camera_fovy)
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

    if args.viewer:
        from dm_control import viewer

        policy = None
        if args.viewer_policy == "random":
            policy = make_random_policy(action_spec, rng)
        print("Launching live dm_control viewer. Press Space to pause/run, Backspace to reset, F1 for help.")
        try:
            viewer.launch(
                env,
                policy=policy,
                title=f"dm_control: {args.domain}/{args.task}",
                width=args.viewer_width,
                height=args.viewer_height,
            )
        except RuntimeError as exc:
            if "Failed to create window" in str(exc):
                print(
                    "Error: could not open the GLFW viewer window. Run this from a desktop session "
                    "or enable X11/Wayland forwarding, and use '--mujoco-gl glfw'.",
                    file=sys.stderr,
                )
                raise SystemExit(1) from exc
            raise
        return

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
                camera_id=camera_id,
            )
            rendered += 1
            if frames_dir is not None:
                assert iio is not None
                iio.imwrite(frames_dir / f"frame_{step:05d}.png", frame)

        if timestep.last():
            timestep = env.reset()

    print(f"Completed {args.steps} random steps; total_reward={total_reward:.3f}; rendered_frames={rendered}")
    if rendered and frames_dir is not None:
        print(f"Saved rendered PNG frames to: {frames_dir}")


if __name__ == "__main__":
    main()
