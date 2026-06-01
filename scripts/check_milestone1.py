"""Diagnostic checks for the analytical environment and NPZ dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ball_rssm.envs import BallBalanceEnv
from ball_rssm.utils.plotting import OBS_LABELS, save_xy_diagnostics
from scripts.collect_dataset import pd_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    env = BallBalanceEnv()
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (6,)
        assert isinstance(info, dict)
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (6,)

        step_obs, reward, terminated, truncated, step_info = env.step(np.zeros(2, dtype=np.float32))
        assert step_obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)

        env.reset(seed=1)
        env.state[:] = 0.0
        for _ in range(20):
            zero_obs, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
            assert np.allclose(zero_obs, 0.0, atol=1e-6)

        env.reset(seed=2)
        for _ in range(100):
            rollout_obs, _, done, trunc, _ = env.step(env.action_space.sample())
            assert np.isfinite(rollout_obs).all()
            if done or trunc:
                break

        obs, _ = env.reset(seed=3)
        for _ in range(100):
            obs, _, done, trunc, _ = env.step(pd_action(obs))
            assert np.isfinite(obs).all()
            if done or trunc:
                break
    finally:
        env.close()

    data = np.load(args.dataset)
    obs_arr = data["obs"]
    action_arr = data["action"]
    assert obs_arr.ndim == 3
    assert action_arr.ndim == 3
    assert obs_arr.shape[2] == 6
    assert action_arr.shape[2] == 2
    assert obs_arr.shape[1] == action_arr.shape[1] + 1

    for key in ("obs", "action"):
        arr = data[key]
        assert np.isfinite(arr).all(), f"{key} contains NaN or Inf"

    dyn_std = obs_arr[:, :, :4].reshape(-1, 4).std(axis=0)
    assert np.all(dyn_std > 1e-4), f"Nontrivial dynamics check failed: std={dyn_std}"

    print("Observation stats:")
    for idx, label in enumerate(OBS_LABELS):
        values = obs_arr[:, :, idx]
        print(f"  {label:8s} min={values.min(): .5f} max={values.max(): .5f} mean={values.mean(): .5f} std={values.std(): .5f}")

    print("Action stats:")
    for idx, label in enumerate(["theta_x_cmd", "theta_y_cmd"]):
        values = action_arr[:, :, idx]
        print(f"  {label:12s} min={values.min(): .5f} max={values.max(): .5f} mean={values.mean(): .5f} std={values.std(): .5f}")

    out = Path(args.out) if args.out else Path(args.dataset).with_name("milestone1_xy_diagnostics.png")
    save_xy_diagnostics(obs_arr, out)
    print(f"Milestone 1 checks passed. Saved diagnostic plot to {out}")


if __name__ == "__main__":
    main()
