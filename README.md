# Ball RSSM Analytical Simulator

This repository contains the first simulation milestone for a PlaNet-style RSSM world model project: a small Gymnasium environment for ball-board balancing, plus rollout and dataset utilities.

The simulator is intentionally analytical and simple. It is meant for debugging offline trajectory collection and later RSSM training, not for high-fidelity rigid-body physics.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The milestone uses Gymnasium, NumPy, and Matplotlib. `pytest` is included only for tests.

## Environment

`BallBalanceEnv` follows the Gymnasium terminated/truncated API.

Observation:

```text
[x, y, vx, vy, theta_x, theta_y]
```

Action:

```text
[theta_x_cmd, theta_y_cmd]
```

The board is a square centered at the origin. The ball falls when `abs(x)` or `abs(y)` exceeds half the board size.

## Run Rollouts

Random smooth actions:

```bash
python scripts/run_random_env.py --episodes 3 --seed 0
```

PD center stabilization:

```bash
python scripts/run_pd_env.py --kp 0.8 --kd 0.25 --seed 0
```

## Collect Dataset

```bash
python scripts/collect_dataset.py \
  --num-episodes 100 \
  --max-episode-steps 300 \
  --mode mixed \
  --seed 0 \
  --out data/ball_balance_dataset.npz
```

Policy modes:

- `random_smooth`: low-pass filtered random actions.
- `pd`: center-stabilizing PD controller.
- `mixed`: per-episode mix of random smooth, PD, and noisy/offset PD.

Saved arrays:

```text
obs:        [num_episodes, max_episode_steps + 1, 6]
action:     [num_episodes, max_episode_steps, 2]
reward:     [num_episodes, max_episode_steps, 1]
terminated: [num_episodes, max_episode_steps, 1]
truncated:  [num_episodes, max_episode_steps, 1]
done:       [num_episodes, max_episode_steps, 1]
```

Convention:

```text
obs[:, t] + action[:, t] -> obs[:, t + 1]
```

If an episode ends early, remaining entries are padded by repeating the final observation and marking `done`.

## Plot Dataset

```bash
python scripts/plot_dataset.py data/ball_balance_dataset.npz --episode 0
```

## Test

```bash
pytest
```
