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

## Milestone 2: RSSM World Model

Before training, check that the environment and collected dataset are sane:

```bash
python scripts/check_milestone1.py --dataset data/ball_balance_v0.npz
```

First collect offline data by

```bash
python scripts/collect_dataset.py   --num-episodes 1000   --max-episode-steps 300   --mode mixed   --seed 0   --out data/ball_balance_v0.npz
```


Train a low-dimensional Gaussian RSSM:

```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_v0.npz \
  --run-dir runs/rssm_ball_v0 \
  --seq-len 50 \
  --batch-size 128 \
  --epochs 100
```

Evaluate posterior reconstruction, one-step prior prediction, and open-loop rollout:

```bash
python scripts/eval_rssm_prediction.py \
  --checkpoint runs/rssm_ball_v0/checkpoints/best.pt \
  --dataset data/ball_balance_v0.npz
```

Visualize true observations, posterior reconstruction, and prior open-loop rollout:

```bash
python scripts/visualize_rssm_rollout.py \
  --checkpoint runs/rssm_ball_v0/checkpoints/best.pt \
  --dataset data/ball_balance_v0.npz
```

Play the decoded imagined future rollout on the board, similar to the environment render:

```bash
python scripts/visualize_rssm_rollout.py \
  --checkpoint runs/rssm_ball_v0/checkpoints/best.pt \
  --dataset data/ball_balance_v0.npz \
  --context-len 10 \
  --horizon 100 \
  --play
```

You can also save the same playback as a GIF:

```bash
python scripts/visualize_rssm_rollout.py \
  --checkpoint runs/rssm_ball_v0/checkpoints/best.pt \
  --dataset data/ball_balance_v0.npz \
  --save-gif runs/rssm_ball_v0/figures/imagined_rollout.gif
```

Evaluate the trained model on fresh bounded random initial conditions rather than replaying an existing dataset episode:

```bash
python scripts/eval_rssm_random_ic_rollout.py \
  --checkpoint runs/rssm_ball_v0/checkpoints/best.pt \
  --num-episodes 128 \
  --context-len 20 \
  --horizon 100 \
  --pos-bound 0.25 \
  --vel-bound 0.20 \
  --angle-bound 0.12 \
  --action-mode mixed
```

This creates true environment rollouts from random bounded initial states `[x, y, vx, vy, theta_x, theta_y]`, then compares decoded RSSM prior imagination against the true future ball trajectory. It saves `random_ic_metrics.json`, error curves, and trajectory plots under the checkpoint run directory.

Prediction modes:

- Posterior reconstruction uses the current observation `obs_t` to infer `z_t`, then decodes the latent state. It checks representation quality but can hide weak dynamics.
- One-step prior prediction predicts `z_t` from the previous latent state and previous action, then compares against `obs_t`.
- Multi-step open-loop prediction warms up the posterior for a context window, then rolls forward using only future actions and the RSSM prior.
- Open-loop prediction is the important dynamics test because future observations are not provided to the model.
