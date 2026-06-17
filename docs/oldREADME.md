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
The default reward is bounded for reward-model regression: centered stable states are near `1`, edge states approach
`0` or below, and falling returns `-1` with `terminated=True`.

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

V0 Version
```bash
python scripts/collect_dataset.py   
--num-episodes 1000   --max-episode-steps 300   
--mode mixed   --seed 0   --out data/ball_balance_v0.npz
```

!!!V1 Version: this works well
```bash
python scripts/collect_dataset.py \
  --num-episodes 5000 \
  --max-episode-steps 300 \
  --mode mpc_cover \
  --pos-bound 0.25 \
  --vel-bound 0.20 \
  --angle-bound 0.12 \
  --target-bound 0.15 \
  --action-noise-std 0.04 \
  --seed 0 \
  --out data/ball_balance_mpc_v1.npz
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

## Reward-Aware PlaNet Workflow

This branch trains an RSSM world model with both an observation decoder and a reward model.
See [docs/reward_aware_planet_workflow.md](docs/reward_aware_planet_workflow.md) for the full workflow.

## Milestone 2: RSSM + Reward World Model

Before training, check that the environment and collected dataset are sane:

```bash
python scripts/check_milestone1.py --dataset data/ball_balance_v0.npz
```

Train a low-dimensional Gaussian RSSM with a reward head:

```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_v0.npz \
  --run-dir runs/rssm_ball_v0 \
  --seq-len 50 \
  --batch-size 128 \
  --epochs 100 \
  --reward-loss-weight 1.0 \
  --continuation-loss-weight 1.0
```

!!! Following parameters work well for mpc control
```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_mpc_v1.npz \
  --run-dir runs/rssm_ball_v3_long_wiz_reward_continual_model \
  --seq-len 200 \
  --batch-size 256 \
  --epochs 100 \
  --reward-loss-weight 1.0 \
  --continuation-loss-weight 1.0
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

## Milestone 3: Reward-Aware RSSM + CEM/MPC

Train an RSSM checkpoint first. MPC does not retrain the model; it only uses the frozen checkpoint for planning.
The default objective is `state_cost`, which uses hand-written costs over decoded observations.
For center stabilization, `--planning-objective reward` can instead maximize the learned environment reward.
For via-point tasks, keep `state_cost` unless you explicitly want the learned center reward to bias the plan.

Center stabilization:

```bash
python scripts/run_rssm_mpc_center.py \
  --checkpoint runs/rssm_ball_v3_long_wiz_reward_continual_model/checkpoints/best.pt \
  --num-episodes 5 \
  --max-steps 300 \
  --horizon 25 \
  --planning-objective state_cost
```

Fixed target stabilization:

```bash
python scripts/run_rssm_mpc_viapoint.py \
  --checkpoint runs/rssm_ball_v3_long_wiz_reward_continual_model/checkpoints/best.pt \
  --target-x 0.15 \
  --target-y -0.10 \
  --planning-objective state_cost
```

Center stabilization with learned reward planning:

```bash
python scripts/run_rssm_mpc_center.py \
  --checkpoint runs/rssm_ball_v3_long_wiz_reward_continual_model/checkpoints/best.pt \
  --num-episodes 5 \
  --max-steps 300 \
  --horizon 25 \
  --planning-objective reward
```

Evaluate many random initial conditions:

```bash
python scripts/eval_rssm_mpc.py \
  --checkpoint runs/rssm_ball_v3_long_wiz_reward_continual_model/checkpoints/best.pt \
  --mode center \
  --num-episodes 50 \
  --planning-objective state_cost
```

Compare PD against RSSM MPC:

```bash
python scripts/compare_pd_vs_rssm_mpc.py \
  --checkpoint runs/rssm_ball_v3_long_wiz_reward_continual_model/checkpoints/best.pt
```

Visualize a closed-loop MPC rollout:

```bash
python scripts/visualize_mpc_rollout.py \
  --checkpoint runs/rssm_ball_v3_long_wiz_reward_continual_model/checkpoints/best.pt \
  --out-dir runs/rssm_ball_v3_long_wiz_reward_continual_model/mpc_visualization
```

Run an online closed-loop visualizer with the gym ball window, sliding-window control input plot,
sliding-window state plot, automatic episode switching, random initial states, and final success rate:

```bash
python scripts/online_mpc_visualizer.py \
  --checkpoint runs/rssm_ball_v5_long_wiz_reward_continual_model/checkpoints/best.pt \
  --task viapoint \
  --num-episodes 5 \
  --max-steps 150 \
  --pos-bound 0.45 \
  --vel-bound 0.10 \
  --angle-bound 0.05 \
  --target-bound 0.18 \
  --planning-objective state_cost
```

Use `--task center` for center stabilization, `--task viapoint` for a random target per episode,
or `--task random` to randomly mix center and via-point episodes. Per-axis initial condition
bounds are also available through `--x-bound`, `--y-bound`, `--vx-bound`, `--vy-bound`,
`--theta-x-bound`, and `--theta-y-bound`.

MPC loop:

```text
real obs_t -> posterior update -> CEM samples future actions
-> RSSM prior rollout -> decode predicted observations
-> predict rewards and continuation from imagined latent states
-> objective on denormalized predictions/rewards with continuation-discounted return -> execute first action only
-> replan at next real step
```

Important details:

- CEM optimizes future action sequences under the learned world model.
- MPC executes only the first action and replans every environment step.
- Posterior update uses real observations up to the current step.
- Future rollout during planning uses RSSM prior imagination only.
- Candidate actions are normalized before entering RSSM.
- Predicted observations are denormalized before computing costs.
- Predicted rewards are denormalized before learned-reward planning.
- Predicted continuation discounts imagined learned-reward return after likely terminal states.
