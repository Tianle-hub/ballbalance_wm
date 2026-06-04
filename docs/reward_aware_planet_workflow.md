# Reward-Aware PlaNet Workflow

This branch trains the RSSM as a reward-aware world model:

```text
obs_t, action_t -> RSSM posterior/prior latent state
latent state -> observation decoder
latent state -> reward model
```

The reward model follows the same idea as PlaNet/Dreamer-style RSSMs: concatenate deterministic state `h`
and stochastic state `z`, then predict scalar reward with an MLP. The implementation stays in the repo's
low-dimensional style while using `Buffer` as the data collection, NPZ I/O, and training dataset entry point.

## 1. Collect Data

The known-good v1 collection recipe remains the default starting point:

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
  --out data/ball_balance_mpc_v1_reward_normalized.npz
```

The dataset stores true environment rewards:

```text
obs:    [N, T + 1, 6]
action: [N, T, 2]
reward: [N, T, 1]
done:   [N, T, 1]
```

`reward[:, t]` is the true reward from applying `action[:, t]` and reaching `obs[:, t + 1]`.

## 2. Train RSSM + Reward Model

```bash
python scripts/train_rssm.py \
  --dataset data/ball_balance_mpc_v1_reward_normalized.npz \
  --run-dir runs/rssm_ball_v2_long_wiz_reward_model \
  --seq-len 200 \
  --batch-size 256 \
  --epochs 100
```

Rerunning the same command resumes automatically from `checkpoints/last.pt` when it exists, or
from the older `checkpoints/latest.pt` name for existing runs. `--epochs` is the total target epoch
count, so a checkpoint saved at epoch 37 continues with epoch 38 and stops after epoch 100. Use
`--no-resume` to intentionally start from scratch, or `--resume-from path/to/checkpoint.pt` to choose
a specific checkpoint.

Useful knobs:

```bash
--reward-loss-weight 1.0
--beta-kl 1.0
--free-nats 1.0
```

Training normalizes observation, action, and reward statistics from the training split. The loss is:

```text
total_loss = observation_reconstruction_mse
           + beta_kl * free_nats_clamped_kl
           + reward_loss_weight * reward_prediction_mse
```

The checkpoint contains one model state dict with encoder, RSSM dynamics, observation decoder, and reward
model weights.

## 3. Evaluate Model Prediction

```bash
python scripts/eval_rssm_prediction.py \
  --checkpoint runs/rssm_ball_v2_long_wiz_reward_model/checkpoints/best.pt \
  --dataset data/ball_balance_mpc_v1_reward_normalized.npz \
  --context-len 10 \
  --horizon 50
```

This writes:

```text
runs/rssm_ball_v2_long_wiz_reward_model/eval_metrics.json
runs/rssm_ball_v2_long_wiz_reward_model/open_loop_mse_curve.png
runs/rssm_ball_v2_long_wiz_reward_model/open_loop_reward_mse_curve.png
```

## 4. Plan With MPC

Planning objectives:

- `state_cost`: default; uses hand-written costs on decoded observations.
- `reward`: maximizes learned environment reward from the reward model.
- `hybrid`: state cost plus learned reward objective.

For center stabilization, all three objectives are meaningful because the environment reward is center
stabilization reward:

```bash
python scripts/run_rssm_mpc_center.py \
  --checkpoint runs/rssm_ball_v2_long_wiz_reward_model/checkpoints/best.pt \
  --num-episodes 5 \
  --max-steps 300 \
  --horizon 25 \
  --planning-objective reward
```

For via-point tasks, keep `state_cost` unless you explicitly want to bias toward the environment's center
reward. The true env reward does not include arbitrary via-point targets.

```bash
python scripts/online_mpc_visualizer.py \
  --checkpoint runs/rssm_ball_v2_long_wiz_reward_model/checkpoints/best.pt \
  --task viapoint \
  --num-episodes 5 \
  --max-steps 150 \
  --pos-bound 0.25 \
  --vel-bound 0.10 \
  --angle-bound 0.05 \
  --target-bound 0.18 \
  --planning-objective state_cost
```

Center reward-planning visualizer:

```bash
python scripts/online_mpc_visualizer.py \
  --checkpoint runs/rssm_ball_v2_long_wiz_reward_model/checkpoints/best.pt \
  --task center \
  --num-episodes 5 \
  --max-steps 150 \
  --planning-objective reward
```

## 5. Interpretation

The reward model is trained from true environment reward, not from task labels. In this environment that reward
penalizes distance from the origin, velocity, board angle, action magnitude, and falling. It is not a universal
goal-conditioned reward. To make learned reward planning work for random via-points, the dataset and model would
need target-conditioned observations or a target-conditioned reward model.
