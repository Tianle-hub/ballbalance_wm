# Ball Balance DreamerV1

This project trains a low-dimensional PyTorch DreamerV1 agent for the analytical ball-board balancing environment.

The previous PlaNet-style workflow has been removed. Dynamics learning, actor learning, and value learning now run in the same training loop. Control uses the learned actor directly from the RSSM belief state; there is no CEM, CEM-GD, or separate planning module.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

The default task is center stabilization. The reward is bounded: centered stable states are near `1`, board-edge states approach `0` or below, and falling returns `-1` with `terminated=True`.

## Collect Bootstrap Data

Dreamer trains from replay batches. Start with a broad offline replay dataset:

```bash
python scripts/collect_dataset.py \
  --num-episodes 5000 \
  --max-episode-steps 300 \
  --mode coverage \
  --pos-bound 0.45 \
  --vel-bound 0.40 \
  --angle-bound 0.12 \
  --target-bound 0.15 \
  --action-noise-std 0.04 \
  --seed 0 \
  --out data/ball_balance_coverage_v0.npz
```

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

## Train Dreamer

```bash
python scripts/train_dreamer.py \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_v0 \
  --seq-len 200 \
  --batch-size 512 \
  --epochs 100 \
  --imagination-horizon 15 \
  --behavior-batch-size 4096
```

long imagine horizon trial:
```bash
python scripts/train_dreamer.py \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_v1 \
  --seq-len 200 \
  --batch-size 512 \
  --epochs 100 \
  --imagination-horizon 50 \
  --behavior-batch-size 4096
```


Each batch performs:

1. RSSM world-model update from reconstruction, reward, continuation, and KL losses.
2. Actor update by backpropagating imagined TD(lambda) returns through frozen RSSM dynamics.
3. Critic update toward target-critic TD(lambda) returns from imagined rollouts.

`--behavior-batch-size` caps how many posterior RSSM states are used as starts for actor/value imagination. The world model still trains on the full sequence batch; this cap only prevents long `seq-len` values from exploding the behavior update.

Checkpoints are written under `runs/dreamer_ball_v0/checkpoints/` and contain the world model, actor, critic, target critic, normalizer, optimizer states, and configs.

## Evaluate Actor

```bash
python scripts/run_dreamer_policy.py \
  --checkpoint runs/dreamer_ball_v0/checkpoints/best.pt \
  --num-episodes 20 \
  --max-steps 300 \
  --save-plots \
  --save-episodes
```

This runs closed-loop control as:

```text
real obs_t -> RSSM posterior update -> actor(action | latent belief) -> env step
```

No action sequence optimization or replanning is used.

## World-Model Diagnostics

Open-loop RSSM quality is still important because actor/value training happens in imagined latent rollouts.

```bash
python scripts/eval_rssm_prediction.py \
  --checkpoint runs/dreamer_ball_v0/checkpoints/best.pt \
  --dataset data/ball_balance_coverage_v0.npz \
  --context-len 10 \
  --horizon 50
```

```bash
python scripts/visualize_rssm_rollout.py \
  --checkpoint runs/dreamer_ball_v0/checkpoints/best.pt \
  --dataset data/ball_balance_coverage_v0.npz \
  --context-len 10 \
  --horizon 100 \
  --play
```

## Docs

- [DreamerV1 workflow](docs/dreamer_v1_workflow.md)
- [Architecture notes](docs/architecture.md)

## Test

```bash
pytest
```
