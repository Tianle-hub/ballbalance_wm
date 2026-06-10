# Ball Balance Dreamer V1/V2

This project trains a low-dimensional PyTorch Dreamer agent for the analytical ball-board balancing environment. Use `--dreamer-version v1` for the continuous Gaussian RSSM or `--dreamer-version v2` for the discrete categorical RSSM with KL balancing.

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

By default, `BallBalanceEnv` returns rendered pixel observations:

```text
[3, 64, 64]
```

For low-dimensional experiments, construct the environment with `config={"observation_mode": "state"}`. The state observation and `info["state"]` are:

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
obs:        [num_episodes, max_episode_steps + 1, 3, 64, 64]
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

## Train Dreamer Offline

```bash
python scripts/train_dreamer.py \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_v0 \
  --dreamer-version v1 \
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

## Train Dreamer Online

Online training starts with seed replay collected into the buffer, then alternates model/actor/value updates with actor-driven data collection, following the loop used in `dreamer-torch-v1v2`.

```bash
python scripts/train_dreamer.py \
  --train-mode online \
  --run-dir runs/dreamer_ball_online_v2_more_data \
  --dreamer-version v2 \
  --seed-episodes 1000 \
  --buffer-episodes 20000 \
  --max-episode-steps 300 \
  --seed-policy-mode coverage \
  --pos-bound 0.45 \
  --vel-bound 0.40 \
  --angle-bound 0.12 \
  --target-bound 0.15 \
  --action-noise-std 0.04 \
  --online-iterations 300 \
  --update-steps 150 \
  --collect-episodes 25 \
  --exploration-noise 0.3 \
  --exploration-decay 0.995 \
  --min-exploration-noise 0.05 \
  --seq-len 150 \
  --batch-size 256
```

If `--dataset` is supplied in online mode, that replay is used as the seed buffer. Otherwise the script collects `--seed-episodes` using `--seed-policy-mode`. Online replay is saved to `runs/.../replay/latest.npz` so resumed runs can continue from the latest actor-collected buffer.

Each batch performs:

1. RSSM world-model update from reconstruction, reward, continuation, and KL losses.
2. Actor update by backpropagating imagined TD(lambda) returns through frozen RSSM dynamics.
3. Critic update toward target-critic TD(lambda) returns from imagined rollouts.

V1 uses the Gaussian posterior/prior KL with free nats. V2 uses a straight-through categorical latent state and the reference KL balance controlled by `--kl-alpha`.

`--behavior-batch-size` caps how many posterior RSSM states are used as starts for actor/value imagination. The world model still trains on the full sequence batch; this cap only prevents long `seq-len` values from exploding the behavior update.

Checkpoints are written under `runs/dreamer_ball_v0/checkpoints/` and contain the world model, actor, critic, target critic, normalizer, optimizer states, and configs.

## Evaluate Actor

```bash
python scripts/run_dreamer_policy.py \
  --checkpoint runs/dreamer_ball_online_v2/checkpoints/best.pt \
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
