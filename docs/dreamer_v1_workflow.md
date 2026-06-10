# Dreamer Workflow

This branch can train DreamerV1-style or DreamerV2-style actor-critic agents over
an RSSM world model.

Use `--dreamer-version v1` for continuous Gaussian latents, free-nats KL, dynamics
backpropagation actor gradients, and external Gaussian collection noise. Use
`--dreamer-version v2` for categorical straight-through latents, KL balancing,
REINFORCE actor gradients, and stochastic policy collection without external
action noise. The behavior choices are controlled by `--actor-gradient auto` and
`--exploration-mode auto`; pass `dynamics`, `reinforce`, `noise`, or
`policy_entropy` explicitly for ablations.

## Main Loop

`scripts/train_dreamer.py` is the main entry point. For every replay batch it:

1. Normalizes observations, actions, and rewards with checkpointed replay statistics.
2. Updates the RSSM world model with reconstruction, reward, continuation, and KL losses.
3. Infers posterior latent states from the same batch.
4. Starts latent imagination from those posterior states.
5. Updates the actor by maximizing imagined TD(lambda) returns.
6. Updates the critic toward target-critic TD(lambda) returns.
7. Soft-updates the target critic.

The actor and critic are not trained in a separate stage. The world model, actor, and critic checkpoints are saved together.

## Data

Collect a broad bootstrap replay dataset:

```bash
python scripts/collect_dataset.py \
  --num-episodes 5000 \
  --max-episode-steps 300 \
  --mode coverage \
  --pos-bound 0.25 \
  --vel-bound 0.20 \
  --angle-bound 0.12 \
  --target-bound 0.15 \
  --action-noise-std 0.04 \
  --seed 0 \
  --out data/ball_balance_coverage_v0.npz
```

The `coverage` mode mixes stabilizing and exploratory actions so the world model sees recoverable states near the center and broader board states.

## Training

```bash
python scripts/train_dreamer.py \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_v0 \
  --dreamer-version v1 \
  --seq-len 50 \
  --batch-size 128 \
  --epochs 100 \
  --world-lr 3e-4 \
  --actor-lr 8e-5 \
  --critic-lr 8e-5 \
  --imagination-horizon 15 \
  --behavior-batch-size 4096 \
  --discount 0.99 \
  --lambda 0.95
```

Useful metrics:

- `recon_loss`: normalized observation reconstruction loss.
- `reward_loss`: normalized reward prediction loss.
- `continuation_loss`: terminal prediction loss.
- `kl_loss` and `raw_kl`: posterior-prior RSSM divergence.
- `actor_objective`: imagined return objective before sign flip.
- `critic_loss`: value regression loss on imagined features.
- `behavior_start_count`: posterior starts available from the full world-model batch.
- `behavior_sample_count`: posterior starts actually used for actor/value imagination.
- `val_open_loop/obs_h*`: multi-step prior prediction quality.

For long world-model sequences, keep `--seq-len` large and use `--behavior-batch-size` to cap actor/value starts. This keeps dynamics training on the full batch while making the behavior update memory-bounded.

## Policy Evaluation

```bash
python scripts/run_dreamer_policy.py \
  --checkpoint runs/dreamer_ball_v0/checkpoints/best.pt \
  --num-episodes 20 \
  --max-steps 300 \
  --save-plots \
  --save-episodes
```

The evaluator stores `metrics.json` plus optional episode NPZ files and plots. The closed-loop policy does one actor forward pass per environment step.

## Removed PlaNet Control Path

The old control path trained a world model first and then controlled the environment with an external optimizer over candidate action sequences. That path has been removed:

- no standalone planning package;
- no receding-horizon optimizer;
- no CEM or CEM-GD;
- no frozen-model control stage.

Dreamer control is the actor learned from imagined RSSM rollouts.
