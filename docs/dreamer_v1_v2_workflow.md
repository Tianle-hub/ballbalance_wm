# Dreamer V1/V2 Workflow

This repository now uses one Dreamer training path with selectable V1-style and
V2-style world-model behavior. The file name is historical; the current workflow
is controlled by `scripts/train_dreamer.py` and `--dreamer-version`.

There is no separate PlaNet controller. The world model, actor, critic, and
target critic are trained together, and closed-loop control uses the learned
actor directly from the RSSM belief state.

For concrete starting values and tuning recipes, see
[dreamer_tuning.md](dreamer_tuning.md).

## PlaNet, Dreamer V1, And Dreamer V2 In This Codebase

The three agents share the RSSM idea, but they differ in how actions are chosen,
how behavior is learned, and which robustness tricks are important.

| Agent | World model | Action selection | Behavior learning | Current codebase status |
| --- | --- | --- | --- | --- |
| PlaNet | RSSM dynamics with reconstruction and reward prediction | Online planning with CEM over action sequences | No learned actor or critic is required for control | Not implemented as a separate controller. The code keeps open-loop prediction utilities but controls with a learned actor. |
| Dreamer V1 | Continuous diagonal Gaussian stochastic state `z` plus deterministic GRU state `h` | Learned actor over RSSM features `[h, z]` | Actor and critic train on imagined latent rollouts with TD(lambda) returns | Implemented through `--dreamer-version v1`. Uses V1-style free nats on KL. |
| Dreamer V2 | Straight-through categorical stochastic state, KL balancing, larger discrete feature space | Learned actor over RSSM features `[h, z]` | Atari used REINFORCE-style actor gradients; continuous control usually prefers dynamics gradients | Implemented through `--dreamer-version v2`. The repo supports `dynamics`, `reinforce`, and `both` actor-gradient modes. |

The code path is now organized around the same conceptual split:

```text
world_model.py -> observation, reward, continuation, and KL losses
behavior.py    -> actor loss, critic loss, TD(lambda), discount weights
trainer.py     -> replay batches, optimizer steps, collection, checkpoints
```

This is different from PlaNet's control story. PlaNet uses the world model for
planning at action time. Dreamer V1 and V2 use the world model for imagination
during training, then deploy the actor directly.

## Robustness Techniques

The DreamerV2 paper reports several changes that helped on Atari. In this repo,
the same ideas map to explicit config knobs, but their best values depend on
whether the task is continuous control, pixel control, or a discrete-action
benchmark.

| Technique | DreamerV1 baseline | DreamerV2 change | Current code knob | Practical note |
| --- | --- | --- | --- | --- |
| Categorical latents | Continuous Gaussian RSSM latents | Straight-through categorical latents | `--dreamer-version v2`, `--stoch-dim`, `--discrete-classes` | Increases latent capacity. Useful for V2 and pixels, but raises feature size to `stoch_dim * discrete_classes`. |
| KL balancing | Free nats on mean posterior-prior KL | Separately balance representation and dynamics KL terms | `--kl-balance`, `--kl-free`, `--kl-free-avg`, `--kl-forward` | Helps the temporal prior learn useful dynamics instead of relying too much on posterior correction. |
| Actor gradient choice | Dynamics backpropagation through imagined rollouts | Atari favored REINFORCE-only; continuous control favored dynamics gradients | `--actor-gradient dynamics|reinforce|both|auto` | For DM-Control continuous actions, keep `auto` or `dynamics` first. Try `reinforce` mostly for discrete-action experiments. |
| Model size | Smaller V1 models | Larger networks and CNN depths | `--deter-dim`, `--stoch-dim`, `--hidden-dim`, `--embed-dim`, `--cnn-depth` | Pixel tasks usually need much larger models than state tasks. |
| Policy entropy | External action noise is common for continuous control | Entropy regularization supports exploration and imagination | `--actor-entropy-scale`, `--exploration-mode policy_entropy` | V2 defaults to policy-entropy exploration. V1 defaults to external Gaussian noise. |
| Action normalization | Environment-dependent | Bounded actions normalized to `[-1, 1]` | DM-Control `NormalizeActionWrapper` | This is already used for DM-Control replay, actor training, and evaluation. |
| Episode reset handling | Depends on environment wrapper | Driver APIs pass reset markers | `is_first` in `StreamReplay` and `RSSM.observe()` | Prevents training the RSSM through artificial reset boundaries. |
| Continuation prediction | Often separate from reward | Discount/continuation head controls imagined horizon | `continuation_loss_weight` and Bernoulli continuation head | Important for imagined rollouts and terminal transitions. |
| Prior stability | Single prior often works on simpler tasks | V2 reference can use ensemble dynamics and normalization | `--rssm-ensemble`, `--layer-norm` | Try these before expecting harder pixel locomotion to match reference behavior. |

Changes that the DreamerV2 paper tried but did not keep should be treated as
ablation ideas, not defaults. In this repo, prefer changes that are already
exposed as flags before adding new architectural variations.

## Network Setting Differences

The most important architecture difference is not just "V1 versus V2"; it is
also "state observations versus pixel observations."

| Component | PlaNet-style view | Dreamer V1 in this repo | Dreamer V2 in this repo |
| --- | --- | --- | --- |
| RSSM deterministic state | GRU memory `h` | Same GRU memory `h` | Same GRU memory `h` |
| RSSM stochastic state | Usually continuous latent | Continuous Gaussian `z` with `stoch_dim` | Straight-through categorical `z` with `stoch_dim * discrete_classes` flattened features |
| Prior | Dynamics prior predicts next latent from `h` | Single prior by default | Configurable prior ensemble through `--rssm-ensemble`; still defaults to `1` |
| Encoder/decoder for vectors | MLP | MLP | MLP |
| Encoder for pixels | CNN | Configurable CNN encoder | Configurable CNN encoder with V2-friendly depth |
| Decoder for pixels | CNN transpose decoder | `1x1 -> 64x64` transpose-conv path when using default kernels | Same decoder path; tune `cnn_depth`, latent dims, and loss weights |
| Reward head | Predicts reward for planning | Fixed-std Normal reward likelihood by default | Same head; optional reward transforms |
| Continuation head | Sometimes absent or implicit | Bernoulli continuation likelihood | Same head |
| Actor/critic | Not needed for CEM control | Learned actor and critic | Learned actor and critic |
| Exploration | Planner samples candidate action sequences | External action noise by default | Policy entropy by default |

## Mode Matrix

`--dreamer-version` selects the RSSM latent family and KL loss:

| Option | RSSM latent | KL loss | Auto actor gradient | Auto online exploration |
| --- | --- | --- | --- | --- |
| `v1` | continuous Gaussian `z` with `stoch_dim` features | mean posterior-prior KL with `free_nats` | `dynamics` | `noise` |
| `v2` | straight-through categorical `z` with `stoch_dim * discrete_classes` features | Dreamer V2 KL with `kl_forward`, `kl_balance`, `kl_free`, and `kl_free_avg` | action-space dependent | `policy_entropy` |

The mode knobs are independent after initialization:

- `--actor-gradient auto` resolves to `reinforce` for discrete actors and
  `dynamics` for continuous actors. Ball balance currently uses a continuous
  action actor, so both V1 and V2 resolve to `dynamics`.
- `--actor-gradient dynamics` backpropagates imagined TD(lambda) returns through
  the frozen world-model transition and critic output into actor actions.
- `--actor-gradient reinforce` samples actions for a score-function objective,
  detaches imagined actions, and weights log-probabilities by imagined
  advantages.
- `--actor-gradient both` combines the score-function term and the
  pathwise/dynamics backpropagation term, matching the DreamerV2 Eq. 6 style
  actor objective.
- `--exploration-mode auto` resolves to `noise` for V1 and `policy_entropy` for
  V2.
- `--exploration-mode noise` keeps stochastic actor collection and adds scheduled
  Gaussian noise in normalized action space.
- `--exploration-mode policy_entropy` samples from the actor distribution but
  suppresses the external Gaussian noise schedule.

This makes hybrid ablations valid. For example, `--dreamer-version v2
--actor-gradient reinforce` keeps categorical latents and KL balancing while
forcing the Atari-style score-function actor gradient.

## Shared Batch Update

Every offline epoch or online update step samples replay sequences and applies
the same Dreamer batch update:

1. Normalize observations, actions, and rewards from replay statistics.
2. Train the RSSM world model on real sequences:
   - observation reconstruction;
   - reward prediction;
   - continuation prediction;
   - V1 or V2 KL loss.
3. Re-run posterior inference with the updated world model.
4. Flatten posterior RSSM states across batch and time, dropping the final state.
5. Use those flattened states as actor/critic imagination starts.
6. Imagine latent rollouts with the actor for `--imagination-horizon` steps.
7. Update the actor from imagined TD(lambda) returns plus entropy bonus.
8. Update the critic toward target-critic TD(lambda) returns.
9. Soft-update the target critic using `--target-tau`.

World-model training always uses the full replay sequence batch. Actor and
critic imagination starts from the flattened posterior state batch.

## Replay Data

Offline Dreamer needs a replay dataset. Online Dreamer either starts from
`--dataset`, resumes `runs/.../replay/latest.npz`, or collects seed episodes with
`--seed-policy-mode`.

Collect a broad bootstrap dataset:

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

The saved replay convention is:

```text
obs[:, t] + action[:, t] -> obs[:, t + 1]
```

Arrays include `obs`, `action`, `reward`, `terminated`, `truncated`, and `done`.
Termination masks keep the terminal transition trainable and mask artificial
post-done padding.

## Offline Training

V1-style continuous RSSM:

```bash
python scripts/train_dreamer.py \
  --train-mode offline \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_offline_v1 \
  --dreamer-version v1 \
  --seq-len 200 \
  --batch-size 512 \
  --epochs 100 \
  --imagination-horizon 15 \
  ```

V2-style categorical RSSM:

```bash
python scripts/train_dreamer.py \
  --train-mode offline \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_offline_v2 \
  --dreamer-version v2 \
  --seq-len 200 \
  --batch-size 512 \
  --epochs 100 \
  --imagination-horizon 15 \
  ```

Training resumes by default from `run-dir/checkpoints/last.pt` or
`run-dir/checkpoints/latest.pt`. Use `--no-resume` to start a fresh run, or
`--resume-from PATH` to load an explicit Dreamer checkpoint.

## Online Training

Online mode alternates update steps with actor-driven collection:

1. Build the initial replay buffer from `--dataset`, resumed replay, or seed
   collection.
2. Run `--train-steps` Dreamer batch updates.
3. Validate and write checkpoints.
4. Collect `--collect-episodes` real environment episodes with the current actor.
5. Append those episodes to replay and save `runs/.../replay/latest.npz`.

V2 online default:

```bash
python scripts/train_dreamer.py \
  --train-mode online \
  --run-dir runs/dreamer_ball_online_v2 \
  --dreamer-version v2 \
  --seed-episodes 200 \
  --buffer-episodes 20000 \
  --max-episode-steps 300 \
  --seed-policy-mode coverage \
  --pos-bound 0.45 \
  --vel-bound 0.40 \
  --angle-bound 0.12 \
  --target-bound 0.15 \
  --action-noise-std 0.04 \
  --online-iterations 300 \
  --train-steps 150 \
  --collect-episodes 50 \
  --seq-len 200 \
  --batch-size 256
```

Coverage-initialized V1 online:

```bash
python scripts/train_dreamer.py \
  --train-mode online \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_online_v1 \
  --dreamer-version v1 \
  --actor-gradient dynamics \
  --exploration-mode noise \
  --buffer-episodes 20000 \
  --max-episode-steps 300 \
  --online-iterations 300 \
  --train-steps 150 \
  --collect-episodes 50 \
  --exploration-noise 0.3 \
  --exploration-decay 0.995 \
  --min-exploration-noise 0.05 \
  --seq-len 200 \
  --batch-size 256
```

V2 latents with V1-style dynamics actor gradients:

```bash
python scripts/train_dreamer.py \
  --train-mode online \
  --dataset data/ball_balance_coverage_v0.npz \
  --run-dir runs/dreamer_ball_online_v2_dynamics \
  --dreamer-version v2 \
  --actor-gradient dynamics \
  --exploration-mode policy_entropy \
  --buffer-episodes 20000 \
  --max-episode-steps 300 \
  --online-iterations 300 \
  --train-steps 150 \
  --collect-episodes 50 \
  --seq-len 200 \
  --batch-size 256
```

In `policy_entropy` exploration, `--exploration-noise`,
`--exploration-decay`, and `--min-exploration-noise` are still stored in config
and metrics, but the effective external noise used for collection is `0.0`.

## Important Metrics

- `total_loss`: world-model loss used for validation and best-checkpoint
  selection.
- `recon_loss`: normalized observation reconstruction loss.
- `reward_loss`: normalized reward prediction loss.
- `continuation_loss`: binary continuation prediction loss.
- `kl_loss`: version-specific KL term after free-nats or KL balancing.
- `raw_kl`: unbalanced posterior-prior KL before V1 free-nats.
- `dynamics_kl_loss` and `representation_kl_loss`: V2 KL-balance components;
  equal to `raw_kl` in V1 logging.
- `posterior_std_mean` and `prior_std_mean`: V1 Gaussian RSSM statistics.
- `posterior_entropy_mean` and `prior_entropy_mean`: V2 categorical RSSM
  statistics.
- `actor_objective`: actor optimization objective before sign flip.
- `actor_imagined_return_objective`: discounted imagined return used for
  diagnostics.
- `actor_dynamics_objective`: nonzero when using `dynamics` or `both` actor
  gradients.
- `actor_reinforce_objective`: nonzero when using `reinforce` or `both` actor
  gradients.
- `actor_gradient_reinforce`: `1.0` for REINFORCE-only, `0.0` otherwise.
- `actor_gradient_both`: `1.0` for the DreamerV2 Eq. 6 style mixed objective,
  `0.0` otherwise.
- `actor_entropy`: actor entropy bonus term.
- `critic_loss`: value regression loss on imagined features.
- `imagined_reward_mean` and `imagined_continue_mean`: rollout-model
  diagnostics for behavior learning.
- `behavior_start_count`: posterior starts used for actor/value imagination.
- `val_prior_rollout/obs_h*` and `val_prior_rollout/reward_h*`: prior rollout
  prediction losses after a posterior context window.
- `collect/*`: online-only collection rewards, replay size, exploration mode,
  configured noise, and effective noise.

## Checkpoints

Dreamer checkpoints are written under `runs/.../checkpoints/` as `last.pt`,
`latest.pt`, and `best.pt`. They contain:

- world model, actor, critic, and target critic state dicts;
- world, actor, and critic optimizer state dicts;
- normalizer state;
- world-model, actor, critic, and Dreamer config dictionaries;
- `train_args`, `epoch`, `best_val_loss`, and `replay_episodes`.

For compatibility with world-model diagnostic scripts, checkpoints also keep
`model_state_dict`, `optimizer_state_dict`, and `config` aliases for the world
model.

## Policy Evaluation

```bash
python scripts/run_dreamer_policy.py \
  --checkpoint runs/dreamer_ball_online_v2/checkpoints/best.pt \
  --num-episodes 20 \
  --max-steps 300 \
  --save-plots \
  --save-episodes
```

The evaluator writes `metrics.json` plus optional episode NPZ files and plots.
The closed-loop policy is:

```text
real obs_t -> RSSM posterior update -> actor(action | h_t, z_t) -> env step
```

By default the evaluator uses the actor mode. Pass `--stochastic` to sample from
the actor distribution instead.

## World-Model Diagnostics

Open-loop RSSM quality still matters because actor and critic training happen in
imagined latent rollouts.

```bash
python scripts/eval_rssm_prediction.py \
  --checkpoint runs/dreamer_ball_offline_v1/checkpoints/best.pt \
  --dataset data/ball_balance_coverage_v0.npz \
  --context-len 10 \
  --horizon 50
```

```bash
python scripts/visualize_rssm_rollout.py \
  --checkpoint runs/dreamer_ball_offline_v1/checkpoints/best.pt \
  --dataset data/ball_balance_coverage_v0.npz \
  --context-len 10 \
  --horizon 100 \
  --play
```

## Removed PlaNet Control Path

The old control path trained a world model first and then controlled the
environment with an external optimizer over candidate action sequences. That
path has been removed:

- no standalone planning package;
- no receding-horizon optimizer;
- no CEM or CEM-GD;
- no frozen-model control stage.

Dreamer control is the actor learned from imagined RSSM rollouts.
