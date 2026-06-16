# Dreamer Hyperparameters And Tuning

This guide collects the knobs that matter most in the current PyTorch codebase.
It is not a claim of benchmark parity with the TensorFlow references. Treat it
as a practical checklist for getting stable DM-Control and ball-balance runs.

## Core Hyperparameter Table

| Area | Parameter | Current default | Increase when | Decrease when | Notes |
| --- | --- | ---: | --- | --- | --- |
| Replay | `seq_len` | `50` for DM-Control, often `200` for ball balance | Long-horizon prediction matters and memory allows it | Pixel training is slow or unstable | World-model loss sees the full sequence. Actor/critic imagination starts from flattened posterior states. |
| Replay | `batch_size` | `128` | Gradients are noisy and memory is available | Pixel model runs out of memory | Pixel runs often need smaller batches first. |
| Online loop | `seed_episodes` | `20` | Replay is narrow or early actor collapses | You need a fast smoke test | Random seed replay is only a bootstrap. |
| Online loop | `train_steps` | `100` | Replay grows faster than model learns | Collection is too slow | More updates per collection improves sample reuse. |
| Online loop | `collect_episodes` | `5` | Policy improves and you want faster replay growth | Model is not learning from current replay | Too much collection with a weak policy can fill replay with poor data. |
| Model size | `deter_dim` | `128` | Pixel tasks, walker tasks, long horizons | Fast smoke testing | Try `200`, `512`, or larger for serious pixel runs. |
| Model size | `stoch_dim` | `16` | Latent bottleneck is too small | KL collapses or training is too heavy | For V2, total stochastic feature size is `stoch_dim * discrete_classes`. |
| Model size | `discrete_classes` | `32` | V2 needs more categorical capacity | Memory or KL is too large | Only used by V2. |
| Model size | `hidden_dim` | `128` | Reward/reconstruction underfit | Training is slow | Dense RSSM/head MLP width. |
| Pixels | `cnn_depth` | `48` | Pixel reconstructions are weak | Memory is tight | Dreamer-style pixel runs usually need more capacity than state runs. |
| Pixels | `encoder_kernels` | `4,4,4,4` | Rarely changed | Image size is not `64x64` | Default encoder matches 64x64-style Dreamer inputs. |
| Pixels | `decoder_kernels` | `5,5,6,6` | Rarely changed | Image size is not `64x64` | Default decoder expands `1x1 -> 64x64`. |
| RSSM prior | `rssm_ensemble` | `1` | Hard pixel/locomotion tasks are unstable | State tasks or smoke tests | Try `5` for V2-style stability experiments. |
| Normalization | `layer_norm` | `False` | Prior/reward/value training is unstable | Simple state tasks already work | Adds layer norm in dense blocks and group norm in CNN blocks. |
| KL V1 | `free_nats` | `1.0` | KL dominates reconstruction early | Latents are unused | V1 path uses mean KL thresholding. |
| KL V2 | `kl_balance` | `0.8` | Prior dynamics lag posterior | Representation quality drops | V2-style balance between representation and dynamics terms. |
| KL V2 | `kl_free` | `0.0` | KL is too noisy or too small | KL loss dominates | V2 free threshold. |
| KL V2 | `kl_free_avg` | `True` | You want reference-style average threshold | Per-step KL spikes are harmful | `False` thresholds each KL entry before averaging. |
| Loss scale | `reward_loss_weight` | `1.0` | Reward prediction is poor | Reward loss overwhelms reconstruction/KL | Actor learns from imagined reward, so this matters. |
| Loss scale | `continuation_loss_weight` | `1.0` | Termination prediction is poor | Continuation dominates | More important in tasks with true terminal states. |
| Distribution | `decoder_dist` | `normal` | Usually leave default | Use `mse` for old ablations | Fixed-std Normal NLL by default. |
| Distribution | `reward_head_dist` | `normal` | Usually leave default | Use `mse` for old ablations | Fixed-std Normal reward likelihood. |
| Distribution | `value_head_dist` | `normal` | Usually leave default | Use `mse` for old ablations | Fixed-std Normal value likelihood. |
| Reward scale | `reward_transform` | `identity` | Rewards have heavy tails | Transformed reward hurts control | Try `symlog` for large reward ranges. |
| Behavior | `imagination_horizon` | `15` | Task needs longer planning | Actor/critic unstable | Longer imagination increases compounding model error. |
| Behavior | `discount` | `0.99` | Long-horizon reward matters | Short tasks or instability | Multiplied by predicted continuation. |
| Behavior | `lambda` | `0.95` | Returns are too biased | Returns have high variance | TD(lambda) return mixing. |
| Actor | `actor-gradient` | `auto` | You need a specific ablation | Usually leave default | DM-Control continuous actions resolve to `dynamics`. |
| Actor | `actor-entropy-scale` | `1e-3` | Policy collapses too early | Actions remain too random | Supports exploration and imagined policy entropy. |
| Collection | `exploration-mode` | `auto` | You want explicit behavior | Usually leave default | V1 resolves to noise, V2 to policy entropy. |
| Collection | `exploration-noise` | `0.3` | V1 collection lacks coverage | Actor is too noisy | Ignored as external noise in policy-entropy mode. |
| Optimization | `world-lr` | `3e-4` | Model learns too slowly | Loss spikes or diverges | Usually tune before actor/critic LR. |
| Optimization | `actor-lr` | `8e-5` | Actor improves too slowly | Actor collapses | Actor gradients are from imagined rollouts. |
| Optimization | `critic-lr` | `8e-5` | Critic lags returns | Critic loss explodes | Target critic soft update also matters. |
| Optimization | `target-tau` | `0.01` | Critic target lags too much | Targets are too noisy | Soft update coefficient. |
| Optimization | `grad-clip` | `100.0` | Rarely increase | Gradients explode | Applies to world, actor, and critic optimizers. |

## Suggested Starting Points

| Run type | Recommended starting configuration | Why |
| --- | --- | --- |
| DM-Control state V1 smoke | Defaults with `--dreamer-version v1`, `cartpole/swingup`, `batch-size 128` | Fastest way to verify replay, actor, critic, and checkpointing. |
| DM-Control state V2 smoke | `--dreamer-version v2 --stoch-dim 16 --discrete-classes 32 --kl-balance 0.8` | Checks categorical RSSM and V2 KL controls without pixel cost. |
| DM-Control pixel V1 | Add `--obs-type pixel --height 64 --width 64 --cnn-depth 48`, reduce batch if needed | Pixel path tests CNN encoder/decoder and centered image preprocessing. |
| DM-Control pixel V2 | Pixel V2 plus `--layer-norm --rssm-ensemble 5` | Closer to V2 robustness settings for hard visual control. |
| Ball-balance offline V1 | `--train-mode offline --dreamer-version v1 --seq-len 200` | Simple continuous-control baseline for low-dimensional observations. |
| V2 actor-gradient ablation | `--dreamer-version v2 --actor-gradient dynamics|reinforce|both` | Separates latent/KL changes from behavior-gradient changes. |

Use a fresh `--run-dir` or add `--no-resume` whenever changing architecture
knobs such as `dreamer-version`, `deter-dim`, `stoch-dim`, `discrete-classes`,
`cnn-depth`, `rssm-ensemble`, or `layer-norm`. Resuming loads the old checkpoint
config and will ignore new architecture flags.

## Tuning Order

1. Make the environment path boring first.

Check that replay shapes, action bounds, and done flags are sane. For
DM-Control, actions should be stored in normalized `[-1, 1]` coordinates and
pixel observations should be centered in `[-0.5, 0.5]`.

2. Stabilize the world model before judging the actor.

Watch `recon_loss`, `reward_loss`, `continuation_loss`, `kl_loss`, `raw_kl`,
and entropy/std metrics. If open-loop losses are poor, actor improvement is not
very meaningful yet.

3. Tune KL and latent capacity.

For V1, start with `free_nats`. For V2, first tune `kl_balance` and `kl_free`.
If KL is near zero and reconstructions are poor, the latent may be unused. If KL
dominates everything, the model may be over-regularized or the posterior/prior
are fighting.

4. Tune behavior learning after the model is plausible.

For continuous DM-Control, prefer `--actor-gradient dynamics` or `auto`.
Increase `actor-entropy-scale` if the policy collapses early. Reduce
`imagination-horizon` if actor and critic losses become unstable before the
world model is good.

5. Scale pixels carefully.

Start with smaller `batch-size` and maybe fewer `train-steps`. Once the run is
stable, increase `deter-dim`, `hidden-dim`, `embed-dim`, `cnn-depth`, and
possibly `rssm-ensemble`. Pixel tasks can look broken simply because the model
is too small.

## Symptoms And Likely Fixes

| Symptom | Likely cause | Try |
| --- | --- | --- |
| `recon_loss` falls but rewards do not improve | Actor/critic not learning useful imagined returns | Check `reward_loss`, increase reward capacity/weight, lower actor LR, inspect imagined rewards. |
| KL collapses near zero | Posterior/prior ignore stochastic state | Lower free threshold, increase latent capacity, inspect decoder strength. |
| KL explodes | Prior cannot track posterior or model too small | Increase model size, try `--layer-norm`, tune `kl_balance`, lower world LR. |
| Actor loss improves but real return does not | Model exploitation or poor replay coverage | Collect more seed data, lower imagination horizon, validate open-loop predictions. |
| Critic loss explodes | Targets too noisy or long-horizon model error | Lower `critic-lr`, lower `imagination-horizon`, lower `target-tau`, improve world model first. |
| Pixel training runs out of memory | CNN/model/batch too large | Lower `batch-size`, lower `cnn-depth`, shorten `seq-len`. |
| V2 pixel task unstable | Prior/categorical dynamics too hard | Add `--layer-norm`, try `--rssm-ensemble 5`, increase `deter-dim`, reduce LR. |
| Collection return flatlines | Exploration too weak or replay too narrow | Increase seed episodes, use policy entropy for V2, tune external noise for V1. |

## Metrics To Plot First

Use `scripts/plot_dm_control_training_losses.py` for DM-Control runs with
`metrics.jsonl`.

Minimum useful plots:

- `train/recon_loss` and `val/recon_loss`
- `train/reward_loss` and `val/reward_loss`
- `train/kl_loss`, `train/raw_kl`, `train/dynamics_kl_loss`
- `train/actor_loss`, `train/critic_loss`
- `collect/collect_avg_reward`
- `val_prior_rollout/obs_h1`, `val_prior_rollout/obs_h5`, `val_prior_rollout/reward_h1`

If these disagree, trust validation and collection metrics over training loss.
Training loss can improve while the actor overfits imagined model errors.
