# Implementation Comparison

This note compares the local PyTorch implementation against the local Danijar Hafner reference code in:

- `../Danijar/dreamer` for Dreamer V1.
- `../Danijar/dreamerv2` for Dreamer V2.

It is meant as an engineering checklist, not a claim of benchmark parity.

## What Matches

The core model boundary now follows the same high-level split:

```text
observation -> encoder -> RSSM -> decoder / reward / continuation / actor / critic
```

The local `RSSM` keeps deterministic GRU memory plus stochastic latent state. For V1 it uses a continuous diagonal Gaussian latent. For V2 it uses straight-through categorical latents.

The actor and critic train from imagined RSSM rollouts with TD(lambda)-style returns, which is the Dreamer behavior-learning idea rather than PlaNet-style CEM planning.

The world model now supports both state observations and pixel observations. State observations use MLP encoder/decoder; pixel observations use convolutional encoder/decoder.

The V2 KL path includes a dynamics/representation split controlled by `kl_alpha`, similar in spirit to Dreamer V2's KL balancing.

## Important Differences

### Environment And Replay

Danijar's implementations use driver-style online interaction, action repeats, action normalization wrappers, `is_first` episode reset flags, and replay sampling around step streams.

This repo uses an episode-major NPZ replay buffer:

```text
obs:        [episodes, T + 1, ...]
action:     [episodes, T, action_dim]
reward:     [episodes, T, 1]
terminated: [episodes, T, 1]
truncated:  [episodes, T, 1]
done:       [episodes, T, 1]
```

Potential issue: the RSSM does not receive an explicit `is_first` flag inside sequence windows. It starts each sampled window from a zero state and masks losses after `done`, but it does not reset hidden state mid-window the way Dreamer V2's `obs_step(..., is_first, ...)` does.

### Observation Preprocessing

Dreamer V2 preprocesses `uint8` images as:

```text
image / 255.0 - 0.5
```

This repo's DM-Control adapter stores pixel replay as float images in `[0, 1]`, then `Normalizer` applies per-pixel mean/std normalization.

Potential issue: per-pixel dataset normalization can work, but it differs from the reference and can make decoder scale and image likelihood behavior harder to compare directly.

### Decoder And Reconstruction Loss

Danijar's decoder heads output probability distributions and train with log likelihoods. For images, the decoder is a Normal distribution with fixed std; for vectors, V2 uses distribution layers.

This repo decodes deterministic tensors and trains reconstruction with MSE.

Potential issue: MSE is simpler and often usable, but it changes loss scale, uncertainty modeling, and the balance between reconstruction, reward, continuation, and KL losses.

### RSSM Prior

Dreamer V2's reference RSSM uses an ensemble prior for dynamics statistics and layer normalization in several dense blocks.

This repo uses a single prior network and no layer normalization by default.

Potential issue: the single-prior model may be less stable on harder DM-Control tasks, especially pixel tasks and long-horizon locomotion.

### KL Objective

Dreamer V1 applies free nats to the KL. Dreamer V2 has configurable forward/reverse KL, balance, and `free_avg` behavior.

This repo:

- Applies `free_nats` to V1.
- Uses KL balancing for V2 through `kl_alpha`.
- Does not currently apply a V2 free-nats/free-avg threshold.

Potential issue: V2 categorical latents may collapse or over-regularize more easily without the exact reference KL controls.

### Reward And Continuation

The reference implementation uses probabilistic reward and discount heads. It can also transform rewards depending on config.

This repo uses:

- MSE reward prediction on normalized rewards.
- BCE continuation prediction from terminal labels.
- Heuristics that treat DM-Control `discount == 0` as termination and time limits as truncation.

Potential issue: reward normalization plus MSE changes the scale used for imagined actor/critic learning. This can work, but tuning from the reference configs will not transfer one-to-one.

### Actor Action Space

Danijar's DM-Control wrappers normalize bounded actions to `[-1, 1]`.

This repo normalizes actions with replay mean/std, then stores environment action bounds in that normalized coordinate system for the actor. At execution, actions are denormalized and clipped to DM-Control bounds.

Potential issue: exploration noise is applied in normalized-stat coordinates, not directly in `[-1, 1]` action coordinates. This can make noise scale task- and dataset-dependent.

### Actor Gradient Modes

The local behavior training supports `dynamics`, `reinforce`, and `both`, with `auto` choosing dynamics for continuous actions.

This is useful, but it is not an exact reproduction of all V1/V2 actor loss details. In particular, entropy terms, stop-gradient placement, and score-function mixing differ from the TensorFlow reference.

### Pixel Architecture

Dreamer V1's image decoder uses transpose-convolution kernels that exactly expand from `1x1` to `64x64`. Dreamer V2 uses configurable CNN depth and separates CNN/MLP keys.

This repo uses a compact CHW PyTorch conv stack:

```text
Conv 32/64/128/256, kernel 4, stride 2
ConvTranspose back to the original image shape
```

Potential issue: this is functional, but not architecture-identical. For benchmark-style results, tune `embed_dim`, `deter_dim`, `stoch_dim`, decoder depth, and loss scales.

## Current DM-Control Path

Implemented scripts:

- `scripts/collect_dm_control_dataset.py`: collect random replay.
- `scripts/train_dm_control_dreamer.py`: online Dreamer training on DM-Control.
- `scripts/run_dm_control_policy.py`: closed-loop checkpoint evaluation with optional rendered frames and GIF.
- `scripts/run_dm_control_env.py`: low-level environment smoke test.

Recommended order:

1. Train state-observation `cartpole/swingup` with V1.
2. Train state-observation `cartpole/swingup` with V2.
3. Try `walker/walk` state observations.
4. Move to pixel observations after state training is stable.

## Issues To Watch

If training fails to improve:

1. Verify replay observations, actions, rewards, and done flags by inspecting `runs/.../replay/latest.npz`.
2. Plot `recon_loss`, `reward_loss`, `continuation_loss`, `kl_loss`, actor loss, and critic loss.
3. Reduce `seq-len`, `batch-size`, and `imagination-horizon` for pixel tasks until the smoke path is stable.
4. Try larger models for pixel tasks: `deter_dim=200`, `stoch_dim=30`, `embed_dim=1024`.
5. Tune reward and continuation loss weights; MSE reward scale differs from reference log-likelihood heads.
6. Consider adding V2 free-nats/free-avg, layer norm, and ensemble priors before expecting reference-level V2 performance.

## Highest-Priority Future Improvements

1. Add exact Dreamer-style image preprocessing option: `image / 255.0 - 0.5` without per-pixel mean/std normalization.
2. Add explicit `is_first` handling in RSSM sequence observation.
3. Add V2 free-nats/free-avg controls.
4. Add distributional decoder/reward/value heads instead of plain MSE heads.
5. Add action normalization mode `[-1, 1]` for DM-Control actions to match the reference wrappers.
6. Add benchmark logging summaries for rendered reconstructions and imagined rollouts.
