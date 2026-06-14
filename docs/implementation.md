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

The V2 KL path includes Dreamer V2-style forward/reverse KL direction, balance, `free`, and `free_avg` controls.

## Important Differences

### Environment And Replay

Danijar's implementations use driver-style online interaction, action repeats, action normalization wrappers, `is_first` episode reset flags, and replay sampling around step streams.

The DM-Control path now follows that shape more closely:

```text
DMControlDriver -> NormalizeActionWrapper -> StreamReplay -> SequenceDataset
```

DM-Control actions are stored in normalized `[-1, 1]` coordinates. `StreamReplay` stores one continuous step stream:

```text
obs:        [1, stream_steps + 1, ...]
action:     [1, stream_steps, action_dim]
reward:     [1, stream_steps, 1]
terminated: [1, stream_steps, 1]
truncated:  [1, stream_steps, 1]
done:       [1, stream_steps, 1]
is_first:   [1, stream_steps + 1, 1]
```

`RSSM.observe()` and one-step posterior updates now consume `is_first`, reset recurrent state at episode starts, and zero the reset action. The world-model loss masks transitions whose next observation is marked `is_first`, so dummy reset transitions are not trained as dynamics.

Remaining difference: the local replay sampler is still a simple fixed-window PyTorch dataset over a single stream. The reference replay stack has richer prefetching, dataset workers, logging, and exact config-driven sampling behavior.

### Observation Preprocessing

Dreamer V2 preprocesses `uint8` images as:

```text
image / 255.0 - 0.5
```

Dreamer V1 uses the same centered image scale. In the local reference, the V1 preprocessing path casts images to float, optionally reduces bit depth, divides by the image bins, adds uniform dequantization noise, then subtracts `0.5`. V2 uses the simpler deterministic `uint8 / 255.0 - 0.5`.

This repo's DM-Control pixel adapter now stores rendered frames as CHW float images in `[-0.5, 0.5]`, and `scripts/train_dm_control_dreamer.py` leaves pixel observations unnormalized by dataset mean/std. State observations still use dataset mean/std normalization.

Remaining difference: the local path does not add Dreamer V1's optional image dequantization noise or bit-depth reduction; it matches the deterministic V2-style centering.

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
- Uses Dreamer V2-style `kl_forward`, `kl_balance`, `kl_free`, and `kl_free_avg`.
- Applies V2 free-nats either to the masked average KL (`free_avg=True`) or to each KL entry before averaging (`free_avg=False`).

Remaining difference: this still uses the local RSSM parameterization and single-prior network, so matching KL controls does not make the world model architecture identical to the reference.

### Reward And Continuation

The reference implementation uses probabilistic reward and discount heads. It can also transform rewards depending on config.

This repo uses:

- MSE reward prediction on normalized rewards.
- BCE continuation prediction from terminal labels.
- Heuristics that treat DM-Control `discount == 0` as termination and time limits as truncation.

Potential issue: reward normalization plus MSE changes the scale used for imagined actor/critic learning. This can work, but tuning from the reference configs will not transfer one-to-one.

### Actor Action Space

Danijar's DM-Control wrappers normalize bounded actions to `[-1, 1]`.

The DM-Control path now matches this convention: `NormalizeActionWrapper` exposes `[-1, 1]` actions to collection, replay, actor training, and policy evaluation, then maps those actions back to the real DM-Control action spec before stepping the environment.

The DM-Control trainer treats this normalized action range as the actor's native action space and no longer routes actor bounds through replay-stat action normalization.

### Actor Gradient Modes (Keep it)

The local behavior training supports `dynamics`, `reinforce`, and `both`, with `auto` choosing dynamics for continuous actions.

This is useful, but it is not an exact reproduction of all V1/V2 actor loss details. In particular, entropy terms, stop-gradient placement, and score-function mixing differ from the TensorFlow reference.

### Pixel Architecture (Keep it currently)

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
6. Consider adding layer norm and ensemble priors before expecting reference-level V2 performance.

## Highest-Priority Future Improvements

1. Add V1-style image dequantization and bit-depth controls.
2. Add distributional decoder/reward/value heads instead of plain MSE heads.
3. Add benchmark logging summaries for rendered reconstructions and imagined rollouts.
4. Add exact reference-style replay prefetching and dataset worker behavior.
