# DM-Control Dreamer V1/V2

PyTorch Dreamer implementation for DeepMind Control Suite tasks. The same code supports:

- Dreamer V1 style continuous Gaussian RSSM latents.
- Dreamer V2 style straight-through categorical RSSM latents with KL balancing.
- State observations from DM-Control observation dictionaries.
- Pixel observations rendered from MuJoCo cameras.

The original ball-balance environment is still available, but this branch is oriented around DM-Control.

DM-Control training uses a driver-style online loop: actions are sampled in `[-1, 1]`, mapped to the real DM-Control action spec by a wrapper, written into a step-stream replay with `is_first` reset markers, and sampled as contiguous sequence windows for RSSM training.

## Install

Use Python 3.12 for DM-Control in this workspace. The existing Python 3.13 `.venv` can make `labmaze` fall back to a Bazel source build.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /usr/bin/python3.12 .venv-dm-control
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv-dm-control/bin/python -e ".[dm-control,dev]"
```

Smoke-test the environment:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py \
  --domain cartpole \
  --task swingup \
  --steps 20
```

On headless machines, use EGL for rendering:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py \
  --domain cartpole \
  --task swingup \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/smoke/cartpole
```

## Train Dreamer V1

Start with state observations. This is the quickest way to verify the world model, actor, critic, replay, and checkpointing loop.

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain cartpole \
  --task swingup \
  --obs-type state \
  --run-dir runs/dmc_cartpole_swingup_v1_0614 \
  --dreamer-version v1 \
  --seed-episodes 20 \
  --buffer-episodes 2000 \
  --max-episode-steps 200 \
  --online-iterations 100 \
  --update-steps 100 \
  --collect-episodes 5 \
  --seq-len 50 \
  --batch-size 128 \
  --imagination-horizon 15
```

Checkpoints and replay are written under:

```text
runs/dmc_cartpole_swingup_v1/checkpoints/
runs/dmc_cartpole_swingup_v1/replay/latest.npz
```

## Train Dreamer V2

V2 uses the categorical RSSM path. For continuous DM-Control actions, `--actor-gradient auto` resolves to dynamics gradients; `--exploration-mode auto` resolves to policy-entropy sampling for V2.

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain cartpole \
  --task swingup \
  --obs-type state \
  --run-dir runs/dmc_cartpole_swingup_v2_0614 \
  --dreamer-version v2 \
  --stoch-dim 16 \
  --discrete-classes 32 \
  --kl-balance 0.8 \
  --kl-free 0.0 \
  --kl-free-avg \
  --decoder-dist normal \
  --reward-head-dist normal \
  --value-head-dist normal \
  --seed-episodes 20 \
  --buffer-episodes 2000 \
  --max-episode-steps 200 \
  --online-iterations 100 \
  --update-steps 100 \
  --collect-episodes 5 \
  --seq-len 50 \
  --batch-size 128 \
  --imagination-horizon 15
```

For harder V2 tasks, especially pixels or locomotion, try the reference-oriented stability knobs:

```bash
  --layer-norm \
  --rssm-ensemble 5
```

## Pixel Training

Pixel observations use `WorldModelConfig(obs_type="pixel")`, `ConvEncoder`, and `ConvDecoder`. In stream replay, image observations are stored as `[1, stream_time, channels, height, width]` with `is_first` markers at episode boundaries. Rendered frames are preprocessed as `image / 255.0 - 0.5`, and pixel observation normalization is left as identity.

The default pixel architecture uses Dreamer-style CNN knobs: encoder kernels `4,4,4,4`, decoder kernels `5,5,6,6`, and `cnn_depth=48`, so `64x64` images decode from `1x1` back to `64x64`.

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain cartpole \
  --task swingup \
  --obs-type pixel \
  --height 64 \
  --width 64 \
  --cnn-depth 48 \
  --encoder-kernels 4,4,4,4 \
  --decoder-kernels 5,5,6,6 \
  --camera-id 0 \
  --mujoco-gl egl \
  --run-dir runs/dmc_cartpole_swingup_pixel_v1_0614 \
  --dreamer-version v1 \
  --seed-episodes 20 \
  --buffer-episodes 1000 \
  --max-episode-steps 200 \
  --online-iterations 100 \
  --update-steps 100 \
  --collect-episodes 5 \
  --seq-len 50 \
  --batch-size 128 \
  --imagination-horizon 15
```

Pixel training is much heavier than state training. Use small `--batch-size` first.

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain cartpole \
  --task swingup \
  --obs-type pixel \
  --height 64 \
  --width 64 \
  --cnn-depth 48 \
  --encoder-kernels 4,4,4,4 \
  --decoder-kernels 5,5,6,6 \
  --camera-id 0 \
  --mujoco-gl egl \
  --run-dir runs/dmc_cartpole_swingup_pixel_v2_0614 \
  --dreamer-version v2 \
  --stoch-dim 16 \
  --discrete-classes 32 \
  --layer-norm \
  --rssm-ensemble 5 \
  --kl-balance 0.8 \
  --kl-free 0.0 \
  --kl-free-avg \
  --seed-episodes 20 \
  --buffer-episodes 1000 \
  --max-episode-steps 200 \
  --online-iterations 100 \
  --update-steps 100 \
  --collect-episodes 5 \
  --seq-len 50 \
  --batch-size 128 \
  --imagination-horizon 15
```

## Training Metrics And Loss Plots

Online DM-Control training writes one JSON record per iteration to:

```text
runs/<run-name>/metrics.jsonl
```

The log includes train and validation losses for the world model and behavior heads, open-loop validation losses, replay size, exploration settings, and the collected return. It is written even when TensorBoard is not installed.

After training, plot the four standard cartpole runs with:

```bash
.venv-dm-control/bin/python scripts/plot_dm_control_training_losses.py
```

The plots are written to:

```text
runs/dm_control_loss_plots/
```

The plotting script creates one figure each for reconstruction loss, reward loss, KL loss, actor loss, critic loss, and return. Each figure has train and validation subplots; return is logged from online collection as `collect/collect_avg_reward`.

Older runs that do not have `metrics.jsonl` cannot reconstruct full curves from checkpoints alone. If you saved terminal output, place it in the run directory as `train.log`, `stdout.log`, or `output.log`; the plotting script will parse lines like `iter=100 train=... val=... recon=...`.


## Evaluate A Checkpoint

Evaluate without rendering:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v1/checkpoints/best.pt \
  --num-episodes 10 \
  --max-steps 200 \
  --device cpu
```

Run the trained policy online in the DM-Control environment and save rendered frames plus a GIF:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v1_0614/checkpoints/best.pt \
  --num-episodes 3 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_v1_0614 \
  --gif-out outputs/dmc_cartpole_swingup_v1_0614.gif \
  --device cpu
```
Evaluate a V2 checkpoint:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v2_0614/checkpoints/best.pt \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_v2_0614 \
  --gif-out outputs/dmc_cartpole_swingup_v2_0614.gif \
  --device cpu
```

Evaluate a V2 pixel checkpoint:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_pixel_v2_0614/checkpoints/best.pt \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_pixel_v2_0614 \
  --gif-out outputs/dmc_cartpole_swingup_pixel_v2_0614.gif \
  --device cpu
```

The policy runner infers `domain`, `task`, `obs_type`, image size, camera, and action repeat from the checkpoint when the checkpoint was produced by `scripts/train_dm_control_dreamer.py`. Override them with CLI flags if needed.

## Random Replay Only

For debugging offline world-model training or inspecting dataset shapes:

```bash
.venv-dm-control/bin/python scripts/collect_dm_control_dataset.py \
  --domain walker \
  --task walk \
  --obs-type state \
  --num-episodes 100 \
  --max-episode-steps 200 \
  --out data/dmc_walker_walk_random_state.npz
```

Then train from that replay with the DM-Control trainer:

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --dataset data/dmc_walker_walk_random_state.npz \
  --run-dir runs/dmc_walker_walk_offline_v1 \
  --domain walker \
  --task walk \
  --obs-type state \
  --dreamer-version v1 \
  --online-iterations 50 \
  --update-steps 100 \
  --collect-episodes 0 \
  --seq-len 50 \
  --batch-size 128
```

## Implementation Notes

See [docs/implementation.md](docs/implementation.md) for differences from Danijar Hafner's local Dreamer V1/V2 reference implementations and likely issues to watch while scaling this implementation.
