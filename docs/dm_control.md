# Applying This Dreamer to DM-Control

The current implementation is built for a low-dimensional Gymnasium environment:

```text
obs_t -> MLP encoder -> RSSM posterior/prior -> MLP decoder, reward head, continuation head
```

DM-Control can be integrated in two stages.

## 1. Start With State Observations

Use `dm_control.suite` observations as flat vectors first. This keeps the current MLP world model unchanged and lets you validate replay collection, action normalization, reward learning, and actor training before adding pixels.

Adapter responsibilities:

1. Convert the DM-Control timestep into Gymnasium-style pieces: `obs`, `reward`, `terminated`, `truncated`, `info`.
2. Flatten the observation dictionary in a stable key order.
3. Read `env.action_spec()` and expose a continuous action range for the actor.
4. Store replay in the same NPZ layout used by `Buffer`: `obs`, `action`, `reward`, `terminated`, `truncated`, `done`.

For example, `cartpole/swingup` has vector observations, so the current `WorldModelConfig(obs_dim=<flat dim>, action_dim=<action dim>)` pattern still applies.

## 2. Move to Pixel Observations

For image-based Dreamer, keep `RSSM` mostly as-is. The RSSM consumes an embedding vector, not raw observations, so the main change belongs in `WorldModel`:

```text
image obs_t -> ConvEncoder -> embed_t -> RSSM
RSSM feature [h_t, z_t] -> ConvDecoder -> image reconstruction
RSSM feature [h_t, z_t] -> MLP reward / continuation / actor / critic
```

Recommended code changes:

1. Add `obs_type: str = "vector"` and `obs_shape: tuple[int, ...]` to `WorldModelConfig`.
2. Keep the existing MLP encoder/decoder for `obs_type == "vector"`.
3. Add `ConvEncoder` for pixel inputs shaped `[batch, time, channels, height, width]`.
4. Add `ConvDecoder` that reconstructs pixels from Dreamer features.
5. Change reconstruction loss for pixels from plain MSE over flat vectors to image loss over `[C, H, W]`; Dreamer commonly uses normalized pixels in `[0, 1]` or `[-0.5, 0.5]`.
6. Keep reward, continuation, actor, and critic heads as MLPs over latent features.

Practical ConvEncoder shape:

```text
Input: 3x64x64 RGB
Conv 32, kernel 4, stride 2 -> ELU
Conv 64, kernel 4, stride 2 -> ELU
Conv 128, kernel 4, stride 2 -> ELU
Conv 256, kernel 4, stride 2 -> ELU
Flatten -> Linear(embed_dim)
```

Practical ConvDecoder shape:

```text
Linear(feature_dim -> 1024)
Reshape to 1024x1x1 or a small spatial grid
ConvTranspose blocks back to 3x64x64
```

## 3. Use the Smoke Script

Install the optional dependencies:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /usr/bin/python3.12 .venv-dm-control
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv-dm-control/bin/python -e ".[dm-control]"
```

Run a state-observation smoke test:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py --domain cartpole --task swingup --steps 100
```

Check RGB rendering:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py \
  --domain walker \
  --task walk \
  --steps 50 \
  --render \
  --frames-out output-smoke/dm-control-walker
```

On headless machines, try:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py --domain cartpole --task swingup --render --mujoco-gl osmesa
```

or:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py --domain cartpole --task swingup --render --mujoco-gl egl
```

The project extra pins `mujoco>=3.8.1,<3.9` because `dm-control 1.0.41` and `mujoco 3.9.0` currently disagree on MuJoCo model fields. Python 3.12 is the smoother local choice; Python 3.13 may require a working Bazel install to build `labmaze`.

## 4. Training Path

A conservative migration path is:

1. Build a `DMControlEnv` adapter and replay collector for flattened observations.
2. Train the existing Dreamer on `cartpole/swingup` or `cheetah/run` from state observations.
3. Add `ConvEncoder` and `ConvDecoder`.
4. Collect pixel replay with `physics.render(height=64, width=64, camera_id=...)`.
5. Train the world model with image reconstruction, reward, continuation, and KL losses.
6. Reuse the actor and critic without changing their input interface, because they already consume RSSM features.

The important boundary is: change the observation model for pixels, not the RSSM dynamics API.
