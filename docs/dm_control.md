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

Implemented model changes:

1. `WorldModelConfig` has `obs_type: str = "vector"` and `obs_shape: tuple[int, ...]`.
2. `obs_type == "vector"` keeps the existing MLP encoder/decoder.
3. `obs_type == "pixel"` uses `ConvEncoder` for inputs shaped `[batch, time, channels, height, width]`.
4. Pixel models use `ConvDecoder` to reconstruct image observations from Dreamer features.
5. Reconstruction loss now reduces over all trailing observation dimensions, so vector and pixel observations both produce per-step MSE.
6. Reward, continuation, actor, and critic heads still use MLPs over RSSM features.

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

Watch the environment in the live viewer:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py \
  --domain walker \
  --task walk \
  --viewer \
  --mujoco-gl glfw
```

The viewer starts paused. Press Space to run or pause, Backspace to reset, and
F1 for the built-in controls.

The project extra pins `mujoco>=3.8.1,<3.9` because `dm-control 1.0.41` and `mujoco 3.9.0` currently disagree on MuJoCo model fields. Python 3.12 is the smoother local choice; Python 3.13 may require a working Bazel install to build `labmaze`.

## 4. Training Path

A conservative migration path is:

1. Build a `DMControlEnv` adapter and replay collector for flattened observations.
2. Train the existing Dreamer on `cartpole/swingup` or `cheetah/run` from state observations.
3. Collect pixel replay with `physics.render(height=64, width=64, camera_id=...)`.
4. Train with `--obs-type pixel` so `WorldModel` selects the convolutional encoder/decoder.
5. Reuse the actor and critic without changing their input interface, because they already consume RSSM features.

The important boundary is: change the observation model for pixels, not the RSSM dynamics API.
