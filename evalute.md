Example

.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v2_0615_imagine30/checkpoints/best.pt \
  --camera-id 1 \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_v2_0615_imagine30 \
  --gif-out outputs/dmc_cartpole_swingup_v2_0615_imagine30.gif \
  --device cpu
```

### Reacher train state-based

.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain reacher \
  --task easy \
  --obs-type state \
  --run-dir runs/dmc_reacher_easy_v2_0616 \
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
  --train-steps 100 \
  --collect-episodes 5 \
  --seq-len 50 \
  --batch-size 128 \
    --imagination-horizon 15

.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_reacher_easy_v2_0616/checkpoints/best.pt \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_reacher_easy_v2_0616 \
  --gif-out outputs/dmc_reacher_easy_v2_0616.gif


### Reacher train pixel-based

.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain reacher \
  --task easy \
  --obs-type pixel \
  --height 64 \
  --width 64 \
  --camera-id 0 \
  --mujoco-gl egl \
  --run-dir runs/dmc_reacher_easy_pixel_v2_0616 \
  --dreamer-version v2 \
  --stoch-dim 16 \
  --discrete-classes 32 \
  --layer-norm \
  --rssm-ensemble 5 \
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
  --train-steps 100 \
  --collect-episodes 5 \
  --seq-len 50 \
  --batch-size 64 \
    --imagination-horizon 15


.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_reacher_easy_pixel_v2_0616/checkpoints/best.pt \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_reacher_easy_pixel_v2_0616 \
  --gif-out outputs/dmc_reacher_easy_pixel_v2_0616.gif