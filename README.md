# DM-Control Dreamer V1/V2

PyTorch-Implementierung von Dreamer für Aufgaben aus der DeepMind Control Suite. Derselbe Code unterstützt:

- Kontinuierliche Gaußsche RSSM-Latents im Stil von Dreamer V1.
- Straight-through kategorische RSSM-Latents mit KL-Balancing im Stil von Dreamer V2.
- Zustandsbeobachtungen aus DM-Control-Observation-Dictionaries.
- Pixelbeobachtungen, die aus MuJoCo-Kameras gerendert werden.

Die ursprüngliche Ball-Balance-Umgebung ist weiterhin vorhanden, aber dieser Branch ist auf DM-Control ausgerichtet.

Das DM-Control-Training nutzt eine driver-artige Online-Schleife: Aktionen werden in `[-1, 1]` gesampelt, durch einen Wrapper auf die echte DM-Control-Action-Spezifikation abgebildet, mit `is_first`-Reset-Markern in ein Step-Stream-Replay geschrieben und anschließend als zusammenhängende Sequenzfenster für das RSSM-Training gesampelt.

## Installation

Verwende in diesem Workspace Python 3.12 für DM-Control. Die bestehende Python-3.13-`.venv` kann dazu führen, dass `labmaze` auf einen Bazel-Source-Build zurückfällt.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python /usr/bin/python3.12 .venv-dm-control
UV_CACHE_DIR=/tmp/uv-cache uv pip install --python .venv-dm-control/bin/python -e ".[dm-control,dev]"
```

Smoke-Test für die Umgebung:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py \
  --domain cartpole \
  --task swingup \
  --steps 20
```

Auf headless Maschinen kannst du EGL fürs Rendering verwenden:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_env.py \
  --domain cartpole \
  --task swingup \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/smoke/cartpole
```

## Dreamer V1 Trainieren

Starte mit Zustandsbeobachtungen. Das ist der schnellste Weg, um World Model, Actor, Critic, Replay und Checkpointing-Schleife zu prüfen.

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain cartpole \
  --task swingup \
  --obs-type state \
  --run-dir runs/dmc_cartpole_swingup_v1_0615 \
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

Checkpoints und Replay werden hier geschrieben:

```text
runs/dmc_cartpole_swingup_v1/checkpoints/
runs/dmc_cartpole_swingup_v1/replay/latest.npz
```

## Dreamer V2 Trainieren

V2 nutzt den kategorischen RSSM-Pfad. Für kontinuierliche DM-Control-Aktionen wird `--actor-gradient auto` zu Dynamics-Gradienten aufgelöst; `--exploration-mode auto` wird für V2 zu Policy-Entropy-Sampling.

```bash
.venv-dm-control/bin/python scripts/train_dm_control_dreamer.py \
  --domain cartpole \
  --task swingup \
  --obs-type state \
  --run-dir runs/dmc_cartpole_swingup_v2_0615 \
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

Für schwierigere V2-Aufgaben, besonders Pixel- oder Lokomotionsaufgaben, probiere die referenzorientierten Stabilitätsoptionen:

```bash
  --layer-norm \
  --rssm-ensemble 5
```

## Pixel-Training

Pixelbeobachtungen verwenden `WorldModelConfig(obs_type="pixel")`, `ConvEncoder` und `ConvDecoder`. Im Stream-Replay werden Bildbeobachtungen als `[1, stream_time, channels, height, width]` mit `is_first`-Markern an Episodengrenzen gespeichert. Gerenderte Frames werden als `image / 255.0 - 0.5` vorverarbeitet; die Pixel-Observation-Normalisierung bleibt die Identität.

Die Standard-Pixelarchitektur nutzt Dreamer-artige CNN-Optionen: Encoder-Kernels `4,4,4,4`, Decoder-Kernels `5,5,6,6` und `cnn_depth=48`. Dadurch werden `64x64`-Bilder von `1x1` zurück auf `64x64` dekodiert.

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
  --run-dir runs/dmc_cartpole_swingup_pixel_v1_0615 \
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

Pixel-Training ist deutlich schwerer als State-Training. Verwende zuerst eine kleine `--batch-size`.

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
  --run-dir runs/dmc_cartpole_swingup_pixel_v2_0615 \
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

## Trainingsmetriken Und Loss-Plots

Das Online-DM-Control-Training schreibt pro Iteration einen JSON-Datensatz nach:

```text
runs/<run-name>/metrics.jsonl
```

Das Log enthält Train- und Validation-Losses für World Model und Behavior-Heads, Open-Loop-Validation-Losses, Replay-Größe, Exploration-Einstellungen und den gesammelten Return. Es wird auch geschrieben, wenn TensorBoard nicht installiert ist.

Nach dem Training kannst du die vier Standard-Cartpole-Runs plotten mit:

```bash
.venv-dm-control/bin/python scripts/plot_dm_control_training_losses.py
```

Die Plots werden hier gespeichert:

```text
runs/dm_control_loss_plots/
```

Das Plot-Skript erzeugt jeweils eine Abbildung für Reconstruction-Loss, Reward-Loss, KL-Loss, Actor-Loss, Critic-Loss und Return. Jede Abbildung hat Train- und Validation-Subplots; Return wird aus der Online-Collection als `collect/collect_avg_reward` geloggt.

Ältere Runs ohne `metrics.jsonl` können vollständige Kurven nicht allein aus Checkpoints rekonstruieren. Wenn du Terminalausgabe gespeichert hast, lege sie im Run-Verzeichnis als `train.log`, `stdout.log` oder `output.log` ab; das Plot-Skript parst Zeilen wie `iter=100 train=... val=... recon=...`.

## Checkpoint Evaluieren

Evaluation ohne Rendering:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v1/checkpoints/best.pt \
  --num-episodes 10 \
  --max-steps 200 \
  --device cpu
```

Trainierte Policy online in der DM-Control-Umgebung ausführen und gerenderte Frames plus GIF speichern:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v1_0615/checkpoints/best.pt \
  --num-episodes 3 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_v1_0615 \
  --gif-out outputs/dmc_cartpole_swingup_v1_0615.gif \
  --device cpu
```

V2-Checkpoint evaluieren:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_v2_0615/checkpoints/best.pt \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_v2_0615 \
  --gif-out outputs/dmc_cartpole_swingup_v2_0615.gif \
  --device cpu
```

V2-Pixel-Checkpoint evaluieren:

```bash
.venv-dm-control/bin/python scripts/run_dm_control_policy.py \
  --checkpoint runs/dmc_cartpole_swingup_pixel_v2_0615/checkpoints/best.pt \
  --num-episodes 1 \
  --max-steps 500 \
  --render \
  --mujoco-gl egl \
  --frames-out outputs/dmc_cartpole_swingup_pixel_v2_0615 \
  --gif-out outputs/dmc_cartpole_swingup_pixel_v2_0615.gif \
  --device cpu
```

Der Policy-Runner leitet `domain`, `task`, `obs_type`, Bildgröße, Kamera und Action-Repeat aus dem Checkpoint ab, wenn der Checkpoint mit `scripts/train_dm_control_dreamer.py` erzeugt wurde. Du kannst diese Werte bei Bedarf mit CLI-Flags überschreiben.

## Nur Zufälliges Replay

Zum Debuggen von Offline-World-Model-Training oder zum Prüfen von Dataset-Shapes:

```bash
.venv-dm-control/bin/python scripts/collect_dm_control_dataset.py \
  --domain walker \
  --task walk \
  --obs-type state \
  --num-episodes 100 \
  --max-episode-steps 200 \
  --out data/dmc_walker_walk_random_state.npz
```

Danach kannst du mit dem DM-Control-Trainer aus diesem Replay trainieren:

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

## Implementierungsnotizen

Siehe [docs/implementation.md](docs/implementation.md) für Unterschiede zu Danijar Hafners lokalen Dreamer-V1/V2-Referenzimplementierungen und für wahrscheinliche Punkte, auf die du beim Skalieren dieser Implementierung achten solltest.
