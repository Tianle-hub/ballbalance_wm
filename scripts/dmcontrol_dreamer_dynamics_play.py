#!/usr/bin/env python3
"""Interactive DMControl vs Dreamer dynamics viewer.

The left pane shows the real environment driven by keyboard or policy actions.
The right pane shows the Dreamer world model's decoded prediction under the
same actions. Use ``--model-mode anchored`` for one-step predictions corrected
by each real observation, or ``--model-mode free`` to let the model roll forward
from the reset frame without further observation updates.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Set

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_config = Path.home() / ".config" / "matplotlib"
    if not os.access(str(matplotlib_config.parent), os.W_OK):
        fallback_config = Path(tempfile.gettempdir()) / "ballbalance_matplotlib"
        fallback_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(fallback_config)


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DMControl vs Dreamer Dynamics</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111418;
      color: #f5f7fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      padding: 16px;
      background: #111418;
    }
    header, footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-width: 0;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      line-height: 1.2;
    }
    #status {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #c8d0da;
      font-size: 13px;
    }
    main {
      display: grid;
      place-items: center;
      min-height: 0;
      overflow: hidden;
    }
    #stream {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      image-rendering: pixelated;
      border: 1px solid #343b45;
      background: #050608;
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    button, select {
      height: 34px;
      border: 1px solid #3b4552;
      background: #1d232b;
      color: #f5f7fa;
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
      font-size: 13px;
    }
    button:active { transform: translateY(1px); }
    kbd {
      min-width: 24px;
      display: inline-block;
      padding: 2px 6px;
      border: 1px solid #3b4552;
      border-bottom-color: #232a33;
      border-radius: 4px;
      background: #1b2027;
      color: #e9edf2;
      text-align: center;
      font-size: 12px;
    }
    footer {
      justify-content: flex-start;
      color: #aeb8c4;
      font-size: 12px;
      flex-wrap: wrap;
    }
  </style>
</head>
<body>
  <header>
    <h1>DMControl vs Dreamer Dynamics</h1>
    <div id="status">connecting...</div>
    <div class="controls">
      <button id="pause">Pause</button>
      <button id="reset">Reset</button>
      <button id="control">Control</button>
      <select id="mode">
        <option value="anchored">anchored</option>
        <option value="free">free</option>
      </select>
    </div>
  </header>
  <main><img id="stream" alt="DMControl and Dreamer dynamics stream" /></main>
  <footer>
    <span><kbd>Space</kbd> pause</span>
    <span><kbd>R</kbd> reset</span>
    <span><kbd>P</kbd> policy/manual</span>
    <span><kbd>&larr;</kbd><kbd>&rarr;</kbd> action dim</span>
    <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> action value</span>
  </footer>
  <script>
    const statusEl = document.getElementById("status");
    const streamEl = document.getElementById("stream");
    const pauseBtn = document.getElementById("pause");
    const resetBtn = document.getElementById("reset");
    const controlBtn = document.getElementById("control");
    const modeSelect = document.getElementById("mode");
    modeSelect.value = "__INITIAL_MODE__";
    function send(data) {
      fetch("/event", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
      }).catch(() => {});
    }
    streamEl.src = "/stream";
    async function pollStatus() {
      try {
        const response = await fetch("/status", {cache: "no-store"});
        const msg = await response.json();
        if (msg.type === "status") {
          statusEl.textContent = msg.text;
          pauseBtn.textContent = msg.paused ? "Resume" : "Pause";
          controlBtn.textContent = msg.control === "policy" ? "Policy" : "Manual";
          modeSelect.value = msg.model_mode;
        }
      } catch (err) {
        statusEl.textContent = "disconnected";
      }
    }
    setInterval(pollStatus, 250);
    pollStatus();
    window.addEventListener("keydown", (event) => {
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(event.code)) {
        event.preventDefault();
      }
      send({type: "keydown", key: event.code});
    });
    window.addEventListener("keyup", (event) => {
      send({type: "keyup", key: event.code});
    });
    pauseBtn.onclick = () => send({type: "toggle_pause"});
    resetBtn.onclick = () => send({type: "reset"});
    controlBtn.onclick = () => send({type: "toggle_control"});
    modeSelect.onchange = () => send({type: "set_mode", model_mode: modeSelect.value});
  </script>
</body>
</html>
"""


def configure_render_backend(backend: str) -> None:
    backend = str(backend).lower()
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ["PYOPENGL_PLATFORM"] = backend
    elif backend == "glfw":
        os.environ.pop("PYOPENGL_PLATFORM", None)


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    return np.clip(image, 0, 255).astype(np.uint8)


def tensor_image_to_hwc(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.dim() == 4:
        image = image[0]
    image = torch.clamp((image + 0.5) * 255.0, 0.0, 255.0)
    return chw_to_hwc(image.numpy())


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int) -> bytes:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def resize_panel(frame: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray(chw_to_hwc(frame), mode="RGB")
    return image.resize((size, size), Image.BILINEAR)


def compose_frame(left: np.ndarray, right: np.ndarray, size: int, status: str) -> np.ndarray:
    label_h = 30
    status_h = 28
    gap = 6
    width = size * 2 + gap
    height = label_h + size + status_h
    canvas = Image.new("RGB", (width, height), (12, 15, 19))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(resize_panel(left, size), (0, label_h))
    canvas.paste(resize_panel(right, size), (size + gap, label_h))
    draw.rectangle((0, 0, width, label_h), fill=(23, 28, 35))
    draw.text((10, 8), "DMControl simulator", fill=(242, 246, 250))
    draw.text((size + gap + 10, 8), "Dreamer dynamics model", fill=(242, 246, 250))
    draw.rectangle((0, label_h + size, width, height), fill=(23, 28, 35))
    draw.text((10, label_h + size + 8), status[:160], fill=(206, 215, 225))
    return np.asarray(canvas, dtype=np.uint8)


@dataclass
class SessionState:
    obs: dict
    prev_state_for_obs: dict
    prev_action: torch.Tensor
    model_state: Optional[dict]
    keys_down: Set[str]
    selected_dim: int
    paused: bool
    step: int
    episode_return: float
    last_reward: float
    pred_reward: float
    last_action_val: float
    control: str
    model_mode: str


class DreamerDynamicsServer:
    def __init__(self, args: argparse.Namespace):
        configure_render_backend(args.render_backend)
        from dreamer import Dreamer, make_env

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
        self.env = make_env(args)
        self.obs_shape = self.env.observation_space["image"].shape
        self.action_size = int(self.env.action_space.shape[0])
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
        self.dreamer = Dreamer(args, self.obs_shape, self.action_size, self.device, restore=True)
        self._set_eval()
        self.lock = threading.Lock()
        self.state = self.new_session()
        self.last_status = self._status(self.state, "ready")
        self.latest_jpeg: Optional[bytes] = None
        self.pending_reset = False

    def _set_eval(self) -> None:
        modules = [
            self.dreamer.rssm,
            self.dreamer.actor,
            self.dreamer.obs_encoder,
            self.dreamer.obs_decoder,
            self.dreamer.reward_model,
            self.dreamer.value_model,
        ]
        if self.args.use_disc_model:
            modules.append(self.dreamer.discount_model)
        for module in modules:
            module.eval()

    def _init_rssm_state(self) -> dict:
        return self.dreamer.rssm.init_state(1, self.device)

    def new_session(self) -> SessionState:
        obs = self.env.reset()
        return SessionState(
            obs=obs,
            prev_state_for_obs=self._init_rssm_state(),
            prev_action=torch.zeros(1, self.action_size, device=self.device),
            model_state=None,
            keys_down=set(),
            selected_dim=0,
            paused=False,
            step=0,
            episode_return=0.0,
            last_reward=0.0,
            pred_reward=0.0,
            last_action_val=0.0,
            control=self.args.control,
            model_mode=self.args.model_mode,
        )

    def _observe_real(self, st: SessionState) -> dict:
        from dreamer import preprocess_obs

        obs = torch.tensor(st.obs["image"].copy(), dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_embed = self.dreamer.obs_encoder(preprocess_obs(obs))
        _, posterior = self.dreamer.rssm.observe_step(st.prev_state_for_obs, st.prev_action, obs_embed)
        return self.dreamer.rssm.detach_state(posterior)

    def _manual_action(self, st: SessionState) -> np.ndarray:
        action = np.zeros(self.action_size, dtype=np.float32)
        if self.action_size <= 0:
            return action
        value = 0.0
        if "ArrowUp" in st.keys_down:
            value += float(self.args.action_scale)
        if "ArrowDown" in st.keys_down:
            value -= float(self.args.action_scale)
        action[st.selected_dim] = value
        return np.clip(action, self.action_low, self.action_high)

    def _policy_action(self, state: dict) -> np.ndarray:
        feat = self.dreamer.rssm.get_feat(state)
        action = self.dreamer.actor(feat, deter=True)
        action = torch.clamp(action, -1.0, 1.0)
        return action[0].detach().cpu().numpy().astype(np.float32)

    def _imagine_step(self, state: dict, action: torch.Tensor) -> dict:
        if self.args.stochastic_model:
            next_state = self.dreamer.rssm.imagine_step(state, action)
        else:
            next_state = self.dreamer._deterministic_imagine_step(state, action)
        return self.dreamer.rssm.detach_state(next_state)

    def _decode_state(self, state: dict) -> tuple[np.ndarray, float]:
        feat = self.dreamer.rssm.get_feat(state)
        obs_mean = self.dreamer.obs_decoder(feat).mean
        reward_mean = self.dreamer.reward_model(feat).mean
        return tensor_image_to_hwc(obs_mean), float(reward_mean.reshape(-1)[0].detach().cpu())

    def _status_text(self, st: SessionState) -> str:
        return (
            f"env={self.args.env} | mode={st.model_mode} | control={st.control} | "
            f"step={st.step} | dim={st.selected_dim}/{max(self.action_size - 1, 0)} | "
            f"a={st.last_action_val:+.2f} | r={st.last_reward:+.3f} | "
            f"pred_r={st.pred_reward:+.3f} | return={st.episode_return:+.2f}"
        )

    def advance(self, st: SessionState) -> tuple[bytes, dict]:
        with torch.no_grad():
            posterior = self._observe_real(st)
            if st.model_state is None or st.model_mode == "anchored":
                base_state = posterior
            else:
                base_state = st.model_state

            if st.paused:
                right_frame, pred_reward = self._decode_state(base_state)
                st.pred_reward = pred_reward
                left_frame = chw_to_hwc(st.obs["image"])
                status_text = self._status_text(st)
                frame = compose_frame(left_frame, right_frame, self.args.image_size, status_text)
                status = self._status(st, status_text)
                self.last_status = status
                return frame_to_jpeg_bytes(frame, self.args.jpeg_quality), status

            if st.control == "policy":
                action_np = self._policy_action(base_state)
            else:
                action_np = self._manual_action(st)
            action_np = np.clip(action_np, self.action_low, self.action_high).astype(np.float32)
            action = torch.tensor(action_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            pred_state = self._imagine_step(base_state, action)
            right_frame, pred_reward = self._decode_state(pred_state)
            next_obs, reward, done, _ = self.env.step(action_np)
            left_frame = chw_to_hwc(next_obs["image"])

            st.step += 1
            st.last_reward = float(reward)
            st.pred_reward = pred_reward
            st.episode_return += float(reward)
            st.last_action_val = float(action_np[st.selected_dim]) if self.action_size > 0 else 0.0

            if done:
                st.obs = self.env.reset()
                st.prev_state_for_obs = self._init_rssm_state()
                st.prev_action = torch.zeros(1, self.action_size, device=self.device)
                st.model_state = None
                st.step = 0
                st.episode_return = 0.0
            else:
                st.obs = next_obs
                st.prev_state_for_obs = posterior
                st.prev_action = action
                st.model_state = pred_state

            status_text = self._status_text(st)
            frame = compose_frame(left_frame, right_frame, self.args.image_size, status_text)
            status = self._status(st, status_text)
            self.last_status = status
            return frame_to_jpeg_bytes(frame, self.args.jpeg_quality), status

    def _status(self, st: SessionState, text: str) -> dict:
        return {
            "type": "status",
            "paused": bool(st.paused),
            "control": st.control,
            "model_mode": st.model_mode,
            "text": text,
        }

    def index_html(self) -> bytes:
        html = HTML.replace("__INITIAL_MODE__", self.args.model_mode)
        return html.encode("utf-8")

    def handle_event(self, data: dict) -> None:
        st = self.state
        kind = str(data.get("type", ""))
        key = str(data.get("key", ""))

        if kind == "keydown":
            if key == "Space":
                st.paused = not st.paused
            elif key in ("KeyR", "r", "R"):
                self.pending_reset = True
            elif key in ("KeyP", "p", "P"):
                st.control = "manual" if st.control == "policy" else "policy"
            elif key in ("ArrowLeft", "ArrowRight") and self.action_size > 0:
                delta = -1 if key == "ArrowLeft" else 1
                st.selected_dim = (st.selected_dim + delta) % self.action_size
            else:
                st.keys_down.add(key)
        elif kind == "keyup":
            st.keys_down.discard(key)
        elif kind == "toggle_pause":
            st.paused = not st.paused
        elif kind == "toggle_control":
            st.control = "manual" if st.control == "policy" else "policy"
        elif kind == "set_mode":
            mode = str(data.get("model_mode", ""))
            if mode in {"anchored", "free"}:
                st.model_mode = mode
                st.model_state = None
                st.prev_state_for_obs = self._init_rssm_state()
                st.prev_action = torch.zeros(1, self.action_size, device=self.device)
        elif kind == "reset":
            self.pending_reset = True

        self.last_status = self._status(st, self._status_text(st))

    def run(self) -> None:
        handler = make_handler(self)
        httpd = ThreadingHTTPServer((self.args.host, self.args.port), handler)
        http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        http_thread.start()
        print(
            f"[web] serving DMControl vs Dreamer dynamics on "
            f"http://{self.args.host}:{self.args.port}  "
            f"(env={self.args.env}, ckpt={self.args.checkpoint_path})"
        )
        dt = 1.0 / max(1e-6, float(self.args.fps))
        try:
            while True:
                start = time.monotonic()
                with self.lock:
                    if self.pending_reset:
                        self.state = self.new_session()
                        self.pending_reset = False
                    self.latest_jpeg, self.last_status = self.advance(self.state)
                time.sleep(max(0.0, dt - (time.monotonic() - start)))
        except KeyboardInterrupt:
            print("\n[web] shutting down")
        finally:
            httpd.shutdown()
            httpd.server_close()


def make_handler(server: DreamerDynamicsServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/?"):
                body = server.index_html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.startswith("/status"):
                with server.lock:
                    body = json.dumps(server.last_status).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                dt = 1.0 / max(1e-6, float(server.args.fps))
                while True:
                    try:
                        with server.lock:
                            jpeg = server.latest_jpeg
                        if jpeg is None:
                            time.sleep(dt)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    time.sleep(dt)

            self.send_error(404)

        def do_POST(self) -> None:
            if not self.path.startswith("/event"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                data = {}
            with server.lock:
                server.handle_event(data)
            self.send_response(204)
            self.end_headers()

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play DMControl and compare it with a trained Dreamer dynamics model."
    )
    parser.add_argument("--checkpoint-path", required=True, help="dreamer.py checkpoint to restore.")
    parser.add_argument("--env", type=str, default="walker-walk")
    parser.add_argument("--algo", type=str, default="Dreamerv1", choices=["Dreamerv1", "Dreamerv2"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=1000)

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7589)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--control", choices=["manual", "policy"], default="manual")
    parser.add_argument("--model-mode", choices=["anchored", "free"], default="anchored")
    parser.add_argument("--stochastic-model", action="store_true", help="Sample RSSM priors instead of using mean/softmax.")
    parser.add_argument(
        "--render-backend",
        choices=["egl", "osmesa", "glfw"],
        default="egl",
        help="MuJoCo/DMControl rendering backend.",
    )

    parser.add_argument("--cnn-activation-function", type=str, default="relu")
    parser.add_argument("--dense-activation-function", type=str, default="elu")
    parser.add_argument("--obs-embed-size", type=int, default=1024)
    parser.add_argument("--num-units", type=int, default=400)
    parser.add_argument("--deter-size", type=int, default=200)
    parser.add_argument("--stoch-size", type=int, default=30)
    parser.add_argument("--discrete-classes", type=int, default=32)
    parser.add_argument("--use-disc-model", action="store_true")

    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-seq-len", type=int, default=50)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--action-noise", type=float, default=0.3)
    parser.add_argument("--actor-grad", type=str, default="dynamics", choices=["dynamics", "reinforce", "both"])
    parser.add_argument("--actor-grad-mix", type=float, default=0.1)
    parser.add_argument("--actor-ent", type=float, default=1e-4)

    parser.add_argument("--free-nats", type=float, default=3.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--kl-loss-coeff", type=float, default=1.0)
    parser.add_argument("--kl-alpha", type=float, default=0.8)
    parser.add_argument("--disc-loss-coeff", type=float, default=10.0)

    parser.add_argument("--model_learning-rate", type=float, default=6e-4)
    parser.add_argument("--actor_learning-rate", type=float, default=8e-5)
    parser.add_argument("--value_learning-rate", type=float, default=8e-5)
    parser.add_argument("--adam-epsilon", type=float, default=1e-7)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.restore = True
    args.train = False
    args.evaluate = True
    args.test = False
    args.max_episode_length = args.time_limit
    args.test_interval = 10000
    args.test_episodes = 1
    args.scalar_freq = 1000
    args.log_video_freq = -1
    args.max_videos_to_save = 1
    args.checkpoint_interval = 10000
    args.experience_replay = ""
    args.render = False
    args.exp_name = "dynamics_play"
    return args


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    configure_render_backend(args.render_backend)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_gpu:
        torch.cuda.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "rssm" not in checkpoint or "obs_decoder" not in checkpoint:
        keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise KeyError(
            "Expected a dreamer.py checkpoint with keys such as 'rssm', "
            f"'obs_encoder', and 'obs_decoder'. Got: {keys}"
        )

    server = DreamerDynamicsServer(args)
    server.run()


if __name__ == "__main__":
    main()
