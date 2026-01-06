import threading
import time
import random
import json
import os
import platform
from datetime import datetime
import importlib.util

# ---------------- Flask / requests ----------------
try:
    from flask import Flask, jsonify, request as flask_request, Response, stream_with_context
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ---------------- GPU / NVML ----------------
try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

# ---------------- Small model ----------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# ---------------- Voice / TTS ----------------
try:
    import speech_recognition as sr
    _HAS_SR = True
except ImportError:
    _HAS_SR = False

try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    _HAS_TTS = False

# ---------------- Tkinter ----------------
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk

# ============================================================
#  CONFIG
# ============================================================

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 5000
ENGINE_BASE = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
ENGINE_STATUS_URL = f"{ENGINE_BASE}/status"
ENGINE_INFER_URL = f"{ENGINE_BASE}/infer"
ENGINE_STREAM_URL = f"{ENGINE_BASE}/stream"
ENGINE_LOAD_URL = f"{ENGINE_BASE}/control/load"
ENGINE_UNLOAD_URL = f"{ENGINE_BASE}/control/unload"

FALLBACK_GPU_NAME = "Virtual GPU"
FALLBACK_MODEL_NAME = "Copilot-Core"
FALLBACK_VRAM_TOTAL_GB = 10.0

DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_GPU_INDEX = 0

MEMORY_FILE_BASE = "copilot_chat_memory"
SYSTEM_PROMPT_FILE = "system_prompts.json"
TTS_CONFIG_FILE = "tts_config.json"
PLUGINS_DIR = "plugins"

DEFAULT_MODELS = [
    "distilgpt2",
    "gpt2",
    "sshleifer/tiny-gpt2"
]

# ============================================================
#  ENGINE: MODEL + SERVER + SYSTEM PROMPTS
# ============================================================

class CopilotEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.model_loaded = False
        self.model_name = DEFAULT_MODEL_NAME
        self.gpu_index = DEFAULT_GPU_INDEX

        self.last_infer_time_ms = 0.0
        self.total_requests = 0

        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._real_model_ready = False

        # system prompt profiles
        self.system_prompts = {
            "default": "You are a helpful assistant.",
        }
        self.active_profile = "default"
        self._load_system_prompts()

    # ---------- System prompts ----------
    def _load_system_prompts(self):
        if os.path.exists(SYSTEM_PROMPT_FILE):
            try:
                with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.system_prompts = data.get("profiles", self.system_prompts)
                    self.active_profile = data.get("active_profile", "default")
            except Exception as e:
                print(f"[ENGINE] Failed to load system prompts: {e}")

    def _save_system_prompts(self):
        data = {
            "profiles": self.system_prompts,
            "active_profile": self.active_profile
        }
        try:
            with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ENGINE] Failed to save system prompts: {e}")

    def set_system_prompt_profile(self, name, text):
        with self.lock:
            self.system_prompts[name] = text
            self.active_profile = name
        self._save_system_prompts()

    def set_active_profile(self, name):
        with self.lock:
            if name in self.system_prompts:
                self.active_profile = name
                self._save_system_prompts()

    def get_system_prompts(self):
        with self.lock:
            return {
                "profiles": self.system_prompts,
                "active_profile": self.active_profile
            }

    def get_active_system_prompt(self):
        with self.lock:
            return self.system_prompts.get(self.active_profile, "")

    # ---------- Model loading ----------
    def load_model(self, model_name=None, gpu_index=None):
        with self.lock:
            if model_name:
                self.model_name = model_name
            if gpu_index is not None:
                self.gpu_index = gpu_index

            self._model = None
            self._tokenizer = None
            self._real_model_ready = False
            self.model_loaded = False
            self.active = False

        if _HAS_TRANSFORMERS:
            try:
                device = "cpu"
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    if 0 <= self.gpu_index < num_gpus:
                        device = f"cuda:{self.gpu_index}"
                    else:
                        device = "cuda:0"

                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                model = model.to(device)
                model.eval()

                with self.lock:
                    self._device = device
                    self._tokenizer = tokenizer
                    self._model = model
                    self._real_model_ready = True
                    self.model_loaded = True
                    self.active = True
            except Exception as e:
                print(f"[ENGINE] Failed to load real model '{self.model_name}': {e}")
                with self.lock:
                    self._model = None
                    self._tokenizer = None
                    self._real_model_ready = False
                    self.model_loaded = True
                    self.active = True
        else:
            with self.lock:
                time.sleep(0.5)
                self._model = None
                self._tokenizer = None
                self._real_model_ready = False
                self.model_loaded = True
                self.active = True

    def unload_model(self):
        with self.lock:
            self._model = None
            self._tokenizer = None
            self._real_model_ready = False
            self.model_loaded = False
            self.active = False

    def get_status(self):
        with self.lock:
            return {
                "active": self.active,
                "model_loaded": self.model_loaded,
                "model_name": self.model_name,
                "gpu_index": self.gpu_index,
                "last_infer_time_ms": self.last_infer_time_ms,
                "total_requests": self.total_requests,
                "real_model": self._real_model_ready,
                "device": self._device if self.model_loaded else "none",
                "active_profile": self.active_profile,
            }

    def _run_inference(self, full_prompt: str, max_new_tokens: int = 64):
        start = time.time()
        with self.lock:
            if not (self.active and self.model_loaded):
                self.last_infer_time_ms = (time.time() - start) * 1000.0
                return "[ENGINE] Model not loaded.", False

            real_model = self._real_model_ready
            model = self._model
            tokenizer = self._tokenizer
            device = self._device

        if real_model and model is not None and tokenizer is not None:
            try:
                inputs = tokenizer(full_prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8
                    )

                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if generated.startswith(full_prompt):
                    reply_text = generated[len(full_prompt):].strip()
                else:
                    reply_text = generated.strip()

                if not reply_text:
                    reply_text = "[ENGINE] Model replied with empty text."
            except Exception as e:
                reply_text = f"[ENGINE] Inference error: {e}"
        else:
            time.sleep(0.05 + random.random() * 0.1)
            reply_text = f"[Simulated reply] You said: {full_prompt[:80]}"

        end = time.time()
        with self.lock:
            self.last_infer_time_ms = (end - start) * 1000.0
            self.total_requests += 1

        return reply_text, real_model

    def infer(self, prompt: str, max_new_tokens: int = 64) -> str:
        system_prompt = self.get_active_system_prompt()
        if system_prompt:
            full_prompt = system_prompt.rstrip() + "\n\nUser: " + prompt + "\nAssistant:"
        else:
            full_prompt = prompt
        reply, _ = self._run_inference(full_prompt, max_new_tokens)
        return reply

    def stream_infer(self, prompt: str, max_new_tokens: int = 64):
        system_prompt = self.get_active_system_prompt()
        if system_prompt:
            full_prompt = system_prompt.rstrip() + "\n\nUser: " + prompt + "\nAssistant:"
        else:
            full_prompt = prompt

        start = time.time()
        with self.lock:
            if not (self.active and self.model_loaded):
                self.last_infer_time_ms = (time.time() - start) * 1000.0
                yield "[ENGINE] Model not loaded."
                return

            real_model = self._real_model_ready
            model = self._model
            tokenizer = self._tokenizer
            device = self._device

        if real_model and model is not None and tokenizer is not None:
            try:
                inputs = tokenizer(full_prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.8
                    )
                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                if generated.startswith(full_prompt):
                    reply_text = generated[len(full_prompt):].strip()
                else:
                    reply_text = generated.strip()
                if not reply_text:
                    reply_text = "[ENGINE] Model replied with empty text."
                # stream by chunks
                chunk_size = max(10, len(reply_text) // 20)
                for i in range(0, len(reply_text), chunk_size):
                    chunk = reply_text[i:i + chunk_size]
                    yield chunk
                    time.sleep(0.03)
            except Exception as e:
                yield f"[ENGINE] Inference error: {e}"
        else:
            time.sleep(0.05 + random.random() * 0.1)
            sim = f"[Simulated reply] You said: {full_prompt[:80]}"
            for i in range(0, len(sim), 15):
                yield sim[i:i + 15]
                time.sleep(0.03)

        end = time.time()
        with self.lock:
            self.last_infer_time_ms = (end - start) * 1000.0
            self.total_requests += 1


engine = CopilotEngine()

# ============================================================
#  FLASK APP
# ============================================================

if _HAS_FLASK:
    app = Flask(__name__)

    @app.route("/status", methods=["GET"])
    def status_endpoint():
        return jsonify(engine.get_status())

    @app.route("/control/load", methods=["POST"])
    def control_load():
        data = flask_request.get_json(silent=True) or {}
        model_name = data.get("model_name", DEFAULT_MODEL_NAME)
        gpu_index = int(data.get("gpu_index", DEFAULT_GPU_INDEX))

        threading.Thread(
            target=engine.load_model,
            args=(model_name, gpu_index),
            daemon=True
        ).start()

        return jsonify({
            "ok": True,
            "message": "Loading model",
            "model_name": model_name,
            "gpu_index": gpu_index
        })

    @app.route("/control/unload", methods=["POST"])
    def control_unload():
        threading.Thread(target=engine.unload_model, daemon=True).start()
        return jsonify({"ok": True, "message": "Unloading model"})

    @app.route("/control/activate", methods=["POST"])
    def control_activate():
        with engine.lock:
            engine.active = True
        return jsonify({"ok": True, "active": True})

    @app.route("/control/deactivate", methods=["POST"])
    def control_deactivate():
        with engine.lock:
            engine.active = False
        return jsonify({"ok": True, "active": False})

    @app.route("/control/system_prompts", methods=["GET", "POST"])
    def control_system_prompts():
        if flask_request.method == "GET":
            return jsonify(engine.get_system_prompts())
        data = flask_request.get_json(silent=True) or {}
        mode = data.get("mode", "set_profile")
        if mode == "set_profile":
            name = data.get("name", "default")
            text = data.get("text", "")
            engine.set_system_prompt_profile(name, text)
        elif mode == "activate":
            name = data.get("name", "default")
            engine.set_active_profile(name)
        return jsonify({"ok": True, "data": engine.get_system_prompts()})

    @app.route("/infer", methods=["POST"])
    def infer_endpoint():
        data = flask_request.get_json(silent=True) or {}
        prompt = data.get("prompt", "").strip()
        max_new_tokens = int(data.get("max_new_tokens", 64))

        if not prompt:
            return jsonify({"ok": False, "error": "No prompt provided"}), 400

        reply = engine.infer(prompt, max_new_tokens=max_new_tokens)
        status = engine.get_status()
        return jsonify({
            "ok": True,
            "reply": reply,
            "last_infer_time_ms": status["last_infer_time_ms"],
            "total_requests": status["total_requests"],
            "real_model": status["real_model"]
        })

    @app.route("/stream", methods=["POST"])
    def stream_endpoint():
        data = flask_request.get_json(silent=True) or {}
        prompt = data.get("prompt", "").strip()
        max_new_tokens = int(data.get("max_new_tokens", 64))
        if not prompt:
            def gen_err():
                yield "[STREAM] No prompt provided."
            return Response(stream_with_context(gen_err()), mimetype="text/plain")

        def generate():
            for chunk in engine.stream_infer(prompt, max_new_tokens=max_new_tokens):
                yield chunk

        return Response(stream_with_context(generate()), mimetype="text/plain")


def start_engine_server():
    if not _HAS_FLASK:
        print("[ENGINE] Flask not installed, engine HTTP server disabled.")
        return
    print("[ENGINE] Loading model and starting server...")
    engine.load_model(model_name=DEFAULT_MODEL_NAME, gpu_index=DEFAULT_GPU_INDEX)

    def run():
        app.run(host=ENGINE_HOST, port=ENGINE_PORT, threaded=True)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"[ENGINE] Server running on {ENGINE_BASE}")

# ============================================================
#  GPU BACKEND FOR TINKLER
# ============================================================

class GPUBackend:
    def __init__(self):
        self.gpus = []
        self.current_index = 0

        if _HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gb = mem_info.total / (1024**3)
                    self.gpus.append({
                        "index": i,
                        "handle": handle,
                        "name": name,
                        "total_gb": total_gb
                    })
            except Exception:
                self.gpus = []

        if not self.gpus:
            self.gpus.append({
                "index": 0,
                "handle": None,
                "name": FALLBACK_GPU_NAME,
                "total_gb": FALLBACK_VRAM_TOTAL_GB
            })

    def get_gpu_count(self):
        return len(self.gpus)

    def set_current_gpu(self, index):
        if 0 <= index < len(self.gpus):
            self.current_index = index

    def get_current_gpu(self):
        return self.gpus[self.current_index]

    def get_gpu_names(self):
        return [g["name"] for g in self.gpus]

    def get_vram_usage(self):
        gpu = self.get_current_gpu()
        total_gb = gpu["total_gb"]
        handle = gpu["handle"]

        if _HAS_PYNVML and handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gb = mem_info.used / (1024**3)
                return used_gb, total_gb
            except Exception:
                pass

        used_gb = round(1.0 + random.random() * 2.0, 2)
        return min(used_gb, total_gb), total_gb


gpu_backend = GPUBackend()

# ============================================================
#  ENGINE BACKEND FOR GUIS
# ============================================================

class EngineBackend:
    def _fetch_status(self):
        if not _HAS_REQUESTS:
            return None
        try:
            r = requests.get(ENGINE_STATUS_URL, timeout=0.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def get_status(self):
        data = self._fetch_status()
        if data is None:
            return {
                "active": True,
                "model_loaded": True,
                "model_name": FALLBACK_MODEL_NAME,
                "gpu_index": gpu_backend.current_index,
                "last_infer_time_ms": 0.0,
                "total_requests": 0,
                "real_model": False,
                "device": "none",
                "active_profile": "default",
            }
        return {
            "active": bool(data.get("active", True)),
            "model_loaded": bool(data.get("model_loaded", True)),
            "model_name": data.get("model_name", FALLBACK_MODEL_NAME),
            "gpu_index": int(data.get("gpu_index", gpu_backend.current_index)),
            "last_infer_time_ms": float(data.get("last_infer_time_ms", 0.0)),
            "total_requests": int(data.get("total_requests", 0)),
            "real_model": bool(data.get("real_model", False)),
            "device": data.get("device", "none"),
            "active_profile": data.get("active_profile", "default"),
        }


engine_backend = EngineBackend()

# ============================================================
#  TINKLER WINDOW (as Toplevel)
# ============================================================

class TinklerWindow:
    def __init__(self, root):
        self.bg_color = "#121212"
        self.fg_color = "#E0E0E0"
        self.accent_color = "#00D1B2"
        self.inactive_color = "#FF5555"

        self.win = tk.Toplevel(root)
        self.win.title("Copilot VRAM Monitor")
        self.win.geometry("360x260")
        self.win.configure(bg=self.bg_color)
        self.win.resizable(False, False)
        self.win.attributes("-alpha", 0.96)

        self.color_mode = tk.BooleanVar(value=True)
        self.always_on_top = tk.BooleanVar(value=True)
        self.mini_mode = tk.BooleanVar(value=False)
        self.bar_mode = tk.BooleanVar(value=False)

        self.last_active = False
        self.last_model_name = FALLBACK_MODEL_NAME

        self.win.attributes("-topmost", self.always_on_top.get())

        self._drag_start_x = 0
        self._drag_start_y = 0
        self.win.bind("<Button-1>", self._on_drag_start)
        self.win.bind("<B1-Motion>", self._on_drag_motion)

        self.top_frame = tk.Frame(self.win, bg=self.bg_color)
        self.top_frame.pack(pady=4, fill="x")

        self.canvas = tk.Canvas(
            self.top_frame, width=26, height=26,
            highlightthickness=0, bg=self.bg_color, bd=0
        )
        self.canvas.pack(side="left", padx=(4, 8))
        self.status_dot = self.canvas.create_oval(4, 4, 22, 22, fill=self.inactive_color, outline="")

        self.label_status = tk.Label(
            self.top_frame, text="Copilot: UNKNOWN",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_color, fg=self.fg_color
        )
        self.label_status.pack(side="left", padx=2)

        self.gpu_var = tk.StringVar(value=gpu_backend.get_gpu_names()[gpu_backend.current_index])
        self.gpu_menu = tk.OptionMenu(
            self.top_frame,
            self.gpu_var,
            *gpu_backend.get_gpu_names(),
            command=self._on_gpu_change
        )
        self.gpu_menu.config(bg=self.bg_color, fg=self.fg_color, highlightthickness=0, bd=1,
                             activebackground="#1E1E1E", activeforeground=self.fg_color)
        self.gpu_menu["menu"].config(bg="#1E1E1E", fg=self.fg_color)
        self.gpu_menu.pack(side="right", padx=6)

        self.mid_frame = tk.Frame(self.win, bg=self.bg_color)
        self.mid_frame.pack(pady=2, fill="x")

        self.label_gpu = tk.Label(
            self.mid_frame, text="", font=("Segoe UI", 9),
            bg=self.bg_color, fg=self.fg_color
        )
        self.label_gpu.pack(anchor="w", padx=10)

        self.label_model = tk.Label(
            self.mid_frame, text="", font=("Segoe UI", 9),
            bg=self.bg_color, fg=self.fg_color
        )
        self.label_model.pack(anchor="w", padx=10)

        self.label_vram = tk.Label(
            self.mid_frame, text="", font=("Segoe UI", 10),
            bg=self.bg_color, fg=self.fg_color
        )
        self.label_vram.pack(anchor="w", padx=10, pady=(4, 0))

        self.label_vram_bar = tk.Label(
            self.mid_frame, text="", font=("Consolas", 10),
            bg=self.bg_color, fg=self.accent_color
        )
        self.label_vram_bar.pack(anchor="w", padx=10)

        self.label_loaded = tk.Label(
            self.mid_frame, text="", font=("Segoe UI", 10),
            bg=self.bg_color, fg=self.fg_color
        )
        self.label_loaded.pack(anchor="w", padx=10, pady=(4, 0))

        self.label_updated = tk.Label(
            self.mid_frame, text="", font=("Segoe UI", 8),
            bg=self.bg_color, fg="#888888"
        )
        self.label_updated.pack(anchor="w", padx=10, pady=(2, 0))

        self.gauge_frame = tk.Frame(self.win, bg=self.bg_color)
        self.gauge_frame.pack(pady=(6, 2))

        self.gauge_canvas = tk.Canvas(
            self.gauge_frame, width=110, height=110,
            bg=self.bg_color, highlightthickness=0, bd=0
        )
        self.gauge_canvas.pack()

        self.gauge_bg_circle = self.gauge_canvas.create_oval(
            10, 10, 100, 100, outline="#333333", width=3
        )
        self.gauge_arc = self.gauge_canvas.create_arc(
            10, 10, 100, 100, start=90, extent=0,
            style="arc", outline=self.accent_color, width=4
        )
        self.gauge_text = self.gauge_canvas.create_text(
            55, 55, text="0%", fill=self.fg_color, font=("Segoe UI", 11, "bold")
        )

        self.controls_frame = tk.Frame(self.win, bg=self.bg_color)
        self.controls_frame.pack(pady=4)

        self.chk_color = tk.Checkbutton(
            self.controls_frame,
            text="Color",
            variable=self.color_mode,
            command=self._on_color_mode_toggle,
            font=("Segoe UI", 8),
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        )
        self.chk_color.grid(row=0, column=0, padx=5)

        self.chk_top = tk.Checkbutton(
            self.controls_frame,
            text="Top",
            variable=self.always_on_top,
            command=self._on_top_toggle,
            font=("Segoe UI", 8),
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        )
        self.chk_top.grid(row=0, column=1, padx=5)

        self.chk_mini = tk.Checkbutton(
            self.controls_frame,
            text="Mini",
            variable=self.mini_mode,
            command=self._on_mini_mode_toggle,
            font=("Segoe UI", 8),
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        )
        self.chk_mini.grid(row=0, column=2, padx=5)

        self.chk_bar = tk.Checkbutton(
            self.controls_frame,
            text="Dock",
            variable=self.bar_mode,
            command=self._on_bar_mode_toggle,
            font=("Segoe UI", 8),
            bg=self.bg_color,
            fg=self.fg_color,
            selectcolor=self.bg_color,
            activebackground=self.bg_color,
            activeforeground=self.fg_color
        )
        self.chk_bar.grid(row=0, column=3, padx=5)

        self.label_mini = tk.Label(
            self.win, text="", font=("Segoe UI", 10),
            bg=self.bg_color, fg=self.fg_color
        )

        self.label_bar = tk.Label(
            self.win, text="", font=("Consolas", 9),
            bg="#181818", fg=self.fg_color,
            bd=1, relief="solid"
        )

        self._update_ui()

    def _on_drag_start(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_motion(self, event):
        x = self.win.winfo_x() + event.x - self._drag_start_x
        y = self.win.winfo_y() + event.y - self._drag_start_y
        self.win.geometry(f"+{x}+{y}")

    def _on_gpu_change(self, selected_name):
        names = gpu_backend.get_gpu_names()
        if selected_name in names:
            idx = names.index(selected_name)
            gpu_backend.set_current_gpu(idx)

    def _on_color_mode_toggle(self):
        self._apply_status_visuals(self.last_active)

    def _on_top_toggle(self):
        self.win.attributes("-topmost", self.always_on_top.get())

    def _on_mini_mode_toggle(self):
        if self.mini_mode.get():
            if self.bar_mode.get():
                self.bar_mode.set(False)
                self.label_bar.pack_forget()
            self._hide_full_widgets()
            self.label_mini.pack(pady=4)
            self.win.geometry("260x70")
        else:
            self.label_mini.pack_forget()
            self._show_full_widgets()
            self.win.geometry("360x260")

    def _on_bar_mode_toggle(self):
        if self.bar_mode.get():
            if self.mini_mode.get():
                self.mini_mode.set(False)
                self.label_mini.pack_forget()
            self._hide_full_widgets()
            self.label_bar.pack(fill="x", padx=4, pady=4)
            self.win.geometry("420x60")
        else:
            self.label_bar.pack_forget()
            self._show_full_widgets()
            self.win.geometry("360x260")

    def _hide_full_widgets(self):
        for w in (self.top_frame, self.mid_frame, self.gauge_frame, self.controls_frame):
            w.pack_forget()

    def _show_full_widgets(self):
        self.top_frame.pack(pady=4, fill="x")
        self.mid_frame.pack(pady=2, fill="x")
        self.gauge_frame.pack(pady=(6, 2))
        self.controls_frame.pack(pady=4)

    def _update_ui(self):
        status = engine_backend.get_status()
        active = status["active"]
        model_loaded = status["model_loaded"]
        model_name = status["model_name"]
        gpu_index_from_engine = status["gpu_index"]
        last_infer_ms = status["last_infer_time_ms"]
        total_requests = status["total_requests"]
        real_model = status["real_model"]
        device = status["device"]

        if 0 <= gpu_index_from_engine < gpu_backend.get_gpu_count():
            gpu_backend.set_current_gpu(gpu_index_from_engine)
            self.gpu_var.set(gpu_backend.get_gpu_names()[gpu_index_from_engine])

        used_gb, total_gb = gpu_backend.get_vram_usage()
        vram_pct = (used_gb / total_gb) * 100 if total_gb > 0 else 0
        timestamp = time.strftime("%H:%M:%S")

        self.last_active = active
        self.last_model_name = model_name

        bar_len = 18
        filled = int(round((vram_pct / 100.0) * bar_len))
        bar = "█" * filled + "░" * (bar_len - filled)

        gpu = gpu_backend.get_current_gpu()
        gpu_name = gpu["name"]

        self.label_status.config(text=f"Copilot: {'ACTIVE' if active else 'INACTIVE'}")
        self.label_gpu.config(text=f"GPU: {gpu_name} ({device})")
        self.label_model.config(
            text=f"Model: {model_name} ({'real' if real_model else 'simulated'})"
        )
        self.label_vram.config(
            text=f"VRAM: {used_gb:.2f} GB / {total_gb:.1f} GB"
        )
        self.label_vram_bar.config(
            text=f"[{bar}] {vram_pct:.0f}%"
        )
        self.label_loaded.config(
            text=f"Model in VRAM: {'YES' if model_loaded else 'NO'} | Last infer: {last_infer_ms:.1f} ms | "
                 f"Reqs: {total_requests}"
        )
        self.label_updated.config(text=f"Last updated: {timestamp}")

        self.label_mini.config(
            text=f"{'●' if active else '○'} {used_gb:.2f}/{total_gb:.1f} GB | {vram_pct:.0f}%"
        )

        self.label_bar.config(
            text=f"Copilot {'ON' if active else 'OFF'} | GPU {gpu_index_from_engine}: {gpu_name} | "
                 f"{used_gb:.2f}/{total_gb:.1f} GB ({vram_pct:.0f}%) | Model: {model_name} "
                 f"| Real: {real_model} | Dev: {device}"
        )

        self._update_gauge(vram_pct)
        self._apply_status_visuals(active)

        self.win.after(1000, self._update_ui)

    def _update_gauge(self, pct):
        extent = -(pct / 100.0) * 360.0
        self.gauge_canvas.itemconfig(self.gauge_arc, extent=extent)
        self.gauge_canvas.itemconfig(self.gauge_text, text=f"{pct:.0f}%")

    def _apply_status_visuals(self, active):
        if self.color_mode.get():
            dot_color = self.accent_color if active else self.inactive_color
        else:
            dot_color = self.fg_color
        self.canvas.itemconfig(self.status_dot, fill=dot_color)

# ============================================================
#  PLUGINS
# ============================================================

class PluginManager:
    def __init__(self, plugins_dir):
        self.plugins_dir = plugins_dir
        self.commands = {}  # name -> handler

        if not os.path.exists(self.plugins_dir):
            os.makedirs(self.plugins_dir, exist_ok=True)

        self._load_plugins()

    def _load_plugins(self):
        for fname in os.listdir(self.plugins_dir):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(self.plugins_dir, fname)
            try:
                spec = importlib.util.spec_from_file_location(fname[:-3], path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "register"):
                    info = mod.register()
                    name = info.get("name")
                    trigger = info.get("trigger")
                    handler = info.get("handler")
                    if name and trigger and handler:
                        self.commands[trigger.lower()] = handler
                        print(f"[PLUGIN] Loaded {name} ({trigger})")
            except Exception as e:
                print(f"[PLUGIN] Failed to load {fname}: {e}")

    def handle(self, cmd_text: str):
        parts = cmd_text.strip().split()
        if not parts:
            return None
        trigger = parts[0].lower()
        args = parts[1:]
        handler = self.commands.get(trigger)
        if handler:
            try:
                return handler(args)
            except Exception as e:
                return f"[Plugin error] {e}"
        return None


plugin_manager = PluginManager(PLUGINS_DIR)

# ============================================================
#  MEMORY BROWSER
# ============================================================

class MemoryBrowser:
    def __init__(self, root, get_memory_file_func):
        self.get_memory_file = get_memory_file_func
        self.bg = "#121212"
        self.fg = "#E0E0E0"

        self.win = tk.Toplevel(root)
        self.win.title("Memory Browser")
        self.win.geometry("600x500")
        self.win.configure(bg=self.bg)
        self.win.resizable(False, False)

        top = tk.Frame(self.win, bg=self.bg)
        top.pack(fill="x", padx=10, pady=(10, 5))

        tk.Label(
            top, text="Search:", bg=self.bg, fg=self.fg,
            font=("Segoe UI", 9)
        ).pack(side="left")

        self.search_var = tk.StringVar()
        self.entry_search = tk.Entry(
            top, textvariable=self.search_var,
            bg="#1E1E1E", fg=self.fg,
            insertbackground=self.fg,
            relief="flat"
        )
        self.entry_search.pack(side="left", fill="x", expand=True, padx=(5, 10))
        self.entry_search.bind("<Return>", lambda e: self._reload())

        self.btn_search = tk.Button(
            top, text="Go", bg="#1E1E1E", fg=self.fg,
            font=("Segoe UI", 8), relief="flat",
            command=self._reload
        )
        self.btn_search.pack(side="left")

        mid = tk.Frame(self.win, bg=self.bg)
        mid.pack(fill="both", expand=True, padx=10, pady=5)

        self.listbox = tk.Listbox(
            mid, bg="#1E1E1E", fg=self.fg,
            font=("Consolas", 9), selectmode=tk.SINGLE
        )
        self.listbox.pack(side="left", fill="y")

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        self.detail = scrolledtext.ScrolledText(
            mid,
            wrap=tk.WORD,
            bg=self.bg,
            fg=self.fg,
            font=("Segoe UI", 10),
            insertbackground=self.fg,
            state=tk.DISABLED
        )
        self.detail.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.records = []
        self._reload()

    def _load_records(self):
        self.records = []
        path = self.get_memory_file()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self.records.append(obj)
                    except Exception:
                        continue
        except Exception:
            pass

    def _reload(self):
        query = self.search_var.get().strip().lower()
        self._load_records()
        self.listbox.delete(0, tk.END)
        for i, rec in enumerate(self.records):
            text = rec.get("text", "")
            role = rec.get("role", "?")
            ts = rec.get("timestamp", "")
            if query and query not in text.lower() and query not in role.lower():
                continue
            label = f"{i:04d} | {role[:6]:<6} | {ts}"
            self.listbox.insert(tk.END, label)

    def _on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = int(self.listbox.get(selection[0]).split("|")[0].strip())
        if 0 <= idx < len(self.records):
            rec = self.records[idx]
            text = json.dumps(rec, ensure_ascii=False, indent=2)
            self.detail.config(state=tk.NORMAL)
            self.detail.delete("1.0", tk.END)
            self.detail.insert(tk.END, text)
            self.detail.config(state=tk.DISABLED)

# ============================================================
#  SYSTEM PROMPT EDITOR
# ============================================================

class SystemPromptEditor:
    def __init__(self, root):
        self.bg = "#121212"
        self.fg = "#E0E0E0"
        self.win = tk.Toplevel(root)
        self.win.title("System Prompt Editor")
        self.win.geometry("600x500")
        self.win.configure(bg=self.bg)
        self.win.resizable(False, False)

        top = tk.Frame(self.win, bg=self.bg)
        top.pack(fill="x", padx=10, pady=(10, 5))

        tk.Label(top, text="Profile:", bg=self.bg, fg=self.fg, font=("Segoe UI", 9)).pack(side="left")

        self.profile_var = tk.StringVar(value="default")
        self.combo_profiles = ttk.Combobox(
            top, textvariable=self.profile_var,
            values=["default"], state="readonly"
        )
        self.combo_profiles.pack(side="left", padx=(5, 10))
        self.combo_profiles.bind("<<ComboboxSelected>>", self._on_profile_change)

        self.btn_new = tk.Button(
            top, text="New", bg="#1E1E1E", fg=self.fg,
            font=("Segoe UI", 8), relief="flat",
            command=self._new_profile
        )
        self.btn_new.pack(side="left", padx=(0, 5))

        self.btn_save = tk.Button(
            top, text="Save", bg="#1E1E1E", fg=self.fg,
            font=("Segoe UI", 8), relief="flat",
            command=self._save_profile
        )
        self.btn_save.pack(side="left")

        self.btn_activate = tk.Button(
            top, text="Activate", bg="#1E1E1E", fg=self.fg,
            font=("Segoe UI", 8), relief="flat",
            command=self._activate_profile
        )
        self.btn_activate.pack(side="left", padx=(5, 0))

        self.text = scrolledtext.ScrolledText(
            self.win, wrap=tk.WORD,
            bg=self.bg, fg=self.fg,
            font=("Segoe UI", 10),
            insertbackground=self.fg
        )
        self.text.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self._load_from_engine()

    def _load_from_engine(self):
        if not _HAS_REQUESTS:
            return
        try:
            r = requests.get(f"{ENGINE_BASE}/control/system_prompts", timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                profiles = data.get("profiles", {})
                active = data.get("active_profile", "default")
                names = list(profiles.keys())
                self.combo_profiles["values"] = names
                self.profile_var.set(active)
                self.text.delete("1.0", tk.END)
                self.text.insert(tk.END, profiles.get(active, ""))
        except Exception:
            pass

    def _on_profile_change(self, event=None):
        name = self.profile_var.get()
        if not _HAS_REQUESTS:
            return
        try:
            r = requests.get(f"{ENGINE_BASE}/control/system_prompts", timeout=1.0)
            if r.status_code == 200:
                data = r.json()
                profiles = data.get("profiles", {})
                self.text.delete("1.0", tk.END)
                self.text.insert(tk.END, profiles.get(name, ""))
        except Exception:
            pass

    def _new_profile(self):
        name = f"profile_{int(time.time())}"
        self.profile_var.set(name)
        vals = list(self.combo_profiles["values"])
        vals.append(name)
        self.combo_profiles["values"] = vals
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "You are a helpful assistant with a new persona.")

    def _save_profile(self):
        name = self.profile_var.get()
        text = self.text.get("1.0", tk.END).strip()
        if not name or not _HAS_REQUESTS:
            return
        try:
            requests.post(
                f"{ENGINE_BASE}/control/system_prompts",
                json={"mode": "set_profile", "name": name, "text": text},
                timeout=2.0
            )
        except Exception:
            pass

    def _activate_profile(self):
        name = self.profile_var.get()
        if not name or not _HAS_REQUESTS:
            return
        try:
            requests.post(
                f"{ENGINE_BASE}/control/system_prompts",
                json={"mode": "activate", "name": name},
                timeout=2.0
            )
        except Exception:
            pass

# ============================================================
#  TTS SETTINGS WINDOW
# ============================================================

class TTSSettingsWindow:
    def __init__(self, root, tts_engine):
        self.tts_engine = tts_engine
        self.bg = "#121212"
        self.fg = "#E0E0E0"

        self.win = tk.Toplevel(root)
        self.win.title("Voice Settings")
        self.win.geometry("400x250")
        self.win.configure(bg=self.bg)
        self.win.resizable(False, False)

        top = tk.Frame(self.win, bg=self.bg)
        top.pack(fill="x", padx=10, pady=(10, 5))

        tk.Label(top, text="Voice:", bg=self.bg, fg=self.fg, font=("Segoe UI", 9)).pack(anchor="w")

        self.voice_var = tk.StringVar(value="")
        self.voices = []
        if self.tts_engine:
            try:
                self.voices = self.tts_engine.getProperty("voices")
            except Exception:
                self.voices = []

        voice_names = [v.name for v in self.voices] if self.voices else []
        self.combo_voices = ttk.Combobox(
            top, textvariable=self.voice_var,
            values=voice_names, state="readonly"
        )
        self.combo_voices.pack(fill="x", pady=(2, 5))

        sliders = tk.Frame(self.win, bg=self.bg)
        sliders.pack(fill="x", padx=10, pady=5)

        tk.Label(sliders, text="Rate:", bg=self.bg, fg=self.fg, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        self.rate_scale = tk.Scale(
            sliders, from_=50, to=300, orient=tk.HORIZONTAL,
            bg=self.bg, fg=self.fg, troughcolor="#1E1E1E",
            highlightthickness=0
        )
        self.rate_scale.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        tk.Label(sliders, text="Volume:", bg=self.bg, fg=self.fg, font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w")
        self.volume_scale = tk.Scale(
            sliders, from_=0, to=100, orient=tk.HORIZONTAL,
            bg=self.bg, fg=self.fg, troughcolor="#1E1E1E",
            highlightthickness=0
        )
        self.volume_scale.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        sliders.columnconfigure(1, weight=1)

        bottom = tk.Frame(self.win, bg=self.bg)
        bottom.pack(fill="x", padx=10, pady=(10, 10))

        self.btn_test = tk.Button(
            bottom, text="Test", bg="#1E1E1E", fg=self.fg,
            font=("Segoe UI", 8), relief="flat",
            command=self._test_voice
        )
        self.btn_test.pack(side="left")

        self.btn_save = tk.Button(
            bottom, text="Save", bg="#1E1E1E", fg=self.fg,
            font=("Segoe UI", 8), relief="flat",
            command=self._save
        )
        self.btn_save.pack(side="right")

        self._load_config()

    def _load_config(self):
        if not self.tts_engine:
            return
        rate = self.tts_engine.getProperty("rate")
        volume = self.tts_engine.getProperty("volume")
        self.rate_scale.set(int(rate))
        self.volume_scale.set(int(volume * 100))

        if os.path.exists(TTS_CONFIG_FILE):
            try:
                with open(TTS_CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                voice_id = cfg.get("voice_id", "")
                rate = cfg.get("rate", rate)
                volume = cfg.get("volume", volume)
                self.rate_scale.set(int(rate))
                self.volume_scale.set(int(volume * 100))
                if voice_id and self.voices:
                    for v in self.voices:
                        if v.id == voice_id or v.name == voice_id:
                            self.voice_var.set(v.name)
                            self.tts_engine.setProperty("voice", v.id)
                            break
            except Exception:
                pass

    def _apply_current(self):
        if not self.tts_engine:
            return
        rate = self.rate_scale.get()
        volume = self.volume_scale.get() / 100.0
        self.tts_engine.setProperty("rate", int(rate))
        self.tts_engine.setProperty("volume", float(volume))

        name = self.voice_var.get()
        if name and self.voices:
            for v in self.voices:
                if v.name == name:
                    self.tts_engine.setProperty("voice", v.id)
                    break

    def _test_voice(self):
        if not self.tts_engine:
            return
        self._apply_current()
        try:
            self.tts_engine.stop()
            self.tts_engine.say("This is a test of the current voice settings.")
            self.tts_engine.runAndWait()
        except Exception:
            pass

    def _save(self):
        if not self.tts_engine:
            return
        self._apply_current()
        voice_id = ""
        name = self.voice_var.get()
        if name and self.voices:
            for v in self.voices:
                if v.name == name:
                    voice_id = v.id
                    break
        cfg = {
            "voice_id": voice_id or name,
            "rate": self.rate_scale.get(),
            "volume": self.volume_scale.get() / 100.0
        }
        try:
            with open(TTS_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# ============================================================
#  CHAT WINDOW (as Toplevel)
# ============================================================

class ChatWindow:
    def __init__(self, root):
        self.bg = "#121212"
        self.fg = "#E0E0E0"
        self.accent = "#00D1B2"
        self.error_color = "#FF5555"

        self.root = root

        self.win = tk.Toplevel(root)
        self.win.title("Copilot Chat")
        self.win.geometry("800x720")
        self.win.configure(bg=self.bg)
        self.win.resizable(False, False)

        # session
        self.session_id = 1
        self.streaming_enabled = tk.BooleanVar(value=True)
        self.continuous_listen = tk.BooleanVar(value=False)

        # TTS
        self.tts_enabled = tk.BooleanVar(value=False)
        if _HAS_TTS:
            self.tts_engine = pyttsx3.init()
            rate = self.tts_engine.getProperty("rate")
            self.tts_engine.setProperty("rate", int(rate * 0.9))
            self._load_tts_config()
        else:
            self.tts_engine = None

        # Voice
        self.recognizer = sr.Recognizer() if _HAS_SR else None
        self.listening = False
        self.continuous_thread = None
        self.continuous_stop = threading.Event()

        # plugins
        self.plugin_manager = plugin_manager

        self._build_top_bar()
        self._build_chat_box()
        self._build_bottom_bar()

        self._load_memory_info()
        self._update_engine_status()

        if self.continuous_listen.get():
            self._start_continuous_listen()

    # ---------- TTS config load ----------
    def _load_tts_config(self):
        if not self.tts_engine:
            return
        if os.path.exists(TTS_CONFIG_FILE):
            try:
                with open(TTS_CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                voice_id = cfg.get("voice_id", "")
                rate = cfg.get("rate", self.tts_engine.getProperty("rate"))
                volume = cfg.get("volume", self.tts_engine.getProperty("volume"))
                self.tts_engine.setProperty("rate", int(rate))
                self.tts_engine.setProperty("volume", float(volume))
                voices = self.tts_engine.getProperty("voices")
                if voice_id and voices:
                    for v in voices:
                        if v.id == voice_id or v.name == voice_id:
                            self.tts_engine.setProperty("voice", v.id)
                            break
            except Exception:
                pass

    # ---------- UI ----------
    def _build_top_bar(self):
        top = tk.Frame(self.win, bg=self.bg)
        top.pack(fill="x", padx=10, pady=(10, 5))

        left = tk.Frame(top, bg=self.bg)
        left.pack(side="left", fill="x", expand=True)

        self.label_status = tk.Label(
            left, text="Engine: unknown",
            bg=self.bg, fg="#AAAAAA",
            font=("Segoe UI", 9)
        )
        self.label_status.pack(anchor="w")

        self.label_memory = tk.Label(
            left, text="Memory: 0 entries (session 1)",
            bg=self.bg, fg="#888888",
            font=("Segoe UI", 9)
        )
        self.label_memory.pack(anchor="w", pady=(2, 0))

        right = tk.Frame(top, bg=self.bg)
        right.pack(side="right")

        # model selector
        tk.Label(
            right, text="Model:", bg=self.bg, fg=self.fg, font=("Segoe UI", 9)
        ).pack(side="left", padx=(0, 3))

        self.model_var = tk.StringVar(value=DEFAULT_MODEL_NAME)
        self.combo_model = ttk.Combobox(
            right, textvariable=self.model_var,
            values=DEFAULT_MODELS, state="readonly",
            width=18
        )
        self.combo_model.pack(side="left")
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_change)

        # stream toggle
        self.chk_stream = tk.Checkbutton(
            right,
            text="Stream",
            variable=self.streaming_enabled,
            bg=self.bg,
            fg=self.fg,
            selectcolor=self.bg,
            activebackground=self.bg,
            activeforeground=self.fg,
            font=("Segoe UI", 8)
        )
        self.chk_stream.pack(side="left", padx=(5, 0))

        # new session button
        self.btn_new_session = tk.Button(
            right,
            text="New Session",
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 8),
            relief="flat",
            command=self._new_session
        )
        self.btn_new_session.pack(side="left", padx=(5, 0))

    def _build_chat_box(self):
        self.chat_box = scrolledtext.ScrolledText(
            self.win,
            wrap=tk.WORD,
            bg=self.bg,
            fg=self.fg,
            font=("Segoe UI", 11),
            insertbackground=self.fg,
            state=tk.DISABLED
        )
        self.chat_box.pack(padx=10, pady=5, fill="both", expand=True)

    def _build_bottom_bar(self):
        bottom = tk.Frame(self.win, bg=self.bg)
        bottom.pack(fill="x", padx=10, pady=(5, 10))

        left = tk.Frame(bottom, bg=self.bg)
        left.pack(side="left", padx=(0, 10))

        self.btn_save_log = tk.Button(
            left,
            text="Save Log...",
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 8),
            relief="flat",
            command=self._save_chat_log
        )
        self.btn_save_log.pack(side="left", padx=3)

        self.btn_clear_mem = tk.Button(
            left,
            text="Clear Memory",
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 8),
            relief="flat",
            command=self._clear_memory
        )
        self.btn_clear_mem.pack(side="left", padx=3)

        self.btn_memory_browser = tk.Button(
            left,
            text="Memory Browser",
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 8),
            relief="flat",
            command=self._open_memory_browser
        )
        self.btn_memory_browser.pack(side="left", padx=3)

        self.btn_sys_prompt = tk.Button(
            left,
            text="System Prompt",
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 8),
            relief="flat",
            command=self._open_system_prompt_editor
        )
        self.btn_sys_prompt.pack(side="left", padx=3)

        self.btn_tts_settings = tk.Button(
            left,
            text="Voice Settings",
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 8),
            relief="flat",
            command=self._open_tts_settings
        )
        self.btn_tts_settings.pack(side="left", padx=3)

        right = tk.Frame(bottom, bg=self.bg)
        right.pack(side="right", fill="x", expand=True)

        if _HAS_SR:
            self.chk_continuous = tk.Checkbutton(
                right,
                text="Auto Listen",
                variable=self.continuous_listen,
                bg=self.bg,
                fg=self.fg,
                selectcolor=self.bg,
                activebackground=self.bg,
                activeforeground=self.fg,
                font=("Segoe UI", 8),
                command=self._on_continuous_toggle
            )
            self.chk_continuous.pack(side="right", padx=(5, 0))

            self.btn_mic = tk.Button(
                right,
                text="🎙",
                bg="#1E1E1E",
                fg=self.fg,
                font=("Segoe UI", 10),
                relief="flat",
                command=self._toggle_listen
            )
            self.btn_mic.pack(side="right", padx=(5, 0))

        if _HAS_TTS:
            self.chk_tts = tk.Checkbutton(
                right,
                text="TTS",
                variable=self.tts_enabled,
                bg=self.bg,
                fg=self.fg,
                selectcolor=self.bg,
                activebackground=self.bg,
                activeforeground=self.fg,
                font=("Segoe UI", 8)
            )
            self.chk_tts.pack(side="right", padx=(5, 5))

        self.send_btn = tk.Button(
            right,
            text="Send",
            bg=self.accent,
            fg="black",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            command=self.send_message
        )
        self.send_btn.pack(side="right", padx=(5, 0))

        self.entry = tk.Entry(
            right,
            bg="#1E1E1E",
            fg=self.fg,
            font=("Segoe UI", 11),
            insertbackground=self.fg,
            relief="flat"
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

    # ---------- Engine status poll ----------
    def _update_engine_status(self):
        if not _HAS_REQUESTS:
            self.label_status.config(text="Engine: requests not installed", fg=self.error_color)
        else:
            try:
                r = requests.get(ENGINE_STATUS_URL, timeout=0.5)
                if r.status_code == 200:
                    data = r.json()
                    active = data.get("active", False)
                    model = data.get("model_name", "?")
                    device = data.get("device", "none")
                    real_model = data.get("real_model", False)
                    profile = data.get("active_profile", "default")
                    txt = f"Engine: {'ACTIVE' if active else 'INACTIVE'} | {model} | {device} | {'real' if real_model else 'sim'} | Profile: {profile}"
                    self.label_status.config(text=txt, fg=self.fg if active else "#AAAAAA")
                else:
                    self.label_status.config(text=f"Engine: HTTP {r.status_code}", fg=self.error_color)
            except Exception:
                self.label_status.config(text="Engine: unreachable", fg=self.error_color)

        self.win.after(2000, self._update_engine_status)

    # ---------- Memory ----------
    def _memory_file(self):
        return f"{MEMORY_FILE_BASE}_session{self.session_id}.jsonl"

    def _append_memory(self, role, text):
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "text": text,
            "session": self.session_id
        }
        try:
            with open(self._memory_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self._append_system(f"[Memory write error] {e}")
        self._load_memory_info()

    def _load_memory_info(self):
        count = 0
        path = self._memory_file()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for _ in f:
                        count += 1
            except Exception:
                pass
        self.label_memory.config(text=f"Memory: {count} entries (session {self.session_id})")

    def _clear_memory(self):
        path = self._memory_file()
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                self._append_system(f"[Memory clear error] {e}")
        self._load_memory_info()
        self._append_system("Conversation memory cleared.")

    # ---------- Chat display ----------
    def _append_chat(self, text, prefix=""):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"{prefix}{text}\n\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _append_system(self, text):
        self._append_chat(text, prefix="⚙ System: ")

    # ---------- Save log ----------
    def _save_chat_log(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            content = self.chat_box.get("1.0", tk.END)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self._append_system(f"Chat log saved to {file_path}")
        except Exception as e:
            self._append_system(f"[Save error] {e}")

    # ---------- Memory browser / System prompt / TTS settings ----------
    def _open_memory_browser(self):
        MemoryBrowser(self.root, self._memory_file)

    def _open_system_prompt_editor(self):
        SystemPromptEditor(self.root)

    def _open_tts_settings(self):
        if not self.tts_engine:
            self._append_system("TTS engine not available.")
            return
        TTSSettingsWindow(self.root, self.tts_engine)

    # ---------- Plugins + built-in commands ----------
    def _handle_command(self, cmd: str) -> bool:
        parts = cmd.strip().split()
        if not parts:
            return False
        name = parts[0].lower()

        if name == "/time":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._append_system(f"Local time: {now}")
            self._append_memory("system", f"/time -> {now}")
            return True

        if name == "/sysinfo":
            info = f"{platform.system()} {platform.release()} | Python {platform.python_version()}"
            self._append_system(f"System info: {info}")
            self._append_memory("system", f"/sysinfo -> {info}")
            return True

        if name == "/help":
            help_text = (
                "Commands:\n"
                "  /time        - show local time\n"
                "  /sysinfo     - show basic system info\n"
                "  /help        - this help\n"
                "  /session N   - switch to session N\n"
            )
            self._append_system(help_text)
            self._append_memory("system", "/help used")
            return True

        if name == "/session" and len(parts) >= 2:
            try:
                n = int(parts[1])
                if n >= 1:
                    self._switch_session(n)
                    return True
            except Exception:
                self._append_system("Usage: /session N  (N >= 1)")
                return True

        # plugin manager
        result = self.plugin_manager.handle(cmd)
        if result is not None:
            self._append_system(result)
            self._append_memory("system", f"plugin: {cmd} -> {result}")
            return True

        return False

    # ---------- Send to engine ----------
    def send_message(self, event=None):
        prompt = self.entry.get().strip()
        if not prompt:
            return

        self.entry.delete(0, tk.END)

        if prompt.startswith("/"):
            handled = self._handle_command(prompt)
            if handled:
                return

        self._append_chat(prompt, prefix="🧑 You: ")
        self._append_memory("user", prompt)

        if not _HAS_REQUESTS:
            self._append_chat("requests not installed; cannot reach engine.", prefix="❌ Error: ")
            return

        if self.streaming_enabled.get():
            self._send_streaming(prompt)
        else:
            self._send_blocking(prompt)

    def _send_blocking(self, prompt: str):
        def worker():
            try:
                start = time.time()
                r = requests.post(
                    ENGINE_INFER_URL,
                    json={"prompt": prompt, "max_new_tokens": 80},
                    timeout=30
                )
                elapsed = (time.time() - start) * 1000

                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "[No reply]")
                    infer_ms = data.get("last_infer_time_ms", 0.0)
                    real_model = data.get("real_model", False)

                    meta = f"(engine {infer_ms:.1f} ms | gui {elapsed:.1f} ms | {'real' if real_model else 'sim'})"
                    full_reply = f"{reply}\n{meta}"

                    def ui():
                        self._append_chat(full_reply, prefix="🤖 Copilot: ")
                        self._append_memory("assistant", reply)
                        if self.tts_enabled.get() and self.tts_engine is not None:
                            self._speak(reply)
                    self.win.after(0, ui)
                else:
                    def ui_err():
                        self._append_chat(
                            f"[Error {r.status_code}] Engine returned an error.",
                            prefix="❌ Error: "
                        )
                    self.win.after(0, ui_err)
            except Exception as e:
                def ui_exc():
                    self._append_chat(f"[Engine unreachable] {e}", prefix="❌ Error: ")
                self.win.after(0, ui_exc)

        threading.Thread(target=worker, daemon=True).start()

    def _send_streaming(self, prompt: str):
        def worker():
            try:
                start = time.time()
                with requests.post(
                    ENGINE_STREAM_URL,
                    json={"prompt": prompt, "max_new_tokens": 80},
                    timeout=60,
                    stream=True
                ) as r:
                    if r.status_code != 200:
                        def ui_err():
                            self._append_chat(
                                f"[Error {r.status_code}] Engine stream error.",
                                prefix="❌ Error: "
                            )
                        self.win.after(0, ui_err)
                        return

                    chunks = []
                    def ui_start():
                        self._append_chat("", prefix="🤖 Copilot: ")
                    self.win.after(0, ui_start)

                    for chunk in r.iter_content(chunk_size=64, decode_unicode=True):
                        if not chunk:
                            continue
                        chunks.append(chunk)
                        text = chunk

                        def ui_chunk(t=text):
                            # insert before last two newlines
                            self.chat_box.config(state=tk.NORMAL)
                            # insert t before the last newline (the placeholder Copilot line)
                            self.chat_box.insert("end-3l", t)
                            self.chat_box.config(state=tk.DISABLED)
                            self.chat_box.see(tk.END)
                        self.win.after(0, ui_chunk)

                    reply = "".join(chunks)
                    elapsed = (time.time() - start) * 1000

                    def ui_done():
                        meta = f"(stream gui {elapsed:.1f} ms)"
                        self._append_chat(meta, prefix="")
                        self._append_memory("assistant", reply)
                        if self.tts_enabled.get() and self.tts_engine is not None:
                            self._speak(reply)
                    self.win.after(0, ui_done)
            except Exception as e:
                def ui_exc():
                    self._append_chat(f"[Stream error] {e}", prefix="❌ Error: ")
                self.win.after(0, ui_exc)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- TTS ----------
    def _speak(self, text: str):
        if not (self.tts_engine and _HAS_TTS):
            return
        def worker():
            try:
                self.tts_engine.stop()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.win.after(0, lambda: self._append_system(f"[TTS error] {e}"))
        threading.Thread(target=worker, daemon=True).start()

    # ---------- Voice ----------
    def _toggle_listen(self):
        if not (_HAS_SR and self.recognizer):
            self._append_system("Speech recognition not available (install 'speech_recognition' and 'pyaudio').")
            return

        if self.listening:
            return

        self.listening = True
        self._append_system("Listening... speak now.")
        self.win.after(100, self._do_listen_once)

    def _do_listen_once(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = self.recognizer.recognize_google(audio)
            self._append_system(f"Heard: {text}")
            self.entry.delete(0, tk.END)
            self.entry.insert(0, text)
        except Exception as e:
            self._append_system(f"[Voice error] {e}")
        finally:
            self.listening = False

    def _continuous_loop(self):
        while not self.continuous_stop.is_set():
            if not (_HAS_SR and self.recognizer):
                break
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                # simple wake-word: "copilot"
                if "copilot" in text.lower():
                    cleaned = text.lower().replace("copilot", "").strip()
                    if not cleaned:
                        continue
                    def ui_cmd(c=cleaned):
                        self._append_system(f"Wake word heard, sending: {c}")
                        self.entry.delete(0, tk.END)
                        self.entry.insert(0, c)
                        self.send_message()
                    self.win.after(0, ui_cmd)
            except Exception:
                pass
            time.sleep(0.5)

    def _start_continuous_listen(self):
        if not (_HAS_SR and self.recognizer):
            self._append_system("Speech recognition not available for auto listening.")
            self.continuous_listen.set(False)
            return
        if self.continuous_thread and self.continuous_thread.is_alive():
            return
        self.continuous_stop.clear()
        self.continuous_thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self.continuous_thread.start()
        self._append_system("Continuous listening started (wake word: 'copilot').")

    def _stop_continuous_listen(self):
        self.continuous_stop.set()
        self._append_system("Continuous listening stopped.")

    def _on_continuous_toggle(self):
        if self.continuous_listen.get():
            self._start_continuous_listen()
        else:
            self._stop_continuous_listen()

    # ---------- Model change ----------
    def _on_model_change(self, event=None):
        model_name = self.model_var.get().strip()
        if not model_name or not _HAS_REQUESTS:
            return

        def worker():
            try:
                requests.post(
                    ENGINE_LOAD_URL,
                    json={"model_name": model_name, "gpu_index": DEFAULT_GPU_INDEX},
                    timeout=2.0
                )
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()
        self._append_system(f"Requested model switch to '{model_name}'")

    # ---------- Sessions ----------
    def _switch_session(self, n: int):
        if n == self.session_id:
            return
        self.session_id = n
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.delete("1.0", tk.END)
        self.chat_box.config(state=tk.DISABLED)
        self._append_system(f"Switched to session {n}.")
        self._load_memory_info()

    def _new_session(self):
        self.session_id += 1
        self._switch_session(self.session_id)

    # ---------- Misc ----------
    def _on_model_loaded(self):
        pass


# ============================================================
#  MAIN ORGANISM STARTUP
# ============================================================

def main():
    start_engine_server()

    root = tk.Tk()
    root.withdraw()  # hide base root window

    TinklerWindow(root)
    ChatWindow(root)

    root.mainloop()


if __name__ == "__main__":
    main()

