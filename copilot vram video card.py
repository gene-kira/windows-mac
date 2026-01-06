import threading
import time
import random
import json
import os
import platform
from datetime import datetime

# ---------------- Flask / requests ----------------
try:
    from flask import Flask, jsonify, request as flask_request
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
from tkinter import scrolledtext, filedialog

# ============================================================
#  CONFIG
# ============================================================

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 5000
ENGINE_BASE = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
ENGINE_STATUS_URL = f"{ENGINE_BASE}/status"
ENGINE_INFER_URL = f"{ENGINE_BASE}/infer"

FALLBACK_GPU_NAME = "Virtual GPU"
FALLBACK_MODEL_NAME = "Copilot-Core"
FALLBACK_VRAM_TOTAL_GB = 10.0

DEFAULT_MODEL_NAME = "distilgpt2"
DEFAULT_GPU_INDEX = 0

MEMORY_FILE = "copilot_chat_memory.jsonl"

# ============================================================
#  ENGINE: MODEL + SERVER
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
            }

    def infer(self, prompt: str, max_new_tokens: int = 64) -> str:
        start = time.time()
        with self.lock:
            if not (self.active and self.model_loaded):
                self.last_infer_time_ms = (time.time() - start) * 1000.0
                return "[ENGINE] Model not loaded."

            real_model = self._real_model_ready
            model = self._model
            tokenizer = self._tokenizer
            device = self._device

        if real_model and model is not None and tokenizer is not None:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
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
                if generated.startswith(prompt):
                    reply_text = generated[len(prompt):].strip()
                else:
                    reply_text = generated.strip()

                if not reply_text:
                    reply_text = "[ENGINE] Model replied with empty text."
            except Exception as e:
                reply_text = f"[ENGINE] Inference error: {e}"
        else:
            time.sleep(0.05 + random.random() * 0.1)
            reply_text = f"[Simulated reply] You said: {prompt[:80]}"

        end = time.time()
        with self.lock:
            self.last_infer_time_ms = (end - start) * 1000.0
            self.total_requests += 1

        return reply_text


engine = CopilotEngine()

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
#  CHAT WINDOW (as Toplevel)
# ============================================================

class ChatWindow:
    def __init__(self, root):
        self.bg = "#121212"
        self.fg = "#E0E0E0"
        self.accent = "#00D1B2"
        self.error_color = "#FF5555"

        self.win = tk.Toplevel(root)
        self.win.title("Copilot Chat")
        self.win.geometry("640x720")
        self.win.configure(bg=self.bg)
        self.win.resizable(False, False)

        self.tts_enabled = tk.BooleanVar(value=False)
        if _HAS_TTS:
            self.tts_engine = pyttsx3.init()
            rate = self.tts_engine.getProperty("rate")
            self.tts_engine.setProperty("rate", int(rate * 0.9))
        else:
            self.tts_engine = None

        self.recognizer = sr.Recognizer() if _HAS_SR else None
        self.listening = False

        self._build_top_bar()
        self._build_chat_box()
        self._build_bottom_bar()

        self._load_memory_info()
        self._update_engine_status()

    def _build_top_bar(self):
        top = tk.Frame(self.win, bg=self.bg)
        top.pack(fill="x", padx=10, pady=(10, 5))

        self.label_status = tk.Label(
            top, text="Engine: unknown",
            bg=self.bg, fg="#AAAAAA",
            font=("Segoe UI", 9)
        )
        self.label_status.pack(side="left")

        self.label_memory = tk.Label(
            top, text="Memory: 0 entries",
            bg=self.bg, fg="#888888",
            font=("Segoe UI", 9)
        )
        self.label_memory.pack(side="right")

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

        right = tk.Frame(bottom, bg=self.bg)
        right.pack(side="right", fill="x", expand=True)

        if _HAS_SR:
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

    # Engine status poll
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
                    txt = f"Engine: {'ACTIVE' if active else 'INACTIVE'} | {model} | {device} | {'real' if real_model else 'sim'}"
                    self.label_status.config(text=txt, fg=self.fg if active else "#AAAAAA")
                else:
                    self.label_status.config(text=f"Engine: HTTP {r.status_code}", fg=self.error_color)
            except Exception:
                self.label_status.config(text="Engine: unreachable", fg=self.error_color)

        self.win.after(2000, self._update_engine_status)

    # Memory
    def _append_memory(self, role, text):
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role": role,
            "text": text
        }
        try:
            with open(MEMORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self._append_system(f"[Memory write error] {e}")
        self._load_memory_info()

    def _load_memory_info(self):
        count = 0
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    for _ in f:
                        count += 1
            except Exception:
                pass
        self.label_memory.config(text=f"Memory: {count} entries")

    def _clear_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                os.remove(MEMORY_FILE)
            except Exception as e:
                self._append_system(f"[Memory clear error] {e}")
        self._load_memory_info()
        self._append_system("Conversation memory cleared.")

    # Chat display
    def _append_chat(self, text, prefix=""):
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"{prefix}{text}\n\n")
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _append_system(self, text):
        self._append_chat(text, prefix="⚙ System: ")

    # Save log
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

    # Plugins
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
                "  /time     - show local time\n"
                "  /sysinfo  - show basic system info\n"
                "  /help     - this help\n"
            )
            self._append_system(help_text)
            self._append_memory("system", "/help used")
            return True

        return False

    # Send to engine
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
                self._append_chat(full_reply, prefix="🤖 Copilot: ")
                self._append_memory("assistant", reply)

                if self.tts_enabled.get() and self.tts_engine is not None:
                    self._speak(reply)
            else:
                self._append_chat(
                    f"[Error {r.status_code}] Engine returned an error.",
                    prefix="❌ Error: "
                )
        except Exception as e:
            self._append_chat(f"[Engine unreachable] {e}", prefix="❌ Error: ")

    # TTS
    def _speak(self, text: str):
        if not (self.tts_engine and _HAS_TTS):
            return
        try:
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            self._append_system(f"[TTS error] {e}")

    # Voice
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

