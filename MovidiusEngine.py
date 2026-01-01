#!/usr/bin/env python
# movidius_turbo_organ.py
# One-click Movidius/OpenVINO "turbocharger" organ with autoloader, auto-restart, and Tkinter GUI

import sys
import os
import subprocess
import threading
import time
import queue
import traceback

# ============================================================
# 1. Autoloader with optional self-restart after install
# ============================================================

REQUIRED_PACKAGES = [
    "numpy",
    "Pillow",
    "openvino"
]

AUTO_RESTART_FLAG = "--_ov_restarted_once"


def run_pip_install(args):
    try:
        print(f"[AUTOLOADER] pip install {' '.join(args)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + args)
        return True
    except Exception as e:
        print(f"[AUTOLOADER] pip install failed: {e}")
        return False


def ensure_package(pkg_name: str, extra_pip_args=None):
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        print(f"[AUTOLOADER] Installing missing package: {pkg_name} ...")
        extra_pip_args = extra_pip_args or [pkg_name]
        ok = run_pip_install(extra_pip_args)
        if not ok:
            return False
        try:
            __import__(pkg_name)
            print(f"[AUTOLOADER] Installed and imported: {pkg_name}")
            return True
        except ImportError as e:
            print(f"[AUTOLOADER] Still cannot import {pkg_name}: {e}")
            return False


def autoload_dependencies_and_maybe_restart():
    """
    Ensures numpy, Pillow, and openvino-dev[myriad] are installed.
    If openvino was just installed in this process and we have not
    restarted yet, restart the script once to let OpenVINO load cleanly.
    """
    imported_before = {}
    for pkg in ["numpy", "Pillow", "openvino"]:
        try:
            __import__(pkg)
            imported_before[pkg] = True
        except ImportError:
            imported_before[pkg] = False

    failed = []

    # numpy
    if not ensure_package("numpy", ["numpy"]):
        failed.append("numpy")

    # Pillow
    if not ensure_package("PIL", ["Pillow"]):
        failed.append("Pillow")

    # openvino-dev[myriad]
    try:
        __import__("openvino")
        ov_imported = True
    except ImportError:
        ov_imported = False

    if not ov_imported:
        # Try to install full dev with MYRIAD extras
        if not run_pip_install(["openvino-dev[myriad]"]):
            failed.append("openvino-dev[myriad]")
        else:
            try:
                __import__("openvino")
                ov_imported = True
            except ImportError as e:
                print(f"[AUTOLOADER] OpenVINO still not importable: {e}")
                failed.append("openvino-dev[myriad]")

    if failed:
        print("\n[AUTOLOADER] WARNING: Some packages could not be installed:")
        for f in failed:
            print(f"  - {f}")
        print("[AUTOLOADER] The GUI may still start, but acceleration may be limited.\n")

    # If OpenVINO was not available before but is now imported, and we haven't restarted yet:
    if not imported_before.get("openvino", False):
        try:
            __import__("openvino")
            just_installed_ov = True
        except ImportError:
            just_installed_ov = False
    else:
        just_installed_ov = False

    if just_installed_ov and AUTO_RESTART_FLAG not in sys.argv:
        print("[AUTOLOADER] OpenVINO was just installed. Restarting script once...")
        # Restart current process with the same arguments + flag
        new_argv = [sys.executable] + sys.argv
        if AUTO_RESTART_FLAG not in new_argv:
            new_argv.append(AUTO_RESTART_FLAG)
        os.execv(sys.executable, new_argv)


autoload_dependencies_and_maybe_restart()

# Safe to import now (or at least attempt)
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except Exception as e:
    print(f"[AUTOLOADER] OpenVINO import failed: {e}")
    Core = None
    OPENVINO_AVAILABLE = False

# ============================================================
# 2. Movidius/OpenVINO-backed engine with CPU fallback
# ============================================================

class MovidiusEngine:
    """
    OpenVINO-backed engine that prefers MYRIAD, falls back to CPU.
    Works even if OpenVINO or MYRIAD are unavailable (no-accel mode).
    """

    def __init__(self):
        self.core = None
        self.device_connected = False
        self.backend_device = None  # "MYRIAD", "CPU", or "NONE"
        self.available_devices = []

        # graph_id -> {
        #   "model_path": str,
        #   "compiled_model": CompiledModel,
        #   "input_name": str,
        #   "output_name": str
        # }
        self.loaded_graphs = {}

        self.total_inferences = 0
        self.latencies_ms = []
        self.queue_depth = 0
        self.lock = threading.Lock()
        self.running = True

        self.init_core()

    def init_core(self):
        if not OPENVINO_AVAILABLE:
            print("[Engine] OpenVINO not available. Running in 'NONE' mode.")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False
            return

        try:
            self.core = Core()
            self.available_devices = self.core.available_devices
            print(f"[Engine] OpenVINO available devices: {self.available_devices}")
        except Exception as e:
            print(f"[Engine] Failed to initialize OpenVINO Core: {e}")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False

    # ---- device & graph management --------------------------------------

    def connect_device(self):
        """
        Prefer MYRIAD; fallback to CPU; fallback to NONE.
        """
        with self.lock:
            if self.core is None:
                self.device_connected = False
                self.backend_device = "NONE"
                return False

            devices = self.available_devices
            chosen = None
            if "MYRIAD" in devices:
                chosen = "MYRIAD"
            elif "GPU" in devices:
                chosen = "GPU"
            elif "CPU" in devices:
                chosen = "CPU"

            if chosen is None:
                self.device_connected = False
                self.backend_device = "NONE"
                return False

            self.backend_device = chosen
            self.device_connected = True
            return True

    def disconnect_device(self):
        with self.lock:
            self.device_connected = False
            self.backend_device = "NONE"
            self.loaded_graphs.clear()

    def load_graph(self, graph_id: str, graph_path: str):
        """
        graph_path should be path to model.xml (IR).
        OpenVINO will expect model.bin next to it.
        """
        with self.lock:
            if not self.device_connected or self.core is None or self.backend_device == "NONE":
                raise RuntimeError("No acceleration device connected")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Model XML not found: {graph_path}")

        model = self.core.read_model(graph_path)
        compiled_model = self.core.compile_model(model=model, device_name=self.backend_device)

        input_layer = compiled_model.inputs[0]
        output_layer = compiled_model.outputs[0]
        input_name = input_layer.get_any_name()
        output_name = output_layer.get_any_name()

        with self.lock:
            self.loaded_graphs[graph_id] = {
                "model_path": graph_path,
                "compiled_model": compiled_model,
                "input_name": input_name,
                "output_name": output_name,
            }

    def unload_graph(self, graph_id: str):
        with self.lock:
            if graph_id in self.loaded_graphs:
                del self.loaded_graphs[graph_id]

    # ---- inference -------------------------------------------------------

    def run_inference(self, graph_id: str, input_array: np.ndarray):
        """
        input_array: numpy array NHWC [1, H, W, C] normalized.
        If your model expects NCHW, you can transpose here.
        """
        with self.lock:
            if not self.device_connected or self.core is None or self.backend_device == "NONE":
                raise RuntimeError("No acceleration device connected")
            if graph_id not in self.loaded_graphs:
                raise RuntimeError(f"Graph not loaded: {graph_id}")
            entry = self.loaded_graphs[graph_id]
            compiled_model = entry["compiled_model"]
            input_name = entry["input_name"]
            self.queue_depth += 1

        # Transpose if necessary. For now we assume the model is NHWC-capable.
        # If your model is NCHW, uncomment this:
        # input_array = np.transpose(input_array, (0, 3, 1, 2))

        start = time.time()
        infer_request = compiled_model.create_infer_request()
        infer_request.infer({input_name: input_array})
        output = infer_request.get_output_tensor().data
        latency_ms = (time.time() - start) * 1000.0

        with self.lock:
            self.queue_depth = max(0, self.queue_depth - 1)
            self.total_inferences += 1
            self.latencies_ms.append(latency_ms)
            self.latencies_ms = self.latencies_ms[-200:]

        output = np.array(output)
        return output, latency_ms

    # ---- metrics ---------------------------------------------------------

    def get_metrics(self):
        with self.lock:
            if self.latencies_ms:
                avg_latency = sum(self.latencies_ms) / len(self.latencies_ms)
                fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
            else:
                avg_latency = 0.0
                fps = 0.0
            metrics = {
                "device_connected": self.device_connected,
                "backend_device": self.backend_device,
                "available_devices": list(self.available_devices),
                "loaded_graphs": {k: v["model_path"] for k, v in self.loaded_graphs.items()},
                "avg_latency_ms": avg_latency,
                "fps": fps,
                "queue_depth": self.queue_depth,
                "total_inferences": self.total_inferences,
            }
        return metrics

    def stop(self):
        self.running = False


# ============================================================
# 3. Backend worker orchestrator (async commands, thread-safe logging)
# ============================================================

class EngineController:
    """
    Threaded controller that serializes operations to the engine
    and returns results via queues/callbacks.
    """

    def __init__(self, engine: MovidiusEngine, log_queue: queue.Queue, event_queue: queue.Queue):
        self.engine = engine
        self.cmd_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.log_queue = log_queue
        self.event_queue = event_queue
        self.thread.start()

    def _log(self, msg: str):
        try:
            self.log_queue.put(msg, block=False)
        except queue.Full:
            pass

    def _event(self, event_type: str, payload: dict):
        try:
            self.event_queue.put((event_type, payload), block=False)
        except queue.Full:
            pass

    def _worker_loop(self):
        self._log("[EngineController] Worker thread started.")
        while self.engine.running:
            try:
                cmd, args, res_queue = self.cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if cmd == "connect_device":
                    ok = self.engine.connect_device()
                    status = "connected" if ok else "failed"
                    self._event("device", {"status": status})
                    res_queue.put((ok, ok))

                elif cmd == "disconnect_device":
                    self.engine.disconnect_device()
                    self._event("device", {"status": "disconnected"})
                    res_queue.put((True, True))

                elif cmd == "load_graph":
                    graph_id, graph_path = args
                    self.engine.load_graph(graph_id, graph_path)
                    self._event("graph", {"graph_id": graph_id, "status": "loaded"})
                    res_queue.put((True, True))

                elif cmd == "unload_graph":
                    graph_id = args[0]
                    self.engine.unload_graph(graph_id)
                    self._event("graph", {"graph_id": graph_id, "status": "unloaded"})
                    res_queue.put((True, True))

                elif cmd == "run_inference":
                    graph_id, input_arr = args
                    output, latency = self.engine.run_inference(graph_id, input_arr)
                    res_queue.put((True, (output, latency)))

                elif cmd == "get_metrics":
                    metrics = self.engine.get_metrics()
                    res_queue.put((True, metrics))

                elif cmd == "stop":
                    self.engine.stop()
                    res_queue.put((True, True))
                    break

                else:
                    raise ValueError(f"Unknown command: {cmd}")

            except Exception as e:
                tb = traceback.format_exc()
                self._log(f"[EngineController] ERROR in cmd '{cmd}': {e}\n{tb}")
                res_queue.put((False, e))

        self._log("[EngineController] Worker thread stopped.")

    def _call(self, cmd, *args, timeout=10.0):
        res_queue = queue.Queue()
        self.cmd_queue.put((cmd, args, res_queue))
        try:
            ok, result = res_queue.get(timeout=timeout)
            return ok, result
        except queue.Empty:
            return False, TimeoutError(f"Timeout waiting for {cmd}")

    # Public API for GUI

    def connect_device(self):
        return self._call("connect_device")

    def disconnect_device(self):
        return self._call("disconnect_device")

    def load_graph(self, graph_id, graph_path):
        return self._call("load_graph", graph_id, graph_path)

    def unload_graph(self, graph_id):
        return self._call("unload_graph", graph_id)

    def run_inference(self, graph_id, input_arr):
        return self._call("run_inference", graph_id, input_arr)

    def get_metrics(self):
        return self._call("get_metrics")

    def stop(self):
        return self._call("stop")


# ============================================================
# 4. Tkinter advanced operator panel (thread-safe)
# ============================================================

class MovidiusGUI(tk.Tk):
    def __init__(self, controller: EngineController, log_queue: queue.Queue, event_queue: queue.Queue):
        super().__init__()
        self.title("Movidius Turbo Organ - Advanced Operator Panel")
        self.geometry("1100x700")
        self.minsize(900, 600)

        self.controller = controller
        self.log_queue = log_queue
        self.event_queue = event_queue
        self.mode = tk.StringVar(value="AUTO")  # AUTO / LEARN / MANUAL

        self._build_style()
        self._build_layout()
        self._start_polling()

    # ---- styling ---------------------------------------------------------

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=10)
        style.configure("Big.TLabel", font=("Segoe UI", 12))
        style.configure("StatusGood.TLabel", foreground="#1faa00", font=("Segoe UI", 11, "bold"))
        style.configure("StatusWarn.TLabel", foreground="#ffc107", font=("Segoe UI", 11, "bold"))
        style.configure("StatusBad.TLabel", foreground="#d32f2f", font=("Segoe UI", 11, "bold"))

    # ---- layout ----------------------------------------------------------

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_top_bar()
        self._build_main_area()
        self._build_bottom_bar()

    def _build_top_bar(self):
        frame = ttk.Frame(self, padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(2, weight=1)

        # Mode indicator
        self.mode_frame = tk.Frame(frame, width=40, height=40, bg="#1faa00")
        self.mode_frame.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        self.mode_frame.grid_propagate(False)

        ttk.Label(frame, text="Movidius Turbo Organ", style="Header.TLabel").grid(
            row=0, column=1, sticky="w"
        )

        self.device_status_var = tk.StringVar(value="Device: UNKNOWN")
        self.backend_status_var = tk.StringVar(value="Backend: NONE")

        self.device_status_label = ttk.Label(frame, textvariable=self.device_status_var, style="StatusBad.TLabel")
        self.device_status_label.grid(row=1, column=1, sticky="w", pady=(5, 0))

        self.backend_status_label = ttk.Label(frame, textvariable=self.backend_status_var, style="Big.TLabel")
        self.backend_status_label.grid(row=1, column=2, sticky="e", padx=(10, 0))

        # Mode buttons
        mode_btn_frame = ttk.Frame(frame)
        mode_btn_frame.grid(row=0, column=2, sticky="ne", pady=(0, 5))

        for text, mode, color in [
            ("AUTO", "AUTO", "#1faa00"),
            ("LEARN", "LEARN", "#ffc107"),
            ("MANUAL", "MANUAL", "#d32f2f"),
        ]:
            b = ttk.Button(mode_btn_frame, text=text, style="Big.TButton",
                           command=lambda m=mode: self._set_mode(m))
            b.pack(side="left", padx=5)

    def _build_main_area(self):
        main = ttk.Frame(self, padding=(10, 0, 10, 0))
        main.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)

        # Left side: notebook (tabs)
        notebook = ttk.Notebook(main)
        notebook.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Status tab
        self.status_tab = ttk.Frame(notebook)
        notebook.add(self.status_tab, text="Status")
        self._build_status_tab(self.status_tab)

        # Models tab
        self.models_tab = ttk.Frame(notebook)
        notebook.add(self.models_tab, text="Models")
        self._build_models_tab(self.models_tab)

        # Metrics tab
        self.metrics_tab = ttk.Frame(notebook)
        notebook.add(self.metrics_tab, text="Metrics")
        self._build_metrics_tab(self.metrics_tab)

        # Right side: logs
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Event & Error Log", style="Header.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )

        self.log_text = tk.Text(
            right,
            height=20,
            wrap="word",
            font=("Consolas", 10),
            bg="#111111",
            fg="#e0e0e0"
        )
        self.log_text.grid(row=1, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(right, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.log_text["yscrollcommand"] = log_scroll.set

    def _build_status_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Device status:", style="Big.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.status_device_value = ttk.Label(parent, text="UNKNOWN", style="StatusBad.TLabel")
        self.status_device_value.grid(row=0, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Backend device:", style="Big.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.status_backend_value = ttk.Label(parent, text="NONE", style="Big.TLabel")
        self.status_backend_value.grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Available devices:", style="Big.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        self.status_avail_value = ttk.Label(parent, text="N/A", style="Big.TLabel")
        self.status_avail_value.grid(row=2, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Loaded graphs:", style="Big.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        self.status_graphs_value = ttk.Label(parent, text="0", style="Big.TLabel")
        self.status_graphs_value.grid(row=3, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Total inferences:", style="Big.TLabel").grid(row=4, column=0, sticky="w", pady=5)
        self.status_inferences_value = ttk.Label(parent, text="0", style="Big.TLabel")
        self.status_inferences_value.grid(row=4, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Last latency (ms):", style="Big.TLabel").grid(row=5, column=0, sticky="w", pady=5)
        self.status_last_latency_value = ttk.Label(parent, text="N/A", style="Big.TLabel")
        self.status_last_latency_value.grid(row=5, column=1, sticky="w", pady=5)

        # Device control buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=(20, 0), sticky="w")

        ttk.Button(btn_frame, text="Connect Device", style="Big.TButton",
                   command=self.on_connect_device).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Disconnect Device", style="Big.TButton",
                   command=self.on_disconnect_device).pack(side="left", padx=5)

    def _build_models_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="Loaded Models", style="Big.TLabel").grid(row=0, column=0, sticky="w")

        self.models_list = tk.Listbox(parent, font=("Segoe UI", 11), height=8)
        self.models_list.grid(row=1, column=0, sticky="nsew", pady=5)
        models_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.models_list.yview)
        models_scroll.grid(row=1, column=1, sticky="ns")
        self.models_list["yscrollcommand"] = models_scroll.set

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="w")

        ttk.Button(btn_frame, text="Load Model...", style="Big.TButton",
                   command=self.on_load_model).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Unload Selected", style="Big.TButton",
                   command=self.on_unload_selected_model).pack(side="left", padx=5)

        # Test inference controls
        test_frame = ttk.LabelFrame(parent, text="Test Inference", padding=10)
        test_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(15, 0))
        test_frame.columnconfigure(1, weight=1)

        ttk.Label(test_frame, text="Input image:", style="Big.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.test_input_path_var = tk.StringVar(value="")
        ttk.Entry(test_frame, textvariable=self.test_input_path_var, width=50).grid(
            row=0, column=1, sticky="ew", pady=5
        )
        ttk.Button(test_frame, text="Browse...", command=self.on_browse_test_image).grid(
            row=0, column=2, padx=5, pady=5
        )

        ttk.Button(test_frame, text="Run Test Inference", style="Big.TButton",
                   command=self.on_run_test_inference).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        self.test_result_var = tk.StringVar(value="Result: N/A")
        ttk.Label(test_frame, textvariable=self.test_result_var, style="Big.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

    def _build_metrics_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Average latency (ms):", style="Big.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.metrics_latency_value = ttk.Label(parent, text="0.0", style="Big.TLabel")
        self.metrics_latency_value.grid(row=0, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Throughput (FPS):", style="Big.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.metrics_fps_value = ttk.Label(parent, text="0.0", style="Big.TLabel")
        self.metrics_fps_value.grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Queue depth:", style="Big.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        self.metrics_queue_value = ttk.Label(parent, text="0", style="Big.TLabel")
        self.metrics_queue_value.grid(row=2, column=1, sticky="w", pady=5)

        ttk.Label(parent, text="Total inferences:", style="Big.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        self.metrics_total_inf_value = ttk.Label(parent, text="0", style="Big.TLabel")
        self.metrics_total_inf_value.grid(row=3, column=1, sticky="w", pady=5)

    def _build_bottom_bar(self):
        bottom = ttk.Frame(self, padding=10)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        ttk.Button(bottom, text="Restart Engine", style="Big.TButton",
                   command=self.on_restart_engine).grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Button(bottom, text="Exit", style="Big.TButton",
                   command=self.on_exit).grid(row=0, column=1, sticky="e")

    # ---- mode handling ---------------------------------------------------

    def _set_mode(self, mode: str):
        self.mode.set(mode)
        if mode == "AUTO":
            self.mode_frame.configure(bg="#1faa00")
        elif mode == "LEARN":
            self.mode_frame.configure(bg="#ffc107")
        else:
            self.mode_frame.configure(bg="#d32f2f")
        self._log(f"[GUI] Mode changed to {mode}")

    # ---- logging & events ------------------------------------------------

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        text = f"[{ts}] {msg}\n"
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _handle_event(self, event_type: str, payload: dict):
        if event_type == "device":
            status = payload.get("status")
            self._log(f"Device event: {status}")
        elif event_type == "graph":
            graph_id = payload.get("graph_id")
            status = payload.get("status")
            self._log(f"Graph event: {graph_id} -> {status}")

    # ---- GUI callbacks ---------------------------------------------------

    def on_connect_device(self):
        ok, result = self.controller.connect_device()
        if not ok or not result:
            messagebox.showwarning("Device", "Failed to connect to any acceleration device.\nCheck MYRIAD/CPU/OpenVINO.")
            self._log("Device connect failed.")
        else:
            self._log("Device connected.")

    def on_disconnect_device(self):
        ok, result = self.controller.disconnect_device()
        if not ok:
            messagebox.showerror("Error", f"Failed to disconnect device: {result}")
        else:
            self._log("Device disconnected.")

    def on_load_model(self):
        path = filedialog.askopenfilename(
            title="Select OpenVINO IR model XML file",
            filetypes=[("OpenVINO IR XML", "*.xml"), ("All Files", "*.*")]
        )
        if not path:
            return
        graph_id = os.path.basename(path)
        ok, result = self.controller.load_graph(graph_id, path)
        if not ok:
            messagebox.showerror("Error", f"Failed to load model:\n{result}")
        else:
            self._log(f"Model loaded: {graph_id}")
            self._refresh_models_list()

    def on_unload_selected_model(self):
        sel = self.models_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "No model selected.")
            return
        graph_id = self.models_list.get(sel[0])
        ok, result = self.controller.unload_graph(graph_id)
        if not ok:
            messagebox.showerror("Error", f"Failed to unload model:\n{result}")
        else:
            self._log(f"Model unloaded: {graph_id}")
            self._refresh_models_list()

    def on_browse_test_image(self):
        path = filedialog.askopenfilename(
            title="Select test input image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All Files", "*.*")]
        )
        if not path:
            return
        self.test_input_path_var.set(path)

    def _prepare_input(self, image_path: str):
        """
        Preprocess image to [1, 224, 224, 3] float32 normalized.
        Adjust to match your model; if it expects NCHW,
        transpose is handled in engine if you uncomment it.
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # NHWC
        return arr

    def on_run_test_inference(self):
        sel = self.models_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "No model selected. Load and select a model first.")
            return
        graph_id = self.models_list.get(sel[0])

        img_path = self.test_input_path_var.get()
        if not img_path or not os.path.exists(img_path):
            messagebox.showinfo("Info", "Select a valid test input image.")
            return

        try:
            arr = self._prepare_input(img_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare input image:\n{e}")
            return

        self._log(f"Running test inference on model '{graph_id}' ...")
        self.test_result_var.set("Result: Running...")
        self.update_idletasks()

        ok, result = self.controller.run_inference(graph_id, arr)
        if not ok:
            messagebox.showerror("Error", f"Inference failed:\n{result}")
            self.test_result_var.set("Result: ERROR")
        else:
            output, latency = result
            try:
                top_idx = int(np.argmax(output))
            except Exception:
                top_idx = -1
            self.test_result_var.set(f"Result: class {top_idx}, latency {latency:.2f} ms")
            self.status_last_latency_value.config(text=f"{latency:.2f}")
            self._log(f"Inference done. class={top_idx}, latency={latency:.2f} ms")

    def on_restart_engine(self):
        self._log("Restarting engine (disconnect + connect) ...")
        self.on_disconnect_device()
        time.sleep(0.5)
        self.on_connect_device()

    def on_exit(self):
        self._log("Exiting...")
        self.controller.stop()
        self.destroy()

    # ---- periodic polling -----------------------------------------------

    def _start_polling(self):
        self._poll_metrics()
        self._poll_logs()
        self._poll_events()

    def _poll_metrics(self):
        ok, metrics = self.controller.get_metrics()
        if ok:
            self._update_metrics(metrics)
        self.after(500, self._poll_metrics)

    def _poll_logs(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._log(msg)
        self.after(200, self._poll_logs)

    def _poll_events(self):
        while True:
            try:
                event_type, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event_type, payload)
        self.after(200, self._poll_events)

    def _update_metrics(self, m: dict):
        device_connected = m.get("device_connected", False)
        backend_device = m.get("backend_device", "NONE")
        avail_devices = m.get("available_devices", [])
        loaded_graphs = m.get("loaded_graphs", {})
        avg_latency = m.get("avg_latency_ms", 0.0)
        fps = m.get("fps", 0.0)
        qd = m.get("queue_depth", 0)
        total_inf = m.get("total_inferences", 0)

        # Top bar
        if not OPENVINO_AVAILABLE:
            self.device_status_var.set("OpenVINO: NOT AVAILABLE")
            self.device_status_label.configure(style="StatusBad.TLabel")
        else:
            if device_connected and backend_device != "NONE":
                self.device_status_var.set("Device: CONNECTED")
                self.device_status_label.configure(style="StatusGood.TLabel")
            else:
                self.device_status_var.set("Device: DISCONNECTED")
                self.device_status_label.configure(style="StatusBad.TLabel")

        self.backend_status_var.set(f"Backend: {backend_device}")
        self.status_device_value.configure(
            text="CONNECTED" if device_connected else "DISCONNECTED",
            style="StatusGood.TLabel" if device_connected else "StatusBad.TLabel"
        )
        self.status_backend_value.configure(text=backend_device)
        self.status_avail_value.configure(text=", ".join(avail_devices) if avail_devices else "None")
        self.status_graphs_value.configure(text=str(len(loaded_graphs)))
        self.status_inferences_value.configure(text=str(total_inf))

        # Metrics tab
        self.metrics_latency_value.configure(text=f"{avg_latency:.2f}")
        self.metrics_fps_value.configure(text=f"{fps:.2f}")
        self.metrics_queue_value.configure(text=str(qd))
        self.metrics_total_inf_value.configure(text=str(total_inf))

        # Models list
        self._refresh_models_list(loaded_graphs)

    def _refresh_models_list(self, loaded_graphs=None):
        if loaded_graphs is None:
            ok, metrics = self.controller.get_metrics()
            if not ok:
                return
            loaded_graphs = metrics.get("loaded_graphs", {})

        current = list(self.models_list.get(0, "end"))
        new_list = list(loaded_graphs.keys())

        if current != new_list:
            self.models_list.delete(0, "end")
            for gid in new_list:
                self.models_list.insert("end", gid)


# ============================================================
# 5. Main entry point
# ============================================================

def main():
    engine = MovidiusEngine()

    log_q = queue.Queue(maxsize=1000)
    event_q = queue.Queue(maxsize=1000)

    controller = EngineController(engine, log_queue=log_q, event_queue=event_q)

    app = MovidiusGUI(controller, log_queue=log_q, event_queue=event_q)

    # Try initial auto-connect
    ok, result = controller.connect_device()
    if not ok or not result:
        print("[Startup] Auto-connect failed; GUI will show DISCONNECTED and CPU/NONE as appropriate.")

    app.mainloop()


if __name__ == "__main__":
    main()

