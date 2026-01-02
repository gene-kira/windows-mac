import os
import time
import threading
import json
import math
from collections import deque
from typing import Dict, Any, Optional, List, Deque

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    Core = None
    OPENVINO_AVAILABLE = False

# GUI
import tkinter as tk
from tkinter import ttk


# =========================
# 🧬 Adaptive Codex / Global
# =========================

if cp is not None:
    agent_weights = cp.array([0.6, -0.8, -0.3])  # can evolve later
else:
    agent_weights = [0.6, -0.8, -0.3]  # CPU fallback

mutation_log: List[Dict[str, Any]] = []


def log_mutation(event: str, details: Dict[str, Any]):
    mutation_entry = {
        "ts": time.time(),
        "event": event,
        "details": details,
    }
    mutation_log.append(mutation_entry)
    print(f"[Mutation] {event}: {details}")


# =======================================================
# 🧠 MovidiusEngine – Predictive, Baseline-aware, Resilient
# =======================================================

class MovidiusEngine:
    """
    OpenVINO-backed engine that prefers MYRIAD, falls back to CPU, and
    still runs in 'NONE' mode (no-accel) while keeping predictive + adaptive logic alive.

    Organs:
    - Predictive Drift Engine
    - Phantom Node Generator (ghost sync handling)
    - Baseline Memory Organ
    - Mutation Governor (self-rewriting purge/meta-logic)
    - Temporal Forecast Loop
    - Resilience Posture Manager
    - Swarm Codex Sync (filesystem-based)
    """

    def __init__(
        self,
        history_window: int = 64,
        drift_window: int = 32,
        ghost_sync_threshold: float = 0.7,
        baseline_alpha: float = 0.05,
        codex_path: Optional[str] = None,
        swarm_sync_dir: Optional[str] = None,
        swarm_node_id: str = "node_local",
    ):
        # Core / backend state
        self.core: Optional[Core] = None
        self.device_connected: bool = False
        self.backend_device: str = None  # "MYRIAD", "GPU", "CPU", or "NONE"
        self.available_devices: List[str] = []

        # Model graphs
        self.loaded_graphs: Dict[str, Dict[str, Any]] = {}

        # Inference metrics
        self.total_inferences: int = 0
        self.latencies_ms: List[float] = []
        self.queue_depth: int = 0

        # Predictive / temporal organs
        self.input_history: Deque[Any] = deque(maxlen=history_window)
        self.output_history: Deque[Any] = deque(maxlen=history_window)
        self.drift_window: int = drift_window

        # Baseline memory
        self.baseline_latency_ms: Optional[float] = None
        self.baseline_queue_depth: Optional[float] = None
        self.baseline_alpha: float = baseline_alpha  # EWMA smoothing

        # Ghost sync + phantom nodes
        self.ghost_sync_threshold: float = ghost_sync_threshold
        self.ghost_sync_score: float = 0.0
        self.phantom_node_count: int = 0
        self.telemetry_retention_seconds: float = 300.0  # default 5min

        # Mutation / codex
        self.codex_rules: Dict[str, Any] = {}
        self.codex_path = codex_path or "movidius_codex.json"

        # Swarm sync
        self.swarm_sync_dir = swarm_sync_dir
        self.swarm_node_id = swarm_node_id
        self.last_swarm_sync_ts: float = 0.0
        self.swarm_sync_interval: float = 15.0  # seconds

        # Resilience posture manager
        # posture: "OPTIMAL", "DEGRADED", "FALLBACK", "NONE"
        self.posture: str = "NONE"
        self.last_device_check_ts: float = 0.0
        self.device_check_interval: float = 10.0  # seconds

        # Concurrency
        self.lock = threading.Lock()
        self.running: bool = True

        # Load codex if present
        self._load_codex()

        # Init core + loops
        self.init_core()
        self._start_resilience_loop()
        self._start_temporal_loop()
        self._start_swarm_loop()

    # -------------------
    # Core init / devices
    # -------------------

    def init_core(self):
        if not OPENVINO_AVAILABLE:
            print("[Engine] OpenVINO not available. Running in 'NONE' mode.")
            self.core = None
            self.available_devices = []
            self.backend_device = "NONE"
            self.device_connected = False
            self.posture = "NONE"
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
            self.posture = "NONE"

    def connect_device(self) -> bool:
        """
        Prefer MYRIAD; fallback to GPU; fallback to CPU; fallback to NONE.
        """
        with self.lock:
            if self.core is None:
                self.device_connected = False
                self.backend_device = "NONE"
                self.posture = "NONE"
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
                self.posture = "NONE"
                return False

            self.backend_device = chosen
            self.device_connected = True
            self.posture = "OPTIMAL" if chosen in ("MYRIAD", "GPU") else "DEGRADED"
            print(f"[Engine] Connected to device: {self.backend_device} (posture={self.posture})")
            return True

    def disconnect_device(self):
        with self.lock:
            print("[Engine] Device disconnected by request or fault.")
            self.device_connected = False
            self.backend_device = "NONE"
            self.loaded_graphs.clear()
            self.posture = "NONE"

    # ------------------------
    # Graph / model management
    # ------------------------

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
            print(f"[Engine] Loaded graph '{graph_id}' on {self.backend_device}")

    def unload_graph(self, graph_id: str):
        with self.lock:
            if graph_id in self.loaded_graphs:
                del self.loaded_graphs[graph_id]
                print(f"[Engine] Unloaded graph '{graph_id}'")

    # -------------
    # Inference API
    # -------------

    def infer(self, graph_id: str, input_data: Any) -> Any:
        """
        Main inference call with:
        - latency tracking
        - queue depth tracking
        - predictive history
        - drift + ghost sync update
        - mutation governor hooks
        """
        with self.lock:
            self.queue_depth += 1

        t0 = time.time()
        try:
            if graph_id not in self.loaded_graphs or not self.device_connected:
                # No accel / no graph – still run predictive skeleton if possible
                output = self._dummy_infer(input_data)
            else:
                graph = self.loaded_graphs[graph_id]
                compiled_model = graph["compiled_model"]
                input_name = graph["input_name"]
                output_name = graph["output_name"]

                result = compiled_model({input_name: input_data})
                output = result[output_name]

            return output
        finally:
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0

            with self.lock:
                self.queue_depth = max(self.queue_depth - 1, 0)
                self.total_inferences += 1
                self.latencies_ms.append(latency_ms)

                self._update_baselines(latency_ms)
                self._record_history(input_data, output)
                self._update_ghost_sync_and_mutations()

    def _dummy_infer(self, input_data: Any) -> Any:
        """
        Fallback no-accel inference stub.
        """
        return {
            "echo": True,
            "ts": time.time(),
            "input_shape": getattr(input_data, "shape", None),
            "posture": self.posture,
        }

    # ==================================
    # Baseline Memory / Drift / GhostSync
    # ==================================

    def _update_baselines(self, latency_ms: float):
        if self.baseline_latency_ms is None:
            self.baseline_latency_ms = latency_ms
        else:
            self.baseline_latency_ms = (
                (1 - self.baseline_alpha) * self.baseline_latency_ms
                + self.baseline_alpha * latency_ms
            )

        if self.baseline_queue_depth is None:
            self.baseline_queue_depth = float(self.queue_depth)
        else:
            self.baseline_queue_depth = (
                (1 - self.baseline_alpha) * self.baseline_queue_depth
                + self.baseline_alpha * float(self.queue_depth)
            )

    def _record_history(self, input_data: Any, output_data: Any):
        self.input_history.append(self._to_small_vector(input_data))
        self.output_history.append(self._to_small_vector(output_data))

    def _to_small_vector(self, data: Any, max_len: int = 16) -> List[float]:
        if data is None:
            return [0.0]

        if np is not None and isinstance(data, np.ndarray):
            flat = data.flatten()
            if flat.size == 0:
                return [0.0]
            sample = flat[:max_len]
            return [float(x) for x in sample]

        if isinstance(data, (int, float)):
            return [float(data)]
        if isinstance(data, (list, tuple)) and data:
            try:
                return [float(x) for x in data[:max_len]]
            except Exception:
                pass

        return [float(len(str(data)))]

    def _compute_drift_score(self) -> float:
        n = min(len(self.output_history), self.drift_window)
        if n < 2:
            return 0.0

        recent = list(self.output_history)[-n:]
        diffs = []
        for i in range(1, n):
            v_prev = recent[i - 1]
            v_curr = recent[i]
            length = min(len(v_prev), len(v_curr))
            if length == 0:
                continue
            delta = sum(abs(v_curr[j] - v_prev[j]) for j in range(length)) / float(length)
            diffs.append(delta)

        if not diffs:
            return 0.0

        mean_diff = sum(diffs) / len(diffs)
        return math.tanh(mean_diff / 10.0)

    def _update_ghost_sync_and_mutations(self):
        drift_score = self._compute_drift_score()

        posture_factor = {
            "OPTIMAL": 0.2,
            "DEGRADED": 0.5,
            "FALLBACK": 0.7,
            "NONE": 0.9,
        }.get(self.posture, 0.5)

        queue_stress = 0.0
        if self.baseline_queue_depth is not None and self.baseline_queue_depth > 0:
            queue_stress = min(
                2.0 * (float(self.queue_depth) / (self.baseline_queue_depth + 1e-6)),
                4.0,
            )

        raw_score = drift_score + posture_factor + 0.1 * queue_stress
        self.ghost_sync_score = math.tanh(raw_score / 2.5)

        if self.ghost_sync_score >= self.ghost_sync_threshold:
            old_ret = self.telemetry_retention_seconds
            self.telemetry_retention_seconds = max(30.0, self.telemetry_retention_seconds * 0.8)
            self.phantom_node_count += 1

            log_mutation(
                "ghost_sync_detected",
                {
                    "ghost_sync_score": self.ghost_sync_score,
                    "drift_score": drift_score,
                    "posture": self.posture,
                    "queue_stress": queue_stress,
                    "old_retention": old_ret,
                    "new_retention": self.telemetry_retention_seconds,
                    "phantom_nodes": self.phantom_node_count,
                },
            )

            self._mutate_codex_for_ghost_sync()

    # ================
    # Phantom codex DNA
    # ================

    def _mutate_codex_for_ghost_sync(self):
        ghost_entry = {
            "timestamp": time.time(),
            "ghost_sync_score": self.ghost_sync_score,
            "telemetry_retention_seconds": self.telemetry_retention_seconds,
            "phantom_node_count": self.phantom_node_count,
            "posture": self.posture,
            "node_id": self.swarm_node_id,
        }

        ghosts = self.codex_rules.setdefault("ghost_sync_events", [])
        ghosts.append(ghost_entry)

        if self.ghost_sync_score > 0.9:
            self.codex_rules["purge_mode"] = "aggressive"
        elif self.ghost_sync_score > 0.75:
            self.codex_rules["purge_mode"] = "adaptive"
        else:
            self.codex_rules.setdefault("purge_mode", "baseline")

        self._save_codex()
        log_mutation("codex_ghost_mutation", {"purge_mode": self.codex_rules.get("purge_mode")})

    def _load_codex(self):
        if not self.codex_path or not os.path.exists(self.codex_path):
            self.codex_rules = {}
            return
        try:
            with open(self.codex_path, "r", encoding="utf-8") as f:
                self.codex_rules = json.load(f)
            print(f"[Engine] Loaded codex from {self.codex_path}")
        except Exception as e:
            print(f"[Engine] Failed to load codex: {e}")
            self.codex_rules = {}

    def _save_codex(self):
        if not self.codex_path:
            return
        try:
            with open(self.codex_path, "w", encoding="utf-8") as f:
                json.dump(self.codex_rules, f, indent=2)
        except Exception as e:
            print(f"[Engine] Failed to save codex: {e}")

    # ===========================
    # Swarm codex sync (filesystem)
    # ===========================

    def _start_swarm_loop(self):
        t = threading.Thread(target=self._swarm_loop, daemon=True)
        t.start()

    def _swarm_loop(self):
        while self.running:
            time.sleep(3.0)
            if not self.swarm_sync_dir:
                continue
            try:
                now = time.time()
                if now - self.last_swarm_sync_ts < self.swarm_sync_interval:
                    continue
                self.last_swarm_sync_ts = now
                self._swarm_sync_push_pull()
            except Exception as e:
                print(f"[Engine] Swarm loop error: {e}")

    def _swarm_sync_push_pull(self):
        os.makedirs(self.swarm_sync_dir, exist_ok=True)

        # Push local codex snapshot
        local_snapshot = {
            "node_id": self.swarm_node_id,
            "timestamp": time.time(),
            "codex_rules": self.codex_rules,
        }
        local_path = os.path.join(self.swarm_sync_dir, f"{self.swarm_node_id}_codex.json")
        try:
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(local_snapshot, f, indent=2)
        except Exception as e:
            print(f"[Engine] Failed to push swarm codex: {e}")

        # Pull others and merge ghost events + adopt strongest purge_mode
        try:
            ghost_events: List[Dict[str, Any]] = self.codex_rules.get("ghost_sync_events", [])
            purge_mode = self.codex_rules.get("purge_mode", "baseline")

            for fname in os.listdir(self.swarm_sync_dir):
                if not fname.endswith("_codex.json"):
                    continue
                full = os.path.join(self.swarm_sync_dir, fname)
                if full == local_path:
                    continue
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        remote = json.load(f)
                except Exception:
                    continue

                rcodex = remote.get("codex_rules", {})
                rghosts = rcodex.get("ghost_sync_events", [])
                ghost_events.extend(rghosts)

                rmode = rcodex.get("purge_mode", "baseline")
                # Simple priority: aggressive > adaptive > baseline
                mode_rank = {"baseline": 0, "adaptive": 1, "aggressive": 2}
                if mode_rank.get(rmode, 0) > mode_rank.get(purge_mode, 0):
                    purge_mode = rmode

            self.codex_rules["ghost_sync_events"] = ghost_events
            self.codex_rules["purge_mode"] = purge_mode
        except Exception as e:
            print(f"[Engine] Failed to pull/merge swarm codex: {e}")

    # =========================================
    # Temporal Forecast Loop (Predictive Engine)
    # =========================================

    def _start_temporal_loop(self):
        t = threading.Thread(target=self._temporal_loop, daemon=True)
        t.start()

    def _temporal_loop(self):
        while self.running:
            time.sleep(2.0)
            try:
                _ = self.forecast_next_output_vector()
            except Exception as e:
                print(f"[Engine] Temporal loop error: {e}")

    def forecast_next_output_vector(self) -> Optional[List[float]]:
        n = min(len(self.output_history), self.drift_window)
        if n < 3:
            return None

        recent = list(self.output_history)[-n:]
        deltas = []
        for i in range(1, n):
            v_prev = recent[i - 1]
            v_curr = recent[i]
            length = min(len(v_prev), len(v_curr))
            if length == 0:
                continue
            deltas.append([v_curr[j] - v_prev[j] for j in range(length)])

        if not deltas:
            return None

        length = min(len(d) for d in deltas)
        avg_delta = []
        for j in range(length):
            avg_delta.append(sum(d[j] for d in deltas) / len(deltas))

        last = recent[-1]
        length = min(len(last), len(avg_delta))
        forecast = [last[j] + avg_delta[j] for j in range(length)]
        return forecast

    # ============================
    # Resilience Posture Management
    # ============================

    def _start_resilience_loop(self):
        t = threading.Thread(target=self._resilience_loop, daemon=True)
        t.start()

    def _resilience_loop(self):
        while self.running:
            time.sleep(3.0)
            try:
                now = time.time()
                if now - self.last_device_check_ts < self.device_check_interval:
                    continue
                self.last_device_check_ts = now

                if not self.device_connected:
                    connected = self.connect_device()
                    if not connected:
                        if self.posture != "NONE":
                            print("[Engine] No device found, entering NONE posture.")
                        self.posture = "NONE"
                    else:
                        print("[Engine] Device reconnected in resilience loop.")
                else:
                    self._update_posture_from_metrics()
            except Exception as e:
                print(f"[Engine] Resilience loop error: {e}")

    def _update_posture_from_metrics(self):
        if self.baseline_latency_ms is None:
            return

        latency = self.latencies_ms[-1] if self.latencies_ms else self.baseline_latency_ms
        latency_ratio = latency / (self.baseline_latency_ms + 1e-6)

        queue_ratio = 1.0
        if self.baseline_queue_depth is not None and self.baseline_queue_depth > 0:
            queue_ratio = float(self.queue_depth) / (self.baseline_queue_depth + 1e-6)

        stress = 0.5 * latency_ratio + 0.5 * queue_ratio

        old_posture = self.posture
        if stress < 1.5:
            if self.backend_device in ("MYRIAD", "GPU"):
                self.posture = "OPTIMAL"
            elif self.backend_device == "CPU":
                self.posture = "DEGRADED"
            else:
                self.posture = "NONE"
        elif stress < 3.0:
            self.posture = "DEGRADED"
        else:
            self.posture = "FALLBACK"

        if old_posture != self.posture:
            log_mutation(
                "posture_change",
                {
                    "old_posture": old_posture,
                    "new_posture": self.posture,
                    "latency_ms": latency,
                    "baseline_latency_ms": self.baseline_latency_ms,
                    "queue_depth": self.queue_depth,
                    "baseline_queue_depth": self.baseline_queue_depth,
                    "stress": stress,
                },
            )

    # -------------
    # Public status
    # -------------

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "backend_device": self.backend_device,
                "device_connected": self.device_connected,
                "posture": self.posture,
                "total_inferences": self.total_inferences,
                "last_latency_ms": self.latencies_ms[-1] if self.latencies_ms else None,
                "baseline_latency_ms": self.baseline_latency_ms,
                "queue_depth": self.queue_depth,
                "baseline_queue_depth": self.baseline_queue_depth,
                "ghost_sync_score": self.ghost_sync_score,
                "telemetry_retention_seconds": self.telemetry_retention_seconds,
                "phantom_node_count": self.phantom_node_count,
                "loaded_graphs": list(self.loaded_graphs.keys()),
                "purge_mode": self.codex_rules.get("purge_mode"),
                "swarm_node_id": self.swarm_node_id,
            }

    def stop(self):
        self.running = False


# ===========================
# Tkinter GUI – Live Organ HUD
# ===========================

class MovidiusHUD:
    def __init__(self, engine: MovidiusEngine):
        self.engine = engine

        self.root = tk.Tk()
        self.root.title("Movidius Organ – Predictive HUD")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        # Slight transparency
        try:
            self.root.attributes("-alpha", 0.93)
        except Exception:
            pass

        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self):
        pad = 4
        main = ttk.Frame(self.root, padding=pad)
        main.grid(row=0, column=0, sticky="nsew")

        # Row 0: Device + Posture
        self.lbl_device = ttk.Label(main, text="Device: ?", width=24)
        self.lbl_device.grid(row=0, column=0, sticky="w")

        self.lbl_posture = ttk.Label(main, text="Posture: ?", width=18)
        self.lbl_posture.grid(row=0, column=1, sticky="w")

        # Row 1: Latency
        self.lbl_latency = ttk.Label(main, text="Latency: ? ms", width=24)
        self.lbl_latency.grid(row=1, column=0, sticky="w")

        self.lbl_baseline_latency = ttk.Label(main, text="Baseline: ? ms", width=18)
        self.lbl_baseline_latency.grid(row=1, column=1, sticky="w")

        # Row 2: Queue
        self.lbl_queue = ttk.Label(main, text="Queue: ?", width=24)
        self.lbl_queue.grid(row=2, column=0, sticky="w")

        self.lbl_baseline_queue = ttk.Label(main, text="Baseline Q: ?", width=18)
        self.lbl_baseline_queue.grid(row=2, column=1, sticky="w")

        # Row 3: Ghost sync + phantom
        self.lbl_ghost = ttk.Label(main, text="Ghost: ?", width=24)
        self.lbl_ghost.grid(row=3, column=0, sticky="w")

        self.lbl_phantoms = ttk.Label(main, text="Phantoms: 0", width=18)
        self.lbl_phantoms.grid(row=3, column=1, sticky="w")

        # Row 4: Telemetry horizon + purge mode
        self.lbl_retention = ttk.Label(main, text="Telemetry: ?s", width=24)
        self.lbl_retention.grid(row=4, column=0, sticky="w")

        self.lbl_purge = ttk.Label(main, text="Purge: ?", width=18)
        self.lbl_purge.grid(row=4, column=1, sticky="w")

        # Row 5: Swarm
        self.lbl_swarm = ttk.Label(main, text="Swarm node: ?", width=24)
        self.lbl_swarm.grid(row=5, column=0, sticky="w")

        self.lbl_graphs = ttk.Label(main, text="Graphs: []", width=18)
        self.lbl_graphs.grid(row=5, column=1, sticky="w")

        # Row 6: Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=6, column=0, columnspan=2, sticky="ew")

        self.btn_refresh = ttk.Button(btn_frame, text="Refresh", command=self._refresh_status)
        self.btn_refresh.grid(row=0, column=0, padx=(0, 4))

        self.btn_exit = ttk.Button(btn_frame, text="Exit", command=self._on_exit)
        self.btn_exit.grid(row=0, column=1, padx=(4, 0))

    def _color_for_posture(self, posture: str) -> str:
        posture = posture or "NONE"
        posture = posture.upper()
        if posture == "OPTIMAL":
            return "#00aa00"
        if posture == "DEGRADED":
            return "#ffaa00"
        if posture == "FALLBACK":
            return "#ff5500"
        return "#777777"

    def _color_for_ghost(self, score: float) -> str:
        if score < 0.4:
            return "#00aa00"
        if score < 0.7:
            return "#ffaa00"
        return "#ff0000"

    def _refresh_status(self):
        status = self.engine.get_status()

        device = status["backend_device"] or "NONE"
        posture = status["posture"] or "NONE"
        last_lat = status["last_latency_ms"]
        base_lat = status["baseline_latency_ms"]
        q = status["queue_depth"]
        qb = status["baseline_queue_depth"]
        ghost = status["ghost_sync_score"]
        telem = status["telemetry_retention_seconds"]
        phantoms = status["phantom_node_count"]
        purge = status.get("purge_mode") or "baseline"
        swarm_node = status.get("swarm_node_id") or "?"

        graphs = status["loaded_graphs"]

        self.lbl_device.config(text=f"Device: {device}")
        self.lbl_posture.config(text=f"Posture: {posture}")
        self.lbl_posture.config(foreground=self._color_for_posture(posture))

        if last_lat is not None:
            self.lbl_latency.config(text=f"Latency: {last_lat:.1f} ms")
        else:
            self.lbl_latency.config(text="Latency: ? ms")

        if base_lat is not None:
            self.lbl_baseline_latency.config(text=f"Baseline: {base_lat:.1f} ms")
        else:
            self.lbl_baseline_latency.config(text="Baseline: ? ms")

        self.lbl_queue.config(text=f"Queue: {q}")
        if qb is not None:
            self.lbl_baseline_queue.config(text=f"Baseline Q: {qb:.1f}")
        else:
            self.lbl_baseline_queue.config(text="Baseline Q: ?")

        self.lbl_ghost.config(text=f"Ghost: {ghost:.2f}")
        self.lbl_ghost.config(foreground=self._color_for_ghost(ghost))

        self.lbl_phantoms.config(text=f"Phantoms: {phantoms}")
        self.lbl_retention.config(text=f"Telemetry: {telem:.0f}s")

        self.lbl_purge.config(text=f"Purge: {purge}")
        self.lbl_swarm.config(text=f"Swarm node: {swarm_node}")
        self.lbl_graphs.config(text=f"Graphs: {graphs}")

    def _schedule_refresh(self):
        self._refresh_status()
        self.root.after(1000, self._schedule_refresh)

    def _on_exit(self):
        self.engine.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# =========
# Demo main
# =========

if __name__ == "__main__":
    # Example: swarm directory in current folder, local node id
    engine = MovidiusEngine(
        codex_path="movidius_codex.json",
        swarm_sync_dir="movidius_swarm",
        swarm_node_id="node_A",
    )

    # Try connecting once at startup
    engine.connect_device()

    # Fire up HUD
    hud = MovidiusHUD(engine)
    hud.run()

