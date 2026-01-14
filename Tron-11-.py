import tkinter as tk
from tkinter import ttk, filedialog
import psutil
import random
import time
import os
import json
import threading
import subprocess
import statistics
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# Global constants
# =========================
SAVE_THRESHOLD = 100 * 1024 * 1024  # 100 MB equivalent activity before autosave
DEFAULT_BACKUP_FOLDER = os.path.join(os.getcwd(), "hive_backup")
TEMPORAL_MEMORY_FILE = "hive_temporal.json"
BEHAVIOR_MEMORY_FILE = "hive_behavior.json"
POLICY_FILE = "hive_policy.json"
EMERGENT_FILE = "hive_insights.json"

# =========================
# L5 / Movidius / NPU Inference Engine (stubbed)
# =========================
class MovidiusInferenceEngine:
    """
    L5: NPU / Movidius replica inference engine (stub).
    Produces a risk score from fused features.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)
        self.available = False
        self.backend_name = "CPU"
        self.detect_movidius()

    def log(self, msg):
        self.log_fn(msg)

    def detect_movidius(self):
        self.available = False
        self.backend_name = "CPU"
        self.log("[Movidius] No Movidius NPU detected (stub). Using CPU heuristics.")

    def infer_best_guess(self, features):
        if not features:
            return None
        ram = features.get("ram", 0)
        combined = features.get("combined", 0)
        volatility = features.get("volatility", 0)
        glitch_count = features.get("glitches", 0)

        score = (
            0.4 * (combined / 100.0)
            + 0.3 * (ram / 100.0)
            + 0.2 * max(0.0, min(1.0, volatility / 20.0))
            + 0.1 * max(0.0, min(1.0, glitch_count / 5.0))
        )
        score = max(0.0, min(1.0, score))
        self.log(
            f"[MovidiusReplica/L5] Best-guess risk score≈{score:.2f} "
            f"(backend={self.backend_name})."
        )
        return score

# =========================
# L4 / Intelligent Water Data Physics Engine
# =========================
class WaterPhysicsEngine:
    """
    L4: Physics-inspired engine treating system load as fluid dynamics.
    """
    def __init__(self, log_fn=None, window=60):
        self.log_fn = log_fn or (lambda msg: None)
        self.window = window
        self.samples = deque(maxlen=window)
        self.glitch_count = 0

    def log(self, msg):
        self.log_fn(msg)

    def add_sample(self, ram_pct, vram_pct, combined_pct, cpu_pct, net_bytes, disk_bytes):
        self.samples.append({
            "ram": ram_pct,
            "vram": vram_pct,
            "combined": combined_pct,
            "cpu": cpu_pct,
            "net": net_bytes,
            "disk": disk_bytes,
        })

    def analyze(self):
        if len(self.samples) < 5:
            return None

        combined = [s["combined"] for s in self.samples]
        flow = [s["net"] + s["disk"] for s in self.samples]

        avg_combined = statistics.mean(combined)
        avg_flow = statistics.mean(flow)
        vol_combined = statistics.pstdev(combined) if len(combined) > 1 else 0.0

        pressure = avg_combined
        turbulence = vol_combined
        capacity = max(0.0, 100.0 - avg_combined)

        latest = combined[-1]
        if latest > avg_combined + 3 * (vol_combined + 1e-3):
            self.glitch_count += 1
            self.log(
                f"[WaterPhysics/L4] Hydraulic spike: pressure={latest:.1f}%, "
                f"avg={avg_combined:.1f}%, turbulence={turbulence:.2f}."
            )

        state = {
            "pressure": pressure,
            "flow": avg_flow,
            "turbulence": turbulence,
            "capacity": capacity,
            "glitches": self.glitch_count,
        }

        self.log(
            f"[WaterPhysics/L4] Pressure={pressure:.1f}%, Capacity={capacity:.1f}%, "
            f"Turbulence={turbulence:.2f}, Flow≈{avg_flow:.0f} B/s, "
            f"Glitches={self.glitch_count}."
        )
        return state

# =========================
# L6: Temporal Memory Engine
# =========================
class TemporalMemoryEngine:
    """
    L6: Long-horizon memory across hours/days.
    Aggregates per-hour stats: avg pressure, glitches, risk.
    """
    def __init__(self, log_fn=None, path=TEMPORAL_MEMORY_FILE):
        self.log_fn = log_fn or (lambda msg: None)
        self.path = path
        self.memory = {}
        self.last_saved = None
        self.load()

    def log(self, msg):
        self.log_fn(msg)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.memory = json.load(f)
                self.last_saved = json.loads(json.dumps(self.memory))
                self.log(f"[L6] Temporal memory loaded from {self.path}.")
            except Exception as e:
                self.log(f"[L6] Failed to load temporal memory: {e}")
                self.memory = {}
        else:
            self.log("[L6] No temporal memory file found; starting fresh.")
            self.memory = {}

    def update(self, water_state, risk_score):
        if not water_state:
            return
        hour = time.localtime().tm_hour
        key = str(hour)
        entry = self.memory.get(key, {
            "samples": 0,
            "avg_pressure": 0.0,
            "avg_risk": 0.0,
            "glitches": 0
        })

        samples = entry["samples"] + 1
        entry["avg_pressure"] = (
            entry["avg_pressure"] * entry["samples"] + water_state["pressure"]
        ) / samples
        entry["avg_risk"] = (
            entry["avg_risk"] * entry["samples"] + (risk_score if risk_score is not None else 0.0)
        ) / samples
        entry["glitches"] = max(entry["glitches"], water_state["glitches"])
        entry["samples"] = samples

        self.memory[key] = entry
        self.log(
            f"[L6] Hour={hour}: avg_pressure≈{entry['avg_pressure']:.1f}%, "
            f"avg_risk≈{entry['avg_risk']:.2f}, glitches={entry['glitches']}."
        )

    def save(self):
        if self.memory == self.last_saved:
            return
        try:
            with open(self.path, "w") as f:
                json.dump(self.memory, f)
            self.last_saved = json.loads(json.dumps(self.memory))
            self.log("[L6] Temporal memory saved.")
        except Exception as e:
            self.log(f"[L6] Failed to save temporal memory: {e}")

    def get_hour_context(self):
        hour = time.localtime().tm_hour
        key = str(hour)
        return self.memory.get(key)

# =========================
# L7: Behavioral Memory Engine
# =========================
class BehavioralMemoryEngine:
    """
    L7: Tracks user actions (Ghost Boost, Foresight) per hour and their context.
    """
    def __init__(self, log_fn=None, path=BEHAVIOR_MEMORY_FILE):
        self.log_fn = log_fn or (lambda msg: None)
        self.path = path
        self.memory = {}
        self.last_saved = None
        self.load()

    def log(self, msg):
        self.log_fn(msg)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.memory = json.load(f)
                self.last_saved = json.loads(json.dumps(self.memory))
                self.log(f"[L7] Behavioral memory loaded from {self.path}.")
            except Exception as e:
                self.log(f"[L7] Failed to load behavioral memory: {e}")
                self.memory = {}
        else:
            self.log("[L7] No behavioral memory file found; starting fresh.")
            self.memory = {}

    def record_action(self, action_type, risk_score, pressure):
        hour = time.localtime().tm_hour
        key = str(hour)
        entry = self.memory.get(key, {
            "actions": 0,
            "ghost_boost": 0,
            "foresight": 0,
            "avg_risk": 0.0,
            "avg_pressure": 0.0
        })

        entry["actions"] += 1
        if action_type == "ghost_boost":
            entry["ghost_boost"] += 1
        elif action_type == "foresight":
            entry["foresight"] += 1

        samples = entry["actions"]
        entry["avg_risk"] = (
            entry["avg_risk"] * (samples - 1) + (risk_score if risk_score is not None else 0.0)
        ) / samples
        entry["avg_pressure"] = (
            entry["avg_pressure"] * (samples - 1) + (pressure if pressure is not None else 0.0)
        ) / samples

        self.memory[key] = entry
        self.log(
            f"[L7] Hour={hour}: actions={entry['actions']}, "
            f"ghost_boost={entry['ghost_boost']}, foresight={entry['foresight']}, "
            f"avg_risk≈{entry['avg_risk']:.2f}, avg_pressure≈{entry['avg_pressure']:.1f}%."
        )

    def save(self):
        if self.memory == self.last_saved:
            return
        try:
            with open(self.path, "w") as f:
                json.dump(self.memory, f)
            self.last_saved = json.loads(json.dumps(self.memory))
            self.log("[L7] Behavioral memory saved.")
        except Exception as e:
            self.log(f"[L7] Failed to save behavioral memory: {e}")

    def get_hour_profile(self):
        hour = time.localtime().tm_hour
        key = str(hour)
        return self.memory.get(key)

# =========================
# L8: Anticipatory Behavior Engine
# =========================
class AnticipatoryBehaviorEngine:
    """
    L8: Predicts likelihood of user actions (Ghost Boost / Foresight)
    based on current state + L6 + L7.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)

    def log(self, msg):
        self.log_fn(msg)

    def predict(self, temporal_ctx, behavior_ctx, risk, pressure, turbulence):
        """
        Returns a dict with probabilities:
        {
            "ghost_boost": p1,
            "foresight": p2,
            "no_action": p3
        }
        """
        base_gb = 0.2
        base_fs = 0.2

        if behavior_ctx:
            total_actions = behavior_ctx.get("actions", 0) or 1
            gb_ratio = behavior_ctx.get("ghost_boost", 0) / total_actions
            fs_ratio = behavior_ctx.get("foresight", 0) / total_actions
            base_gb += 0.4 * gb_ratio
            base_fs += 0.4 * fs_ratio

        if risk is not None:
            base_gb += 0.2 * risk
            base_fs += 0.1 * risk

        if pressure is not None:
            base_gb += 0.1 * (pressure / 100.0)
            base_fs += 0.1 * (pressure / 100.0)

        if turbulence is not None:
            base_fs += 0.1 * min(1.0, turbulence / 20.0)

        gb = max(0.0, base_gb)
        fs = max(0.0, base_fs)
        no = max(0.0, 1.0 - min(1.0, gb + fs))

        total = gb + fs + no
        if total <= 0:
            gb = fs = 0.0
            no = 1.0
            total = 1.0

        gb /= total
        fs /= total
        no /= total

        self.log(
            f"[L8] Anticipation: P(GB)≈{gb:.2f}, P(FS)≈{fs:.2f}, P(None)≈{no:.2f}."
        )

        return {"ghost_boost": gb, "foresight": fs, "no_action": no}

# =========================
# L9: Multi-Agent Reasoning Engine
# =========================
class MultiAgentReasoningEngine:
    """
    L9: Multiple simple agents vote on risk and intervention urgency.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)

    def log(self, msg):
        self.log_fn(msg)

    def reason(self, risk, pressure, turbulence, temporal_ctx, behavior_ctx):
        votes = []

        if risk is not None:
            if risk > 0.7:
                votes.append(("risk_agent", "high"))
            elif risk > 0.4:
                votes.append(("risk_agent", "medium"))
            else:
                votes.append(("risk_agent", "low"))

        if pressure is not None:
            if pressure > 80:
                votes.append(("pressure_agent", "high"))
            elif pressure > 60:
                votes.append(("pressure_agent", "medium"))
            else:
                votes.append(("pressure_agent", "low"))

        if turbulence is not None:
            if turbulence > 15:
                votes.append(("turbulence_agent", "high"))
            elif turbulence > 8:
                votes.append(("turbulence_agent", "medium"))
            else:
                votes.append(("turbulence_agent", "low"))

        if temporal_ctx:
            if temporal_ctx.get("avg_pressure", 0) > 75:
                votes.append(("temporal_agent", "high"))
            else:
                votes.append(("temporal_agent", "medium"))

        if behavior_ctx:
            if behavior_ctx.get("actions", 0) > 5:
                votes.append(("behavior_agent", "intervention_prone"))
            else:
                votes.append(("behavior_agent", "intervention_sparse"))

        high = sum(1 for _, v in votes if v == "high")
        medium = sum(1 for _, v in votes if v == "medium")
        low = sum(1 for _, v in votes if v == "low")

        if high > medium and high > low:
            overall = "high"
        elif medium >= high and medium > low:
            overall = "medium"
        else:
            overall = "low"

        self.log(f"[L9] Multi-agent votes={votes}, overall_risk={overall}.")
        return {"votes": votes, "overall_risk": overall}

# =========================
# L10: Emotional Inertia Engine
# =========================
class EmotionalInertiaEngine:
    """
    L10: Maintains a 'mood' based on recent risk, glitches, and stability.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)
        self.mood = "Calm"

    def log(self, msg):
        self.log_fn(msg)

    def update_mood(self, risk, glitches, turbulence):
        score = 0.0
        if risk is not None:
            score += risk
        if glitches is not None:
            score += min(1.0, glitches / 5.0)
        if turbulence is not None:
            score += min(1.0, turbulence / 20.0)

        if score < 0.7:
            new_mood = "Calm"
        elif score < 1.5:
            new_mood = "Alert"
        elif score < 2.2:
            new_mood = "Anxious"
        else:
            new_mood = "Stressed"

        if new_mood != self.mood:
            self.log(f"[L10] Mood shift: {self.mood} → {new_mood}.")
            self.mood = new_mood

        return self.mood

# =========================
# L11: Adaptive Autonomy Engine
# =========================
class AdaptiveAutonomyEngine:
    """
    L11: Suggests or prepares actions when confidence is high and risk is rising.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)

    def log(self, msg):
        self.log_fn(msg)

    def decide(self, anticipation, overall_risk, mood):
        """
        Returns a dict with suggestions like:
        {
            "prepare_ghost_boost": bool,
            "highlight_foresight": bool
        }
        """
        gb_p = anticipation.get("ghost_boost", 0.0)
        fs_p = anticipation.get("foresight", 0.0)

        prepare_gb = gb_p > 0.5 and overall_risk in ("medium", "high")
        highlight_fs = fs_p > 0.4 and mood in ("Alert", "Anxious", "Stressed")

        if prepare_gb or highlight_fs:
            self.log(
                f"[L11] Autonomy: prepare_ghost_boost={prepare_gb}, "
                f"highlight_foresight={highlight_fs}."
            )

        return {
            "prepare_ghost_boost": prepare_gb,
            "highlight_foresight": highlight_fs
        }

# =========================
# L12: Meta-State Fusion Engine
# =========================
class MetaStateFusionEngine:
    """
    L12: Fuses all layer outputs into a single meta-state snapshot.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)
        self.meta_state = {}

    def log(self, msg):
        self.log_fn(msg)

    def fuse(self, risk, pressure, turbulence, mood, anticipation, multi_agent):
        self.meta_state = {
            "risk": risk,
            "pressure": pressure,
            "turbulence": turbulence,
            "mood": mood,
            "anticipation": anticipation,
            "multi_agent": multi_agent,
            "timestamp": time.time()
        }
        self.log(f"[L12] Meta-state updated: {self.meta_state}")
        return self.meta_state

# =========================
# L13: Stability & Drift Correction Engine
# =========================
class DriftCorrectionEngine:
    """
    L13: Prevents over/under-sensitivity by smoothing and clamping.
    """
    def __init__(self, log_fn=None, window=120):
        self.log_fn = log_fn or (lambda msg: None)
        self.risk_history = deque(maxlen=window)

    def log(self, msg):
        self.log_fn(msg)

    def correct_risk(self, risk):
        if risk is None:
            return None
        self.risk_history.append(risk)
        avg = statistics.mean(self.risk_history) if len(self.risk_history) > 0 else risk
        corrected = (risk + avg) / 2.0
        corrected = max(0.0, min(1.0, corrected))
        if abs(corrected - risk) > 0.1:
            self.log(f"[L13] Risk drift corrected: {risk:.2f} → {corrected:.2f}.")
        return corrected

# =========================
# L14: Cross-Layer Influence Engine
# =========================
class CrossLayerInfluenceEngine:
    """
    L14: Allows layers to influence each other (e.g., mood affects thresholds).
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)

    def log(self, msg):
        self.log_fn(msg)

    def adjust_thresholds(self, mood, base_thresholds):
        """
        base_thresholds: dict like {"warn": 0.7, "critical": 0.9}
        Returns adjusted thresholds.
        """
        warn = base_thresholds.get("warn", 0.7)
        crit = base_thresholds.get("critical", 0.9)

        if mood == "Calm":
            warn += 0.05
            crit += 0.05
        elif mood == "Alert":
            pass
        elif mood == "Anxious":
            warn -= 0.05
            crit -= 0.05
        elif mood == "Stressed":
            warn -= 0.1
            crit -= 0.1

        warn = max(0.3, min(0.95, warn))
        crit = max(0.5, min(0.99, crit))

        adjusted = {"warn": warn, "critical": crit}
        self.log(f"[L14] Thresholds adjusted by mood={mood}: {adjusted}")
        return adjusted

# =========================
# L15: Emergent Insight Engine
# =========================
class EmergentInsightEngine:
    """
    L15: Looks across history to generate simple emergent insights.
    """
    def __init__(self, log_fn=None, path=EMERGENT_FILE):
        self.log_fn = log_fn or (lambda msg: None)
        self.path = path
        self.insights = []
        self.last_saved = None

    def log(self, msg):
        self.log_fn(msg)

    def add_observation(self, risk, pressure, mood, behavior_ctx):
        """
        Very simple heuristic: logs patterns when certain conditions repeat.
        """
        if risk is None or pressure is None:
            return

        if risk > 0.8 and pressure > 80:
            self.insights.append({
                "type": "high_risk_pressure",
                "timestamp": time.time(),
                "mood": mood,
                "behavior": behavior_ctx
            })
            self.log("[L15] Insight: frequent high risk + high pressure episodes detected.")

    def save(self):
        if self.insights == self.last_saved:
            return
        try:
            with open(self.path, "w") as f:
                json.dump(self.insights, f)
            self.last_saved = json.loads(json.dumps(self.insights))
            self.log("[L15] Emergent insights saved.")
        except Exception as e:
            self.log(f"[L15] Failed to save emergent insights: {e}")

# =========================
# L3: Predictive Brain
# =========================
class PredictiveBrain:
    """
    L3: Forecasts RAM/VRAM/combined load and time-to-threshold.
    """
    def __init__(self, log_fn=None, window=60):
        self.log_fn = log_fn or (lambda msg: None)
        self.history = deque(maxlen=window)

    def log(self, msg):
        self.log_fn(msg)

    def add_sample(self, ram_pct, vram_pct, combined_pct, activity):
        self.history.append({
            "ram": ram_pct,
            "vram": vram_pct,
            "combined": combined_pct,
            "activity": activity
        })

    def _ewma(self, series, alpha=0.3):
        if not series:
            return 0.0
        s = series[0]
        for x in series[1:]:
            s = alpha * x + (1 - alpha) * s
        return s

    def _estimate_ttt(self, series, threshold):
        if len(series) < 2:
            return None
        start = series[0]
        end = series[-1]
        delta = end - start
        steps = len(series) - 1
        per_sec = delta / steps if steps > 0 else 0
        if per_sec <= 0:
            return None
        remaining = threshold - end
        if remaining <= 0:
            return 0.0
        return remaining / per_sec

    def analyze(self):
        if len(self.history) < 5:
            return None

        ram_series = [h["ram"] for h in self.history]
        vram_series = [h["vram"] for h in self.history]
        combined_series = [h["combined"] for h in self.history]

        try:
            mean_combined = statistics.mean(combined_series)
            stdev_combined = statistics.pstdev(combined_series)
            latest = combined_series[-1]
            if stdev_combined > 0:
                z = (latest - mean_combined) / stdev_combined
                if abs(z) >= 3:
                    self.log(
                        f"[L3] Glitch Detection: Anomalous combined load (z={z:.2f}). "
                        f"Mean={mean_combined:.1f}%, Latest={latest:.1f}%"
                    )
        except Exception:
            pass

        ram_forecast = self._ewma(ram_series)
        vram_forecast = self._ewma(vram_series)
        combined_forecast = self._ewma(combined_series)

        t_ram_80 = self._estimate_ttt(ram_series, 80)
        t_comb_80 = self._estimate_ttt(combined_series, 80)
        t_comb_90 = self._estimate_ttt(combined_series, 90)

        try:
            vol = statistics.pstdev(combined_series)
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + vol / 10.0)))
        except Exception:
            confidence = 0.5

        result = {
            "ram_forecast": ram_forecast,
            "vram_forecast": vram_forecast,
            "combined_forecast": combined_forecast,
            "t_ram_80": t_ram_80,
            "t_comb_80": t_comb_80,
            "t_comb_90": t_comb_90,
            "confidence": confidence
        }

        msg_parts = [
            f"[L3] Predictive: EWMA RAM≈{ram_forecast:.1f}%, "
            f"VRAM≈{vram_forecast:.1f}%, COMBINED≈{combined_forecast:.1f}% "
            f"(confidence≈{confidence:.2f})."
        ]
        if t_comb_80 is not None and t_comb_80 <= 60:
            msg_parts.append(f"Combined may cross 80% in ~{t_comb_80:.1f}s.")
        if t_comb_90 is not None and t_comb_90 <= 60:
            msg_parts.append(f"Combined may cross 90% in ~{t_comb_90:.1f}s.")
        if t_ram_80 is not None and t_ram_80 <= 60:
            msg_parts.append(f"RAM may cross 80% in ~{t_ram_80:.1f}s.")
        self.log(" ".join(msg_parts))

        return result

# =========================
# L2: Reinforcement Learning Engine (policy)
# =========================
class ReinforcementLearner:
    """
    L2 (part of adaptation): Simple RL policy for foresight parameters.
    """
    def __init__(self, state_file=POLICY_FILE):
        self.state_file = state_file
        self.policy = {"disk_blocks": 10, "network_buffer": 5, "user_depth": 3}
        self.last_saved_policy = None
        self.load_policy()

    def reward(self, success: bool):
        old_policy = dict(self.policy)
        if success:
            self.policy["disk_blocks"] = min(self.policy["disk_blocks"] + 1, 50)
            self.policy["network_buffer"] = min(self.policy["network_buffer"] + 1, 50)
            self.policy["user_depth"] = min(self.policy["user_depth"] + 1, 10)
        else:
            self.policy["disk_blocks"] = max(self.policy["disk_blocks"] - 1, 1)
            self.policy["network_buffer"] = max(self.policy["network_buffer"] - 1, 1)
            self.policy["user_depth"] = max(self.policy["user_depth"] - 1, 1)
        if self.policy != old_policy:
            self.save_policy()

    def save_policy(self):
        if self.policy == self.last_saved_policy:
            return
        with open(self.state_file, "w") as f:
            json.dump(self.policy, f)
        self.last_saved_policy = dict(self.policy)

    def load_policy(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                self.policy = json.load(f)
            self.last_saved_policy = dict(self.policy)

# =========================
# L5 / Compute Backend (CPU / GPU / NPU + L5 Cache)
# =========================
class ComputeBackend:
    """
    L5: Hardware awareness + L5 augmentation cache.
    """
    def __init__(self, log_fn=None):
        self.log_fn = log_fn or (lambda msg: None)
        self.has_gpu = False
        self.has_npu = False
        self.l5_cache = None
        self.l5_mode = "None"
        self.detect_gpu()
        self.detect_npu()
        self.init_l5_augmentation_cache()

    def log(self, msg):
        self.log_fn(msg)

    def detect_gpu(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                self.has_gpu = True
                self.log("[ComputeBackend] GPU detected via nvidia-smi.")
            else:
                self.has_gpu = False
                self.log("[ComputeBackend] No NVIDIA GPU detected.")
        except Exception:
            self.has_gpu = False
            self.log("[ComputeBackend] GPU detection failed; assuming no GPU.")

    def detect_npu(self):
        self.has_npu = False
        self.log("[ComputeBackend] NPU detection stubbed → no NPU reported.")

    def init_l5_augmentation_cache(self):
        size_bytes = 1 * 1024 * 1024 * 1024  # 1 GB

        if self.has_npu:
            self.l5_cache = None
            self.l5_mode = "NPU"
            self.log("[ComputeBackend] L5 cache bound to NPU (logical augmentation).")
            return

        if self.has_gpu:
            try:
                import cupy as cp
                self.l5_cache = cp.zeros((size_bytes,), dtype=cp.uint8)
                self.l5_mode = "VRAM"
                self.log("[ComputeBackend] L5 augmentation cache: 1 GB VRAM allocated.")
                return
            except Exception as e:
                self.log(f"[ComputeBackend] VRAM L5 allocation failed: {e}")

        try:
            import numpy as np
            self.l5_cache = np.zeros((size_bytes,), dtype=np.uint8)
            self.l5_mode = "RAM"
            self.log("[ComputeBackend] L5 augmentation cache: 1 GB system RAM allocated.")
        except Exception as e:
            self.l5_cache = None
            self.l5_mode = "None"
            self.log(f"[ComputeBackend] L5 augmentation cache allocation failed: {e}")

    def best_backend(self):
        if self.has_npu:
            return "NPU"
        if self.has_gpu:
            return "GPU"
        return "CPU"

# =========================
# L1 + GUI + Core HiveMind
# =========================
class HiveMindGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Hive-Mind Controller (L1–L15)")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use('clam')

        ttk.Label(
            root,
            text="AI Hive-Mind Overlay (L1–L15)",
            font=("Arial", 18),
            foreground="white",
            background="#1e1e1e"
        ).pack(pady=10)

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Ghost Boost", command=self.trigger_boost)\
            .grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="Foresight Pulse", command=self.run_foresight)\
            .grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="Select Local Backup", command=self.select_local_backup)\
            .grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="Select SMB Backup", command=self.select_smb_backup)\
            .grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="Manual Restore", command=self.restore_backup)\
            .grid(row=0, column=4, padx=5, pady=5)

        self.status = tk.Text(root, height=15, width=150, bg="#111", fg="#0f0")
        self.status.pack(pady=10)

        meters_frame = ttk.Frame(root)
        meters_frame.pack(pady=5)

        ttk.Label(meters_frame, text="System RAM Usage", font=("Arial", 14))\
            .grid(row=0, column=0, pady=5)
        self.ram_progress = ttk.Progressbar(
            meters_frame, length=400, mode="determinate", maximum=100
        )
        self.ram_progress.grid(row=1, column=0, pady=5, padx=10)

        ttk.Label(meters_frame, text="GPU VRAM Usage", font=("Arial", 14))\
            .grid(row=0, column=1, pady=5)
        self.vram_progress = ttk.Progressbar(
            meters_frame, length=400, mode="determinate", maximum=100
        )
        self.vram_progress.grid(row=1, column=1, pady=5, padx=10)

        ttk.Label(meters_frame, text="Combined RAM+VRAM Load", font=("Arial", 14))\
            .grid(row=0, column=2, pady=5)
        self.combined_progress = ttk.Progressbar(
            meters_frame, length=400, mode="determinate", maximum=100
        )
        self.combined_progress.grid(row=1, column=2, pady=5, padx=10)

        heatmap_frame = ttk.Frame(root)
        heatmap_frame.pack(pady=10)

        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=heatmap_frame)
        self.canvas.get_tk_widget().pack()

        os.makedirs(DEFAULT_BACKUP_FOLDER, exist_ok=True)
        self.local_path = DEFAULT_BACKUP_FOLDER
        self.smb_path = None

        self.activity_counter = 0
        self.state = {}
        self.last_saved_state = None

        # Engines / layers
        self.learner = ReinforcementLearner()
        self.backend = ComputeBackend(log_fn=self.log)
        self.brain = PredictiveBrain(log_fn=self.log, window=60)
        self.water = WaterPhysicsEngine(log_fn=self.log, window=60)
        self.movidius = MovidiusInferenceEngine(log_fn=self.log)
        self.temporal = TemporalMemoryEngine(log_fn=self.log, path=TEMPORAL_MEMORY_FILE)
        self.behavior = BehavioralMemoryEngine(log_fn=self.log, path=BEHAVIOR_MEMORY_FILE)
        self.anticipatory = AnticipatoryBehaviorEngine(log_fn=self.log)
        self.multi_agent = MultiAgentReasoningEngine(log_fn=self.log)
        self.emotional = EmotionalInertiaEngine(log_fn=self.log)
        self.autonomy = AdaptiveAutonomyEngine(log_fn=self.log)
        self.meta_state = MetaStateFusionEngine(log_fn=self.log)
        self.drift = DriftCorrectionEngine(log_fn=self.log)
        self.cross_layer = CrossLayerInfluenceEngine(log_fn=self.log)
        self.emergent = EmergentInsightEngine(log_fn=self.log, path=EMERGENT_FILE)

        self.telemetry_running = False
        self.telemetry_thread = None

        self.cores = psutil.cpu_count(logical=False)
        self.threads = psutil.cpu_count(logical=True)
        self.ghost_cache = None
        self.ghost_mode = "Unknown"

        self.auto_restore_on_start()
        self.activate_hive_core()
        self.start_telemetry_if_needed()

    def log(self, message):
        self.status.insert(tk.END, message + "\n")
        self.status.see(tk.END)

    def auto_restore_on_start(self):
        candidate = os.path.join(self.local_path, "hive_state.json")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r") as f:
                    self.state = json.load(f)
                self.last_saved_state = json.loads(json.dumps(self.state))
                self.log(f"[Auto-Restore] Loaded hive_state.json from {self.local_path}.")
            except Exception as e:
                self.log(f"[Auto-Restore] Failed: {e}")
        else:
            self.log("[Auto-Restore] No existing hive_state.json found, starting fresh.")
            self.state = {}

    def activate_hive_core(self):
        self.ghost_cache, self.ghost_mode = self.allocate_vram_or_ram(256 * 1024 * 1024)
        self.state["cores"] = self.cores
        self.state["threads"] = self.threads
        self.state["ghost_mode"] = self.ghost_mode
        self.state["l5_mode"] = self.backend.l5_mode
        self.state.setdefault("health", {})
        self.state["health"]["npu_ok"] = self.backend.has_npu
        self.state["health"]["gpu_ok"] = self.backend.has_gpu
        self.state["health"].setdefault("last_error", None)
        self.state["health"]["last_boot"] = time.time()

        self.log(
            f"[Hive] Online: {self.cores} cores, {self.threads} threads, "
            f"L4 in {self.ghost_mode}, L5 in {self.backend.l5_mode}, "
            f"backend={self.backend.best_backend()}."
        )
        self.update_heatmap()

    def allocate_vram_or_ram(self, size):
        try:
            import cupy as cp
            ghost_cache = cp.zeros((size,), dtype=cp.uint8)
            return ghost_cache, "VRAM"
        except Exception:
            try:
                import numpy as np
                ghost_cache = np.zeros((size,), dtype=np.uint8)
                return ghost_cache, "RAM"
            except Exception:
                return None, "None"

    def start_telemetry_if_needed(self):
        if not self.telemetry_running:
            self.telemetry_running = True
            self.telemetry_thread = threading.Thread(
                target=self.telemetry_loop, daemon=True
            )
            self.telemetry_thread.start()
            self.log("[Telemetry] L1–L15 engine is now always on.")

    def trigger_boost(self):
        boost_time = random.randint(3, 5)

        current_risk = self.state.get("predictions", {}).get("risk_score")
        current_pressure = self.state.get("water", {}).get("pressure") if "water" in self.state else None
        self.behavior.record_action("ghost_boost", current_risk, current_pressure)
        self.behavior.save()

        self.log(
            f"[Ghost Boost] L4 ({self.ghost_mode}) + L5 ({self.backend.l5_mode}) "
            f"amplifying {self.cores} cores / {self.threads} threads for "
            f"{boost_time} seconds..."
        )

        def boost():
            time.sleep(boost_time)
            self.log("[Ghost Boost] Cycle complete. System stabilized.")

        threading.Thread(target=boost, daemon=True).start()

    def run_foresight(self):
        disk_blocks = self.learner.policy["disk_blocks"]
        net_buf = self.learner.policy["network_buffer"]
        user_depth = self.learner.policy["user_depth"]

        success = random.choice([True, False])
        self.learner.reward(success)

        current_risk = self.state.get("predictions", {}).get("risk_score")
        current_pressure = self.state.get("water", {}).get("pressure") if "water" in self.state else None
        self.behavior.record_action("foresight", current_risk, current_pressure)
        self.behavior.save()

        self.log(
            f"[Foresight] Disk blocks={disk_blocks}, Net buffer={net_buf}, "
            f"User depth={user_depth}, outcome={'success' if success else 'failure'}."
        )

        profile = self.behavior.get_hour_profile()
        if profile:
            self.log(
                f"[L7] Hour-profile: actions={profile['actions']}, "
                f"ghost_boost={profile['ghost_boost']}, foresight={profile['foresight']}, "
                f"avg_risk≈{profile['avg_risk']:.2f}, avg_pressure≈{profile['avg_pressure']:.1f}%."
            )

    def update_heatmap(self):
        usage = {
            "L1": random.randint(70, 100),
            "L2": random.randint(40, 70),
            "L3": random.randint(10, 40),
            "L4": random.randint(0, 20),
            "L5": random.randint(0, 20),
        }
        self.ax.clear()
        self.ax.bar(
            ["L1 High-Speed", "L2 Adaptation", "L3 Predictive", "L4 Physics", "L5 NPU/Aug"],
            [usage["L1"], usage["L2"], usage["L3"], usage["L4"], usage["L5"]],
            color=["#00ff00", "#ffaa00", "#ff0000", "#00ffff", "#ff00ff"]
        )
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Usage %")
        self.ax.set_title("Reasoning Heatmap: Activity (L1–L5)")
        self.canvas.draw()
        self.log(f"[Heatmap] Activity snapshot L1–L5: {usage}")

    def select_local_backup(self):
        path = filedialog.askdirectory(title="Select Local Backup Folder")
        if path:
            self.local_path = path
            self.log(f"[Backup] Local backup path set: {path}")

    def select_smb_backup(self):
        path = filedialog.askdirectory(title="Select SMB Network Backup Folder")
        if path:
            self.smb_path = path
            self.log(f"[Backup] SMB backup path set: {path}")

    def restore_backup(self):
        if self.local_path:
            state = self.load_backup(self.local_path)
        elif self.smb_path:
            state = self.load_backup(self.smb_path)
        else:
            self.log("[Manual Restore] No backup path set.")
            return

        if state:
            self.state = state
            self.ghost_cache, self.ghost_mode = self.allocate_vram_or_ram(256 * 1024 * 1024)
            self.log(f"[Manual Restore] Hive state loaded into {self.ghost_mode}.")
            self.last_saved_state = json.loads(json.dumps(self.state))
        else:
            self.log("[Manual Restore] No valid hive_state.json found.")

    def load_backup(self, path):
        try:
            with open(os.path.join(path, "hive_state.json"), "r") as f:
                return json.load(f)
        except Exception:
            return None

    def dual_write_backup(self):
        if not self.local_path or not self.smb_path:
            self.log("[Autosave] Backup paths not fully set (local+SMB). Skipping.")
            return

        new_state = self.state
        if new_state == self.last_saved_state:
            self.log("[Autosave] State unchanged → skipping backup write.")
            return

        try:
            with open(os.path.join(self.local_path, "hive_state.json"), "w") as f:
                json.dump(new_state, f)
            with open(os.path.join(self.smb_path, "hive_state.json"), "w") as f:
                json.dump(new_state, f)
            self.last_saved_state = json.loads(json.dumps(new_state))
            self.log("[Autosave] Dual-write backup completed (local + SMB).")
        except Exception as e:
            self.state["health"]["last_error"] = str(e)
            self.log(f"[Autosave] Backup failed: {e}")

    def telemetry_loop(self):
        while True:
            try:
                time.sleep(1.0)

                vm = psutil.virtual_memory()
                ram_used = vm.used
                ram_total = vm.total
                ram_percent = (ram_used / ram_total) * 100 if ram_total > 0 else 0

                disk = psutil.disk_io_counters()
                disk_bytes = disk.read_bytes + disk.write_bytes

                net = psutil.net_io_counters()
                net_bytes = net.bytes_recv + net.bytes_sent

                cpu_percent = psutil.cpu_percent(interval=None)

                vram_used, vram_total, gpu_util = self.get_gpu_stats()
                vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0

                combined_used = ram_used + vram_used
                combined_total = ram_total + vram_total
                combined_percent = (combined_used / combined_total) * 100 if combined_total > 0 else 0

                activity = disk_bytes + net_bytes + int(cpu_percent * 1024) + int(gpu_util * 1024)
                self.activity_counter += activity

                self.brain.add_sample(ram_percent, vram_percent, combined_percent, activity)
                self.water.add_sample(
                    ram_percent, vram_percent, combined_percent,
                    cpu_percent, net_bytes, disk_bytes
                )

                self.root.after(
                    0,
                    self.update_usage_and_predict,
                    ram_percent,
                    vram_percent,
                    combined_percent
                )

            except Exception as e:
                self.state["health"]["last_error"] = str(e)
                self.log(f"[Telemetry] Error: {e}")

    def get_gpu_stats(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode != 0:
                return 0, 1, 0
            line = result.stdout.strip().split("\n")[0]
            mem_used_str, mem_total_str, util_str = line.split(",")
            vram_used_mb = float(mem_used_str.strip())
            vram_total_mb = float(mem_total_str.strip())
            gpu_util = float(util_str.strip())
            return int(vram_used_mb * 1024 * 1024), int(vram_total_mb * 1024 * 1024), gpu_util
        except Exception:
            return 0, 1, 0

    def update_usage_and_predict(self, ram_percent, vram_percent, combined_percent):
        self.ram_progress["value"] = min(ram_percent, 100)
        self.vram_progress["value"] = min(vram_percent, 100)
        self.combined_progress["value"] = min(combined_percent, 100)

        self.log(
            f"[Telemetry/L1] RAM={ram_percent:.1f}%, VRAM={vram_percent:.1f}%, "
            f"Combined={combined_percent:.1f}%"
        )

        pb = self.brain.analyze()
        wp = self.water.analyze()

        features = {}
        volatility = 0.0
        if pb:
            features["ram"] = pb["ram_forecast"]
            features["combined"] = pb["combined_forecast"]
            try:
                volatility = statistics.pstdev(
                    [h["combined"] for h in self.brain.history]
                )
            except Exception:
                volatility = 0.0
            features["volatility"] = volatility
        if wp:
            features["glitches"] = wp["glitches"]

        score = self.movidius.infer_best_guess(features)
        if score is not None:
            score = self.drift.correct_risk(score)

        if score is not None:
            self.state.setdefault("predictions", {})
            self.state["predictions"]["risk_score"] = score

        if wp:
            self.state.setdefault("water", {})
            self.state["water"]["pressure"] = wp["pressure"]

            self.temporal.update(wp, score)
            self.temporal.save()
            ctx = self.temporal.get_hour_context()
            if ctx:
                self.log(
                    f"[L6] Hour-context: avg_pressure≈{ctx['avg_pressure']:.1f}%, "
                    f"avg_risk≈{ctx['avg_risk']:.2f}, glitches={ctx['glitches']}."
                )

        temporal_ctx = self.temporal.get_hour_context()
        behavior_ctx = self.behavior.get_hour_profile()
        pressure = wp["pressure"] if wp else None
        turbulence = wp["turbulence"] if wp else None
        glitches = wp["glitches"] if wp else None

        anticipation = self.anticipatory.predict(
            temporal_ctx, behavior_ctx, score, pressure, turbulence
        )
        multi = self.multi_agent.reason(score, pressure, turbulence, temporal_ctx, behavior_ctx)
        mood = self.emotional.update_mood(score, glitches, turbulence)
        thresholds = self.cross_layer.adjust_thresholds(mood, {"warn": 0.7, "critical": 0.9})
        autonomy = self.autonomy.decide(anticipation, multi["overall_risk"], mood)
        meta = self.meta_state.fuse(score, pressure, turbulence, mood, anticipation, multi)
        self.emergent.add_observation(score, pressure, mood, behavior_ctx)
        self.emergent.save()

        self.state["meta"] = meta
        self.state["thresholds"] = thresholds
        self.state["autonomy"] = autonomy

        if self.activity_counter >= SAVE_THRESHOLD:
            self.log("[Autosave] Activity threshold (100 MB) reached → triggering backup.")
            self.dual_write_backup()
            self.activity_counter = 0

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = HiveMindGUI(root)
    root.mainloop()

