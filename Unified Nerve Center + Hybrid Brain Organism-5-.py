#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System Nerve Center — Hybrid Brain Cortex (Evolved + SMB reboot memory)

Key additions in this version:
- SMB-aware reboot memory:
  * HybridBrain state and reinforcement/baseline are mirrored to an SMB/UNC path
    (e.g. \\SERVER\Share\HybridBrainState) if configured.
  * On startup, SMB state is preferred; local state is the fallback.
- GUI:
  * Local memory path (already existed).
  * New SMB memory path field + "Set SMB path" button.
  * Reasoning tail logs SMB sync success/failure.

Everything else from the evolved organism remains:
- Regime detection, multi-engine voting, pattern memory, SelfIntegrityOrgan,
  micro-recovery loops, meta-state evolution, auto-calibration, etc.
"""

import sys
import os
import json
import time
import math
import random
from collections import deque, defaultdict
from datetime import datetime

# ----------------------------
# Auto-loader for dependencies
# ----------------------------

def safe_import(name, pip_name=None):
    try:
        module = __import__(name)
        return module, "ok"
    except ImportError:
        return None, f"missing (pip install {pip_name or name})"


AUTOLOADER_STATUS = {}

psutil, AUTOLOADER_STATUS["psutil"] = safe_import("psutil", "psutil")
pynvml, AUTOLOADER_STATUS["pynvml"] = safe_import("pynvml", "pynvml")
requests, AUTOLOADER_STATUS["requests"] = safe_import("requests", "requests")
numpy, AUTOLOADER_STATUS["numpy"] = safe_import("numpy", "numpy")
onnxruntime, AUTOLOADER_STATUS["onnxruntime"] = safe_import("onnxruntime", "onnxruntime")

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    AUTOLOADER_STATUS["tkinter"] = "ok"
except Exception as e:
    tk = None
    ttk = None
    messagebox = None
    filedialog = None
    AUTOLOADER_STATUS["tkinter"] = f"unavailable: {e}"

# ----------------------------
# Utility helpers
# ----------------------------

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_state")
os.makedirs(STATE_DIR, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def now_ts():
    return time.time()

def safe_percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0

def smooth_ewma(prev, new, alpha):
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev

def hour_bucket():
    return datetime.now().strftime("%H")

def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False

# ----------------------------
# Meta-States & Stances
# ----------------------------

META_STATES = ["Hyper-Flow", "Deep-Dream", "Sentinel", "Recovery-Flow"]
STANCE_MODES = ["Calm", "Balanced", "Aggressive"]

DEFAULT_META_STATE = "Sentinel"
DEFAULT_STANCE = "Balanced"

# ----------------------------
# Base Organ
# ----------------------------

class BaseOrgan:
    def __init__(self, name):
        self.name = name
        self.last_update_ts = 0
        self.metrics = {}
        self.health_score = 1.0
        self.risk_score = 0.0
        self.enabled = True
        self.reason_tail = deque(maxlen=10)

    def update(self):
        self.last_update_ts = now_ts()

    def snapshot(self):
        return {
            "name": self.name,
            "ts": self.last_update_ts,
            "metrics": self.metrics,
            "health": self.health_score,
            "risk": self.risk_score,
            "enabled": self.enabled,
            "reason_tail": list(self.reason_tail),
        }


# ----------------------------
# Organs
# ----------------------------

class DeepRamOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DeepRamOrgan")
        self.appetite = 0.5
        self.target_headroom = 0.25

    def update(self):
        super().update()
        total = used = available = 0
        if psutil:
            try:
                vm = psutil.virtual_memory()
                total = vm.total
                available = vm.available
                used = total - available
            except Exception as e:
                self.reason_tail.append(f"psutil mem error: {e}")
        else:
            total = 16 * 1024 * 1024 * 1024
            used = random.uniform(0.2, 0.8) * total
            available = total - used
            self.reason_tail.append("Synthetic RAM metrics (psutil missing)")

        used_pct = safe_percent(used, total)
        free_pct = safe_percent(available, total)

        diff = abs(free_pct / 100.0 - self.target_headroom)
        self.health_score = clamp(1.0 - diff * 2.0, 0.0, 1.0)
        self.risk_score = clamp((used_pct / 100.0) ** 2, 0.0, 1.0)

        self.metrics = {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "used_pct": used_pct,
            "free_pct": free_pct,
            "appetite": self.appetite,
            "target_headroom": self.target_headroom,
        }


class BackupEngineOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("BackupEngineOrgan")
        self.paths = []
        self.snapshots = deque(maxlen=50)
        self.integrity_score = 1.0
        self.last_known_good = None

    def add_path(self, path):
        if path not in self.paths:
            self.paths.append(path)

    def save_snapshot(self, tag="auto"):
        ts = datetime.now().isoformat()
        snapshot = {
            "ts": ts,
            "tag": tag,
            "integrity": self.integrity_score,
        }
        self.snapshots.append(snapshot)
        self.last_known_good = snapshot

    def auto_rollback(self):
        if self.last_known_good:
            self.reason_tail.append("Auto-rollback to last known good state")
        else:
            self.reason_tail.append("No last known good snapshot for rollback")

    def update(self):
        super().update()
        if len(self.paths) == 0:
            self.integrity_score = 0.2
        else:
            self.integrity_score = clamp(0.3 + 0.1 * len(self.paths), 0.0, 1.0)

        self.health_score = self.integrity_score
        self.risk_score = clamp(1.0 - self.integrity_score, 0.0, 1.0)
        self.metrics = {
            "paths": list(self.paths),
            "snapshot_count": len(self.snapshots),
            "integrity_score": self.integrity_score,
            "last_known_good": self.last_known_good,
        }


class NetworkWatcherOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("NetworkWatcherOrgan")
        self.last_bytes_sent = None
        self.last_bytes_recv = None
        self.last_ts = None

    def update(self):
        super().update()
        if psutil:
            try:
                counters = psutil.net_io_counters()
                now = now_ts()
                sent = counters.bytes_sent
                recv = counters.bytes_recv

                if self.last_ts is not None:
                    dt = now - self.last_ts
                    if dt <= 0:
                        dt = 1e-6
                    up_bps = (sent - self.last_bytes_sent) / dt
                    down_bps = (recv - self.last_bytes_recv) / dt
                else:
                    up_bps = down_bps = 0.0

                self.last_bytes_sent = sent
                self.last_bytes_recv = recv
                self.last_ts = now

                total_bps = abs(up_bps) + abs(down_bps)
                self.risk_score = clamp(math.log10(1 + total_bps) / 10.0, 0.0, 1.0)
                self.health_score = clamp(1.0 - self.risk_score, 0.0, 1.0)

                self.metrics = {
                    "up_bps": up_bps,
                    "down_bps": down_bps,
                    "total_bps": total_bps,
                }
            except Exception as e:
                self.reason_tail.append(f"net error: {e}")
        else:
            self.metrics = {"up_bps": 0.0, "down_bps": 0.0, "total_bps": 0.0}
            self.health_score = 0.8
            self.risk_score = 0.2
            self.reason_tail.append("Synthetic net metrics (psutil missing)")


class GPUCacheOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("GPUCacheOrgan")
        self.cache_fill_pct = 0.2
        self.hit_rate = 0.9

    def update(self):
        super().update()
        self.cache_fill_pct = clamp(self.cache_fill_pct + random.uniform(-0.02, 0.02), 0.0, 1.0)
        self.hit_rate = clamp(self.hit_rate + random.uniform(-0.01, 0.01), 0.0, 1.0)
        self.health_score = clamp(self.hit_rate * (1.0 - self.cache_fill_pct * 0.5), 0.0, 1.0)
        self.risk_score = clamp(self.cache_fill_pct * 0.7, 0.0, 1.0)
        self.metrics = {
            "cache_fill_pct": self.cache_fill_pct,
            "hit_rate": self.hit_rate,
        }


class ThermalOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("ThermalOrgan")
        self.temp_c = None
        self.micro_cool_mode = False

    def update(self):
        super().update()
        temp = None
        if psutil and hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                vals = []
                for name, entries in temps.items():
                    for e in entries:
                        if e.current is not None:
                            vals.append(e.current)
                if vals:
                    temp = sum(vals) / len(vals)
            except Exception as e:
                self.reason_tail.append(f"thermal error: {e}")
        if temp is None:
            temp = 40.0 + random.uniform(-5, 10)
            self.reason_tail.append("Synthetic thermal metrics (no sensors)")

        self.temp_c = temp
        if temp <= 60:
            self.risk_score = 0.1
        elif temp <= 75:
            self.risk_score = 0.4
        elif temp <= 85:
            self.risk_score = 0.7
        else:
            self.risk_score = 0.95

        self.health_score = clamp(1.0 - self.risk_score, 0.0, 1.0)
        self.metrics = {
            "temp_c": self.temp_c,
            "micro_cool_mode": self.micro_cool_mode,
        }


class DiskOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DiskOrgan")
        self.last_read_bytes = None
        self.last_write_bytes = None
        self.last_ts = None
        self.micro_io_relief = False

    def update(self):
        super().update()
        if psutil:
            try:
                io = psutil.disk_io_counters()
                now = now_ts()
                rb = io.read_bytes
                wb = io.write_bytes

                if self.last_ts is not None:
                    dt = now - self.last_ts
                    if dt <= 0:
                        dt = 1e-6
                    rps = (rb - self.last_read_bytes) / dt
                    wps = (wb - self.last_write_bytes) / dt
                else:
                    rps = wps = 0.0

                self.last_read_bytes = rb
                self.last_write_bytes = wb
                self.last_ts = now

                total_io = abs(rps) + abs(wps)
                self.risk_score = clamp(math.log10(1 + total_io) / 10.0, 0.0, 1.0)
                self.health_score = clamp(1.0 - self.risk_score, 0.0, 1.0)

                self.metrics = {
                    "read_bps": rps,
                    "write_bps": wps,
                    "total_bps": total_io,
                    "micro_io_relief": self.micro_io_relief,
                }
            except Exception as e:
                self.reason_tail.append(f"disk error: {e}")
        else:
            self.metrics = {
                "read_bps": 0.0,
                "write_bps": 0.0,
                "total_bps": 0.0,
                "micro_io_relief": self.micro_io_relief,
            }
            self.health_score = 0.8
            self.risk_score = 0.2
            self.reason_tail.append("Synthetic disk metrics (psutil missing)")


class VRAMOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("VRAMOrgan")
        self.vram_used_pct = 0.0
        self.nvml_ok = False
        self.cooldown_mode = False
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_ok = True
            except Exception as e:
                self.reason_tail.append(f"NVML init error: {e}")

    def update(self):
        super().update()
        if self.nvml_ok:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used = info.used
                total = info.total
                self.vram_used_pct = safe_percent(used, total)
            except Exception as e:
                self.reason_tail.append(f"NVML error: {e}")
                self.vram_used_pct = 0.0
        else:
            self.vram_used_pct = random.uniform(0.1, 0.7)
            self.reason_tail.append("Synthetic VRAM metrics (NVML missing)")

        self.risk_score = clamp((self.vram_used_pct / 100.0) ** 1.5, 0.0, 1.0)
        self.health_score = clamp(1.0 - self.risk_score, 0.0, 1.0)
        self.metrics = {
            "vram_used_pct": self.vram_used_pct,
            "cooldown_mode": self.cooldown_mode,
        }


class AICoachOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("AICoachOrgan")
        self.last_tip = "All quiet."

    def update_with_brain_view(self, brain_snapshot):
        super().update()
        risk = brain_snapshot.get("best_guess_risk", 0.0)
        meta_state = brain_snapshot.get("meta_state", DEFAULT_META_STATE)
        stance = brain_snapshot.get("stance_mode", DEFAULT_STANCE)

        if risk > 0.8:
            self.last_tip = "High turbulence forecast: shrink load, favor stability."
        elif risk > 0.5:
            self.last_tip = "Moderate risk ahead: keep stance Balanced, avoid big spikes."
        else:
            if meta_state == "Hyper-Flow":
                self.last_tip = "Conditions good. This is a good time to push."
            elif meta_state == "Deep-Dream":
                self.last_tip = "Background optimization. Avoid sudden heavy loads."
            elif meta_state == "Sentinel":
                self.last_tip = "Guard posture. System ready but cautious."
            else:
                self.last_tip = "Recovery in progress. Let the system settle."

        self.health_score = clamp(1.0 - risk, 0.0, 1.0)
        self.risk_score = risk
        self.metrics = {
            "last_tip": self.last_tip,
            "meta_state": meta_state,
            "stance_mode": stance,
            "best_guess_risk": risk,
        }
        self.reason_tail.append(self.last_tip)


class SwarmNodeOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("SwarmNodeOrgan")
        self.node_id = "local-node"
        self.swarm_health = 1.0

    def update(self):
        super().update()
        self.swarm_health = clamp(self.swarm_health + random.uniform(-0.02, 0.02), 0.0, 1.0)
        self.health_score = self.swarm_health
        self.risk_score = clamp(1.0 - self.swarm_health, 0.0, 1.0)
        self.metrics = {
            "node_id": self.node_id,
            "swarm_health": self.swarm_health,
        }


class Back4BloodAnalyzer(BaseOrgan):
    def __init__(self):
        super().__init__("Back4BloodAnalyzer")
        self.last_game_state = "idle"

    def update_with_organs(self, organs_snapshot):
        super().update()
        avg_risk = sum(o["risk"] for o in organs_snapshot.values()) / max(len(organs_snapshot), 1)
        if avg_risk > 0.7:
            self.last_game_state = "overloaded"
        elif avg_risk < 0.3:
            self.last_game_state = "stable"
        else:
            self.last_game_state = "transient"

        self.health_score = clamp(1.0 - avg_risk, 0.0, 1.0)
        self.risk_score = avg_risk
        self.metrics = {
            "game_state": self.last_game_state,
            "avg_risk_from_organs": avg_risk,
        }
        self.reason_tail.append(f"Game/system fusion: {self.last_game_state}")


class SelfIntegrityOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("SelfIntegrityOrgan")
        self.integrity = 1.0

    def update_with_brain_view(self, brain_snapshot, auto_status):
        super().update()
        issues = 0

        if AUTOLOADER_STATUS.get("psutil", "").startswith("missing"):
            issues += 1
        if AUTOLOADER_STATUS.get("tkinter", "").startswith("unavailable"):
            issues += 1
        if brain_snapshot["model_integrity"] < 0.5:
            issues += 1
        if brain_snapshot["meta_confidence"] < 0.4:
            issues += 1

        self.integrity = clamp(1.0 - 0.25 * issues, 0.0, 1.0)
        self.health_score = self.integrity
        self.risk_score = clamp(1.0 - self.integrity, 0.0, 1.0)

        self.metrics = {
            "integrity": self.integrity,
            "issues": issues,
            "auto_loader_status": auto_status,
        }
        if issues > 0:
            self.reason_tail.append(f"Integrity issues: {issues}")
        else:
            self.reason_tail.append("Integrity stable")


# ----------------------------
# MovidiusInferenceEngine
# ----------------------------

class MovidiusInferenceEngine:
    def __init__(self, enabled=False, model_path=None):
        self.enabled = enabled
        self.model_path = model_path
        self.session = None
        self.use_onnx = False
        self._init_session()

    def _init_session(self):
        self.session = None
        self.use_onnx = False
        if not self.enabled or onnxruntime is None:
            return
        if not self.model_path or not os.path.exists(self.model_path):
            return
        try:
            self.session = onnxruntime.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.use_onnx = True
        except Exception:
            self.session = None
            self.use_onnx = False

    def set_model_path(self, path):
        self.model_path = path
        self._init_session()

    def infer(self, risk_series):
        if not self.enabled or not risk_series:
            last = risk_series[-1] if risk_series else 0.0
            return {
                "pred_1s": last,
                "pred_5s": last,
                "pred_30s": last,
                "pred_120s": last,
                "model_conf": 0.5,
            }

        if self.use_onnx and self.session is not None and numpy is not None:
            try:
                arr = numpy.array(risk_series[-60:], dtype=numpy.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: arr})
                out = outputs[0].flatten().tolist()
                if len(out) >= 5:
                    p1, p5, p30, p120, conf = out[:5]
                else:
                    last = risk_series[-1]
                    p1 = out[0] if len(out) > 0 else last
                    p5 = out[1] if len(out) > 1 else p1
                    p30 = out[2] if len(out) > 2 else p5
                    p120 = out[3] if len(out) > 3 else p30
                    conf = out[4] if len(out) > 4 else 0.5
                return {
                    "pred_1s": clamp(float(p1), 0.0, 1.0),
                    "pred_5s": clamp(float(p5), 0.0, 1.0),
                    "pred_30s": clamp(float(p30), 0.0, 1.0),
                    "pred_120s": clamp(float(p120), 0.0, 1.0),
                    "model_conf": clamp(float(conf), 0.0, 1.0),
                }
            except Exception:
                pass

        try:
            if numpy:
                arr = numpy.array(risk_series[-60:], dtype=float)
                mean = float(arr.mean())
                std = float(arr.std())
            else:
                series = risk_series[-60:]
                mean = sum(series) / len(series)
                var = sum((x - mean) ** 2 for x in series) / len(series)
                std = math.sqrt(var)
            vol_factor = clamp(std * 1.5, 0.0, 1.0)
            return {
                "pred_1s": clamp(mean + 0.1 * vol_factor, 0.0, 1.0),
                "pred_5s": clamp(mean + 0.2 * vol_factor, 0.0, 1.0),
                "pred_30s": clamp(mean + 0.35 * vol_factor, 0.0, 1.0),
                "pred_120s": clamp(mean + 0.45 * vol_factor, 0.0, 1.0),
                "model_conf": clamp(1.0 - std * 1.2, 0.0, 1.0),
            }
        except Exception:
            last = risk_series[-1]
            return {
                "pred_1s": last,
                "pred_5s": last,
                "pred_30s": last,
                "pred_120s": last,
                "model_conf": 0.4,
            }


# ----------------------------
# HybridBrain
# ----------------------------

class HybridBrain:
    def __init__(self, deep_ram_organ, backup_organ, network_organ,
                 gpu_organ, thermal_organ, disk_organ, vram_organ,
                 ai_coach_organ, swarm_organ, integrity_organ,
                 movidius_available=False, movidius_model_path=None,
                 smb_state_dir=None):
        self.deep_ram = deep_ram_organ
        self.backup = backup_organ
        self.network = network_organ
        self.gpu = gpu_organ
        self.thermal = thermal_organ
        self.disk = disk_organ
        self.vram = vram_organ
        self.ai_coach = ai_coach_organ
        self.swarm = swarm_organ
        self.integrity_organ = integrity_organ

        self.movidius_engine = MovidiusInferenceEngine(
            enabled=movidius_available,
            model_path=movidius_model_path
        )

        # SMB reboot memory directory (UNC or mapped path)
        self.smb_state_dir = smb_state_dir  # can be set later
        self.smb_state_ok = False

        self.risk_history = deque(maxlen=240)
        self.timestamp_history = deque(maxlen=240)

        self.ewma_1s = None
        self.ewma_5s = None
        self.ewma_30s = None
        self.ewma_120s = None

        self.mv_pred_1s = None
        self.mv_pred_5s = None
        self.mv_pred_30s = None
        self.mv_pred_120s = None
        self.mv_model_conf = 0.5

        # Local and SMB state filenames
        self.baseline_file_local = os.path.join(STATE_DIR, "baseline.json")
        self.reinforcement_file_local = os.path.join(STATE_DIR, "reinforcement.json")
        self.reboot_file_local = os.path.join(STATE_DIR, "reboot_snapshot.json")

        # SMB paths computed lazily when smb_state_dir is set
        self.baseline_file_smb = None
        self.reinforcement_file_smb = None
        self.reboot_file_smb = None

        self.baseline_by_hour = defaultdict(lambda: {"count": 0, "avg_risk": 0.2})
        self.reinforcement_buckets = {
            "overload": 0,
            "stability": 0,
            "beast_mode_win": 0,
        }
        self.pattern_memory = defaultdict(lambda: {"count": 0, "overload": 0, "stable": 0, "wins": 0})

        self.stance_thresholds = {
            "Calm": 0.4,
            "Balanced": 0.7,
            "Aggressive": 0.9,
        }

        self.current_meta_state = DEFAULT_META_STATE
        self.current_stance = DEFAULT_STANCE
        self.current_regime = "unknown"

        self.reasoning_tail = deque(maxlen=80)

        self.model_integrity = 1.0
        self.health_score = 1.0
        self.risk_score = 0.0
        self.collective_health_score = 1.0
        self.meta_confidence = 0.5
        self.best_guess_risk = 0.0

        self.meta_effects = {
            "pred_horizon_bias": "",
            "ram_appetite": "",
            "thread_expansion": "",
            "cache_behavior": "",
        }

        self.last_auto_calibration_ts = now_ts()

        self._update_smb_paths()
        self.load_state()

    # ---------- SMB path handling ----------

    def set_smb_state_dir(self, path):
        self.smb_state_dir = path.strip() if path else None
        self._update_smb_paths()
        if self.smb_state_dir:
            if safe_mkdir(self.smb_state_dir):
                self.smb_state_ok = True
                self.reasoning_tail.append(f"SMB state dir set: {self.smb_state_dir}")
            else:
                self.smb_state_ok = False
                self.reasoning_tail.append(f"SMB state dir failed: {self.smb_state_dir}")

    def _update_smb_paths(self):
        if self.smb_state_dir:
            self.baseline_file_smb = os.path.join(self.smb_state_dir, "baseline.json")
            self.reinforcement_file_smb = os.path.join(self.smb_state_dir, "reinforcement.json")
            self.reboot_file_smb = os.path.join(self.smb_state_dir, "reboot_snapshot.json")
        else:
            self.baseline_file_smb = None
            self.reinforcement_file_smb = None
            self.reboot_file_smb = None

    # ---------- Persistence ----------

    def _load_json_if_exists(self, path):
        try:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def load_state(self):
        # Prefer SMB if available, else local
        used_source = "none"
        data_baseline = None
        data_reinf = None

        smb_baseline = self._load_json_if_exists(self.baseline_file_smb)
        smb_reinf = self._load_json_if_exists(self.reinforcement_file_smb)

        if smb_baseline or smb_reinf:
            used_source = "smb"
            data_baseline = smb_baseline or self._load_json_if_exists(self.baseline_file_local)
            data_reinf = smb_reinf or self._load_json_if_exists(self.reinforcement_file_local)
        else:
            used_source = "local"
            data_baseline = self._load_json_if_exists(self.baseline_file_local)
            data_reinf = self._load_json_if_exists(self.reinforcement_file_local)

        try:
            if data_baseline:
                self.baseline_by_hour.update(data_baseline.get("baseline_by_hour", {}))
            if data_reinf:
                self.reinforcement_buckets.update(data_reinf.get("reinforcement_buckets", {}))
                # handle both dict and defaultdict-style persisted pattern_memory
                pm = data_reinf.get("pattern_memory", {})
                for k, v in pm.items():
                    self.pattern_memory[k].update(v)
            self.reasoning_tail.append(f"State loaded from {used_source}")
        except Exception as e:
            self.reasoning_tail.append(f"State load error: {e}")

    def _safe_save_json(self, path, payload):
        if not path:
            return False
        try:
            parent = os.path.dirname(path)
            safe_mkdir(parent)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
            return True
        except Exception:
            return False

    def save_state(self):
        baseline_payload = {"baseline_by_hour": self.baseline_by_hour}
        reinf_payload = {
            "reinforcement_buckets": self.reinforcement_buckets,
            "pattern_memory": self.pattern_memory,
        }
        reboot_payload = {
            "timestamp": datetime.now().isoformat(),
            "last_best_guess": self.best_guess_risk,
            "meta_state": self.current_meta_state,
            "stance": self.current_stance,
            "regime": self.current_regime,
        }

        local_ok1 = self._safe_save_json(self.baseline_file_local, baseline_payload)
        local_ok2 = self._safe_save_json(self.reinforcement_file_local, reinf_payload)
        local_ok3 = self._safe_save_json(self.reboot_file_local, reboot_payload)

        smb_ok1 = smb_ok2 = smb_ok3 = False
        if self.smb_state_dir:
            smb_ok1 = self._safe_save_json(self.baseline_file_smb, baseline_payload)
            smb_ok2 = self._safe_save_json(self.reinforcement_file_smb, reinf_payload)
            smb_ok3 = self._safe_save_json(self.reboot_file_smb, reboot_payload)
            self.smb_state_ok = smb_ok1 and smb_ok2 and smb_ok3

        msg = f"State saved local({local_ok1 and local_ok2 and local_ok3})"
        if self.smb_state_dir:
            msg += f" smb({self.smb_state_ok})"
        self.reasoning_tail.append(msg)

    # ---------- Core update ----------

    def update(self, organs_dict):
        total_risk = 0.0
        total_health = 0.0
        count = 0
        for organ in organs_dict.values():
            total_risk += organ.risk_score
            total_health += organ.health_score
            count += 1
        avg_risk = total_risk / count if count > 0 else 0.0
        avg_health = total_health / count if count > 0 else 1.0

        self.risk_history.append(avg_risk)
        self.timestamp_history.append(now_ts())
        self.risk_score = avg_risk
        self.health_score = avg_health

        self.ewma_1s = smooth_ewma(self.ewma_1s, avg_risk, 0.6)
        self.ewma_5s = smooth_ewma(self.ewma_5s, avg_risk, 0.3)
        self.ewma_30s = smooth_ewma(self.ewma_30s, avg_risk, 0.1)
        self.ewma_120s = smooth_ewma(self.ewma_120s, avg_risk, 0.05)

        variance = self._compute_variance(self.risk_history)
        trend = self._compute_trend()
        turbulence = self._water_physics_turbulence(variance, trend)

        mv_out = self.movidius_engine.infer(list(self.risk_history))
        self.mv_pred_1s = mv_out["pred_1s"]
        self.mv_pred_5s = mv_out["pred_5s"]
        self.mv_pred_30s = mv_out["pred_30s"]
        self.mv_pred_120s = mv_out["pred_120s"]
        self.mv_model_conf = mv_out["model_conf"]

        pred_1s = self._fuse_horizon(self.ewma_1s, self.mv_pred_1s)
        pred_5s = self._fuse_horizon(self.ewma_5s, self.mv_pred_5s)
        pred_30s = self._fuse_horizon(self.ewma_30s, self.mv_pred_30s)
        pred_120s = self._fuse_horizon(self.ewma_120s, self.mv_pred_120s)

        self.current_regime = self._detect_regime(avg_risk, variance, trend, turbulence)

        reinforcement_conf = self._reinforcement_conf()
        base_conf = 1.0 - clamp(variance * 1.5 + abs(trend) * 0.5 + turbulence * 0.5, 0.0, 1.0)
        base_conf = (base_conf + reinforcement_conf) / 2.0
        base_conf = (base_conf + self.mv_model_conf) / 2.0
        self.meta_confidence = clamp(base_conf, 0.0, 1.0)

        self._update_regime_baseline(avg_risk)

        fingerprint_tags = self._behavioral_fingerprints(avg_risk, variance, trend, turbulence)

        engine_contribs, self.best_guess_risk = self._multi_engine_vote(
            avg_risk, pred_1s, pred_5s, pred_30s, pred_120s,
            variance, trend, turbulence, reinforcement_conf
        )

        self._update_dynamic_stance_thresholds()

        self.current_meta_state = self._select_meta_state(self.best_guess_risk, variance, trend, turbulence)

        self._apply_meta_state_effects(self.best_guess_risk)

        self.current_stance = self._choose_stance(self.best_guess_risk)

        actions = self._predictive_dampening(self.best_guess_risk)

        self._apply_micro_recovery_loops()

        self.model_integrity = self._estimate_model_integrity()
        self.collective_health_score = clamp(
            (self.health_score * 0.35 +
             (1.0 - self.risk_score) * 0.25 +
             self.model_integrity * 0.2 +
             self.meta_confidence * 0.2),
            0.0, 1.0
        )

        self.integrity_organ.update_with_brain_view(
            self.snapshot(self.best_guess_risk, fingerprint_tags, actions,
                          pred_1s, pred_5s, pred_30s, pred_120s),
            dict(AUTOLOADER_STATUS)
        )

        self.ai_coach.update_with_brain_view(
            self.snapshot(self.best_guess_risk, fingerprint_tags, actions,
                          pred_1s, pred_5s, pred_30s, pred_120s)
        )

        self._append_reasoning(
            avg_risk, self.best_guess_risk, variance, trend,
            turbulence, fingerprint_tags, actions, engine_contribs
        )

        self._auto_calibration()

        if random.random() < 0.01:
            self.save_state()

    # ---------- Internal helpers ----------

    def _compute_variance(self, series):
        if not series:
            return 0.0
        mean = sum(series) / len(series)
        var = sum((x - mean) ** 2 for x in series) / len(series)
        return var

    def _compute_trend(self):
        if len(self.risk_history) < 5:
            return 0.0
        ys = list(self.risk_history)
        xs = list(range(len(ys)))
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope

    def _water_physics_turbulence(self, variance, trend):
        turb = clamp(variance * 2.0 + abs(trend) * 0.5, 0.0, 2.0)
        return clamp(turb / 2.0, 0.0, 1.0)

    def _detect_regime(self, avg_risk, variance, trend, turbulence):
        if variance < 0.01 and abs(trend) < 0.01:
            return "stable"
        if trend > 0.02 and variance < 0.04:
            return "rising"
        if turbulence > 0.6:
            return "chaotic"
        if avg_risk > 0.6 and trend < -0.02:
            return "cooling"
        return "transient"

    def _reinforcement_conf(self):
        total = sum(self.reinforcement_buckets.values()) or 1
        stability = self.reinforcement_buckets["stability"] / total
        overload = self.reinforcement_buckets["overload"] / total
        wins = self.reinforcement_buckets["beast_mode_win"] / total
        return clamp(0.2 + 0.5 * stability + 0.4 * wins - 0.4 * overload, 0.0, 1.0)

    def _update_regime_baseline(self, avg_risk):
        hb = hour_bucket()
        entry = self.baseline_by_hour[hb]
        c = entry["count"]
        avg = entry["avg_risk"]
        new_avg = (avg * c + avg_risk) / (c + 1)
        self.baseline_by_hour[hb] = {"count": c + 1, "avg_risk": new_avg}

    def _behavioral_fingerprints(self, avg_risk, variance, trend, turbulence):
        tags = []

        if avg_risk > 0.8:
            tags.append("overload-pattern")
            self.reinforcement_buckets["overload"] += 1
        elif avg_risk < 0.3 and variance < 0.02:
            tags.append("stability-pattern")
            self.reinforcement_buckets["stability"] += 1

        if 0.4 < avg_risk < 0.7 and variance < 0.02 and abs(trend) < 0.01:
            tags.append("beast-mode-win")
            self.reinforcement_buckets["beast_mode_win"] += 1

        if variance > 0.05:
            tags.append("oscillation")
        if trend > 0.02:
            tags.append("rising-risk")
        elif trend < -0.02:
            tags.append("falling-risk")
        if turbulence > 0.5:
            tags.append("high-turbulence")

        for t in tags:
            pm = self.pattern_memory[t]
            pm["count"] += 1
            if "overload" in t:
                pm["overload"] += 1
            if "stability" in t:
                pm["stable"] += 1
            if "beast" in t:
                pm["wins"] += 1

        return tags

    def _multi_engine_vote(self, avg_risk, p1, p5, p30, p120,
                           variance, trend, turbulence, reinforcement_conf):
        hb = hour_bucket()
        baseline_risk = self.baseline_by_hour[hb]["avg_risk"]
        baseline_dev = avg_risk - baseline_risk

        engines = {}

        engines["ewma_30"] = clamp(p30, 0.0, 1.0)
        engines["trend"] = clamp(avg_risk + trend * 20.0, 0.0, 1.0)
        engines["variance"] = clamp(avg_risk + variance * 2.0, 0.0, 1.0)
        engines["turbulence"] = clamp(avg_risk + turbulence * 0.3, 0.0, 1.0)
        engines["movidius"] = clamp(self.mv_pred_30s, 0.0, 1.0)
        engines["baseline_dev"] = clamp(baseline_risk + baseline_dev * 1.5, 0.0, 1.0)
        engines["reinforcement"] = clamp(avg_risk * (0.5 + 0.5 * reinforcement_conf), 0.0, 1.0)

        weights = {
            "ewma_30": 0.2,
            "trend": 0.1,
            "variance": 0.1,
            "turbulence": 0.1,
            "movidius": 0.25,
            "baseline_dev": 0.15,
            "reinforcement": 0.1,
        }

        if self.current_regime == "stable":
            weights["ewma_30"] += 0.1
            weights["baseline_dev"] += 0.05
        elif self.current_regime == "rising":
            weights["trend"] += 0.1
            weights["movidius"] += 0.05
        elif self.current_regime == "chaotic":
            weights["variance"] += 0.1
            weights["turbulence"] += 0.1
        elif self.current_regime == "cooling":
            weights["baseline_dev"] += 0.05
            weights["ewma_30"] += 0.05

        total_w = sum(weights.values()) or 1.0
        for k in weights:
            weights[k] /= total_w

        guess = 0.0
        contribs = {}
        for name, val in engines.items():
            w = weights.get(name, 0.0)
            guess += w * val
            contribs[name] = (val, w)

        guess = clamp(guess, 0.0, 1.0)
        return contribs, guess

    def _update_dynamic_stance_thresholds(self):
        hb = hour_bucket()
        baseline_risk = self.baseline_by_hour[hb]["avg_risk"]

        overload = self.reinforcement_buckets["overload"]
        stability = self.reinforcement_buckets["stability"] or 1
        overload_ratio = overload / stability

        adjust = clamp(overload_ratio * 0.05, -0.2, 0.2)

        base_calm = clamp(baseline_risk + adjust, 0.2, 0.6)
        base_balanced = clamp(base_calm + 0.2, 0.4, 0.9)
        base_aggressive = clamp(base_balanced + 0.1, 0.7, 1.0)

        if self.current_meta_state == "Hyper-Flow":
            base_calm = clamp(base_calm + 0.05, 0.2, 0.7)
            base_balanced = clamp(base_balanced + 0.05, 0.4, 0.95)
        elif self.current_meta_state == "Deep-Dream":
            base_calm = clamp(base_calm - 0.05, 0.1, 0.5)
            base_balanced = clamp(base_balanced - 0.05, 0.3, 0.8)
        elif self.current_meta_state == "Recovery-Flow":
            base_calm = clamp(base_calm - 0.1, 0.1, 0.5)
            base_balanced = clamp(base_balanced - 0.1, 0.3, 0.8)

        self.stance_thresholds["Calm"] = base_calm
        self.stance_thresholds["Balanced"] = base_balanced
        self.stance_thresholds["Aggressive"] = clamp(base_balanced + 0.1, 0.6, 1.0)

    def _choose_stance(self, best_guess):
        if best_guess < self.stance_thresholds["Calm"]:
            return "Aggressive"
        elif best_guess < self.stance_thresholds["Balanced"]:
            return "Balanced"
        else:
            return "Calm"

    def _select_meta_state(self, best_guess, variance, trend, turbulence):
        if best_guess < 0.3 and variance < 0.02:
            return "Deep-Dream"
        if 0.3 <= best_guess <= 0.6 and variance < 0.03 and trend >= 0:
            return "Hyper-Flow"
        if best_guess > 0.7:
            if trend < 0 or turbulence > 0.6:
                return "Recovery-Flow"
            else:
                return "Sentinel"
        return "Sentinel"

    def _apply_meta_state_effects(self, best_guess):
        appetite_base = 0.5
        horizon_bias = "neutral"
        thread_exp = "normal"
        cache_beh = "normal"

        if self.current_meta_state == "Hyper-Flow":
            self.deep_ram.appetite = clamp(appetite_base + 0.2, 0.1, 1.0)
            horizon_bias = "favor-long (120s) for sustained load"
            thread_exp = "aggressive"
            cache_beh = "hot-cache"
        elif self.current_meta_state == "Deep-Dream":
            self.deep_ram.appetite = clamp(appetite_base - 0.2, 0.1, 1.0)
            horizon_bias = "favor-mid (30s) for gentle waves"
            thread_exp = "slow"
            cache_beh = "background-optim"
        elif self.current_meta_state == "Recovery-Flow":
            self.deep_ram.appetite = clamp(appetite_base - 0.3, 0.1, 1.0)
            horizon_bias = "favor-short (1-5s) for damping spikes"
            thread_exp = "minimal"
            cache_beh = "cool-down"
        else:
            self.deep_ram.appetite = appetite_base
            horizon_bias = "neutral"
            thread_exp = "normal"
            cache_beh = "normal"

        self.meta_effects = {
            "pred_horizon_bias": horizon_bias,
            "ram_appetite": f"{self.deep_ram.appetite:.2f}",
            "thread_expansion": thread_exp,
            "cache_behavior": cache_beh,
        }

    def _predictive_dampening(self, best_guess):
        actions = []
        if best_guess > 0.7:
            old_appetite = self.deep_ram.appetite
            self.deep_ram.appetite = clamp(self.deep_ram.appetite - 0.1, 0.1, 1.0)
            actions.append(f"Shrink DeepRAM appetite {old_appetite:.2f} -> {self.deep_ram.appetite:.2f}")
            self.backup.save_snapshot(tag="pre-risk")
            actions.append("Pre-risk snapshot saved")

        if best_guess > 0.5:
            actions.append("Limit ingestion & thread expansion (conceptual hook)")

        return actions

    def _apply_micro_recovery_loops(self):
        if self.thermal.temp_c and self.thermal.temp_c > 80:
            self.thermal.micro_cool_mode = True
            self.deep_ram.appetite = clamp(self.deep_ram.appetite - 0.1, 0.1, 1.0)
        else:
            self.thermal.micro_cool_mode = False

        if self.disk.metrics.get("total_bps", 0.0) > 10**8:
            self.disk.micro_io_relief = True
        else:
            self.disk.micro_io_relief = False

        if self.vram.vram_used_pct > 90:
            self.vram.cooldown_mode = True
        else:
            self.vram.cooldown_mode = False

    def _estimate_model_integrity(self):
        length_factor = clamp(len(self.risk_history) / 60.0, 0.0, 1.0)
        variance = self._compute_variance(self.risk_history)
        var_factor = 1.0 - clamp(variance * 2.0, 0.0, 1.0)
        conf_factor = self.meta_confidence
        integrity = clamp(0.3 * length_factor + 0.3 * var_factor + 0.4 * conf_factor, 0.0, 1.0)
        return integrity

    def _fuse_horizon(self, ewma_val, mv_val):
        if ewma_val is None and mv_val is None:
            return 0.0
        if ewma_val is None:
            return mv_val
        if mv_val is None:
            return ewma_val
        w_mv = clamp(self.mv_model_conf, 0.2, 0.8)
        w_ewma = 1.0 - w_mv
        return clamp(w_ewma * ewma_val + w_mv * mv_val, 0.0, 1.0)

    def _append_reasoning(self, avg_risk, best_guess, variance, trend,
                          turbulence, tags, actions, engine_contribs):
        contrib_str = ";".join(
            f"{k}:v{v:.2f}*w{w:.2f}"
            for k, (v, w) in engine_contribs.items()
        )
        smb_state = "on" if self.smb_state_dir and self.smb_state_ok else ("cfg" if self.smb_state_dir else "off")
        msg = (
            f"regime={self.current_regime}, risk={avg_risk:.2f}, best={best_guess:.2f}, "
            f"var={variance:.3f}, trend={trend:.3f}, turb={turbulence:.2f}, "
            f"stance={self.current_stance}, meta={self.current_meta_state}, "
            f"SMB={smb_state}, tags={tags}, actions={actions}, mv_conf={self.mv_model_conf:.2f}, "
            f"engines[{contrib_str}]"
        )
        self.reasoning_tail.append(msg)

    def _auto_calibration(self):
        now = now_ts()
        if now - self.last_auto_calibration_ts < 3600:
            return
        self.last_auto_calibration_ts = now

        total = sum(self.reinforcement_buckets.values()) or 1
        stability_ratio = self.reinforcement_buckets["stability"] / total
        overload_ratio = self.reinforcement_buckets["overload"] / total

        delta = clamp((stability_ratio - overload_ratio) * 0.05, -0.05, 0.05)
        self.stance_thresholds["Calm"] = clamp(self.stance_thresholds["Calm"] + delta, 0.2, 0.7)
        self.stance_thresholds["Balanced"] = clamp(self.stance_thresholds["Balanced"] + delta, 0.3, 0.9)
        self.stance_thresholds["Aggressive"] = clamp(self.stance_thresholds["Aggressive"] + delta, 0.5, 1.0)

        self.reasoning_tail.append(
            f"auto-calibration: delta={delta:.3f}, stability={stability_ratio:.2f}, overload={overload_ratio:.2f}"
        )

    # ---------- Public snapshot ----------

    def snapshot(self, best_guess=None, fingerprint_tags=None, actions=None,
                 pred_1s=None, pred_5s=None, pred_30s=None, pred_120s=None):
        if best_guess is None:
            best_guess = self.best_guess_risk
        if fingerprint_tags is None:
            fingerprint_tags = []
        if actions is None:
            actions = []

        p1 = pred_1s if pred_1s is not None else self.ewma_1s
        p5 = pred_5s if pred_5s is not None else self.ewma_5s
        p30 = pred_30s if pred_30s is not None else self.ewma_30s
        p120 = pred_120s if pred_120s is not None else self.ewma_120s

        return {
            "health_score": self.health_score,
            "risk_score": self.risk_score,
            "collective_health_score": self.collective_health_score,
            "meta_confidence": self.meta_confidence,
            "model_integrity": self.model_integrity,
            "meta_state": self.current_meta_state,
            "stance_mode": self.current_stance,
            "stance_thresholds": dict(self.stance_thresholds),
            "predicted_risk_1s": p1,
            "predicted_risk_5s": p5,
            "predicted_risk_30s": p30,
            "predicted_risk_120s": p120,
            "best_guess_risk": best_guess,
            "regime": self.current_regime,
            "baseline_by_hour": dict(self.baseline_by_hour),
            "fingerprint_tags": fingerprint_tags,
            "recent_actions": actions,
            "reasoning_tail": list(self.reasoning_tail),
            "meta_effects": dict(self.meta_effects),
            "mv_model_conf": self.mv_model_conf,
        }


# ----------------------------
# BrainCortexPanel (GUI)
# ----------------------------

class BrainCortexPanel(ttk.Frame):
    def __init__(self, master, hybrid_brain, organs, backup_organ, integrity_organ, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.brain = hybrid_brain
        self.organs = organs
        self.backup_organ = backup_organ
        self.integrity_organ = integrity_organ

        self.pred_history = deque(maxlen=150)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=2)

        self.health_var = tk.StringVar()
        self.risk_var = tk.StringVar()
        self.best_guess_var = tk.StringVar()
        self.meta_conf_var = tk.StringVar()
        self.integrity_var = tk.StringVar()
        self.collective_var = tk.StringVar()

        ttk.Label(top, text="Health:").pack(side="left")
        ttk.Label(top, textvariable=self.health_var, width=6).pack(side="left", padx=(0, 6))

        ttk.Label(top, text="Risk:").pack(side="left")
        ttk.Label(top, textvariable=self.risk_var, width=6).pack(side="left", padx=(0, 6))

        ttk.Label(top, text="Best:").pack(side="left")
        ttk.Label(top, textvariable=self.best_guess_var, width=6).pack(side="left", padx=(0, 6))

        ttk.Label(top, text="Meta-Conf:").pack(side="left")
        ttk.Label(top, textvariable=self.meta_conf_var, width=6).pack(side="left", padx=(0, 6))

        ttk.Label(top, text="Integrity:").pack(side="left")
        ttk.Label(top, textvariable=self.integrity_var, width=6).pack(side="left", padx=(0, 6))

        ttk.Label(top, text="Collective:").pack(side="left")
        ttk.Label(top, textvariable=self.collective_var, width=6).pack(side="left", padx=(0, 6))

        meta_row = ttk.Frame(self)
        meta_row.pack(fill="x", padx=4, pady=2)

        ttk.Label(meta_row, text="Meta-State:").pack(side="left")
        self.meta_state_var = tk.StringVar(value=DEFAULT_META_STATE)
        self.meta_state_combo = ttk.Combobox(meta_row, textvariable=self.meta_state_var,
                                             values=META_STATES, width=14, state="readonly")
        self.meta_state_combo.pack(side="left", padx=(0, 10))

        ttk.Label(meta_row, text="Stance:").pack(side="left")
        self.stance_var = tk.StringVar(value=DEFAULT_STANCE)
        self.stance_combo = ttk.Combobox(meta_row, textvariable=self.stance_var,
                                         values=STANCE_MODES, width=10, state="readonly")
        self.stance_combo.pack(side="left", padx=(0, 10))

        ttk.Label(meta_row, text="Regime:").pack(side="left")
        self.regime_var = tk.StringVar(value="unknown")
        ttk.Label(meta_row, textvariable=self.regime_var, width=10).pack(side="left", padx=(0, 10))

        self.btn_apply_meta = ttk.Button(meta_row, text="Apply overrides",
                                         command=self._apply_overrides)
        self.btn_apply_meta.pack(side="left", padx=(4, 4))

        self.btn_export_json = ttk.Button(meta_row, text="Export JSON snapshot",
                                          command=self._export_json)
        self.btn_export_json.pack(side="left", padx=(4, 4))

        onnx_row = ttk.Frame(self)
        onnx_row.pack(fill="x", padx=4, pady=2)

        ttk.Label(onnx_row, text="ONNX model:").pack(side="left")
        self.onnx_path_var = tk.StringVar(value="")
        self.onnx_entry = ttk.Entry(onnx_row, textvariable=self.onnx_path_var, width=40)
        self.onnx_entry.pack(side="left", padx=(0, 4))

        self.btn_browse_onnx = ttk.Button(onnx_row, text="Browse", command=self._browse_onnx)
        self.btn_browse_onnx.pack(side="left", padx=(2, 2))

        self.btn_reload_onnx = ttk.Button(onnx_row, text="Reload model",
                                          command=self._reload_onnx_model)
        self.btn_reload_onnx.pack(side="left", padx=(2, 2))

        # Memory / SMB paths
        mem_frame = ttk.Frame(self)
        mem_frame.pack(fill="x", padx=4, pady=2)

        ttk.Label(mem_frame, text="Local memory path:").pack(side="left")
        self.memory_path_var = tk.StringVar(value=STATE_DIR)
        self.memory_entry = ttk.Entry(mem_frame, textvariable=self.memory_path_var, width=40)
        self.memory_entry.pack(side="left", padx=(0, 4))

        self.btn_add_backup_path = ttk.Button(mem_frame, text="Add backup path",
                                              command=self._add_backup_path)
        self.btn_add_backup_path.pack(side="left", padx=(4, 4))

        self.btn_save_now = ttk.Button(mem_frame, text="Save snapshot now",
                                       command=self._save_snapshot_now)
        self.btn_save_now.pack(side="left", padx=(4, 4))

        smb_frame = ttk.Frame(self)
        smb_frame.pack(fill="x", padx=4, pady=2)

        ttk.Label(smb_frame, text="SMB memory path:").pack(side="left")
        self.smb_path_var = tk.StringVar(value="")
        self.smb_entry = ttk.Entry(smb_frame, textvariable=self.smb_path_var, width=40)
        self.smb_entry.pack(side="left", padx=(0, 4))

        self.btn_set_smb = ttk.Button(smb_frame, text="Set SMB path",
                                      command=self._set_smb_path)
        self.btn_set_smb.pack(side="left", padx=(4, 4))

        # Altered States row
        alt_frame = ttk.Frame(self)
        alt_frame.pack(fill="x", padx=4, pady=2)

        self.alt_bias_var = tk.StringVar()
        self.alt_appetite_var = tk.StringVar()
        self.alt_thread_var = tk.StringVar()
        self.alt_cache_var = tk.StringVar()

        ttk.Label(alt_frame, text="Altered States: bias").pack(side="left")
        ttk.Label(alt_frame, textvariable=self.alt_bias_var, width=24).pack(side="left")
        ttk.Label(alt_frame, text="RAM appetite").pack(side="left")
        ttk.Label(alt_frame, textvariable=self.alt_appetite_var, width=6).pack(side="left")
        ttk.Label(alt_frame, text="threads").pack(side="left")
        ttk.Label(alt_frame, textvariable=self.alt_thread_var, width=10).pack(side="left")
        ttk.Label(alt_frame, text="cache").pack(side="left")
        ttk.Label(alt_frame, textvariable=self.alt_cache_var, width=12).pack(side="left")

        # Thresholds
        thresh_frame = ttk.Frame(self)
        thresh_frame.pack(fill="x", padx=4, pady=2)

        self.thresh_calm_var = tk.StringVar()
        self.thresh_bal_var = tk.StringVar()
        self.thresh_aggr_var = tk.StringVar()

        ttk.Label(thresh_frame, text="Thresholds: Calm").pack(side="left")
        ttk.Label(thresh_frame, textvariable=self.thresh_calm_var, width=6).pack(side="left")
        ttk.Label(thresh_frame, text="Balanced").pack(side="left")
        ttk.Label(thresh_frame, textvariable=self.thresh_bal_var, width=6).pack(side="left")
        ttk.Label(thresh_frame, text="Aggressive").pack(side="left")
        ttk.Label(thresh_frame, textvariable=self.thresh_aggr_var, width=6).pack(side="left")

        # Organs status
        organ_frame = ttk.LabelFrame(self, text="Organs status")
        organ_frame.pack(fill="x", padx=4, pady=4)

        self.organ_status_vars = {}
        self.organ_spark_vars = {}
        for name in [
            "DeepRamOrgan", "BackupEngineOrgan", "NetworkWatcherOrgan",
            "GPUCacheOrgan", "ThermalOrgan", "DiskOrgan", "VRAMOrgan",
            "SwarmNodeOrgan", "AICoachOrgan", "SelfIntegrityOrgan"
        ]:
            var = tk.StringVar()
            svar = tk.StringVar()
            self.organ_status_vars[name] = var
            self.organ_spark_vars[name] = svar
            sub = ttk.Frame(organ_frame)
            sub.pack(side="left", padx=4)
            ttk.Label(sub, text=name.replace("Organ", "")).pack(side="top")
            ttk.Label(sub, textvariable=var, width=11).pack(side="top")
            ttk.Label(sub, textvariable=svar, width=11, foreground="#00aa88").pack(side="top")

        # Prediction graph + reasoning
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=4, pady=4)

        graph_frame = ttk.LabelFrame(body, text="Best-guess risk graph")
        graph_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.graph_canvas = tk.Canvas(graph_frame, width=320, height=160, bg="black")
        self.graph_canvas.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(body, text="Reasoning tail")
        log_frame.pack(side="left", fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=12, wrap="word")
        self.log_text.pack(fill="both", expand=True)

    # ---- Controls ----

    def _apply_overrides(self):
        self.brain.current_meta_state = self.meta_state_var.get()
        self.brain.current_stance = self.stance_var.get()
        self._append_log(f"[override] meta_state={self.brain.current_meta_state}, stance={self.brain.current_stance}")

    def _add_backup_path(self):
        path = self.memory_path_var.get().strip()
        if path:
            self.backup_organ.add_path(path)
            self._append_log(f"[backup] added path: {path}")

    def _save_snapshot_now(self):
        self.backup_organ.save_snapshot(tag="manual")
        self.brain.save_state()
        self._append_log("[backup] manual snapshot & brain state saved")

    def _set_smb_path(self):
        path = self.smb_path_var.get().strip()
        if path:
            self.brain.set_smb_state_dir(path)
            self._append_log(f"[smb] SMB memory path set to: {path}")
            self.brain.save_state()

    def _export_json(self):
        if not filedialog:
            self._append_log("[export] tkinter filedialog unavailable")
            return
        brain_snap = self.brain.snapshot()
        organs_snap = {name: o.snapshot() for name, o in self.organs.items()}
        merged = {
            "timestamp": datetime.now().isoformat(),
            "brain": brain_snap,
            "organs": organs_snap,
        }
        filename = filedialog.asksaveasfilename(
            title="Export brain snapshot JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, "w") as f:
                    json.dump(merged, f, indent=2)
                self._append_log(f"[export] JSON snapshot saved: {filename}")
            except Exception as e:
                self._append_log(f"[export] error: {e}")

    def _browse_onnx(self):
        if not filedialog:
            self._append_log("[onnx] tkinter filedialog unavailable")
            return
        filename = filedialog.askopenfilename(
            title="Select ONNX model",
            filetypes=[("ONNX models", "*.onnx"), ("All files", "*.*")]
        )
        if filename:
            self.onnx_path_var.set(filename)

    def _reload_onnx_model(self):
        path = self.onnx_path_var.get().strip()
        self.brain.movidius_engine.set_model_path(path)
        if self.brain.movidius_engine.use_onnx:
            self._append_log(f"[onnx] model loaded: {path}")
        else:
            self._append_log(f"[onnx] failed to load or ONNXRuntime unavailable: {path}")

    # ---- UI updates ----

    def _append_log(self, line):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"{ts} {line}\n")
        self.log_text.see("end")

    def update_view(self, brain_snapshot, organs_snapshot):
        self.health_var.set(f"{brain_snapshot['health_score']:.2f}")
        self.risk_var.set(f"{brain_snapshot['risk_score']:.2f}")
        self.best_guess_var.set(f"{brain_snapshot['best_guess_risk']:.2f}")
        self.meta_conf_var.set(f"{brain_snapshot['meta_confidence']:.2f}")
        self.integrity_var.set(f"{brain_snapshot['model_integrity']:.2f}")
        self.collective_var.set(f"{brain_snapshot['collective_health_score']:.2f}")

        self.meta_state_var.set(brain_snapshot["meta_state"])
        self.stance_var.set(brain_snapshot["stance_mode"])
        self.regime_var.set(brain_snapshot.get("regime", "unknown"))

        th = brain_snapshot["stance_thresholds"]
        self.thresh_calm_var.set(f"{th['Calm']:.2f}")
        self.thresh_bal_var.set(f"{th['Balanced']:.2f}")
        self.thresh_aggr_var.set(f"{th['Aggressive']:.2f}")

        eff = brain_snapshot.get("meta_effects", {})
        self.alt_bias_var.set(eff.get("pred_horizon_bias", ""))
        self.alt_appetite_var.set(eff.get("ram_appetite", ""))
        self.alt_thread_var.set(eff.get("thread_expansion", ""))
        self.alt_cache_var.set(eff.get("cache_behavior", ""))

        for name, snap in organs_snapshot.items():
            if name in self.organ_status_vars:
                h = snap["health"]
                r = snap["risk"]
                self.organ_status_vars[name].set(f"H{h:.2f}/R{r:.2f}")
                bar_len = 8
                h_len = int(h * bar_len)
                r_len = int(r * bar_len)
                spark = "H:" + "#" * h_len + "." * (bar_len - h_len) + " R:" + "#" * r_len + "." * (bar_len - r_len)
                self.organ_spark_vars[name].set(spark)

        self.log_text.delete("1.0", "end")
        for line in brain_snapshot["reasoning_tail"][-25:]:
            self._append_log(line)

        best = brain_snapshot.get("best_guess_risk") or 0.0
        self.pred_history.append(best)
        self._draw_graph()

    def _draw_graph(self):
        self.graph_canvas.delete("all")
        w = int(self.graph_canvas.winfo_width() or 320)
        h = int(self.graph_canvas.winfo_height() or 160)
        if len(self.pred_history) < 2:
            return

        self.graph_canvas.create_line(0, h - 1, w, h - 1, fill="#444")
        self.graph_canvas.create_line(0, 0, 0, h, fill="#444")

        xs = list(range(len(self.pred_history)))
        max_x = max(xs) or 1
        scale_x = w / max_x
        prev = None
        for i, val in enumerate(self.pred_history):
            x = i * scale_x
            y = h - (val * h)
            if prev is not None:
                self.graph_canvas.create_line(prev[0], prev[1], x, y, fill="#00ff88", width=2)
            prev = (x, y)


# ----------------------------
# Nerve Center main loop
# ----------------------------

class NerveCenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System Nerve Center — Hybrid Brain Cortex (Evolved + SMB)")

        self.deep_ram = DeepRamOrgan()
        self.backup = BackupEngineOrgan()
        self.network = NetworkWatcherOrgan()
        self.gpu = GPUCacheOrgan()
        self.thermal = ThermalOrgan()
        self.disk = DiskOrgan()
        self.vram = VRAMOrgan()
        self.ai_coach = AICoachOrgan()
        self.swarm = SwarmNodeOrgan()
        self.integrity = SelfIntegrityOrgan()
        self.game_analyzer = Back4BloodAnalyzer()

        self.organs = {
            "DeepRamOrgan": self.deep_ram,
            "BackupEngineOrgan": self.backup,
            "NetworkWatcherOrgan": self.network,
            "GPUCacheOrgan": self.gpu,
            "ThermalOrgan": self.thermal,
            "DiskOrgan": self.disk,
            "VRAMOrgan": self.vram,
            "AICoachOrgan": self.ai_coach,
            "SwarmNodeOrgan": self.swarm,
            "SelfIntegrityOrgan": self.integrity,
        }

        self.hybrid_brain = HybridBrain(
            self.deep_ram, self.backup, self.network,
            self.gpu, self.thermal, self.disk, self.vram,
            self.ai_coach, self.swarm, self.integrity,
            movidius_available=True,
            movidius_model_path=None,
            smb_state_dir=None   # configured via GUI
        )

        self._build_gui()
        self._schedule_tick()

    def _build_gui(self):
        status_frame = ttk.LabelFrame(self.root, text="Autoloader status")
        status_frame.pack(fill="x", padx=4, pady=4)
        text = ", ".join(f"{k}: {v}" for k, v in AUTOLOADER_STATUS.items())
        ttk.Label(status_frame, text=text, wraplength=900, justify="left").pack(fill="x", padx=4, pady=2)

        self.brain_panel = BrainCortexPanel(self.root, self.hybrid_brain, self.organs, self.backup, self.integrity)
        self.brain_panel.pack(fill="both", expand=True, padx=4, pady=4)

    def _schedule_tick(self):
        self.root.after(1000, self._tick)

    def _tick(self):
        for organ in self.organs.values():
            try:
                if isinstance(organ, AICoachOrgan) or isinstance(organ, SelfIntegrityOrgan):
                    continue
                organ.update()
            except Exception as e:
                organ.reason_tail.append(f"update error: {e}")

        organs_snap_simple = {name: o.snapshot() for name, o in self.organs.items()}
        self.game_analyzer.update_with_organs(organs_snap_simple)

        self.hybrid_brain.update(self.organs)

        brain_snap = self.hybrid_brain.snapshot()
        organs_snap = {name: o.snapshot() for name, o in self.organs.items()}

        self.brain_panel.update_view(brain_snap, organs_snap)

        self._schedule_tick()


def main():
    if tk is None:
        print("tkinter is not available; GUI cannot be started.")
        sys.exit(1)

    root = tk.Tk()
    app = NerveCenterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

