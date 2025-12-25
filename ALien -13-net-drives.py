#!/usr/bin/env python3
"""
MagicBox Guardian Brain
Autonomous + Deborah Queen + Borg Probes + Manual Drives on Top
+ Level-3 Organism Persistence
+ Multi-Horizon Predictive Foresight
+ ResourceForecastingOrgan (RAM / CPU / Disk / Drives / Probe Fail Rate)
+ Situational Awareness Cortex (judgment, confidence, predictive, health)
+ Self-Tuning Hybrid Brain (per-risk adaptive weighting)
+ Targeted Risk Signature (what kind of trouble)
+ Cortex Memory (history across reboots)
+ GameAwarenessOrgan (observes game context + bot-like patterns via local telemetry)

SAFE SCOPE FOR GAME AWARENESS:
- Observes local process/telemetry only
- Does NOT control games, bots, accounts, or send inputs
- Uses game context to make the guardian smarter and more predictive
"""

import sys
import subprocess
import threading
import time
import random
import string
import os
import platform
import shutil
import errno
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ============================================
# Auto-loader
# ============================================

REQUIRED_MODULES = [
    "tkinter",
    "psutil",
]

def soft_autoload(cb=print):
    for mod in REQUIRED_MODULES:
        if mod == "tkinter":
            try:
                __import__(mod)
            except ImportError:
                cb("[AutoLoader] tkinter not available on this system.")
            continue

        try:
            __import__(mod)
        except ImportError:
            cb(f"[AutoLoader] Installing missing module: {mod}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", mod])
                cb(f"[AutoLoader] Module {mod} installed.")
            except Exception as e:
                cb(f"[AutoLoader] Failed to install {mod}: {e}")


def hard_autoload(cb=print):
    cb("[AutoLoader] Running deep repair...")
    soft_autoload(cb)
    cb("[AutoLoader] Deep repair complete.")


import psutil
import tkinter as tk
from tkinter import ttk, filedialog

# ============================================
# Utility functions
# ============================================

def adaptive_mutation(mode: str):
    return f"mutation({mode})"


def generate_decoy() -> str:
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def compliance_auditor(items: List[str]) -> str:
    return f"{len(items)} items ✅ compliant"


def reverse_mirror_encrypt(s: str) -> str:
    return "".join(chr((ord(c) + 3) % 126) for _ in s[::-1])


def camouflage(s: str, style: str) -> str:
    return f"{style}:{s}:{style}"


def random_glyph_stream(n: int = 32) -> str:
    glyphs = "⇔⇐⇒∆∑πλΩΨΦ※✶✹✪✦"
    return "".join(random.choice(glyphs) for _ in range(n))


# ============================================
# Brain structures
# ============================================

@dataclass
class Event:
    timestamp: float
    source: str          # "guardian", "rogue", "system", "game"
    level: str           # "info", "warn", "alert", "critical"
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainSnapshot:
    mode: str
    recent_entropy: int
    rogue_trust: float
    guardian_pulse: float
    anomaly_rate: float
    prediction_accuracy: float
    last_judgment: str
    last_confidence: float
    health_score: float
    trend: str           # "improving", "stable", "degrading"

    # predictive health/judgment (multi-horizon)
    predicted_health_short: float
    predicted_judgment_short: str
    predicted_health_medium: float
    predicted_judgment_medium: str

    # resource forecast flags / ETAs (seconds, or None)
    ram_eta: Optional[float]
    cpu_eta: Optional[float]
    disk_eta: Optional[float]
    drive_risk: bool
    probe_failure_risk: bool

    # game risk
    game_risk_score: float
    game_context: str


@dataclass
class CortexView:
    mode: str
    judgment: str
    confidence: float
    health_score: float
    trend: str
    anomaly_rate: float
    rogue_trust: float
    entropy: int
    predicted_judgment_short: str
    predicted_judgment_medium: str
    predicted_health_short: float
    predicted_health_medium: float


class HybridBrain:
    """
    HybridBrain:
    - Integrates system, queen, probe, and game-awareness signals
    - Produces judgment, confidence, health, and predictive foresight
    - Self-tunes both global and per-risk weights
    """

    def __init__(self, max_events: int = 500):
        self.events: deque[Event] = deque(maxlen=max_events)
        self.mode: str = "guardian"
        self.rogue_trust_history: List[float] = []
        self.prediction_hits: int = 0
        self.prediction_total: int = 0
        self._last_snapshot: Optional[BrainSnapshot] = None

        self.weight_anomaly = 1.7
        self.weight_mode = 0.3
        self.weight_rogue_trust = 0.3
        self.weight_prediction_bonus = 35.0
        self.weight_confidence_volume = 1.0

        # per-risk multipliers
        self.risk_weight_ram = 1.0
        self.risk_weight_cpu = 1.0
        self.risk_weight_disk = 1.0
        self.risk_weight_drive = 1.0
        self.risk_weight_probe = 1.0
        self.risk_weight_game = 1.0

        self.health_history: deque[Tuple[float, float]] = deque(maxlen=60)
        self.anomaly_history: deque[Tuple[float, float]] = deque(maxlen=60)

        self.resource_forecasts: Dict[str, Any] = {
            "ram_eta": None,
            "cpu_eta": None,
            "disk_eta": None,
            "drive_risk": False,
            "probe_failure_risk": False,
        }

        # latest game risk summary (fed by GameAwarenessOrgan)
        self.game_risk_score: float = 0.0
        self.game_context: str = "no game"

        # self-tuning tracking
        self.false_calm = 0
        self.false_alarm = 0
        self.good_catch = 0
        self.last_health_for_outcome = None
        self.last_judgment_for_outcome = None
        self.last_judgment_time = None

        # cortex history
        self.cortex_history: deque[CortexView] = deque(maxlen=200)

    # --- situational awareness ---

    def update_mode(self, mode: str):
        self.mode = mode

    def log_event(self, source: str, level: str, message: str, **metadata):
        self.events.append(Event(
            timestamp=time.time(),
            source=source,
            level=level,
            message=message,
            metadata=metadata
        ))

    def record_rogue_trust(self, trust_score: float):
        self.rogue_trust_history.append(trust_score)

    def record_prediction_result(self, hit: bool):
        self.prediction_total += 1
        if hit:
            self.prediction_hits += 1

    # --- anomaly math ---

    def _events_in_window(self, window_sec: float = 60.0) -> List[Event]:
        now = time.time()
        return [e for e in self.events if now - e.timestamp <= window_sec]

    def _estimate_anomaly_rate(self, window_sec: float = 60.0) -> float:
        recent = self._events_in_window(window_sec)
        if not recent:
            return 0.0
        weight_map = {"info": 0.0, "warn": 0.5, "alert": 1.0, "critical": 1.5}
        total = len(recent)
        weighted = sum(weight_map.get(e.level, 0.0) for e in recent)
        return weighted / total

    def _prediction_accuracy(self) -> float:
        if self.prediction_total == 0:
            return 0.5
        return self.prediction_hits / self.prediction_total

    # --- judgment + confidence ---

    def judge(self) -> tuple[str, float]:
        anomaly = self._estimate_anomaly_rate()
        acc = self._prediction_accuracy()
        rogue_trust = self.rogue_trust_history[-1] if self.rogue_trust_history else 0.0

        mode_factor = self.weight_mode if self.mode == "rogue" else 0.0
        score = anomaly * self.weight_anomaly + mode_factor - rogue_trust * self.weight_rogue_trust

        rf = self.resource_forecasts
        if rf.get("ram_eta") is not None and rf["ram_eta"] < 60:
            score += 0.2 * self.risk_weight_ram
        if rf.get("disk_eta") is not None and rf["disk_eta"] < 120:
            score += 0.2 * self.risk_weight_disk
        if rf.get("cpu_eta") is not None and rf["cpu_eta"] < 60:
            score += 0.15 * self.risk_weight_cpu
        if rf.get("drive_risk"):
            score += 0.25 * self.risk_weight_drive
        if rf.get("probe_failure_risk"):
            score += 0.2 * self.risk_weight_probe

        # include game risk
        score += self.game_risk_score * 0.2 * self.risk_weight_game

        if score < 0.2:
            label = "calm"
        elif score < 0.5:
            label = "watchful"
        elif score < 0.9:
            label = "elevated"
        else:
            label = "critical"

        volume_factor = min(len(self.events) / (150.0 / max(self.weight_confidence_volume, 0.1)), 1.0)
        consistency_factor = abs(acc - 0.5) * 2.0
        confidence = 0.15 + 0.45 * volume_factor + 0.4 * consistency_factor
        confidence = max(0.0, min(1.0, confidence))
        return label, confidence

    # --- collective health ---

    def compute_health(self) -> float:
        anomaly = self._estimate_anomaly_rate()
        acc = self._prediction_accuracy()
        rogue_trust = self.rogue_trust_history[-1] if self.rogue_trust_history else 0.0

        health = 100.0
        health -= anomaly * 55.0
        health -= max(0.0, rogue_trust) * 8.0
        if self.mode == "rogue":
            health -= 5.0
        health += (acc - 0.5) * self.weight_prediction_bonus

        rf = self.resource_forecasts
        if rf.get("ram_eta") is not None and rf["ram_eta"] < 60:
            health -= 5.0 * self.risk_weight_ram
        if rf.get("disk_eta") is not None and rf["disk_eta"] < 120:
            health -= 5.0 * self.risk_weight_disk
        if rf.get("cpu_eta") is not None and rf["cpu_eta"] < 60:
            health -= 4.0 * self.risk_weight_cpu
        if rf.get("drive_risk"):
            health -= 4.0 * self.risk_weight_drive
        if rf.get("probe_failure_risk"):
            health -= 4.0 * self.risk_weight_probe

        # game risk can also reduce health (intense match + shaky system)
        health -= min(self.game_risk_score * 10.0 * self.risk_weight_game, 15.0)

        return max(0.0, min(100.0, health))

    # --- slope helper ---

    @staticmethod
    def _estimate_slope(history: deque[Tuple[float, float]], min_points: int = 4) -> float:
        if len(history) < min_points:
            return 0.0
        points = list(history)[-min(10, len(history)):]
        t0, v0 = points[0]
        tn, vn = points[-1]
        dt = max(tn - t0, 1.0)
        return (vn - v0) / dt

    # --- predictive health ---

    def _predict_health_multi(self, current_health: float, anomaly: float) -> Tuple[float, float]:
        now = time.time()
        self.health_history.append((now, current_health))
        self.anomaly_history.append((now, anomaly))

        health_slope = self._estimate_slope(self.health_history)
        anomaly_slope = self._estimate_slope(self.anomaly_history)

        short_horizon = 30.0
        medium_horizon = 120.0

        short = current_health + health_slope * short_horizon - anomaly_slope * 25.0
        medium = current_health + health_slope * medium_horizon - anomaly_slope * 40.0

        short = max(0.0, min(100.0, short))
        medium = max(0.0, min(100.0, medium))
        return short, medium

    def _judgment_from_health(self, health: float) -> str:
        if health > 80:
            return "calm"
        elif health > 60:
            return "watchful"
        elif health > 35:
            return "elevated"
        else:
            return "critical"

    # --- targeted risk signature ---

    def targeted_risk_signature(self) -> str:
        rf = self.resource_forecasts
        parts = []

        if rf.get("ram_eta") is not None and rf["ram_eta"] < 120:
            parts.append("RAM pressure")
        if rf.get("cpu_eta") is not None and rf["cpu_eta"] < 120:
            parts.append("CPU saturation")
        if rf.get("disk_eta") is not None and rf["disk_eta"] < 300:
            parts.append("Disk capacity")
        if rf.get("drive_risk"):
            parts.append("Drive instability")
        if rf.get("probe_failure_risk"):
            parts.append("Probe failure rate")
        if self.game_risk_score > 0.3:
            parts.append(f"Game risk ({self.game_context})")

        recent = list(self._events_in_window(60.0))
        if any("memory" in e.message.lower() for e in recent):
            parts.append("Memory-related anomalies")
        if any("disk" in e.message.lower() for e in recent):
            parts.append("Disk-related anomalies")
        if any(e.source == "rogue" for e in recent):
            parts.append("Rogue pattern escalation")

        if not parts:
            return "No dominant risk vector"
        seen = set()
        uniq = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return ", ".join(uniq)

    # --- self-tuning ---

    def self_tune(self, log_cb):
        if self._last_snapshot is None or self.last_judgment_for_outcome is None:
            return

        health = self._last_snapshot.health_score
        j = self.last_judgment_for_outcome
        bad = health < 40
        good = health > 70

        # coarse anomaly/mode tuning
        if j == "calm" and bad:
            self.false_calm += 1
            delta = 0.05
            self.weight_anomaly += delta
            self.weight_mode += delta * 0.5
            log_cb(f"[BRAIN] Self-tune: false CALM, anomaly -> {self.weight_anomaly:.2f}")
        elif j in ("elevated", "critical") and good:
            self.false_alarm += 1
            delta = 0.05
            self.weight_anomaly = max(0.1, self.weight_anomaly - delta)
            self.weight_mode = max(0.0, self.weight_mode - delta * 0.5)
            log_cb(f"[BRAIN] Self-tune: false alarm, anomaly -> {self.weight_anomaly:.2f}")
        elif j in ("elevated", "critical") and bad:
            self.good_catch += 1
            self.weight_mode = min(1.0, self.weight_mode + 0.02)
            log_cb(f"[BRAIN] Self-tune: good catch, mode penalty -> {self.weight_mode:.2f}")

        # per-risk tuning via risk signature
        risk_sig = self.targeted_risk_signature().lower()

        def adjust(attr: str, token: str, label: str):
            cur = getattr(self, attr)
            if token in risk_sig:
                if bad and j in ("calm", "watchful"):
                    new = min(cur + 0.05, 2.0)
                    setattr(self, attr, new)
                    log_cb(f"[BRAIN] Risk-tune: {label} weight -> {new:.2f}")
                elif good and j in ("elevated", "critical"):
                    new = max(cur - 0.05, 0.3)
                    setattr(self, attr, new)
                    log_cb(f"[BRAIN] Risk-tune: {label} weight -> {new:.2f}")

        adjust("risk_weight_ram", "ram", "RAM")
        adjust("risk_weight_cpu", "cpu", "CPU")
        adjust("risk_weight_disk", "disk", "Disk")
        adjust("risk_weight_drive", "drive", "Drive")
        adjust("risk_weight_probe", "probe", "Probe")
        adjust("risk_weight_game", "game risk", "Game")

    # --- snapshot ---

    def snapshot(self) -> BrainSnapshot:
        anomaly = self._estimate_anomaly_rate()
        acc = self._prediction_accuracy()
        rogue_trust = self.rogue_trust_history[-1] if self.rogue_trust_history else 0.0
        judgment, conf = self.judge()
        health = self.compute_health()

        if self._last_snapshot is None:
            trend = "stable"
        else:
            if health > self._last_snapshot.health_score + 2:
                trend = "improving"
            elif health < self._last_snapshot.health_score - 2:
                trend = "degrading"
            else:
                trend = "stable"

        if self.events:
            last_entropy = int(self.events[-1].metadata.get("entropy", 0))
        else:
            last_entropy = 0

        ph_short, ph_medium = self._predict_health_multi(health, anomaly)
        pj_short = self._judgment_from_health(ph_short)
        pj_medium = self._judgment_from_health(ph_medium)

        rf = self.resource_forecasts

        snap = BrainSnapshot(
            mode=self.mode,
            recent_entropy=last_entropy,
            rogue_trust=rogue_trust,
            guardian_pulse=1.0 if self.mode == "guardian" else 0.5,
            anomaly_rate=anomaly,
            prediction_accuracy=acc,
            last_judgment=judgment,
            last_confidence=conf,
            health_score=health,
            trend=trend,
            predicted_health_short=ph_short,
            predicted_judgment_short=pj_short,
            predicted_health_medium=ph_medium,
            predicted_judgment_medium=pj_medium,
            ram_eta=rf.get("ram_eta"),
            cpu_eta=rf.get("cpu_eta"),
            disk_eta=rf.get("disk_eta"),
            drive_risk=rf.get("drive_risk", False),
            probe_failure_risk=rf.get("probe_failure_risk", False),
            game_risk_score=self.game_risk_score,
            game_context=self.game_context,
        )

        now = time.time()
        self.last_health_for_outcome = health
        self.last_judgment_for_outcome = judgment
        self.last_judgment_time = now

        cv = CortexView(
            mode=snap.mode,
            judgment=snap.last_judgment,
            confidence=snap.last_confidence,
            health_score=snap.health_score,
            trend=snap.trend,
            anomaly_rate=snap.anomaly_rate,
            rogue_trust=snap.rogue_trust,
            entropy=snap.recent_entropy,
            predicted_judgment_short=snap.predicted_judgment_short,
            predicted_judgment_medium=snap.predicted_judgment_medium,
            predicted_health_short=snap.predicted_health_short,
            predicted_health_medium=snap.predicted_health_medium,
        )
        self.cortex_history.append(cv)

        self._last_snapshot = snap
        return snap

    def cortex_view(self) -> CortexView:
        snap = self.snapshot()
        return CortexView(
            mode=snap.mode,
            judgment=snap.last_judgment,
            confidence=snap.last_confidence,
            health_score=snap.health_score,
            trend=snap.trend,
            anomaly_rate=snap.anomaly_rate,
            rogue_trust=snap.rogue_trust,
            entropy=snap.recent_entropy,
            predicted_judgment_short=snap.predicted_judgment_short,
            predicted_judgment_medium=snap.predicted_judgment_medium,
            predicted_health_short=snap.predicted_health_short,
            predicted_health_medium=snap.predicted_health_medium,
        )


# ============================================
# SessionStats
# ============================================

class SessionStats:
    def __init__(self):
        self.start_time = time.time()
        self.current_mode = "guardian"
        self.last_mode_switch = time.time()

        self.guardian_time = 0.0
        self.rogue_time = 0.0
        self.guardian_cycles = 0
        self.rogue_cycles = 0

        self.rogue_trust_sum = 0.0
        self.rogue_trust_count = 0

        self.anomaly_sum = 0.0
        self.anomaly_count = 0

    def switch_mode(self, new_mode: str):
        now = time.time()
        elapsed = now - self.last_mode_switch
        if self.current_mode == "guardian":
            self.guardian_time += elapsed
        else:
            self.rogue_time += elapsed
        self.current_mode = new_mode
        self.last_mode_switch = now

    def tick_cycle(self, mode: str):
        if mode == "guardian":
            self.guardian_cycles += 1
        else:
            self.rogue_cycles += 1

    def record_rogue_trust(self, trust: float):
        self.rogue_trust_sum += trust
        self.rogue_trust_count += 1

    def record_anomaly(self, anomaly: float):
        self.anomaly_sum += anomaly
        self.anomaly_count += 1

    def uptime(self) -> float:
        return time.time() - self.start_time

    def avg_rogue_trust(self) -> float:
        if self.rogue_trust_count == 0:
            return 0.0
        return self.rogue_trust_sum / self.rogue_trust_count

    def avg_anomaly(self) -> float:
        if self.anomaly_count == 0:
            return 0.0
        return self.anomaly_sum / self.anomaly_count


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


# ============================================
# SystemInventory
# ============================================

class SystemInventory:
    def __init__(self):
        self.last_snapshot = None

    def snapshot(self) -> dict:
        info = {}

        info["system"] = {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        total, used, free = shutil.disk_usage(os.path.abspath(os.sep))
        info["disk"] = {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "used_percent": (used / total) * 100.0 if total else 0.0,
        }

        vm = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": vm.total / (1024**3),
            "used_gb": vm.used / (1024**3),
            "used_percent": vm.percent,
        }

        cpu = psutil.cpu_percent(interval=0.0)
        info["cpu"] = {
            "used_percent": cpu,
        }

        processes = []
        for p in psutil.process_iter(["pid", "name", "username", "cpu_percent", "memory_percent"]):
            try:
                processes.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        processes.sort(key=lambda x: x.get("cpu_percent", 0.0), reverse=True)
        info["top_processes"] = processes[:20]

        self.last_snapshot = info
        return info

    def summarize(self, snapshot: dict) -> str:
        mem = snapshot["memory"]["used_percent"]
        disk = snapshot["disk"]["used_percent"]
        cpu = snapshot["cpu"]["used_percent"]
        return f"CPU {cpu:.1f}% | MEM {mem:.1f}% | DISK {disk:.1f}% | TOP {len(snapshot['top_processes'])} proc"


# ============================================
# ResourceForecastingOrgan
# ============================================

class ResourceForecastingOrgan:
    def __init__(self, brain: HybridBrain, log_cb):
        self.brain = brain
        self.log = log_cb

        self.ram_history: deque[Tuple[float, float]] = deque(maxlen=60)
        self.cpu_history: deque[Tuple[float, float]] = deque(maxlen=60)
        self.disk_history: deque[Tuple[float, float]] = deque(maxlen=60)

        self.ram_threshold = 90.0
        self.cpu_threshold = 95.0
        self.disk_threshold = 90.0

        self.drive_risk_flag = False
        self.probe_failure_risk_flag = False

        self.probe_check_count = 0
        self.probe_failure_count = 0

    @staticmethod
    def _eta_to_threshold(history: deque[Tuple[float, float]], threshold: float) -> Optional[float]:
        if len(history) < 4:
            return None
        points = list(history)[-min(10, len(history)):]
        t0, v0 = points[0]
        tn, vn = points[-1]
        dt = max(tn - t0, 1.0)
        dv = vn - v0
        slope = dv / dt

        if slope <= 0:
            return None
        if vn >= threshold:
            return 0.0

        remaining = threshold - vn
        eta = remaining / slope
        if eta < 0:
            return None
        return eta

    def ingest_inventory(self, snapshot: dict):
        now = time.time()
        ram = snapshot["memory"]["used_percent"]
        cpu = snapshot["cpu"]["used_percent"]
        disk = snapshot["disk"]["used_percent"]

        self.ram_history.append((now, ram))
        self.cpu_history.append((now, cpu))
        self.disk_history.append((now, disk))

        ram_eta = self._eta_to_threshold(self.ram_history, self.ram_threshold)
        cpu_eta = self._eta_to_threshold(self.cpu_history, self.cpu_threshold)
        disk_eta = self._eta_to_threshold(self.disk_history, self.disk_threshold)

        if ram_eta is not None and ram_eta < 120:
            self.log(f"[FORECAST] RAM will cross {self.ram_threshold:.0f}% in ~{ram_eta:.0f}s")
        if cpu_eta is not None and cpu_eta < 120:
            self.log(f"[FORECAST] CPU will cross {self.cpu_threshold:.0f}% in ~{cpu_eta:.0f}s")
        if disk_eta is not None and disk_eta < 300:
            self.log(f"[FORECAST] Disk will cross {self.disk_threshold:.0f}% in ~{disk_eta:.0f}s")

        self.brain.resource_forecasts["ram_eta"] = ram_eta
        self.brain.resource_forecasts["cpu_eta"] = cpu_eta
        self.brain.resource_forecasts["disk_eta"] = disk_eta

        self.brain.resource_forecasts["drive_risk"] = self.drive_risk_flag
        self.brain.resource_forecasts["probe_failure_risk"] = self.probe_failure_risk_flag

    def note_drive_status(self, status: Dict[str, bool]):
        if not status:
            self.drive_risk_flag = False
            return
        total = len(status)
        fails = sum(1 for v in status.values() if not v)
        ratio = fails / total
        self.drive_risk_flag = ratio >= 0.4

        if self.drive_risk_flag:
            self.log(f"[FORECAST] Drive instability detected ({fails}/{total} failing/flaky)")

        self.brain.resource_forecasts["drive_risk"] = self.drive_risk_flag

    def note_probe_check(self, ok: bool):
        self.probe_check_count += 1
        if not ok:
            self.probe_failure_count += 1

        if self.probe_check_count >= 10:
            ratio = self.probe_failure_count / self.probe_check_count
            self.probe_failure_risk_flag = ratio >= 0.3
            if self.probe_failure_risk_flag:
                self.log(f"[FORECAST] Probe failure rate high ({ratio*100:.0f}%)")
            self.brain.resource_forecasts["probe_failure_risk"] = self.probe_failure_risk_flag
            self.probe_check_count = 0
            self.probe_failure_count = 0


# ============================================
# DeborahQueen
# ============================================

class DeborahQueen:
    def __init__(self, brain: HybridBrain, log_cb, resource_forecaster: ResourceForecastingOrgan):
        self.brain = brain
        self.log = log_cb
        self.forecaster = resource_forecaster

        self.queen_drive: Optional[str] = None
        self.worker_drives: List[str] = []

        self.last_inventory = None
        self.last_status = "unknown"

        self.max_mem_percent = 90.0
        self.max_disk_percent = 90.0

        self.drive_last_status: Dict[str, bool] = {}
        self.drive_last_ok_time: Dict[str, float] = {}

    def set_queen_drive(self, path: str):
        self.queen_drive = path
        self.log(f"[QUEEN] Queen drive set to: {path}")
        self.brain.log_event("system", "info", "Queen drive set", queen_drive=path)

    def set_worker_drives(self, drives: List[str]):
        self.worker_drives = drives[:]

    def _check_backup_path(self, path: str) -> bool:
        now = time.time()
        last_ok = self.drive_last_ok_time.get(path, 0)

        try:
            if not os.path.exists(path):
                if now - last_ok < 120:
                    self.log(f"[QUEEN] Path {path} temporarily missing, using last-known-good = True")
                    return True
                return False

            test_file = os.path.join(path, ".queen_probe.tmp")
            try:
                with open(test_file, "w") as f:
                    f.write("ok")
                os.remove(test_file)
                self.drive_last_status[path] = True
                self.drive_last_ok_time[path] = now
                return True
            except OSError as e:
                self.drive_last_status[path] = False
                if e.errno in (errno.EACCES, errno.EPERM):
                    self.log(f"[QUEEN] No write permission on backup path: {path}")
                elif e.errno in (errno.ENOENT, errno.ENODEV):
                    self.log(f"[QUEEN] Backup path unreachable: {path}")
                else:
                    self.log(f"[QUEEN] Error probing backup path {path}: {e}")
                return False

        except Exception as e:
            self.log(f"[QUEEN] Unexpected error checking backup path {path}: {e}")
            if path in self.drive_last_status:
                self.log(f"[QUEEN] Using last-known status for {path}: {self.drive_last_status[path]}")
                return self.drive_last_status[path]
            return False

    def verify_all_drives(self) -> Dict[str, bool]:
        paths = []
        if self.queen_drive:
            paths.append(self.queen_drive)
        paths.extend(self.worker_drives)

        status = {}
        for p in paths:
            ok = self._check_backup_path(p)
            status[p] = ok

        if not paths:
            self.brain.log_event("system", "warn", "No backup drives configured")
        else:
            level = "info" if any(status.values()) else "critical"
            self.brain.log_event("system", level, "Backup drives status", **status)

        self.forecaster.note_drive_status(status)

        return status

    def replicate_to_all(self):
        paths = []
        if self.queen_drive:
            paths.append(self.queen_drive)
        paths.extend(self.worker_drives)

        if not paths:
            self.log("[QUEEN] No drives configured for replication.")
            return

        self.log(f"[QUEEN] Full mirror replication pass to {len(paths)} drives.")
        self.brain.log_event("system", "info", "Replication pass", drives=len(paths))

    def assess_inventory(self, snapshot: dict):
        mem = snapshot["memory"]["used_percent"]
        disk = snapshot["disk"]["used_percent"]

        level = "info"
        msg_parts = []

        if mem > self.max_mem_percent:
            level = "alert"
            msg_parts.append(f"High memory usage {mem:.1f}%")
        if disk > self.max_disk_percent:
            level = "alert"
            msg_parts.append(f"High disk usage {disk:.1f}%")

        if not msg_parts:
            msg = "Cave scan clean."
            self.last_status = "clean"
        else:
            msg = "; ".join(msg_parts)
            self.last_status = "suspicious"

        self.brain.log_event("system", level, msg, mem=mem, disk=disk)
        self.log(f"[QUEEN] {msg}")

        self.last_inventory = snapshot


# ============================================
# BorgWorkerProbe
# ============================================

class BorgWorkerProbe(threading.Thread):
    def __init__(self, queen: DeborahQueen, inventory: SystemInventory,
                 forecaster: ResourceForecastingOrgan, log_cb, interval_sec: int = 30):
        super().__init__(daemon=True)
        self.queen = queen
        self.inventory = inventory
        self.forecaster = forecaster
        self.interval_sec = interval_sec
        self.running = True
        self._counter = 0
        self.log = log_cb

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                snapshot = self.inventory.snapshot()
                self.queen.assess_inventory(snapshot)

                self.forecaster.ingest_inventory(snapshot)

                summary = self.inventory.summarize(snapshot)
                self.log(f"[PROBE] {summary}")

                self._counter += 1

                if self._counter % 5 == 0:
                    status = self.queen.verify_all_drives()
                    ok_count = sum(1 for v in status.values() if v)
                    self.log(f"[PROBE] Backup drive status: {ok_count}/{len(status)} OK")
                    self.forecaster.note_probe_check(ok_count == len(status))

                if self._counter % 10 == 0:
                    self.queen.replicate_to_all()
            except Exception as e:
                self.log(f"[PROBE] Worker loop error (recovered): {e}")

            time.sleep(self.interval_sec)


# ============================================
# DualPersonalityBot
# ============================================

class DualPersonalityBot:
    def __init__(self, cb, brain: HybridBrain, stats: SessionStats):
        self.cb = cb
        self.run = False
        self.mode = "guardian"
        self.rogue_weights = [0.2, -0.4, 0.7]
        self.rogue_log = []
        self.brain = brain
        self.stats = stats

        self.loop_interval_sec = 5

    def switch_mode(self):
        new_mode = "rogue" if self.mode == "guardian" else "guardian"
        self.stats.switch_mode(new_mode)
        self.mode = new_mode
        self.brain.update_mode(self.mode)
        msg = f"Personality switched to {self.mode.upper()}"
        self.cb(msg)
        self.brain.log_event("system", "info", msg)

    def guardian_behavior(self):
        adaptive_mutation("guardian")
        decoy = generate_decoy()
        compliance = compliance_auditor([decoy])

        self.cb(f"Guardian audit: {decoy}")
        self.cb(f"Compliance: {compliance}")

        self.brain.log_event(
            "guardian",
            "info",
            "Guardian audit cycle",
            decoy=decoy,
            compliance=compliance
        )
        self.stats.tick_cycle("guardian")

    def rogue_behavior(self):
        entropy = int(time.time()) % 2048
        scrambled = reverse_mirror_encrypt(str(entropy))
        camo = camouflage(str(entropy), "alien")
        glyph_stream = random_glyph_stream()
        unusual_pattern = f"{scrambled[:16]}-{camo}-{glyph_stream[:8]}"

        self.rogue_weights = [
            w + (entropy % 5 - 2) * 0.01 for w in self.rogue_weights
        ]
        self.rogue_log.append(self.rogue_weights)

        score = sum(self.rogue_weights) / len(self.rogue_weights)

        self.cb("Rogue escalation initiated")
        self.cb(f"Rogue pattern: {unusual_pattern}")
        self.cb(f"Rogue weights: {self.rogue_weights} | Trust {score:.3f}")

        severity = "alert" if score > 0.3 else "warn"
        self.brain.record_rogue_trust(score)
        self.brain.log_event(
            "rogue",
            severity,
            "Rogue cycle",
            entropy=entropy,
            pattern=unusual_pattern,
            weights=self.rogue_weights,
            trust=score
        )
        self.stats.tick_cycle("rogue")
        self.stats.record_rogue_trust(score)

    def start(self):
        if self.run:
            return
        self.run = True
        self.brain.update_mode(self.mode)
        self.stats.switch_mode(self.mode)
        t = threading.Thread(target=self.loop, daemon=True)
        t.start()

    def stop(self):
        self.run = False

    def loop(self):
        while self.run:
            try:
                if self.mode == "guardian":
                    self.guardian_behavior()
                else:
                    self.rogue_behavior()
            except Exception as e:
                self.cb(f"[SYSTEM] Bot loop error (recovered): {e}")
                self.brain.log_event("system", "warn", "Bot loop error", error=str(e))
            time.sleep(self.loop_interval_sec)


# ============================================
# GameAwarenessOrgan
# ============================================

class GameAwarenessOrgan(threading.Thread):
    """
    Observes game context via:
    - Configured game process name (e.g., 'cs2.exe', 'valorant.exe')
    - Optional local telemetry file with bot/human events (JSON lines)

    Does NOT control or send input to the game.
    It:
    - Estimates a game_risk_score (0..1)
    - Emits events to the HybridBrain
    """

    def __init__(self, brain: HybridBrain, log_cb, game_process_name: str = "", telemetry_path: str = ""):
        super().__init__(daemon=True)
        self.brain = brain
        self.log = log_cb
        self.game_process_name = game_process_name.strip()
        self.telemetry_path = telemetry_path.strip()
        self.running = True

        # rolling stats
        self.last_game_seen = 0.0
        self.recent_bot_aggression: deque[float] = deque(maxlen=50)
        self.recent_player_pressure: deque[float] = deque(maxlen=50)

        # telemetry file position
        self._telemetry_offset = 0

    def set_game_process_name(self, name: str):
        self.game_process_name = name.strip()
        self.log(f"[SYSTEM] Game process set: {self.game_process_name or '(none)'}")

    def set_telemetry_path(self, path: str):
        self.telemetry_path = path.strip()
        self._telemetry_offset = 0
        self.log(f"[SYSTEM] Game telemetry path set: {self.telemetry_path or '(none)'}")

    def stop(self):
        self.running = False

    def _check_game_running(self) -> bool:
        if not self.game_process_name:
            return False
        for p in psutil.process_iter(["name"]):
            try:
                if p.info["name"] and p.info["name"].lower() == self.game_process_name.lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _read_telemetry(self):
        if not self.telemetry_path or not os.path.isfile(self.telemetry_path):
            return

        try:
            with open(self.telemetry_path, "r", encoding="utf-8") as f:
                f.seek(self._telemetry_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._handle_game_event(ev)
                self._telemetry_offset = f.tell()
        except Exception as e:
            self.log(f"[SYSTEM] Game telemetry read error: {e}")

    def _handle_game_event(self, ev: dict):
        etype = ev.get("type", "")
        if etype == "bot_action":
            agg = float(ev.get("aggression", 0.0))
            self.recent_bot_aggression.append(agg)
            self.brain.log_event(
                "game", "info", "Bot action observed", aggression=agg, detail=ev.get("detail", "")
            )
        elif etype == "player_action":
            pressure = float(ev.get("pressure", 0.0))
            self.recent_player_pressure.append(pressure)
            self.brain.log_event(
                "game", "info", "Player action observed", pressure=pressure, detail=ev.get("detail", "")
            )
        elif etype == "round_end":
            self.brain.log_event(
                "game", "info", "Round end", result=ev.get("result", "unknown")
            )

    def _estimate_game_risk(self, game_running: bool) -> float:
        if not game_running:
            return 0.0

        if self.recent_bot_aggression:
            avg_bot = sum(self.recent_bot_aggression) / len(self.recent_bot_aggression)
        else:
            avg_bot = 0.0

        if self.recent_player_pressure:
            avg_player = sum(self.recent_player_pressure) / len(self.recent_player_pressure)
        else:
            avg_player = 0.0

        risk = (avg_bot * 0.6 + avg_player * 0.4)
        return max(0.0, min(1.0, risk))

    def run(self):
        while self.running:
            try:
                game_running = self._check_game_running()
                if game_running:
                    self.last_game_seen = time.time()
                    game_context = self.game_process_name or "unknown game"
                else:
                    game_context = "no game"

                self._read_telemetry()

                risk = self._estimate_game_risk(game_running)
                self.brain.game_risk_score = risk
                self.brain.game_context = game_context

                if game_running:
                    self.log(f"[SYSTEM] Game context: {game_context}, risk ~{risk:.2f}")

                time.sleep(5.0)
            except Exception as e:
                self.log(f"[SYSTEM] Game awareness loop error (recovered): {e}")
                time.sleep(5.0)


# ============================================
# StateManager
# ============================================

STATE_FILENAME = "magicbox_state.json"


class StateManager:
    def __init__(self, app, log_cb):
        self.app = app
        self.log = log_cb

    def _get_state_path(self) -> str:
        queen_drive = self.app.queen.queen_drive
        if queen_drive:
            try:
                base = os.path.join(queen_drive, "MagicBoxState")
                os.makedirs(base, exist_ok=True)
                return os.path.join(base, STATE_FILENAME)
            except Exception:
                pass
        return os.path.join(os.getcwd(), STATE_FILENAME)

    def save_state(self):
        try:
            state = self._build_state()
            path = self._get_state_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self.log(f"[SYSTEM] State saved to {path}")
        except Exception as e:
            self.log(f"[SYSTEM] Failed to save state: {e}")

    def load_state(self):
        try:
            path = self._get_state_path()
            if not os.path.isfile(path):
                self.log("[SYSTEM] No previous state found (fresh start).")
                return
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._apply_state(state)
            self.log(f"[SYSTEM] State restored from {path}")
        except Exception as e:
            self.log(f"[SYSTEM] Failed to load state: {e}")

    def _build_state(self) -> dict:
        brain = self.app.brain
        stats = self.app.stats
        bot = self.app.bot
        queen = self.app.queen
        probe = self.app.probe_worker

        events = list(brain.events)[-200:]

        return {
            "version": 1,
            "bot": {
                "mode": bot.mode,
                "rogue_weights": bot.rogue_weights,
            },
            "brain": {
                "events": [
                    {
                        "timestamp": e.timestamp,
                        "source": e.source,
                        "level": e.level,
                        "message": e.message,
                        "metadata": e.metadata,
                    }
                    for e in events
                ],
                "rogue_trust_history": brain.rogue_trust_history[-200:],
                "prediction_hits": brain.prediction_hits,
                "prediction_total": brain.prediction_total,
                "weights": {
                    "anomaly": brain.weight_anomaly,
                    "mode": brain.weight_mode,
                    "rogue_trust": brain.weight_rogue_trust,
                    "prediction_bonus": brain.weight_prediction_bonus,
                    "confidence_volume": brain.weight_confidence_volume,
                    "risk_weight_ram": brain.risk_weight_ram,
                    "risk_weight_cpu": brain.risk_weight_cpu,
                    "risk_weight_disk": brain.risk_weight_disk,
                    "risk_weight_drive": brain.risk_weight_drive,
                    "risk_weight_probe": brain.risk_weight_probe,
                    "risk_weight_game": brain.risk_weight_game,
                },
                "cortex_history": [
                    {
                        "mode": cv.mode,
                        "judgment": cv.judgment,
                        "confidence": cv.confidence,
                        "health_score": cv.health_score,
                        "trend": cv.trend,
                        "anomaly_rate": cv.anomaly_rate,
                        "rogue_trust": cv.rogue_trust,
                        "entropy": cv.entropy,
                        "predicted_judgment_short": cv.predicted_judgment_short,
                        "predicted_judgment_medium": cv.predicted_judgment_medium,
                        "predicted_health_short": cv.predicted_health_short,
                        "predicted_health_medium": cv.predicted_health_medium,
                    }
                    for cv in list(brain.cortex_history)
                ],
            },
            "stats": {
                "guardian_time": stats.guardian_time,
                "rogue_time": stats.rogue_time,
                "guardian_cycles": stats.guardian_cycles,
                "rogue_cycles": stats.rogue_cycles,
                "rogue_trust_sum": stats.rogue_trust_sum,
                "rogue_trust_count": stats.rogue_trust_count,
                "anomaly_sum": stats.anomaly_sum,
                "anomaly_count": stats.anomaly_count,
            },
            "queen": {
                "queen_drive": queen.queen_drive,
                "worker_drives": queen.worker_drives,
                "last_status": queen.last_status,
            },
            "probe": {
                "counter": getattr(probe, "_counter", 0),
            },
            "meta": {
                "saved_at": time.time(),
            },
        }

    def _apply_state(self, state: dict):
        if state.get("version") != 1:
            self.log("[SYSTEM] State version mismatch; using defaults.")
            return

        brain = self.app.brain
        stats = self.app.stats
        bot = self.app.bot
        queen = self.app.queen
        probe = self.app.probe_worker

        bot.mode = state["bot"].get("mode", "guardian")
        bot.rogue_weights = state["bot"].get("rogue_weights", bot.rogue_weights)
        brain.update_mode(bot.mode)

        brain.events.clear()
        for e in state["brain"].get("events", []):
            brain.events.append(Event(
                timestamp=e["timestamp"],
                source=e["source"],
                level=e["level"],
                message=e["message"],
                metadata=e.get("metadata", {}),
            ))
        brain.rogue_trust_history = state["brain"].get("rogue_trust_history", [])
        brain.prediction_hits = state["brain"].get("prediction_hits", 0)
        brain.prediction_total = state["brain"].get("prediction_total", 0)

        w = state["brain"].get("weights", {})
        brain.weight_anomaly = w.get("anomaly", brain.weight_anomaly)
        brain.weight_mode = w.get("mode", brain.weight_mode)
        brain.weight_rogue_trust = w.get("rogue_trust", brain.weight_rogue_trust)
        brain.weight_prediction_bonus = w.get("prediction_bonus", brain.weight_prediction_bonus)
        brain.weight_confidence_volume = w.get("confidence_volume", brain.weight_confidence_volume)
        brain.risk_weight_ram = w.get("risk_weight_ram", brain.risk_weight_ram)
        brain.risk_weight_cpu = w.get("risk_weight_cpu", brain.risk_weight_cpu)
        brain.risk_weight_disk = w.get("risk_weight_disk", brain.risk_weight_disk)
        brain.risk_weight_drive = w.get("risk_weight_drive", brain.risk_weight_drive)
        brain.risk_weight_probe = w.get("risk_weight_probe", brain.risk_weight_probe)
        brain.risk_weight_game = w.get("risk_weight_game", brain.risk_weight_game)

        brain.cortex_history.clear()
        for cv in state["brain"].get("cortex_history", []):
            brain.cortex_history.append(CortexView(
                mode=cv["mode"],
                judgment=cv["judgment"],
                confidence=cv["confidence"],
                health_score=cv["health_score"],
                trend=cv["trend"],
                anomaly_rate=cv["anomaly_rate"],
                rogue_trust=cv["rogue_trust"],
                entropy=cv["entropy"],
                predicted_judgment_short=cv["predicted_judgment_short"],
                predicted_judgment_medium=cv["predicted_judgment_medium"],
                predicted_health_short=cv["predicted_health_short"],
                predicted_health_medium=cv["predicted_health_medium"],
            ))

        s = state["stats"]
        stats.guardian_time = s.get("guardian_time", 0.0)
        stats.rogue_time = s.get("rogue_time", 0.0)
        stats.guardian_cycles = s.get("guardian_cycles", 0)
        stats.rogue_cycles = s.get("rogue_cycles", 0)
        stats.rogue_trust_sum = s.get("rogue_trust_sum", 0.0)
        stats.rogue_trust_count = s.get("rogue_trust_count", 0)
        stats.anomaly_sum = s.get("anomaly_sum", 0.0)
        stats.anomaly_count = s.get("anomaly_count", 0)

        q = state["queen"]
        queen.queen_drive = q.get("queen_drive")
        queen.worker_drives = q.get("worker_drives", [])
        queen.last_status = q.get("last_status", "unknown")

        probe._counter = state["probe"].get("counter", 0)

        self.app._restore_drives_from_state()
        self.app._sync_sliders_from_brain()


# ============================================
# MagicBoxApp (GUI)
# ============================================

class MagicBoxApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicBox Guardian Brain (Predictive + Cortex + Self-Learning + Game-Aware)")
        self.root.geometry("1280x800")

        self.root.configure(bg="#101010")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101010")
        style.configure("TLabel", background="#101010", foreground="#f0f0f0", font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#ffd700")
        style.configure("Value.TLabel", font=("Consolas", 14, "bold"), foreground="#00ff99")
        style.configure("Danger.TLabel", font=("Consolas", 14, "bold"), foreground="#ff5555")
        style.configure("Small.TLabel", font=("Consolas", 10), foreground="#aaaaaa")
        style.configure("TButton", font=("Segoe UI", 11, "bold"))

        self.brain = HybridBrain()
        self.stats = SessionStats()
        self.resource_forecaster = ResourceForecastingOrgan(self.brain, self.append_log)
        self.queen = DeborahQueen(self.brain, self.append_log, self.resource_forecaster)
        self.inventory = SystemInventory()
        self.bot = DualPersonalityBot(self.append_log, self.brain, self.stats)

        self.game_organ = GameAwarenessOrgan(self.brain, self.append_log)
        self.game_organ.start()

        self.probe_worker = BorgWorkerProbe(
            self.queen, self.inventory, self.resource_forecaster, self.append_log, interval_sec=30
        )
        self.probe_worker.start()

        self.worker_drives: List[str] = []
        self.worker_row_widgets: Dict[str, Dict[str, Any]] = {}

        self.last_auto_switch_time = 0.0
        self.auto_switch_cooldown_sec = 30

        self._build_layout()

        self.state_manager = StateManager(self, self.append_log)

        self.append_log("Running soft auto-loader...")
        soft_autoload(self.append_log)

        self.state_manager.load_state()
        self._sync_sliders_from_brain()
        self._restore_drives_from_state()

        self.append_log("[SYSTEM] Auto-guardian initializing...")
        self.bot.start()
        self.append_log("[SYSTEM] Bot auto-started in GUARDIAN mode.")
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])

        self.append_log("[SYSTEM] Borg worker probe started.")
        self.append_log("Ready. Continuous guard and scan active.")

        self._schedule_brain_update()
        self._schedule_state_save()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        # Controls
        header = ttk.Label(left, text="MagicBox Control", style="Header.TLabel")
        header.pack(anchor="w", pady=(0, 10))

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(0, 10))

        self.start_button = ttk.Button(btn_frame, text="START", command=self.on_start_click)
        self.start_button.pack(fill="x", pady=2)

        self.stop_button = ttk.Button(btn_frame, text="STOP", command=self.on_stop_click)
        self.stop_button.state(["disabled"])
        self.stop_button.pack(fill="x", pady=2)

        self.mode_button = ttk.Button(btn_frame, text="Switch to ROGUE", command=self.on_switch_mode)
        self.mode_button.pack(fill="x", pady=2)

        self.repair_button = ttk.Button(btn_frame, text="Deep Repair (AutoLoader)", command=self.on_deep_repair)
        self.repair_button.pack(fill="x", pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        # Drives
        backup_header = ttk.Label(left, text="Backup / Network Drives", style="Header.TLabel")
        backup_header.pack(anchor="w", pady=(0, 5))

        queen_row = ttk.Frame(left)
        queen_row.pack(fill="x", pady=2)
        ttk.Label(queen_row, text="Queen Drive:", style="Small.TLabel").grid(row=0, column=0, sticky="w")
        self.queen_drive_label = ttk.Label(queen_row, text="(not set)", style="Small.TLabel")
        self.queen_drive_label.grid(row=0, column=1, sticky="w")
        self.queen_change_button = ttk.Button(queen_row, text="Change", command=self.on_change_queen_drive)
        self.queen_change_button.grid(row=0, column=2, sticky="e", padx=5)

        worker_header = ttk.Label(left, text="Worker Drives", style="Small.TLabel")
        worker_header.pack(anchor="w", pady=(5, 2))

        header_row = ttk.Frame(left)
        header_row.pack(fill="x")
        ttk.Label(header_row, text="Drive Path", style="Small.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header_row, text="Status", style="Small.TLabel").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(header_row, text="Action", style="Small.TLabel").grid(row=0, column=2, sticky="w")

        self.worker_table_frame = ttk.Frame(left)
        self.worker_table_frame.pack(fill="x", pady=(0, 5))

        self.add_worker_button = ttk.Button(left, text="Add Worker Drive…", command=self.on_add_worker_drive)
        self.add_worker_button.pack(anchor="w", pady=(2, 8))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        # Cortex
        brain_header = ttk.Label(left, text="Situational Awareness Cortex", style="Header.TLabel")
        brain_header.pack(anchor="w", pady=(0, 5))

        self.mode_label = ttk.Label(left, text="Mode: -", style="Value.TLabel")
        self.mode_label.pack(anchor="w", pady=2)

        self.judgment_label = ttk.Label(left, text="Judgment: -", style="Value.TLabel")
        self.judgment_label.pack(anchor="w", pady=2)

        self.pred_judgment_label_short = ttk.Label(left, text="Predicted (short): -", style="Value.TLabel")
        self.pred_judgment_label_short.pack(anchor="w", pady=2)

        self.pred_judgment_label_medium = ttk.Label(left, text="Predicted (medium): -", style="Value.TLabel")
        self.pred_judgment_label_medium.pack(anchor="w", pady=2)

        self.conf_label = ttk.Label(left, text="Confidence: -", style="TLabel")
        self.conf_label.pack(anchor="w", pady=2)

        self.health_label = ttk.Label(left, text="Health: -", style="Value.TLabel")
        self.health_label.pack(anchor="w", pady=2)

        self.pred_health_label_short = ttk.Label(left, text="Health (short): -", style="TLabel")
        self.pred_health_label_short.pack(anchor="w", pady=2)

        self.pred_health_label_medium = ttk.Label(left, text="Health (medium): -", style="TLabel")
        self.pred_health_label_medium.pack(anchor="w", pady=2)

        self.trend_label = ttk.Label(left, text="Trend: -", style="TLabel")
        self.trend_label.pack(anchor="w", pady=2)

        self.entropy_label = ttk.Label(left, text="Entropy: -", style="TLabel")
        self.entropy_label.pack(anchor="w", pady=2)

        self.cortex_verdict_label = ttk.Label(left, text="Cortex Verdict: -", style="Small.TLabel")
        self.cortex_verdict_label.pack(anchor="w", pady=2)

        self.cortex_risk_label = ttk.Label(left, text="Likely Risk Vector: -", style="Small.TLabel")
        self.cortex_risk_label.pack(anchor="w", pady=2)

        # Resource forecast
        resource_header = ttk.Label(left, text="Resource Forecast", style="Header.TLabel")
        resource_header.pack(anchor="w", pady=(10, 5))

        self.ram_eta_label = ttk.Label(left, text="RAM ETA: -", style="Small.TLabel")
        self.ram_eta_label.pack(anchor="w", pady=1)

        self.cpu_eta_label = ttk.Label(left, text="CPU ETA: -", style="Small.TLabel")
        self.cpu_eta_label.pack(anchor="w", pady=1)

        self.disk_eta_label = ttk.Label(left, text="Disk ETA: -", style="Small.TLabel")
        self.disk_eta_label.pack(anchor="w", pady=1)

        self.drive_risk_label = ttk.Label(left, text="Drive Risk: -", style="Small.TLabel")
        self.drive_risk_label.pack(anchor="w", pady=1)

        self.probe_risk_label = ttk.Label(left, text="Probe Failure Risk: -", style="Small.TLabel")
        self.probe_risk_label.pack(anchor="w", pady=1)

        # Game awareness
        game_header = ttk.Label(left, text="Game Awareness", style="Header.TLabel")
        game_header.pack(anchor="w", pady=(10, 5))

        self.game_process_entry = ttk.Entry(left)
        self.game_process_entry.insert(0, "cs2.exe")
        self.game_process_entry.pack(fill="x", pady=1)
        self.game_process_button = ttk.Button(left, text="Set Game Process", command=self.on_set_game_process)
        self.game_process_button.pack(fill="x", pady=1)

        self.game_telemetry_entry = ttk.Entry(left)
        self.game_telemetry_entry.insert(0, "")
        self.game_telemetry_entry.pack(fill="x", pady=1)
        self.game_telemetry_button = ttk.Button(left, text="Set Telemetry Path", command=self.on_set_game_telemetry)
        self.game_telemetry_button.pack(fill="x", pady=1)

        self.game_context_label = ttk.Label(left, text="Game: none", style="Small.TLabel")
        self.game_context_label.pack(anchor="w", pady=1)

        self.game_risk_label = ttk.Label(left, text="Game Risk: 0.00", style="Small.TLabel")
        self.game_risk_label.pack(anchor="w", pady=1)

        # Health bar
        ttk.Label(left, text="Health Gauge", style="TLabel").pack(anchor="w", pady=(10, 2))
        self.health_canvas = tk.Canvas(left, width=260, height=28, bg="#202020", highlightthickness=0)
        self.health_canvas.pack(pady=(0, 10))
        self.health_bar_bg = self.health_canvas.create_rectangle(2, 2, 258, 26, fill="#303030", outline="")
        self.health_bar_fg = self.health_canvas.create_rectangle(2, 2, 2, 26, fill="#00ff99", outline="")
        self.health_bar_text = self.health_canvas.create_text(
            130, 14, text="0 / 100", fill="#ffffff", font=("Segoe UI", 11, "bold")
        )

        # Stats
        stats_header = ttk.Label(left, text="Session Stats", style="Header.TLabel")
        stats_header.pack(anchor="w", pady=(10, 5))

        self.uptime_label = ttk.Label(left, text="Uptime: 00:00:00", style="Small.TLabel")
        self.uptime_label.pack(anchor="w", pady=1)

        self.guardian_time_label = ttk.Label(left, text="Guardian Time: 00:00:00", style="Small.TLabel")
        self.guardian_time_label.pack(anchor="w", pady=1)

        self.rogue_time_label = ttk.Label(left, text="Rogue Time: 00:00:00", style="Small.TLabel")
        self.rogue_time_label.pack(anchor="w", pady=1)

        self.guardian_cycles_label = ttk.Label(left, text="Guardian Cycles: 0", style="Small.TLabel")
        self.guardian_cycles_label.pack(anchor="w", pady=1)

        self.rogue_cycles_label = ttk.Label(left, text="Rogue Cycles: 0", style="Small.TLabel")
        self.rogue_cycles_label.pack(anchor="w", pady=1)

        self.avg_trust_label = ttk.Label(left, text="Avg Rogue Trust: 0.00", style="Small.TLabel")
        self.avg_trust_label.pack(anchor="w", pady=1)

        self.avg_anomaly_label = ttk.Label(left, text="Avg Anomaly: 0.00", style="Small.TLabel")
        self.avg_anomaly_label.pack(anchor="w", pady=1)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        # Tuning
        tuning_header = ttk.Label(left, text="Brain Tuning", style="Header.TLabel")
        tuning_header.pack(anchor="w", pady=(0, 5))

        def slider_row(parent, text, from_, to, init, command):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            lbl = ttk.Label(row, text=text, style="Small.TLabel")
            lbl.pack(anchor="w")
            scale = ttk.Scale(row, from_=from_, to=to, orient="horizontal", command=command)
            scale.set(init)
            scale.pack(fill="x")
            return scale

        self.anomaly_slider = slider_row(
            left, "Anomaly Weight", 0.0, 3.0, self.brain.weight_anomaly, self.on_anomaly_weight_change
        )
        self.rogue_penalty_slider = slider_row(
            left, "Rogue Trust Penalty", 0.0, 2.0, self.brain.weight_rogue_trust, self.on_rogue_penalty_change
        )
        self.mode_penalty_slider = slider_row(
            left, "Mode Penalty", 0.0, 1.0, self.brain.weight_mode, self.on_mode_penalty_change
        )
        self.pred_bonus_slider = slider_row(
            left, "Prediction Bonus", 0.0, 60.0, self.brain.weight_prediction_bonus, self.on_pred_bonus_change
        )
        self.conf_vol_slider = slider_row(
            left, "Confidence Volume Weight", 0.2, 2.0, self.brain.weight_confidence_volume, self.on_conf_vol_change
        )

        # Log
        log_header = ttk.Label(right, text="Event Log", style="Header.TLabel")
        log_header.pack(anchor="w", pady=(0, 5))

        self.log_text = tk.Text(
            right,
            bg="#000000",
            fg="#00ff99",
            insertbackground="#ffffff",
            font=("Consolas", 10),
            state="disabled",
            wrap="word"
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_configure("guardian", foreground="#00ff99")
        self.log_text.tag_configure("rogue", foreground="#ff5555")
        self.log_text.tag_configure("system", foreground="#87cefa")
        self.log_text.tag_configure("queen", foreground="#ffcc00")
        self.log_text.tag_configure("probe", foreground="#00bfff")
        self.log_text.tag_configure("forecast", foreground="#ffb347")

    # --- state helpers ---

    def _sync_sliders_from_brain(self):
        try:
            self.anomaly_slider.set(self.brain.weight_anomaly)
            self.rogue_penalty_slider.set(self.brain.weight_rogue_trust)
            self.mode_penalty_slider.set(self.brain.weight_mode)
            self.pred_bonus_slider.set(self.brain.weight_prediction_bonus)
            self.conf_vol_slider.set(self.brain.weight_confidence_volume)
        except Exception:
            pass

    def _restore_drives_from_state(self):
        if self.queen.queen_drive:
            self.queen_drive_label.config(text=self.queen.queen_drive)
        self.worker_drives = list(self.queen.worker_drives)
        for path in self.worker_drives:
            self._add_worker_row(path)

    # --- GUI actions ---

    def on_start_click(self):
        self.bot.start()
        self.append_log("[SYSTEM] Bot started (manual).")
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])

    def on_stop_click(self):
        self.bot.stop()
        self.append_log("[SYSTEM] Bot stopped.")
        self.start_button.state(["!disabled"])
        self.stop_button.state(["disabled"])

    def on_switch_mode(self):
        prev_mode = self.bot.mode
        self.bot.switch_mode()
        new_mode = self.bot.mode
        self.mode_button.config(
            text="Switch to GUARDIAN" if new_mode == "rogue" else "Switch to ROGUE"
        )
        self.append_log(f"[SYSTEM] Mode changed (manual): {prev_mode} -> {new_mode}")

    def on_deep_repair(self):
        self.append_log("[SYSTEM] Deep repair triggered.")
        threading.Thread(target=lambda: hard_autoload(self.append_log), daemon=True).start()

    def on_anomaly_weight_change(self, val):
        self.brain.weight_anomaly = float(val)

    def on_rogue_penalty_change(self, val):
        self.brain.weight_rogue_trust = float(val)

    def on_mode_penalty_change(self, val):
        self.brain.weight_mode = float(val)

    def on_pred_bonus_change(self, val):
        self.brain.weight_prediction_bonus = float(val)

    def on_conf_vol_change(self, val):
        self.brain.weight_confidence_volume = float(val)

    # Drives

    def on_change_queen_drive(self):
        folder = filedialog.askdirectory(title="Select Queen Backup Drive")
        if not folder:
            return
        self.queen_drive_label.config(text=folder)
        self.queen.set_queen_drive(folder)

    def on_add_worker_drive(self):
        folder = filedialog.askdirectory(title="Select Worker Backup Drive")
        if not folder:
            return
        if folder in self.worker_drives:
            self.append_log(f"[SYSTEM] Worker drive already added: {folder}")
            return
        self.worker_drives.append(folder)
        self.queen.set_worker_drives(self.worker_drives)
        self._add_worker_row(folder)

    def _add_worker_row(self, path: str):
        if path in self.worker_row_widgets:
            return

        row = ttk.Frame(self.worker_table_frame)
        row.pack(fill="x", pady=1)

        path_label = ttk.Label(row, text=path, style="Small.TLabel")
        path_label.grid(row=0, column=0, sticky="w")

        status_label = ttk.Label(row, text="unknown", style="Small.TLabel")
        status_label.grid(row=0, column=1, sticky="w", padx=5)

        remove_button = ttk.Button(row, text="Remove", command=lambda p=path: self.on_remove_worker_drive(p))
        remove_button.grid(row=0, column=2, sticky="e")

        self.worker_row_widgets[path] = {
            "frame": row,
            "path_label": path_label,
            "status_label": status_label,
            "remove_button": remove_button,
        }

    def on_remove_worker_drive(self, path: str):
        if path in self.worker_drives:
            self.worker_drives.remove(path)
        self.queen.set_worker_drives(self.worker_drives)

        widgets = self.worker_row_widgets.pop(path, None)
        if widgets:
            widgets["frame"].destroy()

        self.append_log(f"[SYSTEM] Removed worker drive: {path}")

    def _update_worker_drive_status_labels(self, drive_status: Dict[str, bool]):
        for path, widgets in self.worker_row_widgets.items():
            ok = drive_status.get(path, None)
            if ok is True:
                txt = "OK"
            elif ok is False:
                txt = "FAIL"
            else:
                txt = "unknown"
            widgets["status_label"].config(text=txt)

    # Game awareness controls

    def on_set_game_process(self):
        name = self.game_process_entry.get().strip()
        self.game_organ.set_game_process_name(name)

    def on_set_game_telemetry(self):
        path = self.game_telemetry_entry.get().strip()
        self.game_organ.set_telemetry_path(path)

    # Logging

    def append_log(self, message: str):
        if message.startswith("Guardian"):
            tag = "guardian"
        elif message.startswith("Rogue"):
            tag = "rogue"
        elif message.startswith("[QUEEN]"):
            tag = "queen"
        elif message.startswith("[PROBE]"):
            tag = "probe"
        elif message.startswith("[FORECAST]"):
            tag = "forecast"
        elif message.startswith("[SYSTEM]") or message.startswith("[AutoLoader]"):
            tag = "system"
        else:
            tag = "system"

        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # periodic

    def _schedule_brain_update(self):
        self._update_brain_view()
        self.root.after(1000, self._schedule_brain_update)

    def _schedule_state_save(self):
        self.state_manager.save_state()
        self.root.after(60000, self._schedule_state_save)

    def _on_close(self):
        try:
            self.state_manager.save_state()
        except Exception:
            pass
        try:
            self.game_organ.stop()
        except Exception:
            pass
        try:
            self.probe_worker.stop()
        except Exception:
            pass
        self.root.destroy()

    def _update_brain_view(self):
        snap = self.brain.snapshot()
        self.brain.self_tune(self.append_log)
        self.stats.record_anomaly(snap.anomaly_rate)

        self.mode_label.config(text=f"Mode: {snap.mode.upper()}")

        if snap.last_judgment in ("elevated", "critical"):
            style = "Danger.TLabel"
        else:
            style = "Value.TLabel"
        self.judgment_label.config(text=f"Judgment: {snap.last_judgment}", style=style)

        self.pred_judgment_label_short.config(text=f"Predicted (short): {snap.predicted_judgment_short}")
        self.pred_judgment_label_medium.config(text=f"Predicted (medium): {snap.predicted_judgment_medium}")

        self.conf_label.config(text=f"Confidence: {snap.last_confidence*100:.0f}%")
        self.health_label.config(text=f"Health: {snap.health_score:.0f}/100")
        self.pred_health_label_short.config(text=f"Health (short): {snap.predicted_health_short:.0f}/100")
        self.pred_health_label_medium.config(text=f"Health (medium): {snap.predicted_health_medium:.0f}/100")
        self.trend_label.config(text=f"Trend: {snap.trend}")
        self.entropy_label.config(text=f"Entropy: {snap.recent_entropy}")

        cortex = self.brain.cortex_view()
        verdict = f"{cortex.judgment.upper()} @ {cortex.confidence*100:.0f}% | Health {cortex.health_score:.0f}"
        self.cortex_verdict_label.config(text=f"Cortex Verdict: {verdict}")

        risk_signature = self.brain.targeted_risk_signature()
        self.cortex_risk_label.config(text=f"Likely Risk Vector: {risk_signature}")

        def format_eta(label: str, eta: Optional[float]) -> str:
            if eta is None:
                return f"{label}: stable"
            if eta == 0:
                return f"{label}: crossed"
            return f"{label}: ~{eta:.0f}s"

        self.ram_eta_label.config(text=format_eta("RAM ETA", snap.ram_eta))
        self.cpu_eta_label.config(text=format_eta("CPU ETA", snap.cpu_eta))
        self.disk_eta_label.config(text=format_eta("Disk ETA", snap.disk_eta))

        self.drive_risk_label.config(
            text=f"Drive Risk: {'HIGH' if snap.drive_risk else 'normal'}"
        )
        self.probe_risk_label.config(
            text=f"Probe Failure Risk: {'HIGH' if snap.probe_failure_risk else 'normal'}"
        )

        # Game context
        self.game_context_label.config(text=f"Game: {snap.game_context}")
        self.game_risk_label.config(text=f"Game Risk: {snap.game_risk_score:.2f}")

        self._update_health_bar(snap.health_score)
        self._update_stats_view()
        self._auto_mode_logic(snap)

        drive_status = self.queen.verify_all_drives()
        self._update_worker_drive_status_labels(drive_status)

    def _update_health_bar(self, health: float):
        width = 256
        fill_width = int((health / 100.0) * width)
        fill_width = max(0, min(width, fill_width))

        if health < 30:
            color = "#ff3333"
        elif health < 60:
            color = "#ffaa00"
        else:
            color = "#00cc66"

        self.health_canvas.coords(self.health_bar_fg, 2, 2, 2 + fill_width, 26)
        self.health_canvas.itemconfig(self.health_bar_fg, fill=color)
        self.health_canvas.itemconfig(self.health_bar_text, text=f"{health:.0f} / 100")

    def _update_stats_view(self):
        uptime = self.stats.uptime()

        now = time.time()
        elapsed_since_switch = now - self.stats.last_mode_switch
        guardian_time = self.stats.guardian_time
        rogue_time = self.stats.rogue_time
        if self.stats.current_mode == "guardian":
            guardian_time += elapsed_since_switch
        else:
            rogue_time += elapsed_since_switch

        self.uptime_label.config(text=f"Uptime: {format_duration(uptime)}")
        self.guardian_time_label.config(text=f"Guardian Time: {format_duration(guardian_time)}")
        self.rogue_time_label.config(text=f"Rogue Time: {format_duration(rogue_time)}")

        self.guardian_cycles_label.config(text=f"Guardian Cycles: {self.stats.guardian_cycles}")
        self.rogue_cycles_label.config(text=f"Rogue Cycles: {self.stats.rogue_cycles}")

        self.avg_trust_label.config(text=f"Avg Rogue Trust: {self.stats.avg_rogue_trust():.2f}")
        self.avg_anomaly_label.config(text=f"Avg Anomaly: {self.stats.avg_anomaly():.2f}")

    def _auto_mode_logic(self, snap: BrainSnapshot):
        now = time.time()
        if now - self.last_auto_switch_time < self.auto_switch_cooldown_sec:
            return

        high_risk_now = (
            snap.last_judgment in ("elevated", "critical") or snap.health_score < 40
        )

        high_risk_short = (
            snap.predicted_judgment_short in ("elevated", "critical")
            or snap.predicted_health_short < 45
        )

        high_risk_medium = (
            snap.predicted_judgment_medium in ("elevated", "critical")
            or snap.predicted_health_medium < 50
        )

        resource_risk = False
        if snap.ram_eta is not None and snap.ram_eta < 60:
            resource_risk = True
        if snap.cpu_eta is not None and snap.cpu_eta < 60:
            resource_risk = True
        if snap.disk_eta is not None and snap.disk_eta < 120:
            resource_risk = True
        if snap.drive_risk or snap.probe_failure_risk:
            resource_risk = True

        game_risk_flag = snap.game_risk_score > 0.5

        go_rogue = (
            snap.mode == "guardian"
            and (high_risk_now or high_risk_short or high_risk_medium or resource_risk or game_risk_flag)
        )

        safe_now = (snap.last_judgment == "calm" and snap.health_score > 70)
        safe_short = (
            snap.predicted_judgment_short in ("calm", "watchful")
            and snap.predicted_health_short > 65
        )
        safe_medium = (
            snap.predicted_judgment_medium in ("calm", "watchful")
            and snap.predicted_health_medium > 65
        )

        safe_resources = True
        if snap.ram_eta is not None and snap.ram_eta < 120:
            safe_resources = False
        if snap.cpu_eta is not None and snap.cpu_eta < 120:
            safe_resources = False
        if snap.disk_eta is not None and snap.disk_eta < 240:
            safe_resources = False
        if snap.drive_risk or snap.probe_failure_risk:
            safe_resources = False
        if snap.game_risk_score > 0.4:
            safe_resources = False  # treat high game risk as unsafe for return

        go_guardian = (
            snap.mode == "rogue"
            and safe_now and safe_short and safe_medium and safe_resources
        )

        if go_rogue:
            prev_mode = self.bot.mode
            self.bot.switch_mode()
            self.mode_button.config(text="Switch to GUARDIAN")
            self.append_log(
                f"[SYSTEM] Auto-switch: {prev_mode.upper()} -> ROGUE (multi-horizon/resource/game risk)"
            )
            self.last_auto_switch_time = now

        elif go_guardian:
            prev_mode = self.bot.mode
            self.bot.switch_mode()
            self.mode_button.config(text="Switch to ROGUE")
            self.append_log(
                f"[SYSTEM] Auto-switch: {prev_mode.upper()} -> GUARDIAN (all horizons/resources/game stable)"
            )
            self.last_auto_switch_time = now


# ============================================
# Main
# ============================================

def main():
    root = tk.Tk()
    app = MagicBoxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

