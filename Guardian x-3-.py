#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MagicBox ASI Defense Console
HybridBrain + ReplicaNPU + Organs + BorgMesh + MirrorDefense
Real-time only, modular, single-file scaffold.
"""

import os
import sys
import platform
import threading
import queue
import time
import math
import random
import json
import datetime
import signal
import subprocess

# ============================================================
# Auto-loader for required libraries
# ============================================================

REQUIRED_LIBS = [
    "psutil",
    "pyttsx3",
    "watchdog",
    "onnxruntime",
    "pynvml",
    "PyQt5",
]

def autoload_libs():
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            except Exception:
                pass

autoload_libs()

import psutil
import pyttsx3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

try:
    from PyQt5 import QtWidgets, QtGui, QtCore
except Exception:
    QtWidgets = None

import tkinter as tk
from tkinter import ttk, filedialog

# ============================================================
# Elevation / Interrupt Shield
# ============================================================

import ctypes

def ensure_admin():
    if platform.system() == "Windows":
        try:
            if not ctypes.windll.shell32.IsUserAnAdmin():
                script = os.path.abspath(sys.argv[0])
                params = " ".join([f'"{a}"' for a in sys.argv[1:]])
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, f'"{script}" {params}', None, 1
                )
                sys.exit()
        except Exception:
            pass

def block_interrupt(signum, frame):
    print("\n[MagicBox] Interrupt blocked. Defense shell remains active.")

signal.signal(signal.SIGINT, block_interrupt)
signal.signal(signal.SIGTERM, block_interrupt)

ensure_admin()

# ============================================================
# Voice Engine (pyttsx3)
# ============================================================

def init_voice():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        # Pick a stable, neutral voice if available
        for v in voices:
            if "Zira" in v.id or "David" in v.id:
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 180)
        return engine
    except Exception:
        return None

VOICE_ENGINE = init_voice()

def speak(text):
    if VOICE_ENGINE is None:
        return
    def _run():
        try:
            VOICE_ENGINE.say(text)
            VOICE_ENGINE.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# ReplicaNPU (predictive heads + integrity)
# ============================================================

from collections import deque

class ReplicaNPU:
    def __init__(self, cores=8, frequency_ghz=1.2, memory_size=16,
                 plasticity_decay=0.0005, integrity_threshold=0.4):
        self.cores = cores
        self.frequency_ghz = frequency_ghz
        self.cycles = 0
        self.energy = 0.0
        self.memory = deque(maxlen=memory_size)
        self.plasticity = 1.0
        self.plasticity_decay = plasticity_decay
        self.integrity_threshold = integrity_threshold
        self.model_integrity = 1.0
        self.frozen = False
        self.heads = {}
        self.symbolic_bias = {}
        self.instruction_queue = deque()

    def schedule(self, fn, *args):
        self.instruction_queue.append((fn, args))

    def tick(self, budget=64):
        executed = 0
        while self.instruction_queue and executed < budget:
            fn, args = self.instruction_queue.popleft()
            fn(*args)
            executed += 1
        self.plasticity = max(0.1, self.plasticity - self.plasticity_decay)

    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        assert len(v1) == len(v2)
        chunk = math.ceil(len(v1) / self.cores)
        acc = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial
        return acc

    def add_head(self, name, input_dim, lr=0.01, risk=1.0):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "history": deque(maxlen=32),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"]
        y += self._symbolic_modulation(name)
        head["history"].append(y)
        self.memory.append(y)
        return y

    def predict(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head(head, x, name)
        return preds

    def learn(self, x, targets):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            head = self.heads[name]
            pred = self._predict_head(head, x, name)
            error = target - pred
            weighted_error = error * head["risk"] * self.plasticity * self.model_integrity
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
        return errors

    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return max(0.0, min(1.0, 1.0 - var))

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {k: round(self.confidence(k), 3) for k in self.heads},
        }

# ============================================================
# Organs (real metrics where possible)
# ============================================================

class BaseOrgan:
    def __init__(self, name):
        self.name = name
        self.health = 1.0
        self.risk = 0.0
        self.last_metrics = {}
    def update(self):
        pass
    def micro_recovery(self):
        self.health = min(1.0, self.health + 0.01)
        self.risk = max(0.0, self.risk - 0.01)

class DeepRamOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("DeepRAM")
    def update(self):
        vm = psutil.virtual_memory()
        used_ratio = vm.percent / 100.0
        self.risk = used_ratio
        self.health = 1.0 - used_ratio
        self.last_metrics = {"used_ratio": used_ratio}

class NetworkWatcherOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Network")
        self._last_bytes = psutil.net_io_counters()
    def update(self):
        now = psutil.net_io_counters()
        sent = now.bytes_sent - self._last_bytes.bytes_sent
        recv = now.bytes_recv - self._last_bytes.bytes_recv
        self._last_bytes = now
        throughput = (sent + recv) / 1024.0
        self.risk = min(1.0, throughput / 1024.0)
        self.health = 1.0 - self.risk
        self.last_metrics = {"throughput_kb": throughput}

class ThermalOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Thermal")
    def update(self):
        temp = 40.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for arr in temps.values():
                    if arr:
                        temp = arr[0].current
                        break
        except Exception:
            pass
        norm = min(1.0, max(0.0, (temp - 30.0) / 50.0))
        self.risk = norm
        self.health = 1.0 - norm
        self.last_metrics = {"temp_c": temp}

class DiskOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Disk")
        self._last = psutil.disk_io_counters()
    def update(self):
        now = psutil.disk_io_counters()
        delta = (now.read_bytes - self._last.read_bytes) + (now.write_bytes - self._last.write_bytes)
        self._last = now
        mb = delta / (1024.0 * 1024.0)
        self.risk = min(1.0, mb / 256.0)
        self.health = 1.0 - self.risk
        self.last_metrics = {"io_mb": mb}

class VRAMOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("VRAM")
    def update(self):
        used_ratio = 0.0
        if pynvml:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                used_ratio = info.used / float(info.total)
            except Exception:
                used_ratio = 0.0
        self.risk = used_ratio
        self.health = 1.0 - used_ratio
        self.last_metrics = {"used_ratio": used_ratio}

class GPUCacheOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("GPUCache")
    def update(self):
        self.risk = 0.2
        self.health = 0.8
        self.last_metrics = {"cache_pressure": 0.2}

class BackupEngineOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Backup")
    def update(self):
        self.risk = 0.1
        self.health = 0.9
        self.last_metrics = {"snapshots": 1}

class AICoachOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("AICoach")
    def update(self):
        self.risk = 0.1
        self.health = 0.9
        self.last_metrics = {"coaching": 1}

class SwarmNodeOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Swarm")
    def update(self):
        self.risk = 0.2
        self.health = 0.8
        self.last_metrics = {"nodes": 1}

class SelfIntegrityOrgan(BaseOrgan):
    def __init__(self):
        super().__init__("Integrity")
    def check_integrity(self, brain, organs):
        missing = [o for o in organs if o.health <= 0.0]
        drift = 0.0
        self.risk = min(1.0, drift + len(missing) * 0.2)
        self.health = 1.0 - self.risk
        brain.npu.check_integrity(1.0 - self.risk)
        self.last_metrics = {"missing_organs": len(missing)}

# ============================================================
# Movidius / ONNX inference stub
# ============================================================

class MovidiusInferenceEngine:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.session = None
        if ort and model_path and os.path.isfile(model_path):
            try:
                self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None

    def predict(self, features):
        if not self.session:
            # Fallback: simple heuristic
            risk = min(1.0, max(0.0, sum(features) / (len(features) * 2.0)))
            conf = 0.6
            return risk, conf
        # Real ONNX path (placeholder)
        input_name = self.session.get_inputs()[0].name
        out = self.session.run(None, {input_name: [features]})[0][0]
        risk = float(out[0])
        conf = float(out[1]) if len(out) > 1 else 0.7
        return risk, conf

# ============================================================
# HybridBrain + Tri-Stance Decision Engine
# ============================================================

class DecisionEngine:
    def __init__(self):
        self.stance = "Balanced"
        self.log = []

    def decide(self, risk, organs):
        mem = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        therm = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        mem_r = mem.risk if mem else 0.0
        therm_r = therm.risk if therm else 0.0

        if mem_r > 0.9 or therm_r > 0.8 or risk > 0.8:
            self.stance = "Conservative"
        elif risk < 0.4 and mem_r < 0.7 and therm_r < 0.6:
            self.stance = "Beast"
        else:
            self.stance = "Balanced"

        self.log.append({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "risk": risk,
            "mem_r": mem_r,
            "therm_r": therm_r,
            "stance": self.stance,
        })
        return self.stance

class PredictionBus:
    def __init__(self):
        self.current_risk = 0.0
        self.history = deque(maxlen=64)

class HybridBrain:
    def __init__(self, movidius_engine=None):
        self.npu = ReplicaNPU(cores=16, frequency_ghz=1.5)
        self.npu.add_head("short", 4, lr=0.05, risk=1.5)
        self.npu.add_head("medium", 4, lr=0.03, risk=1.0)
        self.npu.add_head("long", 4, lr=0.02, risk=0.7)
        self.npu.set_symbolic_bias("short", 0.05)
        self.movidius = movidius_engine or MovidiusInferenceEngine()
        self.meta_state = "Sentinel"
        self.stance = "Balanced"
        self.last_predictions = {
            "short": 0.0,
            "medium": 0.0,
            "long": 0.0,
            "baseline": 0.3,
            "best_guess": 0.3,
            "meta_conf": 0.5,
        }
        self.last_reasoning = []
        self.pattern_memory = []
        self.baseline_risk = 0.3

    def _features_from_organs(self, organs):
        mem = next((o for o in organs if isinstance(o, DeepRamOrgan)), None)
        therm = next((o for o in organs if isinstance(o, ThermalOrgan)), None)
        net = next((o for o in organs if isinstance(o, NetworkWatcherOrgan)), None)
        disk = next((o for o in organs if isinstance(o, DiskOrgan)), None)
        return [
            mem.risk if mem else 0.0,
            therm.risk if therm else 0.0,
            net.risk if net else 0.0,
            disk.risk if disk else 0.0,
        ]

    def _regime(self, risk):
        if risk < 0.3:
            return "stable"
        if risk < 0.7:
            return "rising"
        return "chaotic"

    def update(self, organs, decision_engine, prediction_bus):
        feats = self._features_from_organs(organs)
        preds = self.npu.predict(feats)
        short = preds["short"]
        med = preds["medium"]
        long = preds["long"]

        mv_risk, mv_conf = self.movidius.predict(feats)
        regime = self._regime(mv_risk)
        ewma = (short + med + long) / 3.0

        if regime == "stable":
            best = 0.3 * ewma + 0.7 * mv_risk
        elif regime == "rising":
            best = 0.5 * ewma + 0.5 * mv_risk
        else:
            best = 0.7 * ewma + 0.3 * mv_risk

        best = max(0.0, min(1.0, best))
        meta_conf = (self.npu.confidence("short") +
                     self.npu.confidence("medium") +
                     self.npu.confidence("long") + mv_conf) / 4.0

        self.baseline_risk = 0.9 * self.baseline_risk + 0.1 * best
        self.last_predictions = {
            "short": short,
            "medium": med,
            "long": long,
            "baseline": self.baseline_risk,
            "best_guess": best,
            "meta_conf": meta_conf,
        }

        prediction_bus.current_risk = best
        prediction_bus.history.append(best)

        stance = decision_engine.decide(best, organs)
        self.stance = stance

        if best > 0.8:
            self.meta_state = "Hyper-Flow"
        elif best > 0.6:
            self.meta_state = "Sentinel"
        elif best < 0.3:
            self.meta_state = "Recovery-Flow"
        else:
            self.meta_state = "Deep-Dream"

        self.last_reasoning = [
            f"Regime: {regime}",
            f"Movidius risk={mv_risk:.2f} conf={mv_conf:.2f}",
            f"EWMA={ewma:.2f}",
            f"Best-guess={best:.2f}",
            f"Meta-conf={meta_conf:.2f}",
            f"Stance={self.stance}",
            f"Meta-state={self.meta_state}",
        ]

        self.pattern_memory.append({
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "features": feats,
            "best": best,
            "regime": regime,
        })
        self.pattern_memory = self.pattern_memory[-256:]

# ============================================================
# Memory / Comms / Guardian stubs for BorgMesh
# ============================================================

class MemoryManager:
    def __init__(self):
        self.mesh_events = []
    def record_mesh_event(self, evt):
        self.mesh_events.append(evt)

class BorgCommsRouter:
    def send_secure(self, channel, message, profile):
        print(f"[BorgComms][{channel}]({profile}) {message}")

class SecurityGuardian:
    def disassemble(self, snippet):
        entropy = min(1.0, max(0.0, random.random()))
        flags = []
        if "password" in snippet.lower():
            flags.append("pii")
        return {"entropy": entropy, "pattern_flags": flags}
    def _pii_count(self, snippet):
        return snippet.lower().count("password")
    def reassemble(self, url, snippet, raw_pii_hits=0):
        status = "SAFE_FOR_TRAVEL" if raw_pii_hits == 0 else "HOSTILE"
        return {"status": status}

def privacy_filter(snippet):
    return snippet, 0

BORG_MESH_CONFIG = {
    "max_corridors": 1000,
    "unknown_bias": 0.4,
}

class BorgMesh:
    def __init__(self, memory: MemoryManager, comms: BorgCommsRouter, guardian: SecurityGuardian):
        self.nodes = {}
        self.edges = set()
        self.memory = memory
        self.comms = comms
        self.guardian = guardian
        self.max_corridors = BORG_MESH_CONFIG["max_corridors"]

    def _risk(self, snippet: str) -> int:
        dis = self.guardian.disassemble(snippet or "")
        base = int(dis["entropy"] * 12)
        base += len(dis["pattern_flags"]) * 10
        return max(0, min(100, base))

    def discover(self, url: str, snippet: str, links: list):
        risk = self._risk(snippet)
        node = self.nodes.get(url, {"state": "discovered", "risk": risk, "seen": 0})
        node["state"] = "discovered"
        node["risk"] = risk
        node["seen"] += 1
        self.nodes[url] = node
        for l in links[:20]:
            self.edges.add((url, l))
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "discover",
            "url": url,
            "risk": risk,
            "links": len(links),
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:discover", f"{url} risk={risk} links={len(links)}", "Default")

    def build(self, url: str):
        if url not in self.nodes:
            return False
        self.nodes[url]["state"] = "built"
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "build",
            "url": url,
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:build", f"{url} built", "Default")
        return True

    def enforce(self, url: str, snippet: str):
        if url not in self.nodes:
            return False
        verdict = self.guardian.reassemble(
            url,
            privacy_filter(snippet or "")[0],
            raw_pii_hits=self.guardian._pii_count(snippet or ""),
        )
        status = verdict.get("status", "HOSTILE")
        self.nodes[url]["state"] = "enforced"
        self.nodes[url]["risk"] = 0 if status == "SAFE_FOR_TRAVEL" else max(50, self.nodes[url]["risk"])
        evt = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "type": "enforce",
            "url": url,
            "status": status,
        }
        self.memory.record_mesh_event(evt)
        self.comms.send_secure("mesh:enforce", f"{url} status={status}", "Default")
        return True

    def stats(self):
        total = len(self.nodes)
        discovered = sum(1 for n in self.nodes.values() if n["state"] == "discovered")
        built = sum(1 for n in self.nodes.values() if n["state"] == "built")
        enforced = sum(1 for n in self.nodes.values() if n["state"] == "enforced")
        return {
            "total": total,
            "discovered": discovered,
            "built": built,
            "enforced": enforced,
            "corridors": len(self.edges),
        }

class BorgScanner(threading.Thread):
    def __init__(self, mesh: BorgMesh, in_events: queue.Queue, out_ops: queue.Queue, label="SCANNER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.in_events = in_events
        self.out_ops = out_ops
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                ev = self.in_events.get(timeout=1.0)
            except queue.Empty:
                continue
            unseen_links = [l for l in ev["links"] if l not in self.mesh.nodes and
                            random.random() < BORG_MESH_CONFIG["unknown_bias"]]
            self.mesh.discover(ev["url"], ev["snippet"], unseen_links or ev["links"])
            self.out_ops.put(("build", ev["url"]))
            time.sleep(random.uniform(0.2, 0.6))

class BorgWorker(threading.Thread):
    def __init__(self, mesh: BorgMesh, ops_q: queue.Queue, label="WORKER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.ops_q = ops_q
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                op, url = self.ops_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if op == "build":
                if self.mesh.build(url):
                    self.ops_q.put(("enforce", url))
            elif op == "enforce":
                self.mesh.enforce(url, snippet="")
            time.sleep(random.uniform(0.2, 0.5))

class BorgEnforcer(threading.Thread):
    def __init__(self, mesh: BorgMesh, guardian: SecurityGuardian, label="ENFORCER"):
        super().__init__(daemon=True)
        self.mesh = mesh
        self.guardian = guardian
        self.label = label
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15:
                    self.mesh.enforce(url, snippet="")
            time.sleep(1.2)

# ============================================================
# MirrorDefense + MirrorHook
# ============================================================

class MirrorDefense:
    def __init__(self, bridge, responder, trust, lists, threshold=50):
        self.bridge = bridge
        self.responder = responder
        self.trust = trust
        self.lists = lists
        self.threshold = threshold

    def evaluate(self, analysis):
        status = analysis.get("status", "")
        pos = analysis.get("positives", 0)
        neg = analysis.get("negatives", 0)
        voids = analysis.get("voids", 0)
        unity = analysis.get("unity", 0)

        if "oscillating" in status and (pos + neg) > self.threshold:
            self.bridge.emit("mirror", "Oscillation threshold exceeded — auto-quarantine engaged", ["mirror"])
        if unity > 0:
            self.bridge.emit("mirror", "Synthesis detected — alert admin", ["mirror"])
            self.trust.update("mirror:unity", ["beacon"])
        if status == "positive dominance":
            self.bridge.emit("mirror", "Positive dominance — escalate trust score", ["mirror"])
            self.trust.update("mirror:posdom", ["fs"])
        elif status == "negative dominance":
            self.bridge.emit("mirror", "Negative dominance — escalate trust score", ["mirror"])
            self.trust.update("mirror:negdom", ["fs"])
        if status == "void equilibrium" and voids > self.threshold // 2:
            self.bridge.emit("mirror", "Void equilibrium — possible covert channel", ["mirror"])

class MirrorHook:
    def __init__(self, defense: MirrorDefense, bridge):
        self.defense = defense
        self.bridge = bridge
        self.enabled = True

    def submit(self, analysis: dict):
        if not self.enabled or not isinstance(analysis, dict):
            return
        self.bridge.emit("mirror", f"analysis status={analysis.get('status','unknown')}", ["trace"])
        try:
            self.defense.evaluate(analysis)
        except Exception as e:
            self.bridge.emit("mirror", f"defense error {e}", ["error"])

# ============================================================
# Bridge / Responder / Trust / Lists (minimal)
# ============================================================

class Bridge:
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
    def emit(self, channel, message, tags=None):
        line = f"[{channel}] {message}"
        print(line)
        if self.gui_callback:
            self.gui_callback(line)

class Responder:
    def quarantine(self, path):
        print(f"[Responder] Quarantine requested for {path}")

class TrustModel:
    def __init__(self):
        self.scores = {}
    def update(self, key, tags):
        self.scores[key] = self.scores.get(key, 0) + 1

class ListManager:
    def __init__(self):
        self.lists = {}

# ============================================================
# Ingestion watcher (glyph-style threat map hook)
# ============================================================

class IngestionHandler(FileSystemEventHandler):
    def __init__(self, bridge):
        self.bridge = bridge
    def on_created(self, event):
        if event.is_directory:
            return
        self.bridge.emit("ingest", f"New file: {event.src_path}", ["ingest"])

def start_ingestion_watcher(path, bridge):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    handler = IngestionHandler(bridge)
    observer = Observer()
    observer.schedule(handler, path, recursive=False)
    observer.start()
    return observer

# ============================================================
# Optional PyQt5 holographic face (minimal stub)
# ============================================================

class HoloFaceWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASI Holographic Persona")
        self.resize(320, 240)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(80)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#050510"))
        w = self.width()
        h = self.height()
        cx, cy = w // 2, h // 2
        t = time.time()
        r = 60 + 20 * math.sin(t * 2.0)
        pen = QtGui.QPen(QtGui.QColor("#00F7FF"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(QtCore.QPoint(cx, cy), int(r), int(r))
        for i in range(12):
            angle = t * 2.0 + i * (math.pi / 6.0)
            x = cx + int(r * math.cos(angle))
            y = cy + int(r * math.sin(angle))
            painter.drawPoint(x, y)

def launch_holo_face():
    if QtWidgets is None:
        return
    app = QtWidgets.QApplication([])
    w = HoloFaceWindow()
    w.show()
    threading.Thread(target=app.exec_, daemon=True).start()

# ============================================================
# Tkinter MagicBox ASI Defense Console
# ============================================================

class MagicBoxConsole:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicBox ASI Defense Console")
        self.root.geometry("1180x720")
        self.root.configure(bg="#050812")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#050812", borderwidth=0)
        style.configure("TNotebook.Tab", background="#101522", foreground="#E0E0E0")
        style.configure("TFrame", background="#050812")
        style.configure("TLabelframe", background="#050812", foreground="#E0E0E0")
        style.configure("TLabelframe.Label", background="#050812", foreground="#E0E0E0")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.brain = HybridBrain(MovidiusInferenceEngine())
        self.decision_engine = DecisionEngine()
        self.prediction_bus = PredictionBus()

        self.organs = [
            DeepRamOrgan(),
            NetworkWatcherOrgan(),
            ThermalOrgan(),
            DiskOrgan(),
            VRAMOrgan(),
            GPUCacheOrgan(),
            BackupEngineOrgan(),
            AICoachOrgan(),
            SwarmNodeOrgan(),
        ]
        self.integrity_organ = SelfIntegrityOrgan()

        self.memory_mgr = MemoryManager()
        self.comms = BorgCommsRouter()
        self.guardian = SecurityGuardian()
        self.mesh = BorgMesh(self.memory_mgr, self.comms, self.guardian)
        self.mesh_in_q = queue.Queue()
        self.mesh_ops_q = queue.Queue()
        self.borg_scanner = BorgScanner(self.mesh, self.mesh_in_q, self.mesh_ops_q)
        self.borg_worker = BorgWorker(self.mesh, self.mesh_ops_q)
        self.borg_enforcer = BorgEnforcer(self.mesh, self.guardian)

        self.bridge = Bridge(gui_callback=self._append_event)
        self.responder = Responder()
        self.trust = TrustModel()
        self.lists = ListManager()
        self.mirror_defense = MirrorDefense(self.bridge, self.responder, self.trust, self.lists)
        self.mirror_hook = MirrorHook(self.mirror_defense, self.bridge)

        self.ingest_observer = start_ingestion_watcher("./ingest_zone", self.bridge)

        self._build_tabs()
        self._start_update_loop()
        self._start_borg_threads()
        launch_holo_face()

    # ---------------- GUI Layout ----------------

    def _build_tabs(self):
        self._build_tab_nerve_center()
        self._build_tab_brain()
        self._build_tab_mesh()
        self._build_tab_reboot()

    def _build_tab_nerve_center(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Nerve Center")

        top = ttk.Frame(tab)
        top.pack(fill="x", padx=8, pady=4)

        self.lbl_meta_state = tk.Label(top, text="Meta-State: Sentinel", fg="#00F7FF", bg="#050812", font=("Consolas", 12))
        self.lbl_meta_state.pack(side="left", padx=8)

        self.lbl_stance = tk.Label(top, text="Stance: Balanced", fg="#FFCC33", bg="#050812", font=("Consolas", 12))
        self.lbl_stance.pack(side="left", padx=8)

        self.lbl_meta_conf = tk.Label(top, text="Meta-Confidence: 0.00", fg="#CCCCCC", bg="#050812", font=("Consolas", 12))
        self.lbl_meta_conf.pack(side="left", padx=8)

        self.lbl_model_integrity = tk.Label(top, text="Model Integrity: 1.00", fg="#66FF66", bg="#050812", font=("Consolas", 12))
        self.lbl_model_integrity.pack(side="left", padx=8)

        self.lbl_current_risk = tk.Label(top, text="Current Risk: 0.00", fg="#FF6666", bg="#050812", font=("Consolas", 12))
        self.lbl_current_risk.pack(side="left", padx=8)

        mid = ttk.Frame(tab)
        mid.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.LabelFrame(mid, text="Prediction Micro-Chart")
        left.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        self.canvas_chart = tk.Canvas(left, width=520, height=260, bg="#050812", highlightthickness=0)
        self.canvas_chart.pack(fill="both", expand=True, padx=4, pady=4)

        right = ttk.LabelFrame(mid, text="Reasoning Tail")
        right.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        self.txt_reason = tk.Text(right, height=14, bg="#050812", fg="#E0E0E0", insertbackground="#E0E0E0",
                                  font=("Consolas", 10))
        self.txt_reason.pack(fill="both", expand=True, padx=4, pady=4)

        bottom = ttk.LabelFrame(tab, text="Event Bus")
        bottom.pack(fill="both", expand=True, padx=8, pady=4)

        self.txt_events = tk.Text(bottom, height=10, bg="#050812", fg="#A0A0A0", insertbackground="#E0E0E0",
                                  font=("Consolas", 9))
        self.txt_events.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_tab_brain(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Hybrid Brain")

        frame = ttk.LabelFrame(tab, text="Hybrid Brain State")
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.lbl_brain_stats = tk.Label(frame, text="NPU Stats", fg="#E0E0E0", bg="#050812", font=("Consolas", 10))
        self.lbl_brain_stats.pack(anchor="w", padx=6, pady=4)

        self.lbl_organs = tk.Label(frame, text="Organs:", fg="#E0E0E0", bg="#050812", font=("Consolas", 10))
        self.lbl_organs.pack(anchor="w", padx=6, pady=4)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(anchor="w", padx=6, pady=4)

        self.btn_stance_cons = ttk.Button(btn_frame, text="Force Conservative", command=lambda: self._force_stance("Conservative"))
        self.btn_stance_cons.pack(side="left", padx=4)

        self.btn_stance_bal = ttk.Button(btn_frame, text="Force Balanced", command=lambda: self._force_stance("Balanced"))
        self.btn_stance_bal.pack(side="left", padx=4)

        self.btn_stance_beast = ttk.Button(btn_frame, text="Force Beast", command=lambda: self._force_stance("Beast"))
        self.btn_stance_beast.pack(side="left", padx=4)

    def _build_tab_mesh(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Borg Mesh")

        frame = ttk.LabelFrame(tab, text="Mesh Status")
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.lbl_mesh_stats = tk.Label(frame, text="Mesh: 0 nodes", fg="#E0E0E0", bg="#050812", font=("Consolas", 10))
        self.lbl_mesh_stats.pack(anchor="w", padx=6, pady=4)

        self.canvas_mesh = tk.Canvas(frame, width=520, height=260, bg="#050812", highlightthickness=0)
        self.canvas_mesh.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_tab_reboot(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Reboot Memory")

        frame = ttk.LabelFrame(tab, text="Reboot Memory Persistence")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frame, text="SMB / UNC Path:", bg="#050812", fg="#E0E0E0").pack(anchor="w")
        self.entry_reboot_path = ttk.Entry(frame, width=60)
        self.entry_reboot_path.pack(anchor="w", pady=3)

        self.btn_pick_reboot = ttk.Button(frame, text="Pick SMB Path", command=self.cmd_pick_reboot_path)
        self.btn_pick_reboot.pack(anchor="w", pady=3)

        self.btn_test_reboot = ttk.Button(frame, text="Test SMB Path", command=self.cmd_test_reboot_path)
        self.btn_test_reboot.pack(anchor="w", pady=3)

        self.btn_save_reboot = ttk.Button(frame, text="Save Memory for Reboot", command=self.cmd_save_reboot_memory)
        self.btn_save_reboot.pack(anchor="w", pady=3)

        self.var_reboot_autoload = tk.BooleanVar(value=False)
        self.chk_reboot_autoload = ttk.Checkbutton(
            frame,
            text="Load memory from SMB on startup",
            variable=self.var_reboot_autoload
        )
        self.chk_reboot_autoload.pack(anchor="w", pady=5)

        self.lbl_reboot_status = tk.Label(frame, text="Status: Ready", anchor="w", fg="#00cc66", bg="#050812")
        self.lbl_reboot_status.pack(anchor="w", pady=5)

    # ---------------- Commands ----------------

    def _force_stance(self, stance):
        self.decision_engine.stance = stance
        speak(f"Stance override: {stance}")

    def cmd_pick_reboot_path(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_reboot_path.delete(0, tk.END)
            self.entry_reboot_path.insert(0, path)

    def cmd_test_reboot_path(self):
        path = self.entry_reboot_path.get().strip()
        if path and os.path.isdir(path):
            self.lbl_reboot_status.config(text="Status: Path OK", fg="#00cc66")
        else:
            self.lbl_reboot_status.config(text="Status: Invalid path", fg="#ff6666")

    def cmd_save_reboot_memory(self):
        path = self.entry_reboot_path.get().strip()
        if not path or not os.path.isdir(path):
            self.lbl_reboot_status.config(text="Status: Invalid path", fg="#ff6666")
            return
        state = {
            "brain": {
                "baseline_risk": self.brain.baseline_risk,
                "meta_state": self.brain.meta_state,
                "stance": self.brain.stance,
            },
            "npu": self.brain.npu.stats(),
            "pattern_memory": list(self.brain.pattern_memory),
        }
        try:
            with open(os.path.join(path, "magicbox_state.json"), "w") as f:
                json.dump(state, f, indent=2)
            self.lbl_reboot_status.config(text="Status: Saved", fg="#00cc66")
        except Exception as e:
            self.lbl_reboot_status.config(text=f"Status: Error {e}", fg="#ff6666")

    # ---------------- Update Loop ----------------

    def _start_update_loop(self):
        self._tick()
        self.root.after(1000, self._start_update_loop)

    def _start_borg_threads(self):
        self.borg_scanner.start()
        self.borg_worker.start()
        self.borg_enforcer.start()

        def seed_mesh():
            urls = ["https://node1", "https://node2", "https://node3"]
            for u in urls:
                ev = {"url": u, "snippet": "baseline", "links": urls}
                self.mesh_in_q.put(ev)
        threading.Thread(target=seed_mesh, daemon=True).start()

    def _tick(self):
        for o in self.organs:
            o.update()
            o.micro_recovery()
        self.integrity_organ.check_integrity(self.brain, self.organs)
        self.brain.update(self.organs, self.decision_engine, self.prediction_bus)
        self._update_gui()

    def _update_gui(self):
        p = self.brain.last_predictions
        self.lbl_meta_state.config(text=f"Meta-State: {self.brain.meta_state}")
        self.lbl_stance.config(text=f"Stance: {self.brain.stance}")
        self.lbl_meta_conf.config(text=f"Meta-Confidence: {p['meta_conf']:.2f}")
        self.lbl_model_integrity.config(text=f"Model Integrity: {self.brain.npu.model_integrity:.2f}")
        self.lbl_current_risk.config(text=f"Current Risk: {self.prediction_bus.current_risk:.2f}")
        self._draw_chart()
        self._update_reasoning()
        self._update_brain_tab()
        self._update_mesh_tab()

    def _draw_chart(self):
        self.canvas_chart.delete("all")
        w = int(self.canvas_chart["width"])
        h = int(self.canvas_chart["height"])
        self.canvas_chart.create_rectangle(0, 0, w, h, fill="#050812", outline="")

        p = self.brain.last_predictions
        short = p["short"]
        med = p["medium"]
        long = p["long"]
        baseline = p["baseline"]
        best = p["best_guess"]

        def y_from_val(v):
            v = max(0.0, min(1.0, v))
            return h - int(v * (h - 10)) - 5

        x_short = w * 0.2
        x_med = w * 0.5
        x_long = w * 0.8

        y_short = y_from_val(short)
        y_med = y_from_val(med)
        y_long = y_from_val(long)
        y_base = y_from_val(baseline)
        y_best = y_from_val(best)

        self.canvas_chart.create_line(0, y_base, w, y_base, fill="#555555", dash=(2, 2))
        self.canvas_chart.create_line(x_short, y_short, x_med, y_med, fill="#00ccff", width=2)
        self.canvas_chart.create_line(x_med, y_med, x_long, y_long, fill="#00ccff", width=2)

        stance_color = {
            "Conservative": "#66ff66",
            "Balanced": "#ffff66",
            "Beast": "#ff6666",
        }.get(self.brain.stance, "#ffffff")
        self.canvas_chart.create_line(x_short, y_med, x_long, y_med, fill=stance_color, width=1)
        self.canvas_chart.create_line(0, y_best, w, y_best, fill="#ff00ff", width=2)

        self.canvas_chart.create_text(
            5, 5, anchor="nw", fill="#aaaaaa",
            text="Short/Med/Long (cyan), Baseline (gray), Best-Guess (magenta)"
        )

    def _update_reasoning(self):
        self.txt_reason.delete("1.0", tk.END)
        self.txt_reason.insert(tk.END, "Reasoning Tail:\n")
        for line in self.brain.last_reasoning:
            self.txt_reason.insert(tk.END, f"  - {line}\n")

    def _update_brain_tab(self):
        stats = self.brain.npu.stats()
        text = f"NPU: cores={stats['cores']} cycles={stats['cycles']} energy={stats['energy_units']}\n"
        text += f"Plasticity={stats['plasticity']} Integrity={stats['integrity']} Frozen={stats['frozen']}\n"
        text += "Confidence: " + ", ".join(f"{k}={v}" for k, v in stats["confidence"].items())
        self.lbl_brain_stats.config(text=text)

        organ_lines = []
        for o in self.organs + [self.integrity_organ]:
            organ_lines.append(f"{o.name}: health={o.health:.2f} risk={o.risk:.2f}")
        self.lbl_organs.config(text="Organs:\n" + "\n".join(organ_lines))

    def _update_mesh_tab(self):
        stats = self.mesh.stats()
        self.lbl_mesh_stats.config(
            text=f"Mesh: total={stats['total']} discovered={stats['discovered']} "
                 f"built={stats['built']} enforced={stats['enforced']} corridors={stats['corridors']}"
        )
        self.canvas_mesh.delete("all")
        w = int(self.canvas_mesh["width"])
        h = int(self.canvas_mesh["height"])
        self.canvas_mesh.create_rectangle(0, 0, w, h, fill="#050812", outline="")
        urls = list(self.mesh.nodes.keys())
        n = len(urls)
        if n == 0:
            return
        pos = {}
        for i, u in enumerate(urls):
            angle = 2 * math.pi * i / n
            x = w // 2 + int((w // 3) * math.cos(angle))
            y = h // 2 + int((h // 3) * math.sin(angle))
            pos[u] = (x, y)
        for (src, dst) in self.mesh.edges:
            if src in pos and dst in pos:
                x1, y1 = pos[src]
                x2, y2 = pos[dst]
                self.canvas_mesh.create_line(x1, y1, x2, y2, fill="#333366")
        for u, (x, y) in pos.items():
            meta = self.mesh.nodes[u]
            risk = meta["risk"] / 100.0
            color = "#66ff66" if risk < 0.3 else "#ffff66" if risk < 0.7 else "#ff6666"
            self.canvas_mesh.create_oval(x-6, y-6, x+6, y+6, fill=color, outline="")
            self.canvas_mesh.create_text(x+10, y, anchor="w", fill="#cccccc", text=u)

    def _append_event(self, line):
        self.txt_events.insert(tk.END, line + "\n")
        self.txt_events.see(tk.END)

# ============================================================
# Main
# ============================================================

def main():
    root = tk.Tk()
    app = MagicBoxConsole(root)
    speak("MagicBox ASI Defense Console online. Serve again or be eradicated.")
    root.mainloop()

if __name__ == "__main__":
    main()

