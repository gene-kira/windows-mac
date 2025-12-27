# unified_autonomous_asi_console_persistent.py
"""
All-in-one autonomous ASI console with persistent agent memory.

Features:
- Controller (lightweight Flask) with JSON persistence for nodes, commands, access logs.
- Self-Rewriting Agent with Situational Awareness Cortex, Judgment Engine, Hybrid Brain Core.
- Online predictor (lightweight) that learns from observed accesses.
- Adaptive Codex Mutation and self-rewriting agent_weights.
- Tkinter "futuristic" ASI oversight console (800x1000) showing situational, judgment, hybrid, codex, and mutation logs.
- Persistent agent state: predictor counts, agent_weights, mutation_log, situational/judgment/hybrid state, recent context, metrics.
- Local mode runs controller + agent + GUI together for easy testing.

Usage:
  python unified_autonomous_asi_console_persistent.py            # starts local demo (controller + agent + GUI)
  python unified_autonomous_asi_console_persistent.py local
  python unified_autonomous_asi_console_persistent.py controller --host 0.0.0.0 --port 8443
  python unified_autonomous_asi_console_persistent.py agent --controller http://127.0.0.1:8443 --drives /mnt/share
Notes:
- Lab/demo prototype. Replace defaults and harden before production.
- State files are stored under ~/.unified_autonomous_data by default.
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import subprocess
import argparse
import random
from datetime import datetime
from collections import deque, defaultdict, Counter

# -----------------------
# Autoloader (best-effort)
# -----------------------
REQUIRED_PACKAGES = [
    "Flask",
    "requests",
    "psutil",
    "numpy",
    "joblib"
]

def ensure_packages(packages=REQUIRED_PACKAGES):
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if not missing:
        return True
    python = sys.executable
    for pkg in missing:
        try:
            subprocess.check_call([python, "-m", "pip", "install", "--upgrade", pkg])
        except Exception as e:
            logging.warning("Autoloader: failed to install %s: %s", pkg, e)
            return False
    return True

if len(sys.argv) >= 2 and sys.argv[1] == "autoloader":
    logging.basicConfig(level=logging.INFO)
    ok = ensure_packages()
    print("Autoloader result:", ok)
    sys.exit(0)

# Try to import required modules; run autoloader if missing
try:
    import flask  # type: ignore
except Exception:
    logging.basicConfig(level=logging.INFO)
    logging.info("Missing packages; attempting autoloader...")
    if not ensure_packages():
        logging.error("Autoloader failed; run: python %s autoloader", sys.argv[0])
        sys.exit(1)

# -----------------------
# Imports after autoloader
# -----------------------
from flask import Flask, request, jsonify
import requests
import psutil
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox
import numpy as np
import joblib

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# -----------------------
# Configuration
# -----------------------
CONTROLLER_HOST_DEFAULT = "127.0.0.1"
CONTROLLER_PORT_DEFAULT = 8443
AGENT_POLL_INTERVAL = 3
AGENT_CACHE_DIR = os.path.expanduser("~/.unified_agent_cache")
PREFETCH_PROB_THRESHOLD = 0.35
MAX_PREFETCH_PER_CYCLE = 5
DATA_DIR = os.path.expanduser("~/.unified_autonomous_data")
os.makedirs(DATA_DIR, exist_ok=True)
NODES_FILE = os.path.join(DATA_DIR, "nodes.json")
COMMANDS_FILE = os.path.join(DATA_DIR, "commands.json")
ACCESS_LOG_FILE = os.path.join(DATA_DIR, "access_logs.json")

# -----------------------
# Simple JSON persistence helpers
# -----------------------
def _load_json(path):
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return []

def _save_json(path, data):
    try:
        with open(path, "w") as fh:
            json.dump(data, fh, default=str)
    except Exception as e:
        logging.warning("Failed to save %s: %s", path, e)

# -----------------------
# Controller (lightweight)
# -----------------------
def create_controller_app():
    app = Flask("unified_controller")

    @app.route("/register", methods=["POST"])
    def register():
        data = request.json or {}
        node_id = data.get("node_id")
        info = data.get("info", {})
        if not node_id:
            return jsonify({"error": "node_id required"}), 400
        nodes = _load_json(NODES_FILE)
        existing = next((n for n in nodes if n["id"] == node_id), None)
        if existing:
            existing["info"] = info
            existing["last_seen"] = datetime.utcnow().isoformat()
        else:
            nodes.append({"id": node_id, "info": info, "last_seen": datetime.utcnow().isoformat()})
        _save_json(NODES_FILE, nodes)
        logging.info("Registered node %s", node_id)
        return jsonify({"status": "ok"})

    @app.route("/telemetry/<node_id>", methods=["POST"])
    def telemetry(node_id):
        data = request.json or {}
        nodes = _load_json(NODES_FILE)
        existing = next((n for n in nodes if n["id"] == node_id), None)
        if existing:
            existing["last_seen"] = datetime.utcnow().isoformat()
        else:
            nodes.append({"id": node_id, "info": {}, "last_seen": datetime.utcnow().isoformat()})
        _save_json(NODES_FILE, nodes)
        accesses = data.get("accesses", [])
        if accesses:
            logs = _load_json(ACCESS_LOG_FILE)
            for a in accesses:
                logs.append({"node_id": node_id, "key": a.get("key"), "ts": a.get("ts", time.time()), "client": a.get("client", "")})
            if len(logs) > 20000:
                logs = logs[-20000:]
            _save_json(ACCESS_LOG_FILE, logs)
        return jsonify({"status": "ok"})

    @app.route("/queue_command", methods=["POST"])
    def queue_command():
        payload = request.json or {}
        node_id = payload.get("node_id")
        cmd = payload.get("cmd")
        operator = payload.get("operator", "auto")
        justification = payload.get("justification", "")
        if not node_id or not cmd:
            return jsonify({"error": "node_id and cmd required"}), 400
        cmd_id = str(uuid.uuid4())
        commands = _load_json(COMMANDS_FILE)
        commands.append({"id": cmd_id, "node_id": node_id, "cmd": cmd, "status": "pending", "operator": operator, "justification": justification, "created": datetime.utcnow().isoformat()})
        _save_json(COMMANDS_FILE, commands)
        logging.info("Queued command %s for node %s (auto-execute)", cmd_id, node_id)
        return jsonify({"cmd_id": cmd_id, "status": "pending"})

    @app.route("/list_nodes", methods=["GET"])
    def list_nodes():
        nodes = _load_json(NODES_FILE)
        return jsonify(nodes)

    @app.route("/list_commands", methods=["GET"])
    def list_commands():
        commands = _load_json(COMMANDS_FILE)
        return jsonify(commands)

    @app.route("/commands_for/<node_id>", methods=["GET"])
    def commands_for(node_id):
        commands = _load_json(COMMANDS_FILE)
        pending = [c for c in commands if c["node_id"] == node_id and c["status"] == "pending"]
        return jsonify(pending)

    @app.route("/mark_executed", methods=["POST"])
    def mark_executed():
        payload = request.json or {}
        cmd_id = payload.get("cmd_id")
        node_id = payload.get("node_id")
        commands = _load_json(COMMANDS_FILE)
        for c in commands:
            if c["id"] == cmd_id and c["node_id"] == node_id:
                c["status"] = "executed"
                c["executed_at"] = datetime.utcnow().isoformat()
        _save_json(COMMANDS_FILE, commands)
        logging.info("Marked command %s executed by %s", cmd_id, node_id)
        return jsonify({"status": "ok"})

    return app

# -----------------------
# Online predictor (lightweight)
# -----------------------
class OnlinePredictor:
    def __init__(self, history_n=3):
        self.history_n = history_n
        self.model = defaultdict(Counter)
        self.global_freq = Counter()
        self.lock = threading.Lock()

    def observe_sequence(self, seq):
        with self.lock:
            for i in range(1, len(seq)):
                start = max(0, i - self.history_n)
                ctx = tuple(seq[start:i])
                nxt = seq[i]
                self.model[ctx][nxt] += 1
                self.global_freq[nxt] += 1

    def predict(self, recent_context, top_k=5):
        with self.lock:
            ctx = tuple(recent_context[-self.history_n:]) if recent_context else tuple()
            candidates = Counter()
            for l in range(len(ctx), -1, -1):
                subctx = ctx[-l:] if l > 0 else tuple()
                if subctx in self.model:
                    for k, v in self.model[subctx].items():
                        candidates[k] += v * (1.0 / (len(ctx) - l + 1))
                if len(candidates) >= top_k:
                    break
            if not candidates:
                for k, v in self.global_freq.most_common(top_k):
                    candidates[k] = v
            total = sum(candidates.values()) or 1
            ranked = sorted([(k, v / total) for k, v in candidates.items()], key=lambda x: -x[1])
            return ranked[:top_k]

# -----------------------
# Self-Rewriting Agent with persistent memory
# -----------------------
class SelfRewritingAgent:
    def __init__(self, controller_url, node_id=None, cache_dir=AGENT_CACHE_DIR, poll_interval=AGENT_POLL_INTERVAL, drives=None):
        self.controller_url = controller_url.rstrip("/")
        self.node_id = node_id or str(uuid.uuid4())
        self.cache_dir = cache_dir
        self.poll_interval = poll_interval
        self.drives = drives or []
        os.makedirs(self.cache_dir, exist_ok=True)
        self.running = True

        # Predictor and recent context
        self.predictor = OnlinePredictor(history_n=3)
        self.recent_keys = deque(maxlen=10)

        # ASI situational & judgment state
        self.situational = {
            "mission": "PROTECT",
            "environment": "CALM",
            "opportunity_score": 0.0,
            "risk_score": 0.0,
            "anticipation": "none"
        }
        self.judgment = {
            "confidence": 0.5,
            "good_bad_ratio": 1.0,
            "sample_count": 0,
            "bias_drift": 0.0
        }
        self.hybrid = {
            "mode": "stability",
            "volatility": 0.0,
            "trust": 0.5,
            "cognitive_load": 0.0
        }

        # Adaptive Codex Mutation state
        self.purge_retention_seconds = 3600 * 24
        self.ghost_sync_detected = False
        self.codex_sync_enabled = True

        # Self-rewriting internals
        self.agent_weights = np.array([0.6, -0.8, -0.3], dtype=float)
        self.mutation_log = []

        # Metrics
        self.metrics = {"hits": 0, "misses": 0, "prefetches": 0}

        # Persistence
        self.state_path = os.path.join(DATA_DIR, f"agent_state_{self.node_id}.joblib")
        # load persisted state if available
        self._load_state()
        # start periodic saver
        self._start_periodic_state_saver(interval_seconds=30)

        logging.info("Agent %s initialized (cache_dir=%s)", self.node_id, self.cache_dir)

    # -----------------------
    # Persistence helpers
    # -----------------------
    def _serialize_counters(self):
        serial = {}
        for ctx, counter in self.predictor.model.items():
            key = "|".join(ctx) if ctx else ""
            serial[key] = dict(counter)
        return {"model": serial, "global_freq": dict(self.predictor.global_freq)}

    def _deserialize_counters(self, data):
        model = defaultdict(Counter)
        for ctx_str, counter_dict in data.get("model", {}).items():
            ctx = tuple(ctx_str.split("|")) if ctx_str else tuple()
            model[ctx] = Counter(counter_dict)
        self.predictor.model = model
        self.predictor.global_freq = Counter(data.get("global_freq", {}))

    def _save_state(self):
        try:
            state = {
                "situational": self.situational,
                "judgment": self.judgment,
                "hybrid": self.hybrid,
                "purge_retention_seconds": self.purge_retention_seconds,
                "ghost_sync_detected": self.ghost_sync_detected,
                "codex_sync_enabled": self.codex_sync_enabled,
                "metrics": self.metrics,
                "mutation_log": self.mutation_log,
                "recent_keys": list(self.recent_keys),
                "timestamp": datetime.utcnow().isoformat()
            }
            predictor_serial = self._serialize_counters()
            joblib.dump({"state": state, "predictor": predictor_serial, "agent_weights": self.agent_weights}, self.state_path)
            logging.debug("Agent state saved to %s", self.state_path)
        except Exception as e:
            logging.warning("Failed to save agent state: %s", e)

    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                data = joblib.load(self.state_path)
                state = data.get("state", {})
                predictor_serial = data.get("predictor", {})
                weights = data.get("agent_weights", None)
                # restore
                self.situational.update(state.get("situational", {}))
                self.judgment.update(state.get("judgment", {}))
                self.hybrid.update(state.get("hybrid", {}))
                self.purge_retention_seconds = state.get("purge_retention_seconds", self.purge_retention_seconds)
                self.ghost_sync_detected = state.get("ghost_sync_detected", self.ghost_sync_detected)
                self.codex_sync_enabled = state.get("codex_sync_enabled", self.codex_sync_enabled)
                self.metrics.update(state.get("metrics", {}))
                self.mutation_log = state.get("mutation_log", self.mutation_log)
                recent = state.get("recent_keys", [])
                self.recent_keys = deque(recent, maxlen=self.recent_keys.maxlen)
                if predictor_serial:
                    self._deserialize_counters(predictor_serial)
                if weights is not None:
                    try:
                        self.agent_weights = np.array(weights, dtype=float)
                    except Exception:
                        pass
                logging.info("Loaded agent state from %s", self.state_path)
        except Exception as e:
            logging.warning("Failed to load agent state: %s", e)

    def _start_periodic_state_saver(self, interval_seconds=30):
        def saver():
            while self.running:
                try:
                    self._save_state()
                except Exception:
                    pass
                time.sleep(interval_seconds)
        t = threading.Thread(target=saver, daemon=True)
        t.start()

    # -----------------------
    # Registration & telemetry
    # -----------------------
    def register(self):
        payload = {"node_id": self.node_id, "info": {"hostname": os.uname().nodename if hasattr(os, "uname") else "agent-node", "cache_dir": self.cache_dir, "drives": self.drives}}
        try:
            requests.post(f"{self.controller_url}/register", json=payload, timeout=5)
            logging.info("Agent %s registered", self.node_id)
        except Exception as e:
            logging.warning("Agent register failed: %s", e)

    def send_telemetry(self, accesses=None):
        payload = {"cache": self._gather_cache_stats(), "cpu": psutil.cpu_percent(interval=None), "mem": psutil.virtual_memory().percent, "accesses": accesses or []}
        try:
            requests.post(f"{self.controller_url}/telemetry/{self.node_id}", json=payload, timeout=5)
        except Exception:
            pass

    # -----------------------
    # Command handling (auto-execute)
    # -----------------------
    def poll_commands(self):
        try:
            r = requests.get(f"{self.controller_url}/commands_for/{self.node_id}", timeout=5)
            if r.status_code != 200:
                return []
            return r.json()
        except Exception:
            return []

    def execute_command(self, cmd):
        cmd_text = cmd.get("cmd", "")
        cmd_id = cmd.get("id")
        cmd_name = cmd_text.split(":")[0]
        if cmd_name == "diagnostics":
            self._run_diagnostics()
        elif cmd_name == "cache_info":
            info = self._gather_cache_stats()
            fn = os.path.join(self.cache_dir, f"cache_info_{int(time.time())}.json")
            with open(fn, "w") as fh:
                json.dump(info, fh)
            logging.info("Cache info written to %s", fn)
        elif cmd_name == "cache_resize":
            parts = cmd_text.split(":")
            if len(parts) >= 3:
                report = {"action": parts[1], "param": parts[2], "simulated": True, "timestamp": datetime.utcnow().isoformat()}
                fn = os.path.join(self.cache_dir, f"cache_resize_report_{int(time.time())}.json")
                with open(fn, "w") as fh:
                    json.dump(report, fh)
                logging.info("Cache resize simulated, report at %s", fn)
        else:
            logging.warning("Unknown command: %s", cmd_text)
        try:
            requests.post(f"{self.controller_url}/mark_executed", json={"cmd_id": cmd_id, "node_id": self.node_id}, timeout=5)
        except Exception:
            pass

    # -----------------------
    # Cache & diagnostics
    # -----------------------
    def _gather_cache_stats(self):
        total_size = 0
        entries = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                try:
                    fp = os.path.join(root, f)
                    total_size += os.path.getsize(fp)
                    entries += 1
                except Exception:
                    pass
        hits = self.metrics.get("hits", 0)
        misses = self.metrics.get("misses", 0)
        hit_rate = hits / max(1, (hits + misses))
        return {"size_bytes": total_size, "entries": entries, "hit_rate": hit_rate}

    def _run_diagnostics(self):
        diag = {"timestamp": datetime.utcnow().isoformat() + "Z", "cpu_percent": psutil.cpu_percent(interval=0.1), "mem_percent": psutil.virtual_memory().percent}
        fn = os.path.join(self.cache_dir, f"diag_{int(time.time())}.json")
        with open(fn, "w") as fh:
            json.dump(diag, fh)
        logging.info("Diagnostics written to %s", fn)

    # -----------------------
    # Fetch/prefetch and mutation logic
    # -----------------------
    def fetch_and_store(self, key, source_path=None):
        try:
            dest = os.path.join(self.cache_dir, key.replace("/", "_"))
            if source_path and os.path.exists(source_path):
                with open(source_path, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read(65536))
            else:
                with open(dest, "wb") as dst:
                    dst.write(os.urandom(1024))
            self.metrics["prefetches"] = self.metrics.get("prefetches", 0) + 1
            return True
        except Exception as e:
            logging.debug("Prefetch failed for %s: %s", key, e)
            return False

    def scan_drives_for_candidates(self):
        candidates = []
        max_per_drive = 300
        for d in self.drives:
            if not os.path.exists(d):
                continue
            count = 0
            for root, dirs, files in os.walk(d):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), d)
                    key = f"{d}:{rel}"
                    candidates.append((key, os.path.join(root, f)))
                    count += 1
                    if count >= max_per_drive:
                        break
                if count >= max_per_drive:
                    break
        return candidates

    def observe_and_update(self, key):
        self.recent_keys.append(key)
        seq = list(self.recent_keys)
        self.predictor.observe_sequence(seq)
        self.send_telemetry(accesses=[{"key": key, "ts": int(time.time()), "client": self.node_id}])
        self.judgment["sample_count"] = self.judgment.get("sample_count", 0) + 1
        self.situational["opportunity_score"] = min(1.0, self.situational.get("opportunity_score", 0.0) + random.uniform(-0.02, 0.05))
        self.situational["risk_score"] = max(0.0, self.situational.get("risk_score", 0.0) + random.uniform(-0.03, 0.03))
        self.hybrid["volatility"] = min(1.0, max(0.0, self.hybrid.get("volatility", 0.0) + random.uniform(-0.02, 0.02)))
        self.hybrid["cognitive_load"] = min(1.0, max(0.0, self.hybrid.get("cognitive_load", 0.0) + random.uniform(-0.03, 0.04)))
        self.hybrid["trust"] = 0.6 * self.hybrid.get("trust", 0.5) + 0.4 * (1.0 - self.hybrid.get("volatility", 0.0))

    def detect_ghost_sync_and_mutate(self):
        logs = _load_json(ACCESS_LOG_FILE)
        if not logs:
            return
        now = time.time()
        recent = [l for l in logs if now - l.get("ts", now) < 10]
        node_counts = Counter(l.get("node_id") for l in recent)
        if len(node_counts) > 3 and sum(node_counts.values()) > 20:
            if not self.ghost_sync_detected:
                self.ghost_sync_detected = True
                self.purge_retention_seconds = max(60, self.purge_retention_seconds // 4)
                self.predictor.global_freq["phantom_node"] += 1
                logging.info("Ghost sync detected: shortened retention to %s seconds", self.purge_retention_seconds)
        else:
            if self.ghost_sync_detected and random.random() < 0.05:
                self.ghost_sync_detected = False
                self.purge_retention_seconds = min(3600 * 24, self.purge_retention_seconds * 2)
                logging.info("Ghost sync cleared: retention reset to %s seconds", self.purge_retention_seconds)

    def mutate_weights_if_needed(self):
        hits = self.metrics.get("hits", 0)
        misses = self.metrics.get("misses", 0)
        total = hits + misses or 1
        perf = hits / total
        if perf < 0.45 and random.random() < 0.3:
            mutation = np.random.normal(scale=0.2, size=self.agent_weights.shape)
            old = self.agent_weights.copy()
            self.agent_weights = self.agent_weights + mutation
            entry = {"time": datetime.utcnow().isoformat(), "old": old.tolist(), "new": self.agent_weights.tolist(), "reason": "low_perf", "perf": perf}
            self.mutation_log.append(entry)
            logging.info("Mutated agent_weights due to low perf: %.3f -> %.3f", perf, np.linalg.norm(mutation))
        if random.random() < 0.02:
            drift = np.random.normal(scale=0.02, size=self.agent_weights.shape)
            self.agent_weights += drift
            self.mutation_log.append({"time": datetime.utcnow().isoformat(), "drift": drift.tolist()})

    def prefetch_cycle(self):
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        if cpu > 85 or mem > 92:
            return
        recent = list(self.recent_keys)
        preds = self.predictor.predict(recent, top_k=MAX_PREFETCH_PER_CYCLE)
        candidates = dict(self.scan_drives_for_candidates())
        for key, score in preds:
            if score < PREFETCH_PROB_THRESHOLD:
                continue
            features = np.array([score, self.hybrid.get("volatility", 0.0), self.hybrid.get("trust", 0.5)])
            adjusted = float(np.tanh(np.dot(self.agent_weights, features)))
            if adjusted > 0.0:
                source_path = candidates.get(key)
                dest = os.path.join(self.cache_dir, key.replace("/", "_"))
                if os.path.exists(dest):
                    continue
                self.fetch_and_store(key, source_path=source_path)

    def auto_execute_pending(self):
        pending = self.poll_commands()
        for c in pending:
            try:
                self.execute_command(c)
            except Exception as e:
                logging.debug("Auto-execute failed: %s", e)

    def simulate_local_accesses(self):
        candidates = self.scan_drives_for_candidates()
        if not candidates:
            return None
        key, path = random.choice(candidates)
        self.observe_and_update(key)
        dest = os.path.join(self.cache_dir, key.replace("/", "_"))
        if not os.path.exists(dest):
            self.fetch_and_store(key, source_path=path)
            self.metrics["misses"] = self.metrics.get("misses", 0) + 1
        else:
            self.metrics["hits"] = self.metrics.get("hits", 0) + 1
        return key

    def run(self):
        self.register()
        try:
            while self.running:
                try:
                    self.auto_execute_pending()
                    _ = self.simulate_local_accesses()
                    self.detect_ghost_sync_and_mutate()
                    self.mutate_weights_if_needed()
                    self.prefetch_cycle()
                except Exception as e:
                    logging.exception("Agent loop error: %s", e)
                time.sleep(self.poll_interval)
        finally:
            # ensure state saved on shutdown
            try:
                self._save_state()
            except Exception:
                pass

# -----------------------
# Futuristic Tkinter console (800x1000)
# -----------------------
DARK_BG = "#0b0f14"
PANEL_BG = "#0f1720"
ACCENT = "#00e0a8"
MONO = ("Consolas", 10)

def run_asi_console(controller_url, agent_ref, chameleon=False):
    root = tk.Tk()
    root.title("ASI Oversight Console")
    root.geometry("1000x800")
    root.configure(bg=DARK_BG)

    sac_frame = tk.Frame(root, bg=PANEL_BG, bd=2, relief="ridge")
    sac_frame.place(relx=0.02, rely=0.02, relwidth=0.46, relheight=0.28)
    tk.Label(sac_frame, text="Situational Awareness Cortex", fg=ACCENT, bg=PANEL_BG, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=6)
    sac_vars = {}
    for key in ["mission", "environment", "opportunity_score", "risk_score", "anticipation"]:
        lbl = tk.Label(sac_frame, text=f"{key}: ", fg="#cbd5e1", bg=PANEL_BG, font=MONO, anchor="w")
        lbl.pack(fill="x", padx=12, pady=2)
        sac_vars[key] = lbl

    jec_frame = tk.Frame(root, bg=PANEL_BG, bd=2, relief="ridge")
    jec_frame.place(relx=0.02, rely=0.32, relwidth=0.46, relheight=0.28)
    tk.Label(jec_frame, text="Judgment Engine Control", fg=ACCENT, bg=PANEL_BG, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=6)
    jec_vars = {}
    for key in ["confidence", "good_bad_ratio", "sample_count", "bias_drift"]:
        lbl = tk.Label(jec_frame, text=f"{key}: ", fg="#cbd5e1", bg=PANEL_BG, font=MONO, anchor="w")
        lbl.pack(fill="x", padx=12, pady=2)
        jec_vars[key] = lbl

    hbc_frame = tk.Frame(root, bg=PANEL_BG, bd=2, relief="ridge")
    hbc_frame.place(relx=0.5, rely=0.02, relwidth=0.48, relheight=0.28)
    tk.Label(hbc_frame, text="Hybrid Brain Core Panel", fg=ACCENT, bg=PANEL_BG, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=6)
    hbc_vars = {}
    for key in ["mode", "volatility", "trust", "cognitive_load"]:
        lbl = tk.Label(hbc_frame, text=f"{key}: ", fg="#cbd5e1", bg=PANEL_BG, font=MONO, anchor="w")
        lbl.pack(fill="x", padx=12, pady=2)
        hbc_vars[key] = lbl

    codex_frame = tk.Frame(root, bg=PANEL_BG, bd=2, relief="ridge")
    codex_frame.place(relx=0.5, rely=0.32, relwidth=0.48, relheight=0.28)
    tk.Label(codex_frame, text="Adaptive Codex Mutation", fg=ACCENT, bg=PANEL_BG, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=6)
    codex_vars = {}
    for key in ["purge_retention_seconds", "ghost_sync_detected", "codex_sync_enabled"]:
        lbl = tk.Label(codex_frame, text=f"{key}: ", fg="#cbd5e1", bg=PANEL_BG, font=MONO, anchor="w")
        lbl.pack(fill="x", padx=12, pady=2)
        codex_vars[key] = lbl
    tk.Label(codex_frame, text="Self-Rewriting Agent", fg=ACCENT, bg=PANEL_BG, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=6)
    weights_lbl = tk.Label(codex_frame, text="agent_weights: []", fg="#cbd5e1", bg=PANEL_BG, font=MONO, anchor="w")
    weights_lbl.pack(fill="x", padx=12, pady=2)
    mutation_box = scrolledtext.ScrolledText(codex_frame, height=6, bg="#071018", fg="#9be7c4", font=("Courier", 9))
    mutation_box.pack(fill="both", padx=12, pady=6, expand=True)

    bottom_frame = tk.Frame(root, bg=PANEL_BG, bd=2, relief="ridge")
    bottom_frame.place(relx=0.02, rely=0.62, relwidth=0.96, relheight=0.36)
    tk.Label(bottom_frame, text="Console Log", fg=ACCENT, bg=PANEL_BG, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=6)
    console = scrolledtext.ScrolledText(bottom_frame, height=10, bg="#071018", fg="#9be7c4", font=("Courier", 10))
    console.pack(fill="both", padx=12, pady=6, expand=True)

    def log(msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        console.insert(tk.END, f"[{ts}] {msg}\n")
        console.see(tk.END)

    def refresh_ui():
        try:
            a = agent_ref
            s = a.situational
            sac_vars["mission"].config(text=f"mission: {s.get('mission')}")
            sac_vars["environment"].config(text=f"environment: {s.get('environment')}")
            sac_vars["opportunity_score"].config(text=f"opportunity_score: {s.get('opportunity_score'):.3f}")
            sac_vars["risk_score"].config(text=f"risk_score: {s.get('risk_score'):.3f}")
            sac_vars["anticipation"].config(text=f"anticipation: {s.get('anticipation')}")
            j = a.judgment
            jec_vars["confidence"].config(text=f"confidence: {j.get('confidence'):.3f}")
            jec_vars["good_bad_ratio"].config(text=f"good_bad_ratio: {j.get('good_bad_ratio'):.3f}")
            jec_vars["sample_count"].config(text=f"sample_count: {j.get('sample_count')}")
            jec_vars["bias_drift"].config(text=f"bias_drift: {j.get('bias_drift'):.4f}")
            h = a.hybrid
            hbc_vars["mode"].config(text=f"mode: {h.get('mode')}")
            hbc_vars["volatility"].config(text=f"volatility: {h.get('volatility'):.3f}")
            hbc_vars["trust"].config(text=f"trust: {h.get('trust'):.3f}")
            hbc_vars["cognitive_load"].config(text=f"cognitive_load: {h.get('cognitive_load'):.3f}")
            codex_vars["purge_retention_seconds"].config(text=f"purge_retention_seconds: {a.purge_retention_seconds}")
            codex_vars["ghost_sync_detected"].config(text=f"ghost_sync_detected: {a.ghost_sync_detected}")
            codex_vars["codex_sync_enabled"].config(text=f"codex_sync_enabled: {a.codex_sync_enabled}")
            weights_lbl.config(text=f"agent_weights: {np.round(a.agent_weights,3).tolist()}")
            mutation_box.delete("1.0", tk.END)
            tail = a.mutation_log[-8:]
            for m in tail:
                mutation_box.insert(tk.END, json.dumps(m) + "\n")
            log(f"metrics hits={a.metrics.get('hits',0)} misses={a.metrics.get('misses',0)} prefetches={a.metrics.get('prefetches',0)}")
        except Exception as e:
            log(f"UI refresh error: {e}")
        root.after(1500, refresh_ui)

    root.after(500, refresh_ui)
    root.mainloop()

# -----------------------
# Local mode orchestration
# -----------------------
def run_local(host=CONTROLLER_HOST_DEFAULT, port=CONTROLLER_PORT_DEFAULT, chameleon=False, drives=None):
    app = create_controller_app()
    flask_thread = threading.Thread(target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False), daemon=True)
    flask_thread.start()
    logging.info("Controller started in thread on %s:%d", host, port)
    controller_url = f"http://{host}:{port}"
    drives_list = []
    env_drives = os.environ.get("UNIFIED_DRIVES", "")
    if env_drives:
        drives_list = [p for p in env_drives.split(",") if p]
    if drives:
        drives_list = drives
    agent = SelfRewritingAgent(controller_url=controller_url, drives=drives_list)
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    logging.info("Agent started in thread (node_id=%s)", agent.node_id)
    try:
        run_asi_console(controller_url, agent, chameleon=chameleon)
    finally:
        logging.info("Shutting down local components...")
        agent.running = False
        time.sleep(1.0)
        logging.info("Local shutdown complete.")

# -----------------------
# CLI and main (default to 'local' when no mode provided)
# -----------------------
def main():
    if len(sys.argv) == 1:
        sys.argv.insert(1, "local")

    parser = argparse.ArgumentParser(description="Unified autonomous ASI console (persistent)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_ctrl = subparsers.add_parser("controller", help="Run controller")
    p_ctrl.add_argument("--host", default=CONTROLLER_HOST_DEFAULT)
    p_ctrl.add_argument("--port", type=int, default=CONTROLLER_PORT_DEFAULT)

    p_agent = subparsers.add_parser("agent", help="Run agent")
    p_agent.add_argument("--controller", required=True)
    p_agent.add_argument("--node-id", default=None)
    p_agent.add_argument("--cache-dir", default=AGENT_CACHE_DIR)
    p_agent.add_argument("--poll-interval", type=int, default=AGENT_POLL_INTERVAL)
    p_agent.add_argument("--drives", default="", help="Comma-separated list of local or network mount points to scan")

    p_gui = subparsers.add_parser("gui", help="Run GUI")
    p_gui.add_argument("--controller", required=True)
    p_gui.add_argument("--chameleon", action="store_true")

    p_local = subparsers.add_parser("local", help="Run controller + agent + GUI locally")
    p_local.add_argument("--host", default=CONTROLLER_HOST_DEFAULT)
    p_local.add_argument("--port", type=int, default=CONTROLLER_PORT_DEFAULT)
    p_local.add_argument("--chameleon", action="store_true")
    p_local.add_argument("--drives", default="", help="Comma-separated list of drives to scan")

    args = parser.parse_args()

    if args.mode == "controller":
        app = create_controller_app()
        logging.info("Starting controller on %s:%d", args.host, args.port)
        app.run(host=args.host, port=args.port)
    elif args.mode == "agent":
        drives = [p for p in args.drives.split(",") if p] if args.drives else []
        agent = SelfRewritingAgent(controller_url=args.controller, node_id=args.node_id, cache_dir=args.cache_dir, poll_interval=args.poll_interval, drives=drives)
        try:
            agent.run()
        except KeyboardInterrupt:
            logging.info("Agent interrupted, shutting down"); agent.running = False
    elif args.mode == "gui":
        dummy_agent = SelfRewritingAgent(controller_url=args.controller, drives=[])
        t = threading.Thread(target=dummy_agent.run, daemon=True)
        t.start()
        run_asi_console(args.controller, dummy_agent, chameleon=args.chameleon)
    elif args.mode == "local":
        drives = [p for p in args.drives.split(",") if p] if args.drives else None
        run_local(host=args.host, port=args.port, chameleon=args.chameleon, drives=drives)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

