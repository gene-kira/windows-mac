import os
import sys
import time
import json
import threading
import queue
import statistics
import sqlite3
import ctypes
import hashlib
import math
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

import psutil

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QMessageBox,
    QTextEdit,
    QComboBox,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, QUrl

# Optional GPU / websockets imports (soft-fail if missing)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import websockets
except Exception:
    websockets = None

# Optional WebEngine import (soft-fail if missing)
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except Exception:
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False

# ============================================================
# GLOBALS / FILES / LOCKS
# ============================================================

APPDATA = os.path.join(os.path.expanduser("~"), ".sentinel_edr")
os.makedirs(APPDATA, exist_ok=True)

LISTS_FILE = os.path.join(APPDATA, "lists.json")
LOG_FILE = os.path.join(APPDATA, "sentinel.log")
BEHAVIOR_FILE = os.path.join(APPDATA, "behavior.json")
STATS_FILE = os.path.join(APPDATA, "stats.json")
CAMPAIGN_FILE = os.path.join(APPDATA, "campaigns.json")

ALLOWLIST = set()
BLOCKLIST = set()
RADIOACTIVE = set()
WHITELIST_NAMES = set()

BEHAVIOR_DB = {}
STATS_DB = {}
CAMPAIGN_DB = {}

LIST_LOCK = threading.Lock()
BEHAVIOR_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()
CAMPAIGN_LOCK = threading.Lock()

BASE_LOCKDOWN_SCORE_THRESHOLD = 70
MIN_LOCKDOWN_THRESHOLD = 40
MAX_LOCKDOWN_THRESHOLD = 90

CHECK_INTERVAL = 2.0
MAX_SCORE_HISTORY = 50
MAX_CPU_HISTORY = 50

# ============================================================
# SIMPLE LINEAGE GRAPH
# ============================================================

class LineageGraph:
    def __init__(self):
        self.lock = threading.Lock()
        self.nodes = {}  # pid -> {name, path, score, ts}
        self.edges = []  # (parent, child)

    def update_process(self, pid, ppid, name, path, score, ts):
        with self.lock:
            self.nodes[pid] = {
                "name": name,
                "path": path,
                "score": score,
                "ts": ts,
            }
            if ppid and ppid != 0:
                self.edges.append((ppid, pid))

    def prune_old(self, now, max_age=3600):
        with self.lock:
            to_del = [pid for pid, n in self.nodes.items() if now - n["ts"] > max_age]
            for pid in to_del:
                self.nodes.pop(pid, None)
            self.edges = [(p, c) for (p, c) in self.edges if p in self.nodes and c in self.nodes]

    def get_subgraph_for_pid(self, pid, depth=3):
        with self.lock:
            if pid not in self.nodes:
                return {}, []
            nodes = {}
            edges = []
            frontier = [(pid, 0)]
            visited = set()
            while frontier:
                cur, d = frontier.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                if cur in self.nodes:
                    nodes[cur] = self.nodes[cur]
                if d >= depth:
                    continue
                for (p, c) in self.edges:
                    if p == cur and c not in visited:
                        edges.append((p, c))
                        frontier.append((c, d + 1))
                    if c == cur and p not in visited:
                        edges.append((p, c))
                        frontier.append((p, d + 1))
            return nodes, edges


LINEAGE_GRAPH = LineageGraph()

# ============================================================
# UTILS
# ============================================================

def norm_path(p):
    if not p:
        return p
    try:
        return os.path.normcase(os.path.abspath(p))
    except Exception:
        return p

def log_event(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(line, end="")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass

def load_lists():
    global ALLOWLIST, BLOCKLIST, RADIOACTIVE, WHITELIST_NAMES
    with LIST_LOCK:
        if not os.path.exists(LISTS_FILE):
            save_lists()
            return
        try:
            with open(LISTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            save_lists()
            return
        ALLOWLIST = set(data.get("allow", []))
        BLOCKLIST = set(data.get("block", []))
        RADIOACTIVE = set(data.get("radio", []))
        WHITELIST_NAMES = set(data.get("whitelist_names", []))

def save_lists():
    with LIST_LOCK:
        data = {
            "allow": sorted(ALLOWLIST),
            "block": sorted(BLOCKLIST),
            "radio": sorted(RADIOACTIVE),
            "whitelist_names": sorted(WHITELIST_NAMES),
        }
        try:
            with open(LISTS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

def load_behavior():
    global BEHAVIOR_DB
    with BEHAVIOR_LOCK:
        if not os.path.exists(BEHAVIOR_FILE):
            save_behavior()
            return
        try:
            with open(BEHAVIOR_FILE, "r", encoding="utf-8") as f:
                BEHAVIOR_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            BEHAVIOR_DB = {}
            save_behavior()

def save_behavior():
    with BEHAVIOR_LOCK:
        try:
            with open(BEHAVIOR_FILE, "w", encoding="utf-8") as f:
                json.dump(BEHAVIOR_DB, f, indent=2)
        except OSError:
            pass

def load_stats():
    global STATS_DB
    with STATS_LOCK:
        if not os.path.exists(STATS_FILE):
            save_stats()
            return
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                STATS_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            STATS_DB = {}
            save_stats()

def save_stats():
    with STATS_LOCK:
        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(STATS_DB, f, indent=2)
        except OSError:
            pass

def load_campaigns():
    global CAMPAIGN_DB
    with CAMPAIGN_LOCK:
        if not os.path.exists(CAMPAIGN_FILE):
            save_campaigns()
            return
        try:
            with open(CAMPAIGN_FILE, "r", encoding="utf-8") as f:
                CAMPAIGN_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            CAMPAIGN_DB = {}
            save_campaigns()

def save_campaigns():
    with CAMPAIGN_LOCK:
        try:
            with open(CAMPAIGN_FILE, "w", encoding="utf-8") as f:
                json.dump(CAMPAIGN_DB, f, indent=2)
        except OSError:
            pass

# ============================================================
# GPU-ACCELERATED ANOMALY DETECTION
# ============================================================

class AnomalyAutoencoder(nn.Module if nn is not None else object):
    def __init__(self, in_dim=16, hidden=8):
        if nn is not None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden, in_dim),
            )
        else:
            pass

    def forward(self, x):
        if nn is None:
            return x
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class GPUAnomalyDetector:
    def __init__(self, in_dim=16, model_path=None, device=None):
        self.enabled = torch is not None and nn is not None
        if not self.enabled:
            self.device = None
            self.model = None
            return
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnomalyAutoencoder(in_dim=in_dim).to(self.device)
        self.model.eval()
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                log_event(f"GPU anomaly model loaded from {model_path}")
            except Exception as e:
                log_event(f"Failed to load GPU anomaly model: {e}")

    def featurize(self, info, behavior_record):
        feats = [
            float(info["cpu"]),
            float(info.get("score", 0.0)),
            float(behavior_record.get("cpu_baseline_mean") or 0.0),
            float(behavior_record.get("cpu_baseline_std") or 0.0),
            float(behavior_record.get("spawn_rate_current") or 0.0),
            float(behavior_record.get("launch_count") or 0.0),
            float(behavior_record.get("max_cpu") or 0.0),
            float(behavior_record.get("total_cpu") or 0.0),
        ]
        while len(feats) < 16:
            feats.append(0.0)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return x

    @torch.no_grad()
    def anomaly_score(self, info, behavior_record):
        if not self.enabled or self.model is None:
            return 0.0
        x = self.featurize(info, behavior_record)
        recon = self.model(x)
        loss = torch.mean((x - recon) ** 2).item()
        return loss

# ============================================================
# SCORING PIPELINE (with time-series baselines + GPU anomaly)
# ============================================================

class ScoringPipeline:
    @staticmethod
    def cluster_and_campaign_boosts(features, cluster_stats, campaign_stats, anomalies):
        cluster_key = features.get("cluster_key")
        cluster_boost = 0
        sibling_burst = False
        campaign_hot = False

        if cluster_key and cluster_key in cluster_stats:
            c = cluster_stats[cluster_key]
            if c.get("recent_members", 0) >= 3:
                cluster_boost += 10
                sibling_burst = True
                anomalies.append("Sibling burst in cluster")
            if c.get("bad_count", 0) >= 2:
                cluster_boost += 10
                anomalies.append("Cluster has multiple bad members")

        if cluster_key and cluster_key in campaign_stats:
            camp = campaign_stats[cluster_key]
            sev = camp.get("severity", 0)
            if sev >= 3:
                cluster_boost += 15
                campaign_hot = True
                anomalies.append("Campaign severity HIGH")
            elif sev == 2:
                cluster_boost += 8
                campaign_hot = True
                anomalies.append("Campaign severity MEDIUM")

        return cluster_boost, sibling_burst, campaign_hot

    @staticmethod
    def momentum_boost(score, score_history, anomalies):
        if not score_history:
            return score, 0.0
        avg = sum(score_history) / len(score_history)
        momentum = score - avg
        if momentum > 10:
            score += 5
            anomalies.append("Positive momentum vs history")
        elif momentum < -10:
            score -= 5
            anomalies.append("Negative momentum vs history")
        return score, momentum

    @staticmethod
    def deviation_scoring(score, behavior_record, info, now, anomalies):
        cpu = info["cpu"]
        mean = behavior_record.get("cpu_baseline_mean")
        std = behavior_record.get("cpu_baseline_std")
        if mean is not None and std is not None and std > 0:
            z = (cpu - mean) / std
            if z > 2.5:
                score += 8
                anomalies.append(f"CPU spike vs baseline (z={z:.1f})")
            elif z > 1.5:
                score += 4
                anomalies.append(f"CPU elevated vs baseline (z={z:.1f})")

        spawn_rate = behavior_record.get("spawn_rate_current")
        spawn_mean = behavior_record.get("spawn_baseline_mean")
        spawn_std = behavior_record.get("spawn_baseline_std")
        if spawn_rate is not None and spawn_mean is not None and spawn_std is not None and spawn_std > 0:
            z_s = (spawn_rate - spawn_mean) / spawn_std
            if z_s > 2.5:
                score += 8
                anomalies.append(f"Spawn rate spike vs baseline (z={z_s:.1f})")
            elif z_s > 1.5:
                score += 4
                anomalies.append(f"Spawn rate elevated vs baseline (z={z_s:.1f})")

        return score

    @staticmethod
    def compute_fingerprint(features, behavior_record, extra_features):
        fp = []
        fp.append(f"parent_type={behavior_record.get('parent_type', 'unknown')}")
        fp.append(f"has_network={behavior_record.get('has_network', False)}")
        fp.append(f"cluster_dir={behavior_record.get('cluster_dir', '')}")
        fp.append(f"cluster_pattern={behavior_record.get('cluster_name_pattern', '')}")
        if extra_features.get("sibling_burst"):
            fp.append("sibling_burst")
        if extra_features.get("campaign_hot"):
            fp.append("campaign_hot")
        if behavior_record.get("cpu_baseline_mean") is not None:
            fp.append("has_cpu_baseline")
        if behavior_record.get("spawn_baseline_mean") is not None:
            fp.append("has_spawn_baseline")
        return fp

    @staticmethod
    def finalize_score(score):
        if score >= 90:
            threat = "HIGH"
        elif score >= 60:
            threat = "MEDIUM"
        elif score >= 30:
            threat = "LOW"
        else:
            threat = "LOW"
        return score, threat

    @staticmethod
    def compute_score(info, behavior_record, bad_fingerprints, cluster_stats, campaign_stats, now, gpu_detector=None):
        anomalies = []
        extra_features = {}

        features = {
            "name": info["name_l"],
            "path": info["path"],
            "cpu": info["cpu"],
            "age_seconds": now - behavior_record.get("first_seen", now),
            "cluster_key": behavior_record.get("cluster_key", ""),
        }

        score = 0

        if info["status"] == "BLOCKED":
            score += 100
            anomalies.append("Explicit blocklist")
        elif info["status"] == "RADIOACTIVE":
            score += 70
            anomalies.append("Radioactive path")
        elif info["status"] == "SUSPICIOUS":
            score += 30
            anomalies.append("Suspicious default classification")

        if behavior_record.get("has_network", False):
            score += 10
            anomalies.append("Has network activity")

        if behavior_record.get("parent_type") in ("browser", "office"):
            score += 10
            anomalies.append(f"Spawned from {behavior_record.get('parent_type')}")

        cluster_boost, sibling_burst, campaign_hot = ScoringPipeline.cluster_and_campaign_boosts(
            features, cluster_stats, campaign_stats, anomalies
        )
        score += cluster_boost
        if sibling_burst:
            extra_features["sibling_burst"] = True
        if campaign_hot:
            extra_features["campaign_hot"] = True

        score = ScoringPipeline.deviation_scoring(score, behavior_record, info, now, anomalies)

        score_history = behavior_record.get("score_history", []) if behavior_record else []
        score, momentum = ScoringPipeline.momentum_boost(score, score_history, anomalies)

        if gpu_detector is not None and gpu_detector.enabled:
            try:
                ae_score = gpu_detector.anomaly_score(info, behavior_record)
                if ae_score > 0.02:
                    score += 10
                    anomalies.append(f"GPU anomaly score high ({ae_score:.4f})")
                elif ae_score > 0.01:
                    score += 5
                    anomalies.append(f"GPU anomaly score elevated ({ae_score:.4f})")
            except Exception as e:
                anomalies.append(f"GPU anomaly error: {e}")

        fp = ScoringPipeline.compute_fingerprint(features, behavior_record, extra_features)

        matched_bad_fp = False
        for bad_fp in bad_fingerprints:
            overlap = len(set(fp) & set(bad_fp))
            if overlap >= 3:
                score += 25
                anomalies.append("Matched bad fingerprint from previous block")
                matched_bad_fp = True
                break

        score, threat = ScoringPipeline.finalize_score(score)

        return score, threat, anomalies, fp, matched_bad_fp, momentum, features["age_seconds"], features["cluster_key"]

# ============================================================
# SENTINEL BRAIN / CAMPAIGNS
# ============================================================

class SentinelBrain:
    def classify(self, proc):
        try:
            name = proc.name()
            exe_raw = proc.exe() or ""
            exe = norm_path(exe_raw) or f"<no-path:{proc.pid}>"
            name_l = name.lower()
            cpu = proc.cpu_percent(interval=None)
            ppid = proc.ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

        if exe in BLOCKLIST or name_l in BLOCKLIST:
            status = "BLOCKED"
        elif exe in ALLOWLIST:
            status = "ALLOW"
        elif exe in RADIOACTIVE:
            status = "RADIOACTIVE"
        elif name_l in WHITELIST_NAMES:
            status = "TRUSTED"
        else:
            status = "SUSPICIOUS"

        threat = self.threat_level(status)

        return {
            "pid": proc.pid,
            "ppid": ppid,
            "name": name,
            "name_l": name_l,
            "cpu": cpu,
            "path": exe,
            "status": status,
            "threat": threat,
            "score": 0,
            "anomalies": [],
            "fingerprint": [],
            "matched_bad_fp": False,
            "momentum": 0.0,
            "age_seconds": 0.0,
            "cluster_stats": {},
            "campaign_id": None,
        }

    @staticmethod
    def threat_level(status):
        if status in ("BLOCKED", "RADIOACTIVE"):
            return "HIGH"
        if status == "SUSPICIOUS":
            return "MEDIUM"
        if status in ("TRUSTED", "ALLOW"):
            return "LOW"
        return "UNKNOWN"

class CampaignEngine:
    @staticmethod
    def update_campaigns(cluster_stats, now):
        with CAMPAIGN_LOCK:
            for ckey, c in cluster_stats.items():
                camp = CAMPAIGN_DB.get(ckey, {
                    "first_seen": now,
                    "last_seen": now,
                    "events": 0,
                    "bad_events": 0,
                    "severity": 0,
                })
                camp["last_seen"] = now
                camp["events"] += c["members"]
                camp["bad_events"] += c["bad_count"]

                severity = 0
                if c["bad_count"] >= 3 or c["avg_score"] >= 80:
                    severity = 3
                elif c["bad_count"] >= 2 or c["avg_score"] >= 60:
                    severity = 2
                elif c["bad_count"] >= 1 or c["avg_score"] >= 40:
                    severity = 1
                camp["severity"] = max(camp["severity"], severity)

                CAMPAIGN_DB[ckey] = camp

            to_del = []
            for ckey, camp in CAMPAIGN_DB.items():
                if now - camp["last_seen"] > 3600:
                    to_del.append(ckey)
            for ckey in to_del:
                CAMPAIGN_DB.pop(ckey, None)

        save_campaigns()

    @staticmethod
    def get_campaign_stats():
        with CAMPAIGN_LOCK:
            stats = {}
            for ckey, camp in CAMPAIGN_DB.items():
                stats[ckey] = {
                    "severity": camp.get("severity", 0),
                    "events": camp.get("events", 0),
                    "bad_events": camp.get("bad_events", 0),
                }
            return stats

# ============================================================
# STORAGE MANAGER
# ============================================================

class StorageManager:
    def __init__(self):
        self.manual_path = None
        self.active_path = None

    def list_local_drives(self):
        drives = []
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for i in range(26):
            if bitmask & (1 << i):
                drive = f"{chr(65+i)}:\\"
                drives.append(drive)
        return drives

    def set_manual_path(self, path):
        if self.validate_path(path):
            self.manual_path = path
            self.active_path = path
            return True
        return False

    def validate_path(self, path):
        return os.path.exists(path)

    def auto_select(self):
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            p = f"{letter}:\\"
            if os.path.exists(p):
                self.active_path = p
                return p
        self.active_path = "C:\\"
        return "C:\\"

# ============================================================
# LLM ENGINES + UNBOUNDED MULTI-LLM MANAGER
# ============================================================

class LocalONNXLlm:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.loaded = False
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[LLM] onnxruntime not installed, local ONNX disabled.")
            return
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.loaded = True
        except Exception as e:
            print(f"[LLM] Failed to load ONNX model: {e}")
            self.loaded = False

    def ask(self, prompt, context=None):
        if not self.loaded:
            return "[Local ONNX] Model not loaded."
        return "[Local ONNX] Inference placeholder for prompt: " + prompt[:200]

class CopilotEngine:
    def ask(self, prompt, context=None):
        return "[Copilot] " + prompt[:400]

class LLMEngineManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.engines = {}
        self.engine_meta = {}
        self.roles = {}
        self.tiers = {}
        self.active_engine = None
        self.active_engines = []
        self.fusion_engine_name = None

    def register_engine(self, name, engine, roles=None, tier=None, is_fusion=False):
        if roles is None:
            roles = []
        with self.lock:
            self.engines[name] = engine
            self.engine_meta[name] = {"roles": roles, "tier": tier}
            for r in roles:
                self.roles.setdefault(r, []).append(name)
            if tier:
                self.tiers.setdefault(tier, []).append(name)
            if self.active_engine is None:
                self.active_engine = name
            if is_fusion or self.fusion_engine_name is None:
                self.fusion_engine_name = name

    def list_models(self):
        with self.lock:
            return sorted(self.engines.keys())

    def set_active(self, name):
        with self.lock:
            if name in self.engines:
                self.active_engine = name

    def set_active_many(self, names):
        with self.lock:
            self.active_engines = [n for n in names if n in self.engines]

    def _ask_single(self, name, prompt, context=None):
        engine = self.engines.get(name)
        if engine is None:
            return f"[{name}] ERROR: engine not registered"
        try:
            return engine.ask(prompt, context=context)
        except Exception as e:
            return f"[{name}] ERROR: {e}"

    def ask(self, prompt, context=None):
        with self.lock:
            name = self.active_engine
        if not name:
            return "[LLM] No active engine configured."
        return self._ask_single(name, prompt, context=context)

    def ask_many(self, prompt, context=None, engines=None):
        with self.lock:
            if engines is None or not engines:
                engines = list(self.active_engines) or ([self.active_engine] if self.active_engine else [])
        results = {}
        threads = []

        def worker(name):
            resp = self._ask_single(name, prompt, context=context)
            results[name] = resp

        for name in engines:
            t = threading.Thread(target=worker, args=(name,))
            t.daemon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        return results

    def ask_role(self, role, prompt, context=None):
        with self.lock:
            engines = list(self.roles.get(role, []))
        if not engines:
            return {}
        return self.ask_many(prompt, context=context, engines=engines)

    def ask_tier(self, tier, prompt, context=None):
        with self.lock:
            engines = list(self.tiers.get(tier, []))
        if not engines:
            return {}
        return self.ask_many(prompt, context=context, engines=engines)

    def fuse(self, multi_results, task_description=""):
        with self.lock:
            fusion_name = self.fusion_engine_name
        if not fusion_name:
            return "[Fusion] No fusion engine configured."
        fusion_prompt = (
            "You are a meta-analyst LLM coordinating multiple security models.\n"
            "You are given responses from several different LLM engines about the same security task.\n"
            "Task description:\n"
            f"{task_description}\n\n"
            "Here are the raw responses (JSON):\n"
            f"{json.dumps(multi_results, indent=2)}\n\n"
            "Please provide:\n"
            "1) Consensus assessment (if any).\n"
            "2) Key disagreements.\n"
            "3) Most likely correct reasoning.\n"
            "4) Recommended action for an EDR/XDR analyst.\n"
        )
        return self._ask_single(fusion_name, fusion_prompt, context=None)

# ============================================================
# STREAMING HUB (WebSocket) – asyncio compatibility shim
# ============================================================

class StreamingHub:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.thread = None
        self.server = None

        if websockets is not None:
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            log_event(f"StreamingHub starting on ws://{host}:{port}")
        else:
            log_event("StreamingHub disabled (websockets not installed).")

    async def _start_server(self):
        """
        Compatibility shim: websockets>=12 calls asyncio.get_running_loop()
        inside Server.__init__, which requires a running loop.

        By awaiting websockets.serve() *inside* this coroutine, which is
        executed via run_until_complete(), we ensure the loop is running
        when websockets asks for it.
        """
        self.server = await websockets.serve(self._handler, self.host, self.port)
        log_event("StreamingHub WebSocket server started.")

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            # Start server inside running loop (compat shim)
            self.loop.run_until_complete(self._start_server())
            self.loop.run_forever()
        except Exception as e:
            log_event(f"StreamingHub loop error: {e}")
        finally:
            try:
                if self.server is not None:
                    self.server.close()
                    self.loop.run_until_complete(self.server.wait_closed())
            except Exception:
                pass
            try:
                self.loop.close()
            except Exception:
                pass
            log_event("StreamingHub loop stopped.")

    async def _handler(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    def broadcast(self, payload: dict):
        if self.loop is None:
            return

        msg = json.dumps(payload)

        async def _send():
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send(), self.loop)
        except RuntimeError:
            # Loop not running yet or already closed – ignore
            pass

# ============================================================
# LLM-DRIVEN AUTO-PATCHING
# ============================================================

class AutoPatcher:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

    def propose_patch(self, recent_events, behavior_db_snapshot):
        prompt = (
            "You are an EDR tuning assistant.\n"
            "Given recent events and behavior records, propose SAFE tuning changes:\n"
            "- which anomalies look like false positives\n"
            "- which paths should be allowlisted\n"
            "- which thresholds could be adjusted\n"
            "Return JSON with keys: allowlist_paths, lower_threshold, raise_threshold, notes.\n\n"
            f"Recent_events: {json.dumps(recent_events, indent=2)}\n\n"
            f"Behavior_snapshot: {json.dumps(behavior_db_snapshot, indent=2)[:8000]}\n"
        )
        resp = self.llm_manager.ask_role("tuning", prompt)
        if not resp:
            resp = {"default": self.llm_manager.ask(prompt)}
        text = next(iter(resp.values())) if isinstance(resp, dict) else resp
        try:
            start = text.find("{")
            end = text.rfind("}")
            data = json.loads(text[start:end+1])
            return data
        except Exception:
            return None

    def apply_patch(self, patch):
        changed = False
        for p in patch.get("allowlist_paths", []):
            np = norm_path(p)
            if np and np not in ALLOWLIST:
                ALLOWLIST.add(np)
                BLOCKLIST.discard(np)
                RADIOACTIVE.discard(np)
                changed = True
                log_event(f"AUTOPATCH: allowlisted {np}")
        if changed:
            save_lists()

# ============================================================
# ENGINE CORE (EDR/XDR + SENTINEL)
# ============================================================

class EngineCore:
    def __init__(self, db_path="engine.db", llm_manager=None):
        self.db_path = db_path
        self.event_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.graph_lock = threading.Lock()
        self.running = True
        self.llm_manager = llm_manager or LLMEngineManager()
        self.storage = StorageManager()
        self.storage.auto_select()

        self.brain = SentinelBrain()
        self.dynamic_lockdown_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD
        self.lockdown_enabled = False
        self.simulate_kills = False

        self.current_rows = []
        self.current_net_rows = []

        self.db_lock = threading.Lock()
        self._init_db()

        self.gpu_anomaly = GPUAnomalyDetector(in_dim=16, model_path="gpu_ae.pt")

        self.streaming = StreamingHub()

        self.auto_patcher = AutoPatcher(self.llm_manager)
        self._last_autopatch_ts = 0

        self._start_threads()

    def _init_db(self):
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=10.0,
            )
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    host TEXT,
                    type TEXT,
                    details TEXT,
                    score REAL DEFAULT 0.0
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    severity TEXT,
                    rule TEXT,
                    score REAL,
                    summary TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    host TEXT,
                    label TEXT,
                    features BLOB
                )
            """)
            self.conn.commit()
        except Exception:
            backup = Path(self.db_path + ".corrupt")
            if Path(self.db_path).exists():
                Path(self.db_path).rename(backup)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)

    def _start_threads(self):
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
        threading.Thread(target=self._distributed_sync_loop, daemon=True).start()
        threading.Thread(target=self._db_worker_loop, daemon=True).start()

    def _db_worker_loop(self):
        while self.running:
            try:
                item = self.event_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                continue
            kind = item.get("kind")
            try:
                if kind == "event":
                    self._db_insert_event(item["data"])
                elif kind == "alert":
                    self._db_insert_alert(item["data"])
            except sqlite3.OperationalError as e:
                log_event(f"DB worker error: {e}")
                time.sleep(0.5)

    def _db_insert_event(self, info):
        with self.db_lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO events (ts, host, type, details, score) VALUES (?, ?, ?, ?, ?)",
                (
                    info["ts"],
                    info["host"],
                    info["type"],
                    info["details"],
                    float(info["score"]),
                ),
            )
            self.conn.commit()

    def _db_insert_alert(self, alert):
        with self.db_lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO alerts (ts, severity, rule, score, summary) VALUES (?, ?, ?, ?, ?)",
                (alert["ts"], alert["severity"], alert["rule"], alert["score"], alert["summary"])
            )
            self.conn.commit()

    def _monitor_loop(self):
        while self.running:
            updated = False
            rows = []
            net_rows = []
            proc_cache = {}
            pid_to_key = {}

            now = time.time()
            bad_fps = self.collect_bad_fingerprints()
            cluster_stats = self.compute_cluster_stats(now)
            CampaignEngine.update_campaigns(cluster_stats, now)
            campaign_stats = CampaignEngine.get_campaign_stats()

            for proc in psutil.process_iter():
                try:
                    info = self.brain.classify(proc)
                    if not info:
                        continue

                    self.update_behavior(proc, info, bad_fps, cluster_stats, campaign_stats, now)

                    key = self._key_for_behavior(info["path"], info["name_l"])
                    pid_to_key[info["pid"]] = key

                    proc_cache[proc.pid] = info["name"]

                    if self.auto_update_lists(info):
                        updated = True

                    self.enforce_lockdown(info)

                    rows.append(info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            try:
                for c in psutil.net_connections(kind="inet"):
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else ""
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else ""
                    status = c.status
                    pid = c.pid or 0
                    name = proc_cache.get(pid, "")
                    net_rows.append({
                        "laddr": laddr,
                        "raddr": raddr,
                        "status": status,
                        "pid": pid,
                        "name": name,
                    })
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass

            self.mark_network_activity(pid_to_key)
            save_behavior()

            self.update_stats_and_threshold(rows)
            LINEAGE_GRAPH.prune_old(now)

            self.current_rows = rows
            self.current_net_rows = net_rows

            time.sleep(CHECK_INTERVAL)

    def _key_for_behavior(self, path, name_l):
        if path and not path.startswith("<no-path:"):
            return path
        return name_l

    @staticmethod
    def name_pattern(name):
        pattern = []
        for c in name:
            if c.isalpha():
                pattern.append("A")
            elif c.isdigit():
                pattern.append("9")
            else:
                pattern.append("_")
        return "".join(pattern)

    def _update_cpu_baseline(self, rec, cpu):
        n = rec.get("cpu_baseline_count", 0)
        mean = rec.get("cpu_baseline_mean", 0.0)
        m2 = rec.get("cpu_baseline_m2", 0.0)

        n += 1
        delta = cpu - mean
        mean += delta / n
        delta2 = cpu - mean
        m2 += delta * delta2

        rec["cpu_baseline_count"] = n
        rec["cpu_baseline_mean"] = mean
        rec["cpu_baseline_m2"] = m2
        if n > 1:
            rec["cpu_baseline_std"] = math.sqrt(m2 / (n - 1))
        else:
            rec["cpu_baseline_std"] = None

    def _update_spawn_baseline(self, rec, now):
        spawn_times = rec.get("spawn_times", [])
        spawn_times.append(now)
        cutoff = now - 600
        spawn_times = [t for t in spawn_times if t >= cutoff]
        rec["spawn_times"] = spawn_times

        current_rate = len([t for t in spawn_times if now - t <= 60])
        rec["spawn_rate_current"] = current_rate

        n = rec.get("spawn_baseline_count", 0)
        mean = rec.get("spawn_baseline_mean", 0.0)
        m2 = rec.get("spawn_baseline_m2", 0.0)

        n += 1
        delta = current_rate - mean
        mean += delta / n
        delta2 = current_rate - mean
        m2 += delta * delta2

        rec["spawn_baseline_count"] = n
        rec["spawn_baseline_mean"] = mean
        rec["spawn_baseline_m2"] = m2
        if n > 1:
            rec["spawn_baseline_std"] = math.sqrt(m2 / (n - 1))
        else:
            rec["spawn_baseline_std"] = None

    def update_behavior(self, proc, info, bad_fingerprints, cluster_stats, campaign_stats, now):
        path = info["path"]
        name_l = info["name_l"]
        key = self._key_for_behavior(path, name_l)

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {
                "first_seen": now,
                "last_seen": now,
                "launch_count": 0,
                "total_cpu": 0.0,
                "max_cpu": 0.0,
                "has_network": False,
                "parent_pids": [],
                "user_label": None,
                "last_score": 0,
                "fingerprints": [],
                "cluster_dir": "",
                "cluster_name_pattern": "",
                "cluster_key": "",
                "score_history": [],
                "cpu_history": [],
                "file_size": None,
                "file_mtime": None,
                "parent_type": None,
                "cpu_baseline_count": 0,
                "cpu_baseline_mean": 0.0,
                "cpu_baseline_m2": 0.0,
                "cpu_baseline_std": None,
                "spawn_times": [],
                "spawn_baseline_count": 0,
                "spawn_baseline_mean": 0.0,
                "spawn_baseline_m2": 0.0,
                "spawn_baseline_std": None,
                "spawn_rate_current": 0.0,
            })

            if rec.get("file_size") is None and not path.startswith("<no-path:"):
                try:
                    st = os.stat(path)
                    rec["file_size"] = st.st_size
                    rec["file_mtime"] = st.st_mtime
                except OSError:
                    rec["file_size"] = None
                    rec["file_mtime"] = None

            rec["last_seen"] = now
            rec["launch_count"] = rec.get("launch_count", 0) + 1
            rec["total_cpu"] = rec.get("total_cpu", 0.0) + info["cpu"]
            rec["max_cpu"] = max(rec.get("max_cpu", 0.0), info["cpu"])

            parent_pids = set(rec.get("parent_pids", []))
            parent_pids.add(info["ppid"])
            rec["parent_pids"] = list(parent_pids)

            parent_type = rec.get("parent_type")
            if not parent_type:
                try:
                    pproc = psutil.Process(info["ppid"])
                    pname = pproc.name().lower()
                    if any(x in pname for x in ["chrome", "edge", "firefox", "brave", "opera"]):
                        parent_type = "browser"
                    elif any(x in pname for x in ["winword", "excel", "powerpnt"]):
                        parent_type = "office"
                    elif "explorer" in pname:
                        parent_type = "shell"
                    else:
                        parent_type = "other"
                    rec["parent_type"] = parent_type
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    rec["parent_type"] = "unknown"

            if not path.startswith("<no-path:"):
                rec["cluster_dir"] = os.path.dirname(path)
            base_name = os.path.basename(path) or name_l
            rec["cluster_name_pattern"] = self.name_pattern(base_name)
            rec["cluster_key"] = f"{rec['cluster_dir']}|{rec['cluster_name_pattern']}|{rec['parent_type']}"

            self._update_cpu_baseline(rec, info["cpu"])
            self._update_spawn_baseline(rec, now)

            BEHAVIOR_DB[key] = rec

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            score, threat, anomalies, fp, matched_bad_fp, momentum, age_seconds, cluster_key = \
                ScoringPipeline.compute_score(
                    info, rec, bad_fingerprints, cluster_stats, campaign_stats, now,
                    gpu_detector=self.gpu_anomaly
                )

            rec["last_score"] = score

            sh = rec.get("score_history", [])
            sh.append(score)
            if len(sh) > MAX_SCORE_HISTORY:
                sh = sh[-MAX_SCORE_HISTORY:]
            rec["score_history"] = sh

            ch = rec.get("cpu_history", [])
            ch.append(info["cpu"])
            if len(ch) > MAX_CPU_HISTORY:
                ch = ch[-MAX_CPU_HISTORY:]
            rec["cpu_history"] = ch

            fps = set(tuple(x) for x in rec.get("fingerprints", []))
            fps.add(tuple(fp))
            rec["fingerprints"] = [list(x) for x in fps]

            BEHAVIOR_DB[key] = rec

        info["score"] = score
        info["threat"] = threat
        info["anomalies"] = anomalies
        info["fingerprint"] = fp
        info["matched_bad_fp"] = matched_bad_fp
        info["momentum"] = momentum
        info["age_seconds"] = age_seconds
        info["cluster_stats"] = cluster_stats.get(cluster_key, {})
        info["campaign_id"] = cluster_key

        LINEAGE_GRAPH.update_process(info["pid"], info["ppid"], info["name"], info["path"], score, now)

        self._store_event_from_info(info)

    def mark_network_activity(self, pid_to_key):
        with BEHAVIOR_LOCK:
            for key in pid_to_key.values():
                rec = BEHAVIOR_DB.get(key)
                if not rec:
                    continue
                if not rec.get("has_network", False):
                    rec["has_network"] = True
                    BEHAVIOR_DB[key] = rec

    def collect_bad_fingerprints(self):
        bad_fps = []
        with BEHAVIOR_LOCK:
            for key, rec in BEHAVIOR_DB.items():
                label = rec.get("user_label")
                last_score = rec.get("last_score", 0)
                if label in ("block", "radio") or last_score >= 80:
                    for fp in rec.get("fingerprints", []):
                        if fp:
                            bad_fps.append(fp)
        return bad_fps

    def compute_cluster_stats(self, now):
        clusters = {}
        with BEHAVIOR_LOCK:
            for key, rec in BEHAVIOR_DB.items():
                ckey = rec.get("cluster_key")
                if not ckey:
                    continue
                c = clusters.setdefault(ckey, {
                    "members": 0,
                    "bad_count": 0,
                    "score_sum": 0.0,
                    "score_count": 0,
                    "recent_members": 0,
                })
                c["members"] += 1
                score = rec.get("last_score", 0)
                c["score_sum"] += score
                c["score_count"] += 1
                label = rec.get("user_label")
                if label in ("block", "radio") or score >= 80:
                    c["bad_count"] += 1
                if now - rec.get("last_seen", now) < 60:
                    c["recent_members"] += 1

        for ckey, c in clusters.items():
            if c["score_count"] > 0:
                c["avg_score"] = c["score_sum"] / c["score_count"]
            else:
                c["avg_score"] = 0.0

        return clusters

    def auto_update_lists(self, info):
        p = info["path"]
        if not p or p.startswith("<no-path:"):
            return False

        np = norm_path(p)
        changed = False

        if info["status"] == "TRUSTED" and np not in ALLOWLIST:
            ALLOWLIST.add(np)
            RADIOACTIVE.discard(np)
            BLOCKLIST.discard(np)
            changed = True
            log_event(f"AUTO: Trusted path added to ALLOWLIST: {np}")

        elif info["status"] == "SUSPICIOUS" and np not in RADIOACTIVE and np not in ALLOWLIST:
            try:
                proc = psutil.Process(info["pid"])
                if proc.create_time() > time.time() - 3:
                    return changed
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

            RADIOACTIVE.add(np)
            changed = True
            log_event(f"AUTO: Suspicious path added to RADIOACTIVE: {np}")

        if changed:
            save_lists()
        return changed

    def enforce_lockdown(self, info):
        if not self.lockdown_enabled:
            return

        status = info["status"]
        path = info["path"]
        pid = info["pid"]
        score = info.get("score", 0)
        anomalies = info.get("anomalies", [])
        matched_bad_fp = info.get("matched_bad_fp", False)

        if status in ("ALLOW", "TRUSTED"):
            return

        np = norm_path(path) if path and not path.startswith("<no-path:") else None
        if np and np in ALLOWLIST:
            return

        if score < self.dynamic_lockdown_threshold and status not in ("BLOCKED", "RADIOACTIVE"):
            return

        reasons = ", ".join(anomalies) if anomalies else "no anomalies listed"

        if self.simulate_kills:
            log_event(
                f"SIMULATED KILL: would have terminated PID {pid} ({info['name']}) "
                f"path={path} status={status} score={score} "
                f"threshold={self.dynamic_lockdown_threshold} reasons=[{reasons}]"
            )
            return

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            msg = (
                f"LOCKDOWN: Terminated PID {pid} ({info['name']}) "
                f"path={path} status={status} score={score} "
                f"threshold={self.dynamic_lockdown_threshold}"
            )
            if matched_bad_fp:
                msg += " (Matched bad fingerprint from previous block)"
            log_event(msg)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            log_event(f"LOCKDOWN: Failed to terminate PID {pid}: {e}")

    def update_stats_and_threshold(self, rows):
        ts = time.time()
        proc_count = len(rows)
        high_count = sum(1 for r in rows if r["threat"] == "HIGH")
        scores = [r["score"] for r in rows]
        avg_score = statistics.mean(scores) if scores else 0.0

        with STATS_LOCK:
            hist = STATS_DB.get("history", [])
            hist.append({
                "ts": ts,
                "proc_count": proc_count,
                "high_count": high_count,
                "avg_score": avg_score,
            })
            if len(hist) > 200:
                hist = hist[-200:]
            STATS_DB["history"] = hist

        save_stats()

        with STATS_LOCK:
            hist = STATS_DB.get("history", [])
            if len(hist) < 20:
                self.dynamic_lockdown_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD
                return

            high_vals = [h["high_count"] for h in hist]
            avg_vals = [h["avg_score"] for h in hist]

            high_avg = statistics.mean(high_vals)
            high_std = statistics.pstdev(high_vals) if len(high_vals) > 1 else 0.0

            current_high = high_count
            hostile_factor = 0.0
            if high_std > 0 and current_high > high_avg + 2 * high_std:
                hostile_factor += 0.3
            elif current_high > high_avg + high_std:
                hostile_factor += 0.15

            avg_avg = statistics.mean(avg_vals)
            avg_std = statistics.pstdev(avg_vals) if len(avg_vals) > 1 else 0.0
            current_avg = avg_score
            if avg_std > 0 and current_avg > avg_avg + 2 * avg_std:
                hostile_factor += 0.3
            elif current_avg > avg_avg + avg_std:
                hostile_factor += 0.15

            hostile_factor = min(1.0, hostile_factor)
            new_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD - hostile_factor * 20
            new_threshold = max(MIN_LOCKDOWN_THRESHOLD, min(MAX_LOCKDOWN_THRESHOLD, new_threshold))
            self.dynamic_lockdown_threshold = new_threshold

        if hostile_factor >= 0.5:
            log_event(
                f"GLOBAL ALERT: Hostile pattern detected. "
                f"high_count={high_count}, avg_score={avg_score:.1f}, "
                f"threshold={self.dynamic_lockdown_threshold:.1f}"
            )

    def whatif_counts(self, rows, threshold):
        c = 0
        for r in rows:
            status = r["status"]
            path = r["path"]
            score = r["score"]
            if status in ("ALLOW", "TRUSTED"):
                continue
            np = norm_path(path) if path and not path.startswith("<no-path:") else None
            if np and np in ALLOWLIST:
                continue
            if score >= threshold:
                c += 1
        return c

    def _store_event_from_info(self, info):
        evt = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "host": "local-host",
            "type": "process",
            "details": f"{info['name']} ({info['path']})",
            "score": float(info["score"]),
        }
        self.event_queue.put({"kind": "event", "data": evt})

        self.streaming.broadcast({
            "type": "event",
            "ts": evt["ts"],
            "host": evt["host"],
            "details": evt["details"],
            "score": evt["score"],
        })

        if info["score"] >= self.dynamic_lockdown_threshold:
            alert = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "HIGH" if info["score"] > 80 else "MEDIUM",
                "rule": "SENTINEL_SCORE",
                "score": info["score"],
                "summary": f"{info['name']} ({info['path']})",
            }
            self._store_alert(alert)

    def _store_alert(self, alert):
        self.event_queue.put({"kind": "alert", "data": alert})
        self.streaming.broadcast({
            "type": "alert",
            "ts": alert["ts"],
            "severity": alert["severity"],
            "summary": alert["summary"],
            "score": alert["score"],
        })

    def fetch_recent_events(self, limit=200):
        with self.db_lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT ts, host, type, details, score FROM events ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return cur.fetchall()

    def fetch_recent_alerts(self, limit=200):
        with self.db_lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT ts, severity, rule, score, summary FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return cur.fetchall()

    def fetch_stats_for_heatmap(self):
        with self.db_lock:
            cur = self.conn.cursor()
            cur.execute("SELECT score FROM events")
            scores = [row[0] for row in cur.fetchall()]
            return scores

    def fetch_samples(self, limit=50):
        with self.db_lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT ts, host, label FROM samples ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return cur.fetchall()

    def _distributed_sync_loop(self):
        while self.running:
            time.sleep(5.0)

    def _watchdog_loop(self):
        while self.running:
            time.sleep(5.0)

            if not os.path.exists(LISTS_FILE):
                log_event("WATCHDOG: Lists file missing, recreating.")
                save_lists()
            with LIST_LOCK:
                try:
                    if os.path.exists(LISTS_FILE):
                        with open(LISTS_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Lists file corrupted, resetting.")
                    try:
                        os.remove(LISTS_FILE)
                    except OSError:
                        pass
                    save_lists()

            with BEHAVIOR_LOCK:
                try:
                    if os.path.exists(BEHAVIOR_FILE):
                        with open(BEHAVIOR_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Behavior DB corrupted, resetting.")
                    try:
                        os.remove(BEHAVIOR_FILE)
                    except OSError:
                        pass
                    save_behavior()

            with STATS_LOCK:
                try:
                    if os.path.exists(STATS_FILE):
                        with open(STATS_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Stats DB corrupted, resetting.")
                    try:
                        os.remove(STATS_FILE)
                    except OSError:
                        pass
                    save_stats()

            with CAMPAIGN_LOCK:
                try:
                    if os.path.exists(CAMPAIGN_FILE):
                        with open(CAMPAIGN_FILE, "r", encoding="utf-8") as f:
                            json.load(f)
                except json.JSONDecodeError:
                    log_event("WATCHDOG: Campaign DB corrupted, resetting.")
                    try:
                        os.remove(CAMPAIGN_FILE)
                    except OSError:
                        pass
                    save_campaigns()

            now = time.time()
            if now - self._last_autopatch_ts > 300:
                self._last_autopatch_ts = now
                recent = self.fetch_recent_events(limit=100)
                with BEHAVIOR_LOCK:
                    snapshot = dict(list(BEHAVIOR_DB.items())[:50])
                patch = self.auto_patcher.propose_patch(recent, snapshot)
                if patch:
                    self.auto_patcher.apply_patch(patch)

    def get_campaign_stats(self):
        return CampaignEngine.get_campaign_stats()

# ============================================================
# PYQT5 GUI
# ============================================================

class EDRConsole(QMainWindow):
    def __init__(self, engine: EngineCore):
        super().__init__()
        self.engine = engine
        self.llm_manager = engine.llm_manager

        self.setWindowTitle("Prototype EDR/XDR – Sentinel Console")
        self.setGeometry(50, 50, 1900, 1000)

        self.sort_column = "score"
        self.sort_reverse = True

        self.filter_var = "ALL"
        self.search_text = ""

        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_all)
        self.timer.start(1000)

    def _build_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        tabs.addTab(self.build_sentinel_panel(), "Sentinel – Processes")
        tabs.addTab(self.build_network_panel(), "Network")
        tabs.addTab(self.build_campaign_panel(), "Campaigns")
        tabs.addTab(self.build_alerts_panel(), "Alerts / Kill-chain")
        tabs.addTab(self.build_heatmap_panel(), "Threat Heatmap")
        tabs.addTab(self.build_training_panel(), "Training Pipeline")
        tabs.addTab(self.build_health_panel(), "Engine Health")
        tabs.addTab(self.build_llm_panel(), "LLM Intelligence")
        tabs.addTab(self.build_storage_panel(), "Storage / Drives")
        tabs.addTab(self.build_streaming_panel(), "Streaming Dashboard")

    # --- Sentinel main panel ---

    def build_sentinel_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self.lockdown_label = QLabel("Lockdown: OFF")
        self.lockdown_btn = QPushButton("Toggle Lockdown")
        self.lockdown_btn.clicked.connect(self.on_lockdown_toggle)
        self.simulate_label = QLabel("Simulate kills: OFF")
        self.simulate_btn = QPushButton("Toggle Simulate")
        self.simulate_btn.clicked.connect(self.on_simulate_toggle)
        self.threshold_label = QLabel("Threshold: ?")

        top.addWidget(self.lockdown_label)
        top.addWidget(self.lockdown_btn)
        top.addWidget(self.simulate_label)
        top.addWidget(self.simulate_btn)
        top.addWidget(self.threshold_label)

        top.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["ALL", "HIGH", "MEDIUM", "LOW"])
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        top.addWidget(self.filter_combo)

        top.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self.on_search_changed)
        top.addWidget(self.search_edit)

        layout.addLayout(top)

        whatif = QHBoxLayout()
        self.whatif_50 = QLabel("T50: 0")
        self.whatif_60 = QLabel("T60: 0")
        self.whatif_70 = QLabel("T70: 0")
        whatif.addWidget(QLabel("What-if kills:"))
        whatif.addWidget(self.whatif_50)
        whatif.addWidget(self.whatif_60)
        whatif.addWidget(self.whatif_70)
        layout.addLayout(whatif)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.proc_table = QTableWidget(0, 7)
        self.proc_table.setHorizontalHeaderLabels(
            ["Name", "PID", "CPU", "Status", "Threat", "Score", "Path"]
        )
        self.proc_table.cellDoubleClicked.connect(self.on_proc_double_click)
        left_layout.addWidget(self.proc_table)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        right_layout.addWidget(QLabel("ALLOWLIST"))
        self.allow_list = QListWidget()
        right_layout.addWidget(self.allow_list)

        right_layout.addWidget(QLabel("BLOCKLIST"))
        self.block_list = QListWidget()
        right_layout.addWidget(self.block_list)

        right_layout.addWidget(QLabel("RADIOACTIVE"))
        self.radio_list = QListWidget()
        right_layout.addWidget(self.radio_list)

        btn_remove = QPushButton("Remove Selected From Lists")
        btn_remove.clicked.connect(self.on_remove_from_lists)
        right_layout.addWidget(btn_remove)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        btn_row = QHBoxLayout()
        self.btn_allow = QPushButton("➕ Add Selected to Allowlist")
        self.btn_block = QPushButton("⛔ Add Selected to Blocklist")
        self.btn_radio = QPushButton("☢ Add Selected to Radioactive")
        self.btn_lineage = QPushButton("🧬 Show Lineage (text)")
        self.btn_explain = QPushButton("🧠 Explain / Compare (multi-LLM)")
        self.btn_allow.clicked.connect(self.on_add_allow)
        self.btn_block.clicked.connect(self.on_add_block)
        self.btn_radio.clicked.connect(self.on_add_radio)
        self.btn_lineage.clicked.connect(self.on_show_lineage)
        self.btn_explain.clicked.connect(self.on_explain_selected)
        btn_row.addWidget(self.btn_allow)
        btn_row.addWidget(self.btn_block)
        btn_row.addWidget(self.btn_radio)
        btn_row.addWidget(self.btn_lineage)
        btn_row.addWidget(self.btn_explain)
        layout.addLayout(btn_row)

        return widget

    # --- Network panel ---

    def build_network_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.net_table = QTableWidget(0, 5)
        self.net_table.setHorizontalHeaderLabels(
            ["Local", "Remote", "Status", "PID", "Name"]
        )
        layout.addWidget(self.net_table)
        return widget

    # --- Campaign panel ---

    def build_campaign_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.campaign_table = QTableWidget(0, 4)
        self.campaign_table.setHorizontalHeaderLabels(
            ["Cluster Key", "Severity", "Events", "Bad Events"]
        )
        layout.addWidget(self.campaign_table)

        self.campaign_heatmap_text = QTextEdit()
        self.campaign_heatmap_text.setReadOnly(True)
        layout.addWidget(QLabel("Campaign Intelligence / Heatmap / LLM Summary"))
        layout.addWidget(self.campaign_heatmap_text)

        self.incident_btn = QPushButton("Generate Campaign Intelligence (LLM role: campaign)")
        self.incident_btn.clicked.connect(self.on_generate_incident_report)
        layout.addWidget(self.incident_btn)

        return widget

    # --- Alerts / kill-chain panel ---

    def build_alerts_panel(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        left = QVBoxLayout()
        self.alert_table = QTableWidget(0, 5)
        self.alert_table.setHorizontalHeaderLabels(
            ["Time", "Severity", "Rule", "Score", "Summary"]
        )
        left.addWidget(self.alert_table)

        right = QVBoxLayout()
        right.addWidget(QLabel("Kill-chain Visualization (lineage + multi-LLM)"))
        self.chain_tree = QTreeWidget()
        self.chain_tree.setHeaderLabels(["Node", "Type", "Score"])
        right.addWidget(self.chain_tree)

        right.addWidget(QLabel("LLM Kill-chain Narrative (multi-LLM + fusion)"))
        self.killchain_explain = QTextEdit()
        self.killchain_explain.setReadOnly(True)
        right.addWidget(self.killchain_explain)

        layout.addLayout(left, 2)
        layout.addLayout(right, 1)
        return widget

    # --- Heatmap panel ---

    def build_heatmap_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Threat Heatmap (score distribution summary)"))
        self.heatmap_text = QTextEdit()
        self.heatmap_text.setReadOnly(True)
        layout.addWidget(self.heatmap_text)
        return widget

    # --- Training panel ---

    def build_training_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Training Pipeline – Collected Samples (from DB)"))
        self.training_table = QTableWidget(0, 3)
        self.training_table.setHorizontalHeaderLabels(["Time", "Host", "Label"])
        layout.addWidget(self.training_table)
        return widget

    # --- Health panel ---

    def build_health_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Engine Health / Watchdog"))
        self.health_text = QTextEdit()
        self.health_text.setReadOnly(True)
        layout.addWidget(self.health_text)
        return widget

    # --- LLM panel (unbounded multi-LLM) ---

    def build_llm_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("LLM Intelligence Control"))

        top = QHBoxLayout()
        top.addWidget(QLabel("Active model:"))
        self.llm_combo = QComboBox()
        self.llm_combo.addItems(self.llm_manager.list_models())
        self.llm_combo.currentTextChanged.connect(self.on_llm_changed)
        top.addWidget(self.llm_combo)

        self.refresh_models_btn = QPushButton("Refresh Models")
        self.refresh_models_btn.clicked.connect(self.refresh_llm_models)
        top.addWidget(self.refresh_models_btn)

        layout.addLayout(top)

        layout.addWidget(QLabel("Multi-engine selection:"))
        self.llm_multi_list = QListWidget()
        self.llm_multi_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for name in self.llm_manager.list_models():
            self.llm_multi_list.addItem(name)
        layout.addWidget(self.llm_multi_list)

        self.llm_fuse_checkbox = QCheckBox("Fuse multi-LLM answers with fusion engine")
        self.llm_fuse_checkbox.setChecked(True)
        layout.addWidget(self.llm_fuse_checkbox)

        layout.addWidget(QLabel("Prompt:"))
        self.llm_prompt = QLineEdit()
        layout.addWidget(self.llm_prompt)

        btn_row = QHBoxLayout()
        self.llm_ask_btn = QPushButton("Ask active model")
        self.llm_ask_btn.clicked.connect(self.on_llm_ask_single)
        btn_row.addWidget(self.llm_ask_btn)

        self.llm_ask_multi_btn = QPushButton("Ask selected models")
        self.llm_ask_multi_btn.clicked.connect(self.on_llm_ask_multi)
        btn_row.addWidget(self.llm_ask_multi_btn)

        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Response:"))
        self.llm_response = QTextEdit()
        self.llm_response.setReadOnly(True)
        layout.addWidget(self.llm_response)

        return widget

    # --- Storage panel ---

    def build_storage_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Active Storage Path:"))
        self.storage_active_label = QLabel(self.engine.storage.active_path)
        layout.addWidget(self.storage_active_label)

        layout.addWidget(QLabel("Select Local Drive:"))
        self.drive_combo = QComboBox()
        self.drive_combo.addItems(self.engine.storage.list_local_drives())
        layout.addWidget(self.drive_combo)

        self.set_drive_btn = QPushButton("Set Local Drive")
        self.set_drive_btn.clicked.connect(self.on_set_local_drive)
        layout.addWidget(self.set_drive_btn)

        layout.addWidget(QLabel("SMB Path (\\\\SERVER\\Share):"))
        self.smb_input = QLineEdit()
        layout.addWidget(self.smb_input)

        self.set_smb_btn = QPushButton("Set SMB Path")
        self.set_smb_btn.clicked.connect(self.on_set_smb)
        layout.addWidget(self.set_smb_btn)

        return widget

    # --- Streaming dashboard panel (WebEngine hybrid mode) ---

    def build_streaming_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Real-time Streaming Dashboard (Web App)"))

        if QWebEngineView is not None:
            self.web_view = QWebEngineView()
            self.web_view.load(QUrl("http://127.0.0.1:8080"))
            layout.addWidget(self.web_view)
        else:
            info = QTextEdit()
            info.setReadOnly(True)
            info.setPlainText(
                "PyQtWebEngine is not installed.\n\n"
                "This tab normally embeds a small web app that subscribes to the WebSocket at ws://127.0.0.1:8765.\n\n"
                "You can:\n"
                "  1) Click the button below to attempt automatic installation (silent pip install), or\n"
                "  2) Manually install:\n"
                "       pip install PyQtWebEngine\n\n"
                "After installation, restart this program to enable the embedded dashboard."
            )
            layout.addWidget(info)

            self.install_webengine_btn = QPushButton("Attempt automatic PyQtWebEngine installation")
            self.install_webengine_btn.clicked.connect(self.on_attempt_webengine_install)
            layout.addWidget(self.install_webengine_btn)

        return widget

    # ========================================================
    # GUI HANDLERS
    # ========================================================

    def on_attempt_webengine_install(self):
        reply = QMessageBox.question(
            self,
            "PyQtWebEngine Missing",
            "PyQtWebEngine is not installed.\n\n"
            "Would you like to attempt automatic installation now?\n\n"
            "This will run a silent 'pip install PyQtWebEngine' using your current Python interpreter.\n"
            "You will need to restart this program after installation.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "PyQtWebEngine"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            QMessageBox.information(
                self,
                "Installation Attempted",
                "An attempt to install PyQtWebEngine has completed.\n\n"
                "If it succeeded, please restart this program to enable the embedded dashboard.\n"
                "If it failed, you may need to install it manually or check your environment."
            )
            log_event("Attempted silent installation of PyQtWebEngine.")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Installation Failed",
                f"Automatic installation failed with error:\n{e}\n\n"
                "Please install PyQtWebEngine manually:\n"
                "    pip install PyQtWebEngine"
            )
            log_event(f"Automatic PyQtWebEngine installation failed: {e}")

    def on_lockdown_toggle(self):
        self.engine.lockdown_enabled = not self.engine.lockdown_enabled
        self.lockdown_label.setText(f"Lockdown: {'ON' if self.engine.lockdown_enabled else 'OFF'}")
        log_event(f"Lockdown mode toggled: {'ON' if self.engine.lockdown_enabled else 'OFF'}")

    def on_simulate_toggle(self):
        self.engine.simulate_kills = not self.engine.simulate_kills
        self.simulate_label.setText(f"Simulate kills: {'ON' if self.engine.simulate_kills else 'OFF'}")
        log_event(f"Simulate pre-emptive kills toggled: {'ON' if self.engine.simulate_kills else 'OFF'}")

    def on_filter_changed(self, text):
        self.filter_var = text
        self.refresh_process_view()

    def on_search_changed(self, text):
        self.search_text = text.lower()
        self.refresh_process_view()

    def refresh_lists_ui(self):
        self.allow_list.clear()
        for item in sorted(ALLOWLIST):
            self.allow_list.addItem(QListWidgetItem(item))
        self.block_list.clear()
        for item in sorted(BLOCKLIST):
            self.block_list.addItem(QListWidgetItem(item))
        self.radio_list.clear()
        for item in sorted(RADIOACTIVE):
            self.radio_list.addItem(QListWidgetItem(item))

    def on_remove_from_lists(self):
        changed = False
        for widget, s in (
            (self.allow_list, ALLOWLIST),
            (self.block_list, BLOCKLIST),
            (self.radio_list, RADIOACTIVE),
        ):
            item = widget.currentItem()
            if item:
                val = item.text()
                if val in s:
                    s.discard(val)
                    log_event(f"Removed from list: {val}")
                    changed = True
        if changed:
            save_lists()
            self.refresh_lists_ui()
            self.refresh_process_view()

    def _selected_proc_rows(self):
        rows = self.proc_table.selectionModel().selectedRows()
        result = []
        for idx in rows:
            r = idx.row()
            vals = []
            for c in range(self.proc_table.columnCount()):
                item = self.proc_table.item(r, c)
                vals.append(item.text() if item else "")
            result.append(vals)
        return result

    def _selected_proc_row(self):
        row = self.proc_table.currentRow()
        if row < 0:
            return None
        vals = []
        for c in range(self.proc_table.columnCount()):
            item = self.proc_table.item(row, c)
            vals.append(item.text() if item else "")
        return vals

    def _selected_path(self):
        vals = self._selected_proc_row()
        if not vals or len(vals) < 7:
            return None
        return vals[6]

    def _selected_name_pid(self):
        vals = self._selected_proc_row()
        if not vals or len(vals) < 2:
            return None, None
        name = vals[0]
        try:
            pid = int(vals[1])
        except (TypeError, ValueError):
            pid = None
        return name, pid

    def on_add_allow(self):
        path_raw = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()
        norm = norm_path(path_raw) if path_raw else name_l
        key = norm

        ALLOWLIST.add(norm)
        BLOCKLIST.discard(norm)
        RADIOACTIVE.discard(norm)
        save_lists()
        log_event(f"ALLOW OVERRIDE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "allow"
            rec["fingerprints"] = []
            rec["last_score"] = 0
            rec["score_history"] = []
            rec["cpu_history"] = []
            BEHAVIOR_DB[key] = rec
        save_behavior()

        self.refresh_lists_ui()
        self.refresh_process_view()

    def on_add_block(self):
        path_raw = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()
        norm = norm_path(path_raw) if path_raw else name_l
        key = norm

        BLOCKLIST.add(norm)
        ALLOWLIST.discard(norm)
        RADIOACTIVE.discard(norm)
        save_lists()
        log_event(f"BLOCK OVERRIDE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "block"
            fps = set(tuple(fp) for fp in rec.get("fingerprints", []))
            rec["fingerprints"] = [list(x) for x in fps]
            rec["last_score"] = max(rec.get("last_score", 0), 90)
            BEHAVIOR_DB[key] = rec
        save_behavior()

        if pid is not None:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                log_event(f"BLOCK: Immediately terminated PID {pid} ({name}) key={key}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                log_event(f"BLOCK: Failed to terminate PID {pid}: {e}")

        self.refresh_lists_ui()
        self.refresh_process_view()

    def on_add_radio(self):
        path_raw = self._selected_path()
        name, pid = self._selected_name_pid()
        if not name:
            return
        name_l = name.lower()
        norm = norm_path(path_raw) if path_raw else name_l
        key = norm

        RADIOACTIVE.add(norm)
        ALLOWLIST.discard(norm)
        BLOCKLIST.discard(norm)
        save_lists()
        log_event(f"RADIO OVERRIDE: {key}")

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            rec["user_label"] = "radio"
            rec["last_score"] = max(rec.get("last_score", 0), 70)
            BEHAVIOR_DB[key] = rec
        save_behavior()

        self.refresh_lists_ui()
        self.refresh_process_view()

    def on_show_lineage(self):
        vals = self._selected_proc_row()
        if not vals or len(vals) < 2:
            return
        name = vals[0]
        try:
            pid = int(vals[1])
        except ValueError:
            return

        nodes, edges = LINEAGE_GRAPH.get_subgraph_for_pid(pid, depth=3)

        text = ""
        text += "Nodes:\n"
        for npid, ninfo in nodes.items():
            text += f"  PID {npid}: {ninfo['name']} [{ninfo['path']}] score={ninfo['score']}\n"
        text += "\nEdges (parent -> child):\n"
        for parent, child in edges:
            text += f"  {parent} -> {child}\n"

        dlg = QMessageBox(self)
        dlg.setWindowTitle(f"Lineage – {name} (PID {pid})")
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

    def on_proc_double_click(self, row, col):
        vals = []
        for c in range(self.proc_table.columnCount()):
            item = self.proc_table.item(row, c)
            vals.append(item.text() if item else "")
        if len(vals) < 7:
            return
        name, pid_str, cpu, status, threat, score, path = vals
        try:
            pid = int(pid_str)
        except ValueError:
            return

        row_info = None
        for r in self.engine.current_rows:
            if r["pid"] == pid and r["path"] == path:
                row_info = r
                break
        if not row_info:
            return

        anomalies = row_info.get("anomalies", [])
        fp = row_info.get("fingerprint", [])
        momentum = row_info.get("momentum", 0.0)
        age_seconds = row_info.get("age_seconds", 0.0)
        cluster_stats = row_info.get("cluster_stats", {})
        campaign_id = row_info.get("campaign_id")

        text = ""
        text += f"Name: {name}\n"
        text += f"PID: {pid}\n"
        text += f"Path: {path}\n"
        text += f"Status: {status}\n"
        text += f"Threat: {threat}\n"
        text += f"Score: {score}\n"
        text += f"CPU: {cpu}%\n"
        text += f"Age: {age_seconds:.1f} seconds\n"
        text += f"Momentum: {momentum:.1f}\n"
        text += f"Campaign: {campaign_id or '(none)'}\n\n"

        text += "Cluster stats:\n"
        if cluster_stats:
            text += f"  Members: {cluster_stats.get('members', 0)}\n"
            text += f"  Bad count: {cluster_stats.get('bad_count', 0)}\n"
            text += f"  Avg score: {cluster_stats.get('avg_score', 0.0):.1f}\n"
            text += f"  Recent members: {cluster_stats.get('recent_members', 0)}\n\n"
        else:
            text += "  (none)\n\n"

        text += "Anomalies:\n"
        if anomalies:
            for a in anomalies:
                text += f"  - {a}\n"
        else:
            text += "  (none)\n"

        text += "\nFingerprint:\n"
        if fp:
            for f in fp:
                text += f"  - {f}\n"
        else:
            text += "  (none)\n"

        dlg = QMessageBox(self)
        dlg.setWindowTitle(f"Explanation – {name} (PID {pid})")
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

    def on_explain_selected(self):
        rows = self._selected_proc_rows()
        if not rows:
            return

        def find_row_info(vals):
            if len(vals) < 7:
                return None, None
            name, pid_str, cpu, status, threat, score, path = vals
            try:
                pid = int(pid_str)
            except ValueError:
                return None, None
            for r in self.engine.current_rows:
                if r["pid"] == pid and r["path"] == path:
                    return r, pid
            return None, None

        selected_engines = [i.text() for i in self.llm_multi_list.selectedItems()]
        self.llm_manager.set_active_many(selected_engines)

        if len(rows) == 1:
            vals = rows[0]
            row_info, pid = find_row_info(vals)
            if not row_info:
                return
            name = row_info["name"]
            anomalies = row_info.get("anomalies", [])
            fp = row_info.get("fingerprint", [])
            momentum = row_info.get("momentum", 0.0)
            age_seconds = row_info.get("age_seconds", 0.0)
            cluster_stats = row_info.get("cluster_stats", {})
            campaign_id = row_info.get("campaign_id")

            prompt = (
                "You are an EDR/XDR analyst LLM. Analyze the following process and explain:\n"
                "1) Why it might be benign or malicious.\n"
                "2) Likely kill-chain stage(s).\n"
                "3) Predicted next behaviors (what happens next).\n"
                "4) Recommended response (block, monitor, isolate, etc.).\n\n"
                f"Name: {name}\n"
                f"PID: {pid}\n"
                f"Path: {row_info['path']}\n"
                f"Status: {row_info['status']}\n"
                f"Threat label: {row_info['threat']}\n"
                f"Score: {row_info['score']}\n"
                f"CPU: {row_info['cpu']}\n"
                f"Age_seconds: {age_seconds}\n"
                f"Momentum: {momentum}\n"
                f"Campaign_id: {campaign_id}\n"
                f"Cluster_stats: {json.dumps(cluster_stats, indent=2)}\n"
                f"Anomalies: {json.dumps(anomalies, indent=2)}\n"
                f"Fingerprint: {json.dumps(fp, indent=2)}\n"
            )

            results = self.llm_manager.ask_many(prompt)
            text = ""
            for name_engine, resp in results.items():
                text += f"===== {name_engine} =====\n{resp}\n\n"

            if self.llm_fuse_checkbox.isChecked() and len(results) > 1:
                fusion = self.llm_manager.fuse(
                    results,
                    task_description="Explain single process with kill-chain and prediction."
                )
                text += "===== FUSION SUMMARY =====\n"
                text += fusion

            self.llm_response.setPlainText(text)

        elif len(rows) == 2:
            vals1, vals2 = rows
            r1, pid1 = find_row_info(vals1)
            r2, pid2 = find_row_info(vals2)
            if not r1 or not r2:
                return

            prompt = (
                "You are an EDR/XDR analyst LLM. Compare these two processes and explain differences in risk, "
                "behavior, and likely kill-chain stages.\n\n"
                "For each process, consider:\n"
                "- anomalies\n"
                "- score and threat label\n"
                "- cluster and campaign context\n"
                "- predicted next behaviors\n\n"
                f"Process A: {json.dumps(r1, default=str, indent=2)}\n\n"
                f"Process B: {json.dumps(r2, default=str, indent=2)}\n\n"
                "Explain:\n"
                "- Which one is more dangerous and why.\n"
                "- Any shared campaign or cluster indicators.\n"
                "- Likely kill-chain stages for each.\n"
                "- Recommended response actions.\n"
            )

            results = self.llm_manager.ask_many(prompt)
            text = ""
            for name_engine, resp in results.items():
                text += f"===== {name_engine} =====\n{resp}\n\n"

            if self.llm_fuse_checkbox.isChecked() and len(results) > 1:
                fusion = self.llm_manager.fuse(
                    results,
                    task_description="Compare two processes and map to kill-chain."
                )
                text += "===== FUSION SUMMARY =====\n"
                text += fusion

            self.llm_response.setPlainText(text)

    def on_llm_changed(self, name):
        self.llm_manager.set_active(name)

    def refresh_llm_models(self):
        models = self.llm_manager.list_models()
        self.llm_combo.clear()
        self.llm_combo.addItems(models)

        self.llm_multi_list.clear()
        for name in models:
            self.llm_multi_list.addItem(name)

    def on_llm_ask_single(self):
        prompt = self.llm_prompt.text().strip()
        if not prompt:
            return
        resp = self.llm_manager.ask(prompt)
        self.llm_response.setPlainText(resp)

    def on_llm_ask_multi(self):
        prompt = self.llm_prompt.text().strip()
        if not prompt:
            return

        selected = [i.text() for i in self.llm_multi_list.selectedItems()]
        self.llm_manager.set_active_many(selected)

        results = self.llm_manager.ask_many(prompt)

        text = ""
        for name, resp in results.items():
            text += f"===== {name} =====\n{resp}\n\n"

        if self.llm_fuse_checkbox.isChecked() and len(results) > 1:
            fusion = self.llm_manager.fuse(results, task_description=prompt)
            text += "===== FUSION SUMMARY =====\n"
            text += fusion

        self.llm_response.setPlainText(text)

    def on_set_local_drive(self):
        drive = self.drive_combo.currentText()
        if self.engine.storage.set_manual_path(drive):
            self.storage_active_label.setText(drive)
        else:
            self.storage_active_label.setText("Invalid drive")

    def on_set_smb(self):
        path = self.smb_input.text().strip()
        if self.engine.storage.set_manual_path(path):
            self.storage_active_label.setText(path)
        else:
            self.storage_active_label.setText("Invalid SMB path")

    def on_generate_incident_report(self):
        stats = self.engine.get_campaign_stats()
        alerts = self.engine.fetch_recent_alerts(limit=50)
        prompt = (
            "You are a SOC analyst LLM focused on campaign-level intelligence.\n"
            "Generate an incident report summarizing current campaigns, their severity, and notable alerts.\n\n"
            f"Campaign_stats: {json.dumps(stats, indent=2)}\n\n"
            f"Recent_alerts: {json.dumps(alerts, default=str, indent=2)}\n\n"
            "Include:\n"
            "- Top campaigns and their characteristics.\n"
            "- Cross-host or cross-process patterns.\n"
            "- Likely threat actors or TTPs (speculative but grounded).\n"
            "- Recommended next steps for the SOC.\n"
        )

        results = self.llm_manager.ask_role("campaign", prompt)
        text = ""
        if results:
            for name_engine, resp in results.items():
                text += f"===== {name_engine} (campaign) =====\n{resp}\n\n"
            if self.llm_fuse_checkbox.isChecked() and len(results) > 1:
                fusion = self.llm_manager.fuse(
                    results,
                    task_description="Campaign-level intelligence and incident report."
                )
                text += "===== FUSION SUMMARY =====\n"
                text += fusion
        else:
            resp = self.llm_manager.ask(prompt)
            text = resp

        self.campaign_heatmap_text.setPlainText(text)

    def refresh_all(self):
        self.refresh_process_view()
        self.refresh_network_view()
        self.refresh_campaign_view()
        self.refresh_alerts_and_chain()
        self.refresh_heatmap()
        self.refresh_training()
        self.refresh_health()
        self.refresh_lists_ui()

    def refresh_process_view(self):
        rows = list(self.engine.current_rows)

        if self.filter_var != "ALL":
            rows = [r for r in rows if r["threat"] == self.filter_var]

        query = self.search_text
        if query:
            filtered = []
            for r in rows:
                if (query in r["name"].lower() or
                    query in r["path"].lower() or
                    any(query in a.lower() for a in r.get("anomalies", []))):
                    filtered.append(r)
            rows = filtered

        def sort_key(r):
            val = r.get(self.sort_column)
            if self.sort_column in ("pid", "cpu", "score"):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0
            return str(val)

        rows = sorted(rows, key=sort_key, reverse=self.sort_reverse)

        self.proc_table.setRowCount(0)
        for i in rows:
            r = self.proc_table.rowCount()
            self.proc_table.insertRow(r)
            vals = [
                i["name"],
                str(i["pid"]),
                f"{i['cpu']:.1f}",
                i["status"],
                i["threat"],
                str(i["score"]),
                i["path"],
            ]
            for c, v in enumerate(vals):
                self.proc_table.setItem(r, c, QTableWidgetItem(v))

        t50 = self.engine.whatif_counts(rows, 50)
        t60 = self.engine.whatif_counts(rows, 60)
        t70 = self.engine.whatif_counts(rows, 70)
        self.whatif_50.setText(f"T50: {t50}")
        self.whatif_60.setText(f"T60: {t60}")
        self.whatif_70.setText(f"T70: {t70}")
        self.threshold_label.setText(f"Threshold: {int(self.engine.dynamic_lockdown_threshold)}")

    def refresh_network_view(self):
        rows = list(self.engine.current_net_rows)
        self.net_table.setRowCount(0)
        for n in rows:
            r = self.net_table.rowCount()
            self.net_table.insertRow(r)
            vals = [
                n["laddr"],
                n["raddr"],
                n["status"],
                str(n["pid"]),
                n["name"],
            ]
            for c, v in enumerate(vals):
                self.net_table.setItem(r, c, QTableWidgetItem(v))

    def refresh_campaign_view(self):
        stats = self.engine.get_campaign_stats()
        self.campaign_table.setRowCount(0)
        for ckey, c in stats.items():
            r = self.campaign_table.rowCount()
            self.campaign_table.insertRow(r)
            vals = [
                ckey,
                str(c["severity"]),
                str(c["events"]),
                str(c["bad_events"]),
            ]
            for col, v in enumerate(vals):
                self.campaign_table.setItem(r, col, QTableWidgetItem(v))

        lines = []
        for ckey, c in stats.items():
            lines.append(
                f"{ckey}: sev={c['severity']} events={c['events']} bad={c['bad_events']}"
            )
        if lines and not self.campaign_heatmap_text.toPlainText().strip():
            self.campaign_heatmap_text.setPlainText("\n".join(lines))

    def refresh_alerts_and_chain(self):
        rows = self.engine.fetch_recent_alerts()
        self.alert_table.setRowCount(0)
        self.chain_tree.clear()
        self.killchain_explain.clear()

        for row in rows:
            r = self.alert_table.rowCount()
            self.alert_table.insertRow(r)
            for c, val in enumerate(row):
                self.alert_table.setItem(r, c, QTableWidgetItem(str(val)))

        if rows:
            top = rows[0]
            summary = top[4]
            name = summary.split(" (")[0]
            pid = None
            for p in self.engine.current_rows:
                if p["name"] == name:
                    pid = p["pid"]
                    break
            if pid is not None:
                nodes, edges = LINEAGE_GRAPH.get_subgraph_for_pid(pid, depth=3)
                root = QTreeWidgetItem([f"PID {pid}", "root", ""])
                self.chain_tree.addTopLevelItem(root)
                for npid, ninfo in nodes.items():
                    if npid == pid:
                        continue
                    child = QTreeWidgetItem(
                        [f"PID {npid}: {ninfo['name']}", "process", str(ninfo["score"])]
                    )
                    root.addChild(child)
                self.chain_tree.expandAll()

                prompt = (
                    "You are an EDR/XDR analyst LLM focused on kill-chain reasoning.\n"
                    "Given this process lineage, explain the likely kill-chain stages and how the processes relate.\n\n"
                    f"Nodes: {json.dumps(nodes, default=str, indent=2)}\n"
                    f"Edges: {json.dumps(edges, default=str, indent=2)}\n\n"
                    "Describe:\n"
                    "- Initial access / execution.\n"
                    "- Persistence / privilege escalation.\n"
                    "- Lateral movement / exfiltration if visible.\n"
                    "- Overall campaign context if inferable.\n"
                )

                results = self.llm_manager.ask_role("killchain", prompt)
                text = ""
                if results:
                    for name_engine, resp in results.items():
                        text += f"===== {name_engine} (killchain) =====\n{resp}\n\n"
                    if self.llm_fuse_checkbox.isChecked() and len(results) > 1:
                        fusion = self.llm_manager.fuse(
                            results,
                            task_description="Kill-chain visualization from lineage graph."
                        )
                        text += "===== FUSION SUMMARY =====\n"
                        text += fusion
                else:
                    text = self.llm_manager.ask(prompt)

                self.killchain_explain.setPlainText(text)

    def refresh_heatmap(self):
        scores = self.engine.fetch_stats_for_heatmap()
        if not scores:
            self.heatmap_text.setPlainText("No scores yet.")
            return
        avg = sum(scores) / len(scores)
        high = len([s for s in scores if s > 70])
        txt = (
            f"Total events: {len(scores)}\n"
            f"Average score: {avg:.2f}\n"
            f"High-risk events (>70): {high}\n"
        )
        self.heatmap_text.setPlainText(txt)

    def refresh_training(self):
        rows = self.engine.fetch_samples()
        self.training_table.setRowCount(0)
        for row in rows:
            r = self.training_table.rowCount()
            self.training_table.insertRow(r)
            for c, val in enumerate(row):
                self.training_table.setItem(r, c, QTableWidgetItem(str(val)))

    def refresh_health(self):
        txt = (
            "Watchdog: active\n"
            f"DB: {self.engine.db_path}\n"
            f"Storage root: {self.engine.storage.active_path}\n"
            "Inference: via multi-LLM manager\n"
            "Distributed sync: placeholder\n"
            f"GPU anomaly: {'ENABLED' if self.engine.gpu_anomaly.enabled else 'DISABLED'}\n"
            f"Streaming hub: {'ENABLED' if self.engine.streaming.loop is not None else 'DISABLED'}\n"
            f"WebEngine: {'AVAILABLE' if QWebEngineView is not None else 'MISSING'}\n"
        )
        self.health_text.setPlainText(txt)

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(APPDATA, exist_ok=True)
    print("Using lists file:", os.path.abspath(LISTS_FILE))
    print("Using log file:", os.path.abspath(LOG_FILE))
    print("Using behavior file:", os.path.abspath(BEHAVIOR_FILE))
    print("Using stats file:", os.path.abspath(STATS_FILE))
    print("Using campaigns file:", os.path.abspath(CAMPAIGN_FILE))

    load_lists()
    load_behavior()
    load_stats()
    load_campaigns()

    llm_manager = LLMEngineManager()

    llm_manager.register_engine(
        "copilot",
        CopilotEngine(),
        roles=["fusion", "killchain", "campaign", "anomaly", "prediction", "tuning"],
        tier="strategic",
        is_fusion=True,
    )

    onnx_path = Path("engine_onnx_model.onnx")
    if onnx_path.exists():
        llm_manager.register_engine(
            "local_onnx_fast",
            LocalONNXLlm(str(onnx_path)),
            roles=["anomaly", "killchain"],
            tier="fast",
            is_fusion=False,
        )

    engine = EngineCore(llm_manager=llm_manager)

    app = QApplication(sys.argv)
    console = EDRConsole(engine)
    console.show()
    ret = app.exec_()

    engine.running = False
    log_event("Sentinel EDR/XDR shutting down.")
    sys.exit(ret)

if __name__ == "__main__":
    main()

