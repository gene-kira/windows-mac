import psutil
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
from datetime import datetime
import statistics
from collections import defaultdict, deque
import hashlib
import argparse
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import requests
except ImportError:
    requests = None

# ============================================================
# CONFIG
# ============================================================

CHECK_INTERVAL = 2.0

APPDATA = os.getenv("APPDATA") or os.path.expanduser("~")
APPDATA = os.path.join(APPDATA, ".sentinel_guardian")
os.makedirs(APPDATA, exist_ok=True)

LISTS_FILE = os.path.join(APPDATA, "sentinel_lists.json")
LOG_FILE = os.path.join(APPDATA, "sentinel_events.log")
BEHAVIOR_FILE = os.path.join(APPDATA, "sentinel_behavior.json")
STATS_FILE = os.path.join(APPDATA, "sentinel_stats.json")
CAMPAIGN_FILE = os.path.join(APPDATA, "sentinel_campaigns.json")
DEFAULT_MODEL_FILE = os.path.join(APPDATA, "sentinel_model.onnx")
MODEL_CONFIG_FILE = os.path.join(APPDATA, "sentinel_model_config.json")

ALLOWLIST = set()
BLOCKLIST = set()
RADIOACTIVE = set()

BEHAVIOR_DB = {}
STATS_DB = {"history": []}
CAMPAIGN_DB = {}

WHITELIST_NAMES = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
}

LIST_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
BEHAVIOR_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()
CAMPAIGN_LOCK = threading.Lock()

BASE_LOCKDOWN_SCORE_THRESHOLD = 70
MIN_LOCKDOWN_THRESHOLD = 40
MAX_LOCKDOWN_THRESHOLD = 90

MAX_SCORE_HISTORY = 20
MAX_CPU_HISTORY = 20
MOMENTUM_WINDOW = 5
NEW_KEY_AGE_SECONDS = 60
NEW_FILE_AGE_SECONDS = 120

LINEAGE_RETENTION_SECONDS = 600  # 10 minutes

MODEL_AUTO_UPDATE_INTERVAL = 6 * 3600  # 6 hours

# Default (placeholder) model acquisition config
DEFAULT_MODEL_CONFIG = {
    "main_url": "",
    "primary_url": "",
    "fallback_urls": [],
    "manifest_url": "",
    "directory_url": "",
    "github_repo": "",
    "auto_update_enabled": False,
}


# ============================================================
# HELPERS
# ============================================================

def norm_path(p: str | None) -> str | None:
    if not p:
        return None
    return os.path.normpath(p).replace("\\", "/").lower()


def log_event(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    with LOG_LOCK:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass


def compute_file_hash(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def load_model_config():
    if not os.path.exists(MODEL_CONFIG_FILE):
        return DEFAULT_MODEL_CONFIG.copy()
    try:
        with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        log_event("WARNING: sentinel_model_config.json unreadable, using defaults.")
        return DEFAULT_MODEL_CONFIG.copy()

    cfg = DEFAULT_MODEL_CONFIG.copy()
    cfg.update({k: v for k, v in data.items() if k in cfg})
    if not isinstance(cfg.get("fallback_urls"), list):
        cfg["fallback_urls"] = []
    return cfg


def save_model_config(cfg):
    tmp = MODEL_CONFIG_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, MODEL_CONFIG_FILE)
    except OSError as e:
        log_event(f"ERROR: Failed to save model config: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


# ============================================================
# LIST STORAGE
# ============================================================

def load_lists():
    global ALLOWLIST, BLOCKLIST, RADIOACTIVE
    with LIST_LOCK:
        if not os.path.exists(LISTS_FILE):
            return
        try:
            with open(LISTS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_lists.json unreadable, starting with empty lists.")
            return

    ALLOWLIST = set(norm_path(x) for x in d.get("allow", []) if x)
    BLOCKLIST = set(norm_path(x) for x in d.get("block", []) if x)
    RADIOACTIVE = set(norm_path(x) for x in d.get("radioactive", []) if x)


def save_lists():
    data = {
        "allow": sorted(ALLOWLIST),
        "block": sorted(BLOCKLIST),
        "radioactive": sorted(RADIOACTIVE),
    }

    tmp_file = LISTS_FILE + ".tmp"
    with LIST_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, LISTS_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save lists: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ============================================================
# BEHAVIOR STORAGE
# ============================================================

def load_behavior():
    global BEHAVIOR_DB
    with BEHAVIOR_LOCK:
        if not os.path.exists(BEHAVIOR_FILE):
            BEHAVIOR_DB = {}
            return
        try:
            with open(BEHAVIOR_FILE, "r", encoding="utf-8") as f:
                BEHAVIOR_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_behavior.json unreadable, starting fresh.")
            BEHAVIOR_DB = {}
            return

        for key, rec in BEHAVIOR_DB.items():
            rec.setdefault("cluster_dir", "")
            rec.setdefault("cluster_name_pattern", "")
            rec.setdefault("cluster_key", "")
            rec.setdefault("score_history", [])
            rec.setdefault("cpu_history", [])
            rec.setdefault("file_size", None)
            rec.setdefault("file_mtime", None)
            rec.setdefault("file_hash", None)
            rec.setdefault("parent_type", None)
            rec.setdefault("first_seen", time.time())
            rec.setdefault("last_seen", rec["first_seen"])
            rec.setdefault("launch_count", 0)
            rec.setdefault("total_cpu", 0.0)
            rec.setdefault("max_cpu", 0.0)
            rec.setdefault("has_network", False)
            rec.setdefault("parent_pids", [])
            rec.setdefault("user_label", None)
            rec.setdefault("last_score", 0)
            rec.setdefault("fingerprints", [])


def save_behavior():
    tmp_file = BEHAVIOR_FILE + ".tmp"
    with BEHAVIOR_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(BEHAVIOR_DB, f, indent=2)
            os.replace(tmp_file, BEHAVIOR_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save behavior DB: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ============================================================
# STATS STORAGE
# ============================================================

def load_stats():
    global STATS_DB
    with STATS_LOCK:
        if not os.path.exists(STATS_FILE):
            STATS_DB = {"history": []}
            return
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                STATS_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_stats.json unreadable, starting fresh.")
            STATS_DB = {"history": []}


def save_stats():
    tmp_file = STATS_FILE + ".tmp"
    with STATS_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(STATS_DB, f, indent=2)
            os.replace(tmp_file, STATS_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save stats DB: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ============================================================
# CAMPAIGN STORAGE
# ============================================================

def load_campaigns():
    global CAMPAIGN_DB
    with CAMPAIGN_LOCK:
        if not os.path.exists(CAMPAIGN_FILE):
            CAMPAIGN_DB = {}
            return
        try:
            with open(CAMPAIGN_FILE, "r", encoding="utf-8") as f:
                CAMPAIGN_DB = json.load(f)
        except (OSError, json.JSONDecodeError):
            log_event("WARNING: sentinel_campaigns.json unreadable, starting fresh.")
            CAMPAIGN_DB = {}


def save_campaigns():
    tmp_file = CAMPAIGN_FILE + ".tmp"
    with CAMPAIGN_LOCK:
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(CAMPAIGN_DB, f, indent=2)
            os.replace(tmp_file, CAMPAIGN_FILE)
        except OSError as e:
            log_event(f"ERROR: Failed to save campaign DB: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass


# ============================================================
# LINEAGE GRAPH
# ============================================================

class LineageGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(set)
        self.reverse_edges = defaultdict(set)
        self.lock = threading.Lock()

    def update_process(self, pid, ppid, name, path, score, now):
        with self.lock:
            node = self.nodes.get(pid, {
                "name": name,
                "path": path,
                "first_seen": now,
                "last_seen": now,
                "score": score,
            })
            node["name"] = name
            node["path"] = path
            node["last_seen"] = now
            node["score"] = score
            self.nodes[pid] = node

            if ppid is not None and ppid > 0:
                self.edges[ppid].add(pid)
                self.reverse_edges[pid].add(ppid)

    def prune_old(self, now):
        with self.lock:
            to_remove = []
            for pid, node in self.nodes.items():
                if now - node["last_seen"] > LINEAGE_RETENTION_SECONDS:
                    to_remove.append(pid)
            for pid in to_remove:
                self.nodes.pop(pid, None)
                if pid in self.edges:
                    self.edges.pop(pid, None)
                if pid in self.reverse_edges:
                    parents = self.reverse_edges.pop(pid, set())
                    for p in parents:
                        if pid in self.edges[p]:
                            self.edges[p].remove(pid)

    def get_subgraph_for_pid(self, pid, depth=3):
        with self.lock:
            visited = set()
            queue = deque([(pid, 0)])
            nodes = {}
            edges = []

            while queue:
                current, d = queue.popleft()
                if current in visited or d > depth:
                    continue
                visited.add(current)
                node = self.nodes.get(current)
                if node:
                    nodes[current] = node

                for parent in self.reverse_edges.get(current, []):
                    edges.append((parent, current))
                    if parent not in visited:
                        queue.append((parent, d + 1))

                for child in self.edges.get(current, []):
                    edges.append((current, child))
                    if child not in visited:
                        queue.append((child, d + 1))

            return nodes, edges


LINEAGE_GRAPH = LineageGraph()


# ============================================================
# MODEL UPDATE MANAGER (AUTO + MANUAL DOWNLOAD)
# ============================================================

class ModelUpdateManager:
    def __init__(self):
        self.cfg = load_model_config()
        self.last_check = 0
        self.lock = threading.Lock()

    def save_cfg(self):
        with self.lock:
            save_model_config(self.cfg)

    def set_main_url(self, url: str):
        with self.lock:
            self.cfg["main_url"] = url.strip()
        self.save_cfg()

    def set_advanced_config(self, primary, fallbacks, manifest, directory, github_repo, auto_update):
        with self.lock:
            self.cfg["primary_url"] = primary.strip()
            self.cfg["fallback_urls"] = [u.strip() for u in fallbacks.split(",") if u.strip()] if isinstance(fallbacks, str) else fallbacks
            self.cfg["manifest_url"] = manifest.strip()
            self.cfg["directory_url"] = directory.strip()
            self.cfg["github_repo"] = github_repo.strip()
            self.cfg["auto_update_enabled"] = bool(auto_update)
        self.save_cfg()

    def get_cfg(self):
        with self.lock:
            return self.cfg.copy()

    def should_auto_update(self):
        with self.lock:
            return self.cfg.get("auto_update_enabled", False)

    def _http_get(self, url, stream=False):
        if requests is None:
            log_event("MODEL: requests module not available, cannot download.")
            return None
        try:
            resp = requests.get(url, timeout=15, stream=stream)
            if resp.status_code == 200:
                return resp
            log_event(f"MODEL: HTTP {resp.status_code} for {url}")
            return None
        except Exception as e:
            log_event(f"MODEL: HTTP error for {url}: {e}")
            return None

    def _download_to_path(self, url, dest_path):
        resp = self._http_get(url, stream=True)
        if resp is None:
            return False
        tmp = dest_path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest_path)
            return True
        except OSError as e:
            log_event(f"MODEL: Failed to write model to {dest_path}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return False

    def _validate_onnx(self, path):
        if ort is None:
            log_event("MODEL: onnxruntime not available, cannot validate model.")
            return False
        if not os.path.exists(path):
            return False
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            ins = sess.get_inputs()
            outs = sess.get_outputs()
            if not ins or not outs:
                log_event(f"MODEL: Invalid ONNX model (no inputs/outputs): {path}")
                return False
            return True
        except Exception as e:
            log_event(f"MODEL: ONNX validation failed for {path}: {e}")
            return False

    def _try_manifest(self, dest_path):
        cfg = self.get_cfg()
        url = cfg.get("manifest_url") or cfg.get("main_url")
        if not url or not url.lower().endswith(".json"):
            return False
        resp = self._http_get(url, stream=False)
        if resp is None:
            return False
        try:
            data = resp.json()
        except Exception as e:
            log_event(f"MODEL: Manifest JSON parse error: {e}")
            return False
        model_url = data.get("url")
        if not model_url:
            log_event("MODEL: Manifest missing 'url' field.")
            return False
        if not self._download_to_path(model_url, dest_path):
            return False
        if not self._validate_onnx(dest_path):
            return False
        log_event(f"MODEL: Downloaded model via manifest from {model_url}")
        return True

    def _try_direct_url(self, url, dest_path):
        if not url:
            return False
        if not self._download_to_path(url, dest_path):
            return False
        if not self._validate_onnx(dest_path):
            return False
        log_event(f"MODEL: Downloaded model from direct URL {url}")
        return True

    def _try_fallbacks(self, dest_path):
        cfg = self.get_cfg()
        urls = cfg.get("fallback_urls", [])
        for u in urls:
            if self._try_direct_url(u, dest_path):
                return True
        return False

    def _try_directory_scan(self, dest_path):
        cfg = self.get_cfg()
        url = cfg.get("directory_url") or cfg.get("main_url")
        if not url:
            return False
        if not url.endswith("/"):
            url = url + "/"
        resp = self._http_get(url, stream=False)
        if resp is None:
            return False
        text = resp.text
        candidates = []
        for line in text.splitlines():
            line = line.strip()
            if ".onnx" in line:
                # naive extraction
                start = line.find("href=")
                if start != -1:
                    start = line.find('"', start)
                    end = line.find('"', start + 1)
                    if start != -1 and end != -1:
                        href = line[start + 1:end]
                        if href.endswith(".onnx"):
                            if href.startswith("http://") or href.startswith("https://"):
                                candidates.append(href)
                            else:
                                candidates.append(url + href.lstrip("/"))
        if not candidates:
            return False
        # pick the last (assuming sorted listing or versioned names)
        candidates = sorted(set(candidates))
        best = candidates[-1]
        if self._try_direct_url(best, dest_path):
            log_event(f"MODEL: Downloaded model via directory scan from {best}")
            return True
        return False

    def _try_github_release(self, dest_path):
        cfg = self.get_cfg()
        repo = cfg.get("github_repo") or cfg.get("main_url")
        if not repo:
            return False
        if "github.com" in repo:
            # normalize to owner/repo
            parts = repo.rstrip("/").split("/")
            if len(parts) >= 2:
                owner = parts[-2]
                name = parts[-1]
                repo = f"{owner}/{name}"
            else:
                return False
        if "/" not in repo:
            return False
        if requests is None:
            return False
        api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        resp = self._http_get(api_url, stream=False)
        if resp is None:
            return False
        try:
            data = resp.json()
        except Exception as e:
            log_event(f"MODEL: GitHub JSON parse error: {e}")
            return False
        assets = data.get("assets", [])
        onnx_assets = [a for a in assets if a.get("name", "").endswith(".onnx")]
        if not onnx_assets:
            return False
        asset = onnx_assets[0]
        dl_url = asset.get("browser_download_url")
        if not dl_url:
            return False
        if self._try_direct_url(dl_url, dest_path):
            log_event(f"MODEL: Downloaded model via GitHub release from {dl_url}")
            return True
        return False

    def download_model_to(self, dest_path):
        """
        Try all strategies (manifest, primary, fallbacks, github, directory).
        """
        # 1) Manifest (if configured)
        if self._try_manifest(dest_path):
            return True

        cfg = self.get_cfg()
        primary = cfg.get("primary_url") or cfg.get("main_url")

        # 2) Primary direct URL
        if self._try_direct_url(primary, dest_path):
            return True

        # 3) Fallback URLs
        if self._try_fallbacks(dest_path):
            return True

        # 4) GitHub releases
        if self._try_github_release(dest_path):
            return True

        # 5) Directory scan
        if self._try_directory_scan(dest_path):
            return True

        return False

    def auto_update_loop(self):
        while True:
            time.sleep(5.0)
            if not self.should_auto_update():
                continue
            now = time.time()
            if now - self.last_check < MODEL_AUTO_UPDATE_INTERVAL:
                continue
            self.last_check = now
            log_event("MODEL: Auto-update check starting.")
            if self.download_model_to(DEFAULT_MODEL_FILE):
                if ScoringPipeline.load_model(DEFAULT_MODEL_FILE):
                    log_event(f"MODEL: Auto-updated and loaded model from {DEFAULT_MODEL_FILE}")
                else:
                    log_event("MODEL: Auto-update downloaded model but failed to load.")
            else:
                log_event("MODEL: Auto-update failed to find/download a model.")


MODEL_MANAGER = ModelUpdateManager()


# ============================================================
# MODULAR SCORING PIPELINE + ML HOOKS
# ============================================================

class ScoringPipeline:
    ML_MODEL = None
    ML_INPUT_NAME = None
    ML_OUTPUT_NAME = None
    ML_FEATURE_ORDER = [
        "cpu",
        "in_temp",
        "consonant_ratio",
        "launches",
        "max_cpu",
        "has_net",
        "lifetime",
        "age_seconds",
        "file_size",
        "file_age",
        "has_file_hash",
        "night_time",
        "cluster_members",
        "cluster_bad_count",
        "cluster_avg_score",
        "cluster_recent_members",
        "campaign_severity",
        "campaign_events",
        "campaign_bad_events",
        "momentum",
    ]
    ML_MODEL_PATH = None

    @staticmethod
    def load_model(path: str):
        global ort
        if ort is None:
            log_event("ML: onnxruntime not available, cannot load model.")
            ScoringPipeline.ML_MODEL = None
            ScoringPipeline.ML_INPUT_NAME = None
            ScoringPipeline.ML_OUTPUT_NAME = None
            ScoringPipeline.ML_MODEL_PATH = None
            return False

        if not os.path.exists(path):
            log_event(f"ML: Model file not found: {path}")
            ScoringPipeline.ML_MODEL = None
            ScoringPipeline.ML_INPUT_NAME = None
            ScoringPipeline.ML_OUTPUT_NAME = None
            ScoringPipeline.ML_MODEL_PATH = None
            return False

        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()
            outputs = sess.get_outputs()
            if not inputs or not outputs:
                log_event(f"ML: Invalid ONNX model (no inputs/outputs): {path}")
                return False

            ScoringPipeline.ML_MODEL = sess
            ScoringPipeline.ML_INPUT_NAME = inputs[0].name
            ScoringPipeline.ML_OUTPUT_NAME = outputs[0].name
            ScoringPipeline.ML_MODEL_PATH = path
            log_event(f"ML: Loaded ONNX model from {path}")
            return True
        except Exception as e:
            log_event(f"ML: Failed to load ONNX model {path}: {e}")
            ScoringPipeline.ML_MODEL = None
            ScoringPipeline.ML_INPUT_NAME = None
            ScoringPipeline.ML_OUTPUT_NAME = None
            ScoringPipeline.ML_MODEL_PATH = None
            return False

    @staticmethod
    def base_features(info, behavior_record, now):
        path = info["path"]
        name_l = info["name_l"]
        cpu = info["cpu"]
        status = info["status"]

        lower_path = path.lower()
        in_temp = any(x in lower_path for x in ["/temp/", "\\temp\\", "\\appdata\\", "/appdata/"])

        base_name = os.path.basename(path)
        if not base_name or base_name.startswith("<no-path:"):
            base_name = name_l

        consonant_ratio = 0.0
        if len(base_name) > 0:
            consonants = sum(1 for c in base_name if c.lower() in "bcdfghjklmnpqrstvwxyz")
            consonant_ratio = consonants / len(base_name)

        launches = behavior_record.get("launch_count", 0) if behavior_record else 0
        max_cpu = behavior_record.get("max_cpu", 0.0) if behavior_record else 0.0
        has_net = behavior_record.get("has_network", False) if behavior_record else False
        first_seen = behavior_record.get("first_seen", now) if behavior_record else now
        last_seen = behavior_record.get("last_seen", now) if behavior_record else now
        lifetime = last_seen - first_seen if last_seen and first_seen else 0
        age_seconds = now - first_seen if first_seen else 0

        file_size = behavior_record.get("file_size") if behavior_record else None
        file_mtime = behavior_record.get("file_mtime") if behavior_record else None
        file_hash = behavior_record.get("file_hash") if behavior_record else None
        file_age = None
        if file_mtime:
            file_age = now - file_mtime

        parent_type = behavior_record.get("parent_type") if behavior_record else None
        cluster_key = behavior_record.get("cluster_key") if behavior_record else None

        hour = datetime.fromtimestamp(now).hour
        night_time = hour < 6 or hour > 22

        return {
            "path": path,
            "name_l": name_l,
            "cpu": cpu,
            "status": status,
            "in_temp": in_temp,
            "base_name": base_name,
            "consonant_ratio": consonant_ratio,
            "launches": launches,
            "max_cpu": max_cpu,
            "has_net": has_net,
            "lifetime": lifetime,
            "age_seconds": age_seconds,
            "file_size": file_size,
            "file_age": file_age,
            "file_hash": file_hash,
            "parent_type": parent_type,
            "cluster_key": cluster_key,
            "night_time": night_time,
        }

    @staticmethod
    def rule_detectors(features):
        score = 0
        anomalies = []

        if features["in_temp"]:
            score += 20
            anomalies.append("Runs from temp/appdata")

        if len(features["base_name"]) > 10 and features["consonant_ratio"] > 0.6:
            score += 15
            anomalies.append("Random-looking name")

        cpu = features["cpu"]
        if cpu > 50:
            score += 15
            anomalies.append(f"High CPU: {cpu:.1f}%")
        if cpu > 80:
            score += 10

        if features["age_seconds"] < NEW_KEY_AGE_SECONDS and features["status"] == "SUSPICIOUS":
            score += 10
            anomalies.append("Early-life suspicious process")

        if features["launches"] <= 1 and features["lifetime"] < 10:
            score += 10
            anomalies.append("Ephemeral process")

        if features["has_net"]:
            score += 25
            anomalies.append("Has network activity")

        if features["max_cpu"] > 20 and cpu > features["max_cpu"] * 2:
            score += 15
            anomalies.append("CPU spike anomaly")

        if features["file_age"] is not None and features["file_age"] < NEW_FILE_AGE_SECONDS:
            score += 10
            anomalies.append("Recently created binary on disk")

        if features["night_time"]:
            anomalies.append("Night-time execution")

        return score, anomalies

    @staticmethod
    def status_adjustments(status, user_label):
        score = 0
        if status == "BLOCKED":
            score += 40
        elif status == "RADIOACTIVE":
            score += 25
        elif status == "SUSPICIOUS":
            score += 10
        elif status == "TRUSTED":
            score -= 20

        if user_label == "block":
            score += 30
        elif user_label == "radio":
            score += 15
        elif user_label == "allow":
            score -= 25

        return score

    @staticmethod
    def compute_fingerprint(features, behavior_record, extra_features):
        fp = []
        if features["in_temp"]:
            fp.append("temp_path")

        if len(features["base_name"]) > 10 and features["consonant_ratio"] > 0.6:
            fp.append("random_name")

        if features["cpu"] > 50:
            fp.append("high_cpu")

        if behavior_record:
            launches = behavior_record.get("launch_count", 0)
            max_cpu = behavior_record.get("max_cpu", 0.0)
            has_net = behavior_record.get("has_network", False)
            first_seen = behavior_record.get("first_seen", 0)
            last_seen = behavior_record.get("last_seen", 0)
            lifetime = last_seen - first_seen if last_seen and first_seen else 0

            if launches <= 1 and lifetime < 10:
                fp.append("ephemeral")
            if has_net:
                fp.append("has_net")
            if features["cpu"] > max_cpu * 2 and max_cpu > 20:
                fp.append("cpu_spike")

        if features["status"] in ("BLOCKED", "RADIOACTIVE"):
            fp.append("user_bad")
        elif features["status"] in ("ALLOW", "TRUSTED"):
            fp.append("user_good")

        if extra_features.get("night_time"):
            fp.append("night_run")
        if extra_features.get("sibling_burst"):
            fp.append("sibling_burst")
        if extra_features.get("new_file"):
            fp.append("new_file")
        if extra_features.get("temp_net"):
            fp.append("temp_net")
        if extra_features.get("parent_type"):
            fp.append(f"parent_{extra_features['parent_type']}")
        if extra_features.get("campaign_hot"):
            fp.append("campaign_hot")

        if features.get("file_hash"):
            fp.append("has_file_hash")

        return sorted(set(fp))

    @staticmethod
    def compute_momentum(score_history):
        if not score_history:
            return 0.0
        window = score_history[-MOMENTUM_WINDOW:]
        if not window:
            return 0.0
        weights = list(range(1, len(window) + 1))
        total_w = sum(weights)
        return sum(s * w for s, w in zip(window, weights)) / total_w

    @staticmethod
    def cluster_and_campaign_boosts(features, cluster_stats, campaign_stats, anomalies):
        score = 0
        cluster_key = features["cluster_key"]
        sibling_burst = False
        campaign_hot = False

        if cluster_key and cluster_key in cluster_stats:
            cstat = cluster_stats[cluster_key]
            if cstat["recent_members"] >= 5:
                sibling_burst = True
                score += 10
                anomalies.append("Cluster burst: many similar processes recently")
            if cstat["bad_count"] >= 2:
                score += 15
                anomalies.append("Cluster has multiple bad members")
            if cstat["avg_score"] >= 60:
                score += 10
                anomalies.append("Cluster average score is high")

        if cluster_key and cluster_key in campaign_stats:
            camp = campaign_stats[cluster_key]
            if camp["severity"] >= 2:
                campaign_hot = True
                score += 15
                anomalies.append("Campaign flagged as hot")

        return score, sibling_burst, campaign_hot

    @staticmethod
    def momentum_boost(score, score_history, anomalies):
        momentum = ScoringPipeline.compute_momentum(score_history + [score])
        if momentum > 60 and score < momentum:
            score += 10
            anomalies.append("Rising threat momentum")
        elif momentum > 40:
            score += 5
            anomalies.append("Elevated threat momentum")
        return score, momentum

    @staticmethod
    def finalize_score(score):
        if score < 0:
            score = 0
        if score > 100:
            score = 100
        if score >= 70:
            threat = "HIGH"
        elif score >= 40:
            threat = "MEDIUM"
        else:
            threat = "LOW"
        return score, threat

    @staticmethod
    def build_feature_vector(features, behavior_record, cluster_stats, campaign_stats, momentum):
        cluster_key = features.get("cluster_key")
        cstat = cluster_stats.get(cluster_key, {}) if cluster_key else {}
        camp = campaign_stats.get(cluster_key, {}) if cluster_key else {}

        vec = {
            "cpu": features["cpu"],
            "in_temp": int(features["in_temp"]),
            "consonant_ratio": features["consonant_ratio"],
            "launches": features["launches"],
            "max_cpu": features["max_cpu"],
            "has_net": int(features["has_net"]),
            "lifetime": features["lifetime"],
            "age_seconds": features["age_seconds"],
            "file_size": features["file_size"] or 0,
            "file_age": features["file_age"] or 0,
            "has_file_hash": int(bool(features.get("file_hash"))),
            "night_time": int(features["night_time"]),
            "cluster_members": cstat.get("members", 0),
            "cluster_bad_count": cstat.get("bad_count", 0),
            "cluster_avg_score": cstat.get("avg_score", 0.0),
            "cluster_recent_members": cstat.get("recent_members", 0),
            "campaign_severity": camp.get("severity", 0),
            "campaign_events": camp.get("events", 0),
            "campaign_bad_events": camp.get("bad_events", 0),
            "momentum": momentum,
        }
        return vec

    @staticmethod
    def ml_model_predict(feature_vector):
        sess = ScoringPipeline.ML_MODEL
        if sess is None or ScoringPipeline.ML_INPUT_NAME is None or ScoringPipeline.ML_OUTPUT_NAME is None:
            return None, None

        try:
            x = []
            for k in ScoringPipeline.ML_FEATURE_ORDER:
                x.append(float(feature_vector.get(k, 0.0)))
            arr = np.array([x], dtype=np.float32)
            inputs = {ScoringPipeline.ML_INPUT_NAME: arr}
            outputs = sess.run([ScoringPipeline.ML_OUTPUT_NAME], inputs)
            ml_score = float(outputs[0].ravel()[0])
            reason = f"ONNX model ({os.path.basename(ScoringPipeline.ML_MODEL_PATH)})"
            return ml_score, reason
        except Exception as e:
            log_event(f"ML: Prediction error: {e}")
            return None, None

    @staticmethod
    def apply_ml_adjustment(score, anomalies, feature_vector):
        ml_score, reason = ScoringPipeline.ml_model_predict(feature_vector)
        if ml_score is None:
            return score

        blended = 0.7 * score + 0.3 * ml_score
        anomalies.append(f"ML model contribution: {reason or 'score'}={ml_score:.1f}")
        return blended

    @staticmethod
    def compute_score(info, behavior_record, bad_fingerprints, cluster_stats, campaign_stats, now):
        status = info["status"]

        if status == "ALLOW":
            return 0, "LOW", [], [], False, 0.0, 0.0, None

        features = ScoringPipeline.base_features(info, behavior_record, now)

        score, anomalies = ScoringPipeline.rule_detectors(features)
        score += ScoringPipeline.status_adjustments(
            features["status"],
            behavior_record.get("user_label") if behavior_record else None
        )

        extra_features = {}
        if features["night_time"]:
            extra_features["night_time"] = True
        if features["in_temp"] and features["has_net"]:
            extra_features["temp_net"] = True
        if features["parent_type"]:
            extra_features["parent_type"] = features["parent_type"]
        if features["file_age"] is not None and features["file_age"] < NEW_FILE_AGE_SECONDS:
            extra_features["new_file"] = True

        cluster_boost, sibling_burst, campaign_hot = ScoringPipeline.cluster_and_campaign_boosts(
            features, cluster_stats, campaign_stats, anomalies
        )
        score += cluster_boost
        if sibling_burst:
            extra_features["sibling_burst"] = True
        if campaign_hot:
            extra_features["campaign_hot"] = True

        score_history = behavior_record.get("score_history", []) if behavior_record else []
        score, momentum = ScoringPipeline.momentum_boost(score, score_history, anomalies)

        feature_vector = ScoringPipeline.build_feature_vector(
            features, behavior_record, cluster_stats, campaign_stats, momentum
        )
        score = ScoringPipeline.apply_ml_adjustment(score, anomalies, feature_vector)

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
# SENTINEL BRAIN
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


# ============================================================
# CAMPAIGN ENGINE
# ============================================================

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
# GUI
# ============================================================

class SentinelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel – Predictive Guardian")

        self.brain = SentinelBrain()
        self.running = True
        self.lockdown = tk.BooleanVar(value=False)
        self.simulate_kills = tk.BooleanVar(value=False)

        self.sort_column = "score"
        self.sort_reverse = True

        self.monitor_thread = None
        self.watchdog_thread = None
        self.model_update_thread = None

        self.current_rows = []
        self.current_net_rows = []

        self.dynamic_lockdown_threshold = BASE_LOCKDOWN_SCORE_THRESHOLD

        self.search_var = tk.StringVar(value="")
        self.model_status_var = tk.StringVar(value="Model: none")
        self.model_url_var = tk.StringVar(value=MODEL_MANAGER.get_cfg().get("main_url", ""))

        self._build_ui()
        self._build_context_menu()
        self.refresh_lists_ui()

        self.start_monitor_thread()
        self.start_watchdog_thread()
        self.start_model_update_thread()

        if ScoringPipeline.ML_MODEL_PATH:
            self.model_status_var.set(f"Model: {os.path.basename(ScoringPipeline.ML_MODEL_PATH)}")
        else:
            self.model_status_var.set("Model: none")

    # ---------------- THREADS ----------------

    def start_monitor_thread(self):
        t = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread = t
        t.start()
        log_event("Monitor thread started.")

    def start_watchdog_thread(self):
        t = threading.Thread(target=self.watchdog_loop, daemon=True)
        self.watchdog_thread = t
        t.start()
        log_event("Watchdog thread started.")

    def start_model_update_thread(self):
        t = threading.Thread(target=MODEL_MANAGER.auto_update_loop, daemon=True)
        self.model_update_thread = t
        t.start()
        log_event("Model auto-update thread started.")

    # ---------------- UI BUILD ----------------

    def _build_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x")

        self.lockdown_check = ttk.Checkbutton(
            top_frame,
            text="Lockdown Mode (adaptive score-based kill)",
            variable=self.lockdown,
            command=self.on_lockdown_toggle,
        )
        self.lockdown_check.pack(side="left", padx=5, pady=2)

        self.simulate_check = ttk.Checkbutton(
            top_frame,
            text="Simulate Pre-emptive Kills",
            variable=self.simulate_kills,
            command=self.on_simulate_toggle,
        )
        self.simulate_check.pack(side="left", padx=5, pady=2)

        self.threshold_label = ttk.Label(top_frame, text="Threshold: ?")
        self.threshold_label.pack(side="left", padx=(10, 2))

        ttk.Label(top_frame, text="Filter:").pack(side="left", padx=(10, 2))
        self.filter_var = tk.StringVar(value="ALL")
        self.filter_combo = ttk.Combobox(
            top_frame,
            textvariable=self.filter_var,
            values=["ALL", "HIGH", "MEDIUM", "LOW"],
            state="readonly",
            width=8,
        )
        self.filter_combo.current(0)
        self.filter_combo.pack(side="left", padx=2)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_process_view())

        ttk.Label(top_frame, text="Search:").pack(side="left", padx=(10, 2))
        self.search_entry = ttk.Entry(top_frame, textvariable=self.search_var, width=20)
        self.search_entry.pack(side="left", padx=2)
        self.search_entry.bind("<KeyRelease>", lambda e: self.refresh_process_view())

        # Model controls
        model_frame = ttk.Frame(self.root)
        model_frame.pack(fill="x", pady=(2, 4))

        self.model_status_label = ttk.Label(model_frame, textvariable=self.model_status_var)
        self.model_status_label.pack(side="left", padx=(5, 10))

        self.load_model_button = ttk.Button(
            model_frame,
            text="Load Model...",
            command=self.on_load_model_clicked
        )
        self.load_model_button.pack(side="left", padx=5)

        ttk.Label(model_frame, text="Model Source URL:").pack(side="left", padx=(10, 2))
        self.model_url_entry = ttk.Entry(model_frame, textvariable=self.model_url_var, width=40)
        self.model_url_entry.pack(side="left", padx=2)

        self.save_url_button = ttk.Button(
            model_frame,
            text="Save URL",
            command=self.on_save_url_clicked
        )
        self.save_url_button.pack(side="left", padx=3)

        self.download_model_button = ttk.Button(
            model_frame,
            text="Download Model",
            command=self.on_download_model_clicked
        )
        self.download_model_button.pack(side="left", padx=3)

        self.auto_update_var = tk.BooleanVar(value=MODEL_MANAGER.get_cfg().get("auto_update_enabled", False))
        self.auto_update_check = ttk.Checkbutton(
            model_frame,
            text="Auto-Update",
            variable=self.auto_update_var,
            command=self.on_auto_update_toggle
        )
        self.auto_update_check.pack(side="left", padx=5)

        self.advanced_model_button = ttk.Button(
            model_frame,
            text="Advanced...",
            command=self.on_advanced_model_clicked
        )
        self.advanced_model_button.pack(side="left", padx=5)

        whatif_frame = ttk.Frame(self.root)
        whatif_frame.pack(fill="x", pady=(2, 4))

        ttk.Label(whatif_frame, text="What-if kills:").pack(side="left", padx=(5, 2))
        self.whatif_50 = ttk.Label(whatif_frame, text="T50: 0")
        self.whatif_50.pack(side="left", padx=5)
        self.whatif_60 = ttk.Label(whatif_frame, text="T60: 0")
        self.whatif_60.pack(side="left", padx=5)
        self.whatif_70 = ttk.Label(whatif_frame, text="T70: 0")
        self.whatif_70.pack(side="left", padx=5)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        proc_frame = ttk.Frame(notebook)
        notebook.add(proc_frame, text="Processes")

        pane = ttk.PanedWindow(proc_frame, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane)
        pane.add(left, weight=3)

        self.tree = ttk.Treeview(
            left,
            columns=("name", "pid", "cpu", "status", "threat", "score", "path"),
            show="headings",
        )
        for c in ("name", "pid", "cpu", "status", "threat", "score", "path"):
            self.tree.heading(c, text=c.upper(), command=lambda col=c: self.on_tree_heading_click(col))
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<Double-1>", self.show_explanation_popup)

        self.tree.tag_configure("HIGH", foreground="red")
        self.tree.tag_configure("MEDIUM", foreground="orange")
        self.tree.tag_configure("LOW", foreground="green")
        self.tree.tag_configure("UNKNOWN", foreground="gray")

        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        self.allow_box = self._make_box(right, "ALLOWLIST")
        self.block_box = self._make_box(right, "BLOCKLIST")
        self.radio_box = self._make_box(right, "RADIOACTIVE")

        ttk.Button(
            right, text="Remove Selected",
            command=self.remove_selected_from_lists
        ).pack(fill="x", pady=5)

        net_frame = ttk.Frame(notebook)
        notebook.add(net_frame, text="Network")

        self.net_tree = ttk.Treeview(
            net_frame,
            columns=("laddr", "raddr", "status", "pid", "name"),
            show="headings",
        )
        for c in ("laddr", "raddr", "status", "pid", "name"):
            self.net_tree.heading(c, text=c.upper())
        self.net_tree.pack(fill="both", expand=True)

        campaign_frame = ttk.Frame(notebook)
        notebook.add(campaign_frame, text="Campaigns")

        self.campaign_tree = ttk.Treeview(
            campaign_frame,
            columns=("cluster_key", "severity", "events", "bad_events"),
            show="headings",
        )
        for c in ("cluster_key", "severity", "events", "bad_events"):
            self.campaign_tree.heading(c, text=c.upper())
        self.campaign_tree.pack(fill="both", expand=True)

    def _make_box(self, parent, title):
        f = ttk.LabelFrame(parent, text=title)
        f.pack(fill="both", expand=True, padx=5, pady=5)

        lb = tk.Listbox(f)
        lb.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(f, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        return lb

    # ---------------- CONTEXT MENU ----------------

    def _build_context_menu(self):
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="➕ Add to Allowlist", command=self.add_selected_allow)
        self.menu.add_command(label="⛔ Add to Blocklist", command=self.add_selected_block)
        self.menu.add_command(label="☢ Add to Radioactive", command=self.add_selected_radio)
        self.menu.add_separator()
        self.menu.add_command(label="🧬 Show Lineage Graph (text)", command=self.show_lineage_popup)

    def show_context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.menu.tk_popup(event.x_root, event.y_root)
            self.menu.grab_release()

    # ---------------- MODEL UI HANDLERS ----------------

    def on_load_model_clicked(self):
        path = filedialog.askopenfilename(
            title="Select ONNX Model",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")]
        )
        if not path:
            return
        ok = ScoringPipeline.load_model(path)
        if ok:
            self.model_status_var.set(f"Model: {os.path.basename(path)}")
            messagebox.showinfo("Model Loaded", f"Loaded model:\n{path}")
        else:
            self.model_status_var.set("Model: none")
            messagebox.showerror("Model Error", f"Failed to load model:\n{path}")

    def on_save_url_clicked(self):
        url = self.model_url_var.get().strip()
        MODEL_MANAGER.set_main_url(url)
        messagebox.showinfo("Model URL Saved", f"Saved main model URL:\n{url or '(empty)'}")

    def on_download_model_clicked(self):
        # Ask where to save
        initial_dir = os.path.dirname(DEFAULT_MODEL_FILE)
        path = filedialog.asksaveasfilename(
            title="Save ONNX Model As",
            defaultextension=".onnx",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=os.path.basename(DEFAULT_MODEL_FILE),
        )
        if not path:
            return

        self.model_status_var.set("Model: downloading...")
        self.root.update_idletasks()

        ok = MODEL_MANAGER.download_model_to(path)
        if not ok:
            self.model_status_var.set("Model: none")
            messagebox.showerror("Download Failed", "Failed to find/download a model using configured sources.")
            return

        if ScoringPipeline.load_model(path):
            self.model_status_var.set(f"Model: {os.path.basename(path)}")
            messagebox.showinfo("Model Downloaded", f"Downloaded and loaded model:\n{path}")
        else:
            self.model_status_var.set("Model: none")
            messagebox.showerror("Model Error", f"Downloaded model but failed to load:\n{path}")

    def on_auto_update_toggle(self):
        cfg = MODEL_MANAGER.get_cfg()
        cfg["auto_update_enabled"] = self.auto_update_var.get()
        MODEL_MANAGER.set_advanced_config(
            cfg.get("primary_url", ""),
            ",".join(cfg.get("fallback_urls", [])),
            cfg.get("manifest_url", ""),
            cfg.get("directory_url", ""),
            cfg.get("github_repo", ""),
            cfg.get("auto_update_enabled", False),
        )
        state = "ON" if self.auto_update_var.get() else "OFF"
        log_event(f"MODEL: Auto-update toggled: {state}")

    def on_advanced_model_clicked(self):
        cfg = MODEL_MANAGER.get_cfg()

        win = tk.Toplevel(self.root)
        win.title("Advanced Model Sources")
        win.geometry("500x260")

        ttk.Label(win, text="Primary URL:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        primary_var = tk.StringVar(value=cfg.get("primary_url", ""))
        primary_entry = ttk.Entry(win, textvariable=primary_var, width=50)
        primary_entry.grid(row=0, column=1, sticky="we", padx=5, pady=3)

        ttk.Label(win, text="Fallback URLs (comma-separated):").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        fallback_var = tk.StringVar(value=",".join(cfg.get("fallback_urls", [])))
        fallback_entry = ttk.Entry(win, textvariable=fallback_var, width=50)
        fallback_entry.grid(row=1, column=1, sticky="we", padx=5, pady=3)

        ttk.Label(win, text="Manifest URL (.json):").grid(row=2, column=0, sticky="w", padx=5, pady=3)
        manifest_var = tk.StringVar(value=cfg.get("manifest_url", ""))
        manifest_entry = ttk.Entry(win, textvariable=manifest_var, width=50)
        manifest_entry.grid(row=2, column=1, sticky="we", padx=5, pady=3)

        ttk.Label(win, text="Directory URL (for scan):").grid(row=3, column=0, sticky="w", padx=5, pady=3)
        directory_var = tk.StringVar(value=cfg.get("directory_url", ""))
        directory_entry = ttk.Entry(win, textvariable=directory_var, width=50)
        directory_entry.grid(row=3, column=1, sticky="we", padx=5, pady=3)

        ttk.Label(win, text="GitHub Repo or URL:").grid(row=4, column=0, sticky="w", padx=5, pady=3)
        github_var = tk.StringVar(value=cfg.get("github_repo", ""))
        github_entry = ttk.Entry(win, textvariable=github_var, width=50)
        github_entry.grid(row=4, column=1, sticky="we", padx=5, pady=3)

        auto_var = tk.BooleanVar(value=cfg.get("auto_update_enabled", False))
        auto_check = ttk.Checkbutton(win, text="Enable Auto-Update", variable=auto_var)
        auto_check.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        def save_and_close():
            MODEL_MANAGER.set_advanced_config(
                primary_var.get(),
                fallback_var.get(),
                manifest_var.get(),
                directory_var.get(),
                github_var.get(),
                auto_var.get(),
            )
            self.auto_update_var.set(auto_var.get())
            messagebox.showinfo("Model Sources Saved", "Advanced model source configuration saved.")
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="Save", command=save_and_close).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side="left", padx=5)

        win.grid_columnconfigure(1, weight=1)

    # ---------------- EXPLANATION POPUP ----------------

    def show_explanation_popup(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0], "values")
        if not item or len(item) < 7:
            return

        name, pid, cpu, status, threat, score, path = item
        pid = int(pid)
        row = None
        for r in self.current_rows:
            if r["pid"] == pid and r["path"] == path:
                row = r
                break
        if not row:
            return

        anomalies = row.get("anomalies", [])
        fp = row.get("fingerprint", [])
        momentum = row.get("momentum", 0.0)
        age_seconds = row.get("age_seconds", 0.0)
        cluster_stats = row.get("cluster_stats", {})
        campaign_id = row.get("campaign_id")

        key = self._key_for_behavior(path, name.lower())
        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            file_hash = rec.get("file_hash")

        win = tk.Toplevel(self.root)
        win.title(f"Explanation – {name} (PID {pid})")
        win.geometry("600x540")

        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True)

        text.insert("end", f"Name: {name}\n")
        text.insert("end", f"PID: {pid}\n")
        text.insert("end", f"Path: {path}\n")
        text.insert("end", f"Status: {status}\n")
        text.insert("end", f"Threat: {threat}\n")
        text.insert("end", f"Score: {score}\n")
        text.insert("end", f"CPU: {cpu}%\n")
        text.insert("end", f"Age: {age_seconds:.1f} seconds\n")
        text.insert("end", f"Momentum: {momentum:.1f}\n")
        text.insert("end", f"Campaign: {campaign_id or '(none)'}\n")
        text.insert("end", f"File hash (SHA-256): {file_hash or '(none)'}\n")
        text.insert("end", f"Model: {os.path.basename(ScoringPipeline.ML_MODEL_PATH) if ScoringPipeline.ML_MODEL_PATH else 'none'}\n\n")

        text.insert("end", "Cluster stats:\n")
        if cluster_stats:
            text.insert("end", f"  Members: {cluster_stats.get('members', 0)}\n")
            text.insert("end", f"  Bad count: {cluster_stats.get('bad_count', 0)}\n")
            text.insert("end", f"  Avg score: {cluster_stats.get('avg_score', 0.0):.1f}\n")
            text.insert("end", f"  Recent members: {cluster_stats.get('recent_members', 0)}\n\n")
        else:
            text.insert("end", "  (none)\n\n")

        text.insert("end", "Anomalies:\n")
        if anomalies:
            for a in anomalies:
                text.insert("end", f"  - {a}\n")
        else:
            text.insert("end", "  (none)\n")

        text.insert("end", "\nFingerprint:\n")
        if fp:
            for f in fp:
                text.insert("end", f"  - {f}\n")
        else:
            text.insert("end", "  (none)\n")

        text.config(state="disabled")

    def show_lineage_popup(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0], "values")
        if not item or len(item) < 2:
            return
        name, pid = item[0], int(item[1])

        nodes, edges = LINEAGE_GRAPH.get_subgraph_for_pid(pid, depth=3)

        win = tk.Toplevel(self.root)
        win.title(f"Lineage – {name} (PID {pid})")
        win.geometry("600x400")

        text = tk.Text(win, wrap="word")
        text.pack(fill="both", expand=True)

        text.insert("end", "Nodes:\n")
        for npid, ninfo in nodes.items():
            text.insert("end", f"  PID {npid}: {ninfo['name']} [{ninfo['path']}] score={ninfo['score']}\n")

        text.insert("end", "\nEdges (parent -> child):\n")
        for parent, child in edges:
            text.insert("end", f"  {parent} -> {child}\n")

        text.config(state="disabled")

    # ---------------- SORTING / FILTERING ----------------

    def on_tree_heading_click(self, column):
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        self.refresh_process_view()

    # ---------------- SELECTION HELPERS ----------------

    def _selected_values(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.item(sel[0], "values")

    def _selected_path(self):
        v = self._selected_values()
        if not v or len(v) < 7:
            return None
        return v[6] if v[6] else None

    def _selected_name_pid(self):
        v = self._selected_values()
        if not v or len(v) < 2:
            return None, None
        name = v[0]
        try:
            pid = int(v[1])
        except (TypeError, ValueError):
            pid = None
        return name, pid

    def _key_for_behavior(self, path, name_l):
        if path and not path.startswith("<no-path:"):
            return path
        return name_l

    # ---------------- LIST OPERATIONS ----------------

    def add_selected_allow(self):
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

    def add_selected_block(self):
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

    def add_selected_radio(self):
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

    def refresh_lists_ui(self):
        for lb, s in (
            (self.allow_box, ALLOWLIST),
            (self.block_box, BLOCKLIST),
            (self.radio_box, RADIOACTIVE),
        ):
            lb.delete(0, "end")
            for item in sorted(s):
                lb.insert("end", item)

    def remove_selected_from_lists(self):
        changed = False
        for lb, s in (
            (self.allow_box, ALLOWLIST),
            (self.block_box, BLOCKLIST),
            (self.radio_box, RADIOACTIVE),
        ):
            sel = lb.curselection()
            if sel:
                val = lb.get(sel[0])
                if val in s:
                    s.discard(val)
                    log_event(f"Removed from list: {val}")
                    changed = True
        if changed:
            save_lists()
            self.refresh_lists_ui()
            self.refresh_process_view()

    # ---------------- AUTO-FILL LISTS ----------------

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

    # ---------------- LOCKDOWN ----------------

    def on_lockdown_toggle(self):
        state = "ON" if self.lockdown.get() else "OFF"
        log_event(f"Lockdown mode toggled: {state}")

    def on_simulate_toggle(self):
        state = "ON" if self.simulate_kills.get() else "OFF"
        log_event(f"Simulate pre-emptive kills toggled: {state}")

    def enforce_lockdown(self, info):
        if not self.lockdown.get():
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

        if self.simulate_kills.get():
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

    # ---------------- BEHAVIOR / CLUSTERS / CAMPAIGNS ----------------

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
                "file_hash": None,
                "parent_type": None,
            })

            if rec.get("file_size") is None and not path.startswith("<no-path:"):
                try:
                    st = os.stat(path)
                    rec["file_size"] = st.st_size
                    rec["file_mtime"] = st.st_mtime
                except OSError:
                    rec["file_size"] = None
                    rec["file_mtime"] = None

            if rec.get("file_hash") is None and not path.startswith("<no-path:"):
                rec["file_hash"] = compute_file_hash(path)

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

            BEHAVIOR_DB[key] = rec

        with BEHAVIOR_LOCK:
            rec = BEHAVIOR_DB.get(key, {})
            score, threat, anomalies, fp, matched_bad_fp, momentum, age_seconds, cluster_key = \
                ScoringPipeline.compute_score(info, rec, bad_fingerprints, cluster_stats, campaign_stats, now)

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

    # ---------------- TEMPORAL / THRESHOLD ----------------

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
                self.threshold_label.config(text=f"Threshold: {int(self.dynamic_lockdown_threshold)}")
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

        self.threshold_label.config(text=f"Threshold: {int(self.dynamic_lockdown_threshold)}")

        if hostile_factor >= 0.5:
            log_event(
                f"GLOBAL ALERT: Hostile pattern detected. "
                f"high_count={high_count}, avg_score={avg_score:.1f}, "
                f"threshold={self.dynamic_lockdown_threshold:.1f}"
            )

    # ---------------- WHAT-IF DASHBOARD ----------------

    def update_whatif_dashboard(self, rows):
        def count_kills(threshold):
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

        t50 = count_kills(50)
        t60 = count_kills(60)
        t70 = count_kills(70)

        self.whatif_50.config(text=f"T50: {t50}")
        self.whatif_60.config(text=f"T60: {t60}")
        self.whatif_70.config(text=f"T70: {t70}")

    # ---------------- MONITOR LOOP ----------------

    def monitor_loop(self):
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

            self.root.after(0, self.update_ui, rows, net_rows, updated)
            time.sleep(CHECK_INTERVAL)

    def update_ui(self, rows, net_rows, updated):
        self.current_rows = rows
        self.current_net_rows = net_rows
        self.refresh_process_view()
        self.refresh_network_view()
        self.update_whatif_dashboard(rows)
        self.refresh_campaign_view()
        if updated:
            self.refresh_lists_ui()

    def refresh_process_view(self):
        rows = list(self.current_rows)

        filt = self.filter_var.get()
        if filt != "ALL":
            rows = [r for r in rows if r["threat"] == filt]

        query = self.search_var.get().strip().lower()
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

        self.tree.delete(*self.tree.get_children())
        for i in rows:
            tag = i["threat"] if i["threat"] in ("HIGH", "MEDIUM", "LOW") else "UNKNOWN"
            self.tree.insert(
                "",
                "end",
                values=(
                    i["name"],
                    i["pid"],
                    f"{i['cpu']:.1f}",
                    i["status"],
                    i["threat"],
                    i["score"],
                    i["path"],
                ),
                tags=(tag,),
            )

    def refresh_network_view(self):
        rows = list(self.current_net_rows)
        self.net_tree.delete(*self.net_tree.get_children())
        for n in rows:
            self.net_tree.insert(
                "",
                "end",
                values=(
                    n["laddr"],
                    n["raddr"],
                    n["status"],
                    n["pid"],
                    n["name"],
                ),
            )

    def refresh_campaign_view(self):
        stats = CampaignEngine.get_campaign_stats()
        self.campaign_tree.delete(*self.campaign_tree.get_children())
        for ckey, c in stats.items():
            self.campaign_tree.insert(
                "",
                "end",
                values=(
                    ckey,
                    c["severity"],
                    c["events"],
                    c["bad_events"],
                ),
            )

    # ---------------- WATCHDOG ----------------

    def watchdog_loop(self):
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

            if self.monitor_thread and not self.monitor_thread.is_alive():
                log_event("WATCHDOG: Monitor thread not alive, restarting.")
                self.start_monitor_thread()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sentinel – Predictive Guardian")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model file")
    args = parser.parse_args()

    print("Using lists file:", os.path.abspath(LISTS_FILE))
    print("Using log file:", os.path.abspath(LOG_FILE))
    print("Using behavior file:", os.path.abspath(BEHAVIOR_FILE))
    print("Using stats file:", os.path.abspath(STATS_FILE))
    print("Using campaigns file:", os.path.abspath(CAMPAIGN_FILE))
    print("Default model file:", os.path.abspath(DEFAULT_MODEL_FILE))
    print("Model config file:", os.path.abspath(MODEL_CONFIG_FILE))

    load_lists()
    load_behavior()
    load_stats()
    load_campaigns()

    model_loaded = False
    if args.model:
        model_loaded = ScoringPipeline.load_model(args.model)
    elif os.path.exists(DEFAULT_MODEL_FILE):
        model_loaded = ScoringPipeline.load_model(DEFAULT_MODEL_FILE)

    if model_loaded:
        print("Model loaded:", ScoringPipeline.ML_MODEL_PATH)
    else:
        print("No model loaded (either missing or failed).")

    root = tk.Tk()
    app = SentinelGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(app, root))
    root.mainloop()


def on_close(app, root):
    app.running = False    # monitor + watchdog threads will exit
    log_event("Sentinel shutting down.")
    root.destroy()


if __name__ == "__main__":
    main()

