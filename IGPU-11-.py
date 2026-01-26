# golden_star_event_horizon_swarm_civilization_raft_game_profiles.py

import sys
import subprocess
import hashlib
import os
import glob
import json
import socket
import uuid
from collections import deque, defaultdict
from datetime import datetime, timedelta
import shutil
import time
import math

from PyQt5.QtCore import QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QSystemTrayIcon, QMenu, QAction, QProgressBar, QPushButton, QHBoxLayout
)
from PyQt5.QtGui import QIcon

# -------------------------------
# Auto-installer
# -------------------------------
def safe_import(module, pip_name=None):
    try:
        return __import__(module)
    except ImportError:
        import subprocess, sys
        pkg = pip_name if pip_name else module
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return __import__(module)

np = safe_import("numpy", "numpy")

try:
    cl = safe_import("pyopencl", "pyopencl")
    OPENCL_AVAILABLE = True
except Exception:
    cl = None
    OPENCL_AVAILABLE = False

try:
    pynvml = safe_import("pynvml", "pynvml")
    NVML_AVAILABLE = True
except Exception:
    pynvml = None
    NVML_AVAILABLE = False

try:
    wmi = safe_import("wmi", "wmi")
    WMI_AVAILABLE = True
except Exception:
    wmi = None
    WMI_AVAILABLE = False

# game detection
try:
    psutil = safe_import("psutil", "psutil")
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


# -------------------------------
# Backplane
# -------------------------------
def choose_backplane_root():
    candidates = ["D:"] + [f"{chr(c)}:" for c in range(ord("F"), ord("Z") + 1)]
    for drive in candidates:
        if os.path.exists(drive):
            try:
                test_path = os.path.join(drive, ".__golden_star_test__")
                with open(test_path, "w") as f:
                    f.write("test")
                os.remove(test_path)
                return drive
            except Exception:
                continue
    return "C:"

BACKPLANE_DRIVE = choose_backplane_root()

BASE_DIR = os.path.join(BACKPLANE_DRIVE, "golden_star_system")

DIRS = {
    "jobs": os.path.join(BASE_DIR, "jobs"),
    "encode": os.path.join(BASE_DIR, "jobs", "encode"),
    "hash": os.path.join(BASE_DIR, "jobs", "hash"),
    "logs": os.path.join(BASE_DIR, "logs"),
    "state": os.path.join(BASE_DIR, "state"),
    "temp": os.path.join(BASE_DIR, "temp"),
    "swarm": os.path.join(BASE_DIR, "swarm"),
    "swarm_jobs": os.path.join(BASE_DIR, "swarm", "jobs"),
    "swarm_files": os.path.join(BASE_DIR, "swarm", "files"),
    "collective": os.path.join(BASE_DIR, "collective"),
    "raft": os.path.join(BASE_DIR, "raft"),
    "profiles": os.path.join(BASE_DIR, "profiles"),
}

ROUTING_STATE_FILE = os.path.join(DIRS["state"], "routing.json")
POLICY_FILE = os.path.join(DIRS["state"], "policy.json")
LEARNING_LOG = os.path.join(DIRS["state"], "learning_log.jsonl")
COLLECTIVE_POLICY_FILE = os.path.join(DIRS["collective"], "policy_collective.json")
COLLECTIVE_MEMORY_FILE = os.path.join(DIRS["collective"], "memory.json")
COLLECTIVE_MODE_FILE = os.path.join(DIRS["collective"], "mode.json")
COLLECTIVE_REPUTATION_FILE = os.path.join(DIRS["collective"], "reputation.json")
COLLECTIVE_BEHAVIOR_FILE = os.path.join(DIRS["collective"], "behavior_regimes.json")
ANOMALY_FILE = os.path.join(DIRS["collective"], "anomalies.json")

GAME_PROFILE_FILE = os.path.join(DIRS["profiles"], "game_profiles.json")

RAFT_STATE_FILE = lambda node_id: os.path.join(DIRS["raft"], f"{node_id}_state.json")
RAFT_LOG_FILE = os.path.join(DIRS["raft"], "raft_log.jsonl")
RAFT_RPC_DIR = os.path.join(DIRS["raft"], "rpc")


def ensure_directories():
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)

    if not os.path.exists(ROUTING_STATE_FILE):
        with open(ROUTING_STATE_FILE, "w") as f:
            json.dump({"encoder": {"MPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0},
                                   "dGPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0}},
                       "compute": {"MPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0},
                                   "CPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0}}}, f)

    if not os.path.exists(POLICY_FILE):
        default_policy = {
            "pressure_weights": {"load": 0.6, "temp": 0.3, "queue": 2.0},
            "temp_limits": {
                "baseline": 85,
                "hyperfocus": 90,
                "defensive": 75,
                "exploratory": 88
            },
            "load_limits": {
                "baseline": 90,
                "hyperfocus": 95,
                "defensive": 60,
                "exploratory": 90
            },
            "exploration_rate": 0.1,
            "last_tune": None,
            "epochs": [],
            "rl_q_table_global": {},
            "rl_q_table_local": {},
            "heuristic_version": 0,
        }
        with open(POLICY_FILE, "w") as f:
            json.dump(default_policy, f, indent=2)

    if not os.path.exists(COLLECTIVE_POLICY_FILE):
        with open(COLLECTIVE_POLICY_FILE, "w") as f:
            json.dump({"epochs": [], "last_merge": None}, f, indent=2)

    if not os.path.exists(COLLECTIVE_MEMORY_FILE):
        with open(COLLECTIVE_MEMORY_FILE, "w") as f:
            json.dump({"events": [], "last_update": None}, f, indent=2)

    if not os.path.exists(COLLECTIVE_MODE_FILE):
        with open(COLLECTIVE_MODE_FILE, "w") as f:
            json.dump({
                "mode": "baseline",
                "reason": "initial",
                "override": False,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

    if not os.path.exists(COLLECTIVE_REPUTATION_FILE):
        with open(COLLECTIVE_REPUTATION_FILE, "w") as f:
            json.dump({"nodes": {}}, f, indent=2)

    if not os.path.exists(COLLECTIVE_BEHAVIOR_FILE):
        with open(COLLECTIVE_BEHAVIOR_FILE, "w") as f:
            json.dump({"snapshots": [], "regimes": []}, f, indent=2)

    if not os.path.exists(ANOMALY_FILE):
        with open(ANOMALY_FILE, "w") as f:
            json.dump({"events": []}, f, indent=2)

    os.makedirs(RAFT_RPC_DIR, exist_ok=True)

    for logname in ["mpu.log", "encode.log", "hash.log", "swarm.log", "policy.log", "collective.log", "raft.log"]:
        logpath = os.path.join(DIRS["logs"], logname)
        if not os.path.exists(logpath):
            with open(logpath, "w") as f:
                f.write("")

    # default game profiles
    if not os.path.exists(GAME_PROFILE_FILE):
        default_profiles = {
            "profiles": [
                {
                    "name": "Generic FPS",
                    "executables": ["cs2.exe", "valorant.exe", "cod.exe"],
                    "preferred_mode": "defensive",
                    "keep_igpu_free": True,
                    "hard_lock_defensive": True
                },
                {
                    "name": "MMO / RPG",
                    "executables": ["wow.exe", "ffxiv_dx11.exe"],
                    "preferred_mode": "baseline",
                    "keep_igpu_free": True,
                    "hard_lock_defensive": False
                },
                {
                    "name": "Media / Streaming",
                    "executables": ["obs64.exe"],
                    "preferred_mode": "hyperfocus",
                    "keep_igpu_free": False,
                    "hard_lock_defensive": False
                }
            ]
        }
        with open(GAME_PROFILE_FILE, "w") as f:
            json.dump(default_profiles, f, indent=2)

ensure_directories()


# -------------------------------
# Utils
# -------------------------------
def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def log_message(name, msg):
    path = os.path.join(DIRS["logs"], f"{name}.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


# -------------------------------
# Routing + Policy
# -------------------------------
def load_routing_stats():
    return load_json(ROUTING_STATE_FILE, {
        "encoder": {"MPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0},
                    "dGPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0}},
        "compute": {"MPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0},
                    "CPU": {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0}}
    })

def save_routing_stats(stats):
    save_json(ROUTING_STATE_FILE, stats)

def load_policy():
    return load_json(POLICY_FILE, None)

def save_policy(policy):
    save_json(POLICY_FILE, policy)

def append_learning_log(entry):
    try:
        with open(LEARNING_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# -------------------------------
# Collective Memory / Policy / Reputation / Behavior / Anomaly
# -------------------------------
def append_collective_memory(event):
    data = load_json(COLLECTIVE_MEMORY_FILE, {"events": [], "last_update": None})
    data["events"].append(event)
    data["last_update"] = datetime.now().isoformat()
    if len(data["events"]) > 5000:
        compress_collective_memory(data)
    save_json(COLLECTIVE_MEMORY_FILE, data)

def compress_collective_memory(data=None):
    if data is None:
        data = load_json(COLLECTIVE_MEMORY_FILE, {"events": [], "last_update": None})
    events = data.get("events", [])
    if len(events) < 2000:
        return
    cutoff = len(events) - 1500
    old = events[:cutoff]
    new = events[cutoff:]
    summary = {
        "type": "summary",
        "timestamp": datetime.now().isoformat(),
        "count": len(old),
        "domains": {},
    }
    for e in old:
        d = e.get("domain", "unknown")
        t = e.get("target", "unknown")
        success = e.get("success", False)
        summary["domains"].setdefault(d, {}).setdefault(t, {"success": 0, "fail": 0})
        if success:
            summary["domains"][d][t]["success"] += 1
        else:
            summary["domains"][d][t]["fail"] += 1
    new.append(summary)
    data["events"] = new
    save_json(COLLECTIVE_MEMORY_FILE, data)
    log_message("collective", "Collective memory compressed")

def merge_policy_into_collective(local_policy, node_id):
    coll = load_json(COLLECTIVE_POLICY_FILE, {"epochs": [], "last_merge": None})
    epoch = {
        "timestamp": datetime.now().isoformat(),
        "node_id": node_id,
        "policy": local_policy
    }
    coll["epochs"].append(epoch)
    coll["last_merge"] = datetime.now().isoformat()
    coll["epochs"] = coll["epochs"][-500:]
    save_json(COLLECTIVE_POLICY_FILE, coll)
    log_message("collective", f"Policy merged from node {node_id}")

def query_collective_memory(
    domain=None,
    target=None,
    success=None,
    since=None,
    min_arousal=None,
    max_arousal=None,
    mode=None
):
    data = load_json(COLLECTIVE_MEMORY_FILE, {"events": [], "last_update": None})
    events = data.get("events", [])
    now = datetime.now()
    cutoff = None
    if since:
        unit = since[-1]
        val = int(since[:-1])
        if unit == "m":
            cutoff = now - timedelta(minutes=val)
        elif unit == "h":
            cutoff = now - timedelta(hours=val)
        elif unit == "d":
            cutoff = now - timedelta(days=val)

    results = []
    for e in events:
        if e.get("type") == "summary":
            continue
        ts = e.get("timestamp")
        try:
            t = datetime.fromisoformat(ts) if ts else None
        except Exception:
            t = None
        if cutoff and t and t < cutoff:
            continue
        if domain is not None and e.get("domain") != domain:
            continue
        if target is not None and e.get("target") != target:
            continue
        if success is not None and e.get("success") != success:
            continue
        c = e.get("context", {})
        if mode is not None and c.get("mode") != mode:
            continue
        ar = c.get("arousal")
        if min_arousal is not None and ar is not None and ar < min_arousal:
            continue
        if max_arousal is not None and ar is not None and ar > max_arousal:
            continue
        results.append(e)
    return results

def load_reputation():
    return load_json(COLLECTIVE_REPUTATION_FILE, {"nodes": {}})

def save_reputation(rep):
    save_json(COLLECTIVE_REPUTATION_FILE, rep)

def update_reputation(node_id, success, stolen_from=False, stolen_by=False):
    rep = load_reputation()
    nodes = rep.setdefault("nodes", {})
    n = nodes.setdefault(node_id, {
        "successes": 0,
        "failures": 0,
        "stolen_from": 0,
        "stolen_by": 0,
        "score": 0.0,
        "last_seen": None
    })
    if success:
        n["successes"] += 1
        n["score"] += 1.0
    else:
        n["failures"] += 1
        n["score"] -= 1.0
    if stolen_from:
        n["stolen_from"] += 1
        n["score"] -= 0.5
    if stolen_by:
        n["stolen_by"] += 1
        n["score"] += 0.2
    n["last_seen"] = datetime.now().isoformat()
    save_reputation(rep)

def record_behavior_snapshot(swarm_c, global_mode):
    data = load_json(COLLECTIVE_BEHAVIOR_FILE, {"snapshots": [], "regimes": []})
    snap = {
        "timestamp": datetime.now().isoformat(),
        "arousal": swarm_c["arousal"],
        "stability": swarm_c["stability"],
        "pressure": swarm_c["pressure"],
        "node_count": swarm_c["node_count"],
        "mode": global_mode
    }
    data["snapshots"].append(snap)
    data["snapshots"] = data["snapshots"][-1000:]
    save_json(COLLECTIVE_BEHAVIOR_FILE, data)

def record_anomaly(event):
    data = load_json(ANOMALY_FILE, {"events": []})
    data["events"].append(event)
    data["events"] = data["events"][-500:]
    save_json(ANOMALY_FILE, data)
    log_message("collective", f"Anomaly detected: {event}")


# -------------------------------
# Game Profiles
# -------------------------------
class GameProfileManager:
    def __init__(self):
        self.profiles = load_json(GAME_PROFILE_FILE, {"profiles": []}).get("profiles", [])
        self.last_active = None
        self.last_profile = None

    def reload(self):
        self.profiles = load_json(GAME_PROFILE_FILE, {"profiles": []}).get("profiles", [])

    def detect_active_game(self):
        if not PSUTIL_AVAILABLE:
            return None, None
        exe_map = {}
        for p in self.profiles:
            for exe in p.get("executables", []):
                exe_map[exe.lower()] = p
        active_game = None
        active_profile = None
        try:
            for proc in psutil.process_iter(["name"]):
                name = proc.info.get("name")
                if not name:
                    continue
                lname = name.lower()
                if lname in exe_map:
                    active_game = name
                    active_profile = exe_map[lname]
                    break
        except Exception:
            pass
        self.last_active = active_game
        self.last_profile = active_profile
        return active_game, active_profile


# -------------------------------
# Consciousness
# -------------------------------
class ConsciousnessState:
    def __init__(self):
        self.arousal = 0.0
        self.confidence = 0.5
        self.curiosity = 0.5
        self.stability = 0.5
        self.last_update = datetime.now()

    def update(self, pressure, temp, queue_depth, success_rate, failures_recent, idle_ticks):
        self.arousal = min(1.0, (pressure / 300.0) + (temp / 120.0) + (queue_depth / 20.0))
        self.confidence = max(0.0, min(1.0, success_rate))
        self.curiosity = max(0.0, min(1.0, idle_ticks / 20.0))
        self.stability = max(0.0, min(1.0, 1.0 - min(1.0, failures_recent / 10.0)))
        self.last_update = datetime.now()

    def choose_mode_and_narrative(self, degraded):
        if degraded or self.stability < 0.3 or self.arousal > 0.9:
            return "defensive", "Stability low, arousal high—entering defensive posture."
        if self.arousal > 0.7 and self.confidence > 0.6:
            return "hyperfocus", "Arousal high, confidence strong—hyperfocus to clear backlog."
        if self.curiosity > 0.6 and self.arousal < 0.5:
            return "exploratory", "Curiosity elevated, pressure low—exploring routing space."
        return "baseline", "Conditions stable—baseline consciousness."

    def to_dict(self):
        return {
            "arousal": self.arousal,
            "confidence": self.confidence,
            "curiosity": self.curiosity,
            "stability": self.stability
        }


# -------------------------------
# Raft Node (file-based RPC)
# -------------------------------
class RaftNode(QThread):
    leader_changed = pyqtSignal(str)
    log_committed = pyqtSignal(dict)

    def __init__(self, node_id, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.state = "follower"
        self.currentTerm = 0
        self.votedFor = None
        self.log = []
        self.commitIndex = -1
        self.lastApplied = -1
        self.nextIndex = {}
        self.matchIndex = {}
        self.leader_id = None
        self._running = True
        self.election_timeout = timedelta(seconds=10)
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 3
        self._load_state()

    def _state_path(self):
        return RAFT_STATE_FILE(self.node_id)

    def _load_state(self):
        data = load_json(self._state_path(), {
            "currentTerm": 0,
            "votedFor": None,
            "log": []
        })
        self.currentTerm = data["currentTerm"]
        self.votedFor = data["votedFor"]
        self.log = data["log"]

    def _save_state(self):
        data = {
            "currentTerm": self.currentTerm,
            "votedFor": self.votedFor,
            "log": self.log
        }
        save_json(self._state_path(), data)

    def stop(self):
        self._running = False

    def _rpc_file(self, rpc_type, src, dst):
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return os.path.join(RAFT_RPC_DIR, f"{rpc_type}_{src}_to_{dst}_{ts}.json")

    def _broadcast_nodes(self):
        nodes = []
        for fn in glob.glob(os.path.join(DIRS["swarm"], "*.json")):
            try:
                with open(fn, "r") as f:
                    data = json.load(f)
                nid = data.get("node_id")
                if nid and nid != self.node_id:
                    nodes.append(nid)
            except Exception:
                continue
        return nodes

    def _write_rpc(self, rpc_type, dst, payload):
        path = self._rpc_file(rpc_type, self.node_id, dst)
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _read_rpcs(self):
        rpcs = []
        for fn in glob.glob(os.path.join(RAFT_RPC_DIR, "*.json")):
            try:
                with open(fn, "r") as f:
                    data = json.load(f)
                rpc_type = data.get("type")
                dst = data.get("dst")
                if dst == self.node_id:
                    rpcs.append((fn, data))
            except Exception:
                continue
        return rpcs

    def _delete_rpc(self, path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _become_follower(self, term, leader_id=None):
        self.state = "follower"
        self.currentTerm = term
        self.votedFor = None
        self.leader_id = leader_id
        self._save_state()
        if leader_id:
            self.leader_changed.emit(leader_id)
            log_message("raft", f"{self.node_id} became follower, leader={leader_id}, term={term}")

    def _become_candidate(self):
        self.state = "candidate"
        self.currentTerm += 1
        self.votedFor = self.node_id
        self._save_state()
        self.last_heartbeat = datetime.now()
        log_message("raft", f"{self.node_id} became candidate, term={self.currentTerm}")
        self._request_votes()

    def _become_leader(self):
        self.state = "leader"
        self.leader_id = self.node_id
        self.nextIndex = {}
        self.matchIndex = {}
        last_index = len(self.log)
        for nid in self._broadcast_nodes():
            self.nextIndex[nid] = last_index
            self.matchIndex[nid] = -1
        self.leader_changed.emit(self.node_id)
        log_message("raft", f"{self.node_id} became leader, term={self.currentTerm}")

    def _last_log_index_term(self):
        if not self.log:
            return -1, 0
        idx = len(self.log) - 1
        return idx, self.log[idx]["term"]

    def _request_votes(self):
        last_index, last_term = self._last_log_index_term()
        payload = {
            "type": "RequestVote",
            "src": self.node_id,
            "dst": None,
            "term": self.currentTerm,
            "candidateId": self.node_id,
            "lastLogIndex": last_index,
            "lastLogTerm": last_term,
        }
        for nid in self._broadcast_nodes():
            p = payload.copy()
            p["dst"] = nid
            self._write_rpc("RequestVote", nid, p)

    def _handle_request_vote(self, rpc):
        term = rpc["term"]
        candidateId = rpc["candidateId"]
        lastLogIndex = rpc["lastLogIndex"]
        lastLogTerm = rpc["lastLogTerm"]

        if term < self.currentTerm:
            granted = False
        else:
            if term > self.currentTerm:
                self._become_follower(term)
            up_to_date = False
            my_last_index, my_last_term = self._last_log_index_term()
            if lastLogTerm > my_last_term or (lastLogTerm == my_last_term and lastLogIndex >= my_last_index):
                up_to_date = True
            if (self.votedFor is None or self.votedFor == candidateId) and up_to_date:
                self.votedFor = candidateId
                self._save_state()
                granted = True
                self.last_heartbeat = datetime.now()
            else:
                granted = False

        reply = {
            "type": "RequestVoteReply",
            "src": self.node_id,
            "dst": rpc["src"],
            "term": self.currentTerm,
            "voteGranted": granted
        }
        self._write_rpc("RequestVoteReply", rpc["src"], reply)

    def _handle_request_vote_reply(self, rpc):
        if self.state != "candidate":
            return
        term = rpc["term"]
        if term > self.currentTerm:
            self._become_follower(term)
            return
        if term < self.currentTerm:
            return
        if rpc["voteGranted"]:
            votes = 1
            try:
                for fn, data in self._read_rpcs():
                    if data.get("type") == "RequestVoteReply" and data.get("dst") == self.node_id and data.get("term") == self.currentTerm and data.get("voteGranted"):
                        votes += 1
            except Exception:
                pass
            total_nodes = len(self._broadcast_nodes()) + 1
            if votes > total_nodes / 2:
                self._become_leader()

    def _append_entries(self, entries):
        for e in entries:
            self.log.append(e)
        self._save_state()

    def _handle_append_entries(self, rpc):
        term = rpc["term"]
        leaderId = rpc["leaderId"]
        prevLogIndex = rpc["prevLogIndex"]
        prevLogTerm = rpc["prevLogTerm"]
        entries = rpc["entries"]
        leaderCommit = rpc["leaderCommit"]

        if term < self.currentTerm:
            success = False
        else:
            if term > self.currentTerm or self.state != "follower":
                self._become_follower(term, leaderId)
            self.last_heartbeat = datetime.now()
            if prevLogIndex >= 0:
                if prevLogIndex >= len(self.log) or self.log[prevLogIndex]["term"] != prevLogTerm:
                    success = False
                else:
                    self.log = self.log[:prevLogIndex + 1]
                    self._append_entries(entries)
                    success = True
            else:
                self.log = []
                self._append_entries(entries)
                success = True

            if leaderCommit > self.commitIndex:
                self.commitIndex = min(leaderCommit, len(self.log) - 1)
                self._apply_commits()

        reply = {
            "type": "AppendEntriesReply",
            "src": self.node_id,
            "dst": rpc["src"],
            "term": self.currentTerm,
            "success": success,
            "matchIndex": self.commitIndex if success else -1
        }
        self._write_rpc("AppendEntriesReply", rpc["src"], reply)

    def _handle_append_entries_reply(self, rpc):
        if self.state != "leader":
            return
        term = rpc["term"]
        if term > self.currentTerm:
            self._become_follower(term)
            return
        if not rpc["success"]:
            return
        nid = rpc["src"]
        matchIndex = rpc["matchIndex"]
        self.matchIndex[nid] = matchIndex
        self.nextIndex[nid] = matchIndex + 1

        for N in range(len(self.log) - 1, self.commitIndex, -1):
            count = 1
            for other in self.matchIndex:
                if self.matchIndex[other] >= N:
                    count += 1
            if count > (len(self._broadcast_nodes()) + 1) / 2 and self.log[N]["term"] == self.currentTerm:
                self.commitIndex = N
                self._apply_commits()
                break

    def _apply_commits(self):
        while self.lastApplied < self.commitIndex:
            self.lastApplied += 1
            entry = self.log[self.lastApplied]
            self.log_committed.emit(entry)
            log_message("raft", f"{self.node_id} applied log index {self.lastApplied}: {entry}")

    def propose_command(self, command):
        if self.state != "leader":
            return False
        entry = {"term": self.currentTerm, "command": command}
        self.log.append(entry)
        self._save_state()
        self._broadcast_append_entries()
        return True

    def _broadcast_append_entries(self):
        last_index = len(self.log) - 1
        for nid in self._broadcast_nodes():
            next_idx = self.nextIndex.get(nid, last_index)
            prev_idx = next_idx - 1
            prev_term = self.log[prev_idx]["term"] if prev_idx >= 0 else 0
            entries = self.log[next_idx:]
            payload = {
                "type": "AppendEntries",
                "src": self.node_id,
                "dst": nid,
                "term": self.currentTerm,
                "leaderId": self.node_id,
                "prevLogIndex": prev_idx,
                "prevLogTerm": prev_term,
                "entries": entries,
                "leaderCommit": self.commitIndex
            }
            self._write_rpc("AppendEntries", nid, payload)

    def _send_heartbeats(self):
        for nid in self._broadcast_nodes():
            prev_idx = len(self.log) - 1
            prev_term = self.log[prev_idx]["term"] if prev_idx >= 0 else 0
            payload = {
                "type": "AppendEntries",
                "src": self.node_id,
                "dst": nid,
                "term": self.currentTerm,
                "leaderId": self.node_id,
                "prevLogIndex": prev_idx,
                "prevLogTerm": prev_term,
                "entries": [],
                "leaderCommit": self.commitIndex
            }
            self._write_rpc("AppendEntries", nid, payload)

    def _process_rpcs(self):
        for path, rpc in self._read_rpcs():
            t = rpc.get("type")
            if t == "RequestVote":
                self._handle_request_vote(rpc)
            elif t == "RequestVoteReply":
                self._handle_request_vote_reply(rpc)
            elif t == "AppendEntries":
                self._handle_append_entries(rpc)
            elif t == "AppendEntriesReply":
                self._handle_append_entries_reply(rpc)
            self._delete_rpc(path)

    def run(self):
        while self._running:
            self._process_rpcs()
            now = datetime.now()
            if self.state in ("follower", "candidate"):
                if now - self.last_heartbeat > self.election_timeout:
                    self._become_candidate()
            elif self.state == "leader":
                if (now - self.last_heartbeat).total_seconds() >= self.heartbeat_interval:
                    self._send_heartbeats()
                    self.last_heartbeat = now
            time.sleep(0.5)


# -------------------------------
# Swarm Consciousness + Global RL + Behavior + Anomaly
# -------------------------------
class SwarmConsciousness(QThread):
    swarm_consciousness_updated = pyqtSignal(dict)

    def __init__(self, node_id, raft_node, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.raft = raft_node
        self._running = True
        self.rl_alpha = 0.2
        self.rl_gamma = 0.9
        self.last_state_action = None

    def stop(self):
        self._running = False

    def _read_nodes(self):
        nodes = {}
        for fn in glob.glob(os.path.join(DIRS["swarm"], "*.json")):
            if os.path.basename(fn).startswith("jobs"):
                continue
            try:
                with open(fn, "r") as f:
                    data = json.load(f)
                nid = data.get("node_id")
                if nid:
                    nodes[nid] = data
            except Exception:
                continue
        return nodes

    def _compute_swarm_consciousness(self, nodes):
        if not nodes:
            return {
                "arousal": 0.0,
                "stability": 1.0,
                "pressure": 0.0,
                "node_count": 0
            }
        loads = []
        temps = []
        queues = []
        for n in nodes.values():
            load = n.get("load", {}).get("igpu_load", 0)
            temp = n.get("load", {}).get("igpu_temp", 0)
            eq = n.get("load", {}).get("encode_queue", 0)
            hq = n.get("load", {}).get("hash_queue", 0)
            loads.append(load)
            temps.append(temp)
            queues.append(eq + hq)
        avg_load = sum(loads) / len(loads)
        avg_temp = sum(temps) / len(temps)
        avg_queue = sum(queues) / len(queues)
        pressure = avg_load * 0.6 + avg_temp * 0.3 + avg_queue * 2.0
        arousal = min(1.0, (pressure / 300.0) + (avg_temp / 120.0) + (avg_queue / 20.0))
        stability = max(0.0, min(1.0, 1.0 - min(1.0, avg_queue / 50.0)))
        return {
            "arousal": arousal,
            "stability": stability,
            "pressure": pressure,
            "node_count": len(nodes)
        }

    def _state_bucket(self, swarm_c):
        a = swarm_c["arousal"]
        s = swarm_c["stability"]
        p = swarm_c["pressure"]
        def bucket(x):
            if x < 0.33: return "low"
            if x < 0.66: return "mid"
            return "high"
        return f"a:{bucket(a)}|s:{bucket(s)}|p:{'low' if p<150 else 'mid' if p<300 else 'high'}"

    def _load_q_table(self):
        policy = load_policy() or {}
        return policy.get("rl_q_table_global", {}), policy

    def _save_q_table(self, q_table, policy):
        policy["rl_q_table_global"] = q_table
        save_policy(policy)
        merge_policy_into_collective(policy, self.node_id)

    def _choose_mode_rl(self, swarm_c):
        q_table, policy = self._load_q_table()
        state = self._state_bucket(swarm_c)
        modes = ["baseline", "hyperfocus", "defensive", "exploratory"]
        if state not in q_table:
            q_table[state] = {m: 0.0 for m in modes}
        eps = policy.get("exploration_rate", 0.1)
        if np.random.rand() < eps:
            mode = np.random.choice(modes)
        else:
            mode = max(q_table[state], key=lambda m: q_table[state][m])
        self.last_state_action = (state, mode)
        self._save_q_table(q_table, policy)
        return mode

    def _compute_reward(self, swarm_c):
        r = 0.0
        r += swarm_c["stability"] * 2.0
        r -= (swarm_c["pressure"] / 300.0)
        if swarm_c["arousal"] > 0.8:
            r -= 0.5
        return r

    def _update_q_table(self, new_swarm_c):
        if not self.last_state_action:
            return
        q_table, policy = self._load_q_table()
        state, mode = self.last_state_action
        if state not in q_table:
            q_table[state] = {m: 0.0 for m in ["baseline", "hyperfocus", "defensive", "exploratory"]}
        reward = self._compute_reward(new_swarm_c)
        new_state = self._state_bucket(new_swarm_c)
        if new_state not in q_table:
            q_table[new_state] = {m: 0.0 for m in ["baseline", "hyperfocus", "defensive", "exploratory"]}
        best_next = max(q_table[new_state].values())
        old = q_table[state][mode]
        q_table[state][mode] = old + self.rl_alpha * (reward + self.rl_gamma * best_next - old)
        self._save_q_table(q_table, policy)
        log_message("policy", f"Global RL update: state={state}, mode={mode}, reward={reward:.3f}")

    def _write_global_mode(self, mode, reason, override):
        cmd = {
            "type": "set_global_mode",
            "mode": mode,
            "reason": reason,
            "override": override,
            "timestamp": datetime.now().isoformat()
        }
        self.raft.propose_command(cmd)

    def _detect_regime(self, swarm_c, mode):
        a = swarm_c["arousal"]
        s = swarm_c["stability"]
        p = swarm_c["pressure"]
        if a > 0.8 and s < 0.4:
            return "panic"
        if p > 300 and s < 0.6:
            return "thrash"
        if p < 150 and s > 0.7 and a < 0.5:
            return "calm_efficiency"
        if p < 150 and s > 0.8 and a < 0.3:
            return "over_cautious"
        return "normal"

    def _behavior_driven_policy_rewrite(self, regime):
        policy = load_policy() or {}
        changed = False
        if regime == "panic":
            policy["pressure_weights"]["queue"] = min(3.0, policy["pressure_weights"]["queue"] + 0.1)
            for k in policy["temp_limits"]:
                policy["temp_limits"][k] = max(70, policy["temp_limits"][k] - 1)
            changed = True
        elif regime == "thrash":
            policy["exploration_rate"] = max(0.01, policy["exploration_rate"] - 0.01)
            changed = True
        elif regime == "over_cautious":
            for k in policy["temp_limits"]:
                policy["temp_limits"][k] = min(95, policy["temp_limits"][k] + 1)
            changed = True
        if changed:
            policy["heuristic_version"] = policy.get("heuristic_version", 0) + 1
            save_policy(policy)
            merge_policy_into_collective(policy, self.node_id)
            log_message("policy", f"Behavior-driven rewrite for regime={regime}, version={policy['heuristic_version']}")

    def _anomaly_detection(self, swarm_c, history):
        if len(history) < 20:
            return
        pressures = [h["pressure"] for h in history]
        mean_p = sum(pressures) / len(pressures)
        var_p = sum((x - mean_p) ** 2 for x in pressures) / len(pressures)
        std_p = math.sqrt(var_p)
        if std_p == 0:
            return
        z = (swarm_c["pressure"] - mean_p) / std_p
        if abs(z) > 3:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": "pressure_anomaly",
                "pressure": swarm_c["pressure"],
                "mean": mean_p,
                "std": std_p,
                "z": z
            }
            record_anomaly(event)
            cmd = {
                "type": "set_global_mode",
                "mode": "defensive",
                "reason": "Swarm-wide anomaly detected (pressure spike)",
                "override": True,
                "timestamp": datetime.now().isoformat()
            }
            self.raft.propose_command(cmd)

    def run(self):
        history = deque(maxlen=100)
        last_rl_update_state = None
        while self._running:
            nodes = self._read_nodes()
            swarm_c = self._compute_swarm_consciousness(nodes)
            self.swarm_consciousness_updated.emit(swarm_c)
            history.append(swarm_c)

            if self.raft.state == "leader":
                mode, reason, override = self._decide_global_mode(swarm_c)
                self._write_global_mode(mode, reason, override)
                record_behavior_snapshot(swarm_c, mode)
                regime = self._detect_regime(swarm_c, mode)
                self._behavior_driven_policy_rewrite(regime)
                self._anomaly_detection(swarm_c, history)
                if last_rl_update_state is not None:
                    self._update_q_table(swarm_c)
                last_rl_update_state = swarm_c

            time.sleep(5)

    def _decide_global_mode(self, swarm_c):
        ar = swarm_c["arousal"]
        st = swarm_c["stability"]
        p = swarm_c["pressure"]

        override = False
        reason = ""

        if p > 350 or (ar > 0.9 and st < 0.4):
            mode = "defensive"
            override = True
            reason = "Swarm pressure critical or stability very low—forcing global defensive override."
        elif p > 280 and st > 0.6:
            mode = "hyperfocus"
            override = True
            reason = "High pressure but stable—forcing global hyperfocus."
        else:
            mode = self._choose_mode_rl(swarm_c)
            reason = "RL-chosen global mode based on swarm state."

        return mode, reason, override


# -------------------------------
# Swarm Node
# -------------------------------
class SwarmNode(QThread):
    swarm_updated = pyqtSignal(dict)

    def __init__(self, node_id, capabilities, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.capabilities = capabilities
        self._running = True
        self.node_file = os.path.join(DIRS["swarm"], f"{self.node_id}.json")
        self.last_summary = {}

    def stop(self):
        self._running = False

    def write_heartbeat(self, load_summary):
        data = {
            "node_id": self.node_id,
            "hostname": socket.gethostname(),
            "capabilities": self.capabilities,
            "load": load_summary,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(self.node_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log_message("swarm", f"Heartbeat write failed: {e}")

    def read_swarm(self):
        nodes = {}
        for fn in glob.glob(os.path.join(DIRS["swarm"], "*.json")):
            if os.path.basename(fn).startswith("jobs"):
                continue
            try:
                with open(fn, "r") as f:
                    data = json.load(f)
                nid = data.get("node_id")
                if nid:
                    nodes[nid] = data
            except Exception:
                continue
        return nodes

    def run(self):
        while self._running:
            nodes = self.read_swarm()
            self.last_summary = nodes
            self.swarm_updated.emit(nodes)
            self.msleep(3000)


# -------------------------------
# Distributed Locking
# -------------------------------
def acquire_lock(lock_path, timeout=5.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.1)
    return False

def release_lock(lock_path):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


# -------------------------------
# Swarm Job Manager
# -------------------------------
class SwarmJobManager(QThread):
    job_received = pyqtSignal(dict)
    job_completed = pyqtSignal(dict)

    def __init__(self, node_id, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self._running = True
        self.jobs_dir = DIRS["swarm_jobs"]
        os.makedirs(self.jobs_dir, exist_ok=True)

    def stop(self):
        self._running = False

    def list_jobs(self):
        jobs = []
        for fn in glob.glob(os.path.join(self.jobs_dir, "*.json")):
            try:
                with open(fn, "r") as f:
                    jobs.append(json.load(f))
            except Exception:
                continue
        return jobs

    def _job_path(self, job_id):
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _lock_path(self, job_id):
        return os.path.join(self.jobs_dir, f"{job_id}.lock")

    def claim_job(self, job):
        job_id = job["job_id"]
        lock_path = self._lock_path(job_id)
        if not acquire_lock(lock_path, timeout=0.5):
            return False
        try:
            if job.get("assigned_to") is not None and job["assigned_to"] != self.node_id:
                release_lock(lock_path)
                return False
            job["assigned_to"] = self.node_id
            job["status"] = "claimed"
            job["claimed_at"] = datetime.now().isoformat()
            with open(self._job_path(job_id), "w") as f:
                json.dump(job, f, indent=2)
            return True
        finally:
            release_lock(lock_path)

    def complete_job(self, job, result):
        job_id = job["job_id"]
        lock_path = self._lock_path(job_id)
        if not acquire_lock(lock_path, timeout=1.0):
            return
        try:
            job["status"] = "done"
            job["result"] = result
            job["completed_at"] = datetime.now().isoformat()
            with open(self._job_path(job_id), "w") as f:
                json.dump(job, f, indent=2)
            self.job_completed.emit(job)
            update_reputation(self.node_id, success=(result.get("status") == "ok"))
        finally:
            release_lock(lock_path)

    def _steal_stuck_jobs(self, jobs, max_age_sec=60):
        now = datetime.now()
        for job in jobs:
            if job.get("status") == "claimed" and job.get("assigned_to") != self.node_id:
                claimed_at = job.get("claimed_at")
                if not claimed_at:
                    continue
                try:
                    t = datetime.fromisoformat(claimed_at)
                except Exception:
                    continue
                age = (now - t).total_seconds()
                if age > max_age_sec:
                    job_id = job["job_id"]
                    lock_path = self._lock_path(job_id)
                    if not acquire_lock(lock_path, timeout=0.5):
                        continue
                    try:
                        prev_node = job.get("assigned_to")
                        job["assigned_to"] = self.node_id
                        job["status"] = "claimed"
                        job["claimed_at"] = datetime.now().isoformat()
                        with open(self._job_path(job_id), "w") as f:
                            json.dump(job, f, indent=2)
                        log_message("swarm", f"Node {self.node_id} stole job {job_id} from {prev_node}")
                        if prev_node:
                            update_reputation(prev_node, success=False, stolen_from=True)
                        update_reputation(self.node_id, success=True, stolen_by=True)
                        self.job_received.emit(job)
                    finally:
                        release_lock(lock_path)

    def run(self):
        while self._running:
            jobs = self.list_jobs()
            for job in jobs:
                if job.get("status") == "pending" and job.get("assigned_to") is None:
                    if self.claim_job(job):
                        self.job_received.emit(job)
            self._steal_stuck_jobs(jobs)
            self.msleep(2000)


# -------------------------------
# Remote file staging
# -------------------------------
def stage_input_file_for_swarm(job):
    if job["type"] == "encode":
        inp = job["payload"]["input"]
        if os.path.exists(inp):
            base = os.path.basename(inp)
            dest = os.path.join(DIRS["swarm_files"], base)
            if not os.path.exists(dest):
                shutil.copy2(inp, dest)
            job["payload"]["input"] = dest
    elif job["type"] == "hash":
        path = job["payload"]["path"]
        if os.path.exists(path):
            base = os.path.basename(path)
            dest = os.path.join(DIRS["swarm_files"], base)
            if not os.path.exists(dest):
                shutil.copy2(path, dest)
            job["payload"]["path"] = dest
    return job


# -------------------------------
# MPU Manager
# -------------------------------
class MPUManager(QObject):
    job_started = pyqtSignal(dict)
    job_finished = pyqtSignal(dict)
    job_failed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.backend = "iGPU"
        self.opencl_ready = False
        self.diagnostics = {
            "pyopencl_installed": OPENCL_AVAILABLE,
            "opencl_device_found": False,
            "opencl_error": None,
            "quick_sync_available": True,
            "hash_backend": "CPU"
        }

        if OPENCL_AVAILABLE:
            self._init_opencl()
        else:
            self.diagnostics["opencl_error"] = "PyOpenCL not installed"

    def _log(self, message, logfile="mpu.log"):
        log_message("mpu", message)

    def _init_opencl(self):
        try:
            platforms = cl.get_platforms()
            self.device = None
            for p in platforms:
                for d in p.get_devices():
                    if "Intel" in d.name and d.type & cl.device_type.GPU:
                        self.device = d
                        break
                if self.device:
                    break

            if self.device is None:
                msg = "No Intel OpenCL GPU found"
                self.diagnostics["opencl_error"] = msg
                return

            self.context = cl.Context(devices=[self.device])
            self.queue = cl.CommandQueue(self.context)

            kernel_src = r"""
            __kernel void xor_hash(__global const uchar *data,
                                   __global uint *partial,
                                   const uint length) {
                uint gid = get_global_id(0);
                uint gsize = get_global_size(0);

                uint acc = 0;
                for (uint i = gid; i < length; i += gsize) {
                    acc ^= (uint)data[i];
                }
                partial[gid] = acc;
            }
            """
            self.program = cl.Program(self.context, kernel_src).build()
            self.opencl_ready = True
            self.diagnostics["opencl_device_found"] = True
            self.diagnostics["hash_backend"] = "OpenCL"

        except Exception as e:
            self.opencl_ready = False
            self.diagnostics["opencl_error"] = str(e)

    def submit_job(self, job):
        self._log(f"Started job: {job}")
        self.job_started.emit(job)
        start = time.time()
        try:
            t = job["type"]
            if t == "encode":
                self._run_encode_qsv(job["payload"])
            elif t == "hash":
                job["result"] = self._run_hash(job["payload"])
            else:
                raise ValueError(f"Unknown job type: {t}")
            job["duration"] = time.time() - start
            self.job_finished.emit(job)
            self._log(f"Finished job: {job}")
        except Exception as e:
            job["error"] = str(e)
            job["duration"] = time.time() - start
            self.job_failed.emit(job)
            self._log(f"Job failed: {job} | error={e}")

    def _run_encode_qsv(self, payload):
        inp = payload["input"]
        out = payload["output"]
        bitrate = payload.get("bitrate", "6000k")
        codec = payload.get("codec", "h264_qsv")

        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel", "qsv",
            "-c:v", codec,
            "-i", inp,
            "-c:v", codec,
            "-b:v", bitrate,
            out
        ]
        subprocess.run(cmd, check=True)

    def _run_hash(self, payload):
        if self.opencl_ready:
            return self._run_hash_opencl(payload)
        else:
            return self._run_hash_cpu(payload)

    def _run_hash_cpu(self, payload):
        path = payload["path"]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _run_hash_opencl(self, payload):
        path = payload["path"]
        with open(path, "rb") as f:
            data = f.read()

        data_np = np.frombuffer(data, dtype=np.uint8)
        length = np.uint32(data_np.size)
        if data_np.size == 0:
            return hashlib.sha256(b"").hexdigest()

        global_size = min(1024 * 64, data_np.size)
        partial_np = np.zeros(global_size, dtype=np.uint32)

        mf = cl.mem_flags
        buf_data = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_np)
        buf_partial = cl.Buffer(self.context, mf.WRITE_ONLY, partial_np.nbytes)

        kernel = self.program.xor_hash
        kernel.set_args(buf_data, buf_partial, length)

        cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), None)
        cl.enqueue_copy(self.queue, partial_np, buf_partial)
        self.queue.finish()

        acc = np.uint32(0)
        for v in partial_np:
            acc ^= v

        h = hashlib.sha256()
        h.update(acc.tobytes())
        return h.hexdigest()


# -------------------------------
# Telemetry
# -------------------------------
class TelemetryEngine(QThread):
    telemetry_updated = pyqtSignal(dict)

    def __init__(self, interval=1.0, parent=None):
        super().__init__(parent)
        self.interval = interval
        self._running = True
        self._nvml_init = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_init = True
            except Exception:
                self._nvml_init = False
        self._wmi = wmi.WMI() if WMI_AVAILABLE else None

    def stop(self):
        self._running = False

    def poll_nvidia(self):
        if self._nvml_init:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                return {
                    "load": util.gpu,
                    "vram": int(mem.used / (1024 * 1024)),
                    "temp": temp,
                    "encoder": getattr(util, "encoder", 0)
                }
            except Exception:
                pass
        return {"load": 45, "vram": 3200, "temp": 62, "encoder": 10}

    def poll_intel_igpu(self):
        if self._wmi:
            try:
                return {"load": 20, "vram": 256, "temp": 48, "encoder": 60}
            except Exception:
                pass
        return {"load": 20, "vram": 256, "temp": 48, "encoder": 60}

    def run(self):
        while self._running:
            d = self.poll_nvidia()
            i = self.poll_intel_igpu()
            self.telemetry_updated.emit({"dGPU": d, "iGPU": i})
            self.msleep(int(self.interval * 1000))


def device_pressure(load, temp, queue_depth, policy):
    w = policy["pressure_weights"]
    return load * w["load"] + temp * w["temp"] + queue_depth * w["queue"]


# -------------------------------
# Task Scheduler (multi-agent RL + game-aware)
# -------------------------------
class TaskScheduler(QObject):
    assignment_changed = pyqtSignal(dict)
    mode_changed = pyqtSignal(str, str)
    consciousness_updated = pyqtSignal(dict)
    game_state_changed = pyqtSignal(str, str)

    def __init__(self, node_id, game_profile_manager, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.game_profile_manager = game_profile_manager
        self.current_assignment = {
            "encoder": "MPU",
            "compute": "MPU",
            "mpu_enabled": True
        }
        self.history = deque(maxlen=60)
        self.mode = "auto"
        self.effective_mode = "baseline"
        self.mode_reason = "Initial state"
        self.encode_queue_depth = 0
        self.hash_queue_depth = 0
        self.routing_stats = load_routing_stats()
        self.degraded_flag = False
        self.idle_ticks = 0
        self.consciousness = ConsciousnessState()
        self.policy = load_policy() or {}
        self.last_policy_tune = datetime.now()
        self.is_leader = False
        self.local_rl_alpha = 0.2
        self.local_rl_gamma = 0.9
        self.last_local_state_action = None
        self.active_game = None
        self.active_game_profile = None

    def set_leader(self, leader_id):
        self.is_leader = (leader_id == self.node_id)
        if self.is_leader:
            log_message("swarm", f"Node {self.node_id} is leader")
        else:
            log_message("swarm", f"Node {self.node_id} is follower")

    def set_mode(self, mode):
        self.mode = mode
        if mode != "auto":
            self.effective_mode = mode
            self.mode_reason = "Manual override"
            self.mode_changed.emit(self.effective_mode, self.mode_reason)

    def update_encode_queue(self, depth):
        self.encode_queue_depth = depth

    def update_hash_queue(self, depth):
        self.hash_queue_depth = depth

    def set_degraded(self, degraded):
        self.degraded_flag = degraded

    def _score_route(self, domain, target):
        stats = self.routing_stats.get(domain, {}).get(target, {"success": 0, "fail": 0})
        s = stats.get("success", 0)
        f = stats.get("fail", 0)
        return (s + 1) / (f + 1)

    def _predict_cost(self, domain, target):
        stats = self.routing_stats.get(domain, {}).get(target, {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0})
        s = stats.get("success", 0)
        f = stats.get("fail", 0)
        c = stats.get("count", 0)
        time_sum = stats.get("time_sum", 0.0)
        avg_time = (time_sum / c) if c > 0 else 1.0
        p_success = (s + 1) / (s + f + 2)
        cost = avg_time / p_success
        rep = load_reputation()
        nodes = rep.get("nodes", {})
        node_rep = nodes.get(self.node_id, {}).get("score", 0.0)
        cost *= (1.0 - min(0.3, node_rep / 100.0))
        return cost

    def _local_state_bucket(self, telemetry):
        i = telemetry["iGPU"]
        load = i["load"]
        temp = i["temp"]
        q = self.encode_queue_depth + self.hash_queue_depth
        def bucket(x):
            if x < 33: return "low"
            if x < 66: return "mid"
            return "high"
        return f"load:{bucket(load)}|temp:{bucket(temp)}|q:{bucket(q)}"

    def _load_local_q_table(self):
        policy = load_policy() or {}
        all_local = policy.get("rl_q_table_local", {})
        table = all_local.get(self.node_id, {})
        return table, policy, all_local

    def _save_local_q_table(self, table, policy, all_local):
        all_local[self.node_id] = table
        policy["rl_q_table_local"] = all_local
        save_policy(policy)
        merge_policy_into_collective(policy, self.node_id)

    def _choose_local_actions(self, telemetry):
        table, policy, all_local = self._load_local_q_table()
        state = self._local_state_bucket(telemetry)
        actions = ["MPU", "CPU"]
        if state not in table:
            table[state] = {a: 0.0 for a in actions}
        eps = policy.get("exploration_rate", 0.1)
        if np.random.rand() < eps:
            enc_action = np.random.choice(actions)
            comp_action = np.random.choice(actions)
        else:
            enc_action = max(table[state], key=lambda a: table[state][a])
            comp_action = max(table[state], key=lambda a: table[state][a])
        self.last_local_state_action = (state, enc_action, comp_action)
        self._save_local_q_table(table, policy, all_local)
        return enc_action, comp_action

    def _update_local_q_table(self, reward):
        if not self.last_local_state_action:
            return
        table, policy, all_local = self._load_local_q_table()
        state, enc_action, comp_action = self.last_local_state_action
        actions = ["MPU", "CPU"]
        if state not in table:
            table[state] = {a: 0.0 for a in actions}
        old = table[state][enc_action]
        best_next = max(table[state].values())
        table[state][enc_action] = old + self.local_rl_alpha * (reward + self.local_rl_gamma * best_next - old)
        self._save_local_q_table(table, policy, all_local)
        log_message("policy", f"Local RL update node={self.node_id}, state={state}, action={enc_action}, reward={reward:.3f}")

    def record_outcome(self, domain, target, success, context=None, duration=None):
        if domain not in self.routing_stats:
            self.routing_stats[domain] = {}
        if target not in self.routing_stats[domain]:
            self.routing_stats[domain][target] = {"success": 0, "fail": 0, "time_sum": 0.0, "count": 0}
        key = "success" if success else "fail"
        self.routing_stats[domain][target][key] += 1
        if duration is not None:
            self.routing_stats[domain][target]["time_sum"] += duration
            self.routing_stats[domain][target]["count"] += 1
        save_routing_stats(self.routing_stats)

        ctx = context or {}
        ctx["mode"] = self.effective_mode
        ctx["arousal"] = self.consciousness.arousal
        if self.active_game:
            ctx["active_game"] = self.active_game
            ctx["game_profile"] = self.active_game_profile.get("name") if self.active_game_profile else None

        entry = {
            "timestamp": datetime.now().isoformat(),
            "node_id": self.node_id,
            "domain": domain,
            "target": target,
            "success": success,
            "context": ctx
        }
        append_learning_log(entry)
        append_collective_memory(entry)
        update_reputation(self.node_id, success=success)

        reward = 1.0 if success else -1.0
        if duration is not None:
            reward -= duration / 60.0
        self._update_local_q_table(reward)

    def _auto_select_mode(self, telemetry):
        d = telemetry["dGPU"]
        i = telemetry["iGPU"]

        if len(self.history) >= 5:
            recent = list(self.history)[-5:]
            avg_igpu = sum(t["iGPU"]["load"] for t in recent) / len(recent)
            avg_temp = sum(t["iGPU"]["temp"] for t in recent) / len(recent)
        else:
            avg_igpu = i["load"]
            avg_temp = i["temp"]

        enc_pressure = device_pressure(i["load"], i["temp"], self.encode_queue_depth, self.policy)
        comp_pressure = device_pressure(i["load"], i["temp"], self.hash_queue_depth, self.policy)
        total_pressure = enc_pressure + comp_pressure

        rising_temp = i["temp"] > avg_temp + 3
        rising_load = i["load"] > avg_igpu + 10

        if self.encode_queue_depth == 0 and self.hash_queue_depth == 0:
            self.idle_ticks += 1
        else:
            self.idle_ticks = 0

        successes = 0
        fails = 0
        for domain in self.routing_stats.values():
            for tgt in domain.values():
                successes += tgt.get("success", 0)
                fails += tgt.get("fail", 0)
        total = successes + fails
        success_rate = (successes / total) if total > 0 else 0.5
        failures_recent = fails if fails < 100 else 100

        self.consciousness.update(
            pressure=total_pressure,
            temp=i["temp"],
            queue_depth=self.encode_queue_depth + self.hash_queue_depth,
            success_rate=success_rate,
            failures_recent=failures_recent,
            idle_ticks=self.idle_ticks
        )

        eff_mode, narrative = self.consciousness.choose_mode_and_narrative(self.degraded_flag)

        if self.degraded_flag or rising_temp or i["temp"] > self.policy["temp_limits"]["defensive"] or total_pressure > 300:
            eff_mode = "defensive"
            narrative = "Degraded state or extreme pressure—hard lock into defensive mode."

        return eff_mode, narrative

    def _tune_policy_if_needed(self):
        now = datetime.now()
        last = self.policy.get("last_tune")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
            except Exception:
                last_dt = now
        else:
            last_dt = now

        if (now - last_dt) < timedelta(minutes=5):
            return

        epoch_summary = {
            "timestamp": now.isoformat(),
            "routing_stats": self.routing_stats,
            "consciousness": self.consciousness.to_dict()
        }
        self.policy.setdefault("epochs", []).append(epoch_summary)

        if self.consciousness.arousal > 0.8 and self.consciousness.stability > 0.6:
            self.policy["pressure_weights"]["queue"] = min(
                3.0, self.policy["pressure_weights"]["queue"] + 0.1
            )
        if self.consciousness.stability < 0.4:
            self.policy["temp_limits"]["defensive"] = max(
                70, self.policy["temp_limits"]["defensive"] - 1
            )

        self.policy["last_tune"] = now.isoformat()
        save_policy(self.policy)
        merge_policy_into_collective(self.policy, self.node_id)
        log_message("policy", f"Policy tuned: {self.policy['pressure_weights']} / {self.policy['temp_limits']}")

    def _read_global_mode(self):
        return load_json(COLLECTIVE_MODE_FILE, {
            "mode": "baseline",
            "reason": "no global mode file",
            "override": False,
            "timestamp": None
        })

    def _apply_global_bias(self, global_mode):
        if global_mode == "hyperfocus":
            for k in self.policy["temp_limits"]:
                self.policy["temp_limits"][k] = min(95, self.policy["temp_limits"][k] + 2)
        elif global_mode == "defensive":
            for k in self.policy["temp_limits"]:
                self.policy["temp_limits"][k] = max(65, self.policy["temp_limits"][k] - 2)
        elif global_mode == "exploratory":
            self.policy["pressure_weights"]["queue"] = max(1.5, self.policy["pressure_weights"]["queue"] - 0.1)

    def _apply_game_bias(self, telemetry):
        game, profile = self.game_profile_manager.detect_active_game()
        if game != self.active_game or profile != self.active_game_profile:
            self.active_game = game
            self.active_game_profile = profile
            name = profile["name"] if profile else "None"
            self.game_state_changed.emit(game or "None", name)
            if game:
                log_message("policy", f"Active game detected: {game} (profile={name})")

        if not profile:
            return None, False, False

        preferred_mode = profile.get("preferred_mode", "defensive")
        keep_igpu_free = profile.get("keep_igpu_free", True)
        hard_lock_defensive = profile.get("hard_lock_defensive", False)

        if hard_lock_defensive:
            eff_mode = "defensive"
            reason = f"Game profile '{profile['name']}' hard-locks defensive mode."
            return (eff_mode, reason, keep_igpu_free), True, keep_igpu_free

        eff_mode = preferred_mode
        reason = f"Game profile '{profile['name']}' prefers {preferred_mode}."
        return (eff_mode, reason, keep_igpu_free), False, keep_igpu_free

    def update_from_telemetry(self, telemetry):
        self.history.append(telemetry)
        self._tune_policy_if_needed()

        global_mode_info = self._read_global_mode()
        g_mode = global_mode_info["mode"]
        g_reason = global_mode_info["reason"]
        g_override = global_mode_info["override"]

        game_bias, game_hard_lock, keep_igpu_free = self._apply_game_bias(telemetry)

        if g_override:
            eff_mode = g_mode
            reason = f"Global override: {g_reason}"
            self.effective_mode = eff_mode
            self.mode_reason = reason
            self.mode_changed.emit(self.effective_mode, self.mode_reason)
        else:
            self._apply_global_bias(g_mode)
            if game_bias:
                eff_mode, game_reason, _ = game_bias
                reason = f"{game_reason} | global bias={g_mode}"
                self.effective_mode = eff_mode
                self.mode_reason = reason
                self.mode_changed.emit(self.effective_mode, self.mode_reason)
            else:
                if self.mode == "auto":
                    eff_mode, reason = self._auto_select_mode(telemetry)
                    reason = f"{reason} | global bias={g_mode}"
                    if eff_mode != self.effective_mode or reason != self.mode_reason:
                        self.effective_mode = eff_mode
                        self.mode_reason = reason
                        self.mode_changed.emit(self.effective_mode, self.mode_reason)
                else:
                    eff_mode = self.effective_mode

        self.consciousness_updated.emit(self.consciousness.to_dict())

        d = telemetry["dGPU"]
        i = telemetry["iGPU"]

        if len(self.history) >= 5:
            recent = list(self.history)[-5:]
            avg_igpu = sum(t["iGPU"]["load"] for t in recent) / len(recent)
        else:
            avg_igpu = i["load"]

        temp_limit = self.policy["temp_limits"][self.effective_mode]
        load_limit = self.policy["load_limits"][self.effective_mode]

        rising = i["load"] > avg_igpu and i["load"] > load_limit
        mpu_enabled = not rising and i["temp"] < temp_limit

        if keep_igpu_free:
            mpu_enabled = False

        enc_action, comp_action = self._choose_local_actions(telemetry)

        if not mpu_enabled:
            encoder_target = "dGPU"
            compute_target = "CPU"
        else:
            encoder_target = enc_action
            compute_target = comp_action

        new_assignment = {
            "encoder": encoder_target,
            "compute": compute_target,
            "mpu_enabled": mpu_enabled
        }

        if new_assignment != self.current_assignment:
            self.current_assignment = new_assignment
            self.assignment_changed.emit(new_assignment)


# -------------------------------
# Workers
# -------------------------------
ENCODE_DIR = DIRS["encode"]
HASH_DIR = DIRS["hash"]

def infer_priority(path):
    name = os.path.basename(path).lower()
    if "prio3" in name or "high" in name:
        return 3
    if "prio2" in name or "med" in name:
        return 2
    return 1

class EncoderWorker(QThread):
    status = pyqtSignal(str)
    queue_updated = pyqtSignal(int)
    outcome = pyqtSignal(str, str, bool, dict, float)
    degraded_state = pyqtSignal(bool)

    def __init__(self, mpu_manager, parent=None):
        super().__init__(parent)
        self.mpu_manager = mpu_manager
        self.target = "MPU"
        self._running = True
        self.failure_count = 0
        self.degraded = False

    def set_target(self, target):
        self.target = target
        self.status.emit(f"Encoder target → {target}")

    def stop(self):
        self._running = False

    def _find_encode_jobs(self):
        patterns = ["*.mkv", "*.mp4", "*.mov", "*.avi"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(ENCODE_DIR, p)))
        files = [(f, infer_priority(f)) for f in files]
        files.sort(key=lambda x: -x[1])
        return [f for f, pr in files]

    def run(self):
        while self._running:
            jobs = self._find_encode_jobs()
            self.queue_updated.emit(len(jobs))

            if not jobs:
                self.status.emit("No encode jobs found in jobs/encode")
                self.msleep(3000)
                continue

            if self.degraded:
                self.status.emit("Encoder in degraded mode, backing off")
                self.degraded_state.emit(True)
                self.msleep(15000)
                continue
            else:
                self.degraded_state.emit(False)

            input_file = jobs[0]
            base, ext = os.path.splitext(os.path.basename(input_file))
            output_file = os.path.join(ENCODE_DIR, f"{base}_qsv{ext}")

            job = {
                "type": "encode",
                "payload": {
                    "input": input_file,
                    "output": output_file,
                    "bitrate": "6000k",
                    "codec": "h264_qsv"
                }
            }

            context = {"job_type": "encode", "input": input_file}
            start = time.time()

            if self.target == "MPU":
                self.status.emit(f"Golden Star MPU: encoding {os.path.basename(input_file)}")
                try:
                    self.mpu_manager.submit_job(job)
                    self.failure_count = 0
                    self.degraded = False
                    duration = time.time() - start
                    self.outcome.emit("encoder", "MPU", True, context, duration)
                except Exception as e:
                    self.failure_count += 1
                    self.status.emit(f"Encode error: {e}")
                    duration = time.time() - start
                    self.outcome.emit("encoder", "MPU", False, context, duration)
                    if self.failure_count >= 3:
                        self.degraded = True
            else:
                self.status.emit("TODO: NVENC path on dGPU")
                duration = time.time() - start
                self.outcome.emit("encoder", "dGPU", True, context, duration)

            self.msleep(2000)


class ComputeWorker(QThread):
    status = pyqtSignal(str)
    queue_updated = pyqtSignal(int)
    outcome = pyqtSignal(str, str, bool, dict, float)
    degraded_state = pyqtSignal(bool)

    def __init__(self, mpu_manager, parent=None):
        super().__init__(parent)
        self.mpu_manager = mpu_manager
        self.target = "MPU"
        self._running = True
        self.failure_count = 0
        self.degraded = False

    def set_target(self, target):
        self.target = target
        self.status.emit(f"Compute target → {target}")

    def stop(self):
        self._running = False

    def _find_hash_jobs(self):
        files = glob.glob(os.path.join(HASH_DIR, "*"))
        files = [(f, infer_priority(f)) for f in files if os.path.isfile(f)]
        files.sort(key=lambda x: -x[1])
        return [f for f, pr in files]

    def run(self):
        while self._running:
            jobs = self._find_hash_jobs()
            self.queue_updated.emit(len(jobs))

            if not jobs:
                self.status.emit("No hash jobs found in jobs/hash")
                self.msleep(4000)
                continue

            if self.degraded:
                self.status.emit("Compute in degraded mode, backing off")
                self.degraded_state.emit(True)
                self.msleep(15000)
                continue
            else:
                self.degraded_state.emit(False)

            path = jobs[0]
            job = {
                "type": "hash",
                "payload": {"path": path}
            }

            context = {"job_type": "hash", "path": path}
            start = time.time()

            if self.target == "MPU":
                self.status.emit(f"Golden Star MPU: hashing {os.path.basename(path)}")
                try:
                    self.mpu_manager.submit_job(job)
                    self.failure_count = 0
                    self.degraded = False
                    duration = time.time() - start
                    self.outcome.emit("compute", "MPU", True, context, duration)
                except Exception as e:
                    self.failure_count += 1
                    self.status.emit(f"Hash error: {e}")
                    duration = time.time() - start
                    self.outcome.emit("compute", "MPU", False, context, duration)
                    if self.failure_count >= 3:
                        self.degraded = True
            else:
                self.status.emit("CPU: hashing job dispatched")
                try:
                    self.mpu_manager._run_hash_cpu(job["payload"])
                    duration = time.time() - start
                    self.outcome.emit("compute", "CPU", True, context, duration)
                except Exception as e:
                    self.status.emit(f"CPU hash error: {e}")
                    duration = time.time() - start
                    self.outcome.emit("compute", "CPU", False, context, duration)

            self.msleep(3000)


# -------------------------------
# Remote job handler
# -------------------------------
def handle_remote_job(job, mpu, swarm_jobs):
    try:
        job = stage_input_file_for_swarm(job)
        if job["type"] == "encode":
            payload = job["payload"]
            mpu._run_encode_qsv(payload)
            result = {"status": "ok", "output": payload["output"]}
        elif job["type"] == "hash":
            payload = job["payload"]
            result = {"status": "ok", "hash": mpu._run_hash(payload)}
        else:
            result = {"status": "error", "msg": "unknown job type"}
    except Exception as e:
        result = {"status": "error", "msg": str(e)}
    swarm_jobs.complete_job(job, result)


# -------------------------------
# Raft command applier
# -------------------------------
def apply_raft_command(entry):
    cmd = entry.get("command", {})
    if cmd.get("type") == "set_global_mode":
        data = {
            "mode": cmd["mode"],
            "reason": cmd["reason"],
            "override": cmd["override"],
            "timestamp": cmd["timestamp"]
        }
        save_json(COLLECTIVE_MODE_FILE, data)
        log_message("raft", f"Applied global mode: {data}")


# -------------------------------
# GUI
# -------------------------------
class BorgControlTower(QWidget):
    def __init__(self, telemetry_engine, scheduler, encoder_worker, compute_worker,
                 mpu_manager, swarm_node, swarm_jobs, swarm_consciousness, raft_node, tray_icon):
        super().__init__()
        self.telemetry_engine = telemetry_engine
        self.scheduler = scheduler
        self.encoder_worker = encoder_worker
        self.compute_worker = compute_worker
        self.mpu_manager = mpu_manager
        self.swarm_node = swarm_node
        self.swarm_jobs = swarm_jobs
        self.swarm_consciousness = swarm_consciousness
        self.raft_node = raft_node
        self.tray_icon = tray_icon

        self.setWindowTitle("Borg Collective Control Tower – Golden Star Event Horizon (Raft + Game Profiles)")
        self.setMinimumWidth(820)
        self.setStyleSheet("""
            QWidget {
                background-color: #020503;
                color: #9eff9e;
                font-family: Consolas, monospace;
            }
            QLabel {
                color: #9eff9e;
            }
            QPushButton {
                background-color: #003300;
                color: #9eff9e;
                border: 1px solid #00ff00;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #005500;
            }
        """)

        layout = QVBoxLayout()

        self.status_label = QLabel("Collective status: Online")
        layout.addWidget(self.status_label)

        self.backplane_label = QLabel(f"Backplane: {BASE_DIR}")
        layout.addWidget(self.backplane_label)

        self.node_label = QLabel(f"Node ID: {scheduler.node_id}")
        layout.addWidget(self.node_label)

        self.leader_label = QLabel("Leader: unknown")
        layout.addWidget(self.leader_label)

        self.swarm_c_label = QLabel("Swarm Consciousness: arousal=0.00, stability=1.00, pressure=0.0, nodes=0")
        layout.addWidget(self.swarm_c_label)

        self.global_mode_label = QLabel("Global Mode: baseline (override=False)")
        layout.addWidget(self.global_mode_label)

        self.game_label = QLabel("Game Mode: None (no active profile)")
        layout.addWidget(self.game_label)

        self.gpu_label = QLabel("Subsystem Telemetry:")
        layout.addWidget(self.gpu_label)

        self.igpu_util_label = QLabel("iGPU Utilization: 0%")
        layout.addWidget(self.igpu_util_label)

        self.igpu_bar = QProgressBar()
        self.igpu_bar.setRange(0, 100)
        self.igpu_bar.setTextVisible(True)
        self.igpu_bar.setStyleSheet("""
            QProgressBar {
                background-color: #001900;
                border: 1px solid #00ff00;
                color: #9eff9e;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
            }
        """)
        layout.addWidget(self.igpu_bar)

        self.assignment_label = QLabel("Resource Routing: pending...")
        layout.addWidget(self.assignment_label)

        self.mpu_label = QLabel("Golden Star MPU: idle")
        layout.addWidget(self.mpu_label)

        self.queue_label = QLabel("Assimilation Queue: encode 0 | hash 0")
        layout.addWidget(self.queue_label)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_label = QLabel("auto / baseline")
        mode_row.addWidget(self.mode_label)

        self.reason_label = QLabel("Reason: Initial state")
        layout.addWidget(self.reason_label)

        self.btn_auto = QPushButton("Auto")
        self.btn_baseline = QPushButton("Baseline")
        self.btn_hyper = QPushButton("Hyperfocus")
        self.btn_def = QPushButton("Defensive")
        self.btn_exp = QPushButton("Exploratory")

        mode_row.addWidget(self.btn_auto)
        mode_row.addWidget(self.btn_baseline)
        mode_row.addWidget(self.btn_hyper)
        mode_row.addWidget(self.btn_def)
        mode_row.addWidget(self.btn_exp)
        layout.addLayout(mode_row)

        self.btn_auto.clicked.connect(lambda: self.set_mode("auto"))
        self.btn_baseline.clicked.connect(lambda: self.set_mode("baseline"))
        self.btn_hyper.clicked.connect(lambda: self.set_mode("hyperfocus"))
        self.btn_def.clicked.connect(lambda: self.set_mode("defensive"))
        self.btn_exp.clicked.connect(lambda: self.set_mode("exploratory"))

        self.conscious_label = QLabel("Consciousness: arousal=0.0, confidence=0.0, curiosity=0.0, stability=0.0")
        layout.addWidget(self.conscious_label)

        self.swarm_label = QLabel("Swarm: 1 node detected (local)")
        layout.addWidget(self.swarm_label)

        self.swarm_jobs_label = QLabel("Swarm Jobs: 0 pending/claimed/done")
        layout.addWidget(self.swarm_jobs_label)

        self.diag_label = QLabel("Diagnostics:")
        layout.addWidget(self.diag_label)

        self.diag_text = QLabel(self._format_diagnostics())
        layout.addWidget(self.diag_text)

        self.diag_button = QPushButton("Refresh Diagnostics")
        self.diag_button.clicked.connect(self.refresh_diagnostics)
        layout.addWidget(self.diag_button)

        self.log_label = QLabel("Event Log:")
        layout.addWidget(self.log_label)

        self.behavior_label = QLabel("Behavior Regime: (emergent patterns not yet clustered)")
        layout.addWidget(self.behavior_label)

        self.setLayout(layout)

        telemetry_engine.telemetry_updated.connect(self.on_telemetry)
        scheduler.assignment_changed.connect(self.on_assignment)
        scheduler.mode_changed.connect(self.on_mode_changed)
        scheduler.consciousness_updated.connect(self.on_consciousness)
        scheduler.game_state_changed.connect(self.on_game_state)
        encoder_worker.status.connect(self.append_log)
        compute_worker.status.connect(self.append_log)
        encoder_worker.queue_updated.connect(self.on_encode_queue)
        compute_worker.queue_updated.connect(self.on_hash_queue)
        encoder_worker.degraded_state.connect(self.on_degraded_state)
        compute_worker.degraded_state.connect(self.on_degraded_state)
        mpu_manager.job_started.connect(self.on_mpu_job_started)
        mpu_manager.job_finished.connect(self.on_mpu_job_finished)
        mpu_manager.job_failed.connect(self.on_mpu_job_failed)
        swarm_node.swarm_updated.connect(self.on_swarm_updated)
        swarm_jobs.job_completed.connect(self.on_swarm_job_completed)
        swarm_consciousness.swarm_consciousness_updated.connect(self.on_swarm_consciousness)
        raft_node.leader_changed.connect(self.on_leader_changed)
        raft_node.log_committed.connect(self.on_raft_log_committed)

        self.encode_queue_count = 0
        self.hash_queue_count = 0

    def set_mode(self, mode):
        self.scheduler.set_mode(mode)
        if mode == "auto":
            self.mode_label.setText(f"auto / {self.scheduler.effective_mode}")
        else:
            self.mode_label.setText(f"manual / {mode}")
        self.append_log(f"Mode set to {mode}")

    def on_mode_changed(self, effective_mode, reason):
        if self.scheduler.mode == "auto":
            self.mode_label.setText(f"auto / {effective_mode}")
        self.reason_label.setText(f"Reason: {reason}")
        self.append_log(f"Mode auto-selected: {effective_mode} ({reason})")

    def on_consciousness(self, c):
        self.conscious_label.setText(
            f"Consciousness: arousal={c['arousal']:.2f}, "
            f"confidence={c['confidence']:.2f}, "
            f"curiosity={c['curiosity']:.2f}, "
            f"stability={c['stability']:.2f}"
        )

    def on_game_state(self, game, profile_name):
        if game == "None":
            self.game_label.setText("Game Mode: None (no active profile)")
        else:
            self.game_label.setText(f"Game Mode: {profile_name} ({game})")

    def on_swarm_consciousness(self, sc):
        self.swarm_c_label.setText(
            f"Swarm Consciousness: arousal={sc['arousal']:.2f}, "
            f"stability={sc['stability']:.2f}, "
            f"pressure={sc['pressure']:.1f}, "
            f"nodes={sc['node_count']}"
        )
        gm = load_json(COLLECTIVE_MODE_FILE, {"mode": "baseline", "override": False})
        self.global_mode_label.setText(
            f"Global Mode: {gm['mode']} (override={gm.get('override', False)})"
        )

    def on_leader_changed(self, leader_id):
        self.leader_label.setText(f"Leader: {leader_id}")
        self.scheduler.set_leader(leader_id)
        self.append_log(f"Leader changed: {leader_id}")

    def on_raft_log_committed(self, entry):
        apply_raft_command(entry)
        gm = load_json(COLLECTIVE_MODE_FILE, {"mode": "baseline", "override": False})
        self.global_mode_label.setText(
            f"Global Mode: {gm['mode']} (override={gm.get('override', False)})"
        )

    def _format_diagnostics(self):
        d = self.mpu_manager.diagnostics
        lines = [
            f"PyOpenCL installed: {d['pyopencl_installed']}",
            f"OpenCL device found: {d['opencl_device_found']}",
            f"OpenCL error: {d['opencl_error']}",
            f"Quick Sync available: {d['quick_sync_available']}",
            f"Hash backend: {d['hash_backend']}",
            f"psutil available (game detection): {PSUTIL_AVAILABLE}",
        ]
        return "\n".join(lines)

    def refresh_diagnostics(self):
        self.diag_text.setText(self._format_diagnostics())
        self.append_log("Diagnostics refreshed")

    def on_encode_queue(self, count):
        self.encode_queue_count = count
        self.scheduler.update_encode_queue(count)
        self._update_queue_label()

    def on_hash_queue(self, count):
        self.hash_queue_count = count
        self.scheduler.update_hash_queue(count)
        self._update_queue_label()

    def _update_queue_label(self):
        self.queue_label.setText(
            f"Assimilation Queue: encode {self.encode_queue_count} | hash {self.hash_queue_count}"
        )

    def on_degraded_state(self, degraded):
        self.scheduler.set_degraded(degraded)

    def on_telemetry(self, t):
        d = t["dGPU"]
        i = t["iGPU"]

        self.gpu_label.setText(
            f"dGPU: {d['load']}% load, {d['vram']}MB, {d['temp']}°C, enc {d['encoder']}%\n"
            f"iGPU: {i['load']}% load, {i['vram']}MB, {i['temp']}°C, enc {i['encoder']}%"
        )

        load = i["load"]
        self.igpu_util_label.setText(f"iGPU Utilization: {load}%")
        self.igpu_bar.setValue(load)

        if load < 60:
            color = "#00ff00"
        elif load < 85:
            color = "#ffaa00"
        else:
            color = "#ff0000"

        self.igpu_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #001900;
                border: 1px solid {color};
                color: #9eff9e;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

        load_summary = {
            "igpu_load": i["load"],
            "igpu_temp": i["temp"],
            "dgpu_load": d["load"],
            "dgpu_temp": d["temp"],
            "encode_queue": self.encode_queue_count,
            "hash_queue": self.hash_queue_count
        }
        self.swarm_node.write_heartbeat(load_summary)

        jobs = glob.glob(os.path.join(DIRS["swarm_jobs"], "*.json"))
        self.swarm_jobs_label.setText(f"Swarm Jobs: {len(jobs)} pending/claimed/done")

    def on_assignment(self, a):
        self.assignment_label.setText(
            f"Encoder → {a['encoder']} | Compute → {a['compute']} | MPU enabled: {a['mpu_enabled']}"
        )
        self.append_log(f"Routing updated: {a}")

    def on_mpu_job_started(self, job):
        t = job["type"]
        if t == "encode":
            msg = "Golden Star MPU: video encode in progress"
        elif t == "hash":
            msg = "Golden Star MPU: data hashing in progress"
        else:
            msg = f"Golden Star MPU: processing {t}"
        self.mpu_label.setText(msg)

    def on_mpu_job_finished(self, job):
        t = job["type"]
        if t == "hash" and "result" in job:
            self.append_log(f"MPU hash result: {job['result']}")
        self.mpu_label.setText("Golden Star MPU: idle")

    def on_mpu_job_failed(self, job):
        self.mpu_label.setText(f"Golden Star MPU: error in {job['type']}")
        self.append_log(f"MPU error: {job.get('error')}")

    def on_swarm_updated(self, nodes):
        count = len(nodes)
        self.swarm_label.setText(f"Swarm: {count} node(s) detected")

    def on_swarm_job_completed(self, job):
        self.append_log(f"Swarm job completed: {job.get('job_id')} status={job.get('result', {}).get('status')}")

    def append_log(self, msg):
        self.log_label.setText(f"Event Log:\n{msg}")

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage(
            "Borg Collective",
            "Control Tower running in background.",
            QSystemTrayIcon.Information,
            2000
        )


# -------------------------------
# Tray
# -------------------------------
def create_tray(app, window):
    icon_path = "borg_icon.png"
    if not os.path.exists(icon_path):
        icon = QIcon()
    else:
        icon = QIcon(icon_path)

    tray = QSystemTrayIcon(icon, app)

    menu = QMenu()
    show_action = QAction("Open Collective Interface")
    quit_action = QAction("Disengage")

    show_action.triggered.connect(window.show)
    quit_action.triggered.connect(app.quit)

    menu.addAction(show_action)
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.show()
    return tray


# -------------------------------
# Main
# -------------------------------
def main():
    app = QApplication(sys.argv)

    node_id = str(uuid.uuid4())[:8]

    telemetry = TelemetryEngine()
    mpu = MPUManager()
    game_profiles = GameProfileManager()
    scheduler = TaskScheduler(node_id=node_id, game_profile_manager=game_profiles)
    encoder = EncoderWorker(mpu)
    compute = ComputeWorker(mpu)
    swarm = SwarmNode(node_id=node_id, capabilities={"igpu": True, "dgpu": True, "cpu": True})
    swarm_jobs = SwarmJobManager(node_id=node_id)
    raft_node = RaftNode(node_id=node_id)
    swarm_consciousness = SwarmConsciousness(node_id=node_id, raft_node=raft_node)

    telemetry.telemetry_updated.connect(scheduler.update_from_telemetry)
    scheduler.assignment_changed.connect(lambda a: encoder.set_target(a["encoder"]))
    scheduler.assignment_changed.connect(lambda a: compute.set_target(a["compute"]))

    encoder.outcome.connect(lambda d, t, s, ctx, dur: scheduler.record_outcome(d, t, s, ctx, dur))
    compute.outcome.connect(lambda d, t, s, ctx, dur: scheduler.record_outcome(d, t, s, ctx, dur))

    swarm_jobs.job_received.connect(lambda job: handle_remote_job(job, mpu, swarm_jobs))

    raft_node.log_committed.connect(apply_raft_command)

    tray_dummy = QSystemTrayIcon()
    window = BorgControlTower(
        telemetry, scheduler, encoder, compute, mpu,
        swarm, swarm_jobs, swarm_consciousness, raft_node, tray_dummy
    )
    tray = create_tray(app, window)
    window.tray_icon = tray

    telemetry.start()
    encoder.start()
    compute.start()
    swarm.start()
    swarm_jobs.start()
    raft_node.start()
    swarm_consciousness.start()

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

