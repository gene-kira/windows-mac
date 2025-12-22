#!/usr/bin/env python3
"""
Hive Guardian - ASI Queen v7 (Predictive Upgrades)

Primary mission:
- System security
- System operations stability

Major features:
- Auto-loader for required libraries
- Hardware + software audits
- Queen brain + workers:
    - Process scanner
    - System metrics
    - Network activity
    - File activity (basic spike detection)
- Habit learning (per-process, time-of-day aware)
- Predictive-ish risk scoring:
    - Behavior vs habits (CPU/MEM ratios)
    - Time-of-day anomaly
    - Network activity anomaly
    - Trust tier (system/installed/user/temp)
    - Global load + file spike context
- Anomaly → action via risk score:
    - WATCH / QUARANTINE / KILL
- Strategy engine (self-tuning thresholds/intervals via simulation)
- Persistence (JSON brain file):
    - Admin password (salted hash + salt)
    - Paranoia level, strategy, profiles, network backup path
- Optional network backup of brain file
- Memory+disk-aware behavior
- Main GUI: Overview / Workers / Events / Process Map
- Decision Panel GUI (admin-only):
    - Password-gated
    - Toggle enforcement (dry run vs real)
    - Adjust paranoia level
    - Change admin password
    - Configure network backup
    - Save brain now
    - Shutdown queen

Usage:
    python hive_guardian_queen.py           # GUI mode
    python hive_guardian_queen.py --service # service mode (no GUI)
"""

import sys
import subprocess
import importlib
import threading
import time
import queue
import json
import os
import hashlib
import getpass
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from random import uniform
from collections import deque

# -------------- AUTO-LOADER FOR LIBRARIES -------------- #

REQUIRED_PACKAGES = [
    "psutil",
    "matplotlib",
    "networkx",
]

def ensure_package(pkg_name: str):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        print(f"[AUTO-LOADER] Installing missing package: {pkg_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        print(f"[AUTO-LOADER] Installed: {pkg_name}")

def ensure_all_packages():
    for pkg in REQUIRED_PACKAGES:
        ensure_package(pkg)

ensure_all_packages()

import platform
import psutil
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

# -------------- PATHS & PERSISTENCE HELPERS -------------- #

def get_default_brain_path() -> str:
    home = os.path.expanduser("~")
    cfg_dir = os.path.join(home, ".hive_guardian")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "hive_guardian_brain.json")

# -------------- SIMPLE PASSWORD HASHING -------------- #

def hash_password(password: str, salt: str) -> str:
    h = hashlib.sha256()
    h.update((salt + password).encode("utf-8"))
    return h.hexdigest()

# -------------- DATA MODELS -------------- #

@dataclass
class NodeInfo:
    id: str
    label: str
    type: str
    cpu: float = 0.0
    memory: float = 0.0
    extra: Dict = field(default_factory=dict)

@dataclass
class EdgeInfo:
    src: str
    dst: str
    type: str
    extra: Dict = field(default_factory=dict)

@dataclass
class SystemSnapshot:
    timestamp: float
    nodes: Dict[str, NodeInfo]
    edges: List[EdgeInfo]

@dataclass
class WorkerStatus:
    name: str
    last_run: float
    last_result: str
    active: bool = True
    interval: float = 0.0

@dataclass
class ProcessProfile:
    name: str
    avg_cpu: float = 0.0
    avg_mem: float = 0.0
    count: int = 0
    hour_counts: Dict[int, int] = field(default_factory=lambda: {h: 0 for h in range(24)})

    def update(self, cpu: float, mem: float, timestamp: float):
        self.count += 1
        alpha = 0.1
        self.avg_cpu = (1 - alpha) * self.avg_cpu + alpha * cpu
        self.avg_mem = (1 - alpha) * self.avg_mem + alpha * mem
        hour = time.localtime(timestamp).tm_hour
        self.hour_counts[hour] = self.hour_counts.get(hour, 0) + 1

    def activity_at_hour(self, hour: int) -> int:
        return self.hour_counts.get(hour, 0)

@dataclass
class Strategy:
    name: str
    scan_interval_factor: float
    cpu_threshold: float
    mem_threshold: float
    enforcement_aggressiveness: float

@dataclass
class StrategyResult:
    strategy: Strategy
    score: float
    notes: str

# -------------- QUEEN BRAIN -------------- #

class QueenBrain:
    def __init__(self, brain_path: Optional[str] = None):
        self.lock = threading.Lock()
        self.snapshots: List[SystemSnapshot] = []
        self.current_graph = nx.DiGraph()
        self.worker_status: Dict[str, WorkerStatus] = {}
        self.event_log: List[str] = []
        self.telemetry_queue: "queue.Queue[SystemSnapshot]" = queue.Queue()
        self.running = True

        self.brain_path = brain_path or get_default_brain_path()
        self.network_backup_path: Optional[str] = None

        self.enforce_actions_for_real = False

        vm = psutil.virtual_memory()
        self.total_mem_bytes = vm.total
        self.total_mem_gb = self.total_mem_bytes / (1024**3)

        self.mem_cap_fraction = 0.50
        self.mem_threshold = 50.0
        self.cpu_threshold = 80.0
        self.disk_high_threshold = 90.0

        self.low_memory_mode = False
        self.low_disk_mode = False

        if self.total_mem_gb <= 4:
            self.max_snapshots_normal = 20
        elif self.total_mem_gb <= 8:
            self.max_snapshots_normal = 40
        else:
            self.max_snapshots_normal = 60
        self.max_snapshots_low_mem = max(10, self.max_snapshots_normal // 2)

        self.last_global_cpu = 0.0
        self.last_global_mem = 0.0
        self.last_global_disk = 0.0

        # Short-term history for simple trend forecasting
        self.global_cpu_history = deque(maxlen=20)   # last N readings
        self.global_mem_history = deque(maxlen=20)
        self.global_disk_history = deque(maxlen=20)

        self.process_profiles: Dict[str, ProcessProfile] = {}

        self.current_strategy = Strategy(
            name="default",
            scan_interval_factor=1.0,
            cpu_threshold=self.cpu_threshold,
            mem_threshold=self.mem_threshold,
            enforcement_aggressiveness=0.3,
        )

        self.critical_process_whitelist = set([
            "System",
            "systemd",
            "init",
            "csrss.exe",
            "wininit.exe",
            "winlogon.exe",
        ])

        self.paranoia_level = 0.3
        self.security_weight = 2.0
        self.ops_weight = 1.0

        self.process_network_activity: Dict[str, Dict[str, int]] = {}
        self.listening_ports: List[Tuple[str, int]] = []
        self.file_spike_events: List[Tuple[str, str, int, int]] = []

        self.admin_salt: Optional[str] = None
        self.admin_password_hash: Optional[str] = None

        # For self-correction logging
        self.recent_actions: List[Dict] = []

        self._load_brain()

        self._run_hardware_audit()
        self._run_software_audit()

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self.persistence_thread = threading.Thread(target=self._persistence_loop, daemon=True)
        self.persistence_thread.start()

    # -------- Persistence -------- #

    def _load_brain(self):
        if not os.path.isfile(self.brain_path):
            print(f"[PERSIST] No brain file at {self.brain_path}. Initializing new brain.")
            self._initialize_admin_credentials()
            return

        try:
            with open(self.brain_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[PERSIST] Failed to load brain file: {e}")
            self._initialize_admin_credentials()
            return

        self.admin_salt = data.get("admin_salt")
        self.admin_password_hash = data.get("admin_password_hash")
        self.network_backup_path = data.get("network_backup_path")

        if not self.admin_salt or not self.admin_password_hash:
            self._initialize_admin_credentials()
        else:
            print("[PERSIST] Loaded admin credentials.")

        self.paranoia_level = data.get("paranoia_level", self.paranoia_level)
        self.security_weight = data.get("security_weight", self.security_weight)
        self.ops_weight = data.get("ops_weight", self.ops_weight)

        strat = data.get("strategy")
        if strat:
            self.current_strategy = Strategy(
                name=strat.get("name", "loaded"),
                scan_interval_factor=strat.get("scan_interval_factor", 1.0),
                cpu_threshold=strat.get("cpu_threshold", self.cpu_threshold),
                mem_threshold=strat.get("mem_threshold", self.mem_threshold),
                enforcement_aggressiveness=strat.get("enforcement_aggressiveness", 0.3),
            )

        profiles_data = data.get("process_profiles", {})
        for name, p in profiles_data.items():
            prof = ProcessProfile(name=name)
            prof.avg_cpu = p.get("avg_cpu", 0.0)
            prof.avg_mem = p.get("avg_mem", 0.0)
            prof.count = p.get("count", 0)
            prof.hour_counts = {int(k): int(v) for k, v in p.get("hour_counts", {}).items()}
            self.process_profiles[name] = prof

        print("[PERSIST] Brain data loaded.")

    def _initialize_admin_credentials(self):
        print("=== Hive Guardian Initial Setup ===")
        print("No admin password found. Please set one now.")
        while True:
            pw1 = getpass.getpass("Admin password: ")
            pw2 = getpass.getpass("Confirm password: ")
            if pw1 and pw1 == pw2:
                break
            print("Passwords do not match or empty. Try again.")

        salt = os.urandom(16).hex()
        pw_hash = hash_password(pw1, salt)

        self.admin_salt = salt
        self.admin_password_hash = pw_hash

        print("Admin password set. This will be stored in the brain file.")
        choice = input("Do you want to configure a network backup path now? (y/N): ").strip().lower()
        if choice == "y":
            p = input("Enter network backup path (e.g., \\\\server\\share\\hive_guardian_brain.json): ").strip()
            if p:
                self.network_backup_path = p
                print(f"Network backup path set to: {p}")
        else:
            print("No network backup configured now. You can change it later from the GUI.")

        self._save_brain()

    def verify_admin_password(self, password: str) -> bool:
        if not self.admin_salt or not self.admin_password_hash:
            return False
        return hash_password(password, self.admin_salt) == self.admin_password_hash

    def change_admin_password(self, old_password: str, new_password: str) -> bool:
        if not self.verify_admin_password(old_password):
            return False
        new_salt = os.urandom(16).hex()
        new_hash = hash_password(new_password, new_salt)
        self.admin_salt = new_salt
        self.admin_password_hash = new_hash
        self._save_brain()
        return True

    def set_network_backup_path(self, path: Optional[str]):
        self.network_backup_path = path or None
        self._save_brain()
        self.log(f"[ADMIN] Network backup path updated to: {self.network_backup_path}")

    def _brain_to_dict(self) -> dict:
        data = {
            "admin_salt": self.admin_salt,
            "admin_password_hash": self.admin_password_hash,
            "network_backup_path": self.network_backup_path,
            "paranoia_level": self.paranoia_level,
            "security_weight": self.security_weight,
            "ops_weight": self.ops_weight,
            "strategy": {
                "name": self.current_strategy.name,
                "scan_interval_factor": self.current_strategy.scan_interval_factor,
                "cpu_threshold": self.current_strategy.cpu_threshold,
                "mem_threshold": self.current_strategy.mem_threshold,
                "enforcement_aggressiveness": self.current_strategy.enforcement_aggressiveness,
            },
            "process_profiles": {},
        }
        for name, p in self.process_profiles.items():
            data["process_profiles"][name] = {
                "avg_cpu": p.avg_cpu,
                "avg_mem": p.avg_mem,
                "count": p.count,
                "hour_counts": {str(h): c for h, c in p.hour_counts.items()},
            }
        return data

    def _save_brain(self):
        data = self._brain_to_dict()
        try:
            with open(self.brain_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[PERSIST] Brain saved to {self.brain_path}")
        except Exception as e:
            print(f"[PERSIST] Failed to save brain: {e}")

        if self.network_backup_path:
            try:
                os.makedirs(os.path.dirname(self.network_backup_path), exist_ok=True)
                with open(self.network_backup_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"[PERSIST] Brain mirrored to network: {self.network_backup_path}")
            except Exception as e:
                print(f"[PERSIST] Failed to save network backup: {e}")

    def _persistence_loop(self):
        while self.running:
            time.sleep(60)
            try:
                self._save_brain()
            except Exception as e:
                print(f"[PERSIST] Error during periodic save: {e}")

    # -------- Logging -------- #

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self.lock:
            self.event_log.append(line)
            if len(self.event_log) > 2000:
                self.event_log.pop(0)
        print(line)

    # -------- Security signals -------- #

    def update_network_signals(self, proc_conn_counts: Dict[str, Dict[str, int]], listening_ports: List[tuple]):
        with self.lock:
            self.process_network_activity = proc_conn_counts
            self.listening_ports = listening_ports

    def update_file_signals(self, events: List[tuple]):
        with self.lock:
            ts = time.strftime("%H:%M:%S")
            for (root, created, deleted) in events:
                self.file_spike_events.append((ts, root, created, deleted))
                self.log(f"[FILE-SEC] Spike in {root}: created={created}, deleted={deleted}")
            self.file_spike_events = self.file_spike_events[-50:]

    # -------- Audits -------- #

    def _run_hardware_audit(self):
        try:
            uname = platform.uname()
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_count_phys = psutil.cpu_count(logical=False) or cpu_count_logical
            vm = psutil.virtual_memory()
            total_ram_gb = vm.total / (1024**3)

            disks_info = []
            for part in psutil.disk_partitions(all=False):
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks_info.append(
                        f"{part.device} ({part.mountpoint}) "
                        f"{usage.total/(1024**3):.1f}GB used={usage.percent:.0f}%"
                    )
                except Exception:
                    pass

            self.log("=== HARDWARE AUDIT START ===")
            self.log(f"System: {uname.system} {uname.release} ({uname.version})")
            self.log(f"Node: {uname.node}")
            self.log(f"Machine: {uname.machine}, Processor: {uname.processor or 'unknown'}")
            self.log(f"CPU cores: physical={cpu_count_phys}, logical={cpu_count_logical}")
            self.log(f"RAM: {total_ram_gb:.2f} GB total")
            if disks_info:
                for d in disks_info:
                    self.log(f"Disk: {d}")
            else:
                self.log("Disk: (no partitions or unable to read usage)")
            self.log("=== HARDWARE AUDIT END ===")
        except Exception as e:
            self.log(f"[ERROR] Hardware audit failed: {e}")

    def _run_software_audit(self):
        try:
            self.log("=== SOFTWARE AUDIT START ===")
            self.log(f"Python: {platform.python_version()} ({sys.executable})")
            self.log(f"Platform: {platform.platform()}")
            self.log("Enumerating running processes (name, pid)...")

            count = 0
            for proc in psutil.process_iter(attrs=["pid", "name"]):
                info = proc.info
                self.log(f"Process: {info.get('name', 'proc')} (pid={info.get('pid')})")
                count += 1

            self.log(f"Total running processes seen during audit: {count}")
            self.log("=== SOFTWARE AUDIT END ===")
        except Exception as e:
            self.log(f"[ERROR] Software audit failed: {e}")

    # -------- Worker management -------- #

    def register_worker(self, name: str, interval: float):
        with self.lock:
            self.worker_status[name] = WorkerStatus(
                name=name,
                last_run=0.0,
                last_result="Pending",
                active=True,
                interval=interval,
            )

    def update_worker_status(self, name: str, result: str, interval: Optional[float] = None):
        with self.lock:
            if name in self.worker_status:
                self.worker_status[name].last_run = time.time()
                self.worker_status[name].last_result = result
                if interval is not None:
                    self.worker_status[name].interval = interval

    # -------- Telemetry -------- #

    def submit_snapshot(self, snapshot: SystemSnapshot):
        self.telemetry_queue.put(snapshot)

    def update_global_metrics(self, cpu: float, mem: float, disk: float):
        with self.lock:
            self.last_global_cpu = cpu
            self.last_global_mem = mem
            self.last_global_disk = disk

            self.global_cpu_history.append(cpu)
            self.global_mem_history.append(mem)
            self.global_disk_history.append(disk)

            self.low_memory_mode = mem > (self.mem_cap_fraction * 100.0)
            self.low_disk_mode = disk > self.disk_high_threshold

    # Simple trend estimate: last - first
    def _estimate_trend(self, values: deque) -> float:
        if len(values) < 3:
            return 0.0
        return values[-1] - values[0]

    # -------- Habits -------- #

    def _update_habit_profiles_from_snapshot(self, snapshot: SystemSnapshot):
        for node in snapshot.nodes.values():
            name = node.extra.get("name", None)
            if not name:
                label = node.label
                if "(" in label:
                    name = label.split("(", 1)[0].strip()
                else:
                    name = label

            profile = self.process_profiles.get(name)
            if profile is None:
                profile = ProcessProfile(name=name)
                self.process_profiles[name] = profile
            profile.update(node.cpu, node.memory, snapshot.timestamp)

    # -------- Strategy evolution -------- #

    def _simulate_strategy(self, strategy: Strategy) -> StrategyResult:
        with self.lock:
            snapshots = list(self.snapshots)

        if not snapshots:
            return StrategyResult(strategy, score=0.0, notes="No data yet")

        cpu_thr = strategy.cpu_threshold
        mem_thr = strategy.mem_threshold

        anomaly_count = 0
        heavy_anomalies = 0
        scan_cost = len(snapshots) / max(strategy.scan_interval_factor, 0.1)

        for snap in snapshots:
            for node in snap.nodes.values():
                if node.cpu > cpu_thr or node.memory > mem_thr:
                    anomaly_count += 1
                    if node.cpu > cpu_thr * 2 or node.memory > mem_thr * 2:
                        heavy_anomalies += 1

        penalty = (
            self.security_weight * heavy_anomalies * 3.0 +
            self.security_weight * anomaly_count * 1.0 +
            self.ops_weight * scan_cost * 0.1
        )
        score = -penalty
        notes = (
            f"anomalies={anomaly_count}, heavy={heavy_anomalies}, "
            f"scan_cost≈{scan_cost:.1f}, score={score:.1f}"
        )
        return StrategyResult(strategy=strategy, score=score, notes=notes)

    def _generate_candidate_strategies(self, n: int = 4) -> List[Strategy]:
        base = self.current_strategy
        candidates: List[Strategy] = []

        for i in range(n):
            s = Strategy(
                name=f"auto_{int(time.time())}_{i}",
                scan_interval_factor=max(0.5, min(2.5, base.scan_interval_factor * uniform(0.7, 1.3))),
                cpu_threshold=max(40.0, min(95.0, base.cpu_threshold * uniform(0.8, 1.2))),
                mem_threshold=max(30.0, min(90.0, base.mem_threshold * uniform(0.8, 1.2))),
                enforcement_aggressiveness=max(
                    0.0,
                    min(1.0, base.enforcement_aggressiveness + uniform(-0.1, 0.1))
                ),
            )
            candidates.append(s)
        return candidates

    def evolve_strategy_if_needed(self):
        with self.lock:
            if len(self.snapshots) < 10:
                return

        candidates = self._generate_candidate_strategies(n=5)
        best_result = self._simulate_strategy(self.current_strategy)

        for s in candidates:
            r = self._simulate_strategy(s)
            if r.score > best_result.score:
                best_result = r

        if best_result.strategy.name != self.current_strategy.name:
            self.log(
                f"[STRATEGY] Adopting new strategy {best_result.strategy.name}: "
                f"{best_result.notes}"
            )
            self.current_strategy = best_result.strategy
        else:
            self.log(
                f"[STRATEGY] Keeping current strategy {self.current_strategy.name}: "
                f"{best_result.notes}"
            )

    # -------- Risk scoring & anomaly classification -------- #

    def _compute_risk_score(self, node: NodeInfo) -> float:
        """
        Combine multiple signals into a single risk score.
        Higher = more dangerous.
        """
        name = node.extra.get("name", node.label)
        profile = self.process_profiles.get(name)
        cpu = node.cpu
        mem = node.memory

        # Base components
        cpu_ratio = 1.0
        mem_ratio = 1.0
        if profile and profile.avg_cpu > 0:
            cpu_ratio = (cpu + 1.0) / (profile.avg_cpu + 1.0)
        if profile and profile.avg_mem > 0:
            mem_ratio = (mem + 1.0) / (profile.avg_mem + 1.0)

        # Time-of-day anomaly
        now_hour = time.localtime().tm_hour
        time_anomaly = 0.0
        if profile and profile.count > 30:
            hour_activity = profile.activity_at_hour(now_hour)
            if hour_activity < (profile.count * 0.02):
                time_anomaly = 1.0  # rare at this hour

        # Network anomaly
        with self.lock:
            net_info = self.process_network_activity.get(node.id, {"outgoing": 0, "listening": 0})
        outgoing = net_info.get("outgoing", 0)
        listening = net_info.get("listening", 0)

        net_anomaly = 0.0
        if outgoing > 20:
            net_anomaly += 1.0
        if listening > 0:
            net_anomaly += 1.0

        # Trust tier
        trust = node.extra.get("trust", "unknown")
        trust_factor = 1.0
        if trust == "system":
            trust_factor = 0.3
        elif trust == "installed":
            trust_factor = 0.6
        elif trust == "user":
            trust_factor = 1.0
        elif trust == "temp":
            trust_factor = 1.5
        else:  # unknown
            trust_factor = 1.3

        # Global stress
        cpu_trend = self._estimate_trend(self.global_cpu_history)
        mem_trend = self._estimate_trend(self.global_mem_history)
        disk_trend = self._estimate_trend(self.global_disk_history)

        global_stress = 0.0
        if self.last_global_cpu > 80 or cpu_trend > 10:
            global_stress += 0.5
        if self.last_global_mem > 70 or mem_trend > 10:
            global_stress += 0.5
        if self.last_global_disk > 90 or disk_trend > 10:
            global_stress += 0.3

        # File spikes recently?
        recent_file_spikes = 1.0 if self.file_spike_events else 0.0

        # Combine components
        # Start from CPU/MEM deviation
        behavior_component = max(cpu_ratio, mem_ratio) - 1.0  # 0 means "like average"
        behavior_component = max(0.0, behavior_component)

        risk = 0.0
        risk += behavior_component * 1.2
        risk += time_anomaly * 0.8
        risk += net_anomaly * 1.0
        risk += recent_file_spikes * 0.5
        risk += global_stress * 0.7

        # Adjust by trust tier
        risk *= trust_factor

        # Adjust by paranoia and strategy aggressiveness
        risk *= (0.5 + self.paranoia_level)   # between 0.5 and 1.5
        risk *= (0.5 + self.current_strategy.enforcement_aggressiveness)  # between 0.5 and 1.5

        # Clamp to sane range
        risk = max(0.0, min(risk, 20.0))
        return risk

    def _risk_to_action(self, risk: float, name: str) -> str:
        """
        Map risk score to action.
        Risk ranges (tunable):
        - 0..3   -> WATCH
        - 3..7   -> QUARANTINE
        - 7+     -> KILL (if allowed)
        """
        is_critical = name in self.critical_process_whitelist

        if risk < 3.0:
            action = "WATCH"
        elif risk < 7.0:
            action = "QUARANTINE"
        else:
            action = "KILL"

        # Safety rails
        if self.paranoia_level < 0.2 and action == "KILL":
            action = "QUARANTINE"
        if self.paranoia_level < 0.05:
            action = "WATCH"

        if self.current_strategy.enforcement_aggressiveness < 0.2 and action == "KILL":
            action = "QUARANTINE"
        if self.current_strategy.enforcement_aggressiveness < 0.05:
            action = "WATCH"

        if is_critical and action == "KILL":
            action = "QUARANTINE"

        return action

    def _record_action(self, node: NodeInfo, risk: float, action: str):
        """
        Self-correction hook: we log key info that could later be used
        to train/evaluate strategy quality.
        """
        name = node.extra.get("name", node.label)
        trust = node.extra.get("trust", "unknown")
        now = time.time()
        entry = {
            "time": now,
            "name": name,
            "pid": node.id,
            "risk": risk,
            "action": action,
            "cpu": node.cpu,
            "mem": node.memory,
            "trust": trust,
        }
        with self.lock:
            self.recent_actions.append(entry)
            self.recent_actions = self.recent_actions[-200:]  # keep last 200
        # In future you could analyze this to adjust thresholds.

    # -------- Main loop -------- #

    def _run_loop(self):
        self.log(
            f"Queen brain started. Total RAM ~ {self.total_mem_gb:.1f} GB. "
            f"Soft cap at {self.mem_cap_fraction*100:.0f}% usage."
        )
        last_evolve = time.time()
        evolve_interval = 60.0

        while self.running:
            try:
                snapshot: SystemSnapshot = self.telemetry_queue.get(timeout=1.0)
                self._ingest_snapshot(snapshot)
            except queue.Empty:
                pass

            now = time.time()
            if now - last_evolve > evolve_interval:
                self.evolve_strategy_if_needed()
                last_evolve = now

            time.sleep(0.1)

    def _ingest_snapshot(self, snapshot: SystemSnapshot):
        self._update_habit_profiles_from_snapshot(snapshot)

        with self.lock:
            max_snapshots = self.max_snapshots_low_mem if self.low_memory_mode else self.max_snapshots_normal
            self.snapshots.append(snapshot)
            while len(self.snapshots) > max_snapshots:
                self.snapshots.pop(0)

            graph = nx.DiGraph()
            nodes_list = list(snapshot.nodes.values())

            if self.low_memory_mode:
                nodes_list = nodes_list[: min(len(nodes_list), 200)]
            else:
                nodes_list = nodes_list[: min(len(nodes_list), 1000)]

            used_ids = set()

            for node in nodes_list:
                graph.add_node(
                    node.id,
                    label=node.label,
                    type=node.type,
                    cpu=node.cpu,
                    memory=node.memory,
                )
                used_ids.add(node.id)

            for edge in snapshot.edges:
                if edge.src in used_ids and edge.dst in used_ids:
                    graph.add_edge(edge.src, edge.dst, type=edge.type)

            self.current_graph = graph

        # Use strategy thresholds as "interestingness" gate
        cpu_thr = self.current_strategy.cpu_threshold
        mem_thr = self.current_strategy.mem_threshold

        hot_nodes = [
            node for node in snapshot.nodes.values()
            if node.cpu > cpu_thr or node.memory > mem_thr
        ]

        for node in hot_nodes:
            name = node.extra.get("name", node.label)
            risk = self._compute_risk_score(node)
            action = self._risk_to_action(risk, name)
            msg = (
                f"[ANOMALY] {node.label} CPU={node.cpu:.1f}% MEM={node.memory:.1f}% "
                f"RISK={risk:.2f} => ACTION={action} via strategy={self.current_strategy.name}"
            )
            self.log(msg)
            self._record_action(node, risk, action)

            if action in ("QUARANTINE", "KILL"):
                self._attempt_enforcement(node, action)

    def _attempt_enforcement(self, node: NodeInfo, action: str):
        pid_str = node.id
        try:
            pid = int(pid_str)
        except ValueError:
            self.log(f"[ACTION] Unable to parse PID from node id={pid_str}")
            return

        name = node.extra.get("name", node.label)
        if not self.enforce_actions_for_real:
            self.log(
                f"[ACTION-DRYRUN] Would {action} process pid={pid} name={name}. "
                f"(Enable real actions via Decision Panel.)"
            )
            return

        try:
            proc = psutil.Process(pid)
            if action == "KILL":
                proc.terminate()
                self.log(f"[ACTION] KILL issued to pid={pid} ({name})")
            elif action == "QUARANTINE":
                proc.suspend()
                self.log(f"[ACTION] QUARANTINE (suspend) issued to pid={pid} ({name})")
        except psutil.NoSuchProcess:
            self.log(f"[ACTION] Process pid={pid} no longer exists.")
        except Exception as e:
            self.log(f"[ACTION-ERROR] Failed to {action} pid={pid}: {e}")

    # -------- Accessors -------- #

    def get_latest_snapshot(self) -> Optional[SystemSnapshot]:
        with self.lock:
            if self.snapshots:
                return self.snapshots[-1]
            return None

    def get_event_log(self) -> List[str]:
        with self.lock:
            return list(self.event_log)

    def get_worker_statuses(self) -> List[WorkerStatus]:
        with self.lock:
            return list(self.worker_status.values())

    def get_graph(self) -> nx.DiGraph:
        with self.lock:
            return self.current_graph.copy()

    def get_global_state(self) -> Tuple[float, float, float, bool, bool]:
        with self.lock:
            return (
                self.last_global_cpu,
                self.last_global_mem,
                self.last_global_disk,
                self.low_memory_mode,
                self.low_disk_mode,
            )

    def get_profiles_summary(self, max_items: int = 8) -> List[str]:
        with self.lock:
            profiles = list(self.process_profiles.values())
        profiles = sorted(profiles, key=lambda p: p.count, reverse=True)[:max_items]
        lines = []
        for p in profiles:
            lines.append(
                f"{p.name}: avgCPU={p.avg_cpu:.1f}% avgMEM={p.avg_mem:.1f}% samples={p.count}"
            )
        return lines

    def get_strategy_summary(self) -> str:
        s = self.current_strategy
        return (
            f"name={s.name}, "
            f"scan_factor={s.scan_interval_factor:.2f}, "
            f"cpu_thr={s.cpu_threshold:.1f}%, "
            f"mem_thr={s.mem_threshold:.1f}%, "
            f"aggro={s.enforcement_aggressiveness:.2f}"
        )

    def stop(self):
        self.running = False
        self.log("Queen brain stopping.")
        try:
            self._save_brain()
        except Exception as e:
            self.log(f"[PERSIST] Error saving brain on stop: {e}")

# -------------- WORKERS -------------- #

class BaseWorker(threading.Thread):
    def __init__(self, name: str, queen: QueenBrain, base_interval: float = 5.0):
        super().__init__(daemon=True)
        self.name = name
        self.queen = queen
        self.base_interval = base_interval
        self.current_interval = base_interval
        self.running = True
        self.queen.register_worker(name, base_interval)

    def run(self):
        self.queen.log(f"Worker {self.name} started.")
        while self.running:
            try:
                self._adapt_interval()
                self.step()
            except Exception as e:
                self.queen.log(f"[ERROR] Worker {self.name}: {e}")
            time.sleep(self.current_interval)

    def _adapt_interval(self):
        cpu, mem, disk, low_mem, low_disk = self.queen.get_global_state()
        strategy = self.queen.current_strategy

        interval = self.base_interval * strategy.scan_interval_factor

        if low_mem or low_disk:
            interval *= 2.0
        elif mem > 40.0:
            interval *= 1.5

        interval = max(2.0, min(interval, 60.0))

        self.current_interval = interval
        self.queen.update_worker_status(
            self.name,
            f"Active (interval={self.current_interval:.1f}s)",
            interval=interval,
        )

    def step(self):
        raise NotImplementedError

    def stop(self):
        self.running = False
        self.queen.log(f"Worker {self.name} stopped.")

class ProcessScannerWorker(BaseWorker):
    def __init__(self, queen: QueenBrain, base_interval: float = 5.0):
        super().__init__("ProcessScanner", queen, base_interval)

    def step(self):
        timestamp = time.time()
        nodes: Dict[str, NodeInfo] = {}
        edges: List[EdgeInfo] = []

        for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent", "ppid", "exe"]):
            info = proc.info
            pid = str(info["pid"])
            ppid = str(info.get("ppid", 0))
            cpu = float(info.get("cpu_percent") or 0.0)
            mem = float(info.get("memory_percent") or 0.0)
            name = info.get("name", "proc")
            exe_path = info.get("exe") or ""
            label = f"{name} ({pid})"

            trust = "unknown"
            exe_low = exe_path.lower()
            if "windows" in exe_low or "system32" in exe_low or "/usr" in exe_low:
                trust = "system"
            elif "program files" in exe_low or "/opt" in exe_low:
                trust = "installed"
            elif "temp" in exe_low or "download" in exe_low:
                trust = "temp"
            else:
                trust = "user"

            nodes[pid] = NodeInfo(
                id=pid,
                label=label,
                type="process",
                cpu=cpu,
                memory=mem,
                extra={"name": name, "exe": exe_path, "trust": trust},
            )

            if ppid != "0":
                edges.append(EdgeInfo(
                    src=ppid,
                    dst=pid,
                    type="parent_child",
                    extra={},
                ))

        snapshot = SystemSnapshot(
            timestamp=timestamp,
            nodes=nodes,
            edges=edges,
        )
        self.queen.submit_snapshot(snapshot)
        self.queen.update_worker_status(self.name, f"Scanned {len(nodes)} processes")

class SystemMetricsWorker(BaseWorker):
    def __init__(self, queen: QueenBrain, base_interval: float = 3.0):
        super().__init__("SystemMetrics", queen, base_interval)

    def step(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        try:
            disk = psutil.disk_usage("/").percent
        except Exception:
            disk = 0.0

        self.queen.update_global_metrics(cpu, mem, disk)

        msg = f"CPU={cpu:.1f}% MEM={mem:.1f}% DISK={disk:.1f}%"
        self.queen.update_worker_status(self.name, msg)

        if cpu > self.queen.cpu_threshold:
            self.queen.log(f"[GLOBAL] High CPU usage: {cpu:.1f}%")
        if mem > self.queen.mem_threshold:
            self.queen.log(f"[GLOBAL] High memory usage (cap=50%): {mem:.1f}%")
        if disk > self.queen.disk_high_threshold:
            self.queen.log(f"[GLOBAL] High disk usage: {disk:.1f}%")

class NetworkScannerWorker(BaseWorker):
    def __init__(self, queen: QueenBrain, base_interval: float = 7.0):
        super().__init__("NetworkScanner", queen, base_interval)

    def step(self):
        proc_conn_counts: Dict[str, Dict[str, int]] = {}
        listening_ports = []

        try:
            connections = psutil.net_connections(kind='inet')
        except Exception as e:
            self.queen.log(f"[ERROR] NetworkScanner: {e}")
            self.queen.update_worker_status(self.name, "Error reading connections")
            return

        for c in connections:
            pid = c.pid
            if pid is None:
                continue
            pid_str = str(pid)

            if pid_str not in proc_conn_counts:
                proc_conn_counts[pid_str] = {"outgoing": 0, "listening": 0}

            if c.status == psutil.CONN_LISTEN:
                proc_conn_counts[pid_str]["listening"] += 1
                if c.laddr:
                    listening_ports.append((pid_str, c.laddr.port))
            else:
                proc_conn_counts[pid_str]["outgoing"] += 1

        self.queen.update_network_signals(proc_conn_counts, listening_ports)
        total_procs = len(proc_conn_counts)
        self.queen.update_worker_status(self.name, f"Scanned net for {total_procs} procs")

class FileActivityWorker(BaseWorker):
    def __init__(self, queen: QueenBrain, base_interval: float = 10.0):
        super().__init__("FileActivity", queen, base_interval)
        self.protected_paths = [
            os.path.expanduser("~"),
        ]
        self.last_snapshots: Dict[str, set] = {}

    def _snapshot_dir(self, root: str) -> set:
        files = set()
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                files.add(os.path.join(dirpath, f))
        return files

    def step(self):
        suspicious_events = []
        for root in self.protected_paths:
            if not os.path.isdir(root):
                continue

            try:
                current = self._snapshot_dir(root)
            except Exception as e:
                self.queen.log(f"[ERROR] FileActivity snapshot {root}: {e}")
                continue

            prev = self.last_snapshots.get(root, set())
            created = current - prev
            deleted = prev - current

            if len(created) + len(deleted) > 50:
                suspicious_events.append((root, len(created), len(deleted)))

            self.last_snapshots[root] = current

        if suspicious_events:
            self.queen.update_file_signals(suspicious_events)
            msg = "; ".join(
                f"{root}: +{c} / -{d}" for (root, c, d) in suspicious_events
            )
            self.queen.update_worker_status(self.name, f"Spike: {msg}")
        else:
            self.queen.update_worker_status(self.name, "No suspicious spikes")

# -------------- MAIN GUI -------------- #

class HiveGUI:
    def __init__(self, queen: QueenBrain):
        self.queen = queen

        self.root = tk.Tk()
        self.root.title("Hive Guardian - Queen & Workers (ASI Security/Ops Guardian)")
        self.root.geometry("1250x780")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_overview_tab()
        self._build_workers_tab()
        self._build_events_tab()
        self._build_map_tab()

        self._build_menu()

        self._schedule_updates()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        control_menu = tk.Menu(menubar, tearoff=0)
        control_menu.add_command(label="Open Decision Panel", command=self._open_decision_panel)
        control_menu.add_command(label="Save Brain Now", command=self._manual_save)
        menubar.add_cascade(label="Control", menu=control_menu)

    def _build_overview_tab(self):
        self.overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_frame, text="Overview")

        self.lbl_overview = tk.Label(
            self.overview_frame,
            text="Waiting for data...",
            anchor="w",
            justify="left",
            font=("Consolas", 11),
        )
        self.lbl_overview.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_workers_tab(self):
        self.workers_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.workers_frame, text="Workers")

        columns = ("name", "interval", "last_run", "last_result", "active")
        self.tree_workers = ttk.Treeview(
            self.workers_frame, columns=columns, show="headings", height=20
        )
        for col in columns:
            self.tree_workers.heading(col, text=col)
            width = 180 if col != "last_result" else 360
            self.tree_workers.column(col, width=width, anchor="w")
        self.tree_workers.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _build_events_tab(self):
        self.events_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.events_frame, text="Events")

        self.txt_events = tk.Text(
            self.events_frame, wrap="none", font=("Consolas", 10)
        )
        self.txt_events.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(
            self.events_frame, command=self.txt_events.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_events["yscrollcommand"] = scrollbar.set

    def _build_map_tab(self):
        self.map_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.map_frame, text="Process Map")

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Process Cave Map")
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def _schedule_updates(self):
        self._update_overview()
        self._update_workers()
        self._update_events()
        self._update_map()
        self.root.after(1000, self._schedule_updates)

    def _update_overview(self):
        snapshot = self.queen.get_latest_snapshot()
        cpu, mem, disk, low_mem_mode, low_disk_mode = self.queen.get_global_state()

        lines = []
        lines.append("Primary mission: SYSTEM SECURITY + SYSTEM OPERATIONS STABILITY")
        lines.append("")
        lines.append(f"Total RAM: ~{self.queen.total_mem_gb:.1f} GB")
        lines.append(
            f"Memory soft cap: {self.queen.mem_cap_fraction*100:.0f}% "
            f"(guardian backs off when system mem > {self.queen.mem_threshold:.0f}%)"
        )
        lines.append(f"Global CPU: {cpu:.1f}%   MEM: {mem:.1f}%   DISK: {disk:.1f}%")
        lines.append(f"Low memory mode: {'ON' if low_mem_mode else 'OFF'}")
        lines.append(f"Low disk mode:   {'ON' if low_disk_mode else 'OFF'}")
        lines.append(
            f"Enforcement mode: "
            f"{'REAL ACTIONS' if self.queen.enforce_actions_for_real else 'DRY RUN ONLY'}"
        )
        lines.append(f"Paranoia level: {self.queen.paranoia_level:.2f}")
        lines.append(f"Strategy: {self.queen.get_strategy_summary()}")
        lines.append("")

        if snapshot is None:
            lines.append("Waiting for first process snapshot...")
        else:
            total_nodes = len(snapshot.nodes)
            cpu_thr = self.queen.current_strategy.cpu_threshold
            mem_thr = self.queen.current_strategy.mem_threshold

            hot_nodes = [
                n for n in snapshot.nodes.values()
                if n.cpu > cpu_thr or n.memory > mem_thr
            ]

            lines.append(
                f"Latest snapshot: "
                f"{time.strftime('%H:%M:%S', time.localtime(snapshot.timestamp))}"
            )
            lines.append(f"Total nodes (processes): {total_nodes}")
            lines.append(
                f"Hot nodes (> {cpu_thr:.0f}% CPU or > {mem_thr:.0f}% MEM): {len(hot_nodes)}"
            )
            lines.append("")
            lines.append("Top 5 CPU processes:")

            sorted_nodes = sorted(
                snapshot.nodes.values(), key=lambda n: n.cpu, reverse=True
            )[:5]
            for node in sorted_nodes:
                lines.append(
                    f"- {node.label}: CPU={node.cpu:.1f}% MEM={node.memory:.1f}%"
                )

        lines.append("")
        lines.append("Learned habit profiles (top by samples):")
        for line in self.queen.get_profiles_summary(max_items=6):
            lines.append(f"  {line}")

        self.lbl_overview.config(text="\n".join(lines))

    def _update_workers(self):
        for item in self.tree_workers.get_children():
            self.tree_workers.delete(item)

        statuses = self.queen.get_worker_statuses()
        for ws in statuses:
            last_run_str = (
                time.strftime("%H:%M:%S", time.localtime(ws.last_run))
                if ws.last_run > 0 else "never"
            )
            self.tree_workers.insert(
                "", tk.END,
                values=(
                    ws.name,
                    f"{ws.interval:.1f}s",
                    last_run_str,
                    ws.last_result,
                    "yes" if ws.active else "no",
                )
            )

    def _update_events(self):
        events = self.queen.get_event_log()
        self.txt_events.delete("1.0", tk.END)
        self.txt_events.insert(tk.END, "\n".join(events))
        self.txt_events.see(tk.END)

    def _update_map(self):
        graph = self.queen.get_graph()
        self.ax.clear()
        self.ax.set_title("Process Cave Map")
        self.ax.axis("off")

        if graph.number_of_nodes() == 0:
            self.canvas.draw()
            return

        try:
            pos = nx.spring_layout(graph, k=0.5, iterations=25)
        except Exception:
            pos = nx.random_layout(graph)

        cpu_values = nx.get_node_attributes(graph, "cpu")
        mem_values = nx.get_node_attributes(graph, "memory")
        colors = []

        cpu_thr = self.queen.current_strategy.cpu_threshold
        mem_thr = self.queen.current_strategy.mem_threshold

        for node in graph.nodes():
            cpu = cpu_values.get(node, 0.0)
            mem = mem_values.get(node, 0.0)
            if cpu > cpu_thr or mem > mem_thr:
                colors.append("red")
            else:
                colors.append("skyblue")

        nx.draw_networkx_nodes(graph, pos, ax=self.ax, node_color=colors, node_size=280)
        nx.draw_networkx_edges(graph, pos, ax=self.ax, arrows=False, alpha=0.4)

        labels = nx.get_node_attributes(graph, "label")
        short_labels = {
            k: (v[:11] + "..." if len(v) > 14 else v)
            for k, v in labels.items()
        }
        nx.draw_networkx_labels(graph, pos, labels=short_labels, ax=self.ax, font_size=7)

        self.canvas.draw()

    def _manual_save(self):
        self.queen.log("[ADMIN] Manual brain save requested.")
        self.queen._save_brain()
        messagebox.showinfo("Save Brain", "Brain state saved.")

    def _open_decision_panel(self):
        DecisionPanelGUI(self.root, self.queen)

    def run(self):
        self.root.mainloop()

# -------------- DECISION PANEL GUI -------------- #

class DecisionPanelGUI:
    LOCKOUT_THRESHOLD = 3
    LOCKOUT_SECONDS = 30

    def __init__(self, parent, queen: QueenBrain):
        self.queen = queen
        self.failed_attempts = 0
        self.lockout_until = 0.0

        if not self._authenticate(parent):
            return

        self.win = tk.Toplevel(parent)
        self.win.title("Queen Decision Panel (Admin)")
        self.win.geometry("470x340")
        self.win.resizable(False, False)

        self._build_ui()
        self._update_ui()

    def _authenticate(self, parent) -> bool:
        now = time.time()
        if now < self.lockout_until:
            remaining = int(self.lockout_until - now)
            messagebox.showerror("Locked Out", f"Too many failed attempts. Try again in {remaining} seconds.")
            return False

        while True:
            pwd = simpledialog.askstring("Admin Authentication", "Enter admin password:", show="*", parent=parent)
            if pwd is None:
                return False

            if self.queen.verify_admin_password(pwd):
                self.queen.log("[ADMIN] Decision Panel access granted.")
                self.failed_attempts = 0
                return True

            self.failed_attempts += 1
            self.queen.log("[ADMIN] Decision Panel access denied (wrong password).")
            if self.failed_attempts >= self.LOCKOUT_THRESHOLD:
                self.lockout_until = time.time() + self.LOCKOUT_SECONDS
                messagebox.showerror("Locked Out", "Too many failed attempts. Locked for 30 seconds.")
                return False
            else:
                messagebox.showerror("Access Denied", "Incorrect password.")

    def _build_ui(self):
        frame = ttk.Frame(self.win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_status = tk.Label(frame, text="", anchor="w", justify="left", font=("Consolas", 9))
        self.lbl_status.pack(fill=tk.X, pady=(0, 8))

        self.enforce_var = tk.BooleanVar(value=self.queen.enforce_actions_for_real)
        enforce_chk = tk.Checkbutton(
            frame,
            text="Enable REAL enforcement (QUARANTINE/KILL)",
            variable=self.enforce_var,
            command=self._on_toggle_enforcement,
        )
        enforce_chk.pack(anchor="w", pady=4)

        tk.Label(frame, text="Paranoia level (0 = cautious, 1 = aggressive):").pack(anchor="w")
        self.paranoia_scale = tk.Scale(
            frame, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL,
            length=250,
            command=self._on_paranoia_change,
        )
        self.paranoia_scale.set(self.queen.paranoia_level)
        self.paranoia_scale.pack(anchor="w", pady=4)

        btn_frame1 = ttk.Frame(frame)
        btn_frame1.pack(anchor="e", pady=6, fill=tk.X)

        ttk.Button(btn_frame1, text="Save Brain Now", command=self._on_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame1, text="Change Admin Password", command=self._on_change_password).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame1, text="Network Backup Path", command=self._on_set_network_backup).pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_frame1, text="Close Panel", command=self.win.destroy).pack(side=tk.RIGHT, padx=4)

        shutdown_btn = ttk.Button(frame, text="Shutdown Queen (exit program)", command=self._on_shutdown)
        shutdown_btn.pack(anchor="e", pady=8)

    def _update_ui(self):
        cpu, mem, disk, low_mem_mode, low_disk_mode = self.queen.get_global_state()
        status_lines = [
            f"CPU={cpu:.1f}% MEM={mem:.1f}% DISK={disk:.1f}%",
            f"Low memory mode: {'ON' if low_mem_mode else 'OFF'}",
            f"Low disk mode:   {'ON' if low_disk_mode else 'OFF'}",
            f"Enforcement: {'REAL ACTIONS' if self.queen.enforce_actions_for_real else 'DRY RUN ONLY'}",
            f"Paranoia: {self.queen.paranoia_level:.2f}",
            f"Strategy: {self.queen.get_strategy_summary()}",
            f"Network backup: {self.queen.network_backup_path or 'None'}",
        ]
        self.lbl_status.config(text="\n".join(status_lines))

    def _on_toggle_enforcement(self):
        val = bool(self.enforce_var.get())
        mode = "REAL ACTIONS" if val else "DRY RUN ONLY"
        self.queen.enforce_actions_for_real = val
        self.queen.log(f"[ADMIN] Enforcement mode changed to: {mode}")
        self._update_ui()

    def _on_paranoia_change(self, value: str):
        try:
            v = float(value)
        except ValueError:
            return
        self.queen.paranoia_level = v
        self.queen.log(f"[ADMIN] Paranoia level set to {v:.2f}")
        self._update_ui()

    def _on_save(self):
        self.queen.log("[ADMIN] Manual brain save from Decision Panel.")
        self.queen._save_brain()
        messagebox.showinfo("Save Brain", "Brain state saved.")
        self._update_ui()

    def _on_set_network_backup(self):
        path = simpledialog.askstring(
            "Network Backup Path",
            "Enter network brain path (leave empty to disable):",
            initialvalue=self.queen.network_backup_path or "",
            show=None,
            parent=self.win,
        )
        if path is None:
            return
        path = path.strip()
        self.queen.set_network_backup_path(path or None)
        self._update_ui()

    def _on_change_password(self):
        current = simpledialog.askstring("Change Password", "Enter current admin password:", show="*", parent=self.win)
        if current is None:
            return

        new1 = simpledialog.askstring("Change Password", "Enter new admin password:", show="*", parent=self.win)
        if new1 is None or not new1.strip():
            messagebox.showerror("Error", "New password cannot be empty.")
            return

        new2 = simpledialog.askstring("Change Password", "Confirm new admin password:", show="*", parent=self.win)
        if new2 is None:
            return

        if new1 != new2:
            messagebox.showerror("Error", "New passwords do not match.")
            return

        if self.queen.change_admin_password(current, new1):
            self.queen.log("[ADMIN] Admin password changed.")
            messagebox.showinfo("Success", "Admin password changed and persisted.")
        else:
            messagebox.showerror("Error", "Current password incorrect.")

        self._update_ui()

    def _on_shutdown(self):
        if messagebox.askyesno("Confirm Shutdown", "Shut down Queen and exit program?"):
            self.queen.log("[ADMIN] Shutting down Queen from Decision Panel.")
            self.queen.stop()
            os._exit(0)

# -------------- MAIN ENTRY -------------- #

def main():
    service_mode = "--service" in sys.argv

    queen = QueenBrain()

    w1 = ProcessScannerWorker(queen, base_interval=5.0)
    w2 = SystemMetricsWorker(queen, base_interval=3.0)
    w3 = NetworkScannerWorker(queen, base_interval=7.0)
    w4 = FileActivityWorker(queen, base_interval=10.0)

    w1.start()
    w2.start()
    w3.start()
    w4.start()

    if service_mode:
        print("[MODE] Running in service mode (no GUI). Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        finally:
            queen.stop()
            w1.stop()
            w2.stop()
            w3.stop()
            w4.stop()
    else:
        gui = HiveGUI(queen)
        try:
            gui.run()
        finally:
            queen.stop()
            w1.stop()
            w2.stop()
            w3.stop()
            w4.stop()

if __name__ == "__main__":
    main()

