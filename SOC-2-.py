#!/usr/bin/env python3
"""
Hive Guardian - ASI Queen v5

Primary mission:
- System security
- System operations stability

Features:
- Auto-loader for required libraries
- Startup hardware + software audit
- Queen brain + workers (process scanner, system metrics)
- Habit learning (per-process behavior profiles)
- Prometheus-style process "cave" scanning
- Memory+disk-aware behavior, with low-memory mode and cautious resource use
- Anomaly detection with WATCH / QUARANTINE / KILL actions
- Strategy engine: queen writes its own operational strategy, simulates it, adopts better ones
- Security/ops priority config: critical whitelist, paranoia level, security vs ops weighting
- Persistence layer: saves brain (profiles, strategy, config) to disk, optional network backup
- GUI with Overview / Workers / Events / Process Map
- Decision Panel GUI (admin-only): toggle enforcement, paranoia, shutdown queen

NOTE:
- By default, enforcement is DRY RUN ONLY.
- Real QUARANTINE/KILL actions require admin enabling via Decision Panel.
"""

import sys
import subprocess
import importlib
import threading
import time
import queue
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from random import uniform

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

# Ensure packages before imports that depend on them
ensure_all_packages()

import platform
import psutil
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

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

    def update(self, cpu: float, mem: float):
        self.count += 1
        alpha = 0.1
        self.avg_cpu = (1 - alpha) * self.avg_cpu + alpha * cpu
        self.avg_mem = (1 - alpha) * self.avg_mem + alpha * mem


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


@dataclass
class PersistenceState:
    process_profiles: Dict[str, Dict]
    current_strategy: Dict
    paranoia_level: float
    security_weight: float
    ops_weight: float
    critical_process_whitelist: List[str]
    enforce_actions_for_real: bool


@dataclass
class PersistenceConfig:
    version: int = 1
    local_file_path: str = ""
    network_file_path: str = ""
    use_network: bool = False


# -------------- PERSISTENCE HELPER -------------- #

class PersistenceManager:
    """
    Handles saving/loading the queen's brain to disk.
    Supports local file + optional network backup.
    Remembers config so it doesn't ask every run.
    """

    def __init__(self, app_name: str = "hive_guardian_queen"):
        self.app_name = app_name

        home = os.path.expanduser("~")
        config_dir = os.path.join(home, ".hive_guardian")
        os.makedirs(config_dir, exist_ok=True)

        self.config_path = os.path.join(config_dir, "config.json")

        # Default locations
        self.local_file_path = os.path.join(config_dir, "queen_brain.json")
        self.network_file_path = ""
        self.use_network = False

        self._load_or_init_config()

    def _load_or_init_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg = PersistenceConfig(**data)
                self.local_file_path = cfg.local_file_path or self.local_file_path
                self.network_file_path = cfg.network_file_path or ""
                self.use_network = cfg.use_network
            except Exception:
                # If config is corrupt, keep defaults and overwrite later
                pass
        else:
            # First time: create default config; optional network can be chosen later
            cfg = PersistenceConfig(
                version=1,
                local_file_path=self.local_file_path,
                network_file_path="",
                use_network=False,
            )
            self._write_config(cfg)

    def _write_config(self, cfg: PersistenceConfig):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(cfg), f, indent=2)
        except Exception as e:
            print(f"[PERSISTENCE] Failed to write config: {e}")

    def set_network_path(self, path: str, use_network: bool):
        self.network_file_path = path
        self.use_network = use_network
        cfg = PersistenceConfig(
            version=1,
            local_file_path=self.local_file_path,
            network_file_path=self.network_file_path,
            use_network=self.use_network,
        )
        self._write_config(cfg)

    def _can_use_network(self) -> bool:
        if not self.use_network or not self.network_file_path:
            return False
        # Simple reachability check: can we write a test?
        try:
            directory = os.path.dirname(self.network_file_path) or "."
            if not os.path.exists(directory):
                return False
            # We won't actually write here — just test directory existence
            return True
        except Exception:
            return False

    def load_state(self) -> Optional[PersistenceState]:
        # Prefer network if configured and reachable
        paths = []
        if self._can_use_network():
            paths.append(self.network_file_path)
        paths.append(self.local_file_path)

        for path in paths:
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PersistenceState(**data)
            except Exception as e:
                print(f"[PERSISTENCE] Failed to load state from {path}: {e}")
        return None

    def save_state(self, state: PersistenceState):
        data = asdict(state)

        # Always save locally
        try:
            with open(self.local_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[PERSISTENCE] Failed to save local state: {e}")

        # Optionally save network copy
        if self._can_use_network():
            try:
                with open(self.network_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[PERSISTENCE] Failed to save network state: {e}")


# -------------- QUEEN BRAIN -------------- #

class QueenBrain:
    """
    Queen: central intelligence.
    """

    def __init__(self, persistence: PersistenceManager):
        self.lock = threading.Lock()
        self.snapshots: List[SystemSnapshot] = []
        self.current_graph = nx.DiGraph()
        self.worker_status: Dict[str, WorkerStatus] = {}
        self.event_log: List[str] = []
        self.telemetry_queue: "queue.Queue[SystemSnapshot]" = queue.Queue()
        self.running = True

        self.persistence = persistence

        # Enforcement mode
        self.enforce_actions_for_real = False

        # System capacity
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

        self._load_persisted_state()

        self._run_hardware_audit()
        self._run_software_audit()

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    # -------- Persistence integration -------- #

    def _load_persisted_state(self):
        state = self.persistence.load_state()
        if not state:
            self.log("[PERSISTENCE] No previous state found; starting fresh.")
            return

        try:
            self.paranoia_level = state.paranoia_level
            self.security_weight = state.security_weight
            self.ops_weight = state.ops_weight
            self.enforce_actions_for_real = state.enforce_actions_for_real

            self.critical_process_whitelist = set(state.critical_process_whitelist)

            self.process_profiles = {}
            for name, pdata in state.process_profiles.items():
                self.process_profiles[name] = ProcessProfile(**pdata)

            self.current_strategy = Strategy(**state.current_strategy)

            self.log("[PERSISTENCE] Loaded previous brain state successfully.")
        except Exception as e:
            self.log(f"[PERSISTENCE] Failed to apply loaded state: {e}")

    def save_state_now(self):
        with self.lock:
            profiles_dict = {name: asdict(p) for name, p in self.process_profiles.items()}
            strategy_dict = asdict(self.current_strategy)
            state = PersistenceState(
                process_profiles=profiles_dict,
                current_strategy=strategy_dict,
                paranoia_level=self.paranoia_level,
                security_weight=self.security_weight,
                ops_weight=self.ops_weight,
                critical_process_whitelist=list(self.critical_process_whitelist),
                enforce_actions_for_real=self.enforce_actions_for_real,
            )
        self.persistence.save_state(state)
        self.log("[PERSISTENCE] Brain state saved.")

    # -------- Logging -------- #

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self.lock:
            self.event_log.append(line)
            if len(self.event_log) > 2000:
                self.event_log.pop(0)
        print(line)

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

    # -------- Telemetry / metrics -------- #

    def submit_snapshot(self, snapshot: SystemSnapshot):
        self.telemetry_queue.put(snapshot)

    def update_global_metrics(self, cpu: float, mem: float, disk: float):
        with self.lock:
            self.last_global_cpu = cpu
            self.last_global_mem = mem
            self.last_global_disk = disk

            self.low_memory_mode = mem > (self.mem_cap_fraction * 100.0)
            self.low_disk_mode = disk > self.disk_high_threshold

    # -------- Habit learning -------- #

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
            profile.update(node.cpu, node.memory)

    # -------- Strategy / simulation -------- #

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

    # -------- Anomaly classification / action -------- #

    def _classify_anomaly(self, node: NodeInfo) -> str:
        name = node.extra.get("name", None)
        if not name:
            if "(" in node.label:
                name = node.label.split("(", 1)[0].strip()
            else:
                name = node.label

        profile = self.process_profiles.get(name)
        cpu = node.cpu
        mem = node.memory

        is_critical = name in self.critical_process_whitelist

        if profile is None or profile.count < 5:
            return "WATCH"

        cpu_ratio = (cpu + 1.0) / (profile.avg_cpu + 1.0)
        mem_ratio = (mem + 1.0) / (profile.avg_mem + 1.0)

        base_action = "WATCH"
        if cpu_ratio > 6.0 or mem_ratio > 6.0:
            base_action = "KILL"
        elif cpu_ratio > 3.0 or mem_ratio > 3.0:
            base_action = "QUARANTINE"

        if self.paranoia_level < 0.2 and base_action == "KILL":
            base_action = "QUARANTINE"
        if self.paranoia_level < 0.05:
            base_action = "WATCH"

        if self.current_strategy.enforcement_aggressiveness < 0.2 and base_action == "KILL":
            base_action = "QUARANTINE"
        if self.current_strategy.enforcement_aggressiveness < 0.05:
            base_action = "WATCH"

        if is_critical and base_action == "KILL":
            base_action = "QUARANTINE"

        return base_action

    # -------- Main loop -------- #

    def _run_loop(self):
        self.log(
            f"Queen brain started. Total RAM ~ {self.total_mem_gb:.1f} GB. "
            f"Soft cap at {self.mem_cap_fraction*100:.0f}% usage."
        )
        last_evolve = time.time()
        last_save = time.time()
        evolve_interval = 60.0
        save_interval = 120.0

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

            if now - last_save > save_interval:
                self.save_state_now()
                last_save = now

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

        cpu_thr = self.current_strategy.cpu_threshold
        mem_thr = self.current_strategy.mem_threshold

        hot_nodes = [
            node for node in snapshot.nodes.values()
            if node.cpu > cpu_thr or node.memory > mem_thr
        ]

        for node in hot_nodes:
            action = self._classify_anomaly(node)
            msg = (
                f"[ANOMALY] {node.label} CPU={node.cpu:.1f}% MEM={node.memory:.1f}% "
                f"=> ACTION={action} via strategy={self.current_strategy.name}"
            )
            self.log(msg)

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
        self.save_state_now()


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

        for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent", "ppid"]):
            info = proc.info
            pid = str(info["pid"])
            ppid = str(info.get("ppid", 0))
            cpu = float(info.get("cpu_percent") or 0.0)
            mem = float(info.get("memory_percent") or 0.0)
            name = info.get("name", "proc")
            label = f"{name} ({pid})"

            nodes[pid] = NodeInfo(
                id=pid,
                label=label,
                type="process",
                cpu=cpu,
                memory=mem,
                extra={"name": name},
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


# -------------- GUI: MAIN -------------- #

class HiveGUI:
    def __init__(self, queen: QueenBrain, persistence: PersistenceManager):
        self.queen = queen
        self.persistence = persistence

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
        control_menu.add_separator()
        control_menu.add_command(label="Save Brain Now", command=self._manual_save)
        control_menu.add_command(label="Configure Network Backup", command=self._configure_network_backup)
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
        lines.append(f"Security weight: {self.queen.security_weight:.1f}   Ops weight: {self.queen.ops_weight:.1f}")
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

    # -------- Menu actions -------- #

    def _manual_save(self):
        self.queen.save_state_now()
        messagebox.showinfo("Save Brain", "Brain state saved.")

    def _configure_network_backup(self):
        top = tk.Toplevel(self.root)
        top.title("Configure Network Backup")
        top.geometry("400x160")
        top.resizable(False, False)

        tk.Label(top, text="Network brain file path:").pack(anchor="w", padx=10, pady=5)
        path_var = tk.StringVar(value=self.persistence.network_file_path)
        entry = tk.Entry(top, textvariable=path_var, width=50)
        entry.pack(anchor="w", padx=10)

        use_var = tk.BooleanVar(value=self.persistence.use_network)
        chk = tk.Checkbutton(top, text="Use network backup (if reachable)", variable=use_var)
        chk.pack(anchor="w", padx=10, pady=5)

        def save_cfg():
            path = path_var.get().strip()
            use = bool(use_var.get())
            self.persistence.set_network_path(path, use)
            self.queen.log(f"[PERSISTENCE] Network backup config updated: use={use}, path={path!r}")
            top.destroy()

        btn_frame = ttk.Frame(top)
        btn_frame.pack(anchor="e", padx=10, pady=10)
        ttk.Button(btn_frame, text="Save", command=save_cfg).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=top.destroy).pack(side=tk.LEFT, padx=5)

    def _open_decision_panel(self):
        DecisionPanelGUI(self.root, self.queen)

    def run(self):
        self.root.mainloop()


# -------------- GUI: DECISION PANEL (ADMIN) -------------- #

class DecisionPanelGUI:
    """
    Smaller GUI for AI actions and admin-only control:
    - Shows basic state
    - Toggle dry run vs real actions
    - Adjust paranoia
    - Shutdown queen
    """

    ADMIN_PASSWORD = "admin"  # change this in real use

    def __init__(self, parent, queen: QueenBrain):
        self.queen = queen

        pwd = simpledialog.askstring("Admin Authentication", "Enter admin password:", show="*")
        if pwd is None:
            return
        if pwd != self.ADMIN_PASSWORD:
            messagebox.showerror("Access Denied", "Incorrect password.")
            return

        self.win = tk.Toplevel(parent)
        self.win.title("Queen Decision Panel (Admin)")
        self.win.geometry("420x260")
        self.win.resizable(False, False)

        self._build_ui()
        self._update_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_status = tk.Label(frame, text="", anchor="w", justify="left", font=("Consolas", 9))
        self.lbl_status.pack(fill=tk.X, pady=(0, 8))

        # Enforcement toggle
        self.enforce_var = tk.BooleanVar(value=self.queen.enforce_actions_for_real)
        enforce_chk = tk.Checkbutton(
            frame,
            text="Enable REAL enforcement (QUARANTINE/KILL)",
            variable=self.enforce_var,
            command=self._on_toggle_enforcement,
        )
        enforce_chk.pack(anchor="w", pady=4)

        # Paranoia slider
        tk.Label(frame, text="Paranoia level (0 = cautious, 1 = aggressive):").pack(anchor="w")
        self.paranoia_scale = tk.Scale(
            frame, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL,
            length=250,
            command=self._on_paranoia_change,
        )
        self.paranoia_scale.set(self.queen.paranoia_level)
        self.paranoia_scale.pack(anchor="w", pady=4)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(anchor="e", pady=10, fill=tk.X)

        ttk.Button(btn_frame, text="Save Brain Now", command=self._on_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Close Panel", command=self.win.destroy).pack(side=tk.RIGHT, padx=4)

        # Shutdown
        shutdown_btn = ttk.Button(frame, text="Shutdown Queen (exit program)", command=self._on_shutdown)
        shutdown_btn.pack(anchor="e", pady=5)

    def _update_ui(self):
        cpu, mem, disk, low_mem_mode, low_disk_mode = self.queen.get_global_state()
        status_lines = [
            f"CPU={cpu:.1f}% MEM={mem:.1f}% DISK={disk:.1f}%",
            f"Low memory mode: {'ON' if low_mem_mode else 'OFF'}",
            f"Low disk mode:   {'ON' if low_disk_mode else 'OFF'}",
            f"Enforcement: {'REAL ACTIONS' if self.queen.enforce_actions_for_real else 'DRY RUN ONLY'}",
            f"Paranoia: {self.queen.paranoia_level:.2f}",
            f"Strategy: {self.queen.get_strategy_summary()}",
        ]
        self.lbl_status.config(text="\n".join(status_lines))

    def _on_toggle_enforcement(self):
        val = bool(self.enforce_var.get())
        self.queen.enforce_actions_for_real = val
        mode = "REAL ACTIONS" if val else "DRY RUN ONLY"
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

    def _on_shutdown(self):
        if messagebox.askyesno("Confirm Shutdown", "Shut down Queen and exit program?"):
            self.queen.log("[ADMIN] Shutting down Queen from Decision Panel.")
            self.queen.stop()
            os._exit(0)

    def _on_save(self):
        self.queen.save_state_now()
        messagebox.showinfo("Save Brain", "Brain state saved.")
        self._update_ui()


# -------------- MAIN ENTRY -------------- #

def main():
    persistence = PersistenceManager()
    queen = QueenBrain(persistence)

    w1 = ProcessScannerWorker(queen, base_interval=5.0)
    w2 = SystemMetricsWorker(queen, base_interval=3.0)
    w1.start()
    w2.start()

    gui = HiveGUI(queen, persistence)
    try:
        gui.run()
    finally:
        queen.stop()
        w1.stop()
        w2.stop()


if __name__ == "__main__":
    main()

