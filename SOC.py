#!/usr/bin/env python3
"""
Hive Guardian - ASI Queen v4

Primary mission:
- System security
- System operations stability

Features:
- Auto-loader for required libraries
- Startup hardware + software audit
- Queen brain + workers
- Prometheus-style process "cave" scanning
- Memory + disk aware (backs off under load, respects ~50% mem cap logic)
- Process habit learning (per-name profiles)
- Anomaly detection with WATCH / QUARANTINE / KILL actions
- Strategy engine: queen writes its own operational strategy, simulates it, adopts better ones
- Security/ops priority config: critical whitelist, paranoia level, security vs ops weighting
- GUI with overview, workers, events, and process map

NOTE:
- By default, enforcement is DRY RUN ONLY.
  To allow real QUARANTINE/KILL actions, set:
      queen.enforce_actions_for_real = True
"""

import sys
import subprocess
import importlib
import threading
import time
import queue
from dataclasses import dataclass, field
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
from tkinter import ttk

# -------------- DATA MODELS -------------- #

@dataclass
class NodeInfo:
    """Represents a node in the system cave: process, interface, etc."""
    id: str
    label: str
    type: str
    cpu: float = 0.0
    memory: float = 0.0
    extra: Dict = field(default_factory=dict)


@dataclass
class EdgeInfo:
    """Represents a relationship between nodes (talks-to, uses, etc.)."""
    src: str
    dst: str
    type: str
    extra: Dict = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """One snapshot of the system at a moment in time."""
    timestamp: float
    nodes: Dict[str, NodeInfo]
    edges: List[EdgeInfo]


@dataclass
class WorkerStatus:
    """Worker state for display."""
    name: str
    last_run: float
    last_result: str
    active: bool = True
    interval: float = 0.0


@dataclass
class ProcessProfile:
    """
    Learned habit profile for a process *name*:
    - Rolling average CPU and memory percent
    - Count of observations
    """
    name: str
    avg_cpu: float = 0.0
    avg_mem: float = 0.0
    count: int = 0

    def update(self, cpu: float, mem: float):
        self.count += 1
        alpha = 0.1  # exponential-like smoothing
        self.avg_cpu = (1 - alpha) * self.avg_cpu + alpha * cpu
        self.avg_mem = (1 - alpha) * self.avg_mem + alpha * mem


@dataclass
class Strategy:
    """
    An operational strategy: this is the 'self-written code' for how the guardian behaves.
    """
    name: str
    scan_interval_factor: float   # multiplies worker base intervals
    cpu_threshold: float          # per-process anomaly CPU threshold
    mem_threshold: float          # per-process anomaly MEM threshold
    enforcement_aggressiveness: float  # 0..1: willingness to QUARANTINE/KILL


@dataclass
class StrategyResult:
    """Result of simulating a strategy over historical data."""
    strategy: Strategy
    score: float
    notes: str


# -------------- QUEEN BRAIN -------------- #

class QueenBrain:
    """
    Queen: central intelligence.
    - Startup hardware and software audits
    - Spawns and coordinates workers
    - Receives telemetry
    - Builds and updates system map
    - Learns process "habits" over time
    - Runs anomaly heuristics
    - Decides proposed actions (watch/quarantine/kill)
    - Self-evolves operational strategy via simulation
    - Respects system capacity: memory+disk aware, low-memory mode
    - Prioritizes system security + operations via policy config
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.snapshots: List[SystemSnapshot] = []
        self.current_graph = nx.DiGraph()
        self.worker_status: Dict[str, WorkerStatus] = {}
        self.event_log: List[str] = []
        self.telemetry_queue: "queue.Queue[SystemSnapshot]" = queue.Queue()
        self.running = True

        # ---- Enforcement mode ----
        # Start safe: only log actions (DRY RUN).
        self.enforce_actions_for_real = False

        # ---- System capacity ----
        vm = psutil.virtual_memory()
        self.total_mem_bytes = vm.total
        self.total_mem_gb = self.total_mem_bytes / (1024**3)

        # Treat ~50% of total memory as practical soft cap.
        self.mem_cap_fraction = 0.50
        self.mem_threshold = 50.0   # used both as global warn and default per-node mem anomaly
        self.cpu_threshold = 80.0
        self.disk_high_threshold = 90.0

        self.low_memory_mode = False
        self.low_disk_mode = False

        # Snapshot history limits adapt based on capacity
        if self.total_mem_gb <= 4:
            self.max_snapshots_normal = 20
        elif self.total_mem_gb <= 8:
            self.max_snapshots_normal = 40
        else:
            self.max_snapshots_normal = 60

        self.max_snapshots_low_mem = max(10, self.max_snapshots_normal // 2)

        # ---- Global metrics ----
        self.last_global_cpu = 0.0
        self.last_global_mem = 0.0
        self.last_global_disk = 0.0

        # ---- Habit profiles ----
        self.process_profiles: Dict[str, ProcessProfile] = {}

        # ---- Strategy / self-written operational code ----
        self.current_strategy = Strategy(
            name="default",
            scan_interval_factor=1.0,
            cpu_threshold=self.cpu_threshold,
            mem_threshold=self.mem_threshold,
            enforcement_aggressiveness=0.3,  # conservative
        )

        # ---- Security / Ops priority configuration ----
        # Critical processes that should not be killed by default.
        # You can add OS-specific names (e.g., "System", "wininit", etc.)
        self.critical_process_whitelist = set([
            "System",
            "systemd",
            "init",
            "csrss.exe",
            "wininit.exe",
            "winlogon.exe",
            "Hive Guardian - Queen & Workers (Audit + Habits)",
        ])

        # Paranoia level:
        # 0.0 = very cautious (rarely kill, more watch)
        # 1.0 = very aggressive (more willing to kill/quarantine)
        self.paranoia_level = 0.3

        # Security vs Ops weighting in strategy scoring
        # Higher security_weight => more negative weight for heavy anomalies
        self.security_weight = 2.0
        self.ops_weight = 1.0

        # ---- Startup audits ----
        self._run_hardware_audit()
        self._run_software_audit()

        # ---- Main loop ----
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    # -------- Logging and events -------- #

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        with self.lock:
            self.event_log.append(line)
            if len(self.event_log) > 2000:
                self.event_log.pop(0)
        print(line)

    # -------- Startup audits -------- #

    def _run_hardware_audit(self):
        """One-time hardware/system capability scan."""
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
        """One-time software/platform scan: OS info + running processes."""
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

    # -------- Telemetry ingestion -------- #

    def submit_snapshot(self, snapshot: SystemSnapshot):
        self.telemetry_queue.put(snapshot)

    # -------- Global metrics from SystemMetricsWorker -------- #

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

    # -------- Strategy / self-written behavior -------- #

    def _simulate_strategy(self, strategy: Strategy) -> StrategyResult:
        """
        Replay stored snapshots and estimate:
        - how many anomalies (security concerns)
        - how many heavy anomalies
        - a simple scan_cost approximation
        This is the queen "simulating" a new version of its operational code.
        """
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

        # Score: lower is better (negative penalties).
        # Weighted by security_weight and ops_weight.
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
        """
        The queen writes a new 'code path' (strategy), simulates it,
        and if it's better for security+ops, adopts it.
        """
        with self.lock:
            if len(self.snapshots) < 10:
                return  # not enough history yet

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

    # -------- Anomaly classification / action selection -------- #

    def _classify_anomaly(self, node: NodeInfo) -> str:
        """
        Decide action for an anomalous node:
        - WATCH: mild deviation or low aggro
        - QUARANTINE: strong deviation but not extreme, and allowed by paranoia
        - KILL: extreme deviation / runaway, allowed by paranoia
        Security + ops priorities + paranoia influence this.
        """
        name = node.extra.get("name", None)
        if not name:
            if "(" in node.label:
                name = node.label.split("(", 1)[0].strip()
            else:
                name = node.label

        profile = self.process_profiles.get(name)
        cpu = node.cpu
        mem = node.memory

        # Critical whitelist: do not kill them by default
        is_critical = name in self.critical_process_whitelist

        # New or rarely seen processes => WATCH first
        if profile is None or profile.count < 5:
            return "WATCH"

        cpu_ratio = (cpu + 1.0) / (profile.avg_cpu + 1.0)
        mem_ratio = (mem + 1.0) / (profile.avg_mem + 1.0)

        # Base classification ignoring paranoia and critical status
        base_action = "WATCH"
        if cpu_ratio > 6.0 or mem_ratio > 6.0:
            base_action = "KILL"
        elif cpu_ratio > 3.0 or mem_ratio > 3.0:
            base_action = "QUARANTINE"

        # Paranoia moderates aggressiveness
        # Low paranoia => downgrade KILL to QUARANTINE/WATCH
        if self.paranoia_level < 0.2 and base_action == "KILL":
            base_action = "QUARANTINE"
        if self.paranoia_level < 0.05:
            base_action = "WATCH"

        # Strategy aggressiveness further modulates
        if self.current_strategy.enforcement_aggressiveness < 0.2 and base_action == "KILL":
            base_action = "QUARANTINE"
        if self.current_strategy.enforcement_aggressiveness < 0.05:
            base_action = "WATCH"

        # Critical processes: never KILL by default
        if is_critical and base_action == "KILL":
            base_action = "QUARANTINE"

        return base_action

    # -------- Queen processing loop -------- #

    def _run_loop(self):
        self.log(
            f"Queen brain started. Total RAM ~ {self.total_mem_gb:.1f} GB. "
            f"Soft cap at {self.mem_cap_fraction*100:.0f}% usage."
        )
        last_evolve = time.time()
        evolve_interval = 60.0  # every 60s, try to evolve strategy

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
        """Update map, learn habits, and run anomaly logic."""
        # Learn habits
        self._update_habit_profiles_from_snapshot(snapshot)

        # Graph + snapshot history with capacity awareness
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

        # Anomaly detection with current strategy thresholds
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
        """
        Try to enforce action against a process:
        - In DRY RUN mode, only logs.
        - In real mode, tries suspend/terminate.
        """
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
                f"(Set queen.enforce_actions_for_real = True to enable.)"
            )
            return

        # Real enforcement
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

    # -------- GUI / public accessors -------- #

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


# -------------- WORKERS -------------- #

class BaseWorker(threading.Thread):
    """
    Base class for workers.
    Each worker runs in a loop and adapts its interval based on system state + strategy.
    """

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
    """
    Worker that scans processes: PID, CPU, memory, parent-child relations.
    This is our Prometheus-style snapshot of the process cave.
    """

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
    """
    Worker that gathers global system metrics (CPU, memory, disk)
    and feeds them to the queen.
    """

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


# -------------- GUI -------------- #

class HiveGUI:
    """
    Tkinter GUI:
    - Overview (global metrics, caps, habits summary, strategy summary)
    - Workers (status list)
    - Events (log)
    - Map (process graph)
    """

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

        self._schedule_updates()

    # -------- Tabs -------- #

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
            width = 180 if col not in ("last_result",) else 360
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

    # -------- Updates -------- #

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
        lines.append(
            f"Strategy: {self.queen.get_strategy_summary()}"
        )
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

    def run(self):
        self.root.mainloop()


# -------------- MAIN ENTRY -------------- #

def main():
    queen = QueenBrain()

    # If you want to allow real enforcement (QUARANTINE/KILL), set:
    # queen.enforce_actions_for_real = True

    w1 = ProcessScannerWorker(queen, base_interval=5.0)
    w2 = SystemMetricsWorker(queen, base_interval=3.0)
    w1.start()
    w2.start()

    gui = HiveGUI(queen)
    try:
        gui.run()
    finally:
        queen.stop()
        w1.stop()
        w2.stop()


if __name__ == "__main__":
    main()

