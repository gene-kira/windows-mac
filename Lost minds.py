#!/usr/bin/env python3
"""
Borg Guardian Organism (Fully Autonomous, Baseline-Aware) with:

- Queen + workers (Borg-style hive)
- Missions, mission stability, presidential mode
- Automatic system boost (5–10s) under congestion
- Reasoning agent (multi-objective, feedback-ready)
- Anomaly watcher that can trigger presidential mode
- System/network scanner
- Ad redirector (TCP proxy)
- Redundant memory organ (primary + local backup + SMB backup)
- System + software inventory at first run
- Identity awareness (user, machine, OS, location)
- Baseline inventory from FIRST successful scan used as normal reference
- Tkinter GUI dashboard for live control/monitoring

Standard library only.
"""

import threading
import time
import queue
import random
import socket
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple

import tkinter as tk
from tkinter import ttk, scrolledtext

# ============================================================
# 0. Configuration: memory paths (edit these to match your system)
# ============================================================

# Primary local memory file
PRIMARY_MEMORY_PATH = os.path.join(os.getcwd(), "agent_memory_primary.json")

# Local backup memory (different folder/drive if possible)
LOCAL_BACKUP_MEMORY_PATH = os.path.join(os.getcwd(), "agent_memory_local_backup.json")

# Network backup memory (Windows SMB UNC path example; change to your real share)
# Example: r"\\MY-SERVER\BorgShare\agent_memory_network_backup.json"
NETWORK_BACKUP_MEMORY_PATH = r"\\MY-SERVER\BorgShare\agent_memory_network_backup.json"


# ============================================================
# 1. Memory Organ: primary + local backup + network SMB backup
# ============================================================

class MemoryOrgan:
    """
    Handles redundant memory:
    - Primary local file
    - Local backup file
    - Network SMB backup file

    Writes:
        primary -> local backup -> SMB backup
    Reads:
        primary first, then local backup, then network backup.
    """

    def __init__(
        self,
        primary_path: str,
        local_backup_path: str,
        network_backup_path: str,
    ):
        self.primary_path = primary_path
        self.local_backup_path = local_backup_path
        self.network_backup_path = network_backup_path

        self.lock = threading.Lock()
        self.last_data: Dict[str, Any] = {}
        self.last_save_ts: float = 0.0
        self.last_errors: Dict[str, Optional[str]] = {
            "primary": None,
            "local_backup": None,
            "network_backup": None,
        }

        self._ensure_dirs()
        self._load_any_existing()

    def _ensure_dirs(self):
        for path in (self.primary_path, self.local_backup_path, self.network_backup_path):
            if not path:
                continue
            folder = os.path.dirname(path)
            if folder and not os.path.exists(folder) and not folder.startswith("\\\\"):
                try:
                    os.makedirs(folder, exist_ok=True)
                except Exception:
                    pass

    def _load_any_existing(self):
        """
        On startup, attempt to load from primary, then local backup, then network backup.
        """
        with self.lock:
            for label, path in [
                ("primary", self.primary_path),
                ("local_backup", self.local_backup_path),
                ("network_backup", self.network_backup_path),
            ]:
                if not path:
                    continue
                try:
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            self.last_data = json.load(f)
                            self.last_errors[label] = None
                            return
                except Exception as e:
                    self.last_errors[label] = str(e)

            # Fresh start – no memory yet
            self.last_data = {
                "decisions": [],
                "baseline_inventory": None,
                "identity": None,
                "baseline_locked": False,  # once True, baseline will NOT be overwritten
            }
            self.last_save_ts = time.time()

    def read(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.last_data))

    def write(self, data: Dict[str, Any]) -> None:
        """
        Save to primary, then local backup, then SMB backup.
        Failures are recorded but not fatal.
        """
        with self.lock:
            self.last_data = data
            self.last_save_ts = time.time()

            # Primary
            try:
                with open(self.primary_path, "w", encoding="utf-8") as f:
                    json.dump(self.last_data, f, indent=2)
                self.last_errors["primary"] = None
            except Exception as e:
                self.last_errors["primary"] = str(e)

            # Local backup
            try:
                with open(self.local_backup_path, "w", encoding="utf-8") as f:
                    json.dump(self.last_data, f, indent=2)
                self.last_errors["local_backup"] = None
            except Exception as e:
                self.last_errors["local_backup"] = str(e)

            # SMB backup (UNC)
            try:
                with open(self.network_backup_path, "w", encoding="utf-8") as f:
                    json.dump(self.last_data, f, indent=2)
                self.last_errors["network_backup"] = None
            except Exception as e:
                self.last_errors["network_backup"] = str(e)

    def health_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "primary_path": self.primary_path,
                "local_backup_path": self.local_backup_path,
                "network_backup_path": self.network_backup_path,
                "last_save_ts": self.last_save_ts,
                "errors": self.last_errors.copy(),
            }


# ============================================================
# 2. Identity + inventory
# ============================================================

def collect_identity() -> Dict[str, Any]:
    """
    Who am I, where am I, what am I running on?
    """
    identity = {
        "user": os.getenv("USERNAME") or os.getenv("USER") or "unknown",
        "home": os.path.expanduser("~"),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    return identity


def collect_software_inventory() -> Dict[str, Any]:
    """
    Basic software/system inventory:
    - PATH entries
    - Program Files dirs (on Windows)
    - Running directory listing
    This is a skeleton; you can extend with registry or package managers later.
    """
    inv: Dict[str, Any] = {}
    inv["timestamp"] = time.time()
    inv["path_entries"] = (os.getenv("PATH") or "").split(os.pathsep)
    inv["cwd"] = os.getcwd()
    inv["cwd_files"] = []
    try:
        inv["cwd_files"] = sorted(os.listdir(os.getcwd()))
    except Exception as e:
        inv["cwd_error"] = str(e)

    if os.name == "nt":
        pf = os.getenv("ProgramFiles") or "C:\\Program Files"
        pf86 = os.getenv("ProgramFiles(x86)") or "C:\\Program Files (x86)"
        inv["program_files_dirs"] = [pf, pf86]
        inv["program_files_contents"] = {}
        for root in [pf, pf86]:
            try:
                if os.path.exists(root):
                    inv["program_files_contents"][root] = sorted(os.listdir(root))[:200]
            except Exception as e:
                inv["program_files_contents"][root] = f"error: {e}"
    else:
        inv["program_files_dirs"] = []
        inv["program_files_contents"] = {}
    return inv


def compare_inventories(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple diff between baseline and current inventories.
    Used for anomaly signals.
    """
    anomalies: Dict[str, Any] = {"added_programs": [], "removed_programs": []}
    if not baseline:
        anomalies["note"] = "No baseline; cannot compare yet."
        return anomalies

    base_pf = baseline.get("program_files_contents", {})
    cur_pf = current.get("program_files_contents", {})
    for root in base_pf:
        if isinstance(base_pf.get(root), list) and isinstance(cur_pf.get(root), list):
            base_set = set(base_pf[root])
            cur_set = set(cur_pf[root])
            added = sorted(cur_set - base_set)
            removed = sorted(base_set - cur_set)
            if added:
                anomalies["added_programs"].append({root: added})
            if removed:
                anomalies["removed_programs"].append({root: removed})

    return anomalies


# ============================================================
# 3. Core data types: Missions, stability, state
# ============================================================

@dataclass
class Mission:
    id: str
    description: str
    priority: int
    mission_type: str = "generic"   # e.g. system_guard, user_request, security, ad_redirect, os_protect, scan
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionStabilityRules:
    max_cpu_percent: float = 80.0
    max_worker_count: int = 16
    must_keep_responsive: bool = True
    allow_aggressive_mode: bool = True


@dataclass
class MissionState:
    active_mission: Optional[Mission] = None
    override_active: bool = False
    override_reason: str = ""
    last_switch_ts: float = field(default_factory=time.time)
    presidential_mode: bool = False
    presidential_reason: str = ""


# ============================================================
# 4. Resource monitor (stub CPU load)
# ============================================================

class ResourceMonitor:
    """
    Simple synthetic CPU load model (replace with real metrics later).
    """
    def __init__(self):
        self._fake_cpu = 10.0

    def read_cpu_percent(self) -> float:
        self._fake_cpu += random.uniform(-3, 3)
        self._fake_cpu = max(0.0, min(100.0, self._fake_cpu))
        return self._fake_cpu

    def snapshot(self) -> Dict[str, float]:
        return {"cpu_percent": self.read_cpu_percent()}


# ============================================================
# 5. Reasoning agent (multi-objective, memory-backed)
# ============================================================

@dataclass
class ReasoningOption:
    id: str
    description: str
    self_safety: float
    others_safety: float
    effort_cost: float
    legality: float
    long_term_benefit: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningSituation:
    id: str
    description: str
    options: List[ReasoningOption]
    objective_weights: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningDecision:
    best_option: ReasoningOption
    option_scores: Dict[str, float]
    side_missions: List[str]
    mode_used: str
    prediction_note: str
    confidence: float
    confidence_rationale: str


class SimpleMemory:
    """
    Thin wrapper around MemoryOrgan for reasoning decisions.
    """

    def __init__(self, organ: MemoryOrgan):
        self.organ = organ
        data = self.organ.read()
        if "decisions" not in data:
            data["decisions"] = []
            self.organ.write(data)

    def _get_data(self) -> Dict[str, Any]:
        data = self.organ.read()
        if "decisions" not in data:
            data["decisions"] = []
        return data

    def record_decision(self, meta_type: str, success: bool, mode: str):
        data = self._get_data()
        data.setdefault("decisions", []).append({
            "meta_type": meta_type,
            "success": 1.0 if success else 0.0,
            "mode": mode,
            "ts": time.time(),
        })
        if len(data["decisions"]) > 2000:
            data["decisions"] = data["decisions"][-1000:]
        self.organ.write(data)

    def summarize_success(self) -> Dict[str, float]:
        data = self._get_data()
        stats: Dict[str, Dict[str, float]] = {}
        for d in data.get("decisions", []):
            mt = d.get("meta_type", "unknown")
            s = float(d.get("success", 0.0))
            info = stats.setdefault(mt, {"count": 0.0, "sum": 0.0})
            info["count"] += 1.0
            info["sum"] += s
        out: Dict[str, float] = {}
        for k, v in stats.items():
            if v["count"] > 0:
                out[k] = v["sum"] / v["count"]
        return out


class ReasoningAgent:
    def __init__(self, memory: SimpleMemory):
        self.memory = memory
        self.mode = "balanced"
        self.default_objectives = {
            "self_safety": 0.4,
            "others_safety": 0.4,
            "effort_cost": 0.1,
            "legality": 0.1,
            "long_term_benefit": 0.0,
        }

    def _effective_weights(self, situation: ReasoningSituation) -> Dict[str, float]:
        w = self.default_objectives.copy()
        w.update(situation.objective_weights)
        if self.mode == "cautious":
            w["self_safety"] = w.get("self_safety", 0.4) + 0.1
        elif self.mode == "altruistic":
            w["others_safety"] = w.get("others_safety", 0.4) + 0.1
        total = sum(abs(v) for v in w.values()) or 1.0
        for k in list(w.keys()):
            w[k] = w[k] / total
        return w

    def _score_option(self, opt: ReasoningOption, w: Dict[str, float], bias: float) -> float:
        components = {
            "self_safety": opt.self_safety,
            "others_safety": opt.others_safety,
            "effort_cost": -abs(opt.effort_cost),
            "legality": opt.legality,
            "long_term_benefit": opt.long_term_benefit,
        }
        score = 0.0
        for k, weight in w.items():
            if k in components:
                score += weight * components[k]
        score += 0.1 * bias
        return score

    def _predictive_bias(self) -> Dict[str, float]:
        stats = self.memory.summarize_success()
        for k in stats:
            stats[k] = max(0.0, min(1.0, stats[k]))
        return stats

    def _compute_confidence(self, scores: Dict[str, float]) -> Tuple[float, str]:
        if not scores:
            return 0.0, "No options."
        vals = sorted(scores.values(), reverse=True)
        best = vals[0]
        second = vals[1] if len(vals) > 1 else best
        margin = best - second
        margin_norm = max(0.0, min(1.0, abs(margin)))

        import math
        exps = [math.exp(v) for v in scores.values()]
        total = sum(exps) or 1.0
        probs = [e / total for e in exps]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 1.0
        sharpness = 1.0 - entropy_norm
        conf = 0.6 * margin_norm + 0.4 * sharpness
        conf = max(0.0, min(1.0, conf))
        rationale = f"margin={margin:.3f}, sharpness={sharpness:.3f}, confidence={conf:.3f}"
        return conf, rationale

    def _side_missions(self, situation: ReasoningSituation, best: ReasoningOption, confidence: float) -> List[str]:
        missions = []
        desc = situation.description.lower()
        ctx = situation.context
        time_of_day = ctx.get("time_of_day", "unknown")
        weather = ctx.get("weather", "unknown")
        crowd = ctx.get("crowd_density", "unknown")

        if "pothole" in desc or "hazard" in desc or "hole" in desc:
            missions.append("Notify local authorities about the hazard.")
            missions.append("Warn nearby people if safe.")
            if time_of_day == "night":
                missions.append("Visibility is low; keep extra distance.")
            if weather in ("rainy", "snowy", "icy"):
                missions.append("Surface may be slippery; widen your path.")
            if crowd in ("medium", "high"):
                missions.append("Guide others around the hazard if possible.")

        if best.others_safety < 0.5:
            missions.append("Consider extra actions to increase safety for others.")
        if best.long_term_benefit < 0.5:
            missions.append("Think about longer-term fixes to prevent this in future.")

        if confidence < 0.4:
            missions.append("Confidence is low; consider reevaluating or seeking input.")
        elif confidence > 0.8:
            missions.append("Confidence is high; still stay alert for changes.")

        missions.append("Reflect whether another mode (cautious/balanced/altruistic) would choose differently.")

        out = []
        seen = set()
        for m in missions:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def deliberate(self, situation: ReasoningSituation) -> ReasoningDecision:
        weights = self._effective_weights(situation)
        type_bias = self._predictive_bias()
        scores: Dict[str, float] = {}
        for opt in situation.options:
            meta_type = opt.meta.get("type", "unknown")
            bias = type_bias.get(meta_type, 0.5) - 0.5
            scores[opt.id] = self._score_option(opt, weights, bias)

        best_id = max(scores, key=scores.get)
        best_opt = next(o for o in situation.options if o.id == best_id)
        conf, rationale = self._compute_confidence(scores)
        side = self._side_missions(situation, best_opt, conf)
        return ReasoningDecision(
            best_option=best_opt,
            option_scores=scores,
            side_missions=side,
            mode_used=self.mode,
            prediction_note="Using historic success where available.",
            confidence=conf,
            confidence_rationale=rationale,
        )


# ============================================================
# 6. Worker + status
# ============================================================

@dataclass
class WorkerStatus:
    id: str
    alive: bool
    current_mission_id: Optional[str]
    last_heartbeat: float
    last_result: Optional[Any] = None
    error: Optional[str] = None
    load: float = 0.0


class Worker(threading.Thread):
    def __init__(
        self,
        worker_id: str,
        task_handler: Callable[[Mission], Any],
        status_callback: Callable[[WorkerStatus], None],
        stop_event: threading.Event,
        idle_sleep: float = 0.5,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_handler = task_handler
        self.status_callback = status_callback
        self.stop_event = stop_event
        self.tasks = queue.Queue()
        self.status = WorkerStatus(
            id=worker_id,
            alive=True,
            current_mission_id=None,
            last_heartbeat=time.time(),
        )
        self.idle_sleep = idle_sleep  # will be shortened during boost

    def assign_mission(self, mission: Mission) -> None:
        self.tasks.put(mission)

    def run(self):
        while not self.stop_event.is_set():
            try:
                mission: Mission = self.tasks.get(timeout=self.idle_sleep)
            except queue.Empty:
                self.status.current_mission_id = None
                self.status.load = max(0.0, self.status.load - 0.05)
                self._heartbeat()
                continue

            self.status.current_mission_id = mission.id
            self.status.load = min(1.0, self.status.load + 0.3)
            self._heartbeat()

            try:
                result = self.task_handler(mission)
                self.status.last_result = result
                self.status.error = None
            except Exception as e:
                self.status.error = str(e)
                self.status.last_result = None
            finally:
                self.status.load = max(0.0, self.status.load - 0.1)
                self._heartbeat()
                self.tasks.task_done()

        self.status.alive = False
        self._heartbeat()

    def _heartbeat(self):
        self.status.last_heartbeat = time.time()
        self.status_callback(self.status)


# ============================================================
# 7. Ad redirector (TCP proxy)
# ============================================================

class AdRedirector(threading.Thread):
    def __init__(self, listen_port: int, sink_host: str, sink_port: int, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.listen_port = listen_port
        self.sink_host = sink_host
        self.sink_port = sink_port
        self.stop_event = stop_event
        self.server_socket: Optional[socket.socket] = None

    def run(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("0.0.0.0", self.listen_port))
            self.server_socket.listen(5)
            print(f"[AD-REDIRECTOR] Listening on {self.listen_port} -> {self.sink_host}:{self.sink_port}")
        except Exception as e:
            print(f"[AD-REDIRECTOR] Failed to start: {e}")
            return

        while not self.stop_event.is_set():
            try:
                self.server_socket.settimeout(1.0)
                client_sock, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            threading.Thread(
                target=self._handle_connection, args=(client_sock, addr), daemon=True
            ).start()

        if self.server_socket:
            self.server_socket.close()

    def _handle_connection(self, client_sock: socket.socket, addr):
        print(f"[AD-REDIRECTOR] Connection from {addr}, redirecting...")
        try:
            sink_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sink_sock.connect((self.sink_host, self.sink_port))
        except Exception as e:
            print(f"[AD-REDIRECTOR] Failed to connect to sink: {e}")
            client_sock.close()
            return

        def forward(src, dst):
            try:
                while True:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
            except Exception:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                dst.close()

        threading.Thread(target=forward, args=(client_sock, sink_sock), daemon=True).start()
        threading.Thread(target=forward, args=(sink_sock, client_sock), daemon=True).start()


# ============================================================
# 8. System/network scanner
# ============================================================

class SystemScanner:
    """
    Very simple scanner:
    - Reports local hostname and IPs
    - Optional basic /24 sweep (ping-like TCP connect on one port)
    """

    def __init__(self):
        self.last_scan_result: Dict[str, Any] = {}

    def scan_local_system(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            hostname = socket.gethostname()
            info["hostname"] = hostname
            info["local_ips"] = list({addr[4][0] for addr in socket.getaddrinfo(hostname, None)})
        except Exception as e:
            info["error"] = str(e)
        self.last_scan_result = info
        return info

    def scan_subnet_stub(self, base_ip: str, port: int = 80, timeout: float = 0.1, max_hosts: int = 32) -> List[str]:
        alive: List[str] = []
        parts = base_ip.split(".")
        if len(parts) != 4:
            return alive
        prefix = ".".join(parts[:3])
        for last in range(1, max_hosts + 1):
            ip = f"{prefix}.{last}"
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            try:
                s.connect((ip, port))
                alive.append(ip)
            except Exception:
                pass
            finally:
                s.close()
        return alive


# ============================================================
# 9. Queen with presidential mode + automatic system boost + baseline
# ============================================================

class Queen:
    def __init__(
        self,
        stability_rules: MissionStabilityRules,
        monitor: ResourceMonitor,
        reasoning: ReasoningAgent,
        memory_organ: MemoryOrgan,
        identity: Dict[str, Any],
        baseline_inventory: Dict[str, Any],
    ):
        self.stability_rules = stability_rules
        self.monitor = monitor
        self.reasoning = reasoning
        self.memory_organ = memory_organ

        self.identity = identity
        self.baseline_inventory = baseline_inventory  # baseline from first successful scan

        self.mission_state = MissionState()
        self.workers: Dict[str, Worker] = {}
        self.worker_status: Dict[str, WorkerStatus] = {}
        self.stop_event = threading.Event()
        self._worker_counter = 0

        self.global_mission_queue: queue.Queue = queue.Queue()
        self.thinking_thread = threading.Thread(target=self._thinking_loop, daemon=True)

        self.ad_redirector: Optional[AdRedirector] = None
        self.scanner = SystemScanner()

        self.current_cpu_percent: float = 0.0

        # Boost mode state
        self.boost_active: bool = False
        self.boost_end_ts: float = 0.0
        self.boost_cooldown_until: float = 0.0

        # Congestion tracking
        self._recent_queue_lengths: List[int] = []

        self.gui_log_callback: Optional[Callable[[str], None]] = None

        # Persist identity + baseline into memory (baseline is locked)
        self._persist_identity_and_baseline()

    # ---------- Logging helper ----------

    def _log(self, msg: str):
        print(msg)
        if self.gui_log_callback:
            self.gui_log_callback(msg)

    def _persist_identity_and_baseline(self):
        data = self.memory_organ.read()
        # If baseline_locked is already True, do NOT overwrite baseline_inventory
        if not data.get("baseline_locked", False):
            data["baseline_inventory"] = self.baseline_inventory
            data["baseline_locked"] = True
            self._log("[QUEEN] Baseline inventory locked as reference for normal operations.")
        if not data.get("identity"):
            data["identity"] = self.identity
        self.memory_organ.write(data)

    # ---------- Presidential mode ----------

    def enter_presidential_mode(self, reason: str) -> None:
        self.mission_state.presidential_mode = True
        self.mission_state.presidential_reason = reason
        self.mission_state.override_active = True
        self.mission_state.override_reason = f"PRESIDENTIAL: {reason}"
        self.mission_state.last_switch_ts = time.time()
        self._log(f"[QUEEN] PRESIDENTIAL MODE ACTIVATED: {reason}")
        self._filter_missions_for_presidential_mode()

    def exit_presidential_mode(self) -> None:
        self.mission_state.presidential_mode = False
        self.mission_state.presidential_reason = ""
        self.mission_state.override_active = False
        self.mission_state.override_reason = ""
        self.mission_state.last_switch_ts = time.time()
        self._log("[QUEEN] Presidential mode deactivated.")

    def _filter_missions_for_presidential_mode(self):
        if not self.mission_state.presidential_mode:
            return
        new_q = queue.Queue()
        while not self.global_mission_queue.empty():
            m: Mission = self.global_mission_queue.get()
            if m.mission_type in ("system_guard", "security", "ad_redirect", "os_protect", "scan"):
                new_q.put(m)
        self.global_mission_queue = new_q

    # ---------- Boost mode ----------

    def _maybe_trigger_boost(self):
        now = time.time()
        if self.boost_active:
            return
        if now < self.boost_cooldown_until:
            return

        queue_len = self.global_mission_queue.qsize()
        self._recent_queue_lengths.append(queue_len)
        if len(self._recent_queue_lengths) > 20:
            self._recent_queue_lengths = self._recent_queue_lengths[-20:]
        avg_len = sum(self._recent_queue_lengths) / max(1, len(self._recent_queue_lengths))

        avg_worker_load = 0.0
        if self.worker_status:
            avg_worker_load = sum(w.load for w in self.worker_status.values()) / len(self.worker_status)

        congestion_score = 0.0
        if queue_len > 0:
            congestion_score += min(1.0, queue_len / 20.0)
        congestion_score += min(1.0, avg_len / 20.0) * 0.5
        congestion_score += avg_worker_load * 0.5

        if congestion_score >= 0.7:
            self._start_boost(congestion_score)

    def _start_boost(self, severity: float):
        now = time.time()
        duration = 5.0 + (severity * 5.0)  # 5 to 10 seconds
        self.boost_active = True
        self.boost_end_ts = now + duration
        self.boost_cooldown_until = now + duration + 10.0
        self._log(f"[QUEEN] BOOST MODE ACTIVATED for {duration:.1f}s (severity={severity:.2f})")

        self.reasoning.mode = "cautious" if self.mission_state.presidential_mode else "balanced"

        while len(self.workers) < min(self.stability_rules.max_worker_count, len(self.workers) + 2):
            self._spawn_worker()

        for w in self.workers.values():
            w.idle_sleep = 0.1

    def _update_boost_state(self):
        if not self.boost_active:
            return
        now = time.time()
        if now >= self.boost_end_ts:
            self.boost_active = False
            self._log("[QUEEN] BOOST MODE ENDED")
            for w in self.workers.values():
                w.idle_sleep = 0.5
            if not self.mission_state.presidential_mode:
                self.reasoning.mode = "balanced"

    # ---------- Mission control ----------

    def set_active_mission(self, mission: Mission) -> None:
        self.mission_state.active_mission = mission
        self.mission_state.last_switch_ts = time.time()
        self._log(f"[QUEEN] Active mission set: {mission.id} ({mission.description})")

    def queue_mission(self, mission: Mission) -> None:
        if self.mission_state.presidential_mode:
            if mission.mission_type not in ("system_guard", "security", "ad_redirect", "os_protect", "scan"):
                self._log(f"[QUEEN] Presidential mode: ignoring mission {mission.id} ({mission.mission_type})")
                return
        self.global_mission_queue.put(mission)

    # ---------- Worker management ----------

    def _spawn_worker(self) -> Worker:
        self._worker_counter += 1
        wid = f"worker-{self._worker_counter}"
        worker = Worker(
            worker_id=wid,
            task_handler=self._handle_worker_task,
            status_callback=self._update_worker_status,
            stop_event=self.stop_event,
            idle_sleep=0.5,
        )
        self.workers[wid] = worker
        worker.start()
        self._log(f"[QUEEN] Spawned {wid}")
        return worker

    def _update_worker_status(self, status: WorkerStatus) -> None:
        self.worker_status[status.id] = status

    # ---------- Worker brain ----------

    def _handle_worker_task(self, mission: Mission) -> Any:
        self._log(f"[{threading.current_thread().name}] Mission {mission.id} ({mission.mission_type})")
        base_sleep = random.uniform(0.05, 0.3)
        if self.boost_active:
            base_sleep *= 0.5
        time.sleep(base_sleep)

        if mission.mission_type in ("system_guard", "os_protect"):
            return {"guard": "ok"}

        if mission.mission_type == "security":
            current_inv = collect_software_inventory()
            anomalies = compare_inventories(self.baseline_inventory, current_inv)
            return {"security_scan": "ok", "anomalies": anomalies}

        if mission.mission_type == "scan":
            mode = mission.params.get("mode", "local")
            if mode == "local":
                result = self.scanner.scan_local_system()
                self.scanner.last_scan_result = result
                return result
            elif mode == "subnet":
                base_ip = mission.params.get("base_ip", "192.168.1.1")
                alive = self.scanner.scan_subnet_stub(base_ip)
                result = {"alive_hosts": alive}
                self.scanner.last_scan_result = result
                return result

        if mission.mission_type == "ad_redirect":
            return {"ad_redirect": "active"}

        if mission.mission_type == "user_request":
            if self.mission_state.presidential_mode:
                return {"denied": "presidential_mode"}
            else:
                situation = ReasoningSituation(
                    id="pothole_example",
                    description="You see a pothole while walking.",
                    options=[
                        ReasoningOption("A", "Go around it.", 0.9, 0.4, 0.2, 1.0, 0.2, {"type": "avoid"}),
                        ReasoningOption("B", "Cover it and walk over.", 0.8, 0.9, 0.6, 0.8, 0.5, {"type": "mitigate"}),
                        ReasoningOption("C", "Jump over it.", 0.4, 0.1, 0.3, 1.0, 0.1, {"type": "risky"}),
                    ],
                    objective_weights={
                        "self_safety": 0.4,
                        "others_safety": 0.4,
                        "effort_cost": 0.1,
                        "legality": 0.05,
                        "long_term_benefit": 0.05,
                    },
                    context={"time_of_day": "day", "weather": "clear", "crowd_density": "low"},
                )
                decision = self.reasoning.deliberate(situation)
                meta_type = decision.best_option.meta.get("type", "unknown")
                self.reasoning.memory.record_decision(meta_type, True, self.reasoning.mode)
                return {
                    "best_option": decision.best_option.description,
                    "confidence": decision.confidence,
                    "side_missions": decision.side_missions,
                }

        return {"status": "unknown_mission_type"}

    # ---------- Thinking loop ----------

    def _thinking_loop(self) -> None:
        while not self.stop_event.is_set():
            snap = self.monitor.snapshot()
            self.current_cpu_percent = snap["cpu_percent"]

            self._update_boost_state()
            self._maybe_trigger_boost()

            if self.current_cpu_percent > self.stability_rules.max_cpu_percent and not self.boost_active:
                self._log(f"[QUEEN] CPU high ({self.current_cpu_percent:.1f}%). Throttling.")

            self._maybe_scale_workers()
            self._assign_missions()

            time.sleep(0.5 if not self.boost_active else 0.25)

    def _maybe_scale_workers(self):
        if not self.global_mission_queue.empty():
            target_workers = self.stability_rules.max_worker_count
            if len(self.workers) < target_workers:
                self._spawn_worker()

    def _assign_missions(self):
        if self.global_mission_queue.empty():
            return
        idle_workers = [w for w in self.workers.values()
                        if w.status.current_mission_id is None]
        for w in idle_workers:
            if self.global_mission_queue.empty():
                break
            m = self.global_mission_queue.get_nowait()
            w.assign_mission(m)

    # ---------- Public control ----------

    def start(self):
        self._log("[QUEEN] Starting thinking loop.")
        self.thinking_thread.start()

    def stop(self):
        self._log("[QUEEN] Stopping organism.")
        self.stop_event.set()
        for w in self.workers.values():
            w.join(timeout=1.0)
        self.thinking_thread.join(timeout=1.0)

    def start_ad_redirector(self, listen_port: int, sink_host: str, sink_port: int):
        if self.ad_redirector:
            self._log("[QUEEN] Ad redirector already running.")
        else:
            self.ad_redirector = AdRedirector(listen_port, sink_host, sink_port, self.stop_event)
            self.ad_redirector.start()
            self._log(f"[QUEEN] Ad redirector started on {listen_port} -> {sink_host}:{sink_port}")


# ============================================================
# 10. Anomaly watcher (inventory-based + synthetic)
# ============================================================

class AnomalyWatcher(threading.Thread):
    """
    Uses:
    - Synthetic anomaly score
    - Inventory drift from baseline
    to decide when to trigger presidential mode.
    """

    def __init__(self, queen: Queen, memory_organ: MemoryOrgan):
        super().__init__(daemon=True)
        self.queen = queen
        self.memory_organ = memory_organ
        self.stop_event = queen.stop_event
        self.anomaly_score = 0.0

    def run(self):
        self.queen._log("[ANOMALY] Watcher started.")
        while not self.stop_event.is_set():
            # Synthetic drift
            self.anomaly_score += random.uniform(-0.15, 0.25)
            self.anomaly_score = max(0.0, min(1.0, self.anomaly_score))

            # Inventory drift check relative to locked baseline
            current_inv = collect_software_inventory()
            baseline = self.queen.baseline_inventory
            anomalies = compare_inventories(baseline, current_inv)
            added = sum(len(x.get(list(x.keys())[0], [])) for x in anomalies.get("added_programs", []))
            removed = sum(len(x.get(list(x.keys())[0], [])) for x in anomalies.get("removed_programs", []))
            if added + removed > 20:
                self.anomaly_score = min(1.0, self.anomaly_score + 0.4)
                self.queen._log("[ANOMALY] Significant inventory change detected vs baseline.")

            if self.anomaly_score > 0.9 and not self.queen.mission_state.presidential_mode:
                self.queen.enter_presidential_mode("Unusual activity / inventory anomaly detected.")

            time.sleep(2.0)
        self.queen._log("[ANOMALY] Watcher stopped.")


# ============================================================
# 11. Tkinter GUI (dashboard)
# ============================================================

class BorgGUI:
    def __init__(self, root: tk.Tk, queen: Queen, anomaly: AnomalyWatcher, memory_organ: MemoryOrgan):
        self.root = root
        self.queen = queen
        self.anomaly = anomaly
        self.memory_organ = memory_organ

        self.root.title("Borg Guardian Organism")
        self.root.geometry("1150x700")

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        # Top frame: mode, CPU, presidential, boost, memory, identity
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=5, pady=5)

        self.cpu_var = tk.StringVar(value="CPU: 0%")
        self.mode_var = tk.StringVar(value="Mode: Normal")
        self.pres_var = tk.StringVar(value="Presidential: OFF")
        self.boost_var = tk.StringVar(value="Boost: OFF")
        self.mem_var = tk.StringVar(value="Memory: OK")
        self.id_var = tk.StringVar(value="Identity: unknown@unknown")

        ttk.Label(top, textvariable=self.cpu_var, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Label(top, textvariable=self.mode_var, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Label(top, textvariable=self.pres_var, width=28).pack(side=tk.LEFT, padx=5)
        ttk.Label(top, textvariable=self.boost_var, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Label(top, textvariable=self.mem_var, width=50).pack(side=tk.LEFT, padx=5)

        id_frame = ttk.Frame(self.root)
        id_frame.pack(fill=tk.X, padx=5)
        ttk.Label(id_frame, textvariable=self.id_var).pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side=tk.RIGHT)

        ttk.Button(btn_frame, text="Enter Presidential", command=self._enter_pres).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Exit Presidential", command=self._exit_pres).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Scan Local", command=self._scan_local).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Scan Subnet", command=self._scan_subnet).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="User Request", command=self._user_request).pack(side=tk.LEFT, padx=2)

        # Middle: workers + scan results + log
        mid = ttk.Frame(self.root)
        mid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Workers tree
        worker_frame = ttk.LabelFrame(mid, text="Workers")
        worker_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.worker_tree = ttk.Treeview(worker_frame, columns=("status", "mission", "load"), show="headings")
        self.worker_tree.heading("status", text="Status")
        self.worker_tree.heading("mission", text="Current Mission")
        self.worker_tree.heading("load", text="Load")
        self.worker_tree.column("status", width=120)
        self.worker_tree.column("mission", width=220)
        self.worker_tree.column("load", width=80)
        self.worker_tree.pack(fill=tk.BOTH, expand=True)

        # Right side: scan results + log
        right_frame = ttk.Frame(mid)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scan_frame = ttk.LabelFrame(right_frame, text="Scan Results / Inventory")
        scan_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.scan_text = scrolledtext.ScrolledText(scan_frame, height=10)
        self.scan_text.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(right_frame, text="Event Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Bind queen logging
        self.queen.gui_log_callback = self._append_log

        # Identity display
        ident = self.queen.identity
        self.id_var.set(f"Identity: {ident.get('user','?')}@{ident.get('hostname','?')} "
                        f"({ident.get('os_system','?')} {ident.get('os_release','?')})")

    def _append_log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _enter_pres(self):
        self.queen.enter_presidential_mode("Manual operator command.")

    def _exit_pres(self):
        self.queen.exit_presidential_mode()

    def _scan_local(self):
        m = Mission(
            id=f"scan-local-{int(time.time())}",
            description="Scan local system",
            priority=8,
            mission_type="scan",
            params={"mode": "local"},
        )
        self.queen.queue_mission(m)
        self._append_log(f"[GUI] Queued local scan {m.id}")

    def _scan_subnet(self):
        base_ip = "192.168.1.1"  # adjust as needed
        m = Mission(
            id=f"scan-subnet-{int(time.time())}",
            description="Scan subnet",
            priority=7,
            mission_type="scan",
            params={"mode": "subnet", "base_ip": base_ip},
        )
        self.queen.queue_mission(m)
        self._append_log(f"[GUI] Queued subnet scan {m.id} base {base_ip}")

    def _user_request(self):
        m = Mission(
            id=f"user-{int(time.time())}",
            description="User reasoning request",
            priority=5,
            mission_type="user_request",
        )
        self.queen.queue_mission(m)
        self._append_log(f"[GUI] Queued user request {m.id}")

    def _schedule_update(self):
        self._update_status()
        self.root.after(500, self._schedule_update)

    def _update_status(self):
        cpu = self.queen.current_cpu_percent
        self.cpu_var.set(f"CPU: {cpu:.1f}%")

        if self.queen.mission_state.presidential_mode:
            self.mode_var.set("Mode: PRESIDENTIAL")
            self.pres_var.set(f"Presidential: ON ({self.queen.mission_state.presidential_reason})")
        else:
            self.mode_var.set("Mode: Normal")
            self.pres_var.set("Presidential: OFF")

        if self.queen.boost_active:
            remaining = max(0.0, self.queen.boost_end_ts - time.time())
            self.boost_var.set(f"Boost: ON ({remaining:.1f}s)")
        else:
            self.boost_var.set("Boost: OFF")

        mem_health = self.memory_organ.health_snapshot()
        errs = mem_health["errors"]
        if any(errs.values()):
            self.mem_var.set(
                f"Memory: issues (P:{errs['primary'] or 'OK'}, "
                f"L:{errs['local_backup'] or 'OK'}, "
                f"N:{errs['network_backup'] or 'OK'})"
            )
        else:
            self.mem_var.set("Memory: OK (primary+local+network)")

        for item in self.worker_tree.get_children():
            self.worker_tree.delete(item)

        for wid, status in self.queen.worker_status.items():
            st = "Alive" if status.alive else "Dead"
            mission = status.current_mission_id or "-"
            load = f"{status.load:.2f}"
            self.worker_tree.insert("", tk.END, values=(st, mission, load))

        if self.queen.scanner.last_scan_result:
            self.scan_text.delete("1.0", tk.END)
            self.scan_text.insert(tk.END, json.dumps(self.queen.scanner.last_scan_result, indent=2))


# ============================================================
# 12. Main: inventory, wire everything, launch GUI
# ============================================================

def main():
    # Build memory organ first
    memory_organ = MemoryOrgan(
        primary_path=PRIMARY_MEMORY_PATH,
        local_backup_path=LOCAL_BACKUP_MEMORY_PATH,
        network_backup_path=NETWORK_BACKUP_MEMORY_PATH,
    )

    # Identity + inventory on startup
    identity = collect_identity()
    current_inventory = collect_software_inventory()

    # Load memory and handle baseline logic:
    data = memory_organ.read()
    baseline = data.get("baseline_inventory")
    baseline_locked = data.get("baseline_locked", False)

    if baseline is None or not baseline_locked:
        # First run (or baseline not locked): use current inventory as baseline
        baseline_inventory = current_inventory
        data["baseline_inventory"] = baseline_inventory
        data["identity"] = identity
        data["baseline_locked"] = True
        memory_organ.write(data)
        print("[MAIN] Baseline inventory created and locked (first scan).")
    else:
        # Existing baseline: use it as the reference, ignore new changes as baseline
        baseline_inventory = baseline
        print("[MAIN] Existing baseline loaded (unchanged).")

    simple_memory = SimpleMemory(memory_organ)

    rules = MissionStabilityRules(
        max_cpu_percent=70.0,
        max_worker_count=8,
        must_keep_responsive=True,
        allow_aggressive_mode=True,
    )
    monitor = ResourceMonitor()
    reasoning = ReasoningAgent(simple_memory)

    queen = Queen(rules, monitor, reasoning, memory_organ, identity, baseline_inventory)
    anomaly = AnomalyWatcher(queen, memory_organ)

    queen.start()
    anomaly.start()

    # Example ad redirector (change ports as needed)
    queen.start_ad_redirector(listen_port=8080, sink_host="127.0.0.1", sink_port=9090)

    # Initial OS protection mission
    os_guard = Mission(
        id="os-guard-001",
        description="Protect OS and guardian organism.",
        priority=10,
        mission_type="os_protect",
    )
    queen.set_active_mission(os_guard)
    queen.queue_mission(os_guard)

    root = tk.Tk()
    gui = BorgGUI(root, queen, anomaly, memory_organ)

    try:
        root.mainloop()
    finally:
        queen.stop()


if __name__ == "__main__":
    main()

