"""
Hive Guardian - Single File Prototype
Windows-first, cross-platform aware (Linux/macOS compatible where psutil works).

External dependencies (install via pip):
  pip install psutil requests

This prototype runs:
- One Queen
- One Node agent (local machine)
- A compact Tkinter GUI showing node status

It uses an in-process message bus rather than real networking to
keep everything in a single file and easy to run.
"""

import os
import sys
import time
import json
import queue
import psutil
import threading
import asyncio
import platform
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Any, List, Set, Optional
import requests

# =========================
# In-process async message bus
# =========================

class InProcessBus:
    """
    Very simple pub/sub bus for this single-file prototype.
    Topics: "telemetry", "commands"
    """

    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {
            "telemetry": asyncio.Queue(),
            "commands": asyncio.Queue(),
        }

    async def publish(self, topic: str, msg: Dict[str, Any]):
        if topic in self.queues:
            await self.queues[topic].put(msg)

    async def subscribe(self, topic: str):
        q = self.queues[topic]
        while True:
            msg = await q.get()
            yield msg


# Shared bus instance for this process
GLOBAL_BUS = InProcessBus()


# =========================
# OS Adapters
# =========================

class OSAdapter(ABC):
    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_disk_health(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_cache_paths(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def identify_node_id(self) -> str:
        pass

    @abstractmethod
    def get_software_profile(self) -> Dict[str, Any]:
        pass


class WindowsAdapter(OSAdapter):
    def get_system_info(self) -> Dict[str, Any]:
        return {
            "os_name": "windows",
            "platform": platform.platform(),
            "version": platform.version(),
            "release": platform.release(),
        }

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory": psutil.virtual_memory()._asdict(),
        }

    def get_disk_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        for part in psutil.disk_partitions(all=False):
            if "cdrom" in part.opts or part.fstype == "":
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
                health[part.mountpoint] = {
                    "device": part.device,
                    "fstype": part.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "percent": usage.percent,
                }
            except PermissionError:
                continue
        return health

    def get_cache_paths(self) -> Dict[str, str]:
        base = os.getenv("APPDATA", None)
        if base is None:
            base = "C:\\hive_cache"
        else:
            base = os.path.join(base, "HiveCache")
        primary = os.path.join(base, "primary")
        secondary = os.path.join(base, "secondary")
        return {"primary": primary, "secondary": secondary}

    def identify_node_id(self) -> str:
        return platform.node()

    def get_software_profile(self) -> Dict[str, Any]:
        program_files = os.getenv("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.getenv("ProgramFiles(x86)", "C:\\Program Files (x86)")
        return {
            "program_files_dirs": [program_files, program_files_x86],
            "note": "Deep registry-based enumeration can be added later.",
        }


class LinuxAdapter(OSAdapter):
    def get_system_info(self) -> Dict[str, Any]:
        return {
            "os_name": "linux",
            "platform": platform.platform(),
            "version": platform.version(),
            "release": platform.release(),
        }

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory": psutil.virtual_memory()._asdict(),
        }

    def get_disk_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                health[part.mountpoint] = {
                    "device": part.device,
                    "fstype": part.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "percent": usage.percent,
                }
            except PermissionError:
                continue
        return health

    def get_cache_paths(self) -> Dict[str, str]:
        base = os.path.expanduser("~/.hive_cache")
        primary = os.path.join(base, "primary")
        secondary = os.path.join(base, "secondary")
        return {"primary": primary, "secondary": secondary}

    def identify_node_id(self) -> str:
        return platform.node()

    def get_software_profile(self) -> Dict[str, Any]:
        return {
            "common_dirs": ["/usr/bin", "/usr/local/bin", "/opt"],
            "note": "Deep package manager integration can be added later.",
        }


class MacOSAdapter(OSAdapter):
    def get_system_info(self) -> Dict[str, Any]:
        return {
            "os_name": "macos",
            "platform": platform.platform(),
            "version": platform.version(),
            "release": platform.release(),
        }

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory": psutil.virtual_memory()._asdict(),
        }

    def get_disk_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                health[part.mountpoint] = {
                    "device": part.device,
                    "fstype": part.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "percent": usage.percent,
                }
            except PermissionError:
                continue
        return health

    def get_cache_paths(self) -> Dict[str, str]:
        base = os.path.expanduser("~/Library/Application Support/HiveCache")
        primary = os.path.join(base, "primary")
        secondary = os.path.join(base, "secondary")
        return {"primary": primary, "secondary": secondary}

    def identify_node_id(self) -> str:
        return platform.node()

    def get_software_profile(self) -> Dict[str, Any]:
        return {
            "applications_dir": "/Applications",
            "note": "Spotlight/Homebrew integration can be added later.",
        }


def get_os_adapter() -> OSAdapter:
    system = platform.system().lower()
    if system == "windows":
        return WindowsAdapter()
    elif system == "darwin":
        return MacOSAdapter()
    else:
        return LinuxAdapter()


# =========================
# Behavior store (persistent brain per node)
# =========================

class BehaviorStore:
    def __init__(self, path: str):
        self.path = path
        self.data = {
            "process_sequences": {},   # process_name -> {next_process_name -> count}
            "proc_resource_usage": {}, # process_name -> stats
            "time_patterns": {},       # "hour:process_name" -> count
            "prefetch_reward": {},     # key -> {"hits": int, "misses": int, "last": ts}
        }
        self.last_process_snapshot: Dict[int, str] = {}
        self._load()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    def update_process_sequences(self, processes: Dict[int, str]):
        seq = self.data["process_sequences"]
        current_names = set(processes.values())
        previous_names = set(self.last_process_snapshot.values())
        started = current_names - previous_names
        stayed = current_names & previous_names

        for prev_name in previous_names:
            if prev_name not in seq:
                seq[prev_name] = {}
            for cur_name in started:
                seq[prev_name][cur_name] = seq[prev_name].get(cur_name, 0) + 1
            for cur_name in stayed:
                seq[prev_name][cur_name] = seq[prev_name].get(cur_name, 0) + 0.1

        self.last_process_snapshot = processes.copy()

    def update_proc_resource_usage(self, processes_info: List[Dict[str, Any]]):
        usage = self.data["proc_resource_usage"]
        for p in processes_info:
            name = (p.get("name") or "").lower()
            if not name:
                continue
            stats = usage.setdefault(name, {
                "cpu_samples": 0,
                "cpu_avg": 0.0,
                "mem_samples": 0,
                "mem_avg": 0.0,
            })
            cpu = p.get("cpu_percent") or 0.0
            mem = p.get("memory_percent") or 0.0

            stats["cpu_samples"] += 1
            stats["cpu_avg"] += (cpu - stats["cpu_avg"]) / stats["cpu_samples"]

            stats["mem_samples"] += 1
            stats["mem_avg"] += (mem - stats["mem_avg"]) / stats["mem_samples"]

    def update_time_patterns(self, processes_info: List[Dict[str, Any]]):
        t = time.localtime()
        hour = t.tm_hour
        tp = self.data["time_patterns"]
        for p in processes_info:
            name = (p.get("name") or "").lower()
            if not name:
                continue
            key = f"{hour}:{name}"
            tp[key] = tp.get(key, 0) + 1

    def record_prefetch_outcome(self, key: str, hit: bool):
        stats = self.data["prefetch_reward"].setdefault(key, {"hits": 0, "misses": 0, "last": 0.0})
        if hit:
            stats["hits"] += 1
        else:
            stats["misses"] += 1
        stats["last"] = time.time()

    def get_prefetch_score(self, key: str) -> float:
        stats = self.data["prefetch_reward"].get(key)
        if not stats:
            return 0.5
        h = stats["hits"]
        m = stats["misses"]
        total = h + m
        if total == 0:
            return 0.5
        return (h + 1) / (total + 2)


# =========================
# Critical cache with dual drives
# =========================

class DriveState:
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class CriticalCache:
    def __init__(self, primary_path: str, secondary_path: str):
        self.primary_path = primary_path
        self.secondary_path = secondary_path
        os.makedirs(self.primary_path, exist_ok=True)
        os.makedirs(self.secondary_path, exist_ok=True)
        self.primary_state = DriveState.HEALTHY
        self.secondary_state = DriveState.HEALTHY

    def _read_file(self, base: str, key: str) -> Optional[bytes]:
        path = os.path.join(base, self._key_to_filename(key))
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError:
            return None

    def _write_file(self, base: str, key: str, data: bytes) -> bool:
        path = os.path.join(base, self._key_to_filename(key))
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
            return True
        except OSError:
            return False

    def _key_to_filename(self, key: str) -> str:
        # Very simple key sanitization
        safe = key.replace(":", "_").replace("/", "_").replace("\\", "_")
        return safe

    def has(self, key: str) -> bool:
        p = os.path.join(self.primary_path, self._key_to_filename(key))
        s = os.path.join(self.secondary_path, self._key_to_filename(key))
        return os.path.exists(p) or os.path.exists(s)

    def get(self, key: str) -> Optional[bytes]:
        if self.primary_state in (DriveState.HEALTHY, DriveState.DEGRADED):
            data = self._read_file(self.primary_path, key)
            if data is not None:
                return data
        if self.secondary_state in (DriveState.HEALTHY, DriveState.DEGRADED):
            data = self._read_file(self.secondary_path, key)
            if data is not None:
                return data
        return None

    def set(self, key: str, data: bytes) -> None:
        primary_ok = False
        secondary_ok = False
        if self.primary_state != DriveState.FAILED:
            primary_ok = self._write_file(self.primary_path, key, data)
            if not primary_ok:
                self.primary_state = DriveState.DEGRADED
        if self.secondary_state != DriveState.FAILED:
            secondary_ok = self._write_file(self.secondary_path, key, data)
            if not secondary_ok:
                self.secondary_state = DriveState.DEGRADED
        if not primary_ok and self.primary_state == DriveState.DEGRADED:
            self.primary_state = DriveState.FAILED
        if not secondary_ok and self.secondary_state == DriveState.DEGRADED:
            self.secondary_state = DriveState.FAILED

    def health(self) -> Dict[str, Any]:
        return {
            "primary": {
                "path": self.primary_path,
                "state": self.primary_state,
            },
            "secondary": {
                "path": self.secondary_path,
                "state": self.secondary_state,
            },
        }


# =========================
# Telemetry collection
# =========================

def collect_processes(limit: int = 200) -> List[Dict[str, Any]]:
    procs: List[Dict[str, Any]] = []
    for proc in psutil.process_iter(
        attrs=["pid", "name", "exe", "username", "cpu_percent", "memory_percent"]
    ):
        try:
            info = proc.info
            p: Dict[str, Any] = {
                "pid": info.get("pid"),
                "name": info.get("name"),
                "exe": info.get("exe") or "",
                "username": info.get("username") or "",
                "cpu_percent": info.get("cpu_percent") or 0.0,
                "memory_percent": info.get("memory_percent") or 0.0,
            }
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if len(procs) >= limit:
            break
    return procs


def collect_connections(limit: int = 300) -> List[Dict[str, Any]]:
    conns: List[Dict[str, Any]] = []
    try:
        net_conns = psutil.net_connections(kind="inet")
    except Exception:
        net_conns = []
    for c in net_conns:
        try:
            laddr = f"{c.laddr.ip}" if c.laddr else ""
            lport = c.laddr.port if c.laddr else 0
            raddr = f"{c.raddr.ip}" if c.raddr else ""
            rport = c.raddr.port if c.raddr else 0
            pid = c.pid or -1
            name = ""
            if pid > 0:
                try:
                    name = psutil.Process(pid).name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            info: Dict[str, Any] = {
                "laddr": laddr,
                "lport": lport,
                "raddr": raddr,
                "rport": rport,
                "status": c.status,
                "pid": pid,
                "process_name": name,
            }
            conns.append(info)
        except Exception:
            continue
        if len(conns) >= limit:
            break
    return conns


def collect_telemetry(node_id: str, os_adapter: OSAdapter) -> Dict[str, Any]:
    os_info = os_adapter.get_system_info()
    hw_info = os_adapter.get_hardware_info()
    disk_health = os_adapter.get_disk_health()
    software_profile = os_adapter.get_software_profile()
    processes = collect_processes()
    connections = collect_connections()

    telemetry: Dict[str, Any] = {
        "node_id": node_id,
        "os": os_info,
        "hardware": hw_info,
        "disks": disk_health,
        "processes": processes,
        "connections": connections,
        "software_profile": software_profile,
    }
    return telemetry


# =========================
# Prefetch policy + engine
# =========================

class PrefetchPolicy:
    def __init__(self, behavior: BehaviorStore):
        self.behavior = behavior
        self.process_url_hints: Dict[str, Set[str]] = defaultdict(set)
        self.prefetch_history: Dict[str, Dict[str, Any]] = {}

    def update_from_telemetry(self, telemetry: Dict[str, Any]) -> None:
        for conn in telemetry.get("connections", []):
            pname = (conn.get("process_name") or "").lower()
            raddr = conn.get("raddr") or ""
            if not pname or not raddr:
                continue
            self.process_url_hints[pname].add(raddr)

    def suggest_tasks(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        processes = telemetry.get("processes", [])
        active_procs = set((p.get("name") or "").lower() for p in processes)

        seq = self.behavior.data["process_sequences"]
        likely_future_procs: Set[str] = set()
        for pname in active_procs:
            nexts = seq.get(pname, {})
            for nname, count in nexts.items():
                if count >= 3:
                    likely_future_procs.add(nname)

        for pname in active_procs.union(likely_future_procs):
            for host in self.process_url_hints.get(pname, []):
                url = f"http://{host}"
                key = f"url:{url}"
                score = self.behavior.get_prefetch_score(key)
                if score < 0.4:
                    continue
                tasks.append({
                    "type": "url",
                    "url": url,
                    "source_process": pname,
                    "reason": "behavior_profile",
                    "score": score,
                })

        now = time.localtime()
        hour = now.tm_hour
        tp = self.behavior.data["time_patterns"]
        for key, count in tp.items():
            h_str, pname = key.split(":", 1)
            if int(h_str) != hour:
                continue
            if count < 5:
                continue
            for host in self.process_url_hints.get(pname, []):
                url = f"http://{host}"
                key = f"url:{url}"
                score = self.behavior.get_prefetch_score(key)
                if score < 0.5:
                    continue
                tasks.append({
                    "type": "url",
                    "url": url,
                    "source_process": pname,
                    "reason": "time_pattern",
                    "score": score,
                })

        seen = set()
        unique: List[Dict[str, Any]] = []
        for t in tasks:
            k = (t["type"], t.get("path") or t.get("url"))
            if k in seen:
                continue
            seen.add(k)
            unique.append(t)

        return unique

    def record_prefetch_result(self, key: str, success: bool) -> None:
        stats = self.prefetch_history.setdefault(key, {"tries": 0, "success": 0, "last": 0.0})
        stats["tries"] += 1
        if success:
            stats["success"] += 1
        stats["last"] = time.time()
        self.behavior.record_prefetch_outcome(key, success)


class PrefetchEngine:
    def __init__(self, cache: CriticalCache, policy: PrefetchPolicy, max_workers: int = 2):
        self.cache = cache
        self.policy = policy
        self.task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.max_workers = max_workers
        self.running = False
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "success": 0,
            "failed": 0,
        }

    def start(self):
        if self.running:
            return
        self.running = True
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

    def stop(self):
        self.running = False

    def submit(self, task: Dict[str, Any]):
        self.stats["submitted"] += 1
        self.task_queue.put(task)

    def _worker_loop(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                success = self._handle_task(task)
                self.stats["completed"] += 1
                if success:
                    self.stats["success"] += 1
                else:
                    self.stats["failed"] += 1
            except Exception:
                self.stats["completed"] += 1
                self.stats["failed"] += 1
                success = False
            finally:
                key = None
                if task.get("type") == "file":
                    key = f"file:{task.get('path')}"
                elif task.get("type") == "url":
                    key = f"url:{task.get('url')}"
                if key:
                    self.policy.record_prefetch_result(key, success)
                self.task_queue.task_done()

    def _handle_task(self, task: Dict[str, Any]) -> bool:
        ttype = task.get("type")
        if ttype == "file":
            return self._prefetch_file(task.get("path", ""))
        elif ttype == "url":
            return self._prefetch_url(task.get("url", ""))
        return False

    def _prefetch_file(self, path: str) -> bool:
        if not path:
            return False
        key = f"file:{path}"
        if self.cache.has(key):
            return True
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.cache.set(key, data)
            return True
        except OSError:
            return False

    def _prefetch_url(self, url: str) -> bool:
        if not url:
            return False
        key = f"url:{url}"
        if self.cache.has(key):
            return True
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok:
                self.cache.set(key, resp.content)
                return True
        except Exception:
            pass
        return False


# =========================
# Anomaly detector using behavior
# =========================

class AnomalyDetector:
    def __init__(self, behavior: BehaviorStore):
        self.behavior = behavior

    def score(self, telemetry: Dict[str, Any]) -> float:
        score = 0.0
        proc_count = len(telemetry.get("processes", []))
        if proc_count > 400:
            score += 0.2

        known_usage = self.behavior.data["proc_resource_usage"]
        unknown_procs = 0
        for p in telemetry.get("processes", []):
            name = (p.get("name") or "").lower()
            if name and name not in known_usage:
                unknown_procs += 1
        if unknown_procs > 5:
            score += 0.2

        seen_hosts = set()
        for k in self.behavior.data.get("prefetch_reward", {}).keys():
            if k.startswith("url:"):
                host_part = k.split("://", 1)[-1]
                seen_hosts.add(host_part)

        new_conn_hosts = 0
        for c in telemetry.get("connections", []):
            raddr = c.get("raddr") or ""
            if not raddr:
                continue
            if raddr not in seen_hosts:
                new_conn_hosts += 1
        if new_conn_hosts > 10:
            score += 0.3

        return min(score, 1.0)


# =========================
# NodeAgent
# =========================

class NodeAgent:
    def __init__(self):
        self.os_adapter = get_os_adapter()
        self.node_id = self.os_adapter.identify_node_id()
        cache_paths = self.os_adapter.get_cache_paths()
        self.critical_cache = CriticalCache(
            primary_path=cache_paths["primary"],
            secondary_path=cache_paths["secondary"],
        )
        behavior_path = os.path.join(os.path.dirname(cache_paths["primary"]), "behavior.json")
        self.behavior = BehaviorStore(behavior_path)
        self.anomaly = AnomalyDetector(self.behavior)
        self.prefetch_policy = PrefetchPolicy(self.behavior)
        self.prefetch_engine = PrefetchEngine(self.critical_cache, self.prefetch_policy)
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.running = True

    async def start(self):
        self.prefetch_engine.start()
        asyncio.create_task(self.send_telemetry_loop())
        asyncio.create_task(self.prefetch_loop())

    async def send_telemetry_loop(self):
        while self.running:
            telemetry = collect_telemetry(self.node_id, self.os_adapter)
            proc_map = {
                p["pid"]: (p.get("name") or "").lower()
                for p in telemetry.get("processes", [])
                if p.get("pid") is not None
            }
            self.behavior.update_process_sequences(proc_map)
            self.behavior.update_proc_resource_usage(telemetry.get("processes", []))
            self.behavior.update_time_patterns(telemetry.get("processes", []))
            self.behavior.save()

            telemetry["critical_cache_health"] = self.critical_cache.health()
            telemetry["anomaly_score"] = self.anomaly.score(telemetry)
            telemetry["prefetch_stats"] = self.prefetch_engine.stats
            telemetry["prefetch_history"] = self.prefetch_policy.prefetch_history

            self.latest_telemetry = telemetry
            await GLOBAL_BUS.publish("telemetry", telemetry)
            await asyncio.sleep(2)

    async def prefetch_loop(self):
        while self.running:
            telemetry = self.latest_telemetry
            if telemetry:
                self.prefetch_policy.update_from_telemetry(telemetry)
                tasks = self.prefetch_policy.suggest_tasks(telemetry)
                for t in tasks:
                    self.prefetch_engine.submit(t)
            await asyncio.sleep(5)


# =========================
# Queen state + controller
# =========================

class QueenState:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.connection_graph: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def update_from_telemetry(self, msg: Dict[str, Any]):
        node_id = msg.get("node_id")
        if not node_id:
            return
        os_info = msg.get("os", {})
        hw = msg.get("hardware", {})
        disks = msg.get("disks", {})
        conns = msg.get("connections", [])
        procs = msg.get("processes", [])
        anomaly = msg.get("anomaly_score", 0.0)
        cache_health = msg.get("critical_cache_health", {})
        prefetch_stats = msg.get("prefetch_stats", {})

        summary: Dict[str, Any] = {
            "node_id": node_id,
            "os_name": os_info.get("os_name", "unknown"),
            "hw_summary": {
                "cpu_count_logical": hw.get("cpu_count_logical"),
                "cpu_count_physical": hw.get("cpu_count_physical"),
                "memory": hw.get("memory", {}),
            },
            "disk_summary": disks,
            "connections_count": len(conns),
            "processes_count": len(procs),
            "anomaly_score": anomaly,
            "critical_cache_health": cache_health,
            "prefetch_stats": prefetch_stats,
        }
        self.nodes[node_id] = summary

        node_edges = self.connection_graph.setdefault(node_id, {})
        now = time.time()
        for c in conns:
            raddr = c.get("raddr") or ""
            if not raddr:
                continue
            ext_id = f"ext:{raddr}"
            edge = node_edges.get(ext_id)
            if not edge:
                edge = {
                    "source": node_id,
                    "target": ext_id,
                    "count": 0,
                    "last_seen": 0.0,
                    "risk": 0.0,
                }
            edge["count"] += 1
            edge["last_seen"] = now
            node_edges[ext_id] = edge


class PolicyEngine:
    def evaluate(self, telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        score = telemetry.get("anomaly_score", 0.0)
        node_id = telemetry.get("node_id", "unknown")
        if score > 0.8:
            return {
                "target_node": node_id,
                "action": "flag_suspect",
                "reason": "high_anomaly_score",
                "score": score,
            }
        return None


class QueenController:
    def __init__(self, state: QueenState):
        self.state = state
        self.policy = PolicyEngine()
        self.running = True

    async def start(self):
        asyncio.create_task(self.listen_for_telemetry())

    async def listen_for_telemetry(self):
        async for msg in GLOBAL_BUS.subscribe("telemetry"):
            node_id = msg.get("node_id")
            anomaly = msg.get("anomaly_score")
            os_info = msg.get("os", {})
            conn_count = len(msg.get("connections", []))
            proc_count = len(msg.get("processes", []))
            print(
                f"[{node_id}] os={os_info.get('os_name')} "
                f"procs={proc_count} conns={conn_count} anomaly={anomaly:.2f}"
            )
            self.state.update_from_telemetry(msg)
            decision = self.policy.evaluate(msg)
            if decision:
                await GLOBAL_BUS.publish("commands", decision)


# =========================
# Compact Tkinter GUI
# =========================

import tkinter as tk
from tkinter import ttk

class HiveGUI:
    """
    Compact GUI:
      - Node summary (OS, CPU, RAM, anomaly)
      - Cache health
      - Prefetch stats
      - Connection & process counts
    Refreshes every 2s from QueenState.
    """

    def __init__(self, root: tk.Tk, state: QueenState):
        self.root = root
        self.state = state
        self.root.title("Hive Guardian")
        self.root.geometry("700x400")

        self.node_select = ttk.Combobox(self.root, state="readonly")
        self.node_select.pack(fill="x", padx=8, pady=4)
        self.node_select.bind("<<ComboboxSelected>>", self.on_node_selected)

        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.labels: Dict[str, tk.Label] = {}
        fields = [
            "OS", "CPU (logical/physical)", "Memory (GB used/total)",
            "Anomaly Score", "Processes", "Connections",
            "Cache Primary", "Cache Secondary",
            "Prefetch Submitted/Completed/Success/Failed",
        ]
        for f in fields:
            lbl_title = tk.Label(self.info_frame, text=f + ":", anchor="w", width=30)
            lbl_title.pack(anchor="w")
            lbl_value = tk.Label(self.info_frame, text="...", anchor="w", fg="#00ff00")
            lbl_value.pack(anchor="w")
            self.labels[f] = lbl_value

        self.selected_node_id: Optional[str] = None
        self.refresh_ui()

    def on_node_selected(self, event=None):
        self.selected_node_id = self.node_select.get()
        self.update_node_view()

    def refresh_ui(self):
        node_ids = list(self.state.nodes.keys())
        current = list(self.node_select["values"])
        if node_ids != list(current):
            self.node_select["values"] = node_ids
            if not self.selected_node_id and node_ids:
                self.selected_node_id = node_ids[0]
                self.node_select.set(self.selected_node_id)
        self.update_node_view()
        self.root.after(2000, self.refresh_ui)

    def update_node_view(self):
        nid = self.selected_node_id
        if not nid or nid not in self.state.nodes:
            return
        summary = self.state.nodes[nid]

        os_name = summary.get("os_name", "unknown")
        hw = summary.get("hw_summary", {})
        mem = hw.get("memory", {})
        mem_total = mem.get("total", 0) / (1024**3) if mem.get("total") else 0
        mem_used = mem.get("used", 0) / (1024**3) if mem.get("used") else 0

        cpu_log = hw.get("cpu_count_logical", 0)
        cpu_phy = hw.get("cpu_count_physical", 0)

        anomaly = summary.get("anomaly_score", 0.0)
        proc_count = summary.get("processes_count", 0)
        conn_count = summary.get("connections_count", 0)
        cache_health = summary.get("critical_cache_health", {})
        p = cache_health.get("primary", {})
        s = cache_health.get("secondary", {})
        prefetch = summary.get("prefetch_stats", {})

        self.labels["OS"]["text"] = os_name
        self.labels["CPU (logical/physical)"]["text"] = f"{cpu_log} / {cpu_phy}"
        self.labels["Memory (GB used/total)"]["text"] = f"{mem_used:.1f} / {mem_total:.1f}"
        self.labels["Anomaly Score"]["text"] = f"{anomaly:.2f}"
        self.labels["Processes"]["text"] = str(proc_count)
        self.labels["Connections"]["text"] = str(conn_count)
        self.labels["Cache Primary"]["text"] = f"{p.get('state', 'unknown')} @ {p.get('path', '')}"
        self.labels["Cache Secondary"]["text"] = f"{s.get('state', 'unknown')} @ {s.get('path', '')}"
        self.labels["Prefetch Submitted/Completed/Success/Failed"]["text"] = (
            f"{prefetch.get('submitted', 0)}/"
            f"{prefetch.get('completed', 0)}/"
            f"{prefetch.get('success', 0)}/"
            f"{prefetch.get('failed', 0)}"
        )


# =========================
# Main runner
# =========================

async def main_async():
    state = QueenState()
    queen = QueenController(state)
    node = NodeAgent()

    await queen.start()
    await node.start()

    # Run Tkinter in a separate thread
    def gui_thread():
        root = tk.Tk()
        gui = HiveGUI(root, state)
        root.mainloop()

    t = threading.Thread(target=gui_thread, daemon=True)
    t.start()

    # Keep the asyncio loop alive
    while True:
        await asyncio.sleep(1)


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

