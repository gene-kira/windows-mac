import os
import sys
import json
import time
import socket
import threading
import subprocess
import platform
import ipaddress
import queue
import tkinter as tk
from tkinter import scrolledtext, simpledialog, filedialog, ttk
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple

# --------------------------------------------------------------------
# Dependency handling (psutil)
# --------------------------------------------------------------------
try:
    import psutil
except ImportError:
    print("[SETUP] psutil not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# --------------------------------------------------------------------
# Global tuning constants
# --------------------------------------------------------------------
BASELINE_AGE_SECONDS = 7 * 24 * 3600
IDLE_STRIKES_THRESHOLD = 5

AI_MODE_POLL_MS = 200
AI_MODE_STABLE_TICKS = 3

BLOCKED_LOG_MAX_ENTRIES = 100

GAME_PROTECT_DOMAINS = [
    "steampowered.com",
    "steamcontent.com",
    "steamstatic.com",
    "steamcdn-a.akamaihd.net",
    "steamserver.net",
    "steamcommunity.com",
    "epicgames.com",
    "epicgames.dev",
    "unrealengine.com",
    "ol.epicgames.com",
    "egs-platform-service-prod.ol.epicgames.com",
]

# --------------------------------------------------------------------
# Memory core
# --------------------------------------------------------------------

DEFAULT_PRIMARY_DIR = os.path.join(os.path.expanduser("~"), ".hive_guardian")
DEFAULT_PRIMARY_FILE = os.path.join(DEFAULT_PRIMARY_DIR, "state.json")


class MemoryCore:
    """
    Persistent JSON state for the entire organism.
    """

    def __init__(
        self,
        primary_file: str = DEFAULT_PRIMARY_FILE,
        backup_file: Optional[str] = None,
    ):
        self.primary_file = primary_file
        self.backup_file = backup_file
        os.makedirs(os.path.dirname(self.primary_file), exist_ok=True)

        self.state: Dict[str, Any] = {
            "version": 1,
            "last_updated": 0,
            "network": {},
            "process": {},
            "sidebrain": {},
            "queen": {},
            "baseline": {
                "process_names": {}
            },
            "modes": {
                "current": "idle",
                "auto_mode_enabled": True
            },
            "booster": {
                "base_seconds": 3.0,
                "max_seconds": 10.0,
            },
            "forecast": {
                "transitions": {},
                "last_event": None,
            },
            "presidential": {
                "enabled": False,
                "target_cpu_share_min": 0.6,
                "target_cpu_share_max": 0.8,
                "auto_trigger": True,
                "last_trigger_source": None,
                "last_trigger_reason": None,
                "last_activated_ts": None,
                "last_deactivated_ts": None,
                "muted_until": 0.0,
            },
            "guardian": {
                "override_enabled": False,
            },
            "ui": {
                "boost_enabled": True,
            },
            "backup_file_path": None,
            "filters": {
                "whitelist": [],
                "blacklist": [],
            },
            "blocked_log": [],
            "prediction": {
                "best_guess_enabled": True,
                "mode_by_hour": {},       # {hour: {mode: count}}
                "last_prediction": None,  # {"mode": str, "confidence": float, "ts": float}
            },
        }

    def _load_json(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[MEMORY] Failed to load {path}: {e}")
            return None

    def _save_json(self, path: str, data: Dict[str, Any]):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MEMORY] Failed to save {path}: {e}")

    def set_backup_path(self, backup_file: Optional[str]):
        self.backup_file = backup_file
        self.state["backup_file_path"] = backup_file

    def load(self):
        primary = self._load_json(self.primary_file)
        backup = self._load_json(self.backup_file) if self.backup_file else None

        chosen = None
        if primary and backup:
            p_ts = primary.get("last_updated", 0)
            b_ts = backup.get("last_updated", 0)
            chosen = primary if p_ts >= b_ts else backup
        elif primary:
            chosen = primary
        elif backup:
            chosen = backup

        if chosen:
            self.state = chosen
            bpath = self.state.get("backup_file_path")
            if bpath:
                self.backup_file = bpath
            print("[MEMORY] Loaded state.")
        else:
            print("[MEMORY] No existing state; starting fresh.")

    def save(self):
        self.state["last_updated"] = time.time()
        self._save_json(self.primary_file, self.state)
        if self.backup_file:
            self._save_json(self.backup_file, self.state)
        print("[MEMORY] State saved.")

    def get_section(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set_section(self, key: str, value: Any):
        self.state[key] = value

    def set_mode(self, mode: str):
        self.state.setdefault("modes", {})
        self.state["modes"]["current"] = mode

    def get_mode(self) -> str:
        return self.state.get("modes", {}).get("current", "idle")

    def get_auto_mode_enabled(self) -> bool:
        return self.state.get("modes", {}).get("auto_mode_enabled", True)

    def set_auto_mode_enabled(self, enabled: bool):
        self.state.setdefault("modes", {})
        self.state["modes"]["auto_mode_enabled"] = bool(enabled)

    def get_booster_config(self):
        return self.state.get("booster", {"base_seconds": 3.0, "max_seconds": 10.0})

    def set_booster_config(self, base_seconds: float, max_seconds: float):
        self.state["booster"] = {
            "base_seconds": float(base_seconds),
            "max_seconds": float(max_seconds),
        }

    def get_forecast_state(self) -> Dict[str, Any]:
        return self.state.setdefault(
            "forecast", {"transitions": {}, "last_event": None}
        )

    def update_forecast_state(self, forecast_state: Dict[str, Any]):
        self.state["forecast"] = forecast_state

    def get_presidential_state(self) -> Dict[str, Any]:
        return self.state.setdefault(
            "presidential",
            {
                "enabled": False,
                "target_cpu_share_min": 0.6,
                "target_cpu_share_max": 0.8,
                "auto_trigger": True,
                "last_trigger_source": None,
                "last_trigger_reason": None,
                "last_activated_ts": None,
                "last_deactivated_ts": None,
                "muted_until": 0.0,
            },
        )

    def update_presidential_state(self, pres_state: Dict[str, Any]):
        self.state["presidential"] = pres_state

    def get_guardian_state(self) -> Dict[str, Any]:
        return self.state.setdefault(
            "guardian",
            {
                "override_enabled": False,
            },
        )

    def update_guardian_state(self, guardian_state: Dict[str, Any]):
        self.state["guardian"] = guardian_state

    def get_baseline_process_info(self) -> Dict[str, Dict[str, Any]]:
        baseline = self.state.setdefault("baseline", {})
        return baseline.setdefault("process_names", {})

    def update_baseline_process_info(self, info: Dict[str, Dict[str, Any]]):
        baseline = self.state.setdefault("baseline", {})
        baseline["process_names"] = info

    def get_ui_state(self) -> Dict[str, Any]:
        return self.state.setdefault("ui", {"boost_enabled": True})

    def update_ui_state(self, ui_state: Dict[str, Any]):
        self.state["ui"] = ui_state

    def get_filters(self) -> Dict[str, List[str]]:
        return self.state.setdefault("filters", {"whitelist": [], "blacklist": []})

    def update_filters(self, filters: Dict[str, List[str]]):
        self.state["filters"] = filters

    def get_blocked_log(self) -> List[Dict[str, Any]]:
        return self.state.setdefault("blocked_log", [])

    def add_blocked_entry(self, domain: str, reason: str, max_entries: int = BLOCKED_LOG_MAX_ENTRIES):
        log = self.get_blocked_log()
        entry = {
            "domain": domain.lower(),
            "reason": reason,
            "ts": time.time(),
        }
        log.append(entry)
        if len(log) > max_entries:
            del log[0 : len(log) - max_entries]
        self.state["blocked_log"] = log
        self.save()

    def get_prediction_state(self) -> Dict[str, Any]:
        return self.state.setdefault(
            "prediction",
            {
                "best_guess_enabled": True,
                "mode_by_hour": {},
                "last_prediction": None,
            },
        )

    def update_prediction_state(self, pstate: Dict[str, Any]):
        self.state["prediction"] = pstate


# --------------------------------------------------------------------
# System scanning
# --------------------------------------------------------------------

@dataclass
class HardwareInfo:
    cpu_count_logical: int
    cpu_count_physical: int
    total_ram: int
    os: str
    os_version: str


@dataclass
class ProcessSnapshot:
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    status: str


def scan_hardware() -> HardwareInfo:
    cpu_logical = psutil.cpu_count(logical=True) or 1
    cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
    vmem = psutil.virtual_memory()
    return HardwareInfo(
        cpu_count_logical=cpu_logical,
        cpu_count_physical=cpu_physical,
        total_ram=vmem.total,
        os=platform.system(),
        os_version=platform.version(),
    )


def scan_processes() -> List[ProcessSnapshot]:
    snapshots = []
    psutil.cpu_percent(interval=None)
    for proc in psutil.process_iter(
        ["pid", "name", "cpu_percent", "memory_percent", "status"]
    ):
        try:
            info = proc.info
            snapshots.append(
                ProcessSnapshot(
                    pid=info["pid"],
                    name=info.get("name") or "",
                    cpu_percent=info.get("cpu_percent") or 0.0,
                    memory_percent=info.get("memory_percent") or 0.0,
                    status=info.get("status") or "",
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return snapshots


# --------------------------------------------------------------------
# Sidebrain
# --------------------------------------------------------------------

@dataclass
class RequestMeta:
    client_addr: str
    method: str
    url: str
    headers: dict


class SideBrain:
    def __init__(self):
        self._in_q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self.stats = {
            "total": 0,
            "ads": 0,
            "normal": 0,
        }

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)

    def submit(self, meta: RequestMeta):
        self._in_q.put(meta)

    def _worker(self):
        while not self._stop.is_set():
            try:
                meta = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._classify_and_log(meta)

    def _is_ad(self, url: str) -> bool:
        import re
        ad_like = any(
            [
                "doubleclick" in url,
                "adservice" in url,
                re.search(r"/ads?[/?]", url or "") is not None,
            ]
        )
        return ad_like

    def _classify_and_log(self, meta: RequestMeta):
        self.stats["total"] += 1
        if self._is_ad(meta.url):
            label = "AD / IRRELEVANT"
            self.stats["ads"] += 1
        else:
            label = "NORMAL"
            self.stats["normal"] += 1
        print(f"[SIDEBRAIN] {label} :: {meta.method} {meta.url}")


# --------------------------------------------------------------------
# Booster
# --------------------------------------------------------------------

@dataclass
class BoostRecord:
    pid: int
    start_time: float
    end_time: float
    original_nice: Optional[int]
    original_affinity: Optional[List[int]]


class BoostAI:
    def __init__(
        self,
        base_seconds: float = 3.0,
        max_seconds: float = 10.0,
        cpu_threshold: float = 40.0,
        max_simultaneous_boosts: int = 3,
    ):
        self.base_seconds = base_seconds
        self.max_seconds = max_seconds
        self.cpu_threshold = cpu_threshold
        self.max_simultaneous_boosts = max_simultaneous_boosts

    def choose_boost_candidates(self) -> List[int]:
        candidates: List[int] = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                info = proc.info
                cpu = info.get("cpu_percent", 0.0)
                name = (info.get("name") or "").lower()
                if any(
                    skip in name
                    for skip in ["system", "idle", "defender", "antivirus"]
                ):
                    continue
                if cpu >= self.cpu_threshold:
                    candidates.append(info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        seen = set()
        unique: List[int] = []
        for pid in candidates:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        return unique[: self.max_simultaneous_boosts]

    def should_extend(self, pid: int, rec: BoostRecord) -> bool:
        try:
            proc = psutil.Process(pid)
            cpu = proc.cpu_percent(interval=0.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

        system_cpu = psutil.cpu_percent(interval=0.0)
        if cpu >= self.cpu_threshold * 0.7 and system_cpu >= 60.0:
            now = time.time()
            current_duration = now - rec.start_time
            return current_duration < self.max_seconds
        return False


class Booster:
    def __init__(
        self,
        memory: Optional[MemoryCore] = None,
        scan_interval: float = 0.5,
    ):
        self.memory = memory or MemoryCore()
        cfg = self.memory.get_booster_config()
        self.base_seconds = cfg.get("base_seconds", 3.0)
        self.max_seconds = cfg.get("max_seconds", 10.0)

        self.scan_interval = scan_interval
        self.ai = BoostAI(
            base_seconds=self.base_seconds,
            max_seconds=self.max_seconds,
            cpu_threshold=40.0,
            max_simultaneous_boosts=3,
        )
        self._boosts: Dict[int, BoostRecord] = {}
        self._stop = threading.Event()
        self._winding_down = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        print(
            f"[BOOSTER] Starting boost controller (base={self.base_seconds}s, max={self.max_seconds}s)..."
        )
        if not self._thread.is_alive():
            self._stop.clear()
            self._winding_down = False
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        print("[BOOSTER] Smart stopping boost controller...")
        self._winding_down = True
        self._stop.set()
        self._thread.join(timeout=2)
        for rec in list(self._boosts.values()):
            self._restore(rec.pid)

    def _loop(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[BOOSTER] Tick error: {e}")
            time.sleep(self.scan_interval)

    def _tick(self):
        now = time.time()
        for pid, rec in list(self._boosts.items()):
            if now >= rec.end_time:
                if not self._winding_down and self.ai.should_extend(pid, rec):
                    new_end = min(
                        rec.end_time + 2.0, rec.start_time + self.max_seconds
                    )
                    if new_end > rec.end_time:
                        print(
                            f"[BOOSTER] Extending boost for PID {pid} to {new_end - rec.start_time:.1f}s total."
                        )
                        rec.end_time = new_end
                    else:
                        self._restore(pid)
                else:
                    self._restore(pid)

        if self._winding_down:
            return

        candidates = self.ai.choose_boost_candidates()
        for pid in candidates:
            if pid not in self._boosts:
                self._apply_boost(pid)

    def _apply_boost(self, pid: int):
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            original_nice: Optional[int] = None
            original_affinity: Optional[List[int]] = None
            try:
                original_nice = proc.nice()
                if psutil.WINDOWS:
                    proc.nice(psutil.HIGH_PRIORITY_CLASS)
                else:
                    proc.nice(0)
            except (psutil.AccessDenied, ValueError):
                pass
            try:
                if hasattr(proc, "cpu_affinity"):
                    original_affinity = proc.cpu_affinity()
                    all_cores = list(range(psutil.cpu_count(logical=True) or 1))
                    proc.cpu_affinity(all_cores)
            except (psutil.AccessDenied, AttributeError, ValueError):
                pass

            now = time.time()
            rec = BoostRecord(
                pid=pid,
                start_time=now,
                end_time=now + self.base_seconds,
                original_nice=original_nice,
                original_affinity=original_affinity,
            )
            self._boosts[pid] = rec
            print(
                f"[BOOSTER] Boosted PID {pid} ({name}) for {self.base_seconds} seconds (adaptive up to {self.max_seconds}s)."
            )
        except psutil.NoSuchProcess:
            return

    def _restore(self, pid: int):
        rec = self._boosts.pop(pid, None)
        if not rec:
            return
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            if rec.original_nice is not None:
                try:
                    proc.nice(rec.original_nice)
                except (psutil.AccessDenied, ValueError):
                    pass
            if rec.original_affinity is not None:
                try:
                    if hasattr(proc, "cpu_affinity"):
                        proc.cpu_affinity(rec.original_affinity)
                except (psutil.AccessDenied, AttributeError, ValueError):
                    pass
            total = rec.end_time - rec.start_time
            print(f"[BOOSTER] Restored PID {pid} ({name}) after {total:.1f}s boost.")
        except psutil.NoSuchProcess:
            pass


# --------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------

class ProcClass(Enum):
    CRITICAL = "critical"
    ACTIVE = "active"
    OCCASIONAL = "occasional"
    JUNK = "junk"


@dataclass
class TrackedProcess:
    snapshot: ProcessSnapshot
    classification: ProcClass
    suspended: bool = False
    idle_strikes: int = 0


class OptimizerAI:
    def __init__(
        self,
        active_cpu_threshold=5.0,
        junk_name_patterns=None,
        critical_name_patterns=None,
    ):
        self.active_cpu_threshold = active_cpu_threshold
        self.junk_name_patterns = junk_name_patterns or [
            "updater",
            "telemetry",
            "assistant",
            "toolbar",
            "adobe updater",
            "auto update",
            "helper",
            "crashreport",
        ]
        self.critical_name_patterns = critical_name_patterns or [
            "system",
            "explorer",
            "csrss",
            "wininit",
            "services",
            "lsass",
            "dwm",
            "winlogon",
            "svchost",
            "defender",
            "antivirus",
            "shellexperiencehost",
            "startmenuexperiencehost",
            "searchui",
            "searchapp",
            "searchhost",
        ]

    def classify(self, snap: ProcessSnapshot) -> ProcClass:
        name = snap.name.lower()
        if any(pat in name for pat in self.critical_name_patterns):
            return ProcClass.CRITICAL
        if any(pat in name for pat in self.junk_name_patterns):
            return ProcClass.JUNK
        if snap.cpu_percent >= self.active_cpu_threshold:
            return ProcClass.ACTIVE
        return ProcClass.OCCASIONAL


class SystemOptimizer:
    def __init__(self, memory: Optional[MemoryCore] = None, scan_interval=2.0):
        self.memory = memory or MemoryCore()
        self.baseline = self.memory.get_baseline_process_info()

        self.scan_interval = scan_interval
        self.ai = OptimizerAI()
        self._tracked: Dict[int, TrackedProcess] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._enabled = False
        self._force_freeze = False
        self._winding_down = False
        self._last_baseline_save = 0.0

    def enable(self):
        if self._enabled:
            return
        self._enabled = True
        self._winding_down = False
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._stop.clear()
            self._thread.start()
        print("[OPTIMIZER] Enabled.")

    def disable(self):
        if not self._enabled and self._winding_down:
            return
        self._enabled = False
        self._winding_down = True
        print("[OPTIMIZER] Smart disabling...")

    def stop(self):
        self._stop.set()
        self.disable()
        self._thread.join(timeout=1)

    def set_force_freeze(self, enabled: bool):
        self._force_freeze = enabled
        mode = "FORCE-FREEZE" if enabled else "normal selective"
        print(f"[OPTIMIZER] Mode changed to {mode} by Presidential controller.")

    def get_suspended_count(self) -> int:
        return sum(1 for tp in self._tracked.values() if tp.suspended)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[OPTIMIZER] Tick error: {e}")
            time.sleep(self.scan_interval)

    def _can_suspend_based_on_baseline(self, snap: ProcessSnapshot, now: float) -> bool:
        name_key = snap.name.lower()
        info = self.baseline.get(name_key)
        if not info:
            return False
        first_seen = info.get("first_seen", now)
        age = now - first_seen
        if age < BASELINE_AGE_SECONDS:
            return False
        return True

    def _update_baseline_for_snapshot(self, snap: ProcessSnapshot, now: float):
        name_key = snap.name.lower()
        info = self.baseline.get(name_key)
        if not info:
            info = {"first_seen": now, "last_seen": now, "total_observed": 1}
        else:
            info["last_seen"] = now
            info["total_observed"] = info.get("total_observed", 0) + 1
        self.baseline[name_key] = info

    def _maybe_save_baseline(self, now: float):
        if now - self._last_baseline_save >= 60.0:
            self.memory.update_baseline_process_info(self.baseline)
            self.memory.save()
            self._last_baseline_save = now

    def _tick(self):
        now = time.time()

        if self._winding_down and not self._enabled:
            any_suspended = False
            resume_quota = 5
            for pid, tp in list(self._tracked.items()):
                if resume_quota <= 0:
                    break
                if tp.suspended:
                    self._resume_process(pid)
                    tp.suspended = False
                    resume_quota -= 1
                    any_suspended = True
            if not any_suspended:
                self._tracked.clear()
                self._winding_down = False
                print("[OPTIMIZER] Smart disable complete; all resumed.")
            self._maybe_save_baseline(now)
            return

        if not self._enabled:
            self._maybe_save_baseline(now)
            return

        snaps = scan_processes()
        current_pids = set()
        for snap in snaps:
            pid = snap.pid
            current_pids.add(pid)
            self._update_baseline_for_snapshot(snap, now)

            tp = self._tracked.get(pid)
            new_class = self.ai.classify(snap)
            if not tp:
                tp = TrackedProcess(snapshot=snap, classification=new_class)
                self._tracked[pid] = tp
            else:
                tp.snapshot = snap
                tp.classification = new_class

            if tp.classification in (ProcClass.JUNK, ProcClass.OCCASIONAL):
                if snap.cpu_percent < 1.0:
                    tp.idle_strikes += 1
                else:
                    tp.idle_strikes = 0

            if self._force_freeze:
                my_pid = os.getpid()
                if pid == my_pid or tp.classification is ProcClass.CRITICAL:
                    continue
                if (
                    not tp.suspended
                    and self._can_suspend_based_on_baseline(snap, now)
                ):
                    self._suspend_process(pid)
                    tp.suspended = True
                elif tp.suspended and (tp.classification is ProcClass.ACTIVE):
                    self._resume_process(pid)
                    tp.suspended = False
            else:
                if tp.classification in (ProcClass.JUNK, ProcClass.OCCASIONAL):
                    if (
                        not tp.suspended
                        and tp.idle_strikes >= IDLE_STRIKES_THRESHOLD
                        and self._can_suspend_based_on_baseline(snap, now)
                    ):
                        self._suspend_process(pid)
                        tp.suspended = True
                else:
                    if tp.suspended:
                        self._resume_process(pid)
                        tp.suspended = False

        for pid in list(self._tracked.keys()):
            if pid not in current_pids:
                del self._tracked[pid]

        self._maybe_save_baseline(now)

    def _suspend_process(self, pid: int):
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            proc.suspend()
            print(f"[OPTIMIZER] Suspended PID {pid} ({name}).")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[OPTIMIZER] Could not suspend {pid}: {e}")

    def _resume_process(self, pid: int):
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            proc.resume()
            print(f"[OPTIMIZER] Resumed PID {pid} ({name}).")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"[OPTIMIZER] Could not resume {pid}: {e}")


# --------------------------------------------------------------------
# Presidential controller
# --------------------------------------------------------------------

class PresidentialController:
    def __init__(
        self, memory: MemoryCore, optimizer: SystemOptimizer, booster: Booster
    ):
        self.memory = memory
        self.optimizer = optimizer
        self.booster = booster
        pres = self.memory.get_presidential_state()
        self._active = pres.get("enabled", False)

    def is_active(self) -> bool:
        return self._active

    def _set_muted_until(self, ts: float):
        pres = self.memory.get_presidential_state()
        pres["muted_until"] = float(ts)
        self.memory.update_presidential_state(pres)
        self.memory.save()

    def is_muted(self) -> bool:
        pres = self.memory.get_presidential_state()
        return time.time() < float(pres.get("muted_until", 0.0))

    def mute_for(self, seconds: float):
        self._set_muted_until(time.time() + float(seconds))
        print(f"[PRESIDENTIAL] Auto-trigger muted for {seconds:.0f} seconds.")

    def activate(self, source: str = "manual", reason: str = "Manual activation"):
        if self._active:
            return
        pres = self.memory.get_presidential_state()
        pres["enabled"] = True
        pres["last_trigger_source"] = source
        pres["last_trigger_reason"] = reason
        pres["last_activated_ts"] = time.time()
        self.memory.update_presidential_state(pres)
        self.memory.save()

        print(f"[PRESIDENTIAL] ACTIVATED ({source}) reason={reason}")
        self._active = True

        self.optimizer.set_force_freeze(True)

        if self.booster is not None:
            print("[PRESIDENTIAL] Booster stays active.")

        try:
            p = psutil.Process(os.getpid())
            if psutil.WINDOWS:
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                p.nice(-5)
            print("[PRESIDENTIAL] Guardian process priority raised.")
        except (psutil.AccessDenied, ValueError) as e:
            print(f"[PRESIDENTIAL] Could not raise priority: {e}")

    def deactivate(self, source: str = "manual", reason: str = "Manual deactivation"):
        if not self._active:
            return
        pres = self.memory.get_presidential_state()
        pres["enabled"] = False
        pres["last_trigger_source"] = source
        pres["last_trigger_reason"] = reason
        pres["last_deactivated_ts"] = time.time()
        self.memory.update_presidential_state(pres)
        self.memory.save()

        print(f"[PRESIDENTIAL] DEACTIVATED ({source}) reason={reason}")
        self._active = False

        self.optimizer.set_force_freeze(False)

        try:
            p = psutil.Process(os.getpid())
            if psutil.WINDOWS:
                p.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                p.nice(0)
            print("[PRESIDENTIAL] Guardian process priority normalized.")
        except (psutil.AccessDenied, ValueError) as e:
            print(f"[PRESIDENTIAL] Could not normalize priority: {e}")


# --------------------------------------------------------------------
# Look-ahead engine (with predictive upgrades)
# --------------------------------------------------------------------

class LookAheadEngine:
    """
    Predictive engine:
    - Learns event transitions (Markov-style).
    - Learns mode usage by hour-of-day.
    - Produces best-guess future mode with confidence.
    - Can preemptively trigger Presidential mode when prediction is strong.
    """

    def __init__(
        self,
        memory: MemoryCore,
        presidential_ref: PresidentialController,
        poll_interval: float = 5.0,
    ):
        self.memory = memory
        self.poll_interval = poll_interval
        self.presidential_ref = presidential_ref
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        print("[FORECAST] Look-ahead engine started.")
        if not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        print("[FORECAST] Stopping look-ahead engine...")
        self._stop.set()
        self._thread.join(timeout=1)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._observe_heavy_process()
                self._record_mode_usage()
                self._maybe_trigger_presidential()
                self._update_last_prediction()
            except Exception as e:
                print(f"[FORECAST] Error in loop: {e}")
            time.sleep(self.poll_interval)

    def _get_forecast_state(self) -> Dict[str, Any]:
        return self.memory.get_forecast_state()

    def _get_prediction_state(self) -> Dict[str, Any]:
        return self.memory.get_prediction_state()

    def register_event(self, event: str):
        fstate = self._get_forecast_state()
        last_event = fstate.get("last_event")
        transitions: Dict[str, Dict[str, int]] = fstate.setdefault(
            "transitions", {}
        )

        if last_event is not None:
            trans_from = transitions.setdefault(last_event, {})
            trans_from[event] = trans_from.get(event, 0) + 1

        fstate["last_event"] = event
        self.memory.update_forecast_state(fstate)
        self.memory.save()
        print(f"[FORECAST] Event recorded: {event} (last={last_event})")

    def _observe_heavy_process(self):
        snaps = scan_processes()
        if not snaps:
            return
        top = max(snaps, key=lambda s: s.cpu_percent)
        if top.cpu_percent < 5.0:
            return
        name = top.name.lower()
        event = f"proc:{name}"
        self.register_event(event)

    def _record_mode_usage(self):
        current_mode = self.memory.get_mode()
        hour = time.localtime().tm_hour
        pstate = self._get_prediction_state()
        mode_by_hour: Dict[str, Dict[str, int]] = pstate.setdefault("mode_by_hour", {})
        slot = mode_by_hour.setdefault(str(hour), {})
        slot[current_mode] = slot.get(current_mode, 0) + 1
        self.memory.update_prediction_state(pstate)
        self.memory.save()

    def _get_predictions_from_transitions(
        self, current: Optional[str], top_n: int = 3
    ) -> List[Tuple[str, int]]:
        fstate = self._get_forecast_state()
        transitions: Dict[str, Dict[str, int]] = fstate.get("transitions", {})
        if current is None or current not in transitions:
            return []
        dests = transitions[current]
        sorted_dests = sorted(dests.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_dests[:top_n]

    def _predict_mode_by_hour(self) -> Tuple[Optional[str], float]:
        pstate = self._get_prediction_state()
        mode_by_hour: Dict[str, Dict[str, int]] = pstate.get("mode_by_hour", {})
        hour = time.localtime().tm_hour
        slot = mode_by_hour.get(str(hour), {})
        if not slot:
            return None, 0.0
        total = sum(slot.values())
        if total <= 0:
            return None, 0.0
        mode, count = max(slot.items(), key=lambda kv: kv[1])
        confidence = count / total
        return mode, confidence

    def best_guess_next_mode(self) -> Tuple[Optional[str], float]:
        """
        Combine:
        - Hour-of-day mode usage
        - Transition table from current event
        """

        by_hour_mode, by_hour_conf = self._predict_mode_by_hour()
        current_mode = self.memory.get_mode()
        current_event = f"mode:{current_mode}"
        trans_preds = self._get_predictions_from_transitions(current_event, top_n=1)

        if not by_hour_mode and not trans_preds:
            return None, 0.0

        # Simple fusion: hour-of-day dominates, transitions refine if available.
        if by_hour_mode:
            fused_mode = by_hour_mode
            fused_conf = by_hour_conf
        else:
            # Only transitions available
            next_event, count = trans_preds[0]
            if next_event.startswith("mode:"):
                fused_mode = next_event.split(":", 1)[1]
            else:
                fused_mode = None
            # rough confidence from transitions
            total = sum(c for _, c in trans_preds)
            fused_conf = (count / total) if total > 0 else 0.0

        # If transitions strongly point to a different mode, blend
        if trans_preds:
            next_event, count = trans_preds[0]
            if next_event.startswith("mode:"):
                t_mode = next_event.split(":", 1)[1]
                if t_mode and by_hour_mode and t_mode != by_hour_mode:
                    fused_conf *= 0.8   # slight penalty for disagreement

        return fused_mode, fused_conf

    def _maybe_trigger_presidential(self):
        if self.presidential_ref.is_muted():
            return

        pres_state = self.memory.get_presidential_state()
        if not pres_state.get("auto_trigger", True):
            return

        current_mode = self.memory.get_mode()
        current_event = f"mode:{current_mode}"
        preds = self._get_predictions_from_transitions(current_event)
        if preds:
            print(f"[FORECAST] From {current_event}, likely next: {preds}")

        predicted_mode, conf = self.best_guess_next_mode()
        if predicted_mode:
            print(f"[FORECAST] Best guess next mode: {predicted_mode} (conf={conf:.2f})")

        system_cpu = psutil.cpu_percent(interval=0.0)

        strong_game_pred = predicted_mode == "game" and conf >= 0.7
        currently_game_heavy = current_mode == "game" and system_cpu > 50.0

        if not self.presidential_ref.is_active():
            if strong_game_pred and system_cpu > 30.0:
                reason = f"Predicted game mode soon (conf={conf:.2f}) with rising CPU"
                print("[FORECAST] Preemptively triggering Presidential Mode (prediction-based).")
                self.presidential_ref.activate(source="forecast", reason=reason)
            elif currently_game_heavy:
                reason = "Game mode with high CPU load"
                print("[FORECAST] Triggering Presidential Mode (current game load).")
                self.presidential_ref.activate(source="forecast", reason=reason)

    def _update_last_prediction(self):
        mode, conf = self.best_guess_next_mode()
        pstate = self._get_prediction_state()
        pstate["last_prediction"] = {
            "mode": mode,
            "confidence": conf,
            "ts": time.time(),
        }
        self.memory.update_prediction_state(pstate)
        self.memory.save()

    def get_learning_stats(self) -> Tuple[int, float]:
        fstate = self._get_forecast_state()
        transitions: Dict[str, Dict[str, int]] = fstate.get("transitions", {})
        total = 0
        max_branch = 0
        for _, dests in transitions.items():
            branch_sum = sum(dests.values())
            total += branch_sum
            if dests:
                local_max = max(dests.values())
                if local_max > max_branch:
                    max_branch = local_max
        confidence = (max_branch / total) if total > 0 else 0.0
        return total, confidence


# --------------------------------------------------------------------
# Network scanning + hive
# --------------------------------------------------------------------

@dataclass
class HostInfo:
    ip: str
    alive: bool
    last_seen: float
    latency_ms: Optional[float]


class PrometheusScanner:
    def __init__(
        self, cidr: str, sweep_interval: float = 10.0, max_workers: int = 64
    ):
        self.network = ipaddress.ip_network(cidr, strict=False)
        self.sweep_interval = sweep_interval
        self.max_workers = max_workers
        self.hosts: Dict[str, HostInfo] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[NETSCAN] Started Prometheus scanner on {self.network}")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        print("[NETSCAN] Stopped Prometheus scanner.")

    def _loop(self):
        while not self._stop.is_set():
            self._sweep_once()
            time.sleep(self.sweep_interval)

    def _sweep_once(self):
        work_q = queue.Queue()
        results_q = queue.Queue()
        for ip in self.network.hosts():
            work_q.put(str(ip))

        threads: List[threading.Thread] = []
        for _ in range(min(self.max_workers, work_q.qsize())):
            t = threading.Thread(
                target=self._worker_ping, args=(work_q, results_q), daemon=True
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        now = time.time()
        while not results_q.empty():
            ip, alive, latency = results_q.get()
            if ip not in self.hosts:
                self.hosts[ip] = HostInfo(
                    ip=ip,
                    alive=alive,
                    last_seen=now if alive else 0.0,
                    latency_ms=latency,
                )
            else:
                hi = self.hosts[ip]
                hi.alive = alive
                if alive:
                    hi.last_seen = now
                hi.latency_ms = latency

    def _worker_ping(self, work_q: queue.Queue, results_q: queue.Queue):
        while True:
            try:
                ip = work_q.get_nowait()
            except queue.Empty:
                break
            alive, latency = self._ping(ip)
            results_q.put((ip, alive, latency))

    def _ping(self, ip: str):
        system = platform.system().lower()
        if system == "windows":
            cmd = ["ping", "-n", "1", "-w", "500", ip]
        else:
            cmd = ["ping", "-c", "1", "-W", "1", ip]
        try:
            start = time.time()
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            end = time.time()
            alive = proc.returncode == 0
            latency = (end - start) * 1000.0 if alive else None
            return alive, latency
        except Exception:
            return False, None

    def snapshot(self) -> List[HostInfo]:
        return list(self.hosts.values())


@dataclass
class WorkerState:
    ip: str
    last_latency_ms: Optional[float]
    avg_latency_ms: Optional[float]
    last_seen: float
    alive: bool


class IPWorker(threading.Thread):
    def __init__(
        self, ip: str, queen_ref: "Queen", poll_interval: float = 5.0
    ):
        super().__init__(daemon=True)
        self.ip = ip
        self.queen_ref = queen_ref
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self.avg_latency_ms: Optional[float] = None

    def run(self):
        while not self._stop.is_set():
            hi = self.queen_ref.get_host_info(self.ip)
            if hi:
                self._update_from_host(hi)
            time.sleep(self.poll_interval)

    def stop(self):
        self._stop.set()

    def _update_from_host(self, hi: HostInfo):
        if hi.latency_ms is not None:
            if self.avg_latency_ms is None:
                self.avg_latency_ms = hi.latency_ms
            else:
                self.avg_latency_ms = (
                    self.avg_latency_ms * 0.7
                ) + (hi.latency_ms * 0.3)

        state = WorkerState(
            ip=self.ip,
            last_latency_ms=hi.latency_ms,
            avg_latency_ms=self.avg_latency_ms,
            last_seen=hi.last_seen,
            alive=hi.alive,
        )
        self.queen_ref.report_worker_state(state)


class Queen:
    def __init__(
        self, cidrs: List[str], memory: MemoryCore, sweep_interval: float = 10.0
    ):
        self.cidrs = cidrs
        self.scanners = [
            PrometheusScanner(c, sweep_interval=sweep_interval) for c in cidrs
        ]
        self.workers: Dict[str, IPWorker] = {}
        self.worker_states: Dict[str, WorkerState] = {}
        self._lock = threading.Lock()
        self.memory = memory

    def start(self):
        for s in self.scanners:
            s.start()
        threading.Thread(
            target=self._manage_workers_loop, daemon=True
        ).start()
        print("[QUEEN] Queen hive started.")

    def stop(self):
        print("[QUEEN] Stopping queen hive...")
        for w in list(self.workers.values()):
            w.stop()
        for s in self.scanners:
            s.stop()

    def _manage_workers_loop(self):
        while True:
            time.sleep(5)
            self._sync_workers_with_hosts()

    def _all_hosts_snapshot(self) -> List[HostInfo]:
        hosts: List[HostInfo] = []
        for s in self.scanners:
            hosts.extend(s.snapshot())
        return hosts

    def _sync_workers_with_hosts(self):
        current_hosts = self._all_hosts_snapshot()
        current_ips = {h.ip for h in current_hosts if h.alive}
        for ip in current_ips:
            if ip not in self.workers:
                w = IPWorker(ip, queen_ref=self)
                self.workers[ip] = w
                w.start()
                print(f"[QUEEN] Worker spawned for {ip}")
        for ip in list(self.workers.keys()):
            if ip not in current_ips:
                self.workers[ip].stop()
                del self.workers[ip]
                print(f"[QUEEN] Worker retired for {ip}")

    def get_host_info(self, ip: str) -> Optional[HostInfo]:
        for h in self._all_hosts_snapshot():
            if h.ip == ip:
                return h
        return None

    def report_worker_state(self, state: WorkerState):
        with self._lock:
            self.worker_states[state.ip] = state
        print(f"[QUEEN] State from {state.ip}: {asdict(state)}")
        self._persist_network_state()

    def _persist_network_state(self):
        with self._lock:
            net_state = {ip: asdict(ws) for ip, ws in self.worker_states.items()}
        self.memory.set_section("network", net_state)
        self.memory.save()

    def get_network_summary(self) -> List[WorkerState]:
        with self._lock:
            return list(self.worker_states.values())


# --------------------------------------------------------------------
# LAN detection
# --------------------------------------------------------------------

def detect_lan_cidrs() -> List[str]:
    cidrs = set()
    addrs = psutil.net_if_addrs()
    for iface, entries in addrs.items():
        for e in entries:
            fam_name = str(e.family)
            if "AF_INET" not in fam_name:
                continue
            ip = e.address
            netmask = e.netmask
            if not ip or not netmask:
                continue
            try:
                network = ipaddress.ip_network(f"{ip}/{netmask}", strict=False)
                if network.is_private:
                    cidrs.add(str(network))
            except Exception:
                continue
    if not cidrs:
        cidrs.update(
            ["192.168.0.0/24", "192.168.1.0/24", "10.0.0.0/24"]
        )
    return sorted(cidrs)


# --------------------------------------------------------------------
# Guardian mode
# --------------------------------------------------------------------

class GuardianMode:
    def __init__(self, memory: MemoryCore):
        self.memory = memory

    def is_override_enabled(self) -> bool:
        return self.memory.get_guardian_state().get("override_enabled", False)

    def set_override(self, enabled: bool):
        g = self.memory.get_guardian_state()
        g["override_enabled"] = enabled
        self.memory.update_guardian_state(g)
        self.memory.save()
        mode = "OVERRIDE (relaxed, manual, RED)" if enabled else "AUTO (strict, GREEN)"
        print(f"[GUARDIAN] Override set: {mode}")

    def _is_suspicious(
        self, method: str, url: str, headers: Dict[str, str], domain: str
    ) -> bool:
        filters = self.memory.get_filters()
        whitelist = {d.lower() for d in filters.get("whitelist", [])}
        if domain.lower() in whitelist:
            return False

        url_l = (url or "").lower()
        danger_patterns = [
            "../",
            "..\\",
            "select%20",
            "union%20",
            "<script",
            "%3cscript",
            "/wp-admin",
            "/phpmyadmin",
            "login",
            "attack",
            "exploit",
        ]
        if any(p in url_l for p in danger_patterns):
            return True

        if method not in ("GET", "POST", "HEAD", "CONNECT"):
            return True
        return False

    def handle_connection(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        client_socket: socket.socket,
        domain: str,
    ) -> bool:
        if self.is_override_enabled():
            return False
        try:
            if self._is_suspicious(method, url, headers, domain):
                print(
                    f"[GUARDIAN] Emergency redirect (U-turn) for suspicious request: {method} {url}"
                )
                resp = (
                    b"HTTP/1.1 302 Found\r\n"
                    b"Location: /\r\n"
                    b"Content-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
                client_socket.sendall(resp)
                return True
        except Exception as e:
            print(f"[GUARDIAN] Error in handle_connection: {e}")
        return False


# --------------------------------------------------------------------
# Stealth proxy
# --------------------------------------------------------------------

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8888
BUFFER_SIZE = 8192


def extract_host_domain(headers: Dict[str, str]) -> str:
    host = ""
    for k, v in headers.items():
        if k.lower() == "host":
            host = v.strip()
            break
    if not host:
        return ""
    if ":" in host:
        host, _ = host.split(":", 1)
    return host.lower()


class StealthProxy:
    def __init__(
        self,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        memory: Optional[MemoryCore] = None,
        cidrs: Optional[List[str]] = None,
    ):
        self.host = host
        self.port = port
        self.sidebrain = SideBrain()
        self.memory = memory or MemoryCore()
        self.memory.load()
        cidrs = cidrs or detect_lan_cidrs()
        self.queen = Queen(cidrs, self.memory)
        self.queen.start()
        self.guardian = GuardianMode(self.memory)
        self._server_socket: Optional[socket.socket] = None
        self._stop = threading.Event()

    def start(self):
        self.sidebrain.start()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(128)
        print(f"[PROXY] Listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                client_socket, addr = self._server_socket.accept()
            except OSError:
                break
            threading.Thread(
                target=self.handle_client, args=(client_socket, addr), daemon=True
            ).start()

    def stop(self):
        self._stop.set()
        if self._server_socket:
            self._server_socket.close()
        self.sidebrain.stop()
        self.queen.stop()

    def _record_block(self, domain: str, reason: str):
        domain = (domain or "").lower()
        if domain:
            self.memory.add_blocked_entry(domain, reason)

    def handle_client(self, client_socket: socket.socket, addr):
        try:
            request = client_socket.recv(BUFFER_SIZE)
            if not request:
                client_socket.close()
                return
            first_line = request.split(b"\r\n", 1)[0].decode(errors="ignore")
            parts = first_line.split()
            if len(parts) < 2:
                client_socket.close()
                return
            method, url = parts[0], parts[1]
            headers: Dict[str, str] = {}
            try:
                header_block = request.split(b"\r\n\r\n", 1)[0].decode(errors="ignore")
                for line in header_block.split("\r\n")[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip()] = v.strip()
            except Exception:
                pass

            domain = extract_host_domain(headers)

            meta = RequestMeta(
                client_addr=str(addr),
                method=method,
                url=url,
                headers=headers,
            )
            self.sidebrain.submit(meta)

            filters = self.memory.get_filters()
            whitelist = {d.lower() for d in filters.get("whitelist", [])}
            blacklist = {d.lower() for d in filters.get("blacklist", [])}

            if domain in whitelist:
                self.forward_request(client_socket, request)
                return

            if domain in blacklist:
                print(f"[PROXY] Domain {domain} is BLACKLISTED. Returning 204.")
                self._record_block(domain, "blacklist")
                self.respond_dummy(client_socket)
                return

            if self.guardian.handle_connection(
                method, url, headers, client_socket, domain
            ):
                self._record_block(domain, "guardian")
                return

            if self.is_irrelevant(url, domain):
                self._record_block(domain, "ad filter")
                self.respond_dummy(client_socket)
                return

            self.forward_request(client_socket, request)
        except Exception as e:
            print(f"[PROXY] Error: {e}")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    def is_irrelevant(self, url: str, domain: str) -> bool:
        url_l = (url or "").lower()
        domain_l = (domain or "").lower()

        ad_domains = [
            "doubleclick.net",
            "googlesyndication.com",
            "adservice.google.com",
            "ads.yahoo.com",
        ]
        if any(domain_l.endswith(ad_dom) for ad_dom in ad_domains):
            print(f"[PROXY] Blocking ad domain: {domain_l}")
            return True

        ad_substrings = [
            "doubleclick",
            "adservice",
            "/ads?",
            "/ad?",
            "/advertising",
        ]
        if any(s in url_l for s in ad_substrings):
            print(f"[PROXY] Blocking ad URL: {url_l}")
            return True

        return False

    def respond_dummy(self, client_socket: socket.socket):
        resp = (
            b"HTTP/1.1 204 No Content\r\n"
            b"Content-Length: 0\r\n"
            b"Connection: close\r\n\r\n"
        )
        try:
            client_socket.sendall(resp)
        except Exception:
            pass

    def forward_request(self, client_socket: socket.socket, request: bytes):
        host_line = [
            h
            for h in request.split(b"\r\n")
            if h.lower().startswith(b"host:")
        ]
        if not host_line:
            return
        host_value = host_line[0].decode().split(":", 1)[1].strip()
        if ":" in host_value:
            remote_host, remote_port_str = host_value.split(":", 1)
            remote_port = int(remote_port_str)
        else:
            remote_host, remote_port = host_value, 80

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as remote_socket:
            remote_socket.connect((remote_host, remote_port))
            remote_socket.sendall(request)
            while True:
                data = remote_socket.recv(BUFFER_SIZE)
                if not data:
                    break
                client_socket.sendall(data)


# --------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------

class StealthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stealth Hive Guardian")

        self.memory = MemoryCore()
        self.memory.load()

        self.proxy: Optional[StealthProxy] = None
        self.proxy_thread: Optional[threading.Thread] = None
        self.running_proxy = False

        self.optimizer: SystemOptimizer = SystemOptimizer(memory=self.memory)
        self.optimizer.enable()

        self.booster: Booster = Booster(memory=self.memory)
        self.boost_enabled = False

        self.presidential = PresidentialController(
            self.memory, self.optimizer, self.booster
        )
        self.lookahead = LookAheadEngine(
            self.memory, self.presidential
        )
        self.lookahead.start()

        self.auto_lan = True
        self.lan_cidrs = detect_lan_cidrs()

        ui_state = self.memory.get_ui_state()
        self.filters = self.memory.get_filters()

        # Auto-protect Steam/Epic
        wl_set = set(d.lower() for d in self.filters.get("whitelist", []))
        changed = False
        for dom in GAME_PROTECT_DOMAINS:
            d = dom.lower()
            if d not in wl_set:
                wl_set.add(d)
                changed = True
        if changed:
            self.filters["whitelist"] = sorted(wl_set)
            self.memory.update_filters(self.filters)
            self.memory.save()

        self.blocked_log: List[Dict[str, Any]] = list(self.memory.get_blocked_log())

        self._auto_mode_enabled = self.memory.get_auto_mode_enabled()
        self._auto_mode_current = self.memory.get_mode()
        self._auto_mode_candidate = self._auto_mode_current
        self._auto_mode_candidate_ticks = 0

        pstate = self.memory.get_prediction_state()
        self.best_guess_enabled = pstate.get("best_guess_enabled", True)

        self._build_layout(ui_state)
        self._log("Ready.")

        if ui_state.get("boost_enabled", True):
            self.booster.start()
            self.boost_enabled = True
            self.boost_var.set(True)
            self._log("Adaptive system boost AUTO-ENABLED from last session.")

        self._schedule_learning_update()
        self._schedule_auto_mode_tick()
        self._schedule_blocked_log_refresh()
        self._schedule_presidential_diagnostics_refresh()
        self._schedule_prediction_status_refresh()

    def _build_layout(self, ui_state: Dict[str, Any]):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=5)

        self.start_btn = tk.Button(
            top_frame, text="Start Proxy", command=self.start_proxy
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(
            top_frame,
            text="Stop Proxy",
            command=self.stop_proxy,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=5)

        self.status_label = tk.Label(top_frame, text="Proxy: Stopped")
        self.status_label.pack(side="left", padx=10)

        self.boost_var = tk.BooleanVar(value=ui_state.get("boost_enabled", True))
        self.boost_check = tk.Checkbutton(
            top_frame,
            text="Adaptive Boost (3–10s)",
            variable=self.boost_var,
            command=self.toggle_boost,
        )
        self.boost_check.pack(side="right", padx=5)

        self.opt_var = tk.BooleanVar(value=True)
        self.opt_check = tk.Checkbutton(
            top_frame,
            text="Freeze Junk/Occasional (baseline-aware)",
            variable=self.opt_var,
            command=self.toggle_optimizer,
        )
        self.opt_check.select()
        self.opt_check.pack(side="right", padx=5)

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(fill="x", padx=10, pady=5)

        self.ai_mode_var = tk.BooleanVar(value=self._auto_mode_enabled)
        self.ai_mode_check = tk.Checkbutton(
            mode_frame,
            text="AI Mode (Automatic)",
            variable=self.ai_mode_var,
            command=self.toggle_ai_mode,
        )
        self.ai_mode_check.pack(side="left", padx=5)

        self.mode_var = tk.StringVar(value=self.memory.get_mode())
        tk.Label(mode_frame, text="Manual Mode:").pack(side="left", padx=5)
        self.mode_buttons: List[tk.Radiobutton] = []
        for m in ["idle", "game", "browse", "work"]:
            rb = tk.Radiobutton(
                mode_frame,
                text=m.capitalize(),
                variable=self.mode_var,
                value=m,
                command=self.change_mode_manual,
            )
            rb.pack(side="left")
            self.mode_buttons.append(rb)

        self.current_mode_label = tk.Label(
            mode_frame, text=f"Current: {self.memory.get_mode()}"
        )
        self.current_mode_label.pack(side="left", padx=10)

        # Best guess mode toggle + status
        self.best_guess_var = tk.BooleanVar(value=self.best_guess_enabled)
        self.best_guess_check = tk.Checkbutton(
            mode_frame,
            text="Best Guess Mode (Predictive)",
            variable=self.best_guess_var,
            command=self.toggle_best_guess_mode,
        )
        self.best_guess_check.pack(side="right", padx=5)

        self._update_mode_controls_state()

        learn_frame = tk.Frame(self.root)
        learn_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(learn_frame, text="Learning Progress:").pack(
            side="left"
        )
        self.learn_bar = ttk.Progressbar(
            learn_frame,
            orient="horizontal",
            length=200,
            mode="determinate",
            maximum=100,
        )
        self.learn_bar.pack(side="left", padx=5)
        self.learn_label = tk.Label(
            learn_frame, text="0 events (0%)"
        )
        self.learn_label.pack(side="left", padx=10)

        # Prediction status
        self.prediction_label_var = tk.StringVar(value="Prediction: (learning...)")
        tk.Label(learn_frame, textvariable=self.prediction_label_var).pack(side="left", padx=10)

        pres_frame = tk.LabelFrame(self.root, text="Presidential Mode")
        pres_frame.pack(fill="x", padx=10, pady=5)

        self.pres_var = tk.BooleanVar(
            value=self.memory.get_presidential_state().get(
                "enabled", False
            )
        )
        self.pres_check = tk.Checkbutton(
            pres_frame,
            text="Presidential Mode (OS-protective)",
            variable=self.pres_var,
            command=self.toggle_presidential_manual,
        )
        self.pres_check.pack(side="left", padx=5)

        self.force_exit_btn = tk.Button(
            pres_frame,
            text="Force Exit Presidential Mode",
            command=self.force_exit_presidential,
        )
        self.force_exit_btn.pack(side="left", padx=5)

        self.pres_status_var = tk.StringVar()
        self.pres_reason_var = tk.StringVar()
        self.pres_freeze_var = tk.StringVar()
        self.pres_suspended_var = tk.StringVar()

        diag_frame = tk.Frame(pres_frame)
        diag_frame.pack(fill="x", padx=5, pady=3)

        tk.Label(diag_frame, text="Status:").grid(row=0, column=0, sticky="w")
        tk.Label(diag_frame, textvariable=self.pres_status_var).grid(row=0, column=1, sticky="w")

        tk.Label(diag_frame, text="Trigger:").grid(row=1, column=0, sticky="w")
        tk.Label(diag_frame, textvariable=self.pres_reason_var).grid(row=1, column=1, sticky="w")

        tk.Label(diag_frame, text="Freeze Mode:").grid(row=2, column=0, sticky="w")
        tk.Label(diag_frame, textvariable=self.pres_freeze_var).grid(row=2, column=1, sticky="w")

        tk.Label(diag_frame, text="Suspended Processes:").grid(row=3, column=0, sticky="w")
        tk.Label(diag_frame, textvariable=self.pres_suspended_var).grid(row=3, column=1, sticky="w")

        self._update_presidential_color()

        guard_frame = tk.Frame(self.root)
        guard_frame.pack(fill="x", padx=10, pady=5)

        guardian_state = self.memory.get_guardian_state()
        self.guard_var = tk.BooleanVar(
            value=guardian_state.get("override_enabled", False)
        )
        self.guard_check = tk.Checkbutton(
            guard_frame,
            text="Guardian Override (RED=Override, GREEN=Auto)",
            variable=self.guard_var,
            command=self.toggle_guardian_override,
        )
        self.guard_check.pack(side="left", padx=5)
        self._update_guardian_color()

        lan_frame = tk.Frame(self.root)
        lan_frame.pack(fill="x", padx=10, pady=5)

        self.auto_lan_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            lan_frame,
            text="Auto-detect LAN ranges",
            variable=self.auto_lan_var,
            command=self.toggle_auto_lan,
        ).pack(side="left")

        self.lan_label = tk.Label(
            lan_frame, text=f"LAN: {', '.join(self.lan_cidrs)}"
        )
        self.lan_label.pack(side="left", padx=10)

        tk.Button(
            lan_frame, text="Add Manual Range", command=self.add_manual_range
        ).pack(side="right")

        filter_frame = tk.LabelFrame(self.root, text="Domain Filters (Whitelist / Blacklist)")
        filter_frame.pack(fill="x", padx=10, pady=5)

        wl_frame = tk.Frame(filter_frame)
        wl_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(wl_frame, text="Whitelist (always allow)").pack(anchor="w")
        self.wl_listbox = tk.Listbox(wl_frame, height=6, selectmode=tk.SINGLE)
        self.wl_listbox.pack(fill="both", expand=True)
        for d in self.filters.get("whitelist", []):
            self.wl_listbox.insert("end", d)

        bl_frame = tk.Frame(filter_frame)
        bl_frame.pack(side="left", fill="both", expand=True, padx=5)

        tk.Label(bl_frame, text="Blacklist (always block)").pack(anchor="w")
        self.bl_listbox = tk.Listbox(bl_frame, height=6, selectmode=tk.SINGLE)
        self.bl_listbox.pack(fill="both", expand=True)
        for d in self.filters.get("blacklist", []):
            self.bl_listbox.insert("end", d)

        btn_frame = tk.Frame(filter_frame)
        btn_frame.pack(side="left", fill="y", padx=5)

        tk.Button(btn_frame, text="Add to Whitelist", command=self.add_to_whitelist).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Remove from Whitelist", command=self.remove_from_whitelist).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Add to Blacklist", command=self.add_to_blacklist).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Remove from Blacklist", command=self.remove_from_blacklist).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="→ To Whitelist", command=self.move_black_to_white).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="→ To Blacklist", command=self.move_white_to_black).pack(fill="x", pady=2)

        blocked_frame = tk.LabelFrame(self.root, text="Recently Blocked (last 100)")
        blocked_frame.pack(fill="x", padx=10, pady=5)

        self.blocked_listbox = tk.Listbox(blocked_frame, height=6, selectmode=tk.SINGLE)
        self.blocked_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=3)

        blocked_btn_frame = tk.Frame(blocked_frame)
        blocked_btn_frame.pack(side="left", fill="y", padx=5)

        tk.Button(blocked_btn_frame, text="Whitelist from Log", command=self.whitelist_from_log).pack(fill="x", pady=2)
        tk.Button(blocked_btn_frame, text="Blacklist from Log", command=self.blacklist_from_log).pack(fill="x", pady=2)
        tk.Button(blocked_btn_frame, text="Clear Log", command=self.clear_blocked_log).pack(fill="x", pady=2)

        self._refresh_blocked_listbox()

        backup_frame = tk.Frame(self.root)
        backup_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(
            backup_frame,
            text="Set Backup Memory Location",
            command=self.set_backup_location,
        ).pack(side="left")

        log_frame = tk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log = scrolledtext.ScrolledText(
            log_frame, wrap="word", height=10
        )
        self.log.pack(fill="both", expand=True)

    def _log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    # ---------- Proxy ----------

    def start_proxy(self):
        if self.running_proxy:
            return
        cidrs = (
            self.lan_cidrs
            if not self.auto_lan_var.get()
            else detect_lan_cidrs()
        )
        self.lan_cidrs = cidrs
        self.lan_label.config(text=f"LAN: {', '.join(self.lan_cidrs)}")

        def run_proxy():
            self.proxy = StealthProxy(
                cidrs=cidrs, memory=self.memory
            )
            if self.proxy:
                self.proxy.guardian.set_override(self.guard_var.get())
            self.proxy.start()

        self.proxy_thread = threading.Thread(
            target=run_proxy, daemon=True
        )
        self.proxy_thread.start()
        self.running_proxy = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Proxy: Running")
        self._log("Proxy started on 127.0.0.1:8888")

    def stop_proxy(self):
        if not self.running_proxy:
            return
        if self.proxy:
            self.proxy.stop()
        self.running_proxy = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Proxy: Stopped")
        self._log("Proxy stopped.")

    # ---------- Booster ----------

    def toggle_boost(self):
        desired = self.boost_var.get()
        ui_state = self.memory.get_ui_state()
        if desired and not self.boost_enabled:
            if self.booster:
                self.booster.start()
            self.boost_enabled = True
            ui_state["boost_enabled"] = True
            self._log("Adaptive system boost ENABLED.")
        elif not desired and self.boost_enabled:
            if self.booster:
                self.booster.stop()
            self.boost_enabled = False
            ui_state["boost_enabled"] = False
            self._log("Adaptive system boost DISABLED.")
        self.memory.update_ui_state(ui_state)
        self.memory.save()

    # ---------- Optimizer ----------

    def toggle_optimizer(self):
        desired = self.opt_var.get()
        if desired and not self.optimizer._enabled:
            self.optimizer.enable()
            self._log("Optimizer ENABLED (baseline-aware).")
        elif not desired and self.optimizer._enabled:
            self.optimizer.disable()
            self._log("Optimizer DISABLED (smart resume).")

    # ---------- AI Mode / Best Guess ----------

    def _update_mode_controls_state(self):
        auto = self.ai_mode_var.get()
        state = tk.DISABLED if auto else tk.NORMAL
        for rb in self.mode_buttons:
            rb.config(state=state)

    def toggle_ai_mode(self):
        enabled = self.ai_mode_var.get()
        self._auto_mode_enabled = enabled
        self.memory.set_auto_mode_enabled(enabled)
        self.memory.save()
        self._update_mode_controls_state()
        if enabled:
            self._log("AI Mode ENABLED.")
        else:
            self._log("AI Mode DISABLED; manual mode.")

    def toggle_best_guess_mode(self):
        self.best_guess_enabled = self.best_guess_var.get()
        pstate = self.memory.get_prediction_state()
        pstate["best_guess_enabled"] = self.best_guess_enabled
        self.memory.update_prediction_state(pstate)
        self.memory.save()
        if self.best_guess_enabled:
            self._log("Best Guess Mode ENABLED (predictive preemptive behavior).")
        else:
            self._log("Best Guess Mode DISABLED.")

    def change_mode_manual(self):
        if self._auto_mode_enabled:
            return
        mode = self.mode_var.get()
        self._set_mode(mode, source="manual")

    def _set_mode(self, mode: str, source: str = "unknown"):
        self.memory.set_mode(mode)
        self.memory.save()
        self.current_mode_label.config(text=f"Current: {mode}")
        self.mode_var.set(mode)
        event = f"mode:{mode}"
        self.lookahead.register_event(event)
        self._log(f"Mode set to: {mode} ({source})")

    def _auto_detect_mode(self) -> str:
        try:
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            cpu = 0.0

        try:
            net = psutil.net_io_counters()
            net_bytes = net.bytes_sent + net.bytes_recv
        except Exception:
            net_bytes = 0

        if not hasattr(self, "_last_net_bytes"):
            self._last_net_bytes = net_bytes
            net_delta = 0
        else:
            net_delta = max(0, net_bytes - self._last_net_bytes)
            self._last_net_bytes = net_bytes

        top_name = ""
        top_cpu = 0.0
        try:
            for p in psutil.process_iter(["name", "cpu_percent"]):
                info = p.info
                name = (info.get("name") or "").lower()
                c = info.get("cpu_percent") or 0.0
                if c > top_cpu:
                    top_cpu = c
                    top_name = name
        except Exception:
            pass

        name = top_name

        is_browser = any(
            x in name
            for x in [
                "chrome",
                "edge",
                "firefox",
                "brave",
                "opera",
                "vivaldi",
            ]
        )
        is_game = any(
            x in name
            for x in [
                "steam",
                "epicgames",
                "eldenring",
                "fortnite",
                "battle.net",
                "leagueoflegends",
                "valorant",
                "genshin",
                "game",
            ]
        )
        is_work = any(
            x in name
            for x in [
                "code",
                "pycharm",
                "idea",
                "devenv",
                "notepad++",
                "sublime",
                "word",
                "excel",
                "powerpnt",
                "onedrive",
                "teams",
                "outlook",
            ]
        )

        if is_game or (cpu > 60 and top_cpu > 50):
            return "game"
        if is_browser or net_delta > 200000:
            return "browse"
        if is_work or (cpu > 25 and top_cpu > 20):
            return "work"
        if cpu < 5 and net_delta < 10000:
            return "idle"

        return self.memory.get_mode()

    def _auto_mode_tick(self):
        if self._auto_mode_enabled:
            predicted = self._auto_detect_mode()
            if predicted == self._auto_mode_candidate:
                self._auto_mode_candidate_ticks += 1
            else:
                self._auto_mode_candidate = predicted
                self._auto_mode_candidate_ticks = 1

            if (
                self._auto_mode_candidate_ticks >= AI_MODE_STABLE_TICKS
                and predicted != self._auto_mode_current
            ):
                self._auto_mode_current = predicted
                self._set_mode(predicted, source="AI")
        self._schedule_auto_mode_tick()

    def _schedule_auto_mode_tick(self):
        self.root.after(AI_MODE_POLL_MS, self._auto_mode_tick)

    # ---------- LAN ----------

    def toggle_auto_lan(self):
        self.auto_lan = self.auto_lan_var.get()
        if self.auto_lan:
            self.lan_cidrs = detect_lan_cidrs()
            self.lan_label.config(text=f"LAN: {', '.join(self.lan_cidrs)}")
            self._log("Auto LAN detection ENABLED.")
        else:
            self._log("Auto LAN detection DISABLED; manual only.")

    def add_manual_range(self):
        cidr = simpledialog.askstring(
            "Manual LAN Range",
            "Enter CIDR (e.g. 192.168.100.0/24):",
            parent=self.root,
        )
        if cidr:
            self.lan_cidrs.append(cidr)
            self.lan_cidrs = sorted(set(self.lan_cidrs))
            self.lan_label.config(text=f"LAN: {', '.join(self.lan_cidrs)}")
            self._log(f"Added manual LAN range: {cidr}")

    # ---------- Filters ----------

    def _save_filters(self):
        self.memory.update_filters(self.filters)
        self.memory.save()

    def add_to_whitelist(self):
        domain = simpledialog.askstring(
            "Add Whitelist Domain",
            "Enter domain (e.g. example.com):",
            parent=self.root,
        )
        if domain:
            d = domain.strip().lower()
            if d and d not in self.filters["whitelist"]:
                self.filters["whitelist"].append(d)
                if d in self.filters["blacklist"]:
                    self.filters["blacklist"].remove(d)
                self.wl_listbox.insert("end", d)
                idxs = [i for i in range(self.bl_listbox.size()) if self.bl_listbox.get(i) == d]
                for i in reversed(idxs):
                    self.bl_listbox.delete(i)
                self._save_filters()
                self._log(f"Added to whitelist: {d}")

    def remove_from_whitelist(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        d = self.wl_listbox.get(idx)
        self.wl_listbox.delete(idx)
        if d in self.filters["whitelist"]:
            self.filters["whitelist"].remove(d)
        self._save_filters()
        self._log(f"Removed from whitelist: {d}")

    def add_to_blacklist(self):
        domain = simpledialog.askstring(
            "Add Blacklist Domain",
            "Enter domain (e.g. ads.badsite.com):",
            parent=self.root,
        )
        if domain:
            d = domain.strip().lower()
            if d and d not in self.filters["blacklist"]:
                self.filters["blacklist"].append(d)
                if d in self.filters["whitelist"]:
                    self.filters["whitelist"].remove(d)
                self.bl_listbox.insert("end", d)
                idxs = [i for i in range(self.wl_listbox.size()) if self.wl_listbox.get(i) == d]
                for i in reversed(idxs):
                    self.wl_listbox.delete(i)
                self._save_filters()
                self._log(f"Added to blacklist: {d}")

    def remove_from_blacklist(self):
        sel = self.bl_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        d = self.bl_listbox.get(idx)
        self.bl_listbox.delete(idx)
        if d in self.filters["blacklist"]:
            self.filters["blacklist"].remove(d)
        self._save_filters()
        self._log(f"Removed from blacklist: {d}")

    def move_black_to_white(self):
        sel = self.bl_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        d = self.bl_listbox.get(idx)
        self.bl_listbox.delete(idx)
        if d in self.filters["blacklist"]:
            self.filters["blacklist"].remove(d)
        if d not in self.filters["whitelist"]:
            self.filters["whitelist"].append(d)
            self.wl_listbox.insert("end", d)
        self._save_filters()
        self._log(f"Moved {d} from blacklist to whitelist")

    def move_white_to_black(self):
        sel = self.wl_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        d = self.wl_listbox.get(idx)
        self.wl_listbox.delete(idx)
        if d in self.filters["whitelist"]:
            self.filters["whitelist"].remove(d)
        if d not in self.filters["blacklist"]:
            self.filters["blacklist"].append(d)
            self.bl_listbox.insert("end", d)
        self._save_filters()
        self._log(f"Moved {d} from whitelist to blacklist")

    # ---------- Blocked log ----------

    def _refresh_blocked_listbox(self):
        self.blocked_listbox.delete(0, "end")
        for entry in self.blocked_log:
            line = f"{entry.get('domain', '')} ({entry.get('reason', '')})"
            self.blocked_listbox.insert("end", line)

    def _refresh_blocked_log_from_memory(self):
        latest = self.memory.get_blocked_log()
        if len(latest) != len(self.blocked_log) or any(
            latest[i] != self.blocked_log[i] for i in range(len(latest))
        ):
            self.blocked_log = list(latest)
            self._refresh_blocked_listbox()

    def _schedule_blocked_log_refresh(self):
        self._refresh_blocked_log_from_memory()
        self.root.after(3000, self._schedule_blocked_log_refresh)

    def whitelist_from_log(self):
        sel = self.blocked_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < 0 or idx >= len(self.blocked_log):
            return
        domain = self.blocked_log[idx].get("domain", "").strip().lower()
        if not domain:
            return
        if domain not in self.filters["whitelist"]:
            self.filters["whitelist"].append(domain)
            if domain in self.filters["blacklist"]:
                self.filters["blacklist"].remove(domain)
            self.wl_listbox.insert("end", domain)
            idxs = [i for i in range(self.bl_listbox.size()) if self.bl_listbox.get(i) == domain]
            for i in reversed(idxs):
                self.bl_listbox.delete(i)
            self._save_filters()
            self._log(f"[BLOCK LOG] {domain} moved to whitelist from blocked log.")

    def blacklist_from_log(self):
        sel = self.blocked_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < 0 or idx >= len(self.blocked_log):
            return
        domain = self.blocked_log[idx].get("domain", "").strip().lower()
        if not domain:
            return
        if domain not in self.filters["blacklist"]:
            self.filters["blacklist"].append(domain)
            if domain in self.filters["whitelist"]:
                self.filters["whitelist"].remove(domain)
            self.bl_listbox.insert("end", domain)
            idxs = [i for i in range(self.wl_listbox.size()) if self.wl_listbox.get(i) == domain]
            for i in reversed(idxs):
                self.wl_listbox.delete(i)
            self._save_filters()
            self._log(f"[BLOCK LOG] {domain} moved to blacklist from blocked log.")

    def clear_blocked_log(self):
        self.blocked_log = []
        self.memory.state["blocked_log"] = []
        self.memory.save()
        self._refresh_blocked_listbox()
        self._log("Blocked activity log cleared.")

    # ---------- Backup ----------

    def set_backup_location(self):
        if self.memory.backup_file:
            self._log(f"Backup memory already set: {self.memory.backup_file}")
            return

        path = filedialog.asksaveasfilename(
            title="Select backup memory file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.memory.set_backup_path(path)
            self.memory.save()
            self._log(f"Backup memory location set: {path}")

    # ---------- Presidential Mode ----------

    def toggle_presidential_manual(self):
        desired = self.pres_var.get()
        if desired and not self.presidential.is_active():
            self.presidential.activate(source="manual", reason="Manual activation")
            self._log("Presidential Mode ACTIVATED manually.")
        elif not desired and self.presidential.is_active():
            self.presidential.deactivate(source="manual", reason="Manual deactivation")
            self._log("Presidential Mode DEACTIVATED manually.")
        self._update_presidential_color()

    def force_exit_presidential(self):
        if self.presidential.is_active():
            self.presidential.deactivate(source="override", reason="Emergency bypass")
            self.presidential.mute_for(30)
            self.pres_var.set(False)
            self._log("Emergency bypass: Presidential Mode disabled for 30 seconds.")
        self._update_presidential_color()

    def _update_presidential_color(self):
        if self.pres_var.get():
            self.pres_check.config(bg="red", activebackground="red", fg="white", selectcolor="red")
        else:
            self.pres_check.config(bg="green", activebackground="green", fg="black", selectcolor="green")

    def _update_presidential_diagnostics(self):
        pres = self.memory.get_presidential_state()
        active = pres.get("enabled", False)
        source = pres.get("last_trigger_source", "None")
        reason = pres.get("last_trigger_reason", "None")
        freeze_mode = "Force-Freeze" if self.optimizer._force_freeze else "Selective"
        suspended = self.optimizer.get_suspended_count()

        self.pres_status_var.set("ACTIVE" if active else "Inactive")
        self.pres_reason_var.set(f"{source} — {reason}")
        self.pres_freeze_var.set(freeze_mode)
        self.pres_suspended_var.set(str(suspended))

    def _schedule_presidential_diagnostics_refresh(self):
        self._update_presidential_diagnostics()
        self.root.after(2000, self._schedule_presidential_diagnostics_refresh)

    # ---------- Guardian ----------

    def toggle_guardian_override(self):
        enabled = self.guard_var.get()
        if self.proxy and self.proxy.guardian:
            self.proxy.guardian.set_override(enabled)
        else:
            g = self.memory.get_guardian_state()
            g["override_enabled"] = enabled
            self.memory.update_guardian_state(g)
            self.memory.save()
        self._update_guardian_color()
        if enabled:
            self._log("Guardian Override ENABLED (RED) — Guardian relaxed.")
        else:
            self._log("Guardian Override DISABLED (GREEN) — Guardian automatic.")

    def _update_guardian_color(self):
        if self.guard_var.get():
            self.guard_check.config(bg="red", activebackground="red", fg="white", selectcolor="red")
        else:
            self.guard_check.config(bg="green", activebackground="green", fg="black", selectcolor="green")

    # ---------- Learning / Prediction ----------

    def _update_learning_meter(self):
        try:
            total_events, confidence = self.lookahead.get_learning_stats()
        except Exception:
            total_events, confidence = 0, 0.0

        capped = min(total_events, 200)
        progress_pct = int((capped / 200.0) * 100) if capped > 0 else 0
        blended = int((progress_pct * 0.7) + (confidence * 100 * 0.3))

        self.learn_bar["value"] = blended
        self.learn_label.config(text=f"{total_events} events ({blended}%)")

    def _schedule_learning_update(self):
        self._update_learning_meter()
        self.root.after(3000, self._schedule_learning_update)

    def _update_prediction_status(self):
        if not self.best_guess_enabled:
            self.prediction_label_var.set("Prediction: disabled")
            return

        pstate = self.memory.get_prediction_state()
        last = pstate.get("last_prediction") or {}
        mode = last.get("mode")
        conf = last.get("confidence", 0.0)
        if mode:
            self.prediction_label_var.set(f"Prediction: {mode} (conf={conf:.2f})")
        else:
            self.prediction_label_var.set("Prediction: gathering data...")

    def _schedule_prediction_status_refresh(self):
        self._update_prediction_status()
        self.root.after(3000, self._schedule_prediction_status_refresh)

    # ---------- Window close ----------

    def on_close(self):
        try:
            self._log("Shutting down…")
            if self.running_proxy and self.proxy:
                self.proxy.stop()
            if self.boost_enabled and self.booster:
                self.booster.stop()
            if self.optimizer:
                self.optimizer.stop()
            if self.lookahead:
                self.lookahead.stop()
        except Exception as e:
            print(f"[CLOSE] Error during shutdown: {e}")
        finally:
            self.root.destroy()


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = StealthGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

