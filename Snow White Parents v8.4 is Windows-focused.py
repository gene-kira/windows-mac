"""
Snow White Parents - MagicBox Refactor (M2 List+Details, R1 Heavy, No Self-Repair)
----------------------------------------------------------------------------------

Architecture:
- Core engine: metrics, delivery, cortex, policy, trust
- MemoryStore: primary + unlimited backups, encryption + JSON, merge, cleanup
- BaselineScanner: Windows inventory
- GUI: Tkinter, operator console style

GUI layout:
- Top: status bar (health, deliveries, blocks)
- Left: navigation (Dashboard, Security, Network, Memory, Settings)
- Right: stacked views
- Memory view (M2 List+Details):
    - Left: list of backup paths
    - Right: details for selected path (type, root status, state)
    - Controls: Add, Remove, Scan Removable, Save

Persistence:
- Encrypted state file: swp_state.json.enc (all state)
- backup_paths.json (plain paths, merged with encrypted list)
- Versioned copies: backup_paths_versions/backup_paths_YYYYMMDD_HHMMSS.json
- Unlimited backups, deduplicated, dead roots removed

Self-repair has been completely removed: no hashing, no program backup, no startup blocking.
"""

import os
import sys
import platform
import threading
import time
import uuid
import json
import base64
import secrets
import hashlib
import datetime

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path

import requests
import psutil
import tkinter as tk
from tkinter import ttk
import urllib.parse as urllib_parse

# Windows registry & drive type
try:
    import winreg  # type: ignore
except ImportError:
    winreg = None

if platform.system().lower() == "windows":
    import ctypes
    from ctypes import wintypes
    GetDriveTypeW = ctypes.windll.kernel32.GetDriveTypeW
    GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    GetDriveTypeW.restype = wintypes.UINT
else:
    GetDriveTypeW = None


# ---------- helpers ----------

def get_system_summary() -> Dict[str, str]:
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def get_drive_type(path: str) -> str:
    if platform.system().lower() != "windows":
        return "Unknown"
    if GetDriveTypeW is None:
        return "Unknown"
    if not path:
        return "Unknown"
    p = Path(path)
    drive = p.drive or (str(p).split(":", 1)[0] + ":\\")
    if not drive.endswith("\\"):
        drive = drive + "\\"
    try:
        dtype = GetDriveTypeW(drive)
    except Exception:
        return "Unknown"
    if dtype == 2:
        return "Removable"
    if dtype == 3:
        return "Fixed"
    if dtype == 4:
        return "Network"
    return "Unknown"


def get_removable_drives() -> List[str]:
    out: List[str] = []
    if platform.system().lower() != "windows":
        return out
    if GetDriveTypeW is None:
        return out
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue
        try:
            dtype = GetDriveTypeW(root)
        except Exception:
            continue
        if dtype == 2:
            out.append(root)
    return out


def path_root_exists(path_str: str) -> bool:
    if not path_str:
        return False
    p = Path(path_str)
    drive = p.drive or (str(p).split(":", 1)[0] + ":\\")
    if not drive.endswith("\\"):
        drive = drive + "\\"
    return os.path.exists(drive)


# ---------- metrics ----------

class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.successes = 0
        self.failures = 0
        self.blocks = 0

    def inc_success(self):
        with self._lock:
            self.successes += 1

    def inc_failure(self):
        with self._lock:
            self.failures += 1

    def inc_block(self):
        with self._lock:
            self.blocks += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "successes": self.successes,
                "failures": self.failures,
                "blocks": self.blocks,
            }


# ---------- policy ----------

class DestinationPolicy:
    def __init__(self):
        self._lock = threading.Lock()
        self.allow = set(["example.com"])
        self.block = set()

    def is_blocked(self, host: str) -> bool:
        h = (host or "").lower()
        with self._lock:
            return h in self.block

    def is_allowed_for_sensitive(self, host: str) -> bool:
        h = (host or "").lower()
        with self._lock:
            if h in self.block:
                return False
            if not self.allow:
                return False
            return h in self.allow

    def add_allow(self, host: str):
        h = (host or "").lower()
        if not h:
            return
        with self._lock:
            self.allow.add(h)
            self.block.discard(h)

    def add_block(self, host: str):
        h = (host or "").lower()
        if not h:
            return
        with self._lock:
            self.block.add(h)
            self.allow.discard(h)

    def snapshot(self) -> Dict[str, List[str]]:
        with self._lock:
            return {
                "allow": sorted(self.allow),
                "block": sorted(self.block),
            }

    def to_dict(self) -> Dict[str, Any]:
        s = self.snapshot()
        return {"allow": s["allow"], "block": s["block"]}

    def from_dict(self, d: Dict[str, Any]):
        allow = d.get("allow", [])
        block = d.get("block", [])
        with self._lock:
            self.allow = set(a.lower() for a in allow)
            self.block = set(b.lower() for b in block)


# ---------- trust ----------

@dataclass
class TrustEntry:
    name: str
    trust: float
    last_update: float


class TrustManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.entries: Dict[str, TrustEntry] = {}

    def ensure(self, process_name: str) -> TrustEntry:
        key = (process_name or "unknown").lower()
        now = time.time()
        with self._lock:
            if key not in self.entries:
                self.entries[key] = TrustEntry(name=process_name, trust=0.90, last_update=now)
            return self.entries[key]

    def adjust(self, process_name: str, delta: float):
        e = self.ensure(process_name)
        now = time.time()
        with self._lock:
            t = max(0.0, min(0.90, e.trust + delta))
            e.trust = t
            e.last_update = now

    def get(self, process_name: str) -> float:
        return self.ensure(process_name).trust

    def snapshot(self) -> List[TrustEntry]:
        with self._lock:
            return list(self.entries.values())

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": [
                    {"name": e.name, "trust": e.trust, "last_update": e.last_update}
                    for e in self.entries.values()
                ]
            }

    def from_dict(self, d: Dict[str, Any]):
        entries = d.get("entries", [])
        with self._lock:
            self.entries.clear()
            for raw in entries:
                name = raw.get("name", "")
                trust = float(raw.get("trust", 0.90))
                last_update = float(raw.get("last_update", time.time()))
                if name:
                    key = name.lower()
                    self.entries[key] = TrustEntry(
                        name=name,
                        trust=max(0.0, min(0.90, trust)),
                        last_update=last_update,
                    )


# ---------- cortex ----------

@dataclass
class Event:
    ts: float
    success: bool
    latency_ms: float
    host: str
    error_type: str
    process: str
    trust: float
    data_type: str


class Cortex:
    def __init__(self, max_events: int = 500):
        self._lock = threading.Lock()
        self.max_events = max_events
        self.events: List[Event] = []

    def record(self, e: Event):
        with self._lock:
            self.events.append(e)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

    def _snapshot(self) -> List[Event]:
        with self._lock:
            return list(self.events)

    def stats(self) -> Dict[str, Any]:
        ev = self._snapshot()
        n = len(ev)
        if n == 0:
            return {
                "success_rate": 1.0,
                "avg_latency": 0.0,
                "per_host": {},
                "avg_trust": 0.90,
            }
        s = sum(1 for e in ev if e.success)
        success_rate = s / n
        avg_lat = sum(e.latency_ms for e in ev) / n
        host_map: Dict[str, Dict[str, int]] = {}
        for e in ev:
            h = e.host
            if h not in host_map:
                host_map[h] = {"tot": 0, "ok": 0}
            host_map[h]["tot"] += 1
            if e.success:
                host_map[h]["ok"] += 1
        per_host = {
            h: (m["ok"] / m["tot"] if m["tot"] > 0 else 1.0)
            for h, m in host_map.items()
        }
        avg_trust = sum(e.trust for e in ev) / n
        return {
            "success_rate": success_rate,
            "avg_latency": avg_lat,
            "per_host": per_host,
            "avg_trust": avg_trust,
        }

    def health(self, host: Optional[str] = None) -> Dict[str, Any]:
        st = self.stats()
        sr = st["success_rate"]
        lat = st["avg_latency"]
        avg_trust = st["avg_trust"]
        per_host = st["per_host"]

        conf = 100.0
        if sr < 0.9:
            conf -= (0.9 - sr) * 50.0
        if host and host in per_host and per_host[host] < 0.7:
            conf -= (0.7 - per_host[host]) * 50.0
        if lat > 1000:
            conf -= 10.0
        if avg_trust < 0.8:
            conf -= (0.8 - avg_trust) * 40.0

        conf = max(0.0, min(100.0, conf))
        if conf >= 80 and sr >= 0.95:
            state = "HEALTHY"
        elif conf >= 50 and sr >= 0.80:
            state = "DEGRADED"
        elif conf >= 30 and sr >= 0.50:
            state = "UNSTABLE"
        else:
            state = "CRITICAL"

        return {
            "confidence": conf,
            "state": state,
            "success_rate": sr,
            "avg_latency": lat,
            "avg_trust": avg_trust,
        }

    def to_dict(self) -> Dict[str, Any]:
        ev = self._snapshot()
        sample = [
            {
                "ts": e.ts,
                "success": e.success,
                "latency_ms": e.latency_ms,
                "host": e.host,
                "error_type": e.error_type,
                "process": e.process,
                "trust": e.trust,
                "data_type": e.data_type,
            }
            for e in ev[-50:]
        ]
        return {"sample": sample}

    def from_dict(self, d: Dict[str, Any]):
        sample = d.get("sample", [])
        arr: List[Event] = []
        for raw in sample:
            arr.append(
                Event(
                    ts=float(raw.get("ts", time.time())),
                    success=bool(raw.get("success", False)),
                    latency_ms=float(raw.get("latency_ms", 0.0)),
                    host=str(raw.get("host", "")),
                    error_type=str(raw.get("error_type", "other")),
                    process=str(raw.get("process", "unknown")),
                    trust=float(raw.get("trust", 0.90)),
                    data_type=str(raw.get("data_type", "general")),
                )
            )
        with self._lock:
            self.events = arr[-self.max_events:]


# ---------- engine ----------

@dataclass
class CapsulePolicy:
    host: str
    ttl: int
    max_attempts: int
    data_type: str


@dataclass
class Capsule:
    id: str
    created: float
    policy: CapsulePolicy
    description: str
    attempts: int = 0
    opened: bool = False
    expired: bool = False
    last_error: str = ""


class Engine:
    def __init__(self, metrics: Metrics, policy: DestinationPolicy, trust: TrustManager, cortex: Cortex):
        self.metrics = metrics
        self.policy = policy
        self.trust = trust
        self.cortex = cortex

    def _host_from_url(self, url: str) -> str:
        parsed = urllib_parse.urlparse(url)
        return parsed.hostname or ""

    def make_capsule(self, host: str, ttl: int, max_attempts: int, data_type: str, desc: str) -> Capsule:
        return Capsule(
            id=str(uuid.uuid4()),
            created=time.time(),
            policy=CapsulePolicy(host=host, ttl=ttl, max_attempts=max_attempts, data_type=data_type),
            description=desc,
        )

    def deliver_get(self, cap: Capsule, url: str, process_name: str, timeout: float = 5.0) -> bool:
        start = time.time()
        host = self._host_from_url(url)
        dt = cap.policy.data_type
        trust_val = self.trust.get(process_name)
        ultra = dt in ("password", "biometric")

        if self.policy.is_blocked(host):
            cap.last_error = f"Host {host} blocked"
            self.metrics.inc_block()
            self._log(False, start, host, "policy", process_name, trust_val, dt)
            self.trust.adjust(process_name, -0.05)
            return False

        if dt in ("personal", "bio", "machine", "password", "biometric"):
            if not self.policy.is_allowed_for_sensitive(host):
                cap.last_error = f"Sensitive '{dt}' not allowed to {host}"
                self.metrics.inc_block()
                self._log(False, start, host, "policy", process_name, trust_val, dt)
                self.trust.adjust(process_name, -0.07)
                return False

        now = time.time()
        if now - cap.created > cap.policy.ttl:
            cap.expired = True
            cap.last_error = "TTL expired"
            self.metrics.inc_failure()
            self._log(False, start, host, "policy", process_name, trust_val, dt)
            self.trust.adjust(process_name, -0.02)
            return False

        if cap.opened or cap.attempts >= cap.policy.max_attempts:
            cap.last_error = "Max attempts exceeded or already opened"
            self.metrics.inc_failure()
            self._log(False, start, host, "policy", process_name, trust_val, dt)
            self.trust.adjust(process_name, -0.01)
            return False

        cap.attempts += 1

        if ultra and trust_val < 0.5:
            cap.last_error = f"Trust {trust_val:.2f} too low for ultra '{dt}'"
            self.metrics.inc_block()
            self._log(False, start, host, "policy", process_name, trust_val, dt)
            self.trust.adjust(process_name, -0.03)
            return False

        success = False
        err_type = "other"
        try:
            resp = requests.get(url, timeout=timeout)
        except Exception as e:
            cap.last_error = f"Network error: {e}"
            err_type = "network"
            self.metrics.inc_failure()
            self.trust.adjust(process_name, -0.005)
            success = False
        else:
            if resp.status_code == 200:
                cap.opened = True
                success = True
                self.metrics.inc_success()
                self.trust.adjust(process_name, +0.002)
            else:
                cap.last_error = f"HTTP {resp.status_code}"
                err_type = "http"
                self.metrics.inc_failure()
                self.trust.adjust(process_name, -0.01)
                success = False

        self._log(success, start, host, err_type, process_name, trust_val, dt)
        return success

    def _log(
        self,
        success: bool,
        start: float,
        host: str,
        err_type: str,
        process_name: str,
        trust: float,
        data_type: str,
    ):
        lat = (time.time() - start) * 1000.0
        self.cortex.record(
            Event(
                ts=time.time(),
                success=success,
                latency_ms=lat,
                host=host,
                error_type=err_type,
                process=process_name,
                trust=trust,
                data_type=data_type,
            )
        )


# ---------- memory store ----------

class MemoryStore:
    """
    Handles:
    - primary_path (string or "")
    - backup_paths (list of strings, unlimited)
    - encrypted_state: swp_state.json.enc
    - backup_paths.json + versioned copies
    """

    def __init__(self, dir_name: str = "SnowWhiteParentsMagicBox"):
        home = Path.home()
        self.base_dir = home / dir_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.key_file = self.base_dir / "state.key"
        self.state_file_name = "swp_state.json.enc"
        self.paths_file = self.base_dir / "backup_paths.json"
        self.paths_versions_dir = self.base_dir / "backup_paths_versions"
        self.paths_versions_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.primary_path: str = ""
        self.backup_paths: List[str] = []

        self._ensure_key()
        self._load_paths_json_initial()

    def _ensure_key(self):
        if not self.key_file.exists():
            key = secrets.token_bytes(32)
            b64 = base64.b64encode(key).decode("utf-8")
            with open(self.key_file, "w", encoding="utf-8") as f:
                f.write(b64)

    def _load_key(self) -> bytes:
        with open(self.key_file, "r", encoding="utf-8") as f:
            b64 = f.read().strip()
        return base64.b64decode(b64.encode("utf-8"))

    def _encrypt(self, data: bytes) -> bytes:
        key = self._load_key()
        out = bytearray()
        for i, b in enumerate(data):
            out.append(b ^ key[i % len(key)])
        return base64.b64encode(bytes(out))

    def _decrypt(self, data: bytes) -> bytes:
        key = self._load_key()
        raw = base64.b64decode(data)
        out = bytearray()
        for i, b in enumerate(raw):
            out.append(b ^ key[i % len(key)])
        return bytes(out)

    def _all_state_dirs(self) -> List[Path]:
        out: List[Path] = []
        if self.primary_path:
            out.append(Path(self.primary_path))
        for p in self.backup_paths:
            out.append(Path(p))
        out.append(self.base_dir)
        return out

    def _load_paths_json_initial(self):
        if not self.paths_file.exists():
            return
        try:
            with open(self.paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            lst = data.get("backup_paths", [])
            if isinstance(lst, list):
                cleaned = []
                for p in lst:
                    p = str(p).strip()
                    if p and p not in cleaned:
                        cleaned.append(p)
                with self._lock:
                    self.backup_paths = cleaned
        except Exception:
            pass

    def _merge_paths_from_json(self):
        if not self.paths_file.exists():
            return
        try:
            with open(self.paths_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            lst = data.get("backup_paths", [])
            if not isinstance(lst, list):
                return
            file_paths = [str(p).strip() for p in lst if str(p).strip()]
            with self._lock:
                merged = self.backup_paths[:]
                for p in file_paths:
                    if p not in merged:
                        merged.append(p)
                self.backup_paths = merged
        except Exception:
            pass

    def _save_paths_json(self):
        try:
            with self._lock:
                lst = list(self.backup_paths)
            data = {"backup_paths": lst}
            with open(self.paths_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pf = self.paths_versions_dir / f"backup_paths_{ts}.json"
            with open(pf, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # public API for primary + backups

    def set_primary(self, path_str: str):
        with self._lock:
            self.primary_path = path_str.strip()

    def get_primary(self) -> str:
        with self._lock:
            return self.primary_path

    def set_backups(self, paths: List[str]):
        cleaned = []
        for p in paths:
            p = (p or "").strip()
            if not p:
                continue
            if not path_root_exists(p):
                continue
            if p not in cleaned:
                cleaned.append(p)
        with self._lock:
            self.backup_paths = cleaned

    def get_backups(self) -> List[str]:
        with self._lock:
            return list(self.backup_paths)

    # save / load

    def save(
        self,
        metrics: Metrics,
        policy: DestinationPolicy,
        trust: TrustManager,
        cortex: Cortex,
        baseline: Dict[str, int],
    ):
        self.set_backups(self.get_backups())

        m_snap = metrics.snapshot()
        p_snap = policy.to_dict()
        t_snap = trust.to_dict()
        c_snap = cortex.to_dict()
        state = {
            "system_summary": get_system_summary(),
            "primary_path": self.get_primary(),
            "backup_paths": self.get_backups(),
            "metrics": m_snap,
            "policy": p_snap,
            "trust": t_snap,
            "cortex": c_snap,
            "baseline": baseline,
            "timestamp": time.time(),
        }
        encoded = json.dumps(state).encode("utf-8")
        enc = self._encrypt(encoded)

        for base_dir in self._all_state_dirs():
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                f = base_dir / self.state_file_name
                with open(f, "wb") as fp:
                    fp.write(enc)
            except Exception:
                continue

        self._save_paths_json()

    def load(
        self,
        policy: DestinationPolicy,
        trust: TrustManager,
        cortex: Cortex,
    ) -> Dict[str, int]:
        baseline = {
            "installed_apps": 0,
            "processes": 0,
            "services": 0,
            "startup_items": 0,
        }

        for base_dir in self._all_state_dirs():
            try:
                f = base_dir / self.state_file_name
                if not f.exists():
                    continue
                with open(f, "rb") as fp:
                    enc = fp.read()
                dec = self._decrypt(enc)
                state = json.loads(dec.decode("utf-8"))
            except Exception:
                continue

            try:
                self.set_primary(state.get("primary_path", "") or "")
                self.set_backups(state.get("backup_paths", []) or [])

                policy.from_dict(state.get("policy", {}))
                trust.from_dict(state.get("trust", {}))
                cortex.from_dict(state.get("cortex", {}))

                baseline = state.get("baseline", baseline)
            except Exception:
                pass
            break

        self._merge_paths_from_json()
        self.set_backups(self.get_backups())
        return baseline


# ---------- baseline scanner ----------

class BaselineScanner:
    def __init__(self):
        self.installed: List[str] = []
        self.procs: List[str] = []
        self.services: List[str] = []
        self.startup: List[str] = []

    def run(self):
        self._scan_installed()
        self._scan_procs()
        self._scan_services()
        self._scan_startup()

    def _scan_installed(self):
        self.installed = []
        if winreg is None:
            return
        paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        for root, subkey in paths:
            try:
                key = winreg.OpenKey(root, subkey)
            except OSError:
                continue
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(key, i)
                    i += 1
                    try:
                        app_key = winreg.OpenKey(root, subkey + "\\" + sub)
                        name, _ = winreg.QueryValueEx(app_key, "DisplayName")
                        if name:
                            self.installed.append(str(name))
                    except OSError:
                        continue
                except OSError:
                    break
            winreg.CloseKey(key)
        self.installed = sorted(set(self.installed))

    def _scan_procs(self):
        names = []
        for p in psutil.process_iter(attrs=["name"]):
            n = p.info.get("name") or ""
            if n:
                names.append(n)
        self.procs = sorted(set(names))

    def _scan_services(self):
        names = []
        try:
            for s in psutil.win_service_iter():
                info = s.as_dict()
                n = info.get("name") or ""
                if n:
                    names.append(n)
        except Exception:
            pass
        self.services = sorted(set(names))

    def _scan_startup(self):
        items: List[str] = []
        if winreg is not None:
            keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            ]
            for root, subkey in keys:
                try:
                    key = winreg.OpenKey(root, subkey)
                except OSError:
                    continue
                i = 0
                while True:
                    try:
                        name, value, _ = winreg.EnumValue(key, i)
                        i += 1
                        if name:
                            items.append(str(name))
                    except OSError:
                        break
                winreg.CloseKey(key)
        self.startup = sorted(set(items))

    def summary(self) -> Dict[str, int]:
        return {
            "installed_apps": len(self.installed),
            "processes": len(self.procs),
            "services": len(self.services),
            "startup_items": len(self.startup),
        }


# ---------- GUI ----------

class AppGUI:
    def __init__(
        self,
        engine: Engine,
        metrics: Metrics,
        policy: DestinationPolicy,
        trust: TrustManager,
        cortex: Cortex,
        store: MemoryStore,
        baseline: Dict[str, int],
    ):
        self.engine = engine
        self.metrics = metrics
        self.policy = policy
        self.trust = trust
        self.cortex = cortex
        self.store = store
        self.baseline = baseline

        self.root = tk.Tk()
        self.root.title("Snow White Parents — Refactor Console (No Self-Repair)")
        self.root.geometry("1200x750")
        self.root.configure(bg="#101018")

        self.bg_main = "#101018"
        self.bg_panel = "#1e1e30"
        self.bg_nav = "#141424"
        self.bg_status = "#181828"
        self.accent = "#4fa3ff"
        self.text_main = "#f0f0ff"
        self.text_dim = "#a0a0c0"

        self._build_layout()
        self._schedule_update()

    def _build_layout(self):
        # status bar
        sb = tk.Frame(self.root, bg=self.bg_status, height=40)
        sb.pack(side="top", fill="x")

        self.status_health = tk.Label(
            sb,
            text="Health: n/a",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_status,
        )
        self.status_health.pack(side="left", padx=10)

        self.status_deliveries = tk.Label(
            sb,
            text="Deliveries: 0 ok / 0 fail / 0 block",
            font=("Segoe UI", 10),
            fg=self.text_dim,
            bg=self.bg_status,
        )
        self.status_deliveries.pack(side="left", padx=20)

        main = tk.Frame(self.root, bg=self.bg_main)
        main.pack(side="top", fill="both", expand=True)

        # navigation
        nav = tk.Frame(main, bg=self.bg_nav, width=180)
        nav.pack(side="left", fill="y")

        self.nav_var = tk.StringVar(value="Dashboard")

        tk.Label(
            nav,
            text="Sections",
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.bg_nav,
            pady=8,
        ).pack(anchor="w", padx=10)

        for name in ["Dashboard", "Security", "Network", "Memory", "Settings"]:
            b = tk.Radiobutton(
                nav,
                text=name,
                value=name,
                variable=self.nav_var,
                indicatoron=False,
                font=("Segoe UI", 10),
                fg=self.text_main,
                bg=self.bg_nav,
                selectcolor=self.accent,
                activebackground=self.accent,
                activeforeground=self.text_main,
                bd=0,
                pady=4,
                command=self._switch_view,
            )
            b.pack(fill="x", padx=8, pady=2)

        # content stack
        self.content = tk.Frame(main, bg=self.bg_main)
        self.content.pack(side="left", fill="both", expand=True)

        self.views: Dict[str, tk.Frame] = {}
        self._build_views()
        self._show_view("Dashboard")

    def _build_views(self):
        self.views["Dashboard"] = self._view_dashboard()
        self.views["Security"] = self._view_security()
        self.views["Network"] = self._view_network()
        self.views["Memory"] = self._view_memory()
        self.views["Settings"] = self._view_settings()
        for v in self.views.values():
            v.pack_forget()

    def _show_view(self, name: str):
        for v in self.views.values():
            v.pack_forget()
        f = self.views.get(name)
        if f:
            f.pack(fill="both", expand=True, padx=10, pady=10)

    def _switch_view(self):
        self._show_view(self.nav_var.get())

    # Views

    def _view_dashboard(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        tk.Label(
            frame,
            text="Dashboard",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=(10, 15))

        # Baseline summary
        base = self.baseline
        tk.Label(
            frame,
            text=(
                f"Installed apps: {base.get('installed_apps', 0)} | "
                f"Processes: {base.get('processes', 0)} | "
                f"Services: {base.get('services', 0)} | "
                f"Startup: {base.get('startup_items', 0)}"
            ),
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=(0, 10))

        # Cortex summary
        self.lbl_dash_cortex = tk.Label(
            frame,
            text="Cortex: n/a",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
            justify="left",
        )
        self.lbl_dash_cortex.pack(anchor="w", padx=15, pady=(0, 10))

        return frame

    def _view_security(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        tk.Label(
            frame,
            text="Security & Policy",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=(10, 15))

        policy_box = tk.Frame(frame, bg=self.bg_panel)
        policy_box.pack(fill="x", padx=15, pady=5)

        tk.Label(
            policy_box,
            text="Destination Policy:",
            font=("Segoe UI", 11, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        row = tk.Frame(policy_box, bg=self.bg_panel)
        row.pack(fill="x", pady=4)

        tk.Label(
            row,
            text="Host:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.entry_policy_host = tk.Entry(row, width=30)
        self.entry_policy_host.insert(0, "example.com")
        self.entry_policy_host.pack(side="left", padx=5)

        btn_allow = tk.Button(
            row,
            text="Allow",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg="#215c2c",
            activebackground="#2b8c3b",
            bd=0,
            padx=8,
            command=self._on_allow_host,
        )
        btn_allow.pack(side="left", padx=3)

        btn_block = tk.Button(
            row,
            text="Block",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg="#7c2020",
            activebackground="#a52a2a",
            bd=0,
            padx=8,
            command=self._on_block_host,
        )
        btn_block.pack(side="left", padx=3)

        self.lbl_allow = tk.Label(
            policy_box,
            text="Allow: (none)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
        )
        self.lbl_allow.pack(anchor="w", pady=2)

        self.lbl_block = tk.Label(
            policy_box,
            text="Block: (none)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
        )
        self.lbl_block.pack(anchor="w", pady=2)

        trust_box = tk.Frame(frame, bg=self.bg_panel)
        trust_box.pack(fill="both", expand=True, padx=15, pady=(10, 10))

        tk.Label(
            trust_box,
            text="Software Trust (top entries)",
            font=("Segoe UI", 11, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        self.list_trust = tk.Listbox(
            trust_box,
            width=80,
            height=14,
            font=("Consolas", 9),
            fg=self.text_main,
            bg="#111120",
            selectbackground=self.accent,
            activestyle="none",
        )
        self.list_trust.pack(fill="both", expand=True, pady=5)

        return frame

    def _view_network(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        tk.Label(
            frame,
            text="Network Delivery Tester",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=(10, 15))

        box = tk.Frame(frame, bg=self.bg_panel)
        box.pack(fill="x", padx=15, pady=5)

        row1 = tk.Frame(box, bg=self.bg_panel)
        row1.pack(fill="x", pady=4)

        tk.Label(
            row1,
            text="URL:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.entry_url = tk.Entry(row1, width=50)
        self.entry_url.insert(0, "https://example.com")
        self.entry_url.pack(side="left", padx=5)

        row2 = tk.Frame(box, bg=self.bg_panel)
        row2.pack(fill="x", pady=4)

        tk.Label(
            row2,
            text="Data type:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.var_data_type = tk.StringVar(value="general")
        self.combo_data_type = ttk.Combobox(
            row2,
            textvariable=self.var_data_type,
            values=["general", "personal", "bio", "machine", "password", "biometric"],
            state="readonly",
            width=18,
        )
        self.combo_data_type.pack(side="left", padx=5)

        tk.Label(
            row2,
            text="Process:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left", padx=(10, 0))

        self.entry_process = tk.Entry(row2, width=20)
        self.entry_process.insert(0, "test_app.exe")
        self.entry_process.pack(side="left", padx=5)

        btn_send = tk.Button(
            box,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.accent,
            activebackground="#6fb5ff",
            bd=0,
            padx=10,
            pady=3,
            command=self._on_send_test,
        )
        btn_send.pack(anchor="w", pady=(8, 4))

        self.lbl_network_result = tk.Label(
            frame,
            text="Last result: (none)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
            wraplength=900,
        )
        self.lbl_network_result.pack(anchor="w", padx=15, pady=(4, 10))

        return frame

    def _view_memory(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        tk.Label(
            frame,
            text="Memory & Backups (List + Details)",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=(10, 15))

        main = tk.Frame(frame, bg=self.bg_panel)
        main.pack(fill="both", expand=True, padx=15, pady=5)

        # top: primary path
        row_p = tk.Frame(main, bg=self.bg_panel)
        row_p.pack(fill="x", pady=(0, 6))

        tk.Label(
            row_p,
            text="Primary memory path:",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.entry_primary = tk.Entry(row_p, width=60)
        self.entry_primary.insert(0, self.store.get_primary())
        self.entry_primary.pack(side="left", padx=5)

        # center: left list, right details
        center = tk.Frame(main, bg=self.bg_panel)
        center.pack(fill="both", expand=True, pady=(6, 6))

        left = tk.Frame(center, bg=self.bg_panel)
        left.pack(side="left", fill="both", expand=False, padx=(0, 8))

        tk.Label(
            left,
            text="Backup paths:",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w")

        self.list_backups = tk.Listbox(
            left,
            width=60,
            height=10,
            font=("Consolas", 9),
            fg=self.text_main,
            bg="#111120",
            selectbackground=self.accent,
            activestyle="none",
        )
        self.list_backups.pack(fill="both", expand=True, pady=(2, 4))
        for p in self.store.get_backups():
            self.list_backups.insert(tk.END, p)
        self.list_backups.bind("<<ListboxSelect>>", lambda e: self._update_backup_details())

        row_add = tk.Frame(left, bg=self.bg_panel)
        row_add.pack(fill="x", pady=(4, 2))

        self.entry_backup_new = tk.Entry(row_add, width=40)
        self.entry_backup_new.pack(side="left", padx=(0, 4))

        btn_add = tk.Button(
            row_add,
            text="Add",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg=self.accent,
            activebackground="#6fb5ff",
            bd=0,
            padx=8,
            command=self._on_add_backup,
        )
        btn_add.pack(side="left", padx=2)

        btn_remove = tk.Button(
            row_add,
            text="Remove Selected",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg="#7c2020",
            activebackground="#a52a2a",
            bd=0,
            padx=8,
            command=self._on_remove_backup,
        )
        btn_remove.pack(side="left", padx=2)

        row_scan = tk.Frame(left, bg=self.bg_panel)
        row_scan.pack(fill="x", pady=(4, 2))

        btn_scan = tk.Button(
            row_scan,
            text="Scan Removable Drives",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg="#215c2c",
            activebackground="#2b8c3b",
            bd=0,
            padx=8,
            command=self._on_scan_removable,
        )
        btn_scan.pack(side="left", padx=(0, 4))

        self.lbl_scan_info = tk.Label(
            row_scan,
            text="(no scan yet)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
        )
        self.lbl_scan_info.pack(side="left")

        # right: details
        right = tk.Frame(center, bg=self.bg_panel, bd=1, relief="sunken")
        right.pack(side="left", fill="both", expand=True)

        tk.Label(
            right,
            text="Backup details",
            font=("Segoe UI", 11, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=10, pady=(6, 6))

        self.lbl_detail_path = tk.Label(
            right,
            text="Path: (none)",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
            justify="left",
            wraplength=500,
        )
        self.lbl_detail_path.pack(anchor="w", padx=10, pady=2)

        self.lbl_detail_type = tk.Label(
            right,
            text="Type: (n/a)",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.lbl_detail_type.pack(anchor="w", padx=10, pady=2)

        self.lbl_detail_root = tk.Label(
            right,
            text="Root: (n/a)",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.lbl_detail_root.pack(anchor="w", padx=10, pady=2)

        self.lbl_detail_state = tk.Label(
            right,
            text="State: (n/a)",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.lbl_detail_state.pack(anchor="w", padx=10, pady=2)

        # bottom: save + status
        btn_save = tk.Button(
            main,
            text="Save Memory Paths",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.accent,
            activebackground="#6fb5ff",
            bd=0,
            padx=10,
            pady=3,
            command=self._on_save_memory,
        )
        btn_save.pack(anchor="w", pady=(6, 4))

        self.lbl_mem_status = tk.Label(
            main,
            text="Status: (not saved yet)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
            wraplength=900,
        )
        self.lbl_mem_status.pack(anchor="w", pady=(2, 8))

        self._update_backup_details()

        return frame

    def _view_settings(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        tk.Label(
            frame,
            text="Settings",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=(10, 15))

        tk.Label(
            frame,
            text="Additional configuration can live here.",
            font=("Segoe UI", 10),
            fg=self.text_dim,
            bg=self.bg_panel,
        ).pack(anchor="w", padx=15, pady=5)

        btn_exit = tk.Button(
            frame,
            text="Exit",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg="#7c2020",
            activebackground="#a52a2a",
            bd=0,
            padx=10,
            pady=3,
            command=self._on_exit,
        )
        btn_exit.pack(anchor="w", padx=15, pady=(15, 10))

        return frame

    # Handlers

    def _on_allow_host(self):
        host = self.entry_policy_host.get().strip()
        if not host:
            return
        self.policy.add_allow(host)
        self._refresh_policy_labels()

    def _on_block_host(self):
        host = self.entry_policy_host.get().strip()
        if not host:
            return
        self.policy.add_block(host)
        self._refresh_policy_labels()

    def _refresh_policy_labels(self):
        snap = self.policy.snapshot()
        allow = ", ".join(snap["allow"]) if snap["allow"] else "(none)"
        block = ", ".join(snap["block"]) if snap["block"] else "(none)"
        self.lbl_allow.config(text=f"Allow: {allow}")
        self.lbl_block.config(text=f"Block: {block}")

    def _on_send_test(self):
        url = self.entry_url.get().strip()
        if not url:
            self.lbl_network_result.config(text="Last result: No URL provided")
            return
        dt = self.var_data_type.get()
        proc = self.entry_process.get().strip() or "unknown"
        parsed = urllib_parse.urlparse(url)
        host = parsed.hostname
        if not host:
            self.lbl_network_result.config(text="Last result: Invalid URL (no host)")
            return

        cap = self.engine.make_capsule(
            host=host,
            ttl=10,
            max_attempts=1,
            data_type=dt,
            desc=f"GET {url}",
        )

        def worker():
            ok = self.engine.deliver_get(cap, url, proc)
            health = self.cortex.health(host=host)
            if ok:
                base_msg = f"SUCCESS to {host} [{dt}, process={proc}]"
            else:
                base_msg = f"FAILURE to {host}: {cap.last_error} [{dt}, process={proc}]"
            host_msg = (
                f"Next for {host}: {health['confidence']:.1f}% "
                f"[{health['state']}] "
                f"(host success_rate={health['success_rate']*100:.1f}%)"
            )
            txt = "Last result: " + base_msg + "\n" + host_msg

            def update():
                self.lbl_network_result.config(text=txt)

            self.root.after(0, update)

        threading.Thread(target=worker, daemon=True).start()

    # memory handlers

    def _on_add_backup(self):
        p = self.entry_backup_new.get().strip()
        if not p:
            return
        existing = [self.list_backups.get(i) for i in range(self.list_backups.size())]
        if p in existing:
            return
        self.list_backups.insert(tk.END, p)
        self.entry_backup_new.delete(0, tk.END)
        self._update_backup_details()

    def _on_remove_backup(self):
        sel = self.list_backups.curselection()
        if not sel:
            return
        for i in reversed(sel):
            self.list_backups.delete(i)
        self._update_backup_details()

    def _on_scan_removable(self):
        drives = get_removable_drives()
        if not drives:
            self.lbl_scan_info.config(text="No removable drives detected.")
            return
        suggested = [f"{d}SnowWhiteParentsBackup" for d in drives]
        existing = [self.list_backups.get(i) for i in range(self.list_backups.size())]
        added = 0
        for p in suggested:
            if p not in existing:
                self.list_backups.insert(tk.END, p)
                added += 1
        if added == 0:
            self.lbl_scan_info.config(
                text=f"Drives: {', '.join(drives)} (paths already listed)"
            )
        else:
            self.lbl_scan_info.config(
                text=f"Added {added} paths from drives: {', '.join(drives)}"
            )
        self._update_backup_details()

    def _sync_memory_from_ui(self):
        primary_str = self.entry_primary.get().strip()
        self.store.set_primary(primary_str)
        backups = [self.list_backups.get(i) for i in range(self.list_backups.size())]
        self.store.set_backups(backups)

    def _on_save_memory(self):
        try:
            self._sync_memory_from_ui()
            self.store.save(
                self.metrics, self.policy, self.trust, self.cortex, self.baseline
            )
            cleaned = self.store.get_backups()
            self.list_backups.delete(0, tk.END)
            for p in cleaned:
                self.list_backups.insert(tk.END, p)
            self.lbl_mem_status.config(
                text=(
                    f"Status: saved\n"
                    f"Primary: {self.store.get_primary() or '(none)'}\n"
                    f"Backup count: {len(cleaned)}"
                )
            )
            self._update_backup_details()
        except Exception as e:
            self.lbl_mem_status.config(text=f"Status: ERROR ({e})")

    def _update_backup_details(self):
        sel = self.list_backups.curselection()
        if not sel:
            self.lbl_detail_path.config(text="Path: (none)")
            self.lbl_detail_type.config(text="Type: (n/a)")
            self.lbl_detail_root.config(text="Root: (n/a)")
            self.lbl_detail_state.config(text="State: (n/a)")
            return
        idx = sel[0]
        path_str = self.list_backups.get(idx)
        self.lbl_detail_path.config(text=f"Path: {path_str}")

        dtype = get_drive_type(path_str)
        self.lbl_detail_type.config(text=f"Type: {dtype}")

        root_ok = path_root_exists(path_str)
        self.lbl_detail_root.config(
            text=f"Root: {'Present' if root_ok else 'Missing'}"
        )

        if root_ok:
            state = "Usable (root available)"
        else:
            state = "Dead (root missing, will be cleaned if saved)"
        self.lbl_detail_state.config(text=f"State: {state}")

    # other

    def _update_status_bar(self):
        m = self.metrics.snapshot()
        h = self.cortex.health()
        self.status_health.config(
            text=(
                f"Health: {h['state']} | "
                f"Conf: {h['confidence']:.1f}% | "
                f"SR: {h['success_rate']*100:.1f}%"
            )
        )
        self.status_deliveries.config(
            text=(
                f"Deliveries: {m['successes']} ok / "
                f"{m['failures']} fail / {m['blocks']} block"
            )
        )
        # Dashboard cortex label
        if "Dashboard" in self.views:
            txt = (
                f"Success rate: {h['success_rate']*100:.1f}% | "
                f"Avg latency: {h['avg_latency']:.1f} ms | "
                f"Avg trust: {h['avg_trust']*100:.1f}%"
            )
            self.lbl_dash_cortex.config(text="Cortex: " + txt)

    def _update_trust_list(self):
        if "Security" not in self.views:
            return
        entries = self.trust.snapshot()
        entries = sorted(entries, key=lambda e: e.trust, reverse=True)
        self.list_trust.delete(0, tk.END)
        for e in entries[:25]:
            line = f"{e.name:40s} trust={e.trust*100:5.1f}%"
            self.list_trust.insert(tk.END, line)

    def _schedule_update(self):
        self._update_status_bar()
        self._refresh_policy_labels()
        self._update_trust_list()
        self.root.after(700, self._schedule_update)

    def _on_exit(self):
        try:
            self._sync_memory_from_ui()
            self.store.save(
                self.metrics, self.policy, self.trust, self.cortex, self.baseline
            )
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
        self.root.mainloop()


# ---------- main ----------

def main():
    if platform.system().lower() != "windows":
        print("Snow White Parents is Windows-focused; exiting on non-Windows.")
        return

    store = MemoryStore()

    metrics = Metrics()
    policy = DestinationPolicy()
    trust = TrustManager()
    cortex = Cortex()

    scanner = BaselineScanner()
    scanner.run()
    baseline = scanner.summary()

    # seed trust with some names
    seed_names = scanner.installed + scanner.procs + ["test_app.exe", "system_service"]
    for n in seed_names:
        trust.ensure(n)

    loaded_baseline = store.load(policy, trust, cortex)
    for k, v in loaded_baseline.items():
        if baseline.get(k, 0) == 0 and v > 0:
            baseline[k] = v

    engine = Engine(metrics, policy, trust, cortex)

    gui = AppGUI(
        engine=engine,
        metrics=metrics,
        policy=policy,
        trust=trust,
        cortex=cortex,
        store=store,
        baseline=baseline,
    )
    gui.run()


if __name__ == "__main__":
    main()

