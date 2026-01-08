"""
Snow White Parents - MagicBox Edition (Tkinter, Hybrid UI, Triple Backup)
-------------------------------------------------------------------------
- Pure Tkinter GUI (no ttkbootstrap, no admin/elevation)
- Magic Hybrid layout: Ribbon (top) + Sidebar (left) + Panels (right)
- Persistent panels (no destroying => no TclError)
- Real HTTP deliveries via requests
- Hybrid Cortex (predictive intelligence, confidence, health)
- Destination allow-list / block-list
- Sensitive data types:
    "general", "personal", "bio", "machine", "password", "biometric"
- Ultra-sensitive:
    "password" and "biometric" only to explicitly allowed hosts
- Software trust model (90% baseline, dynamic adjustments)
- Persistent memory (triple redundant):
    - Primary path (user-selectable)
    - Backup A: local MagicBox folder
    - Backup B: user-selectable optional path (local or SMB)
- Self-repair of main program file (hash + backup)
- Windows baseline:
    - Installed software (registry)
    - Running processes (psutil)
    - Services (psutil)
    - Startup items (registry)
"""

import os
import sys
import platform
import ctypes
import threading
import time
import uuid
import json
import base64
import secrets
import hashlib
import shutil

import requests
import psutil
import tkinter as tk
from tkinter import ttk
import urllib.parse as urllib_parse

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path

# Windows registry
try:
    import winreg  # type: ignore
except ImportError:
    winreg = None


# ---------- helpers ----------

def get_system_summary() -> Dict[str, str]:
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


# ---------- metrics ----------

class GuardianMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.successes = 0
        self.failures = 0
        self.policy_blocks = 0

    def increment_successes(self, n: int = 1):
        with self._lock:
            self.successes += n

    def increment_failures(self, n: int = 1):
        with self._lock:
            self.failures += n

    def increment_policy_blocks(self, n: int = 1):
        with self._lock:
            self.policy_blocks += n

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "successes": self.successes,
                "failures": self.failures,
                "policy_blocks": self.policy_blocks,
            }


# ---------- destination policy ----------

class DestinationPolicy:
    def __init__(self):
        self._lock = threading.Lock()
        self.allow_list = set()
        self.block_list = set()
        self.allow_list.add("example.com")

    def is_blocked(self, host: str) -> bool:
        h = (host or "").lower()
        with self._lock:
            return h in self.block_list

    def is_allowed_for_sensitive(self, host: str) -> bool:
        h = (host or "").lower()
        with self._lock:
            if h in self.block_list:
                return False
            if len(self.allow_list) == 0:
                return False
            return h in self.allow_list

    def add_allow(self, host: str):
        h = (host or "").lower()
        if not h:
            return
        with self._lock:
            self.allow_list.add(h)
            self.block_list.discard(h)

    def add_block(self, host: str):
        h = (host or "").lower()
        if not h:
            return
        with self._lock:
            self.block_list.add(h)
            self.allow_list.discard(h)

    def snapshot(self) -> Dict[str, List[str]]:
        with self._lock:
            return {
                "allow": sorted(self.allow_list),
                "block": sorted(self.block_list),
            }

    def to_dict(self) -> Dict[str, Any]:
        snap = self.snapshot()
        return {"allow": snap["allow"], "block": snap["block"]}

    def from_dict(self, d: Dict[str, Any]):
        allow = d.get("allow", [])
        block = d.get("block", [])
        with self._lock:
            self.allow_list = set(a.lower() for a in allow)
            self.block_list = set(b.lower() for b in block)


# ---------- software trust ----------

@dataclass
class SoftwareTrustEntry:
    name: str
    trust: float
    last_update: float


class SoftwareTrustManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.entries: Dict[str, SoftwareTrustEntry] = {}

    def discover_software(self, names: List[str]):
        now = time.time()
        with self._lock:
            for n in names:
                key = n.lower()
                if key not in self.entries:
                    self.entries[key] = SoftwareTrustEntry(
                        name=n,
                        trust=0.90,
                        last_update=now,
                    )

    def ensure_entry(self, process_name: str) -> SoftwareTrustEntry:
        key = (process_name or "unknown_process").lower()
        now = time.time()
        with self._lock:
            if key not in self.entries:
                self.entries[key] = SoftwareTrustEntry(
                    name=process_name or "unknown_process",
                    trust=0.90,
                    last_update=now,
                )
            return self.entries[key]

    def adjust_trust(self, process_name: str, delta: float):
        entry = self.ensure_entry(process_name)
        now = time.time()
        with self._lock:
            new_trust = entry.trust + delta
            new_trust = max(0.0, min(0.90, new_trust))
            entry.trust = new_trust
            entry.last_update = now

    def get_trust(self, process_name: str) -> float:
        entry = self.ensure_entry(process_name)
        return entry.trust

    def snapshot(self) -> List[SoftwareTrustEntry]:
        with self._lock:
            return list(self.entries.values())

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": [
                    {
                        "name": e.name,
                        "trust": e.trust,
                        "last_update": e.last_update,
                    }
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
                    self.entries[key] = SoftwareTrustEntry(
                        name=name,
                        trust=max(0.0, min(0.90, trust)),
                        last_update=last_update,
                    )


# ---------- hybrid cortex ----------

@dataclass
class DeliveryEvent:
    timestamp: float
    success: bool
    latency_ms: float
    destination_host: str
    error_type: str
    process_name: str
    trust_score: float
    data_type: str


class HybridCortex:
    def __init__(self, max_events: int = 500):
        self._lock = threading.Lock()
        self.max_events = max_events
        self.events: List[DeliveryEvent] = []

    def record_event(self, event: DeliveryEvent):
        with self._lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

    def _snapshot_events(self) -> List[DeliveryEvent]:
        with self._lock:
            return list(self.events)

    def compute_stats(self) -> Dict[str, Any]:
        events = self._snapshot_events()
        n = len(events)
        if n == 0:
            return {
                "success_rate": 1.0,
                "avg_latency_ms": 0.0,
                "error_breakdown": {},
                "per_host": {},
                "avg_trust": 0.90,
            }

        successes = sum(1 for e in events if e.success)
        success_rate = successes / n
        avg_latency = sum(e.latency_ms for e in events) / n

        error_counts: Dict[str, int] = {}
        for e in events:
            if not e.success:
                error_counts[e.error_type] = error_counts.get(e.error_type, 0) + 1
        error_breakdown = {
            k: v / (n - successes) if (n - successes) > 0 else 0.0
            for k, v in error_counts.items()
        }

        host_stats: Dict[str, Dict[str, int]] = {}
        for e in events:
            h = e.destination_host
            if h not in host_stats:
                host_stats[h] = {"total": 0, "success": 0}
            host_stats[h]["total"] += 1
            if e.success:
                host_stats[h]["success"] += 1

        per_host: Dict[str, float] = {}
        for h, st in host_stats.items():
            if st["total"] > 0:
                per_host[h] = st["success"] / st["total"]
            else:
                per_host[h] = 1.0

        avg_trust = sum(e.trust_score for e in events) / n

        return {
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "error_breakdown": error_breakdown,
            "per_host": per_host,
            "avg_trust": avg_trust,
        }

    def predict_confidence(self, host: Optional[str] = None) -> Dict[str, Any]:
        stats = self.compute_stats()

        success_rate = stats["success_rate"]
        avg_latency = stats["avg_latency_ms"]
        error_breakdown = stats["error_breakdown"]
        per_host = stats["per_host"]
        avg_trust = stats["avg_trust"]

        confidence = 100.0

        if success_rate < 0.9:
            confidence -= (0.9 - success_rate) * 50.0

        net_err_ratio = error_breakdown.get("network", 0.0)
        if net_err_ratio > 0.3:
            confidence -= 15.0

        if host is not None and host in per_host:
            host_rate = per_host[host]
            if host_rate < 0.7:
                confidence -= (0.7 - host_rate) * 50.0

        if avg_latency > 1000.0:
            confidence -= 10.0

        if avg_trust < 0.8:
            confidence -= (0.8 - avg_trust) * 40.0

        confidence = max(0.0, min(100.0, confidence))

        if confidence >= 80.0 and success_rate >= 0.95:
            state = "HEALTHY"
        elif confidence >= 50.0 and success_rate >= 0.80:
            state = "DEGRADED"
        elif confidence >= 30.0 and success_rate >= 0.50:
            state = "UNSTABLE"
        else:
            state = "CRITICAL"

        return {
            "confidence": confidence,
            "state": state,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "avg_trust": avg_trust,
        }

    def to_dict(self) -> Dict[str, Any]:
        events = self._snapshot_events()
        stats = self.compute_stats()
        sample = []
        for e in events[-20:]:
            sample.append(
                {
                    "timestamp": e.timestamp,
                    "success": e.success,
                    "latency_ms": e.latency_ms,
                    "destination_host": e.destination_host,
                    "error_type": e.error_type,
                    "process_name": e.process_name,
                    "trust_score": e.trust_score,
                    "data_type": e.data_type,
                }
            )
        return {"stats": stats, "sample": sample}

    def from_dict(self, d: Dict[str, Any]):
        sample = d.get("sample", [])
        restored_events: List[DeliveryEvent] = []
        for raw in sample:
            restored_events.append(
                DeliveryEvent(
                    timestamp=float(raw.get("timestamp", time.time())),
                    success=bool(raw.get("success", False)),
                    latency_ms=float(raw.get("latency_ms", 0.0)),
                    destination_host=str(raw.get("destination_host", "")),
                    error_type=str(raw.get("error_type", "other")),
                    process_name=str(raw.get("process_name", "unknown_process")),
                    trust_score=float(raw.get("trust_score", 0.90)),
                    data_type=str(raw.get("data_type", "general")),
                )
            )
        with self._lock:
            self.events = restored_events[-self.max_events:]


# ---------- capsule + policy ----------

@dataclass
class GuardianPolicy:
    destination_host: str
    ttl_seconds: int
    max_attempts: int
    data_type: str


@dataclass
class GuardianCapsule:
    capsule_id: str
    created_at: float
    policy: GuardianPolicy
    payload_description: str
    attempt_count: int = 0
    opened: bool = False
    expired: bool = False
    tampered: bool = False
    last_error: str = ""


# ---------- memory organ (triple redundant) ----------

class MemoryOrgan:
    def __init__(self, default_dir_name: str = "SnowWhiteParentsMagicBox"):
        home = Path.home()
        self.local_dir = home / default_dir_name
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.key_file = self.local_dir / "memory.key"
        self.memory_file_name = "swp_memory.json.enc"

        self.primary_path: Optional[Path] = None
        self.backup2_path: Optional[Path] = None

        self._lock = threading.Lock()
        self.config: Dict[str, Any] = {
            "primary_memory_path": "",
            "backup2_memory_path": "",  # optional
        }

        self._ensure_key()

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

    # Primary path

    def set_primary_path(self, path_str: str):
        with self._lock:
            self.config["primary_memory_path"] = path_str or ""
            if path_str:
                self.primary_path = Path(path_str)
            else:
                self.primary_path = None

    def get_primary_path(self) -> str:
        with self._lock:
            return self.config.get("primary_memory_path", "")

    # Backup 2 path (optional, can be SMB)

    def set_backup2_path(self, path_str: str):
        with self._lock:
            self.config["backup2_memory_path"] = path_str or ""
            if path_str:
                self.backup2_path = Path(path_str)
            else:
                self.backup2_path = None

    def get_backup2_path(self) -> str:
        with self._lock:
            return self.config.get("backup2_memory_path", "")

    # Path resolution

    def _get_memory_paths(self) -> List[Path]:
        paths: List[Path] = []
        with self._lock:
            if self.primary_path:
                paths.append(self.primary_path)
            if self.backup2_path:
                paths.append(self.backup2_path)
        # Backup A: local dir
        paths.append(self.local_dir)
        return paths

    # Encryption

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

    # Save state (write to all available paths)

    def save_state(
        self,
        dest_policy: DestinationPolicy,
        trust_manager: SoftwareTrustManager,
        cortex: HybridCortex,
        baseline_summary: Dict[str, int],
    ):
        state = {
            "version": 1,
            "timestamp": time.time(),
            "system_summary": get_system_summary(),
            "destination_policy": dest_policy.to_dict(),
            "software_trust": trust_manager.to_dict(),
            "cortex": cortex.to_dict(),
            "config": self.config,
            "baseline_summary": baseline_summary,
        }
        encoded = json.dumps(state).encode("utf-8")
        encrypted = self._encrypt(encoded)

        for base_dir in self._get_memory_paths():
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
                mem_path = base_dir / self.memory_file_name
                with open(mem_path, "wb") as f:
                    f.write(encrypted)
            except Exception:
                continue

    # Load state (try each path until one succeeds)

    def load_state(
        self,
        dest_policy: DestinationPolicy,
        trust_manager: SoftwareTrustManager,
        cortex: HybridCortex,
    ) -> Dict[str, int]:
        baseline_summary = {
            "installed_apps": 0,
            "processes": 0,
            "services": 0,
            "startup_items": 0,
        }

        paths = self._get_memory_paths()
        for base_dir in paths:
            try:
                mem_path = base_dir / self.memory_file_name
                if not mem_path.exists():
                    continue
                with open(mem_path, "rb") as f:
                    encrypted = f.read()
                decoded = self._decrypt(encrypted)
                state = json.loads(decoded.decode("utf-8"))
            except Exception:
                continue

            try:
                if isinstance(state, dict):
                    cfg = state.get("config", {})
                    with self._lock:
                        self.config.update(cfg)
                        primary = self.config.get("primary_memory_path", "")
                        backup2 = self.config.get("backup2_memory_path", "")
                        self.primary_path = Path(primary) if primary else None
                        self.backup2_path = Path(backup2) if backup2 else None

                    dest_dict = state.get("destination_policy", {})
                    dest_policy.from_dict(dest_dict)

                    trust_dict = state.get("software_trust", {})
                    trust_manager.from_dict(trust_dict)

                    cortex_dict = state.get("cortex", {})
                    cortex.from_dict(cortex_dict)

                    baseline_summary = state.get("baseline_summary", baseline_summary)
            except Exception:
                pass
            break

        return baseline_summary


# ---------- self-repair ----------

class SelfRepairOrgan:
    def __init__(self, memory: MemoryOrgan, main_file: str):
        self.memory = memory
        self.main_file = Path(main_file).resolve()
        self.hash_file = self.memory.local_dir / "program.hash"
        self.backup_dir = self.memory.local_dir / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_program = self.backup_dir / "swp_program_backup.py"

    def compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def ensure_baseline_backup(self):
        if not self.hash_file.exists():
            h = self.compute_hash(self.main_file)
            with open(self.hash_file, "w", encoding="utf-8") as f:
                f.write(h)
        if not self.backup_program.exists():
            try:
                shutil.copy2(self.main_file, self.backup_program)
            except Exception:
                pass

    def verify_and_repair(self) -> bool:
        if not self.main_file.exists():
            return False

        if not self.hash_file.exists():
            self.ensure_baseline_backup()
            return True

        try:
            with open(self.hash_file, "r", encoding="utf-8") as f:
                expected = f.read().strip()
            current = self.compute_hash(self.main_file)
        except Exception:
            return False

        if current == expected:
            return True

        if self.backup_program.exists():
            try:
                shutil.copy2(self.backup_program, self.main_file)
                h = self.compute_hash(self.main_file)
                with open(self.hash_file, "w", encoding="utf-8") as f:
                    f.write(h)
                return True
            except Exception:
                return False

        return False


# ---------- Windows baseline scanner ----------

class WindowsBaselineScanner:
    def __init__(self):
        self.installed_apps: List[str] = []
        self.processes: List[str] = []
        self.services: List[str] = []
        self.startup_items: List[str] = []

    def _scan_installed_software(self):
        self.installed_apps = []
        if winreg is None:
            return

        uninstall_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]

        for root, subkey in uninstall_paths:
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
                        display_name, _ = winreg.QueryValueEx(app_key, "DisplayName")
                        if display_name:
                            self.installed_apps.append(str(display_name))
                    except OSError:
                        continue
                except OSError:
                    break
            winreg.CloseKey(key)

        self.installed_apps = sorted(set(self.installed_apps))

    def _scan_processes(self):
        names = []
        for proc in psutil.process_iter(attrs=["name"]):
            n = proc.info.get("name") or ""
            if n:
                names.append(n)
        self.processes = sorted(set(names))

    def _scan_services(self):
        names = []
        try:
            for svc in psutil.win_service_iter():
                info = svc.as_dict()
                name = info.get("name") or ""
                if name:
                    names.append(name)
        except Exception:
            pass
        self.services = sorted(set(names))

    def _scan_startup_items(self):
        items: List[str] = []
        if winreg is not None:
            run_keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
            ]
            for root, subkey in run_keys:
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
        self.startup_items = sorted(set(items))

    def run_baseline(self):
        self._scan_installed_software()
        self._scan_processes()
        self._scan_services()
        self._scan_startup_items()

    def get_baseline_summary(self) -> Dict[str, int]:
        return {
            "installed_apps": len(self.installed_apps),
            "processes": len(self.processes),
            "services": len(self.services),
            "startup_items": len(self.startup_items),
        }


# ---------- guardian forge ----------

class GuardianForge:
    def __init__(
        self,
        metrics: GuardianMetrics,
        cortex: HybridCortex,
        dest_policy: DestinationPolicy,
        trust_manager: SoftwareTrustManager,
    ):
        self.metrics = metrics
        self.cortex = cortex
        self.dest_policy = dest_policy
        self.trust_manager = trust_manager

    def _extract_host(self, url: str) -> str:
        parsed = urllib_parse.urlparse(url)
        return parsed.hostname or ""

    def create_capsule(
        self,
        destination_host: str,
        ttl_seconds: int,
        max_attempts: int,
        data_type: str,
        payload_description: str,
    ) -> GuardianCapsule:
        capsule_id = str(uuid.uuid4())
        policy = GuardianPolicy(
            destination_host=destination_host,
            ttl_seconds=ttl_seconds,
            max_attempts=max_attempts,
            data_type=data_type,
        )
        return GuardianCapsule(
            capsule_id=capsule_id,
            created_at=time.time(),
            policy=policy,
            payload_description=payload_description,
        )

    def deliver_http_get(
        self,
        capsule: GuardianCapsule,
        url: str,
        process_name: str,
        timeout: float = 5.0,
    ) -> bool:
        start = time.time()
        host = self._extract_host(url)
        data_type = capsule.policy.data_type

        trust = self.trust_manager.get_trust(process_name)
        ultra_sensitive = data_type in ("password", "biometric")

        if self.dest_policy.is_blocked(host):
            capsule.last_error = f"Destination {host} is blocked by policy"
            self.metrics.increment_policy_blocks()
            self._log_event(False, start, host, "policy", process_name, trust, data_type)
            self.trust_manager.adjust_trust(process_name, -0.05)
            return False

        if data_type in ("personal", "bio", "machine", "password", "biometric"):
            if not self.dest_policy.is_allowed_for_sensitive(host):
                capsule.last_error = (
                    f"Sensitive data type '{data_type}' cannot be sent to unapproved host '{host}'"
                )
                self.metrics.increment_policy_blocks()
                self._log_event(False, start, host, "policy", process_name, trust, data_type)
                self.trust_manager.adjust_trust(process_name, -0.07)
                return False

        now = time.time()
        if now - capsule.created_at > capsule.policy.ttl_seconds:
            capsule.expired = True
            capsule.last_error = "TTL expired before delivery"
            self.metrics.increment_failures()
            self._log_event(False, start, host, "policy", process_name, trust, data_type)
            self.trust_manager.adjust_trust(process_name, -0.02)
            return False

        if capsule.opened or capsule.attempt_count >= capsule.policy.max_attempts:
            capsule.last_error = "Max attempts exceeded or already opened"
            self.metrics.increment_failures()
            self._log_event(False, start, host, "policy", process_name, trust, data_type)
            self.trust_manager.adjust_trust(process_name, -0.01)
            return False

        capsule.attempt_count += 1

        if ultra_sensitive and trust < 0.50:
            capsule.last_error = (
                f"Process '{process_name}' trust too low ({trust:.2f}) for ultra-sensitive '{data_type}'"
            )
            self.metrics.increment_policy_blocks()
            self._log_event(False, start, host, "policy", process_name, trust, data_type)
            self.trust_manager.adjust_trust(process_name, -0.03)
            return False

        success = False
        error_type = "other"
        try:
            response = requests.get(url, timeout=timeout)
        except Exception as e:
            capsule.last_error = f"Network error: {e}"
            error_type = "network"
            self.metrics.increment_failures()
            success = False
            self.trust_manager.adjust_trust(process_name, -0.005)
        else:
            if response.status_code == 200:
                capsule.opened = True
                success = True
                self.metrics.increment_successes()
                self.trust_manager.adjust_trust(process_name, +0.002)
            else:
                capsule.last_error = f"HTTP status {response.status_code}"
                error_type = "http"
                self.metrics.increment_failures()
                success = False
                self.trust_manager.adjust_trust(process_name, -0.01)

        self._log_event(success, start, host, error_type, process_name, trust, data_type)
        return success

    def _log_event(
        self,
        success: bool,
        start_time: float,
        host: str,
        error_type: str,
        process_name: str,
        trust_score: float,
        data_type: str,
    ):
        latency_ms = (time.time() - start_time) * 1000.0
        event = DeliveryEvent(
            timestamp=time.time(),
            success=success,
            latency_ms=latency_ms,
            destination_host=host or "",
            error_type=error_type,
            process_name=process_name or "unknown_process",
            trust_score=trust_score,
            data_type=data_type,
        )
        self.cortex.record_event(event)


# ---------- MagicBox Hybrid GUI (Ribbon + Sidebar) ----------

class SnowWhiteParentsGUI:
    def __init__(
        self,
        metrics: GuardianMetrics,
        forge: GuardianForge,
        cortex: HybridCortex,
        dest_policy: DestinationPolicy,
        trust_manager: SoftwareTrustManager,
        memory: MemoryOrgan,
        baseline_summary: Dict[str, int],
    ):
        self.metrics = metrics
        self.forge = forge
        self.cortex = cortex
        self.dest_policy = dest_policy
        self.trust_manager = trust_manager
        self.memory = memory
        self.baseline_summary = baseline_summary

        self.root = tk.Tk()
        self.root.title("Snow White Parents — MagicBox Console")
        self.root.geometry("1200x750")

        # MagicBox theme colors
        self.bg_main = "#101018"
        self.bg_ribbon = "#181828"
        self.bg_sidebar = "#141424"
        self.bg_panel = "#1e1e30"
        self.accent = "#4fa3ff"
        self.text_main = "#f0f0ff"
        self.text_dim = "#a0a0c0"

        self.root.configure(bg=self.bg_main)

        # Ribbon
        self.ribbon = tk.Frame(self.root, bg=self.bg_ribbon, height=50)
        self.ribbon.pack(side="top", fill="x")

        # Main area: sidebar + content
        main_area = tk.Frame(self.root, bg=self.bg_main)
        main_area.pack(side="top", fill="both", expand=True)

        self.sidebar = tk.Frame(main_area, bg=self.bg_sidebar, width=180)
        self.sidebar.pack(side="left", fill="y")

        self.content = tk.Frame(main_area, bg=self.bg_main)
        self.content.pack(side="left", fill="both", expand=True)

        self.active_tab = tk.StringVar(value="Home")
        self._build_ribbon()
        self._build_sidebar()

        self.panels: Dict[str, tk.Frame] = {}
        self._build_panels()
        self._show_panel("Home")

        self._schedule_update()

    # Ribbon

    def _build_ribbon(self):
        tabs = ["Home", "Security", "Network", "Memory", "Settings"]
        for name in tabs:
            btn = tk.Button(
                self.ribbon,
                text=name,
                font=("Segoe UI", 11, "bold"),
                fg=self.text_main,
                bg=self.bg_ribbon,
                activebackground=self.accent,
                activeforeground=self.text_main,
                bd=0,
                padx=12,
                pady=6,
                command=lambda n=name: self._on_tab_click(n),
            )
            btn.pack(side="left", padx=4, pady=6)

        title_lbl = tk.Label(
            self.ribbon,
            text="Snow White Parents — MagicBox Edition",
            font=("Segoe UI", 11),
            fg=self.text_dim,
            bg=self.bg_ribbon,
        )
        title_lbl.pack(side="right", padx=10)

    def _on_tab_click(self, name: str):
        self.active_tab.set(name)
        self._show_panel(name)
        self._update_sidebar_title(name)

    # Sidebar

    def _build_sidebar(self):
        self.sidebar_title = tk.Label(
            self.sidebar,
            text="Home",
            font=("Segoe UI", 13, "bold"),
            fg=self.accent,
            bg=self.bg_sidebar,
            pady=10,
        )
        self.sidebar_title.pack(anchor="w", padx=10, pady=(10, 5))

        self.sidebar_info = tk.Label(
            self.sidebar,
            text="MagicBox Console",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_sidebar,
            justify="left",
            wraplength=160,
        )
        self.sidebar_info.pack(anchor="w", padx=10)

        self.status_label = tk.Label(
            self.sidebar,
            text="Status: idle",
            font=("Consolas", 9),
            fg=self.text_main,
            bg=self.bg_sidebar,
            justify="left",
            wraplength=160,
        )
        self.status_label.pack(anchor="w", padx=10, pady=(20, 0))

    def _update_sidebar_title(self, tab_name: str):
        self.sidebar_title.config(text=tab_name)

    # Panels

    def _build_panels(self):
        self.panels["Home"] = self._panel_home()
        self.panels["Security"] = self._panel_security()
        self.panels["Network"] = self._panel_network()
        self.panels["Memory"] = self._panel_memory()
        self.panels["Settings"] = self._panel_settings()

        for p in self.panels.values():
            p.pack_forget()

    def _show_panel(self, name: str):
        for p in self.panels.values():
            p.pack_forget()
        panel = self.panels.get(name)
        if panel:
            panel.pack(fill="both", expand=True, padx=10, pady=10)

    # Home Panel

    def _panel_home(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        header = tk.Label(
            frame,
            text="Dashboard Overview",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        header.pack(anchor="w", padx=15, pady=(10, 15))

        metrics_frame = tk.Frame(frame, bg=self.bg_panel)
        metrics_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            metrics_frame,
            text="Delivery Metrics",
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        self.success_label = tk.Label(
            metrics_frame,
            text="Success: 0",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.success_label.pack(anchor="w", pady=1)

        self.failure_label = tk.Label(
            metrics_frame,
            text="Failures: 0",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.failure_label.pack(anchor="w", pady=1)

        self.policy_block_label = tk.Label(
            metrics_frame,
            text="Policy Blocks: 0",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.policy_block_label.pack(anchor="w", pady=1)

        cortex_frame = tk.Frame(frame, bg=self.bg_panel)
        cortex_frame.pack(fill="x", padx=15, pady=(10, 5))

        tk.Label(
            cortex_frame,
            text="Hybrid Cortex",
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        self.health_label = tk.Label(
            cortex_frame,
            text="Health: n/a",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.health_label.pack(anchor="w", pady=1)

        self.confidence_label = tk.Label(
            cortex_frame,
            text="Confidence: n/a",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        self.confidence_label.pack(anchor="w", pady=1)

        baseline_frame = tk.Frame(frame, bg=self.bg_panel)
        baseline_frame.pack(fill="x", padx=15, pady=(10, 10))

        tk.Label(
            baseline_frame,
            text="Baseline Snapshot",
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        base = self.baseline_summary
        tk.Label(
            baseline_frame,
            text=(
                f"Installed apps: {base.get('installed_apps', 0)} | "
                f"Processes: {base.get('processes', 0)} | "
                f"Services: {base.get('services', 0)} | "
                f"Startup: {base.get('startup_items', 0)}"
            ),
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", pady=2)

        return frame

    # Security Panel

    def _panel_security(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        header = tk.Label(
            frame,
            text="Security & Policy",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        header.pack(anchor="w", padx=15, pady=(10, 15))

        policy_frame = tk.Frame(frame, bg=self.bg_panel)
        policy_frame.pack(fill="x", padx=15, pady=5)

        tk.Label(
            policy_frame,
            text="Destination Policy",
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        row = tk.Frame(policy_frame, bg=self.bg_panel)
        row.pack(fill="x", pady=5)

        tk.Label(
            row,
            text="Host:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.host_policy_entry = tk.Entry(row, width=30)
        self.host_policy_entry.insert(0, "example.com")
        self.host_policy_entry.pack(side="left", padx=5)

        btn_allow = tk.Button(
            row,
            text="Allow",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg="#215c2c",
            activebackground="#2b8c3b",
            bd=0,
            padx=8,
            command=self.on_allow_host,
        )
        btn_allow.pack(side="left", padx=4)

        btn_block = tk.Button(
            row,
            text="Block",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_main,
            bg="#7c2020",
            activebackground="#a52a2a",
            bd=0,
            padx=8,
            command=self.on_block_host,
        )
        btn_block.pack(side="left", padx=4)

        self.allow_list_label = tk.Label(
            policy_frame,
            text="Allow: (none)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
        )
        self.allow_list_label.pack(anchor="w", pady=2)

        self.block_list_label = tk.Label(
            policy_frame,
            text="Block: (none)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
        )
        self.block_list_label.pack(anchor="w", pady=2)

        trust_frame = tk.Frame(frame, bg=self.bg_panel)
        trust_frame.pack(fill="both", expand=True, padx=15, pady=(10, 10))

        tk.Label(
            trust_frame,
            text="Software Trust (Top entries)",
            font=("Segoe UI", 12, "bold"),
            fg=self.accent,
            bg=self.bg_panel,
        ).pack(anchor="w")

        self.trust_list_box = tk.Listbox(
            trust_frame,
            width=80,
            height=14,
            font=("Consolas", 9),
            fg=self.text_main,
            bg="#111120",
            selectbackground=self.accent,
            activestyle="none",
        )
        self.trust_list_box.pack(fill="both", expand=True, pady=5)

        return frame

    # Network Panel

    def _panel_network(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        header = tk.Label(
            frame,
            text="Network Delivery Tester",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        header.pack(anchor="w", padx=15, pady=(10, 15))

        test_frame = tk.Frame(frame, bg=self.bg_panel)
        test_frame.pack(fill="x", padx=15, pady=5)

        row1 = tk.Frame(test_frame, bg=self.bg_panel)
        row1.pack(fill="x", pady=4)

        tk.Label(
            row1,
            text="URL:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.url_entry = tk.Entry(row1, width=50)
        self.url_entry.insert(0, "https://example.com")
        self.url_entry.pack(side="left", padx=5)

        row2 = tk.Frame(test_frame, bg=self.bg_panel)
        row2.pack(fill="x", pady=4)

        tk.Label(
            row2,
            text="Data Type:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left")

        self.data_type_var = tk.StringVar(value="general")
        self.data_type_menu = ttk.Combobox(
            row2,
            textvariable=self.data_type_var,
            values=["general", "personal", "bio", "machine", "password", "biometric"],
            width=17,
            state="readonly",
        )
        self.data_type_menu.pack(side="left", padx=5)

        tk.Label(
            row2,
            text="Process:",
            font=("Segoe UI", 10),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(side="left", padx=(10, 0))

        self.proc_entry = tk.Entry(row2, width=20)
        self.proc_entry.insert(0, "test_app.exe")
        self.proc_entry.pack(side="left", padx=5)

        btn_send = tk.Button(
            test_frame,
            text="Send",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.accent,
            activebackground="#6fb5ff",
            bd=0,
            padx=12,
            pady=4,
            command=self.on_send_test,
        )
        btn_send.pack(anchor="w", pady=(8, 4))

        self.last_result_label = tk.Label(
            frame,
            text="Last result: (none)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
            wraplength=1000,
        )
        self.last_result_label.pack(anchor="w", padx=15, pady=(4, 10))

        return frame

    # Memory Panel (Primary + Backup 2)

    def _panel_memory(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        header = tk.Label(
            frame,
            text="Memory & Persistence (Triple Backup)",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        header.pack(anchor="w", padx=15, pady=(10, 15))

        mem_frame = tk.Frame(frame, bg=self.bg_panel)
        mem_frame.pack(fill="x", padx=15, pady=5)

        # Primary path
        tk.Label(
            mem_frame,
            text="Primary Memory Path:",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w")

        row_primary = tk.Frame(mem_frame, bg=self.bg_panel)
        row_primary.pack(fill="x", pady=4)

        self.mem_path_entry = tk.Entry(row_primary, width=60)
        self.mem_path_entry.insert(0, self.memory.get_primary_path())
        self.mem_path_entry.pack(side="left", padx=(0, 5))

        # Backup 2 path (optional)
        tk.Label(
            mem_frame,
            text="Backup Path 2 (optional, local or SMB):",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        ).pack(anchor="w", pady=(10, 0))

        row_backup2 = tk.Frame(mem_frame, bg=self.bg_panel)
        row_backup2.pack(fill="x", pady=4)

        self.mem_backup2_entry = tk.Entry(row_backup2, width=60)
        self.mem_backup2_entry.insert(0, self.memory.get_backup2_path())
        self.mem_backup2_entry.pack(side="left", padx=(0, 5))

        # Save button
        btn_save = tk.Button(
            mem_frame,
            text="Save Memory Paths",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg=self.accent,
            activebackground="#6fb5ff",
            bd=0,
            padx=10,
            pady=3,
            command=self.on_save_memory_paths,
        )
        btn_save.pack(anchor="w", pady=(10, 4))

        # Info + status
        tk.Label(
            mem_frame,
            text=(
                "Triple redundancy:\n"
                " - Primary: user-selected path\n"
                " - Backup A: local MagicBox folder\n"
                " - Backup B: optional path (may be UNC like \\\\SERVER\\Share)\n"
            ),
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
            wraplength=900,
        ).pack(anchor="w", pady=(6, 4))

        self.mem_status_label = tk.Label(
            mem_frame,
            text="Status: (not tested)",
            font=("Segoe UI", 9),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
            wraplength=900,
        )
        self.mem_status_label.pack(anchor="w", pady=(4, 10))

        return frame

    # Settings Panel

    def _panel_settings(self):
        frame = tk.Frame(self.content, bg=self.bg_panel, bd=1, relief="ridge")

        header = tk.Label(
            frame,
            text="Application Settings",
            font=("Segoe UI", 16, "bold"),
            fg=self.text_main,
            bg=self.bg_panel,
        )
        header.pack(anchor="w", padx=15, pady=(10, 15))

        tk.Label(
            frame,
            text="Configure global options for Snow White Parents.\n(Additional controls can be added here.)",
            font=("Segoe UI", 10),
            fg=self.text_dim,
            bg=self.bg_panel,
            justify="left",
        ).pack(anchor="w", padx=15, pady=5)

        btn_exit = tk.Button(
            frame,
            text="Exit Application",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_main,
            bg="#7c2020",
            activebackground="#a52a2a",
            bd=0,
            padx=12,
            pady=4,
            command=self.on_close,
        )
        btn_exit.pack(anchor="w", padx=15, pady=(15, 10))

        return frame

    # Events & Updates

    def on_send_test(self):
        url = self.url_entry.get().strip()
        data_type = self.data_type_var.get()
        process_name = self.proc_entry.get().strip() or "unknown_process"

        if not url:
            self.last_result_label.config(text="Last result: No URL provided")
            return

        parsed = urllib_parse.urlparse(url)
        host = parsed.hostname
        if not host:
            self.last_result_label.config(text="Last result: Invalid URL (no host)")
            return

        capsule = self.forge.create_capsule(
            destination_host=host,
            ttl_seconds=10,
            max_attempts=1,
            data_type=data_type,
            payload_description=f"HTTP GET to {url} with data_type={data_type}",
        )

        def worker():
            self.status_label.config(text="Status: sending...")
            success = self.forge.deliver_http_get(capsule, url, process_name=process_name)
            if success:
                base_msg = (
                    f"Last result: SUCCESS to {host} (capsule {capsule.capsule_id}) "
                    f"[{data_type}, process={process_name}]"
                )
            else:
                base_msg = (
                    f"Last result: FAILURE to {host} - {capsule.last_error} "
                    f"[{data_type}, process={process_name}]"
                )

            brain_host = self.cortex.predict_confidence(host=host)
            host_msg = (
                f"Next for {host}: {brain_host['confidence']:.1f}% "
                f"[{brain_host['state']}] "
                f"(host success_rate={brain_host['success_rate']*100:.1f}%)"
            )
            full_msg = base_msg + "\n" + host_msg

            def update_labels():
                self.last_result_label.config(text=full_msg)
                self.status_label.config(text="Status: idle")

            self.root.after(0, update_labels)

        threading.Thread(target=worker, daemon=True).start()

    def on_allow_host(self):
        host = self.host_policy_entry.get().strip()
        if not host:
            return
        self.dest_policy.add_allow(host)
        self._update_policy_labels()

    def on_block_host(self):
        host = self.host_policy_entry.get().strip()
        if not host:
            return
        self.dest_policy.add_block(host)
        self._update_policy_labels()

    def _update_policy_labels(self):
        snap = self.dest_policy.snapshot()
        allow_text = ", ".join(snap["allow"]) if snap["allow"] else "(none)"
        block_text = ", ".join(snap["block"]) if snap["block"] else "(none)"
        self.allow_list_label.config(text=f"Allow: {allow_text}")
        self.block_list_label.config(text=f"Block: {block_text}")

    def on_save_memory_paths(self):
        primary_str = self.mem_path_entry.get().strip()
        backup2_str = self.mem_backup2_entry.get().strip()

        try:
            if primary_str:
                p1 = Path(primary_str)
                p1.mkdir(parents=True, exist_ok=True)
                self.memory.set_primary_path(primary_str)
            else:
                self.memory.set_primary_path("")

            if backup2_str:
                p2 = Path(backup2_str)
                p2.mkdir(parents=True, exist_ok=True)
                self.memory.set_backup2_path(backup2_str)
            else:
                self.memory.set_backup2_path("")

            self.memory.save_state(
                self.dest_policy, self.trust_manager, self.cortex, self.baseline_summary
            )

            desc = []
            desc.append(f"Primary: {primary_str or '(none)'}")
            desc.append("Backup A: local MagicBox folder")
            desc.append(f"Backup B: {backup2_str or '(none)'}")

            self.mem_status_label.config(
                text="Status: OK\n" + "\n".join(desc)
            )
        except Exception as e:
            self.mem_status_label.config(text=f"Status: ERROR ({e})")

    def _update_metrics_and_brain(self):
        snapshot = self.metrics.snapshot()
        self.success_label.config(text=f"Success: {snapshot['successes']}")
        self.failure_label.config(text=f"Failures: {snapshot['failures']}")
        self.policy_block_label.config(text=f"Policy Blocks: {snapshot['policy_blocks']}")

        brain = self.cortex.predict_confidence()
        self.health_label.config(
            text=(
                f"Health: {brain['state']} | "
                f"Success rate: {brain['success_rate']*100:.1f}% | "
                f"Avg latency: {brain['avg_latency_ms']:.1f} ms | "
                f"Avg trust: {brain['avg_trust']*100:.1f}%"
            )
        )
        self.confidence_label.config(
            text=f"Confidence: {brain['confidence']:.1f}%"
        )

    def _update_trust_list(self):
        entries = self.trust_manager.snapshot()
        entries_sorted = sorted(entries, key=lambda e: e.trust, reverse=True)
        self.trust_list_box.delete(0, tk.END)
        for e in entries_sorted[:20]:
            label = f"{e.name:40s} trust={e.trust*100:5.1f}%"
            self.trust_list_box.insert(tk.END, label)

    def _schedule_update(self):
        current = self.active_tab.get()

        if current == "Home":
            self._update_metrics_and_brain()

        if current == "Security":
            self._update_policy_labels()
            self._update_trust_list()

        self.root.after(500, self._schedule_update)

    def on_close(self):
        try:
            self.memory.save_state(
                self.dest_policy, self.trust_manager, self.cortex, self.baseline_summary
            )
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------- wiring ----------

def main():
    try:
        if platform.system().lower() != "windows":
            print("Snow White Parents (MagicBox) is Windows-focused. Exiting on non-Windows system.")
            return

        memory = MemoryOrgan(default_dir_name="SnowWhiteParentsMagicBox")
        self_repair = SelfRepairOrgan(memory, main_file=__file__)
        ok = self_repair.verify_and_repair()
        if not ok:
            print("Snow White Parents: integrity check failed and self-repair failed. Exiting.")
            return

        metrics = GuardianMetrics()
        cortex = HybridCortex(max_events=500)
        dest_policy = DestinationPolicy()
        trust_manager = SoftwareTrustManager()

        scanner = WindowsBaselineScanner()
        scanner.run_baseline()
        baseline_summary = scanner.get_baseline_summary()

        seed_names = scanner.installed_apps + scanner.processes + ["test_app.exe", "system_service"]
        trust_manager.discover_software(seed_names)

        loaded_baseline = memory.load_state(dest_policy, trust_manager, cortex)
        for key in baseline_summary.keys():
            if baseline_summary[key] == 0 and loaded_baseline.get(key, 0) > 0:
                baseline_summary[key] = loaded_baseline[key]

        forge = GuardianForge(metrics, cortex, dest_policy, trust_manager)

        gui = SnowWhiteParentsGUI(
            metrics=metrics,
            forge=forge,
            cortex=cortex,
            dest_policy=dest_policy,
            trust_manager=trust_manager,
            memory=memory,
            baseline_summary=baseline_summary,
        )
        gui.run()
    except Exception as e:
        print(f"[Snow White Parents] Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()

