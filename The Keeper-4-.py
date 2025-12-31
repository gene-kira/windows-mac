#!/usr/bin/env python3
"""
UNILATERAL GUARDIAN PROTECTOR - ZERO TOLERANCE / ZERO TRUST
WITH:
- PERSISTENT MEMORY (PRIMARY + BACKUP, LOCAL OR SMB)
- WINDOWS-SPECIFIC TRUST MODEL
- MANUAL WHITELIST/BLACKLIST OVERRIDES
- LIVE GUI TRUST CONTROL

Features:
- Cross-platform backbone (Windows, Linux, macOS, Android via Termux)
- OS-aware organs (Windows primary, Linux second)
- Modes: LEARNING (yellow), GUARDIAN (green), PRESIDENTIAL (red)
- Zero-trust default: all processes untrusted unless explicitly trusted
- Zero-tolerance: untrusted + sensitive/rogue behavior => block + PRESIDENTIAL
- Presidential Mode as OS stability guardian (conceptual keepalive/override)
- Identity vault (chameleon + mirror skeleton)
- Network/ad gatekeeper skeleton
- Process watcher skeleton (zero-trust biased)
- Optimizer / anticipation skeleton
- Startup manager skeleton
- System Adaptive Boost engine (detect slowdown, conceptually boost throughput)
- Learning process bar (events observed + predicted work)
- Persistent memory across reboots (JSON file)
  - Primary memory path (local or SMB)
  - Backup memory path (local or SMB)
  - Configurable via guardian_config.json
- Windows-specific trust model:
  - Very selective whitelist for specific EXEs (manual + auto seeds)
  - Blacklist for hostile binaries
  - Everything else hostile by default
- Live GUI controls to whitelist/blacklist EXE paths
- GUI console (tkinter) with mode indicator and status panels
"""

import os
import sys
import platform
import threading
import queue
import time
import json
import traceback
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None


# ============================================================
# Utility and enums
# ============================================================

class Mode:
    LEARNING = "LEARNING"          # yellow
    GUARDIAN = "GUARDIAN"          # green
    PRESIDENTIAL = "PRESIDENTIAL"  # red


MODE_COLOR = {
    Mode.LEARNING: "#FFD84A",
    Mode.GUARDIAN: "#4CAF50",
    Mode.PRESIDENTIAL: "#F44336",
}


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ============================================================
# Configuration for persistent memory paths
# ============================================================

class MemoryConfig:
    """
    Reads/writes a small config file that tells the guardian where
    to keep its persistent memory.

    guardian_config.json structure (in user home):
    {
      "primary_memory_path": "\\\\SERVER\\Share\\guardian_state.json",
      "backup_memory_path": "D:\\GuardianBackup\\guardian_state_backup.json"
    }

    If not present, defaults to home directory guardian_state.json
    with no backup. User can edit the config file manually to use
    SMB network paths or secondary drives.
    """

    def __init__(self):
        self.home = os.path.expanduser("~")
        self.config_path = os.path.join(self.home, "guardian_config.json")
        self.primary_memory_path = None
        self.backup_memory_path = None
        self._load_or_init_defaults()

    def _load_or_init_defaults(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.primary_memory_path = data.get("primary_memory_path")
                self.backup_memory_path = data.get("backup_memory_path")
                log(f"MemoryConfig loaded from {self.config_path}")
            except Exception as e:
                log(f"MemoryConfig load error, using defaults: {e}")
                traceback.print_exc()
        if not self.primary_memory_path:
            # Default primary to local home file
            self.primary_memory_path = os.path.join(self.home, "guardian_state.json")
        # No backup by default unless user configures it
        self._save()

    def _save(self):
        try:
            data = {
                "primary_memory_path": self.primary_memory_path,
                "backup_memory_path": self.backup_memory_path
            }
            tmp_path = self.config_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.config_path)
        except Exception as e:
            log(f"MemoryConfig save error: {e}")
            traceback.print_exc()


# ============================================================
# Persistent memory store
# ============================================================

class MemoryStore:
    """
    JSON-backed store for persistent memory across reboots.
    Uses MemoryConfig for primary + backup paths.
    Stores:
    - learning_stats
    - baseline_data (summarized)
    - trust_registry (whitelist/blacklist/other trust info)
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.primary_path = self.config.primary_memory_path
        self.backup_path = self.config.backup_memory_path
        self.data = {
            "learning_stats": {
                "events_observed": 0,
                "predictions_made": 0,
                "target_events": 1000
            },
            "baseline_data": {},
            "trust_registry": {},
        }
        self._load()

    def _load_from_path(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load(self):
        # Try primary first
        loaded = False
        try:
            if self.primary_path and os.path.exists(self.primary_path):
                raw = self._load_from_path(self.primary_path)
                if isinstance(raw, dict):
                    for key in self.data:
                        if key in raw and isinstance(raw[key], dict):
                            self.data[key].update(raw[key])
                log(f"MemoryStore loaded from primary: {self.primary_path}")
                loaded = True
        except Exception as e:
            log(f"MemoryStore primary load error: {e}")
            traceback.print_exc()

        # If primary failed and backup exists, try backup
        if not loaded and self.backup_path:
            try:
                if os.path.exists(self.backup_path):
                    raw = self._load_from_path(self.backup_path)
                    if isinstance(raw, dict):
                        for key in self.data:
                            if key in raw and isinstance(raw[key], dict):
                                self.data[key].update(raw[key])
                    log(f"MemoryStore loaded from backup: {self.backup_path}")
                    loaded = True
            except Exception as e:
                log(f"MemoryStore backup load error: {e}")
                traceback.print_exc()

        if not loaded:
            log("MemoryStore: no existing state file found, starting fresh.")

    def _save_to_path(self, path):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp_path, path)

    def save(self):
        # Try primary first
        if self.primary_path:
            try:
                self._save_to_path(self.primary_path)
                return
            except Exception as e:
                log(f"MemoryStore primary save error: {e}")
                traceback.print_exc()

        # Fallback to backup
        if self.backup_path:
            try:
                self._save_to_path(self.backup_path)
                log("MemoryStore saved to backup path after primary failure.")
                return
            except Exception as e:
                log(f"MemoryStore backup save error: {e}")
                traceback.print_exc()

        log("MemoryStore: failed to save to both primary and backup paths.")

    def get_learning_stats(self):
        return self.data.get("learning_stats", {})

    def set_learning_stats(self, stats: dict):
        self.data["learning_stats"] = stats

    def get_baseline_data(self):
        return self.data.get("baseline_data", {})

    def set_baseline_data(self, baseline: dict):
        self.data["baseline_data"] = baseline

    def get_trust_registry(self):
        return self.data.get("trust_registry", {})

    def set_trust_registry(self, registry: dict):
        self.data["trust_registry"] = registry


# ============================================================
# OS detection and capability flags
# ============================================================

class SystemProfile:
    def __init__(self):
        self.os_name = None
        self.os_family = None
        self.arch = None
        self.cpu_count = None
        self.mem_total = None

        self.supports_process_inspection = False
        self.supports_network_inspection = False
        self.supports_port_redirect = False
        self.supports_startup_scan = False
        self.supports_identity_masking = True
        self.supports_gui = False

        self.zero_trust = True

    def detect(self):
        self.os_name = platform.system()
        self.arch = platform.machine()
        self.cpu_count = os.cpu_count() or 1

        if psutil:
            try:
                self.mem_total = psutil.virtual_memory().total
            except Exception:
                self.mem_total = None

        name_lower = self.os_name.lower()
        if "windows" in name_lower:
            self.os_family = "windows"
        elif "linux" in name_lower:
            if "android" in platform.platform().lower() or "termux" in os.environ.get("PREFIX", "").lower():
                self.os_family = "android"
            else:
                self.os_family = "linux"
        elif "darwin" in name_lower or "mac" in name_lower:
            self.os_family = "macos"
        else:
            self.os_family = "unknown"

        if psutil:
            self.supports_process_inspection = True

        if self.os_family in ("windows", "linux", "macos"):
            self.supports_network_inspection = True
            self.supports_startup_scan = True
            self.supports_port_redirect = False

        if tk is not None:
            self.supports_gui = True

        return self


# ============================================================
# Core event bus
# ============================================================

class EventBus:
    def __init__(self):
        self.queue = queue.Queue()

    def emit(self, event_type, payload=None):
        self.queue.put({
            "time": datetime.now().isoformat(),
            "type": event_type,
            "payload": payload or {}
        })

    def get_nowait(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


# ============================================================
# Guardian core
# ============================================================

class GuardianCore:
    def __init__(self, system_profile: SystemProfile, event_bus: EventBus, memory_store: MemoryStore):
        self.system = system_profile
        self.event_bus = event_bus
        self.memory_store = memory_store
        self.mode = Mode.LEARNING
        self.lock = threading.Lock()

        self.baseline_data = self.memory_store.get_baseline_data()
        self.situation_context = {
            "last_presidential_event": None,
            "time_of_day": None,
            "day_of_week": None,
        }

        self.boost_active = False
        self.boost_reason = ""
        self.boost_last_start = None

        stored_stats = self.memory_store.get_learning_stats()
        self.learning_stats = {
            "events_observed": stored_stats.get("events_observed", 0),
            "predictions_made": stored_stats.get("predictions_made", 0),
            "target_events": stored_stats.get("target_events", 1000)
        }

        self.presidential_stability_enforced = False

        self.trust_registry = self.memory_store.get_trust_registry()

        self.win_system_roots = []
        if self.system.os_family == "windows":
            win_dir = os.environ.get("WINDIR", r"C:\Windows")
            self.win_system_roots = [
                os.path.normcase(os.path.join(win_dir)),
                os.path.normcase(os.path.join(win_dir, "System32")),
            ]
            # Seed a tiny default whitelist for core binaries (example)
            for seed in [r"C:\Windows\System32\svchost.exe", r"C:\Windows\System32\lsass.exe"]:
                norm = os.path.normcase(seed)
                if norm not in self.trust_registry:
                    self.trust_registry[norm] = {
                        "trusted": True,
                        "reason": "seed_whitelist_core",
                        "source": "whitelist"
                    }

        self._flush_memory()

    # ------------ Persistence helpers ------------

    def _flush_memory(self):
        self.memory_store.set_learning_stats(self.learning_stats)
        self.memory_store.set_baseline_data(self.baseline_data)
        self.memory_store.set_trust_registry(self.trust_registry)
        self.memory_store.save()

    # ------------ Whitelist/blacklist management ------------

    def add_to_whitelist(self, exe_path, reason="manual_whitelist"):
        exe_norm = os.path.normcase(exe_path)
        self.trust_registry[exe_norm] = {
            "trusted": True,
            "reason": reason,
            "source": "whitelist"
        }
        log(f"Whitelist added: {exe_norm} ({reason})")
        self.event_bus.emit("trust_update", {
            "exe": exe_norm,
            "trusted": True,
            "source": "whitelist",
            "reason": reason
        })
        self._flush_memory()

    def add_to_blacklist(self, exe_path, reason="manual_blacklist"):
        exe_norm = os.path.normcase(exe_path)
        self.trust_registry[exe_norm] = {
            "trusted": False,
            "reason": reason,
            "source": "blacklist"
        }
        log(f"Blacklist added: {exe_norm} ({reason})")
        self.event_bus.emit("trust_update", {
            "exe": exe_norm,
            "trusted": False,
            "source": "blacklist",
            "reason": reason
        })
        self._flush_memory()

    # ------------ Mode management ------------

    def set_mode(self, mode: str, reason: str = ""):
        with self.lock:
            if self.mode != mode:
                log(f"Mode change: {self.mode} → {mode} ({reason})")
                self.mode = mode
                self.event_bus.emit("mode_change", {
                    "mode": mode,
                    "reason": reason
                })
                if mode == Mode.PRESIDENTIAL:
                    self.situation_context["last_presidential_event"] = {
                        "time": datetime.now().isoformat(),
                        "reason": reason
                    }
                    self.presidential_stability_enforced = True
                    self.event_bus.emit("presidential_keepalive", {
                        "message": "Presidential Mode enforcing system stability and zero-tolerance defense"
                    })
                else:
                    self.presidential_stability_enforced = False
                self._flush_memory()

    def get_mode(self):
        with self.lock:
            return self.mode

    # ------------ Awareness and baseline ------------

    def update_situation_awareness(self):
        now = datetime.now()
        self.situation_context["time_of_day"] = now.strftime("%H:%M")
        self.situation_context["day_of_week"] = now.strftime("%A")

    def record_baseline_usage(self, key, value):
        if key not in self.baseline_data:
            self.baseline_data[key] = []
        self.baseline_data[key].append({
            "time": datetime.now().isoformat(),
            "value": value
        })
        self.baseline_data[key] = self.baseline_data[key][-200:]
        self.increment_learning_events(1)
        self._flush_memory()

    # ------------ Learning stats ------------

    def increment_learning_events(self, n=1):
        with self.lock:
            self.learning_stats["events_observed"] += n
            self._emit_learning_update()
            self._flush_memory()

    def increment_predictions(self, n=1):
        with self.lock:
            self.learning_stats["predictions_made"] += n
            self._emit_learning_update()
            self._flush_memory()

    def _emit_learning_update(self):
        events = self.learning_stats["events_observed"]
        target = self.learning_stats["target_events"]
        predictions = self.learning_stats["predictions_made"]
        progress = 0.0
        if target > 0:
            progress = min(100.0, (events / target) * 100.0)

        self.event_bus.emit("learning_update", {
            "events_observed": events,
            "predictions_made": predictions,
            "target_events": target,
            "progress_percent": progress
        })

    # ------------ Zero-trust process trust model ------------

    def is_process_trusted(self, proc_info):
        """
        Trust decision order:
        1. Blacklist in registry -> always untrusted.
        2. Whitelist in registry -> trusted.
        3. Windows system paths -> untrusted unless explicitly whitelisted.
        4. Everything else -> untrusted.
        """
        exe = (proc_info.get("exe") or "").strip()
        if not exe:
            return False, "no_exe_path"

        exe_norm = os.path.normcase(exe)
        reg_entry = self.trust_registry.get(exe_norm)

        if reg_entry is not None:
            if reg_entry.get("source") == "blacklist":
                return False, reg_entry.get("reason", "blacklisted")
            if reg_entry.get("source") == "whitelist":
                return bool(reg_entry.get("trusted", True)), reg_entry.get("reason", "whitelisted")

        if self.system.os_family == "windows":
            for root in self.win_system_roots:
                try:
                    if exe_norm.startswith(root.lower()):
                        return False, "windows_system_path_unwhitelisted"
                except Exception:
                    continue
            return False, "windows_non_system_path_unwhitelisted"

        elif self.system.os_family == "linux":
            return False, "linux_zero_trust_default"

        else:
            return False, "unknown_os_zero_trust_default"

    # ------------ Adaptive boost state management ------------

    def activate_boost(self, reason: str, metrics: dict):
        with self.lock:
            if not self.boost_active:
                self.boost_active = True
                self.boost_reason = reason
                self.boost_last_start = datetime.now().isoformat()
                log(f"Adaptive Boost ACTIVATED: {reason}")
                self.event_bus.emit("boost_start", {
                    "reason": reason,
                    "metrics": metrics
                })

    def deactivate_boost(self, reason: str, metrics: dict):
        with self.lock:
            if self.boost_active:
                self.boost_active = False
                log(f"Adaptive Boost DEACTIVATED: {reason}")
                self.event_bus.emit("boost_end", {
                    "reason": reason,
                    "metrics": metrics
                })

    # ------------ Presidential stability enforcement (conceptual) ------------

    def enforce_presidential_stability(self):
        if self.get_mode() == Mode.PRESIDENTIAL:
            self.event_bus.emit("presidential_keepalive", {
                "message": "Presidential Mode actively ensuring smooth OS-level operation in zero-tolerance posture"
            })

    def presidential_override_decision(self, action_description, details=None):
        if self.get_mode() == Mode.PRESIDENTIAL:
            self.event_bus.emit("presidential_override", {
                "action": action_description,
                "details": details or {}
            })

    def presidential_stability_action(self, description, details=None):
        if self.get_mode() == Mode.PRESIDENTIAL:
            self.event_bus.emit("presidential_stability_action", {
                "description": description,
                "details": details or {}
            })

    # ------------ Policy hooks ------------

    def on_sensitive_access_attempt(self, process_info, data_category):
        trusted, trust_reason = self.is_process_trusted(process_info)
        decision = "allow"
        reason = "trusted_process"

        if not trusted:
            decision = "block"
            reason = f"zero_trust_block_{data_category}"
            if self.mode != Mode.PRESIDENTIAL:
                self.set_mode(Mode.PRESIDENTIAL, reason=f"Untrusted process tried to access {data_category}")
            self.event_bus.emit("nonwhitelist_sensitive_touch", {
                "process": process_info,
                "data_category": data_category,
                "trust_reason": trust_reason
            })

        self.increment_learning_events(1)

        self.event_bus.emit("sensitive_access", {
            "process": process_info,
            "data_category": data_category,
            "decision": decision,
            "reason": reason,
            "trust_reason": trust_reason
        })

        return decision

    def on_rogue_activity_detected(self, description, details=None):
        self.increment_learning_events(1)
        self.set_mode(Mode.PRESIDENTIAL, reason=description)
        self.event_bus.emit("rogue_activity", {
            "description": description,
            "details": details or {}
        })


# ============================================================
# Identity vault
# ============================================================

class IdentityVault:
    def __init__(self, guardian: GuardianCore, event_bus: EventBus):
        self.guardian = guardian
        self.event_bus = event_bus
        self.vault_store = {}

    def classify(self, value: str):
        v = value.strip()
        if len(v) == 11 and v[3] == "-" and v[6] == "-":
            return "SSN"
        if "@" in v and "." in v:
            return "EMAIL"
        if any(c.isdigit() for c in v) and any(c.isalpha() for c in v):
            return "MIXED_ID"
        return "GENERIC"

    def store_sensitive(self, key, value):
        category = self.classify(value)
        self.vault_store[key] = {
            "value": value,
            "category": category,
            "time": datetime.now().isoformat()
        }
        self.event_bus.emit("vault_store", {
            "key": key,
            "category": category
        })

    def retrieve_sensitive(self, key, requesting_process):
        entry = self.vault_store.get(key)
        if not entry:
            return None

        decision = self.guardian.on_sensitive_access_attempt(
            process_info=requesting_process,
            data_category=entry["category"]
        )
        if decision == "allow":
            return entry["value"]
        else:
            return None

    @staticmethod
    def mirror_display(value: str):
        if not value:
            return ""
        rev = value[::-1]
        if len(rev) <= 4:
            return "*" * len(rev)
        return rev[:2] + "*" * (len(rev) - 4) + rev[-2:]


# ============================================================
# Organs
# ============================================================

class BaseOrgan:
    def __init__(self, system: SystemProfile, guardian: GuardianCore, event_bus: EventBus):
        self.system = system
        self.guardian = guardian
        self.event_bus = event_bus
        self.running = False
        self.thread = None

    def start(self):
        if self.thread is None:
            self.running = True
            self.thread = threading.Thread(target=self.run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def run_loop(self):
        raise NotImplementedError


class ProcessWatcher(BaseOrgan):
    def run_loop(self):
        log(f"ProcessWatcher started on {self.system.os_family}")
        while self.running:
            try:
                if psutil and self.system.supports_process_inspection:
                    for proc in psutil.process_iter(['pid', 'name', 'exe', 'username']):
                        name = (proc.info.get('name') or "").lower()
                        exe = (proc.info.get('exe') or "")
                        info = {
                            "pid": proc.info.get('pid'),
                            "name": proc.info.get('name'),
                            "exe": exe,
                            "username": proc.info.get('username'),
                            "trusted": False
                        }

                        if any(bad in name for bad in ("crypt", "miner", "steal", "keylog")):
                            self.guardian.on_rogue_activity_detected(
                                description=f"Suspicious process name detected: {name}",
                                details=info
                            )
                            self.event_bus.emit("process_suspicious", {
                                "process": info
                            })
            except Exception as e:
                log(f"ProcessWatcher error: {e}")
                traceback.print_exc()
            time.sleep(5)


class NetworkGatekeeper(BaseOrgan):
    def run_loop(self):
        log(f"NetworkGatekeeper started on {self.system.os_family}")
        while self.running:
            try:
                self.event_bus.emit("network_heartbeat", {
                    "status": "monitoring_zero_trust",
                    "os": self.system.os_family
                })
                self.guardian.increment_learning_events(1)
            except Exception as e:
                log(f"NetworkGatekeeper error: {e}")
                traceback.print_exc()
            time.sleep(10)


class OptimizerBrain(BaseOrgan):
    def __init__(self, system, guardian, event_bus, memory_budget_min_gb=2, memory_budget_max_gb=10):
        super().__init__(system, guardian, event_bus)
        self.memory_budget_min = memory_budget_min_gb * (1024 ** 3)
        self.memory_budget_max = memory_budget_max_gb * (1024 ** 3)

    def run_loop(self):
        log("OptimizerBrain started")
        while self.running:
            try:
                self.guardian.update_situation_awareness()
                if psutil:
                    procs = list(psutil.process_iter(['pid', 'name', 'cpu_percent']))
                    top = sorted(procs, key=lambda p: p.info.get('cpu_percent') or 0, reverse=True)[:5]
                    snapshot = [{"pid": p.info['pid'], "name": p.info['name'], "cpu": p.info['cpu_percent']} for p in top]
                    self.guardian.record_baseline_usage("top_processes", snapshot)

                self.guardian.increment_predictions(1)
                self.event_bus.emit("anticipation_update", {
                    "message": "Anticipation engine baseline pass complete (zero-trust environment)"
                })
            except Exception as e:
                log(f"OptimizerBrain error: {e}")
                traceback.print_exc()
            time.sleep(15)


class StartupManager(BaseOrgan):
    def run_loop(self):
        log(f"StartupManager started on {self.system.os_family}")
        self.event_bus.emit("startup_scan", {
            "status": "not_implemented_zero_trust",
            "os": self.system.os_family
        })
        self.guardian.increment_learning_events(1)
        while self.running:
            time.sleep(30)


class AdaptiveBoostEngine(BaseOrgan):
    def __init__(self, system: SystemProfile, guardian: GuardianCore, event_bus: EventBus):
        super().__init__(system, guardian, event_bus)
        self.cpu_high_threshold = 80.0
        self.cpu_low_threshold = 50.0
        self.mem_high_threshold = 80.0
        self.mem_low_threshold = 60.0
        self.sustain_seconds = 10
        self.last_high_start = None

    def _get_metrics(self):
        cpu = None
        mem_used_pct = None
        if psutil:
            try:
                cpu = psutil.cpu_percent(interval=0.5)
                vm = psutil.virtual_memory()
                mem_used_pct = vm.percent
            except Exception:
                pass
        return cpu, mem_used_pct

    def run_loop(self):
        log("AdaptiveBoostEngine started")
        while self.running:
            try:
                cpu, mem_used = self._get_metrics()
                if cpu is None or mem_used is None:
                    time.sleep(5)
                    continue

                now = time.time()
                metrics = {
                    "cpu_percent": cpu,
                    "mem_used_percent": mem_used,
                    "timestamp": datetime.now().isoformat()
                }

                if cpu >= self.cpu_high_threshold or mem_used >= self.mem_high_threshold:
                    if self.last_high_start is None:
                        self.last_high_start = now
                    elif (now - self.last_high_start) >= self.sustain_seconds:
                        self.guardian.activate_boost(
                            reason="Sustained high system load",
                            metrics=metrics
                        )
                        self.guardian.increment_learning_events(1)
                else:
                    self.last_high_start = None
                    if self.guardian.boost_active and cpu <= self.cpu_low_threshold and mem_used <= self.mem_low_threshold:
                        self.guardian.deactivate_boost(
                            reason="System load normalized",
                            metrics=metrics
                        )

                if self.guardian.get_mode() == Mode.PRESIDENTIAL:
                    self.guardian.enforce_presidential_stability()

            except Exception as e:
                log(f"AdaptiveBoostEngine error: {e}")
                traceback.print_exc()
            time.sleep(2)


# ============================================================
# GUI console (with trust control tab)
# ============================================================

class GuardianGUI:
    def __init__(self, system: SystemProfile, guardian: GuardianCore, event_bus: EventBus):
        self.system = system
        self.guardian = guardian
        self.event_bus = event_bus

        self.root = tk.Tk()
        self.root.title("Unilateral Guardian Protector - Zero Trust")

        self.mode_frame = None
        self.mode_label = None
        self.mode_reason_label = None

        self.notebook = None

        self.log_text = None
        self.network_text = None
        self.process_text = None
        self.presidential_text = None
        self.optimizer_text = None
        self.boost_text = None
        self.learning_text = None
        self.learning_progressbar = None
        self.learning_stats_label = None

        # Trust control widgets
        self.trust_exe_entry = None
        self.trust_status_label = None

        self.last_mode_reason = ""

        self.build_gui()
        self.root.after(500, self.process_events)

    def build_gui(self):
        self.mode_frame = tk.Frame(self.root, height=40)
        self.mode_frame.pack(fill="x")

        self.mode_label = tk.Label(self.mode_frame, text="Mode: ?", font=("Arial", 14, "bold"))
        self.mode_label.pack(side="left", padx=10, pady=5)

        self.mode_reason_label = tk.Label(self.mode_frame, text="", font=("Arial", 9))
        self.mode_reason_label.pack(side="left", padx=10, pady=5)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        frame_log = ttk.Frame(self.notebook)
        self.log_text = tk.Text(frame_log, wrap="word", state="disabled", height=10)
        self.log_text.pack(fill="both", expand=True)
        self.notebook.add(frame_log, text="System & Identity")

        frame_net = ttk.Frame(self.notebook)
        self.network_text = tk.Text(frame_net, wrap="word", state="disabled", height=10)
        self.network_text.pack(fill="both", expand=True)
        self.notebook.add(frame_net, text="Network & Ads")

        frame_proc = ttk.Frame(self.notebook)
        self.process_text = tk.Text(frame_proc, wrap="word", state="disabled", height=10)
        self.process_text.pack(fill="both", expand=True)
        self.notebook.add(frame_proc, text="Processes & Threats")

        frame_pres = ttk.Frame(self.notebook)
        self.presidential_text = tk.Text(frame_pres, wrap="word", state="disabled", height=10)
        self.presidential_text.pack(fill="both", expand=True)
        self.notebook.add(frame_pres, text="Presidential Mode")

        frame_opt = ttk.Frame(self.notebook)
        self.optimizer_text = tk.Text(frame_opt, wrap="word", state="disabled", height=10)
        self.optimizer_text.pack(fill="both", expand=True)
        self.notebook.add(frame_opt, text="Optimizer & Anticipation")

        frame_boost = ttk.Frame(self.notebook)
        self.boost_text = tk.Text(frame_boost, wrap="word", state="disabled", height=10)
        self.boost_text.pack(fill="both", expand=True)
        self.notebook.add(frame_boost, text="Adaptive Boost")

        frame_learning = ttk.Frame(self.notebook)
        self.learning_progressbar = ttk.Progressbar(frame_learning, orient="horizontal", mode="determinate", length=300)
        self.learning_progressbar.pack(pady=10)
        self.learning_stats_label = tk.Label(frame_learning, text="Events: 0 / 1000 | Predictions: 0")
        self.learning_stats_label.pack(pady=5)
        self.learning_text = tk.Text(frame_learning, wrap="word", state="disabled", height=10)
        self.learning_text.pack(fill="both", expand=True)
        self.notebook.add(frame_learning, text="Learning Progress")

        # Trust Control Tab
        frame_trust = ttk.Frame(self.notebook)
        entry_label = tk.Label(frame_trust, text="EXE path to whitelist/blacklist:")
        entry_label.pack(anchor="w", padx=5, pady=5)
        self.trust_exe_entry = tk.Entry(frame_trust, width=80)
        self.trust_exe_entry.pack(fill="x", padx=5, pady=5)

        btn_frame = tk.Frame(frame_trust)
        btn_frame.pack(fill="x", padx=5, pady=5)

        btn_whitelist = tk.Button(btn_frame, text="Whitelist", command=self.on_whitelist_clicked)
        btn_whitelist.pack(side="left", padx=5)

        btn_blacklist = tk.Button(btn_frame, text="Blacklist", command=self.on_blacklist_clicked)
        btn_blacklist.pack(side="left", padx=5)

        self.trust_status_label = tk.Label(frame_trust, text="", fg="blue")
        self.trust_status_label.pack(anchor="w", padx=5, pady=5)

        self.notebook.add(frame_trust, text="Trust Control")

        self.update_mode_display(initial=True)

    def on_whitelist_clicked(self):
        path = self.trust_exe_entry.get().strip()
        if not path:
            self.trust_status_label.configure(text="No path provided.", fg="red")
            return
        self.guardian.add_to_whitelist(path, reason="gui_manual_whitelist")
        self.trust_status_label.configure(text=f"Whitelisted: {path}", fg="green")

    def on_blacklist_clicked(self):
        path = self.trust_exe_entry.get().strip()
        if not path:
            self.trust_status_label.configure(text="No path provided.", fg="red")
            return
        self.guardian.add_to_blacklist(path, reason="gui_manual_blacklist")
        self.trust_status_label.configure(text=f"Blacklisted: {path}", fg="red")

    def update_mode_display(self, initial=False, reason=""):
        mode = self.guardian.get_mode()
        color = MODE_COLOR.get(mode, "#CCCCCC")
        self.mode_frame.configure(bg=color)
        self.mode_label.configure(bg=color, text=f"Mode: {mode}")
        self.mode_reason_label.configure(bg=color, text=reason)

        if initial:
            self.append_log(
                "System & Identity",
                f"System detected: {self.system.os_name} ({self.system.os_family}), arch={self.system.arch}, zero_trust={self.system.zero_trust}\n"
            )

    def append_text(self, widget, text):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def append_log(self, tab, text):
        if tab == "System & Identity":
            self.append_text(self.log_text, text)
        elif tab == "Network & Ads":
            self.append_text(self.network_text, text)
        elif tab == "Processes & Threats":
            self.append_text(self.process_text, text)
        elif tab == "Presidential Mode":
            self.append_text(self.presidential_text, text)
        elif tab == "Optimizer & Anticipation":
            self.append_text(self.optimizer_text, text)
        elif tab == "Adaptive Boost":
            self.append_text(self.boost_text, text)
        elif tab == "Learning Progress":
            self.append_text(self.learning_text, text)

    def update_learning_display(self, events, target, predictions, progress_percent):
        self.learning_progressbar["maximum"] = 100
        self.learning_progressbar["value"] = progress_percent
        self.learning_stats_label.configure(
            text=f"Events: {events} / {target} | Predictions: {predictions} | Progress: {progress_percent:.1f}%"
        )

    def process_events(self):
        while True:
            evt = self.event_bus.get_nowait()
            if not evt:
                break

            etype = evt["type"]
            payload = evt["payload"]

            if etype == "mode_change":
                reason = payload.get("reason", "")
                self.last_mode_reason = reason
                self.update_mode_display(reason=reason)

            elif etype == "sensitive_access":
                proc = payload.get("process", {})
                cat = payload.get("data_category")
                decision = payload.get("decision")
                reason = payload.get("reason")
                trust_reason = payload.get("trust_reason")
                line = (
                    f"[SensitiveAccess] proc={proc.get('name')} pid={proc.get('pid')} exe={proc.get('exe')} "
                    f"category={cat} decision={decision} reason={reason} trust={trust_reason}\n"
                )
                self.append_log("System & Identity", line)

            elif etype == "nonwhitelist_sensitive_touch":
                proc = payload.get("process", {})
                cat = payload.get("data_category")
                trust_reason = payload.get("trust_reason")
                line = (
                    f"[NonWhitelistSensitiveTouch] proc={proc.get('name')} pid={proc.get('pid')} "
                    f"exe={proc.get('exe')} category={cat} trust={trust_reason}\n"
                )
                self.append_log("Presidential Mode", line)

            elif etype == "rogue_activity":
                desc = payload.get("description", "")
                details = payload.get("details", {})
                line = f"[RogueActivity] {desc} details={json.dumps(details)}\n"
                self.append_log("Processes & Threats", line)
                self.append_log("Presidential Mode", line)

            elif etype == "vault_store":
                line = f"[Vault] Stored key={payload.get('key')} category={payload.get('category')}\n"
                self.append_log("System & Identity", line)

            elif etype == "network_heartbeat":
                line = f"[Network] Status={payload.get('status')} os={payload.get('os')}\n"
                self.append_log("Network & Ads", line)

            elif etype == "process_suspicious":
                proc = payload.get("process", {})
                line = f"[SuspiciousProcess] {proc}\n"
                self.append_log("Processes & Threats", line)

            elif etype == "startup_scan":
                line = f"[Startup] Startup scan status={payload.get('status')} os={payload.get('os')}\n"
                self.append_log("System & Identity", line)

            elif etype == "anticipation_update":
                line = f"[Anticipation] {payload.get('message')}\n"
                self.append_log("Optimizer & Anticipation", line)

            elif etype == "boost_start":
                reason = payload.get("reason", "")
                metrics = payload.get("metrics", {})
                line = f"[BoostStart] reason={reason} metrics={json.dumps(metrics)}\n"
                self.append_log("Adaptive Boost", line)

            elif etype == "boost_end":
                reason = payload.get("reason", "")
                metrics = payload.get("metrics", {})
                line = f"[BoostEnd] reason={reason} metrics={json.dumps(metrics)}\n"
                self.append_log("Adaptive Boost", line)

            elif etype == "learning_update":
                events = payload.get("events_observed", 0)
                predictions = payload.get("predictions_made", 0)
                target = payload.get("target_events", 1000)
                progress = payload.get("progress_percent", 0.0)
                self.update_learning_display(events, target, predictions, progress)
                line = (
                    f"[Learning] events={events}, target={target}, predictions={predictions}, "
                    f"progress={progress:.1f}%\n"
                )
                self.append_log("Learning Progress", line)

            elif etype == "presidential_keepalive":
                msg = payload.get("message", "")
                line = f"[PresidentialKeepalive] {msg}\n"
                self.append_log("Presidential Mode", line)

            elif etype == "presidential_override":
                action = payload.get("action", "")
                details = payload.get("details", {})
                line = f"[PresidentialOverride] action={action} details={json.dumps(details)}\n"
                self.append_log("Presidential Mode", line)

            elif etype == "presidential_stability_action":
                desc = payload.get("description", "")
                details = payload.get("details", {})
                line = f"[PresidentialStability] {desc} details={json.dumps(details)}\n"
                self.append_log("Presidential Mode", line)

            elif etype == "trust_update":
                exe = payload.get("exe", "")
                trusted = payload.get("trusted", False)
                src = payload.get("source", "")
                reason = payload.get("reason", "")
                line = f"[TrustUpdate] exe={exe} trusted={trusted} source={src} reason={reason}\n"
                self.append_log("System & Identity", line)

        self.root.after(500, self.process_events)

    def run(self):
        self.root.mainloop()


# ============================================================
# Wiring it all together
# ============================================================

class UnilateralGuardian:
    def __init__(self):
        self.system = SystemProfile().detect()
        self.config = MemoryConfig()
        self.event_bus = EventBus()
        self.memory_store = MemoryStore(self.config)
        self.guardian = GuardianCore(self.system, self.event_bus, self.memory_store)
        self.vault = IdentityVault(self.guardian, self.event_bus)

        self.process_watcher = ProcessWatcher(self.system, self.guardian, self.event_bus)
        self.network_gatekeeper = NetworkGatekeeper(self.system, self.guardian, self.event_bus)
        self.optimizer_brain = OptimizerBrain(self.system, self.guardian, self.event_bus)
        self.startup_manager = StartupManager(self.system, self.guardian, self.event_bus)
        self.boost_engine = AdaptiveBoostEngine(self.system, self.guardian, self.event_bus)

        self.gui = None

    def start_organs(self):
        self.process_watcher.start()
        self.network_gatekeeper.start()
        self.optimizer_brain.start()
        self.startup_manager.start()
        self.boost_engine.start()

    def start_gui(self):
        if not self.system.supports_gui:
            log("No GUI support detected (tkinter missing). Running headless.")
            return
        self.gui = GuardianGUI(self.system, self.guardian, self.event_bus)
        self.gui.run()

    def demo_seed_vault(self):
        self.vault.store_sensitive("primary_ssn", "123-45-6789")
        self.vault.store_sensitive("user_email", "user@example.com")
        self.vault.store_sensitive("account_id", "A9Z7X3C1")

    def run(self):
        log(f"Starting Unilateral Guardian on {self.system.os_name} ({self.system.os_family})")
        log(
            f"Capabilities: process={self.system.supports_process_inspection}, "
            f"network={self.system.supports_network_inspection}, "
            f"startup={self.system.supports_startup_scan}, gui={self.system.supports_gui}, "
            f"zero_trust={self.system.zero_trust}"
        )
        log(f"Memory primary: {self.config.primary_memory_path}")
        log(f"Memory backup: {self.config.backup_memory_path}")

        self.guardian.set_mode(Mode.LEARNING, reason="Initial learning mode in zero-trust posture")

        self.start_organs()
        self.demo_seed_vault()

        if self.system.supports_gui:
            self.start_gui()
        else:
            log("Running in headless mode (no GUI). Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                log("Shutting down...")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    guardian = UnilateralGuardian()
    guardian.run()

