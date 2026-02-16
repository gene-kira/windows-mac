#!/usr/bin/env python3
"""
Synthetic OS Bubble Organism - Borg Cockpit MAX

Targets:
- Windows (Hyper-V, UIAutomation via pywinauto, Windows Firewall)
- Linux (libvirt/KVM, firejail, iptables/nftables, swtpm hooks)

Organs:
- Autoloader
- Persistence (SQLite)
- BorgTrustCore (local + distributed HTTP RPC)
- TelemetrySpine (events, logs, timeline)
- SyntheticTPM (logical TPM + libvirt/swtpm hooks)
- HyperVVtpmManager (real Hyper-V vTPM status/enable)
- PolicyEngine (safe-mode, quarantine, personas)
- WindowsFirewallManager (per-process + per-VM interface rules)
- HyperVNetworkManager (VM -> NIC/interface mapping)
- DriverCheckOrgan (Hyper-V, libvirt, firewall, firejail presence)
- NICEnumerator (host NICs, Hyper-V switches, Linux interfaces)
- VMHeartbeatMonitor (per-VM liveness + latency)
- WombController (multi-VM, Hyper-V/libvirt, network sandbox hooks)
- SandboxManager (Windows Job Objects + Linux firejail + firewall)
- UIAutomationManager (Windows pywinauto)
- GuestAgent (process telemetry, lineage, sandbox + UI reactions)
- PluginManager (dynamic organs)
- Cockpit GUI (tabs: Overview, VMs, Windows, Logs, Timeline, Override, Plugins, Personas, Host/Drivers)
"""

import importlib
import sys
import time
import threading
import json
import os
import platform
import sqlite3
import socket
from typing import Dict, Any, List, Optional, Callable

# ============================================================
#  Autoloader
# ============================================================

REQUIRED_LIBS = [
    "psutil",
    "watchdog",
    "PySide6",
    "pywinauto",
    "requests",
]

def ensure_libs_installed():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[AUTOLOADER] Missing {lib}, attempting install via pip...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                print(f"[AUTOLOADER] Installed {lib}")
            except Exception as e:
                print(f"[AUTOLOADER] Failed to install {lib}: {e}")

ensure_libs_installed()

import psutil
import requests
from PySide6 import QtWidgets, QtCore, QtGui

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAVE_WATCHDOG = True
except ImportError:
    HAVE_WATCHDOG = False

try:
    import libvirt
    HAVE_LIBVIRT = True
except ImportError:
    HAVE_LIBVIRT = False

IS_WINDOWS = platform.system().lower().startswith("win")
IS_LINUX = platform.system().lower().startswith("linux")

# ============================================================
#  Privilege detection
# ============================================================

def is_admin() -> bool:
    if IS_WINDOWS:
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    elif IS_LINUX:
        return os.geteuid() == 0
    return False

HAS_ADMIN = is_admin()

# ============================================================
#  Persistence (SQLite)
# ============================================================

DB_PATH = "bubble_organism.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trust_nodes (
            name TEXT PRIMARY KEY,
            score REAL,
            flags TEXT,
            last_update REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tpm_keys (
            key_id TEXT PRIMARY KEY,
            data TEXT,
            required_trust REAL,
            created REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            level TEXT,
            source TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_trust_nodes(nodes: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for name, node in nodes.items():
        cur.execute("""
            INSERT INTO trust_nodes (name, score, flags, last_update)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                score=excluded.score,
                flags=excluded.flags,
                last_update=excluded.last_update
        """, (name, node["score"], json.dumps(node["flags"]), node["last_update"]))
    conn.commit()
    conn.close()

def load_trust_nodes() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, score, flags, last_update FROM trust_nodes")
    rows = cur.fetchall()
    conn.close()
    nodes = {}
    for name, score, flags, last_update in rows:
        nodes[name] = {
            "name": name,
            "score": score,
            "flags": json.loads(flags),
            "last_update": last_update,
        }
    return nodes

def save_tpm_key(key_id: str, data: str, required_trust: float, created: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tpm_keys (key_id, data, required_trust, created)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key_id) DO UPDATE SET
            data=excluded.data,
            required_trust=excluded.required_trust,
            created=excluded.created
    """, (key_id, data, required_trust, created))
    conn.commit()
    conn.close()

def load_tpm_keys() -> Dict[str, Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT key_id, data, required_trust, created FROM tpm_keys")
    rows = cur.fetchall()
    conn.close()
    keys = {}
    for key_id, data, required_trust, created in rows:
        keys[key_id] = {
            "data": data,
            "required_trust": required_trust,
            "created": created,
        }
    return keys

def log_event(level: str, source: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (ts, level, source, message)
        VALUES (?, ?, ?, ?)
    """, (time.time(), level, source, message))
    conn.commit()
    conn.close()

def load_recent_events(limit: int = 200) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT ts, level, source, message
        FROM events
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        {"ts": ts, "level": level, "source": source, "message": message}
        for ts, level, source, message in rows
    ]

init_db()

# ============================================================
#  Core Data Structures
# ============================================================

class TrustState:
    def __init__(self, name: str):
        self.name = name
        self.score: float = 1.0
        self.flags: List[str] = []
        self.last_update: float = time.time()

    def update(self, delta: float, reason: str):
        self.score = max(0.0, min(1.0, self.score + delta))
        self.flags.append(reason)
        self.last_update = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "flags": list(self.flags),
            "last_update": self.last_update,
        }

# ============================================================
#  Borg Trust Core
# ============================================================

class BorgTrustCore:
    def __init__(self, node_id: str = None, peers: Optional[List[str]] = None):
        self.node_id = node_id or socket.gethostname()
        self.peers = peers or []
        self.nodes: Dict[str, TrustState] = {}
        self.lock = threading.Lock()
        self.remote_nodes: Dict[str, float] = {}
        loaded = load_trust_nodes()
        for name, data in loaded.items():
            ts = TrustState(name)
            ts.score = data["score"]
            ts.flags = data["flags"]
            ts.last_update = data["last_update"]
            self.nodes[name] = ts

    def ensure_node(self, name: str) -> TrustState:
        with self.lock:
            if name not in self.nodes:
                self.nodes[name] = TrustState(name)
            return self.nodes[name]

    def report_behavior(self, node: str, severity: float, description: str):
        ts = self.ensure_node(node)
        ts.update(severity, description)

    def aggregate_trust(self) -> TrustState:
        with self.lock:
            local_scores = [n.score for n in self.nodes.values()] or [1.0]
            remote_scores = list(self.remote_nodes.values()) or [1.0]
            avg_local = sum(local_scores) / len(local_scores)
            avg_remote = sum(remote_scores) / len(remote_scores)
            global_state = TrustState("global")
            global_state.score = (avg_local * 0.7) + (avg_remote * 0.3)
            global_state.flags = [f"{n.name}:{n.score:.2f}" for n in self.nodes.values()]
            global_state.last_update = time.time()
            return global_state

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            nodes_dict = {k: v.to_dict() for k, v in self.nodes.items()}
        save_trust_nodes(nodes_dict)
        return {
            "nodes": nodes_dict,
            "global": self.aggregate_trust().to_dict(),
            "remote": dict(self.remote_nodes),
            "node_id": self.node_id,
        }

    def update_remote_trust(self, node_id: str, score: float):
        self.remote_nodes[node_id] = max(0.0, min(1.0, score))

    def push_trust_to_peers(self):
        payload = {
            "node_id": self.node_id,
            "score": self.aggregate_trust().score,
        }
        for peer in self.peers:
            url = peer.rstrip("/") + "/trust/update"
            try:
                requests.post(url, json=payload, timeout=1.0)
            except Exception:
                continue

# ============================================================
#  Telemetry Spine
# ============================================================

class TelemetrySpine:
    def __init__(self, trust_core: BorgTrustCore):
        self.trust_core = trust_core
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def publish(self, event: Dict[str, Any]):
        node = event.get("node", "unknown")
        severity = float(event.get("severity", 0.0))
        desc = event.get("description", "no-desc")
        level = "INFO" if severity >= 0 else "WARN"
        source = node
        log_event(level, source, desc)
        self.trust_core.report_behavior(node, severity, desc)
        for sub in self.subscribers:
            try:
                sub(event)
            except Exception as e:
                print(f"[TELEMETRY] Subscriber error: {e}")

    def subscribe(self, fn: Callable[[Dict[str, Any]], None]):
        self.subscribers.append(fn)

# ============================================================
#  Synthetic TPM (logical + libvirt/swtpm hooks)
# ============================================================

class SyntheticTPM:
    def __init__(self, trust_core: BorgTrustCore, libvirt_conn=None):
        self.trust_core = trust_core
        self.pcrs: Dict[int, str] = {i: "0" * 64 for i in range(24)}
        self.keys: Dict[str, Dict[str, Any]] = load_tpm_keys()
        self.libvirt_conn = libvirt_conn

    def _trust_gate(self) -> TrustState:
        return self.trust_core.aggregate_trust()

    def extend_pcr(self, index: int, value: str):
        self.pcrs[index] = value

    def read_pcr(self, index: int) -> str:
        ts = self._trust_gate()
        if ts.score < 0.3:
            return "DEGRADED_" + self.pcrs.get(index, "0" * 64)
        return self.pcrs.get(index, "0" * 64)

    def seal_key(self, key_id: str, data: str, required_trust: float = 0.7):
        meta = {
            "data": data,
            "required_trust": required_trust,
            "created": time.time(),
        }
        self.keys[key_id] = meta
        save_tpm_key(key_id, data, required_trust, meta["created"])

    def unseal_key(self, key_id: str) -> Optional[str]:
        ts = self._trust_gate()
        meta = self.keys.get(key_id)
        if not meta:
            return None
        if ts.score >= meta["required_trust"]:
            return meta["data"]
        return None

    def attest(self) -> Dict[str, Any]:
        ts = self._trust_gate()
        return {
            "trust_score": ts.score,
            "trust_flags": ts.flags,
            "pcrs": dict(self.pcrs),
        }

    # libvirt/swtpm backend hooks (best-effort)
    def configure_libvirt_vtpm(self, vm_name: str):
        if not (IS_LINUX and self.libvirt_conn):
            log_event("INFO", "tpm", f"libvirt vTPM config not available for {vm_name}")
            return
        try:
            dom = self.libvirt_conn.lookupByName(vm_name)
        except Exception as e:
            log_event("WARN", "tpm", f"libvirt domain lookup failed for {vm_name}: {e}")
            return
        try:
            xml = dom.XMLDesc()
            if "<tpm" in xml:
                log_event("INFO", "tpm", f"libvirt domain {vm_name} already has TPM device")
                return
            log_event("INFO", "tpm", f"libvirt vTPM requested for {vm_name} (manual swtpm XML edit required)")
        except Exception as e:
            log_event("WARN", "tpm", f"libvirt XML inspect failed for {vm_name}: {e}")

# ============================================================
#  Hyper-V vTPM Manager
# ============================================================

class HyperVVtpmManager:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry

    def _emit(self, level: str, msg: str, sev: float = 0.0):
        print(f"[HVTPM:{level}] {msg}")
        self.telemetry.publish({
            "node": "hvtpm",
            "severity": sev,
            "description": msg,
        })

    def _run_ps(self, script: str) -> (bool, str):
        import subprocess
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            text = (out + "\n" + err).strip()
            if proc.returncode == 0:
                return True, text
            return False, text
        except Exception as e:
            return False, f"Exception: {e}"

    def get_vtpm_status(self, vm_name: str) -> dict:
        if not IS_WINDOWS:
            return {"exists": False, "enabled": False, "has_key_protector": False,
                    "errors": ["Not Windows"]}

        status = {
            "exists": False,
            "enabled": False,
            "has_key_protector": False,
            "errors": []
        }

        ok, out = self._run_ps(
            f"try {{ Get-VM -Name '{vm_name}' | Select-Object -Property Name | ConvertTo-Json -Depth 2 }} "
            f"catch {{ $_.Exception.Message }}"
        )
        if not ok or not out.strip().startswith("{"):
            status["errors"].append(f"Get-VM failed or VM not found: {out}")
            self._emit("WARN", f"Get-VM {vm_name} failed: {out}", sev=-0.05)
            return status

        status["exists"] = True

        ok, out = self._run_ps(
            f"try {{ Get-VMKeyProtector -VMName '{vm_name}' -ErrorAction Stop | "
            f"Select-Object -First 1 | ConvertTo-Json -Depth 3 }} "
            f"catch {{ 'NO_KEY_PROTECTOR' }}"
        )
        if ok and "NO_KEY_PROTECTOR" not in out:
            status["has_key_protector"] = True
        else:
            status["has_key_protector"] = False

        ok, out = self._run_ps(
            f"try {{ Get-VMTPM -VMName '{vm_name}' -ErrorAction Stop | "
            f"Select-Object -First 1 -Property Enabled | ConvertTo-Json -Depth 3 }} "
            f"catch {{ 'NO_TPM' }}"
        )
        if ok and "NO_TPM" not in out and out.strip().startswith("{"):
            try:
                data = json.loads(out)
                enabled = bool(data.get("Enabled", False))
                status["enabled"] = enabled
            except Exception as e:
                status["errors"].append(f"Parse Get-VMTPM JSON failed: {e}")
        else:
            status["enabled"] = False

        return status

    def enable_vtpm(self, vm_name: str) -> dict:
        if not IS_WINDOWS:
            return {"exists": False, "enabled": False, "has_key_protector": False,
                    "errors": ["Not Windows"], "action": "none"}

        if not HAS_ADMIN:
            msg = "Cannot enable vTPM: not running as admin"
            self._emit("WARN", msg, sev=-0.05)
            return {"exists": False, "enabled": False, "has_key_protector": False,
                    "errors": [msg], "action": "none"}

        status = self.get_vtpm_status(vm_name)
        if not status["exists"]:
            status["action"] = "none"
            return status

        if status["enabled"] and status["has_key_protector"]:
            status["action"] = "already_enabled"
            self._emit("INFO", f"vTPM already enabled for {vm_name}", sev=0.0)
            return status

        ok, out = self._run_ps(
            f"try {{ Set-VMKeyProtector -VMName '{vm_name}' -NewLocalKeyProtector -ErrorAction Stop }} "
            f"catch {{ $_.Exception.Message }}"
        )
        if not ok:
            status["errors"].append(f"Set-VMKeyProtector failed: {out}")
            self._emit("WARN", f"Set-VMKeyProtector {vm_name} failed: {out}", sev=-0.05)
            status["action"] = "failed"
            return status

        ok, out = self._run_ps(
            f"try {{ Enable-VMTPM -VMName '{vm_name}' -ErrorAction Stop }} "
            f"catch {{ $_.Exception.Message }}"
        )
        if not ok:
            status["errors"].append(f"Enable-VMTPM failed: {out}")
            self._emit("WARN", f"Enable-VMTPM {vm_name} failed: {out}", sev=-0.05)
            status["action"] = "failed"
            return status

        self._emit("INFO", f"vTPM enabled for {vm_name}", sev=+0.05)
        new_status = self.get_vtpm_status(vm_name)
        new_status["action"] = "enabled"
        return new_status

# ============================================================
#  Policy Engine
# ============================================================

class PolicyEngine:
    def __init__(self, trust_core: BorgTrustCore):
        self.trust_core = trust_core
        self.safe_mode: bool = False
        self.quarantined: bool = False
        self.operator_override_safe: Optional[bool] = None
        self.operator_override_quarantine: Optional[bool] = None
        self.personas: Dict[str, Dict[str, Any]] = {}

    def set_persona(self, vm_name: str, persona: str):
        self.personas[vm_name] = {"persona": persona, "ts": time.time()}

    def evaluate(self):
        ts = self.trust_core.aggregate_trust()
        if self.operator_override_quarantine is not None:
            self.quarantined = self.operator_override_quarantine
        elif ts.score < 0.2:
            self.quarantined = True
        else:
            self.quarantined = False

        if self.operator_override_safe is not None:
            self.safe_mode = self.operator_override_safe
        elif ts.score < 0.5:
            self.safe_mode = True
        else:
            self.safe_mode = False

    def status(self) -> Dict[str, Any]:
        self.evaluate()
        return {
            "safe_mode": self.safe_mode,
            "quarantined": self.quarantined,
            "trust": self.trust_core.aggregate_trust().to_dict(),
            "personas": dict(self.personas),
        }

# ============================================================
#  Windows Firewall Manager (exe paths + interface alias)
# ============================================================

class WindowsFirewallManager:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry

    def _emit(self, level: str, msg: str, sev: float = 0.0):
        print(f"[WF:{level}] {msg}")
        self.telemetry.publish({
            "node": "winfw",
            "severity": sev,
            "description": msg,
        })

    def _run_netsh(self, args: list) -> (bool, str):
        import subprocess
        cmd = ["netsh"] + args
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            text = (out + "\n" + err).strip()
            if proc.returncode == 0:
                return True, text
            return False, text
        except Exception as e:
            return False, f"Exception: {e}"

    def _run_ps(self, script: str) -> (bool, str):
        import subprocess
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            text = (out + "\n" + err).strip()
            if proc.returncode == 0:
                return True, text
            return False, text
        except Exception as e:
            return False, f"Exception: {e}"

    def rule_exists(self, name: str) -> bool:
        ok, out = self._run_netsh(["advfirewall", "firewall", "show", "rule", f"name={name}"])
        if not ok:
            if "No rules match" in out:
                return False
            self._emit("WARN", f"rule_exists({name}) error: {out}", sev=-0.02)
            return False
        return "Rule Name:" in out

    def add_block_rule_for_program(self, name: str, program: str) -> (bool, str):
        if not HAS_ADMIN:
            msg = "Cannot add firewall rule: not running as admin"
            self._emit("WARN", msg, sev=-0.05)
            return False, msg

        if self.rule_exists(name):
            msg = f"Rule '{name}' already exists"
            self._emit("INFO", msg, sev=0.0)
            return True, msg

        args = [
            "advfirewall", "firewall", "add", "rule",
            f"name={name}",
            "dir=out",
            "action=block",
            "enable=yes",
            f"program={program}",
            "profile=any"
        ]
        ok, out = self._run_netsh(args)
        if ok:
            msg = f"Added firewall rule '{name}' for program '{program}'"
            self._emit("INFO", msg, sev=+0.03)
            return True, msg
        else:
            msg = f"Failed to add rule '{name}': {out}"
            self._emit("WARN", msg, sev=-0.05)
            return False, msg

    def add_block_rule_for_interface(self, name: str, interface_alias: str) -> (bool, str):
        if not HAS_ADMIN:
            msg = "Cannot add firewall rule: not running as admin"
            self._emit("WARN", msg, sev=-0.05)
            return False, msg

        if self.rule_exists(name):
            msg = f"Rule '{name}' already exists"
            self._emit("INFO", msg, sev=0.0)
            return True, msg

        script = (
            f"New-NetFirewallRule -DisplayName '{name}' "
            f"-Direction Outbound -Action Block -Enabled True "
            f"-InterfaceAlias '{interface_alias}' -Profile Any"
        )
        ok, out = self._run_ps(script)
        if ok:
            msg = f"Added firewall rule '{name}' for interface '{interface_alias}'"
            self._emit("INFO", msg, sev=+0.03)
            return True, msg
        else:
            msg = f"Failed to add interface rule '{name}': {out}"
            self._emit("WARN", msg, sev=-0.05)
            return False, msg

    def remove_rule(self, name: str) -> (bool, str):
        if not HAS_ADMIN:
            msg = "Cannot remove firewall rule: not running as admin"
            self._emit("WARN", msg, sev=-0.05)
            return False, msg

        if not self.rule_exists(name):
            msg = f"Rule '{name}' does not exist"
            self._emit("INFO", msg, sev=0.0)
            return True, msg

        ok, out = self._run_netsh(["advfirewall", "firewall", "delete", "rule", f"name={name}"])
        if ok:
            msg = f"Deleted firewall rule '{name}'"
            self._emit("INFO", msg, sev=+0.01)
            return True, msg
        else:
            msg = f"Failed to delete rule '{name}': {out}"
            self._emit("WARN", msg, sev=-0.05)
            return False, msg

# ============================================================
#  Hyper-V Network Manager (VM -> NIC mapping)
# ============================================================

class HyperVNetworkManager:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry

    def _emit(self, level: str, msg: str, sev: float = 0.0):
        print(f"[HVNIC:{level}] {msg}")
        self.telemetry.publish({
            "node": "hvnic",
            "severity": sev,
            "description": msg,
        })

    def _run_ps(self, script: str) -> (bool, str):
        import subprocess
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            text = (out + "\n" + err).strip()
            if proc.returncode == 0:
                return True, text
            return False, text
        except Exception as e:
            return False, f"Exception: {e}"

    def get_vm_interfaces(self, vm_name: str) -> List[str]:
        if not IS_WINDOWS:
            return []
        script = (
            f"try {{ Get-VMNetworkAdapter -VMName '{vm_name}' | "
            f"Select-Object -ExpandProperty Name | ConvertTo-Json -Depth 3 }} "
            f"catch {{ $_.Exception.Message }}"
        )
        ok, out = self._run_ps(script)
        if not ok or not out:
            self._emit("WARN", f"Get-VMNetworkAdapter {vm_name} failed: {out}", sev=-0.05)
            return []
        try:
            data = json.loads(out)
            if isinstance(data, list):
                return data
            elif isinstance(data, str):
                return [data]
            else:
                return []
        except Exception as e:
            self._emit("WARN", f"Parse VMNetworkAdapter JSON failed for {vm_name}: {e}", sev=-0.05)
            return []

    def list_hyperv_switches(self) -> List[Dict[str, Any]]:
        if not IS_WINDOWS:
            return []
        script = (
            "try { Get-VMSwitch | Select-Object Name,SwitchType | ConvertTo-Json -Depth 3 } "
            "catch { $_.Exception.Message }"
        )
        ok, out = self._run_ps(script)
        if not ok or not out:
            self._emit("WARN", f"Get-VMSwitch failed: {out}", sev=-0.05)
            return []
        try:
            data = json.loads(out)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
        except Exception as e:
            self._emit("WARN", f"Parse VMSwitch JSON failed: {e}", sev=-0.05)
            return []

# ============================================================
#  Driver Check Organ
# ============================================================

class DriverCheckOrgan:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry
        self.last_status: Dict[str, Any] = {}

    def _emit(self, level: str, msg: str, sev: float = 0.0):
        print(f"[DRV:{level}] {msg}")
        self.telemetry.publish({
            "node": "driver",
            "severity": sev,
            "description": msg,
        })

    def check_hyperv(self):
        if not IS_WINDOWS:
            return {"available": False, "reason": "Not Windows"}
        import subprocess
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-Command", "Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            enabled = "State : Enabled" in out or "Enabled" in out
            return {"available": enabled, "raw": out}
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def check_libvirt(self):
        if not IS_LINUX:
            return {"available": False, "reason": "Not Linux"}
        return {"available": HAVE_LIBVIRT, "reason": "libvirt module present" if HAVE_LIBVIRT else "libvirt module missing"}

    def check_firewall(self):
        if IS_WINDOWS:
            return {"available": True, "reason": "Windows Firewall via netsh/PowerShell"}
        elif IS_LINUX:
            return {"available": True, "reason": "iptables/nftables assumed present (not verified)"}
        return {"available": False, "reason": "Unknown OS"}

    def check_firejail(self):
        if not IS_LINUX:
            return {"available": False, "reason": "Not Linux"}
        import shutil
        fj = shutil.which("firejail")
        return {"available": fj is not None, "path": fj or ""}

    def run_full_check(self):
        hv = self.check_hyperv()
        lv = self.check_libvirt()
        fw = self.check_firewall()
        fj = self.check_firejail()
        status = {
            "hyperv": hv,
            "libvirt": lv,
            "firewall": fw,
            "firejail": fj,
        }
        self.last_status = status
        self._emit("INFO", f"Driver check: {json.dumps(status)}", sev=0.0)
        return status

# ============================================================
#  NIC Enumerator
# ============================================================

class NICEnumerator:
    def __init__(self, telemetry: TelemetrySpine, hv_nic: Optional[HyperVNetworkManager]):
        self.telemetry = telemetry
        self.hv_nic = hv_nic
        self.last_snapshot: Dict[str, Any] = {}

    def _emit(self, level: str, msg: str, sev: float = 0.0):
        print(f"[NIC:{level}] {msg}")
        self.telemetry.publish({
            "node": "nic",
            "severity": sev,
            "description": msg,
        })

    def snapshot(self):
        host_nics = []
        for iface, addrs in psutil.net_if_addrs().items():
            host_nics.append({
                "name": iface,
                "addresses": [a.address for a in addrs if a.address],
            })

        hv_switches = []
        if self.hv_nic:
            hv_switches = self.hv_nic.list_hyperv_switches()

        snapshot = {
            "host_nics": host_nics,
            "hyperv_switches": hv_switches,
        }
        self.last_snapshot = snapshot
        self._emit("INFO", "NIC snapshot updated", sev=0.0)
        return snapshot

# ============================================================
#  VM Heartbeat Monitor
# ============================================================

class VMHeartbeatMonitor:
    def __init__(self, telemetry: TelemetrySpine, womb: "WombController"):
        self.telemetry = telemetry
        self.womb = womb
        self.heartbeats: Dict[str, Dict[str, Any]] = {}

    def _emit(self, level: str, msg: str, sev: float = 0.0):
        print(f"[HB:{level}] {msg}")
        self.telemetry.publish({
            "node": "vmhb",
            "severity": sev,
            "description": msg,
        })

    def tick(self):
        now = time.time()
        for name, info in self.womb.vms.items():
            running = info.get("running", False)
            hb = self.heartbeats.setdefault(name, {"last_seen": None, "latency": None, "status": "unknown"})
            if running:
                if hb["last_seen"] is None:
                    hb["last_seen"] = now
                    hb["latency"] = 0.0
                    hb["status"] = "starting"
                else:
                    hb["latency"] = now - hb["last_seen"]
                    hb["status"] = "alive"
            else:
                hb["status"] = "stopped"
                hb["latency"] = None

    def mark_heartbeat(self, vm_name: str):
        now = time.time()
        hb = self.heartbeats.setdefault(vm_name, {"last_seen": None, "latency": None, "status": "unknown"})
        hb["last_seen"] = now
        hb["status"] = "alive"
        hb["latency"] = 0.0
        self._emit("INFO", f"Heartbeat from {vm_name}", sev=0.0)

    def status(self) -> Dict[str, Any]:
        return self.heartbeats

# ============================================================
#  Womb / VM Controller
# ============================================================

class WombController:
    def __init__(self, telemetry: TelemetrySpine, policy: PolicyEngine):
        self.telemetry = telemetry
        self.policy = policy
        self.vms: Dict[str, Dict[str, Any]] = {}
        self.libvirt_conn = None
        self.win_fw = WindowsFirewallManager(telemetry) if IS_WINDOWS else None
        self.hv_tpm = HyperVVtpmManager(telemetry) if IS_WINDOWS else None
        self.hv_nic = HyperVNetworkManager(telemetry) if IS_WINDOWS else None

        if IS_LINUX and HAVE_LIBVIRT:
            try:
                self.libvirt_conn = libvirt.open("qemu:///system")
                self.telemetry.publish({
                    "node": "womb",
                    "severity": +0.01,
                    "description": "libvirt connection established",
                })
            except Exception as e:
                self.telemetry.publish({
                    "node": "womb",
                    "severity": -0.1,
                    "description": f"libvirt connection failed: {e}",
                })

        if not HAS_ADMIN:
            self.telemetry.publish({
                "node": "womb",
                "severity": -0.05,
                "description": "Running without admin/root: VM and firewall operations may fail",
            })

    def register_vm(self, name: str):
        if name not in self.vms:
            self.vms[name] = {
                "running": False,
                "snapshots": {},
            }

    def _emit_error(self, where: str, msg: str):
        print(f"[WOMB:{where}] {msg}")
        self.telemetry.publish({
            "node": "womb",
            "severity": -0.1,
            "description": f"{where} error: {msg}",
        })

    def _emit_info(self, where: str, msg: str, sev: float = 0.0):
        print(f"[WOMB:{where}] {msg}")
        self.telemetry.publish({
            "node": "womb",
            "severity": sev,
            "description": f"{where}: {msg}",
        })

    # Hyper-V helpers
    def _hyperv_invoke(self, ps_script: str) -> (bool, str):
        import subprocess
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            if proc.returncode == 0:
                return True, (out or "").strip()
            else:
                return False, (err or out).strip()
        except Exception as e:
            return False, f"Exception: {e}"

    def _hyperv_start_vm(self, name: str) -> bool:
        ps = f"Start-VM -Name '{name}' -ErrorAction Stop"
        ok, out = self._hyperv_invoke(ps)
        if not ok:
            self._emit_error("Hyper-V start", out)
            return False
        self._emit_info("Hyper-V start", out or f"VM {name} started", sev=+0.05)
        return True

    def _hyperv_stop_vm(self, name: str) -> bool:
        ps = f"Stop-VM -Name '{name}' -Force -ErrorAction Stop"
        ok, out = self._hyperv_invoke(ps)
        if not ok:
            self._emit_error("Hyper-V stop", out)
            return False
        self._emit_info("Hyper-V stop", out or f"VM {name} stopped", sev=-0.05)
        return True

    def _hyperv_snapshot(self, name: str, snap: str) -> bool:
        ps = f"Checkpoint-VM -Name '{name}' -SnapshotName '{snap}' -ErrorAction Stop"
        ok, out = self._hyperv_invoke(ps)
        if not ok:
            self._emit_error("Hyper-V snapshot", out)
            return False
        self._emit_info("Hyper-V snapshot", out or f"Checkpoint {snap} created", sev=+0.02)
        return True

    def _hyperv_rollback(self, name: str, snap: str) -> bool:
        ps = (
            f"$cp = Get-VMSnapshot -VMName '{name}' -Name '{snap}' -ErrorAction Stop; "
            f"Restore-VMSnapshot -VMName '{name}' -Name '{snap}' -Confirm:$false -ErrorAction Stop"
        )
        ok, out = self._hyperv_invoke(ps)
        if not ok:
            self._emit_error("Hyper-V rollback", out)
            return False
        self._emit_info("Hyper-V rollback", out or f"Checkpoint {snap} restored", sev=-0.01)
        return True

    # libvirt helpers
    def _libvirt_domain(self, name: str):
        if not self.libvirt_conn:
            return None
        try:
            return self.libvirt_conn.lookupByName(name)
        except Exception as e:
            self._emit_error("libvirt lookup", str(e))
            return None

    def _libvirt_start_vm(self, name: str) -> bool:
        dom = self._libvirt_domain(name)
        if not dom:
            return False
        try:
            if dom.isActive() == 0:
                dom.create()
            self._emit_info("libvirt start", f"Domain {name} started", sev=+0.05)
            return True
        except Exception as e:
            self._emit_error("libvirt start", str(e))
            return False

    def _libvirt_stop_vm(self, name: str) -> bool:
        dom = self._libvirt_domain(name)
        if not dom:
            return False
        try:
            if dom.isActive() == 1:
                dom.destroy()
            self._emit_info("libvirt stop", f"Domain {name} stopped", sev=-0.05)
            return True
        except Exception as e:
            self._emit_error("libvirt stop", str(e))
            return False

    def _libvirt_snapshot(self, name: str, snap: str) -> bool:
        dom = self._libvirt_domain(name)
        if not dom:
            return False
        snap_xml = f"""
        <domainsnapshot>
          <name>{snap}</name>
          <description>Created by Synthetic Bubble Organism</description>
        </domainsnapshot>
        """
        try:
            dom.snapshotCreateXML(snap_xml, 0)
            self._emit_info("libvirt snapshot", f"Snapshot {snap} created", sev=+0.02)
            return True
        except Exception as e:
            self._emit_error("libvirt snapshot", str(e))
            return False

    def _libvirt_rollback(self, name: str, snap: str) -> bool:
        dom = self._libvirt_domain(name)
        if not dom:
            return False
        try:
            s = dom.snapshotLookupByName(snap, 0)
            dom.revertToSnapshot(s, 0)
            self._emit_info("libvirt rollback", f"Snapshot {snap} reverted", sev=-0.01)
            return True
        except Exception as e:
            self._emit_error("libvirt rollback", str(e))
            return False

    # network sandbox hooks
    def apply_network_sandbox(self, vm_name: str, level: str) -> (bool, str):
        if not HAS_ADMIN:
            msg = "No admin/root: cannot apply firewall rules"
            self._emit_error("NetSandbox", msg)
            return False, msg

        if IS_WINDOWS and self.win_fw and self.hv_nic:
            interfaces = self.hv_nic.get_vm_interfaces(vm_name)
            if not interfaces:
                msg = f"No VM network adapters found for {vm_name}"
                self._emit_error("NetSandbox", msg)
                return False, msg
            iface = interfaces[0]
            rule_name = f"Bubble_VM_{vm_name}_{level}"
            ok, msg = self.win_fw.add_block_rule_for_interface(rule_name, iface)
            return ok, msg
        elif IS_LINUX:
            import subprocess
            try:
                subprocess.check_output(
                    ["iptables", "-A", "OUTPUT", "-m", "comment", "--comment", f"Bubble_{vm_name}", "-j", "DROP"],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                msg = f"Applied iptables DROP for {vm_name}"
                self._emit_info("NetSandbox", msg, sev=+0.02)
                return True, msg
            except Exception as e:
                msg = f"iptables rule failed: {e}"
                self._emit_error("NetSandbox", msg)
                return False, msg
        else:
            msg = "Net sandbox not supported on this platform"
            self._emit_error("NetSandbox", msg)
            return False, msg

    # vTPM status + configure
    def vtpm_status(self, name: str) -> dict:
        if IS_WINDOWS and self.hv_tpm:
            return self.hv_tpm.get_vtpm_status(name)
        return {"exists": False, "enabled": False, "has_key_protector": False, "errors": ["Not Windows"]}

    def configure_vtpm_for_vm(self, name: str) -> dict:
        if IS_WINDOWS and self.hv_tpm:
            return self.hv_tpm.enable_vtpm(name)
        else:
            self._emit_info("vTPM", f"vTPM config not supported on this platform for {name}", sev=0.0)
            return {"exists": False, "enabled": False, "has_key_protector": False,
                    "errors": ["Not Windows"], "action": "none"}

    # public VM API
    def start_vm(self, name: str):
        self.register_vm(name)
        ok = False
        if IS_WINDOWS:
            ok = self._hyperv_start_vm(name)
        elif IS_LINUX and self.libvirt_conn:
            ok = self._libvirt_start_vm(name)
        if ok:
            self.vms[name]["running"] = True
            self.telemetry.publish({
                "node": f"vm:{name}",
                "severity": +0.05,
                "description": "VM started",
            })

    def stop_vm(self, name: str):
        self.register_vm(name)
        ok = False
        if IS_WINDOWS:
            ok = self._hyperv_stop_vm(name)
        elif IS_LINUX and self.libvirt_conn:
            ok = self._libvirt_stop_vm(name)
        if ok:
            self.vms[name]["running"] = False
            self.telemetry.publish({
                "node": f"vm:{name}",
                "severity": -0.05,
                "description": "VM stopped",
            })

    def snapshot_vm(self, name: str, snap: str):
        self.register_vm(name)
        ok = False
        if IS_WINDOWS:
            ok = self._hyperv_snapshot(name, snap)
        elif IS_LINUX and self.libvirt_conn:
            ok = self._libvirt_snapshot(name, snap)
        if ok:
            self.vms[name]["snapshots"][snap] = {
                "timestamp": time.time(),
                "policy": self.policy.status(),
            }
            self.telemetry.publish({
                "node": f"vm:{name}",
                "severity": +0.02,
                "description": f"Snapshot created: {snap}",
            })

    def rollback_vm(self, name: str, snap: str):
        self.register_vm(name)
        if snap not in self.vms[name]["snapshots"]:
            self._emit_error("rollback_vm", f"Snapshot {snap} not tracked for {name}")
            return
        ok = False
        if IS_WINDOWS:
            ok = self._hyperv_rollback(name, snap)
        elif IS_LINUX and self.libvirt_conn:
            ok = self._libvirt_rollback(name, snap)
        if ok:
            self.telemetry.publish({
                "node": f"vm:{name}",
                "severity": -0.01,
                "description": f"Rollback to snapshot: {snap}",
            })

    def tick(self):
        status = self.policy.status()
        if status["quarantined"]:
            self.telemetry.publish({
                "node": "womb",
                "severity": -0.01,
                "description": "Quarantine active: restricting guests",
            })
        elif status["safe_mode"]:
            self.telemetry.publish({
                "node": "womb",
                "severity": 0.0,
                "description": "Safe mode: guests running with restrictions",
            })

# ============================================================
#  Sandbox Manager
# ============================================================

class SandboxManager:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry
        self.win_fw = WindowsFirewallManager(telemetry) if IS_WINDOWS else None
        self.pending_linux_sandbox: List[str] = []
        if IS_WINDOWS:
            self._init_windows_job_object()
        else:
            self.job_handle = None
        if IS_LINUX:
            self.firejail_available = self._check_firejail()
        else:
            self.firejail_available = False

        if not HAS_ADMIN:
            self._emit_info("Init", "No admin/root: sandboxing will be best-effort only", sev=-0.01)

    def _emit_error(self, where: str, msg: str):
        print(f"[SANDBOX:{where}] {msg}")
        self.telemetry.publish({
            "node": "sandbox",
            "severity": -0.05,
            "description": f"{where} error: {msg}",
        })

    def _emit_info(self, where: str, msg: str, sev: float = 0.0):
        print(f"[SANDBOX:{where}] {msg}")
        self.telemetry.publish({
            "node": "sandbox",
            "severity": sev,
            "description": f"{where}: {msg}",
        })

    def _init_windows_job_object(self):
        import ctypes
        from ctypes import wintypes
        self.job_handle = None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CreateJobObjectW = kernel32.CreateJobObjectW
        CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        CreateJobObjectW.restype = wintypes.HANDLE
        hJob = CreateJobObjectW(None, "SyntheticBubbleJobObject")
        if not hJob:
            err = ctypes.get_last_error()
            self._emit_error("WinJobInit", f"CreateJobObject failed: {err}")
            return
        self.job_handle = hJob
        self._emit_info("WinJobInit", "Job Object created", sev=+0.01)

    def _windows_attach_process(self, pid: int):
        if not self.job_handle:
            self._emit_error("WinAttach", "Job handle not initialized")
            return
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        OpenProcess.restype = wintypes.HANDLE
        AssignProcessToJobObject = kernel32.AssignProcessToJobObject
        AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        AssignProcessToJobObject.restype = wintypes.BOOL
        PROCESS_ALL_ACCESS = 0x1F0FFF
        hProc = OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if not hProc:
            err = ctypes.get_last_error()
            self._emit_error("WinAttach", f"OpenProcess failed for PID {pid}: {err}")
            return
        ok = AssignProcessToJobObject(self.job_handle, hProc)
        if not ok:
            err = ctypes.get_last_error()
            self._emit_error("WinAttach", f"AssignProcessToJobObject failed for PID {pid}: {err}")
        else:
            self._emit_info("WinAttach", f"Attached PID {pid} to Job Object", sev=+0.01)

    def _check_firejail(self) -> bool:
        import shutil
        return shutil.which("firejail") is not None

    def _linux_launch_in_firejail(self, cmd: List[str]):
        import subprocess
        if not self.firejail_available:
            self._emit_error("Firejail", "firejail not available on system")
            return
        full_cmd = ["firejail", "--quiet"] + cmd
        try:
            subprocess.Popen(full_cmd)
            self._emit_info("Firejail", f"Launched: {' '.join(full_cmd)}", sev=+0.02)
        except Exception as e:
            self._emit_error("Firejail", f"Launch failed: {e}")

    def apply_network_policy(self, proc_name: str, level: str):
        if IS_WINDOWS and self.win_fw:
            proc_name_lower = proc_name.lower()
            seen_paths = set()
            for proc in psutil.process_iter(attrs=["name", "exe"]):
                name = (proc.info.get("name") or "").lower()
                if proc_name_lower in name:
                    exe_path = proc.info.get("exe") or ""
                    if exe_path and exe_path not in seen_paths:
                        seen_paths.add(exe_path)
                        rule_name = f"BubbleProc_{os.path.basename(exe_path)}_{level}"
                        self.win_fw.add_block_rule_for_program(rule_name, exe_path)
            if not seen_paths:
                self._emit_info("NetPolicy", f"No processes found matching '{proc_name}' for firewall rules", sev=0.0)
        elif IS_LINUX:
            import subprocess
            try:
                subprocess.check_output(
                    ["iptables", "-A", "OUTPUT", "-m", "comment", "--comment", f"BubbleProc_{proc_name}", "-j", "DROP"],
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self._emit_info("NetPolicy", f"Applied iptables DROP for {proc_name}", sev=+0.02)
            except Exception as e:
                self._emit_error("NetPolicy", f"iptables rule failed: {e}")
        else:
            self._emit_error("NetPolicy", "Unsupported platform")

    def apply_sandbox(self, proc_name: str, level: str):
        proc_name_lower = proc_name.lower()
        if IS_WINDOWS:
            count = 0
            for proc in psutil.process_iter(attrs=["pid", "name"]):
                name = (proc.info.get("name") or "").lower()
                if proc_name_lower in name:
                    self._windows_attach_process(proc.info["pid"])
                    count += 1
            self._emit_info("ApplySandboxWin",
                            f"Requested sandbox level={level} for '{proc_name}', attached {count} processes",
                            sev=0.0)
        elif IS_LINUX:
            if proc_name_lower not in self.pending_linux_sandbox:
                self.pending_linux_sandbox.append(proc_name_lower)
            self._emit_info("ApplySandboxLinux",
                            f"Requested sandbox level={level} for '{proc_name}'. "
                            f"Marking for firejail on next launch.",
                            sev=0.0)
        else:
            self._emit_error("ApplySandbox", "Unsupported platform")

    def launch_sandboxed_process(self, cmd: List[str]):
        import subprocess
        if IS_LINUX and self.firejail_available:
            self._linux_launch_in_firejail(cmd)
        else:
            try:
                subprocess.Popen(cmd)
                self._emit_info("LaunchPlain", f"Launched: {' '.join(cmd)}", sev=0.0)
            except Exception as e:
                self._emit_error("LaunchPlain", f"Launch failed: {e}")

# ============================================================
#  UI Automation Manager
# ============================================================

class UIAutomationManager:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry
        self.is_windows = IS_WINDOWS
        self.is_linux = IS_LINUX
        if self.is_windows:
            try:
                from pywinauto import Desktop
                self.Desktop = Desktop
                self._emit_info("Init", "pywinauto Desktop ready", sev=+0.01)
            except Exception as e:
                self.Desktop = None
                self._emit_error("Init", f"pywinauto init failed: {e}")
        else:
            self.Desktop = None

    def _emit_error(self, where: str, msg: str):
        print(f"[UIAUTO:{where}] {msg}")
        self.telemetry.publish({
            "node": "uiauto",
            "severity": -0.05,
            "description": f"{where} error: {msg}",
        })

    def _emit_info(self, where: str, msg: str, sev: float = 0.0):
        print(f"[UIAUTO:{where}] {msg}")
        self.telemetry.publish({
            "node": "uiauto",
            "severity": sev,
            "description": f"{where}: {msg}",
        })

    def list_windows(self) -> List[dict]:
        if not (self.is_windows and self.Desktop):
            return []
        wins = []
        try:
            for w in self.Desktop(backend="uia").windows():
                try:
                    title = w.window_text()
                    cls = w.friendly_class_name()
                    if title or cls:
                        wins.append({"title": title, "class": cls})
                except Exception:
                    continue
        except Exception as e:
            self._emit_error("ListWindows", str(e))
        return wins

    def close_windows_matching(self, keyword: str) -> int:
        if not (self.is_windows and self.Desktop):
            return 0
        kw = keyword.lower()
        count = 0
        try:
            for w in self.Desktop(backend="uia").windows():
                try:
                    title = w.window_text() or ""
                    if kw in title.lower():
                        w.close()
                        count += 1
                except Exception:
                    continue
            if count:
                self._emit_info("CloseWindows", f"Closed {count} windows matching '{keyword}'", sev=+0.02)
        except Exception as e:
            self._emit_error("CloseWindows", str(e))
        return count

# ============================================================
#  Plugin Manager
# ============================================================

class PluginManager:
    def __init__(self, telemetry: TelemetrySpine):
        self.telemetry = telemetry
        self.plugins: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def register_plugin(self, name: str, handler: Callable[[Dict[str, Any]], None]):
        self.plugins[name] = handler
        self.telemetry.publish({
            "node": "plugin",
            "severity": +0.01,
            "description": f"Plugin registered: {name}",
        })

    def handle_event(self, event: Dict[str, Any]):
        for name, handler in self.plugins.items():
            try:
                handler(event)
            except Exception as e:
                self.telemetry.publish({
                    "node": "plugin",
                    "severity": -0.05,
                    "description": f"Plugin {name} error: {e}",
                })

# ============================================================
#  Filesystem Telemetry
# ============================================================

class FSHandler(FileSystemEventHandler):
    def __init__(self, telemetry: TelemetrySpine):
        super().__init__()
        self.telemetry = telemetry

    def on_modified(self, event):
        self.telemetry.publish({
            "node": "fs",
            "severity": 0.0,
            "description": f"File modified: {event.src_path}",
        })

def start_fs_monitor(path: str, telemetry: TelemetrySpine):
    if not HAVE_WATCHDOG:
        print("[FS] watchdog not available, skipping FS telemetry")
        return None
    observer = Observer()
    handler = FSHandler(telemetry)
    observer.schedule(handler, path, recursive=True)
    observer.start()
    return observer

# ============================================================
#  Guest Agent
# ============================================================

class GuestAgent:
    def __init__(self, telemetry: TelemetrySpine, sandbox: SandboxManager, ui_auto: UIAutomationManager):
        self.telemetry = telemetry
        self.sandbox = sandbox
        self.ui_auto = ui_auto
        self.lineage: Dict[int, int] = {}

    def report_process_snapshot(self):
        for proc in psutil.process_iter(attrs=["pid", "ppid", "name", "cpu_percent", "memory_percent"]):
            info = proc.info
            pid = info["pid"]
            ppid = info.get("ppid", 0)
            self.lineage[pid] = ppid
            name = info.get("name") or "unknown"
            cpu = info.get("cpu_percent", 0.0)
            mem = info.get("memory_percent", 0.0)
            severity = 0.0
            desc = f"proc={name} pid={pid} ppid={ppid} cpu={cpu} mem={mem}"
            if cpu > 50 or mem > 20:
                severity = -0.05
                desc += " (resource-heavy)"
            self.telemetry.publish({
                "node": f"proc:{name}",
                "severity": severity,
                "description": desc,
            })

    def enforce_sandbox_policies(self):
        suspicious = ["unknown", "cmd.exe", "powershell.exe", "bash", "sh"]
        for proc in psutil.process_iter(attrs=["name"]):
            name = (proc.info.get("name") or "").lower()
            if any(s in name for s in suspicious):
                self.sandbox.apply_sandbox(name, "high")
                self.sandbox.apply_network_policy(name, "high")

        if IS_WINDOWS and self.ui_auto:
            for kw in ["PowerShell", "Command Prompt", "cmd.exe"]:
                self.ui_auto.close_windows_matching(kw)

# ============================================================
#  Trust Graph Widget
# ============================================================

class TrustGraphWidget(QtWidgets.QWidget):
    def __init__(self, trust_core: BorgTrustCore, parent=None):
        super().__init__(parent)
        self.trust_core = trust_core
        self.setMinimumHeight(120)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#111111"))

        trust = self.trust_core.to_dict()
        nodes = trust["nodes"]
        if not nodes:
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "No trust nodes yet")
            return

        bar_width = max(20, rect.width() // max(1, len(nodes)))
        max_height = rect.height() - 20
        x = 10
        for name, data in nodes.items():
            score = data["score"]
            h = int(max_height * score)
            y = rect.bottom() - h - 10
            color = QtGui.QColor("#00ff00") if score > 0.7 else QtGui.QColor("#ffaa00") if score > 0.4 else QtGui.QColor("#ff0000")
            painter.fillRect(QtCore.QRect(x, y, bar_width - 4, h), color)
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.drawText(x, rect.bottom() - 2, name[:6])
            x += bar_width

# ============================================================
#  Cockpit GUI
# ============================================================

class CockpitWindow(QtWidgets.QMainWindow):
    def __init__(self,
                 trust_core: BorgTrustCore,
                 tpm: SyntheticTPM,
                 policy: PolicyEngine,
                 womb: WombController,
                 ui_auto: UIAutomationManager,
                 plugin_mgr: PluginManager,
                 driver_check: DriverCheckOrgan,
                 nic_enum: NICEnumerator,
                 hb_monitor: VMHeartbeatMonitor):
        super().__init__()
        self.trust_core = trust_core
        self.tpm = tpm
        self.policy = policy
        self.womb = womb
        self.ui_auto = ui_auto
        self.plugin_mgr = plugin_mgr
        self.driver_check = driver_check
        self.nic_enum = nic_enum
        self.hb_monitor = hb_monitor

        self.setWindowTitle("Synthetic Bubble Borg Cockpit MAX")
        self.resize(1300, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        if not HAS_ADMIN:
            warn = QtWidgets.QLabel("WARNING: Not running as admin/root. VM, firewall, and sandbox operations may be limited.")
            warn.setStyleSheet("color: #ff5555;")
            layout.addWidget(warn)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self._build_overview_tab()
        self._build_vms_tab()
        self._build_windows_tab()
        self._build_logs_tab()
        self._build_timeline_tab()
        self._build_override_tab()
        self._build_plugins_tab()
        self._build_personas_tab()
        self._build_host_tab()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_all)
        self.timer.start(1000)

    def _build_overview_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Overview")
        layout = QtWidgets.QVBoxLayout(tab)

        self.trust_graph = TrustGraphWidget(self.trust_core)
        layout.addWidget(self.trust_graph)

        self.overview_text = QtWidgets.QPlainTextEdit()
        self.overview_text.setReadOnly(True)
        layout.addWidget(self.overview_text)

        self.safe_label = QtWidgets.QLabel("Safe-mode / Quarantine status")
        layout.addWidget(self.safe_label)

    def _build_vms_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "VMs")
        layout = QtWidgets.QVBoxLayout(tab)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.vm_list = QtWidgets.QListWidget()
        top.addWidget(self.vm_list)

        btns = QtWidgets.QVBoxLayout()
        top.addLayout(btns)

        self.btn_vm_start = QtWidgets.QPushButton("Start VM")
        self.btn_vm_stop = QtWidgets.QPushButton("Stop VM")
        self.btn_vm_snap = QtWidgets.QPushButton("Snapshot VM")
        self.btn_vm_roll = QtWidgets.QPushButton("Rollback VM")
        self.btn_vm_net = QtWidgets.QPushButton("Apply Net Sandbox")
        self.btn_vm_vtpm = QtWidgets.QPushButton("Configure vTPM")

        for b in [self.btn_vm_start, self.btn_vm_stop, self.btn_vm_snap, self.btn_vm_roll, self.btn_vm_net, self.btn_vm_vtpm]:
            btns.addWidget(b)

        self.btn_vm_start.clicked.connect(self.on_vm_start)
        self.btn_vm_stop.clicked.connect(self.on_vm_stop)
        self.btn_vm_snap.clicked.connect(self.on_vm_snapshot)
        self.btn_vm_roll.clicked.connect(self.on_vm_rollback)
        self.btn_vm_net.clicked.connect(self.on_vm_net_sandbox)
        self.btn_vm_vtpm.clicked.connect(self.on_vm_vtpm)

        self.vtpm_status_label = QtWidgets.QLabel("vTPM: n/a")
        layout.addWidget(self.vtpm_status_label)

        self.hb_view = QtWidgets.QPlainTextEdit()
        self.hb_view.setReadOnly(True)
        layout.addWidget(self.hb_view)

    def _current_vm_name(self) -> Optional[str]:
        item = self.vm_list.currentItem()
        return item.text().split(" [", 1)[0] if item else None

    def on_vm_start(self):
        name = self._current_vm_name()
        if name:
            self.womb.start_vm(name)

    def on_vm_stop(self):
        name = self._current_vm_name()
        if name:
            self.womb.stop_vm(name)

    def on_vm_snapshot(self):
        name = self._current_vm_name()
        if not name:
            return
        snap, ok = QtWidgets.QInputDialog.getText(self, "Snapshot", "Name:")
        if ok and snap:
            self.womb.snapshot_vm(name, snap)

    def on_vm_rollback(self):
        name = self._current_vm_name()
        if not name:
            return
        vm = self.womb.vms.get(name, {})
        snaps = list(vm.get("snapshots", {}).keys())
        if not snaps:
            QtWidgets.QMessageBox.information(self, "Rollback", "No snapshots available")
            return
        snap, ok = QtWidgets.QInputDialog.getItem(self, "Rollback", "Snapshot:", snaps, 0, False)
        if ok and snap:
            self.womb.rollback_vm(name, snap)

    def on_vm_net_sandbox(self):
        name = self._current_vm_name()
        if name:
            ok, msg = self.womb.apply_network_sandbox(name, "high")
            QtWidgets.QMessageBox.information(self, "Net Sandbox", msg)

    def on_vm_vtpm(self):
        name = self._current_vm_name()
        if name:
            st = self.womb.configure_vtpm_for_vm(name)
            if st.get("action") == "enabled":
                msg = f"vTPM enabled for {name}"
            elif st.get("action") == "already_enabled":
                msg = f"vTPM already enabled for {name}"
            else:
                msg = f"vTPM operation result for {name}:\n{json.dumps(st, indent=2)}"
            QtWidgets.QMessageBox.information(self, "vTPM", msg)

    def _build_windows_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Windows/UI")
        layout = QtWidgets.QVBoxLayout(tab)

        self.win_list = QtWidgets.QListWidget()
        layout.addWidget(self.win_list)

        bottom = QtWidgets.QHBoxLayout()
        layout.addLayout(bottom)

        self.nuke_pattern_edit = QtWidgets.QLineEdit()
        self.nuke_pattern_edit.setPlaceholderText("Pattern (e.g., PowerShell, cmd.exe)")
        bottom.addWidget(self.nuke_pattern_edit)

        self.btn_nuke = QtWidgets.QPushButton("Nuke Suspicious UIs")
        bottom.addWidget(self.btn_nuke)

        self.btn_nuke.clicked.connect(self.on_nuke_ui)

    def on_nuke_ui(self):
        pattern = self.nuke_pattern_edit.text().strip()
        if not pattern:
            pattern = "PowerShell"
        if IS_WINDOWS:
            closed = self.ui_auto.close_windows_matching(pattern)
            QtWidgets.QMessageBox.information(self, "Nuke UIs", f"Closed {closed} windows matching '{pattern}'")
        else:
            QtWidgets.QMessageBox.information(self, "Nuke UIs", "UI automation not implemented on this platform")

    def _build_logs_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Logs")
        layout = QtWidgets.QVBoxLayout(tab)
        self.logs_view = QtWidgets.QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        layout.addWidget(self.logs_view)

    def _build_timeline_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Timeline")
        layout = QtWidgets.QVBoxLayout(tab)
        self.timeline_view = QtWidgets.QPlainTextEdit()
        self.timeline_view.setReadOnly(True)
        layout.addWidget(self.timeline_view)

    def _build_override_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Override")
        layout = QtWidgets.QVBoxLayout(tab)

        self.chk_force_safe = QtWidgets.QCheckBox("Force Safe Mode")
        self.chk_force_quarantine = QtWidgets.QCheckBox("Force Quarantine")
        layout.addWidget(self.chk_force_safe)
        layout.addWidget(self.chk_force_quarantine)

        self.chk_force_safe.stateChanged.connect(self.on_override_changed)
        self.chk_force_quarantine.stateChanged.connect(self.on_override_changed)

    def on_override_changed(self):
        self.policy.operator_override_safe = self.chk_force_safe.isChecked()
        self.policy.operator_override_quarantine = self.chk_force_quarantine.isChecked()

    def _build_plugins_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Plugins")
        layout = QtWidgets.QVBoxLayout(tab)
        self.plugins_view = QtWidgets.QPlainTextEdit()
        self.plugins_view.setReadOnly(True)
        layout.addWidget(self.plugins_view)

    def _build_personas_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Personas/Distributed")
        layout = QtWidgets.QVBoxLayout(tab)

        self.persona_vm_edit = QtWidgets.QLineEdit()
        self.persona_vm_edit.setPlaceholderText("VM name")
        self.persona_name_edit = QtWidgets.QLineEdit()
        self.persona_name_edit.setPlaceholderText("Persona (e.g., 'RedTeam', 'Banking')")
        self.btn_set_persona = QtWidgets.QPushButton("Set Persona")

        layout.addWidget(self.persona_vm_edit)
        layout.addWidget(self.persona_name_edit)
        layout.addWidget(self.btn_set_persona)

        self.btn_set_persona.clicked.connect(self.on_set_persona)

        self.distributed_view = QtWidgets.QPlainTextEdit()
        self.distributed_view.setReadOnly(True)
        layout.addWidget(self.distributed_view)

    def _build_host_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Host/Drivers")
        layout = QtWidgets.QVBoxLayout(tab)

        self.driver_view = QtWidgets.QPlainTextEdit()
        self.driver_view.setReadOnly(True)
        layout.addWidget(self.driver_view)

        self.nic_view = QtWidgets.QPlainTextEdit()
        self.nic_view.setReadOnly(True)
        layout.addWidget(self.nic_view)

        self.btn_refresh_host = QtWidgets.QPushButton("Refresh Host/Drivers")
        layout.addWidget(self.btn_refresh_host)
        self.btn_refresh_host.clicked.connect(self.on_refresh_host)

    def on_set_persona(self):
        vm = self.persona_vm_edit.text().strip()
        persona = self.persona_name_edit.text().strip()
        if vm and persona:
            self.policy.set_persona(vm, persona)

    def on_refresh_host(self):
        self.driver_check.run_full_check()
        self.nic_enum.snapshot()

    def refresh_all(self):
        self.trust_graph.update()
        self.trust_graph.repaint()

        trust = self.trust_core.to_dict()
        policy = self.policy.status()
        tpm_attest = self.tpm.attest()
        overview = {
            "trust_global": trust["global"],
            "safe_mode": policy["safe_mode"],
            "quarantined": policy["quarantined"],
            "personas": policy["personas"],
            "tpm": tpm_attest,
        }
        self.overview_text.setPlainText(json.dumps(overview, indent=2))
        self.safe_label.setText(f"Safe: {policy['safe_mode']}  |  Quarantined: {policy['quarantined']}")

        self.vm_list.clear()
        for name, info in self.womb.vms.items():
            status = "RUN" if info.get("running") else "OFF"
            self.vm_list.addItem(f"{name} [{status}]")

        cur_vm = self._current_vm_name()
        if cur_vm:
            st = self.womb.vtpm_status(cur_vm)
            if not st["exists"]:
                text = "vTPM: VM not found or unsupported"
            else:
                state = "ENABLED" if st["enabled"] else "DISABLED"
                kp = "KeyProtector: yes" if st["has_key_protector"] else "KeyProtector: no"
                errs = "; ".join(st["errors"]) if st["errors"] else "OK"
                text = f"vTPM: {state} | {kp} | {errs}"
            self.vtpm_status_label.setText(text)
        else:
            self.vtpm_status_label.setText("vTPM: n/a")

        self.win_list.clear()
        wins = self.ui_auto.list_windows() if IS_WINDOWS else []
        for w in wins:
            self.win_list.addItem(f"{w['title']} ({w['class']})")

        events = load_recent_events(100)
        logs_text = "\n".join(
            f"{time.strftime('%H:%M:%S', time.localtime(e['ts']))} [{e['level']}] {e['source']}: {e['message']}"
            for e in reversed(events)
        )
        self.logs_view.setPlainText(logs_text)

        timeline_text = "\n".join(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['ts']))} :: {e['source']} :: {e['message']}"
            for e in reversed(events)
        )
        self.timeline_view.setPlainText(timeline_text)

        self.plugins_view.setPlainText("Registered plugins:\n" + "\n".join(self.plugin_mgr.plugins.keys()))

        dist = self.trust_core.to_dict().get("remote", {})
        self.distributed_view.setPlainText("Remote trust nodes:\n" + json.dumps(dist, indent=2))

        drv = self.driver_check.last_status or {}
        self.driver_view.setPlainText("Driver / Capability Status:\n" + json.dumps(drv, indent=2))

        nic = self.nic_enum.last_snapshot or {}
        self.nic_view.setPlainText("NIC / Switch Snapshot:\n" + json.dumps(nic, indent=2))

        hb = self.hb_monitor.status()
        self.hb_view.setPlainText("VM Heartbeats:\n" + json.dumps(hb, indent=2))

# ============================================================
#  Main
# ============================================================

def main():
    trust_core = BorgTrustCore()
    telemetry = TelemetrySpine(trust_core)
    policy = PolicyEngine(trust_core)

    libvirt_conn = None
    if IS_LINUX and HAVE_LIBVIRT:
        try:
            libvirt_conn = libvirt.open("qemu:///system")
        except Exception:
            libvirt_conn = None

    tpm = SyntheticTPM(trust_core, libvirt_conn=libvirt_conn)
    womb = WombController(telemetry, policy)
    sandbox = SandboxManager(telemetry)
    ui_auto = UIAutomationManager(telemetry)
    plugin_mgr = PluginManager(telemetry)
    driver_check = DriverCheckOrgan(telemetry)
    nic_enum = NICEnumerator(telemetry, womb.hv_nic)
    hb_monitor = VMHeartbeatMonitor(telemetry, womb)

    womb.register_vm("VM1")
    womb.register_vm("VM2")

    if "os_root_key" not in tpm.keys:
        tpm.seal_key("os_root_key", "SECRET_DATA", required_trust=0.7)

    fs_observer = None
    try:
        fs_observer = start_fs_monitor(os.getcwd(), telemetry)
    except Exception as e:
        print(f"[FS] Monitor error: {e}")

    agent = GuestAgent(telemetry, sandbox, ui_auto)

    def background_loop():
        while True:
            agent.report_process_snapshot()
            agent.enforce_sandbox_policies()
            womb.tick()
            hb_monitor.tick()
            nic_enum.snapshot()
            driver_check.run_full_check()
            trust_core.push_trust_to_peers()
            time.sleep(5)

    thread = threading.Thread(target=background_loop, daemon=True)
    thread.start()

    app = QtWidgets.QApplication(sys.argv)
    win = CockpitWindow(trust_core, tpm, policy, womb, ui_auto, plugin_mgr, driver_check, nic_enum, hb_monitor)
    win.show()
    app.exec()

    if fs_observer:
        fs_observer.stop()
        fs_observer.join()

if __name__ == "__main__":
    main()

