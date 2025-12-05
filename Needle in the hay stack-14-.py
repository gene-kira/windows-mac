#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardened Defender — Stable Extended Version (Windows Apps Inventory)
"""

import importlib
import logging
import platform
import threading
import time
import json
import os
import shutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional, Dict, Set

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("defender")

# -----------------------------------------------------------------------------
# Optional imports
# -----------------------------------------------------------------------------
def optional_import(name):
    try:
        m = importlib.import_module(name)
        log.info(f"[autoloader] Loaded {name}")
        return m
    except Exception as e:
        log.warning(f"[autoloader] Missing {name}: {e}")
        return None

psutil = optional_import("psutil")
watchdog = optional_import("watchdog.observers")
watchdog_events = optional_import("watchdog.events")
tkinter = optional_import("tkinter")
winreg = None
if platform.system() == "Windows":
    try:
        import winreg as _winreg
        winreg = _winreg
    except Exception as e:
        log.warning(f"[autoloader] winreg unavailable: {e}")

CAPS = {
    "os": platform.system(),
    "process": psutil is not None,
    "fs_watch": (watchdog is not None and watchdog_events is not None),
    "ui": tkinter is not None,
    "winreg": winreg is not None,
}
log.info(f"[autoloader] Capabilities: {CAPS}")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def safe_load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"[json] load failed {path}: {e}")
    return default

def safe_save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning(f"[json] save failed {path}: {e}")

# -----------------------------------------------------------------------------
# MemoryVault
# -----------------------------------------------------------------------------
class MemoryVault:
    def __init__(self, path="memory_vault.json", autosave_interval=20):
        self.path = path
        self.episodes = deque(maxlen=8000)
        self.lock = threading.Lock()
        self.autosave_interval = autosave_interval
        self._load()
        threading.Thread(target=self._autosave, daemon=True).start()

    def _load(self):
        data = safe_load_json(self.path, {"episodes": []})
        for ep in data.get("episodes", []):
            self.episodes.append(ep)
        log.info(f"[memory] Loaded {len(self.episodes)} episodes")

    def save(self):
        with self.lock:
            safe_save_json(self.path, {"episodes": list(self.episodes)})
        log.info(f"[memory] Saved {len(self.episodes)} episodes")

    def _autosave(self):
        while True:
            time.sleep(self.autosave_interval)
            self.save()

    def add(self, src, msg, tags=None, payload=None):
        ep = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": src,
            "message": msg,
            "tags": tags or [],
            "payload": payload,
        }
        with self.lock:
            self.episodes.append(ep)
        return ep

    def recall(self, include_tags=None, since_seconds=None, limit=200):
        include = set(include_tags or [])
        cutoff = None
        if since_seconds:
            cutoff = datetime.utcnow() - timedelta(seconds=since_seconds)
        out = []
        with self.lock:
            for ep in reversed(self.episodes):
                ts = datetime.fromisoformat(ep["ts"].replace("Z", ""))
                if cutoff and ts < cutoff:
                    continue
                if include and not (include & set(ep.get("tags", []))):
                    continue
                out.append(ep)
                if len(out) >= limit:
                    break
        return list(reversed(out))

# -----------------------------------------------------------------------------
# ACE
# -----------------------------------------------------------------------------
class ACE:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.salt = f"ace-{int(time.time())}"

    def toggle(self, on: bool):
        self.enabled = on

    def mask(self, text: str) -> str:
        if not self.enabled or not text:
            return text
        key = ord(self.salt[-1]) & 0xFF
        masked = "".join(chr(ord(c) ^ key) for c in text)
        return f"MASK:{masked.encode('latin1').hex()}"

    def sanitize_event(self, e: dict) -> dict:
        if not self.enabled or not isinstance(e, dict):
            return e
        out = dict(e)
        for k in ("username", "cmdline", "dest", "src_path"):
            if k in out and isinstance(out[k], str):
                out[k] = self.mask(out[k])
        return out

# -----------------------------------------------------------------------------
# Codex + Trust
# -----------------------------------------------------------------------------
SEVERITY = {"lolbin_misuse": 6, "beacon": 4, "fs_persist": 5}

class Codex:
    def __init__(self, retention_hours=24, memory: Optional[MemoryVault] = None):
        self.version = 1
        self.retention = timedelta(hours=retention_hours)
        self.rules = {"lolbins": {"arg_entropy_threshold": 4.0}, "beacon": {"strict_mode": False}}
        self.memory = memory

    def mutate(self, signals: Dict):
        changed = False
        if signals.get("escalate_defense"):
            self.rules["beacon"]["strict_mode"] = True
            changed = True
        if changed:
            self.version += 1
            if self.memory:
                self.memory.add(
                    "codex",
                    "mutate",
                    tags=["codex", "mutation"],
                    payload={"version": self.version, "rules": self.rules},
                )

class TrustModel:
    def __init__(self):
        self.scores = defaultdict(float)
        self.history = defaultdict(lambda: deque(maxlen=200))

    def update(self, eid, signals, ctx):
        delta = sum(SEVERITY.get(s, 1) for s in signals)
        self.scores[eid] = max(0.0, self.scores[eid] * 0.95 + delta)
        self.history[eid].append((time.time(), self.scores[eid]))
        return self.scores[eid]

# -----------------------------------------------------------------------------
# Bridge
# -----------------------------------------------------------------------------
class Bridge:
    class Signal:
        def __init__(self, name, ui=None, memory=None):
            self.name = name
            self.ui = ui
            self.memory = memory
            self.logger = logging.getLogger(f"signal.{name}")

        def emit(self, msg, tags=None, payload=None):
            self.logger.info(msg)
            if self.ui and hasattr(self.ui, "append_line"):
                self.ui.append_line(f"[{self.name}] {msg}")
            if self.memory:
                self.memory.add(f"signal:{self.name}", msg, tags or [self.name], payload)

    def __init__(self, ui=None, memory=None):
        self.ui = ui
        self.memory = memory
        self.ingest = Bridge.Signal("ingest", ui, memory)
        self.threat = Bridge.Signal("threat", ui, memory)
        self.sync = Bridge.Signal("sync", ui, memory)

    def attach_ui(self, ui):
        self.ui = ui
        self.ingest.ui = ui
        self.threat.ui = ui
        self.sync.ui = ui

# -----------------------------------------------------------------------------
# ObservedRegistry
# -----------------------------------------------------------------------------
class ObservedRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.entities = {}

    def upsert(self, eid, etype, meta=None, tags: Optional[Set[str]] = None):
        with self.lock:
            entry = self.entities.get(eid, {"type": etype, "meta": {}, "tags": set()})
            entry["type"] = etype
            if meta:
                entry["meta"].update(meta)
            if tags:
                entry["tags"].update(tags)
            self.entities[eid] = entry

    def list(self):
        with self.lock:
            return dict(self.entities)

# -----------------------------------------------------------------------------
# ListManager (Windows default-allow + WindowsApps inventory)
# -----------------------------------------------------------------------------
DEFAULT_SOFTWARE_MAP = {
    "WindowsCore": {
        "process_names": ["explorer.exe", "svchost.exe", "lsass.exe", "winlogon.exe", "services.exe", "dwm.exe", "taskmgr.exe"],
        "file_paths_contains": ["\\Windows\\System32", "\\Windows\\SysWOW64", "/Windows/System32", "/Windows/SysWOW64"],
        "network_domains_contains": ["microsoft", "windows", "windowsupdate"],
    },
    "WindowsApps": {
        "process_names": [],
        "file_paths_contains": [],
        "network_domains_contains": [],
    },
}

class ListManager:
    def __init__(
        self,
        memory: MemoryVault,
        observed: ObservedRegistry,
        path_allow="allow_list.json",
        path_disallow="disallow_list.json",
        map_path="software_map.json",
        policy_path="policy.json",
        apps_cache_path="windows_apps.json",
    ):
        self.memory = memory
        self.observed = observed
        self.path_allow = path_allow
        self.path_disallow = path_disallow
        self.map_path = map_path
        self.policy_path = policy_path
        self.apps_cache_path = apps_cache_path

        self.allow_list = set(safe_load_json(path_allow, []))
        self.disallow_list = set(safe_load_json(path_disallow, []))
        self.software_map = safe_load_json(map_path, DEFAULT_SOFTWARE_MAP)

        pol = safe_load_json(policy_path, {"windows_default_allow": True})
        self.windows_default_allow = bool(pol.get("windows_default_allow", True))

        # Windows apps enumeration and integration
        cached = safe_load_json(self.apps_cache_path, {"apps": []})
        if cached.get("apps"):
            apps = cached["apps"]
        else:
            apps = self.enumerate_installed_apps()
            safe_save_json(self.apps_cache_path, {"apps": apps})

        # Integrate into WindowsApps group
        wa = self.software_map.setdefault("WindowsApps", {"process_names": [], "file_paths_contains": [], "network_domains_contains": []})
        for app in apps:
            pn = (app.get("exe_name") or app.get("process_name") or "").strip()
            path = (app.get("install_path") or app.get("path") or "").strip()
            if pn and pn.lower() not in [x.lower() for x in wa["process_names"]]:
                wa["process_names"].append(pn)
            if path and path.lower() not in [x.lower() for x in wa["file_paths_contains"]]:
                wa["file_paths_contains"].append(path)

        self.save()
        log.info(
            f"[lists] allow={len(self.allow_list)} disallow={len(self.disallow_list)} "
            f"windows_default_allow={self.windows_default_allow} WindowsApps={len(wa['process_names'])}"
        )

    def enumerate_installed_apps(self):
        apps = []
        if platform.system() != "Windows" or not winreg:
            return apps

        uninstall_paths = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for subpath in uninstall_paths:
                try:
                    key = winreg.OpenKey(hive, subpath)
                    count = winreg.QueryInfoKey(key)[0]
                    for i in range(count):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey = winreg.OpenKey(key, subkey_name)
                            name = ""
                            install_location = ""
                            display_icon = ""
                            try:
                                name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            except Exception:
                                pass
                            try:
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                            except Exception:
                                pass
                            try:
                                display_icon = winreg.QueryValueEx(subkey, "DisplayIcon")[0]
                            except Exception:
                                pass
                            exe_name = os.path.basename(display_icon) if display_icon else ""
                            if name:
                                apps.append(
                                    {
                                        "name": name,
                                        "install_path": install_location,
                                        "display_icon": display_icon,
                                        "exe_name": exe_name or name,
                                    }
                                )
                        except Exception:
                            continue
                except Exception:
                    continue

        win_apps_dir = r"C:\Program Files\WindowsApps"
        if os.path.isdir(win_apps_dir):
            try:
                for entry in os.listdir(win_apps_dir):
                    if entry:
                        apps.append({"name": entry, "install_path": win_apps_dir, "exe_name": entry})
            except Exception:
                pass

        if psutil:
            try:
                for p in psutil.process_iter(attrs=["name", "exe"]):
                    info = p.info
                    name = info.get("name") or ""
                    exe = info.get("exe") or ""
                    if name:
                        apps.append({"name": name, "install_path": os.path.dirname(exe) if exe else "", "exe_name": name})
            except Exception:
                pass

        uniq = []
        seen = set()
        for a in apps:
            key = (a.get("exe_name") or a.get("name") or "").lower() + "|" + (a.get("install_path") or "").lower()
            if key not in seen:
                seen.add(key)
                uniq.append(a)
        return uniq

    def save(self):
        safe_save_json(self.path_allow, sorted(list(self.allow_list)))
        safe_save_json(self.path_disallow, sorted(list(self.disallow_list)))
        safe_save_json(self.map_path, self.software_map)
        safe_save_json(self.policy_path, {"windows_default_allow": self.windows_default_allow})

    def set_windows_default_allow(self, on: bool):
        self.windows_default_allow = bool(on)
        self.memory.add("lists", f"Windows default allow={self.windows_default_allow}", tags=["lists", "policy", "windows"])
        self.save()

    def is_allowed(self, eid, tags: Optional[Set[str]] = None) -> bool:
        if eid in self.allow_list:
            return True
        if eid in self.disallow_list:
            return False
        if self.windows_default_allow and tags and ("WindowsCore" in tags or "WindowsApps" in tags):
            return True
        return False

    def move_to_allow(self, eid):
        self.disallow_list.discard(eid)
        self.allow_list.add(eid)
        self.memory.add("lists", f"Allow {eid}", tags=["lists", "allow", "override"])
        self.save()

    def move_to_disallow(self, eid):
        self.allow_list.discard(eid)
        self.disallow_list.add(eid)
        self.memory.add("lists", f"Disallow {eid}", tags=["lists", "disallow", "override"])
        self.save()

    def auto_disallow(self, eid, reason: str):
        if eid not in self.allow_list and eid not in self.disallow_list:
            self.disallow_list.add(eid)
            self.memory.add("lists", f"Auto-disallow {eid}", tags=["lists", "disallow", "auto"], payload={"reason": reason})
            self.save()

    def upsert_group(self, name, process_names=None, file_paths_contains=None, network_domains_contains=None):
        name = name.strip()
        if not name:
            return False
        self.software_map.setdefault(name, {"process_names": [], "file_paths_contains": [], "network_domains_contains": []})
        self.software_map[name]["process_names"] = [p.strip() for p in (process_names or []) if p.strip()]
        self.software_map[name]["file_paths_contains"] = [p.strip() for p in (file_paths_contains or []) if p.strip()]
        self.software_map[name]["network_domains_contains"] = [p.strip() for p in (network_domains_contains or []) if p.strip()]
        self.memory.add("lists", f"Upsert group {name}", tags=["lists", "group", "edit"], payload=self.software_map[name])
        self.save()
        return True

    def delete_group(self, name):
        name = name.strip()
        if name in self.software_map:
            del self.software_map[name]
            self.memory.add("lists", f"Deleted group {name}", tags=["lists", "group", "delete"])
            self.save()
            return True
        return False

    def recent(self, limit=200):
        return {
            "allow": sorted(list(self.allow_list))[:limit],
            "disallow": sorted(list(self.disallow_list))[:limit],
            "groups": sorted(list(self.software_map.keys())),
            "policy": {"windows_default_allow": self.windows_default_allow},
        }

    def allow_group(self, group_name: str) -> int:
        group = self.software_map.get(group_name)
        if not group:
            return 0
        count = 0
        for eid, e in self.observed.list().items():
            meta = e.get("meta", {})
            name = (meta.get("name") or meta.get("base") or "").lower()
            path = (meta.get("path") or "").lower()
            dest = (meta.get("dest") or "").lower()
            matched = False
            if e["type"] == "proc" and name in [pn.lower() for pn in group.get("process_names", [])]:
                matched = True
            if e["type"] == "file" and any(sub.lower() in path for sub in group.get("file_paths_contains", [])):
                matched = True
            if e["type"] == "net" and any(sub.lower() in dest for sub in group.get("network_domains_contains", [])):
                matched = True
            if matched:
                self.move_to_allow(eid)
                count += 1
        self.memory.add("lists", f"Group allow {group_name} ({count})", tags=["lists", "group", "allow"], payload={"count": count})
        return count

    def disallow_group(self, group_name: str) -> int:
        group = self.software_map.get(group_name)
        if not group:
            return 0
        count = 0
        for eid, e in self.observed.list().items():
            meta = e.get("meta", {})
            name = (meta.get("name") or meta.get("base") or "").lower()
            path = (meta.get("path") or "").lower()
            dest = (meta.get("dest") or "").lower()
            matched = False
            if e["type"] == "proc" and name in [pn.lower() for pn in group.get("process_names", [])]:
                matched = True
            if e["type"] == "file" and any(sub.lower() in path for sub in group.get("file_paths_contains", [])):
                matched = True
            if e["type"] == "net" and any(sub.lower() in dest for sub in group.get("network_domains_contains", [])):
                matched = True
            if matched:
                self.move_to_disallow(eid)
                count += 1
        self.memory.add("lists", f"Group disallow {group_name} ({count})", tags=["lists", "group", "disallow"], payload={"count": count})
        return count

# -----------------------------------------------------------------------------
# Responder
# -----------------------------------------------------------------------------
class Responder:
    def __init__(self, bridge: Bridge, memory: MemoryVault, ace: ACE, lists: ListManager):
        self.bridge = bridge
        self.memory = memory
        self.ace = ace
        self.lists = lists
        self.process_termination = True
        self.file_quarantine = True
        self.egress_throttle = True
        self.host_isolation = False
        self.quarantine_dir = os.path.join(os.getcwd(), "quarantine")
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def kill_process_tree(self, pid, entity_id=None, tags=None):
        if not psutil:
            return False
        if entity_id and self.lists.is_allowed(entity_id, tags=tags):
            self.bridge.ingest.emit(f"[lists] Allowed: skip kill {entity_id}", tags=["lists", "allow", "skip"])
            return False
        try:
            p = psutil.Process(pid)
            for c in p.children(recursive=True):
                try:
                    c.terminate()
                except Exception:
                    pass
            try:
                p.terminate()
            except Exception:
                pass
            self.bridge.threat.emit(f"Terminated process tree PID={pid}", tags=["responder", "process_kill"], payload={"pid": pid, "entity": entity_id})
            self.memory.add("responder", "process_kill", tags=["responder", "process_kill"], payload={"pid": pid, "entity": entity_id, "tags": list(tags or [])})
            return True
        except Exception as e:
            self.bridge.ingest.emit(f"[responder] kill error: {e}", tags=["responder", "error"])
            return False

    def quarantine_file(self, path, entity_id=None, tags=None):
        try:
            if entity_id and self.lists.is_allowed(entity_id, tags=tags):
                self.bridge.ingest.emit(f"[lists] Allowed: skip quarantine {entity_id}", tags=["lists", "allow", "skip"])
                return False
            if not os.path.isfile(path):
                return False
            dest = os.path.join(self.quarantine_dir, f"{int(time.time())}_{os.path.basename(path)}")
            shutil.copy2(path, dest)
            try:
                os.remove(path)
            except Exception:
                pass
            self.bridge.threat.emit(f"Quarantined file {path} -> {dest}", tags=["responder", "file_quarantine"], payload={"src": path, "dest": dest, "entity": entity_id})
            self.memory.add("responder", "file_quarantine", tags=["responder", "file_quarantine"], payload={"src": path, "dest": dest, "entity": entity_id, "tags": list(tags or [])})
            return True
        except Exception as e:
            self.bridge.ingest.emit(f"[responder] quarantine error: {e}", tags=["responder", "error"])
            return False

    def throttle_dest(self, dest_str, entity_id=None, tags=None):
        if entity_id and self.lists.is_allowed(entity_id, tags=tags):
            self.bridge.ingest.emit(f"[lists] Allowed: skip throttle {entity_id}", tags=["lists", "allow", "skip"])
            return False
        self.bridge.threat.emit(f"Egress throttle applied to {dest_str}", tags=["responder", "egress_throttle"], payload={"dest": dest_str, "entity": entity_id})
        self.memory.add("responder", "egress_throttle", tags=["responder", "egress_throttle"], payload={"dest": dest_str, "entity": entity_id, "tags": list(tags or [])})
        return True

    def set_host_isolation(self, on: bool):
        self.host_isolation = on
        state = "ENABLED" if on else "DISABLED"
        self.bridge.threat.emit(f"Host isolation {state} (temporary)", tags=["responder", "host_isolation"], payload={"state": state})
        self.memory.add("responder", "host_isolation", tags=["responder", "host_isolation"], payload={"state": state})

# -----------------------------------------------------------------------------
# Collectors
# -----------------------------------------------------------------------------
class Collectors:
    def __init__(self, bridge: Bridge, trust: TrustModel, ace: ACE, responder: Responder, lists: ListManager, observed: ObservedRegistry, quarantine_dir: str):
        self.bridge = bridge
        self.trust = trust
        self.ace = ace
        self.responder = responder
        self.lists = lists
        self.observed = observed
        self.running = True
        self.quarantine_dir = os.path.abspath(quarantine_dir)

    def start(self):
        threading.Thread(target=self.process_monitor, daemon=True).start()
        threading.Thread(target=self.network_monitor, daemon=True).start()
        threading.Thread(target=self.filesystem_monitor, daemon=True).start()

    def process_monitor(self):
        if not psutil:
            return
        while self.running:
            try:
                for p in psutil.process_iter(attrs=["pid", "ppid", "name", "cmdline", "exe"]):
                    info = p.info
                    pid = info.get("pid")
                    name = (info.get("name") or "")
                    cmd = " ".join(info.get("cmdline", [])).lower() if info.get("cmdline") else ""
                    exe = (info.get("exe") or "")
                    entity = f"proc:{pid}:{name}"
                    tags = set()
                    for g, spec in self.lists.software_map.items():
                        if name and name.lower() in [pn.lower() for pn in spec.get("process_names", [])]:
                            tags.add(g)
                        if exe and any(sub.lower() in exe.lower() for sub in spec.get("file_paths_contains", [])):
                            tags.add(g)
                    self.observed.upsert(entity, "proc", meta={"pid": pid, "name": name, "path": exe, "cmd": cmd}, tags=tags)
                    if any(x in cmd for x in ["rundll32", "powershell", "mshta", "certutil", "wmic", "bash", "sh", "osascript"]) and cmd:
                        self.lists.auto_disallow(entity, reason="lolbin_misuse")
                        self.trust.update(entity, ["lolbin_misuse"], {"maintenance_window": False})
                        self.bridge.threat.emit(f"LOLBin/Script pid={pid} name={name} cmd={self.ace.mask(cmd)}", tags=["collector", "process", "lolbin"], payload=info)
                        if self.responder.process_termination and not self.lists.is_allowed(entity, tags=tags):
                            self.responder.kill_process_tree(pid, entity_id=entity, tags=tags)
                time.sleep(5)
            except Exception as e:
                self.bridge.ingest.emit(f"[process monitor] error: {e}", tags=["collector", "error"])
                time.sleep(2)

    def network_monitor(self):
        if not psutil:
            return
        while self.running:
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if conn.status == "ESTABLISHED" and conn.raddr:
                        try:
                            dest_str = f"{conn.raddr.ip}:{conn.raddr.port}"
                        except Exception:
                            dest_str = str(conn.raddr)
                        entity = f"net:{dest_str}"
                        tags = set()
                        for g, spec in self.lists.software_map.items():
                            if any(sub.lower() in dest_str.lower() for sub in spec.get("network_domains_contains", [])):
                                tags.add(g)
                        self.observed.upsert(entity, "net", meta={"dest": dest_str, "status": conn.status}, tags=tags)
                        self.lists.auto_disallow(entity, reason="beacon")
                        self.bridge.sync.emit(f"Outbound: {self.ace.mask(dest_str)}", tags=["collector", "network", "outbound"], payload={"dest": dest_str})
                        self.trust.update(entity, ["beacon"], {"maintenance_window": False})
                        if self.responder.egress_throttle and not self.lists.is_allowed(entity, tags=tags):
                            self.responder.throttle_dest(dest_str, entity_id=entity, tags=tags)
                time.sleep(10)
            except Exception as e:
                self.bridge.ingest.emit(f"[network monitor] error: {e}", tags=["collector", "error"])
                time.sleep(2)

    def filesystem_monitor(self):
        if not (watchdog and watchdog_events):
            return
        class Handler(watchdog_events.FileSystemEventHandler):
            def on_modified(_, event):
                try:
                    src_abs = os.path.abspath(event.src_path)
                    if os.path.isdir(src_abs):
                        return
                    if os.path.commonpath([src_abs, self.quarantine_dir]) == self.quarantine_dir:
                        return
                    entity = f"file:{event.src_path}"
                    base = os.path.basename(event.src_path)
                    tags = set()
                    for g, spec in self.lists.software_map.items():
                        if any(sub.lower() in src_abs.lower() for sub in spec.get("file_paths_contains", [])):
                            tags.add(g)
                        if base.lower() in [pn.lower() for pn in spec.get("process_names", [])]:
                            tags.add(g)
                    self.observed.upsert(entity, "file", meta={"path": event.src_path, "base": base}, tags=tags)
                    self.bridge.ingest.emit(f"File modified: {self.ace.mask(event.src_path)}", tags=["collector", "filesystem", "modified"], payload={"src": event.src_path})
                except Exception as ex:
                    self.bridge.ingest.emit(f"[filesystem handler] error: {ex}", tags=["collector", "error"])
        try:
            observer = watchdog.Observer()
            root = "/" if CAPS["os"] != "Windows" else "C:\\"
            observer.schedule(Handler(), path=root, recursive=True)
            observer.start()
            while self.running:
                time.sleep(1)
        except Exception as e:
            self.bridge.ingest.emit(f"[filesystem monitor] error: {e}", tags=["collector", "error"])

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
if CAPS["ui"]:
    import tkinter as tk
    from tkinter import ttk

class DefenderUI:
    def __init__(self, orch):
        self.orch = orch
        self.root = tk.Tk()
        self.root.title("Hardened Defender — Oversight")
        self.root.geometry("1200x800")
        self.root.configure(bg="#111111")
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        # Console
        self.console_tab = tk.Frame(self.tabs, bg="#111111")
        self.tabs.add(self.console_tab, text="Console")
        self.text = tk.Text(self.console_tab, bg="#111111", fg="#E6E6E6", insertbackground="#E6E6E6", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True)
        panel = tk.Frame(self.console_tab, bg="#1a1a1a")
        panel.pack(fill="x")
        tk.Button(panel, text="Switch Mode", command=self.orch.switch_bot_mode, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)
        tk.Button(panel, text="Escalate Defense", command=self.orch.escalate_defense, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        self.iso_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="Host Isolation", variable=self.iso_var, command=lambda: self.orch.responder.set_host_isolation(self.iso_var.get()), bg="#1a1a1a", fg="white", selectcolor="#1a1a1a").pack(side="left", padx=6)
        self.win_var = tk.BooleanVar(value=self.orch.lists.windows_default_allow)
        tk.Checkbutton(panel, text="Trust Windows system components and apps by default", variable=self.win_var, command=lambda: self._toggle_windows(), bg="#1a1a1a", fg="white", selectcolor="#1a1a1a").pack(side="left", padx=6)
        tk.Button(panel, text="Save", command=self._save_all, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(panel, text="Quit", command=self.root.quit, bg="#aa0000", fg="white").pack(side="right", padx=6)

        # Codex
        self.codex_tab = tk.Frame(self.tabs, bg="#111111")
        self.tabs.add(self.codex_tab, text="Codex")
        self.codex_info = tk.Text(self.codex_tab, bg="#111111", fg="#E6E6E6")
        self.codex_info.pack(fill="both", expand=True)

        # Memory
        self.memory_tab = tk.Frame(self.tabs, bg="#111111")
        self.tabs.add(self.memory_tab, text="Memory")
        self.mem_out = tk.Text(self.memory_tab, bg="#111111", fg="#E6E6E6")
        self.mem_out.pack(fill="both", expand=True)

        # Lists + Apps
        self.lists_tab = tk.Frame(self.tabs, bg="#111111")
        self.tabs.add(self.lists_tab, text="Lists + Groups + Apps")
        top = tk.Frame(self.lists_tab, bg="#1a1a1a")
        top.pack(fill="x")
        tk.Label(top, text="Search:", bg="#1a1a1a", fg="white").pack(side="left", padx=6)
        self.search_entry = tk.Entry(top, bg="#222", fg="#E6E6E6", width=30)
        self.search_entry.pack(side="left")
        tk.Button(top, text="Refresh", command=self.refresh_all, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Label(top, text="Group:", bg="#1a1a1a", fg="white").pack(side="left", padx=12)
        self.group_var = tk.StringVar(value="")
        self.group_menu = ttk.Combobox(top, textvariable=self.group_var, values=self.orch.lists.recent()["groups"], width=22)
        self.group_menu.pack(side="left")
        tk.Button(top, text="Allow Group", command=self.group_allow, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(top, text="Disallow Group", command=self.group_disallow, bg="#2e2e2e", fg="white").pack(side="left", padx=6)

        body = tk.Frame(self.lists_tab, bg="#111111")
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#111111")
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(left, text="Observed (show all)", bg="#111111", fg="#00d4aa").pack(anchor="w")
        self.observed_list = tk.Listbox(left, bg="#0f0f0f", fg="#E6E6E6", selectmode="extended")
        self.observed_list.pack(fill="both", expand=True)
        tk.Button(left, text="Allow selected", command=self.allow_selected, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)
        tk.Button(left, text="Disallow selected", command=self.disallow_selected, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)

        middle = tk.Frame(body, bg="#111111")
        middle.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(middle, text="Allow List", bg="#111111", fg="#66ff99").pack(anchor="w")
        self.allow_listbox = tk.Listbox(middle, bg="#0f0f0f", fg="#E6E6E6")
        self.allow_listbox.pack(fill="both", expand=True)

        right = tk.Frame(body, bg="#111111")
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(right, text="Disallow List", bg="#111111", fg="#ff6699").pack(anchor="w")
        self.disallow_listbox = tk.Listbox(right, bg="#0f0f0f", fg="#E6E6E6")
        self.disallow_listbox.pack(fill="both", expand=True)

        apps_frame = tk.LabelFrame(self.lists_tab, text="Windows Apps (allowed by default)", bg="#111111", fg="#E6E6E6")
        apps_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.apps_list = tk.Listbox(apps_frame, bg="#0f0f0f", fg="#E6E6E6", selectmode="extended")
        self.apps_list.pack(fill="both", expand=True)
        btns = tk.Frame(apps_frame, bg="#111111")
        btns.pack(fill="x")
        tk.Button(btns, text="Allow selected apps", command=self.allow_selected_apps, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)
        tk.Button(btns, text="Disallow selected apps", command=self.disallow_selected_apps, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)

        self.orch.bridge.attach_ui(self)
        threading.Thread(target=self._status_loop, daemon=True).start()
        threading.Thread(target=self._update_codex_loop, daemon=True).start()
        self.refresh_all()
        self.root.mainloop()

    def append_line(self, msg):
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def _toggle_windows(self):
        self.orch.lists.set_windows_default_allow(self.win_var.get())
        self.append_line(f"[policy] Windows default allow: {self.win_var.get()}")

    def _save_all(self):
        self.orch.memory.save()
        self.orch.lists.save()
        self.append_line("[persist] Saved memory and lists")

    def _status_loop(self):
        while True:
            try:
                status = f"Allow={len(self.orch.lists.allow_list)} Disallow={len(self.orch.lists.disallow_list)} Observed={len(self.orch.observed.list())}"
                self.append_line("[status] " + status)
                time.sleep(8)
            except Exception:
                time.sleep(8)

    def _update_codex_loop(self):
        while True:
            try:
                mf = {"version": self.orch.codex.version, "rules": self.orch.codex.rules}
                self.codex_info.delete("1.0", "end")
                self.codex_info.insert("end", f"Codex Manifest:\n{json.dumps(mf, indent=2)}\n")
                time.sleep(7)
            except Exception:
                time.sleep(7)

    def _listbox_items_to_entity_ids(self, lb, indices):
        ids = []
        for i in indices:
            text = lb.get(i)
            eid = text.split(" ", 1)[0].strip()
            ids.append(eid)
        return ids

    def allow_selected(self):
        sel = list(self.observed_list.curselection())
        ids = self._listbox_items_to_entity_ids(self.observed_list, sel)
        for eid in ids:
            self.orch.lists.move_to_allow(eid)
            self.append_line(f"[lists] Allowed: {eid}")
        self.refresh_all()

    def disallow_selected(self):
        sel = list(self.observed_list.curselection())
        ids = self._listbox_items_to_entity_ids(self.observed_list, sel)
        for eid in ids:
            self.orch.lists.move_to_disallow(eid)
            self.append_line(f"[lists] Disallowed: {eid}")
        self.refresh_all()

    def allow_selected_apps(self):
        sel = list(self.apps_list.curselection())
        items = [self.apps_list.get(i) for i in sel]
        for text in items:
            name = text.split(" | ", 1)[0]
            for eid, e in self.orch.observed.list().items():
                meta = e.get("meta", {})
                meta_name = (meta.get("name") or "").lower()
                if meta_name == name.lower():
                    self.orch.lists.move_to_allow(eid)
                    self.append_line(f"[apps] Allowed: {eid}")
        self.refresh_all()

    def disallow_selected_apps(self):
        sel = list(self.apps_list.curselection())
        items = [self.apps_list.get(i) for i in sel]
        for text in items:
            name = text.split(" | ", 1)[0]
            for eid, e in self.orch.observed.list().items():
                meta = e.get("meta", {})
                meta_name = (meta.get("name") or "").lower()
                if meta_name == name.lower():
                    self.orch.lists.move_to_disallow(eid)
                    self.append_line(f"[apps] Disallowed: {eid}")
        self.refresh_all()

    def group_allow(self):
        g = self.group_var.get().strip()
        if not g:
            return
        c = self.orch.lists.allow_group(g)
        self.append_line(f"[lists] Group allow {g} ({c})")
        self.refresh_all()

    def group_disallow(self):
        g = self.group_var.get().strip()
        if not g:
            return
        c = self.orch.lists.disallow_group(g)
        self.append_line(f"[lists] Group disallow {g} ({c})")
        self.refresh_all()

    def refresh_all(self):
        query = self.search_entry.get().strip().lower() if hasattr(self, "search_entry") else ""
        # Observed
        self.observed_list.delete(0, "end")
        for eid, e in sorted(self.orch.observed.list().items()):
            label = f"{eid} [{e['type']}] {','.join(sorted(e.get('tags', [])))}"
            if not query or query in label.lower():
                self.observed_list.insert("end", label)
        # Lists
        self.allow_listbox.delete(0, "end")
        self.disallow_listbox.delete(0, "end")
        for a in sorted(list(self.orch.lists.allow_list)):
            self.allow_listbox.insert("end", a)
        for d in sorted(list(self.orch.lists.disallow_list)):
            self.disallow_listbox.insert("end", d)
        # Groups
        try:
            self.group_menu["values"] = self.orch.lists.recent()["groups"]
        except Exception:
            pass
        # WindowsApps inventory
        self.apps_list.delete(0, "end")
        winapps = self.orch.lists.software_map.get("WindowsApps", {})
        pn = winapps.get("process_names", [])
        paths = winapps.get("file_paths_contains", [])
        names_sorted = sorted(set(pn))
        for name in names_sorted:
            # Pick first matching path containing the name as a hint, or empty
            path = ""
            for p in paths:
                if name.lower() in p.lower():
                    path = p
                    break
            self.apps_list.insert("end", f"{name} | {path}")

# -----------------------------------------------------------------------------
# Bot
# -----------------------------------------------------------------------------
class Bot:
    def __init__(self, cb, codex: Codex):
        self.cb = cb
        self.codex = codex
        self.run = True
        self.mode = "guardian"

    def switch(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb("Mode " + self.mode.upper())

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                if self.mode == "guardian":
                    self.codex.mutate({"escalate_defense": False})
                    self.cb("Guardian pulse")
                else:
                    self.cb("Rogue calibration (defensive-only)")
                time.sleep(12)
            except Exception as e:
                self.cb(f"[bot] error: {e}")
                time.sleep(3)

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, use_ui=True):
        self.memory = MemoryVault()
        self.bridge = Bridge(None, memory=self.memory)
        self.codex = Codex(retention_hours=24, memory=self.memory)
        self.trust = TrustModel()
        self.ace = ACE(enabled=True)
        self.observed = ObservedRegistry()
        self.lists = ListManager(memory=self.memory, observed=self.observed)
        self.responder = Responder(self.bridge, self.memory, self.ace, self.lists)
        self.collectors = Collectors(self.bridge, self.trust, self.ace, self.responder, self.lists, self.observed, quarantine_dir=self.responder.quarantine_dir)
        self.bot = Bot(lambda m: self.bridge.ingest.emit(m, tags=["bot"]), self.codex)
        self.use_ui = use_ui and CAPS["ui"]

    def start(self):
        self.bot.start()
        self.collectors.start()
        self.bridge.ingest.emit(
            f"Defender online | OS={CAPS['os']} | UI={self.use_ui} | Windows default allow={self.lists.windows_default_allow}",
            tags=["system", "startup"],
        )
        if self.use_ui:
            DefenderUI(self)
        else:
            log.info("CLI mode active")

    def switch_bot_mode(self):
        self.bot.switch()

    def escalate_defense(self):
        self.codex.mutate({"escalate_defense": True})
        self.bridge.ingest.emit("Defense escalated", tags=["codex", "mutate"])

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
def main():
    orch = Orchestrator(use_ui=True)
    orch.start()

if __name__ == "__main__":
    main()

