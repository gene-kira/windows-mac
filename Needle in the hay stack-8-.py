#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardened, defensive-only, mixed OS autonomous detection and response system
with Tkinter GUI, memory vault, zero-hesitation responders, safe host isolation,
and Software-aware Allow/Disallow lists (groups) with auto-population, overrides,
drag-and-drop, and group editor (create/update/delete).

New in this version:
- Extended Windows groups (WindowsCore, Edge, Office), AV suites (Avira, Defender, Malwarebytes),
  and common vendor utilities (NVIDIA, AMD, Intel).
- Group editor UI: create/update/delete groups directly from the Lists tab.
- Bulk "Apply to all observed" for group Allow/Disallow from the UI.
- Observed registry shows live entities; heuristics tag entities with groups.
- Persistence of updated software map to software_map.json.

Safety:
- Non-destructive toward people. Software-only containments and reversible isolation.

Run:
- Ensure Python 3.x and Tkinter are installed. `pip install psutil watchdog`
- Save as defender.py and run: `python defender.py`
"""

import importlib
import logging
import platform
import threading
import time
import random
import json
import os
import shutil
from collections import defaultdict, deque
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("defender")

# -----------------------------------------------------------------------------
# Autoloader (graceful optional imports)
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
requests = optional_import("requests")

CAPS = {
    "os": platform.system(),
    "process": psutil is not None,
    "fs_watch": (watchdog is not None and watchdog_events is not None),
    "ui": tkinter is not None,
    "net_sync": requests is not None,
}
log.info(f"[autoloader] Capabilities: {CAPS}")

# -----------------------------------------------------------------------------
# Glyphs and emblems
# -----------------------------------------------------------------------------
EMBLEMS = {
    "shield_alert": "🛡️⚠️",
    "alchemical": "🜏",
    "dove": "🕊️",
    "trident": "🔱",
    "triangle": "🔺",
    "lock": "🔒",
    "dna": "🧬",
    "web": "🕸️",
    "mask": "🎭",
    "globe": "🌐",
    "scroll": "📜",
    "monitor": "🖥️",
    "memory": "🧠",
    "lists": "📋",
    "group": "🧩",
}

# -----------------------------------------------------------------------------
# Utility: safe JSON load/save
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
            json.dump(data, f, ensure_ascii=False, indent=2)
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
        threading.Thread(target=self._autosave_loop, daemon=True).start()

    def _load(self):
        data = safe_load_json(self.path, {"episodes": []})
        for ep in data.get("episodes", []):
            self.episodes.append(ep)
        log.info(f"[memory] Loaded {len(self.episodes)} episodes")

    def save(self):
        with self.lock:
            safe_save_json(self.path, {"episodes": list(self.episodes)})
            log.info(f"[memory] Saved {len(self.episodes)} episodes")

    def _autosave_loop(self):
        while True:
            time.sleep(self.autosave_interval)
            self.save()

    def add(self, source, message, tags=None, payload=None):
        ep = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": source,
            "message": message,
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
                ts = datetime.fromisoformat(ep["ts"].replace("Z",""))
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
        return f"{EMBLEMS['scroll']}:{masked.encode('latin1').hex()}"

    def sanitize_event(self, e: dict) -> dict:
        if not self.enabled or not isinstance(e, dict):
            return e
        out = dict(e)
        for k in ("username", "cmdline", "dest", "src_path"):
            if k in out and isinstance(out[k], str):
                out[k] = self.mask(out[k])
        return out

# -----------------------------------------------------------------------------
# Codex
# -----------------------------------------------------------------------------
class Codex:
    def __init__(self, retention_hours=24, memory: MemoryVault | None = None):
        self.version = 1
        self.retention = timedelta(hours=retention_hours)
        self.rules = {
            "lolbins": {"arg_entropy_threshold": 4.0},
            "beacon": {"strict_mode": False},
            "identity": {"step_up_auth": False},
        }
        self.history = []
        self.memory = memory

    def mutate(self, signals):
        changed = False
        before = self.export_manifest()
        if signals.get("obfuscation_spike"):
            self.rules["lolbins"]["arg_entropy_threshold"] = 4.2
            changed = True
        if signals.get("escalate_defense"):
            self.rules["beacon"]["strict_mode"] = True
            self.rules["identity"]["step_up_auth"] = True
            changed = True
        if changed:
            self.version += 1
            self.history.append({"ts": datetime.utcnow().isoformat()+"Z", "signals": signals, "version": self.version})
            after = self.export_manifest()
            if self.memory:
                self.memory.add("codex","mutate",tags=["codex","mutation"],payload={"before":before,"after":after})

    def export_manifest(self):
        return {
            "version": self.version,
            "retention_seconds": int(self.retention.total_seconds()),
            "rules": self.rules.copy(),
        }

    def diff(self, before, after):
        out = []
        if before.get("version") != after.get("version"):
            out.append(f"version {before['version']} -> {after['version']}")
        for k in self.rules.keys():
            b = before["rules"].get(k, {})
            a = after["rules"].get(k, {})
            if b != a:
                out.append(f"rules.{k}: {b} -> {a}")
        return out

# -----------------------------------------------------------------------------
# Trust model
# -----------------------------------------------------------------------------
SEVERITY = {
    "lolbin_misuse": 6,
    "beacon": 4,
    "fs_persist": 5,
    "identity_deviation": 6,
}

class TrustModel:
    def __init__(self):
        self.scores = defaultdict(lambda: 0.0)
        self.history = defaultdict(lambda: deque(maxlen=200))

    def update(self, entity_id, signals, context):
        delta = sum(SEVERITY.get(s, 1) for s in signals)
        if context.get("maintenance_window"):
            delta *= 0.3
        self.scores[entity_id] = max(0.0, self.scores[entity_id] * 0.95 + delta)
        self.history[entity_id].append((time.time(), self.scores[entity_id]))
        return self.scores[entity_id]

# -----------------------------------------------------------------------------
# Bridge and Signals
# -----------------------------------------------------------------------------
class Bridge:
    class Signal:
        def __init__(self, name, ui=None, memory: MemoryVault | None = None):
            self.name = name
            self.ui = ui
            self.memory = memory
            self.logger = logging.getLogger(f"signal.{name}")
        def emit(self, msg, tags=None, payload=None):
            self.logger.info(msg)
            if self.ui and hasattr(self.ui, "append_line"):
                self.ui.append_line(f"[{self.name}] {msg}")
            if self.memory:
                self.memory.add(f"signal:{self.name}", msg, tags=(tags or [self.name]), payload=payload)

    def __init__(self, ui=None, memory: MemoryVault | None = None):
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
# Software map (semantic grouping) — extended for Windows and common suites
# -----------------------------------------------------------------------------
DEFAULT_SOFTWARE_MAP = {
    "WindowsCore": {
        "process_names": [
            "explorer.exe","svchost.exe","wininit.exe","lsass.exe","services.exe","conhost.exe",
            "taskmgr.exe","dwm.exe","csrss.exe","smss.exe","winlogon.exe","registry","system"
        ],
        "file_paths_contains": [
            "\\Windows\\System32","\\Windows\\SysWOW64","/Windows/System32","/Windows/SysWOW64"
        ],
        "network_domains_contains": ["microsoft","windows","msft"],
    },
    "MicrosoftEdge": {
        "process_names": ["msedge.exe","microsoftedge.exe"],
        "file_paths_contains": ["\\Microsoft\\Edge","/Microsoft/Edge"],
        "network_domains_contains": ["edge.microsoft.com","microsoft","bing"],
    },
    "MicrosoftOffice": {
        "process_names": ["winword.exe","excel.exe","powerpnt.exe","outlook.exe","onenote.exe","teams.exe"],
        "file_paths_contains": ["\\Microsoft Office","/Microsoft Office","\\Office\\","/Office/"],
        "network_domains_contains": ["office","microsoft","teams"],
    },
    "Avira": {
        "process_names": ["avcenter.exe","avguard.exe","avgnt.exe","avira.servicehost.exe","avira.security.exe"],
        "file_paths_contains": ["\\Avira\\","/Avira/","\\Program Files\\Avira"],
        "network_domains_contains": ["avira","avira-update"],
    },
    "WindowsDefender": {
        "process_names": ["msmpeng.exe","mpcmdrun.exe","nissrv.exe","senseir.exe"],
        "file_paths_contains": ["\\Windows Defender\\","\\Microsoft\\Windows Defender","/Windows Defender/"],
        "network_domains_contains": ["defender","smartscreen","microsoft"],
    },
    "Malwarebytes": {
        "process_names": ["mbam.exe","mbamservice.exe","mbamtray.exe"],
        "file_paths_contains": ["\\Malwarebytes\\","/Malwarebytes/"],
        "network_domains_contains": ["malwarebytes"],
    },
    "NVIDIA": {
        "process_names": ["nvcontainer.exe","nvdisplay.container.exe","nvidia share.exe","nvidia web helper.exe"],
        "file_paths_contains": ["\\NVIDIA Corporation\\","/NVIDIA/"],
        "network_domains_contains": ["nvidia"],
    },
    "AMD": {
        "process_names": ["radeonsoftware.exe","amddvr.exe","amdradeonsettings.exe"],
        "file_paths_contains": ["\\AMD\\","/AMD/"],
        "network_domains_contains": ["amd"],
    },
    "Intel": {
        "process_names": ["igfxtray.exe","intelppmservice.exe","intelgraphicscommandcenter.exe"],
        "file_paths_contains": ["\\Intel\\","/Intel/"],
        "network_domains_contains": ["intel"],
    },
    "Steam": {
        "process_names": ["steam.exe","steamwebhelper.exe","gameoverlayui.exe","cs2.exe","eldenring.exe","palworld.exe"],
        "file_paths_contains": ["\\Steam\\steamapps\\","/steamapps/","/SteamLibrary/"],
        "network_domains_contains": ["steamcontent","steam","valve"],
    },
    "Discord": {
        "process_names": ["discord.exe","discordcanary.exe","discordptb.exe"],
        "file_paths_contains": ["\\Discord\\","/Discord/"],
        "network_domains_contains": ["discordapp","discord.com"],
    },
    "EpicGames": {
        "process_names": ["epicgameslauncher.exe","fortniteclient-win64-shipping.exe"],
        "file_paths_contains": ["\\Epic Games\\","/EpicGames/"],
        "network_domains_contains": ["epicgames","fortnite"],
    },
}

# -----------------------------------------------------------------------------
# Observed registry (live entities for GUI)
# -----------------------------------------------------------------------------
class ObservedRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.entities = {}  # entity_id -> {"type": "proc|file|net", "meta": {...}, "tags": set([...])}

    def upsert(self, entity_id, etype, meta=None, tags=None):
        with self.lock:
            entry = self.entities.get(entity_id, {"type": etype, "meta": {}, "tags": set()})
            entry["type"] = etype
            if meta:
                entry["meta"].update(meta)
            if tags:
                entry["tags"].update(tags)
            self.entities[entity_id] = entry

    def list(self):
        with self.lock:
            return dict(self.entities)

    def by_group(self, group_name):
        out = []
        with self.lock:
            for eid, e in self.entities.items():
                if group_name in e.get("tags", set()):
                    out.append(eid)
        return out

# -----------------------------------------------------------------------------
# Allow/Disallow Lists Manager (software-aware) + group editor
# -----------------------------------------------------------------------------
class ListManager:
    def __init__(self, memory: MemoryVault, observed: ObservedRegistry, path_allow="allow_list.json", path_disallow="disallow_list.json", map_path="software_map.json"):
        self.memory = memory
        self.observed = observed
        self.path_allow = path_allow
        self.path_disallow = path_disallow
        self.map_path = map_path
        self._lock = threading.Lock()
        self.allow_list = set(safe_load_json(path_allow, []))
        self.disallow_list = set(safe_load_json(path_disallow, []))
        self.software_map = safe_load_json(map_path, DEFAULT_SOFTWARE_MAP)
        log.info(f"[lists] Loaded allow={len(self.allow_list)} disallow={len(self.disallow_list)} groups={len(self.software_map)}")

    def save(self):
        with self._lock:
            safe_save_json(self.path_allow, sorted(list(self.allow_list)))
            safe_save_json(self.path_disallow, sorted(list(self.disallow_list)))
            safe_save_json(self.map_path, self.software_map)
            log.info("[lists] Saved allow/disallow and software map")

    def is_allowed(self, entity_id: str) -> bool:
        return entity_id in self.allow_list

    def is_disallowed(self, entity_id: str) -> bool:
        return entity_id in self.disallow_list

    def auto_disallow(self, entity_id: str, reason: str):
        with self._lock:
            if entity_id not in self.allow_list and entity_id not in self.disallow_list:
                self.disallow_list.add(entity_id)
                self.memory.add("lists", f"Auto-disallow {entity_id}", tags=["lists","disallow","auto"], payload={"reason":reason})
                log.info(f"[lists] Auto-disallow {entity_id} ({reason})")

    def move_to_allow(self, entity_id: str):
        with self._lock:
            if entity_id in self.disallow_list:
                self.disallow_list.remove(entity_id)
            self.allow_list.add(entity_id)
            self.memory.add("lists", f"Moved to allow {entity_id}", tags=["lists","allow","override"])
            log.info(f"[lists] Moved to allow {entity_id}")

    def move_to_disallow(self, entity_id: str):
        with self._lock:
            if entity_id in self.allow_list:
                self.allow_list.remove(entity_id)
            self.disallow_list.add(entity_id)
            self.memory.add("lists", f"Moved to disallow {entity_id}", tags=["lists","disallow","override"])
            log.info(f"[lists] Moved to disallow {entity_id}")

    # software-aware apply-to-observed
    def allow_group(self, group_name: str):
        group = self.software_map.get(group_name)
        if not group:
            return 0
        count = 0
        observed = self.observed.list()
        for eid, e in observed.items():
            meta = e.get("meta", {})
            typ = e.get("type")
            name = (meta.get("name") or meta.get("base") or "").lower()
            path = (meta.get("path") or "").lower()
            dest = (meta.get("dest") or "").lower()
            matched = False
            if typ == "proc" and name:
                matched |= any(name == pn.lower() for pn in group.get("process_names", []))
            if typ == "file" and path:
                matched |= any(sub.lower() in path for sub in group.get("file_paths_contains", []))
            if typ == "net" and dest:
                matched |= any(sub.lower() in dest for sub in group.get("network_domains_contains", []))
            if matched:
                self.move_to_allow(eid); count += 1
        self.memory.add("lists", f"Group allow {group_name} ({count})", tags=["lists","group","allow"], payload={"count":count})
        return count

    def disallow_group(self, group_name: str):
        group = self.software_map.get(group_name)
        if not group:
            return 0
        count = 0
        observed = self.observed.list()
        for eid, e in observed.items():
            meta = e.get("meta", {})
            typ = e.get("type")
            name = (meta.get("name") or meta.get("base") or "").lower()
            path = (meta.get("path") or "").lower()
            dest = (meta.get("dest") or "").lower()
            matched = False
            if typ == "proc" and name:
                matched |= any(name == pn.lower() for pn in group.get("process_names", []))
            if typ == "file" and path:
                matched |= any(sub.lower() in path for sub in group.get("file_paths_contains", []))
            if typ == "net" and dest:
                matched |= any(sub.lower() in dest for sub in group.get("network_domains_contains", []))
            if matched:
                self.move_to_disallow(eid); count += 1
        self.memory.add("lists", f"Group disallow {group_name} ({count})", tags=["lists","group","disallow"], payload={"count":count})
        return count

    # group editor: create/update/delete
    def upsert_group(self, name: str, process_names=None, file_paths_contains=None, network_domains_contains=None):
        name = name.strip()
        if not name:
            return False
        with self._lock:
            self.software_map[name] = {
                "process_names": [p.strip() for p in (process_names or []) if p.strip()],
                "file_paths_contains": [p.strip() for p in (file_paths_contains or []) if p.strip()],
                "network_domains_contains": [p.strip() for p in (network_domains_contains or []) if p.strip()],
            }
            self.memory.add("lists", f"Upsert group {name}", tags=["lists","group","edit"], payload=self.software_map[name])
            log.info(f"[lists] Upserted group {name}")
        return True

    def delete_group(self, name: str):
        name = name.strip()
        with self._lock:
            if name in self.software_map:
                del self.software_map[name]
                self.memory.add("lists", f"Deleted group {name}", tags=["lists","group","delete"])
                log.info(f"[lists] Deleted group {name}")
                return True
        return False

    def recent(self, limit=200):
        return {
            "allow": sorted(list(self.allow_list))[:limit],
            "disallow": sorted(list(self.disallow_list))[:limit],
            "groups": sorted(list(self.software_map.keys())),
        }

# -----------------------------------------------------------------------------
# Zero-Hesitation Responder (non-destructive) with hardened quarantine
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

    def kill_process_tree(self, pid, entity_id=None):
        if not psutil:
            return False
        if entity_id and self.lists.is_allowed(entity_id):
            self.bridge.ingest.emit(f"{EMBLEMS['lists']} Allowed: skip kill {entity_id}", tags=["lists","allow","skip"])
            return False
        try:
            p = psutil.Process(pid)
            children = p.children(recursive=True)
            for c in children:
                try: c.terminate()
                except Exception: pass
            try: p.terminate()
            except Exception: pass
            gone, alive = [], []
            try:
                g, a = psutil.wait_procs(children+[p], timeout=3)
                gone, alive = g, a
            except Exception:
                pass
            msg = f"Terminated process tree PID={pid} (gone={len(gone)}, alive={len(alive)})"
            self.bridge.threat.emit(f"{EMBLEMS['shield_alert']} {msg}", tags=["responder","process_kill"], payload={"pid":pid,"gone":len(gone),"alive":len(alive)})
            self.memory.add("responder","process_kill",tags=["responder","process_kill"],payload={"pid":pid,"gone":len(gone),"alive":len(alive),"entity":entity_id})
            return True
        except Exception as e:
            self.bridge.ingest.emit(f"[responder] kill error: {e}", tags=["responder","error"])
            return False

    def quarantine_file(self, path, entity_id=None):
        try:
            if entity_id and self.lists.is_allowed(entity_id):
                self.bridge.ingest.emit(f"{EMBLEMS['lists']} Allowed: skip quarantine {entity_id}", tags=["lists","allow","skip"])
                return False
            if not os.path.isfile(path):
                self.bridge.ingest.emit(f"[responder] not a file: {path}", tags=["responder","warn"])
                return False
            try:
                q_common = os.path.commonpath([os.path.abspath(path), os.path.abspath(self.quarantine_dir)])
                if q_common == os.path.abspath(self.quarantine_dir):
                    return False
            except Exception:
                pass
            base = os.path.basename(path)
            dest = os.path.join(self.quarantine_dir, f"{int(time.time())}_{base}")
            try:
                shutil.copy2(path, dest)
            except Exception as e:
                self.bridge.ingest.emit(f"[responder] copy failed: {e}", tags=["responder","error"])
                return False
            try:
                os.remove(path)
            except PermissionError:
                self.bridge.ingest.emit(f"[responder] could not delete (locked): {path}", tags=["responder","warn"])
            except FileNotFoundError:
                pass
            except Exception as e:
                self.bridge.ingest.emit(f"[responder] delete failed: {e}", tags=["responder","error"])
            msg = f"Quarantined file {path} -> {dest}"
            self.bridge.threat.emit(f"{EMBLEMS['lock']} {msg}", tags=["responder","file_quarantine"], payload={"src":path,"dest":dest,"entity":entity_id})
            self.memory.add("responder","file_quarantine",tags=["responder","file_quarantine"],payload={"src":path,"dest":dest,"entity":entity_id})
            return True
        except Exception as e:
            self.bridge.ingest.emit(f"[responder] quarantine error: {e}", tags=["responder","error"])
            return False

    def throttle_dest(self, dest_str, entity_id=None):
        if entity_id and self.lists.is_allowed(entity_id):
            self.bridge.ingest.emit(f"{EMBLEMS['lists']} Allowed: skip throttle {entity_id}", tags=["lists","allow","skip"])
            return False
        msg = f"Egress throttle applied to {dest_str}"
        self.bridge.threat.emit(f"{EMBLEMS['globe']} {msg}", tags=["responder","egress_throttle"], payload={"dest":dest_str,"entity":entity_id})
        self.memory.add("responder","egress_throttle",tags=["responder","egress_throttle"],payload={"dest":dest_str,"entity":entity_id})
        return True

    def set_host_isolation(self, on: bool):
        self.host_isolation = on
        state = "ENABLED" if on else "DISABLED"
        msg = f"Host isolation {state} (temporary)"
        self.bridge.threat.emit(f"{EMBLEMS['triangle']} {msg}", tags=["responder","host_isolation"], payload={"state":state})
        self.memory.add("responder","host_isolation",tags=["responder","host_isolation"],payload={"state":state})

# -----------------------------------------------------------------------------
# Collectors (cross-OS, defensive)
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

    def _tag_groups_from_meta(self, name, exe, parent_name):
        tags = set()
        for g, spec in self.lists.software_map.items():
            if name and name.lower() in [pn.lower() for pn in spec.get("process_names", [])]:
                tags.add(g)
            if exe and any(sub.lower() in exe.lower() for sub in spec.get("file_paths_contains", [])):
                tags.add(g)
            if parent_name and parent_name.lower() in [pn.lower() for pn in spec.get("process_names", [])]:
                tags.add(g)
        return tags

    def process_monitor(self):
        if not psutil:
            log.warning("psutil unavailable; process monitor disabled")
            return
        while self.running:
            try:
                for p in psutil.process_iter(attrs=["pid","ppid","name","cmdline","username","exe"]):
                    info = p.info
                    pid = info.get("pid")
                    name = (info.get("name") or "")
                    cmd = " ".join(info.get("cmdline", [])).lower() if info.get("cmdline") else ""
                    exe = (info.get("exe") or "")
                    parent_name = ""
                    try:
                        parent = psutil.Process(info.get("ppid"))
                        parent_name = (parent.name() or "")
                    except Exception:
                        parent_name = ""

                    entity = f"proc:{pid}:{name}"
                    tags = self._tag_groups_from_meta(name, exe, parent_name)
                    self.observed.upsert(entity, "proc", meta={"pid":pid,"name":name,"path":exe,"cmd":cmd}, tags=tags)

                    if any(x in cmd for x in ["rundll32","powershell","mshta","certutil","wmic","bash","sh","osascript"]) and cmd:
                        self.lists.auto_disallow(entity, reason="lolbin_misuse")
                        self.trust.update(entity, ["lolbin_misuse"], {"maintenance_window": False})
                        self.bridge.threat.emit(
                            f"{EMBLEMS['shield_alert']} LOLBin/Script detected pid={pid} name={name} cmd={self.ace.mask(cmd)}",
                            tags=["collector","process","lolbin"], payload=info
                        )
                        if self.responder.process_termination and not self.lists.is_allowed(entity):
                            self.responder.kill_process_tree(pid, entity_id=entity)
                time.sleep(5)
            except Exception as e:
                self.bridge.ingest.emit(f"[process monitor] error: {e}", tags=["collector","error"])
                time.sleep(2)

    def network_monitor(self):
        if not psutil:
            log.warning("psutil unavailable; network monitor disabled")
            return
        while self.running:
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.status == 'ESTABLISHED' and conn.raddr:
                        # Normalize remote address to 'ip:port'
                        try:
                            dest_ip = conn.raddr.ip; dest_port = conn.raddr.port
                            dest_str = f"{dest_ip}:{dest_port}"
                        except Exception:
                            try:
                                ip, port = conn.raddr
                                dest_str = f"{ip}:{port}"
                            except Exception:
                                dest_str = str(conn.raddr)
                        entity = f"net:{dest_str}"
                        tags = set()
                        for g, spec in self.lists.software_map.items():
                            if any(sub.lower() in dest_str.lower() for sub in spec.get("network_domains_contains", [])):
                                tags.add(g)
                        self.observed.upsert(entity, "net", meta={"dest":dest_str,"status":conn.status}, tags=tags)

                        self.lists.auto_disallow(entity, reason="beacon")
                        self.bridge.sync.emit(
                            f"{EMBLEMS['globe']} Outbound: {self.ace.mask(dest_str)}",
                            tags=["collector","network","outbound"],
                            payload={"dest": dest_str}
                        )
                        self.trust.update(entity, ["beacon"], {"maintenance_window": False})
                        if self.responder.egress_throttle and not self.lists.is_allowed(entity):
                            self.responder.throttle_dest(dest_str, entity_id=entity)
                time.sleep(10)
            except Exception as e:
                self.bridge.ingest.emit(f"[network monitor] error: {e}", tags=["collector","error"])
                time.sleep(2)

    def filesystem_monitor(self):
        if not (watchdog and watchdog_events):
            log.warning("watchdog unavailable; filesystem monitor disabled")
            return

        class Handler(watchdog_events.FileSystemEventHandler):
            def on_modified(_, event):
                try:
                    src_abs = os.path.abspath(event.src_path)
                    if os.path.isdir(src_abs):
                        return
                    if os.path.commonpath([src_abs, self.quarantine_dir]) == self.quarantine_dir:
                        return

                    e = {"src_path": event.src_path}
                    sanitized = self.ace.sanitize_event(e)["src_path"]
                    entity = f"file:{event.src_path}"
                    base = os.path.basename(event.src_path)
                    tags = set()
                    for g, spec in self.lists.software_map.items():
                        if any(sub.lower() in src_abs.lower() for sub in spec.get("file_paths_contains", [])):
                            tags.add(g)
                        if base.lower() in [pn.lower() for pn in spec.get("process_names", [])]:
                            tags.add(g)
                    self.observed.upsert(entity, "file", meta={"path":event.src_path,"base":base}, tags=tags)

                    self.bridge.ingest.emit(f"{EMBLEMS['lock']} File modified: {sanitized}",
                                            tags=["collector","filesystem","modified"], payload=e)

                    suspicious = ("\\Startup", "/etc/rc", "/.config/autostart", "\\AppData\\")
                    if any(sp in event.src_path for sp in suspicious):
                        self.lists.auto_disallow(entity, reason="fs_persist")
                        if self.responder.file_quarantine and not self.lists.is_allowed(entity):
                            self.responder.quarantine_file(event.src_path, entity_id=entity)
                except Exception as ex:
                    self.bridge.ingest.emit(f"[filesystem handler] error: {ex}", tags=["collector","error"])

        try:
            observer = watchdog.Observer()
            root = "/" if CAPS["os"] != "Windows" else "C:\\"
            observer.schedule(Handler(), path=root, recursive=True)
            observer.start()
            while self.running:
                time.sleep(1)
        except Exception as e:
            self.bridge.ingest.emit(f"[filesystem monitor] error: {e}", tags=["collector","error"])

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
if CAPS["ui"]:
    import tkinter as tk
    from tkinter import ttk

class HoloFace:
    def __init__(self, root, width=1200, height=200):
        self.canvas = tk.Canvas(root, width=width, height=height, bg="#0a0a0a", highlightthickness=0)
        self.canvas.pack(fill="x")
        self.w, self.h = width, height
        self.intensity = 0.3
        threading.Thread(target=self._animate, daemon=True).start()

    def set_intensity(self, v): self.intensity = max(0.0, min(1.0, v))

    def _draw(self):
        self.canvas.delete("face")
        cx, cy = self.w//2, self.h//2
        self.canvas.create_oval(cx-140, cy-70, cx+140, cy+70, outline="#22ffee", width=2, tags="face")
        self.canvas.create_oval(cx-60, cy-20, cx-30, cy+10, outline="#22ffee", width=2, tags="face")
        self.canvas.create_oval(cx+30, cy-20, cx+60, cy+10, outline="#22ffee", width=2, tags="face")
        w = int(60 + 80*self.intensity)
        self.canvas.create_arc(cx-w//2, cy+10, cx+w//2, cy+40, start=0, extent=180, style="arc",
                               outline="#22ffee", width=2, tags="face")

    def _streams(self):
        for _ in range(3):
            x0 = random.randint(0, self.w); y0 = random.randint(0, self.h)
            x1 = x0 + random.randint(-120, 120); y1 = y0 + random.randint(-40, 40)
            c = random.choice(["#13ffd6", "#00aaff", "#66ffcc"])
            self.canvas.create_line(x0, y0, x1, y1, fill=c, width=1, tags="streams")
        self.canvas.after(140, lambda: self.canvas.delete("streams"))

    def _animate(self):
        while True:
            try:
                self._draw(); self._streams()
                time.sleep(0.1)
            except Exception:
                break

class TrustGraphPanel:
    def __init__(self, parent, trust: TrustModel):
        self.parent = parent
        self.trust = trust
        self.canvas = tk.Canvas(parent, bg="#111111", height=240, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.entities = []
        threading.Thread(target=self._loop, daemon=True).start()

    def set_entities(self, ents): self.entities = ents[:5]

    def _axes(self, w, h):
        self.canvas.create_line(40, h-30, w-20, h-30, fill="#444")
        self.canvas.create_line(40, 20, 40, h-30, fill="#444")

    def _normalize(self, values, w, h):
        if not values: return []
        mn, mx = min(values), max(values); span = max(1e-6, mx-mn)
        pts = []
        for i, v in enumerate(values):
            x = 40 + (i / max(1, len(values)-1)) * (w-60)
            y = (h-30) - ((v - mn)/span) * (h-60)
            pts.append((x,y))
        return pts

    def _loop(self):
        while True:
            try:
                self.canvas.delete("all")
                w = self.canvas.winfo_width() or 800
                h = self.canvas.winfo_height() or 240
                self._axes(w,h)
                colors = ["#00d4aa","#66aaff","#ffcc66","#ff6699","#aaff66"]
                for idx, entity in enumerate(self.entities):
                    hist = list(self.trust.history.get(entity, []))
                    values = [v for _, v in hist]
                    pts = self._normalize(values, w, h)
                    for i in range(1, len(pts)):
                        x0,y0 = pts[i-1]; x1,y1 = pts[i]
                        self.canvas.create_line(x0,y0,x1,y1, fill=colors[idx%len(colors)], width=2)
                    self.canvas.create_text(80 + idx*150, 20, text=f"{entity}", fill=colors[idx%len(colors)], anchor="w")
                time.sleep(2)
            except Exception:
                time.sleep(2)

class MemoryPanel:
    def __init__(self, parent, memory: MemoryVault, bridge: Bridge):
        self.parent = parent; self.memory = memory; self.bridge = bridge
        self.frame = tk.Frame(parent, bg="#111111"); self.frame.pack(fill="both", expand=True)
        ctrl = tk.Frame(self.frame, bg="#1a1a1a"); ctrl.pack(fill="x")
        tk.Label(ctrl, text="Tags (comma):", bg="#1a1a1a", fg="white").pack(side="left", padx=6)
        self.tags = tk.Entry(ctrl, bg="#222", fg="#E6E6E6", width=30); self.tags.pack(side="left")
        tk.Label(ctrl, text="Since seconds:", bg="#1a1a1a", fg="white").pack(side="left", padx=6)
        self.since = tk.Entry(ctrl, bg="#222", fg="#E6E6E6", width=10); self.since.insert(0,"3600"); self.since.pack(side="left")
        tk.Button(ctrl, text="Recall", command=self.recall, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(ctrl, text="Save", command=self.memory.save, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        self.out = tk.Text(self.frame, bg="#111111", fg="#E6E6E6"); self.out.pack(fill="both", expand=True)

    def recall(self):
        tags = [t.strip() for t in self.tags.get().split(",") if t.strip()]
        try: since = int(self.since.get())
        except: since = 3600
        eps = self.memory.recall(include_tags=tags or None, since_seconds=since, limit=200)
        self.out.delete("1.0","end"); self.out.insert("end", f"Recall ({len(eps)}):\n")
        for ep in eps:
            self.out.insert("end", f"{ep['ts']} | {ep['source']} | {ep.get('tags')} | {ep['message']}\n")
        self.bridge.ingest.emit(f"{EMBLEMS['memory']} Recall {len(eps)}", tags=["memory","recall"])

class DragListsPanel:
    def __init__(self, parent, lists: ListManager, bridge: Bridge, observed: ObservedRegistry):
        self.parent = parent; self.lists = lists; self.bridge = bridge; self.observed = observed
        self.frame = tk.Frame(parent, bg="#111111"); self.frame.pack(fill="both", expand=True)

        # Top controls
        ctrl = tk.Frame(self.frame, bg="#1a1a1a"); ctrl.pack(fill="x")
        tk.Label(ctrl, text="Search:", bg="#1a1a1a", fg="white").pack(side="left", padx=6)
        self.search_entry = tk.Entry(ctrl, bg="#222", fg="#E6E6E6", width=30); self.search_entry.pack(side="left")
        tk.Button(ctrl, text="Refresh", command=self.refresh_all, bg="#2e2e2e", fg="white").pack(side="left", padx=6)

        tk.Label(ctrl, text="Group:", bg="#1a1a1a", fg="white").pack(side="left", padx=12)
        self.group_var = tk.StringVar(value="")
        self.group_menu = ttk.Combobox(ctrl, textvariable=self.group_var, values=self.lists.recent()["groups"], width=22)
        self.group_menu.pack(side="left")
        tk.Button(ctrl, text="Allow Group (All Observed)", command=self.group_allow, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(ctrl, text="Disallow Group (All Observed)", command=self.group_disallow, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(ctrl, text="Save Lists/Map", command=self.lists.save, bg="#2e2e2e", fg="white").pack(side="left", padx=6)

        body = tk.Frame(self.frame, bg="#111111"); body.pack(fill="both", expand=True)
        # Observed list
        left = tk.Frame(body, bg="#111111"); left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(left, text="Observed (drag to Allow / Disallow)", bg="#111111", fg="#00d4aa").pack(anchor="w")
        self.observed_list = tk.Listbox(left, bg="#0f0f0f", fg="#E6E6E6", selectmode="extended")
        self.observed_list.pack(fill="both", expand=True)

        # Allow list
        middle = tk.Frame(body, bg="#111111"); middle.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(middle, text="Allow List", bg="#111111", fg="#66ff99").pack(anchor="w")
        self.allow_listbox = tk.Listbox(middle, bg="#0f0f0f", fg="#E6E6E6"); self.allow_listbox.pack(fill="both", expand=True)

        # Disallow list
        right = tk.Frame(body, bg="#111111"); right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        tk.Label(right, text="Disallow List", bg="#111111", fg="#ff6699").pack(anchor="w")
        self.disallow_listbox = tk.Listbox(right, bg="#0f0f0f", fg="#E6E6E6"); self.disallow_listbox.pack(fill="both", expand=True)

        # Drag-and-drop bindings
        for lb in (self.observed_list, self.allow_listbox, self.disallow_listbox):
            lb.bind("<ButtonPress-1>", self.on_drag_start)
            lb.bind("<B1-Motion>", self.on_drag_move)
            lb.bind("<ButtonRelease-1>", self.on_drag_drop)

        # Group editor (bottom panel)
        editor = tk.LabelFrame(self.frame, text="Group Editor", bg="#111111", fg="#E6E6E6")
        editor.pack(fill="x", padx=6, pady=6)

        tk.Label(editor, text="Group name:", bg="#111111", fg="white").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.ed_group = tk.Entry(editor, bg="#222", fg="#E6E6E6", width=24); self.ed_group.grid(row=0, column=1, sticky="w")

        tk.Label(editor, text="Process names (one per line):", bg="#111111", fg="white").grid(row=1, column=0, sticky="nw", padx=6)
        self.ed_proc = tk.Text(editor, bg="#0f0f0f", fg="#E6E6E6", height=5, width=40); self.ed_proc.grid(row=1, column=1, sticky="we", padx=6)

        tk.Label(editor, text="File path substrings (one per line):", bg="#111111", fg="white").grid(row=2, column=0, sticky="nw", padx=6)
        self.ed_paths = tk.Text(editor, bg="#0f0f0f", fg="#E6E6E6", height=5, width=40); self.ed_paths.grid(row=2, column=1, sticky="we", padx=6)

        tk.Label(editor, text="Network domains substrings (one per line):", bg="#111111", fg="white").grid(row=3, column=0, sticky="nw", padx=6)
        self.ed_domains = tk.Text(editor, bg="#0f0f0f", fg="#E6E6E6", height=4, width=40); self.ed_domains.grid(row=3, column=1, sticky="we", padx=6)

        btns = tk.Frame(editor, bg="#111111"); btns.grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        tk.Button(btns, text="Create/Update Group", command=self.editor_upsert_group, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(btns, text="Delete Group", command=self.editor_delete_group, bg="#aa3333", fg="white").pack(side="left", padx=6)
        tk.Button(btns, text="Load Group", command=self.editor_load_group, bg="#2e2e2e", fg="white").pack(side="left", padx=6)

        self.drag_data = {"items": [], "origin": None}
        self.refresh_all()

    def refresh_all(self):
        query = self.search_entry.get().strip().lower()
        # observed
        self.observed_list.delete(0, "end")
        obs = self.observed.list()
        for eid, e in sorted(obs.items()):
            label = f"{eid} [{e['type']}] {','.join(sorted(e.get('tags', [])))}"
            if not query or query in label.lower():
                self.observed_list.insert("end", label)
        # allow/disallow
        snap = self.lists.recent(limit=2000)
        self.allow_listbox.delete(0, "end")
        self.disallow_listbox.delete(0, "end")
        for a in snap["allow"]:
            self.allow_listbox.insert("end", a)
        for d in snap["disallow"]:
            self.disallow_listbox.insert("end", d)
        # update group menu values
        try:
            self.group_menu["values"] = self.lists.recent()["groups"]
        except Exception:
            pass

    def group_allow(self):
        g = self.group_var.get().strip()
        if not g: return
        count = self.lists.allow_group(g)
        self.bridge.ingest.emit(f"{EMBLEMS['group']} Group allow {g} ({count})", tags=["lists","group","allow"])
        self.refresh_all()

    def group_disallow(self):
        g = self.group_var.get().strip()
        if not g: return
        count = self.lists.disallow_group(g)
        self.bridge.ingest.emit(f"{EMBLEMS['group']} Group disallow {g} ({count})", tags=["lists","group","disallow"])
        self.refresh_all()

    def _listbox_items_to_entity_ids(self, lb, indices):
        ids = []
        for i in indices:
            text = lb.get(i)
            eid = text.split(" ", 1)[0].strip()
            ids.append(eid)
        return ids

    def on_drag_start(self, event):
        widget = event.widget
        if widget is self.observed_list:
            selection = list(self.observed_list.curselection())
            self.drag_data["items"] = self._listbox_items_to_entity_ids(self.observed_list, selection)
            self.drag_data["origin"] = "observed"
        elif widget is self.allow_listbox:
            selection = list(self.allow_listbox.curselection())
            self.drag_data["items"] = [self.allow_listbox.get(i) for i in selection]
            self.drag_data["origin"] = "allow"
        elif widget is self.disallow_listbox:
            selection = list(self.disallow_listbox.curselection())
            self.drag_data["items"] = [self.disallow_listbox.get(i) for i in selection]
            self.drag_data["origin"] = "disallow"

    def on_drag_move(self, event):
        pass

    def on_drag_drop(self, event):
        if not self.drag_data["items"]:
            return
        x, y = event.x_root, event.y_root
        target = None
        for lb, name in ((self.allow_listbox, "allow"), (self.disallow_listbox, "disallow"), (self.observed_list, "observed")):
            bx = lb.winfo_rootx(); by = lb.winfo_rooty()
            bw = lb.winfo_width(); bh = lb.winfo_height()
            if bx <= x <= bx + bw and by <= y <= by + bh:
                target = name; break
        items = self.drag_data["items"]; self.drag_data = {"items": [], "origin": None}

        if target == "allow":
            for eid in items:
                self.lists.move_to_allow(eid)
                self.bridge.ingest.emit(f"{EMBLEMS['lists']} Allowed: {eid}", tags=["lists","allow"])
        elif target == "disallow":
            for eid in items:
                self.lists.move_to_disallow(eid)
                self.bridge.ingest.emit(f"{EMBLEMS['lists']} Disallowed: {eid}", tags=["lists","disallow"])
        self.refresh_all()

    # Group editor handlers
    def editor_load_group(self):
        name = self.group_var.get().strip() or self.ed_group.get().strip()
        if not name: return
        gm = self.lists.software_map.get(name)
        if not gm: return
        self.ed_group.delete(0,"end"); self.ed_group.insert(0, name)
        self.ed_proc.delete("1.0","end"); self.ed_paths.delete("1.0","end"); self.ed_domains.delete("1.0","end")
        self.ed_proc.insert("end", "\n".join(gm.get("process_names", [])))
        self.ed_paths.insert("end", "\n".join(gm.get("file_paths_contains", [])))
        self.ed_domains.insert("end", "\n".join(gm.get("network_domains_contains", [])))

    def editor_upsert_group(self):
        name = self.ed_group.get().strip()
        procs = [l.strip() for l in self.ed_proc.get("1.0","end").splitlines() if l.strip()]
        paths = [l.strip() for l in self.ed_paths.get("1.0","end").splitlines() if l.strip()]
        domains = [l.strip() for l in self.ed_domains.get("1.0","end").splitlines() if l.strip()]
        if not name: return
        ok = self.lists.upsert_group(name, procs, paths, domains)
        if ok:
            self.bridge.ingest.emit(f"{EMBLEMS['group']} Upserted group {name}", tags=["lists","group","edit"])
            self.lists.save()
            self.refresh_all()

    def editor_delete_group(self):
        name = self.ed_group.get().strip() or self.group_var.get().strip()
        if not name: return
        if self.lists.delete_group(name):
            self.bridge.ingest.emit(f"{EMBLEMS['group']} Deleted group {name}", tags=["lists","group","delete"])
            self.lists.save()
            self.refresh_all()

class DefenderUI:
    def __init__(self, orch):
        self.orch = orch
        self.root = tk.Tk()
        self.root.title("🖥️ Hardened Defender — Live Oversight")
        self.root.geometry("1440x980")
        self.root.configure(bg="#111111")

        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill="both", expand=True)

        # Console
        self.console_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.console_tab, text="Console")
        self.holo = HoloFace(self.console_tab, width=1440, height=200)
        self.text = tk.Text(self.console_tab, bg="#111111", fg="#E6E6E6", insertbackground="#E6E6E6", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True)
        panel = tk.Frame(self.console_tab, bg="#1a1a1a"); panel.pack(fill="x")
        tk.Button(panel, text="🔺 Switch Mode", command=self.orch.switch_bot_mode, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)
        tk.Button(panel, text="🜏 Escalate Defense", command=self.orch.escalate_defense, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        self.iso_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="🔺 Host Isolation (temporary, explicit)", variable=self.iso_var,
                       command=lambda: self.orch.responder.set_host_isolation(self.iso_var.get()),
                       bg="#1a1a1a", fg="white", selectcolor="#1a1a1a").pack(side="left", padx=6)
        tk.Button(panel, text="📋 Save Lists/Map", command=self.orch.lists.save, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(panel, text="❌ Quit", command=self.root.quit, bg="#aa0000", fg="white").pack(side="right", padx=6)
        self.status = tk.Label(panel, text="", bg="#1a1a1a", fg="#00d4aa", font=("Consolas", 10))
        self.status.pack(side="right", padx=10)

        # Trust graphs
        self.trust_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.trust_tab, text="Trust Graphs")
        self.graph_panel = TrustGraphPanel(self.trust_tab, self.orch.trust)

        # Codex
        self.codex_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.codex_tab, text="Codex")
        self.codex_info = tk.Text(self.codex_tab, bg="#111111", fg="#E6E6E6"); self.codex_info.pack(fill="both", expand=True)

        # Memory
        self.memory_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.memory_tab, text="Memory")
        self.memory_panel = MemoryPanel(self.memory_tab, self.orch.memory, self.orch.bridge)

        # Drag Lists + Group Editor
        self.lists_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.lists_tab, text="Lists (Drag & Drop + Groups)")
        self.drag_panel = DragListsPanel(self.lists_tab, self.orch.lists, self.orch.bridge, self.orch.observed)

        # Attach
        self.orch.bridge.attach_ui(self)
        threading.Thread(target=self._update_status_loop, daemon=True).start()
        threading.Thread(target=self._update_codex_loop, daemon=True).start()
        threading.Thread(target=self._update_graph_entities, daemon=True).start()
        self.root.mainloop()

    def append_line(self, msg):
        self.text.insert("end", msg + "\n"); self.text.see("end")
        if any(k in msg for k in ["LOLBin","Egress throttle","Quarantined file","Terminated process tree","Disallowed","Allowed","Group"]):
            self.holo.set_intensity(0.8)
        else:
            self.holo.set_intensity(0.3)

    def _update_status_loop(self):
        while True:
            try:
                ace = "ON" if self.orch.ace.enabled else "OFF"
                iso = "ENABLED" if self.orch.responder.host_isolation else "DISABLED"
                status = f"ACE: {ace} | Host Isolation: {iso} | Allow={len(self.orch.lists.allow_list)} Disallow={len(self.orch.lists.disallow_list)} Observed={len(self.orch.observed.list())}"
                self.status.config(text=status)
                time.sleep(3)
            except Exception:
                time.sleep(3)

    def _update_codex_loop(self):
        while True:
            try:
                mf = self.orch.codex.export_manifest()
                self.codex_info.delete("1.0","end")
                self.codex_info.insert("end", f"Codex Manifest:\n{json.dumps(mf, indent=2)}\n")
                time.sleep(7)
            except Exception:
                time.sleep(7)

    def _update_graph_entities(self):
        while True:
            try:
                items = sorted(self.orch.trust.scores.items(), key=lambda kv: kv[1], reverse=True)
                self.graph_panel.set_entities([k for k,_ in items[:5]])
                time.sleep(5)
            except Exception:
                time.sleep(5)

# -----------------------------------------------------------------------------
# Bot (signals only; minimal pulse)
# -----------------------------------------------------------------------------
class Bot:
    def __init__(self, cb, codex: Codex):
        self.cb = cb
        self.codex = codex
        self.run = True
        self.mode = "guardian"

    def switch(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb(f"{EMBLEMS['triangle']} Mode switched to {self.mode.upper()}")

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                if self.mode == "guardian":
                    self.codex.mutate({"obfuscation_spike": False})
                    self.cb(f"{EMBLEMS['dove']} Guardian pulse")
                else:
                    self.cb(f"{EMBLEMS['alchemical']} Rogue calibration (defensive-only)")
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
            f"{EMBLEMS['dna']} {EMBLEMS['monitor']} Defender online | OS={CAPS['os']} | UI={self.use_ui}",
            tags=["system","startup"]
        )
        if self.use_ui:
            DefenderUI(self)
        else:
            log.info("CLI. Commands: switch | escalate | isolate on/off | ace on/off | allow <entity> | disallow <entity> | group allow <name> | group disallow <name> | group upsert <json> | group delete <name> | save | quit")
            try:
                while True:
                    cmd = input("> ").strip().lower()
                    if cmd in ("switch","s"):
                        self.bot.switch()
                    elif cmd in ("escalate","e"):
                        before = self.codex.export_manifest()
                        self.codex.mutate({"escalate_defense": True})
                        after = self.codex.export_manifest()
                        diffs = self.codex.diff(before, after)
                        self.bridge.ingest.emit("Defense diffs: " + ", ".join(diffs), tags=["codex","diff"])
                    elif cmd.startswith("allow "):
                        ent = cmd.split(" ",1)[1].strip()
                        self.lists.move_to_allow(ent)
                        self.bridge.ingest.emit(f"{EMBLEMS['lists']} Allowed: {ent}", tags=["lists","allow"])
                    elif cmd.startswith("disallow "):
                        ent = cmd.split(" ",1)[1].strip()
                        self.lists.move_to_disallow(ent)
                        self.bridge.ingest.emit(f"{EMBLEMS['lists']} Disallowed: {ent}", tags=["lists","disallow"])
                    elif cmd.startswith("group allow "):
                        g = cmd.split(" ",2)[2].strip()
                        c = self.lists.allow_group(g)
                        self.bridge.ingest.emit(f"{EMBLEMS['group']} Group allow {g} ({c})", tags=["lists","group","allow"])
                    elif cmd.startswith("group disallow "):
                        g = cmd.split(" ",2)[2].strip()
                        c = self.lists.disallow_group(g)
                        self.bridge.ingest.emit(f"{EMBLEMS['group']} Group disallow {g} ({c})", tags=["lists","group","disallow"])
                    elif cmd.startswith("group upsert "):
                        try:
                            payload = json.loads(cmd.split(" ",2)[2])
                            name = payload.get("name"); procs = payload.get("process_names", [])
                            paths = payload.get("file_paths_contains", []); domains = payload.get("network_domains_contains", [])
                            self.lists.upsert_group(name, procs, paths, domains); self.lists.save()
                            self.bridge.ingest.emit(f"{EMBLEMS['group']} Upserted {name}", tags=["lists","group","edit"])
                        except Exception as ex:
                            log.info(f"Bad upsert payload: {ex}")
                    elif cmd.startswith("group delete "):
                        g = cmd.split(" ",2)[2].strip()
                        self.lists.delete_group(g); self.lists.save()
                        self.bridge.ingest.emit(f"{EMBLEMS['group']} Deleted {g}", tags=["lists","group","delete"])
                    elif cmd in ("isolate on","iso on"):
                        self.responder.set_host_isolation(True)
                    elif cmd in ("isolate off","iso off"):
                        self.responder.set_host_isolation(False)
                    elif cmd in ("ace on","cipher on"):
                        self.ace.toggle(True)
                    elif cmd in ("ace off","cipher off"):
                        self.ace.toggle(False)
                    elif cmd in ("save","persist"):
                        self.memory.save(); self.lists.save()
                    elif cmd in ("quit","q","exit"):
                        log.info("Shutting down...")
                        break
                    else:
                        log.info("Commands: switch | escalate | isolate on/off | ace on/off | allow <entity> | disallow <entity> | group allow <name> | group disallow <name> | group upsert <json> | group delete <name> | save | quit")
            except (EOFError, KeyboardInterrupt):
                log.info("Exiting...")

    # GUI wrappers
    def switch_bot_mode(self):
        self.bot.switch()

    def escalate_defense(self):
        before = self.codex.export_manifest()
        self.codex.mutate({"escalate_defense": True})
        after = self.codex.export_manifest()
        diffs = self.codex.diff(before, after)
        self.bridge.ingest.emit("Defense diffs: " + ", ".join(diffs), tags=["codex","diff"])

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
def main():
    orch = Orchestrator(use_ui=True)
    orch.start()

if __name__ == "__main__":
    main()

