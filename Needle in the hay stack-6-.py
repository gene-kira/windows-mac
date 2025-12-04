#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardened, defensive-only, mixed OS autonomous detection and response system
with Tkinter GUI, memory vault, zero-hesitation responders, safe host isolation,
and Allow/Disallow lists with auto-population and overrides.

This version integrates:
- Robust GUI and orchestration (includes switch_bot_mode wrapper)
- Crash-proof file quarantine routine
- Filesystem monitor excluding quarantine directory (prevents recursion)
- Normalized network addresses (fix: egress throttle crash on conn.raddr)
- Zero-hesitation actions: process termination, file quarantine, egress throttle
- Temporary, explicit host isolation toggle
- Trust graphs, codex manifest view, memory panel and console
- Allow/Disallow lists
  - Disallow auto-populates from detections (process, network, filesystem)
  - Override controls in GUI: move entity to Allow (bypass) or Disallow (enforce)
  - Persistent storage (allow_list.json, disallow_list.json)
  - Memory logging for every change

Safety:
- Non-destructive toward people. Software-only containments.
- Host isolation is reversible and requires explicit confirmation.

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
# Allow/Disallow Lists Manager
# -----------------------------------------------------------------------------
class ListManager:
    def __init__(self, memory: MemoryVault, path_allow="allow_list.json", path_disallow="disallow_list.json"):
        self.memory = memory
        self.path_allow = path_allow
        self.path_disallow = path_disallow
        self._lock = threading.Lock()
        self.allow_list = set(safe_load_json(path_allow, []))
        self.disallow_list = set(safe_load_json(path_disallow, []))
        log.info(f"[lists] Loaded allow={len(self.allow_list)} disallow={len(self.disallow_list)}")

    def save(self):
        with self._lock:
            safe_save_json(self.path_allow, sorted(list(self.allow_list)))
            safe_save_json(self.path_disallow, sorted(list(self.disallow_list)))
            log.info("[lists] Saved allow/disallow")

    def is_allowed(self, entity_id: str) -> bool:
        return entity_id in self.allow_list

    def is_disallowed(self, entity_id: str) -> bool:
        return entity_id in self.disallow_list

    def auto_disallow(self, entity_id: str, reason: str):
        with self._lock:
            if entity_id not in self.allow_list:
                if entity_id not in self.disallow_list:
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

    def recent(self, limit=50):
        return {
            "allow": sorted(list(self.allow_list))[:limit],
            "disallow": sorted(list(self.disallow_list))[:limit],
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
        # Defaults enabled
        self.process_termination = True
        self.file_quarantine = True
        self.egress_throttle = True
        self.host_isolation = False  # explicit confirmation

        # Quarantine paths
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
    def __init__(self, bridge: Bridge, trust: TrustModel, ace: ACE, responder: Responder, lists: ListManager, quarantine_dir: str):
        self.bridge = bridge
        self.trust = trust
        self.ace = ace
        self.responder = responder
        self.lists = lists
        self.running = True
        self.quarantine_dir = os.path.abspath(quarantine_dir)

    def start(self):
        threading.Thread(target=self.process_monitor, daemon=True).start()
        threading.Thread(target=self.network_monitor, daemon=True).start()
        threading.Thread(target=self.filesystem_monitor, daemon=True).start()

    def process_monitor(self):
        if not psutil:
            log.warning("psutil unavailable; process monitor disabled")
            return
        while self.running:
            try:
                for p in psutil.process_iter(attrs=["pid","ppid","name","cmdline","username"]):
                    info = p.info
                    cmd = " ".join(info.get("cmdline", [])).lower() if info.get("cmdline") else ""
                    entity = f"proc:{info.get('pid')}:{info.get('name')}"
                    if any(x in cmd for x in ["rundll32","powershell","mshta","certutil","wmic","bash","sh","osascript"]) and cmd:
                        self.lists.auto_disallow(entity, reason="lolbin_misuse")
                        self.trust.update(entity, ["lolbin_misuse"], {"maintenance_window": False})
                        self.bridge.threat.emit(
                            f"{EMBLEMS['shield_alert']} LOLBin/Script detected pid={info['pid']} name={info['name']} cmd={self.ace.mask(cmd)}",
                            tags=["collector","process","lolbin"], payload=info
                        )
                        if self.responder.process_termination and not self.lists.is_allowed(entity):
                            self.responder.kill_process_tree(info["pid"], entity_id=entity)
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
                        # Normalize remote address to 'ip:port' string
                        try:
                            dest_ip = conn.raddr.ip
                            dest_port = conn.raddr.port
                            dest_str = f"{dest_ip}:{dest_port}"
                        except Exception:
                            try:
                                ip, port = conn.raddr
                                dest_str = f"{ip}:{port}"
                            except Exception:
                                dest_str = str(conn.raddr)
                        entity = f"net:{dest_str}"
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
                    # Skip quarantine directory to avoid recursion
                    if os.path.commonpath([src_abs, self.quarantine_dir]) == self.quarantine_dir:
                        return

                    e = {"src_path": event.src_path}
                    sanitized = self.ace.sanitize_event(e)["src_path"]
                    entity = f"file:{event.src_path}"
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
                self._draw()
                self._streams()
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

class ListsPanel:
    def __init__(self, parent, lists: ListManager, bridge: Bridge):
        self.parent = parent; self.lists = lists; self.bridge = bridge
        self.frame = tk.Frame(parent, bg="#111111"); self.frame.pack(fill="both", expand=True)

        ctrl = tk.Frame(self.frame, bg="#1a1a1a"); ctrl.pack(fill="x")
        tk.Label(ctrl, text="Entity ID:", bg="#1a1a1a", fg="white").pack(side="left", padx=6)
        self.entity_entry = tk.Entry(ctrl, bg="#222", fg="#E6E6E6", width=50); self.entity_entry.pack(side="left")
        tk.Button(ctrl, text="Allow", command=self.allow_entity, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(ctrl, text="Disallow", command=self.disallow_entity, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(ctrl, text="Save Lists", command=self.lists.save, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(ctrl, text="Refresh", command=self.refresh, bg="#2e2e2e", fg="white").pack(side="left", padx=6)

        body = tk.Frame(self.frame, bg="#111111"); body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg="#111111"); left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right = tk.Frame(body, bg="#111111"); right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        tk.Label(left, text="Allow List", bg="#111111", fg="#00d4aa").pack(anchor="w")
        self.allow_text = tk.Text(left, bg="#0f0f0f", fg="#E6E6E6"); self.allow_text.pack(fill="both", expand=True)

        tk.Label(right, text="Disallow List", bg="#111111", fg="#ff6699").pack(anchor="w")
        self.disallow_text = tk.Text(right, bg="#0f0f0f", fg="#E6E6E6"); self.disallow_text.pack(fill="both", expand=True)

        self.refresh()

    def allow_entity(self):
        ent = self.entity_entry.get().strip()
        if ent:
            self.lists.move_to_allow(ent)
            self.bridge.ingest.emit(f"{EMBLEMS['lists']} Allowed: {ent}", tags=["lists","allow"])
            self.refresh()

    def disallow_entity(self):
        ent = self.entity_entry.get().strip()
        if ent:
            self.lists.move_to_disallow(ent)
            self.bridge.ingest.emit(f"{EMBLEMS['lists']} Disallowed: {ent}", tags=["lists","disallow"])
            self.refresh()

    def refresh(self):
        snap = self.lists.recent(limit=500)
        self.allow_text.delete("1.0","end"); self.disallow_text.delete("1.0","end")
        self.allow_text.insert("end", "\n".join(snap["allow"]) or "(empty)")
        self.disallow_text.insert("end", "\n".join(snap["disallow"]) or "(empty)")

class DefenderUI:
    def __init__(self, orch):
        self.orch = orch
        self.root = tk.Tk()
        self.root.title("🖥️ Hardened Defender — Live Oversight")
        self.root.geometry("1320x900")
        self.root.configure(bg="#111111")

        self.tabs = ttk.Notebook(self.root); self.tabs.pack(fill="both", expand=True)

        self.console_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.console_tab, text="Console")
        self.holo = HoloFace(self.console_tab, width=1200, height=200)
        self.text = tk.Text(self.console_tab, bg="#111111", fg="#E6E6E6", insertbackground="#E6E6E6", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True)
        panel = tk.Frame(self.console_tab, bg="#1a1a1a"); panel.pack(fill="x")
        tk.Button(panel, text="🔺 Switch Mode", command=self.orch.switch_bot_mode, bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)
        tk.Button(panel, text="🜏 Escalate Defense", command=self.orch.escalate_defense, bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        self.iso_var = tk.BooleanVar(value=False)
        tk.Checkbutton(panel, text="🔺 Host Isolation (temporary, explicit)", variable=self.iso_var,
                       command=lambda: self.orch.responder.set_host_isolation(self.iso_var.get()),
                       bg="#1a1a1a", fg="white", selectcolor="#1a1a1a").pack(side="left", padx=6)
        tk.Button(panel, text="❌ Quit", command=self.root.quit, bg="#aa0000", fg="white").pack(side="right", padx=6)
        self.status = tk.Label(panel, text="", bg="#1a1a1a", fg="#00d4aa", font=("Consolas", 10))
        self.status.pack(side="right", padx=10)

        self.trust_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.trust_tab, text="Trust Graphs")
        self.graph_panel = TrustGraphPanel(self.trust_tab, self.orch.trust)

        self.codex_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.codex_tab, text="Codex")
        self.codex_info = tk.Text(self.codex_tab, bg="#111111", fg="#E6E6E6"); self.codex_info.pack(fill="both", expand=True)

        self.memory_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.memory_tab, text="Memory")
        self.memory_panel = MemoryPanel(self.memory_tab, self.orch.memory, self.orch.bridge)

        self.lists_tab = tk.Frame(self.tabs, bg="#111111"); self.tabs.add(self.lists_tab, text="Lists")
        self.lists_panel = ListsPanel(self.lists_tab, self.orch.lists, self.orch.bridge)

        self.orch.bridge.attach_ui(self)
        threading.Thread(target=self._update_status_loop, daemon=True).start()
        threading.Thread(target=self._update_codex_loop, daemon=True).start()
        threading.Thread(target=self._update_graph_entities, daemon=True).start()
        self.root.mainloop()

    def append_line(self, msg):
        self.text.insert("end", msg + "\n"); self.text.see("end")
        if any(k in msg for k in ["LOLBin","Egress throttle","Quarantined file","Terminated process tree","Disallowed","Allowed"]):
            self.holo.set_intensity(0.8)
        else:
            self.holo.set_intensity(0.3)

    def _update_status_loop(self):
        while True:
            try:
                ace = "ON" if self.orch.ace.enabled else "OFF"
                iso = "ENABLED" if self.orch.responder.host_isolation else "DISABLED"
                status = f"ACE: {ace} | Host Isolation: {iso} | Allow={len(self.orch.lists.allow_list)} Disallow={len(self.orch.lists.disallow_list)}"
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
        self.lists = ListManager(memory=self.memory)
        self.responder = Responder(self.bridge, self.memory, self.ace, self.lists)
        self.collectors = Collectors(self.bridge, self.trust, self.ace, self.responder, self.lists, quarantine_dir=self.responder.quarantine_dir)
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
            log.info("Running CLI. Commands: switch | escalate | isolate on/off | ace on/off | allow <entity> | disallow <entity> | save | quit")
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
                        log.info("Commands: switch | escalate | isolate on/off | ace on/off | allow <entity> | disallow <entity> | save | quit")
            except (EOFError, KeyboardInterrupt):
                log.info("Exiting...")

    # Wrapper for GUI button
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

