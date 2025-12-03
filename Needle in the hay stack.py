#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified single-file, defensive-only, mixed OS detection system.
Features:
- Autoloader with graceful fallback
- Adaptive Codex mutation (ghost sync, phantom nodes, retention control)
- Cross-OS collectors (process/network/filesystem)
- Trust scoring and compliance auditing
- DualPersonalityBot (guardian/rogue) for safe entropy probes and audits
- Orchestrator and optional Tkinter console

Safety note:
- Strictly defensive. No destructive or harmful actions. "Rogue" mode is a harmless
  entropy probe for detection calibration, not offensive or destructive.
"""

import importlib
import logging
import platform
import threading
import time
import random
from collections import defaultdict
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("unified")

# -----------------------------------------------------------------------------
# Autoloader (graceful, optional imports)
# -----------------------------------------------------------------------------
def optional_import(name):
    try:
        m = importlib.import_module(name)
        log.info(f"[autoloader] Loaded {name}")
        return m
    except Exception as e:
        log.warning(f"[autoloader] Missing {name}: {e}")
        return None

cp = optional_import("cupy")                      # GPU arrays (optional)
np = optional_import("numpy")                     # CPU arrays (fallback, optional)
psutil = optional_import("psutil")                # cross-OS process/network, optional
watchdog = optional_import("watchdog.observers")  # optional
watchdog_events = optional_import("watchdog.events")  # optional
tkinter = optional_import("tkinter")              # optional console UI
yaml = optional_import("yaml")                    # optional codex sync
requests = optional_import("requests")            # optional swarm sync transport

CAPS = {
    "os": platform.system(),
    "gpu": cp is not None,
    "numpy": np is not None,
    "process": psutil is not None,
    "fs_watch": (watchdog is not None and watchdog_events is not None),
    "ui": tkinter is not None,
    "net_sync": requests is not None,
}
log.info(f"[autoloader] Capabilities: {CAPS}")

# -----------------------------------------------------------------------------
# Utility: Glyphs and small helpers
# -----------------------------------------------------------------------------
GLYPHS = "⟁⎈⟡⚚⚝✦✧☄︎⌁"

def random_glyph_stream(n=64):
    return "".join(random.choice(GLYPHS) for _ in range(n))

def camouflage(s, style="alien"):
    return f"{style}:{s[::-1]}"

def reverse_mirror_encrypt(s):
    return "".join(chr((ord(c) ^ 0x2A)) for c in s[::-1])

def generate_decoy():
    return {
        "timestamp": int(time.time()),
        "origin": random.choice(["US", "DE", "JP", "BR", "IN", "GB"]),
        "payload": random_glyph_stream()
    }

# -----------------------------------------------------------------------------
# Adaptive Codex (mutation, ghost sync, swarm sync-ready)
# -----------------------------------------------------------------------------
class Codex:
    def __init__(self, retention_hours=24):
        self.version = 1
        self.retention = timedelta(hours=retention_hours)
        self.rules = {
            "lolbins": {"arg_entropy_threshold": 4.0},
            "rare_edges": {},
            "beacon": {},
            "identity": {},
        }
        self.phantom_nodes = set()
        self.history = []

    def mutate(self, signals):
        """
        Signals can include:
        - ghost_sync: bool
        - obfuscation_spike: bool
        - rare_edge_spike: bool
        """
        changed = False
        if signals.get("ghost_sync"):
            # Defensive response: shorten retention and add phantom node
            self.retention = max(timedelta(hours=6), self.retention - timedelta(hours=6))
            self.phantom_nodes.add(f"phantom-{int(datetime.utcnow().timestamp())}")
            changed = True
        if signals.get("obfuscation_spike"):
            self.rules.setdefault("lolbins", {})["arg_entropy_threshold"] = 4.2
            changed = True
        if signals.get("rare_edge_spike"):
            self.rules.setdefault("rare_edges", {})["sensitivity"] = "high"
            changed = True
        if changed:
            self.version += 1
            self.history.append({"ts": datetime.utcnow(), "signals": signals, "version": self.version})

    def export_manifest(self):
        return {
            "version": self.version,
            "retention_seconds": int(self.retention.total_seconds()),
            "rules": self.rules,
            "phantom_nodes": sorted(list(self.phantom_nodes)),
        }

    def import_manifest(self, manifest):
        # Signature verification would occur externally; here we simply apply
        self.version = manifest.get("version", self.version)
        self.retention = timedelta(seconds=manifest.get("retention_seconds", int(self.retention.total_seconds())))
        self.rules = manifest.get("rules", self.rules)
        self.phantom_nodes = set(manifest.get("phantom_nodes", list(self.phantom_nodes)))

# -----------------------------------------------------------------------------
# Trust model and Compliance auditor
# -----------------------------------------------------------------------------
SEVERITY = {
    "mem_inject": 9,    # metadata indicator only (no memory dumping)
    "cred_ops": 8,
    "lolbin_misuse": 6,
    "rare_edge": 5,
    "beacon": 4,
    "fs_startup": 4,
    "identity_deviation": 6,
}

class TrustModel:
    def __init__(self):
        self.scores = defaultdict(lambda: 0.0)

    def update(self, entity_id, signals, context):
        delta = 0.0
        for sig in signals:
            delta += SEVERITY.get(sig, 1)
        # suppress in maintenance windows
        if context.get("maintenance_window"):
            delta *= 0.3
        # time decay
        self.scores[entity_id] = max(0.0, self.scores[entity_id] * 0.95 + delta)
        return self.scores[entity_id]

def compliance_auditor(events, codex: Codex):
    findings = []
    # Retention check (example: provided events include age_seconds)
    for e in events:
        if e.get("age_seconds", 0) > codex.retention.total_seconds():
            findings.append("retention_violation")
    # Scope check (example threshold)
    if len(events) > 10000:
        findings.append("scope_excess")
    status = "pass" if not findings else f"fail:{','.join(findings)}"
    return {"status": status, "findings": findings}

# -----------------------------------------------------------------------------
# DualPersonalityBot (defensive-only entropy probe)
# -----------------------------------------------------------------------------
class DualPersonalityBot:
    def __init__(self, callback, codex: Codex, trust_model: TrustModel):
        self.cb = callback
        self.run = True
        self.mode = "guardian"  # "guardian" or "rogue"
        self.rogue_weights = [0.2, -0.4, 0.7]
        self.rogue_log = []
        self.codex = codex
        self.trust = trust_model

    def switch_mode(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb(f"🔺 Personality switched to {self.mode.upper()}")

    def guardian_behavior(self):
        # Routine mutation call without ghost sync
        self.codex.mutate({"ghost_sync": False})
        decoy = generate_decoy()
        audit = compliance_auditor([decoy], self.codex)
        self.cb(f"🕊️ Guardian audit: {decoy}")
        self.cb(f"🔱 Compliance: {audit}")

    def rogue_behavior(self):
        # Harmless entropy probe to calibrate detectors
        entropy = int(time.time()) % 2048
        scrambled = reverse_mirror_encrypt(str(entropy))
        camo = camouflage(str(entropy), "alien")
        glyph_stream = random_glyph_stream()
        unusual_pattern = f"{scrambled[:16]}-{camo}-{glyph_stream[:8]}"

        # Safe entropy tweak
        self.rogue_weights = [w + (entropy % 5 - 2) * 0.01 for w in self.rogue_weights]
        self.rogue_log.append(list(self.rogue_weights))
        score = sum(self.rogue_weights) / len(self.rogue_weights)

        self.cb("📈 Rogue entropy probe active (defensive-only)")
        self.cb(f"🜏 Rogue pattern: {unusual_pattern}")
        self.cb(f"📊 Rogue weights: {self.rogue_weights} | Trust {score:.3f}")

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                if self.mode == "guardian":
                    self.guardian_behavior()
                else:
                    self.rogue_behavior()
                time.sleep(10)
            except Exception as e:
                self.cb(f"[bot] error: {e}")
                time.sleep(2)

# -----------------------------------------------------------------------------
# Bridge and optional console
# -----------------------------------------------------------------------------
class ConsoleUI:
    def __init__(self):
        self.lines = []
        self.root = None
        if CAPS["ui"]:
            try:
                self.root = tkinter.Tk()
                self.root.title("ASI Console — Live Oversight")
                self.text = tkinter.Text(self.root, width=120, height=40)
                self.text.pack()
                threading.Thread(target=self._tk_loop, daemon=True).start()
            except Exception as e:
                log.warning(f"[console] Tkinter init failed: {e}")
                self.root = None

    def _tk_loop(self):
        while True:
            try:
                if self.lines:
                    line = self.lines.pop(0)
                    self.text.insert("end", line + "\n")
                    self.text.see("end")
                time.sleep(0.1)
            except Exception as e:
                log.warning(f"[console] loop error: {e}")
                break

    def append_line(self, msg):
        if self.root:
            self.lines.append(msg)
        else:
            # Fallback to log
            log.info(f"[console] {msg}")

class Bridge:
    def __init__(self, ui: ConsoleUI | None):
        self.ui = ui

    class Signal:
        def __init__(self, name, ui):
            self.name = name
            self.ui = ui
        def emit(self, msg):
            log.info(f"[{self.name}] {msg}")
            if self.ui:
                self.ui.append_line(f"[{self.name}] {msg}")

# -----------------------------------------------------------------------------
# Collectors (cross-OS, defensive)
# -----------------------------------------------------------------------------
class CrossOSCollectors:
    def __init__(self, bridge: Bridge, codex: Codex, trust: TrustModel):
        self.bridge = bridge
        self.codex = codex
        self.trust = trust
        self.running = True
        self._threads = []

    def start(self):
        self._start_thread(self.monitor_processes)
        self._start_thread(self.monitor_network)
        self._start_thread(self.monitor_filesystem)

    def _start_thread(self, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self._threads.append(t)

    def monitor_processes(self):
        if not psutil:
            log.warning("psutil unavailable; skipping process monitor")
            return
        while self.running:
            try:
                for p in psutil.process_iter(attrs=["pid", "ppid", "name", "cmdline", "username"]):
                    info = p.info
                    cmd = " ".join(info.get("cmdline", [])).lower() if info.get("cmdline") else ""
                    # LOLBins / script engines detection (cross-OS)
                    if any(x in cmd for x in ["certutil", "mshta", "rundll32", "powershell", "wmic", "bash", "sh", "osascript"]) and cmd:
                        self.bridge.Signal("threat", self.bridge.ui).emit(
                            f"LOLBin/Script usage: pid={info['pid']} name={info['name']} cmd={cmd}"
                        )
                        # Update trust model defensively
                        entity = f"proc:{info['pid']}:{info['name']}"
                        score = self.trust.update(entity, ["lolbin_misuse"], {"maintenance_window": False})
                        self.bridge.Signal("ingest", self.bridge.ui).emit(f"Trust[{entity}]={score:.2f}")
                time.sleep(5)
            except Exception as e:
                log.error(f"[process monitor] error: {e}")
                time.sleep(2)

    def monitor_network(self):
        if not psutil:
            log.warning("psutil unavailable; skipping network monitor")
            return
        while self.running:
            try:
                for conn in psutil.net_connections(kind='inet'):
                    if conn.status == 'ESTABLISHED' and conn.raddr:
                        self.bridge.Signal("sync", self.bridge.ui).emit(f"Outbound: {conn.raddr}")
                        # Simple beacon cadence hint
                        entity = f"net:{conn.raddr}"
                        self.trust.update(entity, ["beacon"], {"maintenance_window": False})
                time.sleep(10)
            except Exception as e:
                log.error(f"[network monitor] error: {e}")
                time.sleep(2)

    def monitor_filesystem(self):
        if not CAPS["fs_watch"]:
            log.warning("watchdog unavailable; skipping filesystem monitor")
            return

        class Handler(watchdog_events.FileSystemEventHandler):
            def on_modified(_, event):
                if not event.is_directory:
                    self.bridge.Signal("ingest", self.bridge.ui).emit(f"File modified: {event.src_path}")

        try:
            observer = watchdog.Observer()
            root = "/" if CAPS["os"] != "Windows" else "C:\\"
            observer.schedule(Handler(), path=root, recursive=True)
            observer.start()
            # Keep thread alive
            while self.running:
                time.sleep(1)
        except Exception as e:
            log.error(f"[filesystem monitor] error: {e}")

# -----------------------------------------------------------------------------
# Orchestrator (wires everything together)
# -----------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, use_ui=True):
        self.ui = ConsoleUI() if (use_ui and CAPS["ui"]) else None
        self.bridge = Bridge(self.ui)
        self.codex = Codex(retention_hours=24)
        self.trust = TrustModel()
        self.bot = DualPersonalityBot(lambda m: self.bridge.Signal("ingest", self.ui).emit(m), self.codex, self.trust)
        self.collectors = CrossOSCollectors(self.bridge, self.codex, self.trust)
        self.running = False

    def start(self):
        self.running = True
        # Start bot and collectors
        self.bot.start()
        self.collectors.start()
        # Announce capabilities and initial state
        self.bridge.Signal("ingest", self.ui).emit(f"System online | OS={CAPS['os']} | GPU={CAPS['gpu']} | UI={self.ui is not None}")
        self.bridge.Signal("ingest", self.ui).emit(f"Codex v{self.codex.version} | Retention={self.codex.retention}")
        # Periodic governance pulse
        threading.Thread(target=self._governance_pulse, daemon=True).start()

    def _governance_pulse(self):
        while self.running:
            try:
                manifest = self.codex.export_manifest()
                self.bridge.Signal("sync", self.ui).emit(f"Codex manifest: {manifest}")
                time.sleep(30)
            except Exception as e:
                self.bridge.Signal("ingest", self.ui).emit(f"[governance] error: {e}")
                time.sleep(5)

    def switch_bot_mode(self):
        self.bot.switch_mode()

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    orch = Orchestrator(use_ui=True)
    orch.start()

    # Simple CLI loop for switching modes if no UI
    log.info("Type 'switch' to toggle bot mode; 'quit' to exit.")
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd in ("switch", "s"):
                orch.switch_bot_mode()
            elif cmd in ("quit", "q", "exit"):
                log.info("Shutting down...")
                break
            else:
                log.info("Commands: switch | quit")
    except (EOFError, KeyboardInterrupt):
        log.info("Exiting...")

if __name__ == "__main__":
    main()

