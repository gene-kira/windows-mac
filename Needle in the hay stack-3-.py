#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified single-file, defensive-only, mixed OS detection system with Tkinter GUI.
Features:
- Autoloader with graceful fallback
- Adaptive Codex mutation (ghost sync, phantom nodes, retention control, escalate defense)
- Cross-OS collectors (process/network/filesystem)
- Trust scoring and compliance auditing
- DualPersonalityBot (guardian/rogue) for safe entropy probes and audits
- Autonomous Cipher Engine (ACE) to anonymize sensitive telemetry
- Orchestrator and ASI-style Tkinter console with buttons, status indicators, and live streaming
- CLI fallback when Tkinter is unavailable

Safety note:
- Strictly defensive. No destructive or harmful actions.
- “Rogue” mode generates harmless entropy patterns for detection calibration only.
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
np = optional_import("numpy")                     # CPU arrays (optional)
psutil = optional_import("psutil")                # cross-OS process/network (optional)
watchdog = optional_import("watchdog.observers")  # filesystem watch (optional)
watchdog_events = optional_import("watchdog.events")
tkinter = optional_import("tkinter")              # console UI (optional)
requests = optional_import("requests")            # swarm sync transport (optional)

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
# Glyphs and emblems
# -----------------------------------------------------------------------------
EMBLEMS = {
    "skull_sords": "💀⚔️",
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
}

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

def EMBLEMS_SAFE(name):
    return EMBLEMS.get(name, "📜")

# -----------------------------------------------------------------------------
# Autonomous Cipher Engine (ACE) — defensive telemetry protection
# -----------------------------------------------------------------------------
class AutonomousCipherEngine:
    """
    Purpose: anonymize sensitive telemetry fields, obfuscate identifiers,
    and apply reversible masking under governance—defensive data hygiene.
    """
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
        tag = EMBLEMS_SAFE("scroll")
        return f"{tag}:{masked.encode('latin1').hex()}"

    def anonymize_event(self, e: dict) -> dict:
        if not self.enabled or not isinstance(e, dict):
            return e
        out = dict(e)
        for k in ("username", "cmdline", "dest", "src_path"):
            if k in out and isinstance(out[k], str):
                out[k] = self.mask(out[k])
        return out

# -----------------------------------------------------------------------------
# Adaptive Codex (mutation, ghost sync, retention, escalate defense)
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
        changed = False
        if signals.get("ghost_sync"):
            self.retention = max(timedelta(hours=6), self.retention - timedelta(hours=6))
            self.phantom_nodes.add(f"phantom-{int(datetime.utcnow().timestamp())}")
            changed = True
        if signals.get("obfuscation_spike"):
            self.rules.setdefault("lolbins", {})["arg_entropy_threshold"] = 4.2
            changed = True
        if signals.get("rare_edge_spike"):
            self.rules.setdefault("rare_edges", {})["sensitivity"] = "high"
            changed = True
        if signals.get("escalate_defense"):
            self.rules.setdefault("beacon", {})["strict_mode"] = True
            self.rules.setdefault("identity", {})["step_up_auth"] = True
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
        self.version = manifest.get("version", self.version)
        self.retention = timedelta(seconds=manifest.get("retention_seconds", int(self.retention.total_seconds())))
        self.rules = manifest.get("rules", self.rules)
        self.phantom_nodes = set(manifest.get("phantom_nodes", list(self.phantom_nodes)))

# -----------------------------------------------------------------------------
# Trust model and Compliance auditor
# -----------------------------------------------------------------------------
SEVERITY = {
    "mem_inject": 9,
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
        if context.get("maintenance_window"):
            delta *= 0.3
        self.scores[entity_id] = max(0.0, self.scores[entity_id] * 0.95 + delta)
        return self.scores[entity_id]

def compliance_auditor(events, codex: Codex):
    findings = []
    for e in events:
        if e.get("age_seconds", 0) > codex.retention.total_seconds():
            findings.append("retention_violation")
    if len(events) > 10000:
        findings.append("scope_excess")
    status = "pass" if not findings else f"fail:{','.join(findings)}"
    return {"status": status, "findings": findings}

# -----------------------------------------------------------------------------
# DualPersonalityBot (guardian/rogue, defensive-only)
# -----------------------------------------------------------------------------
class DualPersonalityBot:
    def __init__(self, callback, codex: Codex, trust_model: TrustModel, ace: AutonomousCipherEngine):
        self.cb = callback
        self.run = True
        self.mode = "guardian"
        self.rogue_weights = [0.2, -0.4, 0.7]
        self.rogue_log = []
        self.codex = codex
        self.trust = trust_model
        self.ace = ace

    def switch_mode(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb(f"{EMBLEMS['triangle']} Personality switched to {self.mode.upper()}")

    def guardian_behavior(self):
        self.codex.mutate({"ghost_sync": False})
        decoy = generate_decoy()
        audit = compliance_auditor([decoy], self.codex)
        msg = f"{EMBLEMS['dove']} Guardian audit: {self.ace.anonymize_event(decoy)}"
        self.cb(msg)
        self.cb(f"{EMBLEMS['trident']} Compliance: {audit}")

    def rogue_behavior(self):
        entropy = int(time.time()) % 2048
        scrambled = reverse_mirror_encrypt(str(entropy))
        camo = camouflage(str(entropy), "alien")
        glyph_stream = random_glyph_stream()
        unusual_pattern = f"{scrambled[:16]}-{camo}-{glyph_stream[:8]}"
        self.rogue_weights = [w + (entropy % 5 - 2) * 0.01 for w in self.rogue_weights]
        self.rogue_log.append(list(self.rogue_weights))
        score = sum(self.rogue_weights) / len(self.rogue_weights)
        self.cb(f"{EMBLEMS['alchemical']} Rogue entropy probe active (defensive-only)")
        self.cb(f"{EMBLEMS['web']} Rogue pattern: {unusual_pattern}")
        self.cb(f"{EMBLEMS['monitor']} Rogue weights: {self.rogue_weights} | Trust {score:.3f}")

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
# Bridge and Signals
# -----------------------------------------------------------------------------
class Bridge:
    class Signal:
        def __init__(self, name, ui=None):
            self.name = name
            self.ui = ui
            self.logger = logging.getLogger(f"signal.{name}")
        def emit(self, msg):
            self.logger.info(msg)
            if self.ui and hasattr(self.ui, "append_line"):
                self.ui.append_line(f"[{self.name}] {msg}")

    def __init__(self, ui=None):
        self.ui = ui
        self.ingest_signal = Bridge.Signal("ingest", ui)
        self.threat_signal = Bridge.Signal("threat", ui)
        self.sync_signal = Bridge.Signal("sync", ui)

    def attach_ui(self, ui):
        self.ui = ui
        self.ingest_signal.ui = ui
        self.threat_signal.ui = ui
        self.sync_signal.ui = ui

# -----------------------------------------------------------------------------
# Collectors (cross-OS, defensive)
# -----------------------------------------------------------------------------
class CrossOSCollectors:
    def __init__(self, bridge: Bridge, codex: Codex, trust: TrustModel, ace: AutonomousCipherEngine):
        self.bridge = bridge
        self.codex = codex
        self.trust = trust
        self.ace = ace
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
                        message = f"{EMBLEMS['shield_alert']} LOLBin/Script: pid={info['pid']} name={info['name']} cmd={cmd}"
                        self.bridge.threat_signal.emit(message)
                        entity = f"proc:{info['pid']}:{info['name']}"
                        score = self.trust.update(entity, ["lolbin_misuse"], {"maintenance_window": False})
                        self.bridge.ingest_signal.emit(f"{EMBLEMS['scroll']} Trust[{entity}]={score:.2f}")
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
                        dest = f"{conn.raddr}"
                        masked = self.ace.mask(str(dest))
                        self.bridge.sync_signal.emit(f"{EMBLEMS['globe']} Outbound: {masked}")
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
                    e = {"src_path": event.src_path}
                    self.bridge.ingest_signal.emit(f"{EMBLEMS['lock']} File modified: {self.ace.anonymize_event(e)['src_path']}")

        try:
            observer = watchdog.Observer()
            root = "/" if CAPS["os"] != "Windows" else "C:\\"
            observer.schedule(Handler(), path=root, recursive=True)
            observer.start()
            while self.running:
                time.sleep(1)
        except Exception as e:
            log.error(f"[filesystem monitor] error: {e}")

# -----------------------------------------------------------------------------
# Tkinter GUI: ASI Console UI
# -----------------------------------------------------------------------------
if CAPS["ui"]:
    import tkinter as tk

class ASIConsoleUI:
    def __init__(self, orchestrator):
        if not CAPS["ui"]:
            raise RuntimeError("Tkinter not available")
        self.orch = orchestrator
        self.root = tk.Tk()
        self.root.title("🖥️ ASI Console — Live Oversight")
        self.root.geometry("1100x650")
        self.root.configure(bg="#111111")

        # Console output
        self.text = tk.Text(self.root, bg="#111111", fg="#E6E6E6", insertbackground="#E6E6E6", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True)

        # Control panel
        self.panel = tk.Frame(self.root, bg="#1a1a1a")
        self.panel.pack(fill="x")

        # Buttons
        tk.Button(self.panel, text="🔺 Switch Mode", command=self.orch.switch_bot_mode,
                  bg="#2e2e2e", fg="white").pack(side="left", padx=6, pady=6)
        tk.Button(self.panel, text="🧬 Mutate Codex", command=self.orch.mutate_again,
                  bg="#2e2e2e", fg="white").pack(side="left", padx=6)
        tk.Button(self.panel, text="🜏 Escalate Defense", command=self.orch.escalate_defense,
                  bg="#2e2e2e", fg="white").pack(side="left", padx=6)

        # Cipher toggle
        self.cipher_var = tk.BooleanVar(value=self.orch.ace.enabled)
        tk.Checkbutton(self.panel, text="🎭 Cipher Engine", variable=self.cipher_var,
                       command=lambda: self.orch.cipher_toggle(self.cipher_var.get()),
                       bg="#1a1a1a", fg="white", selectcolor="#1a1a1a").pack(side="left", padx=6)

        # Quit button
        tk.Button(self.panel, text="❌ Quit", command=self.root.quit, bg="#aa0000", fg="white").pack(side="right", padx=6)

        # Status indicators
        self.status = tk.Label(self.panel, text="", bg="#1a1a1a", fg="#00d4aa", font=("Consolas", 10))
        self.status.pack(side="right", padx=10)

        # Attach UI to bridge for streaming
        self.orch.bridge.attach_ui(self)

        # Background loops
        threading.Thread(target=self._update_status_loop, daemon=True).start()

        # Start mainloop
        self.root.mainloop()

    def append_line(self, msg):
        self.text.insert("end", msg + "\n")
        self.text.see("end")

    def _update_status_loop(self):
        while True:
            try:
                codex = self.orch.codex
                mode = self.orch.bot.mode
                ace_state = "ON" if self.orch.ace.enabled else "OFF"
                status = f"Mode: {mode.upper()} | Codex v{codex.version} | Retention: {codex.retention} | ACE: {ace_state}"
                self.status.config(text=status)
                time.sleep(3)
            except Exception as e:
                self.append_line(f"[status error] {e}")
                time.sleep(5)

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, use_ui=True):
        self.bridge = Bridge(None)  # UI attached later
        self.codex = Codex(retention_hours=24)
        self.trust = TrustModel()
        self.ace = AutonomousCipherEngine(enabled=True)
        # Bot callback routes to ingest signal
        self.bot = DualPersonalityBot(lambda m: self.bridge.ingest_signal.emit(m), self.codex, self.trust, self.ace)
        self.collectors = CrossOSCollectors(self.bridge, self.codex, self.trust, self.ace)
        self.use_ui = use_ui and CAPS["ui"]
        self.running = False

    def start(self):
        self.running = True
        self.bot.start()
        self.collectors.start()
        self.bridge.ingest_signal.emit(
            f"{EMBLEMS['dna']} {EMBLEMS['monitor']} System online | OS={CAPS['os']} | GPU={CAPS['gpu']} | UI={self.use_ui}"
        )
        self.bridge.ingest_signal.emit(
            f"{EMBLEMS['scroll']} Codex v{self.codex.version} | Retention={self.codex.retention}"
        )
        threading.Thread(target=self._governance_pulse, daemon=True).start()

    def _governance_pulse(self):
        while self.running:
            try:
                manifest = self.codex.export_manifest()
                self.bridge.sync_signal.emit(f"{EMBLEMS['trident']} Codex manifest: {manifest}")
                time.sleep(30)
            except Exception as e:
                self.bridge.ingest_signal.emit(f"[governance] error: {e}")
                time.sleep(5)

    # Commands
    def switch_bot_mode(self):
        self.bot.switch_mode()

    def mutate_again(self):
        self.codex.mutate({"ghost_sync": True, "obfuscation_spike": True})
        self.bridge.ingest_signal.emit(f"{EMBLEMS['dna']} mutate again — Codex v{self.codex.version}")

    def escalate_defense(self):
        self.codex.mutate({"escalate_defense": True})
        self.bridge.threat_signal.emit(f"{EMBLEMS['alchemical']} escalate defense — strict mode enabled")

    def cipher_toggle(self, on: bool):
        self.ace.toggle(on)
        state = "ON" if on else "OFF"
        self.bridge.ingest_signal.emit(f"{EMBLEMS['mask']} Autonomous Cipher Engine {state}")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    orch = Orchestrator(use_ui=True)
    orch.start()

    if CAPS["ui"]:
        # Launch Tkinter UI
        ASIConsoleUI(orch)
    else:
        # CLI fallback
        log.info("Tkinter not available; running CLI. Commands: switch | mutate | escalate | cipher on | cipher off | quit")
        try:
            while True:
                cmd = input("> ").strip().lower()
                if cmd in ("switch", "s"):
                    orch.switch_bot_mode()
                elif cmd in ("mutate", "m"):
                    orch.mutate_again()
                elif cmd in ("escalate", "e"):
                    orch.escalate_defense()
                elif cmd in ("cipher on", "ace on", "cipheron"):
                    orch.cipher_toggle(True)
                elif cmd in ("cipher off", "ace off", "cipheroff"):
                    orch.cipher_toggle(False)
                elif cmd in ("quit", "q", "exit"):
                    log.info("Shutting down...")
                    break
                else:
                    log.info("Commands: switch | mutate | escalate | cipher on | cipher off | quit")
        except (EOFError, KeyboardInterrupt):
            log.info("Exiting...")

if __name__ == "__main__":
    main()

