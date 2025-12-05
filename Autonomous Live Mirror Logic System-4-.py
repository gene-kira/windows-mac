#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defender — Compact Full Version
Includes GUI, ACE, Codex, TrustModel, Bridge; Process/Network/FS collectors.
Manual disallow control only.

Additions:
- MirrorDefense: rule-based layer driven by mirror analysis (oscillation/synthesis/dominance/void).
- MirrorHook: minimal interface to receive analysis dicts from your mirror engine and invoke defense.
- Fixed GUI status loop and earlier cut-off issues.
"""

import os, json, time, threading, logging, shutil
from collections import deque, defaultdict
from datetime import datetime

# Optional deps
try:
    import psutil
except Exception:
    psutil = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WD = True
except Exception:
    WD = False

try:
    import tkinter as tk
except Exception:
    tk = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("defender")

# ------------------------- MemoryVault -------------------------
class MemoryVault:
    def __init__(self, path="memory.json"):
        self.path = path
        self.episodes = deque(maxlen=1500)
        if os.path.exists(path):
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
                self.episodes.extend(data.get("episodes", []))
            except Exception:
                pass

    def save(self):
        try:
            json.dump({"episodes": list(self.episodes)}, open(self.path, "w", encoding="utf-8"), indent=2)
        except Exception as e:
            log.warning(f"MemoryVault save error: {e}")

    def add(self, src, msg, tags=None, payload=None):
        self.episodes.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "src": src,
            "msg": msg,
            "tags": tags or [],
            "payload": payload
        })

# --------------------------- ACE -------------------------------
class ACE:
    def __init__(self, key=37):
        self.key = key

    def mask(self, s):
        return "".join(chr(ord(c) ^ self.key) for c in s) if s else s

    def sanitize(self, d):
        return {k: (self.mask(v) if isinstance(v, str) else v) for k, v in (d or {}).items()}

# ---------------------- Codex & Trust --------------------------
SEVERITY = {"lolbin": 6, "beacon": 4, "fs": 3}

class Codex:
    def __init__(self):
        self.version = 1
        self.strict = False

    def mutate(self, escalate=False):
        if escalate:
            self.strict = True
            self.version += 1

class TrustModel:
    def __init__(self):
        self.scores = defaultdict(float)

    def update(self, eid, signals):
        self.scores[eid] += sum(SEVERITY.get(s, 1) for s in signals)
        return self.scores[eid]

# -------------------------- Bridge -----------------------------
class Bridge:
    def __init__(self, ui=None, memory=None):
        self.ui = ui
        self.mem = memory

    def emit(self, chan, msg, tags=None, payload=None):
        log.info(f"[{chan}] {msg}")
        if self.ui:
            try:
                self.ui.append_line(f"[{chan}] {msg}")
            except Exception:
                pass
        if self.mem:
            try:
                self.mem.add(chan, msg, tags, payload)
            except Exception:
                pass

# ------------------------ ListManager --------------------------
class ListManager:
    def __init__(self, mem):
        self.mem = mem
        self.allow = set()
        self.disallow = set()

    def is_allowed(self, eid):
        return eid in self.allow

    def allow_eid(self, eid):
        self.disallow.discard(eid)
        self.allow.add(eid)
        self.mem.add("lists", f"Allow {eid}")

    def disallow_eid(self, eid):
        self.allow.discard(eid)
        self.disallow.add(eid)
        self.mem.add("lists", f"Disallow {eid}")

# ------------------------- Responder ---------------------------
class Responder:
    def __init__(self, bridge, mem, ace, lists):
        self.b = bridge
        self.m = mem
        self.a = ace
        self.l = lists
        self.qdir = os.path.join(os.getcwd(), "quarantine")
        os.makedirs(self.qdir, exist_ok=True)
        self.kill_on_lolbin = True
        self.throttle_beacon = True

    def kill(self, pid, eid=None):
        if not psutil or (eid and self.l.is_allowed(eid)):
            return
        try:
            p = psutil.Process(pid)
            for c in p.children(recursive=True):
                try:
                    c.terminate()
                except Exception:
                    pass
            p.terminate()
            self.b.emit("resp", f"Killed PID={pid}", ["proc"], {"pid": pid})
        except Exception as e:
            self.b.emit("resp", f"Kill error {e}", ["error"])

    def quarantine(self, path, eid=None):
        if eid and self.l.is_allowed(eid):
            return
        if not os.path.isfile(path):
            return
        dest = os.path.join(self.qdir, f"{int(time.time())}_{os.path.basename(path)}")
        try:
            shutil.copy2(path, dest)
            try:
                os.remove(path)
            except Exception:
                pass
            self.b.emit("resp", f"Quarantine {path} -> {dest}", ["file"], {"src": path, "dest": dest})
        except Exception as e:
            self.b.emit("resp", f"Quarantine error {e}", ["error"])

    def throttle(self, dest, eid=None):
        if eid and self.l.is_allowed(eid):
            return
        self.b.emit("resp", f"Egress throttle {dest}", ["net"], {"dest": dest})

# ------------------------- Collectors --------------------------
class Collectors:
    def __init__(self, bridge, trust, ace, resp, lists):
        self.b = bridge
        self.t = trust
        self.a = ace
        self.r = resp
        self.l = lists
        self.running = True

    def start(self):
        if psutil:
            threading.Thread(target=self.process_mon, daemon=True).start()
            threading.Thread(target=self.net_mon, daemon=True).start()
        if WD:
            threading.Thread(target=self.fs_mon, daemon=True).start()

    def process_mon(self):
        lolbins = ("powershell", "rundll32", "mshta", "certutil", "wmic", "wscript", "cscript")
        while self.running:
            try:
                for p in psutil.process_iter(attrs=["pid", "name", "cmdline", "exe"]):
                    pid = p.info["pid"]
                    name = (p.info.get("name") or "").lower()
                    cmd = " ".join(p.info.get("cmdline") or []).lower()
                    eid = f"proc:{pid}"
                    if any(x in name or x in cmd for x in lolbins):
                        self.t.update(eid, ["lolbin"])
                        self.b.emit("threat", f"LOLBin pid={pid} name={name}", ["proc"], self.a.sanitize(p.info))
                        if self.r.kill_on_lolbin and not self.l.is_allowed(eid):
                            self.r.kill(pid, eid=eid)
                time.sleep(5)
            except Exception as e:
                self.b.emit("ingest", f"proc error {e}", ["error"])
                time.sleep(2)

    def net_mon(self):
        while self.running:
            try:
                for c in psutil.net_connections(kind="inet"):
                    if c.status == "ESTABLISHED" and c.raddr:
                        dest = f"{getattr(c.raddr, 'ip', None) or c.raddr[0]}:{getattr(c.raddr, 'port', None) or c.raddr[1]}"
                        eid = f"net:{dest}"
                        self.t.update(eid, ["beacon"])
                        self.b.emit("sync", f"Outbound {self.a.mask(dest)}", ["net"], {"dest": dest, "status": c.status})
                        if self.r.throttle_beacon and not self.l.is_allowed(eid):
                            self.r.throttle(dest, eid=eid)
                time.sleep(10)
            except Exception as e:
                self.b.emit("ingest", f"net error {e}", ["error"])
                time.sleep(2)

    def fs_mon(self):
        class H(FileSystemEventHandler):
            def on_modified(_, event):
                try:
                    path = os.path.abspath(getattr(event, "src_path", ""))
                    if not path or os.path.isdir(path):
                        return
                    eid = f"file:{path}"
                    self.t.update(eid, ["fs"])
                    self.b.emit("ingest", f"File modified {self.a.mask(path)}", ["fs"], {"path": path})
                except Exception as ex:
                    self.b.emit("ingest", f"fs handler {ex}", ["error"])

        try:
            root = os.path.abspath(os.sep)
            obs = Observer()
            obs.schedule(H(), root, recursive=True)
            obs.start()
            while self.running:
                time.sleep(1)
        except Exception as e:
            self.b.emit("ingest", f"fs error {e}", ["error"])

# ---------------------- MirrorDefense --------------------------
class MirrorDefense:
    """
    Rule-based defense driven by mirror analysis dicts:
    analysis = {
        'positives': int, 'negatives': int, 'voids': int, 'unity': int,
        'status': str, 'entropy': 'low'|'high'
    }
    """
    def __init__(self, bridge, responder, trust, lists, threshold=50):
        self.bridge = bridge
        self.responder = responder
        self.trust = trust
        self.lists = lists
        self.threshold = threshold

    def evaluate(self, analysis):
        status = analysis.get("status", "")
        pos = analysis.get("positives", 0)
        neg = analysis.get("negatives", 0)
        voids = analysis.get("voids", 0)
        unity = analysis.get("unity", 0)

        # Oscillation rule: large active polarity count + oscillation signal
        if "oscillating" in status and (pos + neg) > self.threshold:
            self.bridge.emit("mirror", "Oscillation threshold exceeded — auto-quarantine engaged", ["mirror"])
            # Placeholder action: quarantine a known temp file if present (non-destructive)
            suspect = os.path.join(os.getcwd(), "suspicious.tmp")
            if os.path.isfile(suspect):
                self.responder.quarantine(suspect)

        # Synthesis rule: unity present indicates coordinated behavior -> alert and escalate trust
        if unity > 0:
            self.bridge.emit("mirror", "Synthesis detected — alert admin", ["mirror"])
            self.trust.update("mirror:unity", ["beacon"])

        # Dominance rules: escalate trust for prolonged imbalance
        if status == "positive dominance":
            self.bridge.emit("mirror", "Positive dominance — escalate trust score", ["mirror"])
            self.trust.update("mirror:posdom", ["fs"])
        elif status == "negative dominance":
            self.bridge.emit("mirror", "Negative dominance — escalate trust score", ["mirror"])
            self.trust.update("mirror:negdom", ["fs"])

        # Void equilibrium: watch for covert/silent channels
        if status == "void equilibrium" and voids > self.threshold // 2:
            self.bridge.emit("mirror", "Void equilibrium — possible covert channel", ["mirror"])

# ------------------------ MirrorHook ---------------------------
class MirrorHook:
    """
    Minimal interface to accept mirror analysis dicts and route to defense.
    Use this from your mirror engine: hook.submit(analysis)
    """
    def __init__(self, defense: MirrorDefense, bridge: Bridge):
        self.defense = defense
        self.bridge = bridge
        self.enabled = True

    def submit(self, analysis: dict):
        if not self.enabled or not isinstance(analysis, dict):
            return
        # Emit short trace
        self.bridge.emit("mirror", f"analysis status={analysis.get('status','unknown')}", ["trace"])
        try:
            self.defense.evaluate(analysis)
        except Exception as e:
            self.bridge.emit("mirror", f"defense error {e}", ["error"])

# ----------------------------- GUI -----------------------------
class DefenderUI:
    def __init__(self, orch):
        self.orch = orch
        self.root = tk.Tk()
        self.root.title("Defender")
        self.text = tk.Text(self.root, bg="#111", fg="#eee")
        self.text.pack(fill="both", expand=True)
        self.orch.bridge.ui = self

        # Start status loop
        threading.Thread(target=self.status_loop, daemon=True).start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def append_line(self, msg):
        try:
            self.text.insert("end", msg + "\n")
            self.text.see("end")
        except Exception:
            pass

    def status_loop(self):
        while True:
            try:
                a = len(self.orch.lists.allow)
                d = len(self.orch.lists.disallow)
                codex_strict = getattr(self.orch.codex, "strict", False)
                self.append_line(f"[status] allow={a} disallow={d} | codex.strict={codex_strict}")
            except Exception as e:
                self.append_line(f"[status] error {e}")
            time.sleep(8)

    def on_close(self):
        try:
            self.orch.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

# -------------------------- Orchestrator -----------------------
class Orchestrator:
    def __init__(self, use_ui=True):
        self.memory = MemoryVault()
        self.ace = ACE()
        self.codex = Codex()
        self.trust = TrustModel()
        self.lists = ListManager(self.memory)
        self.bridge = Bridge(memory=self.memory)
        self.responder = Responder(self.bridge, self.memory, self.ace, self.lists)
        self.collectors = Collectors(self.bridge, self.trust, self.ace, self.responder, self.lists)

        # MirrorDefense and Hook
        self.mirror_defense = MirrorDefense(self.bridge, self.responder, self.trust, self.lists, threshold=50)
        self.mirror_hook = MirrorHook(self.mirror_defense, self.bridge)

        self.use_ui = bool(use_ui and tk)
        self._running = False

    def start(self):
        self._running = True
        self.collectors.start()
        self.bridge.emit("system", "Defender online")

        # Example: periodic codex strict mode escalation if trust scores exceed threshold
        threading.Thread(target=self.policy_loop, daemon=True).start()

        if self.use_ui:
            DefenderUI(self)

    def stop(self):
        self._running = False
        try:
            self.memory.save()
        except Exception:
            pass
        self.bridge.emit("system", "Defender offline")

    def policy_loop(self):
        """
        Simple governance loop: if aggregate trust exceeds threshold, tighten codex mode.
        """
        while self._running:
            try:
                total_trust = sum(self.trust.scores.values()) if self.trust.scores else 0
                if total_trust > 100 and not self.codex.strict:
                    self.codex.mutate(escalate=True)
                    self.bridge.emit("policy", "Codex escalated to strict due to trust totals", ["policy"], {"total_trust": total_trust})
            except Exception as e:
                self.bridge.emit("policy", f"loop error {e}", ["error"])
            time.sleep(15)

    def get_mirror_hook(self) -> MirrorHook:
        """
        External interface: your mirror engine calls hook.submit(analysis_dict)
        """
        return self.mirror_hook

# ------------------------------ Entry --------------------------
def main():
    orch = Orchestrator(use_ui=True)
    orch.start()

    # Example: feed synthetic analyses (remove this when connecting your real mirror engine)
    # This demonstrates MirrorDefense reacting to oscillation and synthesis.
    def demo_feed():
        # Push a few analyses to show the defense rules
        samples = [
            {"positives": 60, "negatives": 60, "voids": 0, "unity": 0, "status": "oscillating", "entropy": "high"},
            {"positives": 10, "negatives": 10, "voids": 80, "unity": 0, "status": "void equilibrium", "entropy": "low"},
            {"positives": 5, "negatives": 4, "voids": 1, "unity": 3, "status": "synthesis present", "entropy": "high"},
            {"positives": 100, "negatives": 0, "voids": 0, "unity": 0, "status": "positive dominance", "entropy": "low"},
        ]
        hook = orch.get_mirror_hook()
        for s in samples:
            hook.submit(s)
            time.sleep(5)

    threading.Thread(target=demo_feed, daemon=True).start()

if __name__ == "__main__":
    main()

