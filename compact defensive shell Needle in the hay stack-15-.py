#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defender — Compact Full Version
Includes GUI, ACE, Codex, TrustModel, Bridge; Process/Network/FS collectors.
Manual disallow control only.
"""

import os, json, time, threading, logging, shutil
from collections import deque, defaultdict
from datetime import datetime
try: import psutil
except: psutil=None
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WD=True
except: WD=False
try: import tkinter as tk
except: tk=None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("defender")

# ------------------------- MemoryVault -------------------------
class MemoryVault:
    def __init__(self, path="memory.json"):
        self.path, self.episodes = path, deque(maxlen=1500)
        if os.path.exists(path):
            try: self.episodes.extend(json.load(open(path))["episodes"])
            except: pass
    def save(self): json.dump({"episodes":list(self.episodes)}, open(self.path,"w"), indent=2)
    def add(self, src, msg, tags=None, payload=None):
        self.episodes.append({"ts":datetime.utcnow().isoformat(), "src":src, "msg":msg, "tags":tags or [], "payload":payload})

# --------------------------- ACE -------------------------------
class ACE:
    def __init__(self, key=37): self.key=key
    def mask(self, s): return "".join(chr(ord(c)^self.key) for c in s) if s else s
    def sanitize(self, d): return {k:(self.mask(v) if isinstance(v,str) else v) for k,v in (d or {}).items()}

# ---------------------- Codex & Trust --------------------------
SEVERITY = {"lolbin":6, "beacon":4, "fs":3}
class Codex:
    def __init__(self): self.version=1; self.strict=False
    def mutate(self, escalate=False):
        if escalate: self.strict=True; self.version+=1
class TrustModel:
    def __init__(self): self.scores=defaultdict(float)
    def update(self, eid, signals): self.scores[eid]+=sum(SEVERITY.get(s,1) for s in signals); return self.scores[eid]

# -------------------------- Bridge -----------------------------
class Bridge:
    def __init__(self, ui=None, memory=None): self.ui, self.mem = ui, memory
    def emit(self, chan, msg, tags=None, payload=None):
        log.info(f"[{chan}] {msg}")
        if self.ui: self.ui.append_line(f"[{chan}] {msg}")
        if self.mem: self.mem.add(chan, msg, tags, payload)

# ------------------------ ListManager --------------------------
class ListManager:
    def __init__(self, mem):
        self.mem = mem
        self.allow = set()
        self.disallow = set()

    def is_allowed(self, eid): return eid in self.allow
    def allow_eid(self, eid):
        self.disallow.discard(eid); self.allow.add(eid)
        self.mem.add("lists", f"Allow {eid}")
    def disallow_eid(self, eid):
        self.allow.discard(eid); self.disallow.add(eid)
        self.mem.add("lists", f"Disallow {eid}")

# ------------------------- Responder ---------------------------
class Responder:
    def __init__(self, bridge, mem, ace, lists):
        self.b, self.m, self.a, self.l = bridge, mem, ace, lists
        self.qdir = os.path.join(os.getcwd(), "quarantine"); os.makedirs(self.qdir, exist_ok=True)
        self.kill_on_lolbin, self.throttle_beacon = True, True
    def kill(self, pid, eid=None):
        if not psutil or (eid and self.l.is_allowed(eid)): return
        try:
            p = psutil.Process(pid)
            for c in p.children(recursive=True): 
                try: c.terminate()
                except: pass
            p.terminate()
            self.b.emit("resp", f"Killed PID={pid}", ["proc"], {"pid":pid})
        except Exception as e: self.b.emit("resp", f"Kill error {e}", ["error"])
    def quarantine(self, path, eid=None):
        if eid and self.l.is_allowed(eid): return
        if not os.path.isfile(path): return
        dest = os.path.join(self.qdir, f"{int(time.time())}_{os.path.basename(path)}")
        try:
            shutil.copy2(path, dest)
            try: os.remove(path)
            except: pass
            self.b.emit("resp", f"Quarantine {path} -> {dest}", ["file"], {"src":path, "dest":dest})
        except Exception as e: self.b.emit("resp", f"Quarantine error {e}", ["error"])
    def throttle(self, dest, eid=None):
        if eid and self.l.is_allowed(eid): return
        self.b.emit("resp", f"Egress throttle {dest}", ["net"], {"dest":dest})

# ------------------------- Collectors --------------------------
class Collectors:
    def __init__(self, bridge, trust, ace, resp, lists):
        self.b, self.t, self.a, self.r, self.l = bridge, trust, ace, resp, lists
        self.running=True
    def start(self):
        if psutil: threading.Thread(target=self.process_mon, daemon=True).start()
        if psutil: threading.Thread(target=self.net_mon, daemon=True).start()
        if WD: threading.Thread(target=self.fs_mon, daemon=True).start()

    def process_mon(self):
        lolbins = ("powershell","rundll32","mshta","certutil","wmic","wscript","cscript")
        while self.running:
            try:
                for p in psutil.process_iter(attrs=["pid","name","cmdline","exe"]):
                    pid = p.info["pid"]; name = (p.info.get("name") or "").lower()
                    cmd = " ".join(p.info.get("cmdline") or []).lower()
                    eid = f"proc:{pid}"
                    if any(x in name or x in cmd for x in lolbins):
                        self.t.update(eid, ["lolbin"])
                        self.b.emit("threat", f"LOLBin pid={pid} name={name}", ["proc"], self.a.sanitize(p.info))
                        if self.r.kill_on_lolbin and not self.l.is_allowed(eid): self.r.kill(pid, eid=eid)
                time.sleep(5)
            except Exception as e:
                self.b.emit("ingest", f"proc error {e}", ["error"]); time.sleep(2)

    def net_mon(self):
        while self.running:
            try:
                for c in psutil.net_connections(kind="inet"):
                    if c.status == "ESTABLISHED" and c.raddr:
                        dest = f"{getattr(c.raddr,'ip',None) or c.raddr[0]}:{getattr(c.raddr,'port',None) or c.raddr[1]}"
                        eid = f"net:{dest}"
                        self.t.update(eid, ["beacon"])
                        self.b.emit("sync", f"Outbound {self.a.mask(dest)}", ["net"], {"dest":dest, "status":c.status})
                        if self.r.throttle_beacon and not self.l.is_allowed(eid): self.r.throttle(dest, eid=eid)
                time.sleep(10)
            except Exception as e:
                self.b.emit("ingest", f"net error {e}", ["error"]); time.sleep(2)

    def fs_mon(self):
        class H(FileSystemEventHandler):
            def on_modified(_, event):
                try:
                    path = os.path.abspath(getattr(event, "src_path", ""))
                    if not path or os.path.isdir(path): return
                    eid = f"file:{path}"
                    self.t.update(eid, ["fs"])
                    self.b.emit("ingest", f"File modified {self.a.mask(path)}", ["fs"], {"path":path})
                except Exception as ex:
                    self.b.emit("ingest", f"fs handler {ex}", ["error"])
        try:
            root = os.path.abspath(os.sep)
            obs = Observer(); obs.schedule(H(), root, recursive=True); obs.start()
            while self.running: time.sleep(1)
        except Exception as e:
            self.b.emit("ingest", f"fs error {e}", ["error"])

# ----------------------------- GUI -----------------------------
class DefenderUI:
    def __init__(self, orch):
        self.orch=orch; self.root=tk.Tk(); self.root.title("Defender")
        self.text=tk.Text(self.root, bg="#111", fg="#eee"); self.text.pack(fill="both", expand=True)
        self.orch.bridge.ui=self
        threading.Thread(target=self.status_loop, daemon=True).start()
        self.root.mainloop()
    def append_line(self, msg): self.text.insert("end", msg+"\n"); self.text.see("end")
    def status_loop(self):
        while True:
            a=len(self.orch.lists)
def status_loop(self):
        while True:
            a = len(self.orch.lists.allow)
            d = len(self.orch.lists.disallow)
            self.append_line(f"[status] allow={a} disallow={d}")
            time.sleep(8)

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
        self.use_ui = bool(use_ui and tk)

    def start(self):
        self.collectors.start()
        self.bridge.emit("system", "Defender online")
        if self.use_ui:
            DefenderUI(self)

# ------------------------------ Entry --------------------------
def main():
    Orchestrator(use_ui=True).start()

if __name__ == "__main__":
    main()


    