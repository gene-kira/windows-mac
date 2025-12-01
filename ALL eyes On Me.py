"""
Compact Autonomous Engine + Adaptive Codex + Operator Registry + GUI + FastAPI
------------------------------------------------------------------------------
- Continuous system-driven input (no manual submit)
- Adaptive Codex mutation logic preserved
- Operator Registry preserved
- Tkinter GUI for monitoring (always-running dashboard)
- FastAPI layer for remote access
"""

import queue, random, threading, time, uuid, socket
import tkinter as tk
from tkinter import scrolledtext, ttk
from fastapi import FastAPI
import uvicorn, psutil

# ============================================================
# Helpers
# ============================================================

def new_id(prefix): return f"{prefix}_{uuid.uuid4().hex[:8]}"
def now_ts(): return time.time()

# ============================================================
# System Monitor + Solver
# ============================================================

def system_snapshot():
    return {
        "cpu": psutil.cpu_percent(0.2),
        "mem": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
        "net": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv,
        "procs": len(psutil.pids())
    }

def system_solver(snap):
    issues, actions = [], []
    if snap["cpu"] >= 85: issues.append("High CPU"); actions.append("Reduce load")
    if snap["mem"] >= 85: issues.append("High memory"); actions.append("Close apps")
    if snap["disk"] >= 90: issues.append("Disk full"); actions.append("Free space")
    if snap["procs"] > 300: issues.append("Too many processes"); actions.append("Review services")
    if not issues: actions.append("System healthy")
    return {"issues": issues, "actions": actions}

# ============================================================
# Adaptive Codex
# ============================================================

class AdaptiveCodex:
    def __init__(self):
        self.weights = [0.6, -0.8, -0.3]
        self.rules = {"purge_logic": "default"}
        self.telemetry = 60
        self.patterns, self.phantoms = [], []

    def mutate(self):
        delta = [random.uniform(-0.1,0.1) for _ in self.weights]
        self.weights = [w+d for w,d in zip(self.weights,delta)]
        pat = random.choice(["linear","chaotic","fractal","harmonic","stochastic"])
        self.patterns.append(pat); self.rules["purge_logic"] = f"adaptive_{pat}"
        ghost = random.random() < 0.2
        if ghost:
            self.telemetry = max(10,self.telemetry-10)
            self.phantoms.append(f"phantom_{len(self.phantoms)}")
        return {"rules":self.rules.copy(),"telemetry":self.telemetry,
                "patterns":self.patterns.copy(),"phantoms":self.phantoms.copy(),
                "weights":self.weights,"ghost_sync":ghost}

CODEX = AdaptiveCodex()

# ============================================================
# Operator Registry
# ============================================================

class OperatorRegistry:
    def __init__(self): self.ops={}
    def register(self,tier,func): self.ops.setdefault(tier,[]).append(func)
    def get_all(self): return self.ops

REGISTRY = OperatorRegistry()

def op_drive(a): return {"tier":"feasible","steps":["Drive","Arrive at "+a]}
def op_fly(a): return {"tier":"feasible","steps":["Fly","Land at "+a]}
REGISTRY.register("feasible",op_drive); REGISTRY.register("feasible",op_fly)

# ============================================================
# Engine
# ============================================================

class Engine:
    def __init__(self,registry,codex):
        self.registry, self.codex = registry, codex
        self.inbox, self.outbox = queue.Queue(), queue.Queue()
        self.running, self.lock, self.workers = False, threading.Lock(), []

    def submit(self,text,meta=None): self.inbox.put({"text":text,"meta":meta or {}})

    def _process(self,payload):
        snap = system_snapshot(); codex_state = self.codex.mutate()
        bundle = {"id":new_id("bundle"),"answer":payload["text"],
                  "system":snap,"solver":system_solver(snap),"codex":codex_state}
        for tier,funcs in self.registry.get_all().items():
            for f in funcs:
                try: bundle.setdefault(tier,[]).append(f(payload["text"]))
                except Exception as e: bundle.setdefault("notes",[]).append(str(e))
        return bundle

    def _worker(self):
        while self.running:
            try: p=self.inbox.get(timeout=0.5)
            except queue.Empty: continue
            b=self._process(p); self.outbox.put(b); self.inbox.task_done()

    def start(self,n=2):
        with self.lock:
            if self.running: return
            self.running=True
        for _ in range(n):
            t=threading.Thread(target=self._worker,daemon=True); t.start(); self.workers.append(t)

    def fetch(self):
        res=[]
        while True:
            try: b=self.outbox.get_nowait(); res.append(b); self.outbox.task_done()
            except queue.Empty: break
        return res

# ============================================================
# SystemFeed
# ============================================================

class SystemFeed:
    def __init__(self,engine): self.engine,self.running=engine,False
    def _loop(self):
        while self.running:
            snap=system_snapshot()
            self.engine.submit(f"System Snapshot {time.strftime('%H:%M:%S')}",{"snapshot":snap})
            time.sleep(5)
    def start(self):
        if self.running: return
        self.running=True; threading.Thread(target=self._loop,daemon=True).start()

# ============================================================
# GUI
# ============================================================

class EngineGUI:
    def __init__(self,engine):
        self.engine=engine
        self.root=tk.Tk(); self.root.title("Engine Monitor"); self.root.geometry("700x500")
        tk.Label(self.root,text="Running",fg="green",font=("Segoe UI",12)).pack()
        self.progress=ttk.Progressbar(self.root,mode="indeterminate"); self.progress.pack(); self.progress.start(100)
        self.log=scrolledtext.ScrolledText(self.root,width=80,height=20,font=("Consolas",10)); self.log.pack()
        tk.Button(self.root,text="Clear Log",command=lambda:self.log.delete("1.0",tk.END)).pack()
        self.update_loop()

    def update_loop(self):
        for b in self.engine.fetch():
            s=b["system"]; sol=b["solver"]; codex=b["codex"]
            self.log.insert(tk.END,f"{b['id']} :: {b['answer']}\n CPU={s['cpu']}% MEM={s['mem']}% DISK={s['disk']}% PROCS={s['procs']}\n")
            if sol["issues"]: self.log.insert(tk.END,f" Issues: {', '.join(sol['issues'])}\n")
            self.log.insert(tk.END,f" Actions: {sol['actions'][0]}\n Codex purge_logic={codex['rules']['purge_logic']} ghost={codex['ghost_sync']}\n{'-'*60}\n")
            self.log.see(tk.END)
        self.root.after(1000,self.update_loop)

    def run(self): self.root.mainloop()

# ============================================================
# FastAPI
# ============================================================

app=FastAPI(); engine=Engine(REGISTRY,CODEX); feed=SystemFeed(engine)

@app.get("/results")
def results(): return engine.fetch()
@app.get("/system")
def system(): snap=system_snapshot(); return {"snapshot":snap,"solver":system_solver(snap)}

def free_port():
    s=socket.socket(); s.bind(('',0)); port=s.getsockname()[1]; s.close(); return port
def run_api():
    port=free_port(); print(f"[API] FastAPI on port {port}")
    uvicorn.run(app,host="0.0.0.0",port=port)

# ============================================================
# Main
# ============================================================

def main():
    engine.start(3); feed.start()
    threading.Thread(target=run_api,daemon=True).start()
    EngineGUI(engine).run()

if __name__=="__main__": main()

