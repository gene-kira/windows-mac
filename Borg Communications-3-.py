import os, sys, time, json, shutil, socket, threading, logging, importlib
import tkinter as tk
from tkinter import ttk
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# GPU/CPU fallback
try:
    import cupy as cp; gpu=True
except ImportError:
    import numpy as cp; gpu=False

# Config
LOCAL_CACHE = r"C:\CodexCache"; os.makedirs(LOCAL_CACHE, exist_ok=True)
NETWORK_CACHE=None; LOG_FILE=os.path.join(LOCAL_CACHE,"codex.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# Self-Rewriting Agent
agent_weights=cp.array([0.6,-0.8,-0.3]); mutation_log=[]
def gpu_trust_score(v):
    s=cp.dot(cp.array(v),agent_weights)
    return float(s.get()) if gpu else float(s)
def mutate(entropy):
    global agent_weights
    agent_weights+=cp.array([entropy*0.01,-entropy*0.005,entropy*0.002])
    mutation_log.append(agent_weights.get().tolist() if gpu else agent_weights.tolist())

# Cipher Engine
class CipherEngine:
    modes=["🎭","🕸️","🔒","🧬","🌐"]
    def __init__(self,cb): self.cb=cb; self.run=True
    def evolve(self):
        e=int(time.time())%1024; mutate(e)
        score=gpu_trust_score([0.5,-0.2,0.1])
        self.cb(f"{self.modes[e%5]} entropy {e} | Trust {score:.3f}")
    def start(self): threading.Thread(target=self.loop,daemon=True).start()
    def loop(self):
        while self.run: self.evolve(); time.sleep(30)

# Watchdog
class Watchdog:
    def __init__(self,cb): self.cb=cb; self.run=True
    def sync(self):
        if NETWORK_CACHE:
            os.makedirs(NETWORK_CACHE,exist_ok=True)
            for r,_,fs in os.walk(LOCAL_CACHE):
                t=os.path.join(NETWORK_CACHE,os.path.relpath(r,LOCAL_CACHE))
                os.makedirs(t,exist_ok=True)
                [shutil.copy2(os.path.join(r,f),os.path.join(t,f)) for f in fs]
            self.cb("synced")
        else: self.cb("failover")
    def start(self): threading.Thread(target=self.loop,daemon=True).start()
    def loop(self):
        while self.run: self.sync(); time.sleep(30)

# DevourerDaemon
class DevourerDaemon:
    def __init__(self,cb): self.cb=cb; self.run=True
    def start(self):
        threading.Thread(target=self.fs,daemon=True).start()
        threading.Thread(target=self.net,daemon=True).start()
    def fs(self):
        class H(FileSystemEventHandler):
            def on_modified(_,e): 
                if not e.is_directory: self.cb(f"File modified {e.src_path}")
        o=Observer(); o.schedule(H(),path='/',recursive=True); o.start()
    def net(self):
        while self.run:
            for c in psutil.net_connections(kind='inet'):
                if c.status=="ESTABLISHED": self.cb(f"Conn {c.raddr}")
            time.sleep(10)

# GUI
class GUI(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Codex Purge Shell"); self.geometry("800x600")
        self.status=ttk.Label(self,text="Init"); self.status.pack()
        drives=[p.mountpoint for p in psutil.disk_partitions()]
        self.var=tk.StringVar(); ttk.Combobox(self,textvariable=self.var,values=drives).pack()
        ttk.Button(self,text="Set Drive",command=self.set_drive).pack()
        self.text=tk.Text(self,height=20); self.text.pack(fill="both")
        self.wd=self.dev=self.ce=None
    def set_drive(self):
        global NETWORK_CACHE; NETWORK_CACHE=self.var.get(); os.makedirs(NETWORK_CACHE,exist_ok=True)
        self.log(f"Drive set {NETWORK_CACHE}")
        if not self.wd: self.wd=Watchdog(self.update); self.wd.start()
        if not self.dev: self.dev=DevourerDaemon(self.log); self.dev.start()
        if not self.ce: self.ce=CipherEngine(self.log); self.ce.start()
    def update(self,msg): self.after(0,lambda:self.log(msg))
    def log(self,msg): self.text.insert(tk.END,msg+"\n"); self.text.see(tk.END)

if __name__=="__main__": GUI().mainloop()

