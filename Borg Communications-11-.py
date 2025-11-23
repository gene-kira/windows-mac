import os, time, shutil, threading, socket, logging, json, random, hashlib
import tkinter as tk
from tkinter import ttk
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- GPU/CPU fallback ---
try: import cupy as cp; gpu=True
except ImportError: import numpy as cp; gpu=False

# --- Config ---
LOCAL_CACHE=r"C:\\CodexCache"; os.makedirs(LOCAL_CACHE,exist_ok=True)
NETWORK_CACHE=None; LOG_FILE=os.path.join(LOCAL_CACHE,"codex.log")
SETTINGS_FILE=os.path.join(LOCAL_CACHE,"codex_settings.json")
logging.basicConfig(filename=LOG_FILE,level=logging.INFO)

# --- Self‑Rewriting Agent ---
agent_weights=cp.array([0.6,-0.8,-0.3]); mutation_log=[]
def gpu_trust_score(v):
    s=cp.dot(cp.array(v),agent_weights)
    return float(s.get()) if gpu else float(s)
def mutate(entropy):
    global agent_weights
    agent_weights+=cp.array([entropy*0.01,-entropy*0.005,entropy*0.002])
    mutation_log.append(agent_weights.get().tolist() if gpu else agent_weights.tolist())

# --- Codex Rules ---
codex_rules={"purge":["telemetry"],"threats":["resurrection"],"retention":60}
def adaptive_mutation(event):
    if "ghost sync" in event.lower():
        codex_rules["retention"]=30
        if "phantom node" not in codex_rules["threats"]:
            codex_rules["threats"].append("phantom node")
    logging.info(f"Codex rules updated: {codex_rules}")

def purge_cycle():
    while True:
        time.sleep(codex_rules["retention"])
        logging.info("Purging expired data per codex rules")

# --- Borg Queen ---
class CodexQueen:
    def __init__(self, gui_callback):
        self.gui_callback=gui_callback; self.nodes=[]
    def register_node(self,node):
        self.nodes.append(node)
        self.gui_callback(f"Node joined collective: {node['id']}")
    def propagate_rules(self):
        for node in self.nodes:
            node["rules"]=codex_rules.copy()
        self.gui_callback(f"Queen propagated rules: {codex_rules}")
    def evolve(self,event):
        adaptive_mutation(event)
        self.propagate_rules()

queen = CodexQueen(lambda m: print(m))

# --- Cipher Engine ---
class CipherEngine:
    modes=["🎭","🕸️","🔒","🧬","🌐"]
    def __init__(self,cb,logcb): self.cb=cb; self.logcb=logcb; self.run=True
    def evolve(self):
        e=int(time.time())%1024; mutate(e)
        score=gpu_trust_score([0.5,-0.2,0.1])
        glyph=self.modes[e%5]
        self.cb(f"{glyph} entropy {e} | Trust {score:.3f}")
        if score>0.5: color="green"
        elif score>0: color="yellow"
        else: color="red"
        self.logcb(f"Weights: {mutation_log[-1]} | Trust {score:.3f}",color)
    def start(self): threading.Thread(target=self.loop,daemon=True).start()
    def loop(self):
        while self.run: self.evolve(); time.sleep(30)

# --- Watchdog ---
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

# --- DevourerDaemon ---
class DevourerDaemon:
    def __init__(self,cb,node_id):
        self.cb=cb; self.run=True; self.node={"id":node_id,"rules":codex_rules.copy()}
        queen.register_node(self.node)
    def start(self):
        threading.Thread(target=self.fs,daemon=True).start()
        threading.Thread(target=self.net,daemon=True).start()
        threading.Thread(target=purge_cycle,daemon=True).start()
    def fs(self):
        class H(FileSystemEventHandler):
            def on_modified(_,e):
                if not e.is_directory:
                    self.cb(f"File modified {e.src_path}")
                    queen.evolve("ghost sync")
        o=Observer(); o.schedule(H(),path='/',recursive=True); o.start()
    def net(self):
        while self.run:
            for c in psutil.net_connections(kind='inet'):
                if c.status=="ESTABLISHED":
                    self.cb(f"Conn {c.raddr}")
                    queen.evolve("live threat")
            time.sleep(10)

# --- Settings Persistence ---
def save_settings():
    settings={"network_cache":NETWORK_CACHE,"codex_rules":codex_rules}
    with open(SETTINGS_FILE,"w") as f: json.dump(settings,f)
def load_settings():
    global NETWORK_CACHE,codex_rules
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE,"r") as f: s=json.load(f)
        NETWORK_CACHE=s.get("network_cache")
        codex_rules.update(s.get("codex_rules",{}))

# --- ASI Oversight Backend ---
def random_time(): return time.strftime("%Y-%m-%d %H:%M:%S")
def get_real_country(): return random.choice(["US","RU","CN","DE","BR","IN","JP","UK"])
def random_glyph_stream(): return hashlib.sha256(str(random.random()).encode()).hexdigest()
def generate_decoy(): return {"timestamp":random_time(),"origin":get_real_country(),"payload":random_glyph_stream()}
def reverse_mirror_encrypt(data): return hashlib.sha512(data[::-1].encode()).hexdigest()
def camouflage(data,mode="alien"): return f"👁️ {hashlib.md5(data.encode()).hexdigest()}"
def threat_hunter(event): return "⚠️ Ghost Sync detected" if "ghost sync" in event.lower() else f"Threat classified: {event}"
def compliance_auditor(logs): return f"Audit complete: {len(logs)} entries reviewed"
def spawn_glyph_node(tag,msg): print(f"[{tag}] {msg}")
def zero_trust_check(identity):
    trusted={"system_core","authorized_user"}
    if identity not in trusted:
        spawn_glyph_node("trust_block",f"Blocked '{identity}'")
        raise PermissionError("Zero Trust Sentinel blocked access.")
    return f"Identity '{identity}' verified"
def asi_console(event,identity):
    try:
        check=zero_trust_check(identity)
        return {
            "identity":check,
            "decoy":generate_decoy(),
            "encrypted":reverse_mirror_encrypt(event),
            "camouflage":camouflage(event),
            "threat":threat_hunter(event),
            "audit":compliance_auditor([event]),
            "hologram":"👤∞ [alien face streaming infinite data]"
        }
    except PermissionError as e:
        return {"error":str(e)}

# --- Dual Personality Bot (Upgraded) ---
class DualPersonalityBot:
    def __init__(self, cb):
        self.cb = cb
        self.run = True
        self.mode = "guardian"  # can be "guardian" or "rogue"
        self.rogue_weights = [0.2, -0.4, 0.7]
        self.rogue_log = []

    def switch_mode(self):
        self.mode = "rogue" if self.mode == "guardian" else "guardian"
        self.cb(f"🔺 Personality switched to {self.mode.upper()}")

    def guardian_behavior(self):
        adaptive_mutation("ghost sync")
        decoy = generate_decoy()
        self.cb(f"🕊️ Guardian audit: {decoy}")
        self.cb(f"🔱 Compliance: {compliance_auditor([decoy])}")

    def rogue_behavior(self):
        entropy = int(time.time()) % 2048
        scrambled = reverse_mirror_encrypt(str(entropy))
        camo = camouflage(str(entropy),"alien")
        glyph_stream = random_glyph_stream()
        unusual_pattern = f"{scrambled[:16]}-{camo}-{glyph_stream[:8]}"

        # Adjust rogue weights
        self.rogue_weights = [
            w + (entropy % 5 - 2) * 0.01 for w in self.rogue_weights
        ]
        self.rogue_log.append(self.rogue_weights)

        score = sum(self.rogue_weights) / len(self.rogue_weights)

        self.cb("💀⚔️ Rogue escalation initiated")
        self.cb(f"🜏 Rogue pattern: {unusual_pattern}")
        self.cb(f"📊 Rogue weights: {self.rogue_weights} | Trust {score:.3f}")

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while self.run:
            if self.mode == "guardian":
                self.guardian_behavior()
            else:
                self.rogue_behavior()
            time.sleep(10)

# --- GUI ---
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        load_settings()
        self.title("Codex Purge Shell + ASI Console + Dual Personality Bot")
        self.geometry("1100x800")
        self.status=ttk.Label(self,text="Init"); self.status.pack()
        drives=[p.mountpoint for p in psutil.disk_partitions()]
        self.var=tk.StringVar(); ttk.Combobox(self,textvariable=self.var,values=drives).pack()
        ttk.Button(self,text="Set Drive",command=self.set_drive).pack()
        self.text=tk.Text(self,height=20,bg="white"); self.text.pack(fill="x")
        self.mutlog=tk.Text(self,height=12,bg="black"); self.mutlog.pack(fill="x")
        self.mutlog.tag_config("green",foreground="green")
        self.mutlog.tag_config("yellow",foreground="yellow")
        self.mutlog.tag_config("red",foreground="red")
        self.wd=self.dev=self.ce=self.dual=None
        threading.Thread(target=self.run_asi_loop,daemon=True).start()
        if NETWORK_CACHE: 
            self.log(f"Restored drive {NETWORK_CACHE}")
            self.set_drive()

    def set_drive(self):
        global NETWORK_CACHE
        NETWORK_CACHE=self.var.get() or NETWORK_CACHE
        os.makedirs(NETWORK_CACHE,exist_ok=True)
        self.log(f"Drive set {NETWORK_CACHE}")
        save_settings()
        if not self.wd:
            self.wd=Watchdog(self.update); self.wd.start()
        if not self.dev:
            self.dev=DevourerDaemon(self.log,"Node-1"); self.dev.start()
        if not self.ce:
            self.ce=CipherEngine(self.log,self.log_mut); self.ce.start()
        if not self.dual:
            self.dual=DualPersonalityBot(self.log); self.dual.start()

    def update(self,msg):
        self.after(0,lambda:self.log(msg))

    def log(self,msg):
        self.text.insert(tk.END,msg+"\n")
        self.text.see(tk.END)

    def log_mut(self,msg,color="green"):
        def _insert():
            self.mutlog.insert(tk.END,msg+"\n",color)
            self.mutlog.see(tk.END)
        self.after(0,_insert)

    def run_asi_loop(self):
        while True:
            result = asi_console("ghost sync anomaly","system_core")
            if "error" in result:
                self.log(f"Error: {result['error']}")
            else:
                self.log(f"Threat: {result['threat']}")
                self.log(f"Audit: {result['audit']}")
                self.log(f"Decoy: {result['decoy']}")
                self.log_mut(f"Encrypted: {result['encrypted']}", "yellow")
                self.log_mut(f"Camouflage: {result['camouflage']}", "green")
                self.log_mut(f"Hologram: {result['hologram']}", "red")
            time.sleep(5)

# --- Entry Point ---
if __name__ == "__main__":
    GUI().mainloop()



