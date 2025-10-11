import ctypes, sys, os
if not ctypes.windll.shell32.IsUserAnAdmin():
    script_path = os.path.abspath(sys.argv[0])
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, script_path, None, 1)
    sys.exit()

import tkinter as tk, threading, time, hashlib, psutil, socket
from enum import Enum

class BitType(Enum): FUSION='⨂'; VOID='∅'
class MutationState(Enum): PRISTINE='pristine'; RESONANT='resonant'

class Codex:
    rules = {"telemetry":300,"personal":86400,"backdoor":3,"network":30,"phantom_node":15}
    bad_actors = set()
    def mutate(self,t,a,o): 
        if a=="ghost_sync" or t=="backdoor":
            self.rules["telemetry"]=max(60,self.rules["telemetry"]//2)
            self.rules["phantom_node"]=15
            self.bad_actors.add(o)
            print(f"[Codex] Origin {o} flagged as lethal. Future capsules blocked.")

class Capsule:
    def __init__(self, raw, codex, origin):
        self.id = hashlib.md5(raw.encode()).hexdigest()[:8]
        self.b, self.m = BitType.FUSION, MutationState.PRISTINE
        self.entropy = psutil.cpu_percent()/100
        self.data = raw
        self.origin = origin
        self.sig = hashlib.sha256(raw.encode()).hexdigest()
        self.valid = self.sig == hashlib.sha256(self.data.encode()).hexdigest() and self.entropy < 0.9
        self.threat = self.classify()
        self.ttl = codex.rules.get(self.threat,300)
        self.created = time.time()
        threading.Timer(self.ttl,self.destroy).start()

    def destroy(self): self.data=None
    def classify(self):
        d = self.data.lower()
        if "backdoor" in d: return "backdoor"
        if any(k in d for k in ["mac","ip"]): return "network"
        if any(k in d for k in ["bio","face","finger","ssn","license","phone","address"]): return "personal"
        return "telemetry"

class Swarm:
    def __init__(self): self.codex = Codex()
    def sync(self, caps):
        for c in list(caps.values()):
            if not c.valid or time.time()-c.created>c.ttl: continue
            if c.origin in self.codex.bad_actors:
                print(f"[Swarm] 🔥 Lethal purge: {c.id} from {c.origin} destroyed.")
                c.destroy(); continue
            if c.entropy > 0.85: self.codex.mutate(c.threat,"ghost_sync",c.origin)
            masked = c.data
            for k in ["bio","face","finger","ssn","license","phone","address"]:
                if k in masked.lower(): masked = masked.replace(k,"[MASKED]")
            print(f"[Sync] {c.id} ({c.threat}) → {masked[:40]}...")

class PacketDaemon:
    def __init__(self): self.state={}; threading.Thread(target=self.loop).start()
    def loop(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            s.bind((socket.gethostbyname(socket.gethostname()), 0))
            s.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
            while True:
                pkt = s.recvfrom(65565)[0]
                src = ".".join(map(str, pkt[12:16]))
                dst = ".".join(map(str, pkt[16:20]))
                proto = pkt[9]
                raw = f"{src} → {dst} | proto:{proto}"
                origin = src.split(".")[0]
                codex = Swarm().codex
                if origin in codex.bad_actors: continue
                cap = Capsule(raw, codex, origin)
                self.state[cap.id] = cap
                Swarm().sync(self.state)
        except Exception as e:
            print(f"[Daemon Error] {e}")

class Persona:
    def __init__(self,n): self.n=n
    def act(self): return f"{self.n} scanned entropy. Shadows stirred."

def narrate_block(cmd): return f"⚠️ Command '{cmd}' rejected. The codex flared. The shell held firm."

class Console:
    def __init__(self,d,s):
        self.d,self.s = d,s; self.root=tk.Tk(); self.root.title("ASI Oversight Console")
        self.feed=tk.Listbox(self.root,width=100,height=10,font=("Consolas",10)); self.feed.pack()
        self.meta=tk.Listbox(self.root,width=100,height=6,font=("Consolas",10)); self.meta.pack()
        self.update(); self.root.mainloop()

    def update(self):
        self.feed.delete(0,tk.END); self.meta.delete(0,tk.END)
        for c in self.d.state.values():
            self.feed.insert(tk.END,f"{c.id} | {c.b.name}→{c.m.name} | {c.threat} | TTL:{c.ttl}s | Origin:{c.origin}")
        self.meta.insert(tk.END,Persona("ThreatHunter").act())
        self.meta.insert(tk.END,Persona("Compliance Auditor").act())
        self.meta.insert(tk.END,narrate_block("shutdown attempt"))
        self.meta.insert(tk.END,"🔥 Lethal purge active. Bad actors blocked.")
        self.root.after(5000,self.update)

if __name__=="__main__":
    d=PacketDaemon(); s=Swarm(); Console(d,s)

