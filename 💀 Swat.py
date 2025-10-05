import threading, time, random, math, hashlib, platform, psutil, socket, tkinter as tk
from enum import Enum

# 🔧 Symbolic Registry
class BitType(Enum): FUSION='⨂'; XOR='⊕'; TENSOR='⊗'; GRADIENT='∇'; PRIMAL='●'; VOID='∅'
class MutationState(Enum): PRISTINE='pristine'; FUSED='fused'; DECAYED='decayed'; RESONANT='resonant'; CHAOTIC='chaotic'

αB = {
    BitType.FUSION:     {MutationState.PRISTINE:1280, MutationState.FUSED:2048, MutationState.DECAYED:640, MutationState.RESONANT:4096, MutationState.CHAOTIC:8192},
    BitType.XOR:        {MutationState.PRISTINE:320,  MutationState.FUSED:512,  MutationState.DECAYED:160, MutationState.RESONANT:1024, MutationState.CHAOTIC:2048},
    BitType.TENSOR:     {MutationState.PRISTINE:512,  MutationState.FUSED:1024, MutationState.DECAYED:256, MutationState.RESONANT:2048, MutationState.CHAOTIC:4096},
    BitType.GRADIENT:   {MutationState.PRISTINE:256,  MutationState.FUSED:512,  MutationState.DECAYED:128, MutationState.RESONANT:1024, MutationState.CHAOTIC:2048},
    BitType.PRIMAL:     {MutationState.PRISTINE:64,   MutationState.FUSED:128,  MutationState.DECAYED:32,  MutationState.RESONANT:256,  MutationState.CHAOTIC:512},
    BitType.VOID:       {m:0 for m in MutationState}
}

def encode(data,b,m): return f"{b.value}[{m.value}]::{data}::{m.value}[{b.value}]"
def reverse_polarity(data): return ''.join(chr(~ord(c) & 0xFF) for c in data)
def flops(b,m,e,t): return αB[b][m]*1e12*e*t*(psutil.cpu_percent()/100)*(psutil.virtual_memory().percent/100)

# 🧬 Capsule Engine
class Capsule:
    def __init__(self, raw, flop=False):
        self.b = BitType.VOID if flop else random.choice(list(BitType))
        self.m = MutationState.CHAOTIC if flop else random.choice(list(MutationState))
        self.e = round((psutil.cpu_percent()+psutil.virtual_memory().percent+50)/300,2)
        self.t = round(1+math.sin(psutil.getloadavg()[0]/10),2)
        self.f = flops(self.b,self.m,self.e,self.t)
        payload = ''.join(random.choice("X#@$%&*!") for _ in range(32)) if flop else raw
        symbolic = encode(payload,self.b,self.m)
        self.data = reverse_polarity(f"☠ {symbolic} ☠" if "backdoor" in raw.lower() else symbolic)
        self.prio = "decoy" if flop else ("high" if any(k in raw.lower() for k in ["bio","face","finger","ssn","license","phone","address"]) else "low")
        self.timer = 15 if flop else 3 if "backdoor" in raw.lower() else 30 if "no_mac_ip" in raw.lower() else 86400 if self.prio=="high" else 30 if "fake_telemetry" in raw.lower() else 300
        threading.Timer(self.timer,self.destroy).start()

    def destroy(self): self.data=None

# 🔁 Mutation Daemon
class Daemon:
    def __init__(self): self.state={}; threading.Thread(target=self.loop).start()
    def loop(self):
        while True:
            flop = random.random()<0.3
            cid = f"{'flop' if flop else 'capsule'}_{random.randint(1000,9999)}"
            self.state[cid] = Capsule(cid, flop)
            time.sleep(5)

# 📡 Swarm Sync
class Swarm:
    def __init__(self): self.nodes=["192.168.1.10","192.168.1.11"]
    def sync(self): print(f"[Swarm] {socket.gethostbyname(socket.gethostname())} → {platform.system()}")

# 🔄 Replicator Node
class Node:
    def __init__(self,id): self.id=id; self.entropy=random.random()
    def mutate(self): self.entropy=random.random()
    def replicate(self): return Node(f"{self.id}_clone_{int(time.time())}")

# 🎛 GUI Overlay
class GUI:
    def __init__(self):
        self.root=tk.Tk(); self.root.title("Symbolic Overlay")
        self.bit=tk.Label(self.root,font=("Consolas",12)); self.bit.pack()
        self.state=tk.Label(self.root,font=("Consolas",12)); self.state.pack()
        self.flop=tk.Label(self.root,font=("Consolas",12)); self.flop.pack()
        self.update(); self.root.mainloop()

    def update(self):
        b,m=random.choice(list(BitType)),random.choice(list(MutationState))
        e=round((psutil.cpu_percent()+psutil.virtual_memory().percent+50)/300,2)
        t=round(1+math.sin(psutil.getloadavg()[0]/10),2)
        f=flops(b,m,e,t)
        self.bit.config(text=f"Bit: {b.value}"); self.state.config(text=f"State: {m.value}"); self.flop.config(text=f"FLOPs: {f:.2e}")
        self.root.after(5000,self.update)

# 🚀 Ritual Activation
if __name__=="__main__":
    Daemon()
    Swarm().sync()
    n=Node("node_001"); n.mutate(); n.replicate().mutate()
    GUI()

