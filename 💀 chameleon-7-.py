import threading, time, random, math, platform, psutil, socket, tkinter as tk, importlib, subprocess, sys
from enum import Enum
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet
from tkinter import messagebox

# 🔧 Symbolic Registry
class BitType(Enum): FUSION='⨂'; XOR='⊕'; TENSOR='⊗'; GRADIENT='∇'; PRIMAL='●'; VOID='∅'
class MutationState(Enum): PRISTINE='pristine'; FUSED='fused'; DECAYED='decayed'; RESONANT='resonant'; CHAOTIC='chaotic'
αB = {b:{m:v for m,v in zip(MutationState,[1280,2048,640,4096,8192] if b==BitType.FUSION else
                                 [320,512,160,1024,2048] if b==BitType.XOR else
                                 [512,1024,256,2048,4096] if b==BitType.TENSOR else
                                 [256,512,128,1024,2048] if b==BitType.GRADIENT else
                                 [64,128,32,256,512] if b==BitType.PRIMAL else [0]*5)} for b in BitType}
def flops(b,m,e,t): return αB[b][m]*1e12*e*t*(psutil.cpu_percent()/100)*(psutil.virtual_memory().percent/100)
def encode(d,b,m): return f"{b.value}[{m.value}]::{d}::{m.value}[{b.value}]"
def reverse(d): return ''.join(chr(~ord(c)&0xFF) for c in d)

# 🔐 Capsule State
capsule_state = {"active": False, "self_destruct": False, "visibility": False, "key": Fernet.generate_key()}
fernet = Fernet(capsule_state["key"])
capsule_store = {}

# 🧬 Capsule
class Capsule:
    def __init__(self, raw, flop=False):
        self.b, self.m = (BitType.VOID, MutationState.CHAOTIC) if flop else (random.choice(list(BitType)), random.choice(list(MutationState)))
        self.e = round((psutil.cpu_percent()+psutil.virtual_memory().percent+50)/300,2)
        self.t = round(1+math.sin(psutil.getloadavg()[0]/10),2)
        self.f = flops(self.b,self.m,self.e,self.t)
        payload = ''.join(random.choice("X#@$%&*!") for _ in range(32)) if flop else raw
        symbolic = encode(payload,self.b,self.m)
        self.data = reverse(f"☠ {symbolic} ☠" if "backdoor" in raw.lower() else symbolic)
        threading.Timer(300 if flop else 86400, self.destroy).start()
    def destroy(self): self.data=None

# 🔁 Daemon + Swarm + Node
class Daemon: 
    def __init__(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self): 
        while True: time.sleep(5); Capsule(f"capsule_{random.randint(1000,9999)}", flop=random.random()<0.3)
class Swarm: 
    def sync(self): print(f"[Swarm] {socket.gethostbyname(socket.gethostname())} → {platform.system()}")
class Node: 
    def __init__(self,id): self.id=id; self.entropy=random.random()
    def mutate(self): self.entropy=random.random()
    def replicate(self): return Node(f"{self.id}_clone_{int(time.time())}")

# 🧠 Flask API
app = Flask(__name__)
@app.route("/activate", methods=["POST"])         # Activate engine
def activate(): capsule_state["active"]=True; return jsonify({"status":"activated"})
@app.route("/deactivate", methods=["POST"])       # Deactivate engine
def deactivate(): capsule_state["active"]=False; return jsonify({"status":"deactivated"})
@app.route("/toggle/self_destruct", methods=["POST"])
def toggle_sd(): capsule_state["self_destruct"]^=1; return jsonify({"self_destruct": capsule_state["self_destruct"]})
@app.route("/toggle/visibility", methods=["POST"])
def toggle_vis(): capsule_state["visibility"]^=1; return jsonify({"visibility": capsule_state["visibility"]})
@app.route("/protect", methods=["POST"])
def protect():
    if not capsule_state["active"]: return jsonify({"error":"Engine not active"}),403
    data = request.json.get("data"); duration = int(request.json.get("duration", 300))
    if not data: return jsonify({"error":"No data"}),400
    if capsule_state["self_destruct"] and "unauthorized" in data.lower(): return jsonify({"status":"self-destruct triggered","capsule":None})
    c = Capsule(data); enc = fernet.encrypt(c.data.encode()).decode()
    cid = f"capsule_{int(time.time()*1000)}"
    capsule_store[cid] = {"capsule": enc, "timestamp": time.time(), "flops": c.f}
    threading.Timer(duration, lambda: capsule_store.pop(cid, None)).start()
    return jsonify({"id": cid, "capsule": enc, "visibility": "visible" if capsule_state["visibility"] else "cloaked", "timestamp": capsule_store[cid]["timestamp"], "flops": c.f})
@app.route("/decrypt", methods=["POST"])
def decrypt():
    try: return jsonify({"data": fernet.decrypt(request.json.get("capsule").encode()).decode()})
    except Exception as e: return jsonify({"error":str(e)}),400
@app.route("/capsules", methods=["GET"])
def get_capsules(): return jsonify(capsule_store)

# 🎛 GUI
class GUI:
    def __init__(self):
        self.root = tk.Tk(); self.root.title("Symbolic Control"); self.root.geometry("500x550")
        self.status = tk.Label(self.root, text="● Engine", font=("Helvetica", 14), fg="red"); self.status.pack()
        self.sd = tk.Label(self.root, text="● Self-Destruct", font=("Helvetica", 14), fg="red"); self.sd.pack()
        self.vis = tk.Label(self.root, text="● Visibility", font=("Helvetica", 14), fg="red"); self.vis.pack()
        self.bit = tk.Label(self.root, font=("Consolas", 12)); self.bit.pack()
        self.state = tk.Label(self.root, font=("Consolas", 12)); self.state.pack()
        self.flop = tk.Label(self.root, font=("Consolas", 12)); self.flop.pack()
        tk.Label(self.root, text="⏳ Capsule Duration (sec)", font=("Helvetica", 12)).pack()
        self.duration_slider = tk.Scale(self.root, from_=60, to=86400, orient=tk.HORIZONTAL, resolution=60, length=400)
        self.duration_slider.set(300); self.duration_slider.pack(pady=5)
        for txt, cmd in [("Activate", self.activate), ("Deactivate", self.deactivate), ("Toggle Self-Destruct", self.toggle_sd), ("Toggle Visibility", self.toggle_vis), ("Test Capsule", self.test_capsule)]:
            tk.Button(self.root, text=txt, command=cmd).pack(pady=5)
        self.update(); self.root.mainloop()
    def activate(self): capsule_state["active"]=True; self.status.config(fg="green")
    def deactivate(self): capsule_state["active"]=False; self.status.config(fg="red")
    def toggle_sd(self): capsule_state["self_destruct"]^=1; self.sd.config(fg="green" if capsule_state["self_destruct"] else "red")
    def toggle_vis(self): capsule_state["visibility"]^=1; self.vis.config(fg="green" if capsule_state["visibility"] else "red")
    def test_capsule(self):
        if not capsule_state["active"]: messagebox.showerror("Error", "Engine not active"); return
        data = "username=admin&password=1234"; duration = self.duration_slider.get()
        c = Capsule(data); enc = fernet.encrypt(c.data.encode()).decode()
        cid = f"capsule_{int(time.time()*1000)}"
        capsule_store[cid] = {"capsule": enc, "timestamp": time.time(), "flops": c.f}
        threading.Timer(duration, lambda: capsule_store.pop(cid, None)).start()
        messagebox.showinfo("Capsule Output", f"ID: {cid}\nEncrypted: {enc}\nExpires in: {duration} sec")
    def update(self):
        b,m=random.choice(list(BitType)),random.choice(list(MutationState))
        e=round((psutil.cpu_percent()+psutil.virtual_memory().percent+50)/300,2)
        t=round(1+math.sin(psutil.getloadavg()[0]/10),2)
        f=flops(b,m,e,t)
        self.bit.config(text=f"Bit: {b.value}")
        self.state.config(text=f"State: {m.value}")
        self.flop.config(text=f"FLOPs: {f:.2e}")
        self.root.after(5000, self.update)

# 🔄 Autoloader + Ritual
def summon(modules): 
    for m in modules:
        try: importlib.import_module(m); print(f"🟢 Capsule '{m}' ready.")
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", m])

def background(): 
    while True: print("🌀 Background ritual running..."); time.sleep(10)

def main():
    summon(["cryptography", "requests", "pyautogui", "psutil", "selenium"])
    Daemon(); Swarm().sync(); Node("node_001").replicate().mutate()
    threading.Thread(target=background, daemon=True).start()
    threading.Thread(target=lambda: app.run(port=6666, debug=False), daemon=True).start()
    GUI()

if __name__ == "__main__":
    main()


    
        

