# defense_daemon_lcars_extended.py
import os, time, threading, hashlib, base64, secrets, shutil, subprocess, json, queue, socket
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict

import psutil, numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ===== Utilities =====
def now_ts(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def rand_hex(n=16): return secrets.token_hex(n // 2)
def log(channel, msg): print(f"{time.strftime('%H:%M:%S')} [{channel}] {msg}")

def reverse_encrypt(text):
    data=text[::-1].encode("utf-8")
    key=hashlib.sha256(text.encode("utf-8")).digest()
    out=bytes([b ^ key[i%len(key)] for i,b in enumerate(data)])
    return base64.urlsafe_b64encode(out).decode("ascii")

def sha256_file(path: Path):
    try:
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda: f.read(1<<16), b""): h.update(chunk)
        return h.hexdigest()
    except Exception: return None

# ===== Optional signer lib =====
try:
    import pefile
    PEFILE_AVAILABLE = True
except Exception:
    PEFILE_AVAILABLE = False

# ===== Backend =====
class Backend:
    def array(self,x): return np.array(x,dtype=float)
    def dot(self,a,b): return float(np.dot(a,b))
cp=Backend()

# ===== Event bus =====
class EventBus:
    def __init__(self): self.q=queue.Queue(maxsize=5000)
    def publish(self,topic,payload): self.q.put({"topic":topic,"ts":now_ts(),"payload":payload})
    def subscribe(self):
        while True: yield self.q.get()
BUS=EventBus()

# ===== Self-Rewriting Agent =====
class SelfRewritingAgent:
    def __init__(self):
        self.weights=cp.array([0.6,-0.8,-0.3])
        self.events=deque(maxlen=100)
    def mutate(self,event):
        self.events.append(event)
        if event=="ghost_sync": self.weights*=cp.array([0.95,1.05,1.0])
        elif event=="rogue_escalation": self.weights*=cp.array([1.1,0.9,1.05])
        log("agent",f"🧬 Mutation {event} weights={self.weights}")
    def trust_score(self,vec): return cp.dot(cp.array(vec),self.weights)
AGENT=SelfRewritingAgent()

# ===== Cipher engine =====
class CipherEngine:
    def escalate(self,reason): log("cipher",f"🔒 Escalation: {reason}")
CIPHER=CipherEngine()

# ===== Containment (manual-only, reversible) =====
CONTAINMENT_MANIFEST = []  # in-memory panel data

QUAR_DIR=Path.home()/ "Quarantine"; QUAR_DIR.mkdir(exist_ok=True)
def quarantine(path: Path):
    if not path.exists():
        log("contain",f"ℹ️ Not found: {path}")
        return False
    dest=QUAR_DIR/f"{path.name}.quar"
    try:
        h=sha256_file(path)
        shutil.move(str(path),str(dest))
        entry={"type":"quarantine","src":str(path),"dst":str(dest),"sha256":h,"ts":now_ts()}
        CONTAINMENT_MANIFEST.append(entry)
        log("contain",f"🧰 Quarantined: {path} -> {dest} sha256={h}")
        return True
    except Exception as e:
        log("contain",f"❌ Quarantine failed: {e}")
        return False

def firewall_block_exe(exe: Path):
    rule=f"Block-{exe.name}-{rand_hex(8)}"
    cmd=["netsh","advfirewall","firewall","add","rule",f"name={rule}","dir=out","action=block",f"program={exe}","enable=yes"]
    try:
        subprocess.run(cmd,check=True,capture_output=True)
        entry={"type":"firewall_block","rule":rule,"program":str(exe),"ts":now_ts()}
        CONTAINMENT_MANIFEST.append(entry)
        log("contain",f"🛡️ Firewall rule added: {rule}")
        return True
    except Exception as e:
        log("contain",f"❌ Firewall rule failed: {e}")
        return False

def proc_context(pid:int):
    try:
        p=psutil.Process(pid)
        return {"pid":pid,"exe":p.exe(),"name":p.name()}
    except Exception:
        return {"pid":pid,"exe":None}

# ===== Signer verification =====
def signer_info(path: Path):
    if not path.exists():
        return {"signed": False, "signer": None, "error": "file_not_found"}
    if not PEFILE_AVAILABLE:
        return {"signed": False, "signer": None, "error": "pefile_not_available"}
    try:
        pe = pefile.PE(str(path), fast_load=True)
        signed = hasattr(pe, "DIRECTORY_ENTRY_SECURITY")
        # Real chain parsing omitted; we signal presence only
        return {"signed": bool(signed), "signer": "Unknown" if signed else None, "error": None}
    except Exception as e:
        return {"signed": False, "signer": None, "error": str(e)}

# ===== DNS anomaly heuristics =====
def domain_entropy(name: str):
    if not name: return 0.0
    # Shannon-like simple heuristic
    from math import log2
    freq = defaultdict(int)
    for ch in name.lower():
        if ch.isalnum() or ch in "-._": freq[ch]+=1
    n = sum(freq.values()) or 1
    return -sum((c/n)*log2(c/n) for c in freq.values())

def is_suspicious_domain(name: str):
    ent = domain_entropy(name)
    # heuristic: high entropy and long labels, many digits
    has_many_digits = sum(ch.isdigit() for ch in name) >= max(6, len(name)//3)
    long_name = len(name) > 30
    return ent > 4.0 or has_many_digits or long_name

# ===== Sensors =====
class FileSensor:
    def __init__(self,path=None):
        self.path=Path(path or (Path.home()/ "Downloads"))
        self.seen=set(p.name for p in self.path.glob("*"))
    def start(self): threading.Thread(target=self._loop,daemon=True).start()
    def _loop(self):
        while True:
            cur=set(p.name for p in self.path.glob("*")); new=cur-self.seen; self.seen=cur
            for name in new:
                p=str(self.path/name)
                BUS.publish("file.new",{"path":p})
                if p.lower().endswith((".exe",".msi",".bat",".ps1")):
                    BUS.publish("file.exec_new",{"path":p})
            time.sleep(5)

class ProcessSensor:
    def __init__(self): self.last=set()
    def start(self): threading.Thread(target=self._loop,daemon=True).start()
    def _loop(self):
        while True:
            cur=set(psutil.pids()); new=cur-self.last; self.last=cur
            for pid in new: BUS.publish("proc.start",proc_context(pid))
            time.sleep(7)

class NetSensor:
    def __init__(self,window=20,burst=6): self.window=window; self.burst=burst; self.hist=defaultdict(list)
    def start(self): threading.Thread(target=self._loop,daemon=True).start()
    def _loop(self):
        while True:
            now=time.time()
            for c in psutil.net_connections(kind="inet"):
                if c.status!="ESTABLISHED" or not c.raddr: continue
                pid=c.pid; 
                if pid is None: continue
                self.hist[pid].append((now,c.raddr))
                # DNS reverse lookup (best-effort, non-blocking guard)
                raddr = c.raddr
                hostname = None
                try:
                    hostname = socket.getnameinfo((raddr.ip, raddr.port), 0)[0]
                except Exception:
                    try:
                        hostname = socket.gethostbyaddr(raddr.ip)[0]
                    except Exception:
                        hostname = None
                payload = {"pid":pid,"raddr":f"{raddr.ip}:{raddr.port}","host":hostname}
                BUS.publish("net.conn", payload)
                if hostname and is_suspicious_domain(hostname):
                    BUS.publish("dns.anomaly", {"pid": pid, "host": hostname})
            for pid,ev in list(self.hist.items()):
                self.hist[pid]=[(t,r) for (t,r) in ev if now-t<=self.window]
                uniq={r for (_,r) in self.hist[pid]}
                if len(uniq)>=self.burst:
                    BUS.publish("net.burst_pred",{"pid":pid,"count":len(uniq)})
                    self.hist[pid]=[]
            time.sleep(5)

# ===== Bot (LCARS personas) =====
class Bot:
    def __init__(self,agent):
        self.agent=agent; self.mode="guardian"
    def start(self): threading.Thread(target=self._loop,daemon=True).start()
    def switch(self): self.mode="rogue" if self.mode=="guardian" else "guardian"; log("bot",f"🔺 Mode={self.mode}")
    def _loop(self):
        while True:
            if self.mode=="guardian":
                log("bot","🔱 LCARS audit frame"); self.agent.mutate("ghost_sync"); CIPHER.escalate("guardian")
            else:
                pat=f"{reverse_encrypt(str(int(time.time())%2048))[:8]}-{rand_hex(8)}"
                log("bot",f"🜏 Rogue pattern {pat}"); self.agent.mutate("rogue_escalation"); CIPHER.escalate("rogue")
            time.sleep(10)
BOT=Bot(AGENT)

# ===== LCARS UI (Tkinter) =====
LCARS_BG = "#000000"
LCARS_PANEL = "#111111"
LCARS_ACCENT_A = "#FF9966"   # LCARS orange
LCARS_ACCENT_B = "#66CCFF"   # LCARS blue
LCARS_ACCENT_C = "#CC66FF"   # LCARS violet
LCARS_TEXT = "#EEEEEE"
LCARS_WARN = "#FF5555"
LCARS_OK = "#55FFAA"

class LcarsBadge(tk.Canvas):
    def __init__(self, parent, text, bg=LCARS_ACCENT_A, **kw):
        super().__init__(parent, width=160, height=40, bg=LCARS_BG, highlightthickness=0, **kw)
        self.create_arc(0,0,80,80,start=90,extent=180,fill=bg,outline=bg)
        self.create_rectangle(40,0,160,40,fill=bg,outline=bg)
        self.create_text(90,20,text=text,fill=LCARS_TEXT,font=("Segoe UI", 11, "bold"))

class LcarsGauge(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, width=300, height=24, bg=LCARS_PANEL, highlightthickness=0, **kw)
        self.create_rectangle(2,2,298,22,outline=LCARS_ACCENT_B)
        self.fill = self.create_rectangle(2,2,2,22,fill=LCARS_ACCENT_B,outline=LCARS_ACCENT_B)
        self.text = self.create_text(150,12,fill=LCARS_TEXT,font=("Consolas",10),text="trust 0.000")
    def set_value(self, v):
        v = max(0.0, min(1.0, v))
        x = 2 + int(296 * v)
        self.coords(self.fill, 2, 2, x, 22)
        self.itemconfig(self.text, text=f"trust {v:.3f}")
        color = LCARS_OK if v < 0.65 else (LCARS_ACCENT_A if v < 0.85 else LCARS_WARN)
        self.itemconfig(self.fill, fill=color, outline=color)

class DefenseGUI:
    def __init__(self, root):
        self.root=root
        root.title("LCARS ASI Defense Interface")
        root.configure(bg=LCARS_BG)
        root.geometry("1240x760")

        # Header
        header = tk.Frame(root, bg=LCARS_BG)
        header.pack(fill=tk.X, padx=8, pady=8)
        LcarsBadge(header, "ASI DEFENSE").pack(side=tk.LEFT)
        LcarsBadge(header, "MANUAL CONTROL", bg=LCARS_ACCENT_B).pack(side=tk.LEFT, padx=6)
        LcarsBadge(header, "REVERSIBLE", bg=LCARS_ACCENT_C).pack(side=tk.LEFT)
        self.gauge = LcarsGauge(header); self.gauge.pack(side=tk.RIGHT)

        # Main panes
        main = tk.Frame(root, bg=LCARS_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = tk.Frame(main, bg=LCARS_PANEL); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(main, bg=LCARS_PANEL, width=420); right.pack(side=tk.RIGHT, fill=tk.Y)

        # Event feed
        tk.Label(left, text="Event feed", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=(8,0))
        self.feed = scrolledtext.ScrolledText(left, wrap=tk.WORD, height=26, bg="#0F0F0F", fg=LCARS_TEXT, insertbackground=LCARS_TEXT)
        self.feed.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Incident queue
        tk.Label(right, text="Incident queue", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=(8,0))
        self.incidents = tk.Listbox(right, height=8, bg="#0F0F0F", fg=LCARS_TEXT)
        self.incidents.pack(fill=tk.X, padx=8, pady=8)

        # Action bay
        tk.Label(right, text="Action bay", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8)
        ab = tk.Frame(right, bg=LCARS_PANEL); ab.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(ab, text="Quarantine path", bg=LCARS_PANEL, fg=LCARS_TEXT).grid(row=0, column=0, sticky="w")
        self.qpath = tk.Entry(ab, width=36, bg="#101010", fg=LCARS_TEXT, insertbackground=LCARS_TEXT)
        self.qpath.grid(row=1, column=0, padx=(0,8)); tk.Button(ab, text="Quarantine", command=self.do_quarantine, bg=LCARS_ACCENT_A).grid(row=1, column=1)

        tk.Label(ab, text="Block PID", bg=LCARS_PANEL, fg=LCARS_TEXT).grid(row=2, column=0, sticky="w", pady=(8,0))
        self.bpid = tk.Entry(ab, width=12, bg="#101010", fg=LCARS_TEXT, insertbackground=LCARS_TEXT)
        self.bpid.grid(row=3, column=0); tk.Button(ab, text="Block", command=self.do_block, bg=LCARS_ACCENT_B).grid(row=3, column=1)

        tk.Button(ab, text="Switch persona", command=self.do_switch, bg=LCARS_ACCENT_C).grid(row=4, column=0, columnspan=2, pady=(10,0))

        # Signer verification panel
        sv = tk.Frame(right, bg=LCARS_PANEL); sv.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(sv, text="Signer verification", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0,6))
        self.sv_path = tk.Entry(sv, width=30, bg="#101010", fg=LCARS_TEXT, insertbackground=LCARS_TEXT)
        self.sv_path.grid(row=1, column=0, padx=(0,8))
        tk.Button(sv, text="Verify signer", command=self.do_verify_signer, bg=LCARS_ACCENT_B).grid(row=1, column=1)
        self.sv_result = tk.Label(sv, text="", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Consolas", 10))
        self.sv_result.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6,0))

        # DNS anomalies panel
        tk.Label(right, text="DNS anomalies", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8)
        self.dns_list = tk.Listbox(right, height=8, bg="#0F0F0F", fg=LCARS_TEXT)
        self.dns_list.pack(fill=tk.X, padx=8, pady=8)

        # Containment manifest panel
        tk.Label(right, text="Containment manifest", bg=LCARS_PANEL, fg=LCARS_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8)
        self.manifest = tk.Listbox(right, height=10, bg="#0F0F0F", fg=LCARS_TEXT)
        self.manifest.pack(fill=tk.BOTH, padx=8, pady=(8,12), expand=False)

        # Background consumers
        threading.Thread(target=self.consume_events, daemon=True).start()
        threading.Thread(target=self.refresh_trust_loop, daemon=True).start()
        threading.Thread(target=self.refresh_manifest_loop, daemon=True).start()

    def add_event(self, evt):
        line = f"[{evt['ts']}] {evt['topic']}: {json.dumps(evt['payload'])}\n"
        self.feed.insert(tk.END, line)
        self.feed.see(tk.END)
        # Populate incident queue for actionable items
        t = evt["topic"]
        if t in {"file.exec_new", "net.burst_pred", "proc.start"}:
            self.incidents.insert(tk.END, f"{t} :: {evt['payload']}")
        # DNS anomalies panel
        if t == "dns.anomaly":
            host = evt["payload"].get("host")
            self.dns_list.insert(tk.END, f"{evt['payload'].get('pid')} :: {host}")

    def consume_events(self):
        for evt in BUS.subscribe():
            self.root.after(0, self.add_event, evt)

    def refresh_trust_loop(self):
        while True:
            vec=[secrets.randbelow(3)-1 for _ in range(3)]
            score=AGENT.trust_score(vec)
            trust_scaled = max(0.0, min(1.0, (score+1.5)/3.0))
            self.root.after(0, self.gauge.set_value, trust_scaled)
            time.sleep(10)

    def refresh_manifest_loop(self):
        last_len = 0
        while True:
            if len(CONTAINMENT_MANIFEST) != last_len:
                self.manifest.delete(0, tk.END)
                for item in CONTAINMENT_MANIFEST:
                    if item["type"] == "quarantine":
                        line = f"Q: {item['src']} -> {item['dst']} sha256={item['sha256']}"
                    else:
                        line = f"FW: {item['program']} rule={item['rule']}"
                    self.manifest.insert(tk.END, line)
                last_len = len(CONTAINMENT_MANIFEST)
            time.sleep(3)

    def do_quarantine(self):
        path=Path(self.qpath.get().strip())
        if not path:
            messagebox.showinfo("Quarantine", "Enter a valid path.")
            return
        ok = quarantine(path)
        if ok:
            messagebox.showinfo("Quarantine", "File moved to Quarantine.")
        else:
            messagebox.showerror("Quarantine", "Quarantine failed.")

    def do_block(self):
        try:
            pid=int(self.bpid.get().strip())
            ctx=proc_context(pid); exe=ctx.get("exe")
            if exe: 
                ok = firewall_block_exe(Path(exe))
                messagebox.showinfo("Firewall", "Rule added." if ok else "Failed to add rule.")
            else:
                messagebox.showinfo("Firewall", f"No executable for PID {pid}")
        except Exception as e:
            messagebox.showerror("Firewall", f"Error: {e}")

    def do_switch(self):
        BOT.switch()

    def do_verify_signer(self):
        path = Path(self.sv_path.get().strip())
        info = signer_info(path)
        if info["error"]:
            self.sv_result.config(text=f"error={info['error']}", fg=LCARS_WARN)
        else:
            s = "signed" if info["signed"] else "unsigned"
            signer = info["signer"] or "None"
            self.sv_result.config(text=f"{s} | signer={signer}", fg=LCARS_OK if info["signed"] else LCARS_WARN)

# ===== Entrypoint orchestration =====
def start_sensors_and_bot():
    FileSensor().start(); ProcessSensor().start(); NetSensor().start(); BOT.start()

def main():
    log("boot","Starting LCARS ASI Defense Interface (manual-only containment)")
    start_sensors_and_bot()
    root=tk.Tk(); DefenseGUI(root); root.mainloop()

if __name__=="__main__": main()

