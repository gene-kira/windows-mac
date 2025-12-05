# defense_daemon_lolbins_guard.py
# LCARS-inspired defense daemon with Tkinter GUI, ethical autonomy, LOLBins watchlist, lineage checks

import os, time, threading, hashlib, base64, secrets, shutil, subprocess, json, queue, socket
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict

import psutil, numpy as np, yaml
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ===== Utilities =====
def now_ts(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def rand_hex(n=16): return secrets.token_hex(n // 2)
def log(channel, msg): print(f"{time.strftime('%H:%M:%S')} [{channel}] {msg}")

def sha256_file(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

# ===== Obfuscation (reversible) =====
def reverse_mirror(text: str) -> str: return text[::-1]
def reverse_read_encrypt(text: str, key: str) -> str:
    data = text[::-1].encode("utf-8")
    k = hashlib.sha256(key.encode("utf-8")).digest()
    out = bytes([b ^ k[i % len(k)] for i, b in enumerate(data)])
    return base64.urlsafe_b64encode(out).decode("ascii")
def reverse_read_decrypt(blob: str, key: str) -> str:
    raw = base64.urlsafe_b64decode(blob.encode("ascii"))
    k = hashlib.sha256(key.encode("utf-8")).digest()
    data = bytes([b ^ k[i % len(k)] for i, b in enumerate(raw)])
    return data.decode("utf-8")[::-1]
def chameleon_camouflage(text: str, persona: str) -> str:
    salt = hashlib.sha1(persona.encode("utf-8")).digest()
    payload = bytes(a ^ b for a, b in zip(text.encode("utf-8"), salt[:len(text)]))
    return f"{persona}:{base64.urlsafe_b64encode(payload).decode('ascii')}"
def chameleon_decode(blob: str) -> str:
    try:
        persona, data_b64 = blob.split(":", 1)
        salt = hashlib.sha1(persona.encode("utf-8")).digest()
        payload = base64.urlsafe_b64decode(data_b64.encode("ascii"))
        text_bytes = bytes(a ^ b for a, b in zip(payload, salt[:len(payload)]))
        return text_bytes.decode("utf-8")
    except Exception:
        return ""

# ===== Optional signer lib =====
try:
    import pefile
    PEFILE_AVAILABLE = True
except Exception:
    PEFILE_AVAILABLE = False

# ===== Backend =====
class Backend:
    def array(self, x): return np.array(x, dtype=float)
    def dot(self, a, b): return float(np.dot(a, b))
cp = Backend()

# ===== Event bus =====
class EventBus:
    def __init__(self): self.q = queue.Queue(maxsize=5000)
    def publish(self, topic, payload): self.q.put({"topic": topic, "ts": now_ts(), "payload": payload})
    def subscribe(self):
        while True: yield self.q.get()
BUS = EventBus()

# ===== Policies =====
DEFAULT_POLICIES_YAML = """
version: 1
auto:
  enabled: true
  reversible_only: true
  expiry_minutes: 5
thresholds:
  bernoulli_warn: 0.65
  bernoulli_block_ready: 0.85
  cusum_burst: 8
actions:
  quarantine_extensions: [".exe", ".msi", ".bat", ".ps1"]
allowlist:
  signed_only: false
  approved_paths: []
denylist:
  lolbins: ["powershell.exe","wscript.exe","cscript.exe","mshta.exe","rundll32.exe","regsvr32.exe","msiexec.exe","schtasks.exe","sc.exe","taskkill.exe","robocopy.exe","certutil.exe","bitsadmin.exe","installutil.exe","odbcconf.exe","cmd.exe","curl.exe"]
  suspicious_parents: ["winword.exe","excel.exe","outlook.exe","acrord32.exe"]
"""
class Policies:
    def __init__(self, root=None):
        self.root = Path(root or (Path.home() / "DefensePolicies"))
        self.root.mkdir(exist_ok=True)
        self.path = self.root / "policies.yaml"
        if not self.path.exists():
            self.path.write_text(DEFAULT_POLICIES_YAML, encoding="utf-8")
        self.data = yaml.safe_load(self.path.read_text())
    def update(self, updates: dict):
        self.data = {**self.data, **updates}
        self.path.write_text(yaml.safe_dump(self.data), encoding="utf-8")
    def get(self): return self.data
POLICIES = Policies()

# ===== Agent and cipher =====
class SelfRewritingAgent:
    def __init__(self):
        self.weights = cp.array([0.6, -0.8, -0.3]); self.events = deque(maxlen=100)
    def mutate(self, event):
        self.events.append(event)
        if event == "ghost_sync": self.weights = self.weights * cp.array([0.95, 1.05, 1.0])
        elif event == "rogue_escalation": self.weights = self.weights * cp.array([1.1, 0.9, 1.05])
        log("agent", f"🧬 Mutation {event} weights={self.weights}")
    def trust_score(self, vec): return cp.dot(cp.array(vec), self.weights)
AGENT = SelfRewritingAgent()

class CipherEngine:
    def escalate(self, reason): log("cipher", f"🔒 Escalation: {reason}")
CIPHER = CipherEngine()

# ===== Detection: Bernoulli + CUSUM =====
class BernoulliDetector:
    def __init__(self, alpha=1.0, beta=1.0): self.alpha=alpha; self.beta=beta
    def update(self, outcome: int): self.alpha += 1.0 if outcome==1 else 0.0; self.beta += 1.0 if outcome==0 else 0.0
    def probability(self) -> float: return self.alpha / (self.alpha + self.beta)
class CusumBurst:
    def __init__(self, k=1.0, h=8.0): self.k=k; self.h=h; self.s=0.0
    def update(self, x: float) -> bool:
        self.s = max(0.0, self.s + x - self.k); return self.s > self.h
DETECTOR = BernoulliDetector()
CUSUM = CusumBurst(k=1.0, h=float(POLICIES.get()["thresholds"]["cusum_burst"]))

# ===== Containment (manual + reversible with expiry) =====
CONTAINMENT_MANIFEST = []
QUAR_DIR = Path.home() / "Quarantine"; QUAR_DIR.mkdir(exist_ok=True)

def quarantine(path: Path):
    if not path.exists(): log("contain", f"ℹ️ Not found: {path}"); return False
    dest = QUAR_DIR / f"{path.name}.quar"
    try:
        h = sha256_file(path); shutil.move(str(path), str(dest))
        entry = {"type":"quarantine","src":str(path),"dst":str(dest),"sha256":h,"ts":now_ts()}
        CONTAINMENT_MANIFEST.append(entry); log("contain", f"🧰 Quarantined: {path} -> {dest} sha256={h}"); return True
    except Exception as e: log("contain", f"❌ Quarantine failed: {e}"); return False

def restore_quarantine(dst: Path):
    try:
        if not dst.exists(): return False
        src_name = dst.name[:-5] if dst.name.endswith(".quar") else dst.name
        orig = dst.parent.parent / src_name
        shutil.move(str(dst), str(orig)); log("contain", f"♻️ Restored: {dst} -> {orig}"); return True
    except Exception as e: log("contain", f"❌ Restore failed: {e}"); return False

def firewall_block_exe(exe: Path):
    rule=f"Block-{exe.name}-{rand_hex(8)}"
    cmd=["netsh","advfirewall","firewall","add","rule",f"name={rule}","dir=out","action=block",f"program={exe}","enable=yes"]
    try:
        subprocess.run(cmd,check=True, capture_output=True)
        entry={"type":"firewall_block","rule":rule,"program":str(exe),"ts":now_ts()}
        CONTAINMENT_MANIFEST.append(entry); log("contain", f"🛡️ Firewall rule added: {rule}"); return True
    except Exception as e: log("contain", f"❌ Firewall rule failed: {e}"); return False

def firewall_remove_rule(rule_name: str):
    cmd=["netsh","advfirewall","firewall","delete","rule",f"name={rule_name}"]
    try: subprocess.run(cmd,check=True,capture_output=True); log("contain", f"🧹 Firewall rule removed: {rule_name}"); return True
    except Exception as e: log("contain", f"❌ Remove rule failed: {e}"); return False

def signer_info(path: Path):
    if not path.exists(): return {"signed": False, "signer": None, "error": "file_not_found"}
    if not PEFILE_AVAILABLE: return {"signed": False, "signer": None, "error": "pefile_not_available"}
    try:
        pe = pefile.PE(str(path), fast_load=True)
        signed = hasattr(pe, "DIRECTORY_ENTRY_SECURITY")
        return {"signed": bool(signed), "signer": "Unknown" if signed else None, "error": None}
    except Exception as e:
        return {"signed": False, "signer": None, "error": str(e)}

# ===== DNS anomaly heuristics =====
def domain_entropy(name: str):
    if not name: return 0.0
    from math import log2
    freq = defaultdict(int)
    for ch in name.lower():
        if ch.isalnum() or ch in "-._": freq[ch]+=1
    n = sum(freq.values()) or 1
    return -sum((c/n)*log2(c/n) for c in freq.values())
def is_suspicious_domain(name: str):
    ent = domain_entropy(name); has_many_digits = sum(ch.isdigit() for ch in name) >= max(6, len(name)//3); long_name = len(name) > 30
    return ent > 4.0 or has_many_digits or long_name

# ===== Sensors with LOLBins & lineage =====
class FileSensor:
    def __init__(self, path=None):
        self.path = Path(path or (Path.home() / "Downloads")); self.seen = set(p.name for p in self.path.glob("*"))
    def start(self): threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        exts = set(POLICIES.get()["actions"]["quarantine_extensions"])
        while True:
            cur = set(p.name for p in self.path.glob("*")); new = cur - self.seen; self.seen = cur
            for name in new:
                p = str(self.path / name); BUS.publish("file.new", {"path": p})
                if any(p.lower().endswith(e) for e in exts):
                    BUS.publish("file.exec_new", {"path": p}); DETECTOR.update(1)
            time.sleep(5)

def proc_context(pid: int):
    try:
        p = psutil.Process(pid)
        ctx = {"pid": pid, "exe": p.exe(), "name": p.name(), "ppid": p.ppid()}
        try:
            parent = psutil.Process(p.ppid()) if p.ppid() else None
            ctx["parent_name"] = parent.name() if parent else None
            ctx["parent_exe"] = parent.exe() if parent else None
        except Exception:
            pass
        return ctx
    except Exception:
        return {"pid": pid, "exe": None}

class ProcessSensor:
    def __init__(self): self.last = set()
    def start(self): threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        deny = set(n.lower() for n in POLICIES.get()["denylist"]["lolbins"])
        suspicious_parents = set(n.lower() for n in POLICIES.get()["denylist"]["suspicious_parents"])
        while True:
            cur = set(psutil.pids()); new = cur - self.last; self.last = cur
            for pid in new:
                ctx = proc_context(pid)
                BUS.publish("proc.start", ctx)
                name = (ctx.get("name") or "").lower()
                parent = (ctx.get("parent_name") or "").lower()
                if name in deny:
                    BUS.publish("proc.lolbin", ctx); DETECTOR.update(1)
                if parent in suspicious_parents and name in deny:
                    BUS.publish("proc.lolbin_parent", ctx); DETECTOR.update(1)
            time.sleep(7)

class NetSensor:
    def __init__(self, window=20): self.window=window; self.hist=defaultdict(list)
    def start(self): threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while True:
            now = time.time()
            for c in psutil.net_connections(kind="inet"):
                if c.status!="ESTABLISHED" or not c.raddr: continue
                pid=c.pid
                if pid is None: continue
                self.hist[pid].append((now,c.raddr))
                raddr=c.raddr; hostname=None
                try: hostname = socket.getnameinfo((raddr.ip, raddr.port), 0)[0]
                except Exception:
                    try: hostname = socket.gethostbyaddr(raddr.ip)[0]
                    except Exception: hostname = None
                payload={"pid":pid,"raddr":f"{raddr.ip}:{raddr.port}","host":hostname}
                BUS.publish("net.conn", payload)
                if hostname and is_suspicious_domain(hostname):
                    BUS.publish("dns.anomaly", {"pid": pid, "host": hostname}); DETECTOR.update(1)
            for pid, ev in list(self.hist.items()):
                self.hist[pid] = [(t, r) for (t, r) in ev if now - t <= self.window]
                uniq_count = len({r for (_, r) in self.hist[pid]})
                if CUSUM.update(float(uniq_count)):
                    BUS.publish("net.burst_pred", {"pid": pid, "count": uniq_count}); DETECTOR.update(1)
            time.sleep(5)

# ===== Bot =====
class Bot:
    def __init__(self, agent): self.agent=agent; self.mode="guardian"
    def start(self): threading.Thread(target=self._loop, daemon=True).start()
    def switch(self):
        self.mode="rogue" if self.mode=="guardian" else "guardian"; log("bot", f"🔺 Mode={self.mode}")
    def _loop(self):
        while True:
            if self.mode=="guardian":
                phrase=f"LCARS audit {rand_hex(8)}"; camo=chameleon_camouflage(phrase,"guardian")
                log("bot", f"🔱 {reverse_mirror(phrase)} | camo={camo}")
                self.agent.mutate("ghost_sync"); CIPHER.escalate("guardian"); DETECTOR.update(0)
            else:
                raw=f"{reverse_read_encrypt(str(int(time.time())%2048),'rogue-key')}-{rand_hex(8)}"; camo=chameleon_camouflage(raw,"rogue")
                log("bot", f"🜏 pattern={raw} | camo={camo}")
                self.agent.mutate("rogue_escalation"); CIPHER.escalate("rogue")
            time.sleep(10)
BOT = Bot(AGENT)

# ===== Ethical autonomous devourer =====
class DevourerEthical:
    def __init__(self):
        self.running=False; self.policy=POLICIES.get()
        self.last_exec_new = None
        BUS.publish("policies.loaded", self.policy)
    def start(self):
        self.running=True; threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._tap_bus, daemon=True).start()
    def _tap_bus(self):
        # Track last new executable for recommendations
        for evt in BUS.subscribe():
            if evt["topic"]=="file.exec_new": self.last_exec_new = evt["payload"]["path"]
    def _loop(self):
        while self.running:
            p = DETECTOR.probability()
            bern_warn = self.policy["thresholds"]["bernoulli_warn"]
            bern_block = self.policy["thresholds"]["bernoulli_block_ready"]
            auto_enabled = self.policy["auto"]["enabled"]
            expiry_min = int(self.policy["auto"]["expiry_minutes"])
            decision = ("block_ready" if p >= bern_block else ("warn" if p >= bern_warn else "normal"))
            BUS.publish("detector.status", {"p": round(p,4), "decision": decision})
            # Auto reversible behavior: temporary firewall block for LOLBin bursts
            if auto_enabled and decision in {"warn","block_ready"}:
                # Soft quarantine candidate if a new executable appeared
                if self.last_exec_new:
                    path = Path(self.last_exec_new)
                    meta = {"action":"quarantine", "path": str(path)}
                    BUS.publish("devourer.recommend", meta)
                    # Do not auto-quarantine; require manual click in GUI (irreversible risk if user wants to keep it)
                # Example auto temporary block: NONE by default; require manual approval in GUI
            time.sleep(5)
DEVOURER = DevourerEthical()

# ===== LCARS UI =====
LCARS_BG="#000000"; LCARS_PANEL="#111111"; LCARS_ACCENT_A="#FF9966"; LCARS_ACCENT_B="#66CCFF"; LCARS_ACCENT_C="#CC66FF"; LCARS_TEXT="#EEEEEE"; LCARS_WARN="#FF5555"; LCARS_OK="#55FFAA"
class LcarsBadge(tk.Canvas):
    def __init__(self,parent,text,bg=LCARS_ACCENT_A,**kw):
        super().__init__(parent,width=160,height=40,bg=LCARS_BG,highlightthickness=0,**kw)
        self.create_arc(0,0,80,80,start=90,extent=180,fill=bg,outline=bg); self.create_rectangle(40,0,160,40,fill=bg,outline=bg)
        self.create_text(90,20,text=text,fill=LCARS_TEXT,font=("Segoe UI",11,"bold"))
class LcarsGauge(tk.Canvas):
    def __init__(self,parent,**kw):
        super().__init__(parent,width=300,height=24,bg=LCARS_PANEL,highlightthickness=0,**kw)
        self.create_rectangle(2,2,298,22,outline=LCARS_ACCENT_B); self.fill=self.create_rectangle(2,2,2,22,fill=LCARS_ACCENT_B,outline=LCARS_ACCENT_B)
        self.text=self.create_text(150,12,fill=LCARS_TEXT,font=("Consolas",10),text="trust 0.000")
    def set_value(self,v):
        v=max(0.0,min(1.0,v)); x=2+int(296*v); self.coords(self.fill,2,2,x,22); self.itemconfig(self.text,text=f"trust {v:.3f}")
        color = LCARS_OK if v<0.65 else (LCARS_ACCENT_A if v<0.85 else LCARS_WARN); self.itemconfig(self.fill, fill=color, outline=color)
class DefenseGUI:
    def __init__(self,root):
        self.root=root; root.title("LCARS ASI Defense Interface"); root.configure(bg=LCARS_BG); root.geometry("1320x900")
        header=tk.Frame(root,bg=LCARS_BG); header.pack(fill=tk.X,padx=8,pady=8)
        LcarsBadge(header,"ASI DEFENSE").pack(side=tk.LEFT); LcarsBadge(header,"MANUAL CONTROL",bg=LCARS_ACCENT_B).pack(side=tk.LEFT,padx=6)
        LcarsBadge(header,"REVERSIBLE",bg=LCARS_ACCENT_C).pack(side=tk.LEFT); self.gauge=LcarsGauge(header); self.gauge.pack(side=tk.RIGHT)
        main=tk.Frame(root,bg=LCARS_BG); main.pack(fill=tk.BOTH,expand=True,padx=8,pady=8)
        left=tk.Frame(main,bg=LCARS_PANEL); left.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        right=tk.Frame(main,bg=LCARS_PANEL,width=540); right.pack(side=tk.RIGHT,fill=tk.Y)
        tk.Label(left,text="Event feed",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8,pady=(8,0))
        self.feed=scrolledtext.ScrolledText(left,wrap=tk.WORD,height=30,bg="#0F0F0F",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.feed.pack(fill=tk.BOTH,expand=True,padx=8,pady=8)
        tk.Label(right,text="Incident queue",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8,pady=(8,0))
        self.incidents=tk.Listbox(right,height=8,bg="#0F0F0F",fg=LCARS_TEXT); self.incidents.pack(fill=tk.X,padx=8,pady=8)
        tk.Label(right,text="Action bay",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8)
        ab=tk.Frame(right,bg=LCARS_PANEL); ab.pack(fill=tk.X,padx=8,pady=8)
        tk.Label(ab,text="Quarantine path",bg=LCARS_PANEL,fg=LCARS_TEXT).grid(row=0,column=0,sticky="w")
        self.qpath=tk.Entry(ab,width=36,bg="#101010",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.qpath.grid(row=1,column=0,padx=(0,8))
        tk.Button(ab,text="Quarantine",command=self.do_quarantine,bg=LCARS_ACCENT_A).grid(row=1,column=1)
        tk.Label(ab,text="Block PID",bg=LCARS_PANEL,fg=LCARS_TEXT).grid(row=2,column=0,sticky="w",pady=(8,0))
        self.bpid=tk.Entry(ab,width=12,bg="#101010",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.bpid.grid(row=3,column=0)
        tk.Button(ab,text="Block",command=self.do_block,bg=LCARS_ACCENT_B).grid(row=3,column=1)
        tk.Button(ab,text="Switch persona",command=self.do_switch,bg=LCARS_ACCENT_C).grid(row=4,column=0,columnspan=2,pady=(10,0))
        sv=tk.Frame(right,bg=LCARS_PANEL); sv.pack(fill=tk.X,padx=8,pady=8)
        tk.Label(sv,text="Signer verification",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).grid(row=0,column=0,sticky="w",pady=(0,6))
        self.sv_path=tk.Entry(sv,width=30,bg="#101010",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.sv_path.grid(row=1,column=0,padx=(0,8))
        tk.Button(sv,text="Verify signer",command=self.do_verify_signer,bg=LCARS_ACCENT_B).grid(row=1,column=1)
        self.sv_result=tk.Label(sv,text="",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Consolas",10)); self.sv_result.grid(row=2,column=0,columnspan=2,sticky="w",pady=(6,0))
        tk.Label(right,text="DNS anomalies",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8)
        self.dns_list=tk.Listbox(right,height=8,bg="#0F0F0F",fg=LCARS_TEXT); self.dns_list.pack(fill=tk.X,padx=8,pady=8)
        tk.Label(right,text="Containment manifest",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8)
        self.manifest=tk.Listbox(right,height=10,bg="#0F0F0F",fg=LCARS_TEXT); self.manifest.pack(fill=tk.BOTH,padx=8,pady=(8,12),expand=False)
        tk.Label(right,text="Restore",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8)
        rp=tk.Frame(right,bg=LCARS_PANEL); rp.pack(fill=tk.X,padx=8,pady=8)
        self.restore_quar=tk.Entry(rp,width=36,bg="#101010",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.restore_quar.grid(row=0,column=0,padx=(0,8))
        tk.Button(rp,text="Restore quarantine",command=self.do_restore_quarantine,bg=LCARS_ACCENT_B).grid(row=0,column=1)
        self.restore_rule=tk.Entry(rp,width=24,bg="#101010",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.restore_rule.grid(row=1,column=0,padx=(0,8),pady=(8,0))
        tk.Button(rp,text="Remove firewall rule",command=self.do_restore_rule,bg=LCARS_ACCENT_A).grid(row=1,column=1,pady=(8,0))
        tk.Label(right,text="Policies",bg=LCARS_PANEL,fg=LCARS_TEXT,font=("Segoe UI",11,"bold")).pack(anchor="w",padx=8)
        pp=tk.Frame(right,bg=LCARS_PANEL); pp.pack(fill=tk.X,padx=8,pady=8)
        tk.Label(pp,text="Auto enabled",bg=LCARS_PANEL,fg=LCARS_TEXT).grid(row=0,column=0,sticky="w")
        self.auto_enabled=tk.BooleanVar(value=POLICIES.get()["auto"]["enabled"]); tk.Checkbutton(pp,variable=self.auto_enabled,bg=LCARS_PANEL).grid(row=0,column=1)
        tk.Label(pp,text="Expiry minutes",bg=LCARS_PANEL,fg=LCARS_TEXT).grid(row=1,column=0,sticky="w")
        self.expiry_min=tk.Entry(pp,width=6,bg="#101010",fg=LCARS_TEXT,insertbackground=LCARS_TEXT); self.expiry_min.insert(0,str(POLICIES.get()["auto"]["expiry_minutes"])); self.expiry_min.grid(row=1,column=1)
        tk.Button(pp,text="Save policies",command=self.save_policies,bg=LCARS_ACCENT_C).grid(row=2,column=0,columnspan=2,pady=(8,0))
        threading.Thread(target=self.consume_events,daemon=True).start()
        threading.Thread(target=self.refresh_trust_loop,daemon=True).start()
        threading.Thread(target=self.refresh_manifest_loop,daemon=True).start()
    def add_event(self,evt):
        line=f"[{evt['ts']}] {evt['topic']}: {json.dumps(evt['payload'])}\n"; self.feed.insert(tk.END,line); self.feed.see(tk.END)
        t=evt["topic"]
        if t in {"file.exec_new","net.burst_pred","proc.start","proc.lolbin","proc.lolbin_parent"}: self.incidents.insert(tk.END, f"{t} :: {evt['payload']}")
        if t=="dns.anomaly": self.dns_list.insert(tk.END, f"{evt['payload'].get('pid')} :: {evt['payload'].get('host')}")
        if t=="detector.status": self.incidents.insert(tk.END, f"detector :: p={evt['payload'].get('p')} decision={evt['payload'].get('decision')}")
    def consume_events(self):
        for evt in BUS.subscribe(): self.root.after(0,self.add_event,evt)
    def refresh_trust_loop(self):
        while True:
            vec=[secrets.randbelow(3)-1 for _ in range(3)]; score=AGENT.trust_score(vec)
            trust_scaled=max(0.0,min(1.0,(score+1.5)/3.0)); self.root.after(0,self.gauge.set_value,trust_scaled); time.sleep(10)
    def refresh_manifest_loop(self):
        last_len=0
        while True:
            if len(CONTAINMENT_MANIFEST)!=last_len:
                self.manifest.delete(0,tk.END)
                for item in CONTAINMENT_MANIFEST:
                    line = f"Q: {item['src']} -> {item['dst']} sha256={item['sha256']}" if item["type"]=="quarantine" else f"FW: {item['program']} rule={item['rule']}"
                    self.manifest.insert(tk.END,line)
                last_len=len(CONTAINMENT_MANIFEST)
            time.sleep(3)
    # Actions
    def do_quarantine(self):
        path=Path(self.qpath.get().strip())
        if not str(path): messagebox.showinfo("Quarantine","Enter a valid path."); return
        ok=quarantine(path); messagebox.showinfo("Quarantine","File moved to Quarantine." if ok else "Quarantine failed.")
    def do_block(self):
        try:
            pid=int(self.bpid.get().strip()); ctx=proc_context(pid); exe=ctx.get("exe")
            if exe: messagebox.showinfo("Firewall","Rule added." if firewall_block_exe(Path(exe)) else "Failed to add rule.")
            else: messagebox.showinfo("Firewall",f"No executable for PID {pid}")
        except Exception as e: messagebox.showerror("Firewall",f"Error: {e}")
    def do_switch(self): BOT.switch()
    def do_verify_signer(self):
        info=signer_info(Path(self.sv_path.get().strip()))
        if info["error"]: self.sv_result.config(text=f"error={info['error']}", fg=LCARS_WARN)
        else:
            s="signed" if info["signed"] else "unsigned"; signer=info["signer"] or "None"
            self.sv_result.config(text=f"{s} | signer={signer}", fg=LCARS_OK if info["signed"] else LCARS_WARN)
    def do_decode_camo(self):
        decoded=chameleon_decode(self.camo_blob.get().strip()); messagebox.showinfo("Camouflage decode", decoded or "Unable to decode")
    def do_reverse_decrypt(self):
        blob=self.rev_blob.get().strip(); key=self.rev_key.get().strip() or "admin-key"
        try: messagebox.showinfo("Reverse decrypt", reverse_read_decrypt(blob,key))
        except Exception as e: messagebox.showerror("Reverse decrypt", f"Error: {e}")
    def do_restore_quarantine(self):
        ok=restore_quarantine(Path(self.restore_quar.get().strip())); messagebox.showinfo("Restore","Restored." if ok else "Restore failed.")
    def do_restore_rule(self):
        ok=firewall_remove_rule(self.restore_rule.get().strip()); messagebox.showinfo("Firewall","Rule removed." if ok else "Remove failed.")
    def save_policies(self):
        try:
            POLICIES.update({"auto":{"enabled":bool(self.auto_enabled.get()),"reversible_only":True,"expiry_minutes":int(self.expiry_min.get().strip() or POLICIES.get()["auto"]["expiry_minutes"])}})
            messagebox.showinfo("Policies","Saved.")
        except Exception as e: messagebox.showerror("Policies",f"Error: {e}")

# ===== Orchestration =====
def start_all():
    FileSensor().start(); ProcessSensor().start(); NetSensor().start(); BOT.start(); DEVOURER.start()
class DevourerEthical:
    def __init__(self): self.running=False
    def start(self): self.running=True; threading.Thread(target=self._loop, daemon=True).start()
    def _loop(self):
        while self.running:
            p=DETECTOR.probability()
            bern_warn=POLICIES.get()["thresholds"]["bernoulli_warn"]; bern_block=POLICIES.get()["thresholds"]["bernoulli_block_ready"]
            decision=("block_ready" if p>=bern_block else ("warn" if p>=bern_warn else "normal"))
            BUS.publish("detector.status", {"p": round(p,4), "decision": decision})
            time.sleep(5)
DEVOURER = DevourerEthical()

def main():
    log("boot","Starting LCARS ASI Defense (LOLBins guard, ethical autonomy)")
    start_all()
    root=tk.Tk(); DefenseGUI(root); root.mainloop()

if __name__=="__main__":
    main()

