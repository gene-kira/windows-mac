import os, sys, ctypes, subprocess, winreg, tkinter as tk, time, threading, uuid, json, psutil, cv2
from pynput import keyboard, mouse

def elevate():
    try:
        import win32com.shell.shell as shell
        if not ctypes.windll.shell32.IsUserAnAdmin():
            params = ' '.join([f'"{a}"' for a in sys.argv])
            shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=params)
            sys.exit()
    except Exception as e:
        import tkinter.messagebox as mbox
        mbox.showerror("Elevation Failed", f"Admin elevation failed:\n{e}")
        sys.exit(1)

def tag(p, c): return f"{p}::{uuid.uuid4().hex[:6]}_{time.strftime('%H%M%S')}"

class Codex:
    def __init__(self):
        self.trace, self.threats, self.allowed = [], {}, {"US", "CA", "DE"}
        self.ids = {d: f"{d}_{uuid.uuid4().hex[:8]}" for d in ["keyboard", "mouse", "webcam"]}
        self.sync_nodes = [f"NODE_{uuid.uuid4().hex[:4]}" for _ in range(3)]
        self.status = None
        self.memory, self.mutations, self.purge_log = {}, [], []
        self.zero_trust = True
        self.signatures = {"rat.exe", "keylogger", "stealer", "injector", "dropper"}
        self.lineage = {}
        self.mutation_depth = 0

    def log(self, box, msg): self.status.set(msg); box.insert(tk.END, msg + "\n"); box.see(tk.END)
    def add_trace(self, d, e): self.trace.append({"d": d, "e": e, "t": tag(d, e)})
    def tag_ancestry(self, key, origin): return f"{origin}::{key}::R{self.lineage.get(key,0)}::M{self.mutation_depth}"

    def add_threat(self, k, v, origin="US"):
        lvl = "CRITICAL" if any(x in v.lower() for x in self.signatures) else "HIGH" if "camera" in v else "MEDIUM" if "input" in v else "LOW"
        if origin not in self.allowed: lvl = "EXILED"; v += " [Origin Escalated]"
        if k in self.memory:
            self.lineage[k] = self.lineage.get(k, 0) + 1
            if self.lineage[k] > 2:
                lvl = "EXILED"; v += " [EXILED]"
            else:
                lvl = "RESURRECTED"; v += " [RESURRECTED]"
        self.memory[k] = self.memory.get(k, 0) + 1
        self.threats[k] = {"v": v, "lvl": lvl, "origin": origin}
        return lvl

def evolve_signatures(c, sig, log):
    if sig not in c.signatures:
        c.signatures.add(sig)
        log(f"🧬 Signature evolved: {sig}")

def log_mutation(c, key, level):
    c.mutation_depth += 1
    c.mutations.append({
        "id": c.tag_ancestry(key, c.threats[key]["origin"]),
        "level": level,
        "depth": c.mutation_depth,
        "time": time.strftime('%Y-%m-%d %H:%M:%S')
    })

def retain_purge(c, key, level):
    c.purge_log.append({
        "id": c.tag_ancestry(key, c.threats[key]["origin"]),
        "level": level,
        "time": time.strftime('%Y-%m-%d %H:%M:%S')
    })

def swarm_sync(c, log):
    for node in c.sync_nodes:
        log(f"🔁 Swarm Sync → {node} :: {tag('SYNC', node)}")
        log(f"📡 Mutation broadcast: {len(c.signatures)} signatures")

def mutation_cascade(c, key, sig, log):
    evolve_signatures(c, sig, log)
    log_mutation(c, key, c.threats[key]["lvl"])
    swarm_sync(c, log)
    recursive_scan(c, log)

def swarm_resurrection_sync(c, log):
    for k, v in c.threats.items():
        if v["lvl"] == "RESURRECTED":
            ancestry = c.tag_ancestry(k, v["origin"])
            for node in c.sync_nodes:
                log(f"🕸️ Resurrection Sync → {node} :: {ancestry}")

def mutate_velocity(c, log):
    for k, v in c.lineage.items():
        if v > 5:
            new_sig = f"mutant_{k.split('_')[0]}"
            evolve_signatures(c, new_sig, log)
            log(f"⚙️ Velocity Mutation: {new_sig} added")

def threat_prophecy(c, log):
    for k, v in c.lineage.items():
        if v >= 3:
            log(f"🔮 Prophecy: {k} will resurrect again. Mutation depth: {c.mutation_depth}")

def recursive_scan(c, log):
    for p in psutil.process_iter(['pid', 'name']):
        try:
            name = p.info['name']
            if any(sig in name.lower() for sig in c.signatures):
                key = f"proc_{name}_{p.info['pid']}"
                lvl = c.add_threat(key, name, origin="US")
                log(f"🔍 Recursive Scan → {key} [{lvl}]")
                log_mutation(c, key, lvl)
        except: continue

def predictive_scan(c, log):
    for p in psutil.process_iter(['pid', 'name', 'ppid']):
        try:
            name, ppid = p.info['name'], p.info['ppid']
            parent = psutil.Process(ppid).name() if ppid else "Unknown"
            if any(x in name.lower() for x in c.signatures) or "powershell" in parent.lower():
                key = f"prethreat_{name}_{p.info['pid']}"
                lvl = c.add_threat(key, f"{name} via {parent}", origin="US")
                log(f"🧠 Predictive Threat → {key} [{lvl}]")
        except: continue

def recursive_devour(c, log):
    for k, v in list(c.threats.items()):
        if v["lvl"] in ["CRITICAL", "BLOCKED", "RESURRECTED", "EXILED"]:
            try:
                sig = next((s for s in c.signatures if s in v["v"].lower()), None)
                if sig:
                    mutation_cascade(c, k, sig, log)
                if "net_" in k:
                    ip = k.split("_", 1)[1]
                    subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule", f"name=CodexBlock_{ip}", "dir=out", "action=block", f"remoteip={ip}"], check=True)
                    log(f"🚫 IP Blocked: {ip}")
                elif sig:
                    for p in psutil.process_iter(['pid', 'name']):
                        if sig in p.info['name'].lower():
                            psutil.Process(p.info['pid']).kill()
                            log(f"💀 Terminated: {p.info['name']} (PID {p.info['pid']})")
                retain_purge(c, k, v["lvl"])
                swarm_resurrection_sync(c, log)
                mutate_velocity(c, log)
                del c.threats[k]
            except Exception as e:
                log(f"⚠️ Devour failed for {k}: {e}")

def purge_loop(c, log):
    while True:
        if c.zero_trust: recursive_devour(c, log)
        time.sleep(3)

def predictive_loop(c, log):
    while True:
        predictive_scan(c, log)
        time.sleep(10)

def prophecy_loop(c, log):
    while True:
        threat_prophecy(c, log)
        time.sleep(30)

def launch():
    c = Codex()
    root = tk.Tk(); root.title("Codex Devourer Omega Prime"); root.geometry("720x880")
    c.status = tk.StringVar(value="🧠 Sovereign Monitoring Active")
    box = tk.Text(root, height=10, width=85, font=("Consolas", 9)); box.pack()
    log = lambda m: c.log(box, m)

    def scan():
        if os.getlogin().lower() not in ["admin", "codex"]: log("🔒 Persona clearance failed."); return
        try: k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Windows\\CurrentVersion\\Diagnostics"); i = 0; found = []
        except: found = []
        else:
            while True:
                try: found.append(winreg.EnumKey(k, i)); i += 1
                except: break
            k.Close()
            for sub in found: log(f"🧹 Purged: {sub}")
        for n in c.sync_nodes: log(f"🔁 {tag('SYNC', n)}")

    def webcam_loop():
        cap = cv2.VideoCapture(0)
        while True:
            ret, _ = cap.read()
            if ret:
                c.add_trace("webcam", "frame")
                lvl = c.add_threat("camera_access", "Webcam active", origin="US")
                log(f"📷 {tag('WEBCAM', 'frame')} [{lvl}]")
            time.sleep(10)

    def gui():
        progress = tk.DoubleVar()
        tk.Scale(root, variable=progress, from_=0, to=100, orient="horizontal", length=400, showvalue=0).pack(pady=10)
        def animate(ms): steps = 100; interval = ms // steps
        def tick(i=0): progress.set(i); root.after(interval, lambda: tick(i + 1)) if i <= steps else None
        tk.Button(root, text="Run Scan + Sync", command=lambda: scan() or tick(), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Run Defender Scan", command=lambda: defender_scan(c, box), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Export Trace Log", command=lambda: log(json.dumps(c.trace, indent=2)), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Purge All Traces", command=lambda: c.trace.clear() or c.threats.clear() or log("🔥 Traces purged."), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Show Ghost IDs", command=lambda: [log(f"{d}: {c.ids[d]}") for d in c.ids], font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Show Threat Matrix", command=lambda: show_threats(c), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Country Filter Panel", command=lambda: country_panel(c, log), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Show Mutation Registry", command=lambda: show_mutations(c), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Show Escalation Lineage", command=lambda: show_lineage(c), font=("Consolas", 10)).pack(pady=5)
        tk.Button(root, text="Show Sovereign Rituals", command=lambda: show_sovereign(c), font=("Consolas", 10)).pack(pady=5)
        zero_var = tk.BooleanVar(value=c.zero_trust)
        tk.Checkbutton(root, text="☠️ Zero-Trust Mode", variable=zero_var, command=lambda: setattr(c, 'zero_trust', zero_var.get()), font=("Consolas", 10)).pack(pady=5)
        tk.Label(root, textvariable=c.status, wraplength=640, font=("Consolas", 10), fg="blue").pack(pady=10)

    def show_threats(c):
        top = tk.Toplevel(root); top.title("Threat Matrix")
        for k, v in c.threats.items():
            ancestry = c.tag_ancestry(k, v["origin"])
            color = "gray" if v["lvl"] == "BLOCKED" else {"LOW":"green","MEDIUM":"orange","HIGH":"red","CRITICAL":"darkred","RESURRECTED":"purple","EXILED":"black"}[v["lvl"]]
            tk.Label(top, text=f"{ancestry}: {v['lvl']} → {v['v']}", font=("Consolas", 9), fg=color).pack()

    def show_mutations(c):
        top = tk.Toplevel(root); top.title("Mutation Registry")
        for sig in sorted(c.signatures):
            tk.Label(top, text=f"🧬 {sig}", font=("Consolas", 9), fg="purple").pack()

    def show_lineage(c):
        top = tk.Toplevel(root); top.title("Resurrection Audit")
        for k, v in c.lineage.items():
            ancestry = c.tag_ancestry(k, c.threats[k]["origin"])
            tk.Label(top, text=f"{k}: {v} resurrections → {ancestry}", font=("Consolas", 9), fg="darkred").pack()
        tk.Label(top, text=f"Mutation Depth: {c.mutation_depth}", font=("Consolas", 10), fg="blue").pack(pady=10)

    def show_sovereign(c):
        top = tk.Toplevel(root); top.title("Sovereign Rituals")
        for k, v in c.lineage.items():
            if v >= 3:
                ancestry = c.tag_ancestry(k, c.threats[k]["origin"])
                tk.Label(top, text=f"🔮 {k}: {v} resurrections → {ancestry}", font=("Consolas", 9), fg="darkred").pack()
        tk.Label(top, text=f"Mutation Depth: {c.mutation_depth}", font=("Consolas", 10), fg="blue").pack(pady=10)

    threading.Thread(target=lambda: keyboard.Listener(on_press=lambda k: c.add_trace("keyboard", str(k)) or log(f"⌨️ {tag('KEY', str(k))}")).start(), daemon=True).start()
    threading.Thread(target=lambda: mouse.Listener(on_click=lambda x,y,b,p: p and c.add_trace("mouse", f"{b}@{x},{y}") or log(f"🖱️ {tag('MOUSE', str(b))}")).start(), daemon=True).start()
    threading.Thread(target=webcam_loop, daemon=True).start()
    threading.Thread(target=lambda: sniff(c, box), daemon=True).start()
    threading.Thread(target=lambda: monitor(c, box), daemon=True).start()
    threading.Thread(target=lambda: purge_loop(c, log), daemon=True).start()
    threading.Thread(target=lambda: predictive_loop(c, log), daemon=True).start()
    threading.Thread(target=lambda: prophecy_loop(c, log), daemon=True).start()

    gui()
    root.mainloop()

if __name__ == "__main__":
    elevate()
    launch()

