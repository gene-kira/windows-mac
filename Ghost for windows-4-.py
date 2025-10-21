import os, sys, ctypes, subprocess, winreg, tkinter as tk, time, hashlib, threading, uuid, json, socket, psutil, cv2
from pynput import keyboard, mouse

def elevate():
    try: import win32com.shell.shell as shell
    except: subprocess.run([sys.executable,"-m","pip","install","pywin32"]); import win32com.shell.shell as shell
    if not ctypes.windll.shell32.IsUserAnAdmin():
        shell.ShellExecuteEx(lpVerb='runas',lpFile=sys.executable,lpParameters=' '.join([f'"{a}"' for a in sys.argv])); sys.exit()

def tag(p,c): return f"{p}::{hashlib.sha256(c.encode()).hexdigest()[:6]}_{time.strftime('%H%M%S')}"

class Ghost:
    def __init__(self):
        self.ids = {d: f"{d}_{uuid.uuid4().hex[:8]}" for d in ["keyboard", "mouse", "webcam"]}

class Trace:
    def __init__(self):
        self.log = []
    def add(self, device, event):
        self.log.append({"d": device, "e": event, "t": tag(device, event)})
    def purge(self):
        self.log.clear()
    def dump(self):
        return json.dumps(self.log, indent=2)

class Threat:
    def __init__(self):
        self.streams = {}
    def add(self, key, value):
        level = "HIGH" if "camera" in value else "MEDIUM" if "input" in value else "LOW"
        self.streams[key] = {"v": value, "t": tag("STREAM", key), "lvl": level}
        return level
    def purge(self):
        self.streams.clear()

class Sync:
    def __init__(self):
        self.nodes = [f"NODE_{uuid.uuid4().hex[:4]}" for _ in range(3)]
        self.rules = {"purge": ["ScreenClip", "Telemetry"]}
    def sync(self, log):
        for n in self.nodes:
            for r in self.rules["purge"]:
                log(f"🔁 {tag('SYNC', f'{n}:{r}')}")

def set_reg(r,p,n,v,t=winreg.REG_DWORD):
    try:
        k = winreg.CreateKeyEx(r, p, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(k, n, 0, t, v)
        k.Close()
        return True
    except:
        return False

def del_key(r,p,s):
    try:
        winreg.DeleteKey(winreg.OpenKey(r, p, 0, winreg.KEY_ALL_ACCESS), s)
        return True
    except:
        return False

def purge_files(log):
    for d in [r"%LOCALAPPDATA%\Microsoft\Clipboard", r"%LOCALAPPDATA%\Packages"]:
        for r, _, fs in os.walk(os.path.expandvars(d)):
            for f in fs:
                if "ScreenClip" in f or "Clipboard" in r:
                    try: os.remove(os.path.join(r, f)); log(f"🧹 {f}")
                    except: log(f"⚠️ {f}")

def sniff(log):
    for c in psutil.net_connections(kind='inet'):
        if c.status == "ESTABLISHED" and c.raddr:
            ip, port = c.raddr.ip, c.raddr.port
            proc = psutil.Process(c.pid).name() if c.pid else "Unknown"
            log(f"🌐 {tag('NET', f'{ip}:{port}')} → {proc}")

def monitor(log):
    for p in psutil.process_iter(['name', 'username']):
        try: n, u = p.info['name'], p.info['username']
        except: continue
        if "admin" in u.lower() or "elev" in n.lower():
            log(f"⚠️ {tag('PROC', n)} by {u}")

def launch():
    g, t, h, s = Ghost(), Trace(), Threat(), Sync()
    root = tk.Tk(); root.title("Codex Phantom Shell"); root.geometry("720x880")
    status = tk.StringVar(value="🧠 Monitoring...")
    logbox = tk.Text(root, height=10, width=85, font=("Consolas", 9)); logbox.pack()
    def log(m): status.set(m); logbox.insert(tk.END, m + "\n"); logbox.see(tk.END); root.update_idletasks()

    glyphs = ["🧿", "👁️", "🔄"]
    status_label = tk.Label(root, text="🧿 Idle", font=("Consolas", 12), fg="darkgreen"); status_label.pack()
    def set_status(t, c="darkgreen"): status_label.config(text=t, fg=c)
    def pulse(base="Idle", delay=500):
        def loop(i=0): status_label.config(text=f"{glyphs[i % 3]} {base}"); root.after(delay, lambda: loop(i + 1))
        loop()

    pulse("Idle")

    progress = tk.DoubleVar()
    tk.Scale(root, variable=progress, from_=0, to=100, orient="horizontal", length=400, showvalue=0).pack(pady=10)
    def animate(ms):
        steps = 100
        interval = ms // steps
        def tick(i=0):
            if i <= steps:
                progress.set(i)
                root.after(interval, lambda: tick(i + 1))
        tick()

    toggles = tk.Frame(root); toggles.pack()
    def toggle(s):
        f = tk.Frame(toggles); f.pack()
        tk.Label(f, text=f"{s['label']} {tag('REG', s['name'])}", font=("Consolas", 10), width=50, anchor="w").pack(side=tk.LEFT)
        for label, val in [("Allow", s["on"]), ("Deny", s["off"])]:
            tk.Button(f, text=label, width=10, command=lambda v=val: log(f"{s['label']} {label}: {'✅' if set_reg(winreg.HKEY_CURRENT_USER, s['path'], s['name'], v, s.get('type', winreg.REG_DWORD)) else '⚠️'}")).pack(side=tk.LEFT)

    for s in [
        {"label":"Clipboard History","path":r"Software\\Microsoft\\Clipboard","name":"EnableClipboardHistory","on":1,"off":0},
        {"label":"Screen Clipping","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\ScreenClipBlock","name":"BlockScreenClip","on":0,"off":1},
        {"label":"Tailored Experiences","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy","name":"TailoredExperiencesWithDiagnosticDataEnabled","on":1,"off":0},
        {"label":"Advertising ID","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo","name":"Enabled","on":1,"off":0},
        {"label":"Location Services","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location","name":"Value","on":"Allow","off":"Deny","type":winreg.REG_SZ}
    ]: toggle(s)

    def scan():
        set_status("🔍 Scanning...", "orange"); animate(3000); purge_files(log)
        base = r"Software\\Microsoft\\Windows\\CurrentVersion\\Diagnostics"
        try: k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, base); i = 0; found = []
        except: found = []
        else:
            while True:
                try: found.append(winreg.EnumKey(k, i)); i += 1
                except: break
            k.Close()
            for sub in found: log(f"{'🧹 Purged' if del_key(winreg.HKEY_CURRENT_USER, base, sub) else '⚠️'}: {sub}")
        s.sync(log); set_status("🧿 Idle", "darkgreen")

    def start_daemons():
        def on_key(k):
            try: c = k.char if hasattr(k, 'char') and k.char else str(k)
            except: c = str(k)
            t.add("keyboard", c); l = h.add("keyboard_input", c)
            set_status("⌨️ Key", {"LOW":"green","MEDIUM":"orange","HIGH":"red"}[l]); log(f"⌨️ {tag('KEY', c)} [{l}]")

        def on_click(x, y, b, p):
            if p:
                e = f"{b} @({x},{y})"; t.add("mouse", e); l = h.add("mouse_input", e)
                set_status("🖱️ Click", {"LOW":"green","MEDIUM":"orange","HIGH":"red"}[l]); log(f"🖱️ {tag('MOUSE',e)} [{l}]")

        def webcam():
            cap = cv2.VideoCapture(0)
            while True:
                ret, _ = cap.read()
                if ret:
                    t.add("webcam", "frame")
                    l = h.add("camera_access", "Webcam active")
                    set_status("📷 Webcam", {"LOW":"green","MEDIUM":"orange","HIGH":"red"}[l])
                    log(f"📷 {tag('WEBCAM', 'frame')} [{l}]")
                time.sleep(10)

        def net_loop():
            while True:
                sniff(log)
                time.sleep(10)

        def proc_loop():
            while True:
                monitor(log)
                time.sleep(5)

        threading.Thread(target=lambda: keyboard.Listener(on_press=on_key).start(), daemon=True).start()
        threading.Thread(target=lambda: mouse.Listener(on_click=on_click).start(), daemon=True).start()
        threading.Thread(target=webcam, daemon=True).start()
        threading.Thread(target=net_loop, daemon=True).start()
        threading.Thread(target=proc_loop, daemon=True).start()

    start_daemons()

    tk.Button(root, text="Run Scan + Sync", command=scan, font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Export Trace Log", command=lambda: log(t.dump()), font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Purge All Traces", command=lambda: t.purge() or h.purge() or log("🔥 Traces purged."), font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Show Ghost IDs", command=lambda: [log(f"{d}: {g.ids[d]}") for d in g.ids], font=("Consolas", 10)).pack(pady=5)
    tk.Label(root, textvariable=status, wraplength=640, font=("Consolas", 10), fg="blue").pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    elevate()
    launch()



