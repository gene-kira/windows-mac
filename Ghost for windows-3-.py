import os, sys, ctypes, subprocess, winreg, tkinter as tk, time, hashlib, threading, uuid, json
from pynput import keyboard, mouse
import cv2

def elevate():
    try: import win32com.shell.shell as shell
    except: subprocess.run([sys.executable, "-m", "pip", "install", "pywin32"]); import win32com.shell.shell as shell
    if not ctypes.windll.shell32.IsUserAnAdmin():
        shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=' '.join([f'"{a}"' for a in sys.argv])); sys.exit()

def tag(prefix, content):
    h = hashlib.sha256(content.encode()).hexdigest()[:6]
    t = time.strftime('%H%M%S')
    return f"{prefix}::{h}_{t}"

class Ghost:
    def __init__(self):
        self.ids = {d: f"{d}_{uuid.uuid4().hex[:8]}" for d in ["keyboard", "mouse", "webcam"]}

class Trace:
    def __init__(self):
        self.log = []
    def add(self, device, event):
        self.log.append({"device": device, "event": event, "symbol": tag(device, event)})
    def purge(self):
        self.log.clear()
    def dump(self):
        return json.dumps(self.log, indent=2)

class Threat:
    def __init__(self):
        self.streams = {}
    def add(self, key, value):
        level = "HIGH" if "camera" in value else "MEDIUM" if "input" in value else "LOW"
        self.streams[key] = {"value": value, "symbol": tag("STREAM", key), "level": level}
        return level
    def purge(self):
        self.streams.clear()

def set_reg(root, path, name, value, reg_type=winreg.REG_DWORD):
    try:
        key = winreg.CreateKeyEx(root, path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, name, 0, reg_type, value)
        key.Close()
        return True
    except:
        return False

def del_key(root, path, subkey):
    try:
        winreg.DeleteKey(winreg.OpenKey(root, path, 0, winreg.KEY_ALL_ACCESS), subkey)
        return True
    except:
        return False

def purge_files(log):
    for d in [r"%LOCALAPPDATA%\Microsoft\Clipboard", r"%LOCALAPPDATA%\Packages"]:
        for r, _, fs in os.walk(os.path.expandvars(d)):
            for f in fs:
                if "ScreenClip" in f or "Clipboard" in r:
                    try:
                        os.remove(os.path.join(r, f))
                        log(f"🧹 {f}")
                    except:
                        log(f"⚠️ {f}")

def launch():
    ghost, trace, threat = Ghost(), Trace(), Threat()
    root = tk.Tk()
    root.title("Codex Phantom Shell")
    root.geometry("720x880")
    status = tk.StringVar(value="🧠 Monitoring...")
    logbox = tk.Text(root, height=10, width=85, font=("Consolas", 9))
    logbox.pack()

    def log(msg):
        status.set(msg)
        logbox.insert(tk.END, msg + "\n")
        logbox.see(tk.END)
        root.update_idletasks()

    glyphs = ["🧿", "👁️", "🔄"]
    status_label = tk.Label(root, text="🧿 Idle", font=("Consolas", 12), fg="darkgreen")
    status_label.pack()

    def set_status(text, color="darkgreen"):
        status_label.config(text=text, fg=color)

    def pulse(base="Idle", delay=500):
        def loop(i=0):
            status_label.config(text=f"{glyphs[i % len(glyphs)]} {base}")
            root.after(delay, lambda: loop(i + 1))
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

    toggles = tk.Frame(root)
    toggles.pack()

    def toggle(s):
        f = tk.Frame(toggles)
        f.pack()
        tk.Label(f, text=f"{s['label']} {tag('REG', s['name'])}", font=("Consolas", 10), width=50, anchor="w").pack(side=tk.LEFT)
        for label, val in [("Allow", s["on"]), ("Deny", s["off"])]:
            tk.Button(f, text=label, width=10, command=lambda v=val: log(f"{s['label']} {label}: {'✅' if set_reg(winreg.HKEY_CURRENT_USER, s['path'], s['name'], v, s.get('type', winreg.REG_DWORD)) else '⚠️'}")).pack(side=tk.LEFT)

    for s in [
        {"label":"Clipboard History","path":r"Software\\Microsoft\\Clipboard","name":"EnableClipboardHistory","on":1,"off":0},
        {"label":"Screen Clipping","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\ScreenClipBlock","name":"BlockScreenClip","on":0,"off":1},
        {"label":"Tailored Experiences","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy","name":"TailoredExperiencesWithDiagnosticDataEnabled","on":1,"off":0},
        {"label":"Advertising ID","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo","name":"Enabled","on":1,"off":0},
        {"label":"Location Services","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location","name":"Value","on":"Allow","off":"Deny","type":winreg.REG_SZ}
    ]:
        toggle(s)

    def scan():
        set_status("🔍 Scanning...", "orange")
        animate(3000)
        purge_files(log)
        base = r"Software\\Microsoft\\Windows\\CurrentVersion\\Diagnostics"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, base)
            i = 0
            found = []
        except:
            found = []
        else:
            while True:
                try:
                    found.append(winreg.EnumKey(key, i))
                    i += 1
                except:
                    break
            key.Close()
            for sub in found:
                result = del_key(winreg.HKEY_CURRENT_USER, base, sub)
                log(f"{'🧹 Purged' if result else '⚠️'}: {sub}")
        set_status("🧿 Idle", "darkgreen")

    def start_daemons():
        def on_key(key):
            try:
                char = key.char if hasattr(key, 'char') and key.char else str(key)
            except:
                char = str(key)
            trace.add("keyboard", char)
            lvl = threat.add("keyboard_input", char)
            set_status("⌨️ Key", {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[lvl])
            log(f"⌨️ {tag('KEY', char)} [{lvl}]")

        def on_click(x, y, b, p):
            if p:
                event = f"{b} @({x},{y})"
                trace.add("mouse", event)
                lvl = threat.add("mouse_input", event)
                set_status("🖱️ Click", {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[lvl])
                log(f"🖱️ {tag('MOUSE', event)} [{lvl}]")

        def webcam():
            cap = cv2.VideoCapture(0)
            while True:
                ret, _ = cap.read()
                if ret:
                    trace.add("webcam", "frame")
                    lvl = threat.add("camera_access", "Webcam active")
                    set_status("📷 Webcam", {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[lvl])
                    log(f"📷 {tag('WEBCAM', 'frame')} [{lvl}]")
                time.sleep(10)

        threading.Thread(target=lambda: keyboard.Listener(on_press=on_key).start(), daemon=True).start()
        threading.Thread(target=lambda: mouse.Listener(on_click=on_click).start(), daemon=True).start()
        threading.Thread(target=webcam, daemon=True).start()

    start_daemons()

    tk.Button(root, text="Export Trace Log", command=lambda: log(trace.dump()), font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Purge All Traces", command=lambda: trace.purge() or threat.purge() or log("🔥 Traces purged."), font=("Consolas", 10)).pack(pady=5)
    tk.Button(root, text="Show Ghost IDs", command=lambda: [log(f"{d}: {ghost.ids[d]}") for d in ghost.ids], font=("Consolas", 10)).pack(pady=5)
    tk.Label(root, textvariable=status, wraplength=640, font=("Consolas", 10), fg="blue").pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    elevate()
    launch()



