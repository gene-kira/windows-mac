import os, sys, ctypes, subprocess, winreg, tkinter as tk
from tkinter import ttk

def elevate():  # 🔐 Proper UAC elevation
    try:
        import win32com.shell.shell as shell
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
        import win32com.shell.shell as shell
    if not ctypes.windll.shell32.IsUserAnAdmin():
        shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=' '.join([f'"{a}"' for a in sys.argv]))
        sys.exit()

def set_reg(root, path, name, val, typ=winreg.REG_DWORD):
    try: k = winreg.CreateKeyEx(root, path, 0, winreg.KEY_SET_VALUE); winreg.SetValueEx(k, name, 0, typ, val); k.Close(); return True
    except: return False

def del_reg_key(root, path, sub): 
    try: winreg.DeleteKey(winreg.OpenKey(root, path, 0, winreg.KEY_ALL_ACCESS), sub); return True
    except: return False

def purge_clip_and_screenclip(log):
    clip = os.path.join(os.getenv('LOCALAPPDATA'), 'Microsoft', 'Clipboard')
    for f in os.listdir(clip) if os.path.exists(clip) else []:
        try: os.remove(os.path.join(clip, f)); log(f"🧹 Deleted clipboard: {f}")
        except: log(f"⚠️ Failed clipboard: {f}")
    pkg = os.path.join(os.getenv('LOCALAPPDATA'), 'Packages')
    for r, _, fs in os.walk(pkg):
        for f in fs:
            if "ScreenClip" in f:
                try: os.remove(os.path.join(r, f)); log(f"🧹 Deleted screenclip: {f}")
                except: log(f"⚠️ Failed screenclip: {f}")

def launch_gui():
    root = tk.Tk(); root.title("Codex Privacy Sentinel"); root.geometry("620x800")
    status = tk.StringVar(); status.set("🧠 Monitoring...")
    logbox = tk.Text(root, height=10, width=75, font=("Consolas", 9)); logbox.pack(pady=10)
    def log(msg): status.set(msg); logbox.insert(tk.END, msg + "\n"); logbox.see(tk.END); root.update_idletasks()
    overlay = tk.Label(root, text="🧿 Idle", font=("Consolas", 12), fg="darkgreen"); overlay.pack()
    countdown = tk.Label(root, text="", font=("Consolas", 10), fg="darkred"); countdown.pack()
    toggles = tk.Frame(root); toggles.pack(pady=10)
    known = [
        {"label":"Clipboard History","path":r"Software\\Microsoft\\Clipboard","name":"EnableClipboardHistory","on":1,"off":0},
        {"label":"Screen Clipping","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\ScreenClipBlock","name":"BlockScreenClip","on":0,"off":1},
        {"label":"Tailored Experiences","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy","name":"TailoredExperiencesWithDiagnosticDataEnabled","on":1,"off":0},
        {"label":"Advertising ID","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo","name":"Enabled","on":1,"off":0},
        {"label":"Location Services","path":r"Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\location","name":"Value","on":"Allow","off":"Deny","type":winreg.REG_SZ}
    ]
    seen = set()
    def add_toggle(s):
        key = f"{s['path']}::{s['name']}"
        if key in seen: return
        seen.add(key)
        f = tk.Frame(toggles); f.pack()
        tk.Label(f, text=s["label"], font=("Consolas", 10), width=30, anchor="w").pack(side=tk.LEFT)
        tk.Button(f, text="Enable", command=lambda: log(f"{s['label']} enabled: {'✅' if set_reg(winreg.HKEY_CURRENT_USER, s['path'], s['name'], s['on'], s.get('type', winreg.REG_DWORD)) else '⚠️'}"), width=10).pack(side=tk.LEFT)
        tk.Button(f, text="Disable", command=lambda: log(f"{s['label']} disabled: {'✅' if set_reg(winreg.HKEY_CURRENT_USER, s['path'], s['name'], s['off'], s.get('type', winreg.REG_DWORD)) else '⚠️'}"), width=10).pack(side=tk.LEFT)
    for s in known: add_toggle(s)

    def scan_and_purge():
        overlay.config(text="🧿 Scanning...", fg="orange")
        purge_clip_and_screenclip(log)
        base = r"Software\\Microsoft\\Windows\\CurrentVersion\\Diagnostics"
        try:
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, base); i = 0; found = []
            while True:
                try: found.append(winreg.EnumKey(k, i)); i += 1
                except: break
            k.Close()
        except: found = []
        log(f"🔍 Found {len(found)} telemetry keys.")
        for sub in found:
            log(f"{'🧹 Purged' if del_reg_key(winreg.HKEY_CURRENT_USER, base, sub) else '⚠️ Failed'}: {sub}")
        overlay.config(text="🧿 Idle", fg="darkgreen")

    def discover_keys():
        overlay.config(text="🧿 Discovering...", fg="blue")
        for path in [r"Software\\Microsoft\\Windows\\CurrentVersion\\Privacy", r"Software\\Microsoft\\Windows\\CurrentVersion\\AdvertisingInfo", r"Software\\Microsoft\\Clipboard"]:
            try:
                k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path); i = 0
                while True:
                    try:
                        name, val, typ = winreg.EnumValue(k, i)
                        add_toggle({"label":f"{path.split('\\\\')[-1]}: {name}", "path":path, "name":name, "on":val, "off":0 if isinstance(val,int) else "Deny", "type":typ})
                        i += 1
                    except: break
                k.Close()
            except: continue
        overlay.config(text="🧿 Idle", fg="darkgreen"); log("🔍 Key discovery complete.")

    def manual_scan(ms):
        overlay.config(text="⏳ Waiting...", fg="blue")
        t = [ms//1000]
        def tick():
            if t[0] > 0: countdown.config(text=f"⏳ Manual scan in {t[0]}s"); t[0] -= 1; root.after(1000, tick)
            else: countdown.config(text="⏳ Executing..."); scan_and_purge(); discover_keys()
        tick()

    tk.Label(root, text="Manual Scan Presets:", font=("Consolas", 10)).pack()
    f = tk.Frame(root); f.pack()
    for label, ms in [("1min",60000),("1hr",3600000),("1day",86400000)]:
        tk.Button(f, text=label, command=lambda m=ms: manual_scan(m), width=6).pack(side=tk.LEFT, padx=2)

    interval = tk.IntVar(value=60000); auto_t = [interval.get()//1000]
    def auto_tick():
        if auto_t[0] > 0: countdown.config(text=f"⏳ Auto scan in {auto_t[0]}s"); auto_t[0] -= 1; root.after(1000, auto_tick)
        else: countdown.config(text="⏳ Executing auto scan..."); scan_and_purge(); discover_keys(); auto_t[0] = interval.get()//1000; auto_tick(); root.after(interval.get(), daemon)

    def daemon(): overlay.config(text="🧿 Auto-Scan", fg="blue"); auto_tick()
    def set_interval(ms): interval.set(ms); auto_t[0] = ms//1000; log(f"🔁 Auto-scan set to {ms//1000}s")

    tk.Label(root, text="Auto Scan Interval:", font=("Consolas", 10)).pack()
    f = tk.Frame(root); f.pack()
    for label, ms in [("1min",60000),("1hr",3600000),("1day",86400000)]:
        tk.Button(f, text=label, command=lambda m=ms: set_interval(m), width=6).pack(side=tk.LEFT, padx=2)

    tk.Button(root, text="Start Auto-Scan Daemon", command=daemon, font=("Consolas", 10)).pack(pady=10)
    tk.Button(root, text="Scan for New Privacy Keys", command=discover_keys, font=("Consolas", 10)).pack(pady=5)
    tk.Label(root, textvariable=status, wraplength=540, font=("Consolas", 10), fg="blue").pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    elevate()
    launch_gui()



