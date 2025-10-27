import subprocess, os, winreg, tkinter as tk
from tkinter import ttk, messagebox
import threading, time
from datetime import datetime

# 🔮 Utility: Overlay + Clearance
def overlay(symbol): tk.Toplevel().after(1500, lambda w=tk.Canvas(tk.Toplevel(), bg="black").pack(): w.create_text(100,100,text=symbol,font=("Consolas",48),fill="red"))
def clearance(cb): 
    def v(): entry.get()=="mythicpass" and (c.destroy(), cb()) or messagebox.showerror("Denied","Invalid passphrase")
    c, entry = tk.Toplevel(), tk.Entry(tk.Toplevel(), show="*"); tk.Label(c,text="Enter Passphrase").pack(); entry.pack(); tk.Button(c,text="Validate",command=v).pack()

# 🔁 Sync Panel + Bot
class SyncBot:
    def __init__(self, root):
        self.f = ttk.LabelFrame(root, text="🔁 Daemon Sync")
        self.s = tk.Label(self.f, text="Last Scan: --", font=("Consolas", 10)); self.s.pack()
        self.p = tk.Label(self.f, text="🧬", font=("Consolas", 32), fg="green"); self.p.pack()
        self.log = tk.Text(self.f, height=5, width=50); self.log.pack(); self.f.pack(padx=10, pady=10, fill='x')
        threading.Thread(target=self.scan_loop, daemon=True).start()
        threading.Thread(target=self.monitor_loop, daemon=True).start()

    def scan_loop(self):
        while True: self.scan(); time.sleep(86400)

    def scan(self):
        self.s.config(text=f"Last Scan: {datetime.now().strftime('%H:%M:%S')}")
        self.p.config(fg="red"); self.log.insert("end", f"[{datetime.now()}] 🔍 Scan complete\n"); self.log.see("end")
        try:
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Advanced")
            [print(f"[Hidden] {winreg.EnumValue(k,i)[0]}") for i in range(100)]
        except: pass
        if "0x1" in subprocess.run(["reg", "query", r"HKLM\Software\Policies\Microsoft\Windows\DataCollection", "/v", "AllowTelemetry"], capture_output=True, text=True).stdout:
            print("[Privacy Threat] Telemetry ENABLED")

    def monitor_loop(self):
        while True:
            try:
                k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run")
                [print(f"[Startup] {winreg.EnumValue(k,i)[0]}") for i in range(100)]
            except: pass
            time.sleep(300)

# 🧩 Panels
def panel(root, title, actions):
    f = ttk.LabelFrame(root, text=title)
    for label, func, secure, symbol in actions:
        cmd = lambda f=func: [f(), overlay(symbol)]
        ttk.Button(f, text=label, command=(lambda: clearance(cmd) if secure else cmd)).pack(fill='x')
    f.pack(padx=10, pady=10, fill='x')

# 🛡️ Codex Shell
class CodexShell:
    def __init__(self):
        self.root = tk.Tk(); self.root.title("Codex Control Shell"); self.root.geometry("600x800"); self.root.resizable(False, False)
        SyncBot(self.root)

        panel(self.root, "🎮 Gaming Mode", [
            ("Enable Game Mode", lambda: subprocess.run(["reg", "add", r"HKCU\Software\Microsoft\GameBar", "/v", "AllowAutoGameMode", "/t", "REG_DWORD", "/d", "1", "/f"]), False, "🎮"),
            ("Enable GPU Scheduling", lambda: subprocess.run(["reg", "add", r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers", "/v", "HwSchMode", "/t", "REG_DWORD", "/d", "2", "/f"]), False, "⚙️"),
            ("Set High Performance Plan", lambda: subprocess.run(["powercfg", "/setactive", "SCHEME_MAX"]), False, "⚡")
        ])

        panel(self.root, "🔒 Privacy Purge", [
            ("Disable Telemetry", lambda: subprocess.run(["reg", "add", r"HKLM\Software\Policies\Microsoft\Windows\DataCollection", "/v", "AllowTelemetry", "/t", "REG_DWORD", "/d", "0", "/f"]), True, "🚫"),
            ("Disable Ad Tracking", lambda: subprocess.run(["reg", "add", r"HKCU\Software\Microsoft\Windows\CurrentVersion\AdvertisingInfo", "/v", "Enabled", "/t", "REG_DWORD", "/d", "0", "/f"]), False, "🚫"),
            ("Clear Activity History", lambda: subprocess.run(["reg", "delete", r"HKCU\Software\Microsoft\Windows\CurrentVersion\ActivityHistory", "/f"]), False, "🧹")
        ])

        panel(self.root, "🧹 Bloatware Scanner", [
            ("Remove Bloatware", lambda: [subprocess.run(["powershell", "-Command", f"Get-AppxPackage *{app}* | Remove-AppxPackage"]) for app in [
                "Microsoft.ZuneMusic", "Microsoft.XboxGameCallableUI", "Microsoft.Xbox.TCUI",
                "Microsoft.XboxApp", "Microsoft.XboxGameOverlay", "Microsoft.XboxGamingOverlay"
            ]], False, "🔥")
        ])

        panel(self.root, "🧠 Hidden Controls", [
            ("Activate God Mode", lambda: os.makedirs("GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}", exist_ok=True), False, "🧠"),
            ("Enable Clipboard History", lambda: subprocess.run(["reg", "add", r"HKCU\Software\Microsoft\Clipboard", "/v", "EnableClipboardHistory", "/t", "REG_DWORD", "/d", "1", "/f"]), False, "📋")
        ])

    def run(self): self.root.mainloop()

# 🚀 Launch
if __name__ == "__main__": CodexShell().run()

