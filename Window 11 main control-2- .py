import subprocess
import ctypes
import os
import winreg
import tkinter as tk
from tkinter import ttk
import threading
import time
from datetime import datetime

# ⚙️ Autoloader Engine
def autoload_libraries():
    try:
        import winreg, subprocess, ctypes, os, tkinter
        print("[Autoloader] Libraries loaded.")
    except ImportError as e:
        print(f"[Autoloader] Missing: {e}")

# 🎮 Gaming Panel
class GamingPanel:
    def render(self, parent):
        frame = ttk.LabelFrame(parent, text="🎮 Gaming Mode")
        ttk.Button(frame, text="Enable Game Mode", command=self.enable_game_mode).pack(fill='x')
        ttk.Button(frame, text="Enable GPU Scheduling", command=self.enable_gpu_sched).pack(fill='x')
        ttk.Button(frame, text="Set High Performance Plan", command=self.set_power_plan).pack(fill='x')
        return frame

    def enable_game_mode(self):
        subprocess.run(["reg", "add", r"HKCU\Software\Microsoft\GameBar", "/v", "AllowAutoGameMode", "/t", "REG_DWORD", "/d", "1", "/f"])

    def enable_gpu_sched(self):
        subprocess.run(["reg", "add", r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers", "/v", "HwSchMode", "/t", "REG_DWORD", "/d", "2", "/f"])

    def set_power_plan(self):
        subprocess.run(["powercfg", "/setactive", "SCHEME_MAX"])

# 🔒 Privacy Panel
class PrivacyPanel:
    def render(self, parent):
        frame = ttk.LabelFrame(parent, text="🔒 Privacy Purge")
        ttk.Button(frame, text="Disable Telemetry", command=self.disable_telemetry).pack(fill='x')
        ttk.Button(frame, text="Disable Ad Tracking", command=self.disable_ads).pack(fill='x')
        ttk.Button(frame, text="Clear Activity History", command=self.clear_activity).pack(fill='x')
        return frame

    def disable_telemetry(self):
        subprocess.run(["reg", "add", r"HKLM\Software\Policies\Microsoft\Windows\DataCollection", "/v", "AllowTelemetry", "/t", "REG_DWORD", "/d", "0", "/f"])

    def disable_ads(self):
        subprocess.run(["reg", "add", r"HKCU\Software\Microsoft\Windows\CurrentVersion\AdvertisingInfo", "/v", "Enabled", "/t", "REG_DWORD", "/d", "0", "/f"])

    def clear_activity(self):
        subprocess.run(["reg", "delete", r"HKCU\Software\Microsoft\Windows\CurrentVersion\ActivityHistory", "/f"])

# 🧹 Bloatware Panel
class BloatwarePanel:
    def render(self, parent):
        frame = ttk.LabelFrame(parent, text="🧹 Bloatware Scanner")
        ttk.Button(frame, text="Scan & Remove Bloatware", command=self.remove_bloatware).pack(fill='x')
        return frame

    def remove_bloatware(self):
        bloat_list = [
            "Microsoft.ZuneMusic", "Microsoft.XboxGameCallableUI", "Microsoft.Xbox.TCUI",
            "Microsoft.XboxApp", "Microsoft.XboxGameOverlay", "Microsoft.XboxGamingOverlay"
        ]
        for app in bloat_list:
            subprocess.run(["powershell", "-Command", f"Get-AppxPackage *{app}* | Remove-AppxPackage"])

# 🧠 Hidden Settings Panel
class HiddenSettingsPanel:
    def render(self, parent):
        frame = ttk.LabelFrame(parent, text="🧠 Hidden Controls")
        ttk.Button(frame, text="Activate God Mode", command=self.activate_god_mode).pack(fill='x')
        ttk.Button(frame, text="Enable Clipboard History", command=self.enable_clipboard).pack(fill='x')
        return frame

    def activate_god_mode(self):
        os.makedirs("GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}", exist_ok=True)

    def enable_clipboard(self):
        subprocess.run(["reg", "add", r"HKCU\Software\Microsoft\Clipboard", "/v", "EnableClipboardHistory", "/t", "REG_DWORD", "/d", "1", "/f"])

# 🤖 Codex Sentinel Bot
class CodexSentinelBot:
    def __init__(self):
        self.scan_interval = 86400  # 24 hours
        self.running = True
        threading.Thread(target=self.daily_scan_loop, daemon=True).start()
        threading.Thread(target=self.realtime_monitor, daemon=True).start()

    def daily_scan_loop(self):
        while self.running:
            self.full_system_scan()
            time.sleep(self.scan_interval)

    def full_system_scan(self):
        print(f"[{datetime.now()}] 🔍 Daily Scan Initiated")
        self.scan_hidden_features()
        self.scan_privacy_threats()
        self.log_scan("Daily scan complete.")

    def scan_hidden_features(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Advanced")
            index = 0
            while True:
                name, _, _ = winreg.EnumValue(key, index)
                print(f"[Hidden Feature] Found: {name}")
                index += 1
        except Exception:
            pass

    def scan_privacy_threats(self):
        result = subprocess.run(["reg", "query", r"HKLM\Software\Policies\Microsoft\Windows\DataCollection", "/v", "AllowTelemetry"], capture_output=True, text=True)
        if "0x1" in result.stdout:
            print("[Privacy Threat] Telemetry is ENABLED")

    def realtime_monitor(self):
        while self.running:
            self.monitor_registry()
            time.sleep(300)

    def monitor_registry(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run")
            index = 0
            while True:
                name, value, _ = winreg.EnumValue(key, index)
                print(f"[Startup Monitor] {name}: {value}")
                index += 1
        except Exception:
            pass

    def log_scan(self, message):
        with open("codex_logbook.txt", "a") as log:
            log.write(f"[{datetime.now()}] {message}\n")

# 🧬 Codex Control Shell Main
class CodexControlShell:
    def __init__(self):
        autoload_libraries()
        CodexSentinelBot()
        self.root = tk.Tk()
        self.root.title("Codex Control Shell")
        self.root.geometry("600x700")
        self.root.resizable(False, False)

        self.panels = [
            GamingPanel(),
            PrivacyPanel(),
            BloatwarePanel(),
            HiddenSettingsPanel()
        ]

        for panel in self.panels:
            widget = panel.render(self.root)
            widget.pack(padx=10, pady=10, fill='x')

    def run(self):
        self.root.mainloop()

# 🚀 Launch the Shell
if __name__ == "__main__":
    CodexControlShell().run()

