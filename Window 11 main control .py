import subprocess
import ctypes
import os
import winreg
import tkinter as tk
from tkinter import ttk

# 🔥 Autoloader Engine
def autoload_libraries():
    try:
        import winreg, subprocess, ctypes, os, tkinter
        print("[Autoloader] All libraries loaded successfully.")
    except ImportError as e:
        print(f"[Autoloader] Missing library: {e}")

# 🎮 Gaming Panel
class GamingPanel:
    def render(self, parent):
        frame = ttk.LabelFrame(parent, text="🎮 Gaming Mode")
        ttk.Button(frame, text="Enable Game Mode", command=self.enable_game_mode).pack(fill='x')
        ttk.Button(frame, text="Enable GPU Scheduling", command=self.enable_gpu_sched).pack(fill='x')
        ttk.Button(frame, text="Set High Performance Power Plan", command=self.set_power_plan).pack(fill='x')
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

# 🧬 Codex Control Shell Main
class CodexControlShell:
    def __init__(self):
        autoload_libraries()
        self.root = tk.Tk()
        self.root.title("Codex Control Shell")
        self.root.geometry("600x600")
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

