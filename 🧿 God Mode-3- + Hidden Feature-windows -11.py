import tkinter as tk
from tkinter import messagebox
import os, sys, platform, ctypes, winreg

# 🧿 Elevation Ritual
def elevate():
    if "--elevated" not in sys.argv:
        params = " ".join([f'"{arg}"' for arg in sys.argv] + ["--elevated"])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
        sys.exit()

# 🧿 Compatibility Checks
def is_windows_compatible():
    try:
        version = platform.version()
        major = int(version.split('.')[0])
        return major >= 10
    except:
        return False

def has_registry_access():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer", 0, winreg.KEY_WRITE) as key:
            return True
    except:
        return False

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# 🧿 Mode Detection
def detect_safe_mode():
    return not (is_windows_compatible() and has_registry_access() and is_admin())

# 🧿 Ritual Functions
def enable_god_mode():
    try:
        os.makedirs(os.path.expanduser("~/Desktop/GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}"), exist_ok=True)
        feedback("God Mode activated.", success=True)
    except Exception as e:
        feedback(f"God Mode error:\n{e}", success=False)

def enable_hidden_feature():
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "EnableSnapAssistFlyout", 0, winreg.REG_DWORD, 1)
        feedback("Hidden feature enabled.", success=True)
    except Exception as e:
        feedback(f"Registry error:\n{e}", success=False)

# 🧿 Feedback Ritual
def feedback(message, success=True):
    if safe_mode:
        title = "Success" if success else "Error"
    else:
        title = "🧿 Ritual Complete" if success else "⚠️ Ritual Failed"
    messagebox.showinfo(title, message) if success else messagebox.showerror(title, message)

# 🧿 Persona Status Panel
def show_persona_status():
    status = [
        f"{'🧿' if not safe_mode else ''} OS: {platform.system()} {platform.release()}",
        f"{'🧿' if not safe_mode else ''} Version: {platform.version()}",
        f"{'🧿' if not safe_mode else ''} Admin Privileges: {'Yes' if is_admin() else 'No'}",
        f"{'🧿' if not safe_mode else ''} Registry Access: {'Yes' if has_registry_access() else 'No'}",
        f"{'🧿' if not safe_mode else ''} Windows 10+ Compatible: {'Yes' if is_windows_compatible() else 'No'}",
        f"{'🧿' if not safe_mode else ''} Mode: {'Safe Mode' if safe_mode else 'Symbolic Mode'}"
    ]
    messagebox.showinfo("Persona Status Panel", "\n".join(status))

# 🧿 GUI Launcher
def launch_gui():
    root = tk.Tk()
    root.title("🧿 Ritual Shell: God Mode + Persona Status")
    root.geometry("400x260")
    root.resizable(False, False)

    tk.Label(root, text="Activate Rituals" if safe_mode else "🧿 Activate Rituals", font=("Segoe UI", 12, "bold")).pack(pady=10)

    tk.Button(root, text="Enable God Mode", command=enable_god_mode, width=30).pack(pady=5)
    tk.Button(root, text="Enable Hidden Feature", command=enable_hidden_feature, width=30).pack(pady=5)
    tk.Button(root, text="Show Persona Status", command=show_persona_status, width=30).pack(pady=5)

    tk.Label(root, text="Plain feedback enabled." if safe_mode else "Symbolic feedback will guide your path.", font=("Segoe UI", 9, "italic")).pack(pady=10)

    root.mainloop()

# 🧿 Entry Point
if __name__ == "__main__":
    if "--elevated" not in sys.argv:
        elevate()
    safe_mode = detect_safe_mode()
    launch_gui()

