import tkinter as tk
from tkinter import messagebox
import os, platform, ctypes, winreg

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
        os.makedirs("GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}", exist_ok=True)
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

# 🧿 Diagnostics
def run_diagnostics():
    results = []
    results.append(f"{'🧿' if not safe_mode else ''} OS Version: {platform.system()} {platform.release()} ({platform.version()})")
    results.append(f"{'🧿' if not safe_mode else ''} Admin Privileges: {'Yes' if is_admin() else 'No'}")
    results.append(f"{'🧿' if not safe_mode else ''} Registry Access: {'Yes' if has_registry_access() else 'No'}")
    results.append(f"{'🧿' if not safe_mode else ''} Windows 10+ Compatible: {'Yes' if is_windows_compatible() else 'No'}")
    results.append(f"{'🧿' if not safe_mode else ''} Mode: {'Safe Mode' if safe_mode else 'Symbolic Mode'}")

    messagebox.showinfo("System Profile", "\n".join(results))

# 🧿 GUI Ritual Panel
safe_mode = detect_safe_mode()
root = tk.Tk()
root.title("God Mode Diagnostic Panel")
root.geometry("360x240")
root.resizable(False, False)

tk.Label(root, text="Activate Features" if safe_mode else "🧿 Activate Rituals", font=("Segoe UI", 12, "bold")).pack(pady=10)

tk.Button(root, text="Enable God Mode", command=enable_god_mode, width=30).pack(pady=5)
tk.Button(root, text="Enable Hidden Feature", command=enable_hidden_feature, width=30).pack(pady=5)
tk.Button(root, text="Run Diagnostics", command=run_diagnostics, width=30).pack(pady=5)

tk.Label(root, text="Plain feedback enabled." if safe_mode else "Symbolic feedback will guide your path.", font=("Segoe UI", 9, "italic")).pack(pady=10)

root.mainloop()

