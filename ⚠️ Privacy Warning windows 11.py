import os
import sys
import ctypes
import subprocess
import fnmatch
import winreg
import tkinter as tk
from tkinter import messagebox

# 🔁 Auto-install pywin32 if missing
try:
    import win32com.shell.shell as shell
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
    import win32com.shell.shell as shell

# 🔐 Admin check
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# 🔼 Elevate if needed
def elevate_if_needed():
    if not is_admin():
        params = ' '.join([f'"{arg}"' for arg in sys.argv])
        shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=params)
        sys.exit()

# 🧹 Cleanup logic
def cleanup_localappdata_packages(status_callback):
    localappdata = os.getenv('LOCALAPPDATA')
    if not localappdata:
        status_callback("LOCALAPPDATA not found.")
        return []

    target_dir = os.path.join(localappdata, 'Packages')
    patterns = ['*.dat', '*ScreenClip*']
    deleted = []

    for root, dirs, files in os.walk(target_dir):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    deleted.append(file_path)
                except:
                    pass
    status_callback(f"Deleted {len(deleted)} files.")
    return deleted

# 🧬 Registry mutation
def set_registry_value(root, path, name, value, value_type=winreg.REG_DWORD):
    try:
        key = winreg.CreateKeyEx(root, path, 0, winreg.KEY_SET_VALUE)
        winreg.SetValueEx(key, name, 0, value_type, value)
        winreg.CloseKey(key)
        return True
    except:
        return False

def disable_clipboard_and_screenclip(status_callback):
    status_callback("Mutating registry: disabling clipboard history...")
    r1 = set_registry_value(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Clipboard", "EnableClipboardHistory", 0)
    status_callback("Mutating registry: blocking screen clipping...")
    r2 = set_registry_value(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\ScreenClipBlock", "BlockScreenClip", 1)
    return r1 and r2

# 🧙 GUI Ritual
def launch_gui():
    root = tk.Tk()
    root.title("Codex Sentinel: Cleanup Daemon")
    root.geometry("420x320")
    root.resizable(False, False)

    status = tk.StringVar()
    status.set("Ready to purge and mutate...")

    def update_status(msg):
        status.set(msg)
        root.update_idletasks()

    def execute_ritual():
        update_status("🧹 Initiating file purge...")
        deleted = cleanup_localappdata_packages(update_status)
        update_status("🧬 Initiating registry mutation...")
        reg_result = disable_clipboard_and_screenclip(update_status)
        final_msg = f"✅ Ritual complete.\nDeleted {len(deleted)} files.\nRegistry mutation {'succeeded' if reg_result else 'failed'}."
        update_status(final_msg)
        messagebox.showinfo("Codex Sentinel", final_msg)

    tk.Label(root, text="Codex Sentinel Cleanup", font=("Consolas", 16)).pack(pady=20)
    tk.Button(root, text="Execute Ritual", command=execute_ritual, font=("Consolas", 12)).pack(pady=10)
    tk.Label(root, textvariable=status, wraplength=380, font=("Consolas", 10), fg="blue").pack(pady=20)

    root.mainloop()

# 🧨 Entry point
if __name__ == "__main__":
    elevate_if_needed()
    launch_gui()

