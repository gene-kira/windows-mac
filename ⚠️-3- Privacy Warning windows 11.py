import os
import sys
import ctypes
import subprocess
import fnmatch
import winreg
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

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
    r1 = set_registry_value(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Clipboard", "EnableClipboardHistory", 0)
    status_callback("Mutating registry: blocking screen clipping...")
    r2 = set_registry_value(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\ScreenClipBlock", "BlockScreenClip", 1)
    return r1 and r2

# 🧙 GUI Ritual
def launch_gui():
    root = tk.Tk()
    root.title("Codex Sentinel: Autonomous Daemon")
    root.geometry("520x460")
    root.resizable(False, False)

    status = tk.StringVar()
    status.set("Daemon initialized. Watching for glyphs...")

    log = tk.Text(root, height=8, width=65, font=("Consolas", 9))
    log.pack(pady=10)

    def update_status(msg):
        status.set(msg)
        log.insert(tk.END, msg + "\n")
        log.see(tk.END)
        root.update_idletasks()

    progress = ttk.Progressbar(root, orient="horizontal", length=460, mode="determinate")
    progress.pack(pady=10)

    def execute_ritual():
        progress["value"] = 0
        update_status("🧹 Auto-purge triggered...")
        deleted = cleanup_localappdata_packages(update_status)
        progress["value"] = 50
        update_status("🧬 Registry mutation...")
        reg_result = disable_clipboard_and_screenclip(update_status)
        progress["value"] = 100
        final_msg = f"✅ Ritual complete.\nDeleted {len(deleted)} files.\nRegistry mutation {'succeeded' if reg_result else 'failed'}."
        update_status(final_msg)

    def export_log():
        try:
            with open("ritual_log.txt", "w") as f:
                f.write(log.get("1.0", tk.END))
            messagebox.showinfo("Codex Sentinel", "Ritual log exported.")
        except Exception as e:
            messagebox.showerror("Codex Sentinel", f"Export failed: {e}")

    # 🔄 Background watcher thread
    def watcher():
        localappdata = os.getenv('LOCALAPPDATA')
        target_dir = os.path.join(localappdata, 'Packages')
        patterns = ['*.dat', '*ScreenClip*']
        seen = set()
        mutated = False

        while True:
            found = []
            for root_dir, dirs, files in os.walk(target_dir):
                for pattern in patterns:
                    for filename in fnmatch.filter(files, pattern):
                        file_path = os.path.join(root_dir, filename)
                        if file_path not in seen:
                            found.append(file_path)
                            seen.add(file_path)
            if found:
                update_status(f"⚠️ Detected {len(found)} new glyphs. Executing ritual...")
                execute_ritual()
                if not mutated:
                    mutated = True  # Registry mutation only once
            time.sleep(10)

    threading.Thread(target=watcher, daemon=True).start()

    tk.Label(root, text="Codex Sentinel Daemon", font=("Consolas", 16)).pack(pady=10)
    tk.Button(root, text="Manual Ritual Trigger", command=execute_ritual, font=("Consolas", 12)).pack(pady=5)
    tk.Button(root, text="Export Log", command=export_log, font=("Consolas", 10)).pack(pady=5)
    tk.Label(root, textvariable=status, wraplength=480, font=("Consolas", 10), fg="blue").pack(pady=10)

    root.mainloop()

# 🧨 Entry point
if __name__ == "__main__":
    elevate_if_needed()
    launch_gui()

