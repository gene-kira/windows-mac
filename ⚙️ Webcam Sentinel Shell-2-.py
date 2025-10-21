import os
import sys
import ctypes
import threading
import tkinter as tk
from tkinter import messagebox
import subprocess
import time
import datetime

try:
    import winreg
except ImportError:
    messagebox.showerror("Missing Module", "winreg module is required on Windows.")
    sys.exit()

# ─────────────────────────────────────────────
# 🔮 AUTLOADER: Admin Check & Relaunch Fix
# ─────────────────────────────────────────────
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def elevate_if_needed():
    if not is_admin():
        script = os.path.abspath(sys.argv[0])
        params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
        sys.exit()

elevate_if_needed()

# ─────────────────────────────────────────────
# 🧹 PURGE LOGS
# ─────────────────────────────────────────────
def purge_logs():
    try:
        subprocess.run(["wevtutil", "cl", "Microsoft-Windows-DeviceSetupManager/Admin"], check=True)
        status.set("🧹 Logs purged")
    except Exception as e:
        status.set(f"⚠️ Purge failed: {e}")

# ─────────────────────────────────────────────
# 🚫 BLOCK ACCESS
# ─────────────────────────────────────────────
def block_access():
    try:
        class_root = r"SYSTEM\CurrentControlSet\Control\Class"
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as hklm:
            with winreg.OpenKey(hklm, class_root) as root:
                for i in range(winreg.QueryInfoKey(root)[0]):
                    subkey_name = winreg.EnumKey(root, i)
                    try:
                        with winreg.OpenKey(root, subkey_name, 0, winreg.KEY_ALL_ACCESS) as subkey:
                            cls = winreg.QueryValueEx(subkey, "Class")[0]
                            if cls == "Image":
                                winreg.SetValueEx(subkey, "CameraPrivacySetting", 0, winreg.REG_DWORD, 1)
                    except Exception:
                        continue

        subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule",
                        "name=Block Webcam Access", "dir=in", "action=block",
                        "protocol=TCP", "localport=554,8554"], check=True)
        status.set("🔴 Access blocked")
    except Exception as e:
        status.set(f"⚠️ Block failed: {e}")

# ─────────────────────────────────────────────
# 🔍 MONITOR CONNECTIONS
# ─────────────────────────────────────────────
def monitor_connections():
    try:
        result = subprocess.check_output(
            'wmic path Win32_PnPEntity where "Name like \'%Camera%\' or Name like \'%Webcam%\'" get Name,Status,DeviceID',
            shell=True)
        status.set("🟢 Monitoring active devices")
        messagebox.showinfo("Webcam Devices", result.decode(errors="ignore"))
    except Exception as e:
        status.set(f"⚠️ Monitor failed: {e}")

# ─────────────────────────────────────────────
# 🌐 SHOW IP CONNECTIONS
# ─────────────────────────────────────────────
def show_ip_connections():
    try:
        result = subprocess.check_output("netstat -ano", shell=True)
        messagebox.showinfo("IP Connections", result.decode(errors="ignore"))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to get IP connections: {e}")

# ─────────────────────────────────────────────
# 🔎 SHOW WEBCAM PROCESSES
# ─────────────────────────────────────────────
def show_webcam_processes():
    try:
        result = subprocess.check_output(
            'powershell "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*camera*\' -or $_.Name -like \'*camera*\' } | Select Name, ProcessId, CommandLine"',
            shell=True)
        messagebox.showinfo("Webcam Processes", result.decode(errors="ignore"))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to get webcam processes: {e}")

# ─────────────────────────────────────────────
# 🌀 DAEMON THREAD: Autonomous Monitoring
# ─────────────────────────────────────────────
def daemon_monitor():
    while True:
        try:
            result = subprocess.check_output(
                'wmic path Win32_PnPEntity where "Name like \'%Camera%\'" get Name',
                shell=True)
            if result.strip():
                with open(os.path.expanduser("~/webcam_log.txt"), "a") as log:
                    log.write(f"{datetime.datetime.now()}: Webcam active\n")
        except Exception as e:
            with open(os.path.expanduser("~/webcam_log.txt"), "a") as log:
                log.write(f"{datetime.datetime.now()}: Monitor error - {e}\n")
        time.sleep(60)

threading.Thread(target=daemon_monitor, daemon=True).start()

# ─────────────────────────────────────────────
# 🖼️ GUI PANEL
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("Webcam Sentinel Shell")
root.geometry("420x320")
root.resizable(False, False)

status = tk.StringVar()
status.set("🟡 Idle")

tk.Label(root, text="🛡️ Webcam Sentinel Shell", font=("Segoe UI", 16, "bold")).pack(pady=10)
tk.Button(root, text="🧹 Purge Logs", command=purge_logs).pack(pady=5)
tk.Button(root, text="🚫 Block Unauthorized Access", command=block_access).pack(pady=5)
tk.Button(root, text="🔍 Monitor Connections", command=monitor_connections).pack(pady=5)
tk.Button(root, text="🌐 Show IP Connections", command=show_ip_connections).pack(pady=5)
tk.Button(root, text="🔎 Show Webcam Processes", command=show_webcam_processes).pack(pady=5)
tk.Label(root, textvariable=status, font=("Segoe UI", 10, "bold")).pack(pady=20)

root.mainloop()

