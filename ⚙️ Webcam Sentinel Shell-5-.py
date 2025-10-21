import os, sys, ctypes, threading, tkinter as tk
from tkinter import messagebox
import subprocess, time, datetime

def elevate():
    if not ctypes.windll.shell32.IsUserAnAdmin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{os.path.abspath(sys.argv[0])}"', None, 1)
        sys.exit()
elevate()

def run(cmd):
    try: return subprocess.check_output(cmd, shell=True).decode(errors="ignore").strip().splitlines()
    except: return []

def purge_logs(): run("wevtutil cl Microsoft-Windows-DeviceSetupManager/Admin"); status.set("🧹 Logs purged")
def monitor(): messagebox.showinfo("Webcam Devices", "\n".join(run('wmic path Win32_PnPEntity where "Name like \'%Camera%\'" get Name,Status,DeviceID')))
def show_ips(): messagebox.showinfo("IP Connections", "\n".join(run("netstat -ano")))

def block_ip(ip): run(f'netsh advfirewall firewall add rule name="Block {ip}" dir=out action=block remoteip={ip}')
def block_pid(pid): run(f'netsh advfirewall firewall add rule name="Block PID {pid}" dir=out action=block program={pid}')

def threat_panel():
    panel = tk.Toplevel(root); panel.title("Threat Panel"); panel.geometry("520x460")
    tk.Label(panel, text="Suspicious Processes & IPs", font=("Segoe UI", 12)).pack()
    box, scroll = tk.Listbox(panel, font=("Consolas", 10)), tk.Scrollbar(panel)
    box.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
    scroll.config(command=box.yview); box.config(yscrollcommand=scroll.set)

    def refresh(): box.delete(0, tk.END); [box.insert(tk.END, line) for line in run('powershell "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*camera*\' } | Select Name, ProcessId"') + run("netstat -ano")]
    def block_sel(): block_pid(''.join(filter(str.isdigit, box.get(tk.ACTIVE)))); messagebox.showinfo("Blocked", box.get(tk.ACTIVE))
    def allow_sel(): messagebox.showinfo("Allowed", box.get(tk.ACTIVE))
    def block_ip_entry(): block_ip(ip.get()); messagebox.showinfo("Blocked", ip.get())
    def allow_ip_entry(): messagebox.showinfo("Allowed", ip.get())

    for txt, cmd in [("🛡️ Block Selected", block_sel), ("✅ Allow Selected", allow_sel), ("🔁 Refresh", refresh)]: tk.Button(panel, text=txt, command=cmd).pack()
    ip = tk.Entry(panel, font=("Consolas", 10)); ip.pack(pady=5)
    for txt, cmd in [("🛡️ Block IP", block_ip_entry), ("✅ Allow IP", allow_ip_entry)]: tk.Button(panel, text=txt, command=cmd).pack()
    refresh()

def daemon():
    while True:
        active = run('wmic path Win32_PnPEntity where "Name like \'%Camera%\'" get Name')
        suspicious = run('powershell "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*camera*\' }"')
        light.config(bg="red" if active and suspicious else "green" if active else "black")
        time.sleep(5)

root = tk.Tk(); root.title("Webcam Sentinel Shell"); root.geometry("440x460"); root.resizable(False, False)
status = tk.StringVar(); status.set("🟡 Idle")

for txt, cmd in [("🧹 Purge Logs", purge_logs), ("🔍 Monitor Connections", monitor), ("🌐 Show IP Connections", show_ips), ("🛡️ Open Threat Panel", threat_panel)]:
    tk.Button(root, text=txt, command=cmd).pack(pady=5)

tk.Label(root, textvariable=status, font=("Segoe UI", 10)).pack(pady=10)
tk.Label(root, text="Camera Status Light:", font=("Segoe UI", 10)).pack()
light = tk.Label(root, width=20, height=2, bg="black", relief="sunken"); light.pack(pady=5)

threading.Thread(target=daemon, daemon=True).start()
root.mainloop()

