import psutil
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import os
import shutil
import subprocess
import socket
import json
from pathlib import Path

# ---------- CONFIG ----------
CHECK_INTERVAL = 2.0
TIME_DRIFT_THRESHOLD_SECONDS = 10
LOG_FILE = "sentinel_log.txt"
QUARANTINE_DIR = Path("C:/SentinelQuarantine")
LISTS_FILE = str(Path.home() / "sentinel_lists.json")  # Safe location

# Core Windows + Teams whitelist
WHITELIST_NAMES = {
    "system", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
    "searchindexer.exe", "runtimebroker.exe", "dwm.exe",
    "shellexperiencehost.exe", "teams.exe", "ms-teams.exe",
}

# Startup folders where your trusted Python tools live
STARTUP_DIRS = [
    str(Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"),
    "C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Startup",
]

# Official Avira installation directories
AVIRA_DIRS = [
    "C:/Program Files/Avira",
    "C:/Program Files (x86)/Avira",
]

# Dynamic trust lists (exe paths or names)
ALLOWLIST = set()
BLOCKLIST = set()
RADIOACTIVE = set()

# ---------- LIST PERSISTENCE ----------
def load_lists():
    global ALLOWLIST, BLOCKLIST, RADIOACTIVE
    if not os.path.exists(LISTS_FILE):
        return
    try:
        with open(LISTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ALLOWLIST = set(data.get("allow", []))
        BLOCKLIST = set(data.get("block", []))
        RADIOACTIVE = set(data.get("radioactive", []))
    except (PermissionError, json.JSONDecodeError):
        print(f"⚠ Could not read {LISTS_FILE}, skipping load.")

def save_lists():
    try:
        with open(LISTS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "allow": sorted(ALLOWLIST),
                "block": sorted(BLOCKLIST),
                "radioactive": sorted(RADIOACTIVE),
            }, f, indent=2)
    except PermissionError:
        print(f"⚠ Could not write {LISTS_FILE}, check file permissions.")

# ---------- HELPERS ----------
def is_startup_python(proc: psutil.Process) -> bool:
    try:
        name = (proc.name() or "").lower()
        exe = (proc.exe() or "").replace("\\", "/").lower()
        cmdline = " ".join(proc.cmdline() or []).replace("\\", "/").lower()
        if "python" not in name and "python" not in exe:
            return False
        for sdir in STARTUP_DIRS:
            sdir_norm = sdir.replace("\\", "/").lower()
            if exe.startswith(sdir_norm) or cmdline.startswith(sdir_norm):
                return True
        return False
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False

def is_avira_process(proc: psutil.Process) -> bool:
    try:
        exe = (proc.exe() or "").replace("\\", "/").lower()
        for adir in AVIRA_DIRS:
            if exe.startswith(adir.replace("\\", "/").lower()):
                return True
        return False
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False

def get_primary_ethernet_adapter():
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    for name, addr_list in addrs.items():
        if name.lower().startswith("loopback"):
            continue
        if name not in stats or not stats[name].isup:
            continue
        mac, ip = None, None
        for a in addr_list:
            if a.family == psutil.AF_LINK:
                mac = a.address
            elif a.family == socket.AF_INET:
                ip = a.address
        if mac:
            return {"name": name, "mac": mac, "ip": ip or "N/A", "up": stats[name].isup}
    return None

# ---------- BRAIN ----------
class SentinelBrain:
    def __init__(self):
        self.last_wall_time = datetime.datetime.now()
        self.last_monotonic = time.monotonic()

    def log(self, msg: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)

    def check_time(self):
        now_wall = datetime.datetime.now()
        now_mono = time.monotonic()
        expected = self.last_wall_time + datetime.timedelta(seconds=(now_mono - self.last_monotonic))
        drift = (now_wall - expected).total_seconds()
        self.last_wall_time = now_wall
        self.last_monotonic = now_mono
        if abs(drift) > TIME_DRIFT_THRESHOLD_SECONDS:
            msg = f"TIME DRIFT DETECTED: {drift:.1f} seconds"
            self.log(msg)
            return msg
        return None

    def classify_process(self, proc: psutil.Process):
        try:
            name = proc.name()
            pid = proc.pid
            try:
                cpu = proc.cpu_percent(interval=0.0)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                cpu = 0.0
            try:
                exe = proc.exe() or ""
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                exe = ""
            exe_norm = exe.replace("\\", "/").lower()
            name_norm = (name or "").lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None

        # Dynamic lists
        if exe_norm in BLOCKLIST or name_norm in BLOCKLIST:
            status = "BLOCKED"
        elif exe_norm in ALLOWLIST or name_norm in ALLOWLIST:
            status = "ALLOW"
        elif exe_norm in RADIOACTIVE or name_norm in RADIOACTIVE:
            status = "RADIOACTIVE"
        # Static trust
        elif name_norm in (n.lower() for n in WHITELIST_NAMES):
            status = "TRUSTED"
        elif is_avira_process(proc):
            status = "AVIRA"
        elif is_startup_python(proc):
            status = "STARTUP_PY"
        else:
            status = "SUSPICIOUS"

        return {
            "pid": pid,
            "name": name,
            "cpu": cpu,
            "path": exe,
            "status": status,
            "exe_norm": exe_norm,
            "name_norm": name_norm,
        }

# ---------- GUI ----------
class SentinelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Teams-Only Sentinel")
        self.brain = SentinelBrain()
        self.running = True
        self.lockdown_mode = tk.BooleanVar(value=False)
        self._build_ui()
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # --- Tab 1: Processes ---
        proc_tab = ttk.Frame(notebook)
        notebook.add(proc_tab, text="Processes")

        status_frame = ttk.LabelFrame(proc_tab, text="System Status")
        status_frame.pack(fill="x", padx=10, pady=5)
        self.time_label = ttk.Label(status_frame, text="Time: OK")
        self.time_label.pack(anchor="w")
        self.proc_label = ttk.Label(status_frame, text="Processes: Monitoring")
        self.proc_label.pack(anchor="w")
        ttk.Checkbutton(status_frame, text="Lockdown Mode", variable=self.lockdown_mode).pack(anchor="w", pady=5)

        proc_frame = ttk.LabelFrame(proc_tab, text="Top CPU Processes")
        proc_frame.pack(fill="both", expand=True, padx=10, pady=5)
        columns = ("name", "pid", "cpu", "status", "path")
        self.tree = ttk.Treeview(proc_frame, columns=columns, show="headings", height=12)
        for col in columns:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=120 if col != "path" else 400)
        self.tree.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(proc_tab)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="Kill Selected", command=self.kill_selected).pack(side="left")
        ttk.Button(btn_frame, text="Quarantine Selected", command=self.quarantine_selected).pack(side="left")
        ttk.Button(btn_frame, text="Delete Selected", command=self.delete_selected).pack(side="left")
        ttk.Button(btn_frame, text="Quit", command=self.on_quit).pack(side="right")

        # --- Tab 2: Trust Lists ---
        trust_tab = ttk.Frame(notebook)
        notebook.add(trust_tab, text="Trust Lists")
        lists_frame = ttk.Frame(trust_tab)
        lists_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.allow_listbox = tk.Listbox(ttk.LabelFrame(lists_frame, text="Allowlist"), height=15)
        self.allow_listbox.pack(side="left", fill="both", expand=True, padx=5)
        self.block_listbox = tk.Listbox(ttk.LabelFrame(lists_frame, text="Blocklist"), height=15)
        self.block_listbox.pack(side="left", fill="both", expand=True, padx=5)
        self.radio_listbox = tk.Listbox(ttk.LabelFrame(lists_frame, text="Radioactive"), height=15)
        self.radio_listbox.pack(side="left", fill="both", expand=True, padx=5)

        self.refresh_lists_ui()

    # ---------- Trust list helpers ----------
    def refresh_lists_ui(self):
        self.allow_listbox.delete(0, "end")
        for item in sorted(ALLOWLIST):
            self.allow_listbox.insert("end", item)
        self.block_listbox.delete(0, "end")
        for item in sorted(BLOCKLIST):
            self.block_listbox.insert("end", item)
        self.radio_listbox.delete(0, "end")
        for item in sorted(RADIOACTIVE):
            self.radio_listbox.insert("end", item)

    # ---------- Monitor loop ----------
    def monitor_loop(self):
        while self.running:
            drift_msg = self.brain.check_time()
            if drift_msg:
                self.root.after(0, self.time_label.config, {"text": "Time: ALERT"})

            proc_infos = []
            proc_map = {}
            for proc in psutil.process_iter(attrs=["pid", "name"]):
                info = self.brain.classify_process(proc)
                if info:
                    proc_infos.append(info)
                    proc_map[info["pid"]] = (proc, info)
                    # Auto-fill lists
                    if info["status"] in ("TRUSTED", "AVIRA", "STARTUP_PY"):
                        ALLOWLIST.add(info["exe_norm"])
                        RADIOACTIVE.discard(info["exe_norm"])
                    elif info["status"] == "SUSPICIOUS":
                        RADIOACTIVE.add(info["exe_norm"])
                        ALLOWLIST.discard(info["exe_norm"])

            proc_infos.sort(key=lambda x: x["cpu"], reverse=True)
            top = proc_infos[:15]

            def update_views():
                self.tree.delete(*self.tree.get_children())
                for info in top:
                    self.tree.insert("", "end", iid=str(info["pid"]),
                                     values=(info["name"], info["pid"], f"{info['cpu']:.1f}", info["status"], info["path"]))
                self.refresh_lists_ui()

            self.root.after(0, update_views)
            save_lists()
            time.sleep(CHECK_INTERVAL)

    # ---------- Process actions ----------
    def get_selected_pid(self):
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0])
        except ValueError:
            return None

    def kill_selected(self):
        pid = self.get_selected_pid()
        if pid is None:
            return
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            self.brain.log(f"KILLED: {proc.name()} (PID {pid})")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            self.brain.log(f"FAILED to kill PID {pid}")

    def quarantine_selected(self):
        pid = self.get_selected_pid()
        if pid is None:
            return
        try:
            proc = psutil.Process(pid)
            exe = proc.exe()
            proc.terminate()
            time.sleep(0.5)
            dest = QUARANTINE_DIR / Path(exe).name
            shutil.move(exe, dest)
            self.brain.log(f"QUARANTINED: {proc.name()} → {dest}")
        except Exception as e:
            self.brain.log(f"FAILED to quarantine PID {pid}: {e}")

    def delete_selected(self):
        pid = self.get_selected_pid()
        if pid is None:
            return
        try:
            proc = psutil.Process(pid)
            exe = proc.exe()
            proc.terminate()
            time.sleep(0.5)
            os.remove(exe)
            self.brain.log(f"DELETED: {proc.name()} file '{exe}'")
        except Exception as e:
            self.brain.log(f"FAILED to delete PID {pid}: {e}")

    def on_quit(self):
        self.running = False
        self.root.destroy()

# ---------- MAIN ----------
def main():
    if not QUARANTINE_DIR.exists():
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    load_lists()
    root = tk.Tk()
    SentinelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
