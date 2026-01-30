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
LISTS_FILE = "sentinel_lists.json"

WHITELIST_NAMES = {
    "System", "Registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
    "SearchIndexer.exe", "RuntimeBroker.exe", "dwm.exe",
    "ShellExperienceHost.exe", "Teams.exe", "ms-teams.exe",
}

STARTUP_DIRS = [
    str(Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup"),
    "C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Startup",
]

AVIRA_DIRS = [
    "C:/Program Files/Avira",
    "C:/Program Files (x86)/Avira",
]

ALLOWLIST = set()
BLOCKLIST = set()
RADIOACTIVE = set()

# ---------- LIST PERSISTENCE ----------

def load_lists():
    global ALLOWLIST, BLOCKLIST, RADIOACTIVE
    if not os.path.exists(LISTS_FILE):
        return
    with open(LISTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    ALLOWLIST = set(data.get("allow", []))
    BLOCKLIST = set(data.get("block", []))
    RADIOACTIVE = set(data.get("radioactive", []))


def save_lists():
    with open(LISTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "allow": sorted(ALLOWLIST),
            "block": sorted(BLOCKLIST),
            "radioactive": sorted(RADIOACTIVE),
        }, f, indent=2)

# ---------- HELPERS ----------

def is_startup_python(proc):
    try:
        exe = (proc.exe() or "").replace("\\", "/").lower()
        cmd = " ".join(proc.cmdline()).replace("\\", "/").lower()
        if "python" not in exe:
            return False
        return any(exe.startswith(d.lower()) or cmd.startswith(d.lower()) for d in STARTUP_DIRS)
    except Exception:
        return False


def is_avira_process(proc):
    try:
        exe = (proc.exe() or "").replace("\\", "/").lower()
        return any(exe.startswith(d.lower()) for d in AVIRA_DIRS)
    except Exception:
        return False


def get_primary_ethernet_adapter():
    for name, addrs in psutil.net_if_addrs().items():
        stats = psutil.net_if_stats().get(name)
        if not stats or not stats.isup:
            continue
        mac = ip = None
        for a in addrs:
            if a.family == psutil.AF_LINK:
                mac = a.address
            elif a.family == socket.AF_INET:
                ip = a.address
        if mac:
            return {"name": name, "mac": mac, "ip": ip or "N/A", "up": stats.isup}
    return None

# ---------- BRAIN ----------

class SentinelBrain:
    def __init__(self):
        self.last_wall = datetime.datetime.now()
        self.last_mono = time.monotonic()

    def log(self, msg):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now()}] {msg}\n")

    def check_time(self):
        now = datetime.datetime.now()
        mono = time.monotonic()
        drift = (now - (self.last_wall + datetime.timedelta(seconds=mono - self.last_mono))).total_seconds()
        self.last_wall, self.last_mono = now, mono
        return abs(drift) > TIME_DRIFT_THRESHOLD_SECONDS

    def classify(self, proc):
        try:
            name = proc.name()
            exe = proc.exe() or ""
            exe_n = exe.replace("\\", "/").lower()
            name_n = name.lower()
        except Exception:
            return None

        if exe_n in BLOCKLIST or name_n in BLOCKLIST:
            status = "BLOCKED"
        elif exe_n in ALLOWLIST or name_n in ALLOWLIST:
            status = "ALLOW"
        elif exe_n in RADIOACTIVE or name_n in RADIOACTIVE:
            status = "RADIOACTIVE"
        elif name in WHITELIST_NAMES:
            status = "TRUSTED"
        elif is_avira_process(proc):
            status = "AVIRA"
        elif is_startup_python(proc):
            status = "STARTUP_PY"
        else:
            status = "SUSPICIOUS"

        return {
            "pid": proc.pid,
            "name": name,
            "cpu": proc.cpu_percent(0),
            "path": exe,
            "status": status,
        }

# ---------- GUI ----------

class SentinelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Teams-Only Sentinel")
        self.brain = SentinelBrain()
        self.running = True

        self._build_ui()
        self._build_context_menu()

        threading.Thread(target=self.monitor_loop, daemon=True).start()

    # ---------- UI ----------

    def _build_ui(self):
        self.tree = ttk.Treeview(self.root, columns=("name", "pid", "cpu", "status", "path"), show="headings")
        for col in ("name", "pid", "cpu", "status", "path"):
            self.tree.heading(col, text=col.upper())
        self.tree.pack(fill="both", expand=True)

        self.tree.bind("<Button-3>", self.show_context_menu)

    def _build_context_menu(self):
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="➕ Add to Allowlist", command=self.add_selected_allow)
        self.menu.add_command(label="⛔ Add to Blocklist", command=self.add_selected_block)
        self.menu.add_command(label="☢ Add to Radioactive", command=self.add_selected_radio)
        self.menu.add_separator()
        self.menu.add_command(label="❌ Kill", command=self.kill_selected)

    def show_context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.menu.tk_popup(event.x_root, event.y_root)

    # ---------- Trust helpers ----------

    def _get_selected_proc_info(self):
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        if not vals or not vals[4]:
            return None
        return vals[0], vals[4].replace("\\", "/").lower()

    def add_selected_allow(self):
        info = self._get_selected_proc_info()
        if info:
            ALLOWLIST.add(info[1])
            save_lists()

    def add_selected_block(self):
        info = self._get_selected_proc_info()
        if info:
            BLOCKLIST.add(info[1])
            save_lists()

    def add_selected_radio(self):
        info = self._get_selected_proc_info()
        if info:
            RADIOACTIVE.add(info[1])
            save_lists()

    # ---------- Actions ----------

    def kill_selected(self):
        sel = self.tree.selection()
        if sel:
            pid = int(self.tree.item(sel[0], "values")[1])
            psutil.Process(pid).terminate()

    # ---------- Monitor ----------

    def monitor_loop(self):
        while self.running:
            self.tree.delete(*self.tree.get_children())
            for proc in psutil.process_iter():
                info = self.brain.classify(proc)
                if info:
                    self.tree.insert("", "end", values=(
                        info["name"], info["pid"], f"{info['cpu']:.1f}",
                        info["status"], info["path"]
                    ))
            time.sleep(CHECK_INTERVAL)

# ---------- MAIN ----------

def main():
    QUARANTINE_DIR.mkdir(exist_ok=True)
    load_lists()
    root = tk.Tk()
    SentinelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()