# ============================================================
# StartupBridge — Cross-platform startup launcher + LLM bridge
# GUI status + Autoloader + Process monitoring & auto-restart
# Expanded discovery + Config persistence + Threat Matrix
# Python script detection + Add/Delete/Launch selected
# ============================================================

import os
import sys
import time
import json
import queue
import threading
import subprocess
import importlib
import importlib.util
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog

# --- Autoloader ---
def autoload(packages):
    import subprocess as sp
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            try:
                sp.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception as e:
                print(f"[Autoloader] Failed to install {pkg}: {e}")

autoload(["psutil", "openai", "anthropic"])

try:
    import psutil
except Exception:
    psutil = None

try:
    import winreg
except Exception:
    winreg = None

# ============================================================
# Config persistence
# ============================================================
def config_path():
    base = Path.home() / ".startupbridge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "config.json"

DEFAULT_CONFIG = {
    "auto_launch": True,
    "monitor_interval": 5.0,
    "max_restarts": 5,
}

def load_config():
    p = config_path()
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    p = config_path()
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[Config] Failed to save: {e}")

# ============================================================
# Status bus
# ============================================================
class StatusBus:
    def __init__(self):
        self.q = queue.Queue()

    def emit(self, event, payload=None):
        self.q.put({"event": event, "payload": payload or {}})

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

# ============================================================
# OS helpers
# ============================================================
def is_windows(): return os.name == "nt"
def is_mac(): return sys.platform == "darwin"
def is_linux(): return sys.platform.startswith("linux")

# ============================================================
# Startup discovery
# ============================================================
def get_windows_startup_folder_entries():
    entries = []
    startup_dir = os.path.join(
        os.environ.get("APPDATA", ""),
        "Microsoft", "Windows", "Start Menu", "Programs", "Startup"
    )
    if os.path.isdir(startup_dir):
        for item in os.listdir(startup_dir):
            p = os.path.join(startup_dir, item)
            lower = item.lower()
            if lower.endswith(".lnk"):
                entries.append({"path": p, "type": "lnk", "display": item})
            elif lower.endswith(".exe"):
                entries.append({"path": p, "type": "exe", "display": item})
            elif lower.endswith(".py"):
                entries.append({"path": p, "type": "py", "display": item})
            elif lower.endswith(".bat") or lower.endswith(".cmd"):
                entries.append({"path": p, "type": "batch", "display": item})
    return entries

def get_windows_registry_run_entries():
    entries = []
    if not winreg:
        return entries
    run_paths = [
        (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\Windows\CurrentVersion\Run"),
    ]
    for root, subkey in run_paths:
        try:
            with winreg.OpenKey(root, subkey) as key:
                i = 0
                while True:
                    try:
                        name, value, _vtype = winreg.EnumValue(key, i)
                        i += 1
                        entries.append({"path": value, "type": "reg_run", "display": name})
                    except OSError:
                        break
        except Exception:
            pass
    return entries

def get_mac_launch_agents_entries():
    entries = []
    d = os.path.expanduser("~/Library/LaunchAgents")
    if os.path.isdir(d):
        for item in os.listdir(d):
            if item.lower().endswith(".plist"):
                entries.append({"path": os.path.join(d, item), "type": "plist", "display": item})
    return entries

def get_mac_login_items_entries():
    entries = []
    try:
        out = subprocess.check_output(
            ["/usr/bin/osascript", "-e", 'tell application "System Events" to get the name of every login item'],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        names = [n.strip() for n in out.split(",") if n.strip()]
        for n in names:
            entries.append({"path": n, "type": "login_item", "display": n})
    except Exception:
        pass
    return entries

def get_linux_desktop_autostart_entries():
    entries = []
    d = os.path.expanduser("~/.config/autostart")
    if os.path.isdir(d):
        for item in os.listdir(d):
            if item.lower().endswith(".desktop"):
                entries.append({"path": os.path.join(d, item), "type": "desktop", "display": item})
    return entries

def get_linux_systemd_user_units():
    entries = []
    d = os.path.expanduser("~/.config/systemd/user")
    if os.path.isdir(d):
        for item in os.listdir(d):
            if item.endswith(".service"):
                entries.append({"path": os.path.join(d, item), "type": "systemd_service", "display": item})
    return entries

def get_startup_entries():
    if is_windows():
        return get_windows_startup_folder_entries() + get_windows_registry_run_entries()
    if is_mac():
        return get_mac_launch_agents_entries() + get_mac_login_items_entries()
    if is_linux():
        return get_linux_desktop_autostart_entries() + get_linux_systemd_user_units()
    return []

# ============================================================
# Launch entries
# ============================================================
def resolve_windows_lnk(path):
    try:
        if importlib.util.find_spec("win32com"):
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            sc = shell.CreateShortCut(path)
            return sc.Targetpath, sc.Arguments, sc.WorkingDirectory
    except Exception:
        pass
    return None, None, None

def parse_exec_from_desktop(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Exec="):
                    return line.strip().split("=", 1)[1]
    except Exception:
        pass
    return None

def launch_entry(entry, status: StatusBus):
    path, etype, display = entry["path"], entry["type"], entry["display"]
    try:
        if is_windows():
            proc = None
            if etype == "exe":
                proc = subprocess.Popen([path], shell=False)
            elif etype == "lnk":
                t, a, w = resolve_windows_lnk(path)
                if t:
                    cmd = [t] + ([a] if a else [])
                    proc = subprocess.Popen(cmd, cwd=w or None, shell=False)
                else:
                    proc = subprocess.Popen([path], shell=True)
            elif etype == "py":
                proc = subprocess.Popen([sys.executable, path], shell=False)
            elif etype == "batch":
                proc = subprocess.Popen(["cmd", "/c", path], shell=False)
            elif etype == "reg_run":
                # value is a command line; use shell=True to respect quoting
                proc = subprocess.Popen(path, shell=True)
            if proc:
                status.emit("startup_launched", {"name": display, "pid": proc.pid})
                return {"name": display, "pid": proc.pid, "proc": proc, "path": path}

        elif is_mac():
            if etype == "plist":
                subprocess.Popen(["launchctl", "load", "-w", path])
                status.emit("startup_launched", {"name": display, "pid": None})
                return {"name": display, "pid": None, "proc": None, "path": path}
            elif etype == "login_item":
                status.emit("startup_skipped", {"name": display, "reason": "Login item path unavailable"})
                return {"name": display, "pid": None, "proc": None, "path": path}

        elif is_linux():
            if etype == "desktop":
                cmd = parse_exec_from_desktop(path)
                if cmd:
                    proc = subprocess.Popen(cmd.split(), shell=False)
                    status.emit("startup_launched", {"name": display, "pid": proc.pid})
                    return {"name": display, "pid": proc.pid, "proc": proc, "path": cmd}
                else:
                    status.emit("startup_error", {"name": display, "error": "No Exec= found"})
            elif etype == "systemd_service":
                unit_name = os.path.basename(path)
                subprocess.Popen(["systemctl", "--user", "start", unit_name])
                status.emit("startup_launched", {"name": display, "pid": None})
                return {"name": display, "pid": None, "proc": None, "path": path}

    except Exception as e:
        status.emit("startup_error", {"name": display, "error": str(e)})

    return {"name": display, "pid": None, "proc": None, "path": path}

# ============================================================
# Monitor & auto-restart (quarantine after max)
# ============================================================
class ProcessMonitor:
    def __init__(self, status: StatusBus, max_restarts=5):
        self.status = status
        self.lock = threading.Lock()
        self.tracked = {}   # name -> {proc,pid,path,type,fails}
        self.quarantine = set()
        self.running = False
        self.max_restarts = max_restarts

    def add(self, entry_info, entry_type):
        name = entry_info["name"]
        with self.lock:
            self.tracked[name] = {
                "proc": entry_info.get("proc"),
                "pid": entry_info.get("pid"),
                "path": entry_info.get("path"),
                "type": entry_type,
                "fails": 0
            }

    def start(self, interval=5.0):
        if self.running:
            return
        self.running = True
        t = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        t.start()

    def _is_alive(self, info):
        proc = info.get("proc")
        pid = info.get("pid")
        if proc is not None:
            return proc.poll() is None
        if pid and psutil:
            try:
                return psutil.pid_exists(pid)
            except Exception:
                return False
        return False

    def _restart(self, name, info):
        self.status.emit("startup_restarting", {"name": name})
        entry = {"path": info["path"], "type": info["type"], "display": name}
        refreshed = launch_entry(entry, self.status)
        info.update({"proc": refreshed.get("proc"), "pid": refreshed.get("pid")})

    def _loop(self, interval):
        while self.running:
            try:
                with self.lock:
                    for name, info in list(self.tracked.items()):
                        if name in self.quarantine:
                            continue
                        alive = self._is_alive(info)
                        if not alive:
                            info["fails"] += 1
                            self.status.emit("startup_exited", {"name": name, "count": info["fails"]})
                            if info["fails"] <= self.max_restarts:
                                self._restart(name, info)
                            else:
                                self.quarantine.add(name)
                                self.status.emit("startup_failed", {"name": name, "msg": "Max restarts reached; quarantined"})
                time.sleep(interval)
            except Exception as e:
                self.status.emit("monitor_error", {"error": str(e)})
                time.sleep(interval)

    def stop(self):
        self.running = False

# ============================================================
# LLM bridge (auto-detect, simple heartbeat)
# ============================================================
class AutoLLMClient:
    def __init__(self, status: StatusBus):
        self.backend = None
        self.client = None
        try:
            if importlib.util.find_spec("openai") and os.getenv("OPENAI_API_KEY"):
                import openai
                self.client = openai
                self.backend = "OpenAI"
                status.emit("llm_connected", {"backend": "OpenAI"})
            elif importlib.util.find_spec("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
                import anthropic
                self.client = anthropic
                self.backend = "Anthropic"
                status.emit("llm_connected", {"backend": "Anthropic"})
            else:
                self.backend = "Stub"
                status.emit("llm_connected", {"backend": "Stub"})
        except Exception:
            status.emit("llm_error", {})

    def chat_loop(self, status: StatusBus):
        while True:
            try:
                status.emit("llm_tool", {"tool": "heartbeat"})
                time.sleep(10)
            except Exception:
                status.emit("llm_error", {})
                time.sleep(10)

# ============================================================
# GUI (Threat Matrix + Event Bus + Add/Delete/Launch selected)
# ============================================================
class BridgeGUI(tk.Tk):
    def __init__(self, status: StatusBus, config: dict):
        super().__init__()
        self.title("StartupBridge — LLM + Startup Orchestrator")
        self.geometry("1000x680")
        self.status = status
        self.config_data = config

        # Top bar
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)
        self.lbl_ready = ttk.Label(top, text="Bridge: initializing", foreground="black")
        self.lbl_ready.pack(side="left", padx=6)
        self.lbl_llm = ttk.Label(top, text="LLM: not connected", foreground="red")
        self.lbl_llm.pack(side="right", padx=6)

        # Tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        # Startup tab
        self.tab_startup = ttk.Frame(self.nb)
        self.nb.add(self.tab_startup, text="Startup programs")
        self.lst_startup = tk.Listbox(self.tab_startup, selectmode=tk.MULTIPLE)
        self.lst_startup.pack(fill="both", expand=True, padx=10, pady=10)

        # Threat Matrix tab
        self.tab_threat = ttk.Frame(self.nb)
        self.nb.add(self.tab_threat, text="Threat matrix")
        self.tree_threat = ttk.Treeview(self.tab_threat, columns=("fails", "status"), show="headings")
        self.tree_threat.heading("fails", text="Failures")
        self.tree_threat.heading("status", text="Status")
        self.tree_threat.column("fails", width=100, anchor="center")
        self.tree_threat.column("status", width=200, anchor="center")
        self.tree_threat.pack(fill="both", expand=True, padx=10, pady=10)

        # Logs tab
        self.tab_logs = ttk.Frame(self.nb)
        self.nb.add(self.tab_logs, text="Logs")
        self.txt_logs = tk.Text(self.tab_logs, wrap="word")
        self.txt_logs.pack(fill="both", expand=True, padx=10, pady=10)

        # Event Bus tab
        self.tab_bus = ttk.Frame(self.nb)
        self.nb.add(self.tab_bus, text="Event bus")
        self.txt_bus = tk.Text(self.tab_bus, wrap="word", height=10)
        self.txt_bus.pack(fill="both", expand=True, padx=10, pady=10)

        # Controls
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=6)
        self.btn_refresh = ttk.Button(controls, text="Refresh", command=self._refresh_list)
        self.btn_refresh.pack(side="left")
        self.btn_add = ttk.Button(controls, text="Add entry", command=self._add_entry)
        self.btn_add.pack(side="left", padx=8)
        self.btn_delete = ttk.Button(controls, text="Delete selected", command=self._delete_selected)
        self.btn_delete.pack(side="left", padx=8)
        self.btn_launch_selected = ttk.Button(controls, text="Launch selected", command=self._launch_selected)
        self.btn_launch_selected.pack(side="left", padx=8)
        self.btn_launch = ttk.Button(controls, text="Launch all", command=self._launch_all)
        self.btn_launch.pack(side="left", padx=8)

        # Config controls
        cfg_frame = ttk.Frame(self)
        cfg_frame.pack(fill="x", padx=10, pady=6)
        ttk.Label(cfg_frame, text="Monitor interval (sec):").pack(side="left")
        self.var_interval = tk.StringVar(value=str(self.config_data.get("monitor_interval", 5.0)))
        ttk.Entry(cfg_frame, textvariable=self.var_interval, width=6).pack(side="left", padx=5)
        ttk.Label(cfg_frame, text="Max restarts:").pack(side="left", padx=10)
        self.var_max = tk.StringVar(value=str(self.config_data.get("max_restarts", 5)))
        ttk.Entry(cfg_frame, textvariable=self.var_max, width=4).pack(side="left", padx=5)
        self.var_auto = tk.BooleanVar(value=bool(self.config_data.get("auto_launch", True)))
        ttk.Checkbutton(cfg_frame, text="Auto-launch at start", variable=self.var_auto, command=self._save_config).pack(side="left", padx=10)
        ttk.Button(cfg_frame, text="Save config", command=self._save_config).pack(side="left", padx=10)

        self.entries = []
        self.monitor = None

        # Initialize
        self._refresh_list()
        self.status.emit("bridge_ready", {"msg": "Bridge initialized"})
        self.after(300, self._poll_status)

    def _save_config(self):
        try:
            self.config_data["monitor_interval"] = float(self.var_interval.get())
            self.config_data["max_restarts"] = int(self.var_max.get())
            self.config_data["auto_launch"] = bool(self.var_auto.get())
            save_config(self.config_data)
            self._log("Config saved.")
        except Exception as e:
            self._log(f"Config save error: {e}")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.txt_logs.insert("end", f"[{ts}] {msg}\n")
        self.txt_logs.see("end")

    def _bus(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.txt_bus.insert("end", f"[{ts}] {msg}\n")
        self.txt_bus.see("end")

    def _refresh_list(self):
        self.entries = get_startup_entries()
        self.lst_startup.delete(0, tk.END)
        if not self.entries:
            self._log("No startup entries found.")
        for e in self.entries:
            self.lst_startup.insert(tk.END, f"{e['display']}  ({e['type']})")
        self._log(f"Found {len(self.entries)} startup entries.")

        # Refresh Threat Matrix
        for row in self.tree_threat.get_children():
            self.tree_threat.delete(row)
        for e in self.entries:
            self.tree_threat.insert("", "end", iid=e["display"], values=("0", "Idle"))

    def _add_entry(self):
        path = filedialog.askopenfilename(title="Select program/script to add")
        if not path:
            return
        display = os.path.basename(path)
        lower = display.lower()
        if lower.endswith(".py"):
            etype = "py"
        elif lower.endswith(".lnk"):
            etype = "lnk"
        elif lower.endswith(".exe"):
            etype = "exe"
        elif lower.endswith(".bat") or lower.endswith(".cmd"):
            etype = "batch"
        else:
            # Fallback: treat as command (Windows) or executable (others)
            etype = "exe" if is_windows() else "desktop"
        entry = {"path": path, "type": etype, "display": display}
        self.entries.append(entry)
        self.lst_startup.insert(tk.END, f"{display}  ({etype})")
        # Threat matrix row
        self.tree_threat.insert("", "end", iid=display, values=("0", "Idle"))
        self._log(f"Added entry: {display}")

    def _delete_selected(self):
        selected = list(self.lst_startup.curselection())
        if not selected:
            self._log("No entries selected for deletion.")
            return
        for idx in reversed(selected):
            e = self.entries[idx]
            name = e["display"]
            self._log(f"Deleted entry: {name}")
            del self.entries[idx]
            self.lst_startup.delete(idx)
            if name in self.tree_threat.get_children():
                self.tree_threat.delete(name)

    def _ensure_monitor(self):
        interval = float(self.var_interval.get())
        max_restarts = int(self.var_max.get())
        if self.monitor is None:
            self.monitor = ProcessMonitor(self.status, max_restarts=max_restarts)
        self.monitor.start(interval=interval)

    def _launch_selected(self):
        selected = self.lst_startup.curselection()
        if not selected:
            self._log("No entries selected.")
            return
        self._log(f"Launching {len(selected)} selected entries...")
        self._ensure_monitor()
        for idx in selected:
            e = self.entries[idx]
            info = launch_entry(e, self.status)
            self.monitor.add(info, e["type"])
        self._log("Selected entries launched; monitoring started.")

    def _launch_all(self):
        if not self.entries:
            self._log("No entries to launch.")
            return
        self._log("Launching all startup entries...")
        self._ensure_monitor()
        for e in self.entries:
            info = launch_entry(e, self.status)
            self.monitor.add(info, e["type"])
        self._log("All entries launched; monitoring started.")

    def _poll_status(self):
        evt = self.status.get_nowait()
        if evt:
            event = evt["event"]
            payload = evt["payload"]
            self._bus(f"{event}: {payload}")

            if event == "bridge_ready":
                self.lbl_ready.config(text="Bridge: ready", foreground="green")
                self._log("Bridge ready.")
            elif event == "startup_launched":
                self._log(f"Launched: {payload.get('name')} (pid={payload.get('pid')})")
                name = payload.get("name")
                if name and name in self.tree_threat.get_children():
                    self.tree_threat.set(name, "fails", "0")
                    self.tree_threat.set(name, "status", "Running")
            elif event == "startup_skipped":
                self._log(f"Skipped: {payload.get('name')} — {payload.get('reason')}")
                name = payload.get("name")
                if name and name in self.tree_threat.get_children():
                    self.tree_threat.set(name, "status", "Skipped")
            elif event == "startup_error":
                self._log(f"Launch failed: {payload.get('name')} — {payload.get('error')}")
                name = payload.get("name")
                if name and name in self.tree_threat.get_children():
                    self.tree_threat.set(name, "status", "Error")
            elif event == "startup_exited":
                name = payload.get('name')
                count = str(payload.get('count'))
                self._log(f"Exited: {name} (count={count})")
                if name and name in self.tree_threat.get_children():
                    self.tree_threat.set(name, "fails", count)
                    self.tree_threat.set(name, "status", "Exited")
            elif event == "startup_restarting":
                self._log(f"Restarting: {payload.get('name')}")
                name = payload.get("name")
                if name and name in self.tree_threat.get_children():
                    self.tree_threat.set(name, "status", "Restarting")
            elif event == "startup_failed":
                self._log(f"Permanent failure: {payload.get('name')} — {payload.get('msg')}")
                name = payload.get("name")
                if name and name in self.tree_threat.get_children():
                    self.tree_threat.set(name, "status", "Quarantined")
            elif event == "monitor_error":
                self._log(f"Monitor error: {payload.get('error')}")
            elif event == "llm_connected":
                backend = payload.get("backend", "Unknown")
                self.lbl_llm.config(text=f"LLM: {backend}", foreground="green")
                self._log(f"LLM connected: {backend}")
            elif event == "llm_error":
                self.lbl_llm.config(text="LLM: error", foreground="orange")
                self._log("LLM error")
            elif event == "llm_tool":
                tool = payload.get("tool", "")
                self.lbl_llm.config(text=f"LLM: active ({tool})", foreground="blue")
                self._log(f"LLM activity: {tool}")

        self.after(300, self._poll_status)

    def destroy(self):
        try:
            if self.monitor:
                self.monitor.stop()
        except Exception:
            pass
        super().destroy()

# ============================================================
# Entry point
# ============================================================
def main():
    status = StatusBus()
    cfg = load_config()

    app = BridgeGUI(status, cfg)

    def llm_thread():
        try:
            client = AutoLLMClient(status)
            client.chat_loop(status)
        except Exception:
            status.emit("llm_error", {})
            traceback.print_exc()

    t = threading.Thread(target=llm_thread, daemon=True)
    t.start()

    def auto_launch():
        if app.var_auto.get():
            app._launch_all()

    app.after(1500, auto_launch)
    app.mainloop()

if __name__ == "__main__":
    main()

