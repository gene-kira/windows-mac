import sys
import os
import subprocess
from pathlib import Path

# -------------------------
# Auto-loader & elevation (smart + ruthless)
# -------------------------

# core dependency list
REQUIRED_PACKAGES = {
    "psutil": {
        "min_version": "5.9.0"
    },
    "PySide6": {
        "min_version": "6.4.0"
    }
}

# autoloader mode: "safe", "normal", "aggressive"
AUTOLOADER_MODE = os.environ.get("GUARDIAN_AUTOLOADER_MODE", "normal").lower()


def is_windows():
    return os.name == "nt"


def is_admin():
    if not is_windows():
        return False
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def _run_pip(args):
    cmd = [sys.executable, "-m", "pip"] + args
    return subprocess.check_call(cmd)


def ensure_pip():
    """Make sure pip is available."""
    try:
        import pip  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import ensurepip
        ensurepip.bootstrap()
    except Exception as e:
        print(f"[guardian] Failed to bootstrap pip: {e}")
        raise


def parse_version(v: str):
    return tuple(int(p) for p in v.split(".") if p.isdigit())


def get_installed_version(pkg: str):
    try:
        import importlib.metadata as importlib_metadata
    except ImportError:
        import importlib_metadata  # type: ignore
    try:
        return importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def needs_install_or_upgrade(pkg: str, spec: dict):
    installed = get_installed_version(pkg)
    if installed is None:
        return True, "not installed"
    min_ver = spec.get("min_version")
    if not min_ver:
        return False, f"installed {installed}"
    try:
        if parse_version(installed) < parse_version(min_ver):
            return True, f"version {installed} < required {min_ver}"
    except Exception:
        return True, "failed to compare versions"
    return False, f"installed {installed}"


def install_or_upgrade_package(pkg: str, spec: dict):
    ensure_pip()
    should, reason = needs_install_or_upgrade(pkg, spec)
    if not should and AUTOLOADER_MODE in ("safe", "normal"):
        print(f"[guardian] {pkg}: OK ({reason})")
        return
    if AUTOLOADER_MODE == "safe" and get_installed_version(pkg) is not None:
        # in safe mode, don't touch installed packages, only install missing
        print(f"[guardian] {pkg}: installed, safe mode, skipping upgrade ({reason})")
        return

    # in aggressive mode, always install/upgrade
    if AUTOLOADER_MODE == "aggressive":
        should = True
        reason = f"aggressive mode: forcing install/upgrade (current={get_installed_version(pkg)})"

    print(f"[guardian] Installing/upgrading {pkg}: {reason}")
    try:
        _run_pip(["install", "--upgrade", pkg])
        print(f"[guardian] {pkg}: installation/upgrade complete")
    except subprocess.CalledProcessError as e:
        print(f"[guardian] Failed to install/upgrade {pkg}: {e}")
        raise


def ensure_dependencies():
    """
    Ensure all required packages are present and meet minimum versions.
    Behavior depends on AUTOLOADER_MODE:
      - safe: only install missing, never upgrade existing
      - normal: install missing, upgrade if clearly below min version
      - aggressive: always verify and attempt to upgrade to latest
    """
    print(f"[guardian] Autoloader mode: {AUTOLOADER_MODE}")
    for pkg, spec in REQUIRED_PACKAGES.items():
        try:
            install_or_upgrade_package(pkg, spec)
        except Exception as e:
            print(f"[guardian] Autoloader failed for {pkg}: {e}")
            raise


def elevate_if_needed():
    """
    Relaunch the script with admin rights if not already elevated.
    Skip when running in service mode (service already runs as SYSTEM).
    """
    if not is_windows():
        return
    # If this is going to run as a service, let the service wrapper handle rights
    if "--service" in sys.argv:
        return
    if is_admin():
        return

    import ctypes
    params = " ".join([f'"{a}"' for a in sys.argv[1:]])
    try:
        ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            f'"{Path(sys.argv[0]).resolve()}" {params}',
            None,
            1
        )
        sys.exit(0)
    except Exception as e:
        print(f"[guardian] Failed to self-elevate: {e}")
        sys.exit(1)


# run autoloader + elevation as first stage
ensure_dependencies()
elevate_if_needed()

# -------------------------
# Imports after environment ready
# -------------------------

import json
import time
import threading
import traceback
from datetime import datetime

import psutil
import ctypes
from ctypes import wintypes
from PySide6 import QtWidgets, QtCore

# Force Qt to use Windows platform plugin explicitly
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

# -------------------------
# Windows UI / window helpers
# -------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)

EnumWindows = user32.EnumWindows
EnumWindows.argtypes = [ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
EnumWindows.restype = ctypes.c_bool

IsWindowVisible = user32.IsWindowVisible
IsWindowVisible.argtypes = [wintypes.HWND]
IsWindowVisible.restype = wintypes.BOOL

GetWindowThreadProcessId = user32.GetWindowThreadProcessId
GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
GetWindowThreadProcessId.restype = wintypes.DWORD

IsHungAppWindow = user32.IsHungAppWindow
IsHungAppWindow.argtypes = [wintypes.HWND]
IsHungAppWindow.restype = wintypes.BOOL

GetWindowTextW = user32.GetWindowTextW
GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
GetWindowTextW.restype = ctypes.c_int

# -------------------------
# Logging
# -------------------------

class Logger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def log(self, event_type: str, message: str, extra=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "message": message,
            "extra": extra or {}
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

# -------------------------
# Policy engine
# -------------------------

class PolicyEngine:
    def __init__(self, config: dict):
        self.config = config

    def get_global(self):
        return self.config.get("global", {})

    def get_logging(self):
        return self.config.get("logging", {})

    def get_policy_for_process(self, proc_name: str) -> dict:
        g = self.get_global().copy()
        per = self.config.get("per_process", {})
        p = per.get(proc_name.lower()) or per.get(proc_name)
        if p:
            g.update(p)
        return g

    def get_restart_config(self, proc_name: str):
        restartable = self.config.get("restartable_apps", {})
        return restartable.get(proc_name.lower()) or restartable.get(proc_name)

    def get_monitored_services(self):
        services = self.config.get("services", {})
        return services.get("monitor", [])

    def get_action_for_incident(self, proc_name: str, reason: str) -> str:
        """
        Resolve action for a given process + incident reason.
        Priority:
          1) per_process_actions[proc][reason]
          2) actions[reason]
          3) actions['default']
          4) fallback 'kill'
        """
        name_key = proc_name.lower() if proc_name else ""
        per_proc_actions = self.config.get("per_process_actions", {})
        proc_actions = per_proc_actions.get(name_key) or per_proc_actions.get(proc_name or "") or {}
        if reason in proc_actions:
            return proc_actions[reason]

        actions = self.config.get("actions", {})
        if reason in actions:
            return actions[reason]
        if "default" in actions:
            return actions["default"]
        return "kill"

    def is_ruthless(self) -> bool:
        g = self.get_global()
        return bool(g.get("ruthless_mode", False))

# -------------------------
# Guardian core
# -------------------------

class WindowsGuardian:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.policy = PolicyEngine(self.config)

        log_cfg = self.policy.get_logging()
        log_path = Path(log_cfg.get("log_path", "guardian_log.txt"))
        self.log_path = log_path

        # ensure log file exists
        if not self.log_path.exists():
            try:
                self.log_path.write_text("", encoding="utf-8")
            except Exception as e:
                print(f"[guardian] Failed to create log file: {e}")

        self.logger = Logger(log_path)

        self.history = {}              # pid -> list of snapshots
        self.not_responding_since = {} # pid -> timestamp
        self.last_restart_time = {}    # proc_name -> timestamp

        self._stop = False
        self._paused = False
        self._control_path = self.log_path.with_name("guardian_control.json")

    def run(self):
        g = self.policy.get_global()
        interval = g.get("check_interval_sec", 5)
        self.logger.log("info", "WindowsGuardian started", {"interval_sec": interval})
        while not self._stop:
            try:
                self._check_control()
                if not self._paused:
                    self.scan_system()
            except Exception as e:
                self.logger.log("error", "Unhandled exception in scan_system",
                                {"error": str(e), "trace": traceback.format_exc()})
            time.sleep(interval)
        self.logger.log("info", "WindowsGuardian stopped")

    def stop(self):
        self._stop = True

    def pause(self):
        self._paused = True
        self.logger.log("info", "Guardian paused", {})

    def resume(self):
        self._paused = False
        self.logger.log("info", "Guardian resumed", {})

    def _check_control(self):
        """
        Read control file (if exists) for commands: pause/resume/reload.
        """
        if not self._control_path.exists():
            return
        try:
            data = json.loads(self._control_path.read_text(encoding="utf-8"))
        except Exception:
            return

        cmd = data.get("command")
        if cmd == "pause":
            self.pause()
        elif cmd == "resume":
            self.resume()
        elif cmd == "reload_config":
            try:
                self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.policy = PolicyEngine(self.config)
                self.logger.log("info", "Config reloaded via control", {})
            except Exception as e:
                self.logger.log("error", "Failed to reload config", {"error": str(e)})

        # Clear command after processing
        try:
            self._control_path.unlink()
        except Exception:
            pass

    def scan_system(self):
        self._scan_processes()
        self._scan_services()
        self._scan_system_resources()

    def _scan_processes(self):
        hung_pids = self._get_hung_pids()

        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                pid = info["pid"]
                name = (info["name"] or "").lower()
                cpu = info["cpu_percent"]
                mem_bytes = info["memory_info"].rss
                now = time.time()

                snapshot = {
                    "pid": pid,
                    "name": name,
                    "cpu_percent": cpu,
                    "memory_bytes": mem_bytes,
                    "timestamp": now
                }

                self._update_history(snapshot)
                policy = self.policy.get_policy_for_process(name)

                if policy.get("ignore"):
                    continue

                self._check_cpu(proc, snapshot, policy)
                self._check_memory(proc, snapshot, policy)
                self._check_ui_hang(proc, snapshot, policy, hung_pids)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        alive_pids = {p.pid for p in psutil.process_iter(["pid"])}
        self.history = {pid: h for pid, h in self.history.items() if pid in alive_pids}
        self.not_responding_since = {pid: t for pid, t in self.not_responding_since.items() if pid in alive_pids}

    def _scan_services(self):
        try:
            monitor_list = self.policy.get_monitored_services()
            if not monitor_list:
                return
            for svc in psutil.win_service_iter():
                try:
                    name = svc.name()
                    if name not in monitor_list:
                        continue
                    status = svc.status()
                    if status not in ("running", "start_pending"):
                        self.logger.log("service_warn",
                                        f"Service {name} not running (status={status}), attempting restart",
                                        {"service": name, "status": status})
                        try:
                            svc = psutil.win_service_get(name)
                            svc.start()
                            self.logger.log("service_action",
                                            f"Service {name} start requested",
                                            {"service": name})
                        except Exception as e:
                            self.logger.log("service_error",
                                            f"Failed to restart service {name}",
                                            {"service": name, "error": str(e)})
                except Exception:
                    continue
        except Exception as e:
            self.logger.log("error", "Service scan failed", {"error": str(e)})

    def _scan_system_resources(self):
        g = self.policy.get_global()

        # Disk
        try:
            root = Path("/").anchor or "C:\\"
            usage = psutil.disk_usage(root)
            free_percent = 100.0 * usage.free / usage.total
            min_free = g.get("disk_free_percent_min", 10)
            if free_percent < min_free:
                self.logger.log("disk_warn",
                                "Low disk space on system drive",
                                {"free_percent": free_percent, "min_free_percent": min_free})
        except Exception as e:
            self.logger.log("error", "Disk scan failed", {"error": str(e)})

        # Network
        try:
            now = time.time()
            net = psutil.net_io_counters()
            if not hasattr(self, "_prev_net"):
                self._prev_net = (now, net.bytes_sent, net.bytes_recv)
                return
            prev_time, prev_sent, prev_recv = self._prev_net
            dt = max(now - prev_time, 0.001)
            sent_rate = (net.bytes_sent - prev_sent) / dt
            recv_rate = (net.bytes_recv - prev_recv) / dt
            self._prev_net = (now, net.bytes_sent, net.bytes_recv)

            threshold = g.get("network_bytes_per_sec_warn", None)
            if threshold is not None:
                if sent_rate > threshold or recv_rate > threshold:
                    self.logger.log("net_warn",
                                    "High network throughput detected",
                                    {"sent_bytes_per_sec": sent_rate,
                                     "recv_bytes_per_sec": recv_rate,
                                     "threshold": threshold})
        except Exception as e:
            self.logger.log("error", "Network scan failed", {"error": str(e)})

    def _update_history(self, snapshot: dict):
        pid = snapshot["pid"]
        hist = self.history.setdefault(pid, [])
        hist.append(snapshot)
        cutoff = time.time() - 300
        self.history[pid] = [s for s in hist if s["timestamp"] >= cutoff]

    def _is_cpu_excessive(self, pid: int, max_cpu_percent: float, max_seconds: int) -> bool:
        hist = self.history.get(pid, [])
        if not hist:
            return False
        now = time.time()
        relevant = [s for s in hist if now - s["timestamp"] <= max_seconds]
        if not relevant:
            return False
        return all(s["cpu_percent"] >= max_cpu_percent for s in relevant)

    def _check_cpu(self, proc: psutil.Process, snapshot: dict, policy: dict):
        pid = snapshot["pid"]
        max_cpu = policy.get("max_cpu_percent", 95)
        max_cpu_seconds = policy.get("max_cpu_seconds", 60)
        if self._is_cpu_excessive(pid, max_cpu, max_cpu_seconds):
            self._handle_incident(proc, snapshot, reason="cpu_excessive")

    def _check_memory(self, proc: psutil.Process, snapshot: dict, policy: dict):
        max_mem_mb = policy.get("max_memory_mb", None)
        if max_mem_mb is None:
            return
        mem_mb = snapshot["memory_bytes"] / (1024 * 1024)
        if mem_mb > max_mem_mb:
            self._handle_incident(proc, snapshot, reason="memory_excessive")

    def _check_ui_hang(self, proc: psutil.Process, snapshot: dict, policy: dict, hung_pids: set):
        pid = snapshot["pid"]
        g = self.policy.get_global()
        max_not_responding = policy.get("max_not_responding_sec",
                                        g.get("max_not_responding_sec", 30))

        now = time.time()
        if pid in hung_pids:
            if pid not in self.not_responding_since:
                self.not_responding_since[pid] = now
            elapsed = now - self.not_responding_since[pid]
            if elapsed >= max_not_responding:
                self._handle_incident(proc, snapshot, reason="ui_not_responding")
                self.not_responding_since.pop(pid, None)
        else:
            if pid in self.not_responding_since:
                self.logger.log("info",
                                "Process recovered from UI not responding",
                                {"pid": pid, "name": snapshot["name"]})
                self.not_responding_since.pop(pid, None)

    def _severity_for_reason(self, reason: str, ruthless: bool) -> str:
        """
        Simple severity mapping; can be evolved later.
        """
        high = {"cpu_excessive", "memory_excessive", "ui_not_responding"}
        if reason in high:
            return "critical" if ruthless else "high"
        return "medium"

    def _execute_kill_action(self, proc: psutil.Process, info: dict, kill_tree: bool):
        """
        Execute kill or kill_tree, with logging.
        """
        name = info.get("name") or f"pid:{info.get('pid')}"
        try:
            if kill_tree:
                children = proc.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except Exception:
                        pass
                gone, alive = psutil.wait_procs(children, timeout=3)
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass
                self.logger.log("action",
                                f"Children terminated for {name}",
                                {**info, "children_count": len(children)})

            # Kill parent
            proc.terminate()
            gone, alive = psutil.wait_procs([proc], timeout=5)
            if alive:
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass

            self.logger.log("action",
                            f"Process terminated ({'tree' if kill_tree else 'single'}) for {name}",
                            info)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.log("error",
                            "Failed to terminate process",
                            {"error": str(e), **info})

    def _handle_incident(self, proc: psutil.Process, snapshot: dict, reason: str):
        pid = snapshot["pid"]
        name = snapshot["name"]
        proc_name_display = name or f"pid:{pid}"

        # Decide action
        action = self.policy.get_action_for_incident(name, reason)
        ruthless = self.policy.is_ruthless()

        # In ruthless mode, escalate some actions
        if ruthless:
            if action in ("warn", "log_only"):
                action = "kill"
            elif action == "kill":
                action = "kill_tree"

        severity = self._severity_for_reason(reason, ruthless)

        info = {
            "pid": pid,
            "name": name,
            "reason": reason,
            "cpu_percent": snapshot["cpu_percent"],
            "memory_bytes": snapshot["memory_bytes"],
            "action": action,
            "severity": severity
        }

        self.logger.log(
            "incident",
            f"Incident detected ({reason}) on {proc_name_display}, action={action}, severity={severity}",
            info
        )

        # Non-kill actions
        if action == "log_only":
            return

        if action == "warn":
            self.logger.log(
                "warn",
                f"Warning only for {proc_name_display} ({reason})",
                info
            )
            return

        # Kill / Kill-tree
        if action in ("kill", "kill_tree"):
            self._execute_kill_action(proc, info, kill_tree=(action == "kill_tree"))

        # Attempt restart if configured (only after kill/kill_tree)
        if action in ("kill", "kill_tree"):
            restart_cfg = self.policy.get_restart_config(name)
            if restart_cfg:
                self._maybe_restart(name, restart_cfg)

    def _maybe_restart(self, proc_name: str, restart_cfg: dict):
        now = time.time()
        cooldown = restart_cfg.get("cooldown_sec", 30)
        last = self.last_restart_time.get(proc_name)
        if last is not None and now - last < cooldown:
            self.logger.log("info",
                            "Skipping restart due to cooldown",
                            {"proc_name": proc_name, "cooldown_sec": cooldown})
            return

        cmd = restart_cfg.get("launch_cmd")
        if not cmd:
            return

        try:
            subprocess.Popen(cmd, shell=True)
            self.last_restart_time[proc_name] = now
            self.logger.log("action",
                            "Restarted application",
                            {"proc_name": proc_name, "cmd": cmd})
        except Exception as e:
            self.logger.log("error",
                            "Failed to restart application",
                            {"proc_name": proc_name, "cmd": cmd, "error": str(e)})

    def _get_hung_pids(self) -> set:
        hung_pids = set()

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, lparam):
            try:
                if not IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD()
                GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                proc_pid = pid.value
                if proc_pid == 0:
                    return True
                if IsHungAppWindow(hwnd):
                    title_buf = ctypes.create_unicode_buffer(256)
                    GetWindowTextW(hwnd, title_buf, 256)
                    hung_pids.add(proc_pid)
            except Exception:
                pass
            return True

        try:
            EnumWindows(enum_proc, 0)
        except Exception as e:
            self.logger.log("error",
                            "EnumWindows failed for hung detection",
                            {"error": str(e)})

        return hung_pids

# -------------------------
# GUI dashboard (PySide6)
# -------------------------

class GuardianLogModel(QtCore.QAbstractTableModel):
    """
    Simple table model that reads last N log entries from the log file.
    Columns: Time, Type, Message
    """

    def __init__(self, log_path: Path, parent=None):
        super().__init__(parent)
        self.log_path = log_path
        self.entries = []
        self.max_rows = 500
        self.load_initial()

    def load_initial(self):
        self.entries = []
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-self.max_rows:]
            for line in lines:
                try:
                    obj = json.loads(line)
                    self.entries.append(obj)
                except Exception:
                    continue
        except Exception:
            pass

    def refresh(self):
        """Reloads log file and keeps last N entries."""
        old_len = len(self.entries)
        self.load_initial()
        if len(self.entries) != old_len:
            self.layoutChanged.emit()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.entries)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or role != QtCore.Qt.DisplayRole:
            return None
        row = index.row()
        col = index.column()
        entry = self.entries[row]
        if col == 0:
            return entry.get("timestamp", "")
        elif col == 1:
            return entry.get("event_type", "")
        elif col == 2:
            return entry.get("message", "")
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or orientation != QtCore.Qt.Horizontal:
            return None
        if section == 0:
            return "Time"
        if section == 1:
            return "Type"
        if section == 2:
            return "Message"
        return None


class GuardianGUI(QtWidgets.QWidget):
    def __init__(self, config_path: Path):
        super().__init__()
        self.setWindowTitle("Windows Guardian Dashboard")
        self.resize(900, 500)

        # Load config to find log path
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        log_cfg = cfg.get("logging", {})
        log_path = Path(log_cfg.get("log_path", "guardian_log.txt"))

        self.log_path = log_path
        self.control_path = log_path.with_name("guardian_control.json")

        # Ensure log file exists
        if not self.log_path.exists():
            try:
                self.log_path.write_text("", encoding="utf-8")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error",
                                              f"Failed to create log file:\n{e}")

        # Model creation with safety
        try:
            self.model = GuardianLogModel(log_path, self)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "GUI Error",
                                           f"Failed to initialize log model:\n{e}")
            self.model = None

        self.table = QtWidgets.QTableView(self)
        if self.model:
            self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self.btn_pause = QtWidgets.QPushButton("Pause Guardian")
        self.btn_resume = QtWidgets.QPushButton("Resume Guardian")
        self.btn_reload = QtWidgets.QPushButton("Reload Config")
        self.btn_open_log = QtWidgets.QPushButton("Open Log File")

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_reload)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_open_log)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(btn_layout)
        layout.addWidget(self.table)

        self.btn_pause.clicked.connect(self.send_pause)
        self.btn_resume.clicked.connect(self.send_resume)
        self.btn_reload.clicked.connect(self.send_reload)
        self.btn_open_log.clicked.connect(self.open_log)

        # Timer to refresh log view
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_model)
        self.timer.start(3000)

    def refresh_model(self):
        if self.model:
            self.model.refresh()

    def send_command(self, command: str):
        data = {"command": command, "timestamp": datetime.utcnow().isoformat() + "Z"}
        try:
            self.control_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          f"Failed to write control file:\n{e}")

    def send_pause(self):
        self.send_command("pause")

    def send_resume(self):
        self.send_command("resume")

    def send_reload(self):
        self.send_command("reload_config")

    def open_log(self):
        if not self.log_path.exists():
            QtWidgets.QMessageBox.information(self, "Info", "Log file does not exist yet.")
            return
        try:
            os.startfile(str(self.log_path))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          f"Failed to open log file:\n{e}")

# -------------------------
# Main entrypoints
# -------------------------

DEFAULT_CONFIG = {
    "global": {
        "check_interval_sec": 5,
        "max_cpu_percent": 90,
        "max_cpu_seconds": 60,
        "max_memory_mb": 2048,
        "max_not_responding_sec": 30,
        "disk_free_percent_min": 10,
        "network_bytes_per_sec_warn": 12500000,
        "ruthless_mode": False
    },
    "per_process": {
        "chrome.exe": {
            "max_memory_mb": 4096,
            "max_cpu_percent": 95
        },
        "my_game.exe": {
            "ignore": True
        }
    },
    "actions": {
        "cpu_excessive": "kill_tree",
        "memory_excessive": "kill_tree",
        "ui_not_responding": "kill",
        "service_down": "restart_service",
        "default": "kill"
    },
    "per_process_actions": {
        "chrome.exe": {
            "cpu_excessive": "warn",
            "memory_excessive": "kill_tree"
        }
    },
    "restartable_apps": {
        "notepad.exe": {
            "launch_cmd": "notepad.exe",
            "cooldown_sec": 30
        }
    },
    "services": {
        "monitor": [
            "Spooler",
            "wuauserv"
        ]
    },
    "logging": {
        "log_path": "guardian_log.txt"
    }
}


def ensure_config(path: Path):
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


def run_service_mode(config_path: Path):
    guardian = WindowsGuardian(config_path)
    try:
        guardian.run()
    except KeyboardInterrupt:
        guardian.stop()


def run_gui_mode(config_path: Path):
    try:
        app = QtWidgets.QApplication(sys.argv)
        gui = GuardianGUI(config_path)
        gui.show()
        sys.exit(app.exec())
    except Exception:
        print("GUI crashed:")
        traceback.print_exc()
        input("Press Enter to exit...")


def main():
    if not is_windows():
        print("This guardian is designed for Windows only.")
        sys.exit(1)

    config_path = Path("guardian_config.json")
    ensure_config(config_path)

    # Debug: show args
    print("DEBUG sys.argv:", sys.argv)

    args = [a.lower() for a in sys.argv[1:]]

    if "--service" in args or "/service" in args or "-service" in args:
        run_service_mode(config_path)
    elif "--gui" in args or "/gui" in args or "-gui" in args:
        run_gui_mode(config_path)
    else:
        # No valid args: default to GUI (double-click, plain run, etc.)
        print("No mode specified, defaulting to GUI dashboard.")
        run_gui_mode(config_path)


if __name__ == "__main__":
    main()

