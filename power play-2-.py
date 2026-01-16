import sys
import os
import ctypes
import subprocess
import time
import threading
import json
import shutil

import psutil
import winreg
import win32service
import win32serviceutil

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QHBoxLayout, QMessageBox, QHeaderView, QLabel, QToolBar,
    QAction, QTextEdit, QProgressBar, QStyleFactory
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QColor, QPalette

# ---------------- Elevation ----------------

def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False


def relaunch_as_admin():
    params = " ".join(f'"{a}"' for a in sys.argv[1:])
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable,
        f'"{os.path.abspath(sys.argv[0])}" {params}',
        None, 1
    )
    sys.exit(0)


# ---------------- Win32 Installed apps helpers ----------------

UNINSTALL_PATHS = [
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
]


def list_installed_apps():
    apps = []
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for path in UNINSTALL_PATHS:
            try:
                key = winreg.OpenKey(root, path)
            except FileNotFoundError:
                continue
            sub_count, _, _ = winreg.QueryInfoKey(key)
            for i in range(sub_count):
                try:
                    sub_name = winreg.EnumKey(key, i)
                    sub_key = winreg.OpenKey(key, sub_name)
                    name = winreg.QueryValueEx(sub_key, "DisplayName")[0]
                    uninstall = winreg.QueryValueEx(sub_key, "UninstallString")[0]
                    apps.append({"name": name, "uninstall": uninstall})
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                except Exception:
                    continue
    unique = {}
    for app in apps:
        key = (app["name"], app["uninstall"])
        unique[key] = app
    return list(unique.values())


def uninstall_app(uninstall_cmd: str):
    try:
        subprocess.Popen(uninstall_cmd, shell=True)
    except Exception as e:
        print("Uninstall error:", e)


# ---------------- Services helpers ----------------

def list_services():
    services = []
    scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ENUMERATE_SERVICE)
    try:
        statuses = win32service.EnumServicesStatus(scm)
        for (svc_name, display_name, status) in statuses:
            current_state = status[1]
            services.append({
                "name": svc_name,
                "display_name": display_name,
                "state": current_state,
            })
    finally:
        win32service.CloseServiceHandle(scm)
    return services


def service_state_to_text(state: int) -> str:
    mapping = {
        win32service.SERVICE_STOPPED: "Stopped",
        win32service.SERVICE_START_PENDING: "Start Pending",
        win32service.SERVICE_STOP_PENDING: "Stop Pending",
        win32service.SERVICE_RUNNING: "Running",
        win32service.SERVICE_CONTINUE_PENDING: "Continue Pending",
        win32service.SERVICE_PAUSE_PENDING: "Pause Pending",
        win32service.SERVICE_PAUSED: "Paused",
    }
    return mapping.get(state, f"Unknown ({state})")


def start_service(name: str):
    win32serviceutil.StartService(name)


def stop_service(name: str):
    win32serviceutil.StopService(name)


# ---------------- Startup items helpers ----------------

def get_startup_folder():
    from win32com.shell import shell, shellcon
    return shell.SHGetFolderPath(0, shellcon.CSIDL_STARTUP, None, 0)


def list_startup_items():
    items = []
    startup = get_startup_folder()
    if not os.path.isdir(startup):
        return items
    for f in os.listdir(startup):
        full = os.path.join(startup, f)
        if os.path.isfile(full) and f.lower().endswith(".lnk"):
            items.append({"name": f, "path": full, "enabled": True})
    disabled_dir = os.path.join(startup, "_Disabled")
    if os.path.isdir(disabled_dir):
        for f in os.listdir(disabled_dir):
            full = os.path.join(disabled_dir, f)
            if os.path.isfile(full) and f.lower().endswith(".lnk"):
                items.append({"name": f, "path": full, "enabled": False})
    return items


def toggle_startup_item(item):
    startup = get_startup_folder()
    disabled_dir = os.path.join(startup, "_Disabled")
    if not os.path.isdir(disabled_dir):
        os.makedirs(disabled_dir, exist_ok=True)

    path = item["path"]
    name = item["name"]
    if item["enabled"]:
        target = os.path.join(disabled_dir, name)
    else:
        target = os.path.join(startup, name)

    try:
        shutil.move(path, target)
        item["path"] = target
        item["enabled"] = not item["enabled"]
    except Exception as e:
        print("Error toggling startup item:", e)


# ---------------- UWP / Provisioned / Optional features helpers ----------------

def _run_powershell_json(ps_command: str):
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        ps_command + " | ConvertTo-Json"
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
        if not output.strip():
            return []
        data = json.loads(output)
        if isinstance(data, dict):
            return [data]
        return data
    except subprocess.CalledProcessError as e:
        print("PowerShell error:", e)
        return []
    except json.JSONDecodeError:
        return []


def list_uwp_apps():
    # Name, PackageFullName
    return _run_powershell_json("Get-AppxPackage | Select Name, PackageFullName")


def list_provisioned_apps():
    # DisplayName, PackageName
    return _run_powershell_json("Get-AppxProvisionedPackage -Online | Select DisplayName, PackageName")


def list_optional_features():
    # FeatureName, State
    return _run_powershell_json("Get-WindowsOptionalFeature -Online | Select FeatureName, State")


def remove_uwp_package(package_full_name: str):
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"Remove-AppxPackage -Package '{package_full_name}'"
    ]
    subprocess.Popen(cmd, shell=True)


# ---------------- Plugin loader (simple hook) ----------------

def load_plugins():
    plugins_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
    if not os.path.isdir(plugins_dir):
        return []
    sys.path.append(plugins_dir)
    loaded = []
    for f in os.listdir(plugins_dir):
        if f.lower().endswith(".py") and not f.startswith("_"):
            mod_name = os.path.splitext(f)[0]
            try:
                mod = __import__(mod_name)
                loaded.append(mod)
            except Exception as e:
                print("Error loading plugin", mod_name, e)
    return loaded


# ---------------- GUI ----------------

PROTECTED_PROCESSES = {
    "wininit.exe",
    "csrss.exe",
    "lsass.exe",
    "services.exe",
    "smss.exe",
    "winlogon.exe",
}


class SystemConsole(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Control Console (PyQt5, Admin)")
        self.resize(1500, 850)

        self.log_lock = threading.Lock()

        self._apply_dark_theme()
        self.plugins = load_plugins()

        self._build_toolbar()

        self.tabs = QTabWidget()
        self.process_tab = self._build_process_tab()
        self.services_tab = self._build_services_tab()
        self.apps_tab = self._build_apps_tab()
        self.startup_tab = self._build_startup_tab()
        self.sysinfo_tab = self._build_sysinfo_tab()
        self.log_tab = self._build_log_tab()
        self.all_apps_tab = self._build_all_apps_tab()

        self.tabs.addTab(self.process_tab, "Processes")
        self.tabs.addTab(self.services_tab, "Services")
        self.tabs.addTab(self.apps_tab, "Installed Apps")
        self.tabs.addTab(self.startup_tab, "Startup Items")
        self.tabs.addTab(self.sysinfo_tab, "System Info")
        self.tabs.addTab(self.all_apps_tab, "All Windows Apps")
        self.tabs.addTab(self.log_tab, "Action Log")

        self.setCentralWidget(self.tabs)

        self.populate_processes()
        self.populate_services()
        self.populate_apps()
        self.populate_startup()
        self.populate_all_apps()

        self._start_sysinfo_timer()

        self.log("System Console started (admin: %s)" % is_admin())

    # ---------- Theme ----------

    def _apply_dark_theme(self):
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(45, 45, 45))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

    # ---------- Toolbar ----------

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        admin_label = QLabel(" Admin Mode: ON " if is_admin() else " Admin Mode: OFF ")
        admin_label.setStyleSheet(
            "QLabel { color: white; background-color: %s; padding: 4px; }"
            % ("#2e7d32" if is_admin() else "#c62828")
        )
        tb.addWidget(admin_label)

        tb.addSeparator()

        act_refresh_all = QAction("Refresh All", self)
        act_refresh_all.triggered.connect(self.refresh_all)
        tb.addAction(act_refresh_all)

    def refresh_all(self):
        self.populate_processes()
        self.populate_services()
        self.populate_apps()
        self.populate_startup()
        self.populate_all_apps()
        self.log("Refreshed all panels")

    # ---------- Processes tab ----------

    def _build_process_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(6)
        self.proc_table.setHorizontalHeaderLabels(
            ["Name", "PID", "CPU %", "RAM MB", "User", "Actions"]
        )
        self.proc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.proc_table.setSortingEnabled(True)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Processes")
        btn_refresh.clicked.connect(self.populate_processes)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addWidget(self.proc_table)
        return w

    def populate_processes(self):
        self.proc_table.setSortingEnabled(False)
        self.proc_table.setRowCount(0)
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "username"]):
            try:
                info = p.info
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self.proc_table.setRowCount(len(procs))

        for row, info in enumerate(procs):
            name = info.get("name") or ""
            pid = info["pid"]
            cpu = info.get("cpu_percent") or 0
            mem = info.get("memory_info")
            mem_mb = (mem.rss // (1024 * 1024)) if mem else 0
            user = info.get("username") or ""

            self.proc_table.setItem(row, 0, QTableWidgetItem(name))
            self.proc_table.setItem(row, 1, QTableWidgetItem(str(pid)))
            self.proc_table.setItem(row, 2, QTableWidgetItem(str(cpu)))
            self.proc_table.setItem(row, 3, QTableWidgetItem(str(mem_mb)))
            self.proc_table.setItem(row, 4, QTableWidgetItem(user))

            btn_kill = QPushButton("Kill")
            btn_kill.clicked.connect(lambda _, p_name=name, p_pid=pid: self.kill_process(p_name, p_pid))

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(btn_kill)
            h.addStretch()
            self.proc_table.setCellWidget(row, 5, container)

            if name.lower() in PROTECTED_PROCESSES:
                for col in range(5):
                    item = self.proc_table.item(row, col)
                    if item:
                        item.setBackground(QColor(80, 80, 0))

        self.proc_table.setSortingEnabled(True)

    def kill_process(self, name: str, pid: int):
        if name.lower() in PROTECTED_PROCESSES:
            if not self._confirm(
                f"{name} is marked as a protected/system process.\n"
                f"Killing it may crash or destabilize Windows.\n\n"
                f"PID: {pid}\n\nProceed anyway?"
            ):
                return

        if not self._confirm(f"Kill process {name} (PID {pid})?"):
            return

        try:
            p = psutil.Process(pid)
            p.terminate()
            self.log(f"Killed process {name} (PID {pid})")
        except Exception as e:
            self._error(f"Error killing process {name} (PID {pid}): {e}")
            self.log(f"Error killing process {name} (PID {pid}): {e}")
        self.populate_processes()

    # ---------- Services tab ----------

    def _build_services_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.svc_table = QTableWidget()
        self.svc_table.setColumnCount(4)
        self.svc_table.setHorizontalHeaderLabels(
            ["Service Name", "Display Name", "State", "Actions"]
        )
        self.svc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.svc_table.setSortingEnabled(True)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Services")
        btn_refresh.clicked.connect(self.populate_services)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addWidget(self.svc_table)
        return w

    def populate_services(self):
        self.svc_table.setSortingEnabled(False)
        self.svc_table.setRowCount(0)
        try:
            services = list_services()
        except Exception as e:
            self._error(f"Error listing services: {e}")
            self.log(f"Error listing services: {e}")
            return

        self.svc_table.setRowCount(len(services))

        for row, svc in enumerate(services):
            name = svc["name"]
            display = svc["display_name"]
            state_text = service_state_to_text(svc["state"])

            self.svc_table.setItem(row, 0, QTableWidgetItem(name))
            self.svc_table.setItem(row, 1, QTableWidgetItem(display))
            self.svc_table.setItem(row, 2, QTableWidgetItem(state_text))

            btn_start = QPushButton("Start")
            btn_start.clicked.connect(lambda _, s_name=name: self.start_service_ui(s_name))
            btn_stop = QPushButton("Stop")
            btn_stop.clicked.connect(lambda _, s_name=name: self.stop_service_ui(s_name))

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(btn_start)
            h.addWidget(btn_stop)
            h.addStretch()
            self.svc_table.setCellWidget(row, 3, container)

        self.svc_table.setSortingEnabled(True)

    def start_service_ui(self, name: str):
        if not self._confirm(f"Start service {name}?"):
            return
        try:
            start_service(name)
            self.log(f"Started service {name}")
        except Exception as e:
            self._error(f"Error starting service {name}: {e}")
            self.log(f"Error starting service {name}: {e}")
        self.populate_services()

    def stop_service_ui(self, name: str):
        if not self._confirm(f"Stop service {name}?"):
            return
        try:
            stop_service(name)
            self.log(f"Stopped service {name}")
        except Exception as e:
            self._error(f"Error stopping service {name}: {e}")
            self.log(f"Error stopping service {name}: {e}")
        self.populate_services()

    # ---------- Installed apps tab (Win32) ----------

    def _build_apps_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.apps_table = QTableWidget()
        self.apps_table.setColumnCount(3)
        self.apps_table.setHorizontalHeaderLabels(
            ["Name", "Uninstall Command", "Actions"]
        )
        self.apps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.apps_table.setSortingEnabled(True)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Installed Apps")
        btn_refresh.clicked.connect(self.populate_apps)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addWidget(self.apps_table)
        return w

    def populate_apps(self):
        self.apps_table.setSortingEnabled(False)
        self.apps_table.setRowCount(0)
        try:
            apps = list_installed_apps()
        except Exception as e:
            self._error(f"Error listing installed apps: {e}")
            self.log(f"Error listing installed apps: {e}")
            return

        self.apps_table.setRowCount(len(apps))

        for row, app in enumerate(apps):
            name = app["name"]
            uninstall = app["uninstall"]

            self.apps_table.setItem(row, 0, QTableWidgetItem(name))
            self.apps_table.setItem(row, 1, QTableWidgetItem(uninstall))

            btn_uninstall = QPushButton("Uninstall")
            btn_uninstall.clicked.connect(
                lambda _, a_name=name, cmd=uninstall: self.uninstall_app_ui(a_name, cmd)
            )

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(btn_uninstall)
            h.addStretch()
            self.apps_table.setCellWidget(row, 2, container)

        self.apps_table.setSortingEnabled(True)

    def uninstall_app_ui(self, name: str, cmd: str):
        if not self._confirm(
            f"Uninstall application:\n\n{name}\n\nCommand:\n{cmd}\n\nProceed?"
        ):
            return
        try:
            uninstall_app(cmd)
            self.log(f"Triggered uninstall for {name}")
        except Exception as e:
            self._error(f"Error uninstalling {name}: {e}")
            self.log(f"Error uninstalling {name}: {e}")

    # ---------- Startup items tab ----------

    def _build_startup_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.startup_table = QTableWidget()
        self.startup_table.setColumnCount(3)
        self.startup_table.setHorizontalHeaderLabels(
            ["Name", "Status", "Actions"]
        )
        self.startup_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.startup_table.setSortingEnabled(True)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Startup Items")
        btn_refresh.clicked.connect(self.populate_startup)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addWidget(self.startup_table)
        return w

    def populate_startup(self):
        self.startup_table.setSortingEnabled(False)
        self.startup_table.setRowCount(0)
        try:
            items = list_startup_items()
        except Exception as e:
            self._error(f"Error listing startup items: {e}")
            self.log(f"Error listing startup items: {e}")
            return

        self._startup_items = items
        self.startup_table.setRowCount(len(items))

        for row, item in enumerate(items):
            name = item["name"]
            status = "Enabled" if item["enabled"] else "Disabled"

            self.startup_table.setItem(row, 0, QTableWidgetItem(name))
            self.startup_table.setItem(row, 1, QTableWidgetItem(status))

            btn_toggle = QPushButton("Disable" if item["enabled"] else "Enable")
            btn_toggle.clicked.connect(
                lambda _, idx=row: self.toggle_startup_ui(idx)
            )

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(btn_toggle)
            h.addStretch()
            self.startup_table.setCellWidget(row, 2, container)

        self.startup_table.setSortingEnabled(True)

    def toggle_startup_ui(self, index: int):
        if index < 0 or index >= len(self._startup_items):
            return
        item = self._startup_items[index]
        action = "Disable" if item["enabled"] else "Enable"
        if not self._confirm(f"{action} startup item {item['name']}?"):
            return
        try:
            toggle_startup_item(item)
            self.log(f"{action}d startup item {item['name']}")
        except Exception as e:
            self._error(f"Error toggling startup item {item['name']}: {e}")
            self.log(f"Error toggling startup item {item['name']}: {e}")
        self.populate_startup()

    # ---------- System info tab ----------

    def _build_sysinfo_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.cpu_label = QLabel("CPU Usage:")
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)

        self.ram_label = QLabel("RAM Usage:")
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)

        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_bar)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.ram_bar)
        layout.addStretch()

        return w

    def _start_sysinfo_timer(self):
        self.sysinfo_timer = QTimer(self)
        self.sysinfo_timer.timeout.connect(self.update_sysinfo)
        self.sysinfo_timer.start(1000)

    def update_sysinfo(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        self.cpu_bar.setValue(int(cpu))
        self.cpu_label.setText(f"CPU Usage: {cpu:.1f}%")

        self.ram_bar.setValue(int(ram))
        self.ram_label.setText(f"RAM Usage: {ram:.1f}%")

    # ---------- All Windows Apps tab (Win32 + UWP + Provisioned + Features) ----------

    def _build_all_apps_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.all_apps_table = QTableWidget()
        self.all_apps_table.setColumnCount(5)
        self.all_apps_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Status", "Source", "Actions"]
        )
        self.all_apps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.all_apps_table.setSortingEnabled(True)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh All Windows Apps")
        btn_refresh.clicked.connect(self.populate_all_apps)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        layout.addLayout(btn_row)
        layout.addWidget(self.all_apps_table)
        return w

    def build_unified_app_list(self):
        items = []

        # Win32
        try:
            win32_apps = list_installed_apps()
        except Exception as e:
            self.log(f"Error listing Win32 apps: {e}")
            win32_apps = []
        for app in win32_apps:
            items.append({
                "name": app["name"],
                "type": "Win32",
                "source": "Registry",
                "status": "Installed",
                "raw": app,
            })

        # UWP
        uwp = list_uwp_apps()
        for a in uwp:
            items.append({
                "name": a.get("Name", ""),
                "type": "UWP",
                "source": "Appx",
                "status": "Installed",
                "raw": a,
            })

        # Provisioned
        prov = list_provisioned_apps()
        for p in prov:
            items.append({
                "name": p.get("DisplayName", ""),
                "type": "Provisioned",
                "source": "ProvisionedPackage",
                "status": "Provisioned",
                "raw": p,
            })

        # Optional features
        feats = list_optional_features()
        for f in feats:
            items.append({
                "name": f.get("FeatureName", ""),
                "type": "Feature",
                "source": "OptionalFeature",
                "status": f.get("State", ""),
                "raw": f,
            })

        return items

    def populate_all_apps(self):
        self.all_apps_table.setSortingEnabled(False)
        self.all_apps_table.setRowCount(0)

        items = self.build_unified_app_list()
        self._all_apps_items = items
        self.all_apps_table.setRowCount(len(items))

        for row, item in enumerate(items):
            name = item["name"] or ""
            app_type = item["type"]
            status = item["status"]
            source = item["source"]

            name_item = QTableWidgetItem(name)
            type_item = QTableWidgetItem(app_type)
            status_item = QTableWidgetItem(status)
            source_item = QTableWidgetItem(source)

            self.all_apps_table.setItem(row, 0, name_item)
            self.all_apps_table.setItem(row, 1, type_item)
            self.all_apps_table.setItem(row, 2, status_item)
            self.all_apps_table.setItem(row, 3, source_item)

            btn = None
            if app_type == "UWP":
                btn = QPushButton("Uninstall UWP")
                btn.clicked.connect(lambda _, idx=row: self.uninstall_uwp_ui(idx))
            else:
                btn = QPushButton("N/A")
                btn.setEnabled(False)

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(btn)
            h.addStretch()
            self.all_apps_table.setCellWidget(row, 4, container)

            lower_name = name.lower()
            if "xbox" in lower_name or "copilot" in lower_name:
                for col in range(4):
                    cell = self.all_apps_table.item(row, col)
                    if cell:
                        cell.setBackground(QColor(60, 80, 0))

        self.all_apps_table.setSortingEnabled(True)

    def uninstall_uwp_ui(self, index: int):
        if index < 0 or index >= len(self._all_apps_items):
            return
        item = self._all_apps_items[index]
        raw = item["raw"]
        pkg_full = raw.get("PackageFullName")
        name = item["name"]
        if not pkg_full:
            self._error(f"No PackageFullName for UWP app {name}")
            return

        if not self._confirm(
            f"Uninstall UWP app:\n\n{name}\n\nPackage:\n{pkg_full}\n\nProceed?"
        ):
            return

        try:
            remove_uwp_package(pkg_full)
            self.log(f"Triggered UWP uninstall for {name} ({pkg_full})")
        except Exception as e:
            self._error(f"Error uninstalling UWP {name}: {e}")
            self.log(f"Error uninstalling UWP {name}: {e}")

    # ---------- Log tab ----------

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        layout.addWidget(self.log_view)
        return w

    def log(self, message: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        with self.log_lock:
            self.log_view.append(line)

    # ---------- Helpers ----------

    def _confirm(self, text: str) -> bool:
        reply = QMessageBox.question(
            self,
            "Confirm",
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _error(self, text: str):
        QMessageBox.critical(self, "Error", text)


def main():
    if not is_admin():
        relaunch_as_admin()

    app = QApplication(sys.argv)
    win = SystemConsole()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

