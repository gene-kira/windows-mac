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
    QAction, QTextEdit, QProgressBar, QStyleFactory, QDialog,
    QLineEdit, QComboBox, QFormLayout, QDialogButtonBox
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
    for root, root_name in (
        (winreg.HKEY_LOCAL_MACHINE, "HKEY_LOCAL_MACHINE"),
        (winreg.HKEY_CURRENT_USER, "HKEY_CURRENT_USER"),
    ):
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
                    reg_subkey = path + "\\" + sub_name
                    reg_path = root_name + "\\" + reg_subkey
                    apps.append({
                        "name": name,
                        "uninstall": uninstall,
                        "reg_root": root,
                        "reg_root_name": root_name,
                        "reg_subkey": reg_subkey,
                        "reg_path": reg_path,
                    })
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                except Exception:
                    continue
    unique = {}
    for app in apps:
        key = (app["name"], app["uninstall"], app["reg_path"])
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
    return _run_powershell_json("Get-AppxPackage | Select Name, PackageFullName")


def list_provisioned_apps():
    return _run_powershell_json("Get-AppxProvisionedPackage -Online | Select DisplayName, PackageName")


def list_optional_features():
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


# ---------------- Registry helpers ----------------

REG_TYPE_MAP = {
    winreg.REG_SZ: "REG_SZ",
    winreg.REG_EXPAND_SZ: "REG_EXPAND_SZ",
    winreg.REG_BINARY: "REG_BINARY",
    winreg.REG_DWORD: "REG_DWORD",
    winreg.REG_DWORD_LITTLE_ENDIAN: "REG_DWORD",
    winreg.REG_DWORD_BIG_ENDIAN: "REG_DWORD_BIG_ENDIAN",
    winreg.REG_LINK: "REG_LINK",
    winreg.REG_MULTI_SZ: "REG_MULTI_SZ",
    winreg.REG_QWORD: "REG_QWORD",
    winreg.REG_QWORD_LITTLE_ENDIAN: "REG_QWORD",
    winreg.REG_NONE: "REG_NONE",
    winreg.REG_RESOURCE_LIST: "REG_RESOURCE_LIST",
    winreg.REG_FULL_RESOURCE_DESCRIPTOR: "REG_FULL_RESOURCE_DESCRIPTOR",
    winreg.REG_RESOURCE_REQUIREMENTS_LIST: "REG_RESOURCE_REQUIREMENTS_LIST",
}

REG_TYPE_NAME_TO_CONST = {v: k for k, v in REG_TYPE_MAP.items()}


def open_reg_key(root, subkey, access=winreg.KEY_READ | winreg.KEY_WRITE):
    return winreg.OpenKey(root, subkey, 0, access)


def enum_reg_values(root, subkey):
    values = []
    try:
        key = open_reg_key(root, subkey, winreg.KEY_READ)
    except FileNotFoundError:
        return values
    try:
        index = 0
        while True:
            try:
                name, data, vtype = winreg.EnumValue(key, index)
                values.append((name, data, vtype))
                index += 1
            except OSError:
                break
    finally:
        winreg.CloseKey(key)
    return values


def set_reg_value(root, subkey, name, vtype, data):
    key = open_reg_key(root, subkey, winreg.KEY_WRITE)
    try:
        winreg.SetValueEx(key, name, 0, vtype, data)
    finally:
        winreg.CloseKey(key)


def delete_reg_value(root, subkey, name):
    key = open_reg_key(root, subkey, winreg.KEY_WRITE)
    try:
        winreg.DeleteValue(key, name)
    finally:
        winreg.CloseKey(key)


def delete_reg_key(root, subkey):
    # Delete subkey and all children
    def _delete_tree(r, sk):
        try:
            key = open_reg_key(r, sk, winreg.KEY_READ | winreg.KEY_WRITE)
        except FileNotFoundError:
            return
        while True:
            try:
                child = winreg.EnumKey(key, 0)
                _delete_tree(r, sk + "\\" + child)
            except OSError:
                break
        winreg.CloseKey(key)
        winreg.DeleteKey(r, sk)
    _delete_tree(root, subkey)


def open_regedit_at(path_str):
    subprocess.Popen(["regedit.exe"])
    # No direct jump, but path is visible in UI/log.


# ---------------- Registry editor dialog ----------------

class RegValueEditorDialog(QDialog):
    def __init__(self, parent, name="", vtype="REG_SZ", data=""):
        super().__init__(parent)
        self.setWindowTitle("Edit Registry Value")
        self.resize(500, 250)

        self.name_edit = QLineEdit(name)
        self.type_combo = QComboBox()
        self.type_combo.addItems(sorted(REG_TYPE_NAME_TO_CONST.keys()))
        if vtype in REG_TYPE_NAME_TO_CONST:
            self.type_combo.setCurrentText(vtype)

        self.data_edit = QLineEdit()
        self.data_edit.setText(self._data_to_text(vtype, data))

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Type:", self.type_combo)
        form.addRow("Data:", self.data_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def _data_to_text(self, vtype, data):
        if vtype in ("REG_SZ", "REG_EXPAND_SZ"):
            return str(data)
        if vtype == "REG_MULTI_SZ":
            return "\n".join(data) if isinstance(data, list) else str(data)
        if vtype in ("REG_DWORD", "REG_QWORD"):
            try:
                return str(int(data))
            except Exception:
                return str(data)
        if vtype.startswith("REG_"):
            if isinstance(data, (bytes, bytearray)):
                return data.hex(" ")
            return str(data)
        return str(data)

    def _text_to_data(self, vtype, text):
        if vtype in ("REG_SZ", "REG_EXPAND_SZ"):
            return text
        if vtype == "REG_MULTI_SZ":
            return [line for line in text.splitlines()]
        if vtype == "REG_DWORD":
            return int(text, 0)
        if vtype == "REG_QWORD":
            return int(text, 0)
        text = text.strip()
        if not text:
            return b""
        parts = text.replace(",", " ").split()
        return bytes(int(p, 16) for p in parts)

    def get_value(self):
        name = self.name_edit.text()
        vtype_name = self.type_combo.currentText()
        data_text = self.data_edit.text()
        data = self._text_to_data(vtype_name, data_text)
        vtype = REG_TYPE_NAME_TO_CONST.get(vtype_name, winreg.REG_SZ)
        return name, vtype, data


class RegistryEditorWindow(QMainWindow):
    def __init__(self, parent, root, root_name, subkey, log_func):
        super().__init__(parent)
        self.root = root
        self.root_name = root_name
        self.subkey = subkey
        self.log = log_func

        self.setWindowTitle(f"Registry Editor - {root_name}\\{subkey}")
        self.resize(800, 400)

        central = QWidget()
        layout = QVBoxLayout(central)

        self.path_label = QLabel(f"{root_name}\\{subkey}")
        layout.addWidget(self.path_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Data"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Value")
        btn_edit = QPushButton("Edit Value")
        btn_delete = QPushButton("Delete Value")
        btn_del_key = QPushButton("Delete Key")
        btn_regedit = QPushButton("Open in RegEdit")

        btn_add.clicked.connect(self.add_value)
        btn_edit.clicked.connect(self.edit_value)
        btn_delete.clicked.connect(self.delete_value)
        btn_del_key.clicked.connect(self.delete_key)
        btn_regedit.clicked.connect(self.open_in_regedit)

        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_edit)
        btn_row.addWidget(btn_delete)
        btn_row.addWidget(btn_del_key)
        btn_row.addWidget(btn_regedit)
        btn_row.addStretch()

        layout.addLayout(btn_row)

        self.setCentralWidget(central)

        self.populate_values()

    def populate_values(self):
        values = enum_reg_values(self.root, self.subkey)
        self.table.setRowCount(len(values))
        for row, (name, data, vtype) in enumerate(values):
            vtype_name = REG_TYPE_MAP.get(vtype, f"{vtype}")
            data_text = self._data_preview(vtype_name, data)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(vtype_name))
            self.table.setItem(row, 2, QTableWidgetItem(data_text))

    def _data_preview(self, vtype_name, data):
        if vtype_name in ("REG_SZ", "REG_EXPAND_SZ"):
            return str(data)
        if vtype_name == "REG_MULTI_SZ":
            return "\\n".join(data) if isinstance(data, list) else str(data)
        if vtype_name in ("REG_DWORD", "REG_QWORD"):
            try:
                return str(int(data))
            except Exception:
                return str(data)
        if isinstance(data, (bytes, bytearray)):
            hex_str = data.hex(" ")
            if len(hex_str) > 80:
                return hex_str[:80] + " ..."
            return hex_str
        return str(data)

    def _get_selected_row(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return rows[0].row()

    def add_value(self):
        dlg = RegValueEditorDialog(self, name="", vtype="REG_SZ", data="")
        if dlg.exec_() == QDialog.Accepted:
            name, vtype, data = dlg.get_value()
            try:
                set_reg_value(self.root, self.subkey, name, vtype, data)
                self.log(f"Added registry value {name} in {self.root_name}\\{self.subkey}")
                self.populate_values()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error adding value: {e}")

    def edit_value(self):
        row = self._get_selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Select a value to edit.")
            return
        name_item = self.table.item(row, 0)
        type_item = self.table.item(row, 1)
        if not name_item:
            return
        name = name_item.text()
        vtype_name = type_item.text()
        values = enum_reg_values(self.root, self.subkey)
        data = None
        vtype = None
        for n, d, t in values:
            if n == name:
                data = d
                vtype = t
                break
        if data is None:
            QMessageBox.warning(self, "Missing", "Value no longer exists.")
            self.populate_values()
            return
        dlg = RegValueEditorDialog(self, name=name, vtype=vtype_name, data=data)
        if dlg.exec_() == QDialog.Accepted:
            new_name, new_vtype, new_data = dlg.get_value()
            try:
                if new_name != name:
                    delete_reg_value(self.root, self.subkey, name)
                set_reg_value(self.root, self.subkey, new_name, new_vtype, new_data)
                self.log(f"Edited registry value {name} -> {new_name} in {self.root_name}\\{self.subkey}")
                self.populate_values()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error editing value: {e}")

    def delete_value(self):
        row = self._get_selected_row()
        if row is None:
            QMessageBox.information(self, "No selection", "Select a value to delete.")
            return
        name_item = self.table.item(row, 0)
        if not name_item:
            return
        name = name_item.text()
        if QMessageBox.question(
            self,
            "Confirm",
            f"Delete value '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            delete_reg_value(self.root, self.subkey, name)
            self.log(f"Deleted registry value {name} in {self.root_name}\\{self.subkey}")
            self.populate_values()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error deleting value: {e}")

    def delete_key(self):
        if QMessageBox.question(
            self,
            "Confirm",
            f"Delete entire key:\n\n{self.root_name}\\{self.subkey}\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        try:
            delete_reg_key(self.root, self.subkey)
            self.log(f"Deleted registry key {self.root_name}\\{self.subkey}")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error deleting key: {e}")

    def open_in_regedit(self):
        open_regedit_at(f"{self.root_name}\\{self.subkey}")


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
        self.apps_table.setColumnCount(4)
        self.apps_table.setHorizontalHeaderLabels(
            ["Name", "Uninstall Command", "Registry Path", "Actions"]
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

        self._apps_items = apps
        self.apps_table.setRowCount(len(apps))

        for row, app in enumerate(apps):
            name = app["name"]
            uninstall = app["uninstall"]
            reg_path = app["reg_path"]

            self.apps_table.setItem(row, 0, QTableWidgetItem(name))
            self.apps_table.setItem(row, 1, QTableWidgetItem(uninstall))
            self.apps_table.setItem(row, 2, QTableWidgetItem(reg_path))

            btn_uninstall = QPushButton("Uninstall")
            btn_uninstall.clicked.connect(
                lambda _, idx=row: self.uninstall_app_ui_idx(idx)
            )

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(btn_uninstall)
            h.addStretch()
            self.apps_table.setCellWidget(row, 3, container)

        self.apps_table.setSortingEnabled(True)

    def uninstall_app_ui_idx(self, index: int):
        app = self._apps_items[index]
        name = app["name"]
        cmd = app["uninstall"]
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

    # ---------- All Windows Apps tab (with search + filter) ----------

    def _build_all_apps_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # --- Search + Filter Row ---
        top_row = QHBoxLayout()

        self.all_apps_search = QLineEdit()
        self.all_apps_search.setPlaceholderText("Search apps...")
        self.all_apps_search.textChanged.connect(self.filter_all_apps)

        self.all_apps_filter = QComboBox()
        self.all_apps_filter.addItems(["All", "Win32", "UWP", "Provisioned", "Feature"])
        self.all_apps_filter.currentIndexChanged.connect(self.filter_all_apps)

        top_row.addWidget(QLabel("Search:"))
        top_row.addWidget(self.all_apps_search)
        top_row.addWidget(QLabel("Filter:"))
        top_row.addWidget(self.all_apps_filter)
        top_row.addStretch()

        layout.addLayout(top_row)

        # --- Table ---
        self.all_apps_table = QTableWidget()
        self.all_apps_table.setColumnCount(6)
        self.all_apps_table.setHorizontalHeaderLabels(
            ["Name", "Type", "Status", "Source", "Registry Path", "Actions"]
        )
        self.all_apps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.all_apps_table.setSortingEnabled(True)

        layout.addWidget(self.all_apps_table)

        # --- Refresh Button ---
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh All Windows Apps")
        btn_refresh.clicked.connect(self.populate_all_apps)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()

        layout.addLayout(btn_row)

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
                "reg_root": app["reg_root"],
                "reg_root_name": app["reg_root_name"],
                "reg_subkey": app["reg_subkey"],
                "reg_path": app["reg_path"],
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
                "reg_root": None,
                "reg_root_name": "",
                "reg_subkey": "",
                "reg_path": "",
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
                "reg_root": None,
                "reg_root_name": "",
                "reg_subkey": "",
                "reg_path": "",
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
                "reg_root": None,
                "reg_root_name": "",
                "reg_subkey": "",
                "reg_path": "",
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
            reg_path = item.get("reg_path", "")

            name_item = QTableWidgetItem(name)
            type_item = QTableWidgetItem(app_type)
            status_item = QTableWidgetItem(status)
            source_item = QTableWidgetItem(source)
            reg_item = QTableWidgetItem(reg_path)

            self.all_apps_table.setItem(row, 0, name_item)
            self.all_apps_table.setItem(row, 1, type_item)
            self.all_apps_table.setItem(row, 2, status_item)
            self.all_apps_table.setItem(row, 3, source_item)
            self.all_apps_table.setItem(row, 4, reg_item)

            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)

            if app_type == "Win32" and reg_path:
                btn_regedit = QPushButton("Open RegEdit")
                btn_edit = QPushButton("Edit Key")
                btn_del = QPushButton("Delete Key")
                btn_regedit.clicked.connect(lambda _, idx=row: self.open_regedit_ui(idx))
                btn_edit.clicked.connect(lambda _, idx=row: self.edit_regkey_ui(idx))
                btn_del.clicked.connect(lambda _, idx=row: self.delete_regkey_ui(idx))
                h.addWidget(btn_regedit)
                h.addWidget(btn_edit)
                h.addWidget(btn_del)
            elif app_type == "UWP":
                btn_uninstall = QPushButton("Uninstall UWP")
                btn_uninstall.clicked.connect(lambda _, idx=row: self.uninstall_uwp_ui(idx))
                h.addWidget(btn_uninstall)
            else:
                btn_na = QPushButton("N/A")
                btn_na.setEnabled(False)
                h.addWidget(btn_na)

            h.addStretch()
            self.all_apps_table.setCellWidget(row, 5, container)

            lower_name = name.lower()
            if "xbox" in lower_name or "copilot" in lower_name:
                for col in range(5):
                    cell = self.all_apps_table.item(row, col)
                    if cell:
                        cell.setBackground(QColor(60, 80, 0))

        self.all_apps_table.setSortingEnabled(True)

        # Apply current search/filter after refresh (Patch 3)
        self.filter_all_apps()

    def filter_all_apps(self):
        search = self.all_apps_search.text().lower()
        filter_type = self.all_apps_filter.currentText()

        for row in range(self.all_apps_table.rowCount()):
            name_item = self.all_apps_table.item(row, 0)
            type_item = self.all_apps_table.item(row, 1)
            if not name_item or not type_item:
                self.all_apps_table.setRowHidden(row, True)
                continue

            name = name_item.text().lower()
            app_type = type_item.text()

            type_match = (filter_type == "All") or (app_type == filter_type)
            search_match = (search in name)

            self.all_apps_table.setRowHidden(row, not (type_match and search_match))

    def open_regedit_ui(self, index: int):
        item = self._all_apps_items[index]
        reg_path = item.get("reg_path", "")
        if not reg_path:
            return
        open_regedit_at(reg_path)
        self.log(f"Opened RegEdit for {reg_path}")

    def edit_regkey_ui(self, index: int):
        item = self._all_apps_items[index]
        reg_root = item.get("reg_root")
        reg_root_name = item.get("reg_root_name")
        reg_subkey = item.get("reg_subkey")
        if not reg_root or not reg_subkey:
            QMessageBox.warning(self, "No key", "No registry key info for this item.")
            return
        editor = RegistryEditorWindow(self, reg_root, reg_root_name, reg_subkey, self.log)
        editor.show()

    def delete_regkey_ui(self, index: int):
        item = self._all_apps_items[index]
        reg_root = item.get("reg_root")
        reg_root_name = item.get("reg_root_name")
        reg_subkey = item.get("reg_subkey")
        if not reg_root or not reg_subkey:
            QMessageBox.warning(self, "No key", "No registry key info for this item.")
            return
        if not self._confirm(
            f"Delete uninstall registry key:\n\n{reg_root_name}\\{reg_subkey}\n\nThis will remove it from Installed Apps lists."
        ):
            return
        try:
            delete_reg_key(reg_root, reg_subkey)
            self.log(f"Deleted uninstall key {reg_root_name}\\{reg_subkey}")
            self.populate_all_apps()
            self.populate_apps()
        except Exception as e:
            self._error(f"Error deleting key: {e}")
            self.log(f"Error deleting key {reg_root_name}\\{reg_subkey}: {e}")

    def uninstall_uwp_ui(self, index: int):
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

