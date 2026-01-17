import sys
import os
import ctypes
import subprocess
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget, QMessageBox,
    QTableWidget, QTableWidgetItem, QPushButton, QLineEdit, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QLabel, QHeaderView, QInputDialog,
    QSplitter, QGraphicsDropShadowEffect, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

import psutil
import winreg

OVERRIDES_FILE = "unknown_systems_overrides.json"
DEV_MODE_FILE = "dev_mode.json"

# Global health state: "OK", "DEGRADED", "FAILED"
HEALTH_STATE = "OK"

# ============================================================
# Elevation helpers
# ============================================================

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

def relaunch_as_admin():
    ShellExecuteW = ctypes.windll.shell32.ShellExecuteW
    params = " ".join(f'"{arg}"' for arg in sys.argv[1:])
    ret = ShellExecuteW(
        None,
        "runas",
        sys.executable,
        f'"{os.path.abspath(sys.argv[0])}" {params}',
        None,
        1
    )
    if int(ret) <= 32:
        raise RuntimeError(f"Elevation failed, ShellExecuteW returned {ret}")

# ============================================================
# Shell / PowerShell helpers
# ============================================================

def run_ps(cmd):
    result = subprocess.run(
        ["powershell", "-NoLogo", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True
    )
    return result.stdout.strip(), result.stderr.strip()

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout.strip(), result.stderr.strip()

# ============================================================
# Dev mode manager (global shadow/hybrid state)
# ============================================================

class DevModeManager:
    def __init__(self):
        self.enabled = False
        # REAL = real only, SHADOW = shadow only, AUTO = hybrid (real + shadow)
        self.view_mode = "REAL"
        self.load()

    def load(self):
        if not os.path.exists(DEV_MODE_FILE):
            return
        try:
            with open(DEV_MODE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.enabled = bool(data.get("enabled", False))
            self.view_mode = data.get("view_mode", "REAL")
        except Exception:
            pass

    def save(self):
        data = {
            "enabled": self.enabled,
            "view_mode": self.view_mode
        }
        try:
            with open(DEV_MODE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

DEV_MODE = DevModeManager()

# ============================================================
# Windows Apps helpers
# ============================================================

def list_uwp_apps():
    cmd = "Get-AppxPackage | Select Name, PackageFullName | ConvertTo-Json"
    out, err = run_ps(cmd)
    if not out:
        return []
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return [data]
        return data
    except Exception:
        return []

def list_provisioned_apps():
    cmd = "Get-AppxProvisionedPackage -Online | Select DisplayName, PackageName | ConvertTo-Json"
    out, err = run_ps(cmd)
    if not out:
        return []
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return [data]
        return data
    except Exception:
        return []

def remove_uwp(package_full_name):
    cmd = f"Remove-AppxPackage -Package '{package_full_name}'"
    return run_ps(cmd)

def remove_uwp_all_users(package_full_name):
    cmd = f"Get-AppxPackage -AllUsers '{package_full_name}' | Remove-AppxPackage"
    return run_ps(cmd)

def remove_provisioned(package_name):
    cmd = f"Remove-AppxProvisionedPackage -Online -PackageName '{package_name}'"
    return run_ps(cmd)

def force_remove(package_name):
    cmd = (
        f"Get-AppxPackage -AllUsers '{package_name}' | "
        f"Remove-AppxPackage -AllUsers -ErrorAction SilentlyContinue; "
        f"Remove-AppxProvisionedPackage -Online -PackageName '{package_name}' "
        f"-ErrorAction SilentlyContinue"
    )
    return run_ps(cmd)

# ============================================================
# Processes panel
# ============================================================

class ProcessesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "Memory %"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)

        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("⟒⧫ᛟ — Refresh Processes", self)
        self.btn_kill = QPushButton("⧖ᛞ⟟ — Kill Selected", self)

        self.btn_refresh.clicked.connect(self.refresh_processes)
        self.btn_kill.clicked.connect(self.kill_selected)

        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_kill)
        btn_layout.addStretch()

        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

        self.refresh_processes()

    def refresh_processes(self):
        procs = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
        self.table.setRowCount(len(procs))
        for row, p in enumerate(procs):
            self.table.setItem(row, 0, QTableWidgetItem(str(p.info['pid'])))
            self.table.setItem(row, 1, QTableWidgetItem(str(p.info['name'])))
            self.table.setItem(row, 2, QTableWidgetItem(str(p.info['cpu_percent'])))
            mem = p.info['memory_percent'] or 0
            self.table.setItem(row, 3, QTableWidgetItem(str(round(mem, 2))))

    def kill_selected(self):
        if DEV_MODE.enabled:
            QMessageBox.information(self, "Dev Mode", "In Development/Hybrid Mode, process killing is disabled (shadow only).")
            return
        row = self.table.currentRow()
        if row < 0:
            return
        pid_item = self.table.item(row, 0)
        if not pid_item:
            return
        pid = int(pid_item.text())
        reply = QMessageBox.question(
            self,
            "Confirm Kill",
            f"Kill process PID {pid}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            p = psutil.Process(pid)
            p.terminate()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to kill process: {e}")
        self.refresh_processes()

# ============================================================
# Registry panel
# ============================================================

ROOT_KEYS = {
    "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
    "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
    "HKEY_CLASSES_ROOT": winreg.HKEY_CLASSES_ROOT,
    "HKEY_USERS": winreg.HKEY_USERS,
    "HKEY_CURRENT_CONFIG": winreg.HKEY_CURRENT_CONFIG,
}

class RegistryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Registry path (e.g. HKEY_LOCAL_MACHINE\\SOFTWARE)")
        self.btn_load = QPushButton("⧖⟟ᛉ — Go", self)
        self.btn_load.clicked.connect(self.load_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_load)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Key / Value", "Type", "Data"])
        self.tree.itemExpanded.connect(self.on_item_expanded)

        crud_layout = QHBoxLayout()
        self.btn_new_key = QPushButton("⟒ᛟ⧫ — New Key", self)
        self.btn_del_key = QPushButton("⧫ᛟ⟒ — Delete Key", self)
        self.btn_new_val = QPushButton("ᛉ⧗⟒ — New Value", self)
        self.btn_edit_val = QPushButton("⟟ᛞ⧗ — Edit Value", self)
        self.btn_del_val = QPushButton("⧖ᛟ⧫ — Delete Value", self)

        self.btn_new_key.clicked.connect(self.create_key)
        self.btn_del_key.clicked.connect(self.delete_key)
        self.btn_new_val.clicked.connect(self.create_value)
        self.btn_edit_val.clicked.connect(self.edit_value)
        self.btn_del_val.clicked.connect(self.delete_value)

        crud_layout.addWidget(self.btn_new_key)
        crud_layout.addWidget(self.btn_del_key)
        crud_layout.addWidget(self.btn_new_val)
        crud_layout.addWidget(self.btn_edit_val)
        crud_layout.addWidget(self.btn_del_val)
        crud_layout.addStretch()

        layout.addLayout(path_layout)
        layout.addWidget(self.tree)
        layout.addLayout(crud_layout)

        self._populate_roots()

    def _populate_roots(self):
        self.tree.clear()
        for name in ROOT_KEYS.keys():
            item = QTreeWidgetItem([name, "", ""])
            item.setData(0, Qt.UserRole, ("key", name))
            item.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
            child = QTreeWidgetItem(["<loading>", "", ""])
            item.addChild(child)
            self.tree.addTopLevelItem(item)

    def on_item_expanded(self, item):
        if item.childCount() == 1 and item.child(0).text(0) == "<loading>":
            item.takeChildren()
            self._load_subkeys(item)

    def _get_key_path_from_item(self, item):
        parts = []
        current = item
        while current is not None and current.parent() is not None:
            parts.insert(0, current.text(0))
            current = current.parent()
        root_name = current.text(0)
        sub_path = "\\".join(parts)
        return root_name, sub_path

    def _resolve_root_and_subpath(self, root_name, sub_path):
        root_key = ROOT_KEYS.get(root_name)
        if root_key is None:
            raise ValueError(f"Unknown root: {root_name}")
        return root_key, sub_path

    def _type_to_str(self, t):
        mapping = {
            winreg.REG_SZ: "REG_SZ",
            winreg.REG_DWORD: "REG_DWORD",
            winreg.REG_QWORD: "REG_QWORD",
            winreg.REG_BINARY: "REG_BINARY",
            winreg.REG_MULTI_SZ: "REG_MULTI_SZ",
            winreg.REG_EXPAND_SZ: "REG_EXPAND_SZ",
        }
        return mapping.get(t, str(t))

    def _str_to_type(self, s):
        s = s.upper()
        if s == "REG_SZ":
            return winreg.REG_SZ
        if s == "REG_DWORD":
            return winreg.REG_DWORD
        if s == "REG_QWORD":
            return winreg.REG_QWORD
        if s == "REG_BINARY":
            return winreg.REG_BINARY
        if s == "REG_MULTI_SZ":
            return winreg.REG_MULTI_SZ
        if s == "REG_EXPAND_SZ":
            return winreg.REG_EXPAND_SZ
        return winreg.REG_SZ

    def _load_subkeys(self, item):
        root_name, sub_path = self._get_key_path_from_item(item)
        try:
            root_key, base_path = self._resolve_root_and_subpath(root_name, sub_path)
            with winreg.OpenKey(root_key, base_path, 0, winreg.KEY_READ) as key:
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        child = QTreeWidgetItem([subkey_name, "", ""])
                        child.setData(0, Qt.UserRole, ("key", root_name))
                        child.setChildIndicatorPolicy(QTreeWidgetItem.ShowIndicator)
                        child.addChild(QTreeWidgetItem(["<loading>", "", ""]))
                        item.addChild(child)
                        i += 1
                    except OSError:
                        break
                j = 0
                while True:
                    try:
                        val_name, val_data, val_type = winreg.EnumValue(key, j)
                        type_str = self._type_to_str(val_type)
                        val_item = QTreeWidgetItem([val_name, type_str, str(val_data)])
                        val_item.setData(0, Qt.UserRole, ("value", root_name))
                        item.addChild(val_item)
                        j += 1
                    except OSError:
                        break
        except Exception as e:
            QMessageBox.warning(self, "Registry Error", str(e))

    def load_path(self):
        path = self.path_edit.text().strip()
        if not path:
            return
        root_name = None
        for r in ROOT_KEYS.keys():
            if path.upper().startswith(r):
                root_name = r
                break
        if not root_name:
            QMessageBox.warning(self, "Path Error", "Unknown root in path.")
            return
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.text(0).upper() == root_name:
                self.tree.expandItem(item)
                break

    def _get_selected_key_info(self):
        item = self.tree.currentItem()
        if not item:
            return None
        data = item.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data[0] == "value":
            item = item.parent()
        root_name, sub_path = self._get_key_path_from_item(item)
        return root_name, sub_path, item

    def _get_selected_value_info(self):
        item = self.tree.currentItem()
        if not item:
            return None
        data = item.data(0, Qt.UserRole)
        if not (isinstance(data, tuple) and data[0] == "value"):
            return None
        val_name = item.text(0)
        parent = item.parent()
        root_name, sub_path = self._get_key_path_from_item(parent)
        return root_name, sub_path, val_name, item

    def _dev_mode_block(self):
        if DEV_MODE.enabled:
            QMessageBox.information(self, "Dev Mode", "Registry writes are disabled in Development/Hybrid Mode (shadow only).")
            return True
        return False

    def create_key(self):
        if self._dev_mode_block():
            return
        info = self._get_selected_key_info()
        if not info:
            return
        root_name, sub_path, item = info
        name, ok = QInputDialog.getText(self, "New Key", "Key name:")
        if not ok or not name:
            return
        try:
            root_key, base_path = self._resolve_root_and_subpath(root_name, sub_path)
            new_path = base_path + ("\\" if base_path else "") + name
            winreg.CreateKeyEx(root_key, new_path, 0, winreg.KEY_WRITE)
            item.takeChildren()
            item.addChild(QTreeWidgetItem(["<loading>", "", ""]))
            self.tree.expandItem(item)
        except Exception as e:
            QMessageBox.warning(self, "Create Key Error", str(e))

    def delete_key(self):
        if self._dev_mode_block():
            return
        info = self._get_selected_key_info()
        if not info:
            return
        root_name, sub_path, item = info
        if sub_path == "":
            QMessageBox.warning(self, "Delete Key", "Refusing to delete root keys.")
            return
        reply = QMessageBox.question(
            self,
            "Confirm Delete Key",
            f"Delete key:\n{root_name}\\{sub_path} ?\n\nAll subkeys and values may be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            root_key, base_path = self._resolve_root_and_subpath(root_name, sub_path)
            try:
                winreg.DeleteKeyEx(root_key, base_path)
            except AttributeError:
                winreg.DeleteKey(root_key, base_path)
            parent = item.parent()
            parent.removeChild(item)
        except Exception as e:
            QMessageBox.warning(self, "Delete Key Error", str(e))

    def create_value(self):
        if self._dev_mode_block():
            return
        info = self._get_selected_key_info()
        if not info:
            return
        root_name, sub_path, item = info
        val_name, ok = QInputDialog.getText(self, "New Value", "Value name:")
        if not ok or not val_name:
            return
        type_str, ok = QInputDialog.getItem(
            self, "Value Type", "Type:", ["REG_SZ", "REG_DWORD"], 0, False
        )
        if not ok:
            return
        data_str, ok = QInputDialog.getText(self, "Value Data", "Data:")
        if not ok:
            return
        val_type = self._str_to_type(type_str)
        try:
            root_key, base_path = self._resolve_root_and_subpath(root_name, sub_path)
            with winreg.OpenKey(root_key, base_path, 0, winreg.KEY_WRITE) as k:
                if val_type == winreg.REG_DWORD:
                    data = int(data_str, 0)
                else:
                    data = data_str
                winreg.SetValueEx(k, val_name, 0, val_type, data)
            item.takeChildren()
            item.addChild(QTreeWidgetItem(["<loading>", "", ""]))
            self.tree.expandItem(item)
        except Exception as e:
            QMessageBox.warning(self, "Create Value Error", str(e))

    def edit_value(self):
        if self._dev_mode_block():
            return
        info = self._get_selected_value_info()
        if not info:
            return
        root_name, sub_path, val_name, item = info
        type_str = item.text(1)
        old_data = item.text(2)
        new_data, ok = QInputDialog.getText(self, "Edit Value", f"{val_name} ({type_str})", text=old_data)
        if not ok:
            return
        val_type = self._str_to_type(type_str)
        try:
            root_key, base_path = self._resolve_root_and_subpath(root_name, sub_path)
            with winreg.OpenKey(root_key, base_path, 0, winreg.KEY_WRITE) as k:
                if val_type == winreg.REG_DWORD:
                    data = int(new_data, 0)
                else:
                    data = new_data
                winreg.SetValueEx(k, val_name, 0, val_type, data)
            item.setText(2, new_data)
        except Exception as e:
            QMessageBox.warning(self, "Edit Value Error", str(e))

    def delete_value(self):
        if self._dev_mode_block():
            return
        info = self._get_selected_value_info()
        if not info:
            return
        root_name, sub_path, val_name, item = info
        reply = QMessageBox.question(
            self,
            "Confirm Delete Value",
            f"Delete value:\n{val_name} ?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            root_key, base_path = self._resolve_root_and_subpath(root_name, sub_path)
            with winreg.OpenKey(root_key, base_path, 0, winreg.KEY_WRITE) as k:
                winreg.DeleteValue(k, val_name)
            parent = item.parent()
            parent.removeChild(item)
        except Exception as e:
            QMessageBox.warning(self, "Delete Value Error", str(e))

# ============================================================
# PowerShell panel
# ============================================================

class PowerShellPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        cmd_layout = QHBoxLayout()
        self.cmd_edit = QLineEdit(self)
        self.cmd_edit.setPlaceholderText("Enter PowerShell command...")
        self.btn_run = QPushButton("⟟ᛞ⧗ — Run", self)
        self.btn_run.clicked.connect(self.run_command)
        cmd_layout.addWidget(self.cmd_edit)
        cmd_layout.addWidget(self.btn_run)

        self.output = QTextEdit(self)
        self.output.setReadOnly(True)

        layout.addLayout(cmd_layout)
        layout.addWidget(self.output)

    def run_command(self):
        if DEV_MODE.enabled:
            QMessageBox.information(self, "Dev Mode", "PowerShell execution is disabled in Development/Hybrid Mode (shadow only).")
            return
        cmd = self.cmd_edit.text().strip()
        if not cmd:
            return
        out, err = run_ps(cmd)
        text = ""
        if out:
            text += out
        if err:
            text += "\n[stderr]\n" + err
        self.output.setPlainText(text)

# ============================================================
# Windows Apps panel
# ============================================================

class AppsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = []
        self.current_rows = []
        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search by name...")
        self.search_edit.textChanged.connect(self.apply_filter)

        self.btn_refresh = QPushButton("⧖ᛟ⧫ — Refresh", self)
        self.btn_refresh.clicked.connect(self.refresh_lists)

        top_layout.addWidget(QLabel("Filter:"))
        top_layout.addWidget(self.search_edit)
        top_layout.addWidget(self.btn_refresh)

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Display Name", "Package Name", "Type", "Installed?", "Provisioned?"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        btn_layout = QHBoxLayout()
        self.btn_uninstall = QPushButton("⧖ᛞ⟟ — Uninstall (Current User)", self)
        self.btn_uninstall_all = QPushButton("ᛉ⧗⟒ — Uninstall (All Users)", self)
        self.btn_remove_prov = QPushButton("⟒ᛟ⧫ — Remove Provisioned", self)
        self.btn_force = QPushButton("⧫ᛟ⟒ — Force Remove (All)", self)

        self.btn_uninstall.clicked.connect(self.uninstall_current)
        self.btn_uninstall_all.clicked.connect(self.uninstall_all_users)
        self.btn_remove_prov.clicked.connect(self.remove_provisioned_click)
        self.btn_force.clicked.connect(self.force_remove_click)

        btn_layout.addWidget(self.btn_uninstall)
        btn_layout.addWidget(self.btn_uninstall_all)
        btn_layout.addWidget(self.btn_remove_prov)
        btn_layout.addWidget(self.btn_force)

        layout.addLayout(top_layout)
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

        self.refresh_lists()

    def refresh_lists(self):
        uwp_apps = list_uwp_apps()
        prov_apps = list_provisioned_apps()
        self.model = []

        for u in uwp_apps:
            name = u.get("Name") or ""
            pkg = u.get("PackageFullName") or ""
            prov = None
            for p in prov_apps:
                if name and p.get("DisplayName", "").lower() == name.lower():
                    prov = p
                    break
            entry = {
                "display": name,
                "package": pkg,
                "type": "UWP" if not prov else "Both",
                "installed": True,
                "provisioned": prov is not None,
                "prov_pkg": prov.get("PackageName") if prov else None
            }
            self.model.append(entry)

        for p in prov_apps:
            display = p.get("DisplayName") or ""
            pkg = p.get("PackageName") or ""
            if not any(e["prov_pkg"] == pkg or e["package"] == pkg for e in self.model):
                entry = {
                    "display": display,
                    "package": pkg,
                    "type": "Provisioned",
                    "installed": False,
                    "provisioned": True,
                    "prov_pkg": pkg
                }
                self.model.append(entry)

        self.apply_filter()

    def apply_filter(self):
        text = self.search_edit.text().strip().lower()
        rows = []
        for e in self.model:
            if text and text not in (e["display"] or "").lower() and text not in (e["package"] or "").lower():
                continue
            rows.append(e)

        self.table.setRowCount(len(rows))
        for i, e in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(e["display"]))
            self.table.setItem(i, 1, QTableWidgetItem(e["package"]))
            self.table.setItem(i, 2, QTableWidgetItem(e["type"]))
            self.table.setItem(i, 3, QTableWidgetItem("Yes" if e["installed"] else "No"))
            self.table.setItem(i, 4, QTableWidgetItem("Yes" if e["provisioned"] else "No"))

        self.current_rows = rows

    def _get_selected_entry(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.current_rows):
            return None
        return self.current_rows[row]

    def _show_result(self, out, err):
        msg = ""
        if out:
            msg += out
        if err:
            msg += "\n[stderr]\n" + err
        if not msg:
            msg = "Command executed."
        QMessageBox.information(self, "Result", msg)

    def _dev_mode_block(self):
        if DEV_MODE.enabled:
            QMessageBox.information(self, "Dev Mode", "AppX modifications are disabled in Development/Hybrid Mode (shadow only).")
            return True
        return False

    def uninstall_current(self):
        if self._dev_mode_block():
            return
        e = self._get_selected_entry()
        if not e or not e["installed"]:
            return
        pkg = e["package"]
        reply = QMessageBox.question(
            self,
            "Confirm Uninstall",
            f"Uninstall (current user):\n\n{e['display']}\n{pkg}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        out, err = remove_uwp(pkg)
        self._show_result(out, err)
        self.refresh_lists()

    def uninstall_all_users(self):
        if self._dev_mode_block():
            return
        e = self._get_selected_entry()
        if not e or not e["installed"]:
            return
        pkg = e["package"]
        reply = QMessageBox.question(
            self,
            "Confirm Uninstall (All Users)",
            f"Uninstall for ALL users:\n\n{e['display']}\n{pkg}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        out, err = remove_uwp_all_users(pkg)
        self._show_result(out, err)
        self.refresh_lists()

    def remove_provisioned_click(self):
        if self._dev_mode_block():
            return
        e = self._get_selected_entry()
        if not e or not e["provisioned"]:
            return
        pkg = e["prov_pkg"] or e["package"]
        reply = QMessageBox.question(
            self,
            "Confirm Remove Provisioned",
            f"Remove provisioned package:\n\n{e['display']}\n{pkg}?\n\n"
            "This affects new users and may prevent reinstallation.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        out, err = remove_provisioned(pkg)
        self._show_result(out, err)
        self.refresh_lists()

    def force_remove_click(self):
        if self._dev_mode_block():
            return
        e = self._get_selected_entry()
        if not e:
            return
        pkg = e["prov_pkg"] or e["package"]
        reply = QMessageBox.question(
            self,
            "DANGEROUS: Force Remove",
            f"Force remove ALL traces of:\n\n{e['display']}\n{pkg}?\n\n"
            "This may break dependencies. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        out, err = force_remove(pkg)
        self._show_result(out, err)
        self.refresh_lists()

# ============================================================
# Simple Services panel (read-only list)
# ============================================================

class ServicesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Display Name", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("⧖ᛟ⧫ — Refresh Services", self)
        self.btn_refresh.clicked.connect(self.refresh_services)
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch()

        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

        self.refresh_services()

    def refresh_services(self):
        cmd = "Get-Service | Select Name,DisplayName,Status | ConvertTo-Json"
        out, err = run_ps(cmd)
        self.table.setRowCount(0)
        if not out:
            return
        try:
            data = json.loads(out)
            if isinstance(data, dict):
                data = [data]
            for svc in data:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(svc.get("Name", "")))
                self.table.setItem(row, 1, QTableWidgetItem(svc.get("DisplayName", "")))
                self.table.setItem(row, 2, QTableWidgetItem(svc.get("Status", "")))
        except Exception:
            pass

# ============================================================
# UNKNOWN SYSTEMS overrides helpers
# ============================================================

def load_overrides():
    if not os.path.exists(OVERRIDES_FILE):
        return {}
    try:
        with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_overrides(data):
    try:
        with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        QMessageBox.warning(None, "Override Save Error", str(e))

def get_override_key(source, identifier, prop):
    return f"{source}::{identifier}::{prop}"

# ============================================================
# UNKNOWN SYSTEMS panel (hybrid: Real / Shadow / Effective)
# ============================================================

class UnknownSystemsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.overrides = load_overrides()
        self.current_source = ""
        self.current_identifier = ""
        self.current_real_rows = []  # list of dicts with real_value

        layout = QVBoxLayout(self)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        layout.addWidget(self.banner)

        info = QLabel(
            "⧫⧖ᛉ — UNKNOWN SYSTEMS\n"
            "Hybrid shadow of the real system. REAL = live, SHADOW = JSON, AUTO = hybrid overlay.\n"
            "When the real system is injured, hybrid bionic overrides can compensate."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_layout = QHBoxLayout()
        self.btn_wmi = QPushButton("⟒ᛉ⧗ — Deep WMI Probe", self)
        self.btn_hidden_reg = QPushButton("⧖ᛟ⧫ — Hidden Registry Hives", self)
        self.btn_obscure_tasks = QPushButton("ᛉ⧫⟒ — Obscure Scheduled Tasks", self)
        self.btn_unknown_appx = QPushButton("⧫ᛟ⟒ — Unknown AppX Packages", self)

        self.btn_wmi.clicked.connect(self.probe_wmi)
        self.btn_hidden_reg.clicked.connect(self.probe_hidden_reg)
        self.btn_obscure_tasks.clicked.connect(self.probe_tasks)
        self.btn_unknown_appx.clicked.connect(self.probe_appx)

        btn_layout.addWidget(self.btn_wmi)
        btn_layout.addWidget(self.btn_hidden_reg)
        btn_layout.addWidget(self.btn_obscure_tasks)
        btn_layout.addWidget(self.btn_unknown_appx)

        layout.addLayout(btn_layout)

        self.table = QTableWidget(self)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Property", "Real Value", "Shadow Value", "Effective (Hybrid)", "Type", "Source", "Identifier"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        bottom_layout = QHBoxLayout()
        self.btn_apply = QPushButton("⧖ᛞ⟟ — Apply Shadow Changes (Save JSON)", self)
        self.btn_apply.clicked.connect(self.apply_changes)
        bottom_layout.addWidget(self.btn_apply)
        bottom_layout.addStretch()
        layout.addLayout(bottom_layout)

        self.update_banner()
        self.update_editability()

    def update_banner(self):
        global HEALTH_STATE
        health_text = {
            "OK": "SYSTEM HEALTH: OK — Real system stable.",
            "DEGRADED": "SYSTEM HEALTH: DEGRADED — Bionic hybrid assistance recommended.",
            "FAILED": "SYSTEM HEALTH: FAILED — BIONIC OVERRIDE ENGAGED (Hybrid taking over)."
        }.get(HEALTH_STATE, "SYSTEM HEALTH: UNKNOWN")

        if DEV_MODE.enabled:
            self.banner.setText(
                f"{health_text}\n"
                f"DEVELOPMENT / HYBRID MODE ACTIVE — VIEW: {DEV_MODE.view_mode}\n"
                "REAL: Live system (Real column only, read-only)\n"
                "SHADOW: JSON shadow (Shadow column editable)\n"
                "AUTO: Hybrid — Effective column = Real overlaid with Shadow; edits go to Shadow.\n"
                "When the real system is injured, Effective values can act as bionic replacements."
            )
        else:
            self.banner.setText(
                f"{health_text}\n"
                "Development Mode OFF — REAL view only, read-only hybrid display."
            )

    def update_editability(self):
        mode = DEV_MODE.view_mode
        if not DEV_MODE.enabled:
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.btn_apply.setEnabled(False)
            return

        if mode == "REAL":
            self.table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.btn_apply.setEnabled(False)
        else:
            # SHADOW and AUTO (hybrid) allow editing of Shadow column
            self.table.setEditTriggers(QTableWidget.AllEditTriggers)
            self.btn_apply.setEnabled(True)

    def _parse_text_to_rows(self, text, source, identifier):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                prop, val = line.split(":", 1)
                prop = prop.strip()
                val = val.strip()
            else:
                prop = line
                val = ""
            rows.append({
                "property": prop,
                "real_value": val,
                "type": "string",
                "source": source,
                "identifier": identifier
            })
        return rows

    def _load_rows_into_table(self, rows):
        self.table.setRowCount(0)
        for r in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(r["property"]))
            self.table.setItem(row, 1, QTableWidgetItem(r.get("real_value", "")))
            self.table.setItem(row, 2, QTableWidgetItem(r.get("shadow_value", "")))
            eff_item = QTableWidgetItem(r.get("effective_value", ""))
            # Highlight where real is broken but shadow compensates (bionic takeover)
            real_val = r.get("real_value", "")
            shadow_val = r.get("shadow_value", "")
            if HEALTH_STATE in ("DEGRADED", "FAILED") and shadow_val and shadow_val != real_val:
                eff_item.setBackground(QColor(80, 40, 0))
            self.table.setItem(row, 3, eff_item)
            self.table.setItem(row, 4, QTableWidgetItem(r.get("type", "string")))
            self.table.setItem(row, 5, QTableWidgetItem(r["source"]))
            self.table.setItem(row, 6, QTableWidgetItem(r["identifier"]))

    def _build_hybrid_rows(self):
        # Build union of properties from real + shadow for current source/identifier
        props = {}
        # Start with real
        for r in self.current_real_rows:
            key = r["property"]
            props[key] = {
                "property": key,
                "real_value": r.get("real_value", ""),
                "shadow_value": "",
                "type": r.get("type", "string"),
                "source": r["source"],
                "identifier": r["identifier"],
            }
        # Overlay shadow
        for key, data in self.overrides.items():
            src = data.get("source", "")
            ident = data.get("identifier", "")
            if src != self.current_source or ident != self.current_identifier:
                continue
            prop = data.get("property", "")
            if not prop:
                continue
            if prop not in props:
                props[prop] = {
                    "property": prop,
                    "real_value": "",
                    "shadow_value": "",
                    "type": data.get("type", "string"),
                    "source": src,
                    "identifier": ident,
                }
            props[prop]["shadow_value"] = data.get("value", props[prop].get("shadow_value", ""))
            props[prop]["type"] = data.get("type", props[prop].get("type", "string"))

        # Compute effective (hybrid) value
        rows = []
        for prop, r in props.items():
            shadow = r.get("shadow_value", "")
            real = r.get("real_value", "")
            # Default hybrid: shadow if present, else real
            effective = shadow if shadow != "" else real

            # If system is injured, we treat shadow as preferred bionic replacement when it differs
            if HEALTH_STATE in ("DEGRADED", "FAILED") and shadow != "":
                effective = shadow

            r["effective_value"] = effective
            rows.append(r)
        return rows

    def _build_view_rows(self):
        mode = DEV_MODE.view_mode
        if not self.current_real_rows and not self.overrides:
            return []

        rows = self._build_hybrid_rows()

        if not DEV_MODE.enabled or mode == "REAL":
            # REAL mode: show hybrid info but treat Effective as informational
            return rows

        if mode == "SHADOW":
            # SHADOW mode: Effective = Shadow
            for r in rows:
                r["effective_value"] = r.get("shadow_value", "")
            return rows

        if mode == "AUTO":
            # AUTO: true hybrid, already computed with health-aware logic
            return rows

        return rows

    def _probe_generic(self, cmd, source, identifier):
        out, err = run_ps(cmd)
        text = out or err or "No data."
        self.current_source = source
        self.current_identifier = identifier
        self.current_real_rows = self._parse_text_to_rows(text, source, identifier)
        rows = self._build_view_rows()
        self._load_rows_into_table(rows)
        self.update_banner()
        self.update_editability()

    def probe_wmi(self):
        cmd = "Get-WmiObject -List | Select -First 50 | Out-String"
        self._probe_generic(cmd, "WMI", "Get-WmiObject -List")

    def probe_hidden_reg(self):
        cmd = r"Get-Item 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' | Format-List * | Out-String"
        self._probe_generic(cmd, "Registry", r"HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager")

    def probe_tasks(self):
        cmd = "Get-ScheduledTask | Where-Object {$_.TaskPath -notlike '\\Microsoft\\*'} | Select -First 50 | Out-String"
        self._probe_generic(cmd, "Tasks", "Get-ScheduledTask (non-Microsoft)")

    def probe_appx(self):
        cmd = "Get-AppxPackage -AllUsers | Select -First 50 | Out-String"
        self._probe_generic(cmd, "AppX", "Get-AppxPackage -AllUsers")

    def apply_changes(self):
        if not DEV_MODE.enabled:
            QMessageBox.information(self, "Dev Mode", "Development Mode is OFF. No shadow/hybrid writes allowed.")
            return
        mode = DEV_MODE.view_mode
        if mode not in ("SHADOW", "AUTO"):
            QMessageBox.information(self, "Dev Mode", "Edits only apply in SHADOW or AUTO (hybrid) view.")
            return

        rows = self.table.rowCount()
        for row in range(rows):
            prop_item = self.table.item(row, 0)
            shadow_item = self.table.item(row, 2)
            type_item = self.table.item(row, 4)
            src_item = self.table.item(row, 5)
            id_item = self.table.item(row, 6)
            if not (prop_item and shadow_item and src_item and id_item):
                continue
            prop = prop_item.text().strip()
            shadow_val = shadow_item.text()
            type_hint = type_item.text().strip() if type_item else "string"
            source = src_item.text().strip()
            identifier = id_item.text().strip()
            key = get_override_key(source, identifier, prop)

            self.overrides[key] = {
                "value": shadow_val,
                "type": type_hint,
                "source": source,
                "identifier": identifier,
                "property": prop,
            }

        save_overrides(self.overrides)
        QMessageBox.information(self, "Overrides Saved", f"Hybrid shadow changes saved to {OVERRIDES_FILE}.")

# ============================================================
# Development Mode panel (REAL / SHADOW / AUTO = hybrid)
# ============================================================

class DevModePanel(QWidget):
    def __init__(self, parent=None, unknown_panel: UnknownSystemsPanel = None):
        super().__init__(parent)
        self.unknown_panel = unknown_panel

        layout = QVBoxLayout(self)

        title = QLabel("⟒⧫⟒ — DEVELOPMENT / HYBRID MODE")
        title.setWordWrap(True)
        layout.addWidget(title)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        toggle_layout = QHBoxLayout()
        self.btn_enable = QPushButton("Enable Development Mode", self)
        self.btn_disable = QPushButton("Disable Development Mode", self)
        self.btn_enable.clicked.connect(self.enable_dev_mode)
        self.btn_disable.clicked.connect(self.disable_dev_mode)
        toggle_layout.addWidget(self.btn_enable)
        toggle_layout.addWidget(self.btn_disable)
        layout.addLayout(toggle_layout)

        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View Mode:"))
        self.view_combo = QComboBox(self)
        self.view_combo.addItems(["REAL", "SHADOW", "AUTO"])
        self.view_combo.currentTextChanged.connect(self.change_view_mode)
        view_layout.addWidget(self.view_combo)
        view_layout.addStretch()
        layout.addLayout(view_layout)

        desc = QLabel(
            "REAL: Live system, read-only.\n"
            "SHADOW: JSON shadow system, Shadow column editable.\n"
            "AUTO: Hybrid — Effective column = Real overlaid with Shadow; edits go to Shadow.\n"
            "When the real system is injured, AUTO acts as a bionic overlay."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()
        self.refresh_ui()

    def refresh_ui(self):
        global HEALTH_STATE
        self.view_combo.setCurrentText(DEV_MODE.view_mode)
        health_text = {
            "OK": "SYSTEM HEALTH: OK",
            "DEGRADED": "SYSTEM HEALTH: DEGRADED — hybrid assistance recommended.",
            "FAILED": "SYSTEM HEALTH: FAILED — bionic override engaged."
        }.get(HEALTH_STATE, "SYSTEM HEALTH: UNKNOWN")

        if DEV_MODE.enabled:
            self.status_label.setText(
                f"{health_text}\n"
                f"Development/Hybrid Mode is ENABLED.\nCurrent view: {DEV_MODE.view_mode}.\n"
                "All writes are shadow-only (JSON), no real system mutation."
            )
        else:
            self.status_label.setText(
                f"{health_text}\n"
                "Development/Hybrid Mode is DISABLED.\nAll panels operate in REAL, read-only mode."
            )

    def enable_dev_mode(self):
        DEV_MODE.enabled = True
        DEV_MODE.save()
        self.refresh_ui()
        if self.unknown_panel:
            self.unknown_panel.update_banner()
            self.unknown_panel.update_editability()
            rows = self.unknown_panel._build_view_rows()
            self.unknown_panel._load_rows_into_table(rows)

    def disable_dev_mode(self):
        DEV_MODE.enabled = False
        DEV_MODE.save()
        self.refresh_ui()
        if self.unknown_panel:
            self.unknown_panel.update_banner()
            self.unknown_panel.update_editability()
            rows = self.unknown_panel._build_view_rows()
            self.unknown_panel._load_rows_into_table(rows)

    def change_view_mode(self, text):
        DEV_MODE.view_mode = text
        DEV_MODE.save()
        self.refresh_ui()
        if self.unknown_panel:
            self.unknown_panel.update_banner()
            self.unknown_panel.update_editability()
            rows = self.unknown_panel._build_view_rows()
            self.unknown_panel._load_rows_into_table(rows)

# ============================================================
# Alien stylesheet
# ============================================================

ALIEN_QSS = """
QMainWindow {
    background-color: #050506;
}

QWidget {
    color: #f5e642;
    font-family: 'Consolas';
    font-size: 10pt;
}

QSplitter::handle {
    background-color: #3b1010;
    width: 6px;
}

QSplitter::handle:hover {
    background-color: #7a0a0a;
}

QListWidget {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #120308,
        stop:0.5 #2b0f0f,
        stop:1 #120308
    );
    border-right: 2px solid #7a0a0a;
    border-radius: 18px;
    padding: 8px;
}

QListWidget::item {
    padding: 10px;
    margin: 6px;
    border-radius: 16px;
    color: #f5e642;
    border: 1px solid rgba(212,160,23,60);
}

QListWidget::item:selected {
    background-color: #7a0a0a;
    color: #ffffff;
    border: 1px solid #d4a017;
}

QListWidget::item:hover {
    background-color: #2b0f0f;
}

QStackedWidget {
    background-color: #111111;
    border: 2px solid #7a0a0a;
    border-radius: 22px;
}

QTableWidget, QTreeWidget, QTextEdit, QLineEdit {
    background-color: #111;
    color: #ffdddd;
    border: 1px solid #7a0a0a;
    selection-background-color: #7a0a0a;
    selection-color: #ffffff;
    border-radius: 8px;
}

QHeaderView::section {
    background-color: #2b0f0f;
    color: #ffcccc;
    padding: 4px;
    border: 1px solid #7a0a0a;
}

QPushButton {
    background-color: #2b0f0f;
    color: #ffcccc;
    border: 1px solid #d4a017;
    padding: 6px 14px;
    border-radius: 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #7a0a0a;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #a31212;
}

QLabel {
    color: #f5e642;
}
"""

# ============================================================
# Health monitor (bionic trigger)
# ============================================================

class HealthMonitor:
    """
    Very simple health monitor:
    - OK: critical services running, no major PS error
    - DEGRADED: some critical services stopped
    - FAILED: PowerShell health check fails badly
    """
    CRITICAL_SERVICES = ["wuauserv", "bits"]  # example critical services

    def check_health(self):
        # Check critical services
        try:
            svc_filter = ",".join(f"'{s}'" for s in self.CRITICAL_SERVICES)
            cmd = (
                f"$crit = @({svc_filter}); "
                "Get-Service | Where-Object { $_.Name -in $crit -and $_.Status -eq 'Stopped' } | "
                "Select -First 1 | ConvertTo-Json"
            )
            out, err = run_ps(cmd)
            if err and "Get-Service" in err:
                return "FAILED"
            if out:
                # At least one critical service stopped
                return "DEGRADED"
        except Exception:
            return "FAILED"

        # If we reach here, basic health is OK
        return "OK"

# ============================================================
# Main window with alien layout + pulsing nav + health-driven hybrid
# ============================================================

class AlienAdminMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⟒⧫ᛟ  ALIEN ADMIN COCKPIT — HYBRID BIONIC SYSTEM  ⧖ᛉ⧗")
        self.resize(1400, 850)

        central = QWidget(self)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        self.nav = QListWidget(self)
        self.nav.setMinimumWidth(260)

        font = QFont("Consolas", 10)
        font.setBold(True)
        self.nav.setFont(font)

        items = [
            ("⟒⧫ᛟ — Processes", "processes"),
            ("⧖⟟ᛉ — Registry", "registry"),
            ("⟟ᛞ⧗ — PowerShell", "powershell"),
            ("⧫ᛟ⟒ — Windows Apps", "apps"),
            ("ᛉ⧗⟒ — Services", "services"),
            ("⧫⧖ᛉ — UNKNOWN SYSTEMS", "unknown"),
            ("⟒⧫⟒ — Development / Hybrid Mode", "devmode"),
        ]
        self.page_map = {}

        for text, key in items:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, key)
            self.nav.addItem(item)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(122, 10, 10, 180))
        shadow.setOffset(0, 0)
        self.nav.setGraphicsEffect(shadow)

        self.stack = QStackedWidget(self)

        self.processes_panel = ProcessesPanel(self)
        self.registry_panel = RegistryPanel(self)
        self.powershell_panel = PowerShellPanel(self)
        self.apps_panel = AppsPanel(self)
        self.services_panel = ServicesPanel(self)
        self.unknown_panel = UnknownSystemsPanel(self)
        self.devmode_panel = DevModePanel(self, unknown_panel=self.unknown_panel)

        self.page_map["processes"] = self.stack.addWidget(self.processes_panel)
        self.page_map["registry"] = self.stack.addWidget(self.registry_panel)
        self.page_map["powershell"] = self.stack.addWidget(self.powershell_panel)
        self.page_map["apps"] = self.stack.addWidget(self.apps_panel)
        self.page_map["services"] = self.stack.addWidget(self.services_panel)
        self.page_map["unknown"] = self.stack.addWidget(self.unknown_panel)
        self.page_map["devmode"] = self.stack.addWidget(self.devmode_panel)

        self.nav.currentItemChanged.connect(self.on_nav_changed)
        self.nav.setCurrentRow(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.nav)
        splitter.addWidget(self.stack)
        splitter.setSizes([400, 1000])

        root_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._pulse_state = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_nav)
        self._pulse_timer.start(900)

        # Health monitor timer
        self.health_monitor = HealthMonitor()
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self._check_health_and_adapt)
        self.health_timer.start(5000)  # every 5 seconds

        # Initial health check
        self._check_health_and_adapt(initial=True)

    def on_nav_changed(self, current, previous):
        if not current:
            return
        key = current.data(Qt.UserRole)
        idx = self.page_map.get(key)
        if idx is not None:
            self.stack.setCurrentIndex(idx)

    def _pulse_nav(self):
        self._pulse_state = not self._pulse_state
        if DEV_MODE.enabled:
            border_color = "#d4a017" if self._pulse_state else "#ff4444"
        else:
            border_color = "#d4a017" if self._pulse_state else "#7a0a0a"
        extra = f"""
        QListWidget {{
            border-right: 2px solid {border_color};
        }}
        """
        QApplication.instance().setStyleSheet(ALIEN_QSS + extra)

    def _check_health_and_adapt(self, initial=False):
        global HEALTH_STATE
        new_state = self.health_monitor.check_health()
        if new_state == HEALTH_STATE and not initial:
            return

        HEALTH_STATE = new_state

        # Bionic behavior: if system is injured, hybrid mode comes alive
        if HEALTH_STATE in ("DEGRADED", "FAILED"):
            # Enable dev/hybrid mode automatically and switch to AUTO
            DEV_MODE.enabled = True
            DEV_MODE.view_mode = "AUTO"
            DEV_MODE.save()
        # We do NOT auto-disable dev mode when health returns to OK; user stays in control.

        # Refresh panels that depend on health/dev state
        self.devmode_panel.refresh_ui()
        self.unknown_panel.update_banner()
        self.unknown_panel.update_editability()
        rows = self.unknown_panel._build_view_rows()
        self.unknown_panel._load_rows_into_table(rows)

# ============================================================
# Entry point
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(ALIEN_QSS)

    if not is_admin():
        reply = QMessageBox.question(
            None,
            "Admin Required",
            "This console can run with full administrator privileges.\n\n"
            "Do you want to relaunch as Administrator?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                relaunch_as_admin()
            except Exception as e:
                QMessageBox.critical(None, "Elevation Failed", str(e))
            sys.exit(0)

    window = AlienAdminMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

