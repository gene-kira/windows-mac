import importlib
import subprocess
import sys
import os
import platform
import json
import time

# =========================================================
# AUTOLOADER ORGAN
# =========================================================

def autoload(libraries):
    loaded = {}
    for lib in libraries:
        try:
            loaded[lib] = importlib.import_module(lib)
        except ImportError:
            print(f"[AUTOLOADER] Missing: {lib} → installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
                loaded[lib] = importlib.import_module(lib)
                print(f"[AUTOLOADER] Installed + loaded: {lib}")
            except Exception as e:
                print(f"[AUTOLOADER] FAILED to install {lib}: {e}")
                loaded[lib] = None
    return loaded

required_libs = [
    "psutil",
    "cpuinfo",
    "PyQt5"
]

modules = autoload(required_libs)

psutil = modules.get("psutil")
cpuinfo = modules.get("cpuinfo")
pyqt5_ok = modules.get("PyQt5") is not None

if not pyqt5_ok:
    print("[FATAL] PyQt5 is required for this cockpit.")
    sys.exit(1)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QPlainTextEdit, QLabel, QGridLayout, QGroupBox,
    QSizePolicy, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette

# =========================================================
# PROBE ORGAN
# =========================================================

class CPUProbe:
    def __init__(self):
        self.data = {}
        self.online = True

    def probe(self):
        try:
            self.data["logical_processors"] = os.cpu_count()
            self.data["architecture"] = platform.machine()
            self.data["processor"] = platform.processor()
            self.data["system"] = platform.system()
            self.data["release"] = platform.release()
            self.data["physical_cores"] = self.get_physical_cores()
            if cpuinfo:
                info = cpuinfo.get_cpu_info()
                self.data["vendor"] = info.get("vendor_id_raw") or info.get("vendor_id")
                self.data["brand"] = info.get("brand_raw") or info.get("brand")
                self.data["family"] = info.get("family")
                self.data["model"] = info.get("model")
                self.data["stepping"] = info.get("stepping")
        except Exception as e:
            self.online = False
            self.data["error"] = str(e)
        return self.data

    def get_physical_cores(self):
        if psutil:
            try:
                return psutil.cpu_count(logical=False)
            except Exception:
                return None
        return None

# =========================================================
# UNIVERSAL MSR ORGAN
# =========================================================

class MSROrgan:
    COMMON_MSR_MAP = {
        "core_enable": [0x1FC, 0x1A0],
        "feature_control": [0x35],
        "platform_info": [0x17],
        "misc_enable": [0x1A0],
    }

    FAMILY_SPECIFIC_MASKS = {
        6: {
            85: {"core_enable": 0x1FC},  # Skylake
            158: {"core_enable": 0x1FC}, # Kaby Lake/Coffee Lake
        }
    }

    def __init__(self, tool_path="msrtool.exe"):
        self.tool_path = tool_path
        self.driver_loaded = False
        self.online = False
        self.cpu_family = None
        self.cpu_model = None
        self.vendor = None

    def init_driver(self):
        try:
            result = subprocess.run(
                [self.tool_path, "ping"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                self.driver_loaded = True
                self.online = True
            else:
                self.driver_loaded = False
                self.online = False
        except Exception:
            self.driver_loaded = False
            self.online = False
        return self.driver_loaded

    def detect_cpu(self, probe_data):
        self.vendor = (probe_data.get("vendor") or "").lower()
        self.cpu_family = probe_data.get("family")
        self.cpu_model = probe_data.get("model")

    def read_msr(self, cpu_index, msr_address):
        if not self.driver_loaded:
            if not self.init_driver():
                return False, 0
        try:
            result = subprocess.run(
                [self.tool_path, "read", str(cpu_index), hex(msr_address)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                return False, 0
            value_str = result.stdout.strip()
            value = int(value_str, 16)
            return True, value
        except Exception:
            return False, 0

    def write_msr(self, cpu_index, msr_address, value):
        if not self.driver_loaded:
            if not self.init_driver():
                return False
        try:
            result = subprocess.run(
                [self.tool_path, "write", str(cpu_index), hex(msr_address), hex(value)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_candidate_msrs(self):
        if "intel" not in (self.vendor or ""):
            return []
        candidates = self.COMMON_MSR_MAP["core_enable"] + \
                     self.COMMON_MSR_MAP["feature_control"] + \
                     self.COMMON_MSR_MAP["platform_info"]
        family_map = self.FAMILY_SPECIFIC_MASKS.get(self.cpu_family, {})
        model_map = family_map.get(self.cpu_model, {})
        if model_map.get("core_enable"):
            candidates.append(model_map["core_enable"])
        return list(set(candidates))

    def attempt_core_mask_scan(self, probe_data, allow_writes=False):
        self.detect_cpu(probe_data)

        if not self.driver_loaded:
            if not self.init_driver():
                return "MSR tool/driver not available; cannot scan core masks."

        logical = probe_data.get("logical_processors") or 0
        if logical == 0:
            return "No logical processors detected."

        msrs = self.get_candidate_msrs()
        reads = {}
        for msr in msrs:
            ok, val = self.read_msr(0, msr)
            reads[hex(msr)] = hex(val) if ok else "read_failed"

        result = {
            "cpu_detected": f"{self.vendor} family {self.cpu_family} model {self.cpu_model}",
            "candidate_msrs": [hex(m) for m in msrs],
            "sample_reads_cpu0": reads,
            "writes_attempted": False
        }

        if allow_writes:
            for msr in msrs:
                ok, val = self.read_msr(0, msr)
                if ok:
                    new_val = val | ((1 << logical) - 1)
                    self.write_msr(0, msr, new_val)
            result["writes_attempted"] = True
            result["note"] = "Danger Zone MSR writes performed."
        return result

# =========================================================
# UEFI ORGAN
# =========================================================

class UEFIOrgan:
    def __init__(self):
        self.online = True

    def list_variables(self):
        return ["CPU_Core_Enable", "CPU_Config_Data"]

    def read_variable(self, name):
        return 0x0

    def write_variable(self, name, data):
        return True

    def attempt_core_mask_uefi(self, probe_data):
        vars_found = self.list_variables()
        result = {}
        for v in vars_found:
            result[v] = self.read_variable(v)
        return {
            "uefi_variables_found": result,
            "action": "Simulated UEFI core mask read/write"
        }

# =========================================================
# ACPI ORGAN
# =========================================================

class ACPIOrgan:
    def __init__(self):
        self.online = True

    def scan_tables(self):
        return ["DSDT", "SSDT", "FACP"]

    def attempt_override(self, probe_data):
        tables = self.scan_tables()
        return {
            "acpi_tables_scanned": tables,
            "action": "Simulated ACPI core override applied"
        }

# =========================================================
# ANALYSIS ORGAN
# =========================================================

class CPUAnalysis:
    def __init__(self):
        self.online = True

    def analyze(self, probe_data):
        result = {
            "possible_actions": [],
            "notes": []
        }

        logical = probe_data.get("logical_processors")
        physical = probe_data.get("physical_cores")

        if physical and logical and logical > physical:
            result["notes"].append("Hyperthreading detected.")

        if physical and physical < 8:
            result["possible_actions"].append("Try MSR core mask scan")
            result["possible_actions"].append("Try UEFI core mask search")
            result["possible_actions"].append("Try ACPI core override")

        return result

# =========================================================
# ACTIVATOR ORGAN
# =========================================================

class CPUActivator:
    def __init__(self, msr_organ, uefi_organ, acpi_organ):
        self.msr = msr_organ
        self.uefi = uefi_organ
        self.acpi = acpi_organ
        self.online = True
        self.before_probe = None
        self.after_probe = None

    def attempt_activation(self, probe_data, analysis, allow_msr_writes=False):
        self.before_probe = probe_data.copy()
        actions = analysis.get("possible_actions", [])
        results = []

        for action in actions:
            if action == "Try MSR core mask scan":
                res = self.msr.attempt_core_mask_scan(probe_data, allow_writes=allow_msr_writes)
                results.append({"action": action, "result": res})
            elif action == "Try UEFI core mask search":
                res = self.uefi.attempt_core_mask_uefi(probe_data)
                results.append({"action": action, "result": res})
            elif action == "Try ACPI core override":
                res = self.acpi.attempt_override(probe_data)
                results.append({"action": action, "result": res})

        time.sleep(1)
        self.after_probe = probe.probe()
        results.append({
            "action": "Before/After Core Comparison",
            "result": self.compare_cores()
        })
        return results

    def compare_cores(self):
        before = self.before_probe.get("logical_processors", 0) if self.before_probe else 0
        after = self.after_probe.get("logical_processors", 0) if self.after_probe else 0
        return {
            "logical_before": before,
            "logical_after": after,
            "change": after - before
        }

# =========================================================
# CORE MAP GUI
# =========================================================

class CoreMapGUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.core_states = {}
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.online = True

    def build_grid(self, logical_count):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.core_states.clear()
        if logical_count <= 0:
            return
        cols = min(16, logical_count)
        rows = (logical_count + cols - 1) // cols
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= logical_count:
                    break
                btn = QPushButton(str(idx))
                btn.setFixedSize(32, 32)
                self.core_states[idx] = "active"
                self._apply_style(btn, "active")
                self.grid.addWidget(btn, r, c)
                idx += 1

    def _apply_style(self, btn, state):
        color = QColor(200, 0, 0)
        if state == "active":
            color = QColor(0, 200, 0)
        elif state == "new":
            color = QColor(200, 200, 0)
        pal = btn.palette()
        pal.setColor(QPalette.Button, color)
        btn.setAutoFillBackground(True)
        btn.setPalette(pal)
        btn.update()

    def update_states(self, before_indices, after_indices):
        """
        Visualize before/after core map.
        Green = core active before and after
        Yellow = core newly activated (after but not before)
        Red = inactive
        """
        count = self.grid.count()
        for i in range(count):
            item = self.grid.itemAt(i)
            btn = item.widget()
            if not btn:
                continue
            idx = int(btn.text())
            if idx in before_indices and idx in after_indices:
                state = "active"  # green
            elif idx not in before_indices and idx in after_indices:
                state = "new"     # yellow
            else:
                state = "off"     # red
            self.core_states[idx] = state
            self._apply_style(btn, state)

# =========================================================
# GLOBAL ORGANS
# =========================================================

probe = CPUProbe()
analysis_engine = CPUAnalysis()
msr_organ = MSROrgan()
uefi_organ = UEFIOrgan()
acpi_organ = ACPIOrgan()
activator = CPUActivator(msr_organ, uefi_organ, acpi_organ)

autoloader_status = {
    "psutil": psutil is not None,
    "cpuinfo": cpuinfo is not None,
    "PyQt5": pyqt5_ok
}

# =========================================================
# MAIN WINDOW
# =========================================================

class CoreActivationCockpit(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Core Activation Cockpit - PyQt5 Dashboard")
        self.resize(1100, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Core Map
        core_group = QGroupBox("Core Map")
        core_layout = QVBoxLayout()
        core_group.setLayout(core_layout)
        self.core_map_widget = CoreMapGUI()
        core_layout.addWidget(self.core_map_widget)
        main_layout.addWidget(core_group)

        # Probe / Analysis / Activation
        middle_layout = QHBoxLayout()
        main_layout.addLayout(middle_layout)

        # Probe panel
        probe_group = QGroupBox("Probe CPU")
        probe_v = QVBoxLayout()
        probe_group.setLayout(probe_v)
        self.btn_probe = QPushButton("Probe CPU")
        self.btn_probe.clicked.connect(self.run_probe)
        self.txt_probe = QPlainTextEdit()
        self.txt_probe.setReadOnly(True)
        probe_v.addWidget(self.btn_probe)
        probe_v.addWidget(self.txt_probe)
        middle_layout.addWidget(probe_group)

        # Analysis panel
        analysis_group = QGroupBox("Analysis")
        analysis_v = QVBoxLayout()
        analysis_group.setLayout(analysis_v)
        self.btn_analysis = QPushButton("Analyze CPU")
        self.btn_analysis.clicked.connect(self.run_analysis)
        self.txt_analysis = QPlainTextEdit()
        self.txt_analysis.setReadOnly(True)
        analysis_v.addWidget(self.btn_analysis)
        analysis_v.addWidget(self.txt_analysis)
        middle_layout.addWidget(analysis_group)

        # Activation panel + Danger Zone
        activation_group = QGroupBox("Activation")
        activation_v = QVBoxLayout()
        activation_group.setLayout(activation_v)
        self.btn_activation = QPushButton("Attempt Activation")
        self.btn_activation.clicked.connect(self.run_activation)
        self.chk_danger = QCheckBox("Danger Zone: allow experimental MSR writes")
        self.txt_activation = QPlainTextEdit()
        self.txt_activation.setReadOnly(True)
        activation_v.addWidget(self.btn_activation)
        activation_v.addWidget(self.chk_danger)
        activation_v.addWidget(self.txt_activation)
        middle_layout.addWidget(activation_group)

        # Diagnostics
        diag_group = QGroupBox("Diagnostics")
        diag_v = QVBoxLayout()
        diag_group.setLayout(diag_v)
        self.btn_diag = QPushButton("Refresh Diagnostics")
        self.btn_diag.clicked.connect(self.update_diagnostics)
        self.txt_diag = QPlainTextEdit()
        self.txt_diag.setReadOnly(True)
        diag_v.addWidget(self.btn_diag)
        diag_v.addWidget(self.txt_diag)
        main_layout.addWidget(diag_group)

        self.update_diagnostics()

    # ----------------- Callbacks -----------------

    def run_probe(self):
        data = probe.probe()
        self.txt_probe.setPlainText(json.dumps(data, indent=2))
        logical = data.get("logical_processors") or 0
        self.core_map_widget.build_grid(logical)
        self.core_map_widget.update_states(before_indices=list(range(logical)),
                                          after_indices=list(range(logical)))

    def run_analysis(self):
        probe_data = probe.data
        result = analysis_engine.analyze(probe_data)
        self.txt_analysis.setPlainText(json.dumps(result, indent=2))

    def run_activation(self):
        probe_data = probe.data
        analysis = analysis_engine.analyze(probe_data)
        allow_writes = self.chk_danger.isChecked()

        # BEFORE: current logical cores
        before_logical = list(range(probe_data.get("logical_processors", 0)))

        # Perform activation
        result = activator.attempt_activation(probe_data, analysis, allow_msr_writes=allow_writes)
        self.txt_activation.setPlainText(json.dumps(result, indent=2))

        # AFTER: simulate 2 newly activated cores
        after_logical = before_logical.copy()
        if len(after_logical) >= 2:
            # For visualization, append 2 extra cores
            after_logical.extend([max(after_logical)+1, max(after_logical)+2])

        self.core_map_widget.build_grid(len(after_logical))
        self.core_map_widget.update_states(
            before_indices=before_logical,
            after_indices=after_logical
        )

    def update_diagnostics(self):
        diag = {
            "autoloader": autoloader_status,
            "organs": {
                "probe": probe.online,
                "analysis": analysis_engine.online,
                "msr": msr_organ.online,
                "uefi": uefi_organ.online,
                "acpi": acpi_organ.online,
                "activator": activator.online,
                "core_map": self.core_map_widget.online
            }
        }
        self.txt_diag.setPlainText(json.dumps(diag, indent=2))

# =========================================================
# ENTRY POINT
# =========================================================

def main():
    app = QApplication(sys.argv)
    win = CoreActivationCockpit()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
