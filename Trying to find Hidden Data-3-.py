import sys
import os
import subprocess
import importlib
import re
from datetime import datetime

# ============================================================
# AUTOLOADER FOR DEPENDENCIES
# ============================================================

REQUIRED_PACKAGES = [
    ("PyQt5", "PyQt5"),
    ("beautifulsoup4", "bs4"),
    ("lxml", None),
]

def ensure_package(pkg_name, import_name=None):
    if import_name is None:
        import_name = pkg_name
    try:
        importlib.import_module(import_name)
        return
    except ImportError:
        print(f"[AUTOLOADER] Package '{import_name}' not found. Installing '{pkg_name}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        except Exception as e:
            print(f"[AUTOLOADER] Failed to install {pkg_name}: {e}")
        importlib.import_module(import_name)
        print(f"[AUTOLOADER] Successfully imported '{import_name}'.")

def ensure_all_packages():
    for pkg, import_name in REQUIRED_PACKAGES:
        ensure_package(pkg, import_name)

ensure_all_packages()

from PyQt5 import QtWidgets, QtGui, QtCore
from bs4 import BeautifulSoup
from email import message_from_bytes, message_from_string

# ============================================================
# DATA MODEL
# ============================================================

class Finding:
    def __init__(self, ftype, description, snippet, source_type, source_id, severity="info"):
        self.ftype = ftype
        self.description = description
        self.snippet = snippet
        self.source_type = source_type  # "web" / "email" / "raw"
        self.source_id = source_id
        self.severity = severity
        self.timestamp = datetime.utcnow().isoformat(timespec="seconds")

    def to_row(self):
        return [
            self.timestamp,
            self.severity,
            self.source_type,
            self.source_id,
            self.ftype,
            self.description,
        ]

# ============================================================
# DETECTOR
# ============================================================

class HiddenDataDetector:
    def __init__(self):
        self.redirector_domains = {"bit.ly", "t.co", "lnkd.in", "tinyurl.com", "goo.gl"}
        self.suspicious_script_domains = {
            "doubleclick.net",
            "googletagmanager.com",
            "google-analytics.com",
            "facebook.net",
        }
        self.dangerous_ext = (
            ".exe",".js",".vbs",".ps1",".bat",".cmd",".scr",".hta",".jar",".com",".pif",
            ".cpl",".msc",".dll",".sys",".docm",".xlsm",".pptm"
        )

    # -------- main entry points --------

    def scan_html(self, html_text, source_type="web", source_id="N/A"):
        soup = BeautifulSoup(html_text, "lxml")
        findings = []
        findings.extend(self._scan_images(soup, source_type, source_id))
        findings.extend(self._scan_hidden_elements(soup, source_type, source_id))
        findings.extend(self._scan_links(soup, source_type, source_id))
        findings.extend(self._scan_iframes_scripts(soup, source_type, source_id))
        findings.extend(self._scan_auto_forms(soup, source_type, source_id))
        return findings

    def scan_eml(self, data, source_id="Email"):
        if isinstance(data, bytes):
            msg = message_from_bytes(data)
        else:
            msg = message_from_string(data)

        findings = []
        # attachments
        if msg.is_multipart():
            for part in msg.walk():
                filename = part.get_filename()
                if filename:
                    lower = filename.lower()
                    if lower.endswith(self.dangerous_ext):
                        findings.append(
                            Finding(
                                ftype="Dangerous attachment",
                                description=f"Attachment '{filename}' is potentially executable.",
                                snippet=f"Attachment: {filename}",
                                source_type="email",
                                source_id=source_id,
                                severity="critical",
                            )
                        )

        # html parts
        html_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        html_parts.append(payload.decode(charset, errors="ignore"))
                    except Exception:
                        pass
        else:
            if msg.get_content_type() == "text/html":
                try:
                    payload = msg.get_payload(decode=True)
                    charset = msg.get_content_charset() or "utf-8"
                    html_parts.append(payload.decode(charset, errors="ignore"))
                except Exception:
                    pass

        for idx, html in enumerate(html_parts):
            part_id = f"{source_id} (HTML part {idx+1})"
            findings.extend(self.scan_html(html, source_type="email", source_id=part_id))

        return findings

    # -------- helpers --------

    def _scan_images(self, soup, source_type, source_id):
        findings = []
        for img in soup.find_all("img"):
            style = img.get("style", "") or ""
            width = img.get("width")
            height = img.get("height")
            src = img.get("src", "") or ""

            suspicious = False
            reason = []
            severity = "info"

            def small(v):
                try:
                    return int(v) <= 2
                except Exception:
                    return False

            if width and small(width):
                suspicious = True
                reason.append(f"width={width}")
            if height and small(height):
                suspicious = True
                reason.append(f"height={height}")

            st = style.lower()
            if any(flag in st for flag in ["display:none", "visibility:hidden", "opacity:0"]):
                suspicious = True
                reason.append("hidden via CSS")

            tiny_pattern = re.compile(r"(width|height)\s*:\s*([0-9]+)px", re.IGNORECASE)
            for m in tiny_pattern.finditer(style):
                dim_val = m.group(2)
                if small(dim_val):
                    suspicious = True
                    reason.append(f"{m.group(1)}={dim_val}px")

            if suspicious:
                if "http" in src:
                    severity = "suspicious"
                desc = "Potential tracking pixel or hidden image: " + ", ".join(reason)
                snippet = str(img)[:300]
                findings.append(
                    Finding("Tracking pixel", desc, snippet, source_type, source_id, severity)
                )
        return findings

    def _scan_hidden_elements(self, soup, source_type, source_id):
        findings = []
        css_flags = ["display:none", "visibility:hidden", "opacity:0"]
        offscreen = [
            re.compile(r"text-indent\s*:\s*-?\s*1000px", re.IGNORECASE),
            re.compile(r"left\s*:\s*-?\s*1000px", re.IGNORECASE),
            re.compile(r"top\s*:\s*-?\s*1000px", re.IGNORECASE),
        ]

        for el in soup.find_all(True):
            style = (el.get("style", "") or "").lower()
            hidden = False
            reason = []

            if any(flag in style for flag in css_flags):
                hidden = True
                reason.append("hidden by CSS")

            for pat in offscreen:
                if pat.search(style):
                    hidden = True
                    reason.append("off-screen positioning")

            if hidden:
                desc = "Hidden element: " + ", ".join(reason)
                snippet = str(el)[:300]
                findings.append(
                    Finding("Hidden element", desc, snippet, source_type, source_id, "info")
                )
        return findings

    def _scan_links(self, soup, source_type, source_id):
        findings = []
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            text = (a.get_text() or "").strip()
            desc_parts = []
            severity = "info"

            domain = self._extract_domain(href)
            text_domain = self._extract_domain(text) if text else None

            if text_domain and domain and text_domain.lower() != domain.lower():
                desc_parts.append(f"text domain '{text_domain}' != href domain '{domain}'")
                severity = "suspicious"

            if domain and domain.lower() in self.redirector_domains:
                desc_parts.append(f"known redirector '{domain}'")
                severity = "suspicious"

            if "?" in href and len(href) > 120:
                desc_parts.append("long URL with query (possible tracking)")
                if severity == "info":
                    severity = "suspicious"

            if desc_parts:
                desc = "Suspicious/obfuscated link: " + "; ".join(desc_parts)
                snippet = str(a)[:300]
                findings.append(
                    Finding("Obfuscated link", desc, snippet, source_type, source_id, severity)
                )
        return findings

    def _scan_iframes_scripts(self, soup, source_type, source_id):
        findings = []

        for iframe in soup.find_all("iframe"):
            src = (iframe.get("src") or "").strip()
            style = (iframe.get("style") or "").lower()
            hidden = any(flag in style for flag in ["display:none", "visibility:hidden", "opacity:0"])
            if hidden:
                domain = self._extract_domain(src) or "unknown"
                desc = f"Hidden iframe loading from '{domain}'"
                snippet = str(iframe)[:300]
                findings.append(
                    Finding("Hidden iframe", desc, snippet, source_type, source_id, "suspicious")
                )

        for script in soup.find_all("script"):
            src = (script.get("src") or "").strip()
            if not src:
                continue
            domain = self._extract_domain(src) or "unknown"
            if any(d in domain for d in self.suspicious_script_domains):
                desc = f"Script loaded from tracking/suspicious domain '{domain}'"
                snippet = str(script)[:300]
                findings.append(
                    Finding("Suspicious script", desc, snippet, source_type, source_id, "suspicious")
                )

        return findings

    def _scan_auto_forms(self, soup, source_type, source_id):
        findings = []
        for form in soup.find_all("form"):
            form_str = str(form)
            auto = False

            if "onsubmit" in form.attrs:
                val = str(form.get("onsubmit") or "").lower()
                if "submit" in val:
                    auto = True

            for script in form.find_all("script"):
                txt = script.get_text() or ""
                if "submit()" in txt:
                    auto = True

            if auto:
                action = form.get("action", "")
                domain = self._extract_domain(action) or "unknown"
                desc = f"Form with potential auto-submit behavior to '{domain}'"
                snippet = form_str[:300]
                findings.append(
                    Finding("Auto-submit form", desc, snippet, source_type, source_id, "suspicious")
                )
        return findings

    @staticmethod
    def _extract_domain(url):
        if not url:
            return None
        tmp = re.sub(r"^[a-zA-Z]+://", "", url)
        tmp = tmp.split("/", 1)[0]
        tmp = tmp.split("?", 1)[0]
        tmp = tmp.split("@")[-1]
        tmp = tmp.split(":", 1)[0]
        tmp = tmp.strip()
        return tmp or None

# ============================================================
# GUI
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = HiddenDataDetector()
        self.current_source = ""
        self.current_source_type = "raw"
        self.current_source_id = "N/A"
        self.findings = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hidden Data & Behavior Inspector (Single-file)")

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)

        # Top buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load HTML/EML")
        self.btn_paste = QtWidgets.QPushButton("Paste Source")
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_clear = QtWidgets.QPushButton("Clear Results")
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_paste)
        btn_row.addWidget(self.btn_scan)
        btn_row.addWidget(self.btn_clear)
        v.addLayout(btn_row)

        self.info_label = QtWidgets.QLabel("No source loaded.")
        v.addWidget(self.info_label)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Severity", "Source type", "Source id", "Type", "Description"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        v.addWidget(self.table)

        # Snippet
        self.snippet_view = QtWidgets.QPlainTextEdit()
        self.snippet_view.setReadOnly(True)
        self.snippet_view.setPlaceholderText("Select a row to see snippet...")
        v.addWidget(self.snippet_view)

        self.setCentralWidget(central)

        # Signals
        self.btn_load.clicked.connect(self.load_file)
        self.btn_paste.clicked.connect(self.paste_source)
        self.btn_scan.clicked.connect(self.scan_source)
        self.btn_clear.clicked.connect(self.clear_results)
        self.table.itemSelectionChanged.connect(self.update_snippet)

        # Tray icon for popups
        self.tray = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        self.tray.setIcon(icon)
        self.tray.setVisible(True)

    # ---- actions ----

    def load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open HTML or EML",
            "",
            "All Files (*.*);;HTML Files (*.html *.htm);;Email Files (*.eml)"
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read file:\n{e}")
            return

        ext = os.path.splitext(path)[1].lower()
        if ext == ".eml":
            self.current_source = data
            self.current_source_type = "email"
        else:
            try:
                self.current_source = data.decode("utf-8", errors="ignore")
            except Exception:
                self.current_source = data.decode("latin1", errors="ignore")
            self.current_source_type = "web"

        self.current_source_id = os.path.basename(path)
        self.info_label.setText(f"Loaded {self.current_source_type} from file: {self.current_source_id}")

    def paste_source(self):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Paste Source", "Paste raw HTML or email source:"
        )
        if ok and text:
            self.current_source = text
            self.current_source_type = "raw"
            self.current_source_id = "Pasted"
            self.info_label.setText("Loaded pasted source.")

    def scan_source(self):
        if not self.current_source:
            QtWidgets.QMessageBox.warning(self, "No content", "Load or paste content first.")
            return

        if self.current_source_type == "email":
            findings = self.detector.scan_eml(self.current_source, self.current_source_id)
        else:
            findings = self.detector.scan_html(
                self.current_source,
                source_type=self.current_source_type,
                source_id=self.current_source_id,
            )

        self.show_findings(findings)

    def show_findings(self, findings):
        self.findings = findings
        self.table.setRowCount(0)

        suspicious_count = 0
        critical_count = 0

        for f in findings:
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, val in enumerate(f.to_row()):
                item = QtWidgets.QTableWidgetItem(str(val))
                if f.severity == "critical":
                    item.setBackground(QtGui.QColor("#ffcccc"))
                elif f.severity == "suspicious":
                    item.setBackground(QtGui.QColor("#fff2cc"))
                self.table.setItem(row, col, item)

            if f.severity == "suspicious":
                suspicious_count += 1
            elif f.severity == "critical":
                critical_count += 1

        self.table.resizeColumnsToContents()

        total = len(findings)
        if total == 0:
            QtWidgets.QMessageBox.information(self, "Result", "No hidden items detected with current rules.")
        else:
            msg = f"Found {total} items.\nSuspicious: {suspicious_count}\nCritical: {critical_count}"
            QtWidgets.QMessageBox.information(self, "Hidden data detected", msg)
            if suspicious_count or critical_count:
                self.tray.showMessage(
                    "Hidden data detected",
                    msg,
                    QtWidgets.QSystemTrayIcon.Warning,
                    10000,
                )

    def clear_results(self):
        self.table.setRowCount(0)
        self.findings = []
        self.snippet_view.clear()

    def update_snippet(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self.findings):
            self.snippet_view.setPlainText(self.findings[idx].snippet)

# ============================================================
# MAIN
# ============================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

