import sys
import os
import subprocess
import importlib
import threading
import time
import re
import json
import queue
import traceback
from datetime import datetime

# ============================================================
# AUTOLOADER FOR DEPENDENCIES
# ============================================================

REQUIRED_PACKAGES = [
    ("PyQt5", "PyQt5"),
    ("beautifulsoup4", "bs4"),
    ("lxml", None),
    ("flask", "flask"),
    ("requests", "requests"),
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
        # Try import again, let it fail loudly if not working
        importlib.import_module(import_name)
        print(f"[AUTOLOADER] Successfully installed and imported '{import_name}'.")

def ensure_all_packages():
    for pkg, import_name in REQUIRED_PACKAGES:
        ensure_package(pkg, import_name)

ensure_all_packages()

from PyQt5 import QtWidgets, QtGui, QtCore
from bs4 import BeautifulSoup
from email import message_from_bytes, message_from_string
from flask import Flask, request, jsonify
import imaplib

# ============================================================
# DATA MODELS
# ============================================================

class Finding:
    def __init__(self, ftype, description, snippet, location,
                 source_type, source_id, timestamp=None, severity="info"):
        self.ftype = ftype
        self.description = description
        self.snippet = snippet
        self.location = location
        self.source_type = source_type   # "web" / "email"
        self.source_id = source_id       # URL, subject, filename, etc.
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.severity = severity         # "info", "suspicious", "critical"

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
# HIDDEN DATA + SUSPICIOUS BEHAVIOR DETECTOR
# ============================================================

class HiddenDataDetector:
    def __init__(self):
        self.redirector_domains = {
            "bit.ly", "t.co", "lnkd.in", "goo.gl", "tinyurl.com"
        }
        self.suspicious_script_domains = {
            # Add tracking/ad/malicious domains here
            "doubleclick.net",
            "googletagmanager.com",
            "google-analytics.com",
            "facebook.net",
        }

    def scan_html(self, html_text, source_type="web", source_id="N/A"):
        soup = BeautifulSoup(html_text, "lxml")
        findings = []
        findings.extend(self._scan_images(soup, source_type, source_id))
        findings.extend(self._scan_hidden_elements(soup, source_type, source_id))
        findings.extend(self._scan_links(soup, source_type, source_id))
        findings.extend(self._scan_iframes_and_scripts(soup, source_type, source_id))
        findings.extend(self._scan_auto_forms(soup, source_type, source_id))
        return findings

    def scan_eml(self, raw_bytes_or_text, source_id="Email"):
        if isinstance(raw_bytes_or_text, bytes):
            msg = message_from_bytes(raw_bytes_or_text)
        else:
            msg = message_from_string(raw_bytes_or_text)

        findings = []

        # Check attachments (dangerous types)
        dangerous_ext = (".exe", ".js", ".vbs", ".ps1", ".bat", ".cmd",
                         ".scr", ".hta", ".jar", ".com", ".pif",
                         ".cpl", ".msc", ".dll", ".sys",
                         ".docm", ".xlsm", ".pptm")
        if msg.is_multipart():
            for part in msg.walk():
                filename = part.get_filename()
                if filename:
                    lower = filename.lower()
                    if lower.endswith(dangerous_ext):
                        findings.append(
                            Finding(
                                ftype="Dangerous attachment",
                                description=f"Attachment '{filename}' has potentially executable extension.",
                                snippet=f"Attachment: {filename}",
                                location="attachment",
                                source_type="email",
                                source_id=source_id,
                                severity="critical",
                            )
                        )

        # Extract HTML parts
        html_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/html":
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

    # ---------- image / pixel detector ----------

    def _scan_images(self, soup, source_type, source_id):
        findings = []
        for img in soup.find_all("img"):
            style = img.get("style", "") or ""
            width = img.get("width")
            height = img.get("height")

            suspicious = False
            reason_parts = []
            severity = "info"

            def small(v):
                try:
                    return int(v) <= 2
                except Exception:
                    return False

            if width and small(width):
                suspicious = True
                reason_parts.append(f"width={width}")
            if height and small(height):
                suspicious = True
                reason_parts.append(f"height={height}")

            st_lower = style.lower()
            css_flags = ["display:none", "visibility:hidden", "opacity:0"]
            if any(flag in st_lower for flag in css_flags):
                suspicious = True
                reason_parts.append("hidden via CSS")

            tiny_pattern = re.compile(r"(width|height)\s*:\s*([0-9]+)px", re.IGNORECASE)
            for m in tiny_pattern.finditer(style):
                dim_val = m.group(2)
                if small(dim_val):
                    suspicious = True
                    reason_parts.append(f"{m.group(1)}={dim_val}px")

            src = img.get("src", "") or ""
            if suspicious:
                if "http" in src:
                    severity = "suspicious"
                desc = "Potential tracking pixel or hidden image: " + ", ".join(reason_parts)
                snippet = str(img)[:300]
                findings.append(
                    Finding(
                        ftype="Tracking pixel",
                        description=desc,
                        snippet=snippet,
                        location="N/A",
                        source_type=source_type,
                        source_id=source_id,
                        severity=severity,
                    )
                )
        return findings

    # ---------- hidden element detector ----------

    def _scan_hidden_elements(self, soup, source_type, source_id):
        findings = []
        css_hidden_flags = ["display:none", "visibility:hidden", "opacity:0"]
        offscreen_patterns = [
            re.compile(r"text-indent\s*:\s*-?\s*1000px", re.IGNORECASE),
            re.compile(r"left\s*:\s*-?\s*1000px", re.IGNORECASE),
            re.compile(r"top\s*:\s*-?\s*1000px", re.IGNORECASE),
        ]

        for el in soup.find_all(True):
            style = (el.get("style", "") or "").lower()
            hidden = False
            reasons = []

            if any(flag in style for flag in css_hidden_flags):
                hidden = True
                reasons.append("display/visibility/opacity hidden")

            for pat in offscreen_patterns:
                if pat.search(style):
                    hidden = True
                    reasons.append("off-screen positioning")

            if hidden:
                desc = "Hidden element via CSS: " + ", ".join(reasons)
                snippet = str(el)[:300]
                findings.append(
                    Finding(
                        ftype="Hidden element",
                        description=desc,
                        snippet=snippet,
                        location="N/A",
                        source_type=source_type,
                        source_id=source_id,
                        severity="info",
                    )
                )

        return findings

    # ---------- link / redirect detector ----------

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
            url_in_text = self._extract_domain(text) if text else None
            if url_in_text and domain and url_in_text.lower() != domain.lower():
                desc_parts.append(f"Visible text domain '{url_in_text}' != href domain '{domain}'")
                severity = "suspicious"

            if domain and domain.lower() in self.redirector_domains:
                desc_parts.append(f"Known redirector domain '{domain}'")
                severity = "suspicious"

            if "?" in href and len(href) > 120:
                desc_parts.append("Long URL with query parameters (possible tracking or unique ID)")
                if severity == "info":
                    severity = "suspicious"

            if desc_parts:
                desc = "Suspicious/obfuscated link: " + "; ".join(desc_parts)
                snippet = str(a)[:300]
                findings.append(
                    Finding(
                        ftype="Obfuscated link",
                        description=desc,
                        snippet=snippet,
                        location="N/A",
                        source_type=source_type,
                        source_id=source_id,
                        severity=severity,
                    )
                )

        return findings

    # ---------- iframe / script behavior detector ----------

    def _scan_iframes_and_scripts(self, soup, source_type, source_id):
        findings = []

        for iframe in soup.find_all("iframe"):
            src = (iframe.get("src") or "").strip()
            style = (iframe.get("style") or "").lower()
            hidden = False
            reasons = []
            css_flags = ["display:none", "visibility:hidden", "opacity:0"]
            if any(flag in style for flag in css_flags):
                hidden = True
                reasons.append("hidden iframe via CSS")

            domain = self._extract_domain(src) or "unknown"
            if hidden:
                desc = f"Hidden iframe loading from domain '{domain}': " + ", ".join(reasons)
                snippet = str(iframe)[:300]
                severity = "suspicious"
                findings.append(
                    Finding(
                        ftype="Hidden iframe",
                        description=desc,
                        snippet=snippet,
                        location="N/A",
                        source_type=source_type,
                        source_id=source_id,
                        severity=severity,
                    )
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
                    Finding(
                        ftype="Suspicious script",
                        description=desc,
                        snippet=snippet,
                        location="N/A",
                        source_type=source_type,
                        source_id=source_id,
                        severity="suspicious",
                    )
                )

        return findings

    # ---------- auto-form behavior detector ----------

    def _scan_auto_forms(self, soup, source_type, source_id):
        findings = []

        for form in soup.find_all("form"):
            form_str = str(form)
            auto_submit = False
            if "onsubmit" in form.attrs:
                val = str(form.get("onsubmit") or "").lower()
                if "this.submit" in val:
                    auto_submit = True

            has_script = False
            for script in form.find_all("script"):
                stext = script.get_text() or ""
                if "submit()" in stext:
                    has_script = True

            if auto_submit or has_script:
                action = form.get("action", "")
                domain = self._extract_domain(action) or "unknown"
                desc = f"Form with potential auto-submit behavior targeting '{domain}'"
                snippet = form_str[:300]
                findings.append(
                    Finding(
                        ftype="Auto-submit form",
                        description=desc,
                        snippet=snippet,
                        location="N/A",
                        source_type=source_type,
                        source_id=source_id,
                        severity="suspicious",
                    )
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
# EVENT BUS (for daemon -> GUI communication)
# ============================================================

class EventBus:
    def __init__(self):
        self.queue = queue.Queue()

    def publish(self, finding):
        self.queue.put(finding)

    def get_all(self):
        findings = []
        try:
            while True:
                findings.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return findings

EVENT_BUS = EventBus()

# ============================================================
# FLASK BACKGROUND SERVER (LOCAL API)
# ============================================================

flask_app = Flask(__name__)
detector = HiddenDataDetector()

@flask_app.route("/scan_html", methods=["POST"])
def scan_html_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    html = data.get("html") or ""
    source_id = data.get("source_id") or "web_page"
    if not html:
        return jsonify({"error": "missing html"}), 400

    findings = detector.scan_html(html, source_type="web", source_id=source_id)
    for f in findings:
        EVENT_BUS.publish(f)

    return jsonify({
        "status": "ok",
        "findings": [f.__dict__ for f in findings],
    })

@flask_app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "alive"})

def run_flask_server():
    flask_app.run(host="127.0.0.1", port=5005, debug=False, use_reloader=False)

# ============================================================
# IMAP EMAIL WATCHER
# ============================================================

class IMAPWatcher(threading.Thread):
    def __init__(self, host, username, password, folder="INBOX",
                 interval=30, use_ssl=True, enabled=False):
        super().__init__(daemon=True)
        self.host = host
        self.username = username
        self.password = password
        self.folder = folder
        self.interval = interval
        self.use_ssl = use_ssl
        self.enabled = enabled
        self._stop_flag = threading.Event()
        self._last_uid_seen = None

    def stop(self):
        self._stop_flag.set()

    def run(self):
        if not self.enabled:
            return
        while not self._stop_flag.is_set():
            try:
                self.check_mail()
            except Exception:
                traceback.print_exc()
            time.sleep(self.interval)

    def check_mail(self):
        if self.use_ssl:
            M = imaplib.IMAP4_SSL(self.host)
        else:
            M = imaplib.IMAP4(self.host)

        M.login(self.username, self.password)
        M.select(self.folder)

        typ, data = M.uid("search", None, "ALL")
        if typ != "OK":
            M.logout()
            return

        uids = data[0].split()
        new_uids = []
        if self._last_uid_seen is None:
            if uids:
                self._last_uid_seen = uids[-1]
        else:
            for uid in uids:
                if int(uid) > int(self._last_uid_seen):
                    new_uids.append(uid)
            if uids:
                self._last_uid_seen = uids[-1]

        for uid in new_uids:
            typ, msg_data = M.uid("fetch", uid, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw_msg = msg_data[0][1]
            msg = message_from_bytes(raw_msg)
            subject = msg.get("Subject", "(no subject)")
            from_addr = msg.get("From", "(unknown)")
            source_id = f"Email from {from_addr} - {subject}"
            email_findings = detector.scan_eml(raw_msg, source_id=source_id)
            for f in email_findings:
                EVENT_BUS.publish(f)

        M.close()
        M.logout()

# ============================================================
# GUI APPLICATION
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, imap_watcher=None):
        super().__init__()
        self.imap_watcher = imap_watcher
        self._findings_log = []
        self.init_ui()
        self.start_event_timer()

    def init_ui(self):
        self.setWindowTitle("Hidden Data & Behavior Inspector (Autonomous)")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.info_label = QtWidgets.QLabel(
            "Listening on http://127.0.0.1:5005/scan_html for web pages. "
            "IMAP watcher: configured = "
            + ("yes" if self.imap_watcher and self.imap_watcher.enabled else "no")
        )
        layout.addWidget(self.info_label)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Time", "Severity", "Source type", "Source id", "Type", "Description"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        self.snippet_view = QtWidgets.QPlainTextEdit()
        self.snippet_view.setReadOnly(True)
        self.snippet_view.setPlaceholderText("Select a row to view raw snippet / element...")
        layout.addWidget(self.snippet_view)

        btn_layout = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear log")
        self.copy_btn = QtWidgets.QPushButton("Copy snippet")
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.copy_btn)
        layout.addLayout(btn_layout)

        self.setCentralWidget(central)

        self.table.itemSelectionChanged.connect(self.update_snippet_view)
        self.clear_btn.clicked.connect(self.clear_log)
        self.copy_btn.clicked.connect(self.copy_snippet)

        self.tray_icon = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("Hidden Data & Behavior Inspector")
        self.tray_icon.setVisible(True)

    def start_event_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.consume_events)
        self.timer.start(1000)

    def consume_events(self):
        new_findings = EVENT_BUS.get_all()
        if not new_findings:
            return

        for f in new_findings:
            self._findings_log.append(f)
            self.add_row(f)
            if f.severity in ("suspicious", "critical"):
                self.show_popup(f)

    def add_row(self, finding):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, val in enumerate(finding.to_row()):
            item = QtWidgets.QTableWidgetItem(str(val))
            if finding.severity == "critical":
                item.setBackground(QtGui.QColor("#ffcccc"))
            elif finding.severity == "suspicious":
                item.setBackground(QtGui.QColor("#fff2cc"))
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()

    def show_popup(self, finding):
        title = "Hidden data detected" if finding.severity == "suspicious" else "Potentially dangerous behavior detected"
        msg = (
            f"{title}\n\n"
            f"Source: {finding.source_id}\n"
            f"Type: {finding.ftype}\n"
            f"Details: {finding.description}"
        )
        self.tray_icon.showMessage(
            title, msg, QtWidgets.QSystemTrayIcon.Warning, 10000
        )

    def update_snippet_view(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._findings_log):
            f = self._findings_log[idx]
            self.snippet_view.setPlainText(f.snippet)

    def clear_log(self):
        self.table.setRowCount(0)
        self._findings_log = []
        self.snippet_view.clear()

    def copy_snippet(self):
        text = self.snippet_view.toPlainText()
        if not text:
            return
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(text)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    # Optional: IMAP watcher configuration
    # To enable, fill in your IMAP details and set enabled=True
    imap_config = {
        "host": "imap.example.com",
        "username": "user@example.com",
        "password": "password_here",
        "folder": "INBOX",
        "interval": 60,
        "use_ssl": True,
        "enabled": False,  # set to True after configuring
    }

    imap_watcher = IMAPWatcher(
        host=imap_config["host"],
        username=imap_config["username"],
        password=imap_config["password"],
        folder=imap_config["folder"],
        interval=imap_config["interval"],
        use_ssl=imap_config["use_ssl"],
        enabled=imap_config["enabled"],
    )
    imap_watcher.start()

    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(imap_watcher=imap_watcher)
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

