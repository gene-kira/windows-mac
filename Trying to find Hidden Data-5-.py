import sys
import os
import subprocess
import importlib
import re
import threading
import time
from datetime import datetime

# ============================================================
# CONFIG: FOLDERS TO WATCH
# ============================================================

# Edit this list to point to your real folders.
# Example on Windows: r"C:\Users\YourName\Downloads"
WATCH_FOLDERS = [
    # r"C:\Users\YourName\Downloads",
    # r"C:\Users\YourName\SavedEmails",
]

WATCH_INTERVAL_SECONDS = 5  # how often to poll folders

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
    def __init__(self, ftype, description, snippet, source_type, source_id,
                 severity="info", category="generic", extras=None):
        self.ftype = ftype
        self.description = description
        self.snippet = snippet
        self.source_type = source_type      # "web", "email", "raw"
        self.source_id = source_id          # filename, path, label
        self.severity = severity            # "info", "suspicious", "critical"
        self.category = category            # "tracking", "hidden", "link", "script", "attachment", ...
        self.timestamp = datetime.utcnow().isoformat(timespec="seconds")
        self.extras = extras or {}

    def to_row(self):
        return [
            self.timestamp,
            self.severity,
            self.category,
            self.source_type,
            self.source_id,
            self.ftype,
            self.description,
        ]

# ============================================================
# DETECTOR (SCANNER CORE)
# ============================================================

class HiddenDataDetector:
    def __init__(self):
        self.redirector_domains = {
            "bit.ly", "t.co", "lnkd.in", "tinyurl.com", "goo.gl",
        }
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
        self.hidden_class_markers = {
            "hidden", "sr-only", "visually-hidden", "offscreen"
        }
        self.tracking_param_keywords = {
            "uid","id","user","token","track","click","cid","rid",
            "campaign","utm_","session","ref","fingerprint",
        }

    # -------- public entry points --------

    def scan_html(self, html_text, source_type="web", source_id="N/A"):
        soup = BeautifulSoup(html_text, "lxml")
        findings = []
        findings.extend(self._scan_meta_refresh(soup, source_type, source_id))
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

        # Attachments
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
                                category="attachment",
                                extras={"filename": filename},
                            )
                        )

        # HTML parts
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

    # -------- detectors --------

    def _scan_meta_refresh(self, soup, source_type, source_id):
        findings = []
        for meta in soup.find_all("meta"):
            http_equiv = (meta.get("http-equiv") or "").lower()
            content = meta.get("content") or ""
            if http_equiv == "refresh" and "url=" in content.lower():
                desc = f"Meta refresh redirect: {content}"
                snippet = str(meta)[:300]
                findings.append(
                    Finding(
                        ftype="Meta refresh",
                        description=desc,
                        snippet=snippet,
                        source_type=source_type,
                        source_id=source_id,
                        severity="suspicious",
                        category="redirect",
                        extras={"content": content},
                    )
                )
        return findings

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
                domain = self._extract_domain(src) or "unknown"
                desc = "Potential tracking pixel or hidden image: " + ", ".join(reason)
                snippet = str(img)[:300]
                findings.append(
                    Finding(
                        ftype="Tracking pixel",
                        description=desc,
                        snippet=snippet,
                        source_type=source_type,
                        source_id=source_id,
                        severity=severity,
                        category="tracking",
                        extras={"src": src, "domain": domain},
                    )
                )
        return findings

    def _scan_hidden_elements(self, soup, source_type, source_id):
        findings = []
        css_flags = ["display:none", "visibility:hidden", "opacity:0"]
        offscreen_patterns = [
            re.compile(r"text-indent\s*:\s*-?\s*1000px", re.IGNORECASE),
            re.compile(r"left\s*:\s*-?\s*1000px", re.IGNORECASE),
            re.compile(r"top\s*:\s*-?\s*1000px", re.IGNORECASE),
        ]

        for el in soup.find_all(True):
            style = (el.get("style", "") or "").lower()
            classes = set((el.get("class") or []))
            aria_hidden = (el.get("aria-hidden") or "").lower() == "true"

            hidden = False
            reasons = []

            if any(flag in style for flag in css_flags):
                hidden = True
                reasons.append("CSS hidden")
            for pat in offscreen_patterns:
                if pat.search(style):
                    hidden = True
                    reasons.append("off-screen positioning")
            if aria_hidden:
                hidden = True
                reasons.append("aria-hidden=true")
            if classes & self.hidden_class_markers:
                hidden = True
                reasons.append("hidden-related CSS class")

            if hidden:
                desc = "Hidden element: " + ", ".join(reasons)
                snippet = str(el)[:300]
                findings.append(
                    Finding(
                        ftype="Hidden element",
                        description=desc,
                        snippet=snippet,
                        source_type=source_type,
                        source_id=source_id,
                        severity="info",
                        category="hidden",
                        extras={"tag": el.name, "classes": list(classes)},
                    )
                )
        return findings

    def _scan_links(self, soup, source_type, source_id):
        findings = []
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            text = (a.get_text() or "").strip()

            severity = "info"
            reasons = []
            domain = self._extract_domain(href)
            text_domain = self._extract_domain(text) if text else None

            if text_domain and domain and text_domain.lower() != domain.lower():
                reasons.append(f"text domain '{text_domain}' != href domain '{domain}'")
                severity = "suspicious"

            if domain and domain.lower() in self.redirector_domains:
                reasons.append(f"known redirector '{domain}'")
                severity = "suspicious"

            tracking_keys, long_query, base64ish = self._analyze_query(href)
            if tracking_keys:
                reasons.append("tracking parameters: " + ", ".join(sorted(tracking_keys)))
                if severity == "info":
                    severity = "suspicious"
            if long_query:
                reasons.append("long query string")
                if severity == "info":
                    severity = "suspicious"
            if base64ish:
                reasons.append("base64-like value in query")
                if severity == "info":
                    severity = "suspicious"

            if reasons:
                desc = "Suspicious/obfuscated link: " + "; ".join(reasons)
                snippet = str(a)[:300]
                findings.append(
                    Finding(
                        ftype="Obfuscated link",
                        description=desc,
                        snippet=snippet,
                        source_type=source_type,
                        source_id=source_id,
                        severity=severity,
                        category="link",
                        extras={"href": href, "domain": domain},
                    )
                )
        return findings

    def _scan_iframes_scripts(self, soup, source_type, source_id):
        findings = []

        for iframe in soup.find_all("iframe"):
            src = (iframe.get("src") or "").strip()
            style = (iframe.get("style") or "").lower()
            width = iframe.get("width")
            height = iframe.get("height")

            hidden = False
            reasons = []

            if any(flag in style for flag in ["display:none", "visibility:hidden", "opacity:0"]):
                hidden = True
                reasons.append("CSS hidden")
            offscreen_patterns = [
                re.compile(r"left\s*:\s*-?\s*1000px", re.IGNORECASE),
                re.compile(r"top\s*:\s*-?\s*1000px", re.IGNORECASE),
            ]
            for pat in offscreen_patterns:
                if pat.search(style):
                    hidden = True
                    reasons.append("off-screen positioning")

            def small(v):
                try:
                    return int(v) <= 2
                except Exception:
                    return False
            if width and small(width):
                hidden = True
                reasons.append(f"width={width}")
            if height and small(height):
                hidden = True
                reasons.append(f"height={height}")

            if hidden:
                domain = self._extract_domain(src) or "unknown"
                desc = f"Hidden iframe loading from '{domain}': " + ", ".join(reasons)
                snippet = str(iframe)[:300]
                findings.append(
                    Finding(
                        ftype="Hidden iframe",
                        description=desc,
                        snippet=snippet,
                        source_type=source_type,
                        source_id=source_id,
                        severity="suspicious",
                        category="hidden",
                        extras={"src": src, "domain": domain},
                    )
                )

        for script in soup.find_all("script"):
            src = (script.get("src") or "").strip()
            if src:
                domain = self._extract_domain(src) or "unknown"
                reasons = []
                severity = "info"

                if any(d in domain for d in self.suspicious_script_domains):
                    reasons.append(f"script from tracking/suspicious domain '{domain}'")
                    severity = "suspicious"

                _keys, long_query, base64ish = self._analyze_query(src)
                if long_query:
                    reasons.append("long query string in script src")
                    if severity == "info":
                        severity = "suspicious"
                if base64ish:
                    reasons.append("base64-like value in script src")
                    if severity == "info":
                        severity = "suspicious"

                if reasons:
                    desc = "; ".join(reasons)
                    snippet = str(script)[:300]
                    findings.append(
                        Finding(
                            ftype="Suspicious script",
                            description=desc,
                            snippet=snippet,
                            source_type=source_type,
                            source_id=source_id,
                            severity=severity,
                            category="script",
                            extras={"src": src, "domain": domain},
                        )
                    )
            else:
                text = (script.get_text() or "").strip()
                if not text:
                    continue
                patterns = [
                    "new WebSocket(",
                    "navigator.getBattery",
                    "canvas.toDataURL",
                    "Fingerprint",
                    "eval(",
                    "Function('return this')",
                    "atob(",
                ]
                hits = [p for p in patterns if p in text]
                if hits:
                    desc = "Inline script with advanced/dynamic behavior: " + ", ".join(hits)
                    snippet = text[:300]
                    findings.append(
                        Finding(
                            ftype="Advanced inline script",
                            description=desc,
                            snippet=snippet,
                            source_type=source_type,
                            source_id=source_id,
                            severity="suspicious",
                            category="script",
                            extras={"patterns": hits},
                        )
                    )

        return findings

    def _scan_auto_forms(self, soup, source_type, source_id):
        findings = []
        for form in soup.find_all("form"):
            form_str = str(form)
            auto = False
            reasons = []

            if "onsubmit" in form.attrs:
                val = str(form.get("onsubmit") or "").lower()
                if "submit" in val:
                    auto = True
                    reasons.append("onsubmit handler")
            for script in form.find_all("script"):
                txt = script.get_text() or ""
                if "submit()" in txt:
                    auto = True
                    reasons.append("inline script calling submit()")

            if auto:
                action = form.get("action", "")
                domain = self._extract_domain(action) or "unknown"
                desc = f"Form with potential auto-submit behavior targeting '{domain}': " + ", ".join(reasons)
                snippet = form_str[:300]
                findings.append(
                    Finding(
                        ftype="Auto-submit form",
                        description=desc,
                        snippet=snippet,
                        source_type=source_type,
                        source_id=source_id,
                        severity="suspicious",
                        category="form",
                        extras={"action": action, "domain": domain},
                    )
                )

            hidden_inputs = form.find_all("input", {"type": "hidden"})
            if len(hidden_inputs) >= 3:
                tracking_keys_detected = set()
                for inp in hidden_inputs:
                    name = (inp.get("name") or "").lower()
                    if not name:
                        continue
                    for k in self.tracking_param_keywords:
                        if k in name:
                            tracking_keys_detected.add(name)
                if tracking_keys_detected:
                    action = form.get("action", "")
                    domain = self._extract_domain(action) or "unknown"
                    desc = f"Form with multiple hidden tracking fields targeting '{domain}'"
                    snippet = form_str[:300]
                    findings.append(
                        Finding(
                            ftype="Tracking form",
                            description=desc,
                            snippet=snippet,
                            source_type=source_type,
                            source_id=source_id,
                            severity="suspicious",
                            category="form",
                            extras={
                                "action": action,
                                "domain": domain,
                                "hidden_fields": list(tracking_keys_detected),
                            },
                        )
                    )

        return findings

    # -------- helpers --------

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

    def _analyze_query(self, url):
        if "?" not in url:
            return set(), False, False
        qs = url.split("?", 1)[1]
        long_query = len(qs) > 120

        tracking_keys = set()
        base64ish = False
        for pair in qs.split("&"):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k_lower = k.lower()
            for tk in self.tracking_param_keywords:
                if tk in k_lower:
                    tracking_keys.add(k)
            v_clean = re.sub(r"%[0-9a-fA-F]{2}", "", v)
            if len(v_clean) > 30 and re.fullmatch(r"[A-Za-z0-9+/=]+", v_clean or ""):
                base64ish = True
        return tracking_keys, long_query, base64ish

# ============================================================
# BACKGROUND FOLDER WATCHER (AUTONOMOUS FILE SCAN)
# ============================================================

class FolderWatcher(QtCore.QObject):
    newFindings = QtCore.pyqtSignal(list)  # emits list[Finding]

    def __init__(self, detector, folders, interval_seconds=5, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.folders = [f for f in folders if f and os.path.isdir(f)]
        self.interval = interval_seconds
        self._seen_files = {}  # path -> mtime
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.poll)
        if self.folders:
            self._timer.start(self.interval * 1000)

    def poll(self):
        findings_accum = []
        for folder in self.folders:
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if not os.path.isfile(path):
                    continue
                ext = os.path.splitext(path)[1].lower()
                if ext not in (".html", ".htm", ".eml"):
                    continue
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                last_mtime = self._seen_files.get(path)
                if last_mtime is not None and mtime <= last_mtime:
                    continue  # already seen
                self._seen_files[path] = mtime

                # New or changed file: scan it
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    if ext == ".eml":
                        source_type = "email"
                        findings = self.detector.scan_eml(data, source_id=path)
                    else:
                        try:
                            text = data.decode("utf-8", errors="ignore")
                        except Exception:
                            text = data.decode("latin1", errors="ignore")
                        source_type = "web"
                        findings = self.detector.scan_html(text, source_type=source_type, source_id=path)
                    if findings:
                        findings_accum.extend(findings)
                except Exception as e:
                    print(f"[Watcher] Error scanning file {path}: {e}")

        if findings_accum:
            self.newFindings.emit(findings_accum)

# ============================================================
# GUI
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, detector, watcher):
        super().__init__()
        self.detector = detector
        self.watcher = watcher
        self.current_source = ""
        self.current_source_type = "raw"
        self.current_source_id = "N/A"
        self.findings = []
        self.filtered_indices = []
        self.init_ui()

        if self.watcher:
            self.watcher.newFindings.connect(self.on_new_findings)

    def init_ui(self):
        self.setWindowTitle("Hidden Data & Behavior Inspector (Watcher Edition)")
        self.resize(1200, 750)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top row: buttons and filters
        top_row = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load HTML/EML")
        self.btn_paste = QtWidgets.QPushButton("Paste Source")
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_clear = QtWidgets.QPushButton("Clear Results")
        top_row.addWidget(self.btn_load)
        top_row.addWidget(self.btn_paste)
        top_row.addWidget(self.btn_scan)
        top_row.addWidget(self.btn_clear)

        self.severity_filter = QtWidgets.QComboBox()
        self.severity_filter.addItems(["All severities", "info", "suspicious", "critical"])
        self.source_filter = QtWidgets.QComboBox()
        self.source_filter.addItems(["All source types", "web", "email", "raw"])
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Search in description/type/source...")

        top_row.addWidget(QtWidgets.QLabel("Severity:"))
        top_row.addWidget(self.severity_filter)
        top_row.addWidget(QtWidgets.QLabel("Source:"))
        top_row.addWidget(self.source_filter)
        top_row.addWidget(self.search_box)

        main_layout.addLayout(top_row)

        # Summary label
        watched_info = ", ".join(WATCH_FOLDERS) if WATCH_FOLDERS else "No folders configured"
        self.summary_label = QtWidgets.QLabel(
            f"Watching folders (HTML/EML): {watched_info}"
        )
        main_layout.addWidget(self.summary_label)

        # Splitter: table + details/snippet
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        upper_widget = QtWidgets.QWidget()
        upper_layout = QtWidgets.QVBoxLayout(upper_widget)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Severity", "Category", "Source type", "Source id", "Type", "Description"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        upper_layout.addWidget(self.table)
        splitter.addWidget(upper_widget)

        lower_widget = QtWidgets.QWidget()
        lower_layout = QtWidgets.QVBoxLayout(lower_widget)

        self.detail_label = QtWidgets.QLabel("Details:")
        lower_layout.addWidget(self.detail_label)

        self.snippet_view = QtWidgets.QPlainTextEdit()
        self.snippet_view.setReadOnly(True)
        self.snippet_view.setPlaceholderText("Select a row above to see the raw snippet and details...")
        lower_layout.addWidget(self.snippet_view)

        splitter.addWidget(lower_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter)
        self.setCentralWidget(central)

        # Connect signals
        self.btn_load.clicked.connect(self.load_file)
        self.btn_paste.clicked.connect(self.paste_source)
        self.btn_scan.clicked.connect(self.scan_source)
        self.btn_clear.clicked.connect(self.clear_results)
        self.table.itemSelectionChanged.connect(self.update_details)
        self.severity_filter.currentIndexChanged.connect(self.apply_filters)
        self.source_filter.currentIndexChanged.connect(self.apply_filters)
        self.search_box.textChanged.connect(self.apply_filters)

        # Tray icon
        self.tray = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxWarning)
        self.tray.setIcon(icon)
        self.tray.setVisible(True)

    # ---- manual scan actions ----

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

        self.current_source_id = path
        self.summary_label.setText(
            f"Loaded {self.current_source_type} source from file: {self.current_source_id}"
        )

    def paste_source(self):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Paste Source", "Paste raw HTML or email source:"
        )
        if ok and text:
            self.current_source = text
            self.current_source_type = "raw"
            self.current_source_id = "Pasted"
            self.summary_label.setText("Loaded pasted source.")

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

        if findings:
            self.add_findings(findings, show_popup=True)
        else:
            QtWidgets.QMessageBox.information(self, "Result", "No issues detected with current rules.")

    def clear_results(self):
        self.findings = []
        self.filtered_indices = []
        self.table.setRowCount(0)
        self.snippet_view.clear()
        self.detail_label.setText("Details:")

    # ---- autonomous watcher hook ----

    @QtCore.pyqtSlot(list)
    def on_new_findings(self, new_findings):
        self.add_findings(new_findings, show_popup=True, prefix="(Watcher) ")

    # ---- adding & filtering ----

    def add_findings(self, new_findings, show_popup=False, prefix=""):
        self.findings.extend(new_findings)
        self.apply_filters()

        # Summary & popup
        counts = {"info": 0, "suspicious": 0, "critical": 0}
        for f in new_findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        total = len(new_findings)

        if total:
            msg = (
                f"{prefix}New findings: {total} "
                f"(info: {counts['info']}, suspicious: {counts['suspicious']}, critical: {counts['critical']})"
            )
            self.summary_label.setText(msg)
            if show_popup and (counts["suspicious"] or counts["critical"]):
                self.tray.showMessage(
                    "Hidden data detected",
                    msg,
                    QtWidgets.QSystemTrayIcon.Warning,
                    10000,
                )

    def apply_filters(self):
        self.table.setRowCount(0)
        self.filtered_indices = []

        sev_filter = self.severity_filter.currentText()
        if sev_filter == "All severities":
            sev_filter = None
        src_filter = self.source_filter.currentText()
        if src_filter == "All source types":
            src_filter = None
        search_term = self.search_box.text().strip().lower() or None

        for idx, f in enumerate(self.findings):
            if sev_filter and f.severity != sev_filter:
                continue
            if src_filter and f.source_type != src_filter:
                continue

            row_text = " ".join([
                f.severity, f.category, f.source_type, f.source_id, f.ftype, f.description
            ]).lower()
            if search_term and search_term not in row_text:
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.filtered_indices.append(idx)

            for col, val in enumerate(f.to_row()):
                item = QtWidgets.QTableWidgetItem(str(val))
                if f.severity == "critical":
                    item.setBackground(QtGui.QColor("#ffcccc"))
                elif f.severity == "suspicious":
                    item.setBackground(QtGui.QColor("#fff2cc"))
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()

    # ---- details pane ----

    def update_details(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self.filtered_indices:
            return
        table_row = rows[0].row()
        if table_row < 0 or table_row >= len(self.filtered_indices):
            return
        f_idx = self.filtered_indices[table_row]
        f = self.findings[f_idx]

        extras_str = ", ".join(f"{k}={v}" for k, v in f.extras.items()) if f.extras else "None"
        detail_text = (
            f"Type: {f.ftype}\n"
            f"Category: {f.category}\n"
            f"Severity: {f.severity}\n"
            f"Source type: {f.source_type}\n"
            f"Source id: {f.source_id}\n"
            f"Timestamp: {f.timestamp}\n"
            f"Extras: {extras_str}"
        )
        self.detail_label.setText(detail_text)
        self.snippet_view.setPlainText(f.snippet)

# ============================================================
# MAIN
# ============================================================

def main():
    detector = HiddenDataDetector()

    app = QtWidgets.QApplication(sys.argv)

    if WATCH_FOLDERS:
        watcher = FolderWatcher(detector, WATCH_FOLDERS, interval_seconds=WATCH_INTERVAL_SECONDS)
    else:
        watcher = None

    win = MainWindow(detector, watcher)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

