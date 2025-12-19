import sys
import os
import subprocess
import importlib
import re

# -----------------------------
# Auto-loader for dependencies
# -----------------------------

REQUIRED_PACKAGES = [
    ("PyQt5", "PyQt5"),
    ("beautifulsoup4", "bs4"),
    ("lxml", None),
]

def ensure_package(pkg_name, import_name=None):
    """
    Ensure a package is installed. If not, attempt to install via pip.
    pkg_name: name used with pip
    import_name: name used for import (defaults to pkg_name)
    """
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
            # Let the import below raise a clear error
        try:
            importlib.import_module(import_name)
            print(f"[AUTOLOADER] Successfully installed and imported '{import_name}'.")
        except ImportError as e:
            print(f"[AUTOLOADER] ERROR: Could not import '{import_name}' even after installation.")
            raise

def ensure_all_packages():
    for pkg, import_name in REQUIRED_PACKAGES:
        ensure_package(pkg, import_name)

ensure_all_packages()

# Now safe to import third-party libs
from PyQt5 import QtWidgets, QtGui, QtCore
from bs4 import BeautifulSoup
from email import message_from_string

# -----------------------------
# Core data structures
# -----------------------------

class Finding:
    def __init__(self, ftype, description, snippet, location, source_type, source_id):
        self.ftype = ftype               # e.g. "Tracking pixel"
        self.description = description   # Human-readable reason
        self.snippet = snippet           # HTML snippet
        self.location = location         # e.g. "N/A" or "line:col"
        self.source_type = source_type   # "web" or "email" or "raw"
        self.source_id = source_id       # URL, subject, filename, etc.

# -----------------------------
# Hidden data detector
# -----------------------------

class HiddenDataDetector:
    def __init__(self):
        # Known redirector domains (simple example list)
        self.redirector_domains = {
            "bit.ly",
            "t.co",
            "lnkd.in",
            "goo.gl",
            "tinyurl.com",
        }

    def scan_html(self, html_text, source_type="raw", source_id="N/A"):
        soup = BeautifulSoup(html_text, "lxml")
        findings = []
        findings.extend(self._scan_images(soup, source_type, source_id))
        findings.extend(self._scan_hidden_elements(soup, source_type, source_id))
        findings.extend(self._scan_links(soup, source_type, source_id))
        return findings

    def scan_eml(self, eml_text, source_id="Email"):
        """
        Parse .eml, extract HTML part (if any), then run HTML scan.
        """
        msg = message_from_string(eml_text)
        html_parts = []

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/html":
                    try:
                        html_parts.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore"))
                    except Exception:
                        pass
        else:
            if msg.get_content_type() == "text/html":
                try:
                    html_parts.append(msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore"))
                except Exception:
                    pass

        findings = []
        for idx, html in enumerate(html_parts):
            part_id = f"{source_id} (part {idx+1})"
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

            # Size checks
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

            # CSS style-based checks
            st_lower = style.lower()
            css_flags = ["display:none", "visibility:hidden", "opacity:0"]
            if any(flag in st_lower for flag in css_flags):
                suspicious = True
                reason_parts.append("hidden via CSS")

            # Tiny inline styles (width/height in style)
            tiny_pattern = re.compile(r"(width|height)\s*:\s*([0-9]+)px", re.IGNORECASE)
            for m in tiny_pattern.finditer(style):
                dim_val = m.group(2)
                if small(dim_val):
                    suspicious = True
                    reason_parts.append(f"{m.group(1)}={dim_val}px")

            if suspicious:
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

            # Domain extraction
            domain = self._extract_domain(href)

            # Visible text looks like a URL
            url_in_text = self._extract_domain(text) if text else None
            if url_in_text and domain and url_in_text.lower() != domain.lower():
                desc_parts.append(f"Visible text domain '{url_in_text}' != href domain '{domain}'")

            # Known redirector domain
            if domain and domain.lower() in self.redirector_domains:
                desc_parts.append(f"Known redirector domain '{domain}'")

            # Very long query string as possible tracking
            if "?" in href and len(href) > 120:
                desc_parts.append("Long URL with query parameters (possible tracking ID)")

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
                    )
                )

        return findings

    @staticmethod
    def _extract_domain(url):
        """
        Very simple domain extractor (no external deps).
        """
        if not url:
            return None
        # Remove protocol
        tmp = re.sub(r"^[a-zA-Z]+://", "", url)
        # Split by / and ?
        tmp = tmp.split("/", 1)[0]
        tmp = tmp.split("?", 1)[0]
        # Remove credentials
        tmp = tmp.split("@")[-1]
        # Remove port
        tmp = tmp.split(":", 1)[0]
        tmp = tmp.strip()
        return tmp or None

# -----------------------------
# GUI application
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = HiddenDataDetector()
        self.current_source = ""
        self.current_source_type = "raw"
        self.current_source_id = "N/A"
        self._findings = []

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hidden Data Inspector (Autoloader Enabled)")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # --- Top controls ---
        btn_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load HTML/EML")
        self.paste_btn = QtWidgets.QPushButton("Paste Source")
        self.scan_btn = QtWidgets.QPushButton("Scan")
        self.clear_btn = QtWidgets.QPushButton("Clear Results")

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.paste_btn)
        btn_layout.addWidget(self.scan_btn)
        btn_layout.addWidget(self.clear_btn)

        layout.addLayout(btn_layout)

        # Info label
        self.info_label = QtWidgets.QLabel("No source loaded.")
        layout.addWidget(self.info_label)

        # --- Splitter: table + snippet ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Type",
            "Description",
            "Source type",
            "Source id",
            "Snippet"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.snippet_view = QtWidgets.QPlainTextEdit()
        self.snippet_view.setReadOnly(True)
        self.snippet_view.setPlaceholderText("Select a finding to view raw HTML snippet...")

        splitter.addWidget(self.table)
        splitter.addWidget(self.snippet_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        self.setCentralWidget(central)

        # --- Connections ---
        self.load_btn.clicked.connect(self.load_file)
        self.paste_btn.clicked.connect(self.paste_source)
        self.scan_btn.clicked.connect(self.scan_content)
        self.clear_btn.clicked.connect(self.clear_results)
        self.table.itemSelectionChanged.connect(self.update_snippet_view)

    # ---------- actions ----------

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
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                self.current_source = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read file:\n{e}")
            return

        ext = os.path.splitext(path)[1].lower()
        if ext == ".eml":
            self.current_source_type = "email"
        else:
            self.current_source_type = "web"
        self.current_source_id = os.path.basename(path)

        self.info_label.setText(f"Loaded {self.current_source_type} source from file: {self.current_source_id}")
        QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded file: {path}")

    def paste_source(self):
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Paste Source", "Paste raw HTML or email source:"
        )
        if ok and text:
            self.current_source = text
            self.current_source_type = "raw"
            self.current_source_id = "Pasted content"
            self.info_label.setText("Loaded pasted source.")

    def scan_content(self):
        if not self.current_source:
            QtWidgets.QMessageBox.warning(self, "No content", "Load or paste content first.")
            return

        if self.current_source_type == "email":
            findings = self.detector.scan_eml(self.current_source, source_id=self.current_source_id)
        else:
            findings = self.detector.scan_html(
                self.current_source,
                source_type=self.current_source_type,
                source_id=self.current_source_id,
            )

        self.populate_table(findings)

        if findings:
            QtWidgets.QMessageBox.information(
                self,
                "Hidden data detected",
                f"Found {len(findings)} potential hidden items."
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Clean",
                "No hidden items detected with current rules."
            )

    def clear_results(self):
        self.table.setRowCount(0)
        self._findings = []
        self.snippet_view.clear()

    def populate_table(self, findings):
        self._findings = findings
        self.table.setRowCount(len(findings))

        for row, f in enumerate(findings):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f.ftype))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(f.description))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f.source_type))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(f.source_id))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(f.snippet))

        self.table.resizeColumnsToContents()

    def update_snippet_view(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._findings):
            finding = self._findings[idx]
            self.snippet_view.setPlainText(finding.snippet)

# -----------------------------
# Entry point
# -----------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

