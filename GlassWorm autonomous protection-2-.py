# === 🔄 AUTOLOADER ===
try:
    import os, time, json, re, requests
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QApplication, QTextEdit
    from PyQt5.QtCore import Qt
except ImportError as e:
    print(f"[Autoloader] Missing dependency: {e.name}. Please install it via pip.")
    exit(1)

# === 🧠 GLOBALS ===
symbolic_registry = {}
glyph_cache = {}
event_bus = []
AUTONOMOUS_MODE = True
blocked_countries = {"RU", "CN", "IR"}  # Expandable

# === 🧬 INVISIBLE UNICODE SCANNER ===
INVISIBLE_UNICODE = ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF']

def scan_unicode_anomalies(text):
    return [(i, repr(c)) for i, c in enumerate(text) if c in INVISIBLE_UNICODE]

# === 🔄 EVENT BUS SYNC ===
def sync_event_bus(event_type, payload):
    event = {"type": event_type, "payload": payload, "timestamp": time.time()}
    event_bus.append(event)
    print(f"[EventBus] {event_type} → {payload}")

# === 🔥 PURGE DAEMON ===
def log_purge_event(path, threat_level, origin):
    symbolic_registry[path] = {
        "status": "purged",
        "threat_level": threat_level,
        "origin": origin,
        "timestamp": time.time()
    }
    sync_event_bus("purge", {"path": path, "origin": origin})

def purge_extension(path):
    try:
        os.remove(path)
        data = glyph_cache.get(path, {})
        log_purge_event(path, data.get("threat_level", 0), data.get("origin", "unknown"))
        glyph_cache.pop(path, None)
        print(f"[Daemon] Purged: {path}")
    except Exception as e:
        print(f"[Daemon] Purge failed: {e}")

def evaluate_threat(path, data):
    if AUTONOMOUS_MODE:
        if data["threat_level"] >= 8 or data["origin"] in blocked_countries:
            purge_extension(path)
            print(f"[Autonomous] Purged {path} due to threat level or origin.")
    else:
        print(f"[Manual] Threat detected: {path} | Level: {data['threat_level']} | Origin: {data['origin']}")

# === 🌐 EXTENSION SCANNER ===
def scan_extensions(extension_dir):
    suspicious = []
    for root, _, files in os.walk(extension_dir):
        for file in files:
            if file.endswith(('.js', '.json')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        anomalies = scan_unicode_anomalies(content)
                        if anomalies:
                            glyph_cache[path] = {
                                "threat_level": len(anomalies),
                                "origin": "unknown",  # Stub for origin detection
                                "timestamp": time.time()
                            }
                            sync_event_bus("glyph_detected", {"path": path, "glyphs": anomalies})
                            evaluate_threat(path, glyph_cache[path])
                            suspicious.append(path)
                except Exception as e:
                    print(f"[Scanner] Failed to read {path}: {e}")
    return suspicious

# === 🧩 GUI PANEL ===
class CodexPurgeOverlay(QWidget):
    def __init__(self, glyph_cache, purge_callback):
        super().__init__()
        self.setWindowTitle("🧿 CodexPurge Overlay")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        self.label = QLabel("🧿 Glyph Threat Matrix")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.list_widget = QListWidget()
        self.refresh_list()
        self.layout.addWidget(self.list_widget)

        self.purge_button = QPushButton("🔥 Purge Selected")
        self.purge_button.clicked.connect(self.purge_selected)
        self.layout.addWidget(self.purge_button)

        self.toggle_button = QPushButton("🔁 Toggle Mode: Autonomous")
        self.toggle_button.clicked.connect(self.toggle_mode)
        self.layout.addWidget(self.toggle_button)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #111; color: #0f0; font-family: monospace;")
        self.layout.addWidget(QLabel("🧠 ASI Oversight Console"))
        self.layout.addWidget(self.console)

        self.setLayout(self.layout)
        self.purge_callback = purge_callback

    def refresh_list(self):
        self.list_widget.clear()
        for path, data in glyph_cache.items():
            item = QListWidgetItem(f"{path} | Threat: {data['threat_level']} | Origin: {data['origin']}")
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)

    def purge_selected(self):
        selected = self.list_widget.selectedItems()
        for item in selected:
            path = item.data(Qt.UserRole)
            self.purge_callback(path)
            self.console.append(f"[ASI] Manual purge: {path}")
        self.refresh_list()

    def toggle_mode(self):
        global AUTONOMOUS_MODE
        AUTONOMOUS_MODE = not AUTONOMOUS_MODE
        mode = "Autonomous" if AUTONOMOUS_MODE else "Manual"
        self.console.append(f"[ASI] Mode switched to: {mode}")

# === 🚀 MAIN EXECUTION ===
def main():
    extension_dir = os.path.expanduser("~/.vscode/extensions")
    print("[Codex] Scanning extensions...")
    scan_extensions(extension_dir)

    app = QApplication([])
    panel = CodexPurgeOverlay(glyph_cache, purge_extension)
    panel.show()
    app.exec_()

if __name__ == "__main__":
    main()

