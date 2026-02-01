import os
import sys
import time
import json
import base64
import hashlib
import platform
import uuid
import string

import psutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import geoip2.database


# =========================
#  Chameleon Cipher
# =========================

class ChameleonCipher:
    BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    GLYPH_ALPHABET  = "Qw3!rT9#yU0$pO2^aS4&dF6*zG8(hJ)kL1_zX5+cV7-bN@eM"

    b64_to_glyph = dict(zip(BASE64_ALPHABET, GLYPH_ALPHABET))
    glyph_to_b64 = dict(zip(GLYPH_ALPHABET, BASE64_ALPHABET))

    def __init__(self, extra_secret: str = ""):
        self.key = self._derive_machine_key(extra_secret)

    def _derive_machine_key(self, extra_secret: str) -> bytes:
        node = platform.node()
        mac = uuid.getnode()
        raw = f"{node}-{mac}-{extra_secret}".encode("utf-8")
        return hashlib.sha256(raw).digest()

    def _xor_with_key(self, data: bytes) -> bytes:
        key = self.key
        out = bytearray()
        for i, b in enumerate(data):
            out.append(b ^ key[i % len(key)])
        return bytes(out)

    def _mirror(self, data: bytes) -> bytes:
        return data[::-1]

    def _to_glyphs(self, b64_text: str) -> str:
        return "".join(self.b64_to_glyph.get(ch, ch) for ch in b64_text)

    def _from_glyphs(self, glyph_text: str) -> str:
        return "".join(self.glyph_to_b64.get(ch, ch) for ch in glyph_text)

    def encrypt(self, plaintext: str) -> str:
        data = plaintext.encode("utf-8")
        data = self._xor_with_key(data)
        data = self._mirror(data)
        b64 = base64.b64encode(data).decode("ascii")
        glyph = self._to_glyphs(b64)
        return glyph

    def decrypt(self, token: str) -> str:
        b64 = self._from_glyphs(token)
        data = base64.b64decode(b64.encode("ascii"))
        data = self._mirror(data)
        data = self._xor_with_key(data)
        return data.decode("utf-8")


# =========================
#  Storage Selector
# =========================

EXCLUDED = ["C:\\", "E:\\"]

def find_storage_path():
    candidates = []

    # 1. Local drives except C and E
    for part in psutil.disk_partitions(all=False):
        mount = part.mountpoint
        # Windows-style drive root check
        if len(mount) >= 2 and mount[1] == ":":
            if mount.upper() not in EXCLUDED and mount[0].upper() in string.ascii_uppercase:
                if os.access(mount, os.W_OK):
                    candidates.append(mount)

    if candidates:
        return sorted(candidates)[0]

    # 2. SMB / UNC mounts
    for part in psutil.disk_partitions(all=False):
        if part.fstype.lower() == "smb" or part.device.startswith("\\\\"):
            if os.access(part.mountpoint, os.W_OK):
                return part.mountpoint

    # 3. Last resort
    fallback = "C:\\ChameleonFallback"
    os.makedirs(fallback, exist_ok=True)
    return fallback


class ChameleonStorage:
    def __init__(self, cipher: ChameleonCipher):
        self.cipher = cipher

    def save(self, filename: str, plaintext: str) -> str:
        path = find_storage_path()
        full = os.path.join(path, filename)
        encrypted = self.cipher.encrypt(plaintext)
        with open(full, "w", encoding="utf-8") as f:
            f.write(encrypted)
        return full

    def load(self, filename: str) -> str:
        path = find_storage_path()
        full = os.path.join(path, filename)
        with open(full, "r", encoding="utf-8") as f:
            encrypted = f.read()
        return self.cipher.decrypt(encrypted)


# =========================
#  Connection Monitor
# =========================

class ConnectionMonitor(QThread):
    updated = pyqtSignal(dict)

    def __init__(self, geoip_db_path="GeoLite2-City.mmdb"):
        super().__init__()
        self.connections = {}
        try:
            self.geo = geoip2.database.Reader(geoip_db_path)
        except Exception:
            self.geo = None

    def lookup_country(self, ip):
        if self.geo is None:
            return "Unknown"
        try:
            rec = self.geo.city(ip)
            return rec.country.name or "Unknown"
        except Exception:
            return "Unknown"

    def classify(self, proc, country, bytes_out):
        if country in ("Local", "United States"):
            return "GREEN"
        if bytes_out > 50000:
            return "RED"
        return "GRAY"

    def run(self):
        while True:
            snapshot = {}
            for c in psutil.net_connections(kind='inet'):
                if c.raddr:
                    pid = c.pid
                    rip = c.raddr.ip
                    proc = "Unknown"
                    try:
                        if pid:
                            proc = psutil.Process(pid).name()
                    except Exception:
                        pass

                    country = self.lookup_country(rip)
                    key = (pid, rip)

                    if key not in self.connections:
                        self.connections[key] = {
                            "first": time.time(),
                            "bytes_out": 0,
                            "bytes_in": 0,
                        }

                    info = self.connections[key]
                    info["last"] = time.time()

                    try:
                        if pid:
                            io = psutil.Process(pid).io_counters()
                            info["bytes_out"] = io.write_bytes
                            info["bytes_in"] = io.read_bytes
                    except Exception:
                        pass

                    status = self.classify(proc, country, info["bytes_out"])

                    snapshot[key] = {
                        "process": proc,
                        "ip": rip,
                        "country": country,
                        "duration": int(info["last"] - info["first"]),
                        "bytes_out": info["bytes_out"],
                        "bytes_in": info["bytes_in"],
                        "status": status,
                    }

            self.updated.emit(snapshot)
            time.sleep(1)


# =========================
#  PyQt5 GUI
# =========================

class WatcherGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Who's Watching Me")
        self.resize(1000, 600)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Process", "Remote IP", "Country", "Duration (s)",
            "Bytes Out", "Bytes In", "Status"
        ])

        self.save_button = QPushButton("Save Encrypted Snapshot")
        self.save_button.clicked.connect(self.save_snapshot)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.monitor = ConnectionMonitor()
        self.monitor.updated.connect(self.update_table)
        self.monitor.start()

        self.last_snapshot = {}
        self.cipher = ChameleonCipher(extra_secret="sentinel-core")
        self.storage = ChameleonStorage(self.cipher)

    def update_table(self, data):
        self.last_snapshot = data
        self.table.setRowCount(len(data))
        for row, ((pid, ip), info) in enumerate(data.items()):
            self.table.setItem(row, 0, QTableWidgetItem(info["process"]))
            self.table.setItem(row, 1, QTableWidgetItem(info["ip"]))
            self.table.setItem(row, 2, QTableWidgetItem(info["country"]))
            self.table.setItem(row, 3, QTableWidgetItem(str(info["duration"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(info["bytes_out"])))
            self.table.setItem(row, 5, QTableWidgetItem(str(info["bytes_in"])))

            status_item = QTableWidgetItem(info["status"])
            if info["status"] == "GREEN":
                status_item.setBackground(Qt.green)
            elif info["status"] == "GRAY":
                status_item.setBackground(Qt.lightGray)
            else:
                status_item.setBackground(Qt.red)

            self.table.setItem(row, 6, status_item)

        self.table.resizeColumnsToContents()

    def save_snapshot(self):
        if not self.last_snapshot:
            QMessageBox.information(self, "Info", "No data to save yet.")
            return

        try:
            payload = json.dumps(self.last_snapshot, indent=2)
            path = self.storage.save("watcher_snapshot.chm", payload)
            QMessageBox.information(self, "Saved", f"Encrypted snapshot saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save snapshot:\n{e}")


# =========================
#  Main
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = WatcherGUI()
    gui.show()
    sys.exit(app.exec_())

# ===== Imports =====
import os, sys, time, json, base64, hashlib, socket, uuid, platform, string
from collections import defaultdict, deque

import psutil
import geoip2.database

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# ===== Cipher =====
class ChameleonCipher:
    BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    GLYPHS = "Qw3!rT9#yU0$pO2^aS4&dF6*zG8(hJ)kL1_zX5+cV7-bN@eM"
    MAP = dict(zip(BASE64, GLYPHS))
    REV = dict(zip(GLYPHS, BASE64))

    def __init__(self, secret=""):
        raw = f"{platform.node()}-{uuid.getnode()}-{secret}".encode()
        self.key = hashlib.sha256(raw).digest()

    def _xor(self, b):
        return bytes(b[i] ^ self.key[i % len(self.key)] for i in range(len(b)))

    def encrypt(self, text):
        b = self._xor(text.encode())
        return "".join(self.MAP[c] for c in base64.b64encode(b).decode())

    def decrypt(self, token):
        b64 = "".join(self.REV[c] for c in token)
        return self._xor(base64.b64decode(b64)).decode()

# ===== Geo Fusion =====
class GeoFusion:
    def __init__(self, db="GeoLite2-City.mmdb"):
        try:
            self.geo = geoip2.database.Reader(db)
        except:
            self.geo = None
        self.history = defaultdict(int)

    def reverse_dns(self, ip):
        try:
            return socket.gethostbyaddr(ip)[0]
        except:
            return None

    def fuse(self, ip):
        loc = {"city":None,"region":None,"country":"Unknown",
               "lat":None,"lon":None,"provider":None}

        if self.geo:
            try:
                r = self.geo.city(ip)
                loc.update({
                    "city": r.city.name,
                    "region": r.subdivisions.most_specific.name if r.subdivisions else None,
                    "country": r.country.name or "Unknown",
                    "lat": r.location.latitude,
                    "lon": r.location.longitude
                })
            except: pass

        rdns = self.reverse_dns(ip)
        if rdns:
            loc["provider"] = rdns.split('.')[-2]

        self.history[ip] += 1
        loc["seen_before"] = self.history[ip] > 1
        return loc

# ===== Scoring =====
def geo_confidence(loc):
    score = 0
    score += 40 if loc["country"] != "Unknown" else 0
    score += 25 if loc["region"] else 0
    score += 20 if loc["city"] else 0
    score += 10 if loc["lat"] and loc["lon"] else 0
    score += 5 if loc["provider"] else 0
    score += 5 if loc.get("seen_before") else 0
    return min(100, score)

# ===== Behavioral Fingerprinting =====
class BehaviorTracker:
    def __init__(self):
        self.sessions = defaultdict(lambda: deque(maxlen=10))

    def update(self, key, bytes_out, duration):
        self.sessions[key].append((bytes_out, duration))

    def classify(self, key):
        s = self.sessions[key]
        if not s: return "Unknown"
        avg = sum(b/d for b,d in s if d) / len(s)
        if avg > 50000: return "Exfil"
        if len(s) > 5: return "Beacon"
        return "Normal"

# ===== Threat Analyzer =====
def analyze_regions(snapshot):
    regions = defaultdict(list)
    for info in snapshot.values():
        regions[info["country"]].append(info)

    report = {}
    for c, items in regions.items():
        avg_conf = sum(i["confidence"] for i in items)//len(items)
        bytes_out = sum(i["bytes_out"] for i in items)
        threat = "LOW"
        if avg_conf < 40 and bytes_out > 100_000: threat = "HIGH"
        elif bytes_out > 50_000: threat = "MEDIUM"
        report[c] = {"connections":len(items),"threat":threat}
    return report

# ===== Monitor Thread =====
class Monitor(QThread):
    updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.geo = GeoFusion()
        self.behavior = BehaviorTracker()
        self.state = {}

    def run(self):
        while True:
            snap = {}
            for c in psutil.net_connections('inet'):
                if not c.raddr: continue
                pid, ip = c.pid, c.raddr.ip
                proc = psutil.Process(pid).name() if pid else "Unknown"

                loc = self.geo.fuse(ip)
                conf = geo_confidence(loc)
                weak = conf < 60

                key = (pid, ip)
                io = psutil.Process(pid).io_counters() if pid else None
                bout = io.write_bytes if io else 0

                self.behavior.update(key, bout, 1)
                behavior = self.behavior.classify(key)

                snap[key] = {
                    "process": proc,
                    "ip": ip,
                    "country": loc["country"],
                    "confidence": conf,
                    "weak": weak,
                    "bytes_out": bout,
                    "behavior": behavior
                }

            self.updated.emit(snap)
            time.sleep(1)

# ===== GUI =====
class PrometheusGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prometheus Sentinel")
        self.resize(1200,700)

        self.table = QTableWidget(0,6)
        self.table.setHorizontalHeaderLabels(
            ["Process","IP","Country","Confidence","Behavior","Weak"]
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

        self.monitor = Monitor()
        self.monitor.updated.connect(self.update)
        self.monitor.start()

    def update(self, data):
        self.table.setRowCount(len(data))
        for r, info in enumerate(data.values()):
            for c,k in enumerate(
                ["process","ip","country","confidence","behavior","weak"]
            ):
                self.table.setItem(r,c,QTableWidgetItem(str(info[k])))

# ===== Main =====
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PrometheusGUI()
    gui.show()
    sys.exit(app.exec_())
