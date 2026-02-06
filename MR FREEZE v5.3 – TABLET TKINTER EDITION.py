# ===== MR FREEZE v5.3 – TABLET TKINTER EDITION =====
# MagicBox Plasma-Purple theme, tabbed cockpit.
# - Tkinter GUI (no Flask)
# - TCP "ping" (no raw ICMP, no ARP)
# - psutil used only for gaming auto-detect (guarded)
# - Zero-Knowledge Mode: ON by default, toggleable, persists toggle only
# - Outbound Shield: ON by default, toggleable
# - Auto-scan: initial scan at boot, then every hour
# - Gaming Mode: automatic when a game process is detected

import os
import sys
import json
import time
import socket
import threading
import statistics
import platform
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import ttk, messagebox

# psutil is optional; if missing, gaming auto-detect is disabled gracefully
try:
    import psutil
except ImportError:
    psutil = None

# ---------- CONFIG / STORAGE ----------

APP_NAME = "mr_freeze_v5_3_tablet"
STATE_FILE = APP_NAME + ".json.enc"

def get_state_path():
    base = os.path.abspath(".")
    return os.path.join(base, STATE_FILE)

def get_machine_fingerprint():
    data = platform.node() + "|" + platform.system() + "|" + platform.machine()
    return hashlib.sha256(data.encode("utf-8")).digest()

def encrypt_bytes(data: bytes) -> bytes:
    key = get_machine_fingerprint()
    out = bytearray()
    for i, b in enumerate(data):
        out.append(b ^ key[i % len(key)])
    return bytes(out)

def decrypt_bytes(data: bytes) -> bytes:
    return encrypt_bytes(data)

# ---------- TCP "PING" (tablet-safe) ----------

def tcp_ping(ip, port=443, timeout=1.0):
    start = time.time()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            if s.connect_ex((ip, port)) == 0:
                end = time.time()
                return int((end - start) * 1000), True
    except Exception:
        pass
    return None, False

# ---------- MACHINE MODEL ----------

class Machine:
    def __init__(self, address):
        self.address = address
        self.status = "UNKNOWN"  # RUNNING / OFFLINE / FROZEN / UNKNOWN
        self.latency = None

        self.latency_history = []
        self.jitter_history = []
        self.loss_history = []
        self.offline_count = 0
        self.flap_count = 0

        self.behavior_profile = "UNKNOWN"
        self.threat_score = 0
        self.anomaly_score = 0.0
        self.fused_anomaly = 0.0
        self.last_root_cause = "None"

        self.prediction_label = "UNKNOWN"
        self.prediction_confidence = 0.0

        self.auto_frozen = False
        self.manual_frozen = False
        self.last_status_change = datetime.utcnow()
        self.last_seen = None

        self._stop_event = threading.Event()
        self._thread = None

    def is_frozen(self):
        return self.manual_frozen or self.auto_frozen

    def freeze(self, manual=True):
        if manual:
            self.manual_frozen = True
        else:
            self.auto_frozen = True
        if self.status != "OFFLINE":
            self.status = "FROZEN"
        self.last_status_change = datetime.utcnow()

    def unfreeze(self):
        self.manual_frozen = False
        self.auto_frozen = False
        if self.status != "OFFLINE":
            self.status = "RUNNING"
        self.last_status_change = datetime.utcnow()

    def real_ping(self):
        ip = self.address.split(":")[0]
        latency, ok = tcp_ping(ip, port=443, timeout=1.0)
        if ok and latency is not None:
            self.latency = latency
            self.latency_history.append(latency)
            if len(self.latency_history) > 200:
                self.latency_history.pop(0)

            if len(self.latency_history) >= 2:
                j = abs(self.latency_history[-1] - self.latency_history[-2])
                self.jitter_history.append(j)
                if len(self.jitter_history) > 200:
                    self.jitter_history.pop(0)

            self.loss_history.append(0)
            if len(self.loss_history) > 200:
                self.loss_history.pop(0)

            self.last_seen = datetime.utcnow()
            if self.status == "OFFLINE":
                self.flap_count += 1
            if not self.is_frozen():
                self.status = "RUNNING"
        else:
            self.latency = None
            self.offline_count += 1
            self.loss_history.append(1)
            if len(self.loss_history) > 200:
                self.loss_history.pop(0)
            self.status = "OFFLINE"

        self.update_behavior_profile()
        self.update_threat_score()
        self.update_anomaly_score()
        self.update_fused_anomaly()
        self.update_root_cause()

    def update_behavior_profile(self):
        if not self.latency_history:
            self.behavior_profile = "UNKNOWN"
            return

        if self.status == "OFFLINE":
            if self.offline_count > 10:
                self.behavior_profile = "DEAD-PRONE"
            else:
                self.behavior_profile = "UNSTABLE"
            return

        lat = self.latency_history[-50:] if len(self.latency_history) > 50 else self.latency_history
        if len(lat) < 3:
            self.behavior_profile = "UNKNOWN"
            return

        avg = statistics.mean(lat)
        stdev = statistics.pstdev(lat)
        loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))

        if loss_rate > 0.3:
            self.behavior_profile = "UNSTABLE"
        elif stdev < 10 and avg < 80:
            self.behavior_profile = "STABLE"
        elif stdev > 50 and avg < 200:
            self.behavior_profile = "SPIKY"
        elif self.flap_count >= 3:
            self.behavior_profile = "FLAPPY"
        else:
            self.behavior_profile = "UNSTABLE"

    def update_threat_score(self):
        score = 0
        if self.status == "OFFLINE":
            score += 70
        elif self.status == "FROZEN":
            score += 30

        if self.latency is not None:
            if self.latency > 400:
                score += 35
            elif self.latency > 300:
                score += 25
            elif self.latency > 200:
                score += 15
            elif self.latency > 100:
                score += 5

        if self.jitter_history:
            javg = statistics.mean(self.jitter_history[-20:])
            if javg > 60:
                score += 25
            elif javg > 30:
                score += 15

        if self.loss_history:
            loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))
            if loss_rate > 0.4:
                score += 35
            elif loss_rate > 0.2:
                score += 20
            elif loss_rate > 0.1:
                score += 10

        if self.flap_count >= 5:
            score += 25
        elif self.flap_count >= 3:
            score += 15

        if self.behavior_profile in ("UNSTABLE", "DEAD-PRONE", "FLAPPY"):
            score += 15

        self.threat_score = max(0, min(100, score))

    def update_anomaly_score(self):
        if len(self.latency_history) < 10 or self.latency is None:
            self.anomaly_score = 0.0
            return
        window = self.latency_history[-50:]
        avg = statistics.mean(window)
        stdev = statistics.pstdev(window) or 1.0
        z = abs(self.latency - avg) / stdev
        self.anomaly_score = z

    def update_fused_anomaly(self):
        lat_component = min(self.anomaly_score / 5.0, 1.0)
        jitter_component = 0.0
        loss_component = 0.0
        flap_component = min(self.flap_count / 5.0, 1.0)
        threat_component = self.threat_score / 100.0

        if self.jitter_history:
            javg = statistics.mean(self.jitter_history[-20:])
            jitter_component = min(javg / 100.0, 1.0)

        if self.loss_history:
            loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))
            loss_component = min(loss_rate * 3.0, 1.0)

        self.fused_anomaly = max(0.0, min(1.0, 0.3 * lat_component +
                                          0.2 * jitter_component +
                                          0.2 * loss_component +
                                          0.1 * flap_component +
                                          0.2 * threat_component))

    def update_root_cause(self):
        reasons = []
        if self.status == "OFFLINE":
            reasons.append("offline")
        if self.latency is not None and self.latency > 300:
            reasons.append("high latency")
        if self.jitter_history:
            javg = statistics.mean(self.jitter_history[-20:])
            if javg > 40:
                reasons.append("high jitter")
        if self.loss_history:
            loss_rate = sum(self.loss_history[-50:]) / max(1, len(self.loss_history[-50:]))
            if loss_rate > 0.2:
                reasons.append("packet loss")
        if self.flap_count >= 3:
            reasons.append("flapping")

        if not reasons:
            self.last_root_cause = "Stable"
        else:
            self.last_root_cause = ", ".join(reasons)

    def start_ping_loop(self, callback, interval=5.0):
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while not self._stop_event.is_set():
                self.real_ping()
                callback(self)
                time.sleep(interval)

        self._stop_event.clear()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_ping_loop(self):
        self._stop_event.set()

    def to_dict(self):
        return {
            "address": self.address,
            "status": self.status,
            "latency_history": self.latency_history,
            "jitter_history": self.jitter_history,
            "loss_history": self.loss_history,
            "offline_count": self.offline_count,
            "flap_count": self.flap_count,
            "behavior_profile": self.behavior_profile,
            "threat_score": self.threat_score,
            "anomaly_score": self.anomaly_score,
            "fused_anomaly": self.fused_anomaly,
            "last_root_cause": self.last_root_cause,
            "prediction_label": self.prediction_label,
            "prediction_confidence": self.prediction_confidence,
            "auto_frozen": self.auto_frozen,
            "manual_frozen": self.manual_frozen,
        }

    @staticmethod
    def from_dict(d):
        m = Machine(d["address"])
        m.status = d.get("status", "UNKNOWN")
        m.latency_history = d.get("latency_history", [])
        m.jitter_history = d.get("jitter_history", [])
        m.loss_history = d.get("loss_history", [])
        m.offline_count = d.get("offline_count", 0)
        m.flap_count = d.get("flap_count", 0)
        m.behavior_profile = d.get("behavior_profile", "UNKNOWN")
        m.threat_score = d.get("threat_score", 0)
        m.anomaly_score = d.get("anomaly_score", 0.0)
        m.fused_anomaly = d.get("fused_anomaly", 0.0)
        m.last_root_cause = d.get("last_root_cause", "None")
        m.prediction_label = d.get("prediction_label", "UNKNOWN")
        m.prediction_confidence = d.get("prediction_confidence", 0.0)
        m.auto_frozen = d.get("auto_frozen", False)
        m.manual_frozen = d.get("manual_frozen", False)
        return m

# ---------- SIMPLE PREDICTOR ----------

class LocalAIPredictor:
    def predict(self, m: Machine):
        if len(m.latency_history) < 5:
            return "UNKNOWN", 0.0
        lat = m.latency if m.latency is not None else 0.0
        avg_lat = statistics.mean(m.latency_history) if m.latency_history else 0.0
        loss_rate = (sum(m.loss_history) / len(m.loss_history)) if m.loss_history else 0.0
        jitter_avg = statistics.mean(m.jitter_history) if m.jitter_history else 0.0
        fused = m.fused_anomaly
        threat = m.threat_score

        risk = 0.0
        risk += min(lat / 500.0, 1.0) * 0.2
        risk += min(avg_lat / 400.0, 1.0) * 0.2
        risk += min(loss_rate * 3.0, 1.0) * 0.2
        risk += min(jitter_avg / 100.0, 1.0) * 0.1
        risk += fused * 0.2
        risk += (threat / 100.0) * 0.1
        risk = max(0.0, min(1.0, risk))

        if m.status == "OFFLINE":
            return "OFFLINE", 1.0
        if risk >= 0.8:
            return "OFFLINE_RISK", risk
        elif risk >= 0.6:
            return "DEGRADE_SOON", risk
        elif risk >= 0.4:
            return "SPIKE_RISK", risk
        elif risk >= 0.2:
            return "WATCH", 0.5 + risk / 2.0
        else:
            return "STABLE", 0.6 + (0.2 - risk)

# ---------- COCKPIT CORE ----------

class CockpitCore:
    def __init__(self):
        self.machines = []
        self.ai = LocalAIPredictor()

        # Privacy organs
        self.zero_knowledge = True   # ON by default
        self.outbound_shield = True  # ON by default

        # Scanning
        self.auto_scan_subnets = []      # list of strings
        self.auto_scan_interval = 3600   # seconds (1 hour)
        self.last_scan_summary = "Never"
        self.initial_scan_done = False

        # Gaming
        self.gaming_auto = True
        self.gaming_active = False
        self.gaming_known_names = [
            "steam.exe", "csgo.exe", "valorant.exe", "fortniteclient-win64-shipping.exe",
            "eldenring.exe", "wow.exe", "overwatch.exe", "league of legends.exe",
            "dota2.exe", "apex.exe"
        ]

        self.state_dirty = False
        self.load_state()

        # Start ping loops
        for m in self.machines:
            m.start_ping_loop(self.on_machine_update)

        # Start auto-scan loop
        threading.Thread(target=self.auto_scan_loop, daemon=True).start()

        # Start gaming auto-detect loop
        threading.Thread(target=self.gaming_loop, daemon=True).start()

    # ----- STATE -----

    def save_state_if_dirty(self):
        if not self.state_dirty:
            return
        self.save_state()
        self.state_dirty = False

    def save_state(self):
        try:
            data = {
                "zero_knowledge": self.zero_knowledge,
                "outbound_shield": self.outbound_shield,
                "auto_scan_subnets": self.auto_scan_subnets,
                "auto_scan_interval": self.auto_scan_interval,
                "gaming_auto": self.gaming_auto,
            }
            # Zero-knowledge: do NOT persist machines when enabled
            if not self.zero_knowledge:
                data["machines"] = [m.to_dict() for m in self.machines]

            raw = json.dumps(data, indent=2).encode("utf-8")
            enc = encrypt_bytes(raw)
            with open(get_state_path(), "wb") as f:
                f.write(enc)
        except Exception as e:
            print("[STATE] Save failed:", e)

    def load_state(self):
        path = get_state_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                enc = f.read()
            raw = decrypt_bytes(enc)
            data = json.loads(raw.decode("utf-8"))

            self.zero_knowledge = data.get("zero_knowledge", True)
            self.outbound_shield = data.get("outbound_shield", True)
            self.auto_scan_subnets = data.get("auto_scan_subnets", [])
            self.auto_scan_interval = data.get("auto_scan_interval", 3600)
            self.gaming_auto = data.get("gaming_auto", True)

            if not self.zero_knowledge:
                for d in data.get("machines", []):
                    self.machines.append(Machine.from_dict(d))
                print(f"[STATE] Restored {len(self.machines)} machines.")
            else:
                print("[STATE] Zero-Knowledge ON: no machines restored.")
        except Exception as e:
            print("[STATE] Load failed:", e)

    # ----- MACHINES -----

    def add_machine(self, address):
        address = address.strip()
        if not address:
            return
        for m in self.machines:
            if m.address == address:
                return
        m = Machine(address)
        self.machines.append(m)
        m.start_ping_loop(self.on_machine_update)
        self.state_dirty = True
        self.save_state_if_dirty()

    def remove_machine(self, address):
        survivors = []
        for m in self.machines:
            if m.address == address:
                m.stop_ping_loop()
            else:
                survivors.append(m)
        self.machines = survivors
        self.state_dirty = True
        self.save_state_if_dirty()

    def freeze_machine(self, address):
        for m in self.machines:
            if m.address == address:
                m.freeze(manual=True)
                self.state_dirty = True
                self.save_state_if_dirty()
                return

    def unfreeze_machine(self, address):
        for m in self.machines:
            if m.address == address:
                m.unfreeze()
                self.state_dirty = True
                self.save_state_if_dirty()
                return

    # ----- SCANNING -----

    def _estimate_thread_count(self, num_ips):
        min_threads = 10
        max_threads = 100
        try:
            cores = os.cpu_count() or 4
        except Exception:
            cores = 4
        base = max(min_threads, min(max_threads, cores * 5))
        if num_ips < base:
            base = max(min_threads, num_ips)
        return base

    def scan_network(self, subnet):
        print(f"[SCAN] Starting scan: {subnet}")
        ips = []
        try:
            if "-" in subnet:
                base, rng = subnet.rsplit(".", 1)
                start, end = map(int, rng.split("-"))
                ips = [f"{base}.{i}" for i in range(start, end + 1)]
            elif "/" in subnet:
                import ipaddress
                net = ipaddress.ip_network(subnet, strict=False)
                ips = [str(ip) for ip in net.hosts()]
            else:
                ips = [subnet]
        except Exception as e:
            print("[SCAN] Invalid subnet:", e)
            self.last_scan_summary = f"Invalid subnet: {subnet}"
            return

        if not ips:
            self.last_scan_summary = "No IPs to scan."
            return

        num_ips = len(ips)
        threads = self._estimate_thread_count(num_ips)
        live_hosts = []

        def worker(ip):
            m = Machine(ip)
            m.real_ping()
            if m.status == "RUNNING":
                return m
            return None

        try:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                future_map = {executor.submit(worker, ip): ip for ip in ips}
                for future in as_completed(future_map):
                    machine = future.result()
                    if machine is not None:
                        live_hosts.append(machine.address)
                        exists = any(x.address == machine.address for x in self.machines)
                        if not exists:
                            self.machines.append(machine)
                            machine.start_ping_loop(self.on_machine_update)
                            self.state_dirty = True
        except Exception as e:
            print("[SCAN] ERROR:", e)
            self.last_scan_summary = f"Scan error: {e}"
            return

        if live_hosts:
            self.last_scan_summary = "Live: " + ", ".join(live_hosts)
        else:
            self.last_scan_summary = "No live hosts found."
        self.save_state_if_dirty()

    def auto_scan_loop(self):
        # Initial scan at boot (if subnets configured)
        if self.auto_scan_subnets and not self.initial_scan_done:
            for subnet in self.auto_scan_subnets:
                self.scan_network(subnet)
            self.initial_scan_done = True

        while True:
            if self.auto_scan_subnets:
                time.sleep(self.auto_scan_interval)
                for subnet in self.auto_scan_subnets:
                    self.scan_network(subnet)
            else:
                time.sleep(5)

    # ----- GAMING AUTO-DETECT -----

    def gaming_loop(self):
        if psutil is None:
            print("[GAMING] psutil not available; gaming auto-detect disabled.")
            return
        while True:
            try:
                if self.gaming_auto:
                    active = False
                    for p in psutil.process_iter(["name"]):
                        name = (p.info.get("name") or "").lower()
                        if any(g in name for g in self.gaming_known_names):
                            active = True
                            break
                    self.gaming_active = active
                else:
                    self.gaming_active = False
            except Exception:
                # Never crash on tablets
                self.gaming_active = False
            time.sleep(5)

    # ----- UPDATE CALLBACK -----

    def on_machine_update(self, m: Machine):
        label, conf = self.ai.predict(m)
        m.prediction_label = label
        m.prediction_confidence = conf

        # Auto-freeze logic (unless gaming mode is active)
        if not m.is_frozen() and not self.gaming_active:
            if m.latency is not None and m.latency > 400:
                m.freeze(manual=False)
            elif m.fused_anomaly > 0.8 or m.prediction_label == "OFFLINE_RISK":
                m.freeze(manual=False)

        if m.status == "RUNNING" and m.auto_frozen:
            if (datetime.utcnow() - m.last_status_change) > timedelta(seconds=10):
                m.unfreeze()

        self.state_dirty = True
        self.save_state_if_dirty()

    # ----- METRICS -----

    def compute_metrics(self):
        total = len(self.machines)
        running = sum(1 for m in self.machines if m.status == "RUNNING")
        frozen = sum(1 for m in self.machines if m.is_frozen())
        offline = sum(1 for m in self.machines if m.status == "OFFLINE")
        degraded = sum(1 for m in self.machines if m.threat_score >= 40)

        latencies = [m.latency for m in self.machines if m.latency is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        jitters = []
        losses = []
        anomalies = []
        fused = []
        for m in self.machines:
            if m.jitter_history:
                jitters.extend(m.jitter_history[-10:])
            if m.loss_history:
                losses.extend(m.loss_history[-50:])
            anomalies.append(m.anomaly_score)
            fused.append(m.fused_anomaly)

        avg_jitter = statistics.mean(jitters) if jitters else None
        avg_loss = (sum(losses) / len(losses)) if losses else None
        avg_anomaly = statistics.mean(anomalies) if anomalies else 0.0
        avg_fused = statistics.mean(fused) if fused else 0.0

        swarm_health = max(0.0, min(1.0, 1.0 - avg_fused))

        return {
            "total": total,
            "running": running,
            "frozen": frozen,
            "offline": offline,
            "degraded": degraded,
            "avg_latency": avg_latency,
            "avg_jitter": avg_jitter,
            "avg_loss": avg_loss,
            "avg_anomaly": avg_anomaly,
            "swarm_health": swarm_health,
        }

# ---------- TKINTER MAGICBOX (PLASMA-PURPLE) ----------

class MagicBoxApp(tk.Tk):
    def __init__(self, core: CockpitCore):
        super().__init__()
        self.core = core
        self.title("MR FREEZE v5.3 – Tablet Cockpit (MagicBox Plasma-Purple)")
        self.configure(bg="#05030A")
        self.geometry("1100x650")

        # Plasma-purple theme
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background="#05030A", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#12001F",
                        foreground="#E0D0FF",
                        padding=(10, 4),
                        font=("Consolas", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#3A0070")],
                  foreground=[("selected", "#FFFFFF")])

        style.configure("Treeview",
                        background="#080313",
                        foreground="#E0D0FF",
                        fieldbackground="#080313",
                        rowheight=20,
                        font=("Consolas", 9))
        style.configure("Treeview.Heading",
                        background="#1A0030",
                        foreground="#E0D0FF",
                        font=("Consolas", 9, "bold"))

        style.configure("Purple.TButton",
                        background="#3A0070",
                        foreground="#FFFFFF",
                        font=("Consolas", 9, "bold"))
        style.map("Purple.TButton",
                  background=[("active", "#5A00A0")])

        style.configure("Danger.TButton",
                        background="#800020",
                        foreground="#FFFFFF",
                        font=("Consolas", 9, "bold"))
        style.map("Danger.TButton",
                  background=[("active", "#B00030")])

        style.configure("Info.TLabel",
                        background="#05030A",
                        foreground="#C0A0FF",
                        font=("Consolas", 9))

        style.configure("Title.TLabel",
                        background="#05030A",
                        foreground="#FF66FF",
                        font=("Consolas", 12, "bold"))

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.dashboard_frame = ttk.Frame(self.notebook)
        self.scanner_frame = ttk.Frame(self.notebook)
        self.privacy_frame = ttk.Frame(self.notebook)
        self.gaming_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.dashboard_frame, text="Dashboard")
        self.notebook.add(self.scanner_frame, text="Scanner")
        self.notebook.add(self.privacy_frame, text="Privacy")
        self.notebook.add(self.gaming_frame, text="Gaming")

        self.build_dashboard_tab()
        self.build_scanner_tab()
        self.build_privacy_tab()
        self.build_gaming_tab()

        self.after(1000, self.refresh_ui)

    # ----- DASHBOARD TAB -----

    def build_dashboard_tab(self):
        f = self.dashboard_frame
        f.configure(style="TFrame")

        title = ttk.Label(f, text="SWARM STATUS – PLASMA CORE", style="Title.TLabel")
        title.pack(anchor="w", padx=8, pady=4)

        # Metrics
        self.metrics_label = ttk.Label(f, text="", style="Info.TLabel", justify="left")
        self.metrics_label.pack(anchor="w", padx=8, pady=4)

        # Machine table
        cols = ("address", "status", "latency", "threat", "profile", "anomaly", "fused", "prediction", "root")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", selectmode="browse")
        for c, w in zip(cols,
                        (160, 80, 80, 80, 100, 80, 80, 120, 200)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=4)

        # Controls
        ctrl = ttk.Frame(f)
        ctrl.pack(fill="x", padx=8, pady=4)

        self.address_entry = ttk.Entry(ctrl, width=25)
        self.address_entry.grid(row=0, column=0, padx=4)
        add_btn = ttk.Button(ctrl, text="Add Machine", style="Purple.TButton",
                             command=self.on_add_machine)
        add_btn.grid(row=0, column=1, padx=4)

        freeze_btn = ttk.Button(ctrl, text="Freeze", style="Danger.TButton",
                                command=self.on_freeze_selected)
        freeze_btn.grid(row=0, column=2, padx=4)

        unfreeze_btn = ttk.Button(ctrl, text="Unfreeze", style="Purple.TButton",
                                  command=self.on_unfreeze_selected)
        unfreeze_btn.grid(row=0, column=3, padx=4)

        remove_btn = ttk.Button(ctrl, text="Remove", style="Danger.TButton",
                                command=self.on_remove_selected)
        remove_btn.grid(row=0, column=4, padx=4)

    def on_add_machine(self):
        addr = self.address_entry.get().strip()
        if not addr:
            return
        self.core.add_machine(addr)
        self.address_entry.delete(0, tk.END)

    def get_selected_address(self):
        sel = self.tree.selection()
        if not sel:
            return None
        item = self.tree.item(sel[0])
        return item["values"][0]

    def on_freeze_selected(self):
        addr = self.get_selected_address()
        if not addr:
            return
        self.core.freeze_machine(addr)

    def on_unfreeze_selected(self):
        addr = self.get_selected_address()
        if not addr:
            return
        self.core.unfreeze_machine(addr)

    def on_remove_selected(self):
        addr = self.get_selected_address()
        if not addr:
            return
        self.core.remove_machine(addr)

    # ----- SCANNER TAB -----

    def build_scanner_tab(self):
        f = self.scanner_frame

        title = ttk.Label(f, text="NETWORK SCANNER – PLASMA SWEEP", style="Title.TLabel")
        title.pack(anchor="w", padx=8, pady=4)

        frm = ttk.Frame(f)
        frm.pack(anchor="w", padx=8, pady=4)

        ttk.Label(frm, text="Subnet / Range:", style="Info.TLabel").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.subnet_entry = ttk.Entry(frm, width=30)
        self.subnet_entry.grid(row=0, column=1, padx=4, pady=2)

        scan_btn = ttk.Button(frm, text="Scan Now", style="Purple.TButton",
                              command=self.on_scan_now)
        scan_btn.grid(row=0, column=2, padx=4, pady=2)

        ttk.Label(frm, text="Auto-Scan Subnets (one per line):", style="Info.TLabel").grid(
            row=1, column=0, padx=4, pady=4, sticky="nw"
        )
        self.auto_subnets_text = tk.Text(frm, width=40, height=6, bg="#080313", fg="#E0D0FF",
                                         insertbackground="#FF66FF")
        self.auto_subnets_text.grid(row=1, column=1, padx=4, pady=4, columnspan=2, sticky="w")

        ttk.Label(frm, text="Auto-Scan Interval:", style="Info.TLabel").grid(
            row=2, column=0, padx=4, pady=2, sticky="w"
        )
        self.interval_label = ttk.Label(frm, text="Every 1 hour (fixed)", style="Info.TLabel")
        self.interval_label.grid(row=2, column=1, padx=4, pady=2, sticky="w")

        save_btn = ttk.Button(frm, text="Save Auto-Scan Config", style="Purple.TButton",
                              command=self.on_save_autoscan)
        save_btn.grid(row=3, column=1, padx=4, pady=4, sticky="w")

        self.scan_status_label = ttk.Label(f, text="Last Scan: Never", style="Info.TLabel")
        self.scan_status_label.pack(anchor="w", padx=8, pady=4)

        # preload auto-subnets
        if self.core.auto_scan_subnets:
            self.auto_subnets_text.delete("1.0", tk.END)
            self.auto_subnets_text.insert("1.0", "\n".join(self.core.auto_scan_subnets))

    def on_scan_now(self):
        subnet = self.subnet_entry.get().strip()
        if not subnet:
            return
        threading.Thread(target=self.core.scan_network, args=(subnet,), daemon=True).start()

    def on_save_autoscan(self):
        text = self.auto_subnets_text.get("1.0", tk.END).strip()
        subnets = [line.strip() for line in text.splitlines() if line.strip()]
        self.core.auto_scan_subnets = subnets
        self.core.auto_scan_interval = 3600  # fixed 1 hour
        self.core.state_dirty = True
        self.core.save_state_if_dirty()
        messagebox.showinfo("Auto-Scan", "Auto-scan configuration saved.\nInitial scan will run at boot, then hourly.")

    # ----- PRIVACY TAB -----

    def build_privacy_tab(self):
        f = self.privacy_frame

        title = ttk.Label(f, text="PRIVACY ORGANS – ZERO-KNOWLEDGE CORE", style="Title.TLabel")
        title.pack(anchor="w", padx=8, pady=4)

        self.zk_var = tk.BooleanVar(value=self.core.zero_knowledge)
        self.os_var = tk.BooleanVar(value=self.core.outbound_shield)

        zk_frame = ttk.Frame(f)
        zk_frame.pack(anchor="w", padx=8, pady=6, fill="x")

        zk_check = ttk.Checkbutton(zk_frame, text="Zero-Knowledge Mode (ON by default)",
                                   variable=self.zk_var,
                                   command=self.on_toggle_zero_knowledge)
        zk_check.grid(row=0, column=0, sticky="w")

        zk_desc = ttk.Label(zk_frame,
                            text="When ON: no machine telemetry or history is written to disk.\n"
                                 "Only minimal config (like this toggle) is stored.",
                            style="Info.TLabel", justify="left")
        zk_desc.grid(row=1, column=0, sticky="w", pady=2)

        os_frame = ttk.Frame(f)
        os_frame.pack(anchor="w", padx=8, pady=6, fill="x")

        os_check = ttk.Checkbutton(os_frame, text="Outbound Shield (ON by default)",
                                   variable=self.os_var,
                                   command=self.on_toggle_outbound_shield)
        os_check.grid(row=0, column=0, sticky="w")

        os_desc = ttk.Label(os_frame,
                            text="When ON: any future exports / APIs must mask addresses and sensitive fields.\n"
                                 "Local cockpit visuals remain full-fidelity.",
                            style="Info.TLabel", justify="left")
        os_desc.grid(row=1, column=0, sticky="w", pady=2)

    def on_toggle_zero_knowledge(self):
        self.core.zero_knowledge = self.zk_var.get()
        self.core.state_dirty = True
        self.core.save_state_if_dirty()
        messagebox.showinfo(
            "Zero-Knowledge",
            "Zero-Knowledge is now {}.\n\nWhen ON, no machine logs or histories are written to disk.\n"
            "Only minimal config (like this toggle) is remembered across reboot.".format(
                "ON" if self.core.zero_knowledge else "OFF"
            )
        )

    def on_toggle_outbound_shield(self):
        self.core.outbound_shield = self.os_var.get()
        self.core.state_dirty = True
        self.core.save_state_if_dirty()
        messagebox.showinfo(
            "Outbound Shield",
            "Outbound Shield is now {}.\n\nWhen ON, any future exports/APIs must use masked data.".format(
                "ON" if self.core.outbound_shield else "OFF"
            )
        )

    # ----- GAMING TAB -----

    def build_gaming_tab(self):
        f = self.gaming_frame

        title = ttk.Label(f, text="GAMING MODE – AUTONOMIC DETECTOR", style="Title.TLabel")
        title.pack(anchor="w", padx=8, pady=4)

        self.gaming_auto_var = tk.BooleanVar(value=self.core.gaming_auto)
        auto_check = ttk.Checkbutton(f, text="Automatic Gaming Mode (detects game processes)",
                                     variable=self.gaming_auto_var,
                                     command=self.on_toggle_gaming_auto)
        auto_check.pack(anchor="w", padx=8, pady=4)

        self.gaming_status_label = ttk.Label(f, text="", style="Info.TLabel", justify="left")
        self.gaming_status_label.pack(anchor="w", padx=8, pady=4)

        desc = ttk.Label(
            f,
            text="When a known game process is detected:\n"
                 " - Gaming Mode becomes ACTIVE\n"
                 " - Auto-freeze logic is suppressed to avoid disrupting gameplay\n"
                 " - Auto-scan still runs hourly, but freeze decisions are relaxed",
            style="Info.TLabel",
            justify="left"
        )
        desc.pack(anchor="w", padx=8, pady=4)

        if psutil is None:
            warn = ttk.Label(
                f,
                text="psutil is not installed: automatic gaming detection is disabled on this device.",
                style="Info.TLabel",
                justify="left"
            )
            warn.pack(anchor="w", padx=8, pady=4)

    def on_toggle_gaming_auto(self):
        self.core.gaming_auto = self.gaming_auto_var.get()
        self.core.state_dirty = True
        self.core.save_state_if_dirty()

    # ----- UI REFRESH LOOP -----

    def refresh_ui(self):
        # Metrics
        metrics = self.core.compute_metrics()
        txt = []
        txt.append(f"Nodes: {metrics['total']}  Running: {metrics['running']}  "
                   f"Frozen: {metrics['frozen']}  Offline: {metrics['offline']}  "
                   f"Degraded: {metrics['degraded']}")
        if metrics["avg_latency"] is not None:
            txt.append(f"Avg Latency: {int(metrics['avg_latency'])} ms")
        if metrics["avg_jitter"] is not None:
            txt.append(f"Avg Jitter: {int(metrics['avg_jitter'])} ms")
        if metrics["avg_loss"] is not None:
            txt.append(f"Avg Loss: {int(metrics['avg_loss'] * 100)}%")
        txt.append(f"Global Anomaly: {metrics['avg_anomaly']:.2f}")
        txt.append(f"Swarm Health: {int(metrics['swarm_health'] * 100)}%")
        self.metrics_label.config(text="\n".join(txt))

        # Machines table
        self.tree.delete(*self.tree.get_children())
        for m in self.core.machines:
            threat_label = "OK"
            if m.threat_score >= 80:
                threat_label = "CRITICAL"
            elif m.threat_score >= 60:
                threat_label = "HIGH"
            elif m.threat_score >= 40:
                threat_label = "ELEVATED"

            latency_text = "—" if m.latency is None else f"{m.latency} ms"
            anomaly_text = f"{m.anomaly_score:.2f}"
            fused_text = f"{m.fused_anomaly:.2f}"
            pred_text = f"{m.prediction_label} ({int(m.prediction_confidence * 100)}%)"

            self.tree.insert(
                "",
                "end",
                values=(
                    m.address,
                    m.status,
                    latency_text,
                    f"{threat_label} ({m.threat_score})",
                    m.behavior_profile,
                    anomaly_text,
                    fused_text,
                    pred_text,
                    m.last_root_cause,
                )
            )

        # Scanner status
        self.scan_status_label.config(text=f"Last Scan: {self.core.last_scan_summary}")

        # Gaming status
        if self.core.gaming_auto:
            state = "ACTIVE" if self.core.gaming_active else "IDLE"
            self.gaming_status_label.config(
                text=f"Gaming Auto-Detect: ON\nCurrent State: {state}"
            )
        else:
            self.gaming_status_label.config(
                text="Gaming Auto-Detect: OFF\nCurrent State: MANUAL / DISABLED"
            )

        # Reschedule
        self.after(1000, self.refresh_ui)

# ---------- MAIN ----------

def main():
    core = CockpitCore()
    app = MagicBoxApp(core)
    app.mainloop()

if __name__ == "__main__":
    main()

