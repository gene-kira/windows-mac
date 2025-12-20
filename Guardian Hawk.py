#!/usr/bin/env python3
"""
Guardian Unified

Single-file, cross-platform system guardian with:
- Self-bootstrapping dependency install
- Profiling and optimization engine
- Prediction engine (forecasts what the user/system will need)
- Per-program profiles (latency_critical, normal, background)
- Security/anomaly monitor
- Optional local HTTP proxy to block silent downloads
- Global manual override (disable all enforcement)
- Per-program manual override (exempt specific processes from enforcement)
- Persistent memory (SQLite) for settings, overrides, profiles, and learned patterns
- Per-user autostart at login (Windows, Linux, macOS)
- Compact, fixed-size GUI (750x550) with scrollbars (fits on 800x600 screens)

Target: Windows, Linux, macOS.
"""

import sys
import subprocess
import importlib
import threading
import time
import socket
import http.server
import socketserver
import queue
import platform
import traceback
import os
import sqlite3

# =========================
# Dependency autoloader
# =========================

REQUIRED_PACKAGES = [
    "psutil",
    "requests"
]

def ensure_dependencies():
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_dependencies()

import psutil
import requests

# Tkinter import (built-in on most platforms)
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None


# =========================
# Persistent storage (SQLite in script folder)
# =========================

class GuardianStorage:
    """
    Handles persistent storage of:
    - Settings (policy_mode, aggressive, etc.)
    - Predictor usage history (process usage patterns)
    - Per-program overrides
    - Per-program profiles
    """

    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "guardian_memory.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()
        self.lock = threading.Lock()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usage_history (
                process_name TEXT,
                hour INTEGER,
                weekday INTEGER,
                used INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS overrides (
                process_name TEXT PRIMARY KEY
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                process_name TEXT PRIMARY KEY,
                category TEXT
            )
        """)
        self.conn.commit()

    # ----- Settings -----

    def load_settings(self):
        cur = self.conn.cursor()
        cur.execute("SELECT key, value FROM settings")
        rows = cur.fetchall()
        return {k: v for (k, v) in rows}

    def save_setting(self, key, value):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value))
            )
            self.conn.commit()

    # ----- Usage history -----

    def append_usage(self, process_name, hour, weekday, used):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO usage_history(process_name, hour, weekday, used) VALUES(?, ?, ?, ?)",
                (process_name, int(hour), int(weekday), int(used))
            )
            self.conn.commit()

    def load_usage_history(self, limit=5000):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT process_name, hour, weekday, used FROM usage_history "
            "ORDER BY rowid DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
        rows.reverse()
        return rows

    # ----- Overrides -----

    def load_overrides(self):
        cur = self.conn.cursor()
        cur.execute("SELECT process_name FROM overrides")
        rows = cur.fetchall()
        return {name for (name,) in rows}

    def add_override(self, process_name):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO overrides(process_name) VALUES(?) "
                "ON CONFLICT(process_name) DO NOTHING",
                (process_name,)
            )
            self.conn.commit()

    def remove_override(self, process_name):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM overrides WHERE process_name = ?", (process_name,))
            self.conn.commit()

    # ----- Profiles -----

    def load_profiles(self):
        cur = self.conn.cursor()
        cur.execute("SELECT process_name, category FROM profiles")
        rows = cur.fetchall()
        return {name: category for (name, category) in rows}

    def set_profile(self, process_name, category):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO profiles(process_name, category) VALUES(?, ?) "
                "ON CONFLICT(process_name) DO UPDATE SET category=excluded.category",
                (process_name, category)
            )
            self.conn.commit()

    def remove_profile(self, process_name):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM profiles WHERE process_name = ?", (process_name,))
            self.conn.commit()


# =========================
# Core profiling engine
# =========================

class Profiler:
    """
    Collects system and process metrics, builds lightweight profiles.
    """

    def __init__(self):
        self.last_sample_time = None
        self.process_stats = {}
        self.system_profile = {
            "idle_cpu_pattern": [],
            "busy_cpu_pattern": [],
        }
        self.lock = threading.Lock()

    def sample(self):
        with self.lock:
            now = time.time()
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
            except Exception:
                cpu_percent = 0.0

            try:
                procs = list(psutil.process_iter(
                    ["pid", "name", "username", "cpu_percent", "memory_info", "status"]
                ))
            except Exception:
                procs = []

            if cpu_percent < 10:
                self.system_profile["idle_cpu_pattern"].append(cpu_percent)
            else:
                self.system_profile["busy_cpu_pattern"].append(cpu_percent)

            for key in self.system_profile:
                if len(self.system_profile[key]) > 500:
                    self.system_profile[key] = self.system_profile[key][-500:]

            for p in procs:
                info = p.info
                pid = info.get("pid")
                if pid is None:
                    continue
                self.process_stats[pid] = {
                    "name": info.get("name") or f"pid_{pid}",
                    "user": info.get("username") or "unknown",
                    "cpu": info.get("cpu_percent") or 0.0,
                    "rss": getattr(info.get("memory_info"), "rss", 0),
                    "status": info.get("status"),
                    "last_seen": now,
                }

            active_pids = {p.pid for p in procs}
            to_delete = [pid for pid in self.process_stats if pid not in active_pids]
            for pid in to_delete:
                del self.process_stats[pid]

            self.last_sample_time = now

    def get_idle_processes(self, idle_seconds=300, cpu_threshold=1.0):
        idle = []
        now = time.time()
        with self.lock:
            for pid, stats in self.process_stats.items():
                last_seen = stats.get("last_seen", 0)
                cpu_usage = stats.get("cpu", 0.0)
                if (now - last_seen > idle_seconds) and cpu_usage < cpu_threshold:
                    idle.append((pid, stats))
        return idle

    def get_basic_system_stats(self):
        try:
            cpu = psutil.cpu_percent(interval=0.0)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "cpu": cpu,
                "memory_used": mem.used,
                "memory_total": mem.total,
                "disk_used": disk.used,
                "disk_total": disk.total,
            }
        except Exception:
            return {
                "cpu": 0.0,
                "memory_used": 0,
                "memory_total": 0,
                "disk_used": 0,
                "disk_total": 0,
            }

    def get_current_process_names(self):
        with self.lock:
            names = {stats.get("name", f"pid_{pid}") for pid, stats in self.process_stats.items()}
        return sorted(names)


# =========================
# Prediction / forecasting engine
# =========================

class Predictor:
    """
    Learns patterns of app usage and system load to forecast what is needed soon.
    """

    def __init__(self, profiler, storage=None):
        self.profiler = profiler
        self.storage = storage
        self.usage_history = {}  # name -> list of (hour, weekday, used)
        self.lock = threading.Lock()

    def load_usage_history(self, records):
        with self.lock:
            for name, hour, weekday, used in records:
                history = self.usage_history.setdefault(name, [])
                history.append((int(hour), int(weekday), int(used)))
                if len(history) > 1000:
                    self.usage_history[name] = history[-1000:]

    def record_usage_snapshot(self):
        with self.lock:
            now = time.localtime()
            hour = now.tm_hour
            weekday = now.tm_wday

            for pid, stats in self.profiler.process_stats.items():
                name = stats.get("name", f"pid_{pid}")
                cpu = stats.get("cpu", 0.0)
                active = cpu > 1.0
                used_flag = 1 if active else 0

                history = self.usage_history.setdefault(name, [])
                history.append((hour, weekday, used_flag))
                if len(history) > 1000:
                    self.usage_history[name] = history[-1000:]

                if self.storage is not None:
                    try:
                        self.storage.append_usage(name, hour, weekday, used_flag)
                    except Exception:
                        pass

    def _score_time_match(self, history, hour, weekday):
        if not history:
            return 0.0
        score = 0
        count = 0
        for h, w, used in history:
            hour_diff = min(abs(h - hour), 24 - abs(h - hour))
            if w == weekday and hour_diff <= 2:
                weight = 2
            elif hour_diff <= 1:
                weight = 1.5
            elif hour_diff <= 3:
                weight = 1.0
            else:
                weight = 0.2
            score += weight * used
            count += weight
        if count == 0:
            return 0.0
        return score / count

    def get_likely_needed_processes(self, threshold=0.5):
        with self.lock:
            now = time.localtime()
            hour = now.tm_hour
            weekday = now.tm_wday
            likely = set()
            for name, history in self.usage_history.items():
                score = self._score_time_match(history, hour, weekday)
                if score >= threshold:
                    likely.add(name)
            return likely

    def get_forecast_summary(self):
        likely = self.get_likely_needed_processes(threshold=0.6)
        return sorted(list(likely))[:10]


# =========================
# Resource management
# =========================

class ResourceManager:
    """
    Decides when to optimize processes/services, and optionally applies changes.
    Uses prediction, per-program overrides, and per-program profiles.
    """

    def __init__(self, profiler, action_log_queue, aggressive=False, predictor=None, agent=None):
        self.profiler = profiler
        self.action_log_queue = action_log_queue
        self.aggressive = aggressive
        self.os = platform.system()
        self.recommendations = []
        self.lock = threading.Lock()
        self.predictor = predictor
        self.agent = agent

    def optimize(self):
        with self.lock:
            self.recommendations.clear()

        idle_procs = self.profiler.get_idle_processes(
            idle_seconds=300, cpu_threshold=1.0
        )

        likely_needed = set()
        if self.predictor is not None:
            likely_needed = self.predictor.get_likely_needed_processes(threshold=0.5)

        global_override = False
        per_program_overrides = set()
        profiles = {}
        if self.agent is not None:
            global_override = getattr(self.agent, "override_active", False)
            per_program_overrides = getattr(self.agent, "process_overrides", set())
            profiles = getattr(self.agent, "program_profiles", {})

        for pid, stats in idle_procs:
            name = stats.get("name", f"pid_{pid}")
            user = stats.get("user", "unknown")

            if user in ("SYSTEM", "root") or user.lower() in ("local service", "network service"):
                continue

            category = profiles.get(name, "normal")

            if name in per_program_overrides:
                rec = {
                    "action": "override_skip",
                    "pid": pid,
                    "name": name,
                    "reason": "per_program_override",
                    "category": category,
                }
                with self.lock:
                    self.recommendations.append(rec)
                continue

            if name in likely_needed:
                rec = {
                    "action": "keep_warm",
                    "pid": pid,
                    "name": name,
                    "reason": "predicted_needed_soon",
                    "category": category,
                }
                with self.lock:
                    self.recommendations.append(rec)
                continue

            if category == "latency_critical":
                rec = {
                    "action": "keep_critical",
                    "pid": pid,
                    "name": name,
                    "reason": "latency_critical_profile",
                    "category": category,
                }
                with self.lock:
                    self.recommendations.append(rec)
                continue

            effective_action = "consider_suspend"
            if global_override:
                effective_action = "observe_only"
            elif category == "background":
                effective_action = "aggressive_suspend"

            rec = {
                "action": effective_action,
                "pid": pid,
                "name": name,
                "reason": "idle_process_over_300s",
                "category": category,
            }

            with self.lock:
                self.recommendations.append(rec)

            if global_override:
                continue

            if self.aggressive or category == "background":
                self._safe_suspend_process(pid, rec)

    def _safe_suspend_process(self, pid, rec):
        if self.agent is not None:
            if getattr(self.agent, "override_active", False):
                self.action_log_queue.put(
                    f"Global override: skipping suspend of PID {pid} ({rec['name']})"
                )
                return
            per_overrides = getattr(self.agent, "process_overrides", set())
            if rec["name"] in per_overrides:
                self.action_log_queue.put(
                    f"Per-program override: skipping suspend of PID {pid} ({rec['name']})"
                )
                return
            profiles = getattr(self.agent, "program_profiles", {})
            if profiles.get(rec["name"]) == "latency_critical":
                self.action_log_queue.put(
                    f"Latency-critical profile: skipping suspend of PID {pid} ({rec['name']})"
                )
                return

        try:
            proc = psutil.Process(pid)
            if proc.status() in (psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING):
                proc.suspend()
                msg = f"SUSPENDED PID {pid} ({rec['name']}) [{rec.get('category','')}] - {rec['reason']}"
                self.action_log_queue.put(msg)
        except Exception as e:
            self.action_log_queue.put(
                f"FAILED TO SUSPEND PID {pid} ({rec['name']}): {e}"
            )

    def get_recommendations(self):
        with self.lock:
            return list(self.recommendations)


# =========================
# Security / anomaly monitor
# =========================

class SecurityMonitor:
    """
    Looks for suspicious processes and connections.
    Respects per-program overrides (marks as overridden).
    """

    def __init__(self, profiler, alert_queue, agent=None):
        self.profiler = profiler
        self.alert_queue = alert_queue
        self.suspicious_events = []
        self.lock = threading.Lock()
        self.known_safe_ports = {80, 443}
        self.agent = agent

    def scan(self):
        events = []
        try:
            conns = psutil.net_connections()
        except Exception:
            conns = []

        now = time.time()

        per_program_overrides = set()
        profiles = {}
        if self.agent is not None:
            per_program_overrides = getattr(self.agent, "process_overrides", set())
            profiles = getattr(self.agent, "program_profiles", {})

        for conn in conns:
            pid = conn.pid
            if not pid:
                continue
            raddr = conn.raddr
            if not raddr:
                continue

            proc_info = self.profiler.process_stats.get(pid)
            if not proc_info:
                continue

            name = proc_info.get("name", f"pid_{pid}")
            category = profiles.get(name, "normal")

            if raddr.port not in self.known_safe_ports:
                reason = "non_standard_remote_port"
                if name in per_program_overrides:
                    reason += "_overridden"
                if category == "latency_critical":
                    reason += "_latency_critical"

                event = {
                    "timestamp": now,
                    "pid": pid,
                    "name": name,
                    "remote": f"{raddr.ip}:{raddr.port}",
                    "reason": reason,
                    "category": category,
                }
                events.append(event)

        with self.lock:
            self.suspicious_events = events

        for e in events:
            msg = f"SUSPICIOUS: PID {e['pid']} {e['name']} [{e['category']}] -> {e['remote']} ({e['reason']})"
            self.alert_queue.put(msg)

    def get_events(self):
        with self.lock:
            return list(self.suspicious_events)


# =========================
# Local HTTP proxy for silent download control
# =========================

class GuardianHTTPProxyHandler(http.server.BaseHTTPRequestHandler):
    """
    Simple HTTP proxy, blocks some downloads unless global override is on.
    """

    guardian_instance = None

    def _override_active(self):
        gi = GuardianHTTPProxyHandler.guardian_instance
        return bool(gi and getattr(gi, "override_active", False))

    def _should_block(self, path, headers):
        if self._override_active():
            gi = GuardianHTTPProxyHandler.guardian_instance
            if gi:
                gi.log_security(f"Global override: allowing download {path}")
            return False

        host = headers.get("Host", "")
        url = f"http://{host}{path}"
        blocked_ext = (".exe", ".msi", ".bat", ".cmd", ".sh", ".js", ".zip")
        if any(url.lower().endswith(ext) for ext in blocked_ext):
            if GuardianHTTPProxyHandler.guardian_instance:
                GuardianHTTPProxyHandler.guardian_instance.log_security(
                    f"BLOCKED DOWNLOAD: {url} (extension match)"
                )
            return True
        return False

    def do_GET(self):
        if self._should_block(self.path, self.headers):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"Blocked by Guardian")
            return

        host = self.headers.get("Host")
        if not host:
            self.send_response(502)
            self.end_headers()
            return

        url = f"http://{host}{self.path}"
        try:
            resp = requests.get(url, headers=self._filtered_headers(), stream=True, timeout=10)
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() in ("transfer-encoding", "connection", "keep-alive",
                                 "proxy-authenticate", "proxy-authorization", "te",
                                 "trailers", "upgrade"):
                    continue
                self.send_header(k, v)
            self.end_headers()
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    self.wfile.write(chunk)
        except Exception as e:
            if GuardianHTTPProxyHandler.guardian_instance:
                GuardianHTTPProxyHandler.guardian_instance.log_security(
                    f"PROXY ERROR fetching {url}: {e}"
                )
            self.send_response(502)
            self.end_headers()

    def _filtered_headers(self):
        new_headers = {}
        for k, v in self.headers.items():
            if k.lower() == "host":
                continue
            new_headers[k] = v
        return new_headers

    def log_message(self, format, *args):
        if GuardianHTTPProxyHandler.guardian_instance:
            GuardianHTTPProxyHandler.guardian_instance.log_action(
                f"PROXY: " + (format % args)
            )


class GuardianHTTPProxy(threading.Thread):
    def __init__(self, host="127.0.0.1", port=8888):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.httpd = None
        self.running = False

    def run(self):
        try:
            with socketserver.ThreadingTCPServer((self.host, self.port), GuardianHTTPProxyHandler) as httpd:
                httpd.allow_reuse_address = True
                self.httpd = httpd
                self.running = True
                httpd.serve_forever()
        except Exception:
            self.running = False

    def stop(self):
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
        self.running = False


# =========================
# Autostart installer (per-user, all OS)
# =========================

def install_autostart():
    os_name = platform.system()
    script_path = os.path.abspath(__file__)
    python_exe = sys.executable

    if os_name == "Windows":
        try:
            import winreg
        except ImportError:
            print("winreg not available; cannot install autostart on Windows.")
            return

        run_key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        app_name = "GuardianUnified"
        cmd = f'"{python_exe}" "{script_path}"'

        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key_path, 0, winreg.KEY_SET_VALUE)
        except FileNotFoundError:
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, run_key_path)

        winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(key)
        print("Autostart installed in Windows registry (HKCU Run).")

    elif os_name == "Linux":
        home = os.path.expanduser("~")
        autostart_dir = os.path.join(home, ".config", "autostart")
        os.makedirs(autostart_dir, exist_ok=True)
        desktop_path = os.path.join(autostart_dir, "guardian_unified.desktop")

        cmd = f"{python_exe} {script_path}"

        desktop_content = f"""[Desktop Entry]
Type=Application
Exec={cmd}
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=Guardian Unified
Comment=System guardian
"""
        with open(desktop_path, "w", encoding="utf-8") as f:
            f.write(desktop_content)

        print(f"Autostart .desktop file created at: {desktop_path}")

    elif os_name == "Darwin":
        home = os.path.expanduser("~")
        launch_agents_dir = os.path.join(home, "Library", "LaunchAgents")
        os.makedirs(launch_agents_dir, exist_ok=True)
        plist_path = os.path.join(launch_agents_dir, "com.guardian.unified.plist")

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.guardian.unified</string>
    <key>ProgramArguments</key>
    <array>
      <string>{python_exe}</string>
      <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
  </dict>
</plist>
"""
        with open(plist_path, "w", encoding="utf-8") as f:
            f.write(plist_content)

        print(f"LaunchAgent plist created at: {plist_path}")
        print("You may need to run: launchctl load ~/Library/LaunchAgents/com.guardian.unified.plist")

    else:
        print(f"Autostart install not implemented for OS: {os_name}")


# =========================
# Guardian core agent
# =========================

class GuardianAgent:
    """
    Orchestrates profiling, prediction, optimization, security, proxy, and GUI.
    """

    def __init__(self):
        self.os = platform.system()

        self.storage = GuardianStorage()
        self.profiler = Profiler()

        self.action_log_queue = queue.Queue()
        self.security_log_queue = queue.Queue()

        self.override_active = False
        self.process_overrides = set()
        self.program_profiles = {}

        settings = self.storage.load_settings()
        policy_mode = settings.get("policy_mode", "balanced")
        aggressive_flag = settings.get("aggressive", "0")
        try:
            self.process_overrides = self.storage.load_overrides()
        except Exception:
            self.process_overrides = set()
        try:
            self.program_profiles = self.storage.load_profiles()
        except Exception:
            self.program_profiles = {}

        self.predictor = Predictor(self.profiler, storage=self.storage)
        try:
            records = self.storage.load_usage_history(limit=5000)
            self.predictor.load_usage_history(records)
        except Exception:
            pass

        aggressive = aggressive_flag == "1"
        self.resource_manager = ResourceManager(
            self.profiler,
            self.action_log_queue,
            aggressive=aggressive,
            predictor=self.predictor,
            agent=self
        )
        self.security_monitor = SecurityMonitor(
            self.profiler, self.security_log_queue, agent=self
        )

        self.stop_event = threading.Event()

        self.proxy = GuardianHTTPProxy(host="127.0.0.1", port=8888)
        GuardianHTTPProxyHandler.guardian_instance = self

        self.policy_mode = policy_mode

    def log_action(self, message: str):
        self.action_log_queue.put(message)

    def log_security(self, message: str):
        self.security_log_queue.put(message)

    def toggle_override(self):
        self.override_active = not self.override_active
        state = "ON" if self.override_active else "OFF"
        self.log_action(f"Global override toggled: {state}")

    def add_process_override(self, process_name: str):
        process_name = process_name.strip()
        if not process_name:
            return
        if process_name not in self.process_overrides:
            self.process_overrides.add(process_name)
            try:
                self.storage.add_override(process_name)
            except Exception:
                pass
            self.log_action(f"Per-program override added: {process_name}")

    def remove_process_override(self, process_name: str):
        process_name = process_name.strip()
        if process_name in self.process_overrides:
            self.process_overrides.remove(process_name)
            try:
                self.storage.remove_override(process_name)
            except Exception:
                pass
            self.log_action(f"Per-program override removed: {process_name}")

    def set_program_profile(self, process_name: str, category: str):
        process_name = process_name.strip()
        if not process_name:
            return
        if category not in ("latency_critical", "normal", "background"):
            return
        self.program_profiles[process_name] = category
        try:
            self.storage.set_profile(process_name, category)
        except Exception:
            pass
        self.log_action(f"Profile set: {process_name} -> {category}")

    def remove_program_profile(self, process_name: str):
        process_name = process_name.strip()
        if process_name in self.program_profiles:
            del self.program_profiles[process_name]
            try:
                self.storage.remove_profile(process_name)
            except Exception:
                pass
            self.log_action(f"Profile removed: {process_name}")

    def start(self):
        threading.Thread(target=self._profiling_loop, daemon=True).start()
        threading.Thread(target=self._prediction_loop, daemon=True).start()
        threading.Thread(target=self._resource_loop, daemon=True).start()
        threading.Thread(target=self._security_loop, daemon=True).start()
        threading.Thread(target=self._proxy_loop, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        try:
            self.proxy.stop()
        except Exception:
            pass

    def _profiling_loop(self):
        while not self.stop_event.is_set():
            try:
                self.profiler.sample()
            except Exception as e:
                self.log_action(f"Profiler error: {e}")
            time.sleep(2)

    def _prediction_loop(self):
        while not self.stop_event.is_set():
            try:
                self.predictor.record_usage_snapshot()
            except Exception as e:
                self.log_action(f"Predictor error: {e}")
            time.sleep(10)

    def _resource_loop(self):
        while not self.stop_event.is_set():
            try:
                self.resource_manager.optimize()
            except Exception as e:
                self.log_action(f"ResourceManager error: {e}")
            time.sleep(5)

    def _security_loop(self):
        while not self.stop_event.is_set():
            try:
                self.security_monitor.scan()
            except Exception as e:
                self.log_security(f"SecurityMonitor error: {e}")
            time.sleep(3)

    def _proxy_loop(self):
        try:
            self.proxy.start()
            time.sleep(1)
            if self.proxy.running:
                self.log_action("HTTP proxy running on 127.0.0.1:8888")
            else:
                self.log_action("HTTP proxy failed to start")
        except Exception as e:
            self.log_action(f"HTTP proxy error: {e}")

    def get_system_stats(self):
        return self.profiler.get_basic_system_stats()

    def get_recommendations(self):
        return self.resource_manager.get_recommendations()

    def get_security_events(self):
        return self.security_monitor.get_events()

    def get_forecast_summary(self):
        return self.predictor.get_forecast_summary()

    def get_current_process_names(self):
        return self.profiler.get_current_process_names()

    def get_process_overrides(self):
        return sorted(self.process_overrides)

    def get_program_profiles(self):
        return dict(self.program_profiles)

    def set_policy_mode(self, mode: str):
        if mode in ("relaxed", "balanced", "strict"):
            self.policy_mode = mode
            if mode == "relaxed":
                self.resource_manager.aggressive = False
            elif mode == "balanced":
                self.resource_manager.aggressive = False
            elif mode == "strict":
                self.resource_manager.aggressive = True

            try:
                self.storage.save_setting("policy_mode", mode)
                self.storage.save_setting(
                    "aggressive",
                    "1" if self.resource_manager.aggressive else "0"
                )
            except Exception:
                pass

            self.log_action(f"Policy mode set to {mode}")


# =========================
# GUI (compact, fixed-size, scrollable)
# =========================

class GuardianGUI:
    """
    Tkinter GUI:
    - Fixed-size 750x550 (fits on 800x600)
    - Scrollbars for logs and recommendations
    - Global override
    - Per-program overrides and profiles
    """

    def __init__(self, agent: GuardianAgent):
        self.agent = agent
        if tk is None or ttk is None:
            print("Tkinter not available, running headless.")
            self.headless = True
            return
        self.headless = False

        self.root = tk.Tk()
        self.root.title("Guardian Unified")

        # Fixed size, no resize
        self.root.geometry("750x550")
        self.root.resizable(False, False)

        # ========== Top bar (stats, policy, override, forecast) ==========
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side="top", fill="x", padx=5, pady=3)

        self.cpu_label = ttk.Label(top_frame, text="CPU: 0%")
        self.cpu_label.pack(side="left", padx=3)

        self.mem_label = ttk.Label(top_frame, text="Mem: 0 / 0")
        self.mem_label.pack(side="left", padx=3)

        self.disk_label = ttk.Label(top_frame, text="Disk: 0 / 0")
        self.disk_label.pack(side="left", padx=3)

        self.policy_var = tk.StringVar(value=self.agent.policy_mode)
        policy_label = ttk.Label(top_frame, text="Policy:")
        policy_label.pack(side="left", padx=(10, 2))
        policy_menu = ttk.OptionMenu(
            top_frame,
            self.policy_var,
            self.agent.policy_mode,
            "relaxed",
            "balanced",
            "strict",
            command=self._on_policy_change
        )
        policy_menu.config(width=8)
        policy_menu.pack(side="left")

        self.override_button = ttk.Button(
            top_frame,
            text="Override: OFF",
            width=13,
            command=self._on_override_click
        )
        self.override_button.pack(side="left", padx=(10, 2))

        self.forecast_label = ttk.Label(top_frame, text="Forecast: (learning...)")
        self.forecast_label.pack(side="left", padx=(10, 2))

        # ========== Middle: Logs (actions & security) ==========
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(side="top", fill="x", padx=5, pady=3)

        # Actions log (with scrollbar)
        action_frame = ttk.LabelFrame(mid_frame, text="Resource actions")
        action_frame.pack(side="left", fill="both", expand=False, padx=3, pady=3)

        self.action_text = tk.Text(action_frame, height=10, width=40, wrap="none")
        self.action_text.grid(row=0, column=0, sticky="nsew")
        action_scroll_y = ttk.Scrollbar(action_frame, orient="vertical", command=self.action_text.yview)
        action_scroll_y.grid(row=0, column=1, sticky="ns")
        self.action_text.configure(yscrollcommand=action_scroll_y.set)
        action_frame.rowconfigure(0, weight=1)
        action_frame.columnconfigure(0, weight=1)

        # Security log (with scrollbar)
        sec_frame = ttk.LabelFrame(mid_frame, text="Security / anomalies")
        sec_frame.pack(side="left", fill="both", expand=False, padx=3, pady=3)

        self.security_text = tk.Text(sec_frame, height=10, width=40, wrap="none", fg="red")
        self.security_text.grid(row=0, column=0, sticky="nsew")
        sec_scroll_y = ttk.Scrollbar(sec_frame, orient="vertical", command=self.security_text.yview)
        sec_scroll_y.grid(row=0, column=1, sticky="ns")
        self.security_text.configure(yscrollcommand=sec_scroll_y.set)
        sec_frame.rowconfigure(0, weight=1)
        sec_frame.columnconfigure(0, weight=1)

        # ========== Bottom: Recommendations + overrides/profiles ==========
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side="top", fill="both", expand=True, padx=5, pady=3)

        # Left: recommendations (with scrollbar)
        reco_frame = ttk.LabelFrame(bottom_frame, text="Recommendations")
        reco_frame.pack(side="left", fill="both", expand=True, padx=3, pady=3)

        self.reco_text = tk.Text(reco_frame, height=10, width=50, wrap="none")
        self.reco_text.grid(row=0, column=0, sticky="nsew")
        reco_scroll_y = ttk.Scrollbar(reco_frame, orient="vertical", command=self.reco_text.yview)
        reco_scroll_y.grid(row=0, column=1, sticky="ns")
        self.reco_text.configure(yscrollcommand=reco_scroll_y.set)
        reco_frame.rowconfigure(0, weight=1)
        reco_frame.columnconfigure(0, weight=1)

        # Right-click override from recommendation line
        self.reco_text.bind("<Button-3>", self._on_reco_right_click)

        # Right: per-program override/profile controls (fixed)
        control_frame = ttk.LabelFrame(bottom_frame, text="Overrides & Profiles")
        control_frame.pack(side="left", fill="y", expand=False, padx=3, pady=3)

        ttk.Label(control_frame, text="Process:").grid(row=0, column=0, columnspan=2, sticky="w")

        self.proc_override_var = tk.StringVar()
        self.proc_override_combo = ttk.Combobox(
            control_frame,
            textvariable=self.proc_override_var,
            values=[],
            width=24
        )
        self.proc_override_combo.grid(row=1, column=0, columnspan=2, sticky="we", pady=2)

        self.refresh_proc_button = ttk.Button(
            control_frame,
            text="Refresh",
            width=10,
            command=self._refresh_process_list
        )
        self.refresh_proc_button.grid(row=2, column=0, sticky="w", pady=2)

        self.add_override_button = ttk.Button(
            control_frame,
            text="Add override",
            width=12,
            command=self._on_add_proc_override
        )
        self.add_override_button.grid(row=2, column=1, sticky="e", pady=2)

        ttk.Label(control_frame, text="Profile:").grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.profile_var = tk.StringVar(value="normal")
        self.profile_combo = ttk.Combobox(
            control_frame,
            textvariable=self.profile_var,
            values=["latency_critical", "normal", "background"],
            width=22,
            state="readonly"
        )
        self.profile_combo.grid(row=4, column=0, columnspan=2, sticky="we", pady=2)

        self.set_profile_button = ttk.Button(
            control_frame,
            text="Set profile",
            width=12,
            command=self._on_set_profile
        )
        self.set_profile_button.grid(row=5, column=0, sticky="w", pady=2)

        self.remove_profile_button = ttk.Button(
            control_frame,
            text="Remove profile",
            width=12,
            command=self._on_remove_profile
        )
        self.remove_profile_button.grid(row=5, column=1, sticky="e", pady=2)

        ttk.Label(control_frame, text="Overrides:").grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.overrides_listbox = tk.Listbox(control_frame, height=5, width=24)
        self.overrides_listbox.grid(row=7, column=0, columnspan=2, sticky="we")
        self.overrides_listbox.bind("<Double-Button-1>", self._on_overrides_double_click)

        self.remove_override_button = ttk.Button(
            control_frame,
            text="Remove override",
            width=16,
            command=self._on_remove_proc_override
        )
        self.remove_override_button.grid(row=8, column=0, columnspan=2, pady=2)

        ttk.Label(control_frame, text="Profiles:").grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.profiles_listbox = tk.Listbox(control_frame, height=5, width=24)
        self.profiles_listbox.grid(row=10, column=0, columnspan=2, sticky="we")
        self.profiles_listbox.bind("<Double-Button-1>", self._on_profiles_double_click)

        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)

        self._refresh_process_list()
        self._refresh_overrides_list()
        self._refresh_profiles_list()

        self.root.after(1000, self._update_ui)

    def _on_policy_change(self, value):
        try:
            self.agent.set_policy_mode(value)
        except Exception:
            traceback.print_exc()

    def _on_override_click(self):
        try:
            self.agent.toggle_override()
            state = "ON" if self.agent.override_active else "OFF"
            self.override_button.config(text=f"Override: {state}")
        except Exception:
            traceback.print_exc()

    def _refresh_process_list(self):
        try:
            procs = self.agent.get_current_process_names()
            self.proc_override_combo["values"] = procs
        except Exception:
            traceback.print_exc()

    def _refresh_overrides_list(self):
        try:
            self.overrides_listbox.delete(0, tk.END)
            for name in self.agent.get_process_overrides():
                self.overrides_listbox.insert(tk.END, name)
        except Exception:
            traceback.print_exc()

    def _refresh_profiles_list(self):
        try:
            self.profiles_listbox.delete(0, tk.END)
            profiles = self.agent.get_program_profiles()
            for name, cat in sorted(profiles.items()):
                self.profiles_listbox.insert(tk.END, f"{name} [{cat}]")
        except Exception:
            traceback.print_exc()

    def _on_add_proc_override(self):
        try:
            name = self.proc_override_var.get().strip()
            if name:
                self.agent.add_process_override(name)
                self._refresh_overrides_list()
        except Exception:
            traceback.print_exc()

    def _on_remove_proc_override(self):
        try:
            selection = self.overrides_listbox.curselection()
            if not selection:
                return
            index = selection[0]
            name = self.overrides_listbox.get(index)
            self.agent.remove_process_override(name)
            self._refresh_overrides_list()
        except Exception:
            traceback.print_exc()

    def _on_set_profile(self):
        try:
            name = self.proc_override_var.get().strip()
            category = self.profile_var.get().strip()
            if name and category:
                self.agent.set_program_profile(name, category)
                self._refresh_profiles_list()
        except Exception:
            traceback.print_exc()

    def _on_remove_profile(self):
        try:
            selection = self.profiles_listbox.curselection()
            if not selection:
                return
            index = selection[0]
            entry = self.profiles_listbox.get(index)
            name = entry.split("[", 1)[0].strip()
            self.agent.remove_program_profile(name)
            self._refresh_profiles_list()
        except Exception:
            traceback.print_exc()

    def _on_overrides_double_click(self, event):
        self._on_remove_proc_override()

    def _on_profiles_double_click(self, event):
        self._on_remove_profile()

    def _on_reco_right_click(self, event):
        try:
            index = self.reco_text.index(f"@{event.x},{event.y}")
            line_start = f"{index.split('.')[0]}.0"
            line_end = f"{index.split('.')[0]}.end"
            line = self.reco_text.get(line_start, line_end).strip()
            parts = line.split()
            if len(parts) >= 5 and parts[0].startswith("[") and parts[1] == "PID":
                if "-" in parts:
                    dash_index = parts.index("-")
                    name_parts = parts[3:dash_index]
                else:
                    name_parts = parts[3:]
                name = " ".join(name_parts).strip("[]")
                if name:
                    self.agent.add_process_override(name)
                    self._refresh_overrides_list()
        except Exception:
            traceback.print_exc()

    def _update_ui(self):
        if self.headless:
            return

        try:
            stats = self.agent.get_system_stats()
            cpu = stats["cpu"]
            mem_used = stats["memory_used"]
            mem_total = stats["memory_total"]
            disk_used = stats["disk_used"]
            disk_total = stats["disk_total"]

            def fmt_bytes(b):
                for unit in ["B", "KB", "MB", "GB", "TB"]:
                    if b < 1024:
                        return f"{b:.1f}{unit}"
                    b /= 1024
                return f"{b:.1f}PB"

            self.cpu_label.config(text=f"CPU: {cpu:.1f}%")
            self.mem_label.config(
                text=f"Mem: {fmt_bytes(mem_used)} / {fmt_bytes(mem_total)}"
            )
            self.disk_label.config(
                text=f"Disk: {fmt_bytes(disk_used)} / {fmt_bytes(disk_total)}"
            )

            try:
                forecast = self.agent.get_forecast_summary()
                if forecast:
                    self.forecast_label.config(text="Forecast: " + ", ".join(forecast))
                else:
                    self.forecast_label.config(text="Forecast: (no strong pattern yet)")
            except Exception:
                self.forecast_label.config(text="Forecast: (error)")

            state = "ON" if self.agent.override_active else "OFF"
            self.override_button.config(text=f"Override: {state}")

            while not self.agent.action_log_queue.empty():
                msg = self.agent.action_log_queue.get_nowait()
                self.action_text.insert("end", msg + "\n")
                self.action_text.see("end")

            while not self.agent.security_log_queue.empty():
                msg = self.agent.security_log_queue.get_nowait()
                self.security_text.insert("end", msg + "\n")
                self.security_text.see("end")

            self.reco_text.delete("1.0", "end")
            recos = self.agent.get_recommendations()
            for rec in recos:
                line = f"[{rec['action']}] PID {rec['pid']} {rec['name']} [{rec.get('category','')}] - {rec['reason']}\n"
                self.reco_text.insert("end", line)
        except Exception:
            traceback.print_exc()

        self.root.after(1000, self._update_ui)

    def run(self):
        if self.headless:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            self.root.mainloop()


# =========================
# Main entry point
# =========================

def main():
    if "--install-autostart" in sys.argv:
        install_autostart()
        return

    agent = GuardianAgent()
    agent.start()

    gui = GuardianGUI(agent)
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    main()

