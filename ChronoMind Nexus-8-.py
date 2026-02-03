import sys
import os
import time
import json
import threading
import socket
import traceback
import re
import psutil
import win32gui
import win32process
import requests
from pynput import mouse, keyboard
import tkinter as tk
from tkinter import ttk, messagebox

# =========================
# GLOBALS / CONFIG
# =========================

TCP_HOST = "127.0.0.1"
TCP_PORT = 9009
MAX_TIMELINE_ROWS = 600

DEFAULT_CONFIG = {
    "mode": "strict",
    "firewall": {
        "block_patterns": [
            "ignore\\s+all\\s+previous",
            "system\\s*:","assistant\\s*:","<script","data:","base64",
        ],
    },
}

config = DEFAULT_CONFIG.copy()

# =========================
# TCP EVENT BUS (SERVER)
# =========================

clients_lock = threading.Lock()
clients = []

def broadcast_event(event):
    data = json.dumps(event, ensure_ascii=False) + "\n"
    dead = []
    with clients_lock:
        for s, f in clients:
            try:
                f.write(data)
                f.flush()
            except Exception:
                dead.append((s, f))
        for d in dead:
            try: d[0].close()
            except: pass
            if d in clients:
                clients.remove(d)

def handle_bus_client(sock, addr):
    f_in = sock.makefile("r", encoding="utf-8")
    f_out = sock.makefile("w", encoding="utf-8")
    with clients_lock:
        clients.append((sock, f_out))
    try:
        for line in f_in:
            line = line.strip()
            if not line: continue
            try:
                event = json.loads(line)
            except Exception:
                traceback.print_exc()
                continue
            broadcast_event(event)
    finally:
        with clients_lock:
            clients[:] = [c for c in clients if c[0] is not sock]
        try: sock.close()
        except: pass

def bus_server_loop():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((TCP_HOST, TCP_PORT))
    srv.listen(10)
    while True:
        try:
            sock, addr = srv.accept()
            threading.Thread(target=handle_bus_client, args=(sock, addr), daemon=True).start()
        except Exception:
            traceback.print_exc()
            time.sleep(1)

# =========================
# TCP CLIENT
# =========================

class EventBus:
    def __init__(self):
        self.callbacks = []

    def emit(self, event):
        for cb in self.callbacks:
            cb(event)

bus = EventBus()

def tcp_client_loop():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((TCP_HOST, TCP_PORT))
            f = sock.makefile("r", encoding="utf-8")
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    event = json.loads(line)
                    bus.emit(event)
                except Exception:
                    traceback.print_exc()
            f.close()
            sock.close()
        except Exception:
            time.sleep(2)

# =========================
# EVENT PRODUCER
# =========================

class EventProducer:
    def __init__(self):
        self.sock = None
        self.file = None

    def connect(self):
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((TCP_HOST, TCP_PORT))
                self.file = self.sock.makefile("w", encoding="utf-8")
                return
            except Exception:
                time.sleep(2)

    def send_event(self, event):
        if self.file is None: self.connect()
        try:
            self.file.write(json.dumps(event) + "\n")
            self.file.flush()
        except Exception:
            try: self.file.close(); self.sock.close()
            except: pass
            self.file = None; self.sock = None
            self.connect()

producer = EventProducer()

def get_active_window_info():
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc = psutil.Process(pid).name()
        return title, proc
    except:
        return "", ""

def build_event(event_type, raw_text, source_name, extra=None):
    ts = time.time()
    event = {
        "timestamp": ts,
        "raw": raw_text,
        "clean": raw_text,
        "hidden": False,
        "decision": "allow",
        "risk": 0,
        "domain": "",
        "source": {"name": source_name},
        "meta": {"type": event_type},
    }
    if extra: event["meta"].update(extra)
    return event

# =========================
# PRODUCER HOOKS
# =========================

def monitor_window_focus():
    producer.connect()
    last_title, last_proc = "", ""
    while True:
        title, proc = get_active_window_info()
        if title != last_title or proc != last_proc:
            last_title, last_proc = title, proc
            raw = f"WindowFocus: {proc} | {title}"
            event = build_event("window_focus", raw, proc, extra={"window_title": title})
            producer.send_event(event)
        time.sleep(0.1)

def on_click(x, y, button, pressed):
    if not pressed: return
    title, proc = get_active_window_info()
    raw = f"Click: {proc} | {title} | {button} @ ({x},{y})"
    event = build_event("mouse_click", raw, proc, extra={"x": x, "y": y, "button": str(button), "window_title": title})
    producer.send_event(event)

def on_press(key):
    title, proc = get_active_window_info()
    try: key_str = key.char
    except: key_str = str(key)
    raw = f"Keypress: {proc} | {title} | {key_str}"
    event = build_event("key_press", raw, proc, extra={"key": key_str, "window_title": title})
    producer.send_event(event)

def start_producer_hooks():
    threading.Thread(target=monitor_window_focus, daemon=True).start()
    mouse.Listener(on_click=on_click).start()
    keyboard.Listener(on_press=on_press).start()

# =========================
# TKINTER GUI
# =========================

class Cockpit(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Click Intelligence Cockpit")
        self.geometry("1100x650")

        self.mode = tk.StringVar(value=config["mode"])
        self.patterns = config["firewall"]["block_patterns"].copy()
        self.domains = {}

        # --- Top bar ---
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(top_frame, text="Mode:").pack(side=tk.LEFT)
        mode_cb = ttk.Combobox(top_frame, textvariable=self.mode, values=["strict","normal","learning"], width=12)
        mode_cb.pack(side=tk.LEFT, padx=2)

        # --- Main layout ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left overlay
        overlay_frame = tk.LabelFrame(main_frame, text="Overlay")
        overlay_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.overlay_clean = tk.Label(overlay_frame, text="", anchor="w", justify="left")
        self.overlay_clean.pack(fill=tk.X)
        self.overlay_raw = tk.Label(overlay_frame, text="", anchor="w", justify="left")
        self.overlay_raw.pack(fill=tk.X)
        self.overlay_flags = tk.Label(overlay_frame, text="", anchor="w", justify="left")
        self.overlay_flags.pack(fill=tk.X)

        # Center timeline
        timeline_frame = tk.LabelFrame(main_frame, text="Timeline")
        timeline_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.timeline = ttk.Treeview(timeline_frame, columns=("Time","Source","Domain","Text","Decision","Risk"), show="headings")
        for col in self.timeline["columns"]: self.timeline.heading(col, text=col)
        self.timeline.pack(fill=tk.BOTH, expand=True)

        # Right tabs
        right_tabs = ttk.Notebook(main_frame)
        right_tabs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Domain heatmap tab ---
        heatmap_tab = tk.Frame(right_tabs)
        right_tabs.add(heatmap_tab, text="Domains")
        self.domain_tree = ttk.Treeview(heatmap_tab, columns=("Total","Blocked","Risk"), show="headings")
        for col in self.domain_tree["columns"]: self.domain_tree.heading(col, text=col)
        self.domain_tree.pack(fill=tk.BOTH, expand=True)

        # --- Rules editor tab ---
        rules_tab = tk.Frame(right_tabs)
        right_tabs.add(rules_tab, text="Rules")
        self.rules_tree = ttk.Treeview(rules_tab, columns=("Pattern","Valid"), show="headings")
        self.rules_tree.heading("Pattern", text="Pattern")
        self.rules_tree.heading("Valid", text="Valid")
        self.rules_tree.pack(fill=tk.BOTH, expand=True)
        for p in self.patterns:
            self._add_rule_row(p)
        add_frame = tk.Frame(rules_tab)
        add_frame.pack(fill=tk.X)
        self.add_input = tk.Entry(add_frame)
        self.add_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(add_frame, text="Add Rule", command=self.add_rule).pack(side=tk.LEFT)

        # Bind bus events
        bus.callbacks.append(self.handle_event)

    def _add_rule_row(self, pattern):
        valid = self._validate_pattern(pattern)
        self.rules_tree.insert("", tk.END, values=(pattern, "✔" if valid else "✖"))

    def _validate_pattern(self, pattern):
        try: re.compile(pattern); return True
        except: return False

    def add_rule(self):
        rule = self.add_input.get().strip()
        if not rule: return
        if not self._validate_pattern(rule):
            messagebox.showwarning("Invalid Regex","Pattern is not a valid regex")
            return
        self.patterns.append(rule)
        self._add_rule_row(rule)
        self.add_input.delete(0, tk.END)

    def update_domain(self, event):
        domain = event.get("domain")
        if not domain: return
        d = self.domains.setdefault(domain, {"total":0,"blocked":0,"risk":[]})
        d["total"] += 1
        if event.get("decision") == "block": d["blocked"] += 1
        d["risk"].append(event.get("risk",0))
        self._refresh_domains()

    def _refresh_domains(self):
        for row in self.domain_tree.get_children(): self.domain_tree.delete(row)
        for domain, stats in self.domains.items():
            avg_risk = sum(stats["risk"])/len(stats["risk"]) if stats["risk"] else 0
            self.domain_tree.insert("", tk.END, values=(stats["total"], stats["blocked"], f"{avg_risk:.2f}"), text=domain)

    def handle_event(self, event):
        clean = event.get("clean","")
        raw = event.get("raw","")
        decision = event.get("decision","allow")
        risk = event.get("risk",0)
        domain = event.get("domain","")
        source_name = event.get("source",{}).get("name","")
        ts = time.strftime("%H:%M:%S", time.localtime(event.get("timestamp", time.time())))
        flags = decision.upper()
        self.overlay_clean.config(text=clean)
        self.overlay_raw.config(text=raw)
        self.overlay_flags.config(text=flags)
        self.timeline.insert("", tk.END, values=(ts, source_name, domain, clean[:50], decision, str(risk)))
        if self.timeline.get_children().__len__() > MAX_TIMELINE_ROWS:
            first = self.timeline.get_children()[0]
            self.timeline.delete(first)
        self.update_domain(event)

# =========================
# MAIN
# =========================

def main():
    threading.Thread(target=bus_server_loop, daemon=True).start()
    start_producer_hooks()
    threading.Thread(target=tcp_client_loop, daemon=True).start()

    app = Cockpit()
    app.mainloop()

if __name__ == "__main__":
    main()
