# 🔮 Codex Sentinel: Autonomous Data Motor Daemon

# 🧬 Autoloader
import subprocess
import sys

required_libraries = ["tkinter", "hashlib", "threading", "time", "random", "socket", "os", "pyperclip", "psutil", "datetime"]

def autoload_libraries():
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            print(f"📦 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

autoload_libraries()

# 🧠 Imports
import tkinter as tk
from tkinter import ttk
import hashlib
import threading
import time
import random
import socket
import os
import pyperclip
import psutil
import datetime

# ⚙️ Data-as-Motor Daemon
class DataMotorDaemon:
    def __init__(self):
        self.data_stream = []
        self.capacitor_charge = 100
        self.pulse_enabled = True
        self.swarm_nodes = []
        self.log = []
        self.comparisons = []  # (timestamp, raw, mutated, angle)

    def rotor_phase(self, data):
        h = hashlib.sha256(data.encode()).hexdigest()
        return int(h[:4], 16) % 360

    def push_off_window(self, angle):
        return 85 <= angle <= 95

    def inject_compute_pulse(self, data):
        if self.capacitor_charge >= 20 and self.pulse_enabled:
            self.capacitor_charge -= 20
            mutated = data[::-1]
            self.log.append(f"⚡ Pulse injected at {data} → {mutated}")
            return mutated
        return data

    def recharge_capacitor(self):
        if self.capacitor_charge < 100:
            self.capacitor_charge += 1

    def sync_swarm(self, angle):
        for node in self.swarm_nodes:
            node.receive_pulse_signal(angle)

    def process_stream(self):
        processed = []
        for data in self.data_stream:
            angle = self.rotor_phase(data)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            if self.push_off_window(angle):
                result = self.inject_compute_pulse(data)
                self.sync_swarm(angle)
            else:
                result = data
            self.comparisons.append((timestamp, data, result, angle))
            processed.append(result)
        self.data_stream.clear()
        return processed

    def monitor_clipboard(self):
        last_clip = ""
        while True:
            try:
                clip = pyperclip.paste()
                if clip and clip != last_clip:
                    last_clip = clip
                    self.data_stream.append(clip.strip())
            except:
                pass
            time.sleep(2)

    def monitor_network(self):
        while True:
            try:
                conns = psutil.net_connections(kind='tcp')
                for conn in conns:
                    if conn.status == 'ESTABLISHED':
                        info = f"{conn.laddr.ip}:{conn.laddr.port} → {conn.raddr.ip}:{conn.raddr.port}"
                        self.data_stream.append(info)
            except:
                pass
            time.sleep(5)

    def listen_all_ports(self):
        def handle_client(conn):
            try:
                data = conn.recv(1024).decode().strip()
                if data:
                    self.data_stream.append(data)
            except:
                pass
            conn.close()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 0))  # Bind to any available port
        s.listen(100)
        while True:
            try:
                conn, addr = s.accept()
                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
            except:
                pass

# 🧠 Swarm Node
class SwarmNode:
    def receive_pulse_signal(self, angle):
        if 85 <= angle <= 95:
            print(f"Node synced at angle {angle}° — injecting pulse.")

# 🧙 GUI Overlay
def launch_gui(daemon):
    root = tk.Tk()
    root.title("Codex Sentinel: Motor Daemon")
    root.geometry("700x850")
    root.configure(bg="#0a0f1c")

    status = tk.StringVar()
    status.set("Daemon initialized. Monitoring system...")

    tk.Label(root, text="Codex Sentinel", font=("Consolas", 18), fg="cyan", bg="#0a0f1c").pack(pady=10)
    tk.Label(root, textvariable=status, font=("Consolas", 10), fg="orange", bg="#0a0f1c").pack()

    torque_graph = tk.Canvas(root, width=650, height=150, bg="black")
    torque_graph.pack(pady=10)

    capacitor_bar = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
    capacitor_bar.pack(pady=10)
    capacitor_bar["value"] = daemon.capacitor_charge

    log_box = tk.Text(root, height=10, width=85, font=("Consolas", 9), bg="#111", fg="white")
    log_box.pack(pady=10)

    compare_box = tk.Text(root, height=10, width=85, font=("Consolas", 9), bg="#111", fg="lightgreen")
    compare_box.pack(pady=10)

    timing_graph = tk.Canvas(root, width=650, height=150, bg="#222")
    timing_graph.pack(pady=10)

    def refresh_gui():
        while True:
            daemon.recharge_capacitor()
            processed = daemon.process_stream()
            capacitor_bar["value"] = daemon.capacitor_charge
            status.set("⚙️ Stream processed.")
            log_box.delete("1.0", tk.END)
            for entry in daemon.log[-10:]:
                log_box.insert(tk.END, entry + "\n")
            torque_graph.delete("all")
            recent_logs = daemon.log[-10:]
            for i, data in enumerate(processed[-10:]):
                x = i * 60 + 30
                log_entry = recent_logs[i] if i < len(recent_logs) else ""
                height = 100 if "⚡" in log_entry else 40
                color = "orange" if height == 100 else "cyan"
                torque_graph.create_line(x, 120, x, 120 - height, fill=color, width=4)
            compare_box.delete("1.0", tk.END)
            for ts, raw, mutated, angle in daemon.comparisons[-10:]:
                pulse = "⚡" if daemon.push_off_window(angle) else "—"
                compare_box.insert(tk.END, f"{pulse} {angle:3}° | Raw: {raw} → Mutated: {mutated}\n")
            timing_graph.delete("all")
            for i, (ts, raw, mutated, angle) in enumerate(daemon.comparisons[-10:]):
                x = i * 60 + 30
                y = 120
                color = "orange" if daemon.push_off_window(angle) else "cyan"
                timing_graph.create_oval(x-5, y-5, x+5, y+5, fill=color)
                timing_graph.create_text(x, y-15, text=ts, fill="white", font=("Consolas", 8))
            time.sleep(3)

    threading.Thread(target=refresh_gui, daemon=True).start()
    root.mainloop()

# 🧨 Entry Point
if __name__ == "__main__":
    daemon = DataMotorDaemon()
    daemon.swarm_nodes = [SwarmNode() for _ in range(3)]

    threading.Thread(target=daemon.monitor_clipboard, daemon=True).start()
    threading.Thread(target=daemon.monitor_network, daemon=True).start()
    threading.Thread(target=daemon.listen_all_ports, daemon=True).start()
    threading.Thread(target=launch_gui, args=(daemon,), daemon=True).start()

    while True:
        daemon.recharge_capacitor()
        time.sleep(1)

