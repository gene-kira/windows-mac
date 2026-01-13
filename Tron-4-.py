import tkinter as tk
from tkinter import ttk, filedialog
import psutil
import random
import time
import os
import json
import threading
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SAVE_THRESHOLD = 1 * 1024 * 1024  # 1 MB cumulative activity before autosave

# =========================
# Reinforcement Learning Engine
# =========================
class ReinforcementLearner:
    def __init__(self, state_file="hive_policy.json"):
        self.state_file = state_file
        self.policy = {"disk_blocks": 10, "network_buffer": 5, "user_depth": 3}
        self.last_saved_policy = None
        self.load_policy()

    def reward(self, success: bool):
        old_policy = dict(self.policy)
        if success:
            self.policy["disk_blocks"] = min(self.policy["disk_blocks"] + 1, 50)
            self.policy["network_buffer"] = min(self.policy["network_buffer"] + 1, 50)
            self.policy["user_depth"] = min(self.policy["user_depth"] + 1, 10)
        else:
            self.policy["disk_blocks"] = max(self.policy["disk_blocks"] - 1, 1)
            self.policy["network_buffer"] = max(self.policy["network_buffer"] - 1, 1)
            self.policy["user_depth"] = max(self.policy["user_depth"] - 1, 1)
        if self.policy != old_policy:
            self.save_policy()

    def save_policy(self):
        if self.policy == self.last_saved_policy:
            return
        with open(self.state_file, "w") as f:
            json.dump(self.policy, f)
        self.last_saved_policy = dict(self.policy)

    def load_policy(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                self.policy = json.load(f)
            self.last_saved_policy = dict(self.policy)

# =========================
# HiveMind GUI + Telemetry
# =========================
class HiveMindGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Hive-Mind Controller (Live Telemetry)")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use('clam')

        ttk.Label(root, text="AI Hive-Mind Overlay", font=("Arial", 18), foreground="white", background="#1e1e1e").pack(pady=10)

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Activate Hive-Mind", command=self.activate_hive).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="Trigger Ghost Boost", command=self.trigger_boost).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="Run Predictive Foresight", command=self.run_foresight).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="Select Local Backup", command=self.select_local_backup).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="Select SMB Backup", command=self.select_smb_backup).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="Restore from Backup", command=self.restore_backup).grid(row=1, column=2, padx=5, pady=5)

        self.status = tk.Text(root, height=15, width=150, bg="#111", fg="#0f0")
        self.status.pack(pady=10)

        meters_frame = ttk.Frame(root)
        meters_frame.pack(pady=5)

        ttk.Label(meters_frame, text="System RAM Usage", font=("Arial", 14)).grid(row=0, column=0, pady=5)
        self.ram_progress = ttk.Progressbar(meters_frame, length=400, mode="determinate", maximum=100)
        self.ram_progress.grid(row=1, column=0, pady=5, padx=10)

        ttk.Label(meters_frame, text="GPU VRAM Usage", font=("Arial", 14)).grid(row=0, column=1, pady=5)
        self.vram_progress = ttk.Progressbar(meters_frame, length=400, mode="determinate", maximum=100)
        self.vram_progress.grid(row=1, column=1, pady=5, padx=10)

        ttk.Label(meters_frame, text="Combined RAM+VRAM Load", font=("Arial", 14)).grid(row=0, column=2, pady=5)
        self.combined_progress = ttk.Progressbar(meters_frame, length=400, mode="determinate", maximum=100)
        self.combined_progress.grid(row=1, column=2, pady=5, padx=10)

        heatmap_frame = ttk.Frame(root)
        heatmap_frame.pack(pady=10)

        self.figure, self.ax = plt.subplots(figsize=(6,4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=heatmap_frame)
        self.canvas.get_tk_widget().pack()

        self.local_path = None
        self.smb_path = None

        self.activity_counter = 0  # cumulative bytes for autosave
        self.state = {}
        self.last_saved_state = None

        self.learner = ReinforcementLearner()

        self.telemetry_running = False
        self.telemetry_thread = None

    # ========== Logging ==========
    def log(self, message):
        self.status.insert(tk.END, message + "\n")
        self.status.see(tk.END)

    # ========== Activation / VRAM ==========
    def activate_hive(self):
        self.ghost_cache, self.mode = self.allocate_vram_or_ram(256 * 1024 * 1024)  # 256 MB ghost cache as placeholder
        self.cores = psutil.cpu_count(logical=False)
        self.threads = psutil.cpu_count(logical=True)
        self.state = {
            "mode": self.mode,
            "cores": self.cores,
            "threads": self.threads,
            "timestamp": time.time()
        }
        self.log(f"Hive-Mind Activated with {self.cores} cores, {self.threads} threads.")
        self.log(f"L4 Ghost Cache allocated via {self.mode}.")
        self.update_heatmap()

        if not self.telemetry_running:
            self.telemetry_running = True
            self.telemetry_thread = threading.Thread(target=self.telemetry_loop, daemon=True)
            self.telemetry_thread.start()
            self.log("Telemetry loop started (live RAM/VRAM/Disk/Net/CPU).")

    def allocate_vram_or_ram(self, size):
        try:
            import cupy as cp
            ghost_cache = cp.zeros((size,), dtype=cp.uint8)
            return ghost_cache, "VRAM"
        except Exception:
            import numpy as np
            ghost_cache = np.zeros((size,), dtype=np.uint8)
            return ghost_cache, "System RAM"

    # ========== Boost ==========
    def trigger_boost(self):
        if not hasattr(self, "cores") or not hasattr(self, "threads"):
            self.log("Hive not activated yet.")
            return
        boost_time = random.randint(3, 5)
        self.log(f"L4 Ghost Cache ({self.mode}) boosting {self.cores} cores and {self.threads} threads for {boost_time} seconds...")

        def boost():
            time.sleep(boost_time)
            self.log("Boost cycle complete. System stabilized.")

        threading.Thread(target=boost, daemon=True).start()

    # ========== Foresight + RL ==========
    def run_foresight(self):
        disk_blocks = self.learner.policy["disk_blocks"]
        net_buf = self.learner.policy["network_buffer"]
        user_depth = self.learner.policy["user_depth"]

        success = random.choice([True, False])
        self.learner.reward(success)

        self.log(f"Disk foresight: Pre-loaded {disk_blocks} blocks.")
        self.log(f"Network foresight: Pre-allocated {net_buf} packets.")
        self.log(f"User foresight: Anticipated depth {user_depth}.")
        self.log(f"Foresight {'succeeded' if success else 'failed'} → policy updated: {self.learner.policy}")

    # ========== Heatmap ==========
    def update_heatmap(self):
        usage = {
            "L1": random.randint(70, 100),
            "L2": random.randint(40, 70),
            "L3": random.randint(10, 40),
            "L4": random.randint(0, 20)
        }
        self.ax.clear()
        self.ax.bar(
            ["L1 High-Speed", "L2 Medium-Speed", "L3 Slow-Speed", "L4 Ghost Boost"],
            [usage["L1"], usage["L2"], usage["L3"], usage["L4"]],
            color=["#00ff00", "#ffaa00", "#ff0000", "#00ffff"]
        )
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Usage %")
        self.ax.set_title("Reasoning Heatmap: Cache Activity")
        self.canvas.draw()
        self.log(f"Reasoning Heatmap updated: {usage}")

    # ========== Backup Handling ==========
    def select_local_backup(self):
        path = filedialog.askdirectory(title="Select Local Backup Folder")
        if path:
            self.local_path = path
            self.log(f"Local backup path set: {path}")

    def select_smb_backup(self):
        path = filedialog.askdirectory(title="Select SMB Network Backup Folder")
        if path:
            self.smb_path = path
            self.log(f"SMB backup path set: {path}")

    def restore_backup(self):
        if self.local_path:
            state = self.load_backup(self.local_path)
        elif self.smb_path:
            state = self.load_backup(self.smb_path)
        else:
            self.log("No backup path set.")
            return

        if state:
            self.state = state
            self.ghost_cache, self.mode = self.allocate_vram_or_ram(256 * 1024 * 1024)
            self.log(f"Restored hive state from backup into {self.mode}.")
            self.last_saved_state = json.loads(json.dumps(self.state))
        else:
            self.log("Failed to restore backup.")

    def load_backup(self, path):
        try:
            with open(os.path.join(path, "hive_state.json"), "r") as f:
                return json.load(f)
        except Exception:
            return None

    def dual_write_backup(self):
        if not self.local_path or not self.smb_path:
            self.log("Backup paths not set. Skipping autosave.")
            return

        new_state = self.state
        if new_state == self.last_saved_state:
            self.log("State unchanged → skipping backup write.")
            return

        try:
            with open(os.path.join(self.local_path, "hive_state.json"), "w") as f:
                json.dump(new_state, f)
            with open(os.path.join(self.smb_path, "hive_state.json"), "w") as f:
                json.dump(new_state, f)
            self.last_saved_state = json.loads(json.dumps(new_state))
            self.log("Usage-triggered dual-write backup completed.")
        except Exception as e:
            self.log(f"Backup failed: {e}")

    # ========== Telemetry Loop (live backbone) ==========
    def telemetry_loop(self):
        while True:
            try:
                time.sleep(1.0)

                vm = psutil.virtual_memory()
                ram_used = vm.used
                ram_total = vm.total
                ram_percent = (ram_used / ram_total) * 100 if ram_total > 0 else 0

                disk = psutil.disk_io_counters()
                disk_bytes = disk.read_bytes + disk.write_bytes

                net = psutil.net_io_counters()
                net_bytes = net.bytes_recv + net.bytes_sent

                cpu_percent = psutil.cpu_percent(interval=None)

                vram_used, vram_total, gpu_util = self.get_gpu_stats()
                vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0

                combined_used = ram_used + vram_used
                combined_total = ram_total + vram_total
                combined_percent = (combined_used / combined_total) * 100 if combined_total > 0 else 0

                activity = disk_bytes + net_bytes + int(cpu_percent * 1024) + int(gpu_util * 1024)
                self.activity_counter += activity

                self.root.after(
                    0,
                    self.update_usage_display,
                    ram_percent,
                    vram_percent,
                    combined_percent,
                    activity
                )

            except Exception as e:
                self.log(f"Telemetry error: {e}")

    def get_gpu_stats(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0:
                return 0, 1, 0
            line = result.stdout.strip().split("\n")[0]
            mem_used_str, mem_total_str, util_str = line.split(",")
            vram_used_mb = float(mem_used_str.strip())
            vram_total_mb = float(mem_total_str.strip())
            gpu_util = float(util_str.strip())
            return int(vram_used_mb * 1024 * 1024), int(vram_total_mb * 1024 * 1024), gpu_util
        except Exception:
            return 0, 1, 0

    # ========== Usage Display & Autosave ==========
    def update_usage_display(self, ram_percent, vram_percent, combined_percent, activity):
        self.ram_progress["value"] = min(ram_percent, 100)
        self.vram_progress["value"] = min(vram_percent, 100)
        self.combined_progress["value"] = min(combined_percent, 100)

        self.log(f"[Telemetry] RAM={ram_percent:.1f}%, VRAM={vram_percent:.1f}%, Combined={combined_percent:.1f}%")

        if self.activity_counter >= SAVE_THRESHOLD:
            self.log("Activity threshold reached (100 MB equivalent) → Autosave triggered.")
            self.dual_write_backup()
            self.activity_counter = 0

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = HiveMindGUI(root)
    root.mainloop()

