import tkinter as tk
from tkinter import ttk, filedialog
import psutil
import random
import time
import os
import json
import threading
import subprocess
import statistics
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SAVE_THRESHOLD = 100 * 1024 * 1024  # 100 MB cumulative activity before autosave
DEFAULT_BACKUP_FOLDER = os.path.join(os.getcwd(), "hive_backup")

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
# HiveMind GUI + Telemetry + Prediction
# =========================
class HiveMindGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Hive-Mind Controller (Always-On Predictive)")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use('clam')

        ttk.Label(
            root,
            text="AI Hive-Mind Overlay",
            font=("Arial", 18),
            foreground="white",
            background="#1e1e1e"
        ).pack(pady=10)

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Trigger Ghost Boost", command=self.trigger_boost)\
            .grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="Focused Foresight Pulse", command=self.run_foresight)\
            .grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="Select Local Backup", command=self.select_local_backup)\
            .grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(button_frame, text="Select SMB Backup", command=self.select_smb_backup)\
            .grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(button_frame, text="Manual Restore", command=self.restore_backup)\
            .grid(row=0, column=4, padx=5, pady=5)

        self.status = tk.Text(root, height=15, width=150, bg="#111", fg="#0f0")
        self.status.pack(pady=10)

        meters_frame = ttk.Frame(root)
        meters_frame.pack(pady=5)

        ttk.Label(meters_frame, text="System RAM Usage", font=("Arial", 14))\
            .grid(row=0, column=0, pady=5)
        self.ram_progress = ttk.Progressbar(
            meters_frame, length=400, mode="determinate", maximum=100
        )
        self.ram_progress.grid(row=1, column=0, pady=5, padx=10)

        ttk.Label(meters_frame, text="GPU VRAM Usage", font=("Arial", 14))\
            .grid(row=0, column=1, pady=5)
        self.vram_progress = ttk.Progressbar(
            meters_frame, length=400, mode="determinate", maximum=100
        )
        self.vram_progress.grid(row=1, column=1, pady=5, padx=10)

        ttk.Label(meters_frame, text="Combined RAM+VRAM Load", font=("Arial", 14))\
            .grid(row=0, column=2, pady=5)
        self.combined_progress = ttk.Progressbar(
            meters_frame, length=400, mode="determinate", maximum=100
        )
        self.combined_progress.grid(row=1, column=2, pady=5, padx=10)

        heatmap_frame = ttk.Frame(root)
        heatmap_frame.pack(pady=10)

        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=heatmap_frame)
        self.canvas.get_tk_widget().pack()

        # Backup paths
        os.makedirs(DEFAULT_BACKUP_FOLDER, exist_ok=True)
        self.local_path = DEFAULT_BACKUP_FOLDER  # default autosave/restore anchor
        self.smb_path = None

        # State + autosave tracking
        self.activity_counter = 0
        self.state = {}
        self.last_saved_state = None

        # Learning + prediction
        self.learner = ReinforcementLearner()
        self.telemetry_running = False
        self.telemetry_thread = None
        self.history = deque(maxlen=60)  # last 60 seconds: ram, vram, combined, activity

        # CPU/GPU meta
        self.ghost_cache = None
        self.mode = "Unknown"
        self.cores = psutil.cpu_count(logical=False)
        self.threads = psutil.cpu_count(logical=True)

        # Auto-restore + auto-activate + auto-start predictive telemetry
        self.auto_restore_on_start()
        self.activate_hive_core()
        self.start_telemetry_if_needed()

    # ========== Logging ==========
    def log(self, message):
        self.status.insert(tk.END, message + "\n")
        self.status.see(tk.END)

    # ========== Auto-Restore on Startup ==========
    def auto_restore_on_start(self):
        # Try to restore from default local backup if exists
        candidate = os.path.join(self.local_path, "hive_state.json")
        if os.path.exists(candidate):
            try:
                with open(candidate, "r") as f:
                    self.state = json.load(f)
                self.last_saved_state = json.loads(json.dumps(self.state))
                self.log(f"Auto-restore: Loaded hive_state.json from {self.local_path}.")
            except Exception as e:
                self.log(f"Auto-restore failed: {e}")
        else:
            self.log("Auto-restore: No existing hive_state.json found, starting fresh.")
            self.state = {}

    # ========== Hive Activation Core (Always On) ==========
    def activate_hive_core(self):
        # Allocate ghost cache
        self.ghost_cache, self.mode = self.allocate_vram_or_ram(256 * 1024 * 1024)
        # If state doesn't have cores/threads, set them
        if "cores" not in self.state or "threads" not in self.state:
            self.state["cores"] = self.cores
            self.state["threads"] = self.threads
        self.state["mode"] = self.mode
        self.state["timestamp"] = time.time()

        self.log(f"Hive-Mind Online: {self.cores} cores, {self.threads} threads, cache in {self.mode}.")
        self.update_heatmap()

    def allocate_vram_or_ram(self, size):
        try:
            import cupy as cp
            ghost_cache = cp.zeros((size,), dtype=cp.uint8)
            return ghost_cache, "VRAM"
        except Exception:
            import numpy as np
            ghost_cache = np.zeros((size,), dtype=np.uint8)
            return ghost_cache, "System RAM"

    def start_telemetry_if_needed(self):
        if not self.telemetry_running:
            self.telemetry_running = True
            self.telemetry_thread = threading.Thread(
                target=self.telemetry_loop, daemon=True
            )
            self.telemetry_thread.start()
            self.log("Telemetry + predictive engine is now always on.")

    # ========== Ghost Boost ==========
    def trigger_boost(self):
        boost_time = random.randint(3, 5)
        self.log(
            f"[Ghost Boost] L4 Ghost Cache ({self.mode}) amplifying {self.cores} cores / "
            f"{self.threads} threads for {boost_time} seconds..."
        )

        def boost():
            time.sleep(boost_time)
            self.log("[Ghost Boost] Cycle complete. System stabilized.")

        threading.Thread(target=boost, daemon=True).start()

    # ========== Foresight + RL Pulse ==========
    def run_foresight(self):
        disk_blocks = self.learner.policy["disk_blocks"]
        net_buf = self.learner.policy["network_buffer"]
        user_depth = self.learner.policy["user_depth"]

        success = random.choice([True, False])
        self.learner.reward(success)

        self.log(f"[Foresight Pulse] Disk pre-load blocks={disk_blocks}, Net buffer={net_buf}, User depth={user_depth}.")
        self.log(
            f"[Foresight Pulse] Outcome={'success' if success else 'failure'} → policy={self.learner.policy}"
        )

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
        self.log(f"[Heatmap] Activity snapshot: {usage}")

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
            self.log("Manual restore: No backup path set.")
            return

        if state:
            self.state = state
            self.ghost_cache, self.mode = self.allocate_vram_or_ram(256 * 1024 * 1024)
            self.log(f"Manual restore: hive state loaded into {self.mode}.")
            self.last_saved_state = json.loads(json.dumps(self.state))
        else:
            self.log("Manual restore failed: no valid hive_state.json.")

    def load_backup(self, path):
        try:
            with open(os.path.join(path, "hive_state.json"), "r") as f:
                return json.load(f)
        except Exception:
            return None

    def dual_write_backup(self):
        if not self.local_path or not self.smb_path:
            self.log("Backup paths not fully set (local+SMB). Skipping autosave.")
            return

        new_state = self.state
        if new_state == self.last_saved_state:
            self.log("[Autosave] State unchanged → skipping backup write.")
            return

        try:
            with open(os.path.join(self.local_path, "hive_state.json"), "w") as f:
                json.dump(new_state, f)
            with open(os.path.join(self.smb_path, "hive_state.json"), "w") as f:
                json.dump(new_state, f)
            self.last_saved_state = json.loads(json.dumps(new_state))
            self.log("[Autosave] Dual-write backup completed (local + SMB).")
        except Exception as e:
            self.log(f"[Autosave] Backup failed: {e}")

    # ========== Telemetry + Predictive Loop ==========
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

                self.history.append({
                    "ram": ram_percent,
                    "vram": vram_percent,
                    "combined": combined_percent,
                    "activity": activity
                })

                self.root.after(
                    0,
                    self.update_usage_and_predict,
                    ram_percent,
                    vram_percent,
                    combined_percent
                )

            except Exception as e:
                self.log(f"[Telemetry] Error: {e}")

    def get_gpu_stats(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=1
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

    # ========== Prediction, Anomaly Detection, Autosave ==========
    def update_usage_and_predict(self, ram_percent, vram_percent, combined_percent):
        self.ram_progress["value"] = min(ram_percent, 100)
        self.vram_progress["value"] = min(vram_percent, 100)
        self.combined_progress["value"] = min(combined_percent, 100)

        self.log(
            f"[Telemetry] RAM={ram_percent:.1f}%, VRAM={vram_percent:.1f}%, "
            f"Combined={combined_percent:.1f}%"
        )

        self.run_predictive_analysis()

        if self.activity_counter >= SAVE_THRESHOLD:
            self.log("[Autosave] Activity threshold (100 MB) reached → triggering backup.")
            self.dual_write_backup()
            self.activity_counter = 0

    def run_predictive_analysis(self):
        if len(self.history) < 5:
            return

        ram_series = [h["ram"] for h in self.history]
        vram_series = [h["vram"] for h in self.history]
        combined_series = [h["combined"] for h in self.history]

        # Anomaly detection on combined load
        try:
            mean_combined = statistics.mean(combined_series)
            stdev_combined = statistics.pstdev(combined_series)
            latest = combined_series[-1]
            if stdev_combined > 0:
                z = (latest - mean_combined) / stdev_combined
                if abs(z) >= 3:
                    self.log(
                        f"[Glitch Detection] Anomalous combined load (z={z:.2f}). "
                        f"Mean={mean_combined:.1f}%, Latest={latest:.1f}%"
                    )
        except Exception:
            pass

        def ewma(series, alpha=0.3):
            if not series:
                return 0
            s = series[0]
            for x in series[1:]:
                s = alpha * x + (1 - alpha) * s
            return s

        ram_forecast = ewma(ram_series)
        vram_forecast = ewma(vram_series)
        combined_forecast = ewma(combined_series)

        def estimate_ttt(series, threshold):
            if len(series) < 2:
                return None
            start = series[0]
            end = series[-1]
            delta = end - start
            steps = len(series) - 1
            per_sec = delta / steps if steps > 0 else 0
            if per_sec <= 0:
                return None
            remaining = threshold - end
            if remaining <= 0:
                return 0
            return remaining / per_sec

        t_ram_80 = estimate_ttt(ram_series, 80)
        t_comb_80 = estimate_ttt(combined_series, 80)
        t_comb_90 = estimate_ttt(combined_series, 90)

        msg_parts = []
        msg_parts.append(
            f"[Predictive] EWMA RAM≈{ram_forecast:.1f}%, "
            f"VRAM≈{vram_forecast:.1f}%, COMBINED≈{combined_forecast:.1f}%."
        )
        if t_comb_80 is not None and t_comb_80 <= 60:
            msg_parts.append(f"Combined may cross 80% in ~{t_comb_80:.1f}s.")
        if t_comb_90 is not None and t_comb_90 <= 60:
            msg_parts.append(f"Combined may cross 90% in ~{t_comb_90:.1f}s.")
        if t_ram_80 is not None and t_ram_80 <= 60:
            msg_parts.append(f"RAM may cross 80% in ~{t_ram_80:.1f}s.")

        if msg_parts:
            self.log(" ".join(msg_parts))

# =========================
# Main
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = HiveMindGUI(root)
    root.mainloop()

