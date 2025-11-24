import importlib
import subprocess
import sys

# --- Autoloader for required libraries ---
def ensure_libs(libs):
    for lib in libs:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"📦 Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# List of required libraries
required_libs = ["tkinter"]  # tkinter is built-in with Python, but included for clarity
ensure_libs(required_libs)

# --- Now import after ensuring ---
import tkinter as tk
from tkinter import ttk
import time
import threading
import random

# Constants
ENERGY_PER_KG_H2 = 50   # kWh to produce 1 kg H2
ENERGY_RELEASE_H2 = 33  # kWh released per kg H2
H2_RATIO = 0.111
O2_RATIO = 0.889

class HydrogenCycleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Atmospheric Processor: Solar-Hydrogen Cycle")
        self.root.geometry("700x500")
        self.root.configure(bg="#1e1e2e")

        self.h2_storage = 0
        self.o2_storage = 0
        self.energy_available = 0
        self.running = False

        self.create_panels()
        self.log("🌞 System initialized. Ready to begin continuous phoenix loop.")

    def create_panels(self):
        self.title = tk.Label(self.root, text="☀️ Solar-Hydrogen Cycle", font=("Helvetica", 18, "bold"), fg="#ffffff", bg="#1e1e2e")
        self.title.pack(pady=10)

        self.status_frame = tk.Frame(self.root, bg="#1e1e2e")
        self.status_frame.pack(pady=10)

        self.h2_label = tk.Label(self.status_frame, text="Hydrogen: 0.000 kg", font=("Helvetica", 12), fg="#00ffcc", bg="#1e1e2e")
        self.h2_label.grid(row=0, column=0, padx=20)

        self.o2_label = tk.Label(self.status_frame, text="Oxygen: 0.000 kg", font=("Helvetica", 12), fg="#ffcc00", bg="#1e1e2e")
        self.o2_label.grid(row=0, column=1, padx=20)

        self.energy_label = tk.Label(self.status_frame, text="Energy: 0.00 kWh", font=("Helvetica", 12), fg="#ffffff", bg="#1e1e2e")
        self.energy_label.grid(row=0, column=2, padx=20)

        self.log_box = tk.Text(self.root, height=15, width=80, bg="#2e2e3e", fg="#ffffff", font=("Courier", 10))
        self.log_box.pack(pady=10)

        self.start_button = ttk.Button(self.root, text="Start Continuous Cycle 🔄", command=self.toggle_cycle)
        self.start_button.pack(pady=10)

    def log(self, message):
        self.log_box.insert(tk.END, message + "\n")
        self.log_box.see(tk.END)

    def update_status(self):
        self.h2_label.config(text=f"Hydrogen: {self.h2_storage:.3f} kg")
        self.o2_label.config(text=f"Oxygen: {self.o2_storage:.3f} kg")
        self.energy_label.config(text=f"Energy: {self.energy_available:.2f} kWh")

    def toggle_cycle(self):
        self.running = not self.running
        self.start_button.config(text="Stop Continuous Cycle ❌" if self.running else "Start Continuous Cycle 🔄")
        if self.running:
            threading.Thread(target=self.continuous_cycle, daemon=True).start()

    def continuous_cycle(self):
        while self.running:
            self.simulate_cycle()
            time.sleep(3)  # Delay between cycles

    def simulate_cycle(self):
        self.log("🌬️ Atmospheric intake: harvesting moisture...")
        time.sleep(1)
        water_kg = random.uniform(0.8, 1.2)
        self.log(f"💧 Condensed {water_kg:.2f} kg of water vapor.")

        self.log("⚡ Electrolysis: splitting water into H₂ and O₂...")
        time.sleep(1)
        h2 = water_kg * H2_RATIO
        o2 = water_kg * O2_RATIO
        energy_used = h2 * ENERGY_PER_KG_H2
        self.log(f"🔹 Produced {h2:.3f} kg H₂ and {o2:.3f} kg O₂ using {energy_used:.2f} kWh solar energy.")

        self.h2_storage += h2
        self.o2_storage += o2
        self.update_status()

        self.log("🔄 Fuel Cell Rebirth: recombining H₂ and O₂...")
        time.sleep(1)
        energy_released = self.h2_storage * ENERGY_RELEASE_H2
        water_reformed = self.h2_storage / H2_RATIO
        self.log(f"🔥 Released {energy_released:.2f} kWh and reformed {water_reformed:.2f} kg of water.")

        self.energy_available += energy_released
        self.h2_storage = 0
        self.o2_storage = 0
        self.update_status()

        self.log("🌟 Cycle complete. Phoenix loop reborn.\n")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = HydrogenCycleGUI(root)
    root.mainloop()

