# ============================================================
# AUTO-LOADER FOR OPTIONAL LIBRARIES
# ============================================================

import importlib
import subprocess
import sys

def auto_import(package, pip_name=None):
    """
    Attempts to import a package.
    If missing, installs it via pip and then imports again.
    """
    try:
        return importlib.import_module(package)
    except ImportError:
        print(f"[AutoLoader] '{package}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package])
        return importlib.import_module(package)

# Optional libraries for plotting and math
np = auto_import("numpy")
auto_import("matplotlib")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ============================================================
# LIVING BATTERY ADVANCED SIMULATOR
# ============================================================

import tkinter as tk
from tkinter import ttk


class LivingBatteryAdvancedSim:
    def __init__(self, root):
        self.root = root
        self.root.title("Living Battery (Solar + Air + Bio + Organs + Guardian)")

        # --- Core time & environment parameters ---
        self.dt = 0.2                    # time step (seconds, simulated)
        self.t = 0.0

        # --- Energy organs (membrane + core) ---
        self.E_mem_max_nominal = 200.0   # nominal capacity of membrane (fast)
        self.E_core_max_nominal = 800.0  # nominal capacity of core (deep)
        self.health = 1.0                # 1.0 = new, 0.0 = dead

        # Initial energies
        self.E_mem = 50.0
        self.E_core = 300.0

        # --- Temperature model ---
        self.T = 25.0                    # current temperature (°C)
        self.T_ambient = 25.0            # ambient temperature (°C)
        self.T_rise_per_power = 0.03     # temperature rise per unit power (rough)
        self.T_cool_rate = 0.02          # cooling factor toward ambient

        # --- Harvesting efficiencies ---
        self.eta_pv = 0.25               # solar
        self.eta_air = 0.15              # ambient field
        self.eta_bio = 15.0              # bio-engine power gain

        # --- Loss and transfer parameters ---
        self.k_loss_mem = 0.001          # membrane self-loss
        self.k_loss_core = 0.0003        # core self-loss
        self.k_mem_to_core = 0.10        # rate energy flows from membrane to core
        self.k_core_to_mem = 0.02        # backflow (for bursts)

        # --- Aging / health degradation parameters ---
        self.health_decay_base = 1e-5          # baseline health decay per time
        self.health_decay_temp_factor = 5e-5   # added decay * (T - T_ambient) if hot
        self.health_decay_flow_factor = 1e-6   # added decay per |P_flow|

        # --- AI Guardian control weights ---
        self.guardian_weight_low_energy = 1.0
        self.guardian_weight_high_temp = 1.5
        self.guardian_weight_low_health = 2.0

        # History buffers for plotting
        self.history_len = 300
        self.time_hist = []
        self.E_mem_hist = []
        self.E_core_hist = []
        self.T_hist = []
        self.health_hist = []

        # Build GUI
        self._build_controls()
        self._build_display()
        self._build_plots()

        self.running = True
        self._update_sim()

    # ========================================================
    # GUI BUILDING
    # ========================================================

    def _build_controls(self):
        frame = ttk.Frame(self.root, padding=8)
        frame.grid(row=0, column=0, sticky="nsew")

        # Sunlight
        ttk.Label(frame, text="Sunlight P_sun (0–100)").grid(row=0, column=0, sticky="w")
        self.sun_scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal")
        self.sun_scale.set(60)
        self.sun_scale.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        # Ambient field
        ttk.Label(frame, text="Ambient Field P_field (0–100)").grid(row=2, column=0, sticky="w")
        self.field_scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal")
        self.field_scale.set(30)
        self.field_scale.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        # Load power
        ttk.Label(frame, text="Load Power P_load (0–100)").grid(row=4, column=0, sticky="w")
        self.load_scale = ttk.Scale(frame, from_=0, to=100, orient="horizontal")
        self.load_scale.set(40)
        self.load_scale.grid(row=5, column=0, sticky="ew", pady=(0, 6))

        # Manual bio override (guardian modulates around this)
        ttk.Label(frame, text="Bio Activity Bias (0–1)").grid(row=6, column=0, sticky="w")
        self.bio_bias_scale = ttk.Scale(frame, from_=0.0, to=1.0, orient="horizontal")
        self.bio_bias_scale.set(0.5)
        self.bio_bias_scale.grid(row=7, column=0, sticky="ew", pady=(0, 6))

        # Info label
        self.info_label = ttk.Label(frame, text="", justify="left", anchor="w")
        self.info_label.grid(row=8, column=0, sticky="ew", pady=(8, 0))

        frame.columnconfigure(0, weight=1)

    def _build_display(self):
        # Canvas for battery organs
        self.canvas = tk.Canvas(self.root, width=380, height=260, bg="white")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        # Titles
        self.canvas.create_text(95, 20, text="Membrane Energy", anchor="center")
        self.canvas.create_text(285, 20, text="Core Energy", anchor="center")

        # Background bars
        self.canvas.create_rectangle(50, 40, 140, 220, outline="#cccccc", width=2)
        self.canvas.create_rectangle(240, 40, 330, 220, outline="#cccccc", width=2)

        # Active bars
        self.mem_bar = self.canvas.create_rectangle(50, 220, 140, 220,
                                                    fill="#99ccff", outline="")
        self.core_bar = self.canvas.create_rectangle(240, 220, 330, 220,
                                                     fill="#ffcc99", outline="")

        # Power flows / temp / health text
        self.flow_text = self.canvas.create_text(190, 230, text="", anchor="n")

    def _build_plots(self):
        # Matplotlib figure embedded in Tk
        self.fig = Figure(figsize=(6.0, 3.0), dpi=100)
        self.ax_energy = self.fig.add_subplot(2, 1, 1)
        self.ax_state = self.fig.add_subplot(2, 1, 2)

        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.set_xticks([])

        self.ax_state.set_xlabel("Time (s)")
        self.ax_state.set_ylabel("T / Health")

        self.line_E_mem, = self.ax_energy.plot([], [], label="E_mem")
        self.line_E_core, = self.ax_energy.plot([], [], label="E_core")
        self.ax_energy.legend(loc="upper right", fontsize=8)

        self.line_T, = self.ax_state.plot([], [], label="Temperature (°C)")
        self.line_health, = self.ax_state.plot([], [], label="Health (0–1)")
        self.ax_state.legend(loc="upper right", fontsize=8)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_plot.get_tk_widget().grid(row=2, column=0, sticky="nsew",
                                              padx=8, pady=(0, 8))

    # ========================================================
    # AI GUARDIAN LOGIC
    # ========================================================

    def guardian_compute_u_bio(self, fullness_total, temp, health, bias):
        """
        AI guardian chooses u_bio in [0, 1] based on:
        - fullness_total: how full the system is
        - temp: current temperature
        - health: cell health
        - bias: user bias (slider) between 0 and 1
        """
        # More activity when energy is low
        low_energy_drive = (1.0 - fullness_total) * self.guardian_weight_low_energy

        # Less activity when temperature is high
        temp_excess = max(0.0, temp - (self.T_ambient + 5.0))
        high_temp_inhibit = temp_excess * self.guardian_weight_high_temp / 20.0

        # Less activity when health is low
        low_health_inhibit = (1.0 - health) * self.guardian_weight_low_health

        raw = low_energy_drive + bias - high_temp_inhibit - low_health_inhibit
        u_bio = max(0.0, min(1.0, raw))

        debug = {
            "low_energy_drive": low_energy_drive,
            "high_temp_inhibit": high_temp_inhibit,
            "low_health_inhibit": low_health_inhibit,
            "bias": bias,
            "raw": raw
        }
        return u_bio, debug

    # ========================================================
    # SIMULATION UPDATE
    # ========================================================

    def _update_sim(self):
        if not self.running:
            return

        # Read controls
        P_sun = self.sun_scale.get()
        P_field = self.field_scale.get()
        P_load_request = self.load_scale.get()
        bio_bias = self.bio_bias_scale.get()

        # Effective capacities reduced by health
        E_mem_max = self.E_mem_max_nominal * self.health
        E_core_max = self.E_core_max_nominal * self.health

        # Clamp to current max capacities
        if self.E_mem > E_mem_max:
            self.E_mem = E_mem_max
        if self.E_core > E_core_max:
            self.E_core = E_core_max

        # Fullness
        total_E = self.E_mem + self.E_core
        total_E_max = E_mem_max + E_core_max if (E_mem_max + E_core_max) > 0 else 1.0
        fullness_total = total_E / total_E_max

        # Guardian picks u_bio
        u_bio, guardian_debug = self.guardian_compute_u_bio(
            fullness_total, self.T, self.health, bio_bias
        )

        # Solar and air harvesting
        P_pv = self.eta_pv * P_sun
        P_air = self.eta_air * P_field

        # Bio-engine: attached primarily to membrane organ (depends on total fullness)
        P_bio = self.eta_bio * u_bio * (1.0 - fullness_total)
        P_bio = max(0.0, P_bio)

        P_harvest_total = P_pv + P_air + P_bio

        # Assign harvested energy: first fill membrane, then core
        P_to_mem = P_harvest_total * 0.4
        P_to_core = P_harvest_total * 0.6

        # Flow from membrane to core (like slow digestion)
        P_mem_to_core = self.k_mem_to_core * self.E_mem
        # Flow from core to membrane (for bursts)
        P_core_to_mem = self.k_core_to_mem * self.E_core

        # Supply load: membrane first, then core
        P_available_mem = self.E_mem / self.dt if self.dt > 0 else 0.0
        P_from_mem = min(P_load_request, P_available_mem)
        P_remaining_load = P_load_request - P_from_mem

        P_available_core = self.E_core / self.dt if self.dt > 0 else 0.0
        P_from_core = min(P_remaining_load, P_available_core)
        P_unmet = P_load_request - (P_from_mem + P_from_core)
        P_unmet = max(0.0, P_unmet)

        # Losses
        P_loss_mem = self.k_loss_mem * self.E_mem
        P_loss_core = self.k_loss_core * self.E_core

        # Update energies (discrete)
        dE_mem = (P_to_mem + P_core_to_mem - P_mem_to_core
                  - P_from_mem - P_loss_mem) * self.dt
        dE_core = (P_to_core + P_mem_to_core - P_core_to_mem
                   - P_from_core - P_loss_core) * self.dt

        self.E_mem += dE_mem
        self.E_core += dE_core

        # Clamp again to valid ranges
        self.E_mem = max(0.0, min(self.E_mem, E_mem_max))
        self.E_core = max(0.0, min(self.E_core, E_core_max))

        # Temperature update: depends on absolute power flows
        P_flow_mag = (abs(P_harvest_total) + abs(P_from_mem) +
                      abs(P_from_core) + abs(P_mem_to_core) +
                      abs(P_core_to_mem))
        dT = (self.T_rise_per_power * P_flow_mag * self.dt
              - self.T_cool_rate * (self.T - self.T_ambient) * self.dt)
        self.T += dT

        # Health degradation
        temp_excess = max(0.0, self.T - (self.T_ambient + 5.0))
        dHealth = -(
            self.health_decay_base +
            self.health_decay_temp_factor * temp_excess +
            self.health_decay_flow_factor * P_flow_mag
        ) * self.dt
        self.health += dHealth
        self.health = max(0.0, min(1.0, self.health))

        # Advance time
        self.t += self.dt

        # Record history
        self._update_history()

        # Redraw GUI + plots
        self._redraw_display(
            P_pv, P_air, P_bio, P_harvest_total,
            P_load_request, P_from_mem, P_from_core,
            P_unmet, u_bio
        )
        self._redraw_plots()

        # Info label
        self._update_info_label(u_bio, guardian_debug, fullness_total)

        # Schedule next step
        self.root.after(int(self.dt * 1000), self._update_sim)

    # ========================================================
    # HISTORY & PLOTS
    # ========================================================

    def _update_history(self):
        self.time_hist.append(self.t)
        self.E_mem_hist.append(self.E_mem)
        self.E_core_hist.append(self.E_core)
        self.T_hist.append(self.T)
        self.health_hist.append(self.health)

        if len(self.time_hist) > self.history_len:
            self.time_hist = self.time_hist[-self.history_len:]
            self.E_mem_hist = self.E_mem_hist[-self.history_len:]
            self.E_core_hist = self.E_core_hist[-self.history_len:]
            self.T_hist = self.T_hist[-self.history_len:]
            self.health_hist = self.health_hist[-self.history_len:]

    def _redraw_plots(self):
        if not self.time_hist:
            return

        t = np.array(self.time_hist)
        E_mem_arr = np.array(self.E_mem_hist)
        E_core_arr = np.array(self.E_core_hist)
        T_arr = np.array(self.T_hist)
        health_arr = np.array(self.health_hist)

        # Energy subplot
        self.line_E_mem.set_data(t, E_mem_arr)
        self.line_E_core.set_data(t, E_core_arr)
        self.ax_energy.set_xlim(t.min(), t.max())
        y_min = 0.0
        y_max = max(E_mem_arr.max(), E_core_arr.max(), 1.0)
        self.ax_energy.set_ylim(y_min, y_max * 1.05)

        # Temperature + health subplot
        self.line_T.set_data(t, T_arr)
        self.line_health.set_data(t, health_arr)
        self.ax_state.set_xlim(t.min(), t.max())
        y_min2 = min(T_arr.min(), health_arr.min(), 0.0)
        y_max2 = max(T_arr.max(), health_arr.max(), 1.0)
        self.ax_state.set_ylim(y_min2 - 1.0, y_max2 + 1.0)

        self.fig.tight_layout()
        self.canvas_plot.draw()

    # ========================================================
    # DISPLAY UPDATE
    # ========================================================

    def _redraw_display(self, P_pv, P_air, P_bio, P_harvest, P_load,
                        P_from_mem, P_from_core, P_unmet, u_bio):
        # Current max capacities
        E_mem_max = self.E_mem_max_nominal * self.health
        E_core_max = self.E_core_max_nominal * self.health

        # Bar heights
        frac_mem = self.E_mem / E_mem_max if E_mem_max > 0 else 0.0
        frac_core = self.E_core / E_core_max if E_core_max > 0 else 0.0

        frac_mem = max(0.0, min(frac_mem, 1.0))
        frac_core = max(0.0, min(frac_core, 1.0))

        mem_top = 220 - frac_mem * 180
        core_top = 220 - frac_core * 180

        self.canvas.coords(self.mem_bar, 50, mem_top, 140, 220)
        self.canvas.coords(self.core_bar, 240, core_top, 330, 220)

        # Flow text
        text = (
            f"P_pv:{P_pv:5.1f}  P_air:{P_air:5.1f}  P_bio:{P_bio:5.1f}  | "
            f"P_harv:{P_harvest:5.1f}\n"
            f"Load:{P_load:5.1f}  Mem->Load:{P_from_mem:5.1f}  "
            f"Core->Load:{P_from_core:5.1f}  Unmet:{P_unmet:5.1f}\n"
            f"T:{self.T:5.1f}°C  Health:{self.health:4.2f}  u_bio:{u_bio:4.2f}"
        )
        self.canvas.itemconfig(self.flow_text, text=text)

    def _update_info_label(self, u_bio, guardian_debug, fullness_total):
        total_nominal = self.E_mem_max_nominal + self.E_core_max_nominal
        total_capacity_now = total_nominal * self.health
        fullness_pct = 0.0
        if total_capacity_now > 0:
            fullness_pct = 100.0 * (self.E_mem + self.E_core) / total_capacity_now

        info = (
            f"Time: {self.t:6.1f} s\n"
            f"E_mem: {self.E_mem:7.2f} / {self.E_mem_max_nominal * self.health:7.2f}\n"
            f"E_core: {self.E_core:7.2f} / {self.E_core_max_nominal * self.health:7.2f}\n"
            f"Total fullness: {fullness_pct:5.1f}% (raw fullness={fullness_total:4.2f})\n"
            f"T: {self.T:5.2f} °C\n"
            f"Health: {self.health:5.3f}\n"
            f"Guardian terms -> "
            f"LowE:{guardian_debug['low_energy_drive']:4.2f}, "
            f"HighT:{guardian_debug['high_temp_inhibit']:4.2f}, "
            f"LowH:{guardian_debug['low_health_inhibit']:4.2f}, "
            f"Bias:{guardian_debug['bias']:4.2f}, "
            f"Raw:{guardian_debug['raw']:4.2f}"
        )
        self.info_label.config(text=info)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    # Make rows/cols stretch
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.columnconfigure(0, weight=1)

    app = LivingBatteryAdvancedSim(root)
    root.mainloop()

