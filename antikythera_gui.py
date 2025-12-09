# antikythera_gui.py
# High-tech, modular Antikythera-inspired simulator with GUI and AI integration hooks.

import math
import datetime
import threading
import time
import tkinter as tk
from tkinter import ttk

# ---------- Utility: angles, clamp, dates ----------

TAU = 2 * math.pi

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi
def norm_deg(d):
    d = d % 360.0
    return d if d >= 0 else d + 360.0

def days_since_j2000(dt: datetime.datetime) -> float:
    # J2000 epoch: Jan 1, 2000 12:00 TT ~ Jan 1, 2000 11:58:55 UTC; use noon for simplicity
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return (dt - j2000).total_seconds() / 86400.0

# ---------- Ephemeris (simplified) ----------
# Purpose: produce angles similar to what the Mechanism modeled.
# For high precision, swap with JPL ephemerides; this is lightweight.

class Ephemeris:
    # Mean motions (approx): Sun ~ 360°/year, Moon synodic ~ 29.53059 days
    SIDEREAL_YEAR_DAYS = 365.2422
    SYNODIC_MONTH_DAYS = 29.530588
    ANOMALISTIC_MONTH_DAYS = 27.55455  # lunar anomaly pointer

    ZODIAC = [
        "Aries","Taurus","Gemini","Cancer","Leo","Virgo",
        "Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"
    ]

    def __init__(self, now=None):
        self.now = now or datetime.datetime.now(datetime.timezone.utc)

    def update(self, dt=None):
        if dt: self.now = dt

    def sun_ecliptic_longitude_deg(self) -> float:
        d = days_since_j2000(self.now)
        mean = norm_deg(280.460 + 0.9856474 * d)  # mean longitude
        anomaly = norm_deg(357.528 + 0.9856003 * d)  # mean anomaly
        # equation of center (approx)
        lam = mean + 1.915 * math.sin(deg2rad(anomaly)) + 0.020 * math.sin(deg2rad(2 * anomaly))
        return norm_deg(lam)

    def moon_ecliptic_longitude_deg(self) -> float:
        d = days_since_j2000(self.now)
        L0 = norm_deg(218.316 + 13.176396 * d)   # Moon's mean longitude
        M_sun = norm_deg(357.529 + 0.9856003 * d)
        M_moon = norm_deg(134.963 + 13.064993 * d)
        F = norm_deg(93.272 + 13.229350 * d)     # argument of latitude
        # simplified longitude (Meeus-like approximate)
        lam = L0 + 6.289 * math.sin(deg2rad(M_moon)) \
                  + 1.274 * math.sin(deg2rad(2*(L0) - M_moon - M_sun)) \
                  + 0.658 * math.sin(deg2rad(2*(L0) - M_sun)) \
                  + 0.214 * math.sin(deg2rad(2*M_moon)) \
                  + 0.110 * math.sin(deg2rad(L0))
        return norm_deg(lam)

    def lunar_phase_deg(self) -> float:
        # phase angle: Sun->Moon difference
        return norm_deg(self.moon_ecliptic_longitude_deg() - self.sun_ecliptic_longitude_deg())

    def lunar_phase_name(self) -> str:
        a = self.lunar_phase_deg()
        if a < 22.5: return "New"
        elif a < 67.5: return "Waxing Crescent"
        elif a < 112.5: return "First Quarter"
        elif a < 157.5: return "Waxing Gibbous"
        elif a < 202.5: return "Full"
        elif a < 247.5: return "Waning Gibbous"
        elif a < 292.5: return "Last Quarter"
        elif a < 337.5: return "Waning Crescent"
        else: return "New"

    def lunar_anomaly_deg(self) -> float:
        # pointer for lunar anomaly dial (approx)
        d = days_since_j2000(self.now)
        return norm_deg(360.0 * (d % self.ANOMALISTIC_MONTH_DAYS) / self.ANOMALISTIC_MONTH_DAYS)

    def sun_zodiac_index(self) -> int:
        lam = self.sun_ecliptic_longitude_deg()
        return int(lam // 30)

    def predict_eclipse_windows(self, lookahead_days=365*3):
        # Saros ~ 6585.321 days; Exeligmos ~ 3* Saros (~19755.96 days)
        # We mark approximate windows when phase near New or Full and node proximity (simplified).
        now_d = days_since_j2000(self.now)
        windows = []
        step = 5  # days
        for k in range(0, lookahead_days, step):
            d = now_d + k
            dt = self.now + datetime.timedelta(days=k)
            # compute phase angle
            a = self.lunar_phase_deg_on_day(d)
            near_new = a < 10 or a > 350
            near_full = abs(a - 180) < 10
            if near_new or near_full:
                windows.append({
                    "date": dt.date().isoformat(),
                    "type": "Solar" if near_new else "Lunar",
                    "phase_deg": round(a, 1)
                })
        return windows

    def lunar_phase_deg_on_day(self, days):
        # helper for eclipse scanning
        dt = datetime.datetime(2000,1,1,12,0,tzinfo=datetime.timezone.utc) + datetime.timedelta(days=days)
        backup = self.now
        self.now = dt
        a = self.lunar_phase_deg()
        self.now = backup
        return a

# ---------- Gear model ----------
# Declarative, integer-tooth gear trains driving pointers.

class Gear:
    def __init__(self, name, teeth):
        self.name = name
        self.teeth = teeth

class GearMesh:
    def __init__(self):
        # Define a small train: driving Sun pointer and lunar pointer
        # You can swap these ratios for experimental trains.
        self.gears = {
            "driver": Gear("driver", 64),
            "sun": Gear("sun", 38),
            "moon": Gear("moon", 53),
            "phase": Gear("phase", 127),  # classic lunar phase gear count in reconstructions
        }
        # Mesh relationships (driver -> others)
        self.mesh = [
            ("driver", "sun"),
            ("driver", "moon"),
            ("moon", "phase")  # nested path for anomaly/phase coupling
        ]
        self.angles = {g: 0.0 for g in self.gears}  # degrees

    def step(self, driver_deg):
        # Advance driver and compute driven angles based on tooth ratios (neglect idlers for simplicity)
        self.angles["driver"] = norm_deg(self.angles["driver"] + driver_deg)
        for (a, b) in self.mesh:
            ta = self.gears[a].teeth
            tb = self.gears[b].teeth
            # angle transfer: b advances by -a * (ta/tb) for opposite rotation
            delta_b = -driver_deg * (ta / tb) if a == "driver" else - (self.angles[a] / 360.0) * (ta / tb) * 360.0
            self.angles[b] = norm_deg(self.angles[b] + delta_b)

    def get_angle(self, name): return self.angles.get(name, 0.0)

# ---------- AI reconstruction hooks ----------

class ImagingLoader:
    def load(self, path:str):
        # Stub: load imaging data (e.g., micro-CT, RTI, or high-res photos)
        # Return ndarray or image objects in a real setup.
        return {"path": path, "data": None}

class ReconstructionService:
    def __init__(self, model=None):
        self.model = model  # plug in your ML model later

    def infer_gears(self, imaging_blob):
        # Stub: predict gear teeth counts, placements, engravings.
        # Return candidate trains with confidence scores.
        # In a real pipeline: segmentation -> OCR -> param fitting -> constraints.
        return [{
            "train": {"driver":64, "sun":38, "moon":53, "phase":127},
            "confidence": 0.42
        }]

    def apply_constraints(self, candidates):
        # Enforce astronomical constraints (ratios close to synodic, draconic, anomalistic cycles)
        # Return filtered candidates.
        return candidates

# ---------- GUI ----------

class AntikytheraGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Antikythera Simulator — High-Tech Edition")
        self.geometry("920x640")
        self.resizable(False, False)

        self.ephem = Ephemeris()
        self.gears = GearMesh()
        self.running = False
        self.tick_interval_ms = 250
        self.driver_deg_per_tick = 0.5  # small gear advance per tick

        self._build_ui()
        self._draw_dials()

    def _build_ui(self):
        # Controls
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.time_label = ttk.Label(ctrl, text="Time: —")
        self.time_label.pack(side=tk.LEFT, padx=5)

        self.phase_label = ttk.Label(ctrl, text="Phase: —")
        self.phase_label.pack(side=tk.LEFT, padx=5)

        self.zodiac_label = ttk.Label(ctrl, text="Sun in: —")
        self.zodiac_label.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(ctrl, text="Start", command=self.start)
        self.run_btn.pack(side=tk.RIGHT, padx=5)
        self.step_btn = ttk.Button(ctrl, text="Step", command=self.step)
        self.step_btn.pack(side=tk.RIGHT, padx=5)
        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self.stop)
        self.stop_btn.pack(side=tk.RIGHT, padx=5)

        # Canvas
        self.canvas = tk.Canvas(self, width=900, height=520, bg="#111")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)

        # Status
        self.status = tk.Text(self, height=6, bg="#0f0f0f", fg="#cfcfcf")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self._status("Initialized.")

        # Menu for AI
        menubar = tk.Menu(self)
        ai_menu = tk.Menu(menubar, tearoff=0)
        ai_menu.add_command(label="Load imaging...", command=self._load_imaging)
        ai_menu.add_command(label="Infer gears", command=self._infer_gears)
        menubar.add_cascade(label="AI", menu=ai_menu)
        self.config(menu=menubar)

    def _status(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.status.insert(tk.END, f"[{ts}] {msg}\n")
        self.status.see(tk.END)

    def start(self):
        if not self.running:
            self.running = True
            self._status("Simulation started.")
            self.after(self.tick_interval_ms, self._tick)

    def stop(self):
        if self.running:
            self.running = False
            self._status("Simulation stopped.")

    def step(self):
        self._advance_sim()
        self._draw_dials()

    def _tick(self):
        if not self.running: return
        self._advance_sim()
        self._draw_dials()
        self.after(self.tick_interval_ms, self._tick)

    def _advance_sim(self):
        # Advance ephemeris by a small fraction of a day per tick
        self.ephem.update(self.ephem.now + datetime.timedelta(minutes=30))
        # Advance gear train
        self.gears.step(self.driver_deg_per_tick)

    def _draw_dials(self):
        self.canvas.delete("all")

        # Center
        cx, cy = 450, 260
        r_outer = 220
        r_inner = 180
        r_phase = 140

        # Zodiac ring
        self._draw_ring(cx, cy, r_outer, "#333", "#666")
        sun_lon = self.ephem.sun_ecliptic_longitude_deg()
        self._draw_pointer(cx, cy, r_outer, sun_lon, "#ffcc00", width=4)

        # Zodiac labels
        for i, sign in enumerate(Ephemeris.ZODIAC):
            ang = deg2rad(i * 30 + 15)
            tx = cx + (r_outer + 20) * math.cos(ang)
            ty = cy + (r_outer + 20) * math.sin(ang)
            self.canvas.create_text(tx, ty, text=sign, fill="#bbbbbb", font=("Segoe UI", 10,"bold"))

        # Calendar inner ring
        self._draw_ring(cx, cy, r_inner, "#222", "#555")

        # Moon pointer (ecliptic longitude)
        moon_lon = self.ephem.moon_ecliptic_longitude_deg()
        self._draw_pointer(cx, cy, r_inner, moon_lon, "#66aaff", width=3)

        # Lunar phase dial
        phase_ang = self.ephem.lunar_phase_deg()
        self._draw_ring(cx, cy, r_phase, "#222", "#444")
        self._draw_pointer(cx, cy, r_phase, phase_ang, "#ffffff", width=2)

        # Gear debug angles
        self._draw_gear_debug(cx, cy)

        # Labels
        self.time_label.config(text=f"Time: {self.ephem.now.isoformat(timespec='seconds')}")
        self.phase_label.config(text=f"Phase: {self.ephem.lunar_phase_name()}")
        self.zodiac_label.config(text=f"Sun in: {Ephemeris.ZODIAC[self.ephem.sun_zodiac_index()]}")

        # Eclipse windows (short preview)
        wins = self.ephem.predict_eclipse_windows(lookahead_days=120)
        self._status(f"Eclipse windows (next 120d): {', '.join([w['date']+':'+w['type'] for w in wins[:6]])}")

    def _draw_ring(self, cx, cy, r, stroke="#333", tick="#666"):
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=stroke, width=2)
        # ticks every 30°
        for deg in range(0, 360, 30):
            ang = deg2rad(deg)
            x1 = cx + r * math.cos(ang)
            y1 = cy + r * math.sin(ang)
            x2 = cx + (r-10) * math.cos(ang)
            y2 = cy + (r-10) * math.sin(ang)
            self.canvas.create_line(x1, y1, x2, y2, fill=tick)

    def _draw_pointer(self, cx, cy, r, angle_deg, color="#ffcc00", width=3):
        ang = deg2rad(angle_deg)
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        self.canvas.create_line(cx, cy, x, y, fill=color, width=width)
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, outline="")

    def _draw_gear_debug(self, cx, cy):
        g = self.gears
        # show a compact gear angle visualization
        labels = [
            ("Driver", g.get_angle("driver")),
            ("Sun", g.get_angle("sun")),
            ("Moon", g.get_angle("moon")),
            ("Phase", g.get_angle("phase")),
        ]
        y = cy + 260
        x = 40
        for name, angle in labels:
            self.canvas.create_text(x, y, text=f"{name}: {angle:.1f}°", fill="#aaaaaa", anchor="w", font=("Consolas", 10))
            y -= 18

    def _load_imaging(self):
        loader = ImagingLoader()
        blob = loader.load("example_scan.ct")
        self._status(f"Loaded imaging: {blob['path']}")

    def _infer_gears(self):
        recon = ReconstructionService()
        candidates = recon.infer_gears({"path":"example_scan.ct"})
        filtered = recon.apply_constraints(candidates)
        best = max(filtered, key=lambda c: c["confidence"])
        self._status(f"AI candidate train: {best['train']} (conf={best['confidence']:.2f})")
        # Optionally apply to gear mesh
        self._apply_candidate_train(best["train"])

    def _apply_candidate_train(self, train):
        # Rebuild gear mesh with candidate teeth counts
        for k, v in train.items():
            if k in self.gears.gears:
                self.gears.gears[k].teeth = v
        self._status("Applied AI gear train to simulation.")

if __name__ == "__main__":
    app = AntikytheraGUI()
    app.mainloop()

