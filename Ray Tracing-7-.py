import importlib
import subprocess
import sys
import math
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# --- Auto-install minimal deps ---
def auto_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "pip", "install", package])

for pkg in ["numpy", "matplotlib"]:
    auto_install(pkg)

# ===============================
# Physics helpers
# ===============================
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def reflect(I, N):
    return I - 2.0 * np.dot(I, N) * N

def refract(I, N, n1, n2):
    eta = n1 / n2
    cos_i = -np.dot(N, I)
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return None
    return eta * I + (eta * cos_i - math.sqrt(k)) * N

# ===============================
# Triangular prism geometry
# ===============================
class TriangularPrism:
    def __init__(self, height=1.2, length=3.0, apex_angle_deg=60.0):
        self.height = float(height)
        self.length = float(length)
        self.apex_angle_deg = float(apex_angle_deg)
        self._build()

    def _build(self):
        H = self.height
        apex = np.array([0.0, 0.0, +H/2])
        base_z = -H/2
        theta = math.radians(self.apex_angle_deg / 2.0)
        dx = H * math.tan(theta)
        v1 = np.array([-dx, 0.0, base_z])
        v2 = np.array([+dx, 0.0, base_z])

        self.tri_vertices = np.array([apex, v1, v2])
        L = self.length
        self.ymin, self.ymax = -L/2.0, +L/2.0

        self.faces = []
        n1 = np.cross(v1 - apex, np.array([0.0, 1.0, 0.0]))
        n2 = np.cross(v2 - apex, np.array([0.0, 1.0, 0.0]))
        n3 = np.cross(v2 - v1, np.array([0.0, 1.0, 0.0]))
        self.faces.append((apex, normalize(n1), "side1"))
        self.faces.append((apex, normalize(n2), "side2"))
        self.faces.append((v1, normalize(n3), "base"))
        self.faces.append((np.array([0.0, self.ymin, 0.0]), np.array([0.0, -1.0, 0.0]), "end_min"))
        self.faces.append((np.array([0.0, self.ymax, 0.0]), np.array([0.0, +1.0, 0.0]), "end_max"))

    def _inside_triangle(self, p):
        A, B, C = self.tri_vertices
        xz = np.array([p[0], p[2]])
        A2, B2, C2 = np.array([A[0], A[2]]), np.array([B[0], B[2]]), np.array([C[0], C[2]])
        v0, v1, v2 = C2 - A2, B2 - A2, xz - A2
        dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
        dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-9:
            return False
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        return (u >= -1e-6) and (v >= -1e-6) and (u + v <= 1.0 + 1e-6)

    def _point_on_face(self, p, ftype, tol=1e-4):
        if ftype in ("end_min", "end_max"):
            if ftype == "end_min" and abs(p[1] - self.ymin) > 1e-3: return False
            if ftype == "end_max" and abs(p[1] - self.ymax) > 1e-3: return False
            return self._inside_triangle(p)
        else:
            if (p[1] < self.ymin - tol) or (p[1] > self.ymax + tol):
                return False
            return self._inside_triangle(p)

    def ray_intersections(self, p0, d, epsilon=1e-6):
        hits = []
        for point, normal, ftype in self.faces:
            denom = np.dot(normal, d)
            if abs(denom) < 1e-9: continue
            t = np.dot(normal, (point - p0)) / denom
            if t > epsilon:
                pi = p0 + t * d
                if self._point_on_face(pi, ftype):
                    hits.append((t, pi, normal, ftype))
        hits.sort(key=lambda x: x[0])
        return hits

# ===============================
# Ray bundle with dispersion
# ===============================
class RayBundle:
    def __init__(self, origin, direction, n_air=1.000277, n_rgb=(1.513, 1.517, 1.521)):
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))
        self.n_air = float(n_air)
        self.n_rgb = n_rgb
        self.paths = {"R": [self.origin.copy()], "G": [self.origin.copy()], "B": [self.origin.copy()]}

    def trace_through_prism(self, prism, max_bounces=12):
        colors = ["R", "G", "B"]
        indices = {"R": self.n_rgb[0], "G": self.n_rgb[1], "B": self.n_rgb[2]}
        outs = {}
        for c in colors:
            p, d, n1, inside = self.origin.copy(), self.direction.copy(), self.n_air, False
            path = [p.copy()]
            bounces = 0
            while bounces < max_bounces:
                hits = prism.ray_intersections(p, d)
                if not hits:
                    path.append(p + d * 5.0)
                    break
                t, pi, N_out, ftype = hits[0]
                N = N_out if np.dot(N_out, d) <= 0 else -N_out
                path.append(pi.copy())
                if not inside:
                    n2 = indices[c]
                    T = refract(d, N, n1, n2)
                    if T is None:
                        d = reflect(d, N); p = pi + d * 1e-4; bounces += 1; continue
                    d = normalize(T); p = pi + d * 1e-4; inside = True; n1 = n2
                else:
                    n2 = self.n_air
                    T = refract(d, N, n1, n2)
                    if T is None:
                        d = reflect(d, N); p = pi + d * 1e-4; bounces += 1; continue
                    d = normalize(T); p = pi + d * 1e-4; inside = False; n1 = n2
                    path.append(p + d * 5.0); break
            outs[c] = path
            self.paths[c] = path
        return outs

# ===============================
# Visualization
# ===============================
def draw_prism(ax, prism, alpha=0.28):
    A, B, C = prism.tri_vertices
    for yface in [prism.ymin, prism.ymax]:
        poly = np.array([[A[0], yface, A[2]], [B[0], yface, B[2]], [C[0], yface, C[2]]])
        verts = [poly]
        tri_collection = Poly3DCollection(verts, facecolors='lightsteelblue', alpha=alpha, edgecolor='gray', linewidths=0.8)
        ax.add_collection3d(tri_collection)
    edges = [
        (np.array([A[0], prism.ymin, A[2]]), np.array([A[0], prism.ymax, A[2]])),
        (np.array([B[0], prism.ymin, B[2]]), np.array([B[0], prism.ymax, B[2]])),
        (np.array([C[0], prism.ymin, C[2]]), np.array([C[0], prism.ymax, C[2]])),
        (np.array([A[0], prism.ymin, A[2]]), np.array([B[0], prism.ymin, B[2]])),
        (np.array([B[0], prism.ymin, B[2]]), np.array([C[0], prism.ymin, C[2]])),
        (np.array([C[0], prism.ymin, A[2]]), np.array([A[0], prism.ymin, A[2]])),
        (np.array([A[0], prism.ymax, A[2]]), np.array([B[0], prism.ymax, B[2]])),
        (np.array([B[0], prism.ymax, B[2]]), np.array([C[0], prism.ymax, C[2]])),
        (np.array([C[0], prism.ymax, C[2]]), np.array([A[0], prism.ymax, A[2]])),
    ]
    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', linewidth=0.8)

def draw_rays(ax, paths, pickable=False, colors_map=None):
    if colors_map is None:
        colors_map = {"R": "red", "G": "green", "B": "blue"}
    artists = {"R": None, "G": None, "B": None}
    for c, path in paths.items():
        arr = np.array(path)
        line, = ax.plot(arr[:,0], arr[:,1], arr[:,2],
                        color=colors_map[c], linewidth=2, picker=pickable)
        ax.scatter(arr[:,0], arr[:,1], arr[:,2], color=colors_map[c], s=8, alpha=0.7)
        artists[c] = line
    return artists

# ===============================
# Autonomous AI agent
# ===============================
class PrismAI(threading.Thread):
    """
    Autonomous agent that optimizes prism parameters by probing the environment.
    Modes:
      - 'dispersion': maximize angular separation of R, G, B exit rays
      - 'focus': minimize angular spread (align rays)
    Uses stochastic hill-climbing with annealing and safeguards.
    """
    def __init__(self, gui, mode='dispersion', step=1.0, cooldown=0.95, interval=0.8):
        super().__init__(daemon=True)
        self.gui = gui
        self.mode = mode
        self.step = step
        self.cooldown = cooldown
        self.interval = interval
        self._stop = threading.Event()
        self.best_score = None
        self.best_state = None

    def stop(self):
        self._stop.set()

    def run(self):
        self.gui.ai_status("AI starting...")
        # Capture initial state
        state = self._get_state()
        score, meta = self._evaluate(state)
        self.best_score, self.best_state = score, state
        self.gui.ai_status(f"AI initialized: score={score:.3f} mode={self.mode}")
        rng = np.random.default_rng()

        while not self._stop.is_set():
            # Propose small random move in parameters
            proposal = dict(state)
            # Pick parameter to tweak
            choice = rng.choice(["apex_angle", "prism_height", "prism_length", "inc_theta", "inc_phi", "ray_offset"])
            delta = (rng.normal(0, self.step))
            # Parameter-specific scaling and bounds
            if choice == "apex_angle":
                proposal["apex_angle"] = float(np.clip(proposal["apex_angle"] + delta, 25.0, 120.0))
            elif choice == "prism_height":
                proposal["prism_height"] = float(np.clip(proposal["prism_height"] + delta * 0.05, 0.5, 2.5))
            elif choice == "prism_length":
                proposal["prism_length"] = float(np.clip(proposal["prism_length"] + delta * 0.1, 1.0, 6.0))
            elif choice == "inc_theta":
                proposal["inc_theta"] = float(np.clip(proposal["inc_theta"] + delta, -60.0, 60.0))
            elif choice == "inc_phi":
                proposal["inc_phi"] = float((proposal["inc_phi"] + delta * 3.0) % 360.0)
            elif choice == "ray_offset":
                proposal["ray_offset"] = float(np.clip(proposal["ray_offset"] + delta * 0.2, -6.0, 0.0))

            # Evaluate proposal
            score_p, meta_p = self._evaluate(proposal)

            accept = False
            if self.best_score is None:
                accept = True
            else:
                if self.mode == 'dispersion':
                    accept = score_p > self.best_score
                else:  # focus
                    accept = score_p < self.best_score

                # Annealed acceptance for near-neutral steps
                if not accept:
                    temperature = max(0.05, self.step * 0.1)
                    prob = math.exp(-abs(score_p - self.best_score) / max(1e-6, temperature))
                    if rng.random() < prob:
                        accept = True

            if accept:
                # Apply to GUI
                self._apply_state(proposal)
                self.best_score, self.best_state = score_p, dict(proposal)
                self.gui.ai_status(f"AI accepted: {choice} -> score={score_p:.3f}")
                state = dict(proposal)
                # Cool steps over time to stabilize
                self.step *= self.cooldown
                self.step = max(self.step, 0.2)
            else:
                self.gui.ai_status(f"AI rejected: {choice} (score {score_p:.3f} vs {self.best_score:.3f})")

            # Sleep
            time.sleep(self.interval)

    def _get_state(self):
        return {
            "apex_angle": self.gui.apex_angle,
            "prism_height": self.gui.prism_height,
            "prism_length": self.gui.prism_length,
            "inc_theta": self.gui.inc_theta,
            "inc_phi": self.gui.inc_phi,
            "ray_offset": self.gui.ray_offset,
        }

    def _apply_state(self, s):
        # Update GUI safely on main thread via after()
        def apply():
            self.gui.apex_angle = s["apex_angle"]
            self.gui.prism_height = s["prism_height"]
            self.gui.prism_length = s["prism_length"]
            self.gui.inc_theta = s["inc_theta"]
            self.gui.inc_phi = s["inc_phi"]
            self.gui.ray_offset = s["ray_offset"]
            self.gui._render()
        try:
            self.gui.root.after(0, apply)
        except Exception:
            apply()

    def _evaluate(self, s):
        """
        Score based on paths from current state:
          - Dispersion: maximize pairwise angular separation between R,G,B exit rays.
          - Focus: minimize angular separation.
        Penalties: excessive internal bounces, missing exits, unstable directions.
        """
        # Build prism and bundle from proposed state
        prism = TriangularPrism(height=s["prism_height"], length=s["prism_length"], apex_angle_deg=s["apex_angle"])
        origin = np.array([0.0, s["ray_offset"], 0.0], dtype=float)
        th = math.radians(s["inc_theta"])
        ph = math.radians(s["inc_phi"])
        direction = normalize(np.array([math.cos(th)*math.sin(ph), math.cos(th)*math.cos(ph), math.sin(th)], dtype=float))
        bundle = RayBundle(origin, direction, n_air=1.000277, n_rgb=(self.gui.nR, self.gui.nG, self.gui.nB))
        paths = bundle.trace_through_prism(prism, max_bounces=12)

        # Extract exit vectors (last segment direction) for each color
        exit_dirs = {}
        bounces = {}
        for c, path in paths.items():
            arr = np.array(path)
            if len(arr) >= 2:
                v = normalize(arr[-1] - arr[-2])
                exit_dirs[c] = v
                bounces[c] = max(0, len(arr) - 2)
            else:
                exit_dirs[c] = None
                bounces[c] = 0

        # Pairwise angular separation
        def ang_between(u, v):
            if u is None or v is None:
                return 0.0
            dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
            return math.degrees(math.acos(dot))

        sep_RG = ang_between(exit_dirs.get("R"), exit_dirs.get("G"))
        sep_GB = ang_between(exit_dirs.get("G"), exit_dirs.get("B"))
        sep_BR = ang_between(exit_dirs.get("B"), exit_dirs.get("R"))
        sep_sum = sep_RG + sep_GB + sep_BR

        # Penalties for missing exits or too many bounces (internal reflections)
        miss_penalty = sum(1 for c in ["R","G","B"] if exit_dirs.get(c) is None) * 100.0
        bounce_penalty = sum(max(0, bounces.get(c, 0) - 6) for c in ["R","G","B"]) * 5.0

        # Stability penalty: large vertical components make visualization cramped
        vert_penalty = sum(abs(exit_dirs[c][2]) if exit_dirs.get(c) is not None else 0.0 for c in ["R","G","B"]) * 2.0

        # Score
        if self.mode == 'dispersion':
            score = sep_sum - miss_penalty - bounce_penalty - vert_penalty
        else:
            score = -sep_sum - miss_penalty - bounce_penalty - vert_penalty

        meta = {
            "sep_sum": sep_sum,
            "miss_penalty": miss_penalty,
            "bounce_penalty": bounce_penalty,
            "vert_penalty": vert_penalty
        }
        return score, meta

# ===============================
# GUI with interactive canvas and AI
# ===============================
class PrismGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Prism + Autonomous AI")
        self.root.geometry("1020x600")

        # State
        self.apex_angle = 60.0
        self.prism_height = 1.2
        self.prism_length = 3.0
        self.nR = 1.513
        self.nG = 1.517
        self.nB = 1.521
        self.inc_theta = 10.0
        self.inc_phi = 330.0  # use 0..360 for azimuth
        self.ray_offset = -3.0
        self.dragging_origin = False
        self.ray_artists = {}
        self.last_intersections = {}
        self.ai_thread = None
        self.ai_running = False

        # Layout: resizable paned
        paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(paned)
        paned.add(left, minsize=380)

        right = tk.Frame(paned, width=340)
        paned.add(right, minsize=280)

        # Figure compact
        self.fig = plt.Figure(figsize=(4.2, 3.0))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect interactive events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("pick_event", self.on_pick)

        # Scrollable controls
        ctrl_canvas = tk.Canvas(right, borderwidth=0, highlightthickness=0)
        vscroll = tk.Scrollbar(right, orient="vertical", command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ctrl_inner = tk.Frame(ctrl_canvas)
        self.ctrl_inner.bind("<Configure>", lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.create_window((0, 0), window=self.ctrl_inner, anchor="nw")
        ctrl_canvas.bind_all("<MouseWheel>", lambda e: ctrl_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Build controls
        self.status_label = tk.Label(self.ctrl_inner, text="Ready", font=("Arial", 9))
        self.status_label.pack(pady=(6, 4))

        self._section("Prism geometry")
        self._slider("Apex angle (deg)", 20, 120, self.apex_angle, lambda v: self._set("apex_angle", float(v)))
        self._slider("Prism height", 0.4, 2.5, self.prism_height, lambda v: self._set("prism_height", float(v)), res=0.05)
        self._slider("Prism length", 1.0, 6.0, self.prism_length, lambda v: self._set("prism_length", float(v)), res=0.1)

        self._section("Refractive indices (RGB)")
        self._slider("nR (red)", 1.45, 1.60, self.nR, lambda v: self._set("nR", float(v)), res=0.001)
        self._slider("nG (green)", 1.45, 1.60, self.nG, lambda v: self._set("nG", float(v)), res=0.001)
        self._slider("nB (blue)", 1.45, 1.60, self.nB, lambda v: self._set("nB", float(v)), res=0.001)

        self._section("Incident ray")
        self._slider("Elevation θ (deg)", -60, 60, self.inc_theta, lambda v: self._set("inc_theta", float(v)))
        self._slider("Azimuth φ (deg)", 0, 360, self.inc_phi, lambda v: self._set("inc_phi", float(v)))
        self._slider("Ray Y offset", -6.0, 0.0, self.ray_offset, lambda v: self._set("ray_offset", float(v)), res=0.1)

        # AI controls
        self._section("Autonomous AI")
        ai_row = tk.Frame(self.ctrl_inner)
        ai_row.pack(fill=tk.X, pady=3)
        tk.Label(ai_row, text="Mode", font=("Arial", 9), width=10, anchor="w").pack(side=tk.LEFT)
        self.ai_mode_var = tk.StringVar(value="dispersion")
        tk.OptionMenu(ai_row, self.ai_mode_var, "dispersion", "focus").pack(side=tk.LEFT, padx=4)

        ai_btns = tk.Frame(self.ctrl_inner)
        ai_btns.pack(fill=tk.X, pady=(8, 6))
        tk.Button(ai_btns, text="Start AI", command=self.start_ai, font=("Arial", 9)).pack(side=tk.LEFT, padx=4)
        tk.Button(ai_btns, text="Pause AI", command=self.pause_ai, font=("Arial", 9)).pack(side=tk.LEFT, padx=4)
        tk.Button(ai_btns, text="Reset", command=self._reset, font=("Arial", 9)).pack(side=tk.LEFT, padx=4)
        tk.Button(ai_btns, text="Randomize", command=self._randomize, font=("Arial", 9)).pack(side=tk.LEFT, padx=4)

        self.ai_info = tk.Label(self.ctrl_inner, text="AI: idle", font=("Arial", 9), fg="purple")
        self.ai_info.pack(anchor="w", pady=(2, 6))

        # Initial draw
        self._render()

    # Controls helpers
    def _section(self, text):
        tk.Label(self.ctrl_inner, text=text, font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 4))

    def _slider(self, label, frm, to, init, cmd, res=1.0):
        row = tk.Frame(self.ctrl_inner)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=label, font=("Arial", 9), width=14, anchor="w").pack(side=tk.LEFT)
        s = tk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, resolution=res,
                     command=lambda v: cmd(v), length=200)
        s.set(init)
        s.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _set(self, attr, val):
        setattr(self, attr, val)
        self._render()

    def _reset(self):
        self.apex_angle = 60.0
        self.prism_height = 1.2
        self.prism_length = 3.0
        self.nR = 1.513
        self.nG = 1.517
        self.nB = 1.521
        self.inc_theta = 10.0
        self.inc_phi = 330.0
        self.ray_offset = -3.0
        self.status_label.config(text="Reset to defaults")
        self._render()

    def _randomize(self):
        rng = np.random.default_rng()
        self.apex_angle = float(rng.uniform(40, 100))
        self.prism_height = float(rng.uniform(0.6, 2.0))
        self.prism_length = float(rng.uniform(1.5, 5.0))
        self.nR = float(rng.uniform(1.48, 1.56))
        self.nG = float(self.nR + rng.uniform(0.002, 0.008))
        self.nB = float(self.nG + rng.uniform(0.002, 0.008))
        self.inc_theta = float(rng.uniform(-40, 40))
        self.inc_phi = float(rng.uniform(0, 360))
        self.ray_offset = float(rng.uniform(-5.0, -1.0))
        self.status_label.config(text="Randomized parameters")
        self._render()

    # Incident direction from θ, φ
    def _incident_direction(self):
        th = math.radians(self.inc_theta)
        ph = math.radians(self.inc_phi)
        dx = math.cos(th) * math.sin(ph)
        dy = math.cos(th) * math.cos(ph)
        dz = math.sin(th)
        return normalize(np.array([dx, dy, dz], dtype=float))

    # --- AI controls ---
    def start_ai(self):
        if self.ai_running:
            self.ai_status("AI already running")
            return
        self.ai_thread = PrismAI(self, mode=self.ai_mode_var.get(), step=1.2, cooldown=0.93, interval=0.8)
        self.ai_thread.start()
        self.ai_running = True
        self.ai_status("AI started")

    def pause_ai(self):
        if self.ai_thread:
            self.ai_thread.stop()
            self.ai_thread = None
        self.ai_running = False
        self.ai_status("AI paused")

    def ai_status(self, text):
        self.ai_info.config(text=f"AI: {text}")

    # --- Interactive event handlers ---
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.dragging_origin = True
            if event.ydata is not None:
                self.ray_offset = float(event.ydata)
                self.status_label.config(text=f"Dragging origin: y={self.ray_offset:.2f}")
                self._render()

    def on_release(self, event):
        if event.button == 1:
            self.dragging_origin = False
            self.status_label.config(text="Released drag")

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if (event.xdata is not None) and (event.ydata is not None) and (event.zdata is not None):
            self.status_label.config(text=f"Hover ({event.xdata:.2f}, {event.ydata:.2f}, {event.zdata:.2f})")
        if self.dragging_origin and (event.ydata is not None):
            self.ray_offset = float(event.ydata)
            self._render()

    def on_key(self, event):
        if event.key == "r":
            self._reset()
        elif event.key == "w":
            self.inc_theta = max(-60.0, min(60.0, self.inc_theta + 2.0))
            self._render()
        elif event.key == "s":
            self.inc_theta = max(-60.0, min(60.0, self.inc_theta - 2.0))
            self._render()
        elif event.key == "a":
            self.inc_phi = (self.inc_phi - 5.0) % 360.0
            self._render()
        elif event.key == "d":
            self.inc_phi = (self.inc_phi + 5.0) % 360.0
            self._render()
        elif event.key == "m":
            for c, artist in self.ray_artists.items():
                if artist is not None:
                    lw = artist.get_linewidth()
                    artist.set_linewidth(4 if lw < 3 else 2)
            self.canvas.draw()
            self.status_label.config(text="Toggled ray thickness")
        elif event.key == "p":
            # Quick toggle AI
            if self.ai_running: self.pause_ai()
            else: self.start_ai()

    def on_pick(self, event):
        artist = event.artist
        for c, a in self.ray_artists.items():
            if a is artist:
                a.set_color("yellow")
                a.set_linewidth(4)
                self.canvas.draw()
                self.status_label.config(text=f"Picked {c} ray")
            else:
                if a is not None:
                    base_color = {"R":"red","G":"green","B":"blue"}[c]
                    a.set_color(base_color)
                    a.set_linewidth(2)
        self.canvas.draw()

    # --- Render ---
    def _render(self):
        self.ax.clear()
        prism = TriangularPrism(height=self.prism_height, length=self.prism_length, apex_angle_deg=self.apex_angle)
        draw_prism(self.ax, prism, alpha=0.28)

        origin = np.array([0.0, self.ray_offset, 0.0], dtype=float)
        direction = self._incident_direction()

        bundle = RayBundle(origin, direction, n_air=1.000277, n_rgb=(self.nR, self.nG, self.nB))
        paths = bundle.trace_through_prism(prism, max_bounces=12)
        self.ray_artists = draw_rays(self.ax, paths, pickable=True, colors_map={"R":"red","G":"green","B":"blue"})

        # Bounds and view
        X = max(1.5, np.ptp(prism.tri_vertices[:,0]) / 2 + 1.0)
        Z = prism.height/2 + 1.0
        Y = prism.length/2 + 3.0
        self.ax.set_xlim(-X, X)
        self.ax.set_ylim(-Y, Y)
        self.ax.set_zlim(-Z, Z)
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
        self.ax.view_init(elev=18, azim=40)
        self.ax.set_title("Interactive Prism + Autonomous AI", fontsize=10)
        self.canvas.draw()

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interactive Prism Simulator with Autonomous AI")
    # Optional DPI awareness (Windows)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = PrismGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

