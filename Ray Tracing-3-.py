import importlib
import subprocess
import sys
import threading
import time
import math
import socket

# --- Auto-loader for required libraries ---
def auto_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["numpy", "matplotlib", "psutil", "GPUtil"]:
    auto_install(pkg)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import psutil

# Optional GPU via GPUtil (gracefully handles absence)
try:
    import GPUtil
except Exception:
    GPUtil = None

# --- Simulation constants ---
EMPTY, MIRROR, TOWEL, VIEWER, OBJECT = 0, 1, 2, 3, 4

# --- System telemetry collector ---
class MetricsCollector(threading.Thread):
    def __init__(self, interval=1.0, latency_host="8.8.8.8", latency_port=53, latency_timeout=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.metrics = {
            "cpu_percent": 0.0,
            "mem_percent": 0.0,
            "disk_read_bps": 0.0,
            "disk_write_bps": 0.0,
            "net_recv_bps": 0.0,
            "net_sent_bps": 0.0,
            "proc_count": 0,
            "conn_count": 0,
            "gpu_load": 0.0,
            "gpu_mem_percent": 0.0,
            "latency_ms": None,
            "events": []
        }
        self._prev_disk = psutil.disk_io_counters() if psutil.disk_io_counters() else None
        self._prev_net = psutil.net_io_counters() if psutil.net_io_counters() else None
        self._prev_time = time.time()
        self.latency_host = latency_host
        self.latency_port = latency_port
        self.latency_timeout = latency_timeout

    def _probe_latency(self):
        try:
            t0 = time.time()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.latency_timeout)
            s.connect((self.latency_host, self.latency_port))
            s.close()
            return (time.time() - t0) * 1000.0
        except Exception:
            return None

    def run(self):
        psutil.cpu_percent(interval=None)
        # Keep a small window of history to detect spikes
        net_hist = []
        disk_hist = []
        cpu_hist = []
        while not self._stop.is_set():
            start = time.time()
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent if psutil.virtual_memory() else 0.0
            now = time.time()

            disk = psutil.disk_io_counters() if psutil.disk_io_counters() else None
            disk_read_bps = disk_write_bps = 0.0
            if self._prev_disk and disk:
                dt = max(now - self._prev_time, 1e-6)
                disk_read_bps = (disk.read_bytes - self._prev_disk.read_bytes) / dt
                disk_write_bps = (disk.write_bytes - self._prev_disk.write_bytes) / dt
                self._prev_disk = disk

            net = psutil.net_io_counters() if psutil.net_io_counters() else None
            net_recv_bps = net_sent_bps = 0.0
            if self._prev_net and net:
                dt = max(now - self._prev_time, 1e-6)
                net_recv_bps = (net.bytes_recv - self._prev_net.bytes_recv) / dt
                net_sent_bps = (net.bytes_sent - self._prev_net.bytes_sent) / dt
                self._prev_net = net

            self._prev_time = now

            try:
                proc_count = len(psutil.pids())
            except Exception:
                proc_count = 0
            try:
                conn_count = len(psutil.net_connections(kind='inet'))
            except Exception:
                conn_count = 0

            # GPU metrics if available
            gpu_load = 0.0
            gpu_mem_percent = 0.0
            try:
                if GPUtil:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_load = float(np.mean([g.load for g in gpus])) * 100.0
                        gpu_mem_percent = float(np.mean([g.memoryUtil for g in gpus])) * 100.0
            except Exception:
                pass

            # Latency probe (non-blocking-ish, short timeout)
            latency_ms = self._probe_latency()

            # Event detection: track recent values and flag spikes
            net_total = net_recv_bps + net_sent_bps
            disk_total = disk_read_bps + disk_write_bps
            net_hist.append(net_total); disk_hist.append(disk_total); cpu_hist.append(cpu)
            net_hist = net_hist[-10:]; disk_hist = disk_hist[-10:]; cpu_hist = cpu_hist[-10:]

            events = []
            def is_spike(arr, k=2.0):
                if len(arr) < 5:
                    return False
                mu = float(np.mean(arr))
                sigma = float(np.std(arr)) + 1e-6
                return arr[-1] > mu + k * sigma

            if is_spike(net_hist, 2.5): events.append("Network spike")
            if is_spike(disk_hist, 2.5): events.append("Disk spike")
            if is_spike(cpu_hist, 2.5): events.append("CPU spike")
            if latency_ms is not None and latency_ms > 200.0: events.append("High latency")

            with self.lock:
                self.metrics.update({
                    "cpu_percent": float(cpu),
                    "mem_percent": float(mem),
                    "disk_read_bps": float(disk_read_bps),
                    "disk_write_bps": float(disk_write_bps),
                    "net_recv_bps": float(net_recv_bps),
                    "net_sent_bps": float(net_sent_bps),
                    "proc_count": int(proc_count),
                    "conn_count": int(conn_count),
                    "gpu_load": float(gpu_load),
                    "gpu_mem_percent": float(gpu_mem_percent),
                    "latency_ms": None if latency_ms is None else float(latency_ms),
                    "events": events
                })

            elapsed = time.time() - start
            time.sleep(max(0.0, self.interval - elapsed))

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)

    def stop(self):
        self._stop.set()

# --- Multi-plane 3D mirror simulation ---
class MultiPlaneMirrorSimulation:
    def __init__(self, size=(20,20,20), planes=None, exposed_windows=None):
        """
        planes: list of z positions for mirror planes (e.g., [6, 10, 14])
        exposed_windows: list of ((xmin, xmax), (ymin, ymax)) per plane
        """
        self.size = size
        self.grid = np.zeros(size, dtype=int)
        self.viewer_pos = None
        self.object_pos = None
        self.planes = planes or [6, 10, 14]  # storage, network, application layers
        # Default exposed windows centered
        X, Y, _ = self.size
        default = [((X//2-3, X//2+3), (Y//2-3, Y//2+3)) for _ in self.planes]
        self.exposed_windows = exposed_windows or default

    def in_bounds(self,x,y,z):
        X,Y,Z = self.size
        return (0<=x<X) and (0<=y<Y) and (0<=z<Z)

    def is_exposed(self, plane_idx, x, y):
        (xmin, xmax), (ymin, ymax) = self.exposed_windows[plane_idx]
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def place_viewer(self,pos):
        self.viewer_pos = pos
        self.grid[pos] = VIEWER

    def place_object(self,pos):
        self.object_pos = pos
        self.grid[pos] = OBJECT

    def place_planes(self):
        for p_idx, z in enumerate(self.planes):
            for x in range(self.size[0]):
                for y in range(self.size[1]):
                    self.grid[x,y,z] = MIRROR if self.is_exposed(p_idx, x, y) else TOWEL

    def reflect_across_plane(self, point, z_plane):
        x,y,z = point
        return (x, y, 2*z_plane - z)

    def can_see_reflection(self, samples_per_segment=200):
        """
        Multi-bounce rule:
        - Reflect object across planes successively from farthest to nearest to compute a target point Rk.
        - Cast a ray from viewer to Rk.
        - As the ray crosses each plane z, the intersection (rounded to grid) must lie within that plane's exposed window.
        """
        if self.viewer_pos is None or self.object_pos is None: return False
        # Compute composite reflection of object across all planes
        R = self.object_pos
        # Reflect across planes from last to first (far to near)
        for z in reversed(self.planes):
            R = self.reflect_across_plane(R, z)

        V = np.array(self.viewer_pos, dtype=float)
        R = np.array(R, dtype=float)

        # Determine ordered crossings (ascending by z, based on viewer->R ray direction)
        z_set = sorted(self.planes)
        # Sample along ray and detect first occurrences of z-plane crossings
        crossings = {z: None for z in z_set}
        last_z = None
        for t in np.linspace(0.0, 1.0, samples_per_segment):
            P = V + t * (R - V)
            x, y, z = P
            z_round = int(round(z))
            # Detect crossing when we pass near plane z (tolerance)
            for zp in z_set:
                if abs(z - zp) < 0.25:  # tolerance for float sampling
                    ix, iy = int(round(x)), int(round(y))
                    if self.in_bounds(ix, iy, zp) and crossings[zp] is None:
                        crossings[zp] = (ix, iy)
        # Validate exposure at each plane
        for idx, zp in enumerate(z_set):
            if crossings[zp] is None:
                return False
            ix, iy = crossings[zp]
            if not self.is_exposed(idx, ix, iy):
                return False
        return True

    def visualize(self, ax):
        ax.clear()
        # Plot planes
        for p_idx, z in enumerate(self.planes):
            xs_m, ys_m, zs_m = [], [], []
            xs_t, ys_t, zs_t = [], [], []
            for x in range(self.size[0]):
                for y in range(self.size[1]):
                    cell = self.grid[x,y,z]
                    if cell == MIRROR:
                        xs_m.append(x); ys_m.append(y); zs_m.append(z)
                    elif cell == TOWEL:
                        xs_t.append(x); ys_t.append(y); zs_t.append(z)
            ax.scatter(xs_m,ys_m,zs_m,c='cyan',marker='s',s=10,label=f'Plane {p_idx+1} exposed')
            ax.scatter(xs_t,ys_t,zs_t,c='gray',marker='s',s=10,label=f'Plane {p_idx+1} towel')
        # Viewer and object
        if self.viewer_pos:
            ax.scatter(*self.viewer_pos,c='green',s=80,label='Viewer')
        if self.object_pos:
            ax.scatter(*self.object_pos,c='red',s=80,label='Object')
        # Ray path (viewer to composite-reflected object)
        if self.viewer_pos and self.object_pos:
            R = self.object_pos
            for z in reversed(self.planes):
                R = self.reflect_across_plane(R, z)
            ray = np.array([self.viewer_pos, R], dtype=float)
            ax.plot(ray[:,0],ray[:,1],ray[:,2],c='orange',label='Multi-bounce ray')

        X, Y, Z = self.size
        ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_zlim(0, Z)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(loc='upper left')
        ax.set_title("Autonomous Multi-Layer System Mirror")

# --- Mapping: metrics -> scene parameters & windows (multi-plane) ---
def normalize(val, min_v, max_v, size):
    v = 0.0 if max_v <= min_v else (val - min_v) / (max_v - min_v)
    v = max(0.0, min(1.0, v))
    return int(v * (size - 1))

def scaled_window(v, base=2, max_w=8):
    # Log scaling to tame spikes; ensure minimum width
    w = int(min(max_w, max(base, math.log10(v + 1) * (max_w // 2))))
    return max(2, w)

def map_metrics_to_scene_multi(metrics, grid_size, planes):
    X, Y, Z = grid_size
    cpu = metrics["cpu_percent"]
    mem = metrics["mem_percent"]
    net = metrics["net_recv_bps"] + metrics["net_sent_bps"]
    disk = metrics["disk_read_bps"] + metrics["disk_write_bps"]
    procs = metrics["proc_count"]
    conns = metrics["conn_count"]
    gpu_load = metrics.get("gpu_load", 0.0)
    gpu_mem = metrics.get("gpu_mem_percent", 0.0)
    latency_ms = metrics.get("latency_ms", None)

    # Viewer: CPU/MEM on surface layer near first plane
    vx = normalize(cpu, 0, 100, X)
    vy = normalize(mem, 0, 100, Y)
    vz = max(0, min(Z-1, planes[0]-2))

    # Object: deeper internal, influenced by processes/conn and disk
    activity = procs + conns
    ox = normalize(activity, 0, 1000, X)
    oy = normalize(disk, 0, 50_000_000, Y)
    # push deeper with GPU load and latency
    depth_bias = int(min(4, (gpu_load + gpu_mem) / 50.0)) + (1 if (latency_ms and latency_ms > 150) else 0)
    oz = max(0, min(Z-1, planes[-1] + depth_bias))

    viewer_pos = (vx, vy, vz)
    object_pos = (ox, oy, oz)

    # Exposed windows per plane:
    # plane 1 (storage): disk throughput widens window
    # plane 2 (network): net throughput widens window
    # plane 3 (application): CPU/MEM widen window; latency can narrow
    cx, cy = X//2, Y//2
    w_disk = scaled_window(disk, max_w=min(8, min(X, Y)//2))
    w_net  = scaled_window(net,  max_w=min(8, min(X, Y)//2))
    w_app  = scaled_window(cpu + mem, max_w=min(8, min(X, Y)//2))
    if latency_ms and latency_ms > 200:
        w_app = max(2, w_app - 2)

    windows = []
    for idx, z in enumerate(planes):
        if idx == 0:
            w = w_disk
        elif idx == 1:
            w = w_net
        else:
            w = w_app
        xmin, xmax = max(0, cx - w), min(X - 1, cx + w)
        ymin, ymax = max(0, cy - w), min(Y - 1, cy + w)
        windows.append(((xmin, xmax), (ymin, ymax)))

    return viewer_pos, object_pos, windows

# --- GUI wrapper with manual + autonomous control ---
class MirrorGUI:
    def __init__(self, root):
        self.root = root
        self.size = (24,24,24)
        self.planes = [8, 12, 16]  # storage, network, application
        self.metrics_interval = 1.0
        self.collector = MetricsCollector(interval=self.metrics_interval)
        self.collector.start()
        self.autonomous = tk.BooleanVar(value=True)

        # Defaults
        self.viewer_pos = [10,10,self.planes[0]-2]
        self.object_pos = [12,12,self.planes[-1]+2]
        X, Y, _ = self.size
        def win(default_w=4):
            cx, cy = X//2, Y//2
            return [cx-default_w, cx+default_w, cy-default_w, cy+default_w]
        self.windows = [win(4), win(4), win(4)]  # per plane [xmin,xmax,ymin,ymax]

        # Figure
        self.fig = plt.Figure(figsize=(7.5,5.5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Controls
        controls = tk.Frame(root)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(controls, text="Autonomous mode").pack()
        tk.Checkbutton(controls, variable=self.autonomous, command=self.update_scene).pack(pady=4)

        # Viewer sliders
        tk.Label(controls, text="Viewer position").pack()
        self.viewer_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            tk.Label(controls, text=label).pack()
            slider=tk.Scale(controls, from_=0, to=self.size[i]-1, orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_viewer(i,int(val)))
            slider.set(self.viewer_pos[i]); slider.pack()
            self.viewer_sliders.append(slider)

        # Object sliders
        tk.Label(controls, text="Object position").pack(pady=(10,0))
        self.object_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            tk.Label(controls, text=label).pack()
            slider=tk.Scale(controls, from_=0, to=self.size[i]-1, orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_object(i,int(val)))
            slider.set(self.object_pos[i]); slider.pack()
            self.object_sliders.append(slider)

        # Windows per plane
        tk.Label(controls, text="Exposed windows (per plane)").pack(pady=(10,0))
        self.window_sliders = []
        for p_idx in range(len(self.planes)):
            frame = tk.LabelFrame(controls, text=f"Plane {p_idx+1} (z={self.planes[p_idx]})")
            frame.pack(pady=6, fill=tk.X)
            labels=["Xmin","Xmax","Ymin","Ymax"]
            sliders=[]
            for i,label in enumerate(labels):
                tk.Label(frame,text=label).pack()
                slider=tk.Scale(frame, from_=0, to=self.size[0]-1, orient=tk.HORIZONTAL,
                                command=lambda val,p=p_idx,i=i:self.update_window(p,i,int(val)))
                slider.set(self.windows[p_idx][i]); slider.pack()
                sliders.append(slider)
            self.window_sliders.append(sliders)

        # Status + events
        self.status_label = tk.Label(controls, text="Starting...")
        self.status_label.pack(pady=10)
        self.events_label = tk.Label(controls, text="", fg="orange", justify="left")
        self.events_label.pack(pady=4)

        # Buttons
        btn_frame = tk.Frame(controls)
        btn_frame.pack(pady=8, fill=tk.X)
        tk.Button(btn_frame, text="Reset", command=self.reset_defaults).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Pause telemetry", command=self.pause_telemetry)\
            .pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Resume telemetry", command=self.resume_telemetry)\
            .pack(side=tk.LEFT, padx=4)

        # Initial render and schedule
        self.update_scene()
        self.root.after(int(self.metrics_interval*1000), self.autonomous_tick)

    def pause_telemetry(self):
        self.collector.stop()
        self.status_label.config(text="Telemetry paused")

    def resume_telemetry(self):
        if not self.collector.is_alive():
            self.collector = MetricsCollector(interval=self.metrics_interval)
            self.collector.start()
        self.status_label.config(text="Telemetry resumed")

    def reset_defaults(self):
        self.viewer_pos = [10,10,self.planes[0]-2]
        self.object_pos = [12,12,self.planes[-1]+2]
        X, Y, _ = self.size
        cx, cy = X//2, Y//2
        def win_w(w): return [cx-w, cx+w, cy-w, cy+w]
        self.windows = [win_w(4), win_w(4), win_w(4)]
        for i,s in enumerate(self.viewer_sliders): s.set(self.viewer_pos[i])
        for i,s in enumerate(self.object_sliders): s.set(self.object_pos[i])
        for p_idx in range(len(self.planes)):
            for i,s in enumerate(self.window_sliders[p_idx]): s.set(self.windows[p_idx][i])
        self.update_scene()

    def update_viewer(self, i, val):
        self.viewer_pos[i] = val
        self.update_scene()

    def update_object(self, i, val):
        self.object_pos[i] = val
        self.update_scene()

    def update_window(self, p_idx, i, val):
        self.windows[p_idx][i] = val
        self.update_scene()

    def autonomous_tick(self):
        if self.autonomous.get():
            metrics = self.collector.get_metrics()
            vpos, opos, windows = map_metrics_to_scene_multi(metrics, self.size, self.planes)
            self.viewer_pos = list(vpos)
            self.object_pos = list(opos)
            self.windows = [list(w[0]) + list(w[1]) for w in windows]  # flatten for sliders
            for i,s in enumerate(self.viewer_sliders): s.set(self.viewer_pos[i])
            for i,s in enumerate(self.object_sliders): s.set(self.object_pos[i])
            for p_idx in range(len(self.planes)):
                for i,s in enumerate(self.window_sliders[p_idx]): s.set(self.windows[p_idx][i])
            self.update_scene(metrics=metrics)
        self.root.after(int(self.metrics_interval*1000), self.autonomous_tick)

    def update_scene(self, metrics=None):
        # Build exposed windows in proper structure
        windows_struct = []
        for w in self.windows:
            xmin, xmax, ymin, ymax = w
            windows_struct.append(((max(0,xmin), min(self.size[0]-1,xmax)),
                                   (max(0,ymin), min(self.size[1]-1,ymax))))
        sim = MultiPlaneMirrorSimulation(size=self.size, planes=self.planes, exposed_windows=windows_struct)
        sim.place_planes()
        sim.place_viewer(tuple(self.viewer_pos))
        sim.place_object(tuple(self.object_pos))
        sim.visualize(self.ax)
        self.canvas.draw()
        visible = sim.can_see_reflection()
        status = "Visible (all layers)" if visible else "Not visible (blocked at one or more layers)"
        if metrics is None:
            self.status_label.config(text=status)
            self.events_label.config(text="")
        else:
            cpu = metrics["cpu_percent"]
            mem = metrics["mem_percent"]
            net = (metrics["net_recv_bps"] + metrics["net_sent_bps"]) / 1_000_000
            disk = (metrics["disk_read_bps"] + metrics["disk_write_bps"]) / 1_000_000
            gpu = metrics.get("gpu_load", 0.0)
            lat = metrics.get("latency_ms", None)
            lat_txt = f"{lat:.0f} ms" if lat is not None else "n/a"
            self.status_label.config(
                text=f"{status} | CPU {cpu:.0f}% | MEM {mem:.0f}% | NET {net:.2f} MB/s | DISK {disk:.2f} MB/s | GPU {gpu:.0f}% | LAT {lat_txt}"
            )
            events = metrics.get("events", [])
            self.events_label.config(text=("Events: " + ", ".join(events)) if events else "")

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Autonomous Multi-Layer System Telemetry Mirror")
    app = MirrorGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

