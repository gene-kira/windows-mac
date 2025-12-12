import importlib
import subprocess
import sys
import threading
import time
import socket
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import psutil

# --- Auto-loader for required libraries ---
def auto_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["numpy", "matplotlib", "psutil"]:
    auto_install(pkg)

try:
    import GPUtil
except Exception:
    GPUtil = None

EMPTY, MIRROR, TOWEL, VIEWER, OBJECT = 0, 1, 2, 3, 4

# --- System telemetry collector ---
class MetricsCollector(threading.Thread):
    def __init__(self, interval=1.0, latency_host="8.8.8.8", latency_port=53, latency_timeout=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.metrics = {"cpu_percent":0.0,"mem_percent":0.0,
                        "disk_read_bps":0.0,"disk_write_bps":0.0,
                        "net_recv_bps":0.0,"net_sent_bps":0.0,
                        "proc_count":0,"conn_count":0,
                        "gpu_load":0.0,"gpu_mem_percent":0.0,
                        "latency_ms":None}
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
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
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            now = time.time()
            disk = psutil.disk_io_counters()
            disk_read_bps = disk_write_bps = 0.0
            if self._prev_disk and disk:
                dt = max(now - self._prev_time, 1e-6)
                disk_read_bps = (disk.read_bytes - self._prev_disk.read_bytes) / dt
                disk_write_bps = (disk.write_bytes - self._prev_disk.write_bytes) / dt
                self._prev_disk = disk
            net = psutil.net_io_counters()
            net_recv_bps = net_sent_bps = 0.0
            if self._prev_net and net:
                dt = max(now - self._prev_time, 1e-6)
                net_recv_bps = (net.bytes_recv - self._prev_net.bytes_recv) / dt
                net_sent_bps = (net.bytes_sent - self._prev_net.bytes_sent) / dt
                self._prev_net = net
            self._prev_time = now
            proc_count = len(psutil.pids())
            conn_count = len(psutil.net_connections(kind='inet'))
            gpu_load = gpu_mem_percent = 0.0
            if GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_load = float(np.mean([g.load for g in gpus])) * 100.0
                        gpu_mem_percent = float(np.mean([g.memoryUtil for g in gpus])) * 100.0
                except Exception:
                    pass
            latency_ms = self._probe_latency()
            with self.lock:
                self.metrics.update({"cpu_percent":cpu,"mem_percent":mem,
                                     "disk_read_bps":disk_read_bps,"disk_write_bps":disk_write_bps,
                                     "net_recv_bps":net_recv_bps,"net_sent_bps":net_sent_bps,
                                     "proc_count":proc_count,"conn_count":conn_count,
                                     "gpu_load":gpu_load,"gpu_mem_percent":gpu_mem_percent,
                                     "latency_ms":latency_ms})
            time.sleep(self.interval)

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)

    def stop(self):
        self._stop.set()

# --- 3D Mirror Simulation ---
class Mirror3DSimulation:
    def __init__(self, size=(20,20,20), mirror_plane_z=10, exposed_bounds=((6,13),(6,13))):
        self.size = size
        self.grid = np.zeros(size, dtype=int)
        self.viewer_pos = None
        self.object_pos = None
        self.mirror_plane_z = mirror_plane_z
        (self.xmin, self.xmax), (self.ymin, self.ymax) = exposed_bounds

    def in_bounds(self,x,y,z):
        return (0<=x<self.size[0]) and (0<=y<self.size[1]) and (0<=z<self.size[2])

    def is_exposed(self,x,y):
        return (self.xmin<=x<=self.xmax) and (self.ymin<=y<=self.ymax)

    def place_viewer(self,pos):
        self.viewer_pos=pos
        self.grid[pos]=VIEWER

    def place_object(self,pos):
        self.object_pos=pos
        self.grid[pos]=OBJECT

    def place_mirror(self):
        z=self.mirror_plane_z
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.grid[x,y,z]=MIRROR if self.is_exposed(x,y) else TOWEL

    def visualize(self,ax):
        ax.clear()
        z=self.mirror_plane_z
        xs_m,ys_m,zs_m=[],[],[]
        xs_t,ys_t,zs_t=[],[],[]
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.grid[x,y,z]==MIRROR:
                    xs_m.append(x); ys_m.append(y); zs_m.append(z)
                elif self.grid[x,y,z]==TOWEL:
                    xs_t.append(x); ys_t.append(y); zs_t.append(z)
        ax.scatter(xs_m,ys_m,zs_m,c='cyan',marker='s',s=6,label='Exposed')
        ax.scatter(xs_t,ys_t,zs_t,c='gray',marker='s',s=6,label='Towel')
        if self.viewer_pos: ax.scatter(*self.viewer_pos,c='green',s=50,label='Viewer')
        if self.object_pos: ax.scatter(*self.object_pos,c='red',s=50,label='Object')
        ax.set_xlim(0,self.size[0]); ax.set_ylim(0,self.size[1]); ax.set_zlim(0,self.size[2])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title("Compact 3D Mirror Simulation", fontsize=10)

# --- GUI wrapper with scrollable controls ---
class MirrorGUI:
    def __init__(self,root):
        self.root=root
        self.size=(20,20,20)
        self.mirror_plane_z=10
        self.collector=MetricsCollector(interval=1.0)
        self.collector.start()
        self.viewer_pos=[5,5,2]
        self.object_pos=[15,15,18]
        self.exposed_bounds=[6,13,6,13]

        # Paned window for resizable split view
        paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=6)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left: plot
        left_frame = tk.Frame(paned)
        paned.add(left_frame, minsize=300)  # ensure minimum plot area

        self.fig=plt.Figure(figsize=(3.6,2.6))  # 50% smaller but slightly padded
        self.ax=self.fig.add_subplot(111,projection="3d")
        self.canvas=FigureCanvasTkAgg(self.fig,master=left_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Right: scrollable controls
        right_frame = tk.Frame(paned, width=260)  # fixed min width for controls
        paned.add(right_frame, minsize=220)

        # Create a canvas + scrollbar for the controls
        ctrl_canvas = tk.Canvas(right_frame, borderwidth=0, highlightthickness=0)
        vscroll = tk.Scrollbar(right_frame, orient="vertical", command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame that holds actual controls
        self.ctrl_inner = tk.Frame(ctrl_canvas)
        self.ctrl_inner.bind(
            "<Configure>",
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
        )
        ctrl_canvas.create_window((0,0), window=self.ctrl_inner, anchor="nw")

        # Status
        self.status_label=tk.Label(self.ctrl_inner,text="Starting...",font=("Arial",9))
        self.status_label.pack(pady=(6,4))

        # Viewer controls
        self._section_label("Viewer position")
        self.viewer_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            self._slider_row(self.ctrl_inner, label, 0, self.size[i]-1,
                             self.viewer_pos[i], lambda v,i=i: self._set_viewer(i, int(v)),
                             resolution=1)

        # Object controls
        self._section_label("Object position")
        self.object_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            self._slider_row(self.ctrl_inner, label, 0, self.size[i]-1,
                             self.object_pos[i], lambda v,i=i: self._set_object(i, int(v)),
                             resolution=1)

        # Exposed region controls
        self._section_label("Exposed region bounds")
        self.exposed_sliders=[]
        labels=["Xmin","Xmax","Ymin","Ymax"]
        ranges=[(0,self.size[0]-1),(0,self.size[0]-1),(0,self.size[1]-1),(0,self.size[1]-1)]
        for i,label in enumerate(labels):
            self._slider_row(self.ctrl_inner, label, ranges[i][0], ranges[i][1],
                             self.exposed_bounds[i], lambda v,i=i: self._set_exposed(i, int(v)),
                             resolution=1)

        # Mirror plane Z
        self._section_label("Mirror plane Z")
        self._slider_row(self.ctrl_inner, "Z", 1, self.size[2]-2, self.mirror_plane_z,
                         lambda v: self._set_mirror_z(int(v)), resolution=1)

        # Buttons
        btn_frame = tk.Frame(self.ctrl_inner)
        btn_frame.pack(fill=tk.X, pady=(8,6))
        tk.Button(btn_frame, text="Reset", command=self._reset_defaults, font=("Arial",9)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Pause telemetry", command=self._pause, font=("Arial",9)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Resume telemetry", command=self._resume, font=("Arial",9)).pack(side=tk.LEFT, padx=4)

        # Initial render and schedule
        self.update_scene()
        self.root.after(1000, self.tick)

        # Make mouse wheel scroll the controls area
        ctrl_canvas.bind_all("<MouseWheel>", lambda e: ctrl_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    # UI helpers
    def _section_label(self, text):
        tk.Label(self.ctrl_inner, text=text, font=("Arial",9,"bold")).pack(anchor="w", pady=(8,2))

    def _slider_row(self, parent, label, frm, to, init, cmd, resolution=1):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, font=("Arial",9), width=10, anchor="w").pack(side=tk.LEFT)
        s = tk.Scale(row, from_=frm, to=to, orient=tk.HORIZONTAL, command=lambda v: cmd(v),
                     resolution=resolution, length=150, showvalue=True, font=("Arial",9))
        s.set(init)
        s.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return s

    # Telemetry control
    def _pause(self):
        self.collector.stop()
        self.status_label.config(text="Telemetry paused")

    def _resume(self):
        if not self.collector.is_alive():
            self.collector = MetricsCollector(interval=1.0)
            self.collector.start()
        self.status_label.config(text="Telemetry resumed")

    # Slider callbacks
    def _set_viewer(self, i, val):
        self.viewer_pos[i] = val
        self.update_scene()

    def _set_object(self, i, val):
        self.object_pos[i] = val
        self.update_scene()

    def _set_exposed(self, i, val):
        self.exposed_bounds[i] = val
        self.update_scene()

    def _set_mirror_z(self, val):
        self.mirror_plane_z = val
        self.update_scene()

    def _reset_defaults(self):
        self.viewer_pos=[5,5,2]
        self.object_pos=[15,15,18]
        self.exposed_bounds=[6,13,6,13]
        self.mirror_plane_z=10
        self.update_scene()

    # Render tick
    def tick(self):
        metrics=self.collector.get_metrics()
        self.update_scene(metrics)
        self.root.after(1000,self.tick)

    # Scene update
    def update_scene(self,metrics=None):
        bounds=((self.exposed_bounds[0],self.exposed_bounds[1]),
                (self.exposed_bounds[2],self.exposed_bounds[3]))
        sim=Mirror3DSimulation(size=self.size,mirror_plane_z=self.mirror_plane_z,exposed_bounds=bounds)
        sim.place_mirror()
        sim.place_viewer(tuple(self.viewer_pos))
        sim.place_object(tuple(self.object_pos))
        sim.visualize(self.ax)
        self.canvas.draw()

        if metrics:
            cpu = metrics["cpu_percent"]
            mem = metrics["mem_percent"]
            net = (metrics["net_recv_bps"] + metrics["net_sent_bps"]) / 1_000_000
            disk = (metrics["disk_read_bps"] + metrics["disk_write_bps"]) / 1_000_000
            gpu = metrics["gpu_load"]
            lat = metrics["latency_ms"]
            lat_txt = f"{lat:.0f} ms" if lat is not None else "n/a"
            self.status_label.config(
                text=f"CPU {cpu:.0f}% | MEM {mem:.0f}% | NET {net:.2f} MB/s | DISK {disk:.2f} MB/s | GPU {gpu:.0f}% | LAT {lat_txt}"
            )
        else:
            self.status_label.config(text="No metrics yet...")

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Compact 3D System Mirror")
    # Smaller default window, still roomy for scrollable controls
    root.geometry("900x540")
    # Allow high-DPI scaling on Windows (optional)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = MirrorGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

