import importlib
import subprocess
import sys
import threading
import time

# --- Auto-loader for required libraries ---
def auto_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["numpy", "matplotlib", "psutil"]:
    auto_install(pkg)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import psutil

# --- Simulation constants ---
EMPTY, MIRROR, TOWEL, VIEWER, OBJECT = 0, 1, 2, 3, 4

# --- System telemetry collector ---
class MetricsCollector(threading.Thread):
    def __init__(self, interval=1.0):
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
        }
        # Baseline for rate calculations
        self._prev_disk = psutil.disk_io_counters() if psutil.disk_io_counters() else None
        self._prev_net = psutil.net_io_counters() if psutil.net_io_counters() else None
        self._prev_time = time.time()

    def run(self):
        # Warm-up to stabilize CPU percent
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            start = time.time()
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent if psutil.virtual_memory() else 0.0
            now = time.time()

            # Disk throughput
            disk = psutil.disk_io_counters() if psutil.disk_io_counters() else None
            disk_read_bps = disk_write_bps = 0.0
            if self._prev_disk and disk:
                dt = max(now - self._prev_time, 1e-6)
                disk_read_bps = (disk.read_bytes - self._prev_disk.read_bytes) / dt
                disk_write_bps = (disk.write_bytes - self._prev_disk.write_bytes) / dt
                self._prev_disk = disk

            # Net throughput
            net = psutil.net_io_counters() if psutil.net_io_counters() else None
            net_recv_bps = net_sent_bps = 0.0
            if self._prev_net and net:
                dt = max(now - self._prev_time, 1e-6)
                net_recv_bps = (net.bytes_recv - self._prev_net.bytes_recv) / dt
                net_sent_bps = (net.bytes_sent - self._prev_net.bytes_sent) / dt
                self._prev_net = net

            self._prev_time = now

            # Process and connection counts (aggregated)
            try:
                proc_count = len(psutil.pids())
            except Exception:
                proc_count = 0
            try:
                conn_count = len(psutil.net_connections(kind='inet'))
            except Exception:
                conn_count = 0

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
                })

            # Pace loop
            elapsed = time.time() - start
            time.sleep(max(0.0, self.interval - elapsed))

    def get_metrics(self):
        with self.lock:
            return dict(self.metrics)

    def stop(self):
        self._stop.set()

# --- 3D mirror simulation ---
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

    def reflect_point_across_plane(self,point):
        x,y,z=point
        z_m=self.mirror_plane_z
        return (x,y,2*z_m-z)

    def can_see_reflection(self,samples=200):
        if self.viewer_pos is None or self.object_pos is None: return False
        V=np.array(self.viewer_pos,dtype=float)
        R=np.array(self.reflect_point_across_plane(self.object_pos),dtype=float)
        for t in np.linspace(0,1,samples):
            P=V+t*(R-V)
            x,y,z=np.round(P).astype(int)
            if not self.in_bounds(x,y,z): continue
            if z==self.mirror_plane_z:
                return self.is_exposed(x,y)
        return False

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
        ax.scatter(xs_m,ys_m,zs_m,c='cyan',marker='s',s=12,label='Exposed')
        ax.scatter(xs_t,ys_t,zs_t,c='gray',marker='s',s=12,label='Towel')
        if self.viewer_pos: ax.scatter(*self.viewer_pos,c='green',s=80,label='Viewer')
        if self.object_pos: ax.scatter(*self.object_pos,c='red',s=80,label='Object')
        if self.viewer_pos and self.object_pos:
            R=self.reflect_point_across_plane(self.object_pos)
            ray=np.array([self.viewer_pos,R],dtype=float)
            ax.plot(ray[:,0],ray[:,1],ray[:,2],c='orange',label='Ray')
        ax.set_xlim(0,self.size[0]); ax.set_ylim(0,self.size[1]); ax.set_zlim(0,self.size[2])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(loc='upper left')
        ax.set_title("Autonomous System Telemetry Mirror")

# --- Mapping: metrics -> scene parameters ---
def normalize(val, min_v, max_v, size):
    v = 0.0 if max_v <= min_v else (val - min_v) / (max_v - min_v)
    v = max(0.0, min(1.0, v))
    return int(v * (size - 1))

def map_metrics_to_scene(metrics, grid_size):
    X, Y, Z = grid_size
    cpu = metrics["cpu_percent"]               # 0..100
    mem = metrics["mem_percent"]               # 0..100
    net = metrics["net_recv_bps"] + metrics["net_sent_bps"]   # bytes/sec
    disk = metrics["disk_read_bps"] + metrics["disk_write_bps"] # bytes/sec
    procs = metrics["proc_count"]
    conns = metrics["conn_count"]

    # Viewer position: CPU -> X, MEM -> Y, "surface layer" Z low
    vx = normalize(cpu, 0, 100, X)
    vy = normalize(mem, 0, 100, Y)
    vz = 2

    # Object position: processes & connections intensity
    # Combine and normalize to drive X/Y; Z high (deep internal state)
    activity = procs + conns
    ox = normalize(activity, 0, 1000, X)   # heuristic cap at 1000
    oy = normalize(disk, 0, 50_000_000, Y) # 50 MB/s cap for typical desktops
    oz = Z - 2

    # Exposed bounds (telemetry window): widen with network + disk throughput
    # Center window around mid grid; width scales with throughput
    cx, cy = X // 2, Y // 2
    # Scale width from small to large with log to tame spikes
    def scaled_width(v, base=2, max_w=min(X, Y)//2):
        w = int(min(max_w, max(base, np.log10(v + 1) * (max_w // 2))))
        return max(2, w)

    wx = scaled_width(net)
    wy = scaled_width(disk)

    xmin, xmax = max(0, cx - wx), min(X - 1, cx + wx)
    ymin, ymax = max(0, cy - wy), min(Y - 1, cy + wy)

    viewer_pos = (vx, vy, vz)
    object_pos = (ox, oy, oz)
    exposed_bounds = ((xmin, xmax), (ymin, ymax))
    return viewer_pos, object_pos, exposed_bounds

# --- GUI wrapper with manual + autonomous control ---
class MirrorGUI:
    def __init__(self,root):
        self.root = root
        self.size = (20,20,20)
        self.metrics_interval = 1.0
        self.collector = MetricsCollector(interval=self.metrics_interval)
        self.collector.start()
        self.autonomous = tk.BooleanVar(value=True)

        # Default scene parameters
        self.viewer_pos=[10,10,2]
        self.object_pos=[10,10,18]
        self.exposed_bounds=[6,13,6,13]  # xmin, xmax, ymin, ymax

        # Figure
        self.fig=plt.Figure(figsize=(7,5))
        self.ax=self.fig.add_subplot(111,projection="3d")
        self.canvas=FigureCanvasTkAgg(self.fig,master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

        # Controls
        control_frame=tk.Frame(root)
        control_frame.pack(side=tk.RIGHT,fill=tk.Y)

        tk.Label(control_frame,text="Autonomous mode").pack()
        tk.Checkbutton(control_frame,variable=self.autonomous,command=self.update_scene)\
            .pack(pady=4)

        # Viewer sliders
        tk.Label(control_frame,text="Viewer position").pack()
        self.viewer_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            tk.Label(control_frame,text=label).pack()
            slider=tk.Scale(control_frame,from_=0,to=self.size[i]-1,orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_viewer(i,int(val)))
            slider.set(self.viewer_pos[i])
            slider.pack()
            self.viewer_sliders.append(slider)

        # Object sliders
        tk.Label(control_frame,text="Object position").pack(pady=(10,0))
        self.object_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            tk.Label(control_frame,text=label).pack()
            slider=tk.Scale(control_frame,from_=0,to=self.size[i]-1,orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_object(i,int(val)))
            slider.set(self.object_pos[i])
            slider.pack()
            self.object_sliders.append(slider)

        # Exposed region sliders
        tk.Label(control_frame,text="Exposed region bounds").pack(pady=(10,0))
        labels=["Xmin","Xmax","Ymin","Ymax"]
        self.exposed_sliders=[]
        for i,label in enumerate(labels):
            tk.Label(control_frame,text=label).pack()
            slider=tk.Scale(control_frame,from_=0,to=self.size[0]-1,orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_exposed(i,int(val)))
            slider.set(self.exposed_bounds[i])
            slider.pack()
            self.exposed_sliders.append(slider)

        # Status
        self.status_label=tk.Label(control_frame,text="Starting...")
        self.status_label.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(pady=8, fill=tk.X)
        tk.Button(btn_frame,text="Reset",command=self.reset_defaults).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame,text="Pause telemetry",command=self.pause_telemetry)\
            .pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame,text="Resume telemetry",command=self.resume_telemetry)\
            .pack(side=tk.LEFT, padx=4)

        # Initial render and schedule updates
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
        self.viewer_pos=[10,10,2]
        self.object_pos=[10,10,18]
        self.exposed_bounds=[6,13,6,13]
        for i,s in enumerate(self.viewer_sliders): s.set(self.viewer_pos[i])
        for i,s in enumerate(self.object_sliders): s.set(self.object_pos[i])
        for i,s in enumerate(self.exposed_sliders): s.set(self.exposed_bounds[i])
        self.update_scene()

    def update_viewer(self,i,val):
        self.viewer_pos[i]=val
        self.update_scene()

    def update_object(self,i,val):
        self.object_pos[i]=val
        self.update_scene()

    def update_exposed(self,i,val):
        self.exposed_bounds[i]=val
        self.update_scene()

    def autonomous_tick(self):
        if self.autonomous.get():
            metrics = self.collector.get_metrics()
            vpos, opos, bounds = map_metrics_to_scene(metrics, self.size)
            self.viewer_pos = list(vpos)
            self.object_pos = list(opos)
            self.exposed_bounds = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
            # Reflect in sliders (non-disruptive)
            for i,s in enumerate(self.viewer_sliders): s.set(self.viewer_pos[i])
            for i,s in enumerate(self.object_sliders): s.set(self.object_pos[i])
            for i,s in enumerate(self.exposed_sliders): s.set(self.exposed_bounds[i])
            self.update_scene(metrics=metrics)
        # Schedule next tick
        self.root.after(int(self.metrics_interval*1000), self.autonomous_tick)

    def update_scene(self, metrics=None):
        bounds=((self.exposed_bounds[0],self.exposed_bounds[1]),
                (self.exposed_bounds[2],self.exposed_bounds[3]))
        sim=Mirror3DSimulation(size=self.size, exposed_bounds=bounds)
        sim.place_mirror()
        sim.place_viewer(tuple(self.viewer_pos))
        sim.place_object(tuple(self.object_pos))
        sim.visualize(self.ax)
        self.canvas.draw()
        visible=sim.can_see_reflection()
        status = "Visible" if visible else "Not visible"
        if metrics is None:
            self.status_label.config(text=status)
        else:
            # Concise metric overlay
            cpu = metrics["cpu_percent"]
            mem = metrics["mem_percent"]
            net = (metrics["net_recv_bps"] + metrics["net_sent_bps"]) / 1_000_000
            disk = (metrics["disk_read_bps"] + metrics["disk_write_bps"]) / 1_000_000
            self.status_label.config(
                text=f"{status} | CPU {cpu:.0f}% | MEM {mem:.0f}% | NET {net:.2f} MB/s | DISK {disk:.2f} MB/s"
            )

# --- Main ---
if __name__=="__main__":
    root=tk.Tk()
    root.title("Autonomous System Telemetry Mirror")
    app=MirrorGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

