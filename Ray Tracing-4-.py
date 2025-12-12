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
        ax.legend(loc='upper left')
        ax.set_title("Compact 3D Mirror Simulation")

# --- GUI wrapper ---
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

        # Smaller figure size
        self.fig=plt.Figure(figsize=(3.5,2.5))  # 50% smaller
        self.ax=self.fig.add_subplot(111,projection="3d")
        self.canvas=FigureCanvasTkAgg(self.fig,master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

        control_frame=tk.Frame(root,width=150)  # narrower control panel
        control_frame.pack(side=tk.RIGHT,fill=tk.Y)

        self.status_label=tk.Label(control_frame,text="Starting...",font=("Arial",9))
        self.status_label.pack(pady=5)

        self.update_scene()
        self.root.after(1000,self.tick)

    def tick(self):
        metrics=self.collector.get_metrics()
        self.update_scene(metrics)
        self.root.after(1000,self.tick)

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
    root.geometry("800x500")  # scaled down window size
    app = MirrorGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()



