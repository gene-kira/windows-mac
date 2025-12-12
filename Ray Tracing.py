import importlib
import subprocess
import sys

# --- Auto-loader for required libraries ---
def auto_install(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["numpy", "matplotlib"]:
    auto_install(pkg)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# --- Simulation class ---
EMPTY, MIRROR, TOWEL, VIEWER, OBJECT = 0, 1, 2, 3, 4

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
        ax.set_title("3D Mirror/Data Visibility Simulation")

# --- GUI wrapper with sliders ---
class MirrorGUI:
    def __init__(self,root):
        self.root=root
        self.sim=Mirror3DSimulation()
        self.sim.place_mirror()
        self.viewer_pos=[10,10,2]
        self.object_pos=[10,10,18]

        self.fig=plt.Figure(figsize=(6,5))
        self.ax=self.fig.add_subplot(111,projection="3d")
        self.canvas=FigureCanvasTkAgg(self.fig,master=root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH,expand=True)

        control_frame=tk.Frame(root)
        control_frame.pack(side=tk.RIGHT,fill=tk.Y)

        # Sliders for viewer
        tk.Label(control_frame,text="Viewer Position").pack()
        self.viewer_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            tk.Label(control_frame,text=label).pack()
            slider=tk.Scale(control_frame,from_=0,to=self.sim.size[i]-1,orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_viewer(i,int(val)))
            slider.set(self.viewer_pos[i])
            slider.pack()
            self.viewer_sliders.append(slider)

        # Sliders for object
        tk.Label(control_frame,text="Object Position").pack()
        self.object_sliders=[]
        for i,label in enumerate(["X","Y","Z"]):
            tk.Label(control_frame,text=label).pack()
            slider=tk.Scale(control_frame,from_=0,to=self.sim.size[i]-1,orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_object(i,int(val)))
            slider.set(self.object_pos[i])
            slider.pack()
            self.object_sliders.append(slider)

        # Sliders for exposed region bounds
        tk.Label(control_frame,text="Exposed Region Bounds").pack()
        self.exposed_bounds=[6,13,6,13]  # xmin, xmax, ymin, ymax
        labels=["Xmin","Xmax","Ymin","Ymax"]
        self.exposed_sliders=[]
        for i,label in enumerate(labels):
            tk.Label(control_frame,text=label).pack()
            slider=tk.Scale(control_frame,from_=0,to=self.sim.size[0]-1,orient=tk.HORIZONTAL,
                            command=lambda val,i=i:self.update_exposed(i,int(val)))
            slider.set(self.exposed_bounds[i])
            slider.pack()
            self.exposed_sliders.append(slider)

        self.status_label=tk.Label(control_frame,text="")
        self.status_label.pack(pady=10)

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

    def update_scene(self):
        bounds=((self.exposed_bounds[0],self.exposed_bounds[1]),
                (self.exposed_bounds[2],self.exposed_bounds[3]))
        self.sim=Mirror3DSimulation(exposed_bounds=bounds)
        self.sim.place_mirror()
        self.sim.place_viewer(tuple(self.viewer_pos))
        self.sim.place_object(tuple(self.object_pos))
        self.sim.visualize(self.ax)
        self.canvas.draw()
        visible=self.sim.can_see_reflection()
        self.status_label.config(text="Visible" if visible else "Not visible")

if __name__=="__main__":
    root=tk.Tk()
    root.title("Mirror/Data Visibility GUI")
    app=MirrorGUI(root)
    root.mainloop()

