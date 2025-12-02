"""
Adaptive Autonomous Engine + Trends + Anomalies + Dynamic Scaling + GUI Charts + API + Autoloader
-------------------------------------------------------------------------------------------------
- Autoloader with safe checks (runs even if some libs missing)
- System monitoring with rolling history, trend detection, and anomaly flags
- Adaptive Codex influences solver recommendations and feed/engine behavior
- Operator Registry with audit trail
- Dynamic worker scaling based on system load
- Adaptive feed interval under stress
- GUI: text log + live charts (CPU/MEM/DISK), operator audit
- FastAPI: optional endpoints for results and system status
- Self-healing guards for GUI/API threads
"""

# ============================================================
# Autoloader
# ============================================================

import importlib

def autoload(libs):
    loaded = {}
    for lib in libs:
        try:
            loaded[lib] = importlib.import_module(lib)
            print(f"[AUTOLOADER] Loaded {lib}")
        except ImportError:
            print(f"[AUTOLOADER] Missing {lib}, continuing.")
    return loaded

LIBS = autoload([
    "queue","random","threading","time","uuid","socket",
    "tkinter","psutil","fastapi","uvicorn","collections","statistics",
    "matplotlib","matplotlib.backends.backend_tkagg"
])

# Safe aliases
queue     = LIBS.get("queue")
random    = LIBS.get("random")
threading = LIBS.get("threading")
time      = LIBS.get("time")
uuid      = LIBS.get("uuid")
socket    = LIBS.get("socket")
tk        = LIBS.get("tkinter")
psutil    = LIBS.get("psutil")
fastapi   = LIBS.get("fastapi")
uvicorn   = LIBS.get("uvicorn")
collections = LIBS.get("collections")
statistics  = LIBS.get("statistics")
matplotlib   = LIBS.get("matplotlib")
mtkagg       = LIBS.get("matplotlib.backends.backend_tkagg")

# GUI widgets (safe import)
try:
    from tkinter import scrolledtext, ttk
except Exception:
    scrolledtext, ttk = None, None

# FastAPI class (optional)
try:
    from fastapi import FastAPI
except Exception:
    FastAPI = None

# Matplotlib setup (optional)
if matplotlib:
    try:
        matplotlib.use("Agg" if not tk else "TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
else:
    plt = None

FigureCanvasTkAgg = None
if mtkagg:
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except Exception:
        FigureCanvasTkAgg = None

# ============================================================
# Helpers
# ============================================================

def new_id(prefix): return f"{prefix}_{uuid.uuid4().hex[:8]}"
def now_ts(): return time.time()

# ============================================================
# Metric history (trends + anomalies)
# ============================================================

class MetricHistory:
    def __init__(self, window=60):
        self.window = window
        self.values = collections.deque(maxlen=window) if collections else []
    def add(self, val):
        if isinstance(self.values, list):
            self.values.append(val)
            if len(self.values) > self.window:
                self.values.pop(0)
        else:
            self.values.append(val)
    def avg(self):
        vals = list(self.values)
        return sum(vals)/len(vals) if vals else 0
    def std(self):
        vals = list(self.values)
        try:
            return statistics.pstdev(vals) if statistics and len(vals) > 1 else 0
        except Exception:
            return 0
    def trend(self):
        vals = list(self.values)
        if len(vals) < 5: return "flat"
        # simple trend: compare avg of early vs late segments
        k = len(vals)//2
        early = sum(vals[:k])/max(1,k)
        late  = sum(vals[k:])/max(1,len(vals)-k)
        if late > early * 1.05: return "rising"
        if late < early * 0.95: return "falling"
        return "flat"
    def is_anomaly(self, val, factor=2.0):
        mu = self.avg(); sd = self.std()
        if sd == 0: return False
        return abs(val - mu) > factor * sd

cpu_hist = MetricHistory()
mem_hist = MetricHistory()
disk_hist = MetricHistory()

# ============================================================
# System Monitor + Solver (codex-aware)
# ============================================================

def system_snapshot():
    return {
        "cpu": psutil.cpu_percent(0.2) if psutil else 0,
        "mem": psutil.virtual_memory().percent if psutil else 0,
        "disk": psutil.disk_usage('/').percent if psutil else 0,
        "net": (psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv) if psutil else 0,
        "procs": len(psutil.pids()) if psutil else 0,
        "ts": now_ts()
    }

def system_solver(snap, codex_state=None):
    # Update histories
    cpu_hist.add(snap["cpu"])
    mem_hist.add(snap["mem"])
    disk_hist.add(snap["disk"])

    issues, actions = [], []

    # Static thresholds
    if snap["cpu"] >= 85: issues.append("High CPU")
    if snap["mem"] >= 85: issues.append("High memory")
    if snap["disk"] >= 90: issues.append("Disk nearly full")
    if snap["procs"] > 300: issues.append("Too many processes")

    # Trend analysis
    cpu_trend = cpu_hist.trend()
    mem_trend = mem_hist.trend()
    if cpu_trend == "rising": issues.append("CPU rising trend")
    if mem_trend == "rising": issues.append("Memory rising trend")

    # Anomaly detection (Z-score proxy via std)
    if cpu_hist.is_anomaly(snap["cpu"], factor=2.0): issues.append("CPU anomaly")
    if mem_hist.is_anomaly(snap["mem"], factor=2.0): issues.append("Memory anomaly")

    # Actions
    if "Disk nearly full" in issues: actions.append("Free disk space")
    if "High memory" in issues or "Memory rising trend" in issues or "Memory anomaly" in issues:
        actions.append("Close memory-heavy apps")
    if "High CPU" in issues or "CPU rising trend" in issues or "CPU anomaly" in issues:
        actions.append("Reduce workers or close heavy apps")
    if "Too many processes" in issues:
        actions.append("Review background services")

    # Codex-aware recommendation
    if codex_state:
        logic = codex_state["rules"]["purge_logic"]
        ghost = codex_state["ghost_sync"]
        if "chaotic" in logic:
            actions.append("Apply aggressive remediation due to codex chaotic pattern")
        elif "fractal" in logic:
            actions.append("Prefer structured remediation (stepwise) due to codex fractal pattern")
        if ghost:
            actions.append("Ghost sync detected: shorten telemetry retention and verify phantom nodes")

    if not actions:
        actions.append("System healthy")

    return {
        "issues": issues,
        "actions": actions,
        "cpu_avg": cpu_hist.avg(),
        "mem_avg": mem_hist.avg(),
        "disk_avg": disk_hist.avg(),
        "cpu_trend": cpu_trend,
        "mem_trend": mem_trend
    }

# ============================================================
# Adaptive Codex (influences engine & feed adaptivity)
# ============================================================

class AdaptiveCodex:
    def __init__(self):
        self.weights = [0.6, -0.8, -0.3]
        self.rules = {"purge_logic": "default"}
        self.telemetry = 60
        self.patterns, self.phantoms = [], []

    def mutate(self):
        delta = [random.uniform(-0.1, 0.1) for _ in self.weights]
        self.weights = [w + d for w, d in zip(self.weights, delta)]
        pat = random.choice(["linear","chaotic","fractal","harmonic","stochastic"])
        self.patterns.append(pat)
        self.rules["purge_logic"] = f"adaptive_{pat}"
        ghost = random.random() < 0.2
        if ghost:
            self.telemetry = max(10, self.telemetry - 10)
            self.phantoms.append(f"phantom_{len(self.phantoms)}")
        return {
            "rules": self.rules.copy(),
            "telemetry": self.telemetry,
            "patterns": self.patterns.copy(),
            "phantoms": self.phantoms.copy(),
            "weights": self.weights,
            "ghost_sync": ghost
        }

CODEX = AdaptiveCodex()

# ============================================================
# Operator Registry
# ============================================================

class OperatorRegistry:
    def __init__(self): self.ops = {}
    def register(self, tier, func): self.ops.setdefault(tier, []).append(func)
    def get_all(self): return self.ops

REGISTRY = OperatorRegistry()

def op_drive(a): return {"tier": "feasible", "steps": ["Drive", "Arrive at " + a], "operator": "op_drive"}
def op_fly(a):   return {"tier": "feasible", "steps": ["Fly", "Land at " + a], "operator": "op_fly"}
REGISTRY.register("feasible", op_drive)
REGISTRY.register("feasible", op_fly)

# ============================================================
# Engine (dynamic worker scaling)
# ============================================================

class Engine:
    def __init__(self, registry, codex):
        self.registry, self.codex = registry, codex
        self.inbox, self.outbox = queue.Queue(), queue.Queue()
        self.running, self.lock, self.workers = False, threading.Lock(), []
        self.target_workers = 3

    def submit(self, text, meta=None):
        self.inbox.put({"text": text, "meta": meta or {}})

    def _apply_ops(self, payload_text):
        applied = []
        for tier, funcs in self.registry.get_all().items():
            for f in funcs:
                try:
                    applied.append(f(payload_text))
                except Exception as e:
                    applied.append({"tier":"notes","steps":[str(e)],"operator":"error"})
        return applied

    def _process(self, payload):
        snap = system_snapshot()
        codex_state = self.codex.mutate()
        solver = system_solver(snap, codex_state)
        bundle = {
            "id": new_id("bundle"),
            "answer": payload["text"],
            "system": snap,
            "solver": solver,
            "codex": codex_state,
            "operators": self._apply_ops(payload["text"]),
            "ts": now_ts()
        }
        # dynamic scaling hint (used by scaler)
        bundle["scale_hint"] = "down" if snap["cpu"] >= 85 or snap["mem"] >= 85 else "up" if snap["cpu"] < 40 and snap["mem"] < 60 else "steady"
        return bundle

    def _worker(self, idx):
        while self.running:
            try:
                p = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                b = self._process(p)
                self.outbox.put(b)
            except Exception as e:
                self.outbox.put({"id": new_id("bundle"), "error": str(e), "ts": now_ts()})
            finally:
                self.inbox.task_done()

    def _adjust_workers(self):
        # adjust thread count toward target_workers
        current = len(self.workers)
        if self.target_workers > current:
            for _ in range(self.target_workers - current):
                t = threading.Thread(target=self._worker, args=(len(self.workers),), daemon=True)
                t.start()
                self.workers.append(t)
        elif self.target_workers < current:
            # can't kill threads cleanly; reduce target and let them idle
            pass

    def start(self, n=3):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.target_workers = n
        self._adjust_workers()
        # scaler thread
        threading.Thread(target=self._scaler_loop, daemon=True).start()
        print(f"[ENGINE] Started with target {self.target_workers} workers.")

    def _scaler_loop(self):
        while self.running:
            # inspect recent CPU/MEM averages to adjust target workers
            cpu_avg = cpu_hist.avg()
            mem_avg = mem_hist.avg()
            if cpu_avg >= 85 or mem_avg >= 85:
                self.target_workers = max(1, self.target_workers - 1)
            elif cpu_avg < 40 and mem_avg < 60:
                self.target_workers = min(6, self.target_workers + 1)
            self._adjust_workers()
            time.sleep(5)

    def fetch(self):
        res = []
        while True:
            try:
                b = self.outbox.get_nowait()
                res.append(b)
                self.outbox.task_done()
            except queue.Empty:
                break
        return res

# ============================================================
# SystemFeed (adaptive interval)
# ============================================================

class SystemFeed:
    def __init__(self, engine):
        self.engine, self.running = engine, False
        self.interval = 5.0

    def _adapt_interval(self):
        cpu = cpu_hist.avg()
        mem = mem_hist.avg()
        if cpu >= 85 or mem >= 85:
            self.interval = 8.0
        elif cpu < 40 and mem < 60:
            self.interval = 4.0
        else:
            self.interval = 5.0

    def _loop(self):
        while self.running:
            snap = system_snapshot()
            self.engine.submit(f"System Snapshot {time.strftime('%H:%M:%S')}", {"snapshot": snap})
            self._adapt_interval()
            time.sleep(self.interval)

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[FEED] Started with interval {self.interval}s.")

# ============================================================
# GUI (charts + audit)
# ============================================================

class EngineGUI:
    def __init__(self, engine):
        if not tk or not scrolledtext or not ttk:
            raise RuntimeError("Tkinter not available. Install tkinter to use the GUI.")
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Adaptive Engine Monitor")
        self.root.geometry("950x650")

        # Header
        header = tk.Frame(self.root)
        header.pack(fill=tk.X, pady=5)
        tk.Label(header, text="Running", fg="green", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=8)
        self.progress = ttk.Progressbar(header, mode="indeterminate"); self.progress.pack(side=tk.LEFT, padx=8); self.progress.start(100)
        tk.Button(header, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT, padx=8)

        # Charts frame
        charts = tk.Frame(self.root)
        charts.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig = plt.figure(figsize=(6, 3)) if plt else None
        if self.fig and FigureCanvasTkAgg:
            self.ax_cpu = self.fig.add_subplot(131)
            self.ax_mem = self.fig.add_subplot(132)
            self.ax_disk = self.fig.add_subplot(133)
            self.canvas = FigureCanvasTkAgg(self.fig, master=charts)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            self.ax_cpu = self.ax_mem = self.ax_disk = None
            self.canvas = None
            tk.Label(charts, text="Charts disabled (matplotlib/tk backend missing)").pack()

        # Log and audit
        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.log = scrolledtext.ScrolledText(bottom, width=100, height=14, font=("Consolas", 10))
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.audit = scrolledtext.ScrolledText(bottom, width=40, height=14, font=("Consolas", 10))
        self.audit.pack(side=tk.RIGHT, fill=tk.BOTH)

        self._update_loop()

    def _clear_log(self):
        self.log.delete("1.0", tk.END)
        self.audit.delete("1.0", tk.END)

    def _update_charts(self):
        if not self.canvas or not self.ax_cpu: return
        cpu_vals = list(cpu_hist.values) if hasattr(cpu_hist, "values") else []
        mem_vals = list(mem_hist.values) if hasattr(mem_hist, "values") else []
        disk_vals = list(disk_hist.values) if hasattr(disk_hist, "values") else []
        for ax in (self.ax_cpu, self.ax_mem, self.ax_disk):
            ax.clear()
        self.ax_cpu.plot(cpu_vals[-60:], color="red");  self.ax_cpu.set_title("CPU %")
        self.ax_mem.plot(mem_vals[-60:], color="blue"); self.ax_mem.set_title("MEM %")
        self.ax_disk.plot(disk_vals[-60:], color="green"); self.ax_disk.set_title("DISK %")
        self.ax_cpu.set_ylim(0, 100); self.ax_mem.set_ylim(0, 100); self.ax_disk.set_ylim(0, 100)
        self.canvas.draw_idle()

    def _render_bundle(self, b):
        s = b.get("system", {})
        sol = b.get("solver", {})
        codex = b.get("codex", {})
        self.log.insert(tk.END, f"{b.get('id')} :: {b.get('answer')}\n"
                                f" CPU={s.get('cpu',0)}% MEM={s.get('mem',0)}% DISK={s.get('disk',0)}% PROCS={s.get('procs',0)}\n")
        if sol.get("issues"):
            self.log.insert(tk.END, f" Issues: {', '.join(sol['issues'])}\n")
        actions = sol.get("actions", ["System healthy"])
        self.log.insert(tk.END, f" Actions: {actions[0]}\n"
                                f" Codex purge_logic={codex.get('rules',{}).get('purge_logic')} ghost={codex.get('ghost_sync')}\n"
                                f" CPU trend={sol.get('cpu_trend')} MEM trend={sol.get('mem_trend')}\n"
                                f"{'-'*80}\n")
        self.log.see(tk.END)

        # Operator audit trail
        ops = b.get("operators", [])
        self.audit.insert(tk.END, f"{b.get('id')} operators:\n")
        for op in ops[:10]:
            self.audit.insert(tk.END, f" - {op.get('operator')} [{op.get('tier')}]: {', '.join(op.get('steps', []))}\n")
        self.audit.insert(tk.END, "-"*40 + "\n")
        self.audit.see(tk.END)

    def _update_loop(self):
        try:
            bundles = self.engine.fetch()
            for b in bundles:
                self._render_bundle(b)
            self._update_charts()
        except Exception as e:
            self.log.insert(tk.END, f"[GUI ERROR] {e}\n")
        self.root.after(1000, self._update_loop)

    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"[GUI] crashed: {e}")

# ============================================================
# Instances
# ============================================================

engine = Engine(REGISTRY, CODEX)
feed = SystemFeed(engine)

# ============================================================
# API (optional)
# ============================================================

app = None

def free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def run_api():
    if not app or not uvicorn:
        return
    port = free_port()
    print(f"[API] FastAPI on port {port}")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"[API] crashed: {e}")

if fastapi and uvicorn and FastAPI:
    app = FastAPI()

    @app.get("/results")
    def results():
        return engine.fetch()

    @app.get("/system")
    def system():
        snap = system_snapshot()
        codex_state = {"rules": CODEX.rules, "telemetry": CODEX.telemetry,
                       "patterns": CODEX.patterns, "phantoms": CODEX.phantoms,
                       "weights": CODEX.weights, "ghost_sync": False}
        solver = system_solver(snap, codex_state)
        return {"snapshot": snap, "solver": solver}
else:
    print("[WARNING] FastAPI/Uvicorn not available, API layer disabled.")

# ============================================================
# Main
# ============================================================

def main():
    engine.start(3)
    feed.start()

    # Launch API in background if available
    if app and uvicorn:
        threading.Thread(target=run_api, daemon=True).start()

    # Launch GUI
    if tk and scrolledtext and ttk and plt and FigureCanvasTkAgg:
        EngineGUI(engine).run()
    elif tk and scrolledtext and ttk:
        # Run text-only GUI without charts
        print("[INFO] Charts disabled; starting text-only GUI.")
        EngineGUI(engine).run()
    else:
        print("[WARNING] Tkinter/matplotlib not available, GUI disabled. Running headless.")
        # Headless loop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()

