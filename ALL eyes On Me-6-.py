"""
Predictive Adaptive Engine + Trends + Anomalies + Forecasts + Scenarios + GUI Charts + API + Autoloader
-------------------------------------------------------------------------------------------------------
- Autoloader with safe checks (runs even if some libs missing)
- System monitoring with rolling history, trend detection, anomaly flags, and short-horizon forecasts
- Adaptive Codex influences solver and predicts next pattern
- Operator Registry with audit trail and outcome learning hooks
- Dynamic worker scaling based on system load
- Adaptive feed interval under stress
- Scenario Engine: simulate near-future states and proactive actions
- GUI: text log + live charts + forecast panel + operator audit
- FastAPI: optional endpoints for results and system status
- Resilient/headless when GUI/API/matplotlib aren’t available
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
queue       = LIBS.get("queue")
random      = LIBS.get("random")
threading   = LIBS.get("threading")
time        = LIBS.get("time")
uuid        = LIBS.get("uuid")
socket      = LIBS.get("socket")
tk          = LIBS.get("tkinter")
psutil      = LIBS.get("psutil")
fastapi     = LIBS.get("fastapi")
uvicorn     = LIBS.get("uvicorn")
collections = LIBS.get("collections")
statistics  = LIBS.get("statistics")
matplotlib  = LIBS.get("matplotlib")
mtkagg      = LIBS.get("matplotlib.backends.backend_tkagg")

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

# Lightweight forecasting: linear slope extrapolation
def forecast_next(values, steps_ahead=1):
    vals = list(values)
    if not vals:
        return 0
    if len(vals) < 5:
        return vals[-1]
    # simple slope over window
    slope = (vals[-1] - vals[0]) / max(1, len(vals)-1)
    return max(0, min(100, vals[-1] + slope * steps_ahead))

def forecast_series(values, horizon=10):
    base = list(values)
    out = []
    last = base[-1] if base else 0
    for h in range(1, horizon+1):
        last = forecast_next(base + out, steps_ahead=1)
        out.append(last)
    return out

# ============================================================
# Metric history (trends + anomalies)
# ============================================================

class MetricHistory:
    def __init__(self, window=120):
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
        if len(vals) < 8: return "flat"
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
# System Monitor + Solver (codex-aware + predictive)
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

def predictive_flags():
    # short horizon forecasts (e.g., ~10 cycles)
    fc_cpu = forecast_series(cpu_hist.values, horizon=10)
    fc_mem = forecast_series(mem_hist.values, horizon=10)
    fc_disk = forecast_series(disk_hist.values, horizon=10)
    return {
        "cpu_forecast": fc_cpu,
        "mem_forecast": fc_mem,
        "disk_forecast": fc_disk,
        "cpu_future_stress": any(v >= 85 for v in fc_cpu),
        "mem_future_stress": any(v >= 85 for v in fc_mem),
        "disk_future_stress": any(v >= 90 for v in fc_disk)
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
    disk_trend = disk_hist.trend()
    if cpu_trend == "rising": issues.append("CPU rising trend")
    if mem_trend == "rising": issues.append("Memory rising trend")
    if disk_trend == "rising": issues.append("Disk rising trend")

    # Anomaly detection
    if cpu_hist.is_anomaly(snap["cpu"], factor=2.0): issues.append("CPU anomaly")
    if mem_hist.is_anomaly(snap["mem"], factor=2.0): issues.append("Memory anomaly")

    # Predictive flags
    preds = predictive_flags()
    if preds["cpu_future_stress"]: issues.append("Predicted CPU stress")
    if preds["mem_future_stress"]: issues.append("Predicted memory stress")
    if preds["disk_future_stress"]: issues.append("Predicted disk stress")

    # Actions (reactive + predictive)
    if "Disk nearly full" in issues or preds["disk_future_stress"]:
        actions.append("Free disk space / rotate logs")
    if any(k in issues for k in ["High memory","Memory rising trend","Memory anomaly"]) or preds["mem_future_stress"]:
        actions.append("Close memory-heavy apps / enable compression")
    if any(k in issues for k in ["High CPU","CPU rising trend","CPU anomaly"]) or preds["cpu_future_stress"]:
        actions.append("Reduce workers / throttle non-critical tasks")
    if "Too many processes" in issues:
        actions.append("Review background services")

    # Codex-aware recommendation
    if codex_state:
        logic = codex_state["rules"]["purge_logic"]
        ghost = codex_state["ghost_sync"]
        if "chaotic" in logic:
            actions.append("Codex: aggressive remediation (chaotic)")
        elif "fractal" in logic:
            actions.append("Codex: stepwise remediation (fractal)")
        elif "harmonic" in logic:
            actions.append("Codex: balance remediation (harmonic)")
        if ghost:
            actions.append("Ghost sync: verify phantom nodes, shorten telemetry retention")

    if not actions:
        actions.append("System healthy")

    return {
        "issues": issues,
        "actions": actions,
        "cpu_avg": cpu_hist.avg(),
        "mem_avg": mem_hist.avg(),
        "disk_avg": disk_hist.avg(),
        "cpu_trend": cpu_trend,
        "mem_trend": mem_trend,
        "disk_trend": disk_trend,
        "predictions": preds
    }

# ============================================================
# Adaptive Codex (pattern prediction)
# ============================================================

class AdaptiveCodex:
    def __init__(self):
        self.weights = [0.6, -0.8, -0.3]
        self.rules = {"purge_logic": "default"}
        self.telemetry = 60
        self.patterns, self.phantoms = [], []
        self.pattern_transitions = {
            "linear": ["harmonic","fractal"],
            "chaotic": ["stochastic","fractal"],
            "fractal": ["harmonic","linear"],
            "harmonic": ["linear","fractal"],
            "stochastic": ["chaotic","linear"]
        }

    def predict_next_pattern(self):
        if not self.patterns:
            return "linear"
        last = self.patterns[-1]
        candidates = self.pattern_transitions.get(last, ["linear"])
        # bias toward rising trends -> more chaotic/stochastic
        bias = []
        cpu_t, mem_t = cpu_hist.trend(), mem_hist.trend()
        if cpu_t == "rising" or mem_t == "rising":
            bias = ["chaotic","stochastic"]
        for b in bias:
            if b not in candidates:
                candidates.append(b)
        return random.choice(candidates) if candidates else "linear"

    def mutate(self):
        delta = [random.uniform(-0.1, 0.1) for _ in self.weights]
        self.weights = [w + d for w, d in zip(self.weights, delta)]
        # choose next pattern using predictor
        pat = self.predict_next_pattern()
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
            "ghost_sync": ghost,
            "predicted_next": self.predict_next_pattern()
        }

CODEX = AdaptiveCodex()

# ============================================================
# Operator Registry (with outcome learning hook)
# ============================================================

class OperatorRegistry:
    def __init__(self):
        self.ops = {}
        self.stats = {}  # operator_name -> {"used": int, "helpful": int}

    def register(self, tier, func, name):
        self.ops.setdefault(tier, []).append((name, func))
        self.stats.setdefault(name, {"used": 0, "helpful": 0})

    def feedback(self, name, helpful: bool):
        if name in self.stats:
            self.stats[name]["used"] += 1
            if helpful:
                self.stats[name]["helpful"] += 1

    def get_all(self): return self.ops
    def get_stats(self): return self.stats

REGISTRY = OperatorRegistry()

def op_drive(a): return {"tier": "feasible", "steps": ["Drive", "Arrive at " + a], "operator": "op_drive"}
def op_fly(a):   return {"tier": "feasible", "steps": ["Fly", "Land at " + a], "operator": "op_fly"}
REGISTRY.register("feasible", op_drive, "op_drive")
REGISTRY.register("feasible", op_fly, "op_fly")

# ============================================================
# Scenario Engine (near-future simulations)
# ============================================================

class ScenarioEngine:
    def __init__(self, horizon=10):
        self.horizon = horizon

    def simulate(self):
        preds = predictive_flags()
        scenarios = []

        def mk(name, cond, action):
            return {"name": name, "condition": cond, "proposed_action": action}

        if preds["cpu_future_stress"]:
            scenarios.append(mk("CPU spike imminent", True, "Preemptively reduce workers and throttle non-critical tasks"))
        if preds["mem_future_stress"]:
            scenarios.append(mk("Memory pressure imminent", True, "Close memory-heavy apps; enable compression or swap tuning"))
        if preds["disk_future_stress"]:
            scenarios.append(mk("Disk saturation imminent", True, "Rotate logs, purge temp files, move artifacts to backup"))

        # If no stress predicted, propose optimization scenario
        if not scenarios:
            scenarios.append(mk("Healthy horizon", True, "Scale workers up slightly; lower feed interval to increase granularity"))

        return {"horizon": self.horizon, "scenarios": scenarios, "predictions": preds}

SCENARIOS = ScenarioEngine(horizon=10)

# ============================================================
# Engine (dynamic worker scaling + predictive hints)
# ============================================================

class Engine:
    def __init__(self, registry, codex):
        self.registry, self.codex = registry, codex
        self.inbox, self.outbox = queue.Queue(), queue.Queue()
        self.running, self.lock, self.workers = False, threading.Lock(), []
        self.target_workers = 3

    def submit(self, text, meta=None):
        self.inbox.put({"text": text, "meta": meta or {}})

    def _apply_ops(self, payload_text, solver):
        # prioritize operators with better helpful ratio (learning hook)
        stats = self.registry.get_stats()
        ranked = []
        for tier, funcs in self.registry.get_all().items():
            for name, f in funcs:
                s = stats.get(name, {"used": 0, "helpful": 0})
                score = (s["helpful"] + 1) / (s["used"] + 2)  # laplace smoothing
                ranked.append((score, name, f))
        ranked.sort(reverse=True)
        applied = []
        for score, name, f in ranked:
            try:
                res = f(payload_text)
                res["score"] = round(score, 3)
                applied.append(res)
                # naive heuristic: mark helpful if solver suggests remediation
                helpful = any("remediation" in a.lower() or "reduce" in a.lower() for a in solver["actions"])
                self.registry.feedback(name, helpful)
            except Exception as e:
                applied.append({"tier":"notes","steps":[str(e)],"operator":name,"score":0.0})
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
            "operators": self._apply_ops(payload["text"], solver),
            "forecast": solver["predictions"],
            "scenarios": SCENARIOS.simulate(),
            "ts": now_ts()
        }
        # dynamic scaling hint
        stress_now = snap["cpu"] >= 85 or snap["mem"] >= 85
        stress_future = solver["predictions"]["cpu_future_stress"] or solver["predictions"]["mem_future_stress"]
        bundle["scale_hint"] = "down" if (stress_now or stress_future) else ("up" if snap["cpu"] < 40 and snap["mem"] < 60 else "steady")
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
        current = len(self.workers)
        if self.target_workers > current:
            for _ in range(self.target_workers - current):
                t = threading.Thread(target=self._worker, args=(len(self.workers),), daemon=True)
                t.start()
                self.workers.append(t)
        elif self.target_workers < current:
            # cannot kill threads; they will idle
            pass

    def start(self, n=3):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.target_workers = n
        self._adjust_workers()
        threading.Thread(target=self._scaler_loop, daemon=True).start()
        print(f"[ENGINE] Started with target {self.target_workers} workers.")

    def _scaler_loop(self):
        while self.running:
            cpu_avg = cpu_hist.avg()
            mem_avg = mem_hist.avg()
            preds = predictive_flags()
            stress_future = preds["cpu_future_stress"] or preds["mem_future_stress"]
            if cpu_avg >= 85 or mem_avg >= 85 or stress_future:
                self.target_workers = max(1, self.target_workers - 1)
            elif cpu_avg < 40 and mem_avg < 60 and not stress_future:
                self.target_workers = min(8, self.target_workers + 1)
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
# SystemFeed (adaptive interval + anticipatory pacing)
# ============================================================

class SystemFeed:
    def __init__(self, engine):
        self.engine, self.running = engine, False
        self.interval = 5.0

    def _adapt_interval(self):
        preds = predictive_flags()
        cpu = cpu_hist.avg(); mem = mem_hist.avg()
        if cpu >= 85 or mem >= 85 or preds["cpu_future_stress"] or preds["mem_future_stress"]:
            self.interval = 8.0
        elif cpu < 40 and mem < 60 and not (preds["cpu_future_stress"] or preds["mem_future_stress"]):
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
# GUI (charts + forecast + audit)
# ============================================================

class EngineGUI:
    def __init__(self, engine):
        if not tk or not scrolledtext or not ttk:
            raise RuntimeError("Tkinter not available. Install tkinter to use the GUI.")
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Predictive Adaptive Engine Monitor")
        self.root.geometry("1150x750")

        # Header
        header = tk.Frame(self.root)
        header.pack(fill=tk.X, pady=5)
        tk.Label(header, text="Running", fg="green", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=8)
        self.progress = ttk.Progressbar(header, mode="indeterminate"); self.progress.pack(side=tk.LEFT, padx=8); self.progress.start(100)
        tk.Button(header, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT, padx=8)

        # Charts frame
        charts = tk.Frame(self.root)
        charts.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig = plt.figure(figsize=(9, 3)) if plt else None
        if self.fig and FigureCanvasTkAgg:
            self.ax_cpu = self.fig.add_subplot(141)
            self.ax_mem = self.fig.add_subplot(142)
            self.ax_disk = self.fig.add_subplot(143)
            self.ax_fc   = self.fig.add_subplot(144)
            self.canvas = FigureCanvasTkAgg(self.fig, master=charts)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            self.ax_cpu = self.ax_mem = self.ax_disk = self.ax_fc = None
            self.canvas = None
            tk.Label(charts, text="Charts disabled (matplotlib/tk backend missing)").pack()

        # Log and audit
        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.log = scrolledtext.ScrolledText(bottom, width=100, height=18, font=("Consolas", 10))
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.audit = scrolledtext.ScrolledText(bottom, width=45, height=18, font=("Consolas", 10))
        self.audit.pack(side=tk.RIGHT, fill=tk.BOTH)

        self._update_loop()

    def _clear_log(self):
        self.log.delete("1.0", tk.END)
        self.audit.delete("1.0", tk.END)

    def _update_charts(self, preds):
        if not self.canvas or not self.ax_cpu: return
        cpu_vals  = list(cpu_hist.values) if hasattr(cpu_hist, "values") else []
        mem_vals  = list(mem_hist.values) if hasattr(mem_hist, "values") else []
        disk_vals = list(disk_hist.values) if hasattr(disk_hist, "values") else []
        for ax in (self.ax_cpu, self.ax_mem, self.ax_disk, self.ax_fc):
            ax.clear()
        self.ax_cpu.plot(cpu_vals[-120:], color="red");   self.ax_cpu.set_title("CPU %")
        self.ax_mem.plot(mem_vals[-120:], color="blue");  self.ax_mem.set_title("MEM %")
        self.ax_disk.plot(disk_vals[-120:], color="green"); self.ax_disk.set_title("DISK %")
        self.ax_cpu.set_ylim(0, 100); self.ax_mem.set_ylim(0, 100); self.ax_disk.set_ylim(0, 100)

        # Forecast panel (overlay)
        self.ax_fc.plot(preds.get("cpu_forecast", []), color="red", label="CPU→")
        self.ax_fc.plot(preds.get("mem_forecast", []), color="blue", label="MEM→")
        self.ax_fc.plot(preds.get("disk_forecast", []), color="green", label="DISK→")
        self.ax_fc.set_title("Forecast (next horizon)")
        self.ax_fc.set_ylim(0, 100)
        self.ax_fc.legend(loc="upper right", fontsize=8)

        self.canvas.draw_idle()

    def _render_bundle(self, b):
        s = b.get("system", {})
        sol = b.get("solver", {})
        codex = b.get("codex", {})
        preds = sol.get("predictions", {})
        self.log.insert(tk.END, f"{b.get('id')} :: {b.get('answer')}\n"
                                f" CPU={s.get('cpu',0)}% MEM={s.get('mem',0)}% DISK={s.get('disk',0)}% PROCS={s.get('procs',0)}\n")
        if sol.get("issues"):
            self.log.insert(tk.END, f" Issues: {', '.join(sol['issues'])}\n")
        actions = sol.get("actions", ["System healthy"])
        self.log.insert(tk.END, f" Actions: {actions[0]}\n"
                                f" Codex purge_logic={codex.get('rules',{}).get('purge_logic')} ghost={codex.get('ghost_sync')} next={codex.get('predicted_next')}\n"
                                f" Trends: CPU={sol.get('cpu_trend')} MEM={sol.get('mem_trend')} DISK={sol.get('disk_trend')}\n"
                                f" Forecast stress: CPU={preds.get('cpu_future_stress')} MEM={preds.get('mem_future_stress')} DISK={preds.get('disk_future_stress')}\n"
                                f"{'-'*100}\n")
        self.log.see(tk.END)

        # Operator audit trail
        ops = b.get("operators", [])
        self.audit.insert(tk.END, f"{b.get('id')} operators (top 10):\n")
        for op in ops[:10]:
            self.audit.insert(tk.END, f" - {op.get('operator')} [{op.get('tier')} | score={op.get('score')}]: {', '.join(op.get('steps', []))}\n")
        self.audit.insert(tk.END, "-"*50 + "\n")
        self.audit.see(tk.END)

        # Update charts with latest predictions
        self._update_charts(preds)

    def _update_loop(self):
        try:
            bundles = self.engine.fetch()
            for b in bundles:
                self._render_bundle(b)
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
        # Read-only codex state for API
        codex_state = {"rules": CODEX.rules, "telemetry": CODEX.telemetry,
                       "patterns": CODEX.patterns, "phantoms": CODEX.phantoms,
                       "weights": CODEX.weights, "ghost_sync": False,
                       "predicted_next": CODEX.predict_next_pattern()}
        solver = system_solver(snap, codex_state)
        scenarios = SCENARIOS.simulate()
        return {"snapshot": snap, "solver": solver, "scenarios": scenarios}
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
        print("[INFO] Charts disabled; starting text-only GUI.")
        EngineGUI(engine).run()
    else:
        print("[WARNING] Tkinter/matplotlib not available, GUI disabled. Running headless.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()

