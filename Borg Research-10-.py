# ============================================================
# Mythic Swarm Engine + Live GUI + CPU/Storage Controls + Auto LLM Bridge
# With LLM Connection Indicator in GUI
# Single-file runnable demo with automatic LLM backend detection
# ============================================================

# --- Autoloader for optional libs ---
import importlib
import importlib.util  # important for find_spec
import subprocess, sys
def autoload(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception:
                pass
autoload(["numpy", "psutil"])

# --- Imports & globals ---
import os, json, time, random, hashlib, math, zlib, logging, datetime, threading, queue, textwrap, traceback
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import numpy  # type: ignore
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

_shutdown_event = threading.Event()
_archive_lock = threading.Lock()
_storage_dir_lock = threading.Lock()
_storage_dir = os.path.join(os.getcwd(), "projects")

# ============================================================
# Idea engine
# ============================================================
PRIMITIVES = {
    "domains": ["gravity","magnetism","plasma","atmosphere","materials","orbits","energy","optics","computation","biodesign"],
    "mechanisms": ["field_weaving","resonance","gradient_control","phase_modulation","turbulence_guidance","lattice_tuning","feedback_adaptation","nonlinear_coupling","quantized_transport","topology_shaping"],
    "architectures": ["ring_habitat","torsion_drive","beam_array","mesh_network","queen_board_swarm","shield_skin","saddle_coil","shell_stack","spiral_tower","orbital_kite"],
    "intents": ["stability","mobility","harvesting","shielding","navigation","cooling","amplification","damping","sensing","repair"],
    "constraints": ["low_mass","low_power","fail_safe","zero_emissions","self_healing","rapid_deploy","modular","scalable","silent","transparent"]
}
MYTHIC = ["Aegis","Arcadia","Atlas","Aurora","Borealis","Cerberus","Chimera","Chronos","Daedalus","Elysium","Gaea",
          "Helios","Hermes","Hyperion","Icarus","Leviathan","Lyra","Nereid","Nyx","Orion","Pandora","Phoenix",
          "Prometheus","Rhea","Selene","Talos","Titan","Valkyrie","Zephyr"]
VERBS = ["weaver","foundry","engine","loom","array","forge","matrix","mirror","harbor","beacon","horizon","garden",
         "crucible","compass","lattice","spiral","sphere","saddle","shield","tower","kite","swarm","mesh","glider","bridge"]
QUALS = ["prime","delta","x","ghost","weave","nova","flux","arc","pulse","shade","aether","echo"]

def autonomous_project_name(idea: dict, node: str) -> str:
    dom = idea.get("domain","unknown"); mech = idea.get("mechanism","unknown")
    arch = idea.get("architecture","unknown"); intent = idea.get("intent","unknown")
    base = f"{dom}-{mech}-{arch}-{intent}"
    myth = random.choice(MYTHIC); noun = random.choice(VERBS); qual = random.choice(QUALS)
    short = hashlib.sha1((base + node + str(time.time()) + str(random.random())).encode("utf-8")).hexdigest()[:6]
    return f"{myth}-{noun}-{qual}-{short}"

def compose_idea(seed_noise: float = 0.15) -> dict:
    def pick(cat):
        choice = random.choice(PRIMITIVES[cat])
        if random.random() < seed_noise:
            choice = f"{choice}_{random.choice(['delta','x','prime','weave','ghost'])}"
        return choice
    idea = {
        "domain": pick("domains"),
        "mechanism": random.choice(PRIMITIVES["mechanisms"]),
        "architecture": random.choice(PRIMITIVES["architectures"]),
        "intent": random.choice(PRIMITIVES["intents"]),
        "constraints": random.sample(PRIMITIVES["constraints"], k=random.randint(2,5)),
        "params": {
            "scale": round(10 ** random.uniform(-3, 3), 4),
            "power_budget": round(10 ** random.uniform(-1, 6), 3),
            "tolerance": round(random.uniform(0.01, 0.2), 3),
            "mutation_rate": round(random.uniform(0.05, 0.35), 3),
        }
    }
    idea["title"] = f"{idea['domain']}-{idea['mechanism']}-{idea['architecture']}-{idea['intent']}"
    return idea

def ensure_idea_keys(idea: dict) -> dict:
    base = compose_idea(seed_noise=0.0); out = dict(base); out.update(idea or {})
    if "constraints" not in out or not isinstance(out["constraints"], list) or len(out["constraints"]) == 0:
        out["constraints"] = base["constraints"]
    if "params" not in out or not isinstance(out["params"], dict):
        out["params"] = base["params"]
    for p in ["scale","power_budget","tolerance","mutation_rate"]:
        if p not in out["params"]: out["params"][p] = base["params"][p]
    out["title"] = f"{out['domain']}-{out['mechanism']}-{out['architecture']}-{out['intent']}"
    return out

def clamp_params(idea: dict) -> dict:
    idea = ensure_idea_keys(idea); p = idea["params"]
    p["scale"] = float(max(1e-3, min(1e3, p["scale"])))
    p["power_budget"] = float(max(1e-1, min(1e6, p["power_budget"])))
    p["tolerance"] = float(max(0.0, min(0.5, p["tolerance"])))
    p["mutation_rate"] = float(max(0.0, min(1.0, p["mutation_rate"])))
    idea["params"] = p; idea["title"] = f"{idea['domain']}-{idea['mechanism']}-{idea['architecture']}-{idea['intent']}"
    return idea

def mutate_idea(idea: dict) -> dict:
    m = ensure_idea_keys(idea)
    slot = random.choice(["domain","mechanism","architecture","intent"])
    if slot == "domain": m["domain"] = random.choice(PRIMITIVES["domains"])
    elif slot == "mechanism": m["mechanism"] = random.choice(PRIMITIVES["mechanisms"])
    elif slot == "architecture": m["architecture"] = random.choice(PRIMITIVES["architectures"])
    else: m["intent"] = random.choice(PRIMITIVES["intents"])
    for p in ["scale","power_budget","tolerance","mutation_rate"]:
        m["params"][p] = round(float(m["params"].get(p, 1.0)) * random.uniform(0.7, 1.3), 4)
    if random.random() < 0.5:
        m["constraints"] = list(set(m["constraints"] + [random.choice(PRIMITIVES["constraints"]) ]))
    m["title"] = f"{m['domain']}-{m['mechanism']}-{m['architecture']}-{m['intent']}"
    return m

def mutate_idea_safe(idea: dict) -> dict:
    m = mutate_idea(idea); m["constraints"] = list(dict.fromkeys(m["constraints"]))[:7]; return clamp_params(m)

def idea_to_text(idea: dict) -> str:
    return json.dumps(ensure_idea_keys(idea), sort_keys=True)

# ============================================================
# Evaluation
# ============================================================
def text_distance(a: str, b: str) -> float:
    ta, tb = set(a.split('"')), set(b.split('"')); inter = len(ta & tb); union = len(ta | tb)
    return 1.0 - (inter / union if union else 0.0)

def compressibility_score(s: str) -> float:
    comp = zlib.compress(s.encode("utf-8")); return len(comp) / max(1, len(s))

def novelty_score(candidate_text: str, archive_texts: list) -> float:
    if not archive_texts: return 1.0 + compressibility_score(candidate_text)
    if HAS_NUMPY:
        distances = []; step = max(1, len(archive_texts)//1000)
        for t in archive_texts[::step]: distances.append(text_distance(candidate_text, t))
        avg_dist = float(numpy.mean(numpy.array(distances))) if distances else 1.0
    else:
        distances = [text_distance(candidate_text, t) for t in archive_texts]
        avg_dist = sum(distances) / len(distances)
    return avg_dist + 0.5 * compressibility_score(candidate_text)

class Simulator:
    name: str = "base"
    def score(self, idea: dict): raise NotImplementedError

class OrbitalComfortSimulator(Simulator):
    name = "orbital_comfort"
    def score(self, idea: dict):
        p = ensure_idea_keys(idea)["params"]; radius = max(10.0, p["scale"])
        rpm = max(0.1, min(6.0, 60.0 * math.sqrt(9.81 / radius) / (2*math.pi)))
        g = (4 * math.pi**2 * radius * (rpm/60.0)**2)
        comfort_penalty = abs(g - 9.81) / 9.81; coriolis_penalty = max(0.0, (rpm - 2.0) / 4.0)
        power_ok = min(1.0, 1.0 / (1.0 + p["power_budget"]/1e5))
        utility = max(0.0, 1.0 - comfort_penalty - coriolis_penalty) * (0.5 + 0.5*power_ok)
        diag = {"radius_m": radius, "rpm": rpm, "g_mps2": g, "comfort_penalty": comfort_penalty,
                "coriolis_penalty": coriolis_penalty, "power_factor": power_ok}
        return utility, diag

class AtmosphereRoutingSimulator(Simulator):
    name = "atmo_energy_routing"
    def score(self, idea: dict):
        idea = ensure_idea_keys(idea); p = idea["params"]
        intent_bonus = 0.2 if idea["intent"] in ["shielding","damping","navigation"] else 0.0
        constraint_bonus = 0.1 * len([c for c in idea["constraints"] if c in ["low_power","fail_safe","zero_emissions"]])
        power_penalty = 0.0 if idea["intent"] == "amplification" else min(0.6, math.log10(max(1.0, p["power_budget"])) / 10.0)
        stability = max(0.0, 1.0 - p["tolerance"]); utility = max(0.0, stability + intent_bonus + constraint_bonus - power_penalty)
        return utility, {"stability": stability, "intent_bonus": intent_bonus, "constraint_bonus": constraint_bonus, "power_penalty": power_penalty}

SIMULATORS = [OrbitalComfortSimulator(), AtmosphereRoutingSimulator()]
def idea_hash(text: str) -> str: return hashlib.sha1(text.encode("utf-8")).hexdigest()

def impact_score(idea: dict) -> float:
    cons = set(ensure_idea_keys(idea).get("constraints", [])); score = 0.0
    if "low_power" in cons: score += 0.2
    if "zero_emissions" in cons: score += 0.3
    if "fail_safe" in cons: score += 0.2
    if "self_healing" in cons: score += 0.2
    if "modular" in cons: score += 0.1
    if "scalable" in cons: score += 0.1
    if "low_mass" in cons: score += 0.1
    if "silent" in cons: score += 0.1
    if "transparent" in cons: score += 0.1
    return min(1.0, score)

def evaluate_idea_full(idea: dict, archive_texts: list, cfg: dict,
                       perf: dict = None, weights_override: dict = None) -> dict:
    idea = clamp_params(ensure_idea_keys(idea)); text = idea_to_text(idea); novelty = novelty_score(text, archive_texts)
    results = []
    for sim in SIMULATORS:
        try:
            score, diag = sim.score(idea)
        except Exception:
            logging.exception(f"Simulator {sim.name} failed; continuing."); score, diag = 0.0, {"error": "simulator_failure"}
        results.append((sim.name, score, diag))
    utility = sum(s for _, s, _ in results) / len(results) if results else 0.0
    curiosity = novelty * (0.6 + 0.4 * (1.0 - utility)); impact = impact_score(idea)
    w = {"novelty_weight": cfg.get("novelty_weight", 0.25),
         "utility_weight": cfg.get("utility_weight", 0.35),
         "impact_weight": cfg.get("impact_weight", 0.25),
         "curiosity_weight": cfg.get("curiosity_weight", 0.15)}
    if weights_override: w.update({k: weights_override.get(k, w[k]) for k in w.keys()})
    final_score = (w["novelty_weight"]*novelty + w["utility_weight"]*utility + w["impact_weight"]*impact + w["curiosity_weight"]*curiosity)
    return {"idea": idea, "text": text, "novelty": novelty, "utility": utility, "curiosity": curiosity, "impact": impact,
            "final_score": final_score, "weights": w, "sims": results, "hash": idea_hash(text),
            "name": autonomous_project_name(idea, cfg.get("node_name","local")), "objective": final_score}

# ============================================================
# Strategies
# ============================================================
STRATEGY_PATH = os.path.join(os.getcwd(), "strategies.py")
def ensure_strategy_file(cfg: dict):
    if os.path.exists(STRATEGY_PATH): return
    nw, uw, iw, cw = (cfg.get("novelty_weight",0.25), cfg.get("utility_weight",0.35),
                      cfg.get("impact_weight",0.25), cfg.get("curiosity_weight",0.15))
    baseline = f'''# Auto-generated strategies module
weights = {{"novelty_weight": {nw}, "utility_weight": {uw}, "impact_weight": {iw}, "curiosity_weight": {cw}}}
def adjust_weights(perf):
    w = dict(weights)
    cr = perf.get("completion_rate", 0.0)
    avg_obj = perf.get("avg_objective", 0.0)
    feed_entropy = perf.get("feed_entropy", 0.0)
    sys_entropy = perf.get("sys_entropy", 0.0)
    if cr < 0.05:
        w["novelty_weight"] = min(1.2, w["novelty_weight"] + 0.05)
        w["curiosity_weight"] = min(1.2, w["curiosity_weight"] + 0.03)
    if avg_obj > 1.0:
        w["utility_weight"] = min(1.2, w["utility_weight"] + 0.04)
        w["impact_weight"] = min(1.2, w["impact_weight"] + 0.02)
    w["novelty_weight"] = min(1.3, w["novelty_weight"] + 0.01 * feed_entropy)
    w["curiosity_weight"] = min(1.3, w["curiosity_weight"] + 0.01 * sys_entropy)
    return w
'''
    with open(STRATEGY_PATH, "w", encoding="utf-8") as f: f.write(baseline)

def load_strategy():
    try:
        importlib.invalidate_caches(); return importlib.import_module("strategies")
    except Exception:
        logging.exception("Failed to load strategies module."); return None

def mutate_strategy(cfg: dict):
    try:
        with open(STRATEGY_PATH, "r", encoding="utf-8") as f: text = f.read()
        scale = float(cfg.get("strategy_mutation_scale", 0.02))
        def jitter(val: float) -> float: return max(0.0, val + random.uniform(-scale, scale))
        def mutate_line(line: str) -> str:
            for base in ["0.05","0.03","0.04","0.02"]: line = line.replace(base, f"{jitter(float(base)):.3f}")
            return line
        mutated = "\n".join(mutate_line(ln) for ln in text.splitlines())
        with open(STRATEGY_PATH, "w", encoding="utf-8") as f: f.write(mutated)
        importlib.invalidate_caches(); logging.info("Strategy file mutated.")
    except Exception:
        logging.exception("Failed to mutate strategy file.")

# ============================================================
# Archive & storage paths (disk-usage-aware selection)
# ============================================================
def node_paths(cfg: dict) -> dict:
    base = cfg.get("base_dir", os.getcwd())
    arch_path = cfg.get("archive_path", os.path.join(base, "archive.json"))
    storage = cfg.get("storage_dir", os.path.join(base, "projects"))
    return {"archive_path": arch_path, "storage_dir": storage}

def choose_storage_dir(cfg: dict) -> str:
    manual = cfg.get("manual_storage_dir")
    if manual:
        os.makedirs(manual, exist_ok=True); return manual
    root = cfg.get("storage_dir", os.path.join(os.getcwd(), "projects"))
    threshold = float(cfg.get("storage_switch_threshold_pct", 70.0))
    try:
        if psutil:
            usage = psutil.disk_usage("C:\\"); percent = usage.percent
            if percent > threshold:
                alt = cfg.get("alternate_storage_dir")
                if alt:
                    logging.warning(f"C: drive {percent:.1f}% full. Switching to alternate storage: {alt}")
                    os.makedirs(alt, exist_ok=True); root = alt
    except Exception:
        logging.debug("Disk usage check failed; using default storage.")
    if cfg.get("storage_daily_subdir", True):
        day = datetime.datetime.now().strftime("%Y-%m-%d"); return os.path.join(root, day)
    return root

def _set_storage_dir(path: str):
    with _storage_dir_lock:
        global _storage_dir; _storage_dir = path

def get_storage_dir() -> str:
    with _storage_dir_lock: return _storage_dir

def load_archive(paths: dict) -> dict:
    ap = paths.get("archive_path", os.path.join(os.getcwd(), "archive.json"))
    if os.path.exists(ap):
        try:
            with open(ap, "r", encoding="utf-8") as f: data = json.load(f)
            if not isinstance(data, dict): raise ValueError("Archive not dict")
            data.setdefault("ideas", {}); data.setdefault("top", []); data.setdefault("history", [])
            stats = data.setdefault("stats", {})
            stats.setdefault("iterations", 0); stats.setdefault("active_projects", 0); stats.setdefault("completed_projects", 0)
            stats.setdefault("active_project_names", []); stats.setdefault("finished_project_names", [])
            stats.setdefault("feed_entropy", 0.0); stats.setdefault("sys_entropy", 0.0)
            stats.setdefault("storage_dir", paths.get("storage_dir", "")); return data
        except Exception:
            logging.exception("Archive load failed; starting fresh.")
    return {"ideas": {}, "top": [], "history": [],
            "stats": {"iterations": 0, "active_projects": 0, "completed_projects": 0, "active_project_names": [],
                      "finished_project_names": [], "feed_entropy": 0.0, "sys_entropy": 0.0,
                      "storage_dir": paths.get("storage_dir", "")} }

def save_archive(paths: dict, archive: dict):
    ap = paths.get("archive_path", os.path.join(os.getcwd(), "archive.json"))
    try:
        tmp = ap + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: json.dump(archive, f, indent=2, ensure_ascii=False)
        os.replace(tmp, ap)
    except Exception:
        logging.exception("Failed to save archive.")

def enforce_archive_limits(archive: dict, cfg: dict):
    max_size = int(cfg.get("max_archive_size", 1000)); ideas = archive.get("ideas", {})
    if len(ideas) > max_size:
        def score_of(item):
            try:
                ev = item[1].get("eval") or item[1]; return ev.get("final_score", ev.get("objective", 0.0))
            except Exception: return 0.0
        ids_sorted = sorted(ideas.items(), key=score_of, reverse=True)
        keep_ids = set(k for k, _ in ids_sorted[:max_size])
        archive["ideas"] = {i: ideas[i] for i in keep_ids}
        archive["top"] = [t for t in archive.get("top", []) if t.get("id") in keep_ids]
        logging.warning(f"Archive trimmed to max size {max_size}.")

# ============================================================
# Metrics & entropy
# ============================================================
class SystemMetrics:
    def __init__(self):
        self.lock = threading.Lock(); self.snapshot_data = {}
    def poll(self):
        if psutil is None: return
        try:
            cpu = psutil.cpu_percent(interval=0.15)
            mem = psutil.virtual_memory(); disk = psutil.disk_io_counters() if hasattr(psutil, "disk_io_counters") else None
            net = psutil.net_io_counters() if hasattr(psutil, "net_io_counters") else None
            procs = len(psutil.pids()); conns = len(psutil.net_connections(kind="inet")) if hasattr(psutil, "net_connections") else 0
            with self.lock:
                self.snapshot_data = {"cpu": cpu, "mem_used": mem.used if mem else 0, "mem_avail": mem.available if mem else 0,
                                      "disk_read": getattr(disk, "read_bytes", 0) if disk else 0,
                                      "disk_write": getattr(disk, "write_bytes", 0) if disk else 0,
                                      "net_sent": getattr(net, "bytes_sent", 0) if net else 0,
                                      "net_recv": getattr(net, "bytes_recv", 0) if net else 0, "procs": procs, "conns": conns}
        except Exception: pass
    def start(self, interval: float = 2.0):
        def loop():
            while not _shutdown_event.is_set():
                self.poll(); time.sleep(interval)
        threading.Thread(target=loop, daemon=True).start()
    def stop(self): _shutdown_event.set()
    def snapshot(self) -> dict:
        with self.lock: return dict(self.snapshot_data)

def system_entropy(sys_metrics: dict) -> float:
    cpu = sys_metrics.get("cpu", 0.0) / 100.0; procs = sys_metrics.get("procs", 0); conns = sys_metrics.get("conns", 0)
    net = (sys_metrics.get("net_sent", 0) + sys_metrics.get("net_recv", 0)) / 1e9
    disk = (sys_metrics.get("disk_read", 0) + sys_metrics.get("disk_write", 0)) / 1e9
    return min(2.0, 0.5 * cpu + 0.001 * procs + 0.002 * conns + 0.1 * net + 0.1 * disk)

class FeedRegistry:
    def snapshot(self) -> dict: return {"items": 0, "sources": 0}

def feed_entropy(feeds: FeedRegistry) -> float:
    try:
        snap = feeds.snapshot(); items = float(snap.get("items", 0)); sources = float(snap.get("sources", 0))
        return min(2.0, 0.001 * items + 0.05 * sources)
    except Exception: return 0.0

# ============================================================
# Throttle (CPU cap target <= 50%)
# ============================================================
class Throttle:
    def __init__(self, cfg: dict):
        target = min(0.5, float(cfg.get("cpu_target_util", 0.5)))
        self.cfg = {"sleep_min_seconds": cfg.get("sleep_min_seconds", 0.025),
                    "sleep_max_seconds": cfg.get("sleep_max_seconds", 0.8),
                    "cpu_target_util": target}
        self.sleep_s = self.cfg["sleep_min_seconds"]
    def adjust(self):
        if psutil is None:
            self.sleep_s = min(self.cfg["sleep_max_seconds"], self.sleep_s * 1.15 + 0.03); return
        try:
            cpu = psutil.cpu_percent(interval=0.15) / 100.0; target = self.cfg["cpu_target_util"]
            if cpu > target: self.sleep_s = min(self.cfg["sleep_max_seconds"], self.sleep_s * 1.35 + 0.06)
            else: self.sleep_s = max(self.cfg["sleep_min_seconds"], self.sleep_s * 0.92 - 0.01)
        except Exception: pass
    def wait(self): time.sleep(self.sleep_s)

# ============================================================
# Status bus (GUI + LLM bridge)
# ============================================================
class StatusBus:
    def __init__(self): self.q = queue.Queue()
    def emit(self, event: str, payload: dict = None):
        try: self.q.put_nowait({"event": event, "payload": payload or {}})
        except Exception: pass
    def get_nowait(self):
        try: return self.q.get_nowait()
        except queue.Empty: return None

# ============================================================
# Large project builder (1000+ lines)
# ============================================================
class CodeBuilder:
    def __init__(self, name: str, idea: dict, evals: dict):
        self.name = name; self.idea = idea; self.evals = evals; self.lines = []
    def add_header(self):
        hdr = [f"# Project: {self.idea['title']}", f"# Name: {self.name}",
               f"# Constraints: {', '.join(self.idea.get('constraints', []))}",
               f"# Params: scale={self.idea['params']['scale']}, power_budget={self.idea['params']['power_budget']}, tol={self.idea['params']['tolerance']}, mutation_rate={self.idea['params']['mutation_rate']}",
               f"# Scores: final={self.evals.get('final_score', self.evals.get('objective', 0.0)):.4f} novelty={self.evals['novelty']:.4f} utility={self.evals['utility']:.4f} impact={self.evals['impact']:.4f} curiosity={self.evals['curiosity']:.4f}",
               "", '"""', "Generated mythic scaffold with modules, classes, CLI, and stubs.", '"""', "",
               "import sys, os, json, math, time, random, logging, dataclasses, typing", "from typing import Dict, List, Tuple, Optional",
               "logging.basicConfig(level=logging.INFO)", ""]
        self.lines.extend(hdr)
    def add_config(self):
        self.lines.extend(["class Config:",
                           "    def __init__(self):",
                           f"        self.name = '{self.name}'",
                           f"        self.title = '{self.idea['title']}'",
                           "        self.seed = 42", "        self.iterations = 200", "        self.enable_logging = True",
                           "        self.output_dir = os.path.join(os.getcwd(), 'out')", "",
                           "    def ensure(self):", "        os.makedirs(self.output_dir, exist_ok=True)", ""])
    def add_dataclasses(self):
        self.lines.extend(["@dataclasses.dataclass", "class Parameters:", "    scale: float", "    power_budget: float",
                           "    tolerance: float", "    mutation_rate: float", "",
                           "@dataclasses.dataclass", "class Diagnostics:", "    final: float", "    novelty: float",
                           "    utility: float", "    impact: float", "    curiosity: float", ""])
    def add_utils(self):
        self.lines.extend(["def clamp(v, lo, hi):", "    return max(lo, min(hi, v))", "",
                           "def pretty(obj):", "    return json.dumps(obj, indent=2)", "",
                           "def seed_all(seed: int):", "    random.seed(seed)",
                           "    try:", "        import numpy", "        numpy.random.seed(seed)", "    except Exception:", "        pass", ""])
    def add_sim_modules(self):
        self.lines.extend(["class ComfortSim:",
                           "    def run(self, p: Parameters) -> Dict:",
                           "        radius = max(10.0, p.scale)",
                           "        rpm = max(0.1, min(6.0, 60.0 * math.sqrt(9.81 / radius) / (2*math.pi)))",
                           "        g = (4 * math.pi**2 * radius * (rpm/60.0)**2)",
                           "        comfort_penalty = abs(g - 9.81) / 9.81",
                           "        coriolis_penalty = max(0.0, (rpm - 2.0) / 4.0)",
                           "        power_ok = min(1.0, 1.0 / (1.0 + p.power_budget/1e5))",
                           "        utility = max(0.0, 1.0 - comfort_penalty - coriolis_penalty) * (0.5 + 0.5*power_ok)",
                           "        return {'radius': radius, 'rpm': rpm, 'g': g, 'utility': utility}", "",
                           "class RoutingSim:",
                           "    def run(self, p: Parameters, intent: str, constraints: List[str]) -> Dict:",
                           "        intent_bonus = 0.2 if intent in ['shielding','damping','navigation'] else 0.0",
                           "        constraint_bonus = 0.1 * len([c for c in constraints if c in ['low_power','fail_safe','zero_emissions']])",
                           "        power_penalty = 0.0 if intent == 'amplification' else min(0.6, math.log10(max(1.0, p.power_budget)) / 10.0)",
                           "        stability = max(0.0, 1.0 - p.tolerance)",
                           "        utility = max(0.0, stability + intent_bonus + constraint_bonus - power_penalty)",
                           "        return {'stability': stability, 'utility': utility}", ""])
    def add_core_classes(self):
        self.lines.extend(["class ProjectCore:",
                           "    def __init__(self, params: Parameters, title: str, constraints: List[str], intent: str):",
                           "        self.params = params; self.title = title; self.constraints = constraints; self.intent = intent",
                           "        self.comfort = ComfortSim(); self.routing = RoutingSim()", "",
                           "    def step(self) -> Dict:", "        c = self.comfort.run(self.params)",
                           "        r = self.routing.run(self.params, self.intent, self.constraints)", "        return {'comfort': c, 'routing': r}", "",
                           "    def run(self, iterations: int = 100) -> Dict:", "        log = []",
                           "        for i in range(iterations):", "            log.append(self.step())",
                           "        return {'title': self.title, 'log': log, 'constraints': self.constraints}", ""])
    def add_validators(self):
        self.lines.extend(["def validate_params(p: Parameters) -> List[str]:", "    errs = []",
                           "    if not (1e-3 <= p.scale <= 1e3): errs.append('scale out of range')",
                           "    if not (1e-1 <= p.power_budget <= 1e6): errs.append('power_budget out of range')",
                           "    if not (0.0 <= p.tolerance <= 0.5): errs.append('tolerance out of range')",
                           "    if not (0.0 <= p.mutation_rate <= 1.0): errs.append('mutation_rate out of range')",
                           "    return errs", ""])
    def add_cli(self):
        self.lines.extend(["def main():", "    cfg = Config(); cfg.ensure(); seed_all(cfg.seed)",
                           f"    p = Parameters(scale={self.idea['params']['scale']}, power_budget={self.idea['params']['power_budget']}, tolerance={self.idea['params']['tolerance']}, mutation_rate={self.idea['params']['mutation_rate']})",
                           "    errs = validate_params(p)", "    if errs:", "        logging.error('Validation errors: %s', errs); sys.exit(1)",
                           f"    core = ProjectCore(p, '{self.idea['title']}', {self.idea.get('constraints', [])}, '{self.idea['intent']}')",
                           "    out = core.run(iterations=200)", "    path = os.path.join(cfg.output_dir, 'result.json')",
                           "    with open(path, 'w', encoding='utf-8') as f: f.write(pretty(out))", "    logging.info('Wrote %s', path)", "",
                           "if __name__ == '__main__':", "    main()", ""])
    def add_fillers(self, target_lines: int = 1000):
        stub = ["def _stub_fn_{i}(x):", "    return x * {i}", "", "class _StubClass_{i}:", "    def __init__(self):",
                "        self.v = {i}", "    def m(self, y):", "        return self.v + y", ""]
        i = 0
        while len(self.lines) < target_lines:
            self.lines.extend([ln.format(i=i) for ln in stub]); i += 1
    def build(self, min_lines: int = 1000) -> str:
        self.add_header(); self.add_config(); self.add_dataclasses(); self.add_utils()
        self.add_sim_modules(); self.add_core_classes(); self.add_validators(); self.add_cli(); self.add_fillers(min_lines)
        return "\n".join(self.lines)

def generate_large_project_file(idea: dict, evals: dict, cfg: dict, hid: str):
    used_dir = choose_storage_dir(cfg); os.makedirs(used_dir, exist_ok=True)
    name = idea.get("project_name") or evals.get("name") or hid[:8]
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)
    fpath = os.path.join(used_dir, f"{safe}.py"); builder = CodeBuilder(name=safe, idea=idea, evals=evals)
    code = builder.build(min_lines=cfg.get("min_project_lines", 1000))
    with open(fpath, "w", encoding="utf-8") as f: f.write(code)
    return fpath, len(code.splitlines()), used_dir

# ============================================================
# Borg mesh & visualization
# ============================================================
class BorgMesh:
    def __init__(self): self.nodes = {}; self.edges = set()
    def discover(self, url: str, snippet: str, links: list):
        self.nodes.setdefault(url, {"state": "discovered", "risk": random.randint(0, 70), "label": snippet[:22]})
        for l in links:
            self.nodes.setdefault(l, {"state": "discovered", "risk": random.randint(0, 50), "label": l.split("/")[-1][:22]})
            self.edges.add((url, l))
    def build(self, url: str):
        if url in self.nodes: self.nodes[url]["state"] = "built"; return True
        return False
    def enforce(self, url: str):
        if url in self.nodes: self.nodes[url]["state"] = "enforced"; return True
        return False
    def snapshot(self) -> dict: return {"nodes": {u: dict(v) for u, v in self.nodes.items()}, "edges": list(self.edges)}

class MeshGraphPanel(ttk.Frame):
    def __init__(self, parent, get_mesh_snapshot, width=520, height=420):
        super().__init__(parent); self.get_mesh_snapshot = get_mesh_snapshot
        self.canvas = tk.Canvas(self, width=width, height=height, bg="#0e0f13", highlightthickness=0)
        ttk.Label(self, text="Mesh topology").pack(anchor="w"); self.canvas.pack(fill="both", expand=True)
        self.width, self.height = width, height; self._pos = {}; self._running = True; self.after(500, self.refresh)
    def stop(self): self._running = False
    def refresh(self):
        if not self._running: return
        snap = self.get_mesh_snapshot(); self._draw(snap.get("nodes", {}), snap.get("edges", [])); self.after(1000, self.refresh)
    def _draw(self, nodes, edges):
        self.canvas.delete("all"); urls = list(nodes.keys()); n = max(1, len(urls)); margin = 40
        def init_pos(i):
            ang = (2 * math.pi * i) / n; r = min(self.width, self.height) / 2 - margin; cx, cy = self.width / 2, self.height / 2
            return cx + r * math.cos(ang), cy + r * math.sin(ang)
        for i, u in enumerate(urls):
            self._pos[u] = self._pos.get(u, init_pos(i)); x, y = self._pos[u]
            self._pos[u] = (max(margin, min(self.width - margin, x + random.uniform(-0.8, 0.8))),
                            max(margin, min(self.height - margin, y + random.uniform(-0.8, 0.8))))
        for src, dst in edges:
            if src in self._pos and dst in self._pos:
                x1, y1 = self._pos[src]; x2, y2 = self._pos[dst]
                self.canvas.create_line(x1, y1, x2, y2, fill="#3b4252", width=1)
        for u, meta in nodes.items():
            x, y = self._pos.get(u, init_pos(0)); state = meta.get("state", "discovered"); risk = int(meta.get("risk", 0))
            fill = "#81a1c1" if state == "discovered" else ("#88c0d0" if state == "built" else "#a3be8c")
            outline = "#bf616a" if risk >= 60 else ("#d08770" if risk >= 30 else "#5e81ac"); r = 6
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=2)
            label = meta.get("label") or (u.split("//")[-1][:22]); self.canvas.create_text(x + 10, y, text=label, anchor="w", fill="#e5e9f0", font=("Segoe UI", 8))

# ============================================================
# Evolution loop (wired to StatusBus)
# ============================================================
def select_parents(archive: dict, k: int, seed_noise: float) -> list:
    top_ids = [entry["id"] for entry in archive["top"][-50:]]; pool = []
    for _ in range(k):
        if top_ids and random.random() < 0.7: pool.append(archive["ideas"][random.choice(top_ids)]["idea"])
        else: pool.append(compose_idea(seed_noise))
    return pool

def evolve(cfg: dict, archive_ref: dict, feeds: FeedRegistry, sysmon: SystemMetrics,
           status: StatusBus, mesh: BorgMesh, scanner_in: queue.Queue, ops_q: queue.Queue):
    def _score_of(ev_or_dict: dict) -> float: return float(ev_or_dict.get("final_score", ev_or_dict.get("objective", 0.0)))
    paths = node_paths(cfg)
    with _archive_lock: archive = archive_ref["ref"]
    archive_texts = [archive["ideas"][i]["text"] for i in archive["ideas"]]
    no_improve = 0
    best_obj = max([_score_of(archive["ideas"][i].get("eval", archive["ideas"][i])) for i in archive["ideas"]] + [0.0])
    throttle = Throttle(cfg); ensure_strategy_file(cfg); strategy_mod = load_strategy(); last_strategy_mut = time.time(); perf_window = []
    chosen_dir = choose_storage_dir(cfg); _set_storage_dir(chosen_dir)
    with _archive_lock: archive["stats"]["storage_dir"] = chosen_dir
    status.emit("seed", {"msg": "evolution started"})
    while not _shutdown_event.is_set():
        try:
            for step in range(cfg.get("iterations_per_tick", 10)):
                with _archive_lock: archive = archive_ref["ref"]
                chosen_dir = choose_storage_dir(cfg); _set_storage_dir(chosen_dir)
                with _archive_lock: archive["stats"]["storage_dir"] = chosen_dir
                entropy_feed = feed_entropy(feeds); entropy_sys = system_entropy(sysmon.snapshot())
                with _archive_lock: archive["stats"]["feed_entropy"] = entropy_feed; archive["stats"]["sys_entropy"] = entropy_sys
                parents = select_parents(archive, k=max(5, cfg["batch_size"]//4), seed_noise=cfg["seed_noise"])
                candidates = []; active_names = []
                for _ in range(cfg["batch_size"]):
                    base = random.choice(parents)
                    idea = mutate_idea_safe(base) if random.random() < 0.8 else clamp_params(compose_idea(cfg["seed_noise"]))
                    idea["params"]["tolerance"] = round(max(0.003, idea["params"]["tolerance"] * (1.0 - 0.15*entropy_feed - 0.15*entropy_sys)), 4)
                    idea["params"]["mutation_rate"] = round(min(0.8, idea["params"]["mutation_rate"] * (1.0 + 0.25*entropy_feed + 0.25*entropy_sys)), 4)
                    pname = autonomous_project_name(idea, cfg["node_name"]); idea["project_name"] = pname
                    perf = {"avg_objective": (sum(perf_window)/len(perf_window)) if perf_window else 0.0,
                            "new_ideas": len(archive_texts), "completion_rate": (archive["stats"]["completed_projects"]/max(1, archive["stats"]["iterations"])),
                            "feed_entropy": entropy_feed, "sys_entropy": entropy_sys}
                    weights_override = None
                    if strategy_mod and hasattr(strategy_mod, "adjust_weights"):
                        try: weights_override = strategy_mod.adjust_weights(perf)
                        except Exception: logging.exception("Strategy adjust failed."); weights_override = None
                    evals = evaluate_idea_full(idea, archive_texts, cfg, perf=perf, weights_override=weights_override)
                    candidates.append((idea, evals, evals["objective"])); active_names.append(pname); archive_texts.append(evals["text"])
                    ev = {"url": f"https://mesh.local/{pname}", "snippet": idea["title"],
                          "links": [f"https://mesh.local/{pname}/l{i}" for i in range(random.randint(3, 12))]}
                    try: scanner_in.put_nowait(ev); ops_q.put(("build", ev["url"]))
                    except Exception: pass
                with _archive_lock:
                    archive["stats"]["active_projects"] = len(candidates); archive["stats"]["active_project_names"] = active_names
                candidates.sort(key=lambda x: x[2], reverse=True); elites = candidates[:max(5, cfg["batch_size"]//4)]
                improved = False; completed = 0; finished_names = []
                for idea, evals, obj in elites:
                    text = evals["text"]; hid = idea_hash(text)
                    with _archive_lock:
                        if hid not in archive["ideas"]:
                            out_path, line_count, used_dir = generate_large_project_file(idea, evals, cfg, hid)
                            archive["ideas"][hid] = {"id": hid, "idea": ensure_idea_keys(idea), "text": text, "eval": evals,
                                                     "project_name": idea.get("project_name","unnamed"), "code_path": out_path,
                                                     "code_lines": line_count, "storage_dir": used_dir}
                            archive["top"].append({"id": hid, "objective": obj, "novelty": evals["novelty"], "utility": evals["utility"]})
                            archive_texts.append(text); improved = improved or (obj > best_obj); best_obj = max(best_obj, obj)
                            completed += 1; finished_names.append(idea.get("project_name","unnamed")); perf_window.append(obj)
                            if len(perf_window) > cfg.get("strategy_selection_window", 500):
                                perf_window = perf_window[-cfg.get("strategy_selection_window", 500):]
                if completed > 0:
                    with _archive_lock:
                        stats = archive["stats"]; stats["completed_projects"] = stats.get("completed_projects", 0) + completed
                        stats["finished_project_names"] = stats.get("finished_project_names", []) + finished_names
                    status.emit("completed", {"names": finished_names})
                if cfg.get("strategy_enable", True) and (time.time() - last_strategy_mut) > cfg.get("strategy_mutate_interval", 120.0):
                    mutate_strategy(cfg); importlib.invalidate_caches()
                    try:
                        if "strategies" in sys.modules: del sys.modules["strategies"]
                        strategy_mod = importlib.import_module("strategies"); logging.info("Strategy module hot-reloaded.")
                    except Exception: logging.exception("Failed to hot-reload strategies."); strategy_mod = None
                    last_strategy_mut = time.time()
                with _archive_lock:
                    archive["stats"]["iterations"] += 1
                    archive["history"].append({"step": archive["stats"]["iterations"], "best_objective": round(best_obj, 6),
                                               "archive_size": len(archive["ideas"]),
                                               "recent_elites": [e[0].get("project_name", e[0]["title"]) for e in elites],
                                               "node": cfg["node_name"], "active_projects": archive["stats"]["active_projects"],
                                               "active_project_names": archive["stats"]["active_project_names"],
                                               "completed_projects": archive["stats"]["completed_projects"],
                                               "finished_project_names": archive["stats"].get("finished_project_names", []),
                                               "feed_entropy": archive["stats"].get("feed_entropy", 0.0),
                                               "sys_entropy": archive["stats"].get("sys_entropy", 0.0),
                                               "storage_dir": archive["stats"].get("storage_dir", get_storage_dir())})
                status.emit("tick", {"active": archive["stats"]["active_projects"], "finished": archive["stats"]["completed_projects"], "best": best_obj})
                if not improved: no_improve += 1
                else: no_improve = 0
                if no_improve > cfg.get("patience", 50):
                    chaos_names = []; chaos_batch_size = cfg.get("chaos_batch_size", 10)
                    for _ in range(chaos_batch_size):
                        idea = compose_idea(seed_noise=0.2 + 0.2*entropy_sys)
                        idea["params"]["mutation_rate"] = round(min(0.8, idea["params"]["mutation_rate"] * (1.0 + 0.4*entropy_feed + 0.4*entropy_sys)), 4)
                        idea["project_name"] = autonomous_project_name(idea, cfg["node_name"])
                        perf = {"avg_objective": (sum(perf_window)/len(perf_window)) if perf_window else 0.0,
                                "new_ideas": len(archive_texts), "completion_rate": (archive["stats"]["completed_projects"]/max(1, archive["stats"]["iterations"])),
                                "feed_entropy": entropy_feed, "sys_entropy": entropy_sys}
                        evals = evaluate_idea_full(idea, archive_texts, cfg, perf=perf, weights_override=None); hid = idea_hash(evals["text"])
                        with _archive_lock:
                            if hid not in archive["ideas"]:
                                out_path, line_count, used_dir = generate_large_project_file(idea, evals, cfg, hid)
                                archive["ideas"][hid] = {"id": hid, "idea": ensure_idea_keys(idea), "text": evals["text"], "eval": evals,
                                                         "project_name": idea["project_name"], "code_path": out_path,
                                                         "code_lines": line_count, "storage_dir": used_dir}
                                archive["top"].append({"id": hid, "objective": evals["objective"], "novelty": evals["novelty"], "utility": evals["utility"]})
                                archive_texts.append(evals["text"]); chaos_names.append(idea["project_name"])
                    if chaos_names: status.emit("chaos", {"names": chaos_names})
                    enforce_archive_limits(archive, cfg); no_improve = 0
                throttle.adjust(); throttle.wait()
            with _archive_lock: save_archive(paths, archive); archive_ref["ref"] = archive
            status.emit("save", {"size": len(archive["ideas"])})
        except Exception:
            logging.exception("Evolution loop error; continuing with backoff.")
            backoff = cfg.get("backoff_initial", 0.5)
            while backoff < cfg.get("backoff_max", 5.0) and not _shutdown_event.is_set():
                time.sleep(backoff); backoff = min(cfg.get("backoff_max", 5.0), backoff * 2)
    with _archive_lock: save_archive(paths, archive)
    status.emit("stop", {"size": len(archive["ideas"])})
    logging.info("Evolution stopped. Final archive size: %d", len(archive["ideas"]))

# ============================================================
# GUI (Notebook + storage picker)
# ============================================================
class AdvancedGUI(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__(); self.title("Mythic Swarm — Live Dashboard"); self.geometry("1160x760")
        self.cfg = cfg; self.status = StatusBus(); self.mesh = BorgMesh()
        self.paths = node_paths(cfg); self.archive_ref = {"ref": load_archive(self.paths)}
        self.feeds = FeedRegistry(); self.sysmon = SystemMetrics(); self.sysmon.start()
        self.scanner_in, self.ops_q = queue.Queue(), queue.Queue()
        self.scanner = threading.Thread(target=self._scanner_loop, daemon=True)
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.enforcer = threading.Thread(target=self._enforcer_loop, daemon=True)
        self.evolver = threading.Thread(target=self._evolver_loop, daemon=True)
        self._build_ui(); self._start_threads(); self._poll_status_bus()
        # Emit initializing for LLM indicator baseline
        self.status.emit("llm_connected", {"backend": "initializing"})
    def _build_ui(self):
        self.notebook = ttk.Notebook(self); self.notebook.pack(fill="both", expand=True)
        self.tab_overview = ttk.Frame(self.notebook); self.notebook.add(self.tab_overview, text="Overview")
        topbar = ttk.Frame(self.tab_overview); topbar.pack(fill="x", padx=10, pady=10)
        self.lbl_summary = ttk.Label(topbar, text="Starting...", anchor="w"); self.lbl_summary.pack(side="left", padx=5)
        ttk.Button(topbar, text="Pick storage dir", command=self._pick_storage_dir).pack(side="left", padx=8)
        self.lbl_storage = ttk.Label(topbar, text=f"Storage: {get_storage_dir()}", anchor="w"); self.lbl_storage.pack(side="left", padx=8)
        # LLM indicator label
        self.lbl_llm = ttk.Label(topbar, text="LLM: not connected", foreground="red")
        self.lbl_llm.pack(side="right", padx=8)
        self.tab_active = ttk.Frame(self.notebook); self.notebook.add(self.tab_active, text="Active")
        self.active_list = tk.Listbox(self.tab_active); self.active_list.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_finished = ttk.Frame(self.notebook); self.notebook.add(self.tab_finished, text="Finished")
        self.finished_list = tk.Listbox(self.tab_finished); self.finished_list.pack(fill="both", expand=True, padx=10, pady=10)
        btns = ttk.Frame(self.tab_finished); btns.pack(fill="x")
        ttk.Button(btns, text="Open code", command=self._open_selected_code).pack(side="left", padx=8, pady=6)
        self.tab_metrics = ttk.Frame(self.notebook); self.notebook.add(self.tab_metrics, text="Metrics")
        self.txt_metrics = tk.Text(self.tab_metrics, height=12); self.txt_metrics.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_mesh = ttk.Frame(self.notebook); self.notebook.add(self.tab_mesh, text="Mesh")
        self.graph = MeshGraphPanel(self.tab_mesh, get_mesh_snapshot=lambda: self.mesh.snapshot())
        self.graph.pack(fill="both", expand=True, padx=10, pady=10)
    def _start_threads(self): self.scanner.start(); self.worker.start(); self.enforcer.start(); self.evolver.start()
    def _scanner_loop(self):
        while not _shutdown_event.is_set():
            try: ev = self.scanner_in.get(timeout=1.0)
            except queue.Empty: continue
            try: url = ev["url"]; snippet = ev["snippet"]; links = ev["links"]; self.mesh.discover(url, snippet, links); self.ops_q.put(("build", url))
            except Exception: pass
            time.sleep(random.uniform(0.2, 0.6))
    def _worker_loop(self):
        while not _shutdown_event.is_set():
            try: op, url = self.ops_q.get(timeout=1.0)
            except queue.Empty: continue
            if op == "build":
                if self.mesh.build(url): self.ops_q.put(("enforce", url))
            elif op == "enforce": self.mesh.enforce(url)
            time.sleep(random.uniform(0.2, 0.5))
    def _enforcer_loop(self):
        while not _shutdown_event.is_set():
            for url, meta in list(self.mesh.nodes.items()):
                if meta["state"] in ("built", "enforced") and random.random() < 0.15: self.mesh.enforce(url)
            time.sleep(1.2)
    def _evolver_loop(self):
        evolve(self.cfg, self.archive_ref, self.feeds, self.sysmon, self.status, self.mesh, self.scanner_in, self.ops_q)
    def _poll_status_bus(self):
        evt = self.status.get_nowait()
        if evt:
            event = evt["event"]; payload = evt["payload"] or {}; arch = self.archive_ref["ref"]; stats = arch["stats"]
            summary = f"Step {stats['iterations']} | Active {stats['active_projects']} | Finished {stats['completed_projects']} | Best {payload.get('best', 0):.3f}"
            self.lbl_summary.config(text=summary); self.lbl_storage.config(text=f"Storage: {stats.get('storage_dir', get_storage_dir())}")
            self.active_list.delete(0, tk.END); [self.active_list.insert(tk.END, n) for n in stats.get("active_project_names", [])]
            self.finished_list.delete(0, tk.END); [self.finished_list.insert(tk.END, n) for n in stats.get("finished_project_names", [])]
            sysm = self.sysmon.snapshot()
            metrics_txt = textwrap.dedent(f"""
                CPU: {sysm.get('cpu', 0.0):.1f} %
                Procs: {sysm.get('procs', 0)} | Conns: {sysm.get('conns', 0)}
                Disk [R/W]: {sysm.get('disk_read', 0)} / {sysm.get('disk_write', 0)}
                Net [S/R]: {sysm.get('net_sent', 0)} / {sysm.get('net_recv', 0)}
                Feed entropy: {stats.get('feed_entropy', 0.0):.3f}
                Sys entropy:  {stats.get('sys_entropy', 0.0):.3f}
                Storage dir:  {stats.get('storage_dir', get_storage_dir())}
            """).strip()
            self.txt_metrics.delete("1.0", tk.END); self.txt_metrics.insert("1.0", metrics_txt)
            if event == "completed": self.txt_metrics.insert(tk.END, f"\nCompleted: {', '.join(payload.get('names', []))}")
            elif event == "chaos": self.txt_metrics.insert(tk.END, f"\nChaos injected: {', '.join(payload.get('names', []))}")
            elif event == "save": self.txt_metrics.insert(tk.END, f"\nArchive saved: {payload.get('size', 0)} ideas")
            # LLM indicator events
            if event == "llm_connected":
                backend = payload.get("backend", "Unknown")
                self.lbl_llm.config(text=f"LLM: {backend}", foreground="green")
            elif event == "llm_error":
                self.lbl_llm.config(text="LLM: error", foreground="orange")
            elif event == "llm_tool":
                tool = payload.get("tool", "")
                self.lbl_llm.config(text=f"LLM: active ({tool})", foreground="blue")
        self.after(250, self._poll_status_bus)
    def _open_selected_code(self):
        sel = self.finished_list.curselection()
        if not sel: return
        name = self.finished_list.get(sel[0]); storage_dir = get_storage_dir()
        try:
            files = [f for f in os.listdir(storage_dir) if f.startswith(name)]
            if not files: messagebox.showinfo("Code", "No code file found yet."); return
            fpath = os.path.join(storage_dir, files[0]); content = open(fpath, "r", encoding="utf-8").read()
        except Exception:
            messagebox.showinfo("Code", "Failed to open code file."); return
        win = tk.Toplevel(self); win.title(f"Code — {name}")
        txt = tk.Text(win, wrap="none"); txt.pack(fill="both", expand=True); txt.insert("1.0", content)
    def _pick_storage_dir(self):
        chosen = filedialog.askdirectory(title="Select storage directory")
        if chosen:
            self.cfg["manual_storage_dir"] = chosen; _set_storage_dir(chosen)
            with _archive_lock: arch = self.archive_ref["ref"]; arch["stats"]["storage_dir"] = chosen
            self.lbl_storage.config(text=f"Storage: {chosen}")
    def destroy(self):
        try: self.graph.stop()
        except Exception: pass
        super().destroy(); _shutdown_event.set()

# ============================================================
# LLM Bridge: auto-detect client + function-calling adapter + indicator events
# ============================================================
LLM_TOOLS_SCHEMA = [
    {"name": "generate_project", "description": "Generate a large swarm project scaffold",
     "parameters": {"type": "object", "properties": {"min_project_lines": {"type": "integer"}, "seed_noise": {"type": "number"}}}},
    {"name": "get_status", "description": "Get current swarm evolution status",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "mutate_idea", "description": "Mutate a provided idea JSON safely",
     "parameters": {"type": "object", "properties": {"idea": {"type": "object"}}}},
]

class LLMBridge:
    def __init__(self, cfg: dict, archive_ref: dict, status_bus: StatusBus):
        self.cfg = cfg; self.archive_ref = archive_ref; self.status = status_bus
    def generate_project(self, params: dict = None) -> dict:
        local_cfg = dict(self.cfg)
        if params: local_cfg.update(params)
        idea = compose_idea(local_cfg.get("seed_noise", 0.15))
        evals = evaluate_idea_full(idea, [], local_cfg)
        path, lines, dir_used = generate_large_project_file(idea, evals, local_cfg, evals["hash"])
        self.status.emit("llm_tool", {"tool": "generate_project"})
        return {"project_name": idea.get("project_name"), "file": path, "lines": lines, "score": round(evals["final_score"], 4)}
    def get_status(self) -> dict:
        stats = self.archive_ref["ref"]["stats"]
        self.status.emit("llm_tool", {"tool": "get_status"})
        return {"iterations": stats["iterations"], "active": stats["active_projects"], "finished": stats["completed_projects"],
                "storage_dir": stats.get("storage_dir", get_storage_dir())}
    def mutate_idea(self, base_idea: dict) -> dict:
        self.status.emit("llm_tool", {"tool": "mutate_idea"})
        m = mutate_idea_safe(base_idea); return ensure_idea_keys(m)

def handle_llm_tool_call(bridge: LLMBridge, tool_name: str, arguments: dict, status_bus: StatusBus) -> dict:
    try:
        if tool_name == "generate_project": return bridge.generate_project(arguments or {})
        if tool_name == "get_status": return bridge.get_status()
        if tool_name == "mutate_idea": return bridge.mutate_idea(arguments.get("idea", compose_idea()))
        return {"error": f"Unknown tool {tool_name}"}
    except Exception as e:
        status_bus.emit("llm_error", {})
        return {"error": str(e), "traceback": traceback.format_exc()}

class AutoLLMClient:
    """
    Auto-detects an available LLM backend (OpenAI, Anthropic, or stub)
    and emits LLM indicator events to StatusBus.
    """
    def __init__(self, status_bus: StatusBus):
        self.backend = None
        self.client = None
        try:
            if importlib.util.find_spec("openai") and os.getenv("OPENAI_API_KEY"):
                import openai
                self.client = openai
                self.backend = "OpenAI"
                status_bus.emit("llm_connected", {"backend": "OpenAI"})
            elif importlib.util.find_spec("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
                import anthropic
                self.client = anthropic
                self.backend = "Anthropic"
                status_bus.emit("llm_connected", {"backend": "Anthropic"})
            else:
                self.backend = "Stub"
                status_bus.emit("llm_connected", {"backend": "Stub"})
        except Exception:
            status_bus.emit("llm_error", {})

    def chat(self, messages, tools=None, model=None):
        # Stub path: simulate desired tool call prompts
        if self.backend == "Stub":
            user_texts = [m["content"] for m in messages if m.get("role") == "user"]
            wants_generate = any("generate" in t.lower() for t in user_texts)
            wants_status = any("status" in t.lower() for t in user_texts)
            faux = {"choices": [{"message": {"tool_calls": []}}]}
            if wants_generate and tools and any(t["name"] == "generate_project" for t in tools):
                faux["choices"][0]["message"]["tool_calls"] = [{
                    "id": "tool_1",
                    "name": "generate_project",
                    "arguments": {"min_project_lines": 1000, "seed_noise": 0.15}
                }]
            elif wants_status and tools and any(t["name"] == "get_status" for t in tools):
                faux["choices"][0]["message"]["tool_calls"] = [{"id": "tool_2","name": "get_status","arguments": {}}]
            else:
                faux["choices"][0]["message"]["content"] = "No tools requested. Ask me to generate or get status."
            return faux
        # If real backend, you'd route actual API calls here; omitted for single-file demo.
        return {"choices": [{"message": {"content": "Real backend path not implemented in this demo."}}]}

def run_llm_orchestrator(bridge: LLMBridge, tools_schema, cfg: dict, status_bus: StatusBus):
    """
    Background thread: detect LLM and simulate orchestration prompts; handle tool calls.
    """
    client = AutoLLMClient(status_bus)
    messages = [{"role": "system", "content": "You can call tools to control the Mythic Swarm Engine."},
                {"role": "user", "content": "Generate a swarm project and report status."}]
    while not _shutdown_event.is_set():
        try:
            resp = client.chat(messages, tools_schema, model=cfg.get("llm_model"))
            # Normalize response for stub
            tool_calls = []
            if isinstance(resp, dict) and "choices" in resp:
                msg = resp["choices"][0].get("message", {})
                tool_calls = msg.get("tool_calls", [])
                content = msg.get("content")
            else:
                content = str(resp)
            if tool_calls:
                for call in tool_calls:
                    name = call.get("name")
                    args = call.get("arguments", {})
                    result = handle_llm_tool_call(bridge, name, args, status_bus)
                    messages.append({"role": "assistant", "content": f"Tool {name} result: {json.dumps(result)[:2000]}"})
            else:
                messages.append({"role": "assistant", "content": str(content)})
                messages.append({"role": "user", "content": "Get status"})
            time.sleep(4.0)
        except Exception:
            status_bus.emit("llm_error", {})
            logging.exception("LLM orchestrator loop error")
            time.sleep(5.0)

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = {
        # Scoring weights
        "novelty_weight": 0.25, "utility_weight": 0.35, "impact_weight": 0.25, "curiosity_weight": 0.15,
        # Strategy & archive
        "strategy_mutation_scale": 0.02, "max_archive_size": 1000,
        "strategy_enable": True, "strategy_mutate_interval": 120.0, "strategy_selection_window": 500,
        # CPU & throttling (cap at 50%)
        "sleep_min_seconds": 0.025, "sleep_max_seconds": 0.8, "cpu_target_util": 0.5,
        # Evolution
        "iterations_per_tick": 10, "batch_size": 20, "seed_noise": 0.15,
        "node_name": "local", "patience": 50, "save_interval_steps": 20,
        "backoff_initial": 0.5, "backoff_max": 5.0, "chaos_batch_size": 10,
        # Storage config
        "storage_dir": os.path.join(os.getcwd(), "projects"),
        "alternate_storage_dir": os.path.join(os.getcwd(), "projects_alt"),
        "manual_storage_dir": None,
        "storage_daily_subdir": True,
        "storage_switch_threshold_pct": 70.0,
        # Tone & project size
        "tone_enable": False, "min_project_lines": 1000,
        # LLM
        "llm_model": None
    }
    ensure_strategy_file(cfg)

    # GUI app (starts evolution in background)
    app = AdvancedGUI(cfg)

    # LLM bridge bound to the app's archive/status
    bridge = LLMBridge(cfg, app.archive_ref, app.status)

    # Start LLM orchestrator in background (auto-detects backend and calls tools)
    llm_thread = threading.Thread(target=run_llm_orchestrator, args=(bridge, LLM_TOOLS_SCHEMA, cfg, app.status), daemon=True)
    llm_thread.start()

    # Simulate one tool call at startup to reflect activity
    try:
        result = handle_llm_tool_call(bridge, "generate_project", {"min_project_lines": 1200, "seed_noise": 0.2}, app.status)
        logging.info(f"[LLM] Generated project: {result}")
    except Exception:
        logging.exception("LLM bridge demo failed.")

    print("Starting GUI...")
    app.mainloop()

