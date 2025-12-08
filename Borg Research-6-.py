#!/usr/bin/env python3
# imagination_colony_tk_live_ultra.py
# Autonomous imagination engine with:
# - Tkinter GUI (active + finished projects, code preview, open in system editor)
# - Mute/Unmute sound button
# - Live internet feeds (HTTPS, configurable, rate-limited)
# - System metrics ingestion (CPU, memory, disk, net I/O, processes, connections)
# - Self-rewriting strategies module (hot-reload, mutation over time)
# - Self-named projects (mythic+semantic+hash)
# - Impact scoring, adaptive CPU target (~5% busy, up to 40% idle)
# - Robust persistence
# - Ultra project generator (streaming to disk): 50 up to 4,000,000+ lines of Python per project

import os, sys, time, json, random, zlib, math, hashlib, signal, threading, logging, socket, subprocess, shutil, struct
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Tuple, Optional

# ---------------------------
# Optional auto-deps
# ---------------------------
import importlib
def ensure_lib(lib: str) -> Optional[object]:
    try:
        return importlib.import_module(lib)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            return importlib.import_module(lib)
        except Exception:
            return None

psutil = ensure_lib("psutil")
numpy = ensure_lib("numpy")
requests = ensure_lib("requests")

# ---------------------------
# Paths and defaults
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "imagination_config.json")
STRATEGY_PATH = os.path.join(BASE_DIR, "strategies.py")

DEFAULT_CONFIG = {
    "iterations_per_tick": 80,
    "batch_size": 16,
    "patience": 25,
    "save_interval_steps": 8,
    "max_archive_size": 200_000,

    "cpu_target_util": 0.05,
    "sleep_min_seconds": 0.05,
    "sleep_max_seconds": 2.0,

    "backoff_initial": 1.0,
    "backoff_max": 60.0,

    # Objective weights
    "novelty_weight": 0.4,
    "utility_weight": 0.3,
    "impact_weight": 0.2,
    "curiosity_weight": 0.1,

    "seed_noise": 0.15,

    # Logging
    "log_dir": os.path.join(BASE_DIR, "logs"),
    "max_log_bytes": 8_000_000,
    "log_backup_count": 4,

    # Live feeds (HTTPS only)
    "feeds_enable": True,
    "feeds_poll_interval": 30.0,
    "feeds": [
        {"name": "coindesk_btc", "url": "https://api.coindesk.com/v1/bpi/currentprice.json", "timeout": 6.0},
        {"name": "worldtime", "url": "http://worldtimeapi.org/api/ip", "timeout": 6.0},
    ],
    "feeds_max_records": 300,

    # Strategy evolution
    "strategy_enable": True,
    "strategy_mutate_interval": 120.0,
    "strategy_selection_window": 300,
    "strategy_mutation_scale": 0.05,

    # Audible signals (mutable via GUI)
    "audible_signals": True,

    # Project generation target lines (supports 50 up to 4,000,000+)
    "project_target_min_lines": 50,
    "project_target_max_lines": 4_000_000,

    # Code output
    "projects_dir": os.path.join(BASE_DIR, "projects"),
    "preview_max_lines": 1200,

    # GUI
    "gui_refresh_ms": 2000
}

# ---------------------------
# Global state
# ---------------------------
_shutdown_event = threading.Event()
_archive_lock = threading.Lock()

def setup_logging(cfg: Dict):
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_path = os.path.join(cfg["log_dir"], f"{cfg.get('node_name','node')}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rfh = RotatingFileHandler(log_path, maxBytes=cfg["max_log_bytes"], backupCount=cfg["log_backup_count"])
    rfh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.handlers = [rfh, ch]

def _handle_signal(signum, frame):
    logging.warning(f"Received signal {signum}; shutting down...")
    _shutdown_event.set()

for sig in (signal.SIGINT, signal.SIGTERM):
    try: signal.signal(sig, _handle_signal)
    except Exception: pass

# ---------------------------
# Config
# ---------------------------
def load_config() -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                if isinstance(user_cfg, dict):
                    cfg.update(user_cfg)
        except Exception:
            pass
    host = socket.gethostname()
    cfg.setdefault("node_name", f"node-{host}-{random.randint(1000,9999)}")
    cfg.setdefault("data_dir", os.path.join(BASE_DIR, cfg["node_name"]))
    os.makedirs(cfg["data_dir"], exist_ok=True)
    os.makedirs(cfg.get("projects_dir", os.path.join(BASE_DIR, "projects")), exist_ok=True)
    return cfg

def save_config(cfg: Dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        logging.exception("Failed to save config.")

def node_paths(cfg: Dict) -> Dict:
    return {"archive_path": os.path.join(cfg["data_dir"], "imagination_archive.json")}

# ---------------------------
# Idea space & naming
# ---------------------------
PRIMITIVES = {
    "domains": ["gravity","magnetism","plasma","atmosphere","materials","orbits","energy","optics","computation","biodesign"],
    "mechanisms": ["field_weaving","resonance","gradient_control","phase_modulation","turbulence_guidance","lattice_tuning","feedback_adaptation","nonlinear_coupling","quantized_transport","topology_shaping"],
    "architectures": ["ring_habitat","torsion_drive","beam_array","mesh_network","queen_board_swarm","shield_skin","saddle_coil","shell_stack","spiral_tower","orbital_kite"],
    "intents": ["stability","mobility","harvesting","shielding","navigation","cooling","amplification","damping","sensing","repair"],
    "constraints": ["low_mass","low_power","fail_safe","zero_emissions","self_healing","rapid_deploy","modular","scalable","silent","transparent"]
}
MYTHIC = ["Aegis","Arcadia","Atlas","Aurora","Borealis","Cerberus","Chimera","Chronos","Daedalus","Elysium","Gaea","Helios","Hermes","Hyperion","Icarus","Leviathan","Lyra","Nereid","Nyx","Orion","Pandora","Phoenix","Prometheus","Rhea","Selene","Talos","Titan","Valkyrie","Zephyr"]
VERBS = ["weaver","foundry","engine","loom","array","forge","matrix","mirror","harbor","beacon","horizon","garden","crucible","compass","lattice","spiral","sphere","saddle","shield","tower","kite","swarm","mesh","glider","bridge"]
QUALS = ["prime","delta","x","ghost","weave","nova","flux","arc","pulse","shade","aether","echo"]

def autonomous_project_name(idea: Dict, node: str) -> str:
    dom = idea.get("domain","unknown")
    mech = idea.get("mechanism","unknown")
    arch = idea.get("architecture","unknown")
    intent = idea.get("intent","unknown")
    base = f"{dom}-{mech}-{arch}-{intent}"
    myth = random.choice(MYTHIC)
    noun = random.choice(VERBS)
    qual = random.choice(QUALS)
    short = hashlib.sha1((base + node + str(time.time()) + str(random.random())).encode("utf-8")).hexdigest()[:6]
    return f"{myth}-{noun}-{qual}-{short}"

# ---------------------------
# Robust idea handling
# ---------------------------
def compose_idea(seed_noise: float = 0.15) -> Dict:
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

def ensure_idea_keys(idea: Dict) -> Dict:
    base = compose_idea(seed_noise=0.0)
    out = dict(base)
    out.update(idea or {})
    if "constraints" not in out or not isinstance(out["constraints"], list) or len(out["constraints"]) == 0:
        out["constraints"] = base["constraints"]
    if "params" not in out or not isinstance(out["params"], dict):
        out["params"] = base["params"]
    for p in ["scale","power_budget","tolerance","mutation_rate"]:
        if p not in out["params"]:
            out["params"][p] = base["params"][p]
    out["title"] = f"{out['domain']}-{out['mechanism']}-{out['architecture']}-{out['intent']}"
    return out

def mutate_idea(idea: Dict) -> Dict:
    m = ensure_idea_keys(idea)
    slot = random.choice(["domain","mechanism","architecture","intent"])
    if slot == "domain":
        m["domain"] = random.choice(PRIMITIVES["domains"])
    elif slot == "mechanism":
        m["mechanism"] = random.choice(PRIMITIVES["mechanisms"])
    elif slot == "architecture":
        m["architecture"] = random.choice(PRIMITIVES["architectures"])
    else:
        m["intent"] = random.choice(PRIMITIVES["intents"])
    for p in ["scale","power_budget","tolerance","mutation_rate"]:
        m["params"][p] = round(float(m["params"].get(p, 1.0)) * random.uniform(0.7, 1.3), 4)
    if random.random() < 0.5:
        m["constraints"] = list(set(m["constraints"] + [random.choice(PRIMITIVES["constraints"])]))
    m["title"] = f"{m['domain']}-{m['mechanism']}-{m['architecture']}-{m['intent']}"
    return m

def idea_to_text(idea: Dict) -> str:
    return json.dumps(ensure_idea_keys(idea), sort_keys=True)

# ---------------------------
# Novelty & evaluation helpers
# ---------------------------
def text_distance(a: str, b: str) -> float:
    ta, tb = set(a.split('"')), set(b.split('"'))
    inter = len(ta & tb); union = len(ta | tb)
    return 1.0 - (inter / union if union else 0.0)

def compressibility_score(s: str) -> float:
    comp = zlib.compress(s.encode("utf-8"))
    return len(comp) / max(1, len(s))

def novelty_score(candidate_text: str, archive_texts: List[str]) -> float:
    if not archive_texts: return 1.0 + compressibility_score(candidate_text)
    if numpy:
        distances = []
        step = max(1, len(archive_texts)//1000)
        for t in archive_texts[::step]:
            distances.append(text_distance(candidate_text, t))
        avg_dist = float(numpy.mean(numpy.array(distances))) if distances else 1.0
    else:
        distances = [text_distance(candidate_text, t) for t in archive_texts]
        avg_dist = sum(distances) / len(distances)
    return avg_dist + 0.5 * compressibility_score(candidate_text)

# ---------------------------
# Simulators
# ---------------------------
class Simulator:
    name: str = "base"
    def score(self, idea: Dict) -> Tuple[float, Dict]:
        raise NotImplementedError

class OrbitalComfortSimulator(Simulator):
    name = "orbital_comfort"
    def score(self, idea: Dict) -> Tuple[float, Dict]:
        p = ensure_idea_keys(idea)["params"]
        radius = max(10.0, p["scale"])
        rpm = max(0.1, min(6.0, 60.0 * math.sqrt(9.81 / radius) / (2*math.pi)))
        g = (4 * math.pi**2 * radius * (rpm/60.0)**2)
        comfort_penalty = abs(g - 9.81) / 9.81
        coriolis_penalty = max(0.0, (rpm - 2.0) / 4.0)
        power_ok = min(1.0, 1.0 / (1.0 + p["power_budget"]/1e5))
        utility = max(0.0, 1.0 - comfort_penalty - coriolis_penalty) * (0.5 + 0.5*power_ok)
        diag = {"radius_m": radius, "rpm": rpm, "g_mps2": g, "comfort_penalty": comfort_penalty,
                "coriolis_penalty": coriolis_penalty, "power_factor": power_ok}
        return utility, diag

class AtmosphereRoutingSimulator(Simulator):
    name = "atmo_energy_routing"
    def score(self, idea: Dict) -> Tuple[float, Dict]:
        idea = ensure_idea_keys(idea)
        p = idea["params"]
        intent_bonus = 0.2 if idea["intent"] in ["shielding","damping","navigation"] else 0.0
        constraint_bonus = 0.1 * len([c for c in idea["constraints"] if c in ["low_power","fail_safe","zero_emissions"]])
        power_penalty = 0.0 if idea["intent"] == "amplification" else min(0.6, math.log10(max(1.0, p["power_budget"])) / 10.0)
        stability = max(0.0, 1.0 - p["tolerance"])
        utility = max(0.0, stability + intent_bonus + constraint_bonus - power_penalty)
        diag = {"stability": stability, "intent_bonus": intent_bonus, "constraint_bonus": constraint_bonus,
                "power_penalty": power_penalty}
        return utility, diag

SIMULATORS: List[Simulator] = [OrbitalComfortSimulator(), AtmosphereRoutingSimulator()]

def reload_simulators(cfg: Dict):
    global SIMULATORS
    SIMULATORS = [OrbitalComfortSimulator(), AtmosphereRoutingSimulator()]
    logging.info("Simulators reloaded.")

def idea_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ---------------------------
# Global impact scoring
# ---------------------------
def impact_score(idea: Dict) -> float:
    idea = ensure_idea_keys(idea)
    cons = set(idea.get("constraints", []))
    score = 0.0
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

# ---------------------------
# Idea evaluation (fixed and enhanced)
# ---------------------------
def evaluate_idea(idea: Dict, archive_texts: List[str], cfg: Dict,
                  perf: Optional[Dict] = None, weights_override: Optional[Dict] = None) -> Dict:
    idea = ensure_idea_keys(idea)
    text = idea_to_text(idea)
    novelty = novelty_score(text, archive_texts)
    results = []
    for sim in SIMULATORS:
        try:
            score, diag = sim.score(idea)
        except Exception:
            logging.exception(f"Simulator {sim.name} failed; continuing.")
            score, diag = 0.0, {"error": "simulator_failure"}
        results.append((sim.name, score, diag))
    utility = sum(s for _, s, _ in results) / len(results) if results else 0.0
    curiosity = novelty * (0.6 + 0.4 * (1.0 - utility))
    impact = impact_score(idea)

    w = {
        "novelty_weight": cfg["novelty_weight"],
        "utility_weight": cfg["utility_weight"],
        "impact_weight": cfg["impact_weight"],
        "curiosity_weight": cfg["curiosity_weight"]
    }
    if weights_override:
        w.update({k: weights_override.get(k, w[k]) for k in w.keys()})

    # Optional: small nudges from performance signals (if provided)
    if perf:
        cr = perf.get("completion_rate", 0.0)
        avg_obj = perf.get("avg_objective", 0.0)
        feed_entropy = perf.get("feed_entropy", 0.0)
        sys_entropy = perf.get("sys_entropy", 0.0)
        w["novelty_weight"] = min(1.4, w["novelty_weight"] + 0.03 * max(0.0, 0.2 - cr) + 0.01*feed_entropy)
        w["curiosity_weight"] = min(1.4, w["curiosity_weight"] + 0.02 * max(0.0, 0.2 - cr) + 0.01*sys_entropy)
        if avg_obj > 1.0:
            w["utility_weight"] = min(1.4, w["utility_weight"] + 0.02)
            w["impact_weight"] = min(1.4, w["impact_weight"] + 0.01)

    objective = (w["novelty_weight"] * novelty
                 + w["utility_weight"] * utility
                 + w["impact_weight"] * impact
                 + w["curiosity_weight"] * curiosity)

    return {
        "text": text,
        "novelty": novelty,
        "utility": utility,
        "curiosity": curiosity,
        "impact": impact,
        "objective": objective,
        "sims": results
    }

# ---------------------------
# Strategies
# ---------------------------
def ensure_strategy_file(cfg: Dict):
    if os.path.exists(STRATEGY_PATH):
        return
    baseline = f'''# Auto-generated strategies module
weights = {{
    "novelty_weight": {cfg["novelty_weight"]},
    "utility_weight": {cfg["utility_weight"]},
    "impact_weight": {cfg["impact_weight"]},
    "curiosity_weight": {cfg["curiosity_weight"]}
}}

def adjust_weights(perf):
    w = dict(weights)
    if perf.get("completion_rate", 0) < 0.05:
        w["novelty_weight"] = min(1.2, w["novelty_weight"] + 0.05)
        w["curiosity_weight"] = min(1.2, w["curiosity_weight"] + 0.03)
    if perf.get("avg_objective", 0) > 1.0:
        w["utility_weight"] = min(1.2, w["utility_weight"] + 0.04)
        w["impact_weight"] = min(1.2, w["impact_weight"] + 0.02)
    # Data-driven nudges
    w["novelty_weight"] = min(1.3, w["novelty_weight"] + 0.01*perf.get("feed_entropy",0))
    w["curiosity_weight"] = min(1.3, w["curiosity_weight"] + 0.01*perf.get("sys_entropy",0))
    return w
'''
    with open(STRATEGY_PATH, "w", encoding="utf-8") as f:
        f.write(baseline)

def load_strategy():
    try:
        return importlib.import_module("strategies")
    except Exception:
        return None

def mutate_strategy(cfg: Dict):
    try:
        with open(STRATEGY_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        def mutate_line(line: str) -> str:
            return line.replace("0.05", f"{0.05+random.uniform(-cfg['strategy_mutation_scale'], cfg['strategy_mutation_scale']):.3f}")\
                       .replace("0.03", f"{0.03+random.uniform(-cfg['strategy_mutation_scale'], cfg['strategy_mutation_scale']):.3f}")\
                       .replace("0.04", f"{0.04+random.uniform(-cfg['strategy_mutation_scale'], cfg['strategy_mutation_scale']):.3f}")\
                       .replace("0.02", f"{0.02+random.uniform(-cfg['strategy_mutation_scale'], cfg['strategy_mutation_scale']):.3f}")
        mutated = "\n".join(mutate_line(ln) for ln in text.splitlines())
        with open(STRATEGY_PATH, "w", encoding="utf-8") as f:
            f.write(mutated)
        importlib.invalidate_caches()
        logging.info("Strategy file mutated.")
    except Exception:
        logging.exception("Failed to mutate strategy file.")

# ---------------------------
# Archive
# ---------------------------
def load_archive(paths: Dict) -> Dict:
    ap = paths["archive_path"]
    if os.path.exists(ap):
        try:
            with open(ap, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict): raise ValueError("Archive not dict")
                data.setdefault("ideas", {})
                data.setdefault("top", [])
                data.setdefault("history", [])
                stats = data.setdefault("stats", {})
                stats.setdefault("iterations", 0)
                stats.setdefault("active_projects", 0)
                stats.setdefault("completed_projects", 0)
                stats.setdefault("active_project_names", [])
                stats.setdefault("finished_project_names", [])
                stats.setdefault("feed_entropy", 0.0)
                stats.setdefault("sys_entropy", 0.0)
                return data
        except Exception:
            logging.exception("Archive load failed; starting fresh.")
    return {"ideas": {}, "top": [], "history": [], "stats": {
        "iterations": 0, "active_projects": 0, "completed_projects": 0,
        "active_project_names": [], "finished_project_names": [],
        "feed_entropy": 0.0, "sys_entropy": 0.0
    }}

def save_archive(paths: Dict, archive: Dict):
    ap = paths["archive_path"]
    try:
        tmp = ap + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
        os.replace(tmp, ap)
    except Exception:
        logging.exception("Failed to save archive.")

def enforce_archive_limits(archive: Dict, cfg: Dict):
    if len(archive["ideas"]) > cfg["max_archive_size"]:
        keep = cfg["max_archive_size"]
        ids_sorted = sorted(archive["ideas"].keys(),
                            key=lambda k: archive["ideas"][k]["eval"]["objective"],
                            reverse=True)
        keep_ids = set(ids_sorted[:keep])
        archive["ideas"] = {i: archive["ideas"][i] for i in keep_ids}
        archive["top"] = [t for t in archive["top"] if t["id"] in keep_ids]
        logging.warning("Archive trimmed to max size.")

# ---------------------------
# System metrics ingestion
# ---------------------------
class SystemMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.snapshot_data = {}
    def poll(self):
        if psutil is None:
            return
        try:
            cpu = psutil.cpu_percent(interval=0.05)
            mem = psutil.virtual_memory()
            disk = psutil.disk_io_counters() if hasattr(psutil, "disk_io_counters") else None
            net = psutil.net_io_counters() if hasattr(psutil, "net_io_counters") else None
            procs = len(psutil.pids())
            conns = len(psutil.net_connections(kind="inet")) if hasattr(psutil, "net_connections") else 0
            with self.lock:
                self.snapshot_data = {
                    "cpu": cpu,
                    "mem_used": mem.used if mem else 0,
                    "mem_avail": mem.available if mem else 0,
                    "disk_read": getattr(disk, "read_bytes", 0) if disk else 0,
                    "disk_write": getattr(disk, "write_bytes", 0) if disk else 0,
                    "net_sent": getattr(net, "bytes_sent", 0) if net else 0,
                    "net_recv": getattr(net, "bytes_recv", 0) if net else 0,
                    "procs": procs,
                    "conns": conns,
                }
        except Exception:
            pass
    def start(self, interval: float = 2.0):
        def loop():
            while not _shutdown_event.is_set():
                self.poll()
                time.sleep(interval)
        threading.Thread(target=loop, daemon=True).start()
    def snapshot(self) -> Dict:
        with self.lock:
            return dict(self.snapshot_data)

def system_entropy(sys_metrics: Dict) -> float:
    cpu = sys_metrics.get("cpu", 0.0)/100.0
    procs = sys_metrics.get("procs", 0)
    conns = sys_metrics.get("conns", 0)
    net = (sys_metrics.get("net_sent", 0) + sys_metrics.get("net_recv", 0)) / 1e9
    disk = (sys_metrics.get("disk_read", 0) + sys_metrics.get("disk_write", 0)) / 1e9
    return min(2.0, 0.5*cpu + 0.001*procs + 0.002*conns + 0.1*net + 0.1*disk)

# ---------------------------
# Throttling
# ---------------------------
class Throttle:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.sleep_s = cfg["sleep_min_seconds"]
    def adjust(self):
        if psutil is None:
            if self.cfg["cpu_target_util"] <= 0.05:
                self.sleep_s = min(self.cfg["sleep_max_seconds"], self.sleep_s * 1.25 + 0.05)
            else:
                self.sleep_s = max(self.cfg["sleep_min_seconds"], self.sleep_s * 0.9 - 0.01)
            return
        try:
            cpu = psutil.cpu_percent(interval=0.1) / 100.0
            target = self.cfg["cpu_target_util"]
            if cpu > target:
                self.sleep_s = min(self.cfg["sleep_max_seconds"], self.sleep_s * 1.25 + 0.05)
            else:
                self.sleep_s = max(self.cfg["sleep_min_seconds"], self.sleep_s * 0.9 - 0.01)
        except Exception:
            pass
    def wait(self):
        time.sleep(self.sleep_s)

# ---------------------------
# Audible completion
# ---------------------------
def emit_completion_tone(cfg: Dict):
    if not cfg.get("audible_signals", True):
        return
    try:
        import winsound
        winsound.Beep(880, 180)
    except Exception:
        try:
            sys.stdout.write("\a"); sys.stdout.flush()
        except Exception:
            pass

# ---------------------------
# Ultra project generator (streaming to disk, up to 4M+ lines)
# ---------------------------
def generate_ultra_project_file(idea: Dict, evals: Dict, cfg: Dict, hid: str) -> Tuple[str, int]:
    pname = idea.get("project_name", "unnamed_project")
    domain = idea.get("domain", "unknown")
    mechanism = idea.get("mechanism", "unknown")
    architecture = idea.get("architecture", "unknown")
    intent = idea.get("intent", "unknown")
    params = idea.get("params", {})
    target_min = int(cfg.get("project_target_min_lines", 50))
    target_max = int(cfg.get("project_target_max_lines", 4_000_000))
    # Allow external override via env for extreme runs
    env_max = os.environ.get("PROJECT_MAX_LINES")
    if env_max:
        try:
            target_max = max(target_max, int(env_max))
        except Exception:
            pass
    target_lines = random.randint(target_min, target_max)

    out_dir = cfg.get("projects_dir")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pname}_{hid}.py")

    header = f'''#!/usr/bin/env python3
# Generated autonomous project: {pname}
# Domain: {domain} | Mechanism: {mechanism} | Architecture: {architecture} | Intent: {intent}
# Portable ultra-scale module (target ~{target_lines} lines)

import sys, math, random, json, time
from typing import Dict, List, Tuple

META = {{
  "project": "{pname}",
  "domain": "{domain}",
  "mechanism": "{mechanism}",
  "architecture": "{architecture}",
  "intent": "{intent}"
}}

def seed_params():
    return {json.dumps(params, indent=2)}

def _baseline_transform(x: float) -> float:
    return math.sin(x) * 0.5 + math.cos(x*0.5) * 0.25

def _score_block(a: float, b: float, c: float) -> float:
    return max(0.0, (a/(1.0+b)) * (1.0/(1.0+c)))

class Engine:
    def __init__(self, params: Dict):
        self.params = dict(params)
        self.state = {{"ticks": 0, "accum": 0.0, "log": []}}
    def tick(self, x: float) -> float:
        self.state["ticks"] += 1
        val = _baseline_transform(x) + random.uniform(-0.01, 0.01)
        self.state["accum"] += val
        if self.state["ticks"] % 100 == 0:
            self.state["log"].append(val)
        return val

def simulate(params: Dict, steps: int = 2000) -> Dict:
    eng = Engine(params)
    s, p, t = float(params.get("scale",1.0)), float(params.get("power_budget",1000.0)), float(params.get("tolerance",0.1))
    total = 0.0
    for i in range(steps):
        total += eng.tick(i * 0.001)
    score = _score_block(s, p/1e5, t) + max(0.0, total/steps)
    return {{"score": score, "steps": steps, "state": eng.state}}

def main():
    params = seed_params()
    result = simulate(params, steps=3000)
    print(json.dumps({{"project": META, "result": result}}, indent=2))

if __name__ == "__main__":
    main()
'''

    def gen_util(i: int) -> str:
        return (
f"\n# --- Utility block {i}\n"
f"def util_{i}(x: float, y: float, z: float) -> float:\n"
f"    a = math.fabs(x) + (y*y*0.001) - z*0.0001\n"
f"    b = math.sqrt(1.0 + math.fabs(a))\n"
f"    c = math.sin(a*0.1) + math.cos(b*0.2)\n"
f"    return max(0.0, c/(1.0+b))\n"
        )

    def gen_class(i: int) -> str:
        return (
f"\nclass Module_{i}:\n"
f"    def __init__(self, seed: int):\n"
f"        random.seed(seed)\n"
f"        self.hist: List[float] = []\n"
f"    def run(self, n: int) -> float:\n"
f"        acc = 0.0\n"
f"        for k in range(n):\n"
f"            val = util_{i}(random.random(), random.random(), random.random())\n"
f"            self.hist.append(val)\n"
f"            acc += val\n"
f"        return acc/max(1, n)\n"
        )

    # Stream to disk in chunks to avoid memory pressure
    lines_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        lines_written += header.count("\n") + 1
        i = 0
        block_batch = 0
        while lines_written < target_lines and not _shutdown_event.is_set():
            u = gen_util(i)
            c = gen_class(i)
            f.write(u); f.write(c)
            lines_written += u.count("\n") + c.count("\n") + 2
            i += 1; block_batch += 1
            if i % 50 == 0:
                agg = (
f"\n# --- Aggregator {i}\n"
f"def aggregate_{i}(n: int = 100) -> float:\n"
f"    m = Module_{i-1}(seed=n)\n"
f"    return m.run(n)\n"
                )
                f.write(agg)
                lines_written += agg.count("\n") + 1
            # Yield to OS periodically
            if block_batch >= 500:
                block_batch = 0
                time.sleep(0.001)
    return out_path, lines_written

# ---------------------------
# Feeds
# ---------------------------
class FeedRegistry:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.records: List[Dict] = []
    def add_record(self, rec: Dict):
        with self.lock:
            self.records.append({"ts": time.time(), "rec": rec})
            if len(self.records) > self.cfg["feeds_max_records"]:
                self.records = self.records[-self.cfg["feeds_max_records"]:]
    def snapshot(self) -> List[Dict]:
        with self.lock:
            return list(self.records)

def start_feed_loop(cfg: Dict, feeds: FeedRegistry):
    if not cfg.get("feeds_enable", True) or requests is None:
        logging.info("Feeds disabled or requests not available.")
        return
    def loop():
        while not _shutdown_event.is_set():
            for fd in cfg.get("feeds", []):
                try:
                    resp = requests.get(fd.get("url"), timeout=fd.get("timeout", 6.0))
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception:
                            data = {"text": resp.text[:2000]}
                        feeds.add_record({"name": fd.get("name"), "data": data})
                except Exception:
                    logging.info(f"Feed fetch failed for {fd.get('name')}")
            time.sleep(cfg.get("feeds_poll_interval", 30.0))
    threading.Thread(target=loop, daemon=True).start()

def feed_entropy(feeds: FeedRegistry) -> float:
    snap = feeds.snapshot()
    if not snap: return 0.0
    names = set([r["rec"].get("name") for r in snap if r.get("rec")])
    keys = set()
    for r in snap:
        d = r["rec"].get("data")
        if isinstance(d, dict):
            keys |= set(list(d.keys()))
    return min(1.5, 0.2*len(names) + 0.02*len(keys))

# ---------------------------
# Evolution
# ---------------------------
def select_parents(archive: Dict, k: int, seed_noise: float) -> List[Dict]:
    top_ids = [entry["id"] for entry in archive["top"][-50:]]
    pool = []
    for _ in range(k):
        if top_ids and random.random() < 0.7:
            pool.append(archive["ideas"][random.choice(top_ids)]["idea"])
        else:
            pool.append(compose_idea(seed_noise))
    return pool

def evolve(cfg: Dict, archive_ref: Dict, feeds: FeedRegistry, sysmon: SystemMetrics):
    paths = node_paths(cfg)
    with _archive_lock:
        archive = archive_ref["ref"]
    archive_texts = [archive["ideas"][i]["text"] for i in archive["ideas"]]
    no_improve = 0
    best_obj = max([archive["ideas"][i]["eval"]["objective"] for i in archive["ideas"]] + [0.0])
    throttle = Throttle(cfg)

    ensure_strategy_file(cfg)
    strategy_mod = load_strategy()
    last_strategy_mut = time.time()
    perf_window = []

    while not _shutdown_event.is_set():
        try:
            for step in range(cfg["iterations_per_tick"]):
                with _archive_lock:
                    archive = archive_ref["ref"]

                entropy_feed = feed_entropy(feeds)
                entropy_sys = system_entropy(sysmon.snapshot())
                with _archive_lock:
                    archive["stats"]["feed_entropy"] = entropy_feed
                    archive["stats"]["sys_entropy"] = entropy_sys

                parents = select_parents(archive, k=max(5, cfg["batch_size"]//4), seed_noise=cfg["seed_noise"])

                candidates = []
                active_names = []
                for _ in range(cfg["batch_size"]):
                    base = random.choice(parents)
                    idea = mutate_idea(base) if random.random() < 0.8 else compose_idea(cfg["seed_noise"])
                    # Influence from feeds & system metrics
                    idea["params"]["tolerance"] = round(max(0.003, idea["params"]["tolerance"] * (1.0 - 0.15*entropy_feed - 0.15*entropy_sys)), 4)
                    idea["params"]["mutation_rate"] = round(min(0.8, idea["params"]["mutation_rate"] * (1.0 + 0.25*entropy_feed + 0.25*entropy_sys)), 4)
                    pname = autonomous_project_name(idea, cfg["node_name"])
                    idea["project_name"] = pname

                    # Performance snapshot
                    perf = {
                        "avg_objective": (sum(perf_window)/len(perf_window)) if perf_window else 0.0,
                        "new_ideas": len(archive_texts),
                        "completion_rate": (archive["stats"]["completed_projects"]/max(1, archive["stats"]["iterations"])),
                        "feed_entropy": entropy_feed,
                        "sys_entropy": entropy_sys
                    }

                    weights_override = None
                    if strategy_mod and hasattr(strategy_mod, "adjust_weights"):
                        try:
                            weights_override = strategy_mod.adjust_weights(perf)
                        except Exception:
                            logging.exception("Strategy adjust failed.")
                            weights_override = None

                    evals = evaluate_idea(idea, archive_texts, cfg, perf=perf, weights_override=weights_override)
                    obj = evals["objective"]
                    candidates.append((idea, evals, obj))
                    active_names.append(pname)

                with _archive_lock:
                    archive["stats"]["active_projects"] = len(candidates)
                    archive["stats"]["active_project_names"] = active_names
                cfg["cpu_target_util"] = 0.05 if len(candidates) > 0 else 0.40

                candidates.sort(key=lambda x: x[2], reverse=True)
                elites = candidates[:max(5, cfg["batch_size"]//4)]

                improved = False
                completed = 0
                finished_names = []
                for idea, evals, obj in elites:
                    text = evals["text"]
                    hid = idea_hash(text)
                    with _archive_lock:
                        if hid not in archive["ideas"]:
                            # Generate ultra project file on disk (streaming)
                            out_path, line_count = generate_ultra_project_file(idea, evals, cfg, hid)
                            archive["ideas"][hid] = {
                                "id": hid, "idea": ensure_idea_keys(idea),
                                "text": text, "eval": evals,
                                "project_name": idea.get("project_name","unnamed"),
                                "code_path": out_path,
                                "code_lines": line_count
                            }
                            archive["top"].append({"id": hid, "objective": obj,
                                                   "novelty": evals["novelty"], "utility": evals["utility"]})
                            archive_texts.append(text)
                            improved = improved or (obj > best_obj)
                            best_obj = max(best_obj, obj)
                            completed += 1
                            finished_names.append(idea.get("project_name","unnamed"))
                            perf_window.append(obj)
                            if len(perf_window) > cfg["strategy_selection_window"]:
                                perf_window = perf_window[-cfg["strategy_selection_window"]:]

                if completed > 0:
                    with _archive_lock:
                        stats = archive["stats"]
                        stats["completed_projects"] = stats.get("completed_projects", 0) + completed
                        stats["finished_project_names"] = stats.get("finished_project_names", []) + finished_names
                    logging.info(f"Projects completed: {', '.join(finished_names)}")
                    emit_completion_tone(cfg)

                if cfg.get("strategy_enable", True) and (time.time() - last_strategy_mut) > cfg.get("strategy_mutate_interval", 120.0):
                    mutate_strategy(cfg)
                    importlib.invalidate_caches()
                    try:
                        if "strategies" in sys.modules:
                            del sys.modules["strategies"]
                        strategy_mod = importlib.import_module("strategies")
                        logging.info("Strategy module hot-reloaded.")
                    except Exception:
                        logging.exception("Failed to hot-reload strategies.")
                        strategy_mod = None
                    last_strategy_mut = time.time()

                with _archive_lock:
                    archive["stats"]["iterations"] += 1
                    archive["history"].append({
                        "step": archive["stats"]["iterations"],
                        "best_objective": round(best_obj, 6),
                        "archive_size": len(archive["ideas"]),
                        "recent_elites": [e[0].get("project_name", e[0]["title"]) for e in elites],
                        "node": cfg["node_name"],
                        "active_projects": archive["stats"]["active_projects"],
                        "active_project_names": archive["stats"]["active_project_names"],
                        "completed_projects": archive["stats"]["completed_projects"],
                        "finished_project_names": archive["stats"].get("finished_project_names", []),
                        "feed_entropy": archive["stats"].get("feed_entropy", 0.0),
                        "sys_entropy": archive["stats"].get("sys_entropy", 0.0),
                    })

                    if archive["stats"]["iterations"] % cfg["save_interval_steps"] == 0:
                        enforce_archive_limits(archive, cfg)
                        save_archive(paths, archive)
                        archive_ref["ref"] = archive
                        if archive["top"]:
                            best = max(archive["top"], key=lambda e: e["objective"])
                            top = archive["ideas"][best["id"]]
                            logging.info(f"[{archive['stats']['iterations']}] best={best['objective']:.3f} "
                                         f"nov={best['novelty']:.3f} util={best['utility']:.3f} "
                                         f":: {top.get('project_name', top['idea']['title'])} "
                                         f"| active={archive['stats']['active_projects']} "
                                         f"completed={archive['stats']['completed_projects']} "
                                         f"| feed_entropy={archive['stats']['feed_entropy']:.3f} sys_entropy={archive['stats']['sys_entropy']:.3f}")

                if not improved:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve > cfg["patience"]:
                    logging.info("Stagnation detected; injecting chaos batch.")
                    chaos_names = []
                    for _ in range(10):
                        idea = compose_idea(seed_noise=0.3)
                        idea["params"]["mutation_rate"] = round(min(0.8, idea["params"]["mutation_rate"] * (1.0 + 0.4*entropy_feed + 0.4*entropy_sys)), 4)
                        idea["project_name"] = autonomous_project_name(idea, cfg["node_name"])
                        perf = {
                            "avg_objective": (sum(perf_window)/len(perf_window)) if perf_window else 0.0,
                            "new_ideas": len(archive_texts),
                            "completion_rate": (archive["stats"]["completed_projects"]/max(1, archive["stats"]["iterations"])),
                            "feed_entropy": entropy_feed,
                            "sys_entropy": entropy_sys
                        }
                        evals = evaluate_idea(idea, archive_texts, cfg, perf=perf, weights_override=None)
                        hid = idea_hash(evals["text"])
                        with _archive_lock:
                            if hid not in archive["ideas"]:
                                out_path, line_count = generate_ultra_project_file(idea, evals, cfg, hid)
                                archive["ideas"][hid] = {"id": hid, "idea": ensure_idea_keys(idea), "text": evals["text"], "eval": evals,
                                                         "project_name": idea["project_name"], "code_path": out_path, "code_lines": line_count}
                                archive["top"].append({"id": hid, "objective": evals["objective"],
                                                       "novelty": evals["novelty"], "utility": evals["utility"]})
                                archive_texts.append(evals["text"])
                                chaos_names.append(idea["project_name"])
                    if chaos_names:
                        logging.info(f"Chaos batch seeded: {', '.join(chaos_names)}")
                    no_improve = 0

                throttle.adjust()
                throttle.wait()

            with _archive_lock:
                save_archive(paths, archive)
                archive_ref["ref"] = archive

        except Exception:
            logging.exception("Evolution loop error; continuing with backoff.")
            backoff = cfg["backoff_initial"]
            while backoff < cfg["backoff_max"] and not _shutdown_event.is_set():
                time.sleep(backoff)
                backoff = min(cfg["backoff_max"], backoff * 2)

    with _archive_lock:
        save_archive(paths, archive)
    logging.info("Evolution stopped. Final archive size: %d", len(archive["ideas"]))

# ---------------------------
# Tkinter GUI with mute button and huge code preview
# ---------------------------
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

def start_tk_dashboard(cfg: Dict, archive_ref: Dict):
    root = tk.Tk()
    root.title("Imagination Colony Dashboard")

    # Main info
    info_frame = ttk.Frame(root, padding=10)
    info_frame.pack(fill="x")
    node_label = ttk.Label(info_frame, text=f"Node: {cfg.get('node_name')}")
    node_label.pack(side="left")
    iter_label = ttk.Label(info_frame, text="Iterations: 0")
    iter_label.pack(side="left", padx=20)
    size_label = ttk.Label(info_frame, text="Archive size: 0")
    size_label.pack(side="left", padx=20)
    entropy_label = ttk.Label(info_frame, text="Feed entropy: 0.00")
    entropy_label.pack(side="left", padx=20)
    sys_entropy_label = ttk.Label(info_frame, text="System entropy: 0.00")
    sys_entropy_label.pack(side="left", padx=20)

    # Controls
    controls = ttk.Frame(root, padding=10)
    controls.pack(fill="x")
    sound_status = tk.StringVar(value="Sound: ON" if cfg.get("audible_signals", True) else "Sound: OFF")
    def toggle_sound():
        cfg["audible_signals"] = not cfg.get("audible_signals", True)
        sound_status.set("Sound: ON" if cfg["audible_signals"] else "Sound: OFF")
    mute_btn = ttk.Button(controls, textvariable=sound_status, command=toggle_sound)
    mute_btn.pack(side="left")

    target_label = ttk.Label(controls, text=f"Target lines: {cfg.get('project_target_min_lines')}–{cfg.get('project_target_max_lines')}")
    target_label.pack(side="left", padx=20)

    # Active projects panel
    active_frame = ttk.LabelFrame(root, text="Active Projects", padding=10)
    active_frame.pack(fill="both", expand=True, padx=10, pady=5)
    active_list = tk.Listbox(active_frame, height=10)
    active_list.pack(fill="both", expand=True)

    # Finished projects panel
    finished_frame = ttk.LabelFrame(root, text="Finished Projects (double-click to preview code)", padding=10)
    finished_frame.pack(fill="both", expand=True, padx=10, pady=5)
    finished_list = tk.Listbox(finished_frame, height=14)
    finished_list.pack(fill="both", expand=True)

    # Status bar
    status = ttk.Label(root, text="Ready", relief="sunken", anchor="w")
    status.pack(fill="x")

    def show_code(event=None):
        selection = finished_list.curselection()
        if not selection:
            return
        idx = selection[0]
        pname = finished_list.get(idx)
        with _archive_lock:
            archive = archive_ref["ref"]
            target_id = None
            for iid, ideadata in archive.get("ideas", {}).items():
                if ideadata.get("project_name") == pname:
                    target_id = iid
                    break
            if target_id is None:
                status.config(text="Code not found for selected project.")
                return
            code_path = archive["ideas"][target_id].get("code_path")
            code_lines = archive["ideas"][target_id].get("code_lines", 0)

        if not code_path or not os.path.exists(code_path):
            status.config(text="Code file missing.")
            return

        preview_max = int(cfg.get("preview_max_lines", 1200))
        preview_content = []
        lines_read = 0
        try:
            with open(code_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    preview_content.append(line)
                    lines_read += 1
                    if lines_read >= preview_max:
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read code: {e}")
            return

        popup = tk.Toplevel(root)
        popup.title(f"Code preview: {pname} ({lines_read}/{code_lines} lines shown)")
        text = scrolledtext.ScrolledText(popup, wrap="none", width=100, height=35)
        text.pack(fill="both", expand=True)
        text.insert("1.0", "".join(preview_content) + ("\n# ... (truncated preview) ..." if lines_read < code_lines else ""))
        text.configure(state="disabled")

        file_frame = ttk.Frame(popup, padding=6)
        file_frame.pack(fill="x")
        ttk.Label(file_frame, text=f"File: {code_path}").pack(side="left")
        def open_external():
            try:
                if sys.platform.startswith("win"):
                    os.startfile(code_path)  # type: ignore
                elif sys.platform == "darwin":
                    subprocess.call(["open", code_path])
                else:
                    subprocess.call(["xdg-open", code_path])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")
        ttk.Button(file_frame, text="Open in system editor", command=open_external).pack(side="left", padx=10)
        status.config(text=f"Previewed {lines_read}/{code_lines} lines for {pname}")

    finished_list.bind("<Double-Button-1>", show_code)

    def refresh():
        with _archive_lock:
            archive = archive_ref["ref"]
            stats = archive.get("stats", {})
            iterations = stats.get("iterations", 0)
            size = len(archive.get("ideas", {}))
            active_names = stats.get("active_project_names", [])
            finished_names = stats.get("finished_project_names", [])
            entropy = stats.get("feed_entropy", 0.0)
            sys_ent = stats.get("sys_entropy", 0.0)

        iter_label.config(text=f"Iterations: {iterations}")
        size_label.config(text=f"Archive size: {size}")
        entropy_label.config(text=f"Feed entropy: {entropy:.2f}")
        sys_entropy_label.config(text=f"System entropy: {sys_ent:.2f}")

        active_list.delete(0, tk.END)
        for n in active_names:
            active_list.insert(tk.END, n)

        finished_list.delete(0, tk.END)
        for n in finished_names[-300:]:
            finished_list.insert(tk.END, n)

        root.after(int(cfg.get("gui_refresh_ms", 2000)), refresh)

    refresh()
    try:
        root.mainloop()
    finally:
        _shutdown_event.set()

# ---------------------------
# Run
# ---------------------------
def run():
    cfg = load_config()
    setup_logging(cfg)
    if not os.path.exists(CONFIG_PATH):
        save_config(cfg)
    logging.info(f"Starting node {cfg['node_name']} | dir={cfg['data_dir']} | tk=on | feeds={'on' if cfg.get('feeds_enable') else 'off'} | strategies={'on' if cfg.get('strategy_enable') else 'off'} | sound={'on' if cfg.get('audible_signals') else 'off'}")

    paths = node_paths(cfg)
    archive = load_archive(paths)
    archive_ref = {"ref": archive}

    # Start live feeds
    feeds = FeedRegistry(cfg)
    start_feed_loop(cfg, feeds)

    # Start system metrics monitor
    sysmon = SystemMetrics()
    sysmon.start(interval=2.0)

    # Start evolution thread
    evo_thread = threading.Thread(target=evolve, args=(cfg, archive_ref, feeds, sysmon), daemon=True)
    evo_thread.start()

    # Start Tkinter GUI in main thread
    start_tk_dashboard(cfg, archive_ref)

    evo_thread.join(timeout=2.0)
    logging.info("Shutdown complete.")

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    random.seed(time.time())
    run()

