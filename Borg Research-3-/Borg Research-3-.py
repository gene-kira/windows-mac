#!/usr/bin/env python3
# imagination_colony_tk.py
# Fully autonomous imagination engine with:
# - Tkinter GUI (live dashboard): active project names, counts, finished projects
# - Click to view generated Python code for finished projects
# - Self-named projects (mythic+semantic+hash)
# - Impact scoring, adaptive CPU target (~5% busy, up to 40% idle)
# - Robust persistence and optional mesh (UDP/TCP) disabled by default for simplicity

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

# ---------------------------
# Paths and defaults
# ---------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "imagination_config.json")

DEFAULT_CONFIG = {
    "iterations_per_tick": 120,
    "batch_size": 24,
    "patience": 30,
    "save_interval_steps": 10,
    "max_archive_size": 100_000,

    # CPU target adapted each step:
    #  - 0.05 when active projects > 0
    #  - 0.40 when idle
    "cpu_target_util": 0.05,
    "sleep_min_seconds": 0.05,
    "sleep_max_seconds": 2.0,

    "backoff_initial": 1.0,
    "backoff_max": 60.0,
    "auto_update_simulators": True,

    # Objective weights
    "novelty_weight": 0.4,
    "utility_weight": 0.3,
    "impact_weight": 0.2,
    "curiosity_weight": 0.1,

    "seed_noise": 0.15,

    # Logging
    "log_dir": os.path.join(BASE_DIR, "logs"),
    "max_log_bytes": 5_000_000,
    "log_backup_count": 3,

    # Mesh settings (disabled by default to simplify TK operation)
    "node_name": None,
    "data_dir": None,
    "mesh_udp_port": None,
    "mesh_tcp_port": None,
    "mesh_broadcast_interval": 5.0,
    "mesh_peer_timeout": 40.0,
    "mesh_max_peers": 256,
    "mesh_exchange_interval": 15.0,
    "mesh_chunk_limit": 256,
    "mesh_enable": False,            # turn off for TK-only run
    "mesh_allow_external": False,
    "mesh_public_broadcast": False,

    # Replication
    "replicate_enabled": False,
    "replicate_max_children": 3,
    "replicate_check_interval": 120.0,
    "replicate_child_cpu_target": 0.25,
    "replicate_spawn_dir": os.path.join(BASE_DIR, "colony"),
    "replicate_copy_archive": True,
    "replicate_mutation_variation": 0.05,

    # Communications evolution
    "comm_enable": False,            # off in TK-only mode
    "comm_codec_initial": "json+zlib",
    "comm_adapt_interval": 60.0,
    "comm_max_frame": 8_000_000,
    "comm_schema_version": 1,
    "comm_schema_mutate_prob": 0.08,

    # Audible signals
    "audible_signals": True
}

# ---------------------------
# Global state: logging and signals
# ---------------------------
_shutdown_event = threading.Event()
_archive_lock = threading.Lock()

def setup_logging(cfg: Dict):
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_path = os.path.join(cfg["log_dir"], f"{cfg['node_name'] or 'node'}.log")
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
# Config load/save
# ---------------------------
def pick_free_tcp_port(bind_addr: str = "127.0.0.1") -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((bind_addr, 0))
    port = s.getsockname()[1]
    s.close()
    return port

def pick_free_udp_port(bind_addr: str = "127.0.0.1") -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((bind_addr, 0))
    port = s.getsockname()[1]
    s.close()
    return port

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
    if not cfg.get("node_name"):
        cfg["node_name"] = f"node-{host}-{random.randint(1000,9999)}"
    if not cfg.get("data_dir"):
        cfg["data_dir"] = os.path.join(BASE_DIR, cfg["node_name"])
    os.makedirs(cfg["data_dir"], exist_ok=True)
    if not cfg.get("mesh_tcp_port"):
        cfg["mesh_tcp_port"] = pick_free_tcp_port("127.0.0.1")
    if not cfg.get("mesh_udp_port"):
        cfg["mesh_udp_port"] = pick_free_udp_port("127.0.0.1")
    return cfg

def save_config(cfg: Dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        logging.exception("Failed to save config.")

def node_paths(cfg: Dict) -> Dict:
    return {
        "archive_path": os.path.join(cfg["data_dir"], "imagination_archive.json"),
        "log_dir": cfg["log_dir"],
    }

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
# Novelty & evaluation
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

def evaluate_idea(idea: Dict, archive_texts: List[str], cfg: Dict) -> Dict:
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
    objective = (cfg["novelty_weight"] * novelty
                 + cfg["utility_weight"] * utility
                 + cfg["impact_weight"] * impact
                 + cfg["curiosity_weight"] * curiosity)
    return {"text": text, "novelty": novelty, "utility": utility,
            "curiosity": curiosity, "impact": impact,
            "objective": objective, "sims": results}

# ---------------------------
# Archive persistence
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
                return data
        except Exception:
            logging.exception("Archive load failed; starting fresh.")
    return {"ideas": {}, "top": [], "history": [], "stats": {
        "iterations": 0, "active_projects": 0, "completed_projects": 0,
        "active_project_names": [], "finished_project_names": []
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
# Adaptive throttling
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
# Generated code for finished projects
# ---------------------------
def generate_project_code(idea: Dict, evals: Dict) -> str:
    pname = idea.get("project_name", "unnamed_project")
    domain = idea.get("domain", "unknown")
    mechanism = idea.get("mechanism", "unknown")
    architecture = idea.get("architecture", "unknown")
    intent = idea.get("intent", "unknown")
    params = idea.get("params", {})
    code = f'''#!/usr/bin/env python3
# Generated autonomous project: {pname}
# Domain: {domain} | Mechanism: {mechanism} | Architecture: {architecture} | Intent: {intent}
# This module is portable and runnable on general-purpose Python environments.

import sys, math, random, json, time

def simulate(params):
    # Placeholder logic demonstrating portability.
    scale = float(params.get("scale", 1.0))
    power = float(params.get("power_budget", 1000.0))
    tolerance = float(params.get("tolerance", 0.1))
    score = max(0.0, (scale / (1.0 + tolerance)) * (1.0 / (1.0 + power/1e5)))
    return {{
        "score": score,
        "scale": scale,
        "power": power,
        "tolerance": tolerance,
        "timestamp": time.time()
    }}

def main():
    params = {json.dumps(params, indent=2)}
    result = simulate(params)
    print(json.dumps({{"project":"{pname}","result":result}}, indent=2))

if __name__ == "__main__":
    main()
'''
    return code

# ---------------------------
# Evolution loop
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

def evolve(cfg: Dict, archive_ref: Dict):
    paths = node_paths(cfg)
    with _archive_lock:
        archive = archive_ref["ref"]
    archive_texts = [archive["ideas"][i]["text"] for i in archive["ideas"]]
    no_improve = 0
    best_obj = max([archive["ideas"][i]["eval"]["objective"] for i in archive["ideas"]] + [0.0])
    throttle = Throttle(cfg)

    # Optional hot reload disabled in TK mode to keep it simple

    while not _shutdown_event.is_set():
        try:
            for step in range(cfg["iterations_per_tick"]):
                with _archive_lock:
                    archive = archive_ref["ref"]
                parents = select_parents(archive, k=max(5, cfg["batch_size"]//4), seed_noise=cfg["seed_noise"])

                # Generate candidates (projects in progress) with autonomous names
                candidates = []
                active_names = []
                for _ in range(cfg["batch_size"]):
                    base = random.choice(parents)
                    idea = mutate_idea(base) if random.random() < 0.8 else compose_idea(cfg["seed_noise"])
                    pname = autonomous_project_name(idea, cfg["node_name"])
                    idea["project_name"] = pname
                    evals = evaluate_idea(idea, archive_texts, cfg)
                    obj = evals["objective"]
                    candidates.append((idea, evals, obj))
                    active_names.append(pname)

                # Update active projects and CPU target
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
                            code_stub = generate_project_code(idea, evals)
                            archive["ideas"][hid] = {
                                "id": hid, "idea": ensure_idea_keys(idea),
                                "text": text, "eval": evals,
                                "project_name": idea.get("project_name","unnamed"),
                                "code": code_stub
                            }
                            archive["top"].append({"id": hid, "objective": obj,
                                                   "novelty": evals["novelty"], "utility": evals["utility"]})
                            archive_texts.append(text)
                            improved = improved or (obj > best_obj)
                            best_obj = max(best_obj, obj)
                            completed += 1
                            finished_names.append(idea.get("project_name","unnamed"))

                if completed > 0:
                    with _archive_lock:
                        stats = archive["stats"]
                        stats["completed_projects"] = stats.get("completed_projects", 0) + completed
                        stats["finished_project_names"] = stats.get("finished_project_names", []) + finished_names
                    logging.info(f"Projects completed: {', '.join(finished_names)}")
                    emit_completion_tone(cfg)

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
                    })

                    # Save periodically
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
                                         f"completed={archive['stats']['completed_projects']}")

                if not improved:
                    no_improve += 1
                else:
                    no_improve = 0
                if no_improve > cfg["patience"]:
                    logging.info("Stagnation detected; injecting chaos batch.")
                    chaos_names = []
                    for _ in range(10):
                        idea = compose_idea(seed_noise=0.3)
                        idea["project_name"] = autonomous_project_name(idea, cfg["node_name"])
                        evals = evaluate_idea(idea, archive_texts, cfg)
                        hid = idea_hash(evals["text"])
                        with _archive_lock:
                            if hid not in archive["ideas"]:
                                code_stub = generate_project_code(idea, evals)
                                archive["ideas"][hid] = {"id": hid, "idea": ensure_idea_keys(idea), "text": evals["text"], "eval": evals,
                                                         "project_name": idea["project_name"], "code": code_stub}
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
# Tkinter GUI
# ---------------------------
import tkinter as tk
from tkinter import ttk, scrolledtext

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

    # Active projects panel
    active_frame = ttk.LabelFrame(root, text="Active Projects", padding=10)
    active_frame.pack(fill="both", expand=True, padx=10, pady=5)
    active_list = tk.Listbox(active_frame, height=10)
    active_list.pack(fill="both", expand=True)

    # Finished projects panel
    finished_frame = ttk.LabelFrame(root, text="Finished Projects (double-click to view code)", padding=10)
    finished_frame.pack(fill="both", expand=True, padx=10, pady=5)
    finished_list = tk.Listbox(finished_frame, height=10)
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
            # find idea by project_name
            target_id = None
            for iid, ideadata in archive.get("ideas", {}).items():
                if ideadata.get("project_name") == pname:
                    target_id = iid
                    break
            if target_id is None:
                status.config(text="Code not found for selected project.")
                return
            code = archive["ideas"][target_id].get("code", "# No code stored")

        popup = tk.Toplevel(root)
        popup.title(f"Code: {pname}")
        text = scrolledtext.ScrolledText(popup, wrap="none", width=100, height=30)
        text.pack(fill="both", expand=True)
        text.insert("1.0", code)
        text.configure(state="disabled")
        status.config(text=f"Opened code for {pname}")

    finished_list.bind("<Double-Button-1>", show_code)

    def refresh():
        with _archive_lock:
            archive = archive_ref["ref"]
            stats = archive.get("stats", {})
            iterations = stats.get("iterations", 0)
            size = len(archive.get("ideas", {}))
            active_names = stats.get("active_project_names", [])
            finished_names = stats.get("finished_project_names", [])

        iter_label.config(text=f"Iterations: {iterations}")
        size_label.config(text=f"Archive size: {size}")

        active_list.delete(0, tk.END)
        for n in active_names:
            active_list.insert(tk.END, n)

        finished_list.delete(0, tk.END)
        for n in finished_names[-200:]:
            finished_list.insert(tk.END, n)

        root.after(2000, refresh)

    refresh()
    try:
        root.mainloop()
    finally:
        _shutdown_event.set()

# ---------------------------
# Watchdog: start evolution in background, TK in main
# ---------------------------
def run():
    cfg = load_config()
    setup_logging(cfg)
    if not os.path.exists(CONFIG_PATH):
        save_config(cfg)
    logging.info(f"Starting node {cfg['node_name']} | dir={cfg['data_dir']} | tk=on")

    paths = node_paths(cfg)
    archive = load_archive(paths)
    archive_ref = {"ref": archive}

    # Start evolution thread
    evo_thread = threading.Thread(target=evolve, args=(cfg, archive_ref), daemon=True)
    evo_thread.start()

    # Start Tkinter GUI in main thread
    start_tk_dashboard(cfg, archive_ref)

    # Wait for evolution to stop on GUI exit
    evo_thread.join(timeout=2.0)
    logging.info("Shutdown complete.")

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    random.seed(time.time())
    run()

