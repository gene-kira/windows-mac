import os
import sys
import json
import time
import threading
from datetime import datetime
import random
import subprocess
import importlib

# =========================================================
# 0. DEPENDENCIES
# =========================================================

def ensure_libs():
    for lib in ["torch", "numpy"]:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[DEPS] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_libs()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox

# =========================================================
# 1. GRID WORLD ENVIRONMENT (MULTI-AGENT + SITUATIONAL + ASCII)
# =========================================================

class GridWorldEnv:
    """
    Multi-agent 2D grid environment.
    Agents move on a grid, each has its own target.
    Difficulty controls grid size and max steps.

    Observation per agent (situational):
      [ax, ay, tx, ty, dist, difficulty, recent_success, recent_avg_reward]
      all normalized into [0,1] range where reasonable.
    """

    def __init__(self, num_agents=4, base_size=5, max_size=15):
        self.base_size = base_size
        self.max_size = max_size
        self.size = base_size

        self.num_agents = num_agents
        self.agent_positions = None   # list of [x,y]
        self.target_positions = None  # list of [x,y]
        self.max_steps = None
        self.steps = 0

    def set_difficulty(self, difficulty_level):
        difficulty_level = max(1, min(10, difficulty_level))
        self.size = min(self.base_size + difficulty_level - 1, self.max_size)
        self.max_steps = self.size * self.size * 2

    def reset(self, stats=None):
        self.steps = 0
        self.agent_positions = []
        self.target_positions = []
        for _ in range(self.num_agents):
            ax = random.randint(0, self.size - 1)
            ay = random.randint(0, self.size - 1)
            while True:
                tx = random.randint(0, self.size - 1)
                ty = random.randint(0, self.size - 1)
                if not (ax == tx and ay == ty):
                    break
            self.agent_positions.append([ax, ay])
            self.target_positions.append([tx, ty])

        return self._get_obs(stats)

    def _get_obs(self, stats=None):
        """
        Returns an array of shape [num_agents, obs_dim]:
        [axn, ayn, txn, tyn, dist, diff, recent_success, recent_avg]
        """
        if stats is not None:
            snap = stats.snapshot()
            diff = snap["difficulty"] / 10.0
            recent_success = snap["success_rate"]
            recent_avg = (snap["avg_reward"] + 1.0) / 2.0  # crude mapping to [0,1]
        else:
            diff = 0.1
            recent_success = 0.0
            recent_avg = 0.5

        obs_list = []
        for (ax, ay), (tx, ty) in zip(self.agent_positions, self.target_positions):
            axn = ax / max(1, (self.size - 1))
            ayn = ay / max(1, (self.size - 1))
            txn = tx / max(1, (self.size - 1))
            tyn = ty / max(1, (self.size - 1))

            dx = (tx - ax) / max(1, (self.size - 1))
            dy = (ty - ay) / max(1, (self.size - 1))
            dist = (abs(dx) + abs(dy)) / 2.0  # simple normalized manhattan-ish

            obs_list.append([
                axn, ayn, txn, tyn,
                dist,
                diff,
                recent_success,
                recent_avg,
            ])
        return np.array(obs_list, dtype=np.float32)

    def step(self, actions, stats=None):
        """
        actions: array of ints [num_agents], each in {0,1,2,3}
        Returns:
          obs_next [num_agents, obs_dim]
          rewards  [num_agents]
          dones    [num_agents] boolean
          episode_done (True if env should reset)
        """
        self.steps += 1
        rewards = np.full(self.num_agents, -0.01, dtype=np.float32)  # step penalty
        dones = np.zeros(self.num_agents, dtype=bool)

        for i, action in enumerate(actions):
            ax, ay = self.agent_positions[i]

            if action == 0:   # up
                ay = max(0, ay - 1)
            elif action == 1: # down
                ay = min(self.size - 1, ay + 1)
            elif action == 2: # left
                ax = max(0, ax - 1)
            elif action == 3: # right
                ax = min(self.size - 1, ax + 1)

            self.agent_positions[i] = [ax, ay]

            if self.agent_positions[i] == self.target_positions[i]:
                rewards[i] = 1.0
                dones[i] = True

        episode_done = self.steps >= self.max_steps or np.all(dones)
        obs_next = self._get_obs(stats)
        return obs_next, rewards, dones, episode_done

    def ascii_view(self, max_agents_display=4):
        """
        Returns a string showing a simple ASCII grid.
        Displays at most max_agents_display agents.
        """
        size = self.size
        grid = [["." for _ in range(size)] for _ in range(size)]

        for idx, (ax, ay) in enumerate(self.agent_positions[:max_agents_display]):
            ch = chr(ord("A") + idx)  # A, B, C, D...
            grid[ay][ax] = ch

        for idx, (tx, ty) in enumerate(self.target_positions[:max_agents_display]):
            if grid[ty][tx] == ".":
                grid[ty][tx] = "X"
            else:
                grid[ty][tx] = "*"  # agent+target overlap

        lines = [f"Grid {size}x{size}, step {self.steps}"]
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)


# =========================================================
# 2. NEURAL BRAIN (Q + PREDICTIVE HEAD)
# =========================================================

class SimpleBrain(nn.Module):
    """
    Approximates:
      - Q(s, a) for 4 actions  (judgment)
      - Predicted next state + reward (world model / predictive intelligence)
    """
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # Q-values for 4 actions

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Q-head (judgment)
        self.q_head = nn.Linear(hidden_dim, output_dim)

        # Prediction head (next state + reward): input_dim + 1
        self.pred_head = nn.Linear(hidden_dim, input_dim + 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        q = self.q_head(h)
        pred = self.pred_head(h)
        return q, pred

    def rebuild(self, input_dim, output_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        self.__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


# =========================================================
# 3. MEMORY MANAGER (PRIMARY + SECONDARY)
# =========================================================

class MemoryManager:
    def __init__(self, brain, config_path=None):
        self.brain = brain

        if config_path is None:
            home = os.path.expanduser("~")
            config_dir = os.path.join(home, ".predictive_grid_organism")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "memory_config.json")

        self.config_path = config_path
        self.primary_path = None
        self.secondary_path = None

        self.last_status = "Idle"
        self.last_load_source = None
        self.last_save_time = None
        self.last_save_ok = False

        self._load_config()

    def _load_config(self):
        if os.path.isfile(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.primary_path = cfg.get("primary_path") or None
                self.secondary_path = cfg.get("secondary_path") or None
                self.last_status = "Config loaded"
            except Exception as e:
                self.last_status = f"Failed to load config: {e}"
        else:
            self.last_status = "No config yet; please select paths"

    def _save_config(self):
        cfg = {
            "primary_path": self.primary_path,
            "secondary_path": self.secondary_path,
        }
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self.last_status = "Config saved"
        except Exception as e:
            self.last_status = f"Failed to save config: {e}"

    def _ensure_dir(self, path):
        if path is None:
            return False, "Path is None"
        try:
            os.makedirs(path, exist_ok=True)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _memory_files(self, base_path):
        model_path = os.path.join(base_path, "brain_weights.pt")
        meta_path = os.path.join(base_path, "brain_meta.json")
        return model_path, meta_path

    def set_primary_path(self, path):
        self.primary_path = path
        self._save_config()

    def set_secondary_path(self, path):
        self.secondary_path = path
        self._save_config()

    def load_brain(self):
        def try_load_from(label, base_path):
            if base_path is None:
                return False, f"{label} path not set"
            model_path, meta_path = self._memory_files(base_path)
            if not os.path.isfile(model_path):
                return False, f"{label}: model file not found"
            try:
                state = torch.load(model_path, map_location="cpu")
                self.brain.load_state_dict(state)
                if os.path.isfile(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        _meta = json.load(f)
                self.last_load_source = label
                self.last_status = f"Loaded brain from {label}"
                return True, ""
            except Exception as e:
                return False, f"{label} load failed: {e}"

        ok1, msg1 = try_load_from("primary", self.primary_path)
        if ok1:
            return True, self.last_status

        ok2, msg2 = try_load_from("secondary", self.secondary_path)
        if ok2:
            return True, self.last_status

        self.last_status = "Fresh brain (no existing memory). " + " | ".join([msg1, msg2])
        self.last_load_source = None
        return False, self.last_status

    def save_brain(self, extra_meta=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        state = self.brain.state_dict()
        meta = {
            "saved_at": now,
            "model_type": "SimpleBrainPredictiveGridMulti",
        }
        if extra_meta:
            meta.update(extra_meta)

        primary_ok = False
        secondary_ok = False
        messages = []

        if self.primary_path is not None:
            ok, err = self._ensure_dir(self.primary_path)
            if not ok:
                messages.append(f"Primary dir error: {err}")
            else:
                model_path, meta_path = self._memory_files(self.primary_path)
                try:
                    torch.save(state, model_path)
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    primary_ok = True
                    messages.append(f"Primary save OK: {model_path}")
                except Exception as e:
                    messages.append(f"Primary save failed: {e}")
        else:
            messages.append("Primary path not set")

        if self.secondary_path is not None:
            ok, err = self._ensure_dir(self.secondary_path)
            if not ok:
                messages.append(f"Secondary dir error: {err}")
            else:
                model_path, meta_path = self._memory_files(self.secondary_path)
                try:
                    torch.save(state, model_path)
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                    secondary_ok = True
                    messages.append(f"Secondary save OK: {model_path}")
                except Exception as e:
                    messages.append(f"Secondary save failed: {e}")
        else:
            messages.append("Secondary path not set")

        self.last_save_time = now
        self.last_save_ok = primary_ok or secondary_ok

        if self.last_save_ok:
            self.last_status = "Save OK: " + " | ".join(messages)
        else:
            self.last_status = "Save FAILED: " + " | ".join(messages)

        return self.last_save_ok, self.last_status


# =========================================================
# 4. TRAINING STATS + SKILLS + CONFIDENCE
# =========================================================

class TrainingStats:
    def __init__(self):
        self.lock = threading.Lock()

        self.episode = 0
        self.last_reward = 0.0
        self.avg_reward = 0.0
        self.success_rate = 0.0
        self.total_successes = 0
        self.total_episodes = 0

        self.skills = {}

        self.running = False
        self.last_message = "Idle"
        self.difficulty_level = 1

        self.current_learning_rate = 1e-3
        self.current_exploration = 0.5
        self.network_size = 128

        self.last_ascii = ""
        self.confidence = 0.0
        self.state = "exploring"

    def update_episode(self, episode_reward, success_threshold=0.5):
        with self.lock:
            self.episode += 1
            self.last_reward = episode_reward
            self.total_episodes += 1
            if episode_reward >= success_threshold:
                self.total_successes += 1
            self.success_rate = self.total_successes / max(1, self.total_episodes)

    def update_avg_reward(self, avg_reward):
        with self.lock:
            self.avg_reward = avg_reward

    def set_message(self, msg):
        with self.lock:
            self.last_message = msg

    def set_difficulty(self, level):
        with self.lock:
            self.difficulty_level = level

    def set_meta_params(self, learning_rate=None, exploration=None, network_size=None):
        with self.lock:
            if learning_rate is not None:
                self.current_learning_rate = learning_rate
            if exploration is not None:
                self.current_exploration = exploration
            if network_size is not None:
                self.network_size = network_size

    def set_ascii(self, ascii_str):
        with self.lock:
            self.last_ascii = ascii_str

    def set_confidence(self, value):
        with self.lock:
            self.confidence = float(value)

    def set_state(self, value):
        with self.lock:
            self.state = str(value)

    def _skill_level_from_value(self, value):
        if value < 0.2:
            return "none"
        elif value < 0.4:
            return "bronze"
        elif value < 0.6:
            return "silver"
        elif value < 0.8:
            return "gold"
        else:
            return "platinum"

    def update_skill(self, name, performance_value):
        level = self._skill_level_from_value(performance_value)
        confidence = int(performance_value * 100)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.lock:
            self.skills[name] = {
                "level": level,
                "confidence": confidence,
                "last_update": now,
            }

    def snapshot(self):
        with self.lock:
            return {
                "episode": self.episode,
                "last_reward": self.last_reward,
                "avg_reward": self.avg_reward,
                "success_rate": self.success_rate,
                "skills": {k: dict(v) for k, v in self.skills.items()},
                "message": self.last_message,
                "difficulty": self.difficulty_level,
                "learning_rate": self.current_learning_rate,
                "exploration": self.current_exploration,
                "network_size": self.network_size,
                "ascii": self.last_ascii,
                "confidence": self.confidence,
                "state": self.state,
            }


# =========================================================
# 5. CONSCIOUSNESS STATES + SELF-IMPROVEMENT ENGINE
# =========================================================

class ConsciousnessState:
    EXPLORING = "exploring"
    FOCUSED = "focused"
    CAUTIOUS = "cautious"

class SelfImprovementEngine:
    def __init__(self,
                 initial_lr=1e-3,
                 min_lr=1e-5,
                 max_lr=1e-2,
                 initial_exploration=0.5,
                 min_exploration=0.05,
                 max_exploration=0.9):

        self.learning_rate = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.exploration = initial_exploration
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration

        self.difficulty = 1
        self.reward_scale = 1.0
        self.network_hidden_dim = 128

        self._last_avg_reward = None
        self._plateau_count = 0

        self.state = ConsciousnessState.EXPLORING

    def update(self, stats: TrainingStats):
        snap = stats.snapshot()
        avg_reward = snap["avg_reward"]
        success_rate = snap["success_rate"]
        confidence = snap["confidence"]

        if self._last_avg_reward is None:
            self._last_avg_reward = avg_reward
        delta = avg_reward - self._last_avg_reward
        self._last_avg_reward = avg_reward

        if delta > 0.01:
            self.learning_rate = max(self.min_lr, self.learning_rate * 0.95)
        elif delta < -0.01:
            self.learning_rate = min(self.max_lr, self.learning_rate * 1.05)

        if success_rate < 0.3:
            self.exploration = min(self.max_exploration, self.exploration + 0.01)
        elif success_rate > 0.7:
            self.exploration = max(self.min_exploration, self.exploration - 0.01)

        if avg_reward > 0.7 and success_rate > 0.6:
            self.difficulty = min(10, self.difficulty + 1)
        elif avg_reward < 0.3 and success_rate < 0.2:
            self.difficulty = max(1, self.difficulty - 1)

        self.reward_scale = 1.0 + (self.difficulty - 1) * 0.1

        if delta < 0.005 and avg_reward < 0.8 and stats.episode > 200:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        if self._plateau_count > 50 and self.network_hidden_dim < 256:
            self.network_hidden_dim += 16
            self._plateau_count = 0

        # consciousness state logic
        if confidence < 0.3:
            self.state = ConsciousnessState.CAUTIOUS
        elif success_rate > 0.6 and avg_reward > 0.5:
            self.state = ConsciousnessState.FOCUSED
        else:
            self.state = ConsciousnessState.EXPLORING

        stats.set_meta_params(
            learning_rate=self.learning_rate,
            exploration=self.exploration,
            network_size=self.network_hidden_dim
        )
        stats.set_state(self.state)


# =========================================================
# 6. CONFIDENCE ESTIMATION
# =========================================================

def estimate_confidence(brain, obs_tensor, samples=5, noise_std=0.05):
    """
    Crude uncertainty estimate via Q-disagreement.
    Returns a numpy array [N] with confidence in [0,1].
    """
    with torch.no_grad():
        qs = []
        for _ in range(samples):
            noisy = obs_tensor + torch.randn_like(obs_tensor) * noise_std
            q, _ = brain(noisy)
            qs.append(q.unsqueeze(0))  # [1,N,4]
        qs = torch.cat(qs, dim=0)  # [S,N,4]
        std = qs.std(dim=0).mean(dim=1)  # [N]
        conf = 1.0 / (1.0 + std)
        conf = torch.clamp(conf, 0.0, 1.0)
        return conf.numpy()


# =========================================================
# 7. GUI
# =========================================================

class BrainGUI:
    def __init__(self, root, brain, memory_manager, stats: TrainingStats):
        self.root = root
        self.brain = brain
        self.mm = memory_manager
        self.stats = stats

        self.root.title("Predictive Grid Organism - Brain, Memory & Evolution Console")

        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        path_frame = tk.LabelFrame(frame, text="Memory locations", padx=10, pady=10)
        path_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(path_frame, text="Primary memory path:").grid(row=0, column=0, sticky="w")
        self.primary_var = tk.StringVar(value=self.mm.primary_path or "")
        self.primary_entry = tk.Entry(path_frame, textvariable=self.primary_var, width=50)
        self.primary_entry.grid(row=0, column=1, sticky="w")
        self.btn_primary = tk.Button(path_frame, text="Select...", command=self.pick_primary)
        self.btn_primary.grid(row=0, column=2, padx=5)

        tk.Label(path_frame, text="Secondary backup path:").grid(row=1, column=0, sticky="w")
        self.secondary_var = tk.StringVar(value=self.mm.secondary_path or "")
        self.secondary_entry = tk.Entry(path_frame, textvariable=self.secondary_var, width=50)
        self.secondary_entry.grid(row=1, column=1, sticky="w")
        self.btn_secondary = tk.Button(path_frame, text="Select...", command=self.pick_secondary)
        self.btn_secondary.grid(row=1, column=2, padx=5)

        btn_frame = tk.Frame(frame, pady=10)
        btn_frame.pack(fill=tk.X)

        self.btn_load = tk.Button(btn_frame, text="Load brain from memory", command=self.load_brain)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(btn_frame, text="Save brain now", command=self.save_brain)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        training_frame = tk.LabelFrame(frame, text="Learning status", padx=10, pady=10)
        training_frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_episode = tk.Label(training_frame, text="Episode: 0")
        self.lbl_episode.grid(row=0, column=0, sticky="w")

        self.lbl_last_reward = tk.Label(training_frame, text="Last reward: 0.000")
        self.lbl_last_reward.grid(row=1, column=0, sticky="w")

        self.lbl_avg_reward = tk.Label(training_frame, text="Avg reward: 0.000")
        self.lbl_avg_reward.grid(row=2, column=0, sticky="w")

        self.lbl_success_rate = tk.Label(training_frame, text="Success rate: 0.00%")
        self.lbl_success_rate.grid(row=3, column=0, sticky="w")

        self.lbl_difficulty = tk.Label(training_frame, text="Difficulty: 1")
        self.lbl_difficulty.grid(row=4, column=0, sticky="w")

        self.lbl_skills = tk.Label(training_frame, text="Skills: none yet")
        self.lbl_skills.grid(row=5, column=0, sticky="w")

        self.lbl_message = tk.Label(training_frame, text="Status: Idle")
        self.lbl_message.grid(row=6, column=0, sticky="w")

        meta_frame = tk.LabelFrame(frame, text="Self-improvement & state", padx=10, pady=10)
        meta_frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_lr = tk.Label(meta_frame, text="Learning rate: --")
        self.lbl_lr.grid(row=0, column=0, sticky="w")

        self.lbl_exploration = tk.Label(meta_frame, text="Exploration: --")
        self.lbl_exploration.grid(row=1, column=0, sticky="w")

        self.lbl_net_size = tk.Label(meta_frame, text="Network hidden size: --")
        self.lbl_net_size.grid(row=2, column=0, sticky="w")

        self.lbl_confidence = tk.Label(meta_frame, text="Confidence: --")
        self.lbl_confidence.grid(row=3, column=0, sticky="w")

        self.lbl_state = tk.Label(meta_frame, text="Mode: exploring")
        self.lbl_state.grid(row=4, column=0, sticky="w")

        ascii_frame = tk.LabelFrame(frame, text="ASCII world view (sample)", padx=10, pady=10)
        ascii_frame.pack(fill=tk.BOTH, expand=True)

        self.ascii_text = tk.Text(ascii_frame, height=10, width=60, state=tk.DISABLED)
        self.ascii_text.pack(fill=tk.BOTH, expand=True)

        status_frame = tk.LabelFrame(frame, text="System log", padx=10, pady=10)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(status_frame, height=8, width=80, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)

        self.log_status("GUI initialized")
        self.log_status(self.mm.last_status)

        self.root.after(500, self.auto_load_on_start)
        self.root.after(500, self.update_training_view)

    def pick_primary(self):
        path = filedialog.askdirectory(title="Select primary memory directory")
        if path:
            self.primary_var.set(path)
            self.mm.set_primary_path(path)
            self.log_status(f"Primary memory path set to: {path}")

    def pick_secondary(self):
        path = filedialog.askdirectory(title="Select secondary backup directory")
        if path:
            self.secondary_var.set(path)
            self.mm.set_secondary_path(path)
            self.log_status(f"Secondary backup path set to: {path}")

    def load_brain(self):
        ok, msg = self.mm.load_brain()
        self.log_status(msg)
        if ok:
            messagebox.showinfo("Load brain", "Brain loaded from memory.")
        else:
            messagebox.showwarning("Load brain", "No existing brain found. Starting fresh.\n\n" + msg)

    def save_brain(self):
        snap = self.stats.snapshot()
        extra_meta = {
            "skills": snap["skills"],
            "avg_reward": snap["avg_reward"],
            "success_rate": snap["success_rate"],
            "difficulty": snap["difficulty"],
            "learning_rate": snap["learning_rate"],
            "exploration": snap["exploration"],
            "network_size": snap["network_size"],
            "confidence": snap["confidence"],
            "state": snap["state"],
        }
        ok, msg = self.mm.save_brain(extra_meta=extra_meta)
        self.log_status(msg)
        if ok:
            messagebox.showinfo("Save brain", "Brain saved (at least one target OK).")
        else:
            messagebox.showerror("Save brain", "Save failed to both primary and secondary.\n\n" + msg)

    def log_status(self, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}\n"
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, line)
        self.status_text.configure(state=tk.DISABLED)
        self.status_text.see(tk.END)

    def auto_load_on_start(self):
        if self.mm.primary_path or self.mm.secondary_path:
            self.log_status("Auto-loading brain from configured memory paths...")
            ok, msg = self.mm.load_brain()
            self.log_status(msg)
        else:
            self.log_status("No memory paths configured yet; please select primary/secondary.")

    def update_training_view(self):
        snap = self.stats.snapshot()
        self.lbl_episode.config(text=f"Episode: {snap['episode']}")
        self.lbl_last_reward.config(text=f"Last reward: {snap['last_reward']:.3f}")
        self.lbl_avg_reward.config(text=f"Avg reward: {snap['avg_reward']:.3f}")
        self.lbl_success_rate.config(text=f"Success rate: {snap['success_rate']:.2%}")
        self.lbl_difficulty.config(text=f"Difficulty: {snap['difficulty']}")
        self.lbl_message.config(text=f"Status: {snap['message']}")

        if snap["skills"]:
            skills_text = ", ".join(
                f"{name} ({data['level']}, {data['confidence']}%)"
                for name, data in snap["skills"].items()
            )
        else:
            skills_text = "none yet"
        self.lbl_skills.config(text=f"Skills: {skills_text}")

        self.lbl_lr.config(text=f"Learning rate: {snap['learning_rate']:.5f}")
        self.lbl_exploration.config(text=f"Exploration: {snap['exploration']:.3f}")
        self.lbl_net_size.config(text=f"Network hidden size: {snap['network_size']}")

        self.lbl_confidence.config(text=f"Confidence: {snap['confidence']:.3f}")
        self.lbl_state.config(text=f"Mode: {snap['state']}")

        self.ascii_text.configure(state=tk.NORMAL)
        self.ascii_text.delete("1.0", tk.END)
        self.ascii_text.insert(tk.END, snap["ascii"])
        self.ascii_text.configure(state=tk.DISABLED)

        self.root.after(500, self.update_training_view)


# =========================================================
# 8. TRAINING LOOP (MULTI-AGENT, Q + PREDICTIVE, STATE-BASED BEHAVIOR)
# =========================================================

def training_loop(env: GridWorldEnv, brain: SimpleBrain,
                  mm: MemoryManager, stats: TrainingStats,
                  self_engine: SelfImprovementEngine,
                  stop_event: threading.Event):
    stats.set_message("Training running")
    stats.running = True

    optimizer = optim.Adam(brain.parameters(), lr=self_engine.learning_rate)
    gamma = 0.95  # discount factor

    rewards_window = []

    while not stop_event.is_set():
        env.set_difficulty(self_engine.difficulty)
        obs = env.reset(stats)  # [N,obs_dim]
        done_env = False
        episode_reward = 0.0

        episode_conf_accum = 0.0
        episode_conf_count = 0

        while not done_env and not stop_event.is_set():
            if env.steps % 10 == 0:
                stats.set_ascii(env.ascii_view())

            obs_tensor = torch.from_numpy(obs)  # [N,obs_dim]
            num_agents = obs.shape[0]

            # estimate confidence on this batch
            conf_batch = estimate_confidence(brain, obs_tensor)
            episode_conf_accum += float(conf_batch.mean())
            episode_conf_count += 1

            with torch.no_grad():
                q_values, _ = brain(obs_tensor)  # [N,4]

            # epsilon depends on current state
            if self_engine.state == ConsciousnessState.EXPLORING:
                epsilon = self_engine.exploration
            elif self_engine.state == ConsciousnessState.FOCUSED:
                epsilon = self_engine.exploration * 0.5
            else:  # CAUTIOUS
                epsilon = self_engine.exploration * 0.8

            actions = np.zeros(num_agents, dtype=int)
            for i in range(num_agents):
                if np.random.rand() < epsilon:
                    actions[i] = np.random.randint(0, 4)
                else:
                    actions[i] = int(torch.argmax(q_values[i]).item())

            next_obs, rewards, dones, done_env = env.step(actions, stats)
            next_obs_tensor = torch.from_numpy(next_obs)

            q_current, pred_current = brain(obs_tensor)
            with torch.no_grad():
                q_next, _ = brain(next_obs_tensor)
                max_q_next, _ = torch.max(q_next, dim=1)

            rewards_t = torch.from_numpy(rewards).float()
            dones_t = torch.from_numpy(dones.astype(np.float32))
            targets = rewards_t + gamma * max_q_next * (1.0 - dones_t)

            q_chosen = q_current[range(num_agents), actions]

            # prediction targets: next state + reward
            rewards_for_pred = rewards_t.unsqueeze(1)  # [N,1]
            pred_target = torch.cat([next_obs_tensor.float(), rewards_for_pred], dim=1)  # [N,obs_dim+1]

            q_loss = ((q_chosen - targets) ** 2).mean()
            pred_loss = ((pred_current - pred_target) ** 2).mean()

            loss = q_loss + 0.1 * pred_loss * self_engine.reward_scale

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_reward_sum = float(np.sum(rewards))
            episode_reward += step_reward_sum / num_agents

            obs = next_obs

        stats.update_episode(episode_reward)
        rewards_window.append(episode_reward)
        if len(rewards_window) > 100:
            rewards_window.pop(0)
        avg_reward = float(np.mean(rewards_window))
        stats.update_avg_reward(avg_reward)

        if episode_conf_count > 0:
            episode_conf = episode_conf_accum / episode_conf_count
        else:
            episode_conf = 0.5
        stats.set_confidence(episode_conf)

        perf = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))
        stats.update_skill("predictive_grid_navigation", perf)

        self_engine.update(stats)
        stats.set_difficulty(self_engine.difficulty)
        for g in optimizer.param_groups:
            g["lr"] = self_engine.learning_rate

        msg = (
            f"Ep {stats.episode} reward={episode_reward:.3f} "
            f"avg={avg_reward:.3f} diff={self_engine.difficulty} "
            f"conf={episode_conf:.3f} state={self_engine.state}"
        )
        print("[TRAIN]", msg)
        stats.set_message(msg)

        if stats.episode % 50 == 0:
            snap = stats.snapshot()
            extra_meta = {
                "skills": snap["skills"],
                "avg_reward": snap["avg_reward"],
                "success_rate": snap["success_rate"],
                "difficulty": snap["difficulty"],
                "learning_rate": snap["learning_rate"],
                "exploration": snap["exploration"],
                "network_size": snap["network_size"],
                "confidence": snap["confidence"],
                "state": snap["state"],
            }
            ok, save_msg = mm.save_brain(extra_meta=extra_meta)
            print("[SAVE]", save_msg)

    stats.running = False
    stats.set_message("Training stopped")


# =========================================================
# 9. MAIN
# =========================================================

def main():
    print("===========================================")
    print("  PREDICTIVE GRID ORGANISM - PYTHON NODE")
    print("===========================================")

    env = GridWorldEnv(num_agents=4)
    brain = SimpleBrain(input_dim=8, hidden_dim=128, output_dim=4)
    mm = MemoryManager(brain)
    stats = TrainingStats()
    self_engine = SelfImprovementEngine()

    stop_event = threading.Event()
    trainer_thread = threading.Thread(
        target=training_loop,
        args=(env, brain, mm, stats, self_engine, stop_event),
        daemon=True
    )
    trainer_thread.start()

    root = tk.Tk()
    gui = BrainGUI(root, brain, mm, stats)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("[MAIN] GUI interrupted.")

    print("[MAIN] GUI closed. Stopping trainer...")
    stop_event.set()
    time.sleep(1.0)
    print("[MAIN] Shutdown complete. Goodbye.")


if __name__ == "__main__":
    main()

