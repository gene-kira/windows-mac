import os
import sys
import json
import time
import threading
from datetime import datetime
import random

# Try to ensure minimal deps
import subprocess
import importlib

REQUIRED_LIBS = ["torch", "numpy", "tk"]

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
# 1. SIMPLE 2D GRID WORLD ENVIRONMENT
# =========================================================

class GridWorldEnv:
    """
    Simple 2D grid environment.
    Agent moves in a square grid, tries to reach target.
    Difficulty increases by enlarging the grid and adding step penalty.
    """

    def __init__(self, base_size=5, max_size=15):
        self.base_size = base_size
        self.max_size = max_size
        self.size = base_size

        self.agent_pos = None
        self.target_pos = None
        self.max_steps = None
        self.steps = 0

    def set_difficulty(self, difficulty_level):
        # Map difficulty 1-10 to grid size and max steps
        difficulty_level = max(1, min(10, difficulty_level))
        self.size = min(self.base_size + difficulty_level - 1, self.max_size)
        self.max_steps = self.size * self.size * 2

    def reset(self):
        self.steps = 0
        # random positions, not equal
        self.agent_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        while True:
            self.target_pos = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if self.target_pos != self.agent_pos:
                break
        return self._get_obs()

    def _get_obs(self):
        # Observation: [agent_x, agent_y, target_x, target_y] normalized to [0,1]
        ax = self.agent_pos[0] / (self.size - 1)
        ay = self.agent_pos[1] / (self.size - 1)
        tx = self.target_pos[0] / (self.size - 1)
        ty = self.target_pos[1] / (self.size - 1)
        return np.array([ax, ay, tx, ty], dtype=np.float32)

    def step(self, action):
        """
        Action: integer 0-3: up, down, left, right.
        Reward:
          +1.0 on reaching target
          -0.01 per step
        Episode ends on target or max_steps.
        """
        self.steps += 1
        # Move agent
        if action == 0:   # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1: # down
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2: # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3: # right
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        done = False
        reward = -0.01  # step penalty

        if self.agent_pos == self.target_pos:
            reward = 1.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        obs = self._get_obs()
        return obs, reward, done


# =========================================================
# 2. NEURAL BRAIN (DENSE NETWORK)
# =========================================================

class SimpleBrain(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)

    def rebuild(self, input_dim, output_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        self.__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


# =========================================================
# 3. MEMORY MANAGER (PRIMARY + SECONDARY PATHS)
# =========================================================

class MemoryManager:
    def __init__(self, brain, config_path=None):
        self.brain = brain

        if config_path is None:
            home = os.path.expanduser("~")
            config_dir = os.path.join(home, ".living_grid_brain")
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
            "model_type": "SimpleBrainGrid",
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
# 4. TRAINING STATS + SKILL TRACKER
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
        self.network_size = 64

    def update_episode(self, episode_reward, success_threshold=0.9):
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
            }


# =========================================================
# 5. SELF-IMPROVEMENT ENGINE
# =========================================================

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
        self.network_hidden_dim = 64

        self._last_avg_reward = None
        self._plateau_count = 0

    def update(self, stats: TrainingStats):
        snap = stats.snapshot()
        avg_reward = snap["avg_reward"]
        success_rate = snap["success_rate"]

        if self._last_avg_reward is None:
            self._last_avg_reward = avg_reward
        delta = avg_reward - self._last_avg_reward
        self._last_avg_reward = avg_reward

        # Learning rate
        if delta > 0.01:
            self.learning_rate = max(self.min_lr, self.learning_rate * 0.95)
        elif delta < -0.01:
            self.learning_rate = min(self.max_lr, self.learning_rate * 1.05)

        # Exploration
        if success_rate < 0.3:
            self.exploration = min(self.max_exploration, self.exploration + 0.01)
        elif success_rate > 0.7:
            self.exploration = max(self.min_exploration, self.exploration - 0.01)

        # Difficulty
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

        stats.set_meta_params(
            learning_rate=self.learning_rate,
            exploration=self.exploration,
            network_size=self.network_hidden_dim
        )


# =========================================================
# 6. GUI
# =========================================================

class BrainGUI:
    def __init__(self, root, brain, memory_manager, stats: TrainingStats):
        self.root = root
        self.brain = brain
        self.mm = memory_manager
        self.stats = stats

        self.root.title("Grid Organism - Brain, Memory & Evolution Console")

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

        meta_frame = tk.LabelFrame(frame, text="Self-improvement engine", padx=10, pady=10)
        meta_frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_lr = tk.Label(meta_frame, text="Learning rate: --")
        self.lbl_lr.grid(row=0, column=0, sticky="w")

        self.lbl_exploration = tk.Label(meta_frame, text="Exploration: --")
        self.lbl_exploration.grid(row=1, column=0, sticky="w")

        self.lbl_net_size = tk.Label(meta_frame, text="Network hidden size: --")
        self.lbl_net_size.grid(row=2, column=0, sticky="w")

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

        self.root.after(500, self.update_training_view)


# =========================================================
# 7. TRAINING LOOP (PURE PYTHON)
# =========================================================

def training_loop(env: GridWorldEnv, brain: SimpleBrain,
                  mm: MemoryManager, stats: TrainingStats,
                  self_engine: SelfImprovementEngine,
                  stop_event: threading.Event):
    """
    Very simple on-policy learning:
    - epsilon/exploration controls random vs argmax actions
    - reward directly drives a pseudo-loss
    It's not a full RL algorithm, but enough for visible self-improvement.
    """
    stats.set_message("Training running")
    stats.running = True

    optimizer = optim.Adam(brain.parameters(), lr=self_engine.learning_rate)

    rewards_window = []

    while not stop_event.is_set():
        # Sync difficulty from meta-engine into env
        env.set_difficulty(self_engine.difficulty)

        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done and not stop_event.is_set():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            if np.random.rand() < self_engine.exploration:
                action = np.random.randint(0, 4)
            else:
                with torch.no_grad():
                    logits = brain(obs_tensor)
                    action = int(torch.argmax(logits, dim=1).item())

            new_obs, reward, done = env.step(action)

            # Simple "loss": drive policy towards rewarding states
            loss_value = -reward * self_engine.reward_scale
            optimizer.zero_grad()
            loss_tensor = torch.tensor(loss_value, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()

            episode_reward += reward
            obs = new_obs

        # Episode end
        stats.update_episode(episode_reward)
        rewards_window.append(episode_reward)
        if len(rewards_window) > 100:
            rewards_window.pop(0)
        avg_reward = float(np.mean(rewards_window))
        stats.update_avg_reward(avg_reward)

        # map avg_reward roughly into [0,1] for skill
        perf = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))
        stats.update_skill("grid_navigation", perf)

        # self-improvement
        self_engine.update(stats)
        stats.set_difficulty(self_engine.difficulty)
        for g in optimizer.param_groups:
            g["lr"] = self_engine.learning_rate

        msg = (
            f"Ep {stats.episode} reward={episode_reward:.3f} "
            f"avg={avg_reward:.3f} diff={self_engine.difficulty}"
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
            }
            ok, save_msg = mm.save_brain(extra_meta=extra_meta)
            print("[SAVE]", save_msg)

    stats.running = False
    stats.set_message("Training stopped")


# =========================================================
# 8. MAIN
# =========================================================

def main():
    print("===========================================")
    print("  GRID ORGANISM - PURE PYTHON NODE START")
    print("===========================================")

    env = GridWorldEnv()
    brain = SimpleBrain(input_dim=4, hidden_dim=64, output_dim=4)
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

