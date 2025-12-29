import json
import os
import random
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from collections import defaultdict, deque

# ============================================================
#  ASSET MODEL
# ============================================================

class Asset:
    def __init__(self, asset_id, size_mb, kind="generic", meta=None):
        self.id = asset_id                    # string: path or logical ID
        self.size_mb = size_mb
        self.kind = kind                      # "texture", "map", "tab", "video", ...
        self.tier = "disk"                    # "disk", "ram", "vram"
        self.access_count = 0.0
        self.last_access_time = 0.0
        self.meta = meta or {}                # e.g. {"path": "...", "url": "..."}

    def record_access(self, t):
        self.access_count += 1.0
        self.last_access_time = t


# ============================================================
#  MEMORY TIERS
# ============================================================

class Tier:
    def __init__(self, name, capacity_mb):
        self.name = name
        self.capacity_mb = capacity_mb
        self.used_mb = 0.0
        self.assets = set()  # asset IDs

    def can_fit(self, size_mb):
        return self.used_mb + size_mb <= self.capacity_mb

    def add_asset(self, asset):
        if asset.id not in self.assets:
            self.assets.add(asset.id)
            self.used_mb += asset.size_mb

    def remove_asset(self, asset):
        if asset.id in self.assets:
            self.assets.remove(asset.id)
            self.used_mb -= asset.size_mb


# ============================================================
#  SYSTEM HEALTH / SITUATIONAL AWARENESS
# ============================================================

class SystemHealth:
    def __init__(self):
        self.total_accesses = 0
        self.vram_hits = 0
        self.ram_hits = 0
        self.disk_hits = 0
        self.eviction_regrets = 0
        self.last_evicted = {}  # asset_id -> time
        self.window_start = time.time()
        self.window_duration = 10.0  # rolling window

    def record_access(self, asset, tier_name, t):
        self.total_accesses += 1
        if tier_name == "vram":
            self.vram_hits += 1
        elif tier_name == "ram":
            self.ram_hits += 1
        else:
            self.disk_hits += 1

        if asset.id in self.last_evicted:
            age = t - self.last_evicted[asset.id]
            if age < 5.0:  # regret: needed right after eviction
                self.eviction_regrets += 1
                del self.last_evicted[asset.id]

    def record_eviction(self, asset_id, t):
        self.last_evicted[asset_id] = t

    def reset_window_if_needed(self):
        now = time.time()
        if now - self.window_start > self.window_duration:
            self.total_accesses = 0
            self.vram_hits = 0
            self.ram_hits = 0
            self.disk_hits = 0
            self.eviction_regrets = 0
            self.last_evicted = {}
            self.window_start = now

    def health_score(self, vram_used, vram_capacity):
        self.reset_window_if_needed()
        if self.total_accesses == 0:
            return 100.0

        vram_hit_rate = self.vram_hits / max(1, self.total_accesses)
        ram_hit_rate = self.ram_hits / max(1, self.total_accesses)
        regret_rate = self.eviction_regrets / max(1, self.total_accesses)

        util = vram_used / max(1.0, vram_capacity)
        if util < 0.3:
            util_score = util / 0.3
        elif util <= 0.8:
            util_score = 1.0
        else:
            util_score = max(0.0, (1.0 - (util - 0.8) / 0.2))

        base = 0.6 * vram_hit_rate + 0.3 * ram_hit_rate + 0.1 * util_score
        penalty = 0.5 * regret_rate
        score = (base - penalty) * 100.0
        return max(0.0, min(100.0, score))


# ============================================================
#  TRACE LOGGER (ONLINE LEARNING HOOK)
# ============================================================

class TraceLogger:
    """
    Online trace logger + simple per-pattern reward stats.
    """

    def __init__(self, path="traces.log"):
        self.path = path
        self.lock = threading.Lock()
        self.pattern_stats = defaultdict(lambda: {"count": 0.0, "reward": 0.0})

    def log_decision(self, event):
        ts = time.time()
        event = dict(event)
        event["ts"] = ts
        line = json.dumps(event, separators=(",", ":"))
        with self.lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def update_reward(self, pattern_key, reward):
        s = self.pattern_stats[pattern_key]
        s["count"] += 1.0
        s["reward"] += reward

    def get_pattern_score(self, pattern_key):
        s = self.pattern_stats.get(pattern_key)
        if not s or s["count"] == 0:
            return 0.0
        return s["reward"] / s["count"]


# ============================================================
#  HYBRID AI GOVERNOR (OVERMIND)
# ============================================================

class AIGovernor:
    """
    Hybrid overmind:
      - Single hybrid mode
      - Predictive sequences (short/med/long)
      - Phase detection (menu/loading/active)
      - Behavior detection (gaming-like / general-like)
      - Adaptive thresholds
      - Regret-pattern bias per asset kind
      - Hardware-aware risk profiles
      - Bandit across 3 policies
      - Opportunity windows
      - Health feedback
      - Online pattern-based "learned" corrector via TraceLogger
    """

    def __init__(self, assets, tier_disk, tier_ram, tier_vram,
                 health=None, log_fn=None, hw_profile=None, trace_logger=None):
        self.assets = assets
        self.tier_disk = tier_disk
        self.tier_ram = tier_ram
        self.tier_vram = tier_vram
        self.now = time.time
        self.log_fn = log_fn or (lambda *_: None)

        # "internal mode" just for scoring bias; user doesn't select it
        self.mode = "hybrid"

        self.health = health or SystemHealth()
        self.trace = trace_logger or TraceLogger()

        self.hw_profile = hw_profile or "balanced"
        self._init_base_policy_for_hw()

        self.hot_threshold = self.base_hot_threshold
        self.warm_threshold = self.base_warm_threshold

        # Predictive cortex
        self.transition_counts = defaultdict(lambda: {
            "short": defaultdict(int),
            "med": defaultdict(int),
            "long": defaultdict(int)
        })
        self.recent_accesses = deque(maxlen=32)  # (asset, time)

        # Regret-pattern bias: kind -> regret count
        self.kind_regret_counts = defaultdict(int)

        # Phase detection
        self.phase = "unknown"        # menu/loading/active
        self.behavior = "unknown"     # gaming-like / general-like
        self.phase_history = deque(maxlen=50)

        # Policy bandit
        self.policies = ["aggressive", "conservative", "predictive"]
        self.policy_rewards = {p: 0.0 for p in self.policies}
        self.policy_counts = {p: 0.0 for p in self.policies}
        self.current_policy = random.choice(self.policies)
        self.last_policy_switch = self.now()
        self.policy_window = 5.0
        self.epsilon = 0.2

        self.score_history = deque(maxlen=200)

        # Opportunity windows
        self.last_access_time_global = self.now()
        self.last_access_delta = 0.0

        # Learned trust in corrector vs rules
        self.trust_learned = 0.3

    def _init_base_policy_for_hw(self):
        # Base thresholds tuned by HW profile; hybrid logic decides kind bias later
        if self.hw_profile == "bold":
            self.base_hot_threshold = 3.5
            self.base_warm_threshold = 1.2
            self.decay = 0.975
        elif self.hw_profile == "conservative":
            self.base_hot_threshold = 4.5
            self.base_warm_threshold = 1.8
            self.decay = 0.99
        else:
            self.base_hot_threshold = 4.0
            self.base_warm_threshold = 1.5
            self.decay = 0.98

    # ---------- prediction & scoring (time-aware) ----------

    def record_access_for_prediction(self, asset):
        t = self.now()
        if self.recent_accesses:
            prev_asset, prev_t = self.recent_accesses[-1]
            dt = t - prev_t
            if dt < 0.5:
                bucket = "short"
            elif dt < 2.0:
                bucket = "med"
            else:
                bucket = "long"
            self.transition_counts[prev_asset.id][bucket][asset.id] += 1
        self.recent_accesses.append((asset, t))

        self.last_access_delta = t - self.last_access_time_global
        self.last_access_time_global = t

    def predicted_followers(self, asset_id, horizon="short", top_k=3):
        buckets = self.transition_counts.get(asset_id, {})
        counts = buckets.get(horizon, {})
        if not counts:
            return []
        sorted_ids = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [aid for aid, _ in sorted_ids[:top_k]]

    def _base_asset_score(self, asset):
        age = max(0.001, self.now() - asset.last_access_time)
        base = asset.access_count / age

        # Behavior-based bias instead of fixed mode selection
        if self.behavior == "gaming-like":
            if asset.kind in ("texture", "map"):
                base *= 1.2
        elif self.behavior == "general-like":
            if asset.kind in ("tab", "video"):
                base *= 1.2

        regret_bias = 1.0 + min(0.5, self.kind_regret_counts[asset.kind] * 0.02)
        base *= regret_bias

        return base

    def _score_asset(self, asset):
        return self._base_asset_score(asset)

    def promotion_confidence_rules(self, asset):
        base = self._base_asset_score(asset)

        followers_short = self.predicted_followers(asset.id, "short", top_k=3)
        followers_med = self.predicted_followers(asset.id, "med", top_k=2)

        bonus = 0.0
        for fid in followers_short + followers_med:
            fasset = next((a for a in self.assets if a.id == fid), None)
            if fasset:
                bonus += self._base_asset_score(fasset)

        raw = base + 0.4 * bonus

        if self.phase == "loading":
            raw *= 1.2
        elif self.phase == "menu":
            raw *= 0.9

        return max(0.0, min(1.0, raw / 12.0))

    # ---------- learned correction layer ----------

    def pattern_key(self, asset, decision_type):
        return (
            "hybrid",
            self.phase,
            self.behavior,
            self.hw_profile,
            self.current_policy,
            asset.kind,
            decision_type,
        )

    def learned_adjustment(self, asset, decision_type):
        key = self.pattern_key(asset, decision_type)
        score = self.trace.get_pattern_score(key)
        score = max(-5.0, min(5.0, score))
        return (score / 5.0) * 0.2

    def combine_confidence(self, asset, decision_type, base_conf):
        adj = self.learned_adjustment(asset, decision_type)
        learned_conf = max(0.0, min(1.0, base_conf + adj))
        combined = (1 - self.trust_learned) * base_conf + self.trust_learned * learned_conf
        return max(0.0, min(1.0, combined))

    def update_trust_from_health(self):
        if len(self.score_history) < 10:
            return
        recent = list(self.score_history)[-10:]
        avg = sum(recent) / len(recent)
        if avg > 85:
            self.trust_learned = min(1.0, self.trust_learned + 0.01)
        elif avg < 60:
            self.trust_learned = max(0.0, self.trust_learned - 0.01)

    # ---------- phase + behavior detection ----------

    def update_phase_and_behavior(self):
        maps = [a for a in self.assets if a.kind == "map"]
        textures = [a for a in self.assets if a.kind == "texture"]
        tabs = [a for a in self.assets if a.kind == "tab"]
        videos = [a for a in self.assets if a.kind == "video"]

        now = self.now()

        recent_maps = sum(1 for a in maps if now - a.last_access_time < 2.0)
        recent_textures = sum(1 for a in textures if now - a.last_access_time < 2.0)
        recent_tabs = sum(1 for a in tabs if now - a.last_access_time < 2.0)
        recent_videos = sum(1 for a in videos if now - a.last_access_time < 2.0)

        # phase
        new_phase = self.phase
        if recent_maps > 5:
            new_phase = "loading"
        elif (recent_tabs + recent_videos) > 5 and recent_maps < 2:
            new_phase = "active"
        elif recent_textures < 3 and recent_maps < 2 and (recent_tabs + recent_videos) < 3:
            new_phase = "menu"
        else:
            new_phase = "active"

        # behavior (gaming-like vs general-like)
        if (recent_maps + recent_textures) > (recent_tabs + recent_videos):
            new_behavior = "gaming-like"
        elif (recent_tabs + recent_videos) > (recent_maps + recent_textures):
            new_behavior = "general-like"
        else:
            new_behavior = "mixed"

        changed = (new_phase != self.phase) or (new_behavior != self.behavior)
        self.phase = new_phase
        self.behavior = new_behavior
        if changed:
            self.log_fn(f"[STATE] Phase={self.phase}, Behavior={self.behavior}")

        self.phase_history.append(self.phase)

    def is_opportunity_window(self):
        return self.last_access_delta > 1.0

    # ---------- adaptive thresholds ----------

    def update_adaptive_thresholds(self):
        if not self.assets:
            return
        scores = [self._score_asset(a) for a in self.assets]
        scores_sorted = sorted(scores)
        n = len(scores_sorted)

        def percentile(p):
            if n == 0:
                return 0.0
            k = int(p * (n - 1))
            return scores_sorted[k]

        hot_p = percentile(0.8)
        warm_p = percentile(0.5)

        self.hot_threshold = 0.5 * self.base_hot_threshold + 0.5 * hot_p
        self.warm_threshold = 0.5 * self.base_warm_threshold + 0.5 * warm_p
        self.hot_threshold = max(self.warm_threshold + 0.1, self.hot_threshold)

    def apply_decay(self):
        for asset in self.assets:
            asset.access_count *= self.decay

    # ---------- policy bandit ----------

    def _policy_reward_snapshot(self):
        score = self.health.health_score(self.tier_vram.used_mb, self.tier_vram.capacity_mb)
        self.score_history.append(score)
        self.update_trust_from_health()
        return score

    def maybe_switch_policy(self):
        now = self.now()
        if now - self.last_policy_switch < self.policy_window:
            return

        reward = self._policy_reward_snapshot()
        self.policy_rewards[self.current_policy] += reward
        self.policy_counts[self.current_policy] += 1.0

        if random.random() < self.epsilon:
            new_policy = random.choice(self.policies)
        else:
            best_pol = None
            best_val = -1e9
            for p in self.policies:
                c = self.policy_counts[p]
                avg = self.policy_rewards[p] / max(1.0, c)
                if c == 0:
                    avg += 5.0
                if avg > best_val:
                    best_val = avg
                    best_pol = p
            new_policy = best_pol

        self.current_policy = new_policy
        self.last_policy_switch = now
        self.log_fn(f"[BANDIT] Switched policy to: {self.current_policy} (reward={reward:.1f})")

    # ---------- decision making ----------

    def decide_moves(self):
        self.update_adaptive_thresholds()
        self.update_phase_and_behavior()

        moves = []
        sorted_assets = sorted(self.assets, key=self._score_asset, reverse=True)

        for asset in sorted_assets:
            score = self._score_asset(asset)

            if self.current_policy == "aggressive":
                hot_thr = self.hot_threshold * 0.9
                warm_thr = self.warm_threshold * 0.9
            elif self.current_policy == "conservative":
                hot_thr = self.hot_threshold * 1.1
                warm_thr = self.warm_threshold * 1.1
            else:
                hot_thr = self.hot_threshold
                warm_thr = self.warm_threshold

            if self.phase == "loading":
                hot_thr *= 0.9
            elif self.phase == "menu":
                hot_thr *= 1.1

            if score >= hot_thr:
                desired = "vram"
            elif score >= warm_thr:
                desired = "ram"
            else:
                desired = "disk"

            if asset.tier == desired:
                continue

            moves.append((asset, desired))

        return moves

    def _get_tier_by_name(self, name):
        if name == "disk": return self.tier_disk
        if name == "ram": return self.tier_ram
        if name == "vram": return self.tier_vram
        return None

    def execute_moves(self, moves):
        for asset, desired in moves:
            current_tier = asset.tier
            if current_tier == desired:
                continue

            base_conf = self.promotion_confidence_rules(asset)
            decision_type = f"{current_tier}_to_{desired}"
            conf = self.combine_confidence(asset, decision_type, base_conf)

            vram_util = self.tier_vram.used_mb / max(1.0, self.tier_vram.capacity_mb)

            if desired == "vram":
                if self.hw_profile == "conservative":
                    util_limit = 0.85
                elif self.hw_profile == "bold":
                    util_limit = 0.95
                else:
                    util_limit = 0.9

                if vram_util > util_limit and conf < 0.6:
                    self._log_decision(asset, decision_type, conf, vram_util, "skip")
                    self.log_fn(f"[HYBRID][{self.current_policy}] skip: {asset.id} -> vram "
                                f"(conf={conf:.2f}, util={vram_util:.2f})")
                    continue

            src_tier = self._get_tier_by_name(current_tier)
            dst_tier = self._get_tier_by_name(desired)
            if src_tier is None or dst_tier is None:
                continue

            if not dst_tier.can_fit(asset.size_mb):
                self._evict_for(dst_tier, needed_mb=asset.size_mb)

            if dst_tier.can_fit(asset.size_mb):
                src_tier.remove_asset(asset)
                dst_tier.add_asset(asset)
                asset.tier = desired
                self._log_decision(asset, decision_type, conf, vram_util, "move")
                self.log_fn(
                    f"[HYBRID][{self.current_policy}] move: {asset.id} ({asset.kind}) "
                    f"{current_tier} -> {desired} | conf={conf:.2f}"
                )

    def _evict_for(self, tier, needed_mb):
        if tier.name == "disk":
            return

        lower_name = "disk" if tier.name == "ram" else "ram"
        lower_tier = self._get_tier_by_name(lower_name)

        assets_in_tier = [a for a in self.assets if a.tier == tier.name]
        assets_in_tier.sort(key=self._score_asset)

        now = self.now()
        for asset in assets_in_tier:
            if tier.used_mb + needed_mb <= tier.capacity_mb:
                break

            tier.remove_asset(asset)
            if self.health:
                self.health.record_eviction(asset.id, now)

            decision_type = f"{tier.name}_to_{lower_name if lower_tier.can_fit(asset.size_mb) else 'disk'}"

            if lower_tier.can_fit(asset.size_mb):
                lower_tier.add_asset(asset)
                asset.tier = lower_name
                self._log_decision(asset, decision_type, 0.5, tier.used_mb / max(1.0, tier.capacity_mb), "evict")
                self.log_fn(f"[HYBRID][{self.current_policy}] evict: {asset.id} ({asset.kind}) "
                            f"{tier.name} -> {lower_name}")
            else:
                if lower_name != "disk":
                    lower_tier.remove_asset(asset)
                asset.tier = "disk"
                self._log_decision(asset, decision_type, 0.3, tier.used_mb / max(1.0, tier.capacity_mb), "drop")
                self.log_fn(f"[HYBRID][{self.current_policy}] drop: {asset.id} ({asset.kind}) -> disk")

    def _log_decision(self, asset, decision_type, conf, vram_util, action):
        health_now = self.health.health_score(self.tier_vram.used_mb, self.tier_vram.capacity_mb)
        event = {
            "mode": "hybrid",
            "phase": self.phase,
            "behavior": self.behavior,
            "hw_profile": self.hw_profile,
            "policy": self.current_policy,
            "kind": asset.kind,
            "decision_type": decision_type,
            "action": action,
            "conf": round(conf, 3),
            "vram_util": round(vram_util, 3),
            "health": round(health_now, 1),
        }
        self.trace.log_decision(event)
        key = self.pattern_key(asset, decision_type)
        reward = (health_now - 50.0) / 50.0  # -1..+1 approx
        self.trace.update_reward(key, reward)


# ============================================================
#  SIMULATOR
# ============================================================

class GameSimulator:
    def __init__(self, assets, governor, health, log_fn):
        self.assets = assets
        self.governor = governor
        self.health = health
        self.log_fn = log_fn
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        with self.lock:
            self.running = False

    def _loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            now = time.time()
            self._simulate_hybrid_access(now)

            self.governor.apply_decay()
            self.governor.maybe_switch_policy()
            moves = self.governor.decide_moves()
            self.governor.execute_moves(moves)

            time.sleep(0.5)

    def _simulate_hybrid_access(self, now):
        textures = [a for a in self.assets if a.kind == "texture"]
        maps = [a for a in self.assets if a.kind == "map"]
        tabs = [a for a in self.assets if a.kind == "tab"]
        videos = [a for a in self.assets if a.kind == "video"]
        others = [a for a in self.assets if a.kind not in ("texture", "map", "tab", "video")]

        for _ in range(random.randint(8, 25)):
            r = random.random()
            if r < 0.3 and (textures or maps):
                # gaming-like
                if r < 0.2 and textures:
                    asset = random.choice(textures)
                elif maps:
                    asset = random.choice(maps)
                else:
                    asset = random.choice(self.assets)
            elif r < 0.7 and (tabs or videos):
                # general-like
                if r < 0.5 and tabs:
                    asset = random.choice(tabs)
                elif videos:
                    asset = random.choice(videos)
                else:
                    asset = random.choice(self.assets)
            else:
                if others:
                    asset = random.choice(others)
                else:
                    asset = random.choice(self.assets)

            asset.record_access(now)
            self.governor.record_access_for_prediction(asset)
            self.health.record_access(asset, asset.tier, now)


# ============================================================
#  PROFILE PERSISTENCE (JSON)
# ============================================================

class ProfileManager:
    def __init__(self, base_dir="profiles"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.last_profile_path = os.path.join(self.base_dir, "last_profile.json")

    def _profile_path(self, name):
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        return os.path.join(self.base_dir, f"memory_profile_{safe_name}.json")

    def save_profile(self, name, assets, governor, meta, auto_start=True):
        path = self._profile_path(name)
        data = {
            "version": 6,
            "profile_name": name,
            "meta": meta,
            "auto_start": auto_start,
            "policy": {
                "base_hot_threshold": governor.base_hot_threshold,
                "base_warm_threshold": governor.base_warm_threshold,
                "decay": governor.decay,
                "hw_profile": governor.hw_profile,
            },
            "assets": [],
        }

        now = time.time()
        for a in assets:
            data["assets"].append({
                "id": a.id,
                "size_mb": a.size_mb,
                "kind": a.kind,
                "tier": a.tier,
                "access_count": a.access_count,
                "last_access_age": max(0.0, now - a.last_access_time),
                "meta": a.meta,
            })

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save profile: {e}")

        # update last_profile.json
        last = {
            "last_profile_name": name,
            "auto_start": auto_start
        }
        try:
            with open(self.last_profile_path, "w", encoding="utf-8") as f:
                json.dump(last, f, indent=2)
        except Exception as e:
            print(f"Failed to save last_profile: {e}")

    def load_profile(self, name):
        path = self._profile_path(name)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load profile: {e}")
            return None

    def load_last_profile_name(self):
        if not os.path.exists(self.last_profile_path):
            return None, True
        try:
            with open(self.last_profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("last_profile_name", "default"), bool(data.get("auto_start", True))
        except Exception:
            return None, True

    def list_profiles(self):
        profiles = []
        for fname in os.listdir(self.base_dir):
            if fname.startswith("memory_profile_") and fname.endswith(".json"):
                core = fname[len("memory_profile_"):-5]
                profiles.append(core)
        return sorted(profiles)


# ============================================================
#  REAL GAMING: FILE SCAN + PREFETCH
# ============================================================

def scan_game_folder(root_path, log_fn):
    assets = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            try:
                size_bytes = os.path.getsize(full)
            except Exception:
                continue
            size_mb = max(1, int(size_bytes / (1024 * 1024)))
            ext = os.path.splitext(fname)[1].lower()

            kind = "file"
            if ext in (".pak", ".bundle", ".idx", ".map"):
                kind = "map"
            elif ext in (".dds", ".png", ".jpg", ".jpeg", ".tga"):
                kind = "texture"
            elif ext in (".bin", ".dat"):
                kind = "misc"

            asset = Asset(asset_id=full, size_mb=size_mb, kind=kind,
                          meta={"path": full})
            assets.append(asset)

    log_fn(f"Scanned game folder: {root_path} -> {len(assets)} files.")
    return assets


def prewarm_os_cache_for_assets(assets, count, log_fn):
    top = assets[:count]
    log_fn(f"Pre-warming OS cache for {len(top)} assets...")
    for a in top:
        path = a.meta.get("path")
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                _ = f.read(1024 * 1024)  # sample first MB
            log_fn(f"  touched: {os.path.basename(path)} ({a.size_mb} MB)")
        except Exception as e:
            log_fn(f"  failed: {path} ({e})")


# ============================================================
#  TKINTER HUD + GAME LIBRARY
# ============================================================

class MemoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Memory Overmind - Unified Hybrid Cortex + Game Library")
        self.root.geometry("1350x800")

        self.profile_mgr = ProfileManager()
        self.profile_name = "default"

        self.game_root = None
        self.game_exe = None

        self.hw_profile_var = tk.StringVar(value="balanced")

        self.tier_disk = Tier("disk", capacity_mb=20_000)
        self.tier_ram = Tier("ram", capacity_mb=8_000)
        self.tier_vram = Tier("vram", capacity_mb=2_000)

        self.assets = []
        self.health = SystemHealth()
        self.trace = TraceLogger()
        self.governor = None
        self.simulator = None

        self.meta = {"game_root": None, "game_exe": None}
        self.auto_start = True

        self._suppress_game_select_event = False

        self._init_ui()
        self._load_last_profile_and_state()

        self.root.after(500, self._refresh_ui)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # auto-start AI core if auto_start flag is true
        if self.auto_start:
            self.start_simulation()

    # ---------- setup ----------

    def _load_last_profile_and_state(self):
        name, auto = self.profile_mgr.load_last_profile_name()
        if not name:
            name = "default"
        self.profile_name = name
        self.auto_start = auto

        profile = self.profile_mgr.load_profile(self.profile_name)
        if profile is None:
            self._log("No existing profile; creating fresh hybrid assets.")
            self.assets = self._create_placeholder_assets()
            self._reset_tiers()
            for a in self.assets:
                self.tier_disk.add_asset(a)
                a.tier = "disk"
            hw_profile = self.hw_profile_var.get()
            self.governor = AIGovernor(
                self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
                health=self.health, log_fn=self._log,
                hw_profile=hw_profile, trace_logger=self.trace
            )
            self.meta = {"game_root": None, "game_exe": None}
        else:
            self._log(f"Loaded profile '{self.profile_name}'; reconstructing assets and tiers...")
            self._load_from_profile(profile)

        self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)
        self._log(f"Hybrid cortex ready with profile '{self.profile_name}'.")
        self._refresh_game_list(selected=self.profile_name)

    def _reset_tiers(self):
        for t in (self.tier_disk, self.tier_ram, self.tier_vram):
            t.assets.clear()
            t.used_mb = 0.0

    def _load_from_profile(self, profile):
        self.assets = []
        all_assets_data = profile.get("assets", [])
        now = time.time()

        self._reset_tiers()
        self.meta = profile.get("meta", {"game_root": None, "game_exe": None})
        self.game_root = self.meta.get("game_root")
        self.game_exe = self.meta.get("game_exe")

        for a_data in all_assets_data:
            a = Asset(
                a_data["id"],
                a_data["size_mb"],
                kind=a_data.get("kind", "generic"),
                meta=a_data.get("meta", {}),
            )
            a.access_count = float(a_data.get("access_count", 0.0))
            age = float(a_data.get("last_access_age", 0.0))
            a.last_access_time = now - age
            desired_tier = a_data.get("tier", "disk")

            self.assets.append(a)

            tier_obj = self._tier_by_name(desired_tier)
            if tier_obj is None or not tier_obj.can_fit(a.size_mb):
                if desired_tier == "vram":
                    tier_obj = self.tier_ram if self.tier_ram.can_fit(a.size_mb) else self.tier_disk
                elif desired_tier == "ram":
                    tier_obj = self.tier_disk
                else:
                    tier_obj = self.tier_disk

            tier_obj.add_asset(a)
            a.tier = tier_obj.name

        pol = profile.get("policy", {})
        hw_profile = pol.get("hw_profile", self.hw_profile_var.get())
        self.hw_profile_var.set(hw_profile)

        self.governor = AIGovernor(
            self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
            health=self.health, log_fn=self._log,
            hw_profile=hw_profile, trace_logger=self.trace
        )

        self.governor.base_hot_threshold = float(pol.get("base_hot_threshold", self.governor.base_hot_threshold))
        self.governor.base_warm_threshold = float(pol.get("base_warm_threshold", self.governor.base_warm_threshold))
        self.governor.decay = float(pol.get("decay", self.governor.decay))

        self._log(
            f"Profile restored: {len(self.assets)} assets. "
            f"BasePolicy: hot={self.governor.base_hot_threshold:.2f}, "
            f"warm={self.governor.base_warm_threshold:.2f}, decay={self.governor.decay:.3f}, "
            f"HW={self.governor.hw_profile}"
        )

    def _tier_by_name(self, name):
        if name == "disk": return self.tier_disk
        if name == "ram": return self.tier_ram
        if name == "vram": return self.tier_vram
        return None

    def _create_placeholder_assets(self):
        assets = []
        # some gaming-ish assets
        for i in range(15):
            size_mb = random.choice([10, 20, 50])
            assets.append(Asset(f"texture_{i}", size_mb, kind="texture"))
        for i in range(5):
            size_mb = random.choice([100, 200])
            assets.append(Asset(f"map_{i}", size_mb, kind="map"))
        # some general-ish assets
        for i in range(10):
            size_mb = random.choice([5, 10, 15])
            assets.append(Asset(f"tab_{i}", size_mb, kind="tab"))
        for i in range(10):
            size_mb = random.choice([20, 50])
            assets.append(Asset(f"video_seg_{i}", size_mb, kind="video"))
        return assets

    def _switch_hw_profile(self, *_):
        # Changing HW profile should re-init governor but keep assets/tier state
        hw_profile = self.hw_profile_var.get()
        self._log(f"Switching HW profile to {hw_profile}; preserving assets and tiers.")
        self.governor = AIGovernor(
            self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
            health=self.health, log_fn=self._log,
            hw_profile=hw_profile, trace_logger=self.trace
        )

    # ---------- UI ----------

    def _init_ui(self):
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        # top row: left (hybrid + game integration + controls + HUD) and right (game library)
        top = ttk.Frame(main)
        top.pack(fill=tk.X)

        left_top = ttk.Frame(top)
        left_top.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Hybrid cortex panel
        hybrid_frame = ttk.LabelFrame(left_top, text="Hybrid Cortex", padding=5)
        hybrid_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.behavior_label = ttk.Label(hybrid_frame, text="Behavior: unknown", font=("Consolas", 10, "bold"))
        self.behavior_label.pack(anchor="w")

        self.phase_label = ttk.Label(hybrid_frame, text="Phase: unknown", font=("Consolas", 9))
        self.phase_label.pack(anchor="w", pady=(5, 0))

        ttk.Label(hybrid_frame, text="HW Profile:").pack(anchor="w", pady=(10, 0))
        ttk.Radiobutton(hybrid_frame, text="Conservative", value="conservative",
                        variable=self.hw_profile_var, command=self._switch_hw_profile).pack(anchor="w")
        ttk.Radiobutton(hybrid_frame, text="Balanced", value="balanced",
                        variable=self.hw_profile_var, command=self._switch_hw_profile).pack(anchor="w")
        ttk.Radiobutton(hybrid_frame, text="Bold", value="bold",
                        variable=self.hw_profile_var, command=self._switch_hw_profile).pack(anchor="w")

        # Gaming integration
        game_frame = ttk.LabelFrame(left_top, text="Gaming Integration", padding=5)
        game_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(game_frame, text="Set Game Folder", command=self._set_game_folder).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Set Game EXE", command=self._set_game_exe).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Scan Game Files", command=self._scan_game_files).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Pre-warm OS Cache", command=self._prewarm_cache).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Launch Game (Manual)", command=self._launch_game).pack(fill=tk.X, pady=2)

        # Core commands
        ctrl = ttk.LabelFrame(left_top, text="Core Commands", padding=5)
        ctrl.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=5, pady=5)

        self.btn_start = ttk.Button(ctrl, text="ENGAGE AI CORE", command=self.start_simulation, state="normal")
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop = ttk.Button(ctrl, text="DISENGAGE", command=self.stop_simulation, state="normal")
        self.btn_stop.grid(row=0, column=1, padx=5, pady=5)

        self.btn_save_now = ttk.Button(ctrl, text="SAVE MEMORY SIGNATURE", command=self.save_profile)
        self.btn_save_now.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        # HUD
        hud = ttk.LabelFrame(left_top, text="Cortex HUD", padding=5)
        hud.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.disk_var = tk.StringVar()
        self.ram_var = tk.StringVar()
        self.vram_var = tk.StringVar()
        self.health_var = tk.StringVar()
        self.policy_var = tk.StringVar()
        self.hits_var = tk.StringVar()
        self.trust_var = tk.StringVar()

        ttk.Label(hud, text="Disk:").grid(row=0, column=0, sticky="w")
        ttk.Label(hud, textvariable=self.disk_var).grid(row=0, column=1, sticky="w")

        ttk.Label(hud, text="RAM:").grid(row=1, column=0, sticky="w")
        ttk.Label(hud, textvariable=self.ram_var).grid(row=1, column=1, sticky="w")

        ttk.Label(hud, text="VRAM:").grid(row=2, column=0, sticky="w")
        ttk.Label(hud, textvariable=self.vram_var).grid(row=2, column=1, sticky="w")

        ttk.Label(hud, text="Health:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Label(hud, textvariable=self.health_var).grid(row=0, column=3, sticky="w")

        ttk.Label(hud, text="Policy:").grid(row=1, column=2, sticky="w", padx=(20, 0))
        ttk.Label(hud, textvariable=self.policy_var).grid(row=1, column=3, sticky="w")

        ttk.Label(hud, text="Hits (V/R/D):").grid(row=2, column=2, sticky="w", padx=(20, 0))
        ttk.Label(hud, textvariable=self.hits_var).grid(row=2, column=3, sticky="w")

        ttk.Label(hud, text="Trust Learned:").grid(row=3, column=2, sticky="w", padx=(20, 0))
        ttk.Label(hud, textvariable=self.trust_var).grid(row=3, column=3, sticky="w")

        # Game Library on the RIGHT side
        library_frame = ttk.LabelFrame(top, text="Game Library", padding=5)
        library_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.game_listbox = tk.Listbox(library_frame, height=12, exportselection=False)
        self.game_listbox.pack(fill=tk.BOTH, expand=True)
        self.game_listbox.bind("<<ListboxSelect>>", self._on_game_selected)

        btn_frame = ttk.Frame(library_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(btn_frame, text="Add New Game", command=self._add_new_game).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Delete Game", command=self._delete_game).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Refresh List", command=self._refresh_game_list).pack(fill=tk.X, pady=2)

        # VRAM bar
        vram_frame = ttk.LabelFrame(main, text="VRAM Reservoir", padding=5)
        vram_frame.pack(fill=tk.X, padx=5, pady=5)

        self.vram_canvas = tk.Canvas(vram_frame, height=24)
        self.vram_canvas.pack(fill=tk.X, expand=True)

        # Assets table
        middle = ttk.LabelFrame(main, text="Assets (Predictive Cache View)", padding=5)
        middle.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(
            middle,
            columns=("id", "kind", "tier", "size", "access", "score"),
            show="headings",
            height=12
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("kind", text="Kind")
        self.tree.heading("tier", text="Tier")
        self.tree.heading("size", text="Size (MB)")
        self.tree.heading("access", text="Access")
        self.tree.heading("score", text="Score")
        self.tree.column("id", width=350, anchor=tk.W)
        self.tree.column("kind", width=80, anchor=tk.CENTER)
        self.tree.column("tier", width=60, anchor=tk.CENTER)
        self.tree.column("size", width=80, anchor=tk.E)
        self.tree.column("access", width=80, anchor=tk.E)
        self.tree.column("score", width=80, anchor=tk.E)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Log
        bottom = ttk.LabelFrame(main, text="Cortex Log", padding=5)
        bottom.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(bottom, height=8, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state="disabled")

        self.status_strip = ttk.Label(
            self.root, text="CORE INITIALIZING", anchor="w", relief="sunken"
        )
        self.status_strip.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- Game Library logic ----------

    def _refresh_game_list(self, selected=None):
        profiles = self.profile_mgr.list_profiles()
        self._suppress_game_select_event = True
        self.game_listbox.delete(0, tk.END)
        current_idx = -1
        for idx, name in enumerate(profiles):
            self.game_listbox.insert(tk.END, name)
            if selected and name == selected:
                current_idx = idx
        if current_idx >= 0:
            self.game_listbox.select_set(current_idx)
            self.game_listbox.see(current_idx)
        self._suppress_game_select_event = False

    def _on_game_selected(self, event):
        if self._suppress_game_select_event:
            return
        selection = self.game_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        name = self.game_listbox.get(idx)
        if name == self.profile_name:
            return
        self._log(f"Game Library: switching to profile '{name}'")
        self._switch_to_profile(name)

    def _switch_to_profile(self, name):
        # Save current profile
        self.save_profile()

        # Stop current simulator
        if self.simulator:
            self.simulator.stop()

        # Load new profile
        profile = self.profile_mgr.load_profile(name)
        if profile is None:
            self._log(f"Profile '{name}' not found; aborting switch.")
            return

        self.profile_name = name
        self._load_from_profile(profile)

        # Recreate simulator
        self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)

        # Refresh game list and select this one
        self._refresh_game_list(selected=name)

        # Auto-start AI core for this game
        self.start_simulation()

    def _add_new_game(self):
        # Ask for a game name
        name = simpledialog.askstring("New Game Profile", "Enter game name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return

        # Optional: prevent overwriting existing profile unless user confirms
        existing = self.profile_mgr.load_profile(name)
        if existing is not None:
            if not messagebox.askyesno(
                "Overwrite?",
                f"A profile named '{name}' already exists. Overwrite it?",
                parent=self.root
            ):
                return

        # Ask for game folder
        game_folder = filedialog.askdirectory(title="Select Game Folder for this game", parent=self.root)
        if not game_folder:
            return

        # Ask for game EXE
        game_exe = filedialog.askopenfilename(
            title="Select Game Executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            parent=self.root,
        )
        if not game_exe:
            return

        self._log(f"Creating new game profile '{name}'")

        # Scan game folder
        assets = scan_game_folder(game_folder, self._log)
        if not assets:
            if not messagebox.askyesno(
                "No Assets Found",
                "No assets detected in that folder. Create an empty profile anyway?",
                parent=self.root
            ):
                return

        # Stop current simulator
        if self.simulator:
            self.simulator.stop()

        # Bind new assets and meta
        self.profile_name = name
        self.assets = assets if assets else self._create_placeholder_assets()
        self._reset_tiers()
        for a in self.assets:
            self.tier_disk.add_asset(a)
            a.tier = "disk"

        self.game_root = game_folder
        self.game_exe = game_exe
        self.meta = {"game_root": self.game_root, "game_exe": self.game_exe}

        hw_profile = self.hw_profile_var.get()
        self.governor = AIGovernor(
            self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
            health=self.health, log_fn=self._log,
            hw_profile=hw_profile, trace_logger=self.trace
        )
        self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)

        # Save immediately and refresh library, select this game
        self.save_profile()
        self._refresh_game_list(selected=name)

        # Auto-start AI core for this new game
        self.start_simulation()

    def _delete_game(self):
        selection = self.game_listbox.curselection()
        if not selection:
            messagebox.showinfo("Delete Game", "Select a game to delete from the library.", parent=self.root)
            return
        idx = selection[0]
        name = self.game_listbox.get(idx)

        if not messagebox.askyesno(
            "Confirm Delete",
            f"Delete profile '{name}'? This removes its saved memory signature.",
            parent=self.root
        ):
            return

        # If deleting currently active profile, stop simulator first
        if name == self.profile_name and self.simulator:
            self.simulator.stop()

        # Delete profile file
        path = self.profile_mgr._profile_path(name)
        try:
            if os.path.exists(path):
                os.remove(path)
                self._log(f"Deleted profile file for '{name}'.")
        except Exception as e:
            self._log(f"Failed to delete profile '{name}': {e}")

        # If we deleted the active one, fall back to default
        if name == self.profile_name:
            self.profile_name = "default"
            # Try loading default, else create fresh
            profile = self.profile_mgr.load_profile(self.profile_name)
            if profile is None:
                self._log("Active profile deleted; creating fresh default assets.")
                self.assets = self._create_placeholder_assets()
                self._reset_tiers()
                for a in self.assets:
                    self.tier_disk.add_asset(a)
                    a.tier = "disk"
                self.game_root = None
                self.game_exe = None
                self.meta = {"game_root": None, "game_exe": None}
                hw_profile = self.hw_profile_var.get()
                self.governor = AIGovernor(
                    self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
                    health=self.health, log_fn=self._log,
                    hw_profile=hw_profile, trace_logger=self.trace
                )
            else:
                self._load_from_profile(profile)
            self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)
            self.start_simulation()

        self._refresh_game_list(selected=self.profile_name)

    # ---------- gaming integration actions ----------

    def _set_game_folder(self):
        path = filedialog.askdirectory(title="Select Game Folder", parent=self.root)
        if not path:
            return
        self.game_root = path
        self.meta["game_root"] = path
        self._log(f"Game folder set: {path}")

    def _set_game_exe(self):
        path = filedialog.askopenfilename(
            title="Select Game Executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            parent=self.root,
        )
        if not path:
            return
        self.game_exe = path
        self.meta["game_exe"] = path
        self._log(f"Game EXE set: {path}")

    def _scan_game_files(self):
        if not self.game_root:
            messagebox.showwarning("Warning", "Set Game Folder first.", parent=self.root)
            return
        self._log("Scanning real game folder for assets...")
        assets = scan_game_folder(self.game_root, self._log)
        if not assets:
            self._log("No game assets found or scan failed.")
            return
        self.assets = assets
        self._reset_tiers()
        for a in self.assets:
            self.tier_disk.add_asset(a)
            a.tier = "disk"
        hw_profile = self.hw_profile_var.get()
        self.governor = AIGovernor(
            self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
            health=self.health, log_fn=self._log,
            hw_profile=hw_profile, trace_logger=self.trace
        )
        self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)
        self._log(f"Game assets bound to hybrid cortex: {len(self.assets)} files.")

    def _prewarm_cache(self):
        if not self.assets:
            messagebox.showwarning(
                "Warning",
                "No assets present; scan game files or let the cortex learn first.",
                parent=self.root
            )
            return
        sorted_assets = sorted(self.assets, key=self.governor._score_asset, reverse=True)
        prewarm_os_cache_for_assets(sorted_assets, count=min(30, len(sorted_assets)), log_fn=self._log)
        self._log("Pre-warm cycle complete.")

    def _launch_game(self):
        if not self.game_exe or not os.path.isfile(self.game_exe):
            messagebox.showwarning("Warning", "Set a valid Game EXE first.", parent=self.root)
            return
        self._log("Manual game launch requested (no auto-launch on startup).")
        try:
            subprocess.Popen([self.game_exe], cwd=os.path.dirname(self.game_exe))
            self._log(f"Launched: {self.game_exe}")
        except Exception as e:
            self._log(f"Failed to launch game: {e}")
            messagebox.showerror("Launch Error", str(e), parent=self.root)

    # ---------- logging + refresh ----------

    def _log(self, msg):
        self.log_text.config(state="normal")
        t = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{t} - {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _refresh_ui(self):
        if self.assets and self.governor:
            self.disk_var.set(f"{self.tier_disk.used_mb:.0f} / {self.tier_disk.capacity_mb} MB")
            self.ram_var.set(f"{self.tier_ram.used_mb:.0f} / {self.tier_ram.capacity_mb} MB")
            self.vram_var.set(f"{self.tier_vram.used_mb:.0f} / {self.tier_vram.capacity_mb} MB")

            health_score = self.health.health_score(self.tier_vram.used_mb, self.tier_vram.capacity_mb)
            self.health_var.set(f"{health_score:.1f} / 100")
            self.policy_var.set(f"{self.governor.current_policy}")
            self.hits_var.set(
                f"{self.health.vram_hits}/{self.health.ram_hits}/{self.health.disk_hits}"
            )
            self.trust_var.set(f"{self.governor.trust_learned:.2f}")

            self.behavior_label.config(text=f"Behavior: {self.governor.behavior}")
            self.phase_label.config(text=f"Phase: {self.governor.phase}")

            self._draw_vram_bar(health_score)

            for row in self.tree.get_children():
                self.tree.delete(row)

            for asset in sorted(self.assets, key=lambda a: a.id):
                score = self.governor._score_asset(asset)
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        asset.id if isinstance(asset.id, str) else str(asset.id),
                        asset.kind,
                        asset.tier,
                        asset.size_mb,
                        f"{asset.access_count:.1f}",
                        f"{score:.3f}",
                    )
                )

            self.status_strip.config(
                text=f"[{self.profile_name}] VRAM {self.tier_vram.used_mb:.0f}/{self.tier_vram.capacity_mb} MB | "
                     f"Health {health_score:.1f} | Policy {self.governor.current_policy} "
                     f"| Phase {self.governor.phase} | Behavior {self.governor.behavior} "
                     f"| HW {self.governor.hw_profile} | Trust {self.governor.trust_learned:.2f}"
            )

        self.root.after(500, self._refresh_ui)

    def _draw_vram_bar(self, health_score):
        self.vram_canvas.delete("all")
        w = self.vram_canvas.winfo_width() or 400
        h = 20
        used = self.tier_vram.used_mb
        cap = max(1.0, self.tier_vram.capacity_mb)
        frac = max(0.0, min(1.0, used / cap))
        bar_w = int(frac * (w - 4))

        if frac < 0.5:
            color = "#00ff66"
        elif frac < 0.8:
            color = "#ffcc00"
        else:
            color = "#ff4444"

        bg = "#202020"
        self.vram_canvas.create_rectangle(2, 2, w - 2, h + 2, outline="#555555", fill=bg)
        self.vram_canvas.create_rectangle(2, 2, 2 + bar_w, h + 2, outline=color, fill=color)

        self.vram_canvas.create_text(
            w // 2, h // 2 + 2,
            text=f"{used:.0f} / {cap:.0f} MB  |  Health {health_score:.1f}",
            fill="white",
            font=("Consolas", 9)
        )

    # ---------- simulation control ----------

    def start_simulation(self):
        if self.simulator:
            self.simulator.start()
            self._log(f"AI CORE ENGAGED for profile '{self.profile_name}'.")
            self.status_strip.config(text="AI CORE ACTIVE")

    def stop_simulation(self):
        if self.simulator:
            self.simulator.stop()
            self._log("AI CORE DISENGAGED by user.")
        self.status_strip.config(text="AI CORE IDLE")
        self.save_profile()

    # ---------- profile + close ----------

    def save_profile(self):
        if not self.profile_name or not self.assets or not self.governor:
            return
        self.profile_mgr.save_profile(
            self.profile_name,
            self.assets,
            self.governor,
            self.meta,
            auto_start=True  # always auto-resume with core running
        )
        self._log(f"MEMORY SIGNATURE SAVED for profile={self.profile_name}")

    def _on_close(self):
        try:
            if self.simulator:
                self.simulator.stop()
            self.save_profile()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = MemoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

