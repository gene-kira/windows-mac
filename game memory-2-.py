import json
import os
import random
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
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
      - Dual mode (gaming/general)
      - Predictive sequences (short/med/long)
      - Phase detection (menu/loading/active)
      - Adaptive thresholds
      - Regret-pattern bias per asset kind
      - Hardware-aware risk profiles
      - Bandit across 3 policies
      - Opportunity windows
      - Health feedback
      - Online pattern-based "learned" corrector via TraceLogger
    """

    def __init__(self, assets, tier_disk, tier_ram, tier_vram, mode="gaming",
                 health=None, log_fn=None, hw_profile=None, trace_logger=None):
        self.assets = assets
        self.tier_disk = tier_disk
        self.tier_ram = tier_ram
        self.tier_vram = tier_vram
        self.now = time.time
        self.mode = mode
        self.log_fn = log_fn or (lambda *_: None)

        self.health = health or SystemHealth()
        self.trace = trace_logger or TraceLogger()

        self.hw_profile = hw_profile or "balanced"
        self._init_base_policy_for_mode_and_hw()

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
        self.phase = "unknown"
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

    def _init_base_policy_for_mode_and_hw(self):
        if self.mode == "gaming":
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
        else:
            if self.hw_profile == "bold":
                self.base_hot_threshold = 2.5
                self.base_warm_threshold = 1.0
                self.decay = 0.995
            elif self.hw_profile == "conservative":
                self.base_hot_threshold = 3.5
                self.base_warm_threshold = 1.4
                self.decay = 0.997
            else:
                self.base_hot_threshold = 3.0
                self.base_warm_threshold = 1.2
                self.decay = 0.995

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

        if self.mode == "gaming":
            if asset.kind in ("texture", "map"):
                base *= 1.2
        else:
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
            self.mode,
            self.phase,
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

    # ---------- phase detection & opportunity ----------

    def update_phase(self):
        maps = [a for a in self.assets if a.kind == "map"]
        textures = [a for a in self.assets if a.kind == "texture"]
        tabs = [a for a in self.assets if a.kind == "tab"]
        videos = [a for a in self.assets if a.kind == "video"]

        now = self.now()

        recent_maps = sum(1 for a in maps if now - a.last_access_time < 2.0)
        recent_textures = sum(1 for a in textures if now - a.last_access_time < 2.0)
        recent_tabs = sum(1 for a in tabs if now - a.last_access_time < 2.0)
        recent_videos = sum(1 for a in videos if now - a.last_access_time < 2.0)

        new_phase = self.phase
        if recent_maps > 5:
            new_phase = "loading"
        elif (recent_tabs + recent_videos) > 5 and recent_maps < 2:
            new_phase = "active"
        elif recent_textures < 3 and recent_maps < 2 and (recent_tabs + recent_videos) < 3:
            new_phase = "menu"
        else:
            new_phase = "active"

        if new_phase != self.phase:
            self.phase = new_phase
            self.log_fn(f"[PHASE] Detected phase: {self.phase}")

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
        self.update_phase()

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
                    self.log_fn(f"[{self.mode}][{self.current_policy}] skip: {asset.id} -> vram "
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
                    f"[{self.mode}][{self.current_policy}] move: {asset.id} ({asset.kind}) "
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
                self.log_fn(f"[{self.mode}][{self.current_policy}] evict: {asset.id} ({asset.kind}) "
                            f"{tier.name} -> {lower_name}")
            else:
                if lower_name != "disk":
                    lower_tier.remove_asset(asset)
                asset.tier = "disk"
                self._log_decision(asset, decision_type, 0.3, tier.used_mb / max(1.0, tier.capacity_mb), "drop")
                self.log_fn(f"[{self.mode}][{self.current_policy}] drop: {asset.id} ({asset.kind}) -> disk")

    def _log_decision(self, asset, decision_type, conf, vram_util, action):
        health_now = self.health.health_score(self.tier_vram.used_mb, self.tier_vram.capacity_mb)
        event = {
            "mode": self.mode,
            "phase": self.phase,
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
            if self.governor.mode == "gaming":
                self._simulate_gaming_access(now)
            else:
                self._simulate_general_access(now)

            self.governor.apply_decay()
            self.governor.maybe_switch_policy()
            moves = self.governor.decide_moves()
            self.governor.execute_moves(moves)

            time.sleep(0.5)

    def _simulate_gaming_access(self, now):
        textures = [a for a in self.assets if a.kind == "texture"]
        maps = [a for a in self.assets if a.kind == "map"]

        for _ in range(random.randint(8, 25)):
            r = random.random()
            if r < 0.5 and textures:
                asset = random.choice(textures)
            elif r < 0.8 and maps:
                asset = random.choice(maps)
            else:
                asset = random.choice(self.assets)

            asset.record_access(now)
            self.governor.record_access_for_prediction(asset)
            self.health.record_access(asset, asset.tier, now)

    def _simulate_general_access(self, now):
        tabs = [a for a in self.assets if a.kind == "tab"]
        videos = [a for a in self.assets if a.kind == "video"]

        for _ in range(random.randint(8, 25)):
            r = random.random()
            if r < 0.4 and tabs:
                asset = random.choice(tabs)
            elif r < 0.8 and videos:
                asset = random.choice(videos)
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

    def _profile_path(self, mode, name):
        safe_game = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        safe_mode = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in mode)
        return os.path.join(self.base_dir, f"memory_profile_{safe_mode}_{safe_game}.json")

    def save_profile(self, mode, name, assets, governor, meta):
        path = self._profile_path(mode, name)
        data = {
            "version": 5,
            "mode": mode,
            "profile_name": name,
            "meta": meta,
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

    def load_profile(self, mode, name):
        path = self._profile_path(mode, name)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load profile: {e}")
            return None


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
#  TKINTER HUD
# ============================================================

class MemoryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid Memory Overmind - Gaming / General - Predictive VRAM Cache")
        self.root.geometry("1250x780")

        self.profile_mgr = ProfileManager()
        self.mode_var = tk.StringVar(value="gaming")
        self.profile_name = None

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

        self._init_ui()
        self._select_profile_and_load()
        self.root.after(500, self._refresh_ui)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- setup ----------

    def _select_profile_and_load(self):
        name = simpledialog.askstring("Profile", "Enter profile name (game/site):", parent=self.root)
        if not name:
            name = "default"

        self.profile_name = name
        mode = self.mode_var.get()
        hw_profile = self.hw_profile_var.get()
        self._log(f"Mode: {mode} | HW: {hw_profile} | Profile: {self.profile_name}")

        profile = self.profile_mgr.load_profile(mode, self.profile_name)
        if profile is None:
            self._log("No existing profile; creating fresh assets.")
            if mode == "gaming":
                self.assets = self._create_placeholder_gaming_assets()
            else:
                self.assets = self._create_placeholder_general_assets()
            self._reset_tiers()
            for a in self.assets:
                self.tier_disk.add_asset(a)
                a.tier = "disk"
            self.governor = AIGovernor(
                self.assets, self.tier_disk, self.tier_ram, self.tier_vram,
                mode=mode, health=self.health, log_fn=self._log,
                hw_profile=hw_profile, trace_logger=self.trace
            )
            self.meta = {"game_root": None, "game_exe": None}
        else:
            self._log("Loaded profile; reconstructing assets and tiers...")
            self._load_from_profile(profile)

        self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)

    def _reset_tiers(self):
        for t in (self.tier_disk, self.tier_ram, self.tier_vram):
            t.assets.clear()
            t.used_mb = 0.0

    def _load_from_profile(self, profile):
        self.assets = []
        all_assets_data = profile.get("assets", [])
        now = time.time()
        mode = profile.get("mode", self.mode_var.get())

        self._reset_tiers()
        self.meta = profile.get("meta", {"game_root": None, "game_exe": None})

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
            mode=mode, health=self.health, log_fn=self._log,
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

    def _create_placeholder_gaming_assets(self):
        assets = []
        for i in range(20):
            size_mb = random.choice([10, 20, 50])
            assets.append(Asset(f"texture_{i}", size_mb, kind="texture"))
        for i in range(5):
            size_mb = random.choice([100, 200])
            assets.append(Asset(f"map_{i}", size_mb, kind="map"))
        return assets

    def _create_placeholder_general_assets(self):
        assets = []
        for i in range(10):
            size_mb = random.choice([5, 10, 15])
            assets.append(Asset(f"tab_{i}", size_mb, kind="tab"))
        for i in range(10):
            size_mb = random.choice([20, 50])
            assets.append(Asset(f"video_seg_{i}", size_mb, kind="video"))
        return assets

    def _switch_mode_or_hw(self, *_):
        if self.simulator:
            self.simulator.stop()
        self.save_profile()
        self._log("Switching mode/HW; reloading profile with new brain.")
        self._reset_tiers()
        self._select_profile_and_load()

    # ---------- UI ----------

    def _init_ui(self):
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill=tk.X)

        mode_frame = ttk.LabelFrame(top, text="Core Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Radiobutton(mode_frame, text="Gaming Core", value="gaming",
                        variable=self.mode_var, command=self._switch_mode_or_hw).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="General Core (Web/Video)", value="general",
                        variable=self.mode_var, command=self._switch_mode_or_hw).pack(anchor="w")

        ttk.Label(mode_frame, text="HW Profile:").pack(anchor="w", pady=(10, 0))
        ttk.Radiobutton(mode_frame, text="Conservative", value="conservative",
                        variable=self.hw_profile_var, command=self._switch_mode_or_hw).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Balanced", value="balanced",
                        variable=self.hw_profile_var, command=self._switch_mode_or_hw).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Bold", value="bold",
                        variable=self.hw_profile_var, command=self._switch_mode_or_hw).pack(anchor="w")

        self.mode_label = ttk.Label(mode_frame, text="ACTIVE: ?", font=("Consolas", 10, "bold"))
        self.mode_label.pack(anchor="w", pady=(10, 0))

        # Gaming controls
        game_frame = ttk.LabelFrame(top, text="Gaming Integration", padding=5)
        game_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Button(game_frame, text="Set Game Folder", command=self._set_game_folder).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Set Game EXE", command=self._set_game_exe).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Scan Game Files", command=self._scan_game_files).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Pre-warm OS Cache", command=self._prewarm_cache).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Launch Game via AI", command=self._launch_game).pack(fill=tk.X, pady=2)

        # Core commands
        ctrl = ttk.LabelFrame(top, text="Core Commands", padding=5)
        ctrl.pack(side=tk.LEFT, fill=tk.X, expand=False, padx=5, pady=5)

        self.btn_start = ttk.Button(ctrl, text="ENGAGE AI CORE", command=self.start_simulation, state="normal")
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop = ttk.Button(ctrl, text="DISENGAGE", command=self.stop_simulation, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=5, pady=5)

        self.btn_save_now = ttk.Button(ctrl, text="SAVE MEMORY SIGNATURE", command=self.save_profile)
        self.btn_save_now.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        # HUD
        hud = ttk.LabelFrame(top, text="Cortex HUD", padding=5)
        hud.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.disk_var = tk.StringVar()
        self.ram_var = tk.StringVar()
        self.vram_var = tk.StringVar()
        self.health_var = tk.StringVar()
        self.policy_var = tk.StringVar()
        self.hits_var = tk.StringVar()
        self.phase_var = tk.StringVar()
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

        ttk.Label(hud, text="Phase:").grid(row=3, column=0, sticky="w")
        ttk.Label(hud, textvariable=self.phase_var).grid(row=3, column=1, sticky="w")

        ttk.Label(hud, text="Trust Learned:").grid(row=3, column=2, sticky="w", padx=(20, 0))
        ttk.Label(hud, textvariable=self.trust_var).grid(row=3, column=3, sticky="w")

        vram_frame = ttk.LabelFrame(main, text="VRAM Reservoir", padding=5)
        vram_frame.pack(fill=tk.X, padx=5, pady=5)

        self.vram_canvas = tk.Canvas(vram_frame, height=24)
        self.vram_canvas.pack(fill=tk.X, expand=True)

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

        bottom = ttk.LabelFrame(main, text="Cortex Log", padding=5)
        bottom.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(bottom, height=8, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state="disabled")

        self.status_strip = ttk.Label(
            self.root, text="CORE IDLE", anchor="w", relief="sunken"
        )
        self.status_strip.pack(fill=tk.X, side=tk.BOTTOM)

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
        if self.mode_var.get() != "gaming":
            messagebox.showinfo("Info", "Switch to Gaming Core to scan game files.")
            return
        if not self.game_root:
            messagebox.showwarning("Warning", "Set Game Folder first.")
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
            mode="gaming", health=self.health, log_fn=self._log,
            hw_profile=hw_profile, trace_logger=self.trace
        )
        self.simulator = GameSimulator(self.assets, self.governor, self.health, log_fn=self._log)
        self._log(f"Game assets bound to core: {len(self.assets)} files.")

    def _prewarm_cache(self):
        if self.mode_var.get() != "gaming":
            messagebox.showinfo("Info", "Pre-warm is meant for Gaming Core with real files.")
            return
        if not self.assets:
            messagebox.showwarning("Warning", "No assets present; scan game files first.")
            return
        sorted_assets = sorted(self.assets, key=self.governor._score_asset, reverse=True)
        prewarm_os_cache_for_assets(sorted_assets, count=min(30, len(sorted_assets)), log_fn=self._log)
        self._log("Pre-warm cycle complete.")

    def _launch_game(self):
        if self.mode_var.get() != "gaming":
            messagebox.showinfo("Info", "Launch Game is for Gaming Core.")
            return
        if not self.game_exe or not os.path.isfile(self.game_exe):
            messagebox.showwarning("Warning", "Set a valid Game EXE first.")
            return
        self._log("Initiating AI-assisted game launch: pre-warm + start EXE.")
        self._prewarm_cache()
        try:
            subprocess.Popen([self.game_exe], cwd=os.path.dirname(self.game_exe))
            self._log(f"Launched: {self.game_exe}")
        except Exception as e:
            self._log(f"Failed to launch game: {e}")
            messagebox.showerror("Launch Error", str(e))

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
            self.phase_var.set(self.governor.phase)
            self.trust_var.set(f"{self.governor.trust_learned:.2f}")

            mode = self.mode_var.get()
            if mode == "gaming":
                self.mode_label.config(text="ACTIVE: GAMING CORE", foreground="#ff8800")
            else:
                self.mode_label.config(text="ACTIVE: GENERAL CORE", foreground="#00c0ff")

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
                text=f"[{mode.upper()} CORE] VRAM {self.tier_vram.used_mb:.0f}/{self.tier_vram.capacity_mb} MB | "
                     f"Health {health_score:.1f} | Policy {self.governor.current_policy} "
                     f"| Phase {self.governor.phase} | HW {self.governor.hw_profile} "
                     f"| Trust {self.governor.trust_learned:.2f}"
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
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self._log("AI CORE ENGAGED.")
            self.status_strip.config(text="AI CORE ACTIVE")

    def stop_simulation(self):
        if self.simulator:
            self.simulator.stop()
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self._log("AI CORE DISENGAGED.")
        self.status_strip.config(text="AI CORE IDLE")
        self.save_profile()

    # ---------- profile + close ----------

    def save_profile(self):
        if not self.profile_name or not self.assets or not self.governor:
            return
        mode = self.mode_var.get()
        self.profile_mgr.save_profile(mode, self.profile_name, self.assets, self.governor, self.meta)
        self._log(f"MEMORY SIGNATURE SAVED for mode={mode}, profile={self.profile_name}")

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

