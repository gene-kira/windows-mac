import sys
import time
import math
import random
import json
import hashlib
import zlib
import subprocess
import os
from collections import Counter, deque

# ============================================================
# OPTIONAL / AUTO-INSTALLED DEPENDENCIES
# ============================================================

def safe_import(name, auto_install=False, version=None):
    try:
        return __import__(name)
    except ImportError:
        print(f"[Autoloader] Optional module '{name}' not found.")
        if auto_install:
            pkg = name if version is None else f"{name}=={version}"
            print(f"[Autoloader] Attempting to install '{pkg}' via pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"[Autoloader] '{pkg}' installed. Re-importing...")
                return __import__(name)
            except Exception as e:
                print(f"[Autoloader] Failed to install '{pkg}': {e}")
        return None

psutil = safe_import("psutil", auto_install=True)
pyopencl = safe_import("pyopencl", auto_install=True, version=None)
torch = safe_import("torch", auto_install=False)

tk_mod = safe_import("tkinter")
if tk_mod:
    import tkinter as tk
    from tkinter import ttk
else:
    tk = ttk = None


# ============================================================
# PERSISTENCE MANAGER (DRIVE SELECTION + STATUS)
# ============================================================

class PersistenceManager:
    r"""
    Handles persistence of:
      - NPU weights (per-set + VRAM)
      - RAM/VRAM organ cache state
      - Backbone health history

    Drive selection:
      1) If NPU_RAID_PERSIST_PATH is UNC/SMB (\\server\share) -> use that.
      2) Else try local drives in order: D, E, F, G, H.
      3) If none exist -> fallback to C:\NPU_RAID_PERSIST.
      4) If chosen path fails -> persistence disabled.
    """

    def __init__(self, base_dir, enabled=True):
        self.base_dir = base_dir
        self.enabled = enabled
        if self.enabled:
            try:
                os.makedirs(self.base_dir, exist_ok=True)
            except Exception as e:
                print(f"[Persist] Failed to create base dir '{self.base_dir}': {e}")
                self.enabled = False

    @staticmethod
    def _drive_exists_and_writable(root):
        try:
            if not os.path.exists(root):
                return False
            test_path = os.path.join(root, ".npu_raid_test")
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
            return True
        except Exception:
            return False

    @staticmethod
    def _extract_root_from_unc(path):
        # For \\server\share\folder -> \\server\share
        parts = path.strip("\\").split("\\")
        if len(parts) >= 2:
            return "\\\\" + parts[0] + "\\" + parts[1]
        return path

    @staticmethod
    def _extract_drive_from_path(path):
        # UNC path
        if path.startswith("\\\\"):
            return PersistenceManager._extract_root_from_unc(path)
        # Local drive
        drive, _ = os.path.splitdrive(os.path.abspath(path))
        return drive if drive else path

    @classmethod
    def from_env(cls):
        env_path = os.environ.get("NPU_RAID_PERSIST_PATH", "").strip()

        # 1) UNC / SMB path explicitly set
        if env_path.startswith("\\\\"):
            base_dir = env_path
            print(f"[Persist] Using UNC/SMB path for persistence: {base_dir}")
            return cls(base_dir, enabled=True)

        # 2) Try local drives D, E, F, G, H
        candidate_root = None
        for drive in ["D:", "E:", "F:", "G:", "H:"]:
            root = drive + "\\"
            if cls._drive_exists_and_writable(root):
                candidate_root = root
                print(f"[Persist] Using drive {drive} for persistence.")
                break

        # 3) If none found, fallback to C:\NPU_RAID_PERSIST
        if candidate_root is None:
            candidate_root = r"C:\NPU_RAID_PERSIST"
            print(f"[Persist] No D/E/F/G/H found. Falling back to {candidate_root}.")

        # If env_path is a non-empty local path, prefer that under the chosen root?
        # We'll interpret env_path as a folder name if not UNC.
        if env_path and not env_path.startswith("\\\\"):
            # If env_path is absolute, use it directly; else join with candidate_root
            if os.path.isabs(env_path):
                base_dir = env_path
            else:
                base_dir = os.path.join(candidate_root, env_path)
        else:
            # Default folder under chosen root
            if candidate_root.lower().startswith("c:"):
                base_dir = candidate_root
            else:
                base_dir = os.path.join(candidate_root, "NPU_RAID_PERSIST")

        # Try to create; if fail, disable
        try:
            os.makedirs(base_dir, exist_ok=True)
            print(f"[Persist] Final persistence base dir: {base_dir}")
            return cls(base_dir, enabled=True)
        except Exception as e:
            print(f"[Persist] Failed to initialize persistence at '{base_dir}': {e}")
            return cls(base_dir, enabled=False)

    def get_status_string(self):
        if not self.enabled:
            return "PERSIST: OFF"
        drive = self._extract_drive_from_path(self.base_dir)
        return f"PERSIST: ON (drive: {drive})"

    def _path(self, name):
        return os.path.join(self.base_dir, name)

    def save_json(self, name, obj):
        if not self.enabled:
            return
        try:
            path = self._path(name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        except Exception as e:
            print(f"[Persist] Failed to save '{name}': {e}")

    def load_json(self, name):
        if not self.enabled:
            return None
        try:
            path = self._path(name)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Persist] Failed to load '{name}': {e}")
            return None

    def save_npu(self, npu, name):
        if not self.enabled:
            return
        try:
            path = self._path(name)
            npu.save_state(path)
        except Exception as e:
            print(f"[Persist] Failed to save NPU '{name}': {e}")

    def load_npu(self, npu, name):
        if not self.enabled:
            return
        try:
            path = self._path(name)
            if os.path.exists(path):
                npu.load_state(path)
        except Exception as e:
            print(f"[Persist] Failed to load NPU '{name}': {e}")

    def save_backbone(self, backbone):
        if not self.enabled:
            return

        try:
            # Save NPUs
            for i, (_, npu, _) in enumerate(backbone.bay.sets):
                self.save_npu(npu, f"npu_set_{i}.json")
            _, vram_npu, _ = backbone.bay.vram
            self.save_npu(vram_npu, "npu_vram.json")

            # Save organs
            sets_state = []
            for i, (_, _, organ) in enumerate(backbone.bay.sets):
                sets_state.append(organ.to_dict())
            _, _, vram_organ = backbone.bay.vram
            vram_state = vram_organ.to_dict()

            organs_state = {
                "sets": sets_state,
                "vram": vram_state,
            }
            self.save_json("organs.json", organs_state)

            # Save health history
            self.save_json("health_history.json", backbone.health_history)

            # Save backbone meta (interval, etc.)
            meta = {
                "next_interval_ms": backbone.next_interval_ms,
            }
            self.save_json("backbone_meta.json", meta)

        except Exception as e:
            print(f"[Persist] Failed to save backbone: {e}")

    def load_backbone(self, backbone):
        if not self.enabled:
            return

        try:
            # Load NPUs
            for i, (_, npu, _) in enumerate(backbone.bay.sets):
                self.load_npu(npu, f"npu_set_{i}.json")
            _, vram_npu, _ = backbone.bay.vram
            self.load_npu(vram_npu, "npu_vram.json")

            # Load organs
            organs_state = self.load_json("organs.json")
            if organs_state:
                sets_state = organs_state.get("sets", [])
                for i, state in enumerate(sets_state):
                    if i < len(backbone.bay.sets):
                        _, _, organ = backbone.bay.sets[i]
                        organ.from_dict(state)

                vram_state = organs_state.get("vram")
                if vram_state:
                    _, _, vram_organ = backbone.bay.vram
                    vram_organ.from_dict(vram_state)

            # Load health history
            hh = self.load_json("health_history.json")
            if isinstance(hh, list):
                backbone.health_history = hh

            # Load backbone meta
            meta = self.load_json("backbone_meta.json")
            if isinstance(meta, dict):
                backbone.next_interval_ms = meta.get("next_interval_ms", backbone.next_interval_ms)

            print("[Persist] Backbone state restored from persistence.")
        except Exception as e:
            print(f"[Persist] Failed to load backbone: {e}")


# ============================================================
# REPLICA NPU
# ============================================================

class ReplicaNPU:
    def __init__(
        self,
        cores=8,
        frequency_ghz=1.2,
        memory_size=16,
        plasticity_decay=0.0005,
        integrity_threshold=0.4,
    ):
        self.cores = cores
        self.frequency_ghz = frequency_ghz

        self.cycles = 0
        self.energy = 0.0

        self.memory = deque(maxlen=memory_size)

        self.plasticity = 1.0
        self.plasticity_decay = plasticity_decay

        self.integrity_threshold = integrity_threshold
        self.model_integrity = 1.0
        self.frozen = False

        self.heads = {}
        self.symbolic_bias = {}
        self.instruction_queue = deque()

    def schedule(self, fn, *args):
        self.instruction_queue.append((fn, args))

    def tick(self, budget=64):
        executed = 0
        while self.instruction_queue and executed < budget:
            fn, args = self.instruction_queue.popleft()
            fn(*args)
            executed += 1
        self.plasticity = max(0.1, self.plasticity - self.plasticity_decay)

    def mac(self, a, b):
        self.cycles += 1
        self.energy += 0.001
        return a * b

    def vector_mac(self, v1, v2):
        assert len(v1) == len(v2)
        chunk = math.ceil(len(v1) / self.cores)
        acc = 0.0
        for i in range(0, len(v1), chunk):
            partial = 0.0
            for j in range(i, min(i + chunk, len(v1))):
                partial += self.mac(v1[j], v2[j])
            acc += partial
        return acc

    def matmul(self, A, B):
        result = [[0.0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                col = [B[k][j] for k in range(len(B))]
                result[i][j] = self.vector_mac(A[i], col)
        return result

    def relu(self, x):
        self.cycles += 1
        return max(0.0, x)

    def sigmoid(self, x):
        self.cycles += 2
        return 1.0 / (1.0 + math.exp(-x))

    def activate(self, tensor, mode="relu"):
        for i in range(len(tensor)):
            for j in range(len(tensor[0])):
                tensor[i][j] = (
                    self.relu(tensor[i][j])
                    if mode == "relu"
                    else self.sigmoid(tensor[i][j])
                )
        return tensor

    def add_head(self, name, input_dim, lr=0.01, risk=1.0, organ=None):
        self.heads[name] = {
            "w": [random.uniform(-0.1, 0.1) for _ in range(input_dim)],
            "b": 0.0,
            "lr": lr,
            "risk": risk,
            "organ": organ,
            "history": deque(maxlen=12),
        }

    def _symbolic_modulation(self, name):
        return self.symbolic_bias.get(name, 0.0)

    def _predict_head(self, head, x, name):
        y = 0.0
        for i in range(len(x)):
            y += self.mac(x[i], head["w"][i])
        y += head["b"]
        y += self._symbolic_modulation(name)
        head["history"].append(y)
        self.memory.append(y)
        return y

    def predict(self, x):
        preds = {}
        for name, head in self.heads.items():
            preds[name] = self._predict_head(head, x, name)
        return preds

    def learn(self, x, targets):
        if self.frozen:
            return {}
        errors = {}
        for name, target in targets.items():
            head = self.heads[name]
            pred = self._predict_head(head, x, name)
            error = target - pred
            weighted_error = (
                error * head["risk"] * self.plasticity * self.model_integrity
            )
            for i in range(len(head["w"])):
                head["w"][i] += head["lr"] * weighted_error * x[i]
                self.cycles += 1
            head["b"] += head["lr"] * weighted_error
            self.energy += 0.005
            errors[name] = error
        return errors

    def confidence(self, name):
        h = self.heads[name]["history"]
        if len(h) < 2:
            return 0.5
        mean = sum(h) / len(h)
        var = sum((v - mean) ** 2 for v in h) / len(h)
        return max(0.0, min(1.0, 1.0 - var))

    def check_integrity(self, external_integrity=1.0):
        self.model_integrity = external_integrity
        self.frozen = self.model_integrity < self.integrity_threshold

    def micro_recovery(self, rate=0.01):
        self.plasticity = min(1.0, self.plasticity + rate)

    def set_symbolic_bias(self, name, value):
        self.symbolic_bias[name] = value

    def save_state(self, path):
        heads_serializable = {}
        for k, v in self.heads.items():
            heads_serializable[k] = {
                "w": v["w"],
                "b": v["b"],
                "lr": v["lr"],
                "risk": v["risk"],
                "organ": v["organ"],
                "history": list(v["history"]),
            }
        state = {
            "cores": self.cores,
            "frequency_ghz": self.frequency_ghz,
            "cycles": self.cycles,
            "energy": self.energy,
            "plasticity": self.plasticity,
            "plasticity_decay": self.plasticity_decay,
            "integrity_threshold": self.integrity_threshold,
            "model_integrity": self.model_integrity,
            "frozen": self.frozen,
            "heads": heads_serializable,
            "symbolic_bias": self.symbolic_bias,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def load_state(self, path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.cores = state.get("cores", self.cores)
        self.frequency_ghz = state.get("frequency_ghz", self.frequency_ghz)
        self.cycles = state.get("cycles", 0)
        self.energy = state.get("energy", 0.0)
        self.plasticity = state.get("plasticity", 1.0)
        self.plasticity_decay = state.get("plasticity_decay", self.plasticity_decay)
        self.integrity_threshold = state.get("integrity_threshold", self.integrity_threshold)
        self.model_integrity = state.get("model_integrity", 1.0)
        self.frozen = state.get("frozen", False)
        self.symbolic_bias = state.get("symbolic_bias", {})

        heads_loaded = state.get("heads", {})
        self.heads = {}
        for k, v in heads_loaded.items():
            self.heads[k] = {
                "w": v.get("w", []),
                "b": v.get("b", 0.0),
                "lr": v.get("lr", 0.01),
                "risk": v.get("risk", 1.0),
                "organ": v.get("organ", None),
                "history": deque(v.get("history", []), maxlen=12),
            }

    def stats(self):
        time_sec = self.cycles / (self.frequency_ghz * 1e9)
        return {
            "cores": self.cores,
            "cycles": self.cycles,
            "estimated_time_sec": time_sec,
            "energy_units": round(self.energy, 6),
            "plasticity": round(self.plasticity, 3),
            "integrity": round(self.model_integrity, 3),
            "frozen": self.frozen,
            "confidence": {
                k: round(self.confidence(k), 3) for k in self.heads
            },
        }

    def xor_parity(self, blocks):
        if not blocks:
            return []
        length = len(blocks[0])
        parity = [0] * length
        for blk in blocks:
            for i in range(length):
                parity[i] ^= blk[i]
            self.cycles += length
            self.energy += 0.0001 * length
        return parity

    def fast_hash(self, data_bytes: bytes) -> int:
        h = zlib.adler32(data_bytes) & 0xFFFFFFFF
        self.cycles += len(data_bytes)
        self.energy += 0.00005 * len(data_bytes)
        return h

    def strong_hash(self, data_bytes: bytes) -> str:
        h = hashlib.sha256(data_bytes).hexdigest()
        self.cycles += len(data_bytes)
        self.energy += 0.0001 * len(data_bytes)
        return h

    def compression_ratio(self, data_bytes: bytes) -> float:
        if not data_bytes:
            return 1.0
        comp = zlib.compress(data_bytes, level=1)
        self.cycles += len(data_bytes)
        self.energy += 0.00005 * len(data_bytes)
        return len(comp) / len(data_bytes)

    def entropy(self, data_bytes: bytes) -> float:
        if not data_bytes:
            return 0.0
        counts = Counter(data_bytes)
        total = len(data_bytes)
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * math.log2(p)
        self.cycles += len(data_bytes)
        self.energy += 0.00005 * len(data_bytes)
        return ent

    def jaccard_similarity(self, a_bytes: bytes, b_bytes: bytes) -> float:
        set_a = set(a_bytes)
        set_b = set(b_bytes)
        if not set_a and not set_b:
            return 1.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        self.cycles += len(set_a) + len(set_b)
        self.energy += 0.00005 * (len(set_a) + len(set_b))
        return inter / union if union else 1.0

    def dedup_signature(self, data_bytes: bytes) -> str:
        return self.strong_hash(data_bytes)


# ============================================================
# RAID + CACHE ORGAN (RAM)
# ============================================================

class RaidCacheOrgan:
    def __init__(self, npu: ReplicaNPU,
                 level="raid5",
                 spines=4,
                 block_size=1024,
                 capacity_mb=8.0):

        self.npu = npu
        self.level = level.lower()
        self.spines = spines
        self.block_size = block_size
        self.capacity_mb = capacity_mb

        # 0–1 GB UTIL METER
        self.used_bytes = 0
        self.max_bytes = 1_073_741_824  # 1 GB

        self.blocks = [[] for _ in range(spines)]
        self.block_ages = []

        self.access_count = 0
        self.hit_count = 0
        self.evictions = 0
        self.parity_errors = 0
        self.corruption_errors = 0

        self.io_latency_ns_sum = 0
        self.io_latency_ns_count = 0
        self.io_latency_ns_max = 0

        self.dedup_signatures = []
        self.last_similarity = 1.0

        self.storage_health = 1.0
        self.performance_health = 1.0
        self.util_health = 1.0
        self.combined_health = 1.0

    @property
    def utilization(self):
        return min(1.0, self.used_bytes / self.max_bytes)

    @property
    def used_mb(self):
        return self.used_bytes / (1024 * 1024)

    @property
    def max_mb(self):
        return self.max_bytes / (1024 * 1024)

    @property
    def eviction_pressure(self):
        if self.access_count == 0:
            return 0.0
        return min(1.0, self.evictions / self.access_count)

    @property
    def hit_rate(self):
        if self.access_count == 0:
            return 1.0
        return self.hit_count / self.access_count

    @property
    def avg_block_age(self):
        if not self.block_ages:
            return 0.0
        return sum(self.block_ages) / len(self.block_ages)

    @property
    def max_block_age(self):
        if not self.block_ages:
            return 0.0
        return max(self.block_ages)

    @property
    def avg_io_latency_ns(self):
        if self.io_latency_ns_count == 0:
            return 0.0
        return self.io_latency_ns_sum / self.io_latency_ns_count

    def _ensure_block_len(self, data_bytes: bytes):
        arr = list(data_bytes)
        if len(arr) > self.block_size:
            return arr[:self.block_size]
        if len(arr) < self.block_size:
            return arr + [0] * (self.block_size - len(arr))
        return arr

    def _evict_if_needed(self):
        while self.used_bytes >= self.max_bytes:
            if self.block_ages:
                self.block_ages.pop(0)
            for spine in self.blocks:
                if spine:
                    evicted = spine.pop(0)
                    self.used_bytes -= len(evicted)
            if self.dedup_signatures:
                self.dedup_signatures.pop(0)
            self.evictions += 1

    def _append_block(self, data_block):
        self._evict_if_needed()
        if self.level == "raid5":
            parity = self.npu.xor_parity([data_block])
            for i, spine in enumerate(self.blocks):
                if i == 0:
                    spine.append(parity)
                else:
                    spine.append(list(data_block))
        elif self.level == "raid1":
            for spine in self.blocks:
                spine.append(list(data_block))
        elif self.level == "raid0":
            idx = len(self.block_ages) % self.spines
            for i, spine in enumerate(self.blocks):
                if i == idx:
                    spine.append(list(data_block))
                else:
                    if len(spine) < len(self.block_ages) + 1:
                        spine.append([0] * self.block_size)
        elif self.level == "raid10":
            half = self.spines // 2
            stripe = len(self.block_ages) % half
            for i, spine in enumerate(self.blocks):
                if i == stripe or i == stripe + half:
                    spine.append(list(data_block))
                else:
                    if len(spine) < len(self.block_ages) + 1:
                        spine.append([0] * self.block_size)
        else:
            for spine in self.blocks:
                spine.append(list(data_block))

        self.used_bytes += len(data_block)
        self.block_ages.append(0)
        sig = self.npu.dedup_signature(bytes(data_block))
        self.dedup_signatures.append(sig)

    def cache_write_bytes(self, data_bytes: bytes):
        if not data_bytes:
            return
        block = self._ensure_block_len(data_bytes)
        self._append_block(block)
        data_bytes = bytes(block)
        _ = self.npu.fast_hash(data_bytes)
        _ = self.npu.entropy(data_bytes)
        _ = self.npu.compression_ratio(data_bytes)

    def cache_read_block(self, logical_index: int):
        self.access_count += 1
        if not self.block_ages:
            return None
        idx = logical_index % len(self.block_ages)
        self.hit_count += 1
        if self.level == "raid5":
            return self.blocks[1][idx]
        elif self.level == "raid1":
            return self.blocks[0][idx]
        elif self.level == "raid0":
            spine_idx = idx % self.spines
            return self.blocks[spine_idx][idx]
        elif self.level == "raid10":
            half = self.spines // 2
            spine_idx = idx % half
            return self.blocks[spine_idx][idx]
        else:
            return self.blocks[0][idx]

    def record_io_latency_ns(self, latency_ns: int):
        self.io_latency_ns_sum += latency_ns
        self.io_latency_ns_count += 1
        if latency_ns > self.io_latency_ns_max:
            self.io_latency_ns_max = latency_ns

    def deep_npu_cycle(self):
        if not self.block_ages:
            self.storage_health = 1.0
            self.performance_health = 1.0
            self.util_health = 1.0
            self.combined_health = 1.0
            return

        self.block_ages = [age + 1 for age in self.block_ages]

        parity_errors = 0
        if self.level == "raid5":
            for idx in range(len(self.block_ages)):
                data_block = self.blocks[1][idx]
                parity_block = self.blocks[0][idx]
                recomputed = self.npu.xor_parity([data_block])
                if recomputed != parity_block:
                    parity_errors += 1
        self.parity_errors += parity_errors

        entropies = []
        similarities = []
        dedup_map = {}
        prev_bytes = None

        for idx in range(len(self.block_ages)):
            if self.level == "raid5":
                data_block = self.blocks[1][idx]
            else:
                data_block = self.blocks[0][idx]
            data_bytes = bytes(data_block)

            h_strong = self.npu.strong_hash(data_bytes)
            ent = self.npu.entropy(data_bytes)
            _ = self.npu.compression_ratio(data_bytes)

            entropies.append(ent)
            dedup_map.setdefault(h_strong, 0)
            dedup_map[h_strong] += 1

            if prev_bytes is not None:
                sim = self.npu.jaccard_similarity(prev_bytes, data_bytes)
                similarities.append(sim)
            prev_bytes = data_bytes

        total_blocks = len(self.block_ages)
        parity_integrity = 1.0
        if total_blocks > 0:
            parity_integrity = max(0.0, 1.0 - (self.parity_errors / total_blocks))

        eviction_pressure = self.eviction_pressure
        storage_pressure = max(0.0, 1.0 - eviction_pressure)

        max_age = self.max_block_age or 1.0
        age_health = min(1.0, self.avg_block_age / max_age)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        if avg_entropy <= 3.0:
            entropy_health = avg_entropy / 3.0
        elif avg_entropy >= 7.0:
            entropy_health = max(0.0, 1.0 - (avg_entropy - 7.0) / 1.0)
        else:
            entropy_health = 1.0

        storage_health = (
            0.3 * parity_integrity +
            0.3 * storage_pressure +
            0.2 * age_health +
            0.2 * entropy_health
        )

        hit_health = self.hit_rate

        if self.io_latency_ns_max > 0:
            avg_lat = self.avg_io_latency_ns
            worst = self.io_latency_ns_max
            latency_health = max(0.0, 1.0 - (avg_lat / worst))
        else:
            latency_health = 1.0

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
        else:
            avg_sim = 1.0
        self.last_similarity = avg_sim
        similarity_health = avg_sim

        total_dups = sum(c for c in dedup_map.values() if c > 1)
        if total_blocks > 0:
            dedup_ratio = total_dups / total_blocks
        else:
            dedup_ratio = 0.0
        dedup_health = min(1.0, 0.5 + 0.5 * dedup_ratio)

        performance_health = (
            0.4 * hit_health +
            0.3 * latency_health +
            0.2 * similarity_health +
            0.1 * dedup_health
        )

        util = self.utilization
        util_health = max(0.0, 1.0 - util)

        self.storage_health = max(0.0, min(1.0, storage_health))
        self.performance_health = max(0.0, min(1.0, performance_health))
        self.util_health = max(0.0, min(1.0, util_health))
        self.combined_health = (
            0.4 * self.storage_health +
            0.4 * self.performance_health +
            0.2 * self.util_health
        )

    # ---------- Persistence helpers ----------

    def to_dict(self):
        return {
            "level": self.level,
            "spines": self.spines,
            "block_size": self.block_size,
            "capacity_mb": self.capacity_mb,
            "used_bytes": self.used_bytes,
            "max_bytes": self.max_bytes,
            "blocks": self.blocks,
            "block_ages": self.block_ages,
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "evictions": self.evictions,
            "parity_errors": self.parity_errors,
            "corruption_errors": self.corruption_errors,
            "io_latency_ns_sum": self.io_latency_ns_sum,
            "io_latency_ns_count": self.io_latency_ns_count,
            "io_latency_ns_max": self.io_latency_ns_max,
            "dedup_signatures": self.dedup_signatures,
            "last_similarity": self.last_similarity,
            "storage_health": self.storage_health,
            "performance_health": self.performance_health,
            "util_health": self.util_health,
            "combined_health": self.combined_health,
        }

    def from_dict(self, d):
        try:
            self.level = d.get("level", self.level)
            self.spines = d.get("spines", self.spines)
            self.block_size = d.get("block_size", self.block_size)
            self.capacity_mb = d.get("capacity_mb", self.capacity_mb)
            self.used_bytes = d.get("used_bytes", 0)
            self.max_bytes = d.get("max_bytes", self.max_bytes)
            self.blocks = d.get("blocks", [[] for _ in range(self.spines)])
            self.block_ages = d.get("block_ages", [])
            self.access_count = d.get("access_count", 0)
            self.hit_count = d.get("hit_count", 0)
            self.evictions = d.get("evictions", 0)
            self.parity_errors = d.get("parity_errors", 0)
            self.corruption_errors = d.get("corruption_errors", 0)
            self.io_latency_ns_sum = d.get("io_latency_ns_sum", 0)
            self.io_latency_ns_count = d.get("io_latency_ns_count", 0)
            self.io_latency_ns_max = d.get("io_latency_ns_max", 0)
            self.dedup_signatures = d.get("dedup_signatures", [])
            self.last_similarity = d.get("last_similarity", 1.0)
            self.storage_health = d.get("storage_health", 1.0)
            self.performance_health = d.get("performance_health", 1.0)
            self.util_health = d.get("util_health", 1.0)
            self.combined_health = d.get("combined_health", 1.0)
        except Exception as e:
            print(f"[Persist] Failed to restore RaidCacheOrgan: {e}")


# ============================================================
# VRAM CACHE ORGAN
# ============================================================

class VramCacheOrgan:
    def __init__(self, npu: ReplicaNPU, block_size=1024):
        self.npu = npu
        self.block_size = block_size

        self.is_real_vram = False
        self.vram_total_bytes = 1_073_741_824
        self.vram_used_bytes = 0

        self.blocks = []
        self.block_ages = []

        self.storage_health = 1.0
        self.performance_health = 1.0
        self.util_health = 1.0
        self.combined_health = 1.0

        self.io_latency_ns_sum = 0
        self.io_latency_ns_count = 0
        self.io_latency_ns_max = 0

        self.last_similarity = 1.0
        self.evictions = 0
        self.access_count = 0
        self.hit_count = 0

        self.torch_handles = []

        self._init_vram_backend()

    def _init_vram_backend(self):
        self.cl_ctx = None
        self.cl_queue = None
        self.cl_buffer = None

        if not pyopencl:
            print("[VRAM] pyopencl not present – simulating VRAM in RAM.")
            return

        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if not platforms:
                print("[VRAM] No OpenCL platforms found – simulating VRAM.")
                return
            devs = platforms[0].get_devices()
            if not devs:
                print("[VRAM] No OpenCL devices found – simulating VRAM.")
                return
            dev = devs[0]
            self.vram_total_bytes = min(dev.global_mem_size, 1_073_741_824)
            self.cl_ctx = cl.Context([dev])
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
            self.cl_buffer = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_WRITE, size=self.vram_total_bytes)
            self.is_real_vram = True
            print(f"[VRAM] Using OpenCL device: {dev.name}, VRAM mapped: {self.vram_total_bytes / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"[VRAM] OpenCL init failed – simulating VRAM. ({e})")
            self.cl_ctx = None
            self.cl_queue = None
            self.cl_buffer = None
            self.is_real_vram = False

    @property
    def utilization(self):
        return min(1.0, self.vram_used_bytes / self.vram_total_bytes)

    @property
    def used_mb(self):
        return self.vram_used_bytes / (1024 * 1024)

    @property
    def max_mb(self):
        return self.vram_total_bytes / (1024 * 1024)

    @property
    def eviction_pressure(self):
        if self.access_count == 0:
            return 0.0
        return min(1.0, self.evictions / self.access_count)

    @property
    def hit_rate(self):
        if self.access_count == 0:
            return 1.0
        return self.hit_count / self.access_count

    @property
    def avg_io_latency_ns(self):
        if self.io_latency_ns_count == 0:
            return 0.0
        return self.io_latency_ns_sum / self.io_latency_ns_count

    def _ensure_block_len(self, data_bytes: bytes):
        arr = list(data_bytes)
        if len(arr) > self.block_size:
            return arr[:self.block_size]
        if len(arr) < self.block_size:
            return arr + [0] * (self.block_size - len(arr))
        return arr

    def _evict_if_needed(self):
        while self.vram_used_bytes >= self.vram_total_bytes and self.blocks:
            self.blocks.pop(0)
            if self.block_ages:
                self.block_ages.pop(0)
            if self.torch_handles:
                self.torch_handles.pop(0)
            self.evictions += 1

    def cache_write_bytes(self, data_bytes: bytes):
        if not data_bytes:
            return
        block = self._ensure_block_len(data_bytes)
        self._evict_if_needed()

        self.blocks.append(block)
        self.block_ages.append(0)
        self.vram_used_bytes += len(block)

        data_bytes = bytes(block)
        _ = self.npu.fast_hash(data_bytes)
        _ = self.npu.entropy(data_bytes)
        _ = self.npu.compression_ratio(data_bytes)

    def cache_read_block(self, logical_index: int):
        self.access_count += 1
        if not self.blocks:
            return None
        idx = logical_index % len(self.blocks)
        self.hit_count += 1
        return self.blocks[idx]

    def record_io_latency_ns(self, latency_ns: int):
        self.io_latency_ns_sum += latency_ns
        self.io_latency_ns_count += 1
        if latency_ns > self.io_latency_ns_max:
            self.io_latency_ns_max = latency_ns

    def _reserve_bytes(self, size_bytes: int):
        self._evict_if_needed()
        self.vram_used_bytes += size_bytes
        if self.vram_used_bytes > self.vram_total_bytes:
            self.vram_used_bytes = self.vram_total_bytes

    def ingest_torch_tensor(self, tensor):
        if torch is None:
            return
        try:
            if tensor.device.type != "cuda" and torch.cuda.is_available():
                tensor = tensor.to("cuda", non_blocking=True)
            numel = tensor.numel()
            element_size = tensor.element_size()
            size_bytes = numel * element_size
            self._reserve_bytes(size_bytes)
            self.torch_handles.append(tensor)
        except Exception:
            pass

    def ingest_numpy_array(self, arr):
        try:
            import numpy as np
        except ImportError:
            return
        if not isinstance(arr, np.ndarray):
            return
        size_bytes = arr.nbytes
        self._reserve_bytes(size_bytes)
        sample = arr.tobytes()[: self.block_size]
        self.cache_write_bytes(sample)

    def ingest_video_frame_bytes(self, frame_bytes: bytes):
        if not frame_bytes:
            return
        self._reserve_bytes(len(frame_bytes))
        self.cache_write_bytes(frame_bytes[: self.block_size])

    def deep_npu_cycle(self):
        if not self.blocks:
            self.storage_health = 1.0
            self.performance_health = 1.0
            self.util_health = 1.0
            self.combined_health = 1.0
            return

        self.block_ages = [age + 1 for age in self.block_ages]

        entropies = []
        similarities = []
        prev_bytes = None

        for block in self.blocks:
            data_bytes = bytes(block)
            ent = self.npu.entropy(data_bytes)
            _ = self.npu.compression_ratio(data_bytes)
            entropies.append(ent)
            if prev_bytes is not None:
                sim = self.npu.jaccard_similarity(prev_bytes, data_bytes)
                similarities.append(sim)
            prev_bytes = data_bytes

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        if avg_entropy <= 3.0:
            entropy_health = avg_entropy / 3.0
        elif avg_entropy >= 7.0:
            entropy_health = max(0.0, 1.0 - (avg_entropy - 7.0) / 1.0)
        else:
            entropy_health = 1.0

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
        else:
            avg_sim = 1.0
        self.last_similarity = avg_sim

        storage_health = entropy_health
        hit_health = self.hit_rate

        if self.io_latency_ns_max > 0:
            avg_lat = self.avg_io_latency_ns
            worst = self.io_latency_ns_max
            latency_health = max(0.0, 1.0 - (avg_lat / worst))
        else:
            latency_health = 1.0

        performance_health = (
            0.5 * hit_health +
            0.5 * latency_health
        )

        util = self.utilization
        util_health = max(0.0, 1.0 - util)

        self.storage_health = max(0.0, min(1.0, storage_health))
        self.performance_health = max(0.0, min(1.0, performance_health))
        self.util_health = max(0.0, min(1.0, util_health))
        self.combined_health = (
            0.4 * self.storage_health +
            0.4 * self.performance_health +
            0.2 * self.util_health
        )

    # ---------- Persistence helpers ----------

    def to_dict(self):
        return {
            "block_size": self.block_size,
            "is_real_vram": self.is_real_vram,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_used_bytes": self.vram_used_bytes,
            "blocks": self.blocks,
            "block_ages": self.block_ages,
            "storage_health": self.storage_health,
            "performance_health": self.performance_health,
            "util_health": self.util_health,
            "combined_health": self.combined_health,
            "io_latency_ns_sum": self.io_latency_ns_sum,
            "io_latency_ns_count": self.io_latency_ns_count,
            "io_latency_ns_max": self.io_latency_ns_max,
            "last_similarity": self.last_similarity,
            "evictions": self.evictions,
            "access_count": self.access_count,
            "hit_count": self.hit_count,
        }

    def from_dict(self, d):
        try:
            self.block_size = d.get("block_size", self.block_size)
            self.is_real_vram = d.get("is_real_vram", self.is_real_vram)
            self.vram_total_bytes = d.get("vram_total_bytes", self.vram_total_bytes)
            self.vram_used_bytes = d.get("vram_used_bytes", 0)
            self.blocks = d.get("blocks", [])
            self.block_ages = d.get("block_ages", [])
            self.storage_health = d.get("storage_health", 1.0)
            self.performance_health = d.get("performance_health", 1.0)
            self.util_health = d.get("util_health", 1.0)
            self.combined_health = d.get("combined_health", 1.0)
            self.io_latency_ns_sum = d.get("io_latency_ns_sum", 0)
            self.io_latency_ns_count = d.get("io_latency_ns_count", 0)
            self.io_latency_ns_max = d.get("io_latency_ns_max", 0)
            self.last_similarity = d.get("last_similarity", 1.0)
            self.evictions = d.get("evictions", 0)
            self.access_count = d.get("access_count", 0)
            self.hit_count = d.get("hit_count", 0)
        except Exception as e:
            print(f"[Persist] Failed to restore VramCacheOrgan: {e}")


# ============================================================
# RAID BAY
# ============================================================

class RaidBay:
    def __init__(self):
        self.sets = []
        for i in range(6):
            npu = ReplicaNPU(cores=16, frequency_ghz=1.5)
            organ = RaidCacheOrgan(
                npu=npu,
                level="raid5",
                spines=4,
                block_size=1024,
                capacity_mb=8.0
            )
            self.sets.append(("RAM", npu, organ))

        vram_npu = ReplicaNPU(cores=16, frequency_ghz=1.5)
        vram_organ = VramCacheOrgan(vram_npu, block_size=1024)
        self.vram = ("VRAM", vram_npu, vram_organ)

    def cache_write_bytes(self, set_idx: int, data_bytes: bytes):
        if set_idx == 6:
            _, _, organ = self.vram
            organ.cache_write_bytes(data_bytes)
        else:
            _, _, organ = self.sets[set_idx]
            organ.cache_write_bytes(data_bytes)

    def cache_read_block(self, set_idx: int, index: int):
        if set_idx == 6:
            _, _, organ = self.vram
            return organ.cache_read_block(index)
        else:
            _, _, organ = self.sets[set_idx]
            return organ.cache_read_block(index)

    def record_io_latency_ns(self, set_idx: int, latency_ns: int):
        if set_idx == 6:
            _, _, organ = self.vram
            organ.record_io_latency_ns(latency_ns)
        else:
            _, _, organ = self.sets[set_idx]
            organ.record_io_latency_ns(latency_ns)

    def deep_npu_cycle_all(self):
        for _, npu, organ in self.sets:
            organ.deep_npu_cycle()
            npu.micro_recovery(0.001)
            npu.check_integrity(external_integrity=organ.combined_health)

        _, vram_npu, vram_organ = self.vram
        vram_organ.deep_npu_cycle()
        vram_npu.micro_recovery(0.001)
        vram_npu.check_integrity(external_integrity=vram_organ.combined_health)


# ============================================================
# MOVIDIUS-LIKE ENGINE
# ============================================================

class MovidiusEngine:
    def __init__(self):
        self.last_confidence = 0.0
        self.last_decision = "Balanced"

    def infer(self, avg_health, avg_hit, avg_util, load_score):
        score = (
            0.35 * avg_health +
            0.35 * avg_hit +
            0.15 * (1.0 - avg_util) +
            0.15 * (1.0 - load_score)
        )
        score = max(0.0, min(1.0, score))
        self.last_confidence = score

        if score > 0.7:
            decision = "Beast"
        elif score > 0.4:
            decision = "Balanced"
        else:
            decision = "Conservative"

        self.last_decision = decision
        return decision, score


# ============================================================
# AI COACH
# ============================================================

class AICoachBay:
    def __init__(self, bay: RaidBay, movidius: MovidiusEngine):
        self.bay = bay
        self.movidius = movidius
        self.last_decision = "Balanced"
        self.last_confidence = 0.0
        self.last_message = "Coach online."

    def update(self, load_score: float):
        healths = []
        hits = []
        utils = []

        for _, _, organ in self.bay.sets:
            healths.append(organ.combined_health)
            hits.append(organ.hit_rate)
            utils.append(organ.utilization)

        _, _, vram_organ = self.bay.vram
        healths.append(vram_organ.combined_health)
        hits.append(vram_organ.hit_rate)
        utils.append(vram_organ.utilization)

        if not healths:
            return

        avg_health = sum(healths) / len(healths)
        avg_hit = sum(hits) / len(hits)
        avg_util = sum(utils) / len(utils)

        decision, conf = self.movidius.infer(avg_health, avg_hit, avg_util, load_score)
        self.last_decision = decision
        self.last_confidence = conf

        if decision == "Beast":
            if avg_health > 0.55:
                self.last_message = "Coach: Beast stance sustained. Backbone thriving under load."
            else:
                self.last_message = "Coach: Beast stance active, but health dipping. Monitor closely."
        elif decision == "Balanced":
            self.last_message = "Coach: Balanced stance. Throughput and safety in equilibrium."
        else:
            self.last_message = "Coach: Conservative stance. System under stress, backing off deep cycles."


# ============================================================
# NPU RAID BACKBONE
# ============================================================

class NpuRaidBackbone:
    def __init__(self):
        self.bay = RaidBay()
        self.movidius = MovidiusEngine()
        self.coach = AICoachBay(self.bay, self.movidius)

        self.mem_latency_ns_sum = 0
        self.mem_latency_ns_count = 0
        self.mem_latency_ns_max = 0

        self.last_deep_cycle_ns = time.perf_counter_ns()
        self.next_interval_ms = 300

        self.telemetry_health = 1.0 if psutil else 0.2

        # Health history (rolling)
        self.health_history = []

        # Persistence
        self.persistence = PersistenceManager.from_env()
        self._save_counter = 0

        # Try to restore previous state
        self.persistence.load_backbone(self)

    def cache_write_bytes(self, set_idx: int, data_bytes: bytes):
        self.bay.cache_write_bytes(set_idx, data_bytes)

    def cache_read_block(self, set_idx: int, index: int):
        return self.bay.cache_read_block(set_idx, index)

    def record_file_latency_ns(self, set_idx: int, latency_ns: int):
        self.bay.record_io_latency_ns(set_idx, latency_ns)

    def record_memory_latency_ns(self, latency_ns: int):
        self.mem_latency_ns_sum += latency_ns
        self.mem_latency_ns_count += 1
        if latency_ns > self.mem_latency_ns_max:
            self.mem_latency_ns_max = latency_ns

    def _estimate_load(self) -> float:
        cpu_load = 0.0
        mem_load = 0.0
        if psutil:
            cpu_load = psutil.cpu_percent(interval=0.0) / 100.0
            vm = psutil.virtual_memory()
            mem_load = vm.percent / 100.0

        evictions = []
        utils = []
        for _, _, organ in self.bay.sets:
            evictions.append(organ.eviction_pressure)
            utils.append(organ.utilization)
        _, _, vram_organ = self.bay.vram
        evictions.append(vram_organ.eviction_pressure)
        utils.append(vram_organ.utilization)

        cache_pressure = 0.0
        if evictions:
            cache_pressure = 0.5 * (sum(evictions) / len(evictions)) + 0.5 * (sum(utils) / len(utils))

        file_lat = []
        for _, _, organ in self.bay.sets:
            if organ.io_latency_ns_max > 0:
                file_lat.append(organ.avg_io_latency_ns / organ.io_latency_ns_max)
        if vram_organ.io_latency_ns_max > 0:
            file_lat.append(vram_organ.avg_io_latency_ns / vram_organ.io_latency_ns_max)

        if file_lat:
            file_lat_load = sum(file_lat) / len(file_lat)
        else:
            file_lat_load = 0.0

        if self.mem_latency_ns_max > 0 and self.mem_latency_ns_count > 0:
            avg_mem_lat = self.mem_latency_ns_sum / self.mem_latency_ns_count
            mem_lat_load = avg_mem_lat / self.mem_latency_ns_max
        else:
            mem_lat_load = 0.0

        latency_load = max(0.0, min(1.0, 0.6 * file_lat_load + 0.4 * mem_lat_load))

        load = (
            0.35 * cpu_load +
            0.25 * mem_load +
            0.20 * cache_pressure +
            0.20 * latency_load
        )
        return max(0.0, min(1.0, load))

    def maybe_deep_cycle(self):
        now_ns = time.perf_counter_ns()
        elapsed_ms = (now_ns - self.last_deep_cycle_ns) / 1_000_000.0
        if elapsed_ms < self.next_interval_ms:
            return

        self.bay.deep_npu_cycle_all()
        self.last_deep_cycle_ns = now_ns

        load = self._estimate_load()

        if load < 0.7:
            t_min, t_max = 100, 300
            scaled = load / 0.7
            self.next_interval_ms = t_min + (t_max - t_min) * scaled
        elif load < 0.9:
            t_min, t_max = 300, 800
            scaled = (load - 0.7) / 0.2
            self.next_interval_ms = t_min + (t_max - t_min) * scaled
        else:
            t_min, t_max = 800, 2000
            scaled = (load - 0.9) / 0.1
            if scaled > 1.0:
                scaled = 1.0
            self.next_interval_ms = t_min + (t_max - t_min) * scaled

        self.coach.update(load)

        snap = self.snapshot()
        self.health_history.append({
            "ts": time.time(),
            "global_util": snap["global_util"],
            "coach_decision": snap["coach_decision"],
            "coach_confidence": snap["coach_confidence"],
            "global_used_bytes": snap["global_used_bytes"],
            "global_max_bytes": snap["global_max_bytes"],
        })
        if len(self.health_history) > 500:
            self.health_history.pop(0)

        self._save_counter += 1
        if self._save_counter >= 10:
            self._save_counter = 0
            self.persistence.save_backbone(self)

    def snapshot(self):
        sets = []
        total_used_bytes = 0
        total_max_bytes = 0

        for idx, (_, npu, organ) in enumerate(self.bay.sets):
            sets.append({
                "index": idx,
                "type": "RAM",
                "capacity_mb": organ.max_mb,
                "used_mb": organ.used_mb,
                "used_bytes": organ.used_bytes,
                "max_bytes": organ.max_bytes,
                "hit_rate": organ.hit_rate,
                "utilization": organ.utilization,
                "storage_health": organ.storage_health,
                "performance_health": organ.performance_health,
                "util_health": organ.util_health,
                "combined_health": organ.combined_health,
                "eviction_pressure": organ.eviction_pressure,
                "avg_latency_ns": organ.avg_io_latency_ns,
                "max_latency_ns": organ.io_latency_ns_max,
                "similarity": organ.last_similarity,
            })
            total_used_bytes += organ.used_bytes
            total_max_bytes += organ.max_bytes

        _, _, vram_organ = self.bay.vram
        sets.append({
            "index": 6,
            "type": "VRAM",
            "capacity_mb": vram_organ.max_mb,
            "used_mb": vram_organ.used_mb,
            "used_bytes": vram_organ.vram_used_bytes,
            "max_bytes": vram_organ.vram_total_bytes,
            "hit_rate": vram_organ.hit_rate,
            "utilization": vram_organ.utilization,
            "storage_health": vram_organ.storage_health,
            "performance_health": vram_organ.performance_health,
            "util_health": vram_organ.util_health,
            "combined_health": vram_organ.combined_health,
            "eviction_pressure": vram_organ.eviction_pressure,
            "avg_latency_ns": vram_organ.avg_io_latency_ns,
            "max_latency_ns": vram_organ.io_latency_ns_max,
            "similarity": vram_organ.last_similarity,
        })
        total_used_bytes += vram_organ.vram_used_bytes
        total_max_bytes += vram_organ.vram_total_bytes

        global_util = 0.0
        if total_max_bytes > 0:
            global_util = min(1.0, total_used_bytes / total_max_bytes)

        return {
            "sets": sets,
            "coach_decision": self.coach.last_decision,
            "coach_confidence": self.coach.last_confidence,
            "coach_message": self.coach.last_message,
            "next_interval_ms": self.next_interval_ms,
            "global_used_bytes": total_used_bytes,
            "global_max_bytes": total_max_bytes,
            "global_util": global_util,
            "telemetry_health": self.telemetry_health,
            "psutil_present": bool(psutil),
        }

    def attach_file_cache_layer(self):
        return FileCacheLayer(self)

    def attach_memory_cache_layer(self):
        return MemoryCacheLayer(self)


# ============================================================
# FILE CACHE LAYER
# ============================================================

class FileCacheLayer:
    def __init__(self, backbone: NpuRaidBackbone):
        self.backbone = backbone
        self.path_map = {
            "C:": 0,
            "D:": 1,
            "E:": 2,
            "F:": 3,
            "G:": 4,
            "H:": 5,
        }

    def _pick_set_for_path(self, path: str) -> int:
        path_upper = path.upper()
        for prefix, idx in self.path_map.items():
            if path_upper.startswith(prefix):
                return idx
        return 0

    def open(self, path, mode="r", *args, **kwargs):
        set_idx = self._pick_set_for_path(path)
        raw_f = open(path, mode, *args, **kwargs)

        is_binary = "b" in mode
        backbone = self.backbone

        class WrappedFile:
            def __init__(self, f, idx, binary):
                self._f = f
                self._idx = idx
                self._binary = binary

            def read(self, *r_args, **r_kwargs):
                start = time.perf_counter_ns()
                data = self._f.read(*r_args, **r_kwargs)
                end = time.perf_counter_ns()
                backbone.record_file_latency_ns(self._idx, end - start)

                if self._binary:
                    if isinstance(data, (bytes, bytearray)):
                        backbone.cache_write_bytes(self._idx, bytes(data))
                else:
                    if isinstance(data, str):
                        backbone.cache_write_bytes(self._idx, data.encode("utf-8", errors="ignore"))
                return data

            def write(self, data):
                start = time.perf_counter_ns()
                res = self._f.write(data)
                end = time.perf_counter_ns()
                backbone.record_file_latency_ns(self._idx, end - start)

                if self._binary:
                    if isinstance(data, (bytes, bytearray)):
                        backbone.cache_write_bytes(self._idx, bytes(data))
                else:
                    if isinstance(data, str):
                        backbone.cache_write_bytes(self._idx, data.encode("utf-8", errors="ignore"))
                return res

            def __getattr__(self, item):
                return getattr(self._f, item)

            def __enter__(self):
                self._f.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return self._f.__exit__(exc_type, exc_val, exc_tb)

        return WrappedFile(raw_f, set_idx, is_binary)


# ============================================================
# MEMORY CACHE LAYER
# ============================================================

class MemoryCacheLayer:
    def __init__(self, backbone: NpuRaidBackbone):
        self.backbone = backbone
        self._synthetic_counter = 0

    def _record_and_feed(self, set_idx: int, payload: bytes, start_ns: int, end_ns: int, source: str):
        self.backbone.record_memory_latency_ns(end_ns - start_ns)
        self.backbone.cache_write_bytes(set_idx, payload)

    def _synthetic_payload(self) -> bytes:
        self._synthetic_counter += 1
        return f"SYNTH|{self._synthetic_counter}|{time.time()}".encode("utf-8")

    def sample_process_memory(self, set_idx: int):
        if not psutil:
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "proc_synth")
            return
        try:
            start = time.perf_counter_ns()
            p = psutil.Process()
            mem = p.memory_info()
            payload = str(mem.rss).encode("utf-8")
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "proc_mem")
        except Exception:
            pass

    def sample_system_memory(self, set_idx: int):
        if not psutil:
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "sys_synth")
            return
        try:
            start = time.perf_counter_ns()
            vm = psutil.virtual_memory()
            payload = str(vm.used).encode("utf-8")
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "sys_mem")
        except Exception:
            pass

    def sample_disk_io(self, set_idx: int):
        if not psutil:
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "disk_syn")
            return
        try:
            start = time.perf_counter_ns()
            dio = psutil.disk_io_counters()
            if dio is None:
                return
            payload = str(dio.read_bytes + dio.write_bytes).encode("utf-8")
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "disk_io")
        except Exception:
            pass

    def sample_cpu(self, set_idx: int):
        if not psutil:
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "cpu_syn")
            return
        try:
            start = time.perf_counter_ns()
            cpu = psutil.cpu_percent(interval=0.0)
            payload = str(int(cpu)).encode("utf-8")
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "cpu")
        except Exception:
            pass

    def sample_net(self, set_idx: int):
        if not psutil:
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "net_syn")
            return
        try:
            start = time.perf_counter_ns()
            net = psutil.net_io_counters()
            if net is None:
                return
            payload = str(net.bytes_sent + net.bytes_recv).encode("utf-8")
            end = time.perf_counter_ns()
            self._record_and_feed(set_idx, payload, start, end, "net")
        except Exception:
            pass

    def feed_vram_with_real_tensors(self):
        if torch is None:
            return
        if not torch.cuda.is_available():
            return

        try:
            _, _, vram_organ = self.backbone.bay.vram
            for _ in range(3):
                t = torch.randn(1024, 1024, device="cuda")
                vram_organ.ingest_torch_tensor(t)
        except Exception:
            pass

    def high_volume_burst(self):
        for i in range(6):
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(i, payload, start, end, f"ram{i}_syn")

        self.sample_process_memory(0)
        self.sample_system_memory(1)
        self.sample_disk_io(2)
        self.sample_cpu(3)
        self.sample_net(4)

        for _ in range(2):
            start = time.perf_counter_ns()
            payload = self._synthetic_payload()
            end = time.perf_counter_ns()
            self._record_and_feed(6, payload, start, end, "vram_syn")

        self.feed_vram_with_real_tensors()


# ============================================================
# TKINTER CONSOLE (COMPACT NUMERIC COCKPIT + PERSIST STATUS)
# ============================================================

class NpuRaidWindow:
    def __init__(self, root, backbone: NpuRaidBackbone, mem_layer: MemoryCacheLayer):
        self.root = root
        self.backbone = backbone
        self.mem_layer = mem_layer

        self.root.title("NPU RAID – Backbone Console (Compact Cockpit)")
        self.root.configure(bg="#05050A")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Telemetry.Horizontal.TProgressbar",
                        troughcolor="#111111",
                        background="#FF4444" if not psutil else "#00FF44",
                        bordercolor="#111111",
                        lightcolor="#FF8888" if not psutil else "#88FFAA",
                        darkcolor="#FF0000" if not psutil else "#00AA44")

        self._build_ui()

    def _build_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#05050A")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title = tk.Label(
            self.main_frame,
            text="NPU RAID BACKBONE – 6x RAM + 1x VRAM – COMPACT NUMERIC COCKPIT",
            fg="#00FFAA",
            bg="#05050A",
            font=("Consolas", 14, "bold")
        )
        title.pack(anchor="w", pady=(0, 2))

        # PERSISTENCE STATUS LINE (under title)
        self.lbl_persist = tk.Label(
            self.main_frame,
            text=self.backbone.persistence.get_status_string(),
            fg="#8888FF",
            bg="#05050A",
            font=("Consolas", 10, "bold")
        )
        self.lbl_persist.pack(anchor="w", pady=(0, 6))

        tele_frame = tk.Frame(self.main_frame, bg="#05050A")
        tele_frame.pack(fill="x", pady=(0, 6))

        tele_label = tk.Label(
            tele_frame,
            text="TELEMETRY HEALTH",
            fg="#CCCCCC",
            bg="#05050A",
            font=("Consolas", 10, "bold")
        )
        tele_label.grid(row=0, column=0, sticky="w")

        self.tele_bar = ttk.Progressbar(
            tele_frame,
            style="Telemetry.Horizontal.TProgressbar",
            length=200,
            maximum=100
        )
        self.tele_bar.grid(row=0, column=1, sticky="w", padx=(8, 8))

        if psutil:
            tele_status_text = "LIVE psutil – real system telemetry"
            tele_status_color = "#00FF44"
        else:
            tele_status_text = "SYNTHETIC ONLY – psutil install failed"
            tele_status_color = "#FF4444"

        self.tele_status_label = tk.Label(
            tele_frame,
            text=tele_status_text,
            fg=tele_status_color,
            bg="#05050A",
            font=("Consolas", 10, "bold")
        )
        self.tele_status_label.grid(row=0, column=2, sticky="w", padx=(8, 0))

        debug_frame = tk.Frame(self.main_frame, bg="#05050A")
        debug_frame.pack(fill="both", expand=True, pady=(4, 8))

        lbl_debug = tk.Label(
            debug_frame,
            text="COCKPIT LINE – ALL SETS (RAM/VRAM, MB+GB, UTIL, HIT, LAT)",
            fg="#8888FF",
            bg="#05050A",
            font=("Consolas", 10, "bold")
        )
        lbl_debug.pack(anchor="w")

        self.txt_debug = tk.Text(
            debug_frame,
            height=14,
            bg="#05050A",
            fg="#AAAAAA",
            insertbackground="#CCCCCC",
            font=("Consolas", 9)
        )
        self.txt_debug.pack(fill="both", expand=True, pady=(4, 0))

        coach_frame = tk.Frame(self.main_frame, bg="#05050A")
        coach_frame.pack(fill="x", pady=(4, 0))

        self.lbl_coach_decision = tk.Label(
            coach_frame,
            text="COACH DECISION: -",
            fg="#FF00FF",
            bg="#05050A",
            font=("Consolas", 11, "bold")
        )
        self.lbl_coach_decision.pack(anchor="w")

        self.lbl_coach_conf = tk.Label(
            coach_frame,
            text="CONFIDENCE: -",
            fg="#FF00FF",
            bg="#05050A",
            font=("Consolas", 10)
        )
        self.lbl_coach_conf.pack(anchor="w")

        self.lbl_interval = tk.Label(
            coach_frame,
            text="DEEP CYCLE INTERVAL: - ms",
            fg="#00FFAA",
            bg="#05050A",
            font=("Consolas", 10)
        )
        self.lbl_interval.pack(anchor="w")

        self.txt_coach = tk.Text(
            coach_frame,
            height=3,
            bg="#05050A",
            fg="#CCCCCC",
            insertbackground="#CCCCCC",
            font=("Consolas", 9)
        )
        self.txt_coach.pack(fill="x", pady=(4, 0))

    def start_periodic_updates(self):
        self._schedule_tick()

    def _schedule_tick(self):
        self._tick_once()
        self.root.after(200, self._schedule_tick)

    def _tick_once(self):
        self.mem_layer.high_volume_burst()
        self.backbone.maybe_deep_cycle()
        self._update_gui()

    def _update_gui(self):
        try:
            snap = self.backbone.snapshot()

            # Update persistence status line
            self.lbl_persist.config(text=self.backbone.persistence.get_status_string())

            tele_val = int(snap["telemetry_health"] * 100)
            self.tele_bar.config(value=tele_val)

            self.txt_debug.delete("1.0", tk.END)

            g_used_bytes = snap["global_used_bytes"]
            g_max_bytes = snap["global_max_bytes"] if snap["global_max_bytes"] > 0 else 1
            g_used_mb = g_used_bytes / (1024 * 1024)
            g_max_mb = g_max_bytes / (1024 * 1024)
            g_used_gb = g_used_bytes / (1024 * 1024 * 1024)
            g_max_gb = g_max_bytes / (1024 * 1024 * 1024)
            g_util = snap["global_util"] * 100.0

            header = (
                "GLOBAL | "
                f"{g_used_mb:7.1f} MB / {g_max_mb:7.1f} MB | "
                f"{g_used_gb:5.2f} GB / {g_max_gb:5.2f} GB | "
                f"{g_util:5.1f}% UTIL\n"
            )
            self.txt_debug.insert(tk.END, header)
            self.txt_debug.insert(tk.END, "-" * 90 + "\n")

            for s in snap["sets"]:
                idx = s["index"]
                kind = s["type"]
                used_bytes = s["used_bytes"]
                max_bytes = s["max_bytes"] if s["max_bytes"] > 0 else 1

                used_mb = used_bytes / (1024 * 1024)
                max_mb = max_bytes / (1024 * 1024)

                used_gb = used_bytes / (1024 * 1024 * 1024)
                max_gb = max_bytes / (1024 * 1024 * 1024)

                util_pct = s["utilization"] * 100.0
                hit = s["hit_rate"]
                if s["max_latency_ns"] > 0:
                    avg_ms = s["avg_latency_ns"] / 1_000_000.0
                    lat_str = f"{avg_ms:6.2f} ms"
                else:
                    lat_str = "   -   "

                line = (
                    f"SET {idx} | {kind:4} | "
                    f"{used_mb:7.1f} MB / {max_mb:7.1f} MB | "
                    f"{used_gb:5.2f} GB / {max_gb:5.2f} GB | "
                    f"{util_pct:5.1f}% UTIL | "
                    f"HIT {hit:5.2f} | LAT {lat_str}\n"
                )
                self.txt_debug.insert(tk.END, line)

            self.lbl_coach_decision.config(text=f"COACH DECISION: {snap['coach_decision']}")
            self.lbl_coach_conf.config(text=f"CONFIDENCE: {snap['coach_confidence']:.2f}")
            self.lbl_interval.config(text=f"DEEP CYCLE INTERVAL: {snap['next_interval_ms']:.0f} ms")
            self.txt_coach.delete("1.0", tk.END)
            self.txt_coach.insert(tk.END, snap["coach_message"])
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():
    if not tk_mod:
        print("tkinter not available. Install tkinter to run the GUI.")
        return

    root = tk.Tk()
    root.geometry("1100x700")

    backbone = NpuRaidBackbone()
    file_cache = backbone.attach_file_cache_layer()
    mem_layer = backbone.attach_memory_cache_layer()

    gui = NpuRaidWindow(root, backbone, mem_layer)

    root.after(300, gui.start_periodic_updates)
    root.mainloop()


if __name__ == "__main__":
    main()

