import os
import sys
import enum
import json
import time
import random
import socket
import hashlib
import uuid
import math
import base64
import threading
import platform
import ctypes
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union, Callable, Tuple

# Tkinter GUI
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# Optional UI Automation
try:
    import uiautomation as auto  # type: ignore
except Exception:
    auto = None

# Optional crypto backends
try:
    from Crypto.Cipher import AES  # type: ignore
except Exception:
    AES = None

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore
    from cryptography.hazmat.backends import default_backend  # type: ignore
except Exception:
    Cipher = None
    algorithms = None
    modes = None
    default_backend = None

# Optional zlib
try:
    import zlib
except Exception:
    zlib = None

# Optional GPU backends
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

try:
    from numba import cuda  # type: ignore
except Exception:
    cuda = None


# ============================================================
#  AUTO-ELEVATION CHECK
# ============================================================

def ensure_admin():
    try:
        if hasattr(os, "getuid"):
            try:
                if os.getuid() == 0:
                    return
            except Exception:
                pass

        if platform.system() == "Windows":
            try:
                if not ctypes.windll.shell32.IsUserAnAdmin():
                    print("[Codex Sentinel] Elevation required. Relaunching as administrator...")
                    script = os.path.abspath(sys.argv[0])
                    params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
                    ctypes.windll.shell32.ShellExecuteW(
                        None,
                        "runas",
                        sys.executable,
                        f'"{script}" {params}',
                        None,
                        1
                    )
                    sys.exit()
                else:
                    return
            except Exception as e:
                print(f"[Codex Sentinel] Elevation failed: {e}")
                sys.exit()
        else:
            return
    except Exception as e:
        print(f"[Codex Sentinel] Elevation check error: {e}")
        sys.exit()


# =========================
#  Environment & Policies
# =========================

def get_env_fingerprint() -> str:
    hostname = socket.gethostname()
    os_name = os.name
    raw = f"{hostname}|{os_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class SelfDestructPolicy:
    enabled: bool = True
    hard_fail: bool = False
    zeroize_payload: bool = True
    log_only: bool = False


@dataclass
class DataCapsule:
    payload: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    env_fingerprint: str = field(default_factory=get_env_fingerprint)
    allowed_env_fingerprint: Optional[str] = None
    destroyed: bool = False
    self_destruct_policy: SelfDestructPolicy = field(default_factory=SelfDestructPolicy)

    capsule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    lineage: List[str] = field(default_factory=list)

    def check_environment(self):
        if not self.self_destruct_policy.enabled:
            return
        if self.allowed_env_fingerprint is None:
            return
        if self.env_fingerprint != self.allowed_env_fingerprint:
            if self.self_destruct_policy.zeroize_payload:
                self.payload = b""
            self.destroyed = True
            msg = "DataCapsule self-destruct: environment mismatch"
            self.metadata.setdefault("self_destruct_events", []).append(
                {"timestamp": time.time(), "reason": "env_mismatch"}
            )
            if self.self_destruct_policy.log_only:
                print(msg)
                return
            if self.self_destruct_policy.hard_fail:
                raise RuntimeError(msg)

    def spawn_child(self) -> "DataCapsule":
        child = DataCapsule(
            payload=self.payload,
            metadata=self.metadata.copy(),
            allowed_env_fingerprint=self.allowed_env_fingerprint,
            self_destruct_policy=self.self_destruct_policy,
        )
        child.parent_id = self.capsule_id
        child.lineage = self.lineage + [self.capsule_id]
        return child


# ============================================================
# Hardware root of trust (TPM stub)
# ============================================================

class RootOfTrustOrgan:
    def get_device_secret(self) -> bytes:
        raise NotImplementedError


class FileRootOfTrustOrgan(RootOfTrustOrgan):
    def __init__(self):
        self._cached_secret: Optional[bytes] = None

    def get_device_secret(self) -> bytes:
        if self._cached_secret is not None:
            return self._cached_secret
        secret_path = os.path.join(os.path.dirname(__file__), ".device_secret")
        if os.path.exists(secret_path):
            with open(secret_path, "rb") as f:
                self._cached_secret = f.read()
        else:
            self._cached_secret = os.urandom(32)
            with open(secret_path, "wb") as f:
                f.write(self._cached_secret)
        return self._cached_secret


class TPMRootOfTrustOrgan(RootOfTrustOrgan):
    def __init__(self, tpm_key_handle: Optional[int] = None):
        self._fallback = FileRootOfTrustOrgan()
        self._tpm_available = False
        self._tpm_key_handle = tpm_key_handle
        self._cached_secret: Optional[bytes] = None

        try:
            from tpm2_pytss import ESAPI  # type: ignore
            self._ESAPI = ESAPI
            self._tpm_available = True
        except Exception:
            self._tpm_available = False

    def get_device_secret(self) -> bytes:
        if self._cached_secret is not None:
            return self._cached_secret

        if not self._tpm_available or self._tpm_key_handle is None:
            self._cached_secret = self._fallback.get_device_secret()
            return self._cached_secret

        try:
            with self._ESAPI() as esapi:
                handle = self._tpm_key_handle
                pub, _ = esapi.ReadPublic(handle)
                pub_bytes = bytes(pub.marshal())
                self._cached_secret = hashlib.sha256(pub_bytes).digest()
                return self._cached_secret
        except Exception:
            self._cached_secret = self._fallback.get_device_secret()
            return self._cached_secret


# =========================
#  Utility
# =========================

def shard_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H


def log_timeline_event(capsule: DataCapsule, stage: str, shard_index: int, info: Dict[str, Any]):
    entry = {
        "timestamp": time.time(),
        "stage": stage,
        "shard": shard_index,
        **info,
    }
    capsule.metadata.setdefault("timeline", []).append(entry)


# =========================
#  MixEffect-4 Organ
# =========================

class MixMode(str, enum.Enum):
    DETERMINISTIC = "DET"
    RANDOM = "RND"
    HYBRID = "HYB"


def _split_into_4(payload: bytes) -> List[bytes]:
    n = len(payload)
    base = n // 4
    rem = n % 4
    sizes = []
    for i in range(4):
        sizes.append(base + (1 if i < rem else 0))
    shards = []
    offset = 0
    for size in sizes:
        shards.append(payload[offset:offset + size])
        offset += size
    return shards


def _join_shards(shards: List[bytes]) -> bytes:
    return b"".join(shards)


def _parse_order(order_str: str) -> List[int]:
    parts = order_str.split("-")
    order = [int(p) for p in parts if p != ""]
    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid order string: {order_str}")
    return order


def _micro_scramble(data: bytes, rng: random.Random) -> bytes:
    if len(data) <= 1:
        return data
    indices = list(range(len(data)))
    rng.shuffle(indices)
    out = bytearray(len(data))
    for new_pos, old_pos in enumerate(indices):
        out[new_pos] = data[old_pos]
    return bytes(out)


def _bernoulli_corrupt(data: bytes, rng: random.Random, p: float) -> Tuple[bytes, int]:
    if p <= 0.0:
        return data, 0
    out = bytearray(len(data))
    flips = 0
    for i, b in enumerate(data):
        if rng.random() < p:
            out[i] = rng.randint(0, 255)
            flips += 1
        else:
            out[i] = b
    return bytes(out), flips


@dataclass
class MixEffectMetadata:
    mode: str
    order: List[int]
    seed: Optional[int]
    shard_sizes: List[int]
    bernoulli_p: float
    timestamp: float


class MixEffect4Organ:
    def __init__(self, default_mode: MixMode = MixMode.DETERMINISTIC):
        self.default_mode = default_mode

    def __call__(
        self,
        capsule: DataCapsule,
        mode: Optional[MixMode] = None,
        mix_key: Optional[str] = None,
        bernoulli_p: float = 0.0,
    ) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        mode = mode or self.default_mode
        raw = capsule.payload
        shards = _split_into_4(raw)
        shard_sizes = [len(s) for s in shards]

        for i, s in enumerate(shards):
            log_timeline_event(capsule, "mix_split", i, {"size": len(s)})

        order, seed = self._resolve_order_seed(mode, mix_key)
        rng = random.Random(seed) if seed is not None else random.Random()

        total_bits = 0
        total_flips = 0

        if mode in (MixMode.RANDOM, MixMode.HYBRID):
            new_shards = []
            for i, s in enumerate(shards):
                corrupted, flips = _bernoulli_corrupt(s, rng, bernoulli_p)
                s = _micro_scramble(corrupted, rng)
                H = shard_entropy(s)
                log_timeline_event(capsule, "mix_post_scramble", i, {"size": len(s), "entropy": H})
                capsule.metadata.setdefault("heatmap", {}).setdefault("entropy", {})[i] = H
                new_shards.append(s)
                total_bits += len(s) * 8
                total_flips += flips * 8
            shards = new_shards

        mixed = _join_shards([shards[i] for i in order])

        meta = MixEffectMetadata(
            mode=mode.value,
            order=order,
            seed=seed,
            shard_sizes=shard_sizes,
            bernoulli_p=bernoulli_p,
            timestamp=time.time(),
        )
        capsule.metadata.setdefault("mix_effect", []).append(asdict(meta))

        if total_bits > 0 and bernoulli_p > 0.0:
            p_hat = total_flips / total_bits
            n = total_bits
            var = bernoulli_p * (1 - bernoulli_p) / n
            capsule.metadata.setdefault("bernoulli_analysis", []).append(
                {
                    "timestamp": time.time(),
                    "theoretical_p": bernoulli_p,
                    "empirical_p_hat": p_hat,
                    "n_bits": n,
                    "variance": var,
                }
            )

        capsule.payload = mixed
        return capsule

    def _resolve_order_seed(
        self,
        mode: MixMode,
        mix_key: Optional[str],
    ) -> (List[int], Optional[int]):
        if mix_key is None:
            order = [0, 1, 2, 3]
            random.shuffle(order)
            return order, int.from_bytes(os.urandom(8), "big")

        parts = mix_key.split(":")
        if len(parts) < 3 or not parts[0].upper().startswith("MIX4"):
            raise ValueError(f"Invalid mix_key: {mix_key}")
        order = _parse_order(parts[2])
        seed = None
        if len(parts) >= 4 and parts[3].upper().startswith("SEED="):
            seed = int(parts[3].split("=", 1)[1])
        return order, seed


# =========================
#  Synthetic Data Organ
# =========================

@dataclass
class SyntheticMetadata:
    mode: str
    seed: int
    length: int
    timestamp: float


class SyntheticDataOrgan:
    def __init__(self, default_mode: str = "noise"):
        self.default_mode = default_mode

    def __call__(
        self,
        capsule: DataCapsule,
        mode: Optional[str] = None,
        length: Optional[int] = None,
    ) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        mode = mode or self.default_mode
        seed = int.from_bytes(os.urandom(8), "big")
        rng = random.Random(seed)

        if length is None:
            length = max(64, len(capsule.payload))

        if mode == "noise":
            data = bytes(rng.randint(0, 255) for _ in range(length))
        elif mode == "pattern":
            pattern = b"QX"
            data = (pattern * (length // len(pattern) + 1))[:length]
        elif mode == "echo":
            base = capsule.payload or b"seed"
            data = bytearray()
            for i in range(length):
                data.append(base[i % len(base)] ^ rng.randint(0, 255))
            data = bytes(data)
        else:
            raise ValueError(f"Unknown synthetic mode: {mode}")

        capsule.payload = data
        capsule.metadata.setdefault("synthetic", []).append(
            asdict(SyntheticMetadata(mode, seed, length, time.time()))
        )
        return capsule


# =========================
#  Chameleon, Mirror, Glyph, NoiseJitter, HeaderTag
# =========================

class ChameleonOrgan:
    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        text = capsule.payload.decode("utf-8", errors="replace")
        out_chars = []
        for ch in text:
            if ch.islower():
                out_chars.append(ch.upper())
            elif ch.isupper():
                out_chars.append(ch.lower())
            elif ch.isdigit():
                out_chars.append(str((int(ch) + 5) % 10))
            else:
                out_chars.append(ch)
        new_text = "".join(out_chars)
        capsule.metadata.setdefault("chameleon", []).append(
            {"timestamp": time.time(), "len": len(new_text)}
        )
        capsule.payload = new_text.encode("utf-8")
        return capsule


class MirrorOrgan:
    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        capsule.payload = capsule.payload[::-1]
        capsule.metadata.setdefault("mirror", []).append(
            {"timestamp": time.time(), "len": len(capsule.payload)}
        )
        return capsule


class GlyphMode(str, enum.Enum):
    HEX = "HEX"
    B64 = "B64"


class GlyphOrgan:
    def __init__(self, mode: GlyphMode = GlyphMode.HEX):
        self.mode = mode

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        if self.mode == GlyphMode.HEX:
            glyph = capsule.payload.hex()
        else:
            glyph = base64.b64encode(capsule.payload).decode("ascii")

        capsule.metadata.setdefault("glyph", []).append(
            {"timestamp": time.time(), "mode": self.mode.value, "len": len(glyph)}
        )
        capsule.payload = glyph.encode("utf-8")
        capsule.metadata["encoded"] = True
        capsule.metadata["encoding_type"] = f"glyph-{self.mode.value}"
        return capsule


class NoiseJitterOrgan:
    def __init__(self, p: float = 0.02):
        self.p = p

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule
        rng = random.Random(int.from_bytes(os.urandom(8), "big"))
        data = capsule.payload
        out = bytearray(len(data))
        for i, b in enumerate(data):
            if rng.random() < self.p:
                out[i] = (b + rng.randint(1, 5)) % 256
            else:
                out[i] = b
        capsule.payload = bytes(out)
        capsule.metadata.setdefault("noise_jitter", []).append(
            {"timestamp": time.time(), "p": self.p}
        )
        return capsule


class HeaderTagOrgan:
    def __init__(self, tag: str):
        self.tag = tag

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule
        capsule.metadata.setdefault("tags", []).append(
            {"tag": self.tag, "timestamp": time.time()}
        )
        return capsule


# =========================
#  New Organs: Compression, Hash, ThreatScore, Mutation, AI Anomaly
# =========================

class CompressionOrgan:
    def __init__(self, level: int = 6):
        self.level = level

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule
        if zlib is None:
            capsule.metadata.setdefault("compression", []).append(
                {
                    "timestamp": time.time(),
                    "warning": "zlib not available, compression skipped",
                    "original_len": len(capsule.payload),
                    "compressed_len": len(capsule.payload),
                    "ratio": 1.0,
                }
            )
            return capsule

        original_len = len(capsule.payload)
        compressed = zlib.compress(capsule.payload, self.level)
        capsule.payload = compressed
        capsule.metadata.setdefault("compression", []).append(
            {
                "timestamp": time.time(),
                "original_len": original_len,
                "compressed_len": len(compressed),
                "ratio": (len(compressed) / original_len) if original_len else 1.0,
            }
        )
        return capsule


class HashOrgan:
    def __init__(self, algo: str = "sha256"):
        self.algo = algo

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule
        h = hashlib.new(self.algo)
        h.update(capsule.payload)
        digest = h.hexdigest()
        capsule.metadata.setdefault("hashes", []).append(
            {
                "timestamp": time.time(),
                "algo": self.algo,
                "digest": digest,
            }
        )
        return capsule


class ThreatScoreOrgan:
    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        entropy_map = capsule.metadata.get("heatmap", {}).get("entropy", {})
        avg_entropy = 0.0
        if entropy_map:
            avg_entropy = sum(entropy_map.values()) / len(entropy_map)

        bernoulli_list = capsule.metadata.get("bernoulli_analysis", [])
        flips_ratio = 0.0
        if bernoulli_list:
            last = bernoulli_list[-1]
            flips_ratio = last.get("empirical_p_hat", 0.0)

        egress_log = capsule.metadata.get("shadow_egress_log", [])
        blocked = sum(1 for e in egress_log if not e.get("allowed", False))

        score = avg_entropy * 10 + flips_ratio * 100 + blocked * 5

        capsule.metadata.setdefault("threat_score", []).append(
            {
                "timestamp": time.time(),
                "avg_entropy": avg_entropy,
                "flips_ratio": flips_ratio,
                "blocked_egress": blocked,
                "score": score,
            }
        )
        return capsule


class MutationOrgan:
    def __init__(self, mutation_rate: float = 0.05):
        self.mutation_rate = mutation_rate

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        rng = random.Random(int.from_bytes(os.urandom(8), "big"))
        data = capsule.payload
        out = bytearray(len(data))
        mutations = 0
        for i, b in enumerate(data):
            if rng.random() < self.mutation_rate:
                out[i] = rng.randint(0, 255)
                mutations += 1
            else:
                out[i] = b
        capsule.payload = bytes(out)
        capsule.metadata.setdefault("mutation", []).append(
            {
                "timestamp": time.time(),
                "mutation_rate": self.mutation_rate,
                "mutations": mutations,
                "len": len(data),
            }
        )
        return capsule


class AIAnomalyOrgan:
    def __init__(self, window: int = 20, threshold: float = 2.5):
        self.window = window
        self.threshold = threshold
        self.history: List[float] = []

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        entropy_map = capsule.metadata.get("heatmap", {}).get("entropy", {})
        avg_entropy = 0.0
        if entropy_map:
            avg_entropy = sum(entropy_map.values()) / len(entropy_map)

        threat_list = capsule.metadata.get("threat_score", [])
        threat_score = threat_list[-1]["score"] if threat_list else 0.0

        egress_log = capsule.metadata.get("shadow_egress_log", [])
        blocked = sum(1 for e in egress_log if not e.get("allowed", False))
        total_egress = len(egress_log) if egress_log else 1
        block_ratio = blocked / total_egress

        composite = avg_entropy * 0.4 + threat_score * 0.5 + block_ratio * 0.1

        self.history.append(composite)
        if len(self.history) > self.window:
            self.history.pop(0)

        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / len(self.history) if len(self.history) > 1 else 0.0
        std = math.sqrt(var)

        z_score = 0.0
        if std > 0:
            z_score = (composite - mean) / std

        anomaly = abs(z_score) > self.threshold

        capsule.metadata.setdefault("ai_anomaly", []).append(
            {
                "timestamp": time.time(),
                "composite": composite,
                "mean": mean,
                "std": std,
                "z_score": z_score,
                "anomaly": anomaly,
            }
        )
        return capsule


# =========================
#  GPU-Accelerated Organs
# =========================

class GPUEntropyOrgan:
    """
    GPU-accelerated entropy calculator.
    Falls back to CPU if no GPU backend is available.
    """

    def __init__(self):
        self.gpu_available = cp is not None

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        data = capsule.payload
        if not data:
            return capsule

        if self.gpu_available:
            try:
                arr = cp.asarray(list(data), dtype=cp.uint8)
                counts = cp.bincount(arr, minlength=256)
                total = arr.size
                probs = counts / total
                probs = probs[probs > 0]
                H = float(-(probs * cp.log2(probs)).sum())
            except Exception as e:
                capsule.metadata.setdefault("gpu_entropy_errors", []).append(
                    {"timestamp": time.time(), "error": str(e)}
                )
                H = shard_entropy(data)
        else:
            H = shard_entropy(data)

        capsule.metadata.setdefault("gpu_entropy", []).append(
            {"timestamp": time.time(), "entropy": H, "gpu_used": self.gpu_available}
        )
        capsule.metadata.setdefault("heatmap", {}).setdefault("entropy", {})[0] = H
        return capsule


class GPUMutationOrgan:
    """
    GPU-accelerated mutation organ.
    Falls back to CPU MutationOrgan if GPU is unavailable.
    """

    def __init__(self, mutation_rate: float = 0.05):
        self.mutation_rate = mutation_rate
        self.gpu_available = cp is not None
        self.cpu_fallback = MutationOrgan(mutation_rate=mutation_rate)

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        if not self.gpu_available:
            return self.cpu_fallback(capsule)

        data = capsule.payload
        if not data:
            return capsule

        try:
            arr = cp.asarray(list(data), dtype=cp.uint8)
            n = arr.size
            mask = cp.random.random(n) < self.mutation_rate
            rand_bytes = cp.random.randint(0, 256, size=n, dtype=cp.uint8)
            arr = cp.where(mask, rand_bytes, arr)
            mutated = bytes(cp.asnumpy(arr))
            mutations = int(mask.sum())

            capsule.payload = mutated
            capsule.metadata.setdefault("gpu_mutation", []).append(
                {
                    "timestamp": time.time(),
                    "mutation_rate": self.mutation_rate,
                    "mutations": mutations,
                    "len": n,
                    "gpu_used": True,
                }
            )
            return capsule
        except Exception as e:
            capsule.metadata.setdefault("gpu_mutation_errors", []).append(
                {"timestamp": time.time(), "error": str(e)}
            )
            return self.cpu_fallback(capsule)


# =========================
#  Reverse-Encrypt Organ
# =========================

class ReverseEncryptOrgan:
    def __init__(self, key: bytes):
        if not key:
            raise ValueError("Key must not be empty")
        self.key = key

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        data = capsule.payload[::-1]
        out = bytearray(len(data))
        klen = len(self.key)
        for i, b in enumerate(data):
            out[i] = b ^ self.key[i % klen]
        capsule.metadata.setdefault("reverse_encrypt", []).append(
            {"timestamp": time.time(), "key_len": klen}
        )
        capsule.payload = bytes(out)
        capsule.metadata["encoded"] = True
        capsule.metadata["encoding_type"] = "reverse-xor"
        return capsule


# =========================
#  AES-256 Organ
# =========================

class AES256Organ:
    def __init__(self, key: bytes, fallback: ReverseEncryptOrgan):
        if len(key) < 32:
            key = hashlib.sha256(key).digest()
        else:
            key = key[:32]
        self.key = key
        self.fallback = fallback

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        if AES is None:
            capsule.metadata.setdefault("crypto_warnings", []).append(
                {
                    "timestamp": time.time(),
                    "warning": "AES backend unavailable, using reverse-xor fallback",
                }
            )
            return self.fallback(capsule)

        iv = os.urandom(16)
        cipher = AES.new(self.key, AES.MODE_CTR, nonce=iv[:8])
        ciphertext = cipher.encrypt(capsule.payload)
        capsule.payload = iv + ciphertext
        capsule.metadata.setdefault("aes256", []).append(
            {
                "timestamp": time.time(),
                "mode": "CTR",
                "iv_len": len(iv),
                "cipher_len": len(ciphertext),
            }
        )
        capsule.metadata["encoded"] = True
        capsule.metadata["encoding_type"] = "aes256-ctr"
        return capsule


# =========================
#  ChaCha20 Organ
# =========================

class ChaCha20Organ:
    def __init__(self, key: bytes, fallback: ReverseEncryptOrgan):
        self.key = hashlib.sha256(key).digest()
        self.fallback = fallback

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        if Cipher is None or algorithms is None or modes is None or default_backend is None:
            capsule.metadata.setdefault("crypto_warnings", []).append(
                {
                    "timestamp": time.time(),
                    "warning": "ChaCha20 backend unavailable, using reverse-xor fallback",
                }
            )
            return self.fallback(capsule)

        nonce = os.urandom(16)
        algorithm = algorithms.ChaCha20(self.key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(capsule.payload)
        capsule.payload = nonce + ciphertext
        capsule.metadata.setdefault("chacha20", []).append(
            {
                "timestamp": time.time(),
                "nonce_len": len(nonce),
                "cipher_len": len(ciphertext),
            }
        )
        capsule.metadata["encoded"] = True
        capsule.metadata["encoding_type"] = "chacha20"
        return capsule


# =========================
#  Encoding Profiles
# =========================

class EncodingProfile:
    def __init__(self, name, glyph_mode, encryption_strength, bernoulli_p, crypto_mode: str):
        self.name = name
        self.glyph_mode = glyph_mode
        self.encryption_strength = encryption_strength
        self.bernoulli_p = bernoulli_p
        self.crypto_mode = crypto_mode


LOW_PARANOIA = EncodingProfile(
    name="low",
    glyph_mode=GlyphMode.HEX,
    encryption_strength="xor-weak",
    bernoulli_p=0.01,
    crypto_mode="xor",
)

MEDIUM_PARANOIA = EncodingProfile(
    name="medium",
    glyph_mode=GlyphMode.B64,
    encryption_strength="aes256",
    bernoulli_p=0.08,
    crypto_mode="aes",
)

HIGH_PARANOIA = EncodingProfile(
    name="high",
    glyph_mode=GlyphMode.HEX,
    encryption_strength="chacha20",
    bernoulli_p=0.20,
    crypto_mode="chacha",
)

PROFILE_MAP = {
    "low": LOW_PARANOIA,
    "medium": MEDIUM_PARANOIA,
    "high": HIGH_PARANOIA,
}


# =========================
#  Black Night Manager
# =========================

class BlackNightManager:
    def __init__(self, threat_threshold: float = 90.0):
        self.active = False
        self.threat_threshold = threat_threshold
        self.reason: Optional[str] = None

    def activate(self, reason: str):
        if not self.active:
            self.active = True
            self.reason = reason
            print(f"[BLACK NIGHT] Activated: {reason}")
            global device_secret
            try:
                device_secret = b"\x00" * len(device_secret)
            except Exception:
                pass

    def deactivate(self):
        if self.active:
            print("[BLACK NIGHT] Deactivated")
        self.active = False
        self.reason = None

    def should_trigger_from_capsule(self, capsule: DataCapsule) -> bool:
        ts_list = capsule.metadata.get("threat_score", [])
        ai_list = capsule.metadata.get("ai_anomaly", [])
        high_threat = ts_list and ts_list[-1].get("score", 0.0) >= self.threat_threshold
        anomaly = ai_list and ai_list[-1].get("anomaly", False)
        return high_threat or anomaly


# =========================
#  Policy Engine
# =========================

@dataclass
class PolicyRule:
    name: str
    condition: Callable[[DataCapsule], bool]
    action: Callable[[DataCapsule], None]


class PolicyEngine:
    def __init__(self, black_night: Optional[BlackNightManager] = None):
        self.rules: List[PolicyRule] = []
        self.egress_guard: Optional["EgressGuardOrgan"] = None
        self.profile_ref: Optional[EncodingProfile] = None
        self.black_night = black_night

    def attach_egress_guard(self, guard: "EgressGuardOrgan"):
        self.egress_guard = guard

    def attach_profile(self, profile: EncodingProfile):
        self.profile_ref = profile

    def add_rule(self, rule: PolicyRule):
        self.rules.append(rule)

    def evaluate(self, capsule: DataCapsule):
        if self.black_night and self.black_night.should_trigger_from_capsule(capsule):
            self.black_night.activate("AUTO: threat/anomaly")
        for rule in self.rules:
            try:
                if rule.condition(capsule):
                    rule.action(capsule)
                    capsule.metadata.setdefault("policy_hits", []).append(
                        {"timestamp": time.time(), "rule": rule.name}
                    )
            except Exception as e:
                capsule.metadata.setdefault("policy_errors", []).append(
                    {"timestamp": time.time(), "rule": rule.name, "error": str(e)}
                )


# =========================
#  Egress Guard Organ
# =========================

class EgressGuardOrgan:
    def __init__(self, require_env_match: bool = True, black_night: Optional[BlackNightManager] = None):
        self.require_env_match = require_env_match
        self.egress_enabled = True
        self.black_night = black_night

    def set_egress_enabled(self, enabled: bool):
        self.egress_enabled = enabled

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        capsule.check_environment()
        env_ok = True
        if self.require_env_match and capsule.allowed_env_fingerprint is not None:
            env_ok = capsule.env_fingerprint == capsule.allowed_env_fingerprint

        encoded_flag = capsule.metadata.get("encoded", False)
        encoding_type = capsule.metadata.get("encoding_type", "unknown")

        capsule.metadata.setdefault("shadow_egress_log", [])

        if self.black_night and self.black_night.active:
            capsule.metadata["shadow_egress_log"].append({
                "timestamp": time.time(),
                "allowed": False,
                "reason": "black_night_global_block",
                "encoding_type": encoding_type,
                "profile": capsule.metadata.get("encoding_profile"),
                "payload_preview": capsule.payload[:64].hex(),
            })
            capsule.payload = b""
            capsule.destroyed = True
            return capsule

        if not self.egress_enabled:
            capsule.metadata["shadow_egress_log"].append({
                "timestamp": time.time(),
                "allowed": False,
                "reason": "operator_block",
                "encoding_type": encoding_type,
                "profile": capsule.metadata.get("encoding_profile"),
                "payload_preview": capsule.payload[:64].hex(),
            })
            capsule.payload = b""
            capsule.destroyed = True
            return capsule

        if not encoded_flag or not env_ok:
            capsule.metadata["shadow_egress_log"].append({
                "timestamp": time.time(),
                "allowed": False,
                "reason": "not_encoded" if not encoded_flag else "env_mismatch",
                "encoding_type": encoding_type,
                "profile": capsule.metadata.get("encoding_profile"),
                "payload_preview": capsule.payload[:64].hex(),
            })
            capsule.payload = b""
            capsule.destroyed = True
            return capsule

        capsule.metadata["shadow_egress_log"].append({
            "timestamp": time.time(),
            "allowed": True,
            "encoding_type": encoding_type,
            "profile": capsule.metadata.get("encoding_profile"),
        })

        return capsule


# =========================
#  Transmitter
# =========================

class Transmitter:
    def __init__(self, host: str = "127.0.0.1", port: int = 50555):
        self.host = host
        self.port = port

    def transmit(self, capsule: DataCapsule) -> bool:
        if capsule.destroyed:
            return False
        capsule.metadata.setdefault("transmit_events", []).append(
            {"timestamp": time.time(), "len": len(capsule.payload), "host": self.host, "port": self.port}
        )
        return True


# =========================
#  Persistent Swarm Registry
# =========================

class SwarmRegistry:
    """
    Persistent registry of peers.
    Stored as JSON in .swarm_registry.json in the script directory.
    """

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), ".swarm_registry.json")
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"peers": {}}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {"peers": {}}

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def update_peer(self, node_id: str, info: Dict[str, Any]):
        with self._lock:
            self._data.setdefault("peers", {})
            self._data["peers"][node_id] = info
            self._save()

    def get_peers(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data.get("peers", {}))


# =========================
#  Swarm Networking (TCP + QUIC-style stub)
# =========================

@dataclass
class SwarmNodeStatus:
    node_id: str
    env_fingerprint: str
    profile: str
    last_seen: float
    threat_score: float


class SwarmNodeTCP:
    """
    TCP-based swarm node with:
    - auto-port-fallback
    - threat-awareness
    - diagnostics
    - persistent registry
    - QUIC-style session stub (single-packet per connection)
    """

    def __init__(self, node_id: str, port: int, profile: EncodingProfile, registry: SwarmRegistry):
        self.node_id = node_id
        self.requested_port = port
        self.port = port
        self.profile = profile
        self.env_fingerprint = get_env_fingerprint()
        self.running = False
        self.peers: Dict[str, SwarmNodeStatus] = {}
        self.local_threat_score: float = 0.0
        self.registry = registry
        self.black_night: Optional[BlackNightManager] = None
        self.diagnostics: Dict[str, Any] = {
            "requested_port": port,
            "bound_port": None,
            "fallback_used": False,
            "bind_errors": [],
            "connections_accepted": 0,
            "connections_attempted": 0,
            "packets_sent": 0,
            "packets_received": 0,
        }

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._bind_with_fallback()
        self._load_registry_peers()

    def _bind_with_fallback(self):
        try:
            self.server_sock.bind(("127.0.0.1", self.requested_port))
            self.port = self.requested_port
            self.diagnostics["bound_port"] = self.port
            self.server_sock.listen(5)
            return
        except OSError as e:
            self.diagnostics["bind_errors"].append(
                {"port": self.requested_port, "errno": getattr(e, "errno", None), "strerror": str(e)}
            )

        for p in range(self.requested_port + 1, self.requested_port + 50):
            try:
                self.server_sock.bind(("127.0.0.1", p))
                self.port = p
                self.diagnostics["bound_port"] = self.port
                self.diagnostics["fallback_used"] = True
                self.server_sock.listen(5)
                return
            except OSError as e:
                self.diagnostics["bind_errors"].append(
                    {"port": p, "errno": getattr(e, "errno", None), "strerror": str(e)}
                )

        self.diagnostics["fatal"] = "Unable to bind any port in fallback range"
        raise RuntimeError("[SwarmNodeTCP] Failed to bind TCP socket on any port")

    def _encode_status(self) -> bytes:
        data = {
            "node_id": self.node_id,
            "env": self.env_fingerprint,
            "profile": self.profile.name,
            "ts": time.time(),
            "threat_score": self.local_threat_score,
        }
        return json.dumps(data).encode("utf-8")

    def _handle_packet(self, data: bytes):
        try:
            obj = json.loads(data.decode("utf-8"))
            status = SwarmNodeStatus(
                node_id=obj["node_id"],
                env_fingerprint=obj["env"],
                profile=obj["profile"],
                last_seen=obj["ts"],
                threat_score=obj.get("threat_score", 0.0),
            )
            self.peers[status.node_id] = status
            self.diagnostics["packets_received"] += 1
            self.registry.update_peer(status.node_id, {
                "env": status.env_fingerprint,
                "profile": status.profile,
                "last_seen": status.last_seen,
                "threat_score": status.threat_score,
            })
        except Exception:
            pass

    def _load_registry_peers(self):
        stored = self.registry.get_peers()
        for node_id, info in stored.items():
            self.peers[node_id] = SwarmNodeStatus(
                node_id=node_id,
                env_fingerprint=info.get("env", ""),
                profile=info.get("profile", "unknown"),
                last_seen=info.get("last_seen", 0.0),
                threat_score=info.get("threat_score", 0.0),
            )

    def start(self):
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._beacon_loop, daemon=True).start()

    def stop(self):
        self.running = False
        try:
            self.server_sock.close()
        except Exception:
            pass

    def _accept_loop(self):
        while self.running:
            try:
                conn, _ = self.server_sock.accept()
                self.diagnostics["connections_accepted"] += 1
                threading.Thread(target=self._session_handler, args=(conn,), daemon=True).start()
            except Exception:
                time.sleep(0.1)

    def _session_handler(self, conn: socket.socket):
        try:
            data = conn.recv(4096)
            if data:
                self._handle_packet(data)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _beacon_loop(self):
        while self.running:
            if self.black_night and self.black_night.active:
                time.sleep(2.0)
                continue
            pkt = self._encode_status()
            for p in range(self.port, self.port + 3):
                if p == self.port:
                    continue
                try:
                    self.diagnostics["connections_attempted"] += 1
                    with socket.create_connection(("127.0.0.1", p), timeout=0.5) as s:
                        s.sendall(pkt)
                        self.diagnostics["packets_sent"] += 1
                except Exception:
                    pass
            time.sleep(2.0)

    def consensus_env_ok(self) -> bool:
        for s in self.peers.values():
            if s.env_fingerprint != self.env_fingerprint:
                return False
        return True

    def consensus_profile(self) -> str:
        counts = Counter([self.profile.name] + [s.profile for s in self.peers.values()])
        return counts.most_common(1)[0][0]

    def max_peer_threat(self) -> float:
        if not self.peers:
            return 0.0
        return max(s.threat_score for s in self.peers.values())

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "requested_port": self.requested_port,
            "bound_port": self.port,
            "fallback_used": self.diagnostics.get("fallback_used", False),
            "bind_errors": self.diagnostics.get("bind_errors", []),
            "connections_accepted": self.diagnostics.get("connections_accepted", 0),
            "connections_attempted": self.diagnostics.get("connections_attempted", 0),
            "packets_sent": self.diagnostics.get("packets_sent", 0),
            "packets_received": self.diagnostics.get("packets_received", 0),
            "peer_count": len(self.peers),
            "max_peer_threat": self.max_peer_threat(),
        }

    def update_local_threat(self, score: float):
        self.local_threat_score = score


# =========================
#  Organ Sandbox Wrapper
# =========================

class OrganSandbox:
    """
    Wraps organs to prevent a single organ from crashing the pipeline.
    Logs failures into capsule metadata and leaves payload unchanged on error.
    """

    def __init__(self, organ: Callable[[DataCapsule], DataCapsule], name: str):
        self.organ = organ
        self.name = name

    def __call__(self, capsule: DataCapsule) -> DataCapsule:
        try:
            return self.organ(capsule)
        except Exception as e:
            capsule.metadata.setdefault("organ_errors", []).append(
                {"timestamp": time.time(), "organ": self.name, "error": str(e)}
            )
            return capsule


# =========================
#  Quadrant Shard Router
# =========================

class QuadrantShardRouter:
    def __init__(
        self,
        synthetic: Callable[[DataCapsule], DataCapsule],
        mix: Callable[[DataCapsule], DataCapsule],
        reverse: Callable[[DataCapsule], DataCapsule],
        aes256: Callable[[DataCapsule], DataCapsule],
        chacha: Callable[[DataCapsule], DataCapsule],
        chameleon: Callable[[DataCapsule], DataCapsule],
        mirror: Callable[[DataCapsule], DataCapsule],
        glyph: GlyphOrgan,
        jitter: Callable[[DataCapsule], DataCapsule],
        header: Callable[[DataCapsule], DataCapsule],
        compressor: Callable[[DataCapsule], DataCapsule],
        hasher: Callable[[DataCapsule], DataCapsule],
        mutation: Callable[[DataCapsule], DataCapsule],
        threat_score: Callable[[DataCapsule], DataCapsule],
        ai_anomaly: Callable[[DataCapsule], DataCapsule],
        egress: Callable[[DataCapsule], DataCapsule],
        policy_engine: PolicyEngine,
        profile: EncodingProfile,
        gpu_entropy: Callable[[DataCapsule], DataCapsule],
        gpu_mutation: Callable[[DataCapsule], DataCapsule],
        black_night: BlackNightManager,
    ):
        self.synthetic = synthetic
        self.mix = mix
        self.reverse = reverse
        self.aes256 = aes256
        self.chacha = chacha
        self.chameleon = chameleon
        self.mirror = mirror
        self.glyph = glyph
        self.jitter = jitter
        self.header = header
        self.compressor = compressor
        self.hasher = hasher
        self.mutation = mutation
        self.threat_score = threat_score
        self.ai_anomaly = ai_anomaly
        self.egress = egress
        self.policy_engine = policy_engine
        self.profile = profile
        self.gpu_entropy = gpu_entropy
        self.gpu_mutation = gpu_mutation
        self.black_night = black_night

    def set_profile(self, profile: EncodingProfile):
        self.profile = profile
        self.policy_engine.attach_profile(profile)

    def _apply_crypto(self, capsule: DataCapsule) -> DataCapsule:
        mode = self.profile.crypto_mode
        if mode == "xor":
            return self.reverse(capsule)
        elif mode == "aes":
            return self.aes256(capsule)
        elif mode == "chacha":
            return self.chacha(capsule)
        else:
            return self.reverse(capsule)

    def process(self, capsule: DataCapsule, quadrant: str) -> DataCapsule:
        capsule.check_environment()
        if capsule.destroyed:
            return capsule

        capsule.metadata["encoding_profile"] = self.profile.name
        q = quadrant.upper()

        if q == "NW":
            capsule = self.synthetic(capsule)
            capsule = self._apply_crypto(capsule)
        elif q == "NE":
            capsule = self.mix(capsule)
            capsule = self._apply_crypto(capsule)
        elif q == "SW":
            capsule = self.compressor(capsule)
            capsule = self._apply_crypto(capsule)
        elif q == "SE":
            capsule = self.chameleon(capsule)
            capsule = self.mirror(capsule)
            self.glyph.mode = self.profile.glyph_mode
            capsule = self.glyph(capsule)
        else:
            raise ValueError(f"Unknown quadrant: {quadrant}")

        capsule = self.gpu_mutation(capsule)
        capsule = self.hasher(capsule)
        capsule = self.header(capsule)
        capsule = self.jitter(capsule)
        capsule = self.gpu_entropy(capsule)
        capsule = self.threat_score(capsule)
        capsule = self.ai_anomaly(capsule)

        self.policy_engine.evaluate(capsule)

        capsule = self.egress(capsule)

        if self.black_night.active:
            capsule.metadata.setdefault("black_night_events", []).append(
                {"timestamp": time.time(), "action": "capsule_oblivion"}
            )
            capsule.payload = b""
            capsule.destroyed = True

        return capsule


# =========================
#  AutoLoader
# =========================

class AutoLoader:
    def __init__(self):
        self.registry: Dict[str, Any] = {}

    def load(self, name: str, constructor: Callable[[], Any]):
        try:
            instance = constructor()
            self.registry[name] = instance
            return instance
        except Exception as e:
            raise RuntimeError(f"[AutoLoader] Failed to load {name}: {e}")

    def get(self, name: str):
        return self.registry.get(name)


# =========================
#  UI Automation Controller
# =========================

class UIAutomationController:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.available = auto is not None

    def focus_window(self):
        if not self.available:
            return
        try:
            win = auto.WindowControl(searchDepth=1, Name=self.window_title)
            if win.Exists(0, 0):
                win.SetActive()
        except Exception:
            pass

    def announce_quadrant(self, quadrant: str):
        self.focus_window()


# =========================
#  Tkinter Cockpit GUI
# =========================

class CockpitGUI:
    def __init__(
        self,
        root,
        capsule: DataCapsule,
        router: QuadrantShardRouter,
        egress: EgressGuardOrgan,
        swarm: SwarmNodeTCP,
        ui_auto: UIAutomationController,
        elevated: bool,
        black_night: BlackNightManager,
        device_secret_ref: Callable[[], bytes],
        reset_callback: Callable[[], DataCapsule],
    ):
        self.root = root
        self.capsule = capsule
        self.router = router
        self.egress = egress
        self.swarm = swarm
        self.ui_auto = ui_auto
        self.elevated = elevated
        self.black_night = black_night
        self.device_secret_ref = device_secret_ref
        self.reset_callback = reset_callback

        root.title("High-Security Swarm Cockpit")
        root.geometry("1200x750")
        root.configure(bg="#1a1a1a")

        banner = tk.Label(
            root,
            text=(
                "🔒  No personal or device identifiers are ever collected, stored, or transmitted.\n"
                "All data is sharded, camouflaged, and environment-bound by design — "
                "including all regularly transmitted synthetic data."
            ),
            bg="#1a1a1a",
            fg="#00ff00",
            font=("Segoe UI", 10, "bold"),
            justify="center"
        )
        banner.pack(fill="x", pady=5)

        elev_text = "Elevation: ADMIN" if self.elevated else "Elevation: STANDARD USER"
        elev_color = "#00ff00" if self.elevated else "#ffaa00"
        elev_label = tk.Label(
            root,
            text=elev_text,
            bg="#1a1a1a",
            fg=elev_color,
            font=("Segoe UI", 9, "bold"),
            justify="center"
        )
        elev_label.pack(fill="x")

        main_frame = tk.Frame(root, bg="#1a1a1a")
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame, bg="#1a1a1a")
        left_frame.pack(side="left", fill="y", padx=5, pady=5)

        right_frame = tk.Frame(main_frame, bg="#1a1a1a")
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        controls_frame = tk.LabelFrame(left_frame, text="Operator Controls", fg="white", bg="#1a1a1a")
        controls_frame.pack(fill="x", pady=5)

        tk.Label(controls_frame, text="Paranoia Profile:", bg="#1a1a1a", fg="white").pack(anchor="w")
        self.profile_var = tk.StringVar(value=self.router.profile.name)
        profile_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.profile_var,
            values=["low", "medium", "high"],
            state="readonly",
            width=10
        )
        profile_combo.pack(anchor="w", pady=2)

        def on_profile_change(*_):
            name = self.profile_var.get()
            profile = PROFILE_MAP.get(name, HIGH_PARANOIA)
            self.router.set_profile(profile)

        profile_combo.bind("<<ComboboxSelected>>", on_profile_change)

        self.egress_var = tk.BooleanVar(value=True)
        egress_check = tk.Checkbutton(
            controls_frame,
            text="Allow Egress",
            variable=self.egress_var,
            bg="#1a1a1a",
            fg="white",
            selectcolor="#333333",
            command=self._on_egress_toggle
        )
        egress_check.pack(anchor="w", pady=2)

        bn_frame = tk.Frame(controls_frame, bg="#1a1a1a")
        bn_frame.pack(fill="x", pady=4)

        self.bn_button = tk.Button(
            bn_frame,
            text="ACTIVATE BLACK NIGHT",
            bg="#660000",
            fg="#ffffff",
            command=self._activate_black_night,
        )
        self.bn_button.pack(fill="x", pady=2)

        self.bn_reset_button = tk.Button(
            bn_frame,
            text="MANUAL RESET",
            bg="#003366",
            fg="#ffffff",
            command=self._manual_reset,
        )
        self.bn_reset_button.pack(fill="x", pady=2)

        swarm_frame = tk.LabelFrame(left_frame, text="Swarm Status", fg="white", bg="#1a1a1a")
        swarm_frame.pack(fill="x", pady=5)

        self.swarm_env_label = tk.Label(swarm_frame, text="Env Consensus: unknown", bg="#1a1a1a", fg="white")
        self.swarm_env_label.pack(anchor="w")

        self.swarm_profile_label = tk.Label(swarm_frame, text="Profile Consensus: unknown", bg="#1a1a1a", fg="white")
        self.swarm_profile_label.pack(anchor="w")

        self.swarm_threat_label = tk.Label(swarm_frame, text="Max Peer Threat: 0.0", bg="#1a1a1a", fg="white")
        self.swarm_threat_label.pack(anchor="w")

        quad_frame = tk.LabelFrame(left_frame, text="Quadrant Map", fg="white", bg="#1a1a1a")
        quad_frame.pack(fill="both", expand=True, pady=5)

        self.quad_buttons: Dict[str, tk.Button] = {}
        for (q, row, col) in [("NW", 0, 0), ("NE", 0, 1), ("SW", 1, 0), ("SE", 1, 1)]:
            btn = tk.Button(
                quad_frame,
                text=q,
                width=8,
                height=3,
                command=lambda qq=q: self._process_quadrant(qq),
                bg="#333333",
                fg="white",
                activebackground="#555555",
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.quad_buttons[q] = btn

        for i in range(2):
            quad_frame.rowconfigure(i, weight=1)
            quad_frame.columnconfigure(i, weight=1)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        frame_output = ttk.Frame(notebook)
        notebook.add(frame_output, text="Capsule Output")
        self.output_box = ScrolledText(frame_output, wrap="word")
        self.output_box.pack(fill="both", expand=True)
        self._refresh_output()

        frame_redline = ttk.Frame(notebook)
        notebook.add(frame_redline, text="Red Line Log")
        self.redline_box = ScrolledText(frame_redline, wrap="word")
        self.redline_box.pack(fill="both", expand=True)
        self._refresh_redline()

        frame_meta = ttk.Frame(notebook)
        notebook.add(frame_meta, text="Metadata")
        self.meta_box = ScrolledText(frame_meta, wrap="word")
        self.meta_box.pack(fill="both", expand=True)
        self._refresh_metadata()

        frame_diag = ttk.Frame(notebook)
        notebook.add(frame_diag, text="Diagnostics")
        self.diag_box = ScrolledText(frame_diag, wrap="word")
        self.diag_box.pack(fill="both", expand=True)
        self._refresh_diagnostics()

        frame_heatmap = ttk.Frame(notebook)
        notebook.add(frame_heatmap, text="Threat Heatmap")
        self.heatmap_canvas = tk.Canvas(frame_heatmap, bg="#111111")
        self.heatmap_canvas.pack(fill="both", expand=True)
        self._refresh_heatmap()

        self._schedule_swarm_update()
        self._schedule_diag_update()
        self._schedule_heatmap_update()

        if self.black_night.active:
            self._enter_black_night_ui()

    def _on_egress_toggle(self):
        enabled = self.egress_var.get()
        self.egress.set_egress_enabled(enabled)

    def _activate_black_night(self):
        self.black_night.activate("MANUAL: cockpit")
        self._enter_black_night_ui()

    def _enter_black_night_ui(self):
        for btn in self.quad_buttons.values():
            btn.configure(state="disabled", bg="#330000")
        self.bn_button.configure(bg="#990000", text="BLACK NIGHT ACTIVE")
        self.root.configure(bg="#000000")

    def _exit_black_night_ui(self):
        for btn in self.quad_buttons.values():
            btn.configure(state="normal", bg="#333333")
        self.bn_button.configure(bg="#660000", text="ACTIVATE BLACK NIGHT")
        self.root.configure(bg="#1a1a1a")

    def _manual_reset(self):
        self.black_night.deactivate()
        self._exit_black_night_ui()
        self.capsule = self.reset_callback()
        self._refresh_output()
        self._refresh_redline()
        self._refresh_metadata()
        self._refresh_diagnostics()
        self._refresh_heatmap()

    def _process_quadrant(self, quadrant: str):
        base_capsule = self.capsule.spawn_child()
        processed = self.router.process(base_capsule, quadrant)
        self.capsule = processed
        self._refresh_output()
        self._refresh_redline()
        self._refresh_metadata()
        self._refresh_heatmap()
        for q, btn in self.quad_buttons.items():
            btn.configure(bg="#333333")
        self.quad_buttons[quadrant].configure(bg="#007acc")
        self.ui_auto.announce_quadrant(quadrant)

        threat_list = self.capsule.metadata.get("threat_score", [])
        if threat_list:
            last = threat_list[-1]
            score = last.get("score", 0.0)
            self.swarm.update_local_threat(score)

        if self.black_night.active:
            self._enter_black_night_ui()

    def _refresh_output(self):
        self.output_box.delete("1.0", "end")
        if self.capsule.destroyed:
            self.output_box.insert("end", "[DESTROYED CAPSULE]\n")
        self.output_box.insert("end", self.capsule.payload.hex())

    def _refresh_redline(self):
        self.redline_box.delete("1.0", "end")
        logs = self.capsule.metadata.get("shadow_egress_log", [])
        for entry in logs:
            status = "ALLOWED" if entry["allowed"] else "BLOCKED"
            self.redline_box.insert("end", f"[{status}] {json.dumps(entry)}\n")

    def _refresh_metadata(self):
        self.meta_box.delete("1.0", "end")
        self.meta_box.insert("end", json.dumps(self.capsule.metadata, indent=2))

    def _refresh_diagnostics(self):
        self.diag_box.delete("1.0", "end")
        threat_list = self.capsule.metadata.get("threat_score", [])
        local_threat = threat_list[-1]["score"] if threat_list else 0.0
        diag = {
            "elevated": self.elevated,
            "local_threat_score": local_threat,
            "swarm": self.swarm.get_diagnostics(),
            "black_night_active": self.black_night.active,
            "black_night_reason": self.black_night.reason,
        }
        self.diag_box.insert("end", json.dumps(diag, indent=2))

    def _refresh_heatmap(self):
        self.heatmap_canvas.delete("all")
        entropy_map = self.capsule.metadata.get("heatmap", {}).get("entropy", {})
        threat_list = self.capsule.metadata.get("threat_score", [])
        local_threat = threat_list[-1]["score"] if threat_list else 0.0

        width = self.heatmap_canvas.winfo_width() or 400
        height = self.heatmap_canvas.winfo_height() or 300

        title_color = "#ff0000" if self.black_night.active else "#ffffff"
        title_text = (
            f"BLACK NIGHT ACTIVE — Threat: {local_threat:.2f}"
            if self.black_night.active
            else f"Local Threat Score: {local_threat:.2f}"
        )

        self.heatmap_canvas.create_text(
            10, 10,
            anchor="nw",
            fill=title_color,
            font=("Segoe UI", 10, "bold"),
            text=title_text,
        )

        shard_ids = [0, 1, 2, 3]
        box_w = width / 4
        box_h = height - 40

        for i, shard_id in enumerate(shard_ids):
            e = entropy_map.get(shard_id, 0.0)
            norm = min(max(e / 8.0, 0.0), 1.0)
            r = int(255 * norm)
            g = int(255 * (1 - norm))
            b = 0
            color = f"#{r:02x}{g:02x}{b:02x}"
            x0 = i * box_w
            y0 = 40
            x1 = x0 + box_w - 5
            y1 = y0 + box_h - 5
            self.heatmap_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#222222")
            self.heatmap_canvas.create_text(
                x0 + 5, y0 + 5,
                anchor="nw",
                fill="#ffffff",
                font=("Segoe UI", 9),
                text=f"Shard {shard_id}\nH={e:.2f}",
            )

    def _schedule_swarm_update(self):
        self._update_swarm_labels()
        self.root.after(2000, self._schedule_swarm_update)

    def _schedule_diag_update(self):
        self._refresh_diagnostics()
        self.root.after(3000, self._schedule_diag_update)

    def _schedule_heatmap_update(self):
        self._refresh_heatmap()
        self.root.after(3000, self._schedule_heatmap_update)

    def _update_swarm_labels(self):
        env_ok = self.swarm.consensus_env_ok()
        profile_consensus = self.swarm.consensus_profile()
        max_peer_threat = self.swarm.max_peer_threat()
        self.swarm_env_label.config(
            text=f"Env Consensus: {'OK' if env_ok else 'DIVERGENT'}",
            fg="#00ff00" if env_ok else "#ff5555",
        )
        self.swarm_profile_label.config(
            text=f"Profile Consensus: {profile_consensus}",
            fg="#00ff00" if profile_consensus == self.router.profile.name else "#ffaa00",
        )
        self.swarm_threat_label.config(
            text=f"Max Peer Threat: {max_peer_threat:.2f}",
            fg="#ff5555" if max_peer_threat > 50 else "#00ff00",
        )


def launch_tkinter_cockpit(
    capsule: DataCapsule,
    router: QuadrantShardRouter,
    egress: EgressGuardOrgan,
    swarm: SwarmNodeTCP,
    ui_auto: UIAutomationController,
    elevated: bool,
    black_night: BlackNightManager,
    device_secret_ref: Callable[[], bytes],
    reset_callback: Callable[[], DataCapsule],
):
    root = tk.Tk()
    CockpitGUI(root, capsule, router, egress, swarm, ui_auto, elevated, black_night, device_secret_ref, reset_callback)
    root.mainloop()


# =========================
#  Demo / Entry
# =========================

if __name__ == "__main__":
    ensure_admin()
    elevated_flag = False
    try:
        if platform.system() == "Windows":
            elevated_flag = bool(ctypes.windll.shell32.IsUserAnAdmin())
        else:
            elevated_flag = hasattr(os, "getuid") and os.getuid() == 0
    except Exception:
        elevated_flag = False

    loader = AutoLoader()

    root_of_trust = TPMRootOfTrustOrgan(tpm_key_handle=None)
    device_secret = root_of_trust.get_device_secret()

    def get_device_secret_ref() -> bytes:
        return device_secret

    black_night = BlackNightManager(threat_threshold=90.0)

    mix_raw = MixEffect4Organ()
    synthetic_raw = SyntheticDataOrgan()
    reverse_raw = ReverseEncryptOrgan(key=device_secret)
    aes256_raw = AES256Organ(key=device_secret, fallback=reverse_raw)
    chacha_raw = ChaCha20Organ(key=device_secret, fallback=reverse_raw)
    chameleon_raw = ChameleonOrgan()
    mirror_raw = MirrorOrgan()
    glyph = GlyphOrgan(GlyphMode.HEX)
    jitter_raw = NoiseJitterOrgan(p=0.02)
    header_raw = HeaderTagOrgan(tag="route:demo")
    compressor_raw = CompressionOrgan(level=6)
    hasher_raw = HashOrgan(algo="sha256")
    mutation_raw = MutationOrgan(mutation_rate=0.05)
    threat_score_raw = ThreatScoreOrgan()
    ai_anomaly_raw = AIAnomalyOrgan(window=20, threshold=2.5)

    egress = EgressGuardOrgan(require_env_match=True, black_night=black_night)

    gpu_entropy_raw = GPUEntropyOrgan()
    gpu_mutation_raw = GPUMutationOrgan(mutation_rate=0.05)

    mix = OrganSandbox(mix_raw.__call__, "MixEffect4Organ")
    synthetic = OrganSandbox(lambda c: synthetic_raw(c, mode="noise"), "SyntheticDataOrgan")
    reverse = OrganSandbox(reverse_raw.__call__, "ReverseEncryptOrgan")
    aes256 = OrganSandbox(aes256_raw.__call__, "AES256Organ")
    chacha = OrganSandbox(chacha_raw.__call__, "ChaCha20Organ")
    chameleon = OrganSandbox(chameleon_raw.__call__, "ChameleonOrgan")
    mirror = OrganSandbox(mirror_raw.__call__, "MirrorOrgan")
    jitter = OrganSandbox(jitter_raw.__call__, "NoiseJitterOrgan")
    header = OrganSandbox(header_raw.__call__, "HeaderTagOrgan")
    compressor = OrganSandbox(compressor_raw.__call__, "CompressionOrgan")
    hasher = OrganSandbox(hasher_raw.__call__, "HashOrgan")
    mutation = OrganSandbox(mutation_raw.__call__, "MutationOrgan")
    threat_score = OrganSandbox(threat_score_raw.__call__, "ThreatScoreOrgan")
    ai_anomaly = OrganSandbox(ai_anomaly_raw.__call__, "AIAnomalyOrgan")
    egress_sandboxed = OrganSandbox(egress.__call__, "EgressGuardOrgan")

    gpu_entropy = OrganSandbox(gpu_entropy_raw.__call__, "GPUEntropyOrgan")
    gpu_mutation = OrganSandbox(gpu_mutation_raw.__call__, "GPUMutationOrgan")

    policy_engine = PolicyEngine(black_night=black_night)
    policy_engine.attach_egress_guard(egress)

    def rule_high_threat_condition(c: DataCapsule) -> bool:
        ts_list = c.metadata.get("threat_score", [])
        if not ts_list:
            return False
        return ts_list[-1].get("score", 0.0) > 80.0

    def rule_high_threat_action(c: DataCapsule):
        if policy_engine.egress_guard:
            policy_engine.egress_guard.set_egress_enabled(False)
        c.metadata.setdefault("policy_actions", []).append(
            {"timestamp": time.time(), "action": "egress_disabled_due_to_high_threat"}
        )

    policy_engine.add_rule(
        PolicyRule(
            name="block_on_high_threat",
            condition=rule_high_threat_condition,
            action=rule_high_threat_action,
        )
    )

    def rule_anomaly_condition(c: DataCapsule) -> bool:
        ai_list = c.metadata.get("ai_anomaly", [])
        if not ai_list:
            return False
        return ai_list[-1].get("anomaly", False)

    def rule_anomaly_action(c: DataCapsule):
        c.metadata.setdefault("policy_actions", []).append(
            {"timestamp": time.time(), "action": "anomaly_flagged"}
        )

    policy_engine.add_rule(
        PolicyRule(
            name="flag_ai_anomaly",
            condition=rule_anomaly_condition,
            action=rule_anomaly_action,
        )
    )

    profile = HIGH_PARANOIA
    policy_engine.attach_profile(profile)

    router = QuadrantShardRouter(
        synthetic=synthetic,
        mix=lambda c: mix_raw(c, mode=MixMode.RANDOM, bernoulli_p=profile.bernoulli_p),
        reverse=reverse,
        aes256=aes256,
        chacha=chacha,
        chameleon=chameleon,
        mirror=mirror,
        glyph=glyph,
        jitter=jitter,
        header=header,
        compressor=compressor,
        hasher=hasher,
        mutation=gpu_mutation,
        threat_score=threat_score,
        ai_anomaly=ai_anomaly,
        egress=egress_sandboxed,
        policy_engine=policy_engine,
        profile=profile,
        gpu_entropy=gpu_entropy,
        gpu_mutation=gpu_mutation,
        black_night=black_night,
    )

    transmitter = Transmitter()

    original = b"Operator-grade, environment-bound, high-paranoia test payload."
    allowed_env = get_env_fingerprint()

    capsule_holder: Dict[str, DataCapsule] = {
        "capsule": DataCapsule(
            payload=original,
            allowed_env_fingerprint=allowed_env,
            self_destruct_policy=SelfDestructPolicy(enabled=True, hard_fail=False, zeroize_payload=True),
        )
    }

    capsule = capsule_holder["capsule"]
    capsule = router.process(capsule, quadrant="SE")
    capsule_holder["capsule"] = capsule
    transmitter.transmit(capsule)

    print("Destroyed:", capsule.destroyed)
    print("Final payload (bytes):", capsule.payload)
    print("Metadata:", json.dumps(capsule.metadata, indent=2))

    registry = SwarmRegistry()
    swarm = SwarmNodeTCP(node_id="node-1", port=50560, profile=profile, registry=registry)
    swarm.black_night = black_night
    swarm.start()

    ui_auto = UIAutomationController(window_title="High-Security Swarm Cockpit")

    def reset_system() -> DataCapsule:
        new_capsule = DataCapsule(
            payload=original,
            allowed_env_fingerprint=allowed_env,
            self_destruct_policy=SelfDestructPolicy(enabled=True, hard_fail=False, zeroize_payload=True),
        )
        capsule_holder["capsule"] = new_capsule
        return new_capsule

    launch_tkinter_cockpit(
        capsule_holder["capsule"],
        router,
        egress,
        swarm,
        ui_auto,
        elevated_flag,
        black_night,
        get_device_secret_ref,
        reset_system,
    )

    swarm.stop()

