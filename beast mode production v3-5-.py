import os
import time
import math
import platform
import multiprocessing
import concurrent.futures
import numpy as np

try:
    import psutil
    PSUTIL = True
except ImportError:
    PSUTIL = False


def beast_log(msg: str):
    print(f"[BEAST] {msg}", flush=True)


# ================== CORE WORKLOAD (STABLE) ==================
def beast_vector_cycle(size: int) -> float:
    x = np.random.rand(size).astype("float64")
    y = np.random.rand(size).astype("float64")
    s = np.sin(x)
    c = np.cos(y)
    out = s * c
    return float(out.sum())


# ================== AFFINITY HELPERS ==================
def set_affinity_for_current_process(core_id: int):
    """
    Cross-platform best-effort affinity:
    - Windows/Linux: use psutil if available
    - macOS or no psutil: no affinity (OS decides)
    """
    if not PSUTIL:
        return

    try:
        p = psutil.Process(os.getpid())
        system = platform.system().lower()

        if system in ("windows", "linux"):
            p.cpu_affinity([core_id])
        # macOS: no affinity support; silently ignore
    except Exception:
        # Best-effort only; never crash on affinity
        pass


# ================== ORGANISM / SWARM ==================
class BeastOrganism:
    def __init__(self, name: str, target_util: float = 0.5, size: int = 1_000_000):
        self.name = name
        self.target_util = max(0.05, min(target_util, 1.0))
        self.size = size
        self.last_util = 0.0

    def adjust(self, measured_util: float):
        if measured_util <= 0:
            return
        target = self.target_util * 100
        diff = measured_util - target
        if diff > 5:
            self.size = max(200_000, int(self.size * 0.85))
        elif diff < -5:
            self.size = min(10_000_000, int(self.size * 1.15))
        self.last_util = measured_util


def worker_entry(args):
    """
    Runs in each process:
    - Optionally pins to a specific core (Windows/Linux + psutil)
    - Runs one vector cycle
    """
    core_id, name, size = args
    set_affinity_for_current_process(core_id)
    res = beast_vector_cycle(size)
    return core_id, name, size, res


class BeastSwarm:
    def __init__(self, organisms):
        self.organisms = organisms
        self.cycle = 0
        self.total_cores = max(1, multiprocessing.cpu_count())

        if PSUTIL:
            psutil.cpu_percent(interval=None)

    def telemetry(self):
        if not PSUTIL:
            return "Telemetry unavailable (psutil not installed)"
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        lines = []
        for i, v in enumerate(per_core):
            bar = "#" * int(v / 5)
            lines.append(f"Core {i:02d}: {v:5.1f}% | {bar}")
        return "\n".join(lines)

    def run(self):
        beast_log(f"Total logical cores: {self.total_cores}")
        beast_log(f"Platform: {platform.system()} | psutil: {PSUTIL}")

        while True:
            self.cycle += 1
            start = time.time()

            jobs = []
            # one job per core, mapped round-robin to organisms
            for core_id in range(self.total_cores):
                org = self.organisms[core_id % len(self.organisms)]
                jobs.append((core_id, org.name, org.size))

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.total_cores) as ex:
                results = list(ex.map(worker_entry, jobs))

            end = time.time()
            elapsed = end - start
            util = psutil.cpu_percent(interval=None) if PSUTIL else -1

            for org in self.organisms:
                org.adjust(util)

            print("\n" + "=" * 60)
            beast_log(f"Cycle {self.cycle} | CPU={util if util>=0 else 'N/A'}% | Time={elapsed:.3f}s")

            # summarize per-organism (first occurrence per name)
            seen = set()
            for core_id, name, size, res in results:
                if name in seen:
                    continue
                seen.add(name)
                org = next(o for o in self.organisms if o.name == name)
                beast_log(
                    f"[{name}] size={size:,} | target={int(org.target_util*100)}% | "
                    f"last_util={org.last_util:.1f}% | example_core={core_id} | sum={res:.4f}"
                )

            print("-" * 60)
            print(self.telemetry())
            print("=" * 60)


# ================== MAIN ==================
def main():
    beast_log("Starting BEAST UNIFIED CROSS-PLATFORM")

    organisms = [
        BeastOrganism("Alpha", target_util=0.4),
        BeastOrganism("Beta", target_util=0.6),
    ]

    swarm = BeastSwarm(organisms)
    swarm.run()


if __name__ == "__main__":
    main()

