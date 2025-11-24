import importlib
import subprocess
import sys
import json
import os
import random
import threading
import time
import psutil

# -------------------------------
# Manifest-driven autoloader
# -------------------------------
MANIFEST_FILE = "swarm_manifest.json"

default_manifest = {"libraries": ["psutil"]}  # baseline requirement
if not os.path.exists(MANIFEST_FILE):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(default_manifest, f, indent=4)

def ensure_libraries_from_manifest():
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    for lib in manifest.get("libraries", []):
        try:
            importlib.import_module(lib)
            print(f"[Autoloader] Library '{lib}' already available.")
        except ImportError:
            print(f"[Autoloader] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

def declare_library(lib_name):
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    if lib_name not in manifest["libraries"]:
        manifest["libraries"].append(lib_name)
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=4)
        print(f"[Swarm] Declared new library: {lib_name}")
        ensure_libraries_from_manifest()

ensure_libraries_from_manifest()

# -------------------------------
# Governors
# -------------------------------
MAX_MEMORY_PERCENT = 60   # memory cap
MAX_CPU_PERCENT = 70      # CPU cap

# -------------------------------
# Collective Bus
# -------------------------------
class CollectiveBus:
    def __init__(self):
        self.lock = threading.Lock()
        self.results = []
    def broadcast(self, result):
        with self.lock:
            self.results.append(result)
    def consensus(self):
        with self.lock:
            if not self.results:
                return None
            numeric = [r for r in self.results if isinstance(r, (int, float))]
            if numeric:
                avg = sum(numeric) / len(numeric)
                if avg > 50: return "multiply"
                elif avg > 20: return "add"
                else: return "mutate"
            else:
                return "mutate"

# -------------------------------
# Memory Cell
# -------------------------------
class MemoryCell:
    def __init__(self, address, data, operation=None, priority=5):
        self.address = address
        self.data = data
        self.operation = operation
        self.priority = priority  # hierarchy level
        self.lock = threading.Lock()
        self.replications = 0

    def run(self, swarm, bus):
        while True:
            with self.lock:
                if self.operation:
                    result = self.operation(self.data, random.randint(1, 10))
                else:
                    result = self.data
                bus.broadcast(result)
                self.adapt(bus.consensus(), result)
                self.replications += 1
                if self.replications >= 5:
                    self.replicate(swarm, bus)
                    self.replications = 0
            time.sleep(random.uniform(0.5, 1.5))

    def replicate(self, swarm, bus):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)

        # Arbitration: only replicate if governors allow AND priority is high
        if mem.percent > MAX_MEMORY_PERCENT or cpu > MAX_CPU_PERCENT:
            print(f"[Cell {self.address}] Replication paused (Memory {mem.percent}% / CPU {cpu}%)")
            return
        if self.priority < 5:  # low priority cells blocked
            print(f"[Cell {self.address}] Replication denied (priority {self.priority})")
            return

        new_address = len(swarm)
        new_data = self.data ^ random.randint(1, 255) if isinstance(self.data, int) else self.data
        new_op = self.choose_op(bus.consensus())
        new_priority = max(1, min(10, self.priority + random.choice([-1, 0, 1])))  # mutate priority
        new_cell = MemoryCell(new_address, new_data, new_op, new_priority)
        swarm.append(new_cell)
        t = threading.Thread(target=new_cell.run, args=(swarm, bus), daemon=True)
        t.start()
        print(f"[Cell {self.address}] Replicated -> New Cell {new_address} (priority {new_priority})")

    def adapt(self, consensus, result):
        # Adjust priority based on alignment with consensus
        if consensus == "add" and isinstance(result, int) and result < 50:
            self.priority = min(10, self.priority + 1)
        elif consensus == "multiply" and isinstance(result, int) and result > 100:
            self.priority = min(10, self.priority + 1)
        else:
            if random.random() < 0.2:
                self.priority = max(1, self.priority - 1)

        # Mutate operation occasionally
        if random.random() < 0.3:
            new_op = self.choose_op(consensus)
            self.operation = new_op
            # Borg-style: declare new library when mutating
            lib_choice = random.choice(["numpy", "cryptography", "matplotlib", "requests"])
            declare_library(lib_choice)

    def choose_op(self, consensus):
        if consensus == "add": return add_op
        elif consensus == "multiply": return multiply_op
        else: return mutate_op

# -------------------------------
# Operations
# -------------------------------
def add_op(data, input_value): return data + (input_value or 0)
def multiply_op(data, input_value): return data * (input_value or 1)
def mutate_op(data, input_value): return data ^ random.randint(1, 255)

# -------------------------------
# Initialize swarm + bus
# -------------------------------
bus = CollectiveBus()
swarm = [
    MemoryCell(0, 5, add_op, priority=7),
    MemoryCell(1, 10, multiply_op, priority=6),
    MemoryCell(2, 42, mutate_op, priority=4)
]

for cell in swarm:
    t = threading.Thread(target=cell.run, args=(swarm, bus), daemon=True)
    t.start()

print("⚡ Borg swarm with manifest autoloader + dual governor + priority hierarchy running. Press Ctrl+C to stop.")
while True:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    avg_priority = sum(c.priority for c in swarm) / len(swarm)
    print(f"Swarm size: {len(swarm)} | Consensus: {bus.consensus()} | Memory: {mem.percent}% | CPU: {cpu}% | Avg Priority: {avg_priority:.2f}")
    time.sleep(10)

