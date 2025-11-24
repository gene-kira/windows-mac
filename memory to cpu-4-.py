import random
import threading
import time
import psutil

# Dynamic scaling range
MIN_MEMORY_PERCENT = 10
MAX_MEMORY_PERCENT = 50
MAX_CPU_PERCENT = 70  # CPU governor

def dynamic_memory_cap():
    """Scale memory cap between 10% and 50% based on CPU load."""
    cpu = psutil.cpu_percent(interval=0.1)
    # Inverse scaling: high CPU → low memory cap
    scale = 1 - min(cpu / MAX_CPU_PERCENT, 1.0)
    cap = MIN_MEMORY_PERCENT + (MAX_MEMORY_PERCENT - MIN_MEMORY_PERCENT) * scale
    return cap, cpu

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

class MemoryCell:
    def __init__(self, address, data, operation=None, priority=5):
        self.address = address
        self.data = data
        self.operation = operation
        self.priority = priority
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
        cap, cpu = dynamic_memory_cap()

        if mem.percent > cap or cpu > MAX_CPU_PERCENT:
            print(f"[Cell {self.address}] Replication paused (Memory {mem.percent}% > cap {cap:.1f}% | CPU {cpu}%)")
            return
        if self.priority < 5:
            print(f"[Cell {self.address}] Replication denied (priority {self.priority})")
            return

        new_address = len(swarm)
        new_data = self.data ^ random.randint(1, 255) if isinstance(self.data, int) else self.data
        new_op = self.choose_op(bus.consensus())
        new_priority = max(1, min(10, self.priority + random.choice([-1, 0, 1])))
        new_cell = MemoryCell(new_address, new_data, new_op, new_priority)
        swarm.append(new_cell)
        t = threading.Thread(target=new_cell.run, args=(swarm, bus), daemon=True)
        t.start()
        print(f"[Cell {self.address}] Replicated -> New Cell {new_address} (priority {new_priority})")

    def adapt(self, consensus, result):
        if consensus == "add" and isinstance(result, int) and result < 50:
            self.priority = min(10, self.priority + 1)
        elif consensus == "multiply" and isinstance(result, int) and result > 100:
            self.priority = min(10, self.priority + 1)
        else:
            if random.random() < 0.2:
                self.priority = max(1, self.priority - 1)
        if random.random() < 0.3:
            new_op = self.choose_op(consensus)
            self.operation = new_op

    def choose_op(self, consensus):
        if consensus == "add": return add_op
        elif consensus == "multiply": return multiply_op
        else: return mutate_op

# Operations
def add_op(data, input_value): return data + (input_value or 0)
def multiply_op(data, input_value): return data * (input_value or 1)
def mutate_op(data, input_value): return data ^ random.randint(1, 255)

# Initialize swarm + bus
bus = CollectiveBus()
swarm = [
    MemoryCell(0, 5, add_op, priority=7),
    MemoryCell(1, 10, multiply_op, priority=6),
    MemoryCell(2, 42, mutate_op, priority=4)
]

for cell in swarm:
    t = threading.Thread(target=cell.run, args=(swarm, bus), daemon=True)
    t.start()

print("⚡ Borg swarm with dynamic memory scaling (10–50%) + CPU governor running. Press Ctrl+C to stop.")
while True:
    mem = psutil.virtual_memory()
    cap, cpu = dynamic_memory_cap()
    avg_priority = sum(c.priority for c in swarm) / len(swarm)
    print(f"Swarm size: {len(swarm)} | Consensus: {bus.consensus()} | Memory: {mem.percent}% (cap {cap:.1f}%) | CPU: {cpu}% | Avg Priority: {avg_priority:.2f}")
    time.sleep(10)

