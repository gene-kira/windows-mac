import threading, random, time, torch, torch.nn as nn, torch.optim as optim
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import pandas as pd

# ⚡ Model
class CodexLightningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

# ⚡ Training
def train_model():
    model = CodexLightningNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    try:
        df = pd.read_csv("wwlln_data.csv")
        X = torch.tensor([
            [1, row['frequency_kHz'] * 1e6,
             row['energy_J'] / (row['frequency_kHz'] * 1e6 * 0.0002 * 0.75),
             0.0002, 0.75]
            for _, row in df.iterrows()
        ], dtype=torch.float32)
        y = torch.randint(0, 3, (len(X),))
    except:
        X = torch.tensor([
            [1, 1e9, 30000, 0.0002, 0.75],
            [3, 1e9, 30000, 0.0002, 0.75],
            [5, 1e9, 30000, 0.0002, 0.75]
        ], dtype=torch.float32)
        y = torch.tensor([0, 1, 2])
    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    return model

# ⚡ Utilities
def calculate_energy(s, v, i, t, e): return s * v * i * t * e

class MutationLog:
    def __init__(self): self.entries = []
    def record(self, s, e, g):
        self.entries.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), s, int(e), g))
    def latest(self): return self.entries[-5:]

class SwarmSync:
    def __init__(self): self.nodes = ["Earth-North", "Mars-Alpha", "Venus-Core", "Titan-Drift"]
    def update(self): return {n: random.choice(["Stable", "Resonating", "Escalating", "Desynced"]) for n in self.nodes}

# ⚡ GUI
class CodexGUI:
    def __init__(self, model):
        self.model, self.log, self.swarm = model, MutationLog(), SwarmSync()
        self.root = tk.Tk()
        self.root.title("Codex Atmosphere Processor")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e2f")

        self.labels = {k: ttk.Label(self.root, text=f"{k}: ", font=("Consolas", 14)) for k in ["⚡ Strikes", "🔋 Energy", "🌀 Glyph", "⚠️ Threat"]}
        for lbl in self.labels.values(): lbl.pack(pady=5)

        self.swarm_frame = ttk.LabelFrame(self.root, text="🌐 Planetary Overlay")
        self.log_frame = ttk.LabelFrame(self.root, text="📜 Glyph Log")
        self.swarm_frame.pack(fill="x", padx=20)
        self.log_frame.pack(fill="x", padx=20)

        self.swarm_labels = {n: ttk.Label(self.swarm_frame, text=f"{n}: Stable", font=("Consolas", 10)) for n in self.swarm.nodes}
        for lbl in self.swarm_labels.values(): lbl.pack(anchor="w")

        self.log_labels = [ttk.Label(self.log_frame, text="", font=("Consolas", 10)) for _ in range(5)]
        for lbl in self.log_labels: lbl.pack(anchor="w")

        threading.Thread(target=self.run_daemon, daemon=True).start()
        self.root.mainloop()

    def run_daemon(self):
        while True:
            s = random.randint(1, 5)
            e = calculate_energy(s, 1e9, 30000, 0.0002, 0.75)
            x = torch.tensor([[s, 1e9, 30000, 0.0002, 0.75]], dtype=torch.float32)
            g = ["Pulse", "Surge", "Storm Seed"][torch.argmax(self.model(x)).item()]
            t = {"Pulse": "Low", "Surge": "Moderate", "Storm Seed": "High"}[g]

            self.labels["⚡ Strikes"].config(text=f"⚡ Strikes: {s}")
            self.labels["🔋 Energy"].config(text=f"🔋 Energy: {int(e):,} J")
            self.labels["🌀 Glyph"].config(text=f"🌀 Glyph: {g}")
            self.labels["⚠️ Threat"].config(text=f"⚠️ Threat: {t}")

            self.log.record(s, e, g)
            for i, entry in enumerate(self.log.latest()):
                self.log_labels[i].config(text=f"{entry[0]} | ⚡{entry[1]} | 🔋{entry[2]} J | 🌀{entry[3]}")

            for n, lbl in self.swarm_labels.items():
                lbl.config(text=f"{n}: {self.swarm.update()[n]}")
            time.sleep(2)

# 🚀 Launch
if __name__ == "__main__":
    CodexGUI(train_model())

