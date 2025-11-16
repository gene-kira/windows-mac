import time
import random

class MemoryDriftPanel:
    def __init__(self):
        self.sync_drift = 0.0
        self.glyph_echoes = 0
        self.paradox_residue = 0
        self.drift_index = 0.0
        self.alpha = 1.0
        self.beta = 0.01
        self.gamma = 0.05
        self.max_echoes = 10000
        self.max_residue = 500

    def update_metrics(self):
        try:
            self.sync_drift = round(random.uniform(10.0, 15.0), 2)
            self.glyph_echoes += random.randint(50, 150)
            self.paradox_residue += random.randint(1, 10)
            self.glyph_echoes = min(self.glyph_echoes, self.max_echoes)
            self.paradox_residue = min(self.paradox_residue, self.max_residue)
        except Exception as e:
            print(f"[ERROR] update_metrics: {e}")

    def calculate_drift_index(self):
        try:
            self.drift_index = (
                self.alpha * self.sync_drift +
                self.beta * self.glyph_echoes +
                self.gamma * self.paradox_residue
            )
            return round(self.drift_index, 3)
        except Exception as e:
            print(f"[ERROR] calculate_drift_index: {e}")
            return 0.0

    def display_panel(self):
        try:
            print("\n=== MEMORY DRIFT PANEL ===")
            print(f"Sync Drift %     : {self.sync_drift}%")
            print(f"Glyph Echoes     : {self.glyph_echoes}")
            print(f"Paradox Residue  : {self.paradox_residue}")
            print(f"Drift Index      : {self.calculate_drift_index()}")
            print("==========================")
        except Exception as e:
            print(f"[ERROR] display_panel: {e}")

    def run(self, delay=2):
        cycle = 1
        try:
            while True:
                print(f"\n[Cycle {cycle}]")
                self.update_metrics()
                self.display_panel()
                time.sleep(delay)
                cycle += 1
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Drift panel halted by user.")
        except Exception as e:
            print(f"[CRITICAL] run loop crashed: {e}")

if __name__ == "__main__":
    panel = MemoryDriftPanel()
    panel.run()

