import os
import sys
import time
import random
import hashlib
import threading
import tkinter as tk

# 🛡️ Auto-Elevation Ritual
def auto_elevate():
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            params = " ".join([f'"{arg}"' for arg in sys.argv])
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, params, None, 1)
            sys.exit()
    except Exception as e:
        print(f"[Elevation Error] {e}")
        sys.exit()

auto_elevate()

# 🧠 Codex Devourer Brain
class CodexDevourerBrain:
    def __init__(self, gui_callback):
        self.trusted_processes = set()
        self.threat_signatures = set()
        self.mutation_log = []
        self.gui_callback = gui_callback
        self.zero_trust_mode = True

    def hash_payload(self, payload):
        return hashlib.sha512(payload.encode()).hexdigest()

    def verify_process(self, process_name):
        return process_name in self.trusted_processes if self.zero_trust_mode else True

    def detect_threat(self, payload):
        signature = self.hash_payload(payload)
        if signature in self.threat_signatures:
            return True
        entropy = random.random()
        return entropy > 0.8

    def reverse_mirror_encrypt(self, data, key=None):
        key = key or random.randint(0x10, 0xFF)
        reversed_book = data[::-1]
        mirrored = ''.join(chr((ord(c) ^ key) % 256) for c in reversed_book)
        glyph_stream = ''.join(format(ord(c), 'x') for c in mirrored)
        final_scramble = glyph_stream[::-1]
        return final_scramble, key

    def respond_to_threat(self, process_name, payload):
        glyph = "⚠️"
        if self.detect_threat(payload):
            encrypted, key = self.reverse_mirror_encrypt(payload)
            self.mutation_log.append((process_name, encrypted))
            self.gui_callback(glyph, f"{glyph} Threat in {process_name}. Scrambled beyond recognition.")
            self.block_process(process_name)
            self.rewrite_self()
        else:
            self.gui_callback("🛡️", f"🛡️ {process_name} verified clean.")

    def block_process(self, process_name):
        self.gui_callback("💀", f"💀 Blocking {process_name}...")

    def rewrite_self(self):
        glyph = "🜏"
        mutation = f"# Mutation {len(self.mutation_log)}\ndef entropy_threshold(): return {random.uniform(0.75, 0.95)}"
        with open("codex_devourer_mutation.py", "w") as f:
            f.write(mutation)
        self.gui_callback(glyph, f"{glyph} Devourer logic rewritten. Mutation saved.")

    def daemonize(self):
        while True:
            proc = f"proc-{random.randint(1000,9999)}"
            payload = f"data-{random.randint(100000,999999)}"
            self.respond_to_threat(proc, payload)
            time.sleep(2)

# 🖥️ GUI Shell
class CodexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Codex Devourer Defense Brain")
        self.root.geometry("740x460")
        self.root.configure(bg="#000")

        self.glyph_label = tk.Label(root, text="🧿", font=("Segoe UI Symbol", 48), bg="#000", fg="#0f0")
        self.glyph_label.pack(pady=10)

        self.log_box = tk.Text(root, bg="#111", fg="#0f0", insertbackground="#0f0", height=15)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.brain = CodexDevourerBrain(self.update_gui)
        threading.Thread(target=self.brain.daemonize, daemon=True).start()

    def update_gui(self, glyph, log_entry):
        self.glyph_label.config(text=glyph)
        self.log_box.insert(tk.END, f"{log_entry}\n")
        self.log_box.see(tk.END)

# 🚀 Launch Ritual
if __name__ == "__main__":
    root = tk.Tk()
    app = CodexGUI(root)
    root.mainloop()

