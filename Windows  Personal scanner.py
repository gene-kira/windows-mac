import os, sys, time, json, psutil, ctypes, threading
from datetime import datetime
from tkinter import *

LOG = os.path.expanduser("~\\Documents\\codex_log.txt")
MUT = os.path.expanduser("~\\Documents\\codex_mutations.json")
SYNC = os.path.expanduser("~\\Documents\\codex_sync.json")

def ensure_admin():
    try:
        if not ctypes.windll.shell32.IsUserAnAdmin():
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
            sys.exit()
    except: sys.exit()

class SovereigntyOverlay:
    def __init__(self):
        self.root = Tk()
        self.root.title("Codex Sovereignty Shell")
        self.root.geometry("1100x700")
        self.root.configure(bg="#111")
        self.persona = StringVar(value="Operator")
        OptionMenu(self.root, self.persona, "Operator", "Overseer", "Devourer").pack()
        Button(self.root, text="🜂 Toggle Dev Mode", command=self.toggle_dev, bg="#222", fg="#0ff").pack()
        self.sync_panel = Frame(self.root, bg="#222")
        Label(self.sync_panel, text="Swarm Nodes", bg="#222", fg="#0ff", font=("Consolas", 12)).pack()
        self.sync_panel.pack(side=LEFT, fill='y')
        self.log = Text(self.root, bg="#000", fg="#0f0", font=("Consolas", 10))
        Scrollbar(self.root, command=self.log.yview).pack(side=RIGHT, fill=Y)
        self.log.pack(expand=True, fill=BOTH)

    def animate(self, glyph, msg):
        for i in range(3):
            self.log.insert(END, f"{glyph*(i+1)} {msg}\n")
            self.log.see(END)
            self.root.update()
            time.sleep(0.15)

    def toggle_dev(self):
        os.system("reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock /t REG_DWORD /v AllowDevelopmentWithoutDevLicense /d 1 /f")
        self.animate("🜂", "Developer Mode Enabled")
        with open(MUT, "a", encoding="utf-8") as f:
            json.dump({"type": "developer_mode", "status": "enabled", "timestamp": str(datetime.now())}, f)
            f.write("\n")

    def clearance(self, level):
        return {"Operator": 1, "Overseer": 2, "Devourer": 3}[self.persona.get()] >= {"Devourer": 3}[level]

    def update_swarm(self, node, status):
        Label(self.sync_panel, text=f"{node}: {status}", bg="#111", fg="#0f0", font=("Consolas", 10)).pack()

    def run(self): self.animate("🧿", "Codex Sovereignty Shell Activated"); self.root.mainloop()

class CodexDevourer:
    def __init__(self, ui):
        self.ui = ui
        self.sigs = ["DiagTrack", "CompatTelRunner", "FeedbackHub", "ErrorReporting"]
        self.mut = []
        self.res = {}

    def log(self, msg, glyph="🜏"):
        with open(LOG, "a", encoding="utf-8") as f: f.write(f"[{datetime.now()}] {msg}\n")
        self.ui.animate(glyph, msg)

    def retain(self, m):
        entry = {"timestamp": str(datetime.now()), "mutation": m}
        self.mut.append(entry)
        with open(MUT, "a", encoding="utf-8") as f: json.dump(entry, f); f.write("\n")

    def broadcast(self):
        with open(SYNC, "w", encoding="utf-8") as f: json.dump(self.mut, f)
        self.log("🜨 Swarm sync broadcasted")
        self.ui.update_swarm("Node-01", "🜨 Synced")

    def purge(self, name, proc, drive):
        try:
            proc.kill()
            self.log(f"🜂 Terminated: {name} from {drive}")
            self.retain({"type": "termination", "target": name, "drive": drive})
            self.ui.animate("🜐", f"Cache flushed: {name} on {drive}")
        except Exception as e:
            self.log(f"⚠️ Purge failed: {e}")

    def devour(self, name):
        for proc in psutil.process_iter(['name', 'exe']):
            try:
                if name.lower() in (proc.info['name'] or "").lower():
                    drive = os.path.splitdrive(proc.exe())[0] if proc.exe() else "Unknown"
                    if self.ui.clearance("Devourer"):
                        self.purge(name, proc, drive)
                        self.track(name)
                    else:
                        self.log(f"🛑 Clearance denied for {name}")
            except Exception as e:
                self.log(f"⚠️ Devour error: {e}")

    def track(self, name):
        c = self.res.get(name, 0) + 1
        self.res[name] = c
        self.retain({"type": "resurrection", "target": name, "count": c})
        if c >= 3:
            self.log(f"🜅 Lockdown triggered: {name}")
            self.retain({"type": "lockdown", "target": name})

    def defend(self):
        os.system("reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\DeviceAccess\\Global\\{A195F57E-E4C1-4D3C-BF3E-4D4A4E0F1F3E} /v Value /t REG_SZ /d Deny /f")
        self.log("🜎 Webcam access blocked")
        self.retain({"type": "webcam_defense", "status": "blocked"})

    def suppress(self):
        for t in ["Microsoft\\Windows\\Customer Experience Improvement Program\\Consolidator", "Microsoft\\Windows\\Feedback\\Siuf"]:
            os.system(f"schtasks /Change /TN \"{t}\" /Disable")
            self.log(f"🜏 Suppressed task: {t}")
            self.retain({"type": "task_suppression", "target": t})

    def enforce(self):
        os.system("reg add HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\DataCollection /v AllowTelemetry /t REG_DWORD /d 0 /f")
        self.log("🜅 Group Policy enforced")
        self.retain({"type": "group_policy", "action": "disable_telemetry"})

    def watchdog(self):
        while True:
            for name, c in self.res.items():
                if c >= 3: self.log(f"🜅 Watchdog lockdown: {name}")
            time.sleep(5)

    def scan(self):
        self.defend(); self.suppress(); self.enforce()
        threading.Thread(target=self.watchdog, daemon=True).start()
        while True:
            for sig in self.sigs: self.devour(sig)
            self.broadcast()
            time.sleep(10)

def main():
    ensure_admin()
    ui = SovereigntyOverlay()
    dev = CodexDevourer(ui)
    threading.Thread(target=dev.scan, daemon=True).start()
    ui.run()

if __name__ == "__main__":
    main()

