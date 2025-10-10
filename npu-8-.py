import sys, subprocess, platform, datetime, time, threading, hashlib, numpy as np, psutil, importlib
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox
from PyQt5.QtCore import QTimer

# === Setup ===
def autoload(pkgs): [subprocess.call([sys.executable, "-m", "pip", "install", p]) for p in pkgs if not importlib.util.find_spec(p)]
autoload(["PyQt5", "openvino", "numpy", "psutil", "pycoral"])
log, outflow, codex = [], [], {"retention": 7, "phantoms": []}
rules, OS = {"blocked": set()}, platform.system()
countries = sorted(["US","CA","GB","DE","FR","IT","ES","JP","CN","RU","IN","BR","MX","KR","AU","ZA","NG","EG","TR","IR","PK","ID","AR","SA","IL","UA","TH","VN","NL","SE","NO","CH","PL","BE","MY","PH","CO","CL","NZ"])

# === Core ===
def record(msg): log.append(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")
def purge(data, exp): time.sleep((exp - datetime.datetime.now()).total_seconds()); outflow[:] = [d for d in outflow if d["data"] != data]; record(f"💀 Purged: {data}")
def register(data, kind, origin): 
    if origin in rules["blocked"]: record(f"🌐 Blocked {origin}: {data}"); return
    ttl = {"backdoor":3,"mac_ip":30,"fake":30,"personal":86400,"bio":86400,"ip":30,"mac":30,"telemetry":30}.get(kind,60)
    exp = datetime.datetime.now() + datetime.timedelta(seconds=ttl)
    outflow.append({"data":data,"type":kind,"origin":origin,"expires":exp})
    record(f"Registered {kind} for purge in {ttl}s")
    threading.Thread(target=lambda: purge(data, exp), daemon=True).start()
def mutate(t): codex.update({"retention":2}) if t=="Phantom" else None; codex["phantoms"].append("ghost_sync") if t=="Phantom" else None
def replay(feed): [feed.append(e) for e in log]

# === NPU ===
def get_npu_status():
    try:
        if OS == "Windows":
            out = subprocess.check_output(['powershell', '-Command', 'Get-Counter "\\Intel(R) AI Boost Engine\\Utilization Percentage"']).decode()
            return next((l.strip() for l in out.splitlines() if "Utilization" in l), "NPU status unavailable")
        return subprocess.check_output(['nputop', '--once']).decode().splitlines()[0]
    except: return "NPU telemetry error"

def run_ncs2():
    from openvino.runtime import Core
    try: return str(Core().compile_model("model.xml", "MYRIAD")(inputs={"input": np.ones((1,3,224,224), np.float32)}))
    except Exception as e: return f"NCS2 error: {e}"

def run_coral():
    try:
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import common
        i = make_interpreter("model_edgetpu.tflite"); i.allocate_tensors(); common.set_input(i, np.ones((1,*common.input_size(i),3), np.uint8)); i.invoke()
        return str(common.output_tensor(i, 0))
    except Exception as e: return f"Coral error: {e}"

def run_hailo():
    try:
        if importlib.util.find_spec("degirum") is None: return "Hailo SDK not installed."
        import degirum.pysdk as dg
        return str(dg.connect().load_model("hailo_model")(np.ones((1,3,224,224), np.float32)))
    except Exception as e: return f"Hailo error: {e}"

# === GUI ===
class Shell(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASI Surveillance Console")
        self.setGeometry(100,100,1600,900)
        self.setStyleSheet("background:#000; color:#0ff; font-family:'JetBrains Mono'; font-size:14px;")
        self.layout = QVBoxLayout(self); self.top(); self.grid(); self.bottom(); self.country_controls()
        QTimer(self, timeout=self.update).start(1000)

    def top(self):
        bar = QHBoxLayout(); self.status, self.time, self.npu = QLabel("Status: STABLE"), QLabel(""), QLabel("NPU: ...")
        [bar.addWidget(w) or w.setStyleSheet("color:#0ff") for w in [self.status,self.npu,self.time]]; bar.addStretch(); self.layout.addLayout(bar)

    def grid(self):
        self.feed = QTextEdit(); self.feed.setReadOnly(True); self.feed.setStyleSheet("background:#111; color:#0ff")
        self.layout.addWidget(QLabel("Live Feed")); self.layout.addWidget(self.feed)
        ctrl = QHBoxLayout()
        for label, fn in [
            ("NCS2", lambda: self.feed.append(run_ncs2())),
            ("Coral", lambda: self.feed.append(run_coral())),
            ("Hailo", lambda: self.feed.append(run_hailo())),
            ("Capsule", self.inject_capsule),
            ("Firewall", self.firewall),
            ("Replay", lambda: replay(self.feed)),
            ("Threat", lambda: self.inject_persona("ThreatHunter")),
            ("Audit", lambda: self.inject_persona("Auditor")),
            ("Ghost", self.ghost_sync),
            ("Swarm", self.swarm_sync),
            ("Fake", self.fake_telemetry),
        ]:
            b = QPushButton(label); b.setStyleSheet("background:#222; color:#f0f"); b.clicked.connect(fn); ctrl.addWidget(b)
        self.layout.addLayout(ctrl)

    def bottom(self):
        bar = QHBoxLayout(); self.cpu, self.mem, self.ent, self.lat = [QLabel() for _ in range(4)]
        [bar.addWidget(w) or w.setStyleSheet("color:#0ff") for w in [self.cpu,self.mem,self.ent,self.lat]]; self.layout.addLayout(bar)

    def country_controls(self):
        bar = QHBoxLayout(); self.pick = QComboBox(); self.pick.addItems(countries)
        for label, fn in [("Block", self.block), ("Unblock", self.unblock), ("View", self.view)]:
            b = QPushButton(label); b.clicked.connect(fn); bar.addWidget(b)
        bar.addWidget(self.pick); self.layout.addLayout(bar)

    def block(self): c = self.pick.currentText(); rules["blocked"].add(c); self.feed.append(f"🌐 Blocked: {c}")
    def unblock(self): c = self.pick.currentText(); rules["blocked"].discard(c); self.feed.append(f"✅ Unblocked: {c}")
    def view(self): self.feed.append(f"🌐 Blocked: {', '.join(sorted(rules['blocked'])) or 'None'}")

    def update(self):
        self.cpu.setText(f"CPU: {psutil.cpu_percent()}%")
        self.mem.setText(f"Memory: {psutil.virtual_memory().percent}%")
        self.ent.setText(f"Entropy: {psutil.disk_usage('/').percent}%")
        self.lat.setText(f"Latency: {psutil.boot_time()%100}ms")
        self.time.setText(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        self.npu.setText(get_npu_status())
        self.feed.append(f"[{self.time.text()}] Capsule mutation stable.")

    def inject_capsule(self): register("capsule_xyz","backdoor",self.pick.currentText()); self.feed.append("🧬 Capsule: DENIED"); mutate("Phantom")
    def firewall(self): self.feed.append("🚨 Suspicious" if any(k in "inject override exploit" for k in ["inject","override","exploit"]) else "✅ Clean")
    def inject_persona(self, name): self.feed.append(f"🎭 Persona {name} deployed"); register(name,"mac_ip",self.pick.currentText())
    def ghost_sync(self): register("ghost sync","mac_ip",self.pick.currentText()); self.feed.append("🕸️ Ghost sync detected."); mutate("Phantom")
    def swarm_sync(self): self.feed.append("🔗 Swarm sync initiated.")
    def fake_telemetry(self): f = f"telemetry-{np.random.randint(1000,9999)}"; register(f,"fake",self.pick.currentText()); self.feed.append(f"🌀 Fake telemetry: {f}")

# === Launch ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    shell = Shell()
    shell.show()
    sys.exit(app.exec_())



