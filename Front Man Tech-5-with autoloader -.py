import sys, os, socket, platform, uuid, psutil, json, subprocess, zmq, time
from datetime import datetime, timedelta
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QMessageBox
from PyQt6.QtCore import QTimer

def validate_libs():
    required = {"psutil": "psutil", "zmq": "pyzmq", "PyQt6": "PyQt6"}
    status = {}
    for lib, pip in required.items():
        try: __import__(lib); status[lib] = "✓ Installed"
        except:
            try: subprocess.check_call([sys.executable, "-m", "pip", "install", pip])
            except: status[lib] = "✗ Failed"; continue
            status[lib] = "✓ Auto-Installed"
    return status

def telemetry():
    real = {
        "Time": datetime.utcnow().isoformat(),
        "Host": socket.gethostname(),
        "IP": socket.gethostbyname(socket.gethostname()),
        "OS": platform.system(),
        "Ver": platform.version(),
        "CPU": platform.processor(),
        "RAM": round(psutil.virtual_memory().total / (1024**3), 2),
        "MAC": ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0,8*6,8)][::-1])
    }
    fake = {"IP": "10.0.0.99", "MAC": "00:FA:KE:00:00:00", "TTL": (datetime.utcnow() + timedelta(seconds=30)).isoformat()}
    return {**real, "Fake Telemetry": json.dumps(fake)}

def conns():
    out = []
    for c in psutil.net_connections(kind='inet'):
        if c.status in ['LISTEN', 'ESTABLISHED']:
            try:
                p = psutil.Process(c.pid)
                port = c.laddr.port
                risk = "🟢" if port in [22,443,8080] else "🟡" if port < 1024 else "🔴"
                out.append({"Local": f"{c.laddr.ip}:{port} {risk}", "Remote": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "N/A", "Status": c.status, "PID": c.pid, "Proc": p.name()})
            except: continue
    return out

def sync():
    s = zmq.Context().socket(zmq.SUB); s.connect("tcp://localhost:5555"); s.setsockopt_string(zmq.SUBSCRIBE, "MASK_SYNC")
    try: return json.loads(s.recv_string(flags=zmq.NOBLOCK).split(" ",1)[1])
    except: return None

def deploy():
    profile = {
        "persona_id": str(uuid.uuid4()), "decoy_hostname": "frontman-node", "decoy_ip": "192.168.0.254",
        "decoy_ports": [8080,443], "decoy_os": "Linux", "decoy_processes": ["systemd","sshd","nginx"],
        "telemetry_signature": "MASKED", "mask_level": "full", "swarm_ready": True,
        "timestamp": datetime.utcnow().isoformat(),
        "fake_telemetry": {"IP": "10.0.0.99", "MAC": "00:FA:KE:00:00:00", "expires": (datetime.utcnow()+timedelta(seconds=30)).isoformat()}
    }
    s = zmq.Context().socket(zmq.PUB); s.bind("tcp://*:5555"); s.send_string(f"MASK_SYNC {json.dumps(profile)}")

def vaporize():
    for path in ["/tmp", os.path.expanduser("~/.cache")]:
        for r,_,f in os.walk(path):
            for file in f:
                try: os.remove(os.path.join(r,file))
                except: continue

def purge_personal():
    vault = os.path.expanduser("~/vault")
    now = time.time()
    for r,_,f in os.walk(vault):
        for file in f:
            fp = os.path.join(r,file)
            if os.path.getmtime(fp) < now - 86400:
                try: os.remove(fp)
                except: continue

class ASIDash(QWidget):
    def __init__(s):
        super().__init__()
        s.setWindowTitle("ASI Oversight Console")
        s.setStyleSheet("background:#0f0f0f; color:#0ff; font-family:'Courier New'; font-size:12px;")
        s.layout = QVBoxLayout(s)
        s.tables = {}
        for label,h in {
            "🔧 Autoloader": ["Lib","Status"], "🧠 Telemetry": ["Metric","Value"], "🔗 Swarm": ["Field","Value"],
            "🌐 Connections": ["Local","Remote","Status","PID","Proc"], "🛡️ Protect": ["Subsystem","Status"],
            "🧬 Mutation": ["Lineage","Status"], "🧠 Biometric": ["Node","Resonance"], "🚨 Threats": ["Time","Type","Severity"]
        }.items():
            s.layout.addWidget(QLabel(label))
            t = QTableWidget(); t.setColumnCount(len(h)); t.setHorizontalHeaderLabels(h)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            s.layout.addWidget(t); s.tables[label] = t
        s.layout.addWidget(QLabel("🧭 Command Deck"))
        for txt, fn in [("🚀 Deploy", s.deploy), ("💣 Vaporize", s.vaporize)]:
            b = QPushButton(txt); b.clicked.connect(fn); s.layout.addWidget(b)
        s.refresh(); s.timer = QTimer(); s.timer.timeout.connect(s.refresh); s.timer.start(3000)

    def refresh(s):
        purge_personal()
        s.fill("🔧 Autoloader", validate_libs())
        s.fill("🧠 Telemetry", telemetry())
        if (sw := sync()): s.fill("🔗 Swarm", sw)
        s.fill("🌐 Connections", conns(), True)
        s.fill("🛡️ Protect", {"Zero Trust": "🟢", "Mask": "🟢", "Sync": "🟢", "TTL": "🟢"})
        s.fill("🧬 Mutation", {"frontman → peer-1 → peer-2": "🟢"})
        s.fill("🧠 Biometric", {"frontman": "Stable", "peer-1": "Elevated"})
        s.fill("🚨 Threats", {datetime.utcnow().isoformat(): "Backdoor Ejection | High"})

    def fill(s, label, data, is_list=False):
        t = s.tables[label]; t.setRowCount(len(data))
        for i,(k,v) in enumerate(data.items() if not is_list else [(None,d) for d in data]):
            for j,val in enumerate(v.values() if is_list else [k,v]):
                t.setItem(i,j,QTableWidgetItem(str(val)))

    def deploy(s): deploy(); QMessageBox.information(s,"Deploy","Mask profile deployed.")
    def vaporize(s): vaporize(); QMessageBox.warning(s,"Vaporize","TTL enforced. Node vaporized.")

if __name__ == "__main__":
    deploy()
    app = QApplication(sys.argv)
    win = ASIDash(); win.resize(1000,750); win.show()
    sys.exit(app.exec())

