# === AUTO-ELEVATION (CROSS-PLATFORM) ===
import os, sys, platform, ctypes

def is_admin():
    try:
        return os.getuid() == 0
    except AttributeError:
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

def ensure_admin():
    if is_admin():
        return
    system = platform.system()
    if system == "Windows":
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
    elif system in ("Linux", "Darwin"):
        script = os.path.abspath(sys.argv[0])
        os.execvp("sudo", ["sudo", sys.executable, script] + sys.argv[1:])
    else:
        print(f"Unsupported OS for elevation: {system}")
        sys.exit()

ensure_admin()

# === STANDARD IMPORTS ===
import subprocess
import importlib
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import json

# -----------------------------
# SETTINGS / PERSISTENCE
# -----------------------------
SETTINGS_FILE = "settings.json"

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return None
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[SETTINGS] Failed to save settings: {e}")

# -----------------------------
# AUTOLOADER FOR DEPENDENCIES
# -----------------------------
REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "skopt": "scikit-optimize",
    "psutil": "psutil",
    "pynvml": "pynvml",
}

def ensure_dependencies():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[AUTOLOADER] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    else:
        print("[AUTOLOADER] All required packages already installed.")

ensure_dependencies()

# -----------------------------
# IMPORTS AFTER DEP CHECK
# -----------------------------
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_large

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skimage.util import img_as_ubyte

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
DEVICE = torch.device("cpu")
NUM_CLASSES = 10

# -----------------------------
# DATA GENERATION / DATASET
# -----------------------------
def generate_dummy_data(num_samples=1000, height=32, width=32):
    X = np.random.rand(num_samples, height, width, 3).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, size=(num_samples,))
    return X, y

class NumpyImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        img_uint8 = img_as_ubyte(img)
        if self.transform is not None:
            img_tensor = self.transform(img_uint8)
        else:
            img_tensor = T.ToTensor()(img_uint8)
        return img_tensor, label

# -----------------------------
# AUGMENTATION PIPELINES
# -----------------------------
def get_transforms(augment_level):
    if augment_level == "Low":
        train_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
        ])
    elif augment_level == "Medium":
        train_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
        ])
    else:
        train_tf = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop((64, 64), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(25),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.RandomPerspective(distortion_scale=0.4, p=0.5),
            T.ToTensor(),
        ])
    val_tf = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    return train_tf, val_tf

# -----------------------------
# MODEL CREATION
# -----------------------------
def create_mobilenet_v3_large(dropout=0.2):
    model = mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    for layer in model.classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = float(dropout)
    model.to(DEVICE)
    return model

# -----------------------------
# TRAINING / EVAL
# -----------------------------
def train_one_model(
    learning_rate,
    dropout,
    epochs,
    batch_size,
    augment_level,
    log=None,
    max_train_samples=800,
    max_val_samples=200,
):
    if log is None:
        log = print
    log(f"[TRAIN] Preparing data (augment={augment_level}, epochs={epochs}, bs={batch_size})")
    X, y = generate_dummy_data(num_samples=max_train_samples + max_val_samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=max_val_samples, random_state=42
    )
    train_tf, val_tf = get_transforms(augment_level)
    train_ds = NumpyImageDataset(X_train, y_train, transform=train_tf)
    val_ds = NumpyImageDataset(X_val, y_val, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = create_mobilenet_v3_large(dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        val_acc = evaluate_model(model, val_loader)
        log(
            f"[TRAIN] Epoch {epoch + 1}/{epochs} "
            f"Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  ValAcc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    log(f"[TRAIN] Finished. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# -----------------------------
# BAYESIAN OPTIMIZATION
# -----------------------------
dimensions = [
    Real(low=1e-5, high=1e-2, name="learning_rate"),
    Real(low=0.1, high=0.5, name="dropout"),
]

@use_named_args(dimensions=dimensions)
def objective_function(learning_rate, dropout):
    def silent_log(*args, **kwargs):
        pass
    val_acc = train_one_model(
        learning_rate=learning_rate,
        dropout=dropout,
        epochs=3,
        batch_size=32,
        augment_level="Medium",
        log=silent_log,
        max_train_samples=600,
        max_val_samples=200,
    )
    return -val_acc

def run_hyperparameter_optimization(gui_log):
    gui_log("[OPT] Starting Bayesian optimization (this will take a while)...")
    result = gp_minimize(
        objective_function,
        dimensions,
        n_calls=10,
        random_state=42,
    )
    best_lr = result.x[0]
    best_dropout = result.x[1]
    best_val_acc = -result.fun
    gui_log(f"[OPT] Best LR: {best_lr:.6f}")
    gui_log(f"[OPT] Best Dropout: {best_dropout:.3f}")
    gui_log(f"[OPT] Best Val Acc: {best_val_acc:.4f}")
    return best_lr, best_dropout, best_val_acc

def run_augmentation_only(augment_level, gui_log):
    gui_log(f"[AUG] Running augmentation-only test at level: {augment_level}")
    X, y = generate_dummy_data(num_samples=256)
    train_tf, _ = get_transforms(augment_level)
    ds = NumpyImageDataset(X, y, transform=train_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    count = 0
    for imgs, labels in loader:
        count += imgs.size(0)
    gui_log(f"[AUG] Augmentation passed through {count} samples.")

# -----------------------------
# PRESETS
# -----------------------------
PRESETS = {
    "conservative": {
        "learning_rate": 0.0005,
        "dropout1": 0.4,
        "dropout2": 0.4,
        "epochs": 5,
        "batch": 32,
        "augment": "Low",
    },
    "balanced": {
        "learning_rate": 0.001,
        "dropout1": 0.3,
        "dropout2": 0.3,
        "epochs": 10,
        "batch": 32,
        "augment": "Medium",
    },
    "aggressive": {
        "learning_rate": 0.005,
        "dropout1": 0.2,
        "dropout2": 0.2,
        "epochs": 20,
        "batch": 64,
        "augment": "High",
    },
}

# -----------------------------
# GUI CLASS
# -----------------------------
class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Training Control Panel (PyTorch / MobileNetV3-Large / CPU)")
        self.root.geometry("1000x800")
        self.cpu_temp_var = tk.StringVar(value="CPU Temp: N/A")
        self.gpu_temp_var = tk.StringVar(value="GPU Temp: N/A")
        self.gpu_usage_var = tk.StringVar(value="GPU: N/A")
        self.vram_var = tk.StringVar(value="VRAM: N/A")
        self.ram_usage_var = tk.StringVar(value="RAM: N/A")
        self.mode = tk.StringVar(value="ai")
        self.ai_setting = tk.StringVar(value="balanced")
        self.manual_setting = tk.StringVar(value="balanced")
        self.learning_rate = tk.StringVar()
        self.dropout1 = tk.StringVar()
        self.dropout2 = tk.StringVar()
        self.epochs = tk.StringVar()
        self.batch_size = tk.StringVar()
        self.augment_intensity = tk.StringVar()
        self.build_gui()
        self.apply_preset("balanced")
        self.update_mode()
        self.setup_autosave_traces()
        self.load_saved_settings()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_stats()

    def build_gui(self):
        stats_frame = ttk.Frame(self.root)
        stats_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(stats_frame, textvariable=self.cpu_temp_var, width=18).pack(side="left", padx=5)
        ttk.Label(stats_frame, textvariable=self.gpu_temp_var, width=18).pack(side="left", padx=5)
        ttk.Label(stats_frame, textvariable=self.gpu_usage_var, width=18).pack(side="left", padx=5)
        ttk.Label(stats_frame, textvariable=self.vram_var, width=22).pack(side="left", padx=5)
        ttk.Label(stats_frame, textvariable=self.ram_usage_var, width=18).pack(side="left", padx=5)

        mode_frame = ttk.LabelFrame(self.root, text="Mode Selection")
        mode_frame.pack(fill="x", padx=10, pady=10)
        ttk.Radiobutton(
            mode_frame, text="AI Mode", variable=self.mode, value="ai",
            command=self.update_mode
        ).pack(side="left", padx=10)
        ttk.Radiobutton(
            mode_frame, text="Manual Mode", variable=self.mode, value="manual",
            command=self.update_mode
        ).pack(side="left", padx=10)

        self.ai_frame = ttk.LabelFrame(self.root, text="AI Mode Settings")
        self.ai_frame.pack(fill="x", padx=10, pady=10)
        ttk.Radiobutton(
            self.ai_frame, text="Conservative", variable=self.ai_setting,
            value="conservative", command=self.ai_preset_selected
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.ai_frame, text="Balanced", variable=self.ai_setting,
            value="balanced", command=self.ai_preset_selected
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.ai_frame, text="Aggressive", variable=self.ai_setting,
            value="aggressive", command=self.ai_preset_selected
        ).pack(anchor="w")

        self.manual_frame = ttk.LabelFrame(self.root, text="Manual Settings")
        self.manual_frame.pack(fill="x", padx=10, pady=10)
        ttk.Radiobutton(
            self.manual_frame, text="Conservative", variable=self.manual_setting,
            value="conservative", command=self.manual_preset_selected
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.manual_frame, text="Balanced", variable=self.manual_setting,
            value="balanced", command=self.manual_preset_selected
        ).pack(anchor="w")
        ttk.Radiobutton(
            self.manual_frame, text="Aggressive", variable=self.manual_setting,
            value="aggressive", command=self.manual_preset_selected
        ).pack(anchor="w")

        self.build_param_fields()

        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(button_frame, text="Run Optimization",
                   command=self.run_optimization).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Train Model",
                   command=self.run_training).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Augment Only",
                   command=self.run_augmentation).pack(side="left", padx=5)

        console_frame = ttk.LabelFrame(self.root, text="Console Output")
        console_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.console = ScrolledText(console_frame, height=15)
        self.console.pack(fill="both", expand=True)

    def build_param_fields(self):
        param_frame = ttk.LabelFrame(self.root, text="Parameters")
        param_frame.pack(fill="x", padx=10, pady=10)

        lr_row = ttk.Frame(param_frame); lr_row.pack(fill="x", pady=4)
        ttk.Label(lr_row, text="Learning Rate", width=20).pack(side="left")
        lr_slider = ttk.Scale(
            lr_row, from_=0.000001, to=0.01, orient="horizontal",
            command=lambda v: self.learning_rate.set(f"{float(v):.6f}")
        )
        lr_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Entry(lr_row, textvariable=self.learning_rate, width=10).pack(side="left")

        d1_row = ttk.Frame(param_frame); d1_row.pack(fill="x", pady=4)
        ttk.Label(d1_row, text="Dropout 1", width=20).pack(side="left")
        d1_slider = ttk.Scale(
            d1_row, from_=0.0, to=0.7, orient="horizontal",
            command=lambda v: self.dropout1.set(f"{float(v):.2f}")
        )
        d1_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Entry(d1_row, textvariable=self.dropout1, width=10).pack(side="left")

        d2_row = ttk.Frame(param_frame); d2_row.pack(fill="x", pady=4)
        ttk.Label(d2_row, text="Dropout 2", width=20).pack(side="left")
        d2_slider = ttk.Scale(
            d2_row, from_=0.0, to=0.7, orient="horizontal",
            command=lambda v: self.dropout2.set(f"{float(v):.2f}")
        )
        d2_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Entry(d2_row, textvariable=self.dropout2, width=10).pack(side="left")

        ep_row = ttk.Frame(param_frame); ep_row.pack(fill="x", pady=4)
        ttk.Label(ep_row, text="Epochs", width=20).pack(side="left")
        ep_slider = ttk.Scale(
            ep_row, from_=1, to=100, orient="horizontal",
            command=lambda v: self.epochs.set(str(int(float(v))))
        )
        ep_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Entry(ep_row, textvariable=self.epochs, width=10).pack(side="left")

        bs_row = ttk.Frame(param_frame); bs_row.pack(fill="x", pady=4)
        ttk.Label(bs_row, text="Batch Size", width=20).pack(side="left")
        bs_slider = ttk.Scale(
            bs_row, from_=8, to=256, orient="horizontal",
            command=lambda v: self.batch_size.set(str(int(float(v))))
        )
        bs_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Entry(bs_row, textvariable=self.batch_size, width=10).pack(side="left")

        aug_row = ttk.Frame(param_frame); aug_row.pack(fill="x", pady=4)
        ttk.Label(aug_row, text="Augment Intensity", width=20).pack(side="left")
        def update_aug(v):
            v = int(float(v))
            levels = ["Low", "Medium", "High"]
            self.augment_intensity.set(levels[v])
        aug_slider = ttk.Scale(
            aug_row, from_=0, to=2, orient="horizontal",
            command=update_aug
        )
        aug_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Entry(aug_row, textvariable=self.augment_intensity, width=10).pack(side="left")

        self.sliders = {
            "lr": lr_slider,
            "d1": d1_slider,
            "d2": d2_slider,
            "epochs": ep_slider,
            "batch": bs_slider,
            "augment": aug_slider,
        }

    # ---------- SETTINGS PERSISTENCE ----------
    def setup_autosave_traces(self):
        def autosave(*args):
            save_settings(self.collect_settings())
        self.learning_rate.trace_add("write", autosave)
        self.dropout1.trace_add("write", autosave)
        self.dropout2.trace_add("write", autosave)
        self.epochs.trace_add("write", autosave)
        self.batch_size.trace_add("write", autosave)
        self.augment_intensity.trace_add("write", autosave)
        self.mode.trace_add("write", autosave)
        self.ai_setting.trace_add("write", autosave)
        self.manual_setting.trace_add("write", autosave)

    def collect_settings(self):
        return {
            "mode": self.mode.get(),
            "preset": self.ai_setting.get() if self.mode.get() == "ai" else self.manual_setting.get(),
            "learning_rate": self.learning_rate.get(),
            "dropout1": self.dropout1.get(),
            "dropout2": self.dropout2.get(),
            "epochs": self.epochs.get(),
            "batch_size": self.batch_size.get(),
            "augment_intensity": self.augment_intensity.get()
        }

    def load_saved_settings(self):
        data = load_settings()
        if not data:
            return
        try:
            self.mode.set(data.get("mode", "ai"))
            self.update_mode()
            preset = data.get("preset", "balanced")
            self.apply_preset(preset)
            self.ai_setting.set(preset)
            self.manual_setting.set(preset)

            if data.get("learning_rate") is not None:
                self.learning_rate.set(data.get("learning_rate"))
                self.sliders["lr"].set(float(data.get("learning_rate")))
            if data.get("dropout1") is not None:
                self.dropout1.set(data.get("dropout1"))
                self.sliders["d1"].set(float(data.get("dropout1")))
            if data.get("dropout2") is not None:
                self.dropout2.set(data.get("dropout2"))
                self.sliders["d2"].set(float(data.get("dropout2")))
            if data.get("epochs") is not None:
                self.epochs.set(data.get("epochs"))
                self.sliders["epochs"].set(int(data.get("epochs")))
            if data.get("batch_size") is not None:
                self.batch_size.set(data.get("batch_size"))
                self.sliders["batch"].set(int(data.get("batch_size")))

            aug = data.get("augment_intensity", "Medium")
            self.augment_intensity.set(aug)
            aug_map = {"Low": 0, "Medium": 1, "High": 2}
            self.sliders["augment"].set(aug_map.get(aug, 1))
        except Exception as e:
            print(f"[SETTINGS] Failed to load settings: {e}")

    def on_close(self):
        save_settings(self.collect_settings())
        self.root.destroy()

    # ---------- MODES / PRESETS ----------
    def update_mode(self):
        if self.mode.get() == "ai":
            self.ai_frame.state(["!disabled"])
            self.manual_frame.state(["disabled"])
        else:
            self.ai_frame.state(["disabled"])
            self.manual_frame.state(["!disabled"])

    def ai_preset_selected(self):
        self.apply_preset(self.ai_setting.get())

    def manual_preset_selected(self):
        self.apply_preset(self.manual_setting.get())

    def apply_preset(self, preset_name):
        preset = PRESETS[preset_name]
        self.learning_rate.set(preset["learning_rate"])
        self.dropout1.set(preset["dropout1"])
        self.dropout2.set(preset["dropout2"])
        self.epochs.set(preset["epochs"])
        self.batch_size.set(preset["batch"])
        self.augment_intensity.set(preset["augment"])
        self.sliders["lr"].set(preset["learning_rate"])
        self.sliders["d1"].set(preset["dropout1"])
        self.sliders["d2"].set(preset["dropout2"])
        self.sliders["epochs"].set(preset["epochs"])
        self.sliders["batch"].set(preset["batch"])
        aug_map = {"Low": 0, "Medium": 1, "High": 2}
        self.sliders["augment"].set(aug_map[preset["augment"]])
        save_settings(self.collect_settings())

    # ---------- LOGGING ----------
    def log(self, msg):
        self.console.insert("end", msg + "\n")
        self.console.see("end")

    # ---------- THREAD HELP ----------
    def run_threaded(self, target):
        threading.Thread(target=target, daemon=True).start()

    # ---------- PARAM GRAB ----------
    def get_current_params(self):
        try:
            lr = float(self.learning_rate.get())
        except ValueError:
            lr = 0.001
        try:
            d1 = float(self.dropout1.get())
        except ValueError:
            d1 = 0.3
        try:
            ep = int(self.epochs.get())
        except ValueError:
            ep = 10
        try:
            bs = int(self.batch_size.get())
        except ValueError:
            bs = 32
        aug = self.augment_intensity.get()
        if aug not in ["Low", "Medium", "High"]:
            aug = "Medium"
        return lr, d1, ep, bs, aug

    # ---------- BUTTON ACTIONS ----------
    def run_optimization(self):
        self.log("[GUI] Starting optimization...")
        self.run_threaded(self._run_optimization)

    def run_training(self):
        self.log("[GUI] Starting training with current parameters...")
        self.run_threaded(self._run_training)

    def run_augmentation(self):
        self.log("[GUI] Running augmentation-only test...")
        self.run_threaded(self._run_augmentation)

    def _run_optimization(self):
        best_lr, best_dropout, best_acc = run_hyperparameter_optimization(self.log)
        self.learning_rate.set(best_lr)
        self.dropout1.set(best_dropout)
        self.sliders["lr"].set(best_lr)
        self.sliders["d1"].set(best_dropout)
        save_settings(self.collect_settings())
        self.log(f"[GUI] Optimization complete. Best Val Acc: {best_acc:.4f}")

    def _run_training(self):
        lr, d1, ep, bs, aug = self.get_current_params()
        self.log(f"[GUI] Params -> LR={lr}, Dropout={d1}, Epochs={ep}, Batch={bs}, Aug={aug}")
        start = time.time()
        val_acc = train_one_model(
            learning_rate=lr,
            dropout=d1,
            epochs=ep,
            batch_size=bs,
            augment_level=aug,
            log=self.log,
        )
        self.log(f"[GUI] Training finished. Final Val Acc: {val_acc:.4f}")
        self.log(f"[GUI] Elapsed: {time.time() - start:.1f} seconds")

    def _run_augmentation(self):
        _, _, _, _, aug = self.get_current_params()
        run_augmentation_only(aug, self.log)
        self.log("[GUI] Augmentation-only test finished.")

    # ---------- STATS UPDATE ----------
    def update_stats(self):
        vm = psutil.virtual_memory()
        self.ram_usage_var.set(f"RAM: {vm.percent:.0f}%")
        cpu_temp_str = "N/A"
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        cpu_temp_str = f"{entries[0].current:.0f}°C"
                        break
        except Exception:
            pass
        self.cpu_temp_var.set(f"CPU Temp: {cpu_temp_str}")

        if HAS_NVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_temp_str = f"{temp}°C"
                gpu_usage_str = f"{util.gpu}%"
                vram_str = f"{mem.used / (1024**2):.0f}/{mem.total / (1024**2):.0f} MB"
            except Exception:
                gpu_temp_str = "N/A"
                gpu_usage_str = "N/A"
                vram_str = "N/A"
        else:
            gpu_temp_str = "N/A"
            gpu_usage_str = "N/A"
            vram_str = "N/A"

        self.gpu_temp_var.set(f"GPU Temp: {gpu_temp_str}")
        self.gpu_usage_var.set(f"GPU: {gpu_usage_str}")
        self.vram_var.set(f"VRAM: {vram_str}")
        self.root.after(1000, self.update_stats)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = TrainingGUI(root)
    root.mainloop()

