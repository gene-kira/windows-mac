import importlib
import subprocess
import sys
import time
import threading
import platform
import json
import os

# ============================================================
# AUTO-LOADER FOR REQUIRED PACKAGES (WITH STATUS MESSAGES)
# ============================================================

def auto_install(package, import_name=None):
    """
    Auto-installs a missing package and imports it.
    Prints status messages so the user knows what's happening.
    """
    try:
        return importlib.import_module(import_name or package)
    except ImportError:
        print(f"[AutoLoader] Missing package '{package}'. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[AutoLoader] Successfully installed '{package}'.")
        return importlib.import_module(import_name or package)


# Core libs
auto_install("numpy")
auto_install("psutil")

# GPU monitor (optional)
try:
    auto_install("gputil", "GPUtil")
    import GPUtil
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    GPUtil = None
    print("[AutoLoader] GPUtil not available. GPU monitoring limited.")

# ML stack
auto_install("scikit-image", "skimage")
auto_install("scikit-optimize", "skopt")
auto_install("tensorflow")

# ============================================================
# STANDARD IMPORTS AFTER AUTO-INSTALL
# ============================================================

import numpy as np
import psutil

from sklearn.model_selection import train_test_split

from skimage.transform import rotate, rescale
from skimage.util import random_noise, img_as_ubyte

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import tkinter as tk
from tkinter import ttk

# OS detection
IS_WINDOWS = platform.system() == "Windows"

# ============================================================
# SETTINGS PERSISTENCE (settings.json in script folder)
# ============================================================

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "cpu_throttle": 0.20,
    "gpu_throttle": 0.20,
    "auto_mode": False,
    "governor_mode": "balanced",

    "cpu_power_strategy": "saver",
    "cpu_custom_idle": 0.20,
    "cpu_custom_burst": 0.80,
    "cpu_custom_duration": 1.5,
    "cpu_custom_cooldown": 0.05,
    "cpu_custom_trigger": 60.0,

    "gpu_power_strategy": "saver",
    "gpu_custom_idle": 0.15,
    "gpu_custom_ramp_up": 0.02,
    "gpu_custom_ramp_down": 0.03,
    "gpu_custom_trigger": 60.0,
}

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        # merge with defaults to avoid missing keys
        settings = DEFAULT_SETTINGS.copy()
        settings.update(data)
        return settings
    except Exception as e:
        print(f"[Settings] Failed to load settings.json: {e}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[Settings] Failed to save settings.json: {e}")

# ============================================================
# GLOBAL THROTTLE + GOVERNOR + CPU/GPU POWER STRATEGY STATE
# ============================================================

CPU_THROTTLE = DEFAULT_SETTINGS["cpu_throttle"]
GPU_THROTTLE = DEFAULT_SETTINGS["gpu_throttle"]

AUTO_MODE = DEFAULT_SETTINGS["auto_mode"]

GOVERNOR_MODE = DEFAULT_SETTINGS["governor_mode"]

GOV_CONFIG = {
    "conservative": {
        "gpu_target": 60.0,
        "step_up": 0.005,
        "step_down": 0.01,
    },
    "balanced": {
        "gpu_target": 80.0,
        "step_up": 0.01,
        "step_down": 0.02,
    },
    "aggressive": {
        "gpu_target": 95.0,
        "step_up": 0.02,
        "step_down": 0.04,
    },
}

CPU_POWER_STRATEGY = DEFAULT_SETTINGS["cpu_power_strategy"]
CPU_POWER_CONFIG = {
    "saver": {
        "idle_baseline": 0.15,
        "burst_throttle": 0.80,
        "burst_duration": 1.0,
        "cooldown_step": 0.05,
        "burst_trigger": 60.0,
    },
    "balanced": {
        "idle_baseline": 0.30,
        "burst_throttle": 0.90,
        "burst_duration": 1.5,
        "cooldown_step": 0.04,
        "burst_trigger": 55.0,
    },
    "aggressive": {
        "idle_baseline": 0.40,
        "burst_throttle": 1.00,
        "burst_duration": 2.0,
        "cooldown_step": 0.03,
        "burst_trigger": 50.0,
    },
}

GPU_POWER_STRATEGY = DEFAULT_SETTINGS["gpu_power_strategy"]
GPU_POWER_CONFIG = {
    "saver": {
        "idle_baseline": 0.10,
        "ramp_up": 0.01,
        "ramp_down": 0.03,
        "load_trigger": 70.0,
        "thermal_sensitivity": "high",
    },
    "balanced": {
        "idle_baseline": 0.20,
        "ramp_up": 0.02,
        "ramp_down": 0.02,
        "load_trigger": 60.0,
        "thermal_sensitivity": "medium",
    },
    "aggressive": {
        "idle_baseline": 0.30,
        "ramp_up": 0.04,
        "ramp_down": 0.01,
        "load_trigger": 50.0,
        "thermal_sensitivity": "low",
    },
}


def set_cpu_throttle_from_slider(val):
    global CPU_THROTTLE
    CPU_THROTTLE = max(0.01, min(1.0, float(val) / 100.0))


def set_gpu_throttle_from_slider(val):
    global GPU_THROTTLE
    GPU_THROTTLE = max(0.01, min(1.0, float(val) / 100.0))


def set_auto_mode(on: bool):
    global AUTO_MODE
    AUTO_MODE = bool(on)


def set_governor_mode(mode: str):
    global GOVERNOR_MODE
    if mode in GOV_CONFIG:
        GOVERNOR_MODE = mode


def set_cpu_power_strategy(mode: str):
    global CPU_POWER_STRATEGY
    if mode in ("saver", "balanced", "aggressive", "custom"):
        CPU_POWER_STRATEGY = mode


def set_gpu_power_strategy(mode: str):
    global GPU_POWER_STRATEGY
    if mode in ("saver", "balanced", "aggressive", "custom"):
        GPU_POWER_STRATEGY = mode


# ============================================================
# THROTTLE HELPERS
# ============================================================

def cpu_throttle_sleep(base_active_time=0.01):
    global CPU_THROTTLE
    throttle = max(0.01, min(1.0, CPU_THROTTLE))
    active = base_active_time
    sleep_time = active * (1.0 / throttle - 1.0)
    if sleep_time > 0:
        time.sleep(sleep_time)


def gpu_throttle_sleep(last_batch_time):
    global GPU_THROTTLE
    throttle = max(0.01, min(1.0, GPU_THROTTLE))
    active = max(1e-4, last_batch_time)
    sleep_time = active * (1.0 / throttle - 1.0)
    if sleep_time > 0:
        time.sleep(sleep_time)


# ============================================================
# DATA + AUGMENTATION
# ============================================================

def load_data():
    num_samples = 1000
    img_height, img_width = 32, 32
    X = np.random.rand(num_samples, img_height, img_width, 3) * 255.0
    y = np.random.randint(0, 10, size=(num_samples,))
    return X.astype(np.float32), y.astype(np.int64)


def augment_image(image):
    angle = np.random.uniform(-30, 30)
    scale = np.random.uniform(0.8, 1.2)

    augmented = rotate(image, angle, mode='reflect', preserve_range=True)
    cpu_throttle_sleep(0.003)

    augmented = rescale(
        augmented,
        scale,
        anti_aliasing=True,
        channel_axis=-1,
        preserve_range=True
    )
    cpu_throttle_sleep(0.003)

    target_h, target_w = image.shape[:2]
    h, w = augmented.shape[:2]

    if h > target_h:
        start_h = (h - target_h) // 2
        augmented = augmented[start_h:start_h + target_h, :, :]
    elif h < target_h:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        augmented = np.pad(augmented, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='reflect')

    cpu_throttle_sleep(0.002)

    h, w = augmented.shape[:2]
    if w > target_w:
        start_w = (w - target_w) // 2
        augmented = augmented[:, start_w:start_w + target_w, :]
    elif w < target_w:
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        augmented = np.pad(augmented, ((0, 0), (pad_left, pad_right), (0, 0)), mode='reflect')

    cpu_throttle_sleep(0.002)

    augmented = random_noise(augmented)
    augmented = img_as_ubyte(augmented)

    cpu_throttle_sleep(0.002)
    return augmented.astype(np.float32)


def augment_dataset(X):
    out = []
    for img in X:
        out.append(augment_image(img))
    return np.array(out, dtype=np.float32)


# ============================================================
# MODEL
# ============================================================

def build_model(input_shape, num_classes, dropout_rate1, dropout_rate2, learning_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate1),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# GPU WARMUP
# ============================================================

def gpu_warmup():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU detected by TensorFlow.")
            return
        print(f"GPU detected: {gpus}")
    except Exception as e:
        print(f"Error checking GPUs: {e}")
        return

    a = tf.random.normal((2048, 2048))
    b = tf.random.normal((2048, 2048))
    c = tf.matmul(a, b)
    _ = tf.reduce_sum(c).numpy()
    print("GPU warmup completed.")


# ============================================================
# BAYESIAN OPTIMIZATION OBJECTIVE
# ============================================================

space = [
    Real(low=1e-4, high=1e-2, name='learning_rate'),
    Real(low=0.1, high=0.5, name='dropout_rate1'),
    Real(low=0.1, high=0.5, name='dropout_rate2')
]


@use_named_args(dimensions=space)
def objective_function(learning_rate, dropout_rate1, dropout_rate2):
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_aug = augment_dataset(X_train)
    X_val_aug = augment_dataset(X_val)

    input_shape = X_train_aug.shape[1:]
    num_classes = len(np.unique(y))

    model = build_model(
        input_shape,
        num_classes,
        dropout_rate1,
        dropout_rate2,
        learning_rate
    )

    history = model.fit(
        X_train_aug, y_train,
        epochs=3,
        batch_size=32,
        validation_data=(X_val_aug, y_val),
        verbose=0
    )

    best_val_acc = max(history.history['val_accuracy'])
    print(
        f"Trial lr={learning_rate:.6f}, "
        f"dr1={dropout_rate1:.3f}, dr2={dropout_rate2:.3f}, "
        f"val_acc={best_val_acc:.4f}"
    )
    return -best_val_acc


# ============================================================
# TRAINING WITH GPU THROTTLE
# ============================================================

def train_with_throttle(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    import math
    num_samples = X_train.shape[0]
    num_batches = math.ceil(num_samples / batch_size)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b+1) * batch_size, num_samples)
            xb = X_train[start_idx:end_idx]
            yb = y_train[start_idx:end_idx]

            start_time = time.time()
            model.train_on_batch(xb, yb)
            end_time = time.time()

            batch_time = end_time - start_time
            gpu_throttle_sleep(batch_time)

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    gpu_warmup()

    print("Starting hyperparameter optimization...")
    result = gp_minimize(
        objective_function,
        space,
        n_calls=5,
        n_initial_points=5,
        random_state=42
    )

    best_lr, best_dr1, best_dr2 = result.x
    best_val_acc = -result.fun
    print("Optimization finished.")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Best Dropout 1: {best_dr1}")
    print(f"Best Dropout 2: {best_dr2}")
    print(f"Best Validation Accuracy (BO): {best_val_acc}")

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    start_aug = time.time()
    X_train_aug = augment_dataset(X_train)
    X_val_aug = augment_dataset(X_val)
    end_aug = time.time()
    print(f"Data augmentation took {end_aug - start_aug:.2f} seconds")

    input_shape = X_train_aug.shape[1:]
    num_classes = len(np.unique(y))
    model = build_model(input_shape, num_classes, best_dr1, best_dr2, best_lr)

    start_train = time.time()
    train_with_throttle(
        model,
        X_train_aug,
        y_train,
        X_val_aug,
        y_val,
        epochs=10,
        batch_size=32
    )
    end_train = time.time()
    print(f"Final model training took {end_train - start_train:.2f} seconds")


# ============================================================
# GUI
# ============================================================

class MonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("System Monitor (CPU/GPU Power Strategies + AutoLoader + Persist)")

        self.root.tk.call('tk', 'scaling', 1.0)
        self.root.geometry("900x420")

        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)

        self.cpu_burst_active = False
        self.cpu_burst_end_time = 0.0

        # Load settings early
        self.settings = load_settings()

        # TOP: metrics
        numbers_frame = ttk.Frame(main)
        numbers_frame.pack(fill="x", pady=5)

        self.cpu_label = ttk.Label(numbers_frame, text="CPU: 0%", font=("Arial", 12))
        self.cpu_label.pack(side="left", padx=8)

        self.cpu_temp_label = ttk.Label(
            numbers_frame,
            text="CPU Temp: --°C",
            font=("Arial", 12),
            foreground="black"
        )
        self.cpu_temp_label.pack(side="left", padx=8)

        if GPU_AVAILABLE and GPUtil is not None:
            self.gpu_label = ttk.Label(numbers_frame, text="GPU: 0%", font=("Arial", 12))
            self.gpu_label.pack(side="left", padx=8)

            self.gpu_temp_label = ttk.Label(numbers_frame, text="GPU Temp: --°C", font=("Arial", 12))
            self.gpu_temp_label.pack(side="left", padx=8)

            self.vram_label = ttk.Label(numbers_frame, text="VRAM: --%", font=("Arial", 12))
            self.vram_label.pack(side="left", padx=8)
        else:
            self.gpu_label = ttk.Label(numbers_frame, text="GPU: N/A", font=("Arial", 12))
            self.gpu_label.pack(side="left", padx=8)

            self.gpu_temp_label = ttk.Label(numbers_frame, text="GPU Temp: N/A", font=("Arial", 12))
            self.gpu_temp_label.pack(side="left", padx=8)

            self.vram_label = ttk.Label(numbers_frame, text="VRAM: N/A", font=("Arial", 12))
            self.vram_label.pack(side="left", padx=8)

        self.ram_label = ttk.Label(numbers_frame, text="RAM: 0%", font=("Arial", 12))
        self.ram_label.pack(side="left", padx=8)

        # SLIDERS + AUTO + GOVERNOR
        slider_frame = ttk.Frame(main)
        slider_frame.pack(fill="x", pady=5)

        ttk.Label(slider_frame, text="CPU Throttle (%)", font=("Arial", 11)).grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.cpu_slider = ttk.Scale(
            slider_frame,
            from_=1,
            to=100,
            orient="horizontal",
            command=self.on_cpu_slider_changed
        )
        self.cpu_slider.grid(row=0, column=1, sticky="we", padx=5)

        ttk.Label(slider_frame, text="GPU Throttle (%)", font=("Arial", 11)).grid(
            row=0, column=2, sticky="w", padx=5
        )
        self.gpu_slider = ttk.Scale(
            slider_frame,
            from_=1,
            to=100,
            orient="horizontal",
            command=self.on_gpu_slider_changed
        )
        self.gpu_slider.grid(row=0, column=3, sticky="we", padx=5)

        slider_frame.columnconfigure(1, weight=1)
        slider_frame.columnconfigure(3, weight=1)

        auto_frame = ttk.Frame(main)
        auto_frame.pack(fill="x", pady=5)

        self.auto_var = tk.BooleanVar(value=False)
        self.auto_check = ttk.Checkbutton(
            auto_frame,
            text="Auto AI Mode",
            variable=self.auto_var,
            command=self.on_auto_mode_changed
        )
        self.auto_check.grid(row=0, column=0, sticky="w", padx=5)

        self.gov_mode_var = tk.StringVar(value="balanced")

        gov_label = ttk.Label(auto_frame, text="Governor Mode:", font=("Arial", 11))
        gov_label.grid(row=1, column=0, sticky="w", padx=5, pady=(2, 0))

        self.gov_radio_conservative = ttk.Radiobutton(
            auto_frame,
            text="Conservative",
            value="conservative",
            variable=self.gov_mode_var,
            command=self.on_governor_mode_changed
        )
        self.gov_radio_conservative.grid(row=1, column=1, sticky="w", padx=10, pady=(2, 0))

        self.gov_radio_balanced = ttk.Radiobutton(
            auto_frame,
            text="Balanced",
            value="balanced",
            variable=self.gov_mode_var,
            command=self.on_governor_mode_changed
        )
        self.gov_radio_balanced.grid(row=1, column=2, sticky="w", padx=10, pady=(2, 0))

        self.gov_radio_aggressive = ttk.Radiobutton(
            auto_frame,
            text="Aggressive",
            value="aggressive",
            variable=self.gov_mode_var,
            command=self.on_governor_mode_changed
        )
        self.gov_radio_aggressive.grid(row=1, column=3, sticky="w", padx=10, pady=(2, 0))

        # CPU POWER STRATEGY
        cpu_ps_frame = ttk.Frame(main)
        cpu_ps_frame.pack(fill="x", pady=5)

        self.cpu_power_strategy_var = tk.StringVar(value="saver")

        cpu_ps_label = ttk.Label(cpu_ps_frame, text="CPU Power Strategy:", font=("Arial", 11))
        cpu_ps_label.grid(row=0, column=0, sticky="w", padx=5)

        self.cpu_ps_radio_saver = ttk.Radiobutton(
            cpu_ps_frame,
            text="Saver",
            value="saver",
            variable=self.cpu_power_strategy_var,
            command=self.on_cpu_power_strategy_changed
        )
        self.cpu_ps_radio_saver.grid(row=0, column=1, sticky="w", padx=8)

        self.cpu_ps_radio_balanced = ttk.Radiobutton(
            cpu_ps_frame,
            text="Balanced",
            value="balanced",
            variable=self.cpu_power_strategy_var,
            command=self.on_cpu_power_strategy_changed
        )
        self.cpu_ps_radio_balanced.grid(row=0, column=2, sticky="w", padx=8)

        self.cpu_ps_radio_aggressive = ttk.Radiobutton(
            cpu_ps_frame,
            text="Aggressive",
            value="aggressive",
            variable=self.cpu_power_strategy_var,
            command=self.on_cpu_power_strategy_changed
        )
        self.cpu_ps_radio_aggressive.grid(row=0, column=3, sticky="w", padx=8)

        self.cpu_ps_radio_custom = ttk.Radiobutton(
            cpu_ps_frame,
            text="Custom",
            value="custom",
            variable=self.cpu_power_strategy_var,
            command=self.on_cpu_power_strategy_changed
        )
        self.cpu_ps_radio_custom.grid(row=0, column=4, sticky="w", padx=8)

        self.cpu_custom_frame = ttk.Frame(main)

        ttk.Label(self.cpu_custom_frame, text="CPU Idle baseline (%)", font=("Arial", 10)).grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.cpu_custom_idle = ttk.Scale(
            self.cpu_custom_frame,
            from_=5,
            to=60,
            orient="horizontal",
            command=self.on_cpu_custom_changed
        )
        self.cpu_custom_idle.grid(row=0, column=1, sticky="we", padx=5)

        ttk.Label(self.cpu_custom_frame, text="CPU Burst throttle (%)", font=("Arial", 10)).grid(
            row=0, column=2, sticky="w", padx=5
        )
        self.cpu_custom_burst = ttk.Scale(
            self.cpu_custom_frame,
            from_=40,
            to=100,
            orient="horizontal",
            command=self.on_cpu_custom_changed
        )
        self.cpu_custom_burst.grid(row=0, column=3, sticky="we", padx=5)

        ttk.Label(self.cpu_custom_frame, text="CPU Burst duration (s)", font=("Arial", 10)).grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.cpu_custom_duration = ttk.Scale(
            self.cpu_custom_frame,
            from_=0.5,
            to=5.0,
            orient="horizontal",
            command=self.on_cpu_custom_changed
        )
        self.cpu_custom_duration.grid(row=1, column=1, sticky="we", padx=5)

        ttk.Label(self.cpu_custom_frame, text="CPU Cooldown speed", font=("Arial", 10)).grid(
            row=1, column=2, sticky="w", padx=5
        )
        self.cpu_custom_cooldown = ttk.Scale(
            self.cpu_custom_frame,
            from_=0.01,
            to=0.10,
            orient="horizontal",
            command=self.on_cpu_custom_changed
        )
        self.cpu_custom_cooldown.grid(row=1, column=3, sticky="we", padx=5)

        ttk.Label(self.cpu_custom_frame, text="CPU Burst trigger (%)", font=("Arial", 10)).grid(
            row=2, column=0, sticky="w", padx=5
        )
        self.cpu_custom_trigger = ttk.Scale(
            self.cpu_custom_frame,
            from_=30,
            to=90,
            orient="horizontal",
            command=self.on_cpu_custom_changed
        )
        self.cpu_custom_trigger.grid(row=2, column=1, sticky="we", padx=5)

        self.cpu_custom_frame.columnconfigure(1, weight=1)
        self.cpu_custom_frame.columnconfigure(3, weight=1)

        # GPU POWER STRATEGY
        gpu_ps_frame = ttk.Frame(main)
        gpu_ps_frame.pack(fill="x", pady=5)

        self.gpu_power_strategy_var = tk.StringVar(value="saver")

        gpu_ps_label = ttk.Label(gpu_ps_frame, text="GPU Power Strategy:", font=("Arial", 11))
        gpu_ps_label.grid(row=0, column=0, sticky="w", padx=5)

        self.gpu_ps_radio_saver = ttk.Radiobutton(
            gpu_ps_frame,
            text="Saver",
            value="saver",
            variable=self.gpu_power_strategy_var,
            command=self.on_gpu_power_strategy_changed
        )
        self.gpu_ps_radio_saver.grid(row=0, column=1, sticky="w", padx=8)

        self.gpu_ps_radio_balanced = ttk.Radiobutton(
            gpu_ps_frame,
            text="Balanced",
            value="balanced",
            variable=self.gpu_power_strategy_var,
            command=self.on_gpu_power_strategy_changed
        )
        self.gpu_ps_radio_balanced.grid(row=0, column=2, sticky="w", padx=8)

        self.gpu_ps_radio_aggressive = ttk.Radiobutton(
            gpu_ps_frame,
            text="Aggressive",
            value="aggressive",
            variable=self.gpu_power_strategy_var,
            command=self.on_gpu_power_strategy_changed
        )
        self.gpu_ps_radio_aggressive.grid(row=0, column=3, sticky="w", padx=8)

        self.gpu_ps_radio_custom = ttk.Radiobutton(
            gpu_ps_frame,
            text="Custom",
            value="custom",
            variable=self.gpu_power_strategy_var,
            command=self.on_gpu_power_strategy_changed
        )
        self.gpu_ps_radio_custom.grid(row=0, column=4, sticky="w", padx=8)

        self.gpu_custom_frame = ttk.Frame(main)

        ttk.Label(self.gpu_custom_frame, text="GPU Idle baseline (%)", font=("Arial", 10)).grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.gpu_custom_idle = ttk.Scale(
            self.gpu_custom_frame,
            from_=5,
            to=60,
            orient="horizontal",
            command=self.on_gpu_custom_changed
        )
        self.gpu_custom_idle.grid(row=0, column=1, sticky="we", padx=5)

        ttk.Label(self.gpu_custom_frame, text="GPU Ramp up", font=("Arial", 10)).grid(
            row=0, column=2, sticky="w", padx=5
        )
        self.gpu_custom_ramp_up = ttk.Scale(
            self.gpu_custom_frame,
            from_=0.005,
            to=0.10,
            orient="horizontal",
            command=self.on_gpu_custom_changed
        )
        self.gpu_custom_ramp_up.grid(row=0, column=3, sticky="we", padx=5)

        ttk.Label(self.gpu_custom_frame, text="GPU Ramp down", font=("Arial", 10)).grid(
            row=1, column=0, sticky="w", padx=5
        )
        self.gpu_custom_ramp_down = ttk.Scale(
            self.gpu_custom_frame,
            from_=0.005,
            to=0.10,
            orient="horizontal",
            command=self.on_gpu_custom_changed
        )
        self.gpu_custom_ramp_down.grid(row=1, column=1, sticky="we", padx=5)

        ttk.Label(self.gpu_custom_frame, text="GPU Load trigger (%)", font=("Arial", 10)).grid(
            row=1, column=2, sticky="w", padx=5
        )
        self.gpu_custom_trigger = ttk.Scale(
            self.gpu_custom_frame,
            from_=30,
            to=90,
            orient="horizontal",
            command=self.on_gpu_custom_changed
        )
        self.gpu_custom_trigger.grid(row=1, column=3, sticky="we", padx=5)

        self.gpu_custom_frame.columnconfigure(1, weight=1)
        self.gpu_custom_frame.columnconfigure(3, weight=1)

        # BOTTOM: start + status
        bottom_frame = ttk.Frame(main)
        bottom_frame.pack(fill="x", pady=10)

        self.start_button = ttk.Button(
            bottom_frame,
            text="Start CPU+GPU Pipeline",
            command=self.start_pipeline_thread
        )
        self.start_button.pack(side="left", padx=10)

        self.status_label = ttk.Label(bottom_frame, text="Status: Idle", font=("Arial", 11))
        self.status_label.pack(side="left", padx=20)

        # Apply settings to UI & globals
        self.apply_settings_to_state_and_ui()

        self.update_usage()

    # --------------------------------------------------------
    # Settings helpers
    # --------------------------------------------------------
    def gather_settings(self):
        return {
            "cpu_throttle": CPU_THROTTLE,
            "gpu_throttle": GPU_THROTTLE,
            "auto_mode": AUTO_MODE,
            "governor_mode": GOVERNOR_MODE,

            "cpu_power_strategy": self.cpu_power_strategy_var.get(),
            "cpu_custom_idle": self.cpu_custom_idle.get() / 100.0,
            "cpu_custom_burst": self.cpu_custom_burst.get() / 100.0,
            "cpu_custom_duration": self.cpu_custom_duration.get(),
            "cpu_custom_cooldown": self.cpu_custom_cooldown.get(),
            "cpu_custom_trigger": self.cpu_custom_trigger.get(),

            "gpu_power_strategy": self.gpu_power_strategy_var.get(),
            "gpu_custom_idle": self.gpu_custom_idle.get() / 100.0,
            "gpu_custom_ramp_up": self.gpu_custom_ramp_up.get(),
            "gpu_custom_ramp_down": self.gpu_custom_ramp_down.get(),
            "gpu_custom_trigger": self.gpu_custom_trigger.get(),
        }

    def apply_settings_to_state_and_ui(self):
        global CPU_THROTTLE, GPU_THROTTLE, AUTO_MODE, GOVERNOR_MODE
        global CPU_POWER_STRATEGY, GPU_POWER_STRATEGY

        s = self.settings

        CPU_THROTTLE = s.get("cpu_throttle", DEFAULT_SETTINGS["cpu_throttle"])
        GPU_THROTTLE = s.get("gpu_throttle", DEFAULT_SETTINGS["gpu_throttle"])
        AUTO_MODE = s.get("auto_mode", DEFAULT_SETTINGS["auto_mode"])
        GOVERNOR_MODE = s.get("governor_mode", DEFAULT_SETTINGS["governor_mode"])

        CPU_POWER_STRATEGY = s.get("cpu_power_strategy", DEFAULT_SETTINGS["cpu_power_strategy"])
        GPU_POWER_STRATEGY = s.get("gpu_power_strategy", DEFAULT_SETTINGS["gpu_power_strategy"])

        # Sliders
        self.cpu_slider.set(CPU_THROTTLE * 100.0)
        self.gpu_slider.set(GPU_THROTTLE * 100.0)

        # Auto mode
        self.auto_var.set(AUTO_MODE)

        # Governor mode
        self.gov_mode_var.set(GOVERNOR_MODE)

        # CPU strategy
        self.cpu_power_strategy_var.set(CPU_POWER_STRATEGY)
        if CPU_POWER_STRATEGY == "custom":
            self.cpu_custom_frame.pack(fill="x", pady=5)
        else:
            self.cpu_custom_frame.pack_forget()

        self.cpu_custom_idle.set(s.get("cpu_custom_idle", DEFAULT_SETTINGS["cpu_custom_idle"]) * 100.0)
        self.cpu_custom_burst.set(s.get("cpu_custom_burst", DEFAULT_SETTINGS["cpu_custom_burst"]) * 100.0)
        self.cpu_custom_duration.set(s.get("cpu_custom_duration", DEFAULT_SETTINGS["cpu_custom_duration"]))
        self.cpu_custom_cooldown.set(s.get("cpu_custom_cooldown", DEFAULT_SETTINGS["cpu_custom_cooldown"]))
        self.cpu_custom_trigger.set(s.get("cpu_custom_trigger", DEFAULT_SETTINGS["cpu_custom_trigger"]))

        # GPU strategy
        self.gpu_power_strategy_var.set(GPU_POWER_STRATEGY)
        if GPU_POWER_STRATEGY == "custom":
            self.gpu_custom_frame.pack(fill="x", pady=5)
        else:
            self.gpu_custom_frame.pack_forget()

        self.gpu_custom_idle.set(s.get("gpu_custom_idle", DEFAULT_SETTINGS["gpu_custom_idle"]) * 100.0)
        self.gpu_custom_ramp_up.set(s.get("gpu_custom_ramp_up", DEFAULT_SETTINGS["gpu_custom_ramp_up"]))
        self.gpu_custom_ramp_down.set(s.get("gpu_custom_ramp_down", DEFAULT_SETTINGS["gpu_custom_ramp_down"]))
        self.gpu_custom_trigger.set(s.get("gpu_custom_trigger", DEFAULT_SETTINGS["gpu_custom_trigger"]))

    def persist_now(self):
        self.settings = self.gather_settings()
        save_settings(self.settings)

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    def on_cpu_slider_changed(self, v):
        set_cpu_throttle_from_slider(v)
        self.persist_now()

    def on_gpu_slider_changed(self, v):
        set_gpu_throttle_from_slider(v)
        self.persist_now()

    def on_auto_mode_changed(self):
        set_auto_mode(self.auto_var.get())
        self.persist_now()

    def on_governor_mode_changed(self):
        set_governor_mode(self.gov_mode_var.get())
        self.persist_now()

    def on_cpu_power_strategy_changed(self):
        mode = self.cpu_power_strategy_var.get()
        set_cpu_power_strategy(mode)
        if mode == "custom":
            self.cpu_custom_frame.pack(fill="x", pady=5)
        else:
            self.cpu_custom_frame.pack_forget()
        self.cpu_burst_active = False
        self.cpu_burst_end_time = 0.0
        self.persist_now()

    def on_gpu_power_strategy_changed(self):
        mode = self.gpu_power_strategy_var.get()
        set_gpu_power_strategy(mode)
        if mode == "custom":
            self.gpu_custom_frame.pack(fill="x", pady=5)
        else:
            self.gpu_custom_frame.pack_forget()
        self.persist_now()

    def on_cpu_custom_changed(self, v):
        # any CPU custom slider moved
        self.persist_now()

    def on_gpu_custom_changed(self, v):
        # any GPU custom slider moved
        self.persist_now()

    # --------------------------------------------------------
    # Pipeline controls
    # --------------------------------------------------------
    def start_pipeline_thread(self):
        self.status_label.config(text="Status: Running pipeline...", foreground="black")
        self.start_button.config(state=tk.DISABLED)
        t = threading.Thread(target=self.run_pipeline_wrapper, daemon=True)
        t.start()

    def run_pipeline_wrapper(self):
        try:
            run_pipeline()
            self.status_label.config(text="Status: Pipeline finished.", foreground="black")
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {e}", foreground="red")
        finally:
            self.start_button.config(state=tk.NORMAL)

    # --------------------------------------------------------
    # Power params
    # --------------------------------------------------------
    def get_cpu_power_params(self):
        mode = self.cpu_power_strategy_var.get()
        if mode in ("saver", "balanced", "aggressive"):
            cfg = CPU_POWER_CONFIG[mode]
            return (
                cfg["idle_baseline"],
                cfg["burst_throttle"],
                cfg["burst_duration"],
                cfg["cooldown_step"],
                cfg["burst_trigger"],
            )
        idle = self.cpu_custom_idle.get() / 100.0
        burst = self.cpu_custom_burst.get() / 100.0
        duration = self.cpu_custom_duration.get()
        cooldown = self.cpu_custom_cooldown.get()
        trigger = self.cpu_custom_trigger.get()
        return idle, burst, duration, cooldown, trigger

    def get_gpu_power_params(self):
        mode = self.gpu_power_strategy_var.get()
        if mode in ("saver", "balanced", "aggressive"):
            cfg = GPU_POWER_CONFIG[mode]
            return (
                cfg["idle_baseline"],
                cfg["ramp_up"],
                cfg["ramp_down"],
                cfg["load_trigger"],
                cfg["thermal_sensitivity"],
            )
        idle = self.gpu_custom_idle.get() / 100.0
        ramp_up = self.gpu_custom_ramp_up.get()
        ramp_down = self.gpu_custom_ramp_down.get()
        trigger = self.gpu_custom_trigger.get()
        return idle, ramp_up, ramp_down, trigger, "medium"

    # --------------------------------------------------------
    # Auto governor
    # --------------------------------------------------------
    def auto_adjust_throttles(self, cpu_percent, gpu_percent, cpu_temp, gpu_temp):
        global CPU_THROTTLE, GPU_THROTTLE

        if not AUTO_MODE:
            return

        # CPU strategy with burst
        idle_baseline, burst_throttle, burst_duration, cooldown_step, burst_trigger = self.get_cpu_power_params()
        now = time.time()

        if self.cpu_burst_active and now >= self.cpu_burst_end_time:
            self.cpu_burst_active = False

        if (not self.cpu_burst_active) and (cpu_percent >= burst_trigger):
            self.cpu_burst_active = True
            self.cpu_burst_end_time = now + burst_duration
            CPU_THROTTLE = burst_throttle

        if self.cpu_burst_active:
            CPU_THROTTLE = max(CPU_THROTTLE, burst_throttle)
        else:
            if CPU_THROTTLE > idle_baseline:
                CPU_THROTTLE = max(idle_baseline, CPU_THROTTLE - cooldown_step)
            elif CPU_THROTTLE < idle_baseline * 0.9:
                CPU_THROTTLE = min(idle_baseline, CPU_THROTTLE + cooldown_step / 2.0)

        # GPU strategy, conservative
        if GPU_AVAILABLE and GPUtil is not None:
            idle_g, ramp_up, ramp_down, trigger, therm_sens = self.get_gpu_power_params()

            if gpu_percent < trigger * 0.7:
                GPU_THROTTLE = max(idle_g, GPU_THROTTLE - ramp_down)
            elif trigger * 0.7 <= gpu_percent <= trigger * 1.2:
                if gpu_percent > trigger:
                    GPU_THROTTLE = max(idle_g, GPU_THROTTLE - ramp_down / 2.0)
                else:
                    GPU_THROTTLE = min(1.0, GPU_THROTTLE + ramp_up / 4.0)
            else:
                GPU_THROTTLE = min(1.0, GPU_THROTTLE + ramp_up)

            # governor tweaks
            cfg = GOV_CONFIG.get(GOVERNOR_MODE, GOV_CONFIG["balanced"])
            gpu_target = cfg["gpu_target"]
            step_up = cfg["step_up"]
            step_down = cfg["step_down"]
            gpu_upper = gpu_target + 15
            gpu_lower = gpu_target - 15

            if gpu_percent > gpu_upper:
                GPU_THROTTLE = max(0.05, GPU_THROTTLE - step_down)
            elif gpu_percent < gpu_lower:
                GPU_THROTTLE = min(1.0, GPU_THROTTLE + step_up)

        overheating = False

        if cpu_temp is not None:
            if cpu_temp > 85:
                CPU_THROTTLE = max(0.05, CPU_THROTTLE - 0.10)
                overheating = True
            elif cpu_temp > 80:
                CPU_THROTTLE = max(0.05, CPU_THROTTLE - 0.05)
                overheating = True

        if gpu_temp is not None:
            if gpu_temp > 85:
                GPU_THROTTLE = max(0.05, GPU_THROTTLE - 0.10)
                overheating = True
            elif gpu_temp > 80:
                GPU_THROTTLE = max(0.05, GPU_THROTTLE - 0.05)
                overheating = True

        if overheating:
            self.status_label.config(text="Status: Cooldown mode (thermal limit)", foreground="orange")
        else:
            if "Error" not in self.status_label.cget("text") and "Running" not in self.status_label.cget("text"):
                if "Cooldown" in self.status_label.cget("text"):
                    self.status_label.config(text="Status: Idle", foreground="black")

        self.cpu_slider.set(CPU_THROTTLE * 100.0)
        self.gpu_slider.set(GPU_THROTTLE * 100.0)

        # persist throttles as well
        self.persist_now()

    # --------------------------------------------------------
    # periodic update
    # --------------------------------------------------------
    def update_usage(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_label.config(text=f"CPU: {cpu_percent:.0f}%")

        cpu_temp = None
        if IS_WINDOWS:
            self.cpu_temp_label.config(text="CPU Temp: N/A", foreground="black")
        else:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    if "coretemp" in temps:
                        cpu_temp = temps["coretemp"][0].current
                    elif "cpu-thermal" in temps:
                        cpu_temp = temps["cpu-thermal"][0].current

                if cpu_temp is not None:
                    self.cpu_temp_label.config(text=f"CPU Temp: {cpu_temp:.0f}°C")
                    if cpu_temp > 85:
                        self.cpu_temp_label.config(foreground="red")
                    elif cpu_temp > 80:
                        self.cpu_temp_label.config(foreground="orange")
                    else:
                        self.cpu_temp_label.config(foreground="black")
                else:
                    self.cpu_temp_label.config(text="CPU Temp: N/A", foreground="black")
            except NotImplementedError:
                self.cpu_temp_label.config(text="CPU Temp: N/A", foreground="black")
            except Exception:
                self.cpu_temp_label.config(text="CPU Temp: Error", foreground="red")

        ram_percent = psutil.virtual_memory().percent
        self.ram_label.config(text=f"RAM: {ram_percent:.0f}%")

        gpu_percent = 0.0
        gpu_temp = None

        if GPU_AVAILABLE and GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_percent = gpu.load * 100
                    self.gpu_label.config(text=f"GPU: {gpu_percent:.0f}%")

                    gpu_temp = gpu.temperature
                    self.gpu_temp_label.config(text=f"GPU Temp: {gpu_temp}°C")

                    vram_used = gpu.memoryUsed
                    vram_total = gpu.memoryTotal
                    vram_percent = (vram_used / vram_total) * 100
                    self.vram_label.config(text=f"VRAM: {vram_percent:.0f}%")
                else:
                    self.gpu_label.config(text="GPU: N/A")
                    self.gpu_temp_label.config(text="GPU Temp: N/A")
                    self.vram_label.config(text="VRAM: N/A")
            except Exception:
                self.gpu_label.config(text="GPU: Error")
                self.gpu_temp_label.config(text="GPU Temp: Error")
                self.vram_label.config(text="VRAM: Error")
        else:
            self.gpu_label.config(text="GPU: N/A")

        self.auto_adjust_throttles(cpu_percent, gpu_percent, cpu_temp, gpu_temp)
        self.root.after(500, self.update_usage)


# ============================================================
# MAIN
# ============================================================

def main():
    root = tk.Tk()
    gui = MonitorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

