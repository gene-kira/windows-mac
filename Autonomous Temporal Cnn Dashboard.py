# -------------------------------------------------------------------
# Auto-loader for required packages
# -------------------------------------------------------------------
import importlib
import subprocess
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "matplotlib",
    "scikit-learn",
    "tensorflow",
    "optuna",
    "tensorflow_model_optimization"
    # Note: tkinter is part of the Python standard library. On Linux, you may need system package python3-tk.
]

def auto_loader(packages):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"[OK] {pkg} already installed.")
        except ImportError:
            print(f"[MISSING] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"[DONE] {pkg} installed.")

auto_loader(REQUIRED_PACKAGES)

# -------------------------------------------------------------------
# Imports (safe after auto-loader)
# -------------------------------------------------------------------
import os
import json
import hashlib
import logging
import numpy as np
import random
import threading
from datetime import datetime
from queue import Queue

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tensorflow as tf
import optuna
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    roc_auc_score
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
)
from tensorflow_model_optimization.sparsity.keras import (
    prune_low_magnitude, PolynomialDecay, UpdatePruningStep, strip_pruning
)

# -------------------------------------------------------------------
# Reproducibility and directories
# -------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------------------------------
# Config hashing and secure logging stub
# -------------------------------------------------------------------
def hash_config(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()

def secure_log_config(cfg: dict, path="secure_config_log.json"):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "config_hash": hash_config(cfg),
        "config": cfg
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# -------------------------------------------------------------------
# Data and augmentation
# -------------------------------------------------------------------
def load_and_preprocess_data(seq_length=3, num_samples=1000, h=64, w=64, c=3):
    X = np.random.rand(num_samples, seq_length, h, w, c).astype(np.float32)  # [0,1]
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int32)
    return X, y

def sequence_augment(x, y, flip_prob=0.5, bright=0.1, cl=0.9, cu=1.1):
    # Deterministic-style params per sample
    do_flip = tf.less(tf.random.uniform([], seed=SEED), flip_prob)
    b = tf.random.uniform([], -bright, bright, seed=SEED)
    c = tf.random.uniform([], cl, cu, seed=SEED)

    def apply_frame(f):
        f = tf.cond(do_flip, lambda: tf.image.flip_left_right(f), lambda: f)
        f = tf.image.adjust_brightness(f, b)
        f = tf.image.adjust_contrast(f, c)
        return tf.clip_by_value(f, 0.0, 1.0)

    x_aug = tf.map_fn(apply_frame, x, fn_output_signature=tf.float32)
    return x_aug, y

def make_datasets(Xtr, ytr, Xv, yv, batch=32, augment=False):
    def _prep(x, y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(len(Xtr), seed=SEED).map(
        _prep, num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment:
        ds_tr = ds_tr.map(sequence_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds_tr = ds_tr.batch(batch).prefetch(tf.data.AUTOTUNE)
    ds_v = tf.data.Dataset.from_tensor_slices((Xv, yv)).map(
        _prep, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds_tr, ds_v

# -------------------------------------------------------------------
# Model (Functional API) with safe pooling
# -------------------------------------------------------------------
def build_temporal_cnn(input_shape, num_classes):
    inp = tf.keras.Input(shape=input_shape)  # (T,H,W,C)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(inp)
    x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inp, outputs=out)

# -------------------------------------------------------------------
# Plotting helpers (compact figures)
# -------------------------------------------------------------------
def fig_loss(history):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(history.history.get('loss', []), label='train')
    ax.plot(history.history.get('val_loss', []), label='val')
    ax.set_title("Loss")
    ax.legend()
    return fig

def fig_acc(history):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(history.history.get('accuracy', []), label='train')
    ax.plot(history.history.get('val_accuracy', []), label='val')
    ax.set_title("Accuracy")
    ax.legend()
    return fig

def fig_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], '--')
    ax.set_title("ROC")
    ax.legend()
    return fig

def fig_pr(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, label=f"AP={ap:.2f}")
    ax.set_title("PR")
    ax.legend()
    return fig

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_model(model, Xv, yv):
    y_scores = model.predict(Xv).ravel()
    y_pred = (y_scores > 0.5).astype(int)

    logging.info("\n" + classification_report(yv, y_pred))
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(yv, y_pred)))

    return {
        "fig_roc": fig_roc(yv, y_scores),
        "fig_pr": fig_pr(yv, y_scores)
    }

# -------------------------------------------------------------------
# Optuna objective
# -------------------------------------------------------------------
def objective(trial):
    global X_train, X_val, y_train, y_val
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    model = build_temporal_cnn(input_shape, num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    ds_tr, ds_v = make_datasets(X_train, y_train, X_val, y_val, batch=32, augment=False)
    model.fit(ds_tr, epochs=5, validation_data=ds_v, verbose=0)
    _, val_acc = model.evaluate(ds_v, verbose=0)
    return val_acc

# -------------------------------------------------------------------
# Representative dataset for int8 quantization
# -------------------------------------------------------------------
def representative_dataset_gen(X_calib, batch_size=1):
    for i in range(min(100, len(X_calib))):
        sample = X_calib[i:i+batch_size]
        yield [sample.astype(np.float32)]

# -------------------------------------------------------------------
# Autonomous GUI (50% smaller, thread-safe logs via queue)
# -------------------------------------------------------------------
class AutonomousGUI:
    def __init__(self, root, pipeline_fn):
        self.root = root
        self.root.title("Autonomous Temporal CNN Dashboard")
        self.root.geometry("550x400")  # 50% smaller

        # Status
        self.status_label = ttk.Label(root, text="Initializing...", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Log window
        self.log_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tabs for plots
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.tab_loss = ttk.Frame(self.tabs); self.tabs.add(self.tab_loss, text="Loss")
        self.tab_acc  = ttk.Frame(self.tabs); self.tabs.add(self.tab_acc,  text="Accuracy")
        self.tab_roc  = ttk.Frame(self.tabs); self.tabs.add(self.tab_roc,  text="ROC")
        self.tab_pr   = ttk.Frame(self.tabs); self.tabs.add(self.tab_pr,   text="PR")

        # Thread-safe logging
        self.log_queue = Queue()
        self.root.after(200, self._drain_log_queue)

        # Auto-run pipeline
        threading.Thread(target=self._run_pipeline, args=(pipeline_fn,), daemon=True).start()

    def log(self, msg):
        self.log_queue.put(msg)

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.root.after(200, self._drain_log_queue)

    def _embed_fig(self, fig, tab):
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_pipeline(self, pipeline_fn):
        self.status_label.config(text="Training...")
        try:
            results = pipeline_fn(self.log)
            self._embed_fig(results["fig_loss"], self.tab_loss)
            self._embed_fig(results["fig_acc"],  self.tab_acc)
            self._embed_fig(results["fig_roc"],  self.tab_roc)
            self._embed_fig(results["fig_pr"],   self.tab_pr)
            self.status_label.config(text="Done")
            self.log("Pipeline completed successfully.")
        except Exception as e:
            self.status_label.config(text="Error")
            self.log(f"ERROR: {e}")

# -------------------------------------------------------------------
# Pipeline: data -> optuna -> train -> eval -> prune -> eval -> quantize
# -------------------------------------------------------------------
def training_pipeline(log_fn):
    global X_train, X_val, y_train, y_val

    # Config
    run_cfg = {
        "seed": SEED,
        "seq_length": 3,
        "height": 64,
        "width": 64,
        "channels": 3,
        "batch_size": 32,
        "optuna_trials": 10,
        "epochs": 50,
        "prune_epochs": 10
    }
    secure_log_config(run_cfg)
    log_fn(f"Run config hash: {hash_config(run_cfg)}")

    # Data
    X, y = load_and_preprocess_data(
        seq_length=run_cfg["seq_length"],
        h=run_cfg["height"],
        w=run_cfg["width"],
        c=run_cfg["channels"]
    )
    split = int(0.8 * len(X))
    X_train, X_val, y_train, y_val = X[:split], X[split:], y[:split], y[split:]
    log_fn(f"Dataset split: train={len(X_train)} val={len(X_val)}")

    # Optuna
    study = optuna.create_study(direction='maximize')
    log_fn("Optuna study started.")
    study.optimize(objective, n_trials=run_cfg["optuna_trials"])
    best_trial = study.best_trial
    log_fn(f"Best trial #{best_trial.number} acc={best_trial.value:.4f} lr={best_trial.params['learning_rate']:.6f}")

    # Build best model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    model = build_temporal_cnn(input_shape, num_classes=1)
    optimizer = tf.keras.optimizers.Adam(best_trial.params['learning_rate'])
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    ds_tr, ds_v = make_datasets(X_train, y_train, X_val, y_val, batch=run_cfg["batch_size"], augment=True)

    # Telemetry callbacks
    tb_callback = TensorBoard(log_dir="logs/tensorboard", histogram_freq=1)
    csv_logger = CSVLogger("logs/training_metrics.csv", append=True)
    cp_callback = ModelCheckpoint(
        filepath="checkpoints/best_weights.h5",
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    class GuiLogger(tf.keras.callbacks.Callback):
        def __init__(self, fn): super().__init__(); self.fn = fn
        def on_epoch_end(self, epoch, logs=None):
            self.fn(f"Epoch {epoch+1}: "
                    f"loss={logs.get('loss', 0):.4f}, acc={logs.get('accuracy', 0):.4f}, "
                    f"val_loss={logs.get('val_loss', 0):.4f}, val_acc={logs.get('val_accuracy', 0):.4f}")

    gui_logger = GuiLogger(log_fn)

    # Train
    log_fn("Training started.")
    history = model.fit(
        ds_tr,
        epochs=run_cfg["epochs"],
        validation_data=ds_v,
        callbacks=[tb_callback, csv_logger, cp_callback, early_stopping, gui_logger],
        verbose=0
    )
    log_fn("Training finished.")

    # Evaluate
    eval_figs = evaluate_model(model, X_val, y_val)
    log_fn("Evaluation complete.")

    # Pruning and fine-tuning
    log_fn("Pruning started.")
    pruning_params = {
        'pruning_schedule': PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=2000
        )
    }
    pruned_model = prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    pruned_model.fit(
        ds_tr,
        epochs=run_cfg["prune_epochs"],
        validation_data=ds_v,
        callbacks=[UpdatePruningStep(), tb_callback, csv_logger, cp_callback, early_stopping, gui_logger],
        verbose=0
    )
    pruned_model = strip_pruning(pruned_model)
    log_fn("Pruning finished and stripped.")
    pr_eval_figs = evaluate_model(pruned_model, X_val, y_val)
    log_fn("Pruned model evaluation complete.")

    # Quantization: dynamic-range
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()
    with open('pruned_quantized_model_dynamic.tflite', 'wb') as f:
        f.write(tflite_dynamic)
    log_fn("Saved dynamic-range TFLite: pruned_quantized_model_dynamic.tflite")

    # Quantization: full int8
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = lambda: representative_dataset_gen(X_val)
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8
    tflite_int8 = converter_int8.convert()
    with open('pruned_quantized_model_int8.tflite', 'wb') as f:
        f.write(tflite_int8)
    log_fn("Saved full-int8 TFLite: pruned_quantized_model_int8.tflite")

    # Figures for GUI embedding (use original training curves)
    return {
        "fig_loss": fig_loss(history),
        "fig_acc": fig_acc(history),
        "fig_roc": eval_figs["fig_roc"],
        "fig_pr": eval_figs["fig_pr"]
    }

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
def main():
    root = tk.Tk()
    gui = AutonomousGUI(root, training_pipeline)
    root.mainloop()

if __name__ == "__main__":
    main()

