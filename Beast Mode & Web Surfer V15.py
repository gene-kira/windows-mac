#!/usr/bin/env python3
import sys
import os
import subprocess
import importlib
import time
import threading
import platform

IS_WEB_SURFER = "--websurfer" in sys.argv
IS_TRAINER = "--trainer" in sys.argv
IS_GUI = not IS_WEB_SURFER and not IS_TRAINER

IS_WINDOWS = (platform.system().lower() == "windows")

def get_installed_packages():
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"]
        ).decode().lower().splitlines()
        return {line.split("==")[0] for line in out if "==" in line}
    except Exception:
        return set()

def ensure_lib(name, pip_name=None):
    if pip_name is None:
        pip_name = name
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            importlib.invalidate_caches()
            importlib.import_module(name)
            return True
        except Exception as e:
            print(f"[Autoloader] Failed to install {pip_name}: {e}")
            return False

if IS_GUI:
    GUI_LIBS = ["psutil", "PyQt5"]
    installed = get_installed_packages()
    for lib in GUI_LIBS:
        if lib.lower() not in installed:
            ensure_lib(lib)

    import psutil
    from PyQt5 import QtCore, QtGui, QtWidgets

if IS_WEB_SURFER:
    WEB_LIBS = [
        "flask",
        "flask-wtf",
        "flask-login",
        "sqlalchemy",
        "flask_sqlalchemy",
        "flask-talisman",
        "werkzeug",
        "requests",
        "beautifulsoup4",
        "sentence-transformers",
        "python-dotenv",
        "psutil",
    ]
    installed = get_installed_packages()
    for lib in WEB_LIBS:
        if lib == "python-dotenv":
            if "python-dotenv" not in installed and "dotenv" not in installed:
                ensure_lib("dotenv", "python-dotenv")
        else:
            if lib.lower() not in installed:
                ensure_lib(lib)

    from flask import Flask, request, render_template_string, redirect, url_for, flash
    from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
    from flask_wtf.csrf import CSRFProtect
    from wtforms import Form, StringField, PasswordField, validators
    import requests as http_requests
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer, util
    from werkzeug.security import generate_password_hash, check_password_hash
    from flask_sqlalchemy import SQLAlchemy
    from flask_talisman import Talisman
    from dotenv import load_dotenv
    from datetime import datetime, timedelta
    import psutil

if IS_TRAINER:
    TRAIN_LIBS = [
        "numpy",
        "scikit-learn",
        "tensorflow",
        "scikit-image",
        "scikit-optimize",
        "tf-keras",
        "psutil",
        "multiprocessing-logging",
    ]
    installed = get_installed_packages()
    for lib in TRAIN_LIBS:
        if lib.lower() not in installed:
            ensure_lib(lib)

    import numpy as np
    from sklearn.model_selection import train_test_split
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from skimage.transform import rotate, rescale
    from skimage.util import random_noise, img_as_ubyte
    import concurrent.futures
    import time as _time
    import psutil
    import multiprocessing
    import tensorflow as tf

# -------------------------
# Web Surfer (Flask app)
# -------------------------
def create_websurfer_app():
    load_dotenv()

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL', 'sqlite:///site.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db = SQLAlchemy(app)
    login_manager = LoginManager(app)
    csrf = CSRFProtect(app)
    talisman = Talisman(app)

    class User(db.Model, UserMixin):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(20), unique=True, nullable=False)
        password_hash = db.Column(db.String(128), nullable=False)

        def set_password(self, password):
            self.password_hash = generate_password_hash(password)

        def check_password(self, password):
            return check_password_hash(self.password_hash, password)

    with app.app_context():
        db.create_all()

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    class RegistrationForm(Form):
        username = StringField('Username', [validators.Length(min=4, max=25)])
        password = PasswordField('Password', [
            validators.DataRequired(),
            validators.EqualTo('confirm', message='Passwords must match')
        ])
        confirm = PasswordField('Confirm Password')

    class LoginForm(Form):
        username = StringField('Username', [validators.Length(min=4, max=25)])
        password = PasswordField('Password', [validators.DataRequired()])

    class ScrapeForm(Form):
        url = StringField('URL', [validators.URL(), validators.Length(min=4, max=256)])

    class SearchForm(Form):
        query = StringField('Query', [validators.Length(min=1, max=256)])

    documents = [
        {'id': 1, 'title': 'Document 1', 'content': 'This is the content of document one.'},
        {'id': 2, 'title': 'Document 2', 'content': 'This document contains important information.'}
    ]

    email_cache = {
        1: {'id': 1, 'subject': 'Promotion', 'content': 'Congratulations! You have won a prize.', 'safe': False},
        2: {'id': 2, 'subject': 'Meeting Reminder', 'content': 'Reminder about tomorrow\'s meeting.', 'safe': True}
    }

    scraped_cache = {}

    def fetch_emails():
        return [em for em in email_cache.values() if not em['safe']]

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        form = RegistrationForm(request.form)
        if request.method == 'POST' and form.validate():
            user = User(username=form.username.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Register</h2>
                <form method="post">
                    {{ form.hidden_tag() }}
                    <div class="form-group">
                        {{ form.username.label }}<br>
                        {{ form.username(size=32) }}<br>
                        {% for error in form.username.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <div class="form-group">
                        {{ form.password.label }}<br>
                        {{ form.password() }}<br>
                        {% for error in form.password.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <div class="form-group">
                        {{ form.confirm.label }}<br>
                        {{ form.confirm() }}<br>
                        {% for error in form.confirm.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <input type="submit" class="btn btn-primary" value="Register">
                </form>
            </div>
        ''', form=form)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        form = LoginForm(request.form)
        if request.method == 'POST' and form.validate():
            user = User.query.filter_by(username=form.username.data).first()
            if user and user.check_password(form.password.data):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password', 'danger')
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Login</h2>
                <form method="post">
                    {{ form.hidden_tag() }}
                    <div class="form-group">
                        {{ form.username.label }}<br>
                        {{ form.username(size=32) }}<br>
                        {% for error in form.username.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <div class="form-group">
                        {{ form.password.label }}<br>
                        {{ form.password() }}<br>
                        {% for error in form.password.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <input type="submit" class="btn btn-primary" value="Login">
                </form>
            </div>
        ''', form=form)

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login'))

    @app.route('/')
    @login_required
    def index():
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Welcome to the Web Surfing Assistant</h2>
                <a href="{{ url_for('scrape') }}" class="btn btn-primary">Scrape Website</a>
                <a href="{{ url_for('search') }}" class="btn btn-secondary">Search Documents</a>
                <a href="{{ url_for('email_protection') }}" class="btn btn-danger">Email Protection</a>
            </div>
        ''')

    @app.route('/scrape', methods=['GET', 'POST'])
    @login_required
    def scrape():
        form = ScrapeForm(request.form)
        if request.method == 'POST' and form.validate():
            url = form.url.data
            if url in scraped_cache and scraped_cache[url]['timestamp'] + timedelta(hours=1) > datetime.utcnow():
                content = scraped_cache[url]['content']
            else:
                try:
                    response = http_requests.get(url, timeout=5)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = [p.get_text() for p in soup.find_all('p')[:5]]
                    content = '\n'.join(paragraphs)
                    scraped_cache[url] = {'content': content, 'timestamp': datetime.utcnow()}
                except http_requests.RequestException as e:
                    flash(f'Error fetching the URL: {e}', 'danger')
                    return redirect(url_for('scrape'))
            return render_template_string('''
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
                <div class="container mt-5">
                    <h2>Scraped Content</h2>
                    <pre>{{ content }}</pre>
                    <a href="{{ url_for('scrape') }}" class="btn btn-secondary">Back to Scrape</a>
                </div>
            ''', content=content)
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Scrape Website</h2>
                <form method="post">
                    {{ form.hidden_tag() }}
                    <div class="form-group">
                        {{ form.url.label }}<br>
                        {{ form.url(size=32) }}<br>
                        {% for error in form.url.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <input type="submit" class="btn btn-primary" value="Scrape">
                </form>
            </div>
        ''', form=form)

    @app.route('/search', methods=['GET', 'POST'])
    @login_required
    def search():
        form = SearchForm(request.form)
        if request.method == 'POST' and form.validate():
            query = form.query.data.lower()
            results = []
            for doc in documents:
                if query in doc['content'].lower():
                    results.append(doc)
            return render_template_string('''
                <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
                <div class="container mt-5">
                    <h2>Search Results</h2>
                    {% if results %}
                        {% for result in results %}
                            <div class="card mb-3">
                                <div class="card-body">
                                    <h5 class="card-title">{{ result['title'] }}</h5>
                                    <p class="card-text">{{ result['content'][:100] }}...</p>
                                </div>
                            </div>
                        {% endfor %}
                    {% else %}
                        <p>No results found.</p>
                    {% endif %}
                    <a href="{{ url_for('search') }}" class="btn btn-secondary">Back to Search</a>
                </div>
            ''', results=results)
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Search Documents</h2>
                <form method="post">
                    {{ form.hidden_tag() }}
                    <div class="form-group">
                        {{ form.query.label }}<br>
                        {{ form.query(size=32) }}<br>
                        {% for error in form.query.errors %}
                            <span style="color: red;">[{{ error }}]</span>
                        {% endfor %}
                    </div>
                    <input type="submit" class="btn btn-primary" value="Search">
                </form>
            </div>
        ''', form=form)

    @app.route('/email_protection')
    @login_required
    def email_protection():
        emails = fetch_emails()
        return render_template_string('''
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
            <div class="container mt-5">
                <h2>Email Protection</h2>
                {% if emails %}
                    {% for email in emails %}
                        <div class="card mb-3">
                            <div class="card-body">
                                <h5 class="card-title">{{ email['subject'] }}</h5>
                                <p class="card-text">{{ email['content'][:100] }}...</p>
                                <a href="{{ url_for('mark_safe', email_id=email['id']) }}" class="btn btn-success">Mark as Safe</a>
                            </div>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>No emails found.</p>
                {% endif %}
            </div>
        ''', emails=emails)

    @app.route('/mark_safe/<int:email_id>')
    @login_required
    def mark_safe(email_id):
        if email_id in email_cache:
            email_cache[email_id]['safe'] = True
        flash(f'Email {email_id} marked as safe.', 'success')
        return redirect(url_for('email_protection'))

    return app

def run_websurfer():
    try:
        p = psutil.Process()
        total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        if total_cores > 2:
            beast_cores = list(range(2, total_cores))
            p.cpu_affinity(beast_cores)
            print(f"[WebSurfer] Affinity set to Beast cores: {beast_cores}")
        else:
            print("[WebSurfer] Not enough cores to reserve 2 for background; using all cores.")
    except Exception as e:
        print(f"[WebSurfer] Failed to set CPU affinity: {e}")

    app = create_websurfer_app()
    app.run(debug=False, host="127.0.0.1", port=5000)

# -------------------------
# Trainer
# -------------------------
if IS_TRAINER:
    def augment_image(image):
        angle = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.8, 1.2)
        flip_axis = np.random.choice([0, 1])

        augmented_image = rotate(image, angle)
        augmented_image = rescale(augmented_image, scale, anti_aliasing=True)
        if flip_axis == 0:
            augmented_image = np.flip(augmented_image, axis=0)
        else:
            augmented_image = np.flip(augmented_image, axis=1)

        augmented_image = random_noise(augmented_image)
        augmented_image = img_as_ubyte(augmented_image)
        return augmented_image

# -------------------------
# Hardware detection helpers
# -------------------------
def detect_npu_devices():
    devices = []
    try:
        try:
            ov = importlib.import_module("openvino.runtime")
            core = ov.Core()
            for dev in core.available_devices:
                devices.append(f"OpenVINO device: {str(dev)}")
        except Exception:
            pass
        try:
            importlib.import_module("tflite_runtime.interpreter")
            devices.append("Coral / Edge TPU (runtime present)")
        except Exception:
            pass
        try:
            importlib.import_module("hailo_platform")
            devices.append("Hailo accelerator (SDK present)")
        except Exception:
            pass
    except Exception:
        pass
    return devices

def detect_movidius_status():
    try:
        ov = importlib.import_module("openvino.runtime")
        core = ov.Core()
        devs = [str(d) for d in core.available_devices]
        if any("MYRIAD" in d.upper() for d in devs):
            return True, "Movidius MYRIAD (USB) detected via OpenVINO"
        return False, "Movidius MYRIAD not found in OpenVINO devices"
    except Exception as e:
        return False, f"Movidius detection failed: {e}"

def detect_coral_tpu_status():
    try:
        import tflite_runtime.interpreter as tflite
        return True, "Coral / Edge TPU runtime detected (tflite_runtime available)"
    except Exception as e:
        return False, f"Coral / Edge TPU not available: {e}"

def detect_gpu_devices_and_vram():
    gpus = []
    free_vram_mb = None
    try:
        import tensorflow as tf
        physical_gpus = tf.config.list_physical_devices('GPU')
        for g in physical_gpus:
            gpus.append(str(g))
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()
        if out:
            free_vram_mb = int(out[0])
    except Exception:
        pass

    return gpus, free_vram_mb

def detect_usb_devices_summary():
    summary = []
    try:
        if IS_WINDOWS:
            out = subprocess.check_output(
                ["wmic", "path", "Win32_PnPEntity", "get", "Name"],
                stderr=subprocess.DEVNULL
            ).decode(errors="ignore").splitlines()
            for line in out:
                line = line.strip().replace("\x00", "")
                if line and "usb" in line.lower():
                    summary.append(line)
        else:
            try:
                out = subprocess.check_output(
                    ["lsusb"],
                    stderr=subprocess.DEVNULL
                ).decode(errors="ignore").splitlines()
                for line in out:
                    summary.append(line.replace("\x00", ""))
            except Exception:
                pass
    except Exception:
        pass
    return summary

# -------------------------
# Water-physics predictor
# -------------------------
class WaterPhysicsPredictor:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed = None
        self.last_time = None
        self.velocity = 0.0

    def update(self, t, value):
        if value is None:
            return None, None

        v = float(value)
        if self.smoothed is None:
            self.smoothed = v
            self.last_time = t
            self.velocity = 0.0
            return self.smoothed, self.smoothed

        dt = max(0.1, t - self.last_time)
        prev = self.smoothed
        self.smoothed = self.alpha * v + (1.0 - self.alpha) * self.smoothed
        self.velocity = (self.smoothed - prev) / dt
        self.last_time = t

        horizon = 5.0
        forecast = self.smoothed + self.velocity * horizon
        forecast = max(0.0, min(100.0, forecast))
        return self.smoothed, forecast

# -------------------------
# GPU/TPU load balancer policy
# -------------------------
class ResourcePolicy:
    def __init__(self):
        self.last_decision = None
        self.last_change_time = 0.0
        self.cooldown = 5.0

    def choose_engine(self, now, beast_forecast, gpu_free_mb, movidius_ok, coral_ok):
        if beast_forecast is None:
            decision = "CPU"
        else:
            hot_future = beast_forecast > 70.0
            medium_future = beast_forecast > 50.0

            if gpu_free_mb is not None and gpu_free_mb >= 2048 and hot_future:
                decision = "GPU"
            elif movidius_ok and medium_future:
                decision = "MOVIDIUS"
            elif coral_ok and medium_future:
                decision = "CORAL"
            else:
                decision = "CPU"

        if self.last_decision is not None and decision != self.last_decision:
            if now - self.last_change_time < self.cooldown:
                decision = self.last_decision
            else:
                self.last_change_time = now

        if decision != self.last_decision:
            print(f"[Scheduler] Engine decision changed: {self.last_decision} -> {decision}")
        self.last_decision = decision
        return decision

# -------------------------
# Trainer main
# -------------------------
def trainer_main():
    try:
        p = psutil.Process()
        total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        if total_cores > 2:
            beast_cores = list(range(2, total_cores))
            p.cpu_affinity(beast_cores)
            print(f"[Trainer] Affinity set to Beast cores: {beast_cores}")
        else:
            print("[Trainer] Not enough cores to reserve 2 for background; using all cores.")
    except Exception as e:
        print(f"[Trainer] Failed to set CPU affinity: {e}")

    try:
        total_threads = multiprocessing.cpu_count()
        beast_threads = max(1, total_threads - 2)
        tf.config.threading.set_intra_op_parallelism_threads(beast_threads)
        tf.config.threading.set_inter_op_parallelism_threads(beast_threads)
        print(f"[Trainer] Beast Mode threads set to: {beast_threads} (of {total_threads})")
    except Exception as e:
        print(f"[Trainer] Failed to configure TensorFlow threads: {e}")

    use_gpu = False
    try:
        gpus, free_vram_mb = detect_gpu_devices_and_vram()
        if gpus and (free_vram_mb is None or free_vram_mb >= 2048):
            use_gpu = True
            print(f"Using GPU for training: {gpus[0]} (free VRAM: {free_vram_mb} MB)")
        else:
            print("GPU present but not enough free VRAM or not detected; using CPU.")
    except Exception:
        print("GPU detection failed; using CPU.")

    movidius_detected, movidius_desc = detect_movidius_status()
    print(movidius_desc)

    coral_detected, coral_desc = detect_coral_tpu_status()
    print(coral_desc)

    try:
        npu_devices = detect_npu_devices()
        if npu_devices:
            print("Detected NPU devices:")
            for d in npu_devices:
                print("  -", d)
        else:
            print("No additional NPU devices detected.")
    except Exception:
        print("NPU detection failed.")

    ov_core = None
    movidius_device_name = None
    if movidius_detected:
        try:
            import openvino.runtime as ov
            ov_core = ov.Core()
            for dev in ov_core.available_devices:
                dev_str = str(dev)
                if "MYRIAD" in dev_str.upper():
                    movidius_device_name = dev_str
                    break
            if movidius_device_name:
                print(f"Movidius integration active on device: {movidius_device_name}")
            else:
                print("Movidius reported but MYRIAD device name not found.")
        except Exception as e:
            print(f"Failed to initialize OpenVINO Core for Movidius: {e}")
            ov_core = None

    coral_interpreter = None
    if coral_detected:
        try:
            import tflite_runtime.interpreter as tflite
            coral_model_path = os.environ.get("CORAL_TPU_MODEL", "")
            if coral_model_path and os.path.exists(coral_model_path):
                delegate_lib = "edgetpu.dll" if IS_WINDOWS else "libedgetpu.so.1"
                coral_interpreter = tflite.Interpreter(
                    model_path=coral_model_path,
                    experimental_delegates=[tflite.load_delegate(delegate_lib)]
                )
                coral_interpreter.allocate_tensors()
                print(f"Coral TPU interpreter initialized with model: {coral_model_path}")
            else:
                print("Coral TPU runtime present, but no model path set (env CORAL_TPU_MODEL).")
        except Exception as e:
            print(f"Failed to initialize Coral TPU interpreter: {e}")
            coral_interpreter = None

    def load_data():
        num_samples = 1000
        img_height, img_width = 32, 32
        X = np.random.rand(num_samples, img_height, img_width, 3) * 255.0
        y = np.random.randint(0, 10, size=(num_samples,))
        return X, y

    def augment_dataset(X):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            X_augmented = list(executor.map(augment_image, X))
        return np.array(X_augmented)

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

    @use_named_args(dimensions=[
        Real(low=1e-6, high=1e-2, name='learning_rate'),
        Real(low=0.1, high=0.5, name='dropout_rate1'),
        Real(low=0.1, high=0.5, name='dropout_rate2')
    ])
    def objective_function(learning_rate, dropout_rate1, dropout_rate2):
        X, y = load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_augmented = augment_dataset(X_train)
        X_val_augmented = augment_dataset(X_val)

        input_shape = X_train_augmented.shape[1:]
        num_classes = len(np.unique(y))
        model = build_model(
            input_shape, num_classes,
            dropout_rate1, dropout_rate2, learning_rate
        )

        history = model.fit(
            X_train_augmented, y_train,
            epochs=10, batch_size=32,
            validation_data=(X_val_augmented, y_val),
            verbose=0
        )

        return -np.max(history.history['val_accuracy'])

    dimensions = [
        Real(low=1e-6, high=1e-2, name='learning_rate'),
        Real(low=0.1, high=0.5, name='dropout_rate1'),
        Real(low=0.1, high=0.5, name='dropout_rate2')
    ]

    print("Starting hyperparameter optimization...")
    result = gp_minimize(objective_function, dimensions, n_calls=30, random_state=42)

    best_learning_rate = result.x[0]
    best_dropout_rate1 = result.x[1]
    best_dropout_rate2 = result.x[2]
    best_val_accuracy = -result.fun

    print(f"Best Learning Rate: {best_learning_rate}")
    print(f"Best Dropout Rate 1: {best_dropout_rate1}")
    print(f"Best Dropout Rate 2: {best_dropout_rate2}")
    print(f"Best Validation Accuracy: {best_val_accuracy}")

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    start_time = _time.time()
    X_train_augmented = augment_dataset(X_train)
    X_val_augmented = augment_dataset(X_val)
    end_time = _time.time()
    print(f"Data augmentation took {end_time - start_time} seconds")

    input_shape = X_train_augmented.shape[1:]
    num_classes = len(np.unique(y))
    model = build_model(
        input_shape, num_classes,
        best_dropout_rate1, best_dropout_rate2, best_learning_rate
    )

    start_time = _time.time()
    model.fit(
        X_train_augmented, y_train,
        epochs=30, batch_size=32,
        validation_data=(X_val_augmented, y_val)
    )
    end_time = _time.time()
    print(f"Model training took {end_time - start_time} seconds")

# -------------------------
# GUI side
# -------------------------
if IS_GUI:
    import psutil
    from PyQt5 import QtCore, QtGui, QtWidgets

    class WebSurferController(QtCore.QObject):
        stateChanged = QtCore.pyqtSignal(dict)
        logEvent = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._proc = None
            self._running = False
            self._thread = None
            self._auto_respawn = False  # OFF until user turns ON

        def start(self):
            self._auto_respawn = True
            if self._running:
                self.logEvent.emit("Web Surfer monitor already running")
                return
            self._running = True
            self._start_monitor()

        def stop(self):
            # Disable auto-respawn
            self._auto_respawn = False
            # Stop monitor loop
            self._running = False
            # Kill the process if alive
            if self._proc and self._proc.poll() is None:
                self.logEvent.emit("Powering OFF Web Surfer...")
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            # Clear process reference immediately
            self._proc = None
            # Emit OFF state instantly so GUI updates immediately
            self.stateChanged.emit({
                "mode": "off",
                "pid": None,
                "cpu": 0.0,
                "threads": 0
            })

        def _spawn(self):
            if not self._auto_respawn:
                return
            if self._proc and self._proc.poll() is None:
                return
            cmd = [sys.executable, os.path.abspath(__file__), "--websurfer"]
            self.logEvent.emit(f"Spawning Web Surfer: {cmd}")
            try:
                self._proc = subprocess.Popen(cmd)
            except Exception as e:
                self.logEvent.emit(f"Failed to spawn Web Surfer: {e}")
                self._proc = None

        def _start_monitor(self):
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self):
            # Start the process immediately
            self._spawn()

            while True:
                # Exit instantly when OFF is pressed
                if not self._running:
                    break

                mode = "off" if not self._auto_respawn else "stopped"
                pid = None
                cpu = 0.0
                threads = 0

                if self._proc:
                    if self._proc.poll() is None:
                        mode = "running"
                        pid = self._proc.pid
                        try:
                            p = psutil.Process(pid)
                            cpu = p.cpu_percent(interval=0.1)
                            threads = p.num_threads()
                        except Exception:
                            pass
                    else:
                        if self._auto_respawn:
                            mode = "restarting..."
                            self.logEvent.emit("Web Surfer crashed or exited, respawning...")
                            self._spawn()
                            time.sleep(0.5)
                        else:
                            mode = "off"

                self.stateChanged.emit({
                    "mode": mode,
                    "pid": pid,
                    "cpu": cpu,
                    "threads": threads
                })

                # Interruptible micro-sleeps for responsiveness
                for _ in range(20):  # 20 × 0.1s = 2 seconds total
                    if not self._running:
                        break
                    time.sleep(0.1)

    class TrainerController(QtCore.QObject):
        stateChanged = QtCore.pyqtSignal(dict)
        logEvent = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._proc = None
            self._running = False
            self._thread = None
            self._auto_respawn = True

        def start(self):
            if self._running:
                self.logEvent.emit("Trainer monitor already running")
                return
            self._running = True
            self._auto_respawn = True
            self._start_monitor()

        def stop(self):
            self._auto_respawn = False
            self._running = False
            if self._proc and self._proc.poll() is None:
                self.logEvent.emit("Stopping Trainer...")
                try:
                    self._proc.terminate()
                except Exception:
                    pass

        def _spawn(self):
            if not self._auto_respawn:
                return
            if self._proc and self._proc.poll() is None:
                return
            cmd = [sys.executable, os.path.abspath(__file__), "--trainer"]
            self.logEvent.emit(f"Spawning Trainer: {cmd}")
            try:
                self._proc = subprocess.Popen(cmd)
            except Exception as e:
                self.logEvent.emit(f"Failed to spawn Trainer: {e}")
                self._proc = None

        def _start_monitor(self):
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self):
            self._spawn()
            while self._running:
                mode = "stopped"
                pid = None
                cpu = 0.0
                threads = 0

                if self._proc:
                    if self._proc.poll() is None:
                        mode = "running"
                        pid = self._proc.pid
                        try:
                            p = psutil.Process(pid)
                            cpu = p.cpu_percent(interval=0.1)
                            threads = p.num_threads()
                        except Exception:
                            pass
                    else:
                        if self._auto_respawn:
                            mode = "restarting..."
                            self.logEvent.emit("Trainer crashed or exited, respawning...")
                            self._spawn()
                            time.sleep(1)
                        else:
                            mode = "stopped"

                state = {
                    "mode": str(mode),
                    "pid": int(pid) if pid is not None else None,
                    "cpu": float(cpu),
                    "threads": int(threads),
                }
                self.stateChanged.emit(state)
                time.sleep(5)

    class PredictionGraph(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._history = []
            self._forecast = None
            self.setMinimumHeight(120)

        def set_data(self, history, forecast):
            safe = []
            if history:
                for v in history:
                    try:
                        fv = float(v)
                        if fv == fv and 0.0 <= fv <= 100.0:
                            safe.append(fv)
                    except Exception:
                        continue
            self._history = safe[-120:]

            try:
                f = float(forecast)
                if f == f:
                    self._forecast = max(0.0, min(100.0, f))
                else:
                    self._forecast = None
            except Exception:
                self._forecast = None

            self.update()

        def paintEvent(self, event):
            try:
                painter = QtGui.QPainter(self)
                painter.setRenderHint(QtGui.QPainter.Antialiasing)

                rect = self.rect().adjusted(8, 8, -8, -8)
                painter.fillRect(rect, QtGui.QColor(20, 20, 20))
                painter.setPen(QtGui.QPen(QtGui.QColor(80, 80, 80), 1))
                painter.drawRect(rect)

                if len(self._history) < 2:
                    painter.setPen(QtGui.QColor(180, 180, 180))
                    painter.drawText(rect, QtCore.Qt.AlignCenter, "Beast load history")
                    return

                vals = self._history
                max_val = max(vals)
                min_val = min(vals)
                if max_val == min_val:
                    max_val += 1.0

                n = len(vals)

                def map_point(i, val):
                    x = rect.left() + (rect.width() * i) / (n - 1)
                    norm = (val - min_val) / (max_val - min_val)
                    y = rect.bottom() - norm * rect.height()
                    return QtCore.QPointF(x, y)

                path = QtGui.QPainterPath()
                path.moveTo(map_point(0, vals[0]))
                for i in range(1, n):
                    path.lineTo(map_point(i, vals[i]))

                painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 0), 2))
                painter.drawPath(path)

                if self._forecast is not None:
                    pt = map_point(n - 1, self._forecast)
                    painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 0), 1, QtCore.Qt.DashLine))
                    painter.drawLine(pt.x(), rect.bottom(), pt.x(), rect.top())
                    painter.setBrush(QtGui.QColor(200, 200, 0))
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1))
                    painter.drawEllipse(pt, 4, 4)

                    painter.setPen(QtGui.QColor(200, 200, 0))
                    painter.drawText(
                        rect.adjusted(4, 4, -4, -4),
                        QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
                        f"Forecast: {self._forecast:.1f}%"
                    )

            except Exception as e:
                print("[Graph] paintEvent error:", e)
                return

    class SystemWorker(QtCore.QObject):
        metricsReady = QtCore.pyqtSignal(dict)
        hardwareReady = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._running = True
            self._fast_interval = 1.0
            self._slow_interval = 10.0
            self._last_slow = 0.0
            self._last_hw = {}
            self._cpu_history = []
            self._history_window = 20.0
            self._beast_history = []
            self._beast_history_window = 60.0
            self._policy = ResourcePolicy()
            self._water = WaterPhysicsPredictor()
            self._last_gpu_free_mb = None
            self._last_movidius = False
            self._last_coral = False
            self._cached_system_threads = 0

        def stop(self):
            self._running = False

        def _compute_beast_metrics(self):
            if not self._cpu_history:
                return None, None, False

            recent = self._cpu_history[-5:]
            beast_avgs = []

            for _, per_core in recent:
                if len(per_core) > 2:
                    beast_slice = per_core[2:]
                else:
                    beast_slice = per_core
                if beast_slice:
                    avg = sum(beast_slice) / len(beast_slice)
                    beast_avgs.append(avg)

            if not beast_avgs:
                return None, None, False

            current = beast_avgs[-1]
            past = beast_avgs[0]
            slope = current - past

            active = False
            if current >= 70.0:
                active = True
            elif current >= 40.0 and slope is not None and slope > 5.0:
                active = True

            return current, slope, active

        @QtCore.pyqtSlot()
        def run(self):
            while self._running:
                start = time.time()
                try:
                    cpu_perc = psutil.cpu_percent(interval=None, percpu=True)
                    ram = psutil.virtual_memory()
                    cpu_logical = psutil.cpu_count(logical=True) or 0
                    cpu_physical = psutil.cpu_count(logical=False) or 0

                    now = time.time()
                    self._cpu_history.append((now, cpu_perc))
                    self._cpu_history = [
                        (t, v) for (t, v) in self._cpu_history
                        if now - t <= self._history_window
                    ]

                    beast_avg, beast_trend, beast_active = self._compute_beast_metrics()

                    if beast_avg is not None:
                        self._beast_history.append((now, beast_avg))
                        self._beast_history = [
                            (t, v) for (t, v) in self._beast_history
                            if now - t <= self._beast_history_window
                        ]
                    else:
                        self._beast_history = []

                    _, beast_forecast = self._water.update(now, beast_avg)

                    if now - self._last_slow >= self._slow_interval:
                        self._last_slow = now
                        try:
                            total_threads = 0
                            for pid in psutil.pids():
                                try:
                                    p = psutil.Process(pid)
                                    total_threads += p.num_threads()
                                except Exception:
                                    continue
                            self._cached_system_threads = total_threads
                        except Exception:
                            pass

                        hw = {}
                        try:
                            npu = detect_npu_devices()
                        except Exception:
                            npu = []
                        try:
                            movidius_detected, movidius_desc = detect_movidius_status()
                        except Exception:
                            movidius_detected, movidius_desc = False, "Movidius detection failed"
                        try:
                            coral_detected, coral_desc = detect_coral_tpu_status()
                        except Exception:
                            coral_detected, coral_desc = False, "Coral / Edge TPU detection failed"
                        try:
                            gpus, free_vram_mb = detect_gpu_devices_and_vram()
                        except Exception:
                            gpus, free_vram_mb = [], None
                        try:
                            usb = detect_usb_devices_summary()
                        except Exception:
                            usb = []

                        hw["npu"] = [str(x) for x in npu]
                        hw["movidius_detected"] = bool(movidius_detected)
                        hw["movidius_desc"] = str(movidius_desc)
                        hw["coral_detected"] = bool(coral_detected)
                        hw["coral_desc"] = str(coral_desc)
                        hw["gpus"] = [str(g) for g in gpus]
                        hw["gpu_free_vram_mb"] = int(free_vram_mb) if free_vram_mb is not None else None
                        hw["usb"] = [str(u) for u in usb]

                        self._last_gpu_free_mb = hw["gpu_free_vram_mb"]
                        self._last_movidius = hw["movidius_detected"]
                        self._last_coral = hw["coral_detected"]

                        if hw != self._last_hw:
                            self._last_hw = hw
                            self.hardwareReady.emit(hw)

                    engine_choice = self._policy.choose_engine(
                        now,
                        beast_forecast,
                        self._last_gpu_free_mb,
                        self._last_movidius,
                        self._last_coral
                    )

                    history_vals = [float(v) for (_, v) in self._beast_history]

                    metrics = {
                        "cpu_perc": [float(x) for x in cpu_perc],
                        "ram_used": int(ram.used),
                        "ram_total": int(ram.total),
                        "ram_percent": float(ram.percent),
                        "cpu_logical": int(cpu_logical),
                        "cpu_physical": int(cpu_physical),
                        "system_threads": int(self._cached_system_threads),
                        "beast_avg": float(beast_avg) if beast_avg is not None else None,
                        "beast_trend": float(beast_trend) if beast_trend is not None else None,
                        "beast_active": bool(beast_active),
                        "beast_forecast": float(beast_forecast) if beast_forecast is not None else None,
                        "beast_history": history_vals,
                        "engine_choice": str(engine_choice),
                    }
                    self.metricsReady.emit(metrics)

                except Exception as e:
                    print("[SystemWorker] loop error:", e)

                elapsed = time.time() - start
                sleep_time = max(0.1, self._fast_interval - elapsed)
                time.sleep(sleep_time)

    class SystemMonitor(QtCore.QObject):
        metricsChanged = QtCore.pyqtSignal(dict)
        hardwareChanged = QtCore.pyqtSignal(dict)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._thread = QtCore.QThread()
            self._worker = SystemWorker()
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.metricsReady.connect(self.metricsChanged)
            self._worker.hardwareReady.connect(self.hardwareChanged)
            self._thread.start()

        def stop(self):
            self._worker.stop()
            self._thread.quit()
            self._thread.wait()

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, websurfer: WebSurferController, trainer: TrainerController, monitor: SystemMonitor, parent=None):
            super().__init__(parent)
            self.websurfer = websurfer
            self.trainer = trainer
            self.monitor = monitor
            self._last_ws_toggle = 0.0

            self.setWindowTitle("Beast Mode – Event Horizon Nerve Center")
            self.resize(1350, 780)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)

            main_layout = QtWidgets.QHBoxLayout(central)

            left_panel = QtWidgets.QVBoxLayout()
            main_layout.addLayout(left_panel, 2)

            right_panel = QtWidgets.QVBoxLayout()
            main_layout.addLayout(right_panel, 1)

            top_row = QtWidgets.QHBoxLayout()
            self.status_label = QtWidgets.QLabel("Status: Idle")
            font = self.status_label.font()
            font.setPointSize(14)
            font.setBold(True)
            self.status_label.setFont(font)

            self.beast_led_label = QtWidgets.QLabel("Beast Mode: IDLE")
            beast_font = self.beast_led_label.font()
            beast_font.setPointSize(12)
            beast_font.setBold(True)
            self.beast_led_label.setFont(beast_font)
            self._set_beast_led(False)

            self.engine_label = QtWidgets.QLabel("Engine: CPU")
            eng_font = self.engine_label.font()
            eng_font.setPointSize(10)
            self.engine_label.setFont(eng_font)

            top_row.addWidget(self.status_label)
            top_row.addStretch(1)
            top_row.addWidget(self.beast_led_label)
            top_row.addWidget(self.engine_label)

            left_panel.addLayout(top_row)

            self.cpu_group = QtWidgets.QGroupBox("CPU Cores")
            self.cpu_layout = QtWidgets.QVBoxLayout(self.cpu_group)
            left_panel.addWidget(self.cpu_group)

            self.cpu_bars = []
            cpu_count = psutil.cpu_count(logical=True)
            for i in range(cpu_count):
                bar = QtWidgets.QProgressBar()
                bar.setRange(0, 100)
                bar.setFormat(f"Core {i}: %p%")
                self.cpu_layout.addWidget(bar)
                self.cpu_bars.append(bar)

            self.ram_label = QtWidgets.QLabel("RAM: 0 / 0 (0%)")
            left_panel.addWidget(self.ram_label)

            ws_group = QtWidgets.QGroupBox("Web Surfer")
            ws_layout = QtWidgets.QGridLayout(ws_group)

            self.ws_status_label = QtWidgets.QLabel("Mode: off")
            self.ws_pid_label = QtWidgets.QLabel("PID: -")
            self.ws_cpu_label = QtWidgets.QLabel("CPU: 0.0%")
            self.ws_threads_label = QtWidgets.QLabel("Threads: 0")

            self.ws_toggle_btn = QtWidgets.QPushButton("Turn ON Web Surfer")

            ws_layout.addWidget(self.ws_status_label, 0, 0, 1, 2)
            ws_layout.addWidget(self.ws_pid_label, 1, 0, 1, 2)
            ws_layout.addWidget(self.ws_cpu_label, 2, 0)
            ws_layout.addWidget(self.ws_threads_label, 2, 1)
            ws_layout.addWidget(self.ws_toggle_btn, 3, 0, 1, 2)

            left_panel.addWidget(ws_group)

            tr_group = QtWidgets.QGroupBox("Beast Mode Trainer")
            tr_layout = QtWidgets.QGridLayout(tr_group)

            self.tr_status_label = QtWidgets.QLabel("Mode: stopped")
            self.tr_pid_label = QtWidgets.QLabel("PID: -")
            self.tr_cpu_label = QtWidgets.QLabel("CPU: 0.0%")
            self.tr_threads_label = QtWidgets.QLabel("Threads: 0")

            self.tr_start_btn = QtWidgets.QPushButton("Start Trainer")
            self.tr_stop_btn = QtWidgets.QPushButton("Stop Trainer")

            tr_layout.addWidget(self.tr_status_label, 0, 0, 1, 2)
            tr_layout.addWidget(self.tr_pid_label, 1, 0, 1, 2)
            tr_layout.addWidget(self.tr_cpu_label, 2, 0)
            tr_layout.addWidget(self.tr_threads_label, 2, 1)
            tr_layout.addWidget(self.tr_start_btn, 3, 0)
            tr_layout.addWidget(self.tr_stop_btn, 3, 1)

            left_panel.addWidget(tr_group)

            left_panel.addStretch(1)

            stats_group = QtWidgets.QGroupBox("System CPU & Thread Stats")
            stats_layout = QtWidgets.QVBoxLayout(stats_group)
            self.sys_cpu_logical_label = QtWidgets.QLabel("Logical CPUs: -")
            self.sys_cpu_physical_label = QtWidgets.QLabel("Physical CPUs: -")
            self.sys_threads_label = QtWidgets.QLabel("System Threads: -")
            stats_layout.addWidget(self.sys_cpu_logical_label)
            stats_layout.addWidget(self.sys_cpu_physical_label)
            stats_layout.addWidget(self.sys_threads_label)
            right_panel.addWidget(stats_group)

            proc_stats_group = QtWidgets.QGroupBox("Process CPU & Threads")
            proc_layout = QtWidgets.QVBoxLayout(proc_stats_group)
            self.ws_proc_label = QtWidgets.QLabel("Web Surfer: CPU 0.0%, Threads 0")
            self.tr_proc_label = QtWidgets.QLabel("Trainer: CPU 0.0%, Threads 0")
            proc_layout.addWidget(self.ws_proc_label)
            proc_layout.addWidget(self.tr_proc_label)
            right_panel.addWidget(proc_stats_group)

            graph_group = QtWidgets.QGroupBox("Beast Load – Prediction Horizon")
            graph_layout = QtWidgets.QVBoxLayout(graph_group)
            self.graph_widget = PredictionGraph()
            graph_layout.addWidget(self.graph_widget)
            right_panel.addWidget(graph_group)

            hw_group = QtWidgets.QGroupBox("Hardware Accelerators & USB Devices")
            hw_layout = QtWidgets.QVBoxLayout(hw_group)

            self.npu_label = QtWidgets.QLabel("NPU: none detected")
            self.movidius_label = QtWidgets.QLabel("Movidius: status unknown")
            self.coral_label = QtWidgets.QLabel("Coral / Edge TPU: status unknown")
            self.gpu_label = QtWidgets.QLabel("GPU: none detected")
            self.gpu_vram_label = QtWidgets.QLabel("GPU free VRAM: unknown")
            hw_layout.addWidget(self.npu_label)
            hw_layout.addWidget(self.movidius_label)
            hw_layout.addWidget(self.coral_label)
            hw_layout.addWidget(self.gpu_label)
            hw_layout.addWidget(self.gpu_vram_label)

            self.usb_list = QtWidgets.QListWidget()
            self.usb_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            hw_layout.addWidget(QtWidgets.QLabel("USB devices (summary):"))
            hw_layout.addWidget(self.usb_list)

            right_panel.addWidget(hw_group, 2)

            self.ws_toggle_btn.clicked.connect(self.toggle_websurfer)
            self.tr_start_btn.clicked.connect(self.trainer.start)
            self.tr_stop_btn.clicked.connect(self.trainer.stop)

            self.websurfer.stateChanged.connect(self.on_websurfer_state)
            self.trainer.stateChanged.connect(self.on_trainer_state)
            self.monitor.metricsChanged.connect(self.on_metrics)
            self.monitor.hardwareChanged.connect(self.on_hardware)

        def _set_beast_led(self, active: bool):
            if active:
                self.beast_led_label.setText("Beast Mode: ACTIVE")
                self.beast_led_label.setStyleSheet("color: white; background-color: green; padding: 4px; border-radius: 4px;")
            else:
                self.beast_led_label.setText("Beast Mode: IDLE")
                self.beast_led_label.setStyleSheet("color: black; background-color: lightgray; padding: 4px; border-radius: 4px;")

        def update_global_status(self):
            ws_mode = self.ws_status_label.text()
            tr_mode = self.tr_status_label.text()

            ws_running = ws_mode.startswith("Mode: running")
            tr_running = tr_mode.startswith("Mode: running")

            if ws_running and tr_running:
                self.status_label.setText("Status: System fully engaged")
            elif ws_running:
                self.status_label.setText("Status: Web Surfer active")
            elif tr_running:
                self.status_label.setText("Status: Beast Mode active")
            else:
                self.status_label.setText("Status: Idle")

        def toggle_websurfer(self):
            now = time.time()
            if now - self._last_ws_toggle < 1.0:
                return
            self._last_ws_toggle = now

            mode_text = self.ws_status_label.text().lower()
            if "running" in mode_text or "restarting" in mode_text:
                self.websurfer.stop()
                self.ws_toggle_btn.setText("Turn ON Web Surfer")
                QtWidgets.QToolTip.showText(self.mapToGlobal(self.rect().center()), "Web Surfer deactivated")
            else:
                self.websurfer.start()
                self.ws_toggle_btn.setText("Turn OFF Web Surfer")
                QtWidgets.QToolTip.showText(self.mapToGlobal(self.rect().center()), "Web Surfer activated")

        @QtCore.pyqtSlot(dict)
        def on_websurfer_state(self, state: dict):
            try:
                mode = str(state.get("mode", "off"))
                pid = state.get("pid")
                cpu = float(state.get("cpu", 0.0))
                threads = int(state.get("threads", 0))

                self.ws_status_label.setText(f"Mode: {mode}")
                self.ws_pid_label.setText(f"PID: {pid if pid else '-'}")
                self.ws_cpu_label.setText(f"CPU: {cpu:.1f}%")
                self.ws_threads_label.setText(f"Threads: {threads}")
                self.ws_proc_label.setText(f"Web Surfer: CPU {cpu:.1f}%, Threads {threads}")

                if mode == "running" or mode == "restarting...":
                    self.ws_toggle_btn.setText("Turn OFF Web Surfer")
                else:
                    self.ws_toggle_btn.setText("Turn ON Web Surfer")

                self.update_global_status()
            except Exception as e:
                print("[GUI] on_websurfer_state error:", e)

        @QtCore.pyqtSlot(dict)
        def on_trainer_state(self, state: dict):
            try:
                mode = str(state.get("mode", "stopped"))
                pid = state.get("pid")
                cpu = float(state.get("cpu", 0.0))
                threads = int(state.get("threads", 0))

                self.tr_status_label.setText(f"Mode: {mode}")
                self.tr_pid_label.setText(f"PID: {pid if pid else '-'}")
                self.tr_cpu_label.setText(f"CPU: {cpu:.1f}%")
                self.tr_threads_label.setText(f"Threads: {threads}")
                self.tr_proc_label.setText(f"Trainer: CPU {cpu:.1f}%, Threads {threads}")

                self.update_global_status()
            except Exception as e:
                print("[GUI] on_trainer_state error:", e)

        @QtCore.pyqtSlot(dict)
        def on_metrics(self, metrics: dict):
            try:
                cpu_perc = metrics.get("cpu_perc", [])
                ram_used = metrics.get("ram_used", 0)
                ram_total = metrics.get("ram_total", 1)
                ram_percent = metrics.get("ram_percent", 0.0)
                cpu_logical = metrics.get("cpu_logical", 0)
                cpu_physical = metrics.get("cpu_physical", 0)
                system_threads = metrics.get("system_threads", 0)
                beast_active = metrics.get("beast_active", False)
                engine_choice = metrics.get("engine_choice", "CPU")
                beast_history = metrics.get("beast_history", [])
                beast_forecast = metrics.get("beast_forecast", None)

                for i, bar in enumerate(self.cpu_bars):
                    if i < len(cpu_perc):
                        try:
                            val = int(cpu_perc[i])
                        except Exception:
                            val = 0
                        bar.setValue(max(0, min(100, val)))
                    else:
                        bar.setValue(0)

                self._set_beast_led(bool(beast_active))
                self.engine_label.setText(f"Engine: {engine_choice}")

                self.graph_widget.set_data(beast_history, beast_forecast)

                def fmt_bytes(b):
                    b = float(b)
                    for unit in ["B", "KB", "MB", "GB", "TB"]:
                        if b < 1024:
                            return f"{b:.1f} {unit}"
                        b /= 1024
                    return f"{b:.1f} PB"

                self.ram_label.setText(
                    f"RAM: {fmt_bytes(ram_used)} / {fmt_bytes(ram_total)} ({ram_percent:.1f}%)"
                )
                self.sys_cpu_logical_label.setText(f"Logical CPUs: {cpu_logical}")
                self.sys_cpu_physical_label.setText(f"Physical CPUs: {cpu_physical}")
                self.sys_threads_label.setText(f"System Threads: {system_threads}")
            except Exception as e:
                print("[GUI] on_metrics error:", e)

        @QtCore.pyqtSlot(dict)
        def on_hardware(self, hw: dict):
            try:
                npu = hw.get("npu", [])
                movidius_detected = bool(hw.get("movidius_detected", False))
                movidius_desc = str(hw.get("movidius_desc", "Movidius: status unknown"))
                coral_detected = bool(hw.get("coral_detected", False))
                coral_desc = str(hw.get("coral_desc", "Coral / Edge TPU: status unknown"))
                gpus = hw.get("gpus", [])
                free_vram_mb = hw.get("gpu_free_vram_mb", None)
                usb = hw.get("usb", [])

                if npu:
                    self.npu_label.setText("NPU: " + "; ".join(str(x) for x in npu))
                else:
                    self.npu_label.setText("NPU: none detected")

                if movidius_detected:
                    self.movidius_label.setText("Movidius: ACTIVE – " + movidius_desc)
                else:
                    self.movidius_label.setText("Movidius: " + movidius_desc)

                if coral_detected:
                    self.coral_label.setText("Coral / Edge TPU: ACTIVE – " + coral_desc)
                else:
                    self.coral_label.setText("Coral / Edge TPU: " + coral_desc)

                if gpus:
                    self.gpu_label.setText("GPU: " + "; ".join(str(g) for g in gpus))
                else:
                    self.gpu_label.setText("GPU: none detected")

                if free_vram_mb is not None:
                    self.gpu_vram_label.setText(f"GPU free VRAM: {free_vram_mb} MB")
                else:
                    self.gpu_vram_label.setText("GPU free VRAM: unknown")

                self.usb_list.clear()
                if usb:
                    for line in usb:
                        self.usb_list.addItem(str(line))
                else:
                    self.usb_list.addItem("No USB summary available")
            except Exception as e:
                print("[GUI] on_hardware error:", e)

    def main_gui():
        try:
            p = psutil.Process()
            total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            if total_cores >= 2:
                bg_cores = [0, 1]
            else:
                bg_cores = [0]
            p.cpu_affinity(bg_cores)
            print(f"[GUI] Affinity set to background cores: {bg_cores}")
        except Exception as e:
            print(f"[GUI] Failed to set CPU affinity: {e}")

        app = QtWidgets.QApplication(sys.argv)

        websurfer = WebSurferController()
        trainer = TrainerController()
        monitor = SystemMonitor()

        window = MainWindow(websurfer, trainer, monitor)
        window.show()

        ret = app.exec_()
        websurfer.stop()
        trainer.stop()
        monitor.stop()
        sys.exit(ret)

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    if IS_WEB_SURFER:
        run_websurfer()
    elif IS_TRAINER:
        trainer_main()
    else:
        main_gui()

