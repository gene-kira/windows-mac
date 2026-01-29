import sys
import subprocess
import importlib
import time
import os
import json
import math
import ctypes
from ctypes import wintypes

# =========================
# REQUIRED LIBS
# =========================

REQUIRED_LIBS = [
    "PySide6",
    "psutil",
    "requests",
    "mss",
    "Pillow",
    "pytesseract",
]


def ensure_libraries():
    for lib in REQUIRED_LIBS:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[AUTO-LOADER] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])


# =========================
# AI ENGINE MANAGER
# =========================

class AIEngineManager:
    def __init__(self):
        import requests
        self.requests = requests
        self.engines = []
        self.preferred_engine_name = None
        self.load_builtin_engines()
        self.load_local_plugins()
        self.load_cloud_plugins()

    def load_builtin_engines(self):
        self.engines.append({
            "name": "WindowsCopilot",
            "type": "system",
            "priority": 1,
            "available": self.detect_windows_copilot()
        })

    def detect_windows_copilot(self):
        return os.name == "nt"

    def load_local_plugins(self):
        folder = os.path.join("plugins", "local")
        if not os.path.exists(folder):
            return
        for file in os.listdir(folder):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                        config = json.load(f)
                        config["available"] = True
                        config.setdefault("priority", 3)
                        config["type"] = "local"
                        self.engines.append(config)
                except Exception as e:
                    print(f"[AIEngineManager] Failed to load local plugin {file}: {e}")

    def load_cloud_plugins(self):
        folder = os.path.join("plugins", "cloud")
        if not os.path.exists(folder):
            return
        for file in os.listdir(folder):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                        config = json.load(f)
                        config["available"] = True
                        config.setdefault("priority", 4)
                        config["type"] = "cloud"
                        self.engines.append(config)
                except Exception as e:
                    print(f"[AIEngineManager] Failed to load cloud plugin {file}: {e}")

    def set_preferred_engine(self, name: str):
        self.preferred_engine_name = name

    def get_engine_by_name(self, name: str):
        for e in self.engines:
            if e.get("name") == name:
                return e
        return None

    def get_best_engine(self):
        if self.preferred_engine_name:
            eng = self.get_engine_by_name(self.preferred_engine_name)
            if eng and eng.get("available"):
                return eng

        available = [e for e in self.engines if e.get("available")]
        if not available:
            return None
        return sorted(available, key=lambda x: x["priority"])[0]

    def ask(self, prompt: str) -> str:
        engine = self.get_best_engine()
        if not engine:
            return "[AI] No engines available."

        if engine["name"] == "WindowsCopilot":
            return self.ask_windows_copilot(prompt)
        if engine["type"] == "local":
            return self.ask_local_engine(engine, prompt)
        if engine["type"] == "cloud":
            return self.ask_cloud_engine(engine, prompt)

        return "[AI] Unsupported engine type."

    def ask_windows_copilot(self, prompt: str) -> str:
        # Stub – replace with real integration if available
        return f"{prompt}"

    def ask_local_engine(self, engine, prompt: str) -> str:
        try:
            r = self.requests.post(
                engine["endpoint"],
                json={"prompt": prompt},
                timeout=30
            )
            data = r.json()
            return (
                data.get("response")
                or data.get("text")
                or "[Local AI] No response field."
            )
        except Exception as e:
            return f"[Local AI Error] {e}"

    def ask_cloud_engine(self, engine, prompt: str) -> str:
        try:
            headers = {}
            if "api_key" in engine:
                headers["Authorization"] = f"Bearer {engine['api_key']}"
            r = self.requests.post(
                engine["endpoint"],
                json={"prompt": prompt},
                headers=headers,
                timeout=30
            )
            data = r.json()
            return (
                data.get("response")
                or data.get("text")
                or "[Cloud AI] No response field."
            )
        except Exception as e:
            return f"[Cloud AI Error] {e}"


# =========================
# MAIN OVERLAY
# =========================

def main():
    ensure_libraries()

    import psutil
    import mss
    from PIL import Image
    import pytesseract

    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QSlider,
        QLabel,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
    )
    from PySide6.QtGui import QPainter, QColor, QFont, QMouseEvent
    from PySide6.QtCore import Qt, QRect, QTimer, QPoint

    user32 = ctypes.windll.user32

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class SystemMonitor:
        def __init__(self):
            self.last_net_bytes = None
            self.last_net_time = None
            self.cpu_history = []
            self.ram_history = []
            self.max_history = 30

        def get_power_info(self):
            try:
                batt = psutil.sensors_battery()
                if batt is None:
                    return "UNKNOWN", None, None
                power_source = "AC" if batt.power_plugged else "DC"
                percent = int(batt.percent)
                secs_left = batt.secsleft if batt.secsleft not in (
                    psutil.POWER_TIME_UNLIMITED,
                    psutil.POWER_TIME_UNKNOWN
                ) else None
                return power_source, percent, secs_left
            except Exception:
                return "UNKNOWN", None, None

        def get_cpu_usage(self):
            try:
                val = int(psutil.cpu_percent(interval=None))
                self.cpu_history.append(val)
                if len(self.cpu_history) > self.max_history:
                    self.cpu_history.pop(0)
                return val
            except Exception:
                return None

        def get_ram_usage(self):
            try:
                mem = psutil.virtual_memory()
                val = int(mem.percent)
                self.ram_history.append(val)
                if len(self.ram_history) > self.max_history:
                    self.ram_history.pop(0)
                return val
            except Exception:
                return None

        def get_net_status(self):
            try:
                net = psutil.net_io_counters()
                now = time.time()
                if self.last_net_bytes is None:
                    self.last_net_bytes = net.bytes_sent + net.bytes_recv
                    self.last_net_time = now
                    return "INIT"

                delta_bytes = (net.bytes_sent + net.bytes_recv) - self.last_net_bytes
                delta_time = (now - self.last_net_time) if self.last_net_time else 1.0

                self.last_net_bytes = net.bytes_sent + net.bytes_recv
                self.last_net_time = now

                rate = delta_bytes / max(delta_time, 0.001)
                if rate > 50_000:
                    return "ACTIVE"
                elif rate > 5_000:
                    return "LIGHT"
                else:
                    return "IDLE"
            except Exception:
                return "UNKNOWN"

    class OverlayWindow(QWidget):
        def __init__(self):
            super().__init__()

            # Settings file path
            self.settings_path = os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])),
                "visor_settings.json"
            )
            self.settings = {}

            # HUD geometry
            self.base_bar_height = 40
            self.close_button_size = 32
            self.close_button_margin = 10
            self.close_button_rect = QRect()

            # Horizontal control
            self.hud_width_pct = 100  # 20–100 via slider
            self.hud_offset_x = 0     # draggable
            self.dragging_hud = False
            self.drag_offset_x = 0

            # Modes
            self.game_mode_auto = False
            self.config_mode = False

            # Game intelligence
            self.current_game_exe = None
            self.game_profiles = {
                "eldenring.exe": {"hud_opacity": 220, "show_net": False},
                "cs2.exe": {"hud_opacity": 200, "show_net": True},
            }

            # Default slider state (overridden by settings)
            self.hud_opacity = 170
            self.game_aggression = 70
            self.cursor_scale = 0.35
            self.panel_scale = 0.5
            self.glow_intensity = 60
            self.beast_threshold = 75
            self.hud_thickness_pct = 0

            # Load settings
            self.load_settings()

            # Translation / explanation
            self.translation_text = ""
            self.explanation_text = ""
            self.show_translation_panel = False
            self.show_explanation_panel = False
            self.ocr_error = None

            # AI + hint feed
            self.ai = AIEngineManager()
            self.hint_lines = []  # newest last

            # Cursor + click intelligence state
            self.last_cursor_pos = None
            self.last_cursor_comment_time = 0.0
            self.cursor_move_threshold = 20
            self.cursor_comment_cooldown = 1.0
            self.last_lbutton_down = False

            # Input intelligence state
            self.last_input_sample_time = time.time()
            self.last_hint_time = 0.0
            self.hint_cooldown = 2.0  # seconds
            self.idle_threshold = 1.0  # seconds
            self.last_any_input_time = time.time()

            self.last_w_state = False
            self.last_a_state = False
            self.last_s_state = False
            self.last_d_state = False
            self.last_x_state = False
            self.last_shift_state = False
            self.last_ctrl_state = False
            self.last_space_state = False
            self.last_rmb_state = False
            self.last_mmb_state = False
            self.last_x1_state = False
            self.last_x2_state = False

            self.last_click_times = []
            self.click_window = 2.0  # seconds for rhythm
            self.sprint_start_time = None
            self.crouch_start_time = None
            self.jump_times = []

            # Combat / danger tracking
            self.last_combat_input_time = 0.0
            self.last_ocr_danger_time = 0.0

            # Ultra aim engine state
            self.last_mouse_pos = None
            self.last_mouse_time = None
            self.aim_vel_history = []      # list of speeds
            self.aim_dir_history = []      # list of (dx, dy) normalized
            self.aim_history_window = 0.8  # seconds
            self.mouse_sample_interval_ms = 33  # adaptive
            self.mouse_in_combat = False

            # System
            self.monitor = SystemMonitor()
            self.hud_text = "Initializing HUD..."
            self.event_log = []

            # Window setup
            self.setWindowFlags(
                Qt.FramelessWindowHint |
                Qt.WindowStaysOnTopHint |
                Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

            # Screen geometry
            screen = QApplication.primaryScreen()
            self.screen_geom = screen.geometry()

            # Initial window geometry
            self.update_window_geometry()

            # Timers
            self.update_timer = QTimer(self)
            self.update_timer.timeout.connect(self.update_hud_data)
            self.update_timer.start(1000)

            self.game_detect_timer = QTimer(self)
            self.game_detect_timer.timeout.connect(self.detect_game_mode_aggressive)
            self.game_detect_timer.start(1500)

            # AI hint loop (game OCR)
            self.hint_timer = QTimer(self)
            self.hint_timer.timeout.connect(self.update_ai_hint_loop)
            self.hint_timer.start(2000)

            # Cursor commentary timer
            self.cursor_timer = QTimer(self)
            self.cursor_timer.timeout.connect(self.update_cursor_commentary)
            self.cursor_timer.start(120)

            # Click intelligence timer (LMB)
            self.click_timer = QTimer(self)
            self.click_timer.timeout.connect(self.update_click_intel)
            self.click_timer.start(60)

            # Input intelligence timer (WASD + mouse buttons + sprint/crouch/jump)
            self.input_timer = QTimer(self)
            self.input_timer.timeout.connect(self.update_input_intel)
            self.input_timer.start(80)

            # Ultra aim engine timer (adaptive)
            self.mouse_timer = QTimer(self)
            self.mouse_timer.timeout.connect(self.update_mouse_aim_intel)
            self.mouse_timer.start(self.mouse_sample_interval_ms)

            self.update_hud_data()

            # OCR helpers
            self.mss = mss.mss()
            self.pytesseract = pytesseract
            self.Image = Image

            # Floating slider panel
            self.slider_panel = SliderPanel(self)
            geom = self.screen_geom
            sx, sy, sw, sh = geom.x(), geom.y(), geom.width(), geom.height()
            panel_w, panel_h = self.slider_panel.width(), self.slider_panel.height()
            px = sx + (sw - panel_w) // 2
            py = sy + (sh - panel_h) // 2
            self.slider_panel.move(px, py)
            self.slider_panel.show()

        # ---------- SETTINGS PERSISTENCE ----------

        def load_settings(self):
            try:
                if not os.path.exists(self.settings_path):
                    return
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
            except Exception:
                self.settings = {}

            self.hud_opacity = int(self.settings.get("hud_opacity", self.hud_opacity))
            self.game_aggression = int(self.settings.get("game_aggression", self.game_aggression))
            self.cursor_scale = float(self.settings.get("cursor_scale", self.cursor_scale))
            self.panel_scale = float(self.settings.get("panel_scale", self.panel_scale))
            self.glow_intensity = int(self.settings.get("glow_intensity", self.glow_intensity))
            self.beast_threshold = int(self.settings.get("beast_threshold", self.beast_threshold))
            self.hud_thickness_pct = int(self.settings.get("hud_thickness_pct", self.hud_thickness_pct))
            self.hud_width_pct = int(self.settings.get("hud_width_pct", self.hud_width_pct))

        def save_settings(self):
            try:
                data = {
                    "hud_opacity": int(self.hud_opacity),
                    "game_aggression": int(self.game_aggression),
                    "cursor_scale": float(self.cursor_scale),
                    "panel_scale": float(self.panel_scale),
                    "glow_intensity": int(self.glow_intensity),
                    "beast_threshold": int(self.beast_threshold),
                    "hud_thickness_pct": int(self.hud_thickness_pct),
                    "hud_width_pct": int(self.hud_width_pct),
                }
                with open(self.settings_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print("Error saving settings:", e)

        # ---------- WINDOW GEOMETRY ----------

        def update_window_geometry(self):
            geom = self.screen_geom
            sw, sh = geom.width(), geom.height()

            pct = max(20, min(100, self.hud_width_pct))
            hud_w = int(sw * (pct / 100.0))
            hud_w = max(200, hud_w)

            max_x = sw - hud_w
            self.hud_offset_x = max(0, min(self.hud_offset_x, max_x))

            self.setGeometry(self.hud_offset_x, 0, hud_w, sh)
            self.update_close_button_rect()

        def update_close_button_rect(self):
            try:
                w = self.width()
                size = self.close_button_size
                margin = self.close_button_margin
                self.close_button_rect = QRect(
                    w - size - margin,
                    margin,
                    size,
                    size
                )
            except Exception as e:
                print("update_close_button_rect error:", e)

        def resizeEvent(self, event):
            self.update_close_button_rect()
            super().resizeEvent(event)

        # ---------- LOGGING ----------

        def log_event(self, msg: str):
            ts = time.strftime("%H:%M:%S")
            entry = f"[{ts}] {msg}"
            self.event_log.append(entry)
            if len(self.event_log) > 100:
                self.event_log.pop(0)

        # ---------- AGGRESSIVE GAME DETECTION ----------

        def detect_game_mode_aggressive(self):
            try:
                hwnd_foreground = user32.GetForegroundWindow()
                if not hwnd_foreground:
                    self.game_mode_auto = False
                    self.current_game_exe = None
                    self.update_hud_data()
                    return

                hwnd_self = int(self.winId())
                if hwnd_foreground == hwnd_self:
                    self.game_mode_auto = False
                    self.current_game_exe = None
                    self.update_hud_data()
                    return

                rect = wintypes.RECT()
                user32.GetWindowRect(hwnd_foreground, ctypes.byref(rect))
                win_w = rect.right - rect.left
                win_h = rect.bottom - rect.top

                screen = QApplication.primaryScreen()
                screen_geom = screen.geometry()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()

                tol = 80
                fullscreen_like = (
                    abs(win_w - screen_w) <= tol and
                    abs(win_h - screen_h) <= tol
                )

                pid = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(hwnd_foreground, ctypes.byref(pid))
                fg_pid = pid.value

                ignore_names = {
                    "explorer.exe",
                    "dwm.exe",
                    "shellexperiencehost.exe",
                    "searchui.exe",
                    "startmenuexperiencehost.exe",
                    "applicationframehost.exe",
                }

                aggr = self.game_aggression / 100.0
                cpu_threshold = 5.0 - 3.0 * aggr
                ram_threshold = (700 - 400 * aggr) * 1024 * 1024
                score_threshold = 3 - int(aggr * 1)

                candidate_exe = None
                candidate_score = 0

                for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                    try:
                        name = (p.info["name"] or "").lower()
                        if not name:
                            continue
                        if name in ignore_names:
                            continue

                        cpu = p.info["cpu_percent"] or 0.0
                        mem = p.info["memory_info"].rss if p.info["memory_info"] else 0

                        heavy = (cpu > cpu_threshold or mem > ram_threshold)

                        score = 0
                        if heavy:
                            score += 2
                        if p.info["pid"] == fg_pid:
                            score += 3
                        if fullscreen_like and p.info["pid"] == fg_pid:
                            score += 3

                        gameish_tokens = ["game", "steam", "epic", "unity", "unreal", "launcher"]
                        if any(tok in name for tok in gameish_tokens):
                            score += 1

                        if score > candidate_score:
                            candidate_score = score
                            candidate_exe = name
                    except Exception:
                        continue

                if candidate_exe and candidate_score >= score_threshold:
                    if candidate_exe != self.current_game_exe:
                        self.log_event(f"[AGGRESSIVE] Game detected: {candidate_exe} (score={candidate_score})")
                    self.current_game_exe = candidate_exe
                    self.game_mode_auto = True
                else:
                    if self.game_mode_auto or self.current_game_exe:
                        self.log_event("[AGGRESSIVE] No game detected; reverting to desktop.")
                    self.game_mode_auto = False
                    self.current_game_exe = None

            except Exception as e:
                print("detect_game_mode_aggressive error:", e)
                self.game_mode_auto = False
                self.current_game_exe = None

            self.update_hud_data()

        def is_game_mode(self):
            return self.game_mode_auto

        def get_current_profile(self):
            if not self.is_game_mode() or not self.current_game_exe:
                return None
            return self.game_profiles.get(self.current_game_exe)

        # ---------- HUD DATA ----------

        def update_hud_data(self):
            power_source, batt_percent, _ = self.monitor.get_power_info()
            cpu = self.monitor.get_cpu_usage()
            ram = self.monitor.get_ram_usage()
            net = self.monitor.get_net_status()

            parts = []

            parts.append("[GAME]" if self.is_game_mode() else "[DESKTOP]")

            if power_source != "UNKNOWN":
                if batt_percent is not None:
                    parts.append(f"{power_source} {batt_percent}%")
                else:
                    parts.append(f"{power_source}")
            else:
                parts.append("PWR ?")

            parts.append(f"CPU {cpu if cpu is not None else '--'}%")
            parts.append(f"RAM {ram if ram is not None else '--'}%")

            profile = self.get_current_profile()
            show_net = True
            if profile and "show_net" in profile:
                show_net = profile["show_net"]

            if show_net:
                parts.append(f"NET {net}")

            self.hud_text = "  •  ".join(parts)
            self.update()

        # ---------- CAPTURE HELPERS ----------

        def capture_region_around_cursor(self):
            try:
                pt = POINT()
                if not user32.GetCursorPos(ctypes.byref(pt)):
                    self.ocr_error = "Failed to get cursor position."
                    return None

                screen = QApplication.primaryScreen()
                geom = screen.geometry()
                sw, sh = geom.width(), geom.height()

                scale = self.cursor_scale
                region_w = int(sw * scale)
                region_h = int(sh * (scale * 0.6))

                x = pt.x - region_w // 2
                y = pt.y - region_h // 2

                x = max(0, min(x, sw - region_w))
                y = max(0, min(y, sh - region_h))

                monitor = {
                    "top": y,
                    "left": x,
                    "width": region_w,
                    "height": region_h,
                }
                img = self.mss.grab(monitor)
                img_pil = self.Image.frombytes("RGB", img.size, img.rgb)
                return img_pil
            except Exception as e:
                self.ocr_error = f"OCR capture error: {e}"
                return None

        def capture_game_region_for_hint(self):
            try:
                screen = QApplication.primaryScreen()
                geom = screen.geometry()
                sw, sh = geom.width(), geom.height()

                region_w = int(sw * 0.5)
                region_h = int(sh * 0.4)
                x = (sw - region_w) // 2
                y = (sh - region_h) // 2

                monitor = {
                    "top": y,
                    "left": x,
                    "width": region_w,
                    "height": region_h,
                }
                img = self.mss.grab(monitor)
                img_pil = self.Image.frombytes("RGB", img.size, img.rgb)
                return img_pil
            except Exception as e:
                self.ocr_error = f"Hint capture error: {e}"
                return None

        # ---------- TRANSLATE / EXPLAIN ----------

        def translate_under_cursor(self):
            self.log_event("Translate under cursor.")
            img = self.capture_region_around_cursor()
            if img is None:
                self.translation_text = self.ocr_error or "OCR capture failed."
                self.show_translation_panel = True
                self.update()
                return

            try:
                ocr_text = self.pytesseract.image_to_string(img).strip()
                if not ocr_text:
                    self.translation_text = "No text detected under cursor."
                    self.show_translation_panel = True
                    self.update()
                    return
                prompt = f"OCR:\n{ocr_text}\n\nTranslate to English, under 8 words."
                self.translation_text = self.ai.ask(prompt)
            except Exception as e:
                self.translation_text = f"OCR/AI error: {e}"

            self.show_translation_panel = True
            self.update()

        def explain_under_cursor(self):
            self.log_event("Explain under cursor.")
            img = self.capture_region_around_cursor()
            if img is None:
                self.explanation_text = self.ocr_error or "OCR capture failed."
                self.show_explanation_panel = True
                self.update()
                return

            try:
                ocr_text = self.pytesseract.image_to_string(img).strip()
                if not ocr_text:
                    prompt = "Give a generic UI explanation under 8 words."
                else:
                    prompt = f"OCR:\n{ocr_text}\n\nGive UI explanation under 8 words."
                self.explanation_text = self.ai.ask(prompt)
            except Exception as e:
                self.explanation_text = f"OCR/AI error: {e}"

            self.show_explanation_panel = True
            self.update()

        # ---------- CONFIG TOGGLE ----------

        def toggle_config_panel(self):
            self.config_mode = not self.config_mode
            self.log_event(f"Config panel: {'OPEN' if self.config_mode else 'CLOSED'}")
            self.update()

        # ---------- HUD THICKNESS ----------

        def get_bar_height(self):
            SM_CYSCREEN = 1
            SM_CYFULLSCREEN = 17
            total_h = user32.GetSystemMetrics(SM_CYSCREEN)
            full_h = user32.GetSystemMetrics(SM_CYFULLSCREEN)
            taskbar_h = total_h - full_h if total_h > full_h else 0
            max_height = total_h - taskbar_h
            max_height = max(self.base_bar_height, max_height)
            pct = max(0, min(100, self.hud_thickness_pct))
            return int(self.base_bar_height + (max_height - self.base_bar_height) * (pct / 100.0))

        # ---------- GAME OCR HINT LOOP ----------

        def update_ai_hint_loop(self):
            if not self.is_game_mode():
                return

            now = time.time()
            if now - self.last_hint_time < self.hint_cooldown:
                return

            img = self.capture_game_region_for_hint()
            if img is None:
                ocr_text = ""
            else:
                try:
                    ocr_text = self.pytesseract.image_to_string(img).strip()
                except Exception:
                    ocr_text = ""

            danger = False
            if ocr_text:
                danger_tokens = ["ENEMY", "DANGER", "WARNING", "ALERT", "ATTACK", "NEARBY"]
                upper = ocr_text.upper()
                if any(tok in upper for tok in danger_tokens):
                    danger = True
                    self.last_ocr_danger_time = now
                    prompt = f"OCR:\n{ocr_text}\n\nGive tactical danger hint under 8 words."
                else:
                    prompt = f"OCR:\n{ocr_text}\n\nGive tactical game hint under 8 words."
            else:
                prompt = "Give tactical game hint under 8 words."

            try:
                hint = self.ai.ask(prompt)
            except Exception as e:
                hint = f"[AI hint error] {e}"

            hint = hint.strip()
            if not hint:
                return

            timestamp = time.strftime("%H:%M:%S")
            tag = "[GAME]" if not danger else "[DANGER]"
            line = f"{timestamp} {tag} {hint}"
            self.hint_lines.append(line)
            self.trim_hints_to_fit()
            self.last_hint_time = time.time()
            self.update()

        # ---------- CURSOR COMMENTARY ----------

        def update_cursor_commentary(self):
            try:
                pt = POINT()
                if not user32.GetCursorPos(ctypes.byref(pt)):
                    return

                cur_pos = (pt.x, pt.y)
                now = time.time()

                if self.last_cursor_pos is None:
                    self.last_cursor_pos = cur_pos
                    return

                dx = cur_pos[0] - self.last_cursor_pos[0]
                dy = cur_pos[1] - self.last_cursor_pos[1]
                dist = math.hypot(dx, dy)

                if dist < self.cursor_move_threshold:
                    return

                if now - self.last_cursor_comment_time < self.cursor_comment_cooldown:
                    self.last_cursor_pos = cur_pos
                    return

                self.last_cursor_pos = cur_pos
                self.last_cursor_comment_time = now

                img = self.capture_region_around_cursor()
                if img is None:
                    ocr_text = ""
                else:
                    try:
                        ocr_text = self.pytesseract.image_to_string(img).strip()
                    except Exception:
                        ocr_text = ""

                if ocr_text:
                    prompt = f"OCR:\n{ocr_text}\n\nGive UI hint under 8 words."
                else:
                    prompt = "Give generic UI hint under 8 words."

                try:
                    comment = self.ai.ask(prompt)
                except Exception as e:
                    comment = f"[AI cursor error] {e}"

                comment = comment.strip()
                if not comment:
                    return

                timestamp = time.strftime("%H:%M:%S")
                line = f"{timestamp} [CURSOR] {comment}"
                self.hint_lines.append(line)
                self.trim_hints_to_fit()
                self.update()
            except Exception:
                pass

        # ---------- CLICK INTELLIGENCE (LMB) ----------

        def update_click_intel(self):
            try:
                state = user32.GetAsyncKeyState(0x01) & 0x8000  # LMB
                is_down = bool(state)
                now = time.time()

                if is_down and not self.last_lbutton_down:
                    # Record click time for rhythm
                    self.last_click_times.append(now)
                    self.last_click_times = [t for t in self.last_click_times if now - t <= self.click_window]

                    # Mark combat input
                    self.last_combat_input_time = now

                    img = self.capture_region_around_cursor()
                    if img is None:
                        ocr_text = ""
                    else:
                        try:
                            ocr_text = self.pytesseract.image_to_string(img).strip()
                        except Exception:
                            ocr_text = ""

                    if now - self.last_hint_time >= self.hint_cooldown:
                        if ocr_text:
                            prompt = f"OCR:\n{ocr_text}\n\nGive firing hint under 8 words."
                        else:
                            prompt = "Give firing hint under 8 words."

                        try:
                            comment = self.ai.ask(prompt)
                        except Exception as e:
                            comment = f"[AI click error] {e}"

                        comment = comment.strip()
                        if comment:
                            timestamp = time.strftime("%H:%M:%S")
                            line = f"{timestamp} [CLICK] {comment}"
                            self.hint_lines.append(line)
                            self.trim_hints_to_fit()
                            self.last_hint_time = now
                            self.update()

                self.last_lbutton_down = is_down
            except Exception:
                pass

        # ---------- INPUT INTELLIGENCE (WASD + MOUSE BUTTONS + SPRINT/CROUCH/JUMP) ----------

        def get_key_down(self, vk):
            return bool(user32.GetAsyncKeyState(vk) & 0x8000)

        def update_input_intel(self):
            now = time.time()

            # Keyboard
            w = self.get_key_down(0x57)  # W
            a = self.get_key_down(0x41)  # A
            s = self.get_key_down(0x53)  # S
            d = self.get_key_down(0x44)  # D
            x = self.get_key_down(0x58)  # X
            shift = self.get_key_down(0x10)  # Shift
            ctrl = self.get_key_down(0x11)   # Ctrl
            space = self.get_key_down(0x20)  # Space

            # Mouse buttons
            lmb = self.get_key_down(0x01)
            rmb = self.get_key_down(0x02)
            mmb = self.get_key_down(0x04)
            x1 = self.get_key_down(0x05)
            x2 = self.get_key_down(0x06)

            any_input = (
                w or a or s or d or x or
                lmb or rmb or mmb or x1 or x2 or
                shift or ctrl or space
            )
            if any_input:
                self.last_any_input_time = now

            # Combat input (for adaptive sampling)
            if lmb or rmb or mmb or x1 or x2:
                self.last_combat_input_time = now

            # Sprint tracking (high sensitivity)
            if shift and self.sprint_start_time is None:
                self.sprint_start_time = now
            if not shift and self.sprint_start_time is not None:
                self.sprint_start_time = None

            # Crouch tracking
            if ctrl and self.crouch_start_time is None:
                self.crouch_start_time = now
            if not ctrl and self.crouch_start_time is not None:
                self.crouch_start_time = None

            # Jump tracking
            if space and not self.last_space_state:
                self.jump_times.append(now)
                self.jump_times = [t for t in self.jump_times if now - t <= 3.0]

            # Decide if we should emit a movement hint (respect cooldown)
            if now - self.last_hint_time >= self.hint_cooldown:
                # Idle detection
                idle_time = now - self.last_any_input_time
                if idle_time >= self.idle_threshold and self.is_game_mode():
                    prompt = "Idle 1s. Give movement hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[MOVE]")
                    return

                # Sprint pattern
                if self.sprint_start_time is not None:
                    prompt = "Sprint active. Give sprint hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[SPRINT]")
                    return

                # Crouch pattern
                if self.crouch_start_time is not None:
                    prompt = "Crouch active. Give crouch hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[CROUCH]")
                    return

                # Jump spam
                if len(self.jump_times) >= 3:
                    prompt = "Jump spam. Give movement hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[JUMP]")
                    self.jump_times.clear()
                    return

                # Movement predictability (simple heuristic)
                if w and not (a or s or d):
                    prompt = "Forward only. Give movement hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[MOVE]")
                    return
                if a and not (w or s or d):
                    prompt = "Left strafe only. Give movement hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[MOVE]")
                    return
                if d and not (w or a or s):
                    prompt = "Right strafe only. Give movement hint under 8 words."
                    self.emit_movement_hint(prompt, tag="[MOVE]")
                    return

            # Store last states
            self.last_w_state = w
            self.last_a_state = a
            self.last_s_state = s
            self.last_d_state = d
            self.last_x_state = x
            self.last_shift_state = shift
            self.last_ctrl_state = ctrl
            self.last_space_state = space
            self.last_rmb_state = rmb
            self.last_mmb_state = mmb
            self.last_x1_state = x1
            self.last_x2_state = x2

        def emit_movement_hint(self, prompt: str, tag: str = "[MOVE]"):
            now = time.time()
            if now - self.last_hint_time < self.hint_cooldown:
                return

            try:
                hint = self.ai.ask(prompt)
            except Exception as e:
                hint = f"[AI move error] {e}"

            hint = hint.strip()
            if not hint:
                return

            timestamp = time.strftime("%H:%M:%S")
            line = f"{timestamp} {tag} {hint}"
            self.hint_lines.append(line)
            self.trim_hints_to_fit()
            self.last_hint_time = now
            self.update()

        # ---------- ULTRA AIM ENGINE (MOUSE MOVEMENT) ----------

        def is_combat_mode(self):
            now = time.time()
            # D: OCR danger OR any combat input
            if now - self.last_ocr_danger_time < 3.0:
                return True
            if now - self.last_combat_input_time < 3.0:
                return True
            return False

        def update_mouse_sampling_interval(self):
            # Adaptive: 120 Hz in combat, 60 Hz in movement, 30 Hz idle
            now = time.time()
            idle_time = now - self.last_any_input_time
            combat = self.is_combat_mode()

            if combat:
                target_ms = 8   # ~120 Hz
            elif idle_time < 3.0:
                target_ms = 16  # ~60 Hz
            else:
                target_ms = 33  # ~30 Hz

            if target_ms != self.mouse_sample_interval_ms:
                self.mouse_sample_interval_ms = target_ms
                self.mouse_timer.start(self.mouse_sample_interval_ms)

        def update_mouse_aim_intel(self):
            try:
                self.update_mouse_sampling_interval()

                pt = POINT()
                if not user32.GetCursorPos(ctypes.byref(pt)):
                    return

                now = time.time()
                cur_pos = (pt.x, pt.y)

                if self.last_mouse_pos is None or self.last_mouse_time is None:
                    self.last_mouse_pos = cur_pos
                    self.last_mouse_time = now
                    return

                dt = now - self.last_mouse_time
                if dt <= 0:
                    dt = 0.001

                dx = cur_pos[0] - self.last_mouse_pos[0]
                dy = cur_pos[1] - self.last_mouse_pos[1]
                dist = math.hypot(dx, dy)
                speed = dist / dt

                # Normalize direction
                if dist > 0:
                    ndx = dx / dist
                    ndy = dy / dist
                else:
                    ndx = 0.0
                    ndy = 0.0

                # Store in history
                self.aim_vel_history.append((now, speed))
                self.aim_dir_history.append((now, ndx, ndy))

                # Trim history
                cutoff = now - self.aim_history_window
                self.aim_vel_history = [v for v in self.aim_vel_history if v[0] >= cutoff]
                self.aim_dir_history = [d for d in self.aim_dir_history if d[0] >= cutoff]

                self.last_mouse_pos = cur_pos
                self.last_mouse_time = now

                # Only analyze in game mode
                if not self.is_game_mode():
                    return

                # Ultra sensitivity: detect jitter, flicks, oscillation, panic
                if now - self.last_hint_time < self.hint_cooldown:
                    return

                speeds = [v[1] for v in self.aim_vel_history]
                if len(speeds) < 5:
                    return

                avg_speed = sum(speeds) / len(speeds)
                var_speed = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
                std_speed = math.sqrt(var_speed)

                # Direction changes
                dir_changes = 0
                last_dir = None
                for _, x, y in self.aim_dir_history:
                    if last_dir is None:
                        last_dir = (x, y)
                        continue
                    dot = last_dir[0] * x + last_dir[1] * y
                    if dot < 0.0:
                        dir_changes += 1
                        last_dir = (x, y)

                # Heuristics
                jitter = std_speed > 300 and avg_speed < 800
                flick = avg_speed > 1000
                oscillation = dir_changes >= 4
                panic = std_speed > 600 and dir_changes >= 3

                # Ultra: any of these can trigger
                prompt = None
                tag = "[AIM]"

                if panic:
                    prompt = "Panic aim. Give aim hint under 8 words."
                elif jitter:
                    prompt = "High jitter. Give aim hint under 8 words."
                elif flick:
                    prompt = "Flick detected. Give aim hint under 8 words."
                elif oscillation:
                    prompt = "Oscillation. Give aim hint under 8 words."

                if prompt:
                    self.emit_aim_hint(prompt, tag)
            except Exception:
                pass

        def emit_aim_hint(self, prompt: str, tag: str = "[AIM]"):
            now = time.time()
            if now - self.last_hint_time < self.hint_cooldown:
                return

            try:
                hint = self.ai.ask(prompt)
            except Exception as e:
                hint = f"[AI aim error] {e}"

            hint = hint.strip()
            if not hint:
                return

            timestamp = time.strftime("%H:%M:%S")
            line = f"{timestamp} {tag} {hint}"
            self.hint_lines.append(line)
            self.trim_hints_to_fit()
            self.last_hint_time = now
            self.update()

        # ---------- HINT FEED TRIM ----------

        def trim_hints_to_fit(self):
            bar_height = self.get_bar_height()
            available = max(0, bar_height - 30)
            if available <= 0:
                self.hint_lines = []
                return

            line_height = 16
            max_lines = max(1, available // line_height)

            if len(self.hint_lines) > max_lines:
                self.hint_lines = self.hint_lines[-max_lines:]

        # ---------- PAINT ----------

        def paintEvent(self, event):
            painter = QPainter(self)

            bar_height = self.get_bar_height()
            profile = self.get_current_profile()
            base_opacity = self.hud_opacity
            if profile and "hud_opacity" in profile:
                base_opacity = profile["hud_opacity"]

            glow = self.glow_intensity
            if self.is_game_mode():
                painter.fillRect(
                    0,
                    0,
                    self.width(),
                    bar_height,
                    QColor(5, 5, 20 + glow // 2, base_opacity)
                )
            else:
                painter.fillRect(
                    0,
                    0,
                    self.width(),
                    bar_height,
                    QColor(0, 0, 0, base_opacity)
                )

            # Main HUD text
            painter.setPen(QColor(200, 255, 255, 230))
            painter.setFont(QFont("Segoe UI", 10))
            hud_margin = 20
            hud_rect = QRect(
                hud_margin,
                4,
                self.width() - 2 * hud_margin,
                24
            )
            painter.drawText(
                hud_rect,
                Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
                self.hud_text
            )

            # Close button
            painter.setPen(QColor(255, 255, 255, 230))
            painter.setBrush(QColor(80 + glow // 3, 0, 0, 200))
            painter.drawRect(self.close_button_rect)
            painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
            painter.drawText(self.close_button_rect, Qt.AlignCenter, "X")

            # AI stacked feed
            if self.hint_lines:
                painter.setFont(QFont("Segoe UI", 9))
                painter.setPen(QColor(180, 230, 255, 230))

                x = 20
                y = 32
                available_width = self.width() - 40
                line_spacing = 2

                for line in self.hint_lines:
                    if y >= bar_height - 12:
                        break
                    rect = QRect(x, y, available_width, bar_height - y)
                    painter.drawText(rect, Qt.AlignLeft | Qt.TextWordWrap, line)
                    metrics = painter.fontMetrics()
                    wrapped_height = metrics.boundingRect(rect, Qt.AlignLeft | Qt.TextWordWrap, line).height()
                    y += wrapped_height + line_spacing

            # Translation panel
            if self.show_translation_panel and self.translation_text:
                panel_scale = self.panel_scale
                if self.is_game_mode():
                    overlay_width = int(self.width() * (0.3 + 0.3 * panel_scale))
                    overlay_height = 160
                    x = (self.width() - overlay_width) // 2
                    y = int(self.height() * 0.7)
                else:
                    overlay_width = int(self.width() * (0.35 + 0.35 * panel_scale))
                    overlay_height = 180
                    x = (self.width() - overlay_width) // 2
                    y = (self.height() - overlay_height) // 2

                rect = QRect(x, y, overlay_width, overlay_height)
                painter.setBrush(QColor(0, 0, 0, 210))
                painter.setPen(QColor(0, 200, 255, 230))
                painter.drawRect(rect)

                painter.setFont(QFont("Segoe UI", 11))
                painter.setPen(QColor(220, 255, 255, 255))

                text = f"Translation\n\n{self.translation_text}"
                painter.drawText(
                    rect.adjusted(10, 10, -10, -10),
                    Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                    text
                )

            # Explanation panel
            if self.show_explanation_panel and self.explanation_text:
                panel_scale = self.panel_scale
                overlay_width = int(self.width() * (0.35 + 0.35 * panel_scale))
                overlay_height = 220
                x = 20
                y = bar_height + 20

                rect = QRect(x, y, overlay_width, overlay_height)
                painter.setBrush(QColor(0, 0, 0, 210))
                painter.setPen(QColor(0, 180, 120, 230))
                painter.drawRect(rect)

                painter.setFont(QFont("Segoe UI", 10))
                painter.setPen(QColor(200, 255, 220, 255))

                text = f"Explanation\n\n{self.explanation_text}"
                painter.drawText(
                    rect.adjusted(10, 10, -10, -10),
                    Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                    text
                )

            # Config panel
            if self.config_mode:
                overlay_width = 360
                overlay_height = 280
                x = self.width() - overlay_width - 20
                y = bar_height + 20

                rect = QRect(x, y, overlay_width, overlay_height)
                painter.setBrush(QColor(0, 0, 0, 230))
                painter.setPen(QColor(180, 180, 180, 230))
                painter.drawRect(rect)

                painter.setFont(QFont("Segoe UI", 9))
                painter.setPen(QColor(230, 230, 230, 255))

                lines = []
                lines.append("Config Panel")
                lines.append("")
                lines.append(f"Game EXE (aggressive): {self.current_game_exe or 'None'}")
                profile = self.get_current_profile()
                if profile:
                    lines.append(
                        f"Profile: ACTIVE (opacity={profile.get('hud_opacity')}, "
                        f"net={profile.get('show_net')})"
                    )
                else:
                    lines.append("Profile: None")

                lines.append("")
                lines.append("Sliders:")
                lines.append(f"HUD Opacity: {self.hud_opacity}")
                lines.append(f"Game Aggression: {self.game_aggression}")
                lines.append(f"Cursor Radius: {int(self.cursor_scale * 100)}")
                lines.append(f"Panel Size: {int(self.panel_scale * 100)}")
                lines.append(f"Glow Intensity: {self.glow_intensity}")
                lines.append(f"Beast Threshold: {self.beast_threshold}")
                lines.append(f"HUD Thickness: {self.hud_thickness_pct}%")
                lines.append(f"HUD Width: {self.hud_width_pct}%")

                lines.append("")
                lines.append("Actions:")
                lines.append("Shift+F8       → Translate under cursor")
                lines.append("Shift+F7       → Explain under cursor")
                lines.append("Shift+Alt+F10  → Toggle this panel")
                lines.append("Shift+Alt+F9   → Toggle slider mini-panel")
                lines.append("ESC            → Close visor")

                lines.append("")
                lines.append("AI Engines:")
                if self.ai.engines:
                    for idx, eng in enumerate(self.ai.engines):
                        mark = "* " if eng.get("name") == self.ai.preferred_engine_name else "  "
                        status = "OK" if eng.get("available") else "OFF"
                        lines.append(f"{mark}{idx}: {eng.get('name')} [{status}]")
                    lines.append("Use number keys 0-9 to set preferred engine.")
                else:
                    lines.append("No engines loaded.")

                text = "\n".join(lines)
                painter.drawText(
                    rect.adjusted(10, 10, -10, -10),
                    Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                    text
                )

            painter.end()

        # ---------- MOUSE INPUT ----------

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                if self.close_button_rect.contains(event.pos()):
                    self.close()
                    return

                bar_height = self.get_bar_height()
                if 0 <= event.pos().y() <= bar_height:
                    self.dragging_hud = True
                    self.drag_offset_x = event.globalPosition().toPoint().x() - self.x()
                    event.accept()
                    return

            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self.dragging_hud:
                global_x = event.globalPosition().toPoint().x()
                new_x = global_x - self.drag_offset_x

                sw = self.screen_geom.width()
                hud_w = self.width()
                max_x = sw - hud_w
                self.hud_offset_x = max(0, min(new_x, max_x))

                self.update_window_geometry()
                event.accept()
                return

            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.LeftButton and self.dragging_hud:
                self.dragging_hud = False
                event.accept()
                return
            super().mouseReleaseEvent(event)

        # ---------- KEY INPUT ----------

        def keyPressEvent(self, event):
            key = event.key()
            mods = event.modifiers()

            if key == Qt.Key_Escape:
                self.close()
                return

            if key == Qt.Key_F8 and (mods & Qt.ShiftModifier):
                self.translate_under_cursor()
                return

            if key == Qt.Key_F7 and (mods & Qt.ShiftModifier):
                self.explain_under_cursor()
                return

            if key == Qt.Key_F10 and (mods & Qt.ShiftModifier) and (mods & Qt.AltModifier):
                self.toggle_config_panel()
                return

            if key == Qt.Key_F9 and (mods & Qt.ShiftModifier) and (mods & Qt.AltModifier):
                if self.slider_panel.isVisible():
                    self.slider_panel.hide()
                else:
                    self.slider_panel.show()
                return

            if self.config_mode and Qt.Key_0 <= key <= Qt.Key_9:
                idx = key - Qt.Key_0
                if 0 <= idx < len(self.ai.engines):
                    eng = self.ai.engines[idx]
                    self.ai.set_preferred_engine(eng.get("name"))
                    self.log_event(f"Preferred AI engine set to: {eng.get('name')}")
                    self.update()
                return

            super().keyPressEvent(event)

    class SliderPanel(QWidget):
        """
        Floating mini-panel with sliders and control buttons.
        Draggable, visible on startup.
        """
        def __init__(self, overlay):
            super().__init__(overlay)
            self.overlay = overlay
            self.dragging = False
            self.drag_offset = QPoint(0, 0)

            self.setWindowFlags(
                Qt.FramelessWindowHint |
                Qt.Tool |
                Qt.WindowStaysOnTopHint
            )
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_NoSystemBackground, True)

            self.bg_color = QColor(0, 0, 0, 220)
            self.border_color = QColor(0, 200, 255, 220)

            layout = QVBoxLayout()
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            title_row = QHBoxLayout()
            lbl_title = QLabel("Visor Controls")
            lbl_title.setStyleSheet("color: #00e0ff; font-weight: bold;")
            btn_close = QPushButton("X")
            btn_close.setFixedWidth(24)
            btn_close.clicked.connect(self.hide)
            btn_close.setStyleSheet(
                "QPushButton { color: white; background-color: #550000; border: 1px solid #aa0000; }"
                "QPushButton:hover { background-color: #770000; }"
            )
            title_row.addWidget(lbl_title)
            title_row.addStretch()
            title_row.addWidget(btn_close)
            layout.addLayout(title_row)

            btn_row = QHBoxLayout()
            self.btn_translate = QPushButton("Translate")
            self.btn_explain = QPushButton("Explain")
            self.btn_config = QPushButton("Config")

            for b in (self.btn_translate, self.btn_explain, self.btn_config):
                b.setStyleSheet(
                    "QPushButton { color: white; background-color: #202020; "
                    "border: 1px solid #00a0c0; font-size: 9pt; }"
                    "QPushButton:hover { background-color: #303030; }"
                )

            self.btn_translate.clicked.connect(self.overlay.translate_under_cursor)
            self.btn_explain.clicked.connect(self.overlay.explain_under_cursor)
            self.btn_config.clicked.connect(self.overlay.toggle_config_panel)

            btn_row.addWidget(self.btn_translate)
            btn_row.addWidget(self.btn_explain)
            btn_row.addWidget(self.btn_config)
            layout.addLayout(btn_row)

            def add_slider(label_text, min_val, max_val, init_val, on_change, fmt=lambda v: str(v)):
                row = QVBoxLayout()
                top = QHBoxLayout()
                lbl = QLabel(f"{label_text}: {fmt(init_val)}")
                lbl.setStyleSheet("color: white; font-size: 9pt;")
                top.addWidget(lbl)
                top.addStretch()
                row.addLayout(top)

                sld = QSlider(Qt.Horizontal)
                sld.setMinimum(min_val)
                sld.setMaximum(max_val)
                sld.setValue(init_val)

                def handler(v):
                    lbl.setText(f"{label_text}: {fmt(v)}")
                    on_change(v)
                    self.overlay.save_settings()

                sld.valueChanged.connect(handler)
                row.addWidget(sld)
                layout.addLayout(row)
                return sld, lbl

            self.slider_hud_opacity, _ = add_slider(
                "HUD Opacity",
                80,
                255,
                self.overlay.hud_opacity,
                self.on_hud_opacity_changed,
            )

            self.slider_game_aggr, _ = add_slider(
                "Game Aggression",
                0,
                100,
                self.overlay.game_aggression,
                self.on_game_aggression_changed,
            )

            self.slider_cursor_radius, _ = add_slider(
                "Cursor Radius",
                20,
                60,
                int(self.overlay.cursor_scale * 100),
                self.on_cursor_radius_changed,
            )

            self.slider_panel_size, _ = add_slider(
                "Panel Size",
                30,
                70,
                int(self.overlay.panel_scale * 100),
                self.on_panel_size_changed,
            )

            self.slider_glow, _ = add_slider(
                "Glow Intensity",
                0,
                100,
                self.overlay.glow_intensity,
                self.on_glow_changed,
            )

            self.slider_beast, _ = add_slider(
                "Beast Threshold",
                0,
                100,
                self.overlay.beast_threshold,
                self.on_beast_changed,
            )

            self.slider_thickness, _ = add_slider(
                "HUD Thickness",
                0,
                100,
                self.overlay.hud_thickness_pct,
                self.on_thickness_changed,
                fmt=lambda v: f"{v}%",
            )

            self.slider_width, _ = add_slider(
                "HUD Width",
                20,
                100,
                self.overlay.hud_width_pct,
                self.on_width_changed,
                fmt=lambda v: f"{v}%",
            )

            self.setLayout(layout)
            self.resize(320, 360)

        def on_hud_opacity_changed(self, val):
            self.overlay.hud_opacity = val
            self.overlay.update()

        def on_game_aggression_changed(self, val):
            self.overlay.game_aggression = val

        def on_cursor_radius_changed(self, val):
            self.overlay.cursor_scale = val / 100.0

        def on_panel_size_changed(self, val):
            self.overlay.panel_scale = val / 100.0

        def on_glow_changed(self, val):
            self.overlay.glow_intensity = val
            self.overlay.update()

        def on_beast_changed(self, val):
            self.overlay.beast_threshold = val

        def on_thickness_changed(self, val):
            self.overlay.hud_thickness_pct = val
            self.overlay.trim_hints_to_fit()
            self.overlay.update()

        def on_width_changed(self, val):
            self.overlay.hud_width_pct = val
            self.overlay.update_window_geometry()
            self.overlay.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            rect = self.rect()
            painter.setBrush(self.bg_color)
            painter.setPen(self.border_color)
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 8, 8)
            painter.end()

        def mousePressEvent(self, event: QMouseEvent):
            if event.button() == Qt.LeftButton:
                self.dragging = True
                self.drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event: QMouseEvent):
            if self.dragging:
                new_pos = event.globalPosition().toPoint() - self.drag_offset
                self.move(new_pos)
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent):
            if event.button() == Qt.LeftButton:
                self.dragging = False
                event.accept()
            else:
                super().mouseReleaseEvent(event)

    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

