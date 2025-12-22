import math
from collections import deque

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import NumericProperty, ObjectProperty, StringProperty
from kivy.graphics import Color, Line, Ellipse, Rectangle


# ==============================
# 1. Simulation engine
# ==============================

class LivingBatterySimulator:
    """
    Simulates:
      - Three series capacitors (C1, C2, C3) -> combined into C_outer.
      - Primary LC tank: Lp + Cp_inner + C_outer = Cp_eff.
      - Secondary LC tank: Ls + Cs + load R_load.
      - Mutual inductance M = k * sqrt(Lp * Ls).
      - Primary driven by a voltage pulse V_in(t).

    State variables:
      ip: primary inductor current
      is_: secondary inductor current
      vp: primary capacitor voltage (Cp_eff)
      vs: secondary capacitor voltage (Cs, also across R_load)
    """

    def __init__(self):
        # Default parameters
        self.C1 = 1e-9
        self.C2 = 2e-9
        self.C3 = 4e-9

        self.Lp = 100e-6
        self.Cp_inner = 1e-9

        self.Ls = 100e-6
        self.Cs = 1e-9

        self.k = 0.9
        self.R_load = 200.0

        self.V_drive_peak = 5.0
        self.drive_duration = 2e-6

        self.t = 0.0
        self.dt = 1e-8  # internal simulation step

        self.reset_state()

    def set_params(self, C1, C2, C3, Lp, Cp_inner, Ls, Cs, k, R_load,
                   V_drive_peak, drive_duration):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.Lp = Lp
        self.Cp_inner = Cp_inner
        self.Ls = Ls
        self.Cs = Cs
        self.k = max(0.0, min(0.999, k))
        self.R_load = max(1.0, R_load)
        self.V_drive_peak = V_drive_peak
        self.drive_duration = max(1e-8, drive_duration)
        self.reset_state()

    def reset_state(self):
        self.t = 0.0
        self.C_outer = self.combine_series_capacitors(self.C1, self.C2, self.C3)
        self.Cp_eff = self.Cp_inner + self.C_outer
        self.M = self.k * math.sqrt(self.Lp * self.Ls)

        # State: ip, is, vp, vs
        self.ip = 0.0
        self.is_ = 0.0
        self.vp = 0.0
        self.vs = 0.0

    @staticmethod
    def combine_series_capacitors(c1, c2, c3):
        inv = 1.0 / c1 + 1.0 / c2 + 1.0 / c3
        return 1.0 / inv

    def V_in(self, t):
        return self.V_drive_peak if t < self.drive_duration else 0.0

    def derivatives(self, t, state):
        ip, is_, vp, vs = state

        # Capacitor equations
        dvp_dt = ip / self.Cp_eff
        dvs_dt = (is_ - vs / self.R_load) / self.Cs

        Vin_t = self.V_in(t)

        # Inductor equations with mutual inductance
        # Lp * dip/dt + M * dis/dt = Vin_t - vp
        # M * dip/dt + Ls * dis/dt = -vs
        det = self.Lp * self.Ls - self.M * self.M
        if abs(det) < 1e-24:
            dip_dt = 0.0
            dis_dt = 0.0
        else:
            dip_dt = ((Vin_t - vp) * self.Ls - self.M * (-vs)) / det
            dis_dt = (self.Lp * (-vs) - self.M * (Vin_t - vp)) / det

        return dip_dt, dis_dt, dvp_dt, dvs_dt

    def rk4_step(self):
        t = self.t
        dt = self.dt
        state = (self.ip, self.is_, self.vp, self.vs)

        k1 = self.derivatives(t, state)
        k2 = self.derivatives(t + 0.5 * dt,
                              tuple(s + 0.5 * dt * k1i for s, k1i in zip(state, k1)))
        k3 = self.derivatives(t + 0.5 * dt,
                              tuple(s + 0.5 * dt * k2i for s, k2i in zip(state, k2)))
        k4 = self.derivatives(t + dt,
                              tuple(s + dt * k3i for s, k3i in zip(state, k3)))

        new_state = [
            s + dt / 6.0 * (k1i + 2 * k2i + 2 * k3i + k4i)
            for s, k1i, k2i, k3i, k4i in zip(state, k1, k2, k3, k4)
        ]

        self.ip, self.is_, self.vp, self.vs = new_state
        self.t += dt

    def step(self, steps=10):
        """Advance simulation by 'steps' internal RK4 steps."""
        for _ in range(steps):
            self.rk4_step()

        # Energies
        E_Cp = 0.5 * self.Cp_eff * self.vp * self.vp
        E_Cs = 0.5 * self.Cs * self.vs * self.vs
        E_Lp = 0.5 * self.Lp * self.ip * self.ip
        E_Ls = 0.5 * self.Ls * self.is_ * self.is_
        E_total = E_Cp + E_Cs + E_Lp + E_Ls

        return {
            "t": self.t,
            "vp": self.vp,
            "vs": self.vs,
            "ip": self.ip,
            "is": self.is_,
            "E_total": E_total,
            "E_Cp": E_Cp,
            "E_Cs": E_Cs,
            "E_Lp": E_Lp,
            "E_Ls": E_Ls,
        }


# ==============================
# 2. GUI widgets
# ==============================

class EnergyDiagramWidget(Widget):
    """
    Field view:
      - Nested capacitors as rings
      - Primary coil (left)
      - Secondary coil (right)
      - Color intensity based on |vp|, |vs|, |ip|, |is|
    """

    vp = NumericProperty(0.0)
    vs = NumericProperty(0.0)
    ip = NumericProperty(0.0)
    is_ = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            pass
        self.bind(pos=self._trigger_redraw, size=self._trigger_redraw,
                  vp=self._trigger_redraw, vs=self._trigger_redraw,
                  ip=self._trigger_redraw, is_=self._trigger_redraw)

    def _trigger_redraw(self, *args):
        self.canvas.clear()
        self.draw_diagram()

    def draw_diagram(self):
        w, h = self.width, self.height
        cx, cy = self.center_x, self.center_y
        r_outer = min(w, h) * 0.4

        vp_norm = min(1.0, abs(self.vp) / 10.0)
        vs_norm = min(1.0, abs(self.vs) / 10.0)
        ip_norm = min(1.0, abs(self.ip) / 1.0)
        is_norm = min(1.0, abs(self.is_) / 1.0)

        with self.canvas:
            # Background
            Color(0.05, 0.05, 0.08, 1)
            Rectangle(pos=self.pos, size=self.size)

            # Nested capacitors: three concentric rings
            for i, intensity in enumerate([0.3, 0.6, 1.0]):
                ri = r_outer * (0.35 + 0.15 * i)
                Color(0.2, 0.6 * intensity * vp_norm + 0.1, 0.9, 1)
                Line(circle=(cx, cy, ri), width=1.5)

            # Primary coil – left
            coil_w = w * 0.15
            coil_h = h * 0.4
            coil_x = self.x + w * 0.1
            coil_y = cy - coil_h / 2

            Color(0.8 * ip_norm + 0.1, 0.3, 0.1, 1)
            n_turns = 6
            dx = coil_w / n_turns
            points = []
            for i in range(n_turns):
                x = coil_x + i * dx
                points.extend([x, coil_y, x, coil_y + coil_h])
            Line(points=points, width=2)

            # Secondary coil – right
            coil_x2 = self.x + w * 0.75
            Color(0.1, 0.3, 0.8 * is_norm + 0.1, 1)
            points = []
            for i in range(n_turns):
                x = coil_x2 + i * dx
                points.extend([x, coil_y, x, coil_y + coil_h])
            Line(points=points, width=2)

            # Central core (energy pulse)
            core_r = r_outer * 0.18
            core_intensity = min(1.0, vp_norm + vs_norm + 0.3)
            Color(0.4 + 0.6 * core_intensity, 0.2, 0.7 * core_intensity, 0.9)
            Ellipse(pos=(cx - core_r, cy - core_r), size=(2 * core_r, 2 * core_r))


class CircuitDiagramWidget(Widget):
    """
    Circuit schematic view:
      - C1, C2, C3 in series
      - Primary coil + Cp_inner
      - Secondary coil + Cs
      - Transformer coupling
      - Load resistor
      - Annotated with:
        - vp, vs, ip, is_ (numeric overlays)
        - E_Cp, E_Cs, E_Lp, E_Ls (numeric + mini bars + glow)
    """

    vp = NumericProperty(0.0)
    vs = NumericProperty(0.0)
    ip = NumericProperty(0.0)
    is_ = NumericProperty(0.0)

    E_Cp = NumericProperty(0.0)
    E_Cs = NumericProperty(0.0)
    E_Lp = NumericProperty(0.0)
    E_Ls = NumericProperty(0.0)
    E_total = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            pass

        # Numeric overlay labels
        self.lbl_vp = Label(text="vp=0.00V", font_size=11, size_hint=(None, None))
        self.lbl_vs = Label(text="vs=0.00V", font_size=11, size_hint=(None, None))
        self.lbl_ip = Label(text="ip=0.00A", font_size=11, size_hint=(None, None))
        self.lbl_is = Label(text="is=0.00A", font_size=11, size_hint=(None, None))

        self.lbl_E_Cp = Label(text="E_Cp=0.00J", font_size=10, size_hint=(None, None))
        self.lbl_E_Cs = Label(text="E_Cs=0.00J", font_size=10, size_hint=(None, None))
        self.lbl_E_Lp = Label(text="E_Lp=0.00J", font_size=10, size_hint=(None, None))
        self.lbl_E_Ls = Label(text="E_Ls=0.00J", font_size=10, size_hint=(None, None))

        for lbl in [self.lbl_vp, self.lbl_vs, self.lbl_ip, self.lbl_is,
                    self.lbl_E_Cp, self.lbl_E_Cs, self.lbl_E_Lp, self.lbl_E_Ls]:
            self.add_widget(lbl)

        self.bind(pos=self._trigger_redraw, size=self._trigger_redraw,
                  vp=self._trigger_redraw, vs=self._trigger_redraw,
                  ip=self._trigger_redraw, is_=self._trigger_redraw,
                  E_Cp=self._trigger_redraw, E_Cs=self._trigger_redraw,
                  E_Lp=self._trigger_redraw, E_Ls=self._trigger_redraw,
                  E_total=self._trigger_redraw)

    def _trigger_redraw(self, *args):
        self.canvas.clear()
        self.draw_schematic()
        self.update_labels()

    def _energy_fraction(self, E):
        if self.E_total <= 0:
            return 0.0
        return max(0.0, min(1.0, E / self.E_total))

    def draw_schematic(self):
        w, h = self.width, self.height
        x0, y0 = self.x, self.y

        f_Cp = self._energy_fraction(self.E_Cp)
        f_Cs = self._energy_fraction(self.E_Cs)
        f_Lp = self._energy_fraction(self.E_Lp)
        f_Ls = self._energy_fraction(self.E_Ls)

        with self.canvas:
            # Background
            Color(0.04, 0.04, 0.06, 1)
            Rectangle(pos=self.pos, size=self.size)

            # Common Y center
            yc = y0 + h * 0.5

            # Series capacitors C1, C2, C3
            Color(0.8, 0.8, 0.9, 1)
            seg_w = w * 0.18
            x_start = x0 + w * 0.05
            x_c1 = x_start + seg_w
            x_c2 = x_c1 + seg_w
            x_c3 = x_c2 + seg_w

            # Wires before/after series caps
            Line(points=[x_start, yc, x_c1 - 10, yc], width=1.3)
            Line(points=[x_c1 + 10, yc, x_c2 - 10, yc], width=1.3)
            Line(points=[x_c2 + 10, yc, x_c3 - 10, yc], width=1.3)
            Line(points=[x_c3 + 10, yc, x_c3 + seg_w * 0.6, yc], width=1.3)

            def draw_cap(xc):
                cap_h = h * 0.14
                Color(0.9, 0.9, 0.1, 1)
                Line(points=[xc - 10, yc - cap_h / 2, xc - 10, yc + cap_h / 2],
                     width=1.4)
                Line(points=[xc + 10, yc - cap_h / 2, xc + 10, yc + cap_h / 2],
                     width=1.4)

            draw_cap(x_c1)
            draw_cap(x_c2)
            draw_cap(x_c3)

            # Primary coil + Cp (below)
            self._coil_x = coil_x = x0 + w * 0.15
            self._coil_y = coil_y = y0 + h * 0.18
            coil_w = w * 0.18
            coil_h = h * 0.18
            n_turns = 5
            dx = coil_w / n_turns

            Color(0.6 + 0.4 * f_Lp, 0.2, 0.1, 1)  # glow based on E_Lp
            pts = []
            for i in range(n_turns):
                x = coil_x + i * dx
                pts.extend([x, coil_y, x, coil_y + coil_h])
            Line(points=pts, width=1.8)

            # Mini energy bar for Lp
            bar_w = 8
            bar_h_max = h * 0.2
            Color(0.9, 0.4, 0.1, 1)
            Rectangle(pos=(coil_x - 15, coil_y),
                      size=(bar_w, bar_h_max * f_Lp))

            # Cp across primary coil
            self._cp_x = cap_x = coil_x + coil_w + 20
            cap_yc = coil_y + coil_h / 2
            cap_h = coil_h * 0.6
            Color(0.8 + 0.2 * f_Cp, 0.8, 0.1, 1)  # glow based on E_Cp
            Line(points=[cap_x - 8, cap_yc - cap_h / 2, cap_x - 8, cap_yc + cap_h / 2],
                 width=1.3)
            Line(points=[cap_x + 8, cap_yc - cap_h / 2, cap_x + 8, cap_yc + cap_h / 2],
                 width=1.3)
            Line(points=[coil_x, cap_yc, coil_x, coil_y + coil_h], width=1.2)
            Line(points=[coil_x + coil_w, cap_yc, coil_x + coil_w, coil_y + coil_h],
                 width=1.2)
            Line(points=[coil_x, cap_yc, cap_x - 8, cap_yc], width=1.2)
            Line(points=[coil_x + coil_w, cap_yc, cap_x + 8, cap_yc], width=1.2)

            # Mini energy bar for Cp
            Color(0.9, 0.9, 0.2, 1)
            Rectangle(pos=(cap_x + 18, cap_yc - bar_h_max * f_Cp / 2),
                      size=(bar_w, bar_h_max * f_Cp))

            # Secondary coil + Cs + load (above)
            self._s_coil_x = s_coil_x = x0 + w * 0.62
            self._s_coil_y = s_coil_y = y0 + h * 0.52
            s_coil_w = w * 0.18
            s_coil_h = h * 0.18

            Color(0.2, 0.4, 0.8 + 0.2 * f_Ls, 1)  # glow based on E_Ls
            pts = []
            for i in range(n_turns):
                x = s_coil_x + i * (s_coil_w / n_turns)
                pts.extend([x, s_coil_y, x, s_coil_y + s_coil_h])
            Line(points=pts, width=1.8)

            # Mini energy bar for Ls
            Color(0.3, 0.6, 1.0, 1)
            Rectangle(pos=(s_coil_x - 15, s_coil_y),
                      size=(bar_w, bar_h_max * f_Ls))

            # Cs across secondary coil
            self._cs_x = s_cap_x = s_coil_x + s_coil_w + 20
            s_cap_yc = s_coil_y + s_coil_h / 2
            s_cap_h = s_coil_h * 0.6
            Color(0.8 + 0.2 * f_Cs, 0.9, 0.3, 1)  # glow based on E_Cs
            Line(points=[s_cap_x - 8, s_cap_yc - s_cap_h / 2, s_cap_x - 8, s_cap_yc + s_cap_h / 2],
                 width=1.3)
            Line(points=[s_cap_x + 8, s_cap_yc - s_cap_h / 2, s_cap_x + 8, s_cap_yc + s_cap_h / 2],
                 width=1.3)
            Line(points=[s_coil_x, s_cap_yc, s_coil_x, s_coil_y + s_coil_h], width=1.2)
            Line(points=[s_coil_x + s_coil_w, s_cap_yc,
                         s_coil_x + s_coil_w, s_coil_y + s_coil_h], width=1.2)
            Line(points=[s_coil_x, s_cap_yc, s_cap_x - 8, s_cap_yc], width=1.2)
            Line(points=[s_coil_x + s_coil_w, s_cap_yc, s_cap_x + 8, s_cap_yc],
                 width=1.2)

            # Mini energy bar for Cs
            Color(0.9, 0.9, 0.4, 1)
            Rectangle(pos=(s_cap_x + 18, s_cap_yc - bar_h_max * f_Cs / 2),
                      size=(bar_w, bar_h_max * f_Cs))

            # Load resistor to right of secondary cap
            r_x1 = s_cap_x + 20
            r_x2 = r_x1 + 40
            r_y1 = s_cap_yc + 20
            r_y2 = s_cap_yc - 20
            Color(0.9, 0.7, 0.3, 1)
            Line(points=[s_cap_x + 8, s_cap_yc, r_x1, s_cap_yc], width=1.2)
            Line(points=[r_x1, s_cap_yc, r_x1, r_y1], width=1.2)
            Line(points=[r_x1, r_y1, r_x2, r_y2], width=1.2)
            Line(points=[r_x2, r_y2, r_x2, s_cap_yc], width=1.2)
            Line(points=[r_x2, s_cap_yc, r_x2 + 20, s_cap_yc], width=1.2)

            # Transformer core indication
            core_x = x0 + w * 0.5
            core_y1 = coil_y + coil_h
            core_y2 = s_coil_y
            Color(0.7, 0.7, 0.7, 1)
            Line(points=[core_x - 5, core_y1, core_x - 5, core_y2], width=1.2)
            Line(points=[core_x + 5, core_y1, core_x + 5, core_y2], width=1.2)

    def update_labels(self):
        # Position overlays relative to stored feature positions
        w, h = self.width, self.height

        # Primary coil label (ip) near primary coil
        self.lbl_ip.text = f"ip={self.ip: .2e} A"
        self.lbl_ip.size = (120, 20)
        self.lbl_ip.pos = (self._coil_x - 10, self._coil_y - 25)

        # Primary capacitor label (vp) near Cp
        self.lbl_vp.text = f"vp={self.vp: .2f} V"
        self.lbl_vp.size = (130, 20)
        self.lbl_vp.pos = (self._cp_x - 40, self._coil_y + h * 0.22)

        # Secondary coil label (is) near secondary coil
        self.lbl_is.text = f"is={self.is_: .2e} A"
        self.lbl_is.size = (130, 20)
        self.lbl_is.pos = (self._s_coil_x - 10, self._s_coil_y + h * 0.26)

        # Secondary capacitor label (vs) near Cs
        self.lbl_vs.text = f"vs={self.vs: .2f} V"
        self.lbl_vs.size = (130, 20)
        self.lbl_vs.pos = (self._cs_x - 40, self._s_coil_y - 25)

        # Energy labels near their respective elements
        self.lbl_E_Lp.text = f"E_Lp={self.E_Lp: .2e} J"
        self.lbl_E_Lp.size = (150, 18)
        self.lbl_E_Lp.pos = (self._coil_x - 10, self._coil_y + h * 0.03)

        self.lbl_E_Cp.text = f"E_Cp={self.E_Cp: .2e} J"
        self.lbl_E_Cp.size = (150, 18)
        self.lbl_E_Cp.pos = (self._cp_x - 10, self._coil_y + h * 0.10)

        self.lbl_E_Ls.text = f"E_Ls={self.E_Ls: .2e} J"
        self.lbl_E_Ls.size = (150, 18)
        self.lbl_E_Ls.pos = (self._s_coil_x - 10, self._s_coil_y - h * 0.05)

        self.lbl_E_Cs.text = f"E_Cs={self.E_Cs: .2e} J"
        self.lbl_E_Cs.size = (150, 18)
        self.lbl_E_Cs.pos = (self._cs_x - 10, self._s_coil_y + h * 0.03)


class GraphWidget(Widget):
    """
    Simple oscilloscope-like graph using Kivy canvas.
    Holds a deque of points for each trace:
      trace1: vp
      trace2: vs
      trace3: E_total
    """

    max_points = 300

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace1 = deque(maxlen=self.max_points)
        self.trace2 = deque(maxlen=self.max_points)
        self.trace3 = deque(maxlen=self.max_points)
        self.bind(pos=self._trigger_redraw, size=self._trigger_redraw)

    def add_sample(self, vp, vs, E_total):
        self.trace1.append(vp)
        self.trace2.append(vs)
        self.trace3.append(E_total)
        self._trigger_redraw()

    def _trigger_redraw(self, *args):
        self.canvas.clear()
        self.draw_graphs()

    def draw_graphs(self):
        if not self.trace1:
            return

        w, h = self.width, self.height
        x0, y0 = self.x, self.y

        with self.canvas:
            # background
            Color(0.02, 0.02, 0.04, 1)
            Rectangle(pos=self.pos, size=self.size)

            # Horizontal separators
            Color(0.3, 0.3, 0.4, 1)
            Line(points=[x0 + 5, y0 + h * 0.33, x0 + w - 5, y0 + h * 0.33], width=1)
            Line(points=[x0 + 5, y0 + h * 0.66, x0 + w - 5, y0 + h * 0.66], width=1)

            def norm_trace(trace):
                if not trace:
                    return [0.0] * self.max_points
                max_abs = max(abs(v) for v in trace) or 1.0
                return [v / max_abs for v in trace]

            n1 = norm_trace(self.trace1)
            n2 = norm_trace(self.trace2)
            n3 = norm_trace(self.trace3)

            def draw_single(norm_vals, y_center, color):
                Color(*color)
                pts = []
                for i, v in enumerate(norm_vals):
                    x = x0 + 5 + (w - 10) * (i / max(1, len(norm_vals) - 1))
                    y = y0 + y_center + v * (h * 0.14)
                    pts.extend([x, y])
                if len(pts) >= 4:
                    Line(points=pts, width=1.3)

            # vp (top band)
            draw_single(n1, h * 0.17, (0.7, 0.9, 0.2, 1))
            # vs (middle band)
            draw_single(n2, h * 0.50, (0.2, 0.7, 0.9, 1))
            # E_total (bottom band)
            draw_single(n3, h * 0.83, (0.9, 0.4, 0.6, 1))


class ControlsPanel(GridLayout):
    """
    Middle panel with sliders, speed selector, and control buttons.
    Exposes:
      - get_params()
      - speed_mode (string)
    """

    simulator_ref = ObjectProperty(None)
    speed_mode = StringProperty("Normal")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 4
        self.padding = 5
        self.spacing = 5

        def add_slider_row(label_text, min_val, max_val, init, scale=1.0):
            lbl = Label(text=label_text, size_hint_x=0.9, font_size=12)
            sld = Slider(min=min_val, max=max_val, value=init,
                         size_hint_x=3.0)
            val_lbl = Label(text=f"{init*scale:.2e}", size_hint_x=1.3, font_size=10)

            def on_value(instance, value):
                val_lbl.text = f"{value * scale:.2e}"

            sld.bind(value=on_value)

            self.add_widget(lbl)
            self.add_widget(sld)
            self.add_widget(val_lbl)
            self.add_widget(Widget(size_hint_x=0.2))

            return sld

        # Capacitances (F)
        self.s_C1 = add_slider_row("C1 (nF)", 0.5, 5.0, 1.0, scale=1e-9)
        self.s_C2 = add_slider_row("C2 (nF)", 0.5, 5.0, 2.0, scale=1e-9)
        self.s_C3 = add_slider_row("C3 (nF)", 0.5, 5.0, 4.0, scale=1e-9)

        self.s_Cp = add_slider_row("Cp (nF)", 0.5, 5.0, 1.0, scale=1e-9)
        self.s_Cs = add_slider_row("Cs (nF)", 0.5, 5.0, 1.0, scale=1e-9)

        # Inductances (uH)
        self.s_Lp = add_slider_row("Lp (uH)", 10, 500, 100, scale=1e-6)
        self.s_Ls = add_slider_row("Ls (uH)", 10, 500, 100, scale=1e-6)

        # Coupling
        self.s_k = add_slider_row("k (0-1)", 0.1, 0.99, 0.9, scale=1.0)

        # Load (ohms)
        self.s_R = add_slider_row("R_load (ohm)", 10, 1000, 200, scale=1.0)

        # Drive
        self.s_V = add_slider_row("V_drive (V)", 1, 20, 5, scale=1.0)
        self.s_T = add_slider_row("drive T (us)", 0.1, 20, 2, scale=1e-6)

        # Buttons + speed
        self.btn_start = Button(text="Start", size_hint_x=1.0)
        self.btn_pause = Button(text="Pause", size_hint_x=1.0)
        self.btn_reset = Button(text="Reset & Apply", size_hint_x=1.8)

        self.speed_spinner = Spinner(
            text="Normal",
            values=("Slow", "Normal", "Overdrive"),
            size_hint_x=1.2,
        )

        def on_speed_change(spinner, value):
            self.speed_mode = value

        self.speed_spinner.bind(text=on_speed_change)

        self.add_widget(self.btn_start)
        self.add_widget(self.btn_pause)
        self.add_widget(self.btn_reset)
        self.add_widget(self.speed_spinner)

    def get_params(self):
        C1 = self.s_C1.value * 1e-9
        C2 = self.s_C2.value * 1e-9
        C3 = self.s_C3.value * 1e-9
        Cp = self.s_Cp.value * 1e-9
        Cs = self.s_Cs.value * 1e-9
        Lp = self.s_Lp.value * 1e-6
        Ls = self.s_Ls.value * 1e-6
        k = self.s_k.value
        R = self.s_R.value
        V = self.s_V.value
        T = self.s_T.value * 1e-6
        return C1, C2, C3, Lp, Cp, Ls, Cs, k, R, V, T


class ReadoutPanel(GridLayout):
    """
    Compact readout bar for:
      - t
      - vp, vs
      - ip, is
      - E_total
      and current speed mode.
    """

    t_text = StringProperty("t = 0.00 us")
    vp_text = StringProperty("vp = 0.00 V")
    vs_text = StringProperty("vs = 0.00 V")
    ip_text = StringProperty("ip = 0.00 A")
    is_text = StringProperty("is = 0.00 A")
    E_text = StringProperty("E = 0.00 J")
    speed_text = StringProperty("Speed: Normal")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 7
        self.padding = 2
        self.spacing = 4

        self.add_widget(Label(text=self.t_text, font_size=11))
        self.add_widget(Label(text=self.vp_text, font_size=11))
        self.add_widget(Label(text=self.vs_text, font_size=11))
        self.add_widget(Label(text=self.ip_text, font_size=11))
        self.add_widget(Label(text=self.is_text, font_size=11))
        self.add_widget(Label(text=self.E_text, font_size=11))
        self.add_widget(Label(text=self.speed_text, font_size=11))

        # Keep references to labels for manual updates
        self.lbls = self.children[::-1]  # children is reversed

    def update_values(self, t, vp, vs, ip, is_, E_total, speed_mode):
        self.t_text = f"t = {t * 1e6:6.2f} us"
        self.vp_text = f"vp = {vp:7.3f} V"
        self.vs_text = f"vs = {vs:7.3f} V"
        self.ip_text = f"ip = {ip:7.3e} A"
        self.is_text = f"is = {is_:7.3e} A"
        self.E_text = f"E = {E_total:7.3e} J"
        self.speed_text = f"Speed: {speed_mode}"

        # Update label texts
        self.lbls[0].text = self.t_text
        self.lbls[1].text = self.vp_text
        self.lbls[2].text = self.vs_text
        self.lbls[3].text = self.ip_text
        self.lbls[4].text = self.is_text
        self.lbls[5].text = self.E_text
        self.lbls[6].text = self.speed_text


# ==============================
# 3. Root layout and app
# ==============================

class LivingBatteryRoot(BoxLayout):
    def __init__(self, simulator, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.simulator = simulator

        # -------------------
        # Top: Toggle + views
        # -------------------
        top_container = BoxLayout(orientation="vertical", size_hint_y=0.40)
        self.add_widget(top_container)

        # Toggle row
        toggle_row = BoxLayout(orientation="horizontal", size_hint_y=0.18, padding=4, spacing=4)
        self.btn_field = ToggleButton(text="Field View", state="down")
        self.btn_circuit = ToggleButton(text="Circuit View", state="down")
        toggle_row.add_widget(Label(text="View:", size_hint_x=0.3))
        toggle_row.add_widget(self.btn_field)
        toggle_row.add_widget(self.btn_circuit)

        top_container.add_widget(toggle_row)

        # Views row
        views_row = BoxLayout(orientation="horizontal", size_hint_y=0.82, spacing=4, padding=4)
        self.energy_diagram = EnergyDiagramWidget()
        self.circuit_diagram = CircuitDiagramWidget()
        views_row.add_widget(self.energy_diagram)
        views_row.add_widget(self.circuit_diagram)

        top_container.add_widget(views_row)

        # Bind toggle behavior
        self.btn_field.bind(on_release=self._update_view_visibility)
        self.btn_circuit.bind(on_release=self._update_view_visibility)

        # -------------------
        # Middle: controls
        # -------------------
        self.controls = ControlsPanel(size_hint_y=0.34)
        self.add_widget(self.controls)

        # Readout bar
        self.readouts = ReadoutPanel(size_hint_y=0.08)
        self.add_widget(self.readouts)

        # Bottom: graphs
        self.graph = GraphWidget(size_hint_y=0.18)
        self.add_widget(self.graph)

        # Control state
        self.running = False
        self._clock_event = None

        # Wire buttons
        self.controls.btn_start.bind(on_release=self.on_start)
        self.controls.btn_pause.bind(on_release=self.on_pause)
        self.controls.btn_reset.bind(on_release=self.on_reset)

        # Initial visibility
        self._update_view_visibility()

    def _update_view_visibility(self, *args):
        # If off, shrink to almost zero width
        if self.btn_field.state == "down":
            self.energy_diagram.size_hint_x = 1
            self.energy_diagram.opacity = 1
        else:
            self.energy_diagram.size_hint_x = 0.0001
            self.energy_diagram.opacity = 0

        if self.btn_circuit.state == "down":
            self.circuit_diagram.size_hint_x = 1
            self.circuit_diagram.opacity = 1
        else:
            self.circuit_diagram.size_hint_x = 0.0001
            self.circuit_diagram.opacity = 0

    def on_start(self, *args):
        if not self.running:
            self.running = True
            if self._clock_event is None:
                self._clock_event = Clock.schedule_interval(self.update, 1 / 60.0)

    def on_pause(self, *args):
        self.running = False

    def on_reset(self, *args):
        # Apply slider parameters to simulator and reset
        C1, C2, C3, Lp, Cp, Ls, Cs, k, R, V, T = self.controls.get_params()
        self.simulator.set_params(C1, C2, C3, Lp, Cp, Ls, Cs, k, R, V, T)
        # Clear graph traces
        self.graph.trace1.clear()
        self.graph.trace2.clear()
        self.graph.trace3.clear()

    def _steps_for_speed(self):
        mode = self.controls.speed_mode
        if mode == "Slow":
            return 5
        if mode == "Overdrive":
            return 50
        return 10  # Normal

    def update(self, dt):
        if not self.running:
            return

        steps = self._steps_for_speed()
        result = self.simulator.step(steps=steps)

        vp = result["vp"]
        vs = result["vs"]
        ip = result["ip"]
        is_ = result["is"]
        E_total = result["E_total"]
        E_Cp = result["E_Cp"]
        E_Cs = result["E_Cs"]
        E_Lp = result["E_Lp"]
        E_Ls = result["E_Ls"]
        t = result["t"]

        # Update field view
        self.energy_diagram.vp = vp
        self.energy_diagram.vs = vs
        self.energy_diagram.ip = ip
        self.energy_diagram.is_ = is_

        # Update circuit schematic view (values + overlays)
        self.circuit_diagram.vp = vp
        self.circuit_diagram.vs = vs
        self.circuit_diagram.ip = ip
        self.circuit_diagram.is_ = is_
        self.circuit_diagram.E_Cp = E_Cp
        self.circuit_diagram.E_Cs = E_Cs
        self.circuit_diagram.E_Lp = E_Lp
        self.circuit_diagram.E_Ls = E_Ls
        self.circuit_diagram.E_total = E_total

        # Update graph traces
        self.graph.add_sample(vp, vs, E_total)

        # Update readouts
        self.readouts.update_values(t, vp, vs, ip, is_, E_total,
                                    self.controls.speed_mode)


class LivingBatteryApp(App):
    def build(self):
        self.title = "Living Battery – Nested Capacitors + Coupled LC (Kivy GUI)"
        simulator = LivingBatterySimulator()
        root = LivingBatteryRoot(simulator)
        return root


if __name__ == "__main__":
    LivingBatteryApp().run()

