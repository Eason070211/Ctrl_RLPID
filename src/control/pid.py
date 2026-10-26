"""Discrete PID controller with anti-windup and derivative filtering.

Error convention: e = T_set - T_measured.  Output u is signed TEC command in
[-1, 1] (u>0 cooling).  The derivative acts on the *measurement* to avoid
derivative kick on setpoint changes; the integral is clamped (anti-windup).
"""
from __future__ import annotations
import numpy as np


class DiscretePID:
    def __init__(self, Kp=1.2, Ki=0.25, Kd=0.05, dt=1.0,
                 out_min=-1.0, out_max=1.0, i_limit=3.0, d_alpha=0.15):
        self.set_gains(Kp, Ki, Kd)
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
        self.i_limit = i_limit
        self.d_alpha = d_alpha          # derivative low-pass factor
        self.reset()

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)

    def reset(self):
        self.integral = 0.0
        self.e_prev = 0.0
        self.d_filt = 0.0
        self.u_prev = 0.0

    # ------------------------------------------------------------------
    def step(self, T_set, T_meas):
        # Error convention matches the actuator: u>0 = cooling.
        # e = T_meas - T_set : too hot -> positive error -> cooling u>0.
        e = T_meas - T_set
        self.integral += e * self.dt
        self.integral = float(np.clip(self.integral, -self.i_limit,
                                       self.i_limit))
        # derivative on measurement (avoid kick on setpoint changes)
        d_raw = (T_meas - self.e_prev) / self.dt if self.dt > 0 else 0.0
        self.e_prev = T_meas
        self.d_filt = (1 - self.d_alpha) * self.d_filt + self.d_alpha * d_raw

        u = self.Kp * e + self.Ki * self.integral + self.Kd * self.d_filt
        u = float(np.clip(u, self.out_min, self.out_max))
        # back-calculate anti-windup: if saturated, pull integral back
        if u >= self.out_max and e > 0:
            self.integral -= e * self.dt
        elif u <= self.out_min and e < 0:
            self.integral -= e * self.dt
        self.u_prev = u
        return u
