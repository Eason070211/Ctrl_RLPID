"""Traditional baseline controllers.

All controllers share::

    ctl.reset()
    u = ctl.act(obs: dict)    # obs has T, T_set, t, T_air, ...

u is signed TEC command in [-1, 1] (u<0 heats, u>0 cools).
"""
from __future__ import annotations
import numpy as np
from .pid import DiscretePID


class BaseController:
    name = "base"

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError


class FixedPID(BaseController):
    name = "fixed_pid"

    def __init__(self, gains, pid_kwargs=None):
        self.gains = gains
        self.pid = DiscretePID(**(pid_kwargs or {}))
        self.reset()

    def reset(self):
        self.pid.set_gains(*self.gains)
        self.pid.reset()

    def act(self, obs):
        return self.pid.step(obs["T_set"], obs["T"])


class BangBang(BaseController):
    """Fastest on/off control with deadband hysteresis."""
    name = "bang_bang"

    def __init__(self, deadband=0.5):
        self.deadband = deadband
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def act(self, obs):
        e = obs["T_set"] - obs["T"]
        if e > self.deadband:
            self.state = -1.0      # too cold -> heat
        elif e < -self.deadband:
            self.state = +1.0      # too hot -> cool
        return self.state


class StagedPID(BaseController):
    """Fast heat -> thermal-inertia coast -> active brake -> steady PID.

    Encodes the user's idea: use the plant's thermal mass to coast toward the
    setpoint instead of immediately reversing (energy saving), and brake early.
    """
    name = "staged_pid"

    def __init__(self, gains, pid_kwargs=None, near_band=2.0, coast_margin=1.2):
        self.pid = DiscretePID(**(pid_kwargs or {}))
        self.gains = gains
        self.near_band = near_band
        self.coast_margin = coast_margin
        self.T_prev = None
        self.reset()

    def reset(self):
        self.pid.set_gains(*self.gains)
        self.pid.reset()
        self.T_prev = None

    def act(self, obs):
        T, T_set = obs["T"], obs["T_set"]
        dt = obs.get("dt", 1.0)
        e = T_set - T
        dT = 0.0 if self.T_prev is None else (T - self.T_prev) / dt
        self.T_prev = T

        if abs(e) < self.near_band:
            return self.pid.step(T_set, T)          # steady PID
        # far from setpoint: drive hard
        if e > 0:
            # need heating: estimate coast time; if inertia carries us past,
            # cut power and coast (no energy spent)
            if dT > 1e-4 and e / max(dT, 1e-6) < self.coast_margin * dt:
                return 0.0                          # coast on thermal inertia
            return -1.0 if e > self.near_band * 2 else self.pid.step(T_set, T)
        else:
            return +1.0 if -e > self.near_band * 2 else self.pid.step(T_set, T)


class GainScheduledPID(BaseController):
    """Human-designed gain scheduling by error band (no RL)."""
    name = "gain_scheduled_pid"

    def __init__(self, pid_kwargs=None):
        # wide-band aggressive gains, narrow-band soft gains
        self.bands = [(0.0, 2.0, (2.0, 0.10, 0.02)),     # near setpoint
                      (2.0, 1e9, (0.8, 0.40, 0.05))]      # far: gentler Ki
        self.pid = DiscretePID(**(pid_kwargs or {}))
        self.reset()

    def reset(self):
        self.pid.reset()

    def act(self, obs):
        e = abs(obs["T_set"] - obs["T"])
        g = self.bands[-1][2]
        for lo, hi, gains in self.bands:
            if lo <= e < hi:
                g = gains
                break
        self.pid.set_gains(*g)
        return self.pid.step(obs["T_set"], obs["T"])
