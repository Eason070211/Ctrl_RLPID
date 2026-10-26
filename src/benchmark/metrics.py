"""Closed-loop performance metrics."""
from __future__ import annotations
import numpy as np


def compute_metrics(t, T, T_set, u, energy, dt=1.0):
    t = np.asarray(t); T = np.asarray(T); T_set = np.asarray(T_set)
    u = np.asarray(u)
    e = T - T_set
    IAE = float(np.sum(np.abs(e)) * dt)
    ITAE = float(np.sum(np.abs(e) * t) * dt)
    RMSE = float(np.sqrt(np.mean(e ** 2)))
    MAE = float(np.mean(np.abs(e)))

    # step response to the *final* setpoint (assume step at t=0)
    Tss = float(np.mean(T[-int(len(T)*0.1):]))
    Tf = T_set[-1]
    overshoot = float(max(0.0, (T.max() - Tf) / max(abs(Tf - T[0]), 1e-6) * 100.0))
    # rise time 10-90%
    try:
        lo = T[0] + 0.1 * (Tf - T[0]); hi = T[0] + 0.9 * (Tf - T[0])
        i10 = np.argmax(T >= lo); i90 = np.argmax(T >= hi)
        rise = float((i90 - i10) * dt) if i90 > i10 else np.nan
    except Exception:
        rise = np.nan
    # settling time: time after which |e| stays within 2% of step
    band = 0.02 * max(abs(Tf - T[0]), 1e-6)
    settle = float(t[-1])
    for i in range(len(T)):
        if np.all(np.abs(e[i:]) <= band):
            settle = float(t[i]); break
    control_effort = float(np.sum(np.abs(np.diff(u))) * dt)
    energy_kJ = float(np.sum(energy) / 1000.0)
    violate = float(np.sum(np.abs(e) > 3.0) * dt)
    return dict(IAE=IAE, ITAE=ITAE, RMSE=RMSE, MAE=MAE,
                overshoot_pct=overshoot, rise_time_s=rise,
                settle_time_s=settle, energy_kJ=energy_kJ,
                control_effort=control_effort, violate_s=violate,
                T_final=float(T[-1]))
