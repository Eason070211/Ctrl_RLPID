"""Closed-loop performance metrics."""
from __future__ import annotations
import numpy as np


def compute_metrics(t, T, T_set, u, energy, dt=1.0,
                    ramp_end_s=150.0):
    """Report both ramp-phase and steady-state metrics separately.

    Settle band is a fixed ±2 °C absolute (not 2% of step) so it does not
    saturate for large setpoint changes.
    """
    t = np.asarray(t); T = np.asarray(T); T_set = np.asarray(T_set)
    u = np.asarray(u)
    e = T - T_set

    def block(es, ts):
        return dict(
            IAE=float(np.sum(np.abs(es)) * dt),
            MAE=float(np.mean(np.abs(es))),
            RMSE=float(np.sqrt(np.mean(es ** 2))),
        )

    ramp = t <= ramp_end_s
    steady = t > ramp_end_s
    m_ramp = block(e[ramp], t[ramp])
    m_ss = block(e[steady], t[steady])

    # overshoot relative to first setpoint change
    Tf = T_set[-1]
    overshoot = float(max(0.0, (T.max() - Tf) / max(abs(Tf - T[0]), 1e-6) * 100.0))
    # rise time 10-90%
    try:
        lo = T[0] + 0.1 * (Tf - T[0]); hi = T[0] + 0.9 * (Tf - T[0])
        i10 = int(np.argmax(T >= lo)); i90 = int(np.argmax(T >= hi))
        rise = float((i90 - i10) * dt) if i90 > i10 else np.nan
    except Exception:
        rise = np.nan
    # settle time: first index after which |e| stays within ±2 C forever
    band = 2.0
    settle = float(t[-1])
    for i in range(len(T)):
        if np.all(np.abs(e[i:]) <= band):
            settle = float(t[i]); break

    return dict(
        IAE=m_ramp["IAE"] + m_ss["IAE"],
        MAE=m_ramp["MAE"],
        MAE_steady=m_ss["MAE"],
        RMSE=m_ss["RMSE"],
        overshoot_pct=overshoot,
        rise_time_s=rise,
        settle_time_s=settle,
        energy_kJ=float(np.sum(energy) / 1000.0),
        control_effort=float(np.sum(np.abs(np.diff(u))) * dt),
        violate_s=float(np.sum(np.abs(e) > 3.0) * dt),
        T_final=float(T[-1]),
    )
