"""Grid-search the best fixed PID gains, then re-benchmark with multiple seeds."""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.thermal.heat_solver import PlatePlant
from src.control.baselines import FixedPID
from src.benchmark.metrics import compute_metrics

cfg = load_config(); c = cfg["control"]

def run_pid(gains, Tair=25.0, v=0.5, rh=0.4, Tset=200.0, seed=0):
    p = PlatePlant(cfg); p.reset(25.0, Tair)
    pid = FixedPID(gains, dict(dt=1, out_min=-1, out_max=1, i_limit=3))
    ts, Ts, us, Es = [], [], [], []
    for k in range(800):
        u = pid.act({"T": p.observed_temperature(), "T_set": Tset, "dt": 1.0})
        T = p.step(u, Tair, v, rh, 1.0)
        ts.append(k); Ts.append(T); us.append(u); Es.append(p.p_electric)
    return compute_metrics(ts, Ts, np.full_like(Ts, Tset), us, Es)

print("=== PID grid search (nominal) ===")
best = None
for Kp in [0.3, 0.6, 1.0, 1.5, 2.0]:
    for Ki in [0.02, 0.05, 0.1, 0.2]:
        for Kd in [0.0, 0.02, 0.05]:
            m = run_pid((Kp, Ki, Kd))
            score = m["MAE_steady"] + 0.01*m["energy_kJ"]
            if best is None or score < best[0]:
                best = (score, (Kp, Ki, Kd), m)
            print(f"  Kp={Kp} Ki={Ki} Kd={Kd}: MAE_ss={m['MAE_steady']:.3f} "
                  f"over={m['overshoot_pct']:.1f}% E={m['energy_kJ']:.1f}")
print(f"\nBest: Kp={best[1][0]} Ki={best[1][1]} Kd={best[1][2]}")
print(f"  MAE_ss={best[2]['MAE_steady']:.3f} over={best[2]['overshoot_pct']:.1f}% "
      f"E={best[2]['energy_kJ']:.1f}kJ")
