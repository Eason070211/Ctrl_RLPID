"""Zoomed comparison: fixed PID vs ppo_fast_pid, rise + steady detail."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from src.config import load_config
from src.env.tec_env import TECEnv
from src.thermal.heat_solver import PlatePlant
from src.control.baselines import FixedPID

cfg = load_config()


def run_pid(v=3.0):
    p = PlatePlant(cfg); p.reset(25, 25)
    pid = FixedPID((0.6, 0.1, 0.02), dict(dt=1, out_min=-1, out_max=1, i_limit=3))
    ts, Ts, us = [], [], []
    for k in range(400):
        u = pid.act({"T": p.observed_temperature(), "T_set": 200.0, "dt": 1.0})
        T = p.step(u, 25.0, v, 0.4, 1.0)
        ts.append(k); Ts.append(T); us.append(u)
    return np.array(ts), np.array(Ts), np.array(us)


def run_rl(v=3.0):
    m = PPO.load("results/models/ppo_fast_pid.zip")
    env = TECEnv(cfg, mode="fast_pid", scenario=None, seed=0)
    env.fixed = dict(T_air0=25, v=v, rh=0.4, T_set=200)
    env._bc = lambda: (25.0, v, 0.4, 200.0)
    env.record = True; env.traj = []; env.reset()
    done = False
    while not done:
        a, _ = m.predict(env._obs(), deterministic=True)
        _, _, term, trunc, _ = env.step(a); done = term or trunc
    tr = np.array(env.traj)
    return tr[:, 0], tr[:, 1], tr[:, 3]


# ---- strong wind v=3: this is where RL wins ----
tp, Tp, up = run_pid(v=3.0)
tr, Tr, ur = run_rl(v=3.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: rise 0-150s
ax = axes[0]
ax.plot(tp, Tp, label="fixed PID", color="C1", lw=1.8)
ax.plot(tr, Tr, label="PPO-fast-PID", color="C5", lw=1.8, ls="--")
ax.axhline(200, color="k", ls=":", lw=0.8)
ax.axhspan(198, 202, alpha=0.12, color="g")
ax.set_xlim(0, 150); ax.set_ylim(20, 210)
ax.set_xlabel("t [s]"); ax.set_ylabel("T [°C]")
ax.set_title("(a) Rise 25→200°C, v=3 m/s")
ax.legend(loc="lower right"); ax.grid(True, which="both", alpha=0.3)

# Panel B: steady detail 200-300s (wind cooling causes PID to sag)
ax = axes[1]
mask_p = (tp >= 200) & (tp <= 300)
mask_r = (tr >= 200) & (tr <= 300)
ax.plot(tp[mask_p], Tp[mask_p], label="fixed PID", color="C1", lw=1.8, marker="o", ms=3)
ax.plot(tr[mask_r], Tr[mask_r], label="PPO-fast-PID", color="C5", lw=1.8, ls="--", marker="s", ms=3)
ax.axhline(200, color="k", ls=":", lw=0.8)
ax.axhspan(198, 202, alpha=0.12, color="g", label="±2°C band")
ax.set_xlim(200, 300); ax.set_ylim(195, 202)
ax.set_xlabel("t [s]"); ax.set_ylabel("T [°C]")
ax.set_title("(b) Steady detail (wind cools plate), v=3 m/s")
ax.legend(loc="lower right"); ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Zoomed comparison: fixed PID vs PPO-fast-PID (strong wind v=3 m/s)")
fig.tight_layout()
fig.savefig("results/figures/zoomed_comparison.png", dpi=150)
print("saved results/figures/zoomed_comparison.png")
#（注：内容由AI生成）
