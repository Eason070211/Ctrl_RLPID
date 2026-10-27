"""Run all controllers over scenarios and collect metrics."""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.config import load_config
from src.thermal.heat_solver import PlatePlant
from src.control.baselines import (FixedPID, BangBang, StagedPID,
                                    GainScheduledPID)
from src.env.tec_env import TECEnv
from src.benchmark.metrics import compute_metrics


# ---------------------------------------------------------------------------
def scenario_nominal(t):
    return 25.0, 0.5, 0.4, 200.0

def scenario_hot_air_step(t):
    return (35.0 if t >= 300.0 else 25.0), 0.5, 0.4, 200.0

def scenario_setpoint_step(t):
    return 25.0, 0.5, 0.4, (220.0 if t >= 300.0 else 200.0)

def scenario_humidity(rh):
    def _f(t):
        return 25.0, 0.5, rh, 200.0
    return _f

def scenario_windspeed(v):
    def _f(t):
        return 25.0, v, 0.4, 200.0
    return _f


# ---------------------------------------------------------------------------
def run_traditional(ctl_factory, scenario, cfg, episode_time=None):
    c = cfg["control"]
    episode_time = episode_time or c["episode_time"]
    p = PlatePlant(cfg)
    T_air0, v, rh, Tset = scenario(0.0)
    p.reset(T_init=T_air0, T_air=T_air0)
    ctl = ctl_factory()
    dt = c["dt_ctrl"]
    n = int(episode_time / dt)
    ts, Ts, us, Es, Tset_hist = [], [], [], [], []
    for k in range(n):
        t = k * dt
        T_air, v, rh, Tset = scenario(t)
        T = p.observed_temperature()
        u = ctl.act({"T": T, "T_set": Tset, "dt": dt, "t": t})
        T = p.step(u, T_air, v, rh, dt)
        ts.append(t); Ts.append(T); us.append(u); Es.append(p.p_electric)
        Tset_hist.append(Tset)
    return dict(t=np.array(ts), T=np.array(Ts), T_set=np.array(Tset_hist),
                u=np.array(us), energy=np.array(Es))


def run_rl(model, mode, scenario, cfg, episode_time=None):
    c = cfg["control"]
    episode_time = episode_time or c["episode_time"]
    env = TECEnv(cfg, mode=mode, scenario=None, seed=0)
    env.fixed = dict(T_air0=scenario(0.0)[0], v=scenario(0.0)[1],
                      rh=scenario(0.0)[2], T_set=scenario(0.0)[3])
    env._bc = lambda: scenario(env.t)
    env.record = True
    env.traj = []
    env.reset()
    done = False
    while not done:
        obs = env._obs()
        a, _ = model.predict(obs, deterministic=True)
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
    traj = np.array(env.traj)
    return dict(t=traj[:, 0], T=traj[:, 1], T_set=traj[:, 2],
                u=traj[:, 3], energy=traj[:, 7])


# ---------------------------------------------------------------------------
def build_controllers(cfg):
    c = cfg["control"]
    pid_kw = dict(dt=c["dt_pid"], out_min=c["u_min"], out_max=c["u_max"],
                  i_limit=c["i_limit"])
    g = (c["Kp"], c["Ki"], c["Kd"])
    return {
        "fixed_pid": lambda: FixedPID(g, pid_kw),
        "bang_bang": lambda: BangBang(0.5),
        "staged_pid": lambda: StagedPID(g, pid_kw),
        "gain_scheduled": lambda: GainScheduledPID(pid_kw),
    }
