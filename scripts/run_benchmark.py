"""Run the full benchmark: traditional controllers + trained PPO agents."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.config import load_config
from src.benchmark.runner import (build_controllers, run_traditional, run_rl,
                                   scenario_nominal, scenario_hot_air_step,
                                   scenario_setpoint_step,
                                   scenario_humidity, scenario_windspeed)
from src.benchmark.metrics import compute_metrics


def safe_load(mode, path):
    if not os.path.exists(path):
        print(f"[bench] model missing: {path}")
        return None
    from stable_baselines3 import PPO
    return PPO.load(path)


def main():
    cfg = load_config()
    rows = []
    traj = {}
    ctrls = build_controllers(cfg)
    rl_modes = {
        "ppo_direct": safe_load("direct", "results/models/ppo_direct.zip"),
        "ppo_fast_pid": safe_load("fast_pid", "results/models/ppo_fast_pid.zip"),
        "ppo_event_pid": safe_load("event_pid",
                                   "results/models/ppo_event_pid.zip"),
        "ppo_bang_rl": safe_load("bang_rl", "results/models/ppo_bang_rl.zip"),
    }

    scenarios = {
        "nominal": scenario_nominal,
        "hot_air_step": scenario_hot_air_step,
        "setpoint_step": scenario_setpoint_step,
    }
    for rh in cfg["benchmark"]["humidity_sweep"]:
        scenarios[f"RH{int(rh*100)}"] = scenario_humidity(rh)
    for v in [0.0, 1.0, 2.0, 3.0]:
        scenarios[f"wind{v:.1f}"] = scenario_windspeed(v)

    for sc_name, scen in scenarios.items():
        print(f"\n=== scenario: {sc_name} ===")
        for name, factory in ctrls.items():
            out = run_traditional(factory, scen, cfg)
            m = compute_metrics(out["t"], out["T"], out["T_set"], out["u"],
                                out["energy"])
            m.update(controller=name, scenario=sc_name)
            rows.append(m)
            if sc_name == "nominal":
                traj[name] = out
            print(f"  {name:18s} MAE_ss={m['MAE_steady']:.3f} IAE={m['IAE']:.0f} "
                  f"E={m['energy_kJ']:.1f}kJ over={m['overshoot_pct']:.1f}%")
        for name, model in rl_modes.items():
            if model is None:
                continue
            mode = name.replace("ppo_", "")
            out = run_rl(model, mode, scen, cfg)
            m = compute_metrics(out["t"], out["T"], out["T_set"], out["u"],
                                out["energy"])
            m.update(controller=name, scenario=sc_name)
            rows.append(m)
            if sc_name == "nominal":
                traj[name] = out
            print(f"  {name:18s} MAE_ss={m['MAE_steady']:.3f} IAE={m['IAE']:.0f} "
                  f"E={m['energy_kJ']:.1f}kJ over={m['overshoot_pct']:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/benchmark.csv", index=False)
    print(f"\n[bench] wrote results/tables/benchmark.csv ({len(df)} rows)")

    # ---- plot: nominal trajectories ----
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, out in traj.items():
        axes[0].plot(out["t"], out["T"], label=name, lw=1.2)
        axes[1].plot(out["t"], out["u"], label=name, lw=0.8, alpha=0.8)
    axes[0].axhline(200, ls="--", c="k", lw=0.8, label="setpoint")
    axes[0].set_ylabel("T [°C]"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("u (signed)"); axes[1].set_xlabel("t [s]")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Nominal scenario: temperature and control effort")
    fig.tight_layout(); fig.savefig("results/figures/nominal_traj.png", dpi=130)
    plt.close(fig)

    # ---- humidity robustness: MAE vs RH ----
    hum = df[df.scenario.str.startswith("RH")].copy()
    hum["RH"] = hum.scenario.str.extract(r"RH(\d+)").astype(float) / 100.0
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, g in hum.groupby("controller"):
        g = g.sort_values("RH")
        ax.plot(g.RH, g.MAE_steady, marker="o", label=name)
    ax.set_xlabel("Relative humidity"); ax.set_ylabel("MAE_steady [°C]")
    ax.set_title("Humidity robustness: MAE vs RH"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("results/figures/humidity_sweep.png", dpi=130)
    plt.close(fig)

    # ---- wind speed robustness: MAE vs v_air ----
    wind = df[df.scenario.str.startswith("wind")].copy()
    wind["v"] = wind.scenario.str.extract(r"wind([0-9.]+)").astype(float)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, g in wind.groupby("controller"):
        g = g.sort_values("v")
        ax.plot(g.v, g.MAE_steady, marker="s", label=name)
    ax.set_xlabel("Air speed [m/s]"); ax.set_ylabel("MAE_steady [°C]")
    ax.set_title("Wind robustness: MAE vs v_air"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("results/figures/wind_sweep.png", dpi=130)
    plt.close(fig)

    # ---- table summary (nominal + hot air) ----
    summary = df[df.scenario.isin(["nominal", "hot_air_step"])].pivot_table(
        index="controller", columns="scenario",
        values=["MAE_steady", "IAE", "energy_kJ", "overshoot_pct", "settle_time_s"])
    summary.to_csv("results/tables/summary.csv")
    print("[bench] wrote results/tables/summary.csv")
    print(summary.round(2))


if __name__ == "__main__":
    main()
