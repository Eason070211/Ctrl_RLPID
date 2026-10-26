"""Train 3 direct-PPO variants with different reward weightings.

A: speed-heavy   (w_temp=2.5, w_energy=0.02, w_time=0.15)
B: energy-heavy  (w_temp=0.5, w_energy=0.60, w_time=0.0)
C: balanced      (w_temp=1.0, w_energy=0.15, w_time=0.03)
"""
import os, sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from src.config import load_config
from src.env.tec_env import TECEnv, Scenario

VARIANTS = {
    "speed":   dict(w_temp=2.5, w_energy=0.02, w_time=0.15),
    "energy":  dict(w_temp=0.5, w_energy=0.60, w_time=0.0),
    "balance": dict(w_temp=1.0, w_energy=0.15, w_time=0.03),
}


def _make(tag, seed):
    ov = VARIANTS[tag]
    def _init():
        return TECEnv(load_config(), mode="direct",
                      scenario=Scenario(seed=seed), seed=seed,
                      reward_override=ov)
    return _init


def main():
    cfg = load_config(); tr = cfg["train"]
    for tag, ov in VARIANTS.items():
        print(f"\n=== training direct PPO, reward={tag} {ov} ===")
        venv = make_vec_env(_make(tag, 42), n_envs=tr["n_envs"])
        model = PPO("MlpPolicy", venv,
                    learning_rate=tr["lr"], n_steps=tr["n_steps"],
                    batch_size=tr["batch_size"], n_epochs=tr["n_epochs"],
                    gamma=tr["gamma"], gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, verbose=0, seed=42, device="cpu",
                    policy_kwargs=dict(net_arch=[256, 256]))
        model.learn(total_timesteps=300000)
        path = f"results/models/ppo_direct_{tag}.zip"
        model.save(path)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
