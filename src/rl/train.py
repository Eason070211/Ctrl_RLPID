"""Train PPO agents for the three control modes with stable-baselines3."""
from __future__ import annotations
import os, sys, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from src.config import load_config
from src.env.tec_env import TECEnv, Scenario


def _make_env(mode, seed):
    def _init():
        return TECEnv(load_config(), mode=mode, scenario=Scenario(seed=seed),
                      seed=seed)
    return _init


class _Progress(BaseCallback):
    def __init__(self, total):
        super().__init__(); self.total = total; self.t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 < 2048:
            el = time.time() - self.t0
            sps = self.num_timesteps / max(el, 1)
            print(f"  [{self.model.__class__.__name__}] {self.num_timesteps}/{self.total} "
                  f"steps, {sps:.0f} st/s, {el/60:.1f} min")
        return True


def train(mode: str, total_steps: int = None, seed: int = 42,
          out_dir: str = "results/models"):
    cfg = load_config()
    tr = cfg["train"]
    total_steps = total_steps or tr["total_steps"]
    os.makedirs(out_dir, exist_ok=True)

    venv = make_vec_env(_make_env(mode, seed), n_envs=tr["n_envs"])
    model = PPO(
        "MlpPolicy", venv,
        learning_rate=tr["lr"],
        n_steps=tr["n_steps"],
        batch_size=tr["batch_size"],
        n_epochs=tr["n_epochs"],
        gamma=tr["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
        device="cpu",
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    print(f"[train] mode={mode}, total={total_steps}, envs={tr['n_envs']}")
    model.learn(total_timesteps=total_steps, callback=_Progress(total_steps))
    path = os.path.join(out_dir, f"ppo_{mode}.zip")
    model.save(path)
    print(f"[train] saved {path}")
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["direct", "fast_pid", "event_pid", "bang_rl"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.mode, args.steps, args.seed)
