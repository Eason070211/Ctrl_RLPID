"""Gymnasium environment for TEC temperature control.

Three controller modes share the same plant and reward:

  * ``direct``   : PPO outputs the signed TEC command u in [-1,1] directly.
  * ``fast_pid`` : inner PID runs every ctrl step; PPO (upper loop, every
                   ``ppo_every_steps`` ctrl steps) outputs gain increments.
  * ``event_pid``: PID gains are locked; PPO acts only every
                   ``event_every_steps`` and a deadband implements "hold if
                   already good" (long-horizon event-triggered adaptation).

Every ``env.step`` advances the plant for ``inner_steps`` control intervals.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import load_config
from src.thermal.heat_solver import PlatePlant
from src.control.pid import DiscretePID


# ---------------------------------------------------------------------------
class Scenario:
    """Disturbance schedule: returns (T_air, v_air, RH, T_set) at time t."""

    def __init__(self, kind="nominal", seed=0):
        self.kind = kind
        self.rng = np.random.default_rng(seed)

    def sample_train(self):
        T_air0 = self.rng.uniform(15.0, 35.0)
        v = self.rng.uniform(0.0, 3.0)
        rh = self.rng.uniform(0.1, 0.9)
        T_set = self.rng.uniform(150.0, 220.0)
        return dict(T_air0=T_air0, v=v, rh=rh, T_set=T_set,
                    air_step=self.rng.uniform(15.0, 30.0),
                    air_step_delta=self.rng.choice([-5.0, 5.0, 10.0]))

    def at(self, t, s):
        T_air = s["T_air0"] + (s["air_step_delta"] if t >= s["air_step"] else 0.0)
        return T_air, s["v"], s["rh"], s["T_set"]


class TECEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict | None = None, mode: str = "direct",
                 scenario: Scenario | None = None, seed: int = 0,
                 fixed_scenario: dict | None = None):
        super().__init__()
        self.cfg = cfg or load_config()
        self.mode = mode
        self.scenario = scenario
        self.fixed = fixed_scenario
        self.c = self.cfg["control"]
        self.rl = self.cfg["rl"]
        self.rew_w = self.rl["reward"]

        self.dt_ctrl = self.c["dt_ctrl"]
        self.episode_time = self.c["episode_time"]

        # inner cadence
        if mode in ("direct", "bang_rl"):
            self.inner = 1
        elif mode == "fast_pid":
            self.inner = int(self.rl["ppo_every_steps"])
        elif mode == "event_pid":
            self.inner = int(self.rl["event_every_steps"])
        else:
            raise ValueError(mode)

        # PID object
        self.pid = DiscretePID(
            self.c["Kp"], self.c["Ki"], self.c["Kd"],
            dt=self.c["dt_pid"], out_min=self.c["u_min"],
            out_max=self.c["u_max"], i_limit=self.c["i_limit"])
        self.base_gains = np.array([self.c["Kp"], self.c["Ki"], self.c["Kd"]])
        self.gains = self.base_gains.copy()
        self.Kp_lo, self.Kp_hi = self.c["Kp_range"]
        self.Ki_lo, self.Ki_hi = self.c["Ki_range"]
        self.Kd_lo, self.Kd_hi = self.c["Kd_range"]

        # action / observation spaces
        if mode == "direct":
            self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        elif mode == "bang_rl":
            self.action_space = spaces.Discrete(3)  # 0=heat,1=coast,2=cool
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        n_obs = 7 + (3 if mode in ("fast_pid", "event_pid") else 0) \
                + (2 if mode == "event_pid" else 0)
        self.observation_space = spaces.Box(-10.0, 10.0, (n_obs,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.plant = PlatePlant(self.cfg)
        self.u_prev = 0.0
        self.dT_filt = 0.0
        self.t = 0.0
        self.health = 0.0
        self.since_update = 999
        self._r = None
        self.record = False
        self.traj = []

    # ------------------------------------------------------------------
    def _obs(self):
        T = self.plant.observed_temperature()
        T_air, v, rh, T_set = self._bc()
        o = [
            (T - T_set) / 30.0,
            (T - 25.0) / 100.0,
            self.dT_filt / 2.0,
            (T_air - 25.0) / 10.0,
            float(rh),
            float(v) / 3.0,
            self.u_prev,
        ]
        if self.mode in ("fast_pid", "event_pid"):
            o += [self.gains[0] / 2.0, self.gains[1] / 0.5, self.gains[2] / 0.1]
        if self.mode == "event_pid":
            o += [self.since_update / float(self.inner), self.health]
        return np.array(o, dtype=np.float32)

    def _bc(self):
        if self.fixed is not None:
            s = self.fixed
        else:
            s = self._train_s
        return self.scenario.at(self.t, s) if self.scenario else \
            (s["T_air0"], s["v"], s["rh"], s["T_set"])

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.scenario is not None:
            self._train_s = self.scenario.sample_train()
        T_air0 = self._train_s["T_air0"] if self.scenario else \
            (self.fixed or dict(T_air0=self.c["t_air"], v=self.c["v_air"],
                                rh=self.c["RH"], T_set=self.c["t_setpoint"]))["T_air0"]
        self.plant.reset(T_init=T_air0, T_air=T_air0)
        self.gains = self.base_gains.copy()
        self.pid.set_gains(*self.gains)
        self.pid.reset()
        self.u_prev = 0.0
        self.dT_filt = 0.0
        self.t = 0.0
        self.health = 0.0
        self.since_update = 999
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _decode_u(self, action):
        if self.mode == "direct":
            return float(np.clip(action[0], -1, 1))
        if self.mode == "bang_rl":
            return [-1.0, 0.0, 1.0][int(action)]
        return None  # PID modes: inner loop calls self.pid

    def _apply_action(self, action):
        if self.mode in ("direct", "bang_rl"):
            return
        # delta gains
        dp, di, dd = action
        new = self.gains + np.array([
            self.c["dKp_scale"] * dp,
            self.c["dKi_scale"] * di,
            self.c["dKd_scale"] * dd])
        new[0] = float(np.clip(new[0], self.Kp_lo, self.Kp_hi))
        new[1] = float(np.clip(new[1], self.Ki_lo, self.Ki_hi))
        new[2] = float(np.clip(new[2], self.Kd_lo, self.Kd_hi))
        if self.mode == "event_pid":
            if np.max(np.abs(action)) < 0.30:
                new = self.gains.copy()
            else:
                self.since_update = 0
        self.gains = new
        self.pid.set_gains(*new)

    def step(self, action):
        self._apply_action(action)
        w = self.rew_w
        sub_rew = 0.0
        last_T = self.plant.observed_temperature()
        for _ in range(self.inner):
            T_air, v, rh, T_set = self._bc()
            if self.mode in ("direct", "bang_rl"):
                u = self._decode_u(action)
            else:
                u = self.pid.step(T_set, self.plant.observed_temperature())
            T = self.plant.step(u, T_air, v, rh, self.dt_ctrl)
            self.dT_filt = 0.9 * self.dT_filt + 0.1 * (T - last_T) / self.dt_ctrl
            last_T = T
            e = (T - T_set) / 30.0
            E = self.plant.p_electric / (self.dt_ctrl * 4.0 * 40.0)
            sub = -w["w_temp"] * e * e \
                  - w["w_energy"] * E * E \
                  - w["w_du"] * (u - self.u_prev) ** 2
            band = 5.0 if self.mode == "bang_rl" else 3.0
            if abs(T - T_set) > band:
                sub -= w["violate_penalty"]
            sub_rew += sub
            if self.record:
                g = self.gains
                self.traj.append((self.t, T, T_set, u,
                                  float(g[0]), float(g[1]), float(g[2]),
                                  self.plant.p_electric))
            self.u_prev = u
            self.t += self.dt_ctrl
        if self.mode not in ("direct", "bang_rl"):
            dK = np.abs(action).mean()
            sub_rew -= w["w_dK"] * dK
        R = sub_rew / self.inner
        self.health = np.clip(abs(last_T - T_set) / 10.0, 0, 2)
        self.since_update += self.inner
        truncated = self.t >= self.episode_time
        info = {"T": last_T, "T_set": T_set, "t": self.t,
                "u": self.u_prev, "gains": self.gains.copy(),
                "energy": self.plant.p_electric * self.inner,
                "dT_filt": self.dT_filt}
        return self._obs(), R, False, truncated, info
