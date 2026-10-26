"""2-D transient heat-conduction plant for a thin copper plate with a 2x2
array of thermoelectric coolers (TECs), convective + evaporative top cooling.

The plate is thin (2 mm) and high-conductivity (copper); the lateral 2-D field
is resolved on a structured Cartesian grid with explicit Euler time stepping
(Numba-JIT).  Each of the four TEC patches drives the local plate temperature
with the standard Peltier energy balance, solved quasi-statically for the hot
side.  The command ``u`` is a single signed current fraction in [-1, 1];
u > 0 cools the plate, u < 0 heats it (reverse polarity).

Physics
-------
rho cp Lz dT/dt = k Lz laplacian(T) + q_tec/A_patch - h_c (T-T_air) - q_evap

q_evap (wet-surface, humidity dependent) = h_evap * (1 - RH) * max(0, T-T_air)
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Patch layout: 2x2 TEC modules, each over a 2x2 cell block.
# Grid is (ny, nx); default nx = ny = 11 cells -> dx = dy = 0.01 m.
# ---------------------------------------------------------------------------
def _patch_indices(nx: int, ny: int):
    """Return list of (slice_y, slice_x) patch footprints (four quadrants)."""
    cx = nx // 2
    cy = ny // 2
    # 2x2 cell blocks around the four quadrant centres
    return [
        (slice(cy - 2, cy),     slice(cx - 2, cx)),      # SW
        (slice(cy - 2, cy),     slice(cx, cx + 2)),      # SE
        (slice(cy, cy + 2),     slice(cx - 2, cx)),      # NW
        (slice(cy, cy + 2),     slice(cx, cx + 2)),      # NE
    ]


@njit(cache=True, fastmath=True)
def _tec_heat_load(T_c_local, I, T_air, alpha, R, K, G_hs):
    """Quasi-steady Peltier heat-pump absorbed at the cold side [W/module]."""
    denom = K + G_hs - alpha * I
    T_h = (K * T_c_local + G_hs * T_air + 0.5 * I * I * R) / denom
    q_c = alpha * I * T_c_local - 0.5 * I * I * R - K * (T_h - T_c_local)
    P_el = alpha * I * (T_h - T_c_local) + I * I * R
    return q_c, P_el, T_h


@njit(cache=True, fastmath=True)
def step_plate(T, dx, dy, dt, Lz, rho_cp, k,
               patch_mask,
               I, T_air, v_air, RH,
               h_wind, h_evap,
               P_heat_max, P_cool_max):
    """Advance the plate field one substep.

    Parameters follow the module docstring.  ``patch_mask`` is (ny,nx) int,
    1 where a TEC patch sits (each contiguous block = one module).
    Returns (new_T, q_total_w, p_electric_total_w).
    """
    ny, nx = T.shape
    lap = np.zeros_like(T)
    q_out = np.zeros_like(T)
    # interior diffusion (5-point stencil, Neumann-insulated borders)
    for j in range(ny):
        jm = j - 1 if j > 0 else j
        jp = j + 1 if j < ny - 1 else j
        for i in range(nx):
            im = i - 1 if i > 0 else i
            ip = i + 1 if i < nx - 1 else i
            lap[j, i] = ((T[j, ip] - 2.0 * T[j, i] + T[j, im]) / (dx * dx)
                         + (T[jp, i] - 2.0 * T[j, i] + T[jm, i]) / (dy * dy))

    # ---- Heat loss model (W/m^2 of plate area), temperature-dependent ----
    # 1) Forced convection: h_nat baseline + wind-slope
    # 2) Natural convection Churchill-Chu (horizontal plate, laminar)
    #    Nu = 0.54 Ra_L^0.25,  Ra = g beta dT L^3 / (nu alpha)
    # 3) Radiation: q = eps sigma (T_K^4 - Tair_K^4)
    # 4) Evaporation: only while the surface is wet (T < 90 C); dries out
    #    above boiling, so RH no longer matters at high T.
    g = 9.81
    L_char = 0.1
    nu_air = 1.6e-5
    alpha_air = 2.2e-5
    k_air = 0.026
    eps = 0.25
    sigma = 5.67e-8
    for j in range(ny):
        for i in range(nx):
            dT = T[j, i] - T_air
            # natural convection
            Tf_K = (T[j, i] + T_air) / 2.0 + 273.15
            beta = 1.0 / max(Tf_K, 200.0)
            Ra = g * beta * max(dT, 0.0) * L_char**3 / (nu_air * alpha_air)
            Nu = 0.54 * Ra**0.25 if Ra > 1e3 else 1.0
            h_n = Nu * k_air / L_char
            h_c = h_n + h_wind * v_air
            q_conv = h_c * dT
            # radiation
            Tk = T[j, i] + 273.15
            Tairk = T_air + 273.15
            q_rad = eps * sigma * (Tk**4 - Tairk**4)
            # evaporation (wet pad dries above 90 C)
            wet = 1.0 if T[j, i] < 90.0 else 0.0
            q_evap = h_evap * (1.0 - RH) * wet * max(dT, 0.0)
            q_out[j, i] = q_conv + q_rad + q_evap

    # TEC cold-side load: q_c per module; distribute over patch cells
    q_tec_area = np.zeros_like(T)
    p_total = 0.0
    # four modules share I; each patch uses its own local mean T_c
    # patches are contiguous 2x2 blocks: hard-code centres via mask sums.
    # We compute per-module loads by looping over the four block origins.
    # Actuator: bidirectional patch source.
    # u < 0 -> resistive heating (P_heat_max per module, realistic up to 200 C)
    # u > 0 -> active cooling  (P_cool_max per module)
    # Asymmetric on purpose: heating is cheaper than active cooling.
    for pc in range(4):
        if I < 0:
            q_mod = I * P_heat_max        # I negative -> negative * P = heating into plate
        else:
            q_mod = I * P_cool_max        # I positive -> cooling out of plate
        # electrical consumption: heating ~ resistive (I^2 R) scaled,
        # cooling uses ~ half the heating power for same heat lift
        P_el_mod = abs(q_mod) if I < 0 else abs(q_mod) * 0.5
        p_total += P_el_mod
        area_patch = (2.0 * dx) * (2.0 * dy)
        for j in range(ny):
            for i in range(nx):
                if patch_mask[j, i] == pc + 1:
                    q_tec_area[j, i] = -q_mod / area_patch

    dTdt = k * lap + (q_tec_area - q_out) / Lz
    T_new = T + dt / rho_cp * dTdt
    return T_new, float(np.sum(q_tec_area) * dx * dy), p_total


class PlatePlant:
    """Thin copper plate + 2x2 TEC array.  All temperatures in deg C."""

    def __init__(self, cfg: dict):
        c = cfg["plant"]
        self.nx = int(c["nx"])
        self.ny = int(c["ny"])
        self.Lx = float(c["Lx"])
        self.Ly = float(c["Ly"])
        self.Lz = float(c["Lz"])
        self.k = float(c["k_copper"])
        self.rho = float(c["rho_copper"])
        self.cp = float(c["cp_copper"])
        self.rho_cp = self.rho * self.cp

        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)

        # Actuator limits per patch (W). Heating is a resistive heater that
        # can run to 200 C; cooling is a chiller.
        t = cfg["tec"]
        self.P_heat_max = float(t.get("P_heat_max_w", 40.0))
        self.P_cool_max = float(t.get("P_cool_max_w", 25.0))
        self.I_max = 1.0   # u in [-1,1] directly

        # convection / evaporation
        e = cfg["environment"]
        self.h_wind = float(e["h_wind_slope"])
        self.h_evap = float(e["h_evap_dry"])

        self.dt_sub = float(c["dt_sub"])

        # patch mask
        self.patch_mask = np.zeros((self.ny, self.nx), dtype=np.int64)
        for pid, (sl_y, sl_x) in enumerate(_patch_indices(self.nx, self.ny), 1):
            self.patch_mask[sl_y, sl_x] = pid

        self.T = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, T_init: float | None = None, T_air: float = 25.0):
        T0 = T_init if T_init is not None else T_air
        self.T = np.full((self.ny, self.nx), T0, dtype=np.float64)
        self.t = 0.0
        self.q_tec_total = 0.0
        self.p_electric = 0.0
        self.T_air = T_air
        return self.observed_temperature()

    def observed_temperature(self):
        """Single measurement = area-average plate temperature (sensor)."""
        return float(self.T.mean())

    def spatial_mean(self):
        return float(self.T.mean())

    def spatial_range(self):
        return float(self.T.max() - self.T.min())

    # ------------------------------------------------------------------
    def step(self, u: float, T_air: float, v_air: float, RH: float,
             dt_ctrl: float):
        """Drive one control interval; u in [-1,1] signed actuator command."""
        I = float(np.clip(u, -1.0, 1.0))
        n_sub = max(1, int(round(dt_ctrl / self.dt_sub)))
        h = dt_ctrl / n_sub
        q_tec_total = 0.0
        p_total = 0.0
        for _ in range(n_sub):
            self.T, q_tec, p_el = step_plate(
                self.T, self.dx, self.dy, h, self.Lz, self.rho_cp, self.k,
                self.patch_mask,
                I, T_air, v_air, RH,
                self.h_wind, self.h_evap,
                self.P_heat_max, self.P_cool_max)
            q_tec_total += q_tec * h
            p_total += p_el * h
        self.t += dt_ctrl
        self.T_air = T_air
        self.q_tec_total = q_tec_total
        self.p_electric = p_total          # energy this step [J]
        return self.observed_temperature()
