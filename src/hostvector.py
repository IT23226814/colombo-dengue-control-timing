"""Host-vector extension: human SEIR coupled to mosquito S_v, E_v, I_v compartments,
with the mosquito carrying capacity driven by lagged weekly rainfall.

Units: time in weeks, rates per week. Mosquito numbers are scaled by the mean
mosquito population, so x_s + x_e + x_i = 1 when the population is at its mean
carrying capacity; only products of the biting and transmission parameters with
the mosquito density matter, and these are absorbed into q.

Humans (N people):
    dS/dt = -lam_h S            dE/dt = lam_h S - sigma E
    dI/dt = sigma E - gamma I   dR/dt = gamma I            dC/dt = sigma E
Mosquitoes (fractions of the mean mosquito population):
    dx_s/dt = mu k(t) - lam_v x_s - mu x_s
    dx_e/dt = lam_v x_s - (nu + mu) x_e
    dx_i/dt = nu x_e - mu x_i
Forces of infection:
    lam_h = u(t) q x_i          (bites by infectious mosquitoes on humans)
    lam_v = A I / N             (bites by mosquitoes on infectious humans)
Relative carrying capacity k(t), constant within each week:
    k = 1 - w + w * Rtilde / mean(Rtilde),
    Rtilde = weighted mean of rainfall in the previous weeks, weights exp(-(j-1)/tau), j = 1, 2, ...

The human-to-human reproduction number at mean mosquito density for a fully
susceptible population is  T0 = q A nu / (gamma mu (nu + mu));  the sampled
quantity is T_e = T0 * s0, the analogue of the SEIR model's R_e.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

try:  # in the notebook these names are already defined by the model and MCMC cells
    from seir import SIGMA_DEFAULT, GAMMA_DEFAULT, control, rk4_step
    from mcmc import Model, negbin_logpmf
except ImportError:
    pass

MU_V_DEFAULT = 7 * 0.243   # adult Aedes aegypti field mortality: 0.064 + 0.179 per day [Brady et al. 2013]
NU_DEFAULT = 7 / 10.0      # 1 / extrinsic incubation period of 10 days (8-12 days [WHO])
A_DEFAULT = 7 * 0.25       # bites on humans per mosquito x infection probability (results insensitive, see tests)


@dataclass
class HVParams:
    """Host-vector model parameters. Floats or broadcastable NumPy arrays."""

    N: float
    q: float                      # human infection rate per infectious-mosquito fraction (1/week)
    A: float = A_DEFAULT          # mosquito infection rate per infectious-human fraction (1/week)
    mu: float = MU_V_DEFAULT      # mosquito mortality (1/week)
    nu: float = NU_DEFAULT        # 1 / extrinsic incubation period (1/week)
    sigma: float = 7 / 5.9        # 1 / intrinsic incubation period (1/week)
    gamma: float = 7 / 5.0        # 1 / infectious period (1/week)
    rho: float = 1.0              # reporting rate
    s0: float = 1.0               # initial susceptible fraction of humans
    e0: float = 0.0               # initial exposed humans
    i0: float = 1.0               # initial infectious humans
    start: float = np.inf         # intervention start (weeks)
    end: float = np.inf           # intervention end (weeks)
    reduction: float = 0.0        # fractional cut in lam_h during the intervention

    def with_(self, **changes) -> "HVParams":
        return replace(self, **changes)

    def arrays(self) -> dict[str, np.ndarray]:
        names = [f.name for f in fields(self)]
        vals = np.broadcast_arrays(*[np.asarray(getattr(self, n), dtype=float) for n in names])
        return dict(zip(names, vals))


def q_from(T_e, s0, A=A_DEFAULT, mu=MU_V_DEFAULT, nu=NU_DEFAULT, gamma=7 / 5.0):
    """q giving the effective reproduction number T_e at mean mosquito density."""
    return T_e * gamma * mu * (nu + mu) / (A * nu * s0)


def capacity(rain: np.ndarray, n_lead: int, w, tau) -> np.ndarray:
    """Relative carrying capacity k for each week after the n_lead lead-in weeks.

    rain: weekly rainfall, the first n_lead values precede week 1.
    w, tau: weight of rainfall (0..1) and memory (weeks); floats or arrays.
    Returns k with shape (len(rain) - n_lead, *broadcast shape of w and tau); the
    mean of k over these weeks is 1.
    """
    rain = np.asarray(rain, dtype=float)
    w, tau = np.broadcast_arrays(np.asarray(w, dtype=float), np.asarray(tau, dtype=float))
    lags = np.arange(1, n_lead + 1).reshape((-1,) + (1,) * tau.ndim)
    kernel = np.exp(-(lags - 1) / tau)
    kernel = kernel / kernel.sum(axis=0)
    weeks = len(rain) - n_lead
    past = np.stack([rain[n_lead + k - lags.reshape(-1)] for k in range(weeks)])   # (weeks, lags)
    smooth = np.tensordot(past, kernel, axes=(1, 0))                            # (weeks, *shape)
    k = (smooth / smooth.mean(axis=0)) ** w
    return k / k.mean(axis=0)


def rhs_hv(t, y, p, u, k):
    """Right-hand side; y = (S, E, I, R, C, x_s, x_e, x_i), u = intervention multiplier, k = capacity."""
    S, E, I, R, C, xs, xe, xi = y
    lam_h = u * p["q"] * xi
    lam_v = p["A"] * I / p["N"]
    inf_h = lam_h * S
    return np.stack([
        -inf_h,
        inf_h - p["sigma"] * E,
        p["sigma"] * E - p["gamma"] * I,
        p["gamma"] * I,
        p["sigma"] * E,
        p["mu"] * k - lam_v * xs - p["mu"] * xs,
        lam_v * xs - (p["nu"] + p["mu"]) * xe,
        p["nu"] * xe - p["mu"] * xi,
    ])


def initial_state_hv(p, k0):
    """Humans as in the SEIR model (immune people in R); mosquitoes at the balance implied by the initial infectious humans."""
    S0 = p["s0"] * p["N"] - p["e0"] - p["i0"]
    R0 = p["N"] - S0 - p["e0"] - p["i0"]                  # people not susceptible start as immune
    lam_v = p["A"] * p["i0"] / p["N"]
    xs = k0 * np.ones_like(S0)
    xe = lam_v * xs / (p["nu"] + p["mu"])
    xi = p["nu"] * xe / p["mu"]
    return np.stack([S0, p["e0"], p["i0"], R0, np.zeros_like(S0), xs, xe, xi])


def simulate_hv(params: HVParams, k_weekly, weeks: int, steps_per_week: int = 7) -> dict[str, np.ndarray]:
    """Integrate for `weeks` weeks with RK4. k_weekly: capacity per week (rows); after the
    last row it is held at its last value. u and k are taken at each step's midpoint."""
    p = params.arrays()
    k_weekly = np.asarray(k_weekly, dtype=float)
    h = 1.0 / steps_per_week
    n = weeks * steps_per_week
    y = initial_state_hv(p, k_weekly[0])
    out = np.empty((n + 1,) + y.shape)
    out[0] = y
    for s in range(n):
        t = s * h
        k = k_weekly[min(int(t + h / 2), len(k_weekly) - 1)]
        y = rk4_step(rhs_hv, t, y, h, p, control(t + h / 2, p), k)
        out[s + 1] = y
    infections = np.diff(out[::steps_per_week, 4], axis=0)
    return {"t": np.arange(n + 1) * h, "y": out, "weekly_infections": infections,
            "weekly_reported": p["rho"] * infections}


def hv_posterior(y_obs, rain, n_lead, priors, fixed, steps_per_week=2):
    """Model (priors + negative-binomial likelihood) for T_e, w, tau, rho, s0, i0, k, bg."""
    y_obs = np.asarray(y_obs, dtype=float)
    weeks = len(y_obs)

    def log_likelihood(theta):
        n_chains = len(theta["T_e"])
        kw = capacity(rain, n_lead, theta["w"], theta["tau"])                  # (weeks, chains)
        base = HVParams(q=1.0, **fixed)
        a = base.arrays()
        i0 = theta["i0"].copy()
        ratio = a["gamma"] / a["sigma"]
        bad = theta["s0"] * a["N"] < i0 * (1.0 + ratio)
        i0[bad] = 1.0
        p = base.with_(q=q_from(theta["T_e"], theta["s0"], a["A"], a["mu"], a["nu"], a["gamma"]),
                       rho=theta["rho"], s0=theta["s0"], i0=i0, e0=i0 * ratio)
        mean = simulate_hv(p, kw, weeks, steps_per_week)["weekly_reported"] + theta["bg"]
        ll = negbin_logpmf(y_obs[:, None], mean.reshape(weeks, n_chains), theta["k"]).sum(axis=0)
        return np.where(bad, -np.inf, ll)

    return Model(priors=priors, log_likelihood=log_likelihood)
