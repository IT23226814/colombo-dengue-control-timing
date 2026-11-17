"""Seasonal SEIR model of dengue transmission, solved with a hand-written RK4.

Units used throughout
---------------------
time            weeks (t = 0 is the start of the first simulated week)
rates           1/week (beta0, sigma, gamma)
compartments    people (S, E, I, R and N)
C               cumulative new symptomatic infections (people), dC/dt = sigma*E

Model
-----
    dS/dt = -beta(t) S I / N
    dE/dt =  beta(t) S I / N - sigma E
    dI/dt =  sigma E - gamma I
    dR/dt =  gamma I
    dC/dt =  sigma E                      (counts every E -> I transition)

    beta(t) = beta0 * [1 + amp * cos(2 pi (t - phase) / period)] * u(t)
    u(t)    = 1 - reduction  while start <= t < end  (intervention), else 1

Weekly new infections in week k (from t = k to k + 1) are C(k+1) - C(k), and
the expected *reported* cases are rho times that.

This is a host-only model: the mosquito stage is not modelled explicitly, so
transmission through mosquitoes is absorbed into beta(t).

Every parameter may be a float or a NumPy array. Arrays are broadcast against
each other, so one call can run many parameter sets (e.g. MCMC samples) at once.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

DAYS_PER_WEEK = 7.0

# Literature defaults for the natural history (converted to 1/week):
#   mean intrinsic incubation period 5.9 days  [Chan & Johansson 2012, PLoS ONE 7(11):e50972,
#                                               doi:10.1371/journal.pone.0050972]
#   human viraemic (infectious) period ~5 days [WHO dengue fact sheet: "about 4-5 days"]
SIGMA_DEFAULT = DAYS_PER_WEEK / 5.9
GAMMA_DEFAULT = DAYS_PER_WEEK / 5.0


@dataclass(frozen=True)
class Params:
    """Model parameters. Floats or broadcastable NumPy arrays."""

    N: float                      # population (people)
    beta0: float                  # mean transmission rate (1/week)
    amp: float = 0.0              # seasonal amplitude, 0 <= amp < 1 (dimensionless)
    phase: float = 0.0            # week of peak transmission (weeks)
    period: float = 52.0          # length of the seasonal cycle (weeks)
    sigma: float = SIGMA_DEFAULT  # 1 / incubation period (1/week)
    gamma: float = GAMMA_DEFAULT  # 1 / infectious period (1/week)
    rho: float = 1.0              # reporting rate: reported / symptomatic infections
    s0: float = 1.0               # initial susceptible fraction of N
    e0: float = 0.0               # initial exposed (people)
    i0: float = 1.0               # initial infectious (people)
    start: float = np.inf         # intervention start (weeks)
    end: float = np.inf           # intervention end (weeks)
    reduction: float = 0.0        # fractional cut in beta during the intervention

    def with_(self, **changes) -> "Params":
        """Return a copy with some parameters changed."""
        return replace(self, **changes)

    def arrays(self) -> dict[str, np.ndarray]:
        """All parameters as NumPy arrays broadcast to one common shape."""
        names = [f.name for f in fields(self)]
        vals = np.broadcast_arrays(*[np.asarray(getattr(self, n), dtype=float) for n in names])
        return dict(zip(names, vals))


def control(t: float, p: dict[str, np.ndarray]) -> np.ndarray:
    """Intervention multiplier u(t): 1 - reduction while start <= t < end, else 1."""
    active = (t >= p["start"]) & (t < p["end"])
    return np.where(active, 1.0 - p["reduction"], 1.0)


def beta(t: float, p: dict[str, np.ndarray], u: np.ndarray | None = None) -> np.ndarray:
    """Transmission rate at time t (1/week). u is the intervention multiplier;
    if None it is evaluated at t."""
    seasonal = p["beta0"] * (1.0 + p["amp"] * np.cos(2.0 * np.pi * (t - p["phase"]) / p["period"]))
    return seasonal * (control(t, p) if u is None else u)


def basic_reproduction_number(t: float, params: Params) -> np.ndarray:
    """Instantaneous R0(t) = beta(t) / gamma for a fully susceptible population."""
    p = params.arrays()
    return beta(t, p) / p["gamma"]


def rhs(t: float, y: np.ndarray, p: dict[str, np.ndarray], u: np.ndarray | None = None) -> np.ndarray:
    """Right-hand side f(t, y); y has shape (5, ...) = (S, E, I, R, C)."""
    S, E, I, R, C = y
    infection = beta(t, p, u) * S * I / p["N"]
    incidence = p["sigma"] * E
    recovery = p["gamma"] * I
    return np.stack([-infection, infection - incidence, incidence - recovery, recovery, incidence])


def rk4_step(f, t: float, y: np.ndarray, h: float, *args) -> np.ndarray:
    """One classical 4th-order Runge-Kutta step of size h for y' = f(t, y, *args)."""
    k1 = f(t, y, *args)
    k2 = f(t + h / 2, y + h / 2 * k1, *args)
    k3 = f(t + h / 2, y + h / 2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def initial_state(p: dict[str, np.ndarray]) -> np.ndarray:
    """(S, E, I, R, C) at t = 0. People not susceptible, exposed or infectious start in R."""
    S = p["s0"] * p["N"] - p["e0"] - p["i0"]
    R = p["N"] - S - p["e0"] - p["i0"]
    if np.any(S < 0):
        raise ValueError("s0 * N must be at least e0 + i0")
    return np.stack([S, p["e0"], p["i0"], R, np.zeros_like(S)])


def simulate(params: Params, weeks: int, steps_per_week: int = 7) -> dict[str, np.ndarray]:
    """Integrate the model for `weeks` weeks with fixed RK4 steps of 1/steps_per_week weeks.

    Returns a dict with
      t                 (weeks*steps_per_week + 1,) time grid in weeks
      y                 (len(t), 5, ...) states S, E, I, R, C on the grid
      weekly_infections (weeks, ...) new symptomatic infections in each week
      weekly_reported   (weeks, ...) expected reported cases = rho * weekly_infections

    The intervention multiplier u is held constant over each step, at its value
    at the step midpoint, so the right-hand side is smooth within every step.
    Start/end times on the step grid (e.g. whole weeks) are therefore handled
    exactly and keep full 4th-order accuracy.
    """
    p = params.arrays()
    h = 1.0 / steps_per_week
    n = weeks * steps_per_week
    y = initial_state(p)
    out = np.empty((n + 1,) + y.shape)
    out[0] = y
    for k in range(n):
        t = k * h
        y = rk4_step(rhs, t, y, h, p, control(t + h / 2, p))
        out[k + 1] = y
    C_weekly = out[::steps_per_week, 4]
    infections = np.diff(C_weekly, axis=0)
    return {
        "t": np.arange(n + 1) * h,
        "y": out,
        "weekly_infections": infections,
        "weekly_reported": p["rho"] * infections,
    }
