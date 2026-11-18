"""Tests for src/hostvector.py: rainfall capacity, conservation and limiting cases."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hostvector as hv  # noqa: E402
import seir  # noqa: E402

N = 1_000_000
RAIN = np.abs(np.sin(np.arange(40) / 3.0)) * 80 + 5   # 40 weeks of made-up rainfall
LEAD = 10


def base(**changes):
    p = hv.HVParams(N=N, q=hv.q_from(2.0, 0.6), s0=0.6, e0=50.0, i0=50.0)
    return p.with_(**changes)


def test_capacity_has_mean_one_and_is_flat_without_rain_effect():
    k = hv.capacity(RAIN, LEAD, np.array([0.0, 0.7, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert k.shape == (len(RAIN) - LEAD, 3)
    np.testing.assert_allclose(k.mean(axis=0), 1.0)
    np.testing.assert_allclose(k[:, 0], 1.0)
    assert k[:, 2].std() > k[:, 1].std() > 0


def test_capacity_uses_only_past_rainfall():
    rain = RAIN.copy()
    rain[LEAD + 5] += 500.0                       # extra rain in week index 5
    k_new = hv.capacity(rain, LEAD, 1.0, 2.0)
    k_old = hv.capacity(RAIN, LEAD, 1.0, 2.0)
    ratio = k_new / k_new.mean() * k_old.mean() / k_old
    np.testing.assert_allclose(ratio[:6] / ratio[0], 1.0, rtol=1e-12)   # weeks 0-5 see no change in shape
    assert ratio[6] > ratio[5]


def test_conservation_and_mosquito_total():
    k = hv.capacity(RAIN, LEAD, 1.0, 2.0)
    out = hv.simulate_hv(base(), k, 30, 28)
    y = out["y"]
    np.testing.assert_allclose(y[:, :4].sum(axis=1), N, rtol=1e-12)                 # S + E + I + R = N
    np.testing.assert_allclose(y[:, 4], (y[0, 0] + y[0, 1]) - (y[:, 0] + y[:, 1]), rtol=1e-9, atol=1e-6)  # C
    # total mosquitoes X = x_s + x_e + x_i solves dX/dt = mu (k - X): exact solution week by week
    X = y[:, 5:].sum(axis=1)
    mu = hv.MU_V_DEFAULT
    exact = [X[0]]
    for w in range(30):
        kw = k[min(w, len(k) - 1)]
        exact.append(kw + (exact[-1] - kw) * np.exp(-mu))
    np.testing.assert_allclose(X[::28], exact, rtol=1e-7)
    assert y.min() > -1e-9


def test_only_product_of_mosquito_rates_matters_when_few_humans_are_infectious():
    # Colombo-like setting (small susceptible pool): doubling A and halving q changes little,
    # because mosquito infection stays far from saturation
    k = hv.capacity(RAIN, LEAD, 1.0, 2.0)
    p = hv.HVParams(N=N, q=hv.q_from(2.0, 0.07), s0=0.07, e0=50.0, i0=50.0)
    a = hv.simulate_hv(p, k, 40)["weekly_infections"].sum()
    b = hv.simulate_hv(p.with_(A=p.A * 2, q=p.q / 2), k, 40)["weekly_infections"].sum()
    assert abs(a - b) / a < 5e-3


def test_fast_mosquito_limit_reduces_to_seir():
    # with mosquito lifetime and EIP of minutes, lam_h -> q A nu / (mu (nu + mu)) * I/N = beta I/N
    mu, nu, beta = 100.0, 1000.0, 2.5
    q = beta * mu * (nu + mu) / (hv.A_DEFAULT * nu)
    p = hv.HVParams(N=N, q=q, mu=mu, nu=nu, s0=0.6, e0=50.0, i0=50.0)
    a = hv.simulate_hv(p, np.ones(1), 40, 400)["weekly_infections"]     # small step: fast rates are stiff
    s = seir.simulate(seir.Params(N=N, beta0=beta, s0=0.6, e0=50.0, i0=50.0), 40, 28)["weekly_infections"]
    assert abs(a.sum() - s.sum()) / s.sum() < 0.01
    assert abs(np.argmax(a) - np.argmax(s)) <= 1


def test_no_outbreak_below_threshold():
    k = np.ones(1)
    out = hv.simulate_hv(base(q=hv.q_from(0.8, 0.6)), k, 52)["weekly_infections"]
    assert out.sum() < 50 * 20                   # a few times the initial infections, no epidemic
    assert out[-1] < out[2]


def test_intervention_reduces_infections():
    k = hv.capacity(RAIN, LEAD, 1.0, 2.0)
    none = hv.simulate_hv(base(), k, 40)["weekly_infections"].sum()
    cut = hv.simulate_hv(base(start=0.0, reduction=0.5), k, 40)["weekly_infections"].sum()
    assert cut < none
