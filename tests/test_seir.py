"""Basic correctness tests for src/seir.py. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import seir  # noqa: E402

N = 2_375_415  # Colombo District, Census 2024
BASE = seir.Params(N=N, beta0=2.0, amp=0.3, phase=27.0, rho=0.1, s0=0.4, e0=50.0, i0=50.0)


def test_population_is_conserved():
    out = seir.simulate(BASE, weeks=104)
    total = out["y"][:, :4].sum(axis=1)
    np.testing.assert_allclose(total, N, rtol=1e-12)


def test_cumulative_infections_equal_people_leaving_s_and_e():
    """C(t) counts E->I moves, so C = (S0 - S) - (E - E0) exactly."""
    out = seir.simulate(BASE, weeks=52)
    S, E, C = out["y"][:, 0], out["y"][:, 1], out["y"][:, 4]
    np.testing.assert_allclose(C, (S[0] - S) - (E - E[0]), rtol=1e-9, atol=1e-6)


def test_matches_reference_solver():
    """Hand-written RK4 agrees with SciPy's adaptive solver at tight tolerance."""
    weeks = 52
    out = seir.simulate(BASE, weeks=weeks, steps_per_week=28)
    p = BASE.arrays()
    ref = solve_ivp(lambda t, y: seir.rhs(t, y, p), (0, weeks), seir.initial_state(p),
                    method="DOP853", rtol=1e-11, atol=1e-8, t_eval=out["t"])
    np.testing.assert_allclose(out["y"], ref.y.T, rtol=1e-6, atol=1e-3)


def test_rk4_error_falls_at_fourth_order():
    """Halving the step should cut the error by about 2^4 = 16."""
    p = BASE.with_(beta0=3.0, phase=10.0, s0=1.0)  # fast-growing outbreak, error well above round-off

    def weekly_states(spw):
        return seir.simulate(p, weeks=26, steps_per_week=spw)["y"][::spw]

    ref = weekly_states(1024)
    errs = [np.abs(weekly_states(s) - ref).max() for s in (4, 8, 16, 32)]
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all((orders > 3.7) & (orders < 4.3)), orders


def test_no_outbreak_when_r0_below_one():
    p = BASE.with_(beta0=0.8 * seir.GAMMA_DEFAULT, amp=0.0, s0=1.0)
    assert seir.basic_reproduction_number(0.0, p) < 1
    inc = seir.simulate(p, weeks=52)["weekly_infections"]
    assert np.all(np.diff(inc[1:]) < 0)  # strictly declining after the first week
    assert inc.sum() < 10 * (p.e0 + p.i0)


def test_intervention_reduces_cases_and_earlier_is_better():
    base = seir.simulate(BASE, weeks=52)["weekly_reported"].sum()
    late = seir.simulate(BASE.with_(start=24, reduction=0.3), weeks=52)["weekly_reported"].sum()
    early = seir.simulate(BASE.with_(start=16, reduction=0.3), weeks=52)["weekly_reported"].sum()
    assert early < late < base


def test_intervention_has_no_effect_before_it_starts():
    a = seir.simulate(BASE, weeks=30)["y"]
    b = seir.simulate(BASE.with_(start=20, reduction=0.5), weeks=30)["y"]
    np.testing.assert_array_equal(a[: 20 * 7 + 1], b[: 20 * 7 + 1])
    assert not np.allclose(a[-1], b[-1])


def test_vectorised_matches_scalar_runs():
    betas = np.array([1.5, 2.0, 2.5])
    batch = seir.simulate(BASE.with_(beta0=betas), weeks=40)["weekly_reported"]
    for j, b in enumerate(betas):
        single = seir.simulate(BASE.with_(beta0=b), weeks=40)["weekly_reported"]
        np.testing.assert_allclose(batch[:, j], single, rtol=1e-12)


def test_rejects_impossible_initial_state():
    with pytest.raises(ValueError):
        seir.simulate(BASE.with_(s0=1e-6, i0=100.0), weeks=1)


@pytest.mark.skipif(seir.numba is None, reason="Numba not installed")
def test_compiled_loop_matches_numpy_loop():
    rng = np.random.default_rng(7)
    n = 50
    p = seir.Params(N=1e6, beta0=rng.uniform(0.5, 6, n), amp=rng.uniform(0, 0.9, n), phase=rng.uniform(0, 52, n),
                    s0=rng.uniform(0.05, 1, n), e0=rng.uniform(0, 500, n), i0=rng.uniform(1, 500, n),
                    start=rng.integers(0, 40, n).astype(float), end=rng.integers(41, 60, n).astype(float),
                    reduction=rng.uniform(0, 0.8, n))
    for spw in (2, 7):                                              # the steps the notebook uses
        a = seir.simulate(p, 52, spw, compiled=True)["y"]
        b = seir.simulate(p, 52, spw, compiled=False)["y"]
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)
    grid = p.with_(start=np.arange(5)[:, None] * 4.0)              # 2-D broadcast, as in the scenario grid
    np.testing.assert_allclose(seir.simulate(grid, 30)["y"], seir.simulate(grid, 30, compiled=False)["y"],
                               rtol=1e-12, atol=1e-9)
