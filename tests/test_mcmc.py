"""Tests for src/mcmc.py: transforms, priors, likelihood, sampler and diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mcmc  # noqa: E402
import seir  # noqa: E402

PRIORS = {
    "bounded": mcmc.Prior("uniform", lo=2.0, hi=5.0),
    "positive": mcmc.Prior("lognormal", lo=0.0, mu=0.5, sd=0.7),
    "real": mcmc.Prior("normal", mu=1.0, sd=2.0),
    "beta": mcmc.Prior("beta", lo=0.0, hi=1.0, a=2.0, b=5.0),
}


@pytest.mark.parametrize("name", list(PRIORS))
def test_transform_round_trip_and_jacobian(name):
    p = PRIORS[name]
    z = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(p.to_z(p.to_theta(z)), z, atol=1e-9)
    h = 1e-6
    numeric = np.log(np.abs(p.to_theta(z + h) - p.to_theta(z - h)) / (2 * h))
    np.testing.assert_allclose(p.log_jacobian(z), numeric, atol=1e-6)


def test_prior_densities_match_scipy_up_to_a_constant():
    x = np.linspace(0.05, 0.95, 7)
    diff = PRIORS["beta"].log_density(x) - stats.beta(2, 5).logpdf(x)
    np.testing.assert_allclose(diff, diff[0])
    x = np.linspace(0.2, 6, 7)
    diff = PRIORS["positive"].log_density(x) - stats.lognorm(s=0.7, scale=np.exp(0.5)).logpdf(x)
    np.testing.assert_allclose(diff, diff[0])
    assert np.all(np.isinf(PRIORS["bounded"].log_density(np.array([1.9, 5.1]))))


def test_negbin_matches_scipy():
    y = np.arange(0, 60)
    mean, k = 17.3, 4.2
    ours = mcmc.negbin_logpmf(y, mean, k)
    ref = stats.nbinom(k, k / (k + mean)).logpmf(y)
    np.testing.assert_allclose(ours, ref, rtol=1e-10)


def gaussian_model(mean, cov):
    prec = np.linalg.inv(cov)
    priors = {"x": mcmc.Prior("uniform", lo=-50, hi=50), "y": mcmc.Prior("uniform", lo=-50, hi=50)}

    def loglik(theta):
        d = np.stack([theta["x"], theta["y"]], axis=-1) - mean
        return -0.5 * np.einsum("ni,ij,nj->n", d, prec, d)

    return mcmc.Model(priors=priors, log_likelihood=loglik)


def test_sampler_recovers_a_correlated_gaussian():
    mean = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.8], [0.8, 2.0]])
    model = gaussian_model(mean, cov)
    rng = np.random.default_rng(7)
    z0 = model.to_z({"x": rng.uniform(-8, 8, 6), "y": rng.uniform(-8, 8, 6)})
    res = mcmc.metropolis(model, z0, n_burn=3000, n_keep=6000, rng=rng)
    xs = np.stack([res.flat("x"), res.flat("y")], axis=1)
    for j, name in enumerate(["x", "y"]):
        ess = mcmc.effective_sample_size(res.theta[name])
        se = np.sqrt(cov[j, j] / ess)
        assert abs(xs[:, j].mean() - mean[j]) < 4 * se
        assert mcmc.split_rhat(res.theta[name]) < 1.02
    np.testing.assert_allclose(np.cov(xs, rowvar=False), cov, rtol=0.15, atol=0.1)
    assert np.all((res.acceptance > 0.15) & (res.acceptance < 0.45)), res.acceptance


def test_rhat_flags_chains_that_disagree():
    rng = np.random.default_rng(0)
    good = rng.standard_normal((1000, 4))
    bad = good + np.array([0.0, 0.0, 3.0, 3.0])
    assert mcmc.split_rhat(good) < 1.01
    assert mcmc.split_rhat(bad) > 1.5


def test_ess_of_independent_and_correlated_draws():
    rng = np.random.default_rng(1)
    iid = rng.standard_normal((4000, 4))
    assert 0.8 * iid.size < mcmc.effective_sample_size(iid) < 1.2 * iid.size
    ar = np.zeros((4000, 4))
    for t in range(1, 4000):                       # AR(1) with phi = 0.9: ESS ~ n (1-phi)/(1+phi)
        ar[t] = 0.9 * ar[t - 1] + rng.standard_normal(4)
    expected = ar.size * 0.1 / 1.9
    assert 0.6 * expected < mcmc.effective_sample_size(ar) < 1.6 * expected


def test_proposal_is_frozen_after_burn_in():
    model = gaussian_model(np.zeros(2), np.eye(2))
    rng = np.random.default_rng(3)
    res = mcmc.metropolis(model, np.zeros((4, 2)), n_burn=500, n_keep=10, rng=rng)
    res2 = mcmc.metropolis(model, np.zeros((4, 2)), n_burn=500, n_keep=2000, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(res.proposal_cov, res2.proposal_cov)


def test_seir_posterior_is_vectorised_and_rejects_impossible_starts():
    N = 2_375_415
    truth = seir.Params(N=N, beta0=2.8, amp=0.5, phase=22.0, rho=0.1, s0=0.5,
                        i0=1500.0, e0=1500.0 * seir.GAMMA_DEFAULT / seir.SIGMA_DEFAULT)
    y = np.round(seir.simulate(truth, 30)["weekly_reported"])
    priors = {"beta0": mcmc.Prior("uniform", lo=0.5, hi=6.0), "amp": mcmc.Prior("uniform", lo=0.0, hi=0.95),
              "s0": mcmc.Prior("uniform", lo=0.0, hi=1.0), "k": mcmc.Prior("lognormal", lo=0.0, mu=3.0, sd=1.0)}
    model = mcmc.seir_model(y, priors, dict(N=N, phase=22.0, rho=0.1, i0=1500.0), seir.simulate, seir.Params)
    theta = {"beta0": np.array([2.8, 2.0, 3.5]), "amp": np.array([0.5, 0.3, 0.6]),
             "s0": np.array([0.5, 0.6, 1e-6]), "k": np.array([20.0, 20.0, 20.0])}
    z = model.to_z(theta)
    batch = model.log_posterior(z)
    singles = [model.log_posterior(z[i:i + 1])[0] for i in range(3)]
    np.testing.assert_allclose(batch[:2], singles[:2], rtol=1e-12)
    assert np.isneginf(batch[2]) and np.isneginf(singles[2])       # s0 * N < initial infected
    assert batch[0] > batch[1]                                     # truth fits better than a wrong guess


def test_background_adds_a_constant_to_expected_cases():
    N = 2_375_415
    y = np.full(10, 300.0)
    base = {"beta0": mcmc.Prior("uniform", lo=0.5, hi=6.0)}
    with_bg = dict(base, bg=mcmc.Prior("uniform", lo=0.0, hi=1000.0))
    fixed = dict(N=N, s0=0.5, i0=10.0, rho=0.1, k=50.0)
    m0 = mcmc.seir_model(y, base, fixed, seir.simulate, seir.Params)
    m1 = mcmc.seir_model(y, with_bg, fixed, seir.simulate, seir.Params)
    # tiny epidemic (R0 < 1): without background the mean is far below 300, with bg = 300 it fits
    lp0 = m0.log_posterior(m0.to_z({"beta0": np.array([0.9])}))
    lp1 = m1.log_posterior(m1.to_z({"beta0": np.array([0.9]), "bg": np.array([300.0])}))
    assert lp1[0] > lp0[0] + 100
