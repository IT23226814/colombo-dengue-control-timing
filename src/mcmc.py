"""Random-walk Metropolis MCMC, priors on transformed scales, convergence
diagnostics, and the posterior for the seasonal SEIR model.

Everything is written from scratch with NumPy; SciPy is used only for the
log-gamma function. Several chains run side by side as rows of one array, so
each iteration evaluates the log-posterior of all chains in one vectorised call
(the SEIR model broadcasts over parameter arrays).

Sampling space
--------------
Each parameter theta is sampled on an unconstrained scale z:
    bounded (lo, hi)  theta = lo + (hi - lo) * logistic(z)
    positive          theta = exp(z)
    real              theta = z
The density on z includes the log-Jacobian |d theta / d z|, so the prior on
theta is exactly the one stated.

Metropolis algorithm (per chain)
--------------------------------
    propose z' = z + L e,  e ~ N(0, I)          (symmetric proposal)
    accept with probability min(1, exp(log p(z') - log p(z)))
During burn-in the proposal covariance L L^T is adapted from the chains'
recent samples (scaled by 2.38^2 / d) and its overall scale is tuned towards
an acceptance rate of about 0.25. After burn-in the proposal is frozen, so
the kept samples come from a fixed Metropolis kernel with the posterior as its
stationary distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Priors and transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prior:
    """Prior for one parameter.

    kind: "uniform" (on lo..hi), "normal" (mu, sd, truncated to lo..hi),
          "lognormal" (mu, sd of log theta; positive), "beta" (a, b on lo..hi).
    The sampling transform follows from the support: bounded -> logit,
    positive -> log, otherwise identity.
    """

    kind: str
    lo: float = -np.inf
    hi: float = np.inf
    mu: float = 0.0
    sd: float = 1.0
    a: float = 1.0
    b: float = 1.0

    @property
    def transform(self) -> str:
        if np.isfinite(self.lo) and np.isfinite(self.hi):
            return "logit"
        if self.lo == 0.0 and not np.isfinite(self.hi):
            return "log"
        return "identity"

    def log_density(self, theta: np.ndarray) -> np.ndarray:
        """log p(theta) up to an additive constant; -inf outside the support."""
        theta = np.asarray(theta, dtype=float)
        inside = (theta > self.lo) & (theta < self.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kind == "uniform":
                lp = np.zeros_like(theta)
            elif self.kind == "normal":
                lp = -0.5 * ((theta - self.mu) / self.sd) ** 2
            elif self.kind == "lognormal":
                lp = -np.log(theta) - 0.5 * ((np.log(theta) - self.mu) / self.sd) ** 2
            elif self.kind == "beta":
                x = (theta - self.lo) / (self.hi - self.lo)
                lp = (self.a - 1) * np.log(x) + (self.b - 1) * np.log1p(-x)
            else:
                raise ValueError(f"unknown prior kind {self.kind!r}")
        return np.where(inside, lp, -np.inf)

    def to_theta(self, z: np.ndarray) -> np.ndarray:
        if self.transform == "logit":
            return self.lo + (self.hi - self.lo) / (1.0 + np.exp(-z))
        if self.transform == "log":
            return np.exp(z)
        return z

    def to_z(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if self.transform == "logit":
            x = (theta - self.lo) / (self.hi - self.lo)
            return np.log(x) - np.log1p(-x)
        if self.transform == "log":
            return np.log(theta)
        return theta

    def log_jacobian(self, z: np.ndarray) -> np.ndarray:
        """log |d theta / d z|."""
        if self.transform == "logit":
            # d/dz logistic(z) = s (1 - s); computed stably with logaddexp
            return np.log(self.hi - self.lo) - np.logaddexp(0.0, z) - np.logaddexp(0.0, -z)
        if self.transform == "log":
            return z
        return np.zeros_like(z)


@dataclass(frozen=True)
class Model:
    """A named set of priors and a log-likelihood log p(data | theta).

    log_likelihood receives a dict {name: array of shape (n_chains,)} and
    returns an array of shape (n_chains,).
    """

    priors: dict[str, Prior]
    log_likelihood: object = field(repr=False)

    @property
    def names(self) -> list[str]:
        return list(self.priors)

    def to_theta(self, z: np.ndarray) -> dict[str, np.ndarray]:
        return {n: p.to_theta(z[..., j]) for j, (n, p) in enumerate(self.priors.items())}

    def to_z(self, theta: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([p.to_z(theta[n]) for n, p in self.priors.items()], axis=-1)

    def log_posterior(self, z: np.ndarray) -> np.ndarray:
        """log p(z | data) up to a constant, for z of shape (n_chains, d)."""
        theta = self.to_theta(z)
        lp = np.zeros(z.shape[0])
        for j, (n, p) in enumerate(self.priors.items()):
            lp = lp + p.log_density(theta[n]) + p.log_jacobian(z[:, j])
        ok = np.isfinite(lp)
        if ok.any():
            ll = np.full(z.shape[0], -np.inf)
            sub = {n: v[ok] for n, v in theta.items()}
            ll[ok] = self.log_likelihood(sub)
            lp = np.where(ok, lp + ll, -np.inf)
        return np.where(np.isnan(lp), -np.inf, lp)


# ---------------------------------------------------------------------------
# Likelihood for counts
# ---------------------------------------------------------------------------


def negbin_logpmf(y: np.ndarray, mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """log P(Y = y) for a negative binomial with the given mean and dispersion k.

    Var(Y) = mean + mean^2 / k, so k -> infinity recovers the Poisson.
    """
    mean = np.maximum(mean, 1e-12)
    return (gammaln(y + k) - gammaln(k) - gammaln(y + 1)
            + k * (np.log(k) - np.log(k + mean)) + y * (np.log(mean) - np.log(k + mean)))


# ---------------------------------------------------------------------------
# Metropolis sampler
# ---------------------------------------------------------------------------


@dataclass
class MCMCResult:
    names: list[str]
    z: np.ndarray            # (n_keep, n_chains, d) kept samples on the sampling scale
    theta: dict              # {name: (n_keep, n_chains)} kept samples on the natural scale
    log_post: np.ndarray     # (n_keep, n_chains)
    acceptance: np.ndarray   # (n_chains,) acceptance rate after burn-in
    proposal_cov: np.ndarray # (d, d) frozen proposal covariance used after burn-in
    burn_in_trace: dict      # {name: (n_burn, n_chains)} for trace plots

    def flat(self, name: str) -> np.ndarray:
        """All kept samples of one parameter, chains pooled."""
        return self.theta[name].reshape(-1)


def metropolis(model: Model, z0: np.ndarray, n_burn: int, n_keep: int, rng: np.random.Generator,
               init_scale: float = 0.1, adapt_every: int = 100, target_accept: float = 0.25,
               thin: int = 1) -> MCMCResult:
    """Run len(z0) random-walk Metropolis chains in parallel.

    z0: (n_chains, d) starting points on the sampling scale (use different,
        over-dispersed starts so convergence can be checked).
    n_burn: iterations with proposal adaptation (discarded).
    n_keep: iterations kept after burn-in (before thinning).
    """
    z = np.array(z0, dtype=float)
    n_chains, d = z.shape
    lp = model.log_posterior(z)
    if not np.all(np.isfinite(lp)):
        raise ValueError("every chain must start where the posterior is finite")

    cov = np.eye(d) * init_scale**2          # proposal covariance before adaptation
    log_scale = 0.0                          # multiplies the covariance by exp(2 * log_scale)
    burn = np.empty((n_burn, n_chains, d))
    window_accepts = 0

    def step(z, lp, chol):
        proposal = z + rng.standard_normal((n_chains, d)) @ chol.T
        lp_new = model.log_posterior(proposal)
        accept = np.log(rng.random(n_chains)) < lp_new - lp
        z = np.where(accept[:, None], proposal, z)
        lp = np.where(accept, lp_new, lp)
        return z, lp, accept

    chol = np.linalg.cholesky(cov)
    for i in range(n_burn):
        z, lp, acc = step(z, lp, np.exp(log_scale) * chol)
        burn[i] = z
        window_accepts += acc.sum()
        if (i + 1) % adapt_every == 0:
            rate = window_accepts / (adapt_every * n_chains)
            log_scale += (rate - target_accept) * 2.0 / np.sqrt((i + 1) / adapt_every)
            window_accepts = 0
            # empirical covariance of the second half of burn-in so far, all chains pooled
            recent = burn[(i + 1) // 2: i + 1].reshape(-1, d)
            if recent.shape[0] > 10 * d:
                emp = np.cov(recent, rowvar=False) + 1e-8 * np.eye(d)
                cov = 2.38**2 / d * emp
                chol = np.linalg.cholesky(cov)

    final_chol = np.exp(log_scale) * chol    # frozen from here on
    keep = np.empty((n_keep // thin, n_chains, d))
    keep_lp = np.empty((n_keep // thin, n_chains))
    accepted = np.zeros(n_chains)
    for i in range(n_keep):
        z, lp, acc = step(z, lp, final_chol)
        accepted += acc
        if (i + 1) % thin == 0:
            keep[(i + 1) // thin - 1] = z
            keep_lp[(i + 1) // thin - 1] = lp

    theta = model.to_theta(keep)
    return MCMCResult(
        names=model.names, z=keep, theta=theta, log_post=keep_lp,
        acceptance=accepted / n_keep, proposal_cov=final_chol @ final_chol.T,
        burn_in_trace=model.to_theta(burn),
    )


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


def split_rhat(x: np.ndarray) -> float:
    """Split-R-hat (Gelman et al.) for samples x of shape (n_iter, n_chains).

    Each chain is split in half; values close to 1 (below about 1.01-1.05)
    indicate that the chains agree with each other and with themselves.
    """
    n = x.shape[0] // 2
    halves = np.concatenate([x[:n], x[n:2 * n]], axis=1)       # (n, 2 * n_chains)
    chain_means = halves.mean(axis=0)
    W = halves.var(axis=0, ddof=1).mean()                       # within-chain variance
    B = n * chain_means.var(ddof=1)                             # between-chain variance
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def effective_sample_size(x: np.ndarray) -> float:
    """Effective sample size for x of shape (n_iter, n_chains), using the
    multi-chain autocorrelation and Geyer's initial positive sequence."""
    n, m = x.shape
    centred = x - x.mean(axis=0)
    f = np.fft.rfft(centred, n=2 * n, axis=0)
    acov = np.fft.irfft(f * np.conj(f), axis=0)[:n] / n          # (n, m) autocovariance per chain
    W = x.var(axis=0, ddof=1).mean()
    var_plus = (n - 1) / n * W + x.mean(axis=0).var(ddof=1) if m > 1 else W
    rho = 1.0 - (W - acov.mean(axis=1)) / var_plus
    rho[0] = 1.0
    tau = -1.0
    for t in range(0, n - 1, 2):                                 # sum pairs while positive
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(n * m / max(tau, 1e-12))


def summarise(result: MCMCResult) -> list[dict]:
    """Posterior mean, 95% credible interval, split-R-hat and ESS for each parameter."""
    rows = []
    for name in result.names:
        x = result.theta[name]
        lo, hi = np.percentile(x, [2.5, 97.5])
        rows.append(dict(parameter=name, mean=x.mean(), sd=x.std(ddof=1), ci_2_5=lo, ci_97_5=hi,
                         rhat=split_rhat(x), ess=effective_sample_size(x)))
    return rows


# ---------------------------------------------------------------------------
# Posterior for the seasonal SEIR model
# ---------------------------------------------------------------------------


def seir_model(y_obs: np.ndarray, priors: dict[str, Prior], fixed: dict, simulate, Params,
               steps_per_week: int = 7) -> Model:
    """Negative-binomial likelihood of weekly reported cases under the SEIR model.

    Expected reported cases in week k are  rho * (new infections in week k) + bg,
    where bg is an optional constant background of reported cases per week.

    priors: priors for any of beta0, amp, phase, rho, s0, i0 (initial
            infectious people), k (negative-binomial dispersion) and bg
            (background reported cases per week; omit it to fix bg = 0).
    fixed:  values for everything not estimated: Params fields (must include
            N) and, optionally, k and bg. Without k (estimated or fixed) the
            likelihood is Poisson.
    Initial exposed people are set to e0 = i0 * gamma / sigma, the ratio at
    which E and I are in balance when incidence is roughly steady.
    simulate, Params: the model's functions, passed in so this module does
    not depend on how the model is imported.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    weeks = len(y_obs)
    observation_fields = {"k", "bg"}
    fixed_model = {n: v for n, v in fixed.items() if n not in observation_fields}
    fixed_obs = {n: v for n, v in fixed.items() if n in observation_fields}

    def log_likelihood(theta: dict) -> np.ndarray:
        kwargs = dict(fixed_model)
        kwargs.update({n: v for n, v in theta.items() if n not in observation_fields})
        p = Params(**kwargs)
        arr = p.arrays()
        n_chains = len(next(iter(theta.values())))
        i0 = np.broadcast_to(arr["i0"], (n_chains,)).copy()
        ratio = np.broadcast_to(arr["gamma"] / arr["sigma"], (n_chains,))
        # Parameter sets with more initial infected than susceptible people are impossible:
        # give them a harmless value for the run and a log-likelihood of -inf afterwards.
        bad = np.broadcast_to(arr["s0"] * arr["N"], (n_chains,)) < i0 * (1.0 + ratio)
        i0[bad] = 1.0
        p = p.with_(i0=i0, e0=i0 * ratio)
        mean = simulate(p, weeks, steps_per_week)["weekly_reported"]
        mean = np.broadcast_to(mean.reshape(weeks, -1), (weeks, n_chains))  # (weeks, n_chains)
        obs = {**fixed_obs, **{n: v for n, v in theta.items() if n in observation_fields}}
        if "bg" in obs:
            mean = mean + obs["bg"]
        if "k" in obs:
            ll = negbin_logpmf(y_obs[:, None], mean, obs["k"]).sum(axis=0)
        else:                                                          # Poisson likelihood
            m = np.maximum(mean, 1e-12)
            ll = (y_obs[:, None] * np.log(m) - m - gammaln(y_obs[:, None] + 1)).sum(axis=0)
        return np.where(bad, -np.inf, ll)

    return Model(priors=priors, log_likelihood=log_likelihood)
