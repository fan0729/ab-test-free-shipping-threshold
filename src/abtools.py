"""Reusable A/B-testing toolkit (no experiment-specific logic lives here).

Every function takes plain numpy arrays so the same code runs on the simulated
experiment and on a real dataset (e.g. Criteo uplift).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

Z975 = stats.norm.ppf(0.975)


# ---------------------------------------------------------------- validity
def srm_test(n_treat, n_ctrl, expected_ratio=0.5):
    """Chi-square goodness-of-fit for the assignment ratio."""
    n = n_treat + n_ctrl
    chi2 = ((n_treat - n * expected_ratio) ** 2 / (n * expected_ratio)
            + (n_ctrl - n * (1 - expected_ratio)) ** 2 / (n * (1 - expected_ratio)))
    return dict(chi2=float(chi2), p=float(stats.chi2.sf(chi2, 1)),
                n_treat=int(n_treat), n_ctrl=int(n_ctrl),
                share_treat=float(n_treat / n))


def smd(x_t, x_c):
    """Standardised mean difference (pooled SD). |SMD| > 0.1 is the usual flag."""
    sd = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
    return float((np.mean(x_t) - np.mean(x_c)) / sd) if sd > 0 else 0.0


# ---------------------------------------------------------------- inference
def welch(y_t, y_c):
    """Difference in means with Welch SE; returns dict with CI and p-value."""
    d = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = d / se
    return dict(diff=float(d), se=float(se), lo=float(d - Z975 * se), hi=float(d + Z975 * se),
                p=float(2 * stats.norm.sf(abs(z))), mean_c=float(y_c.mean()), mean_t=float(y_t.mean()))


def two_prop_z(y_t, y_c):
    """Pooled two-proportion z-test; CI uses the unpooled SE."""
    p1, p0, n1, n0 = y_t.mean(), y_c.mean(), len(y_t), len(y_c)
    pp = (y_t.sum() + y_c.sum()) / (n1 + n0)
    z = (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return dict(diff=float(p1 - p0), se=float(se), lo=float(p1 - p0 - Z975 * se),
                hi=float(p1 - p0 + Z975 * se), p=float(2 * stats.norm.sf(abs(z))),
                mean_c=float(p0), mean_t=float(p1))


def _sparse_boot_means(y, B, rng):
    """Exact nonparametric bootstrap of the mean for zero-inflated data (fast).

    Resampling n rows == drawing the number of non-zeros ~ Binomial(n, k/n) and then
    resampling that many values from the non-zero rows.
    """
    n = len(y)
    nz = y[y != 0]
    k = len(nz)
    m = rng.binomial(n, k / n, size=B)
    out = np.empty(B)
    for b in range(B):
        out[b] = nz[rng.integers(0, k, m[b])].sum() / n
    return out


def bootstrap_diff(y_t, y_c, B=2000, seed=0):
    """Percentile bootstrap CI for difference in means (y arrays may be zero-inflated)."""
    rng = np.random.default_rng(seed)
    d = _sparse_boot_means(y_t, B, rng) - _sparse_boot_means(y_c, B, rng)
    return dict(diff=float(y_t.mean() - y_c.mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), se=float(d.std(ddof=1)))


def delta_ratio(num_t, den_t, num_c, den_c):
    """Delta-method difference of ratio-of-means (e.g. AOV = sum(basket)/sum(orders)).

    num/den are per-USER arrays (the randomisation unit), so the variance accounts for the
    fact that orders are clustered within users.
    """
    def one(num, den):
        n = len(num)
        mn, md = num.mean(), den.mean()
        r = mn / md
        var = (num.var(ddof=1) + r ** 2 * den.var(ddof=1) - 2 * r * np.cov(num, den, ddof=1)[0, 1]) / (n * md ** 2)
        return r, var

    r1, v1 = one(num_t, den_t)
    r0, v0 = one(num_c, den_c)
    d, se = r1 - r0, np.sqrt(v1 + v0)
    return dict(diff=float(d), se=float(se), lo=float(d - Z975 * se), hi=float(d + Z975 * se),
                p=float(2 * stats.norm.sf(abs(d / se))), mean_c=float(r0), mean_t=float(r1))


# ---------------------------------------------------------------- variance reduction
def cuped_adjust(y, x, pooled_theta=None):
    """CUPED: y_adj = y - theta * (x - mean(x)),  theta = cov(y, x) / var(x) over BOTH arms.

    theta must be estimated on pooled data (or pre-period only) so it does not depend on
    treatment assignment.  Returns adjusted y and diagnostics.
    """
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1) if pooled_theta is None else pooled_theta
    y_adj = y - theta * (x - x.mean())
    return y_adj, dict(theta=float(theta), corr=float(np.corrcoef(y, x)[0, 1]),
                       var_reduction=float(1 - y_adj.var(ddof=1) / y.var(ddof=1)))


def regression_adjust(df, y, treat="variant", covars=()):
    """OLS with robust SE: y ~ treat + centred covariates (Lin-style, covariates interacted with treat)."""
    X = pd.DataFrame({treat: df[treat].values})
    for c in covars:
        xc = df[c].values.astype(float)
        xc = xc - xc.mean()
        X[c] = xc
        X[c + ":t"] = xc * df[treat].values
    X = sm.add_constant(X)
    res = sm.OLS(df[y].values, X).fit(cov_type="HC1")
    b, se = res.params[treat], res.bse[treat]
    return dict(diff=float(b), se=float(se), lo=float(b - Z975 * se), hi=float(b + Z975 * se),
                p=float(res.pvalues[treat]), r2=float(res.rsquared))


# ---------------------------------------------------------------- multiple testing
def adjust_pvalues(pvals, method):
    """method: 'holm' or 'fdr_bh'."""
    return multipletests(pvals, alpha=0.05, method=method)[1]


# ---------------------------------------------------------------- sequential testing
def simulate_null_z_paths(n_looks, n_paths, seed=0):
    """Z-statistic paths for equally spaced looks under H0 (equal information per look)."""
    rng = np.random.default_rng(seed)
    inc = rng.standard_normal((n_paths, n_looks))
    s = np.cumsum(inc, axis=1)
    k = np.arange(1, n_looks + 1)
    return s / np.sqrt(k)


def calibrate_boundaries(n_looks, alpha=0.05, shape="obf", n_paths=400_000, seed=1):
    """Two-sided critical z per look, calibrated by simulation to total alpha.

    shape='obf'    : b_k = c / sqrt(t_k)   (O'Brien-Fleming-like: very strict early)
    shape='pocock' : b_k = c               (constant)
    """
    z = np.abs(simulate_null_z_paths(n_looks, n_paths, seed))
    t = np.arange(1, n_looks + 1) / n_looks
    scale = 1 / np.sqrt(t) if shape == "obf" else np.ones(n_looks)
    c = np.quantile((z / scale).max(axis=1), 1 - alpha)
    return c * scale


# ---------------------------------------------------------------- unified metric comparison
def arm_stats(num, den=None):
    """(estimate, variance-of-estimate) for a per-user mean (den=None) or a ratio of means."""
    n = len(num)
    if den is None:
        return float(num.mean()), float(num.var(ddof=1) / n)
    mn, md = num.mean(), den.mean()
    r = mn / md
    var = (num.var(ddof=1) + r ** 2 * den.var(ddof=1) - 2 * r * np.cov(num, den, ddof=1)[0, 1]) / (n * md ** 2)
    return float(r), float(var)


def compare(num_t, num_c, den_t=None, den_c=None):
    """Absolute and relative effect (treat vs ctrl) with delta-method CIs and a z-test p-value."""
    mt, vt = arm_stats(num_t, den_t)
    mc, vc = arm_stats(num_c, den_c)
    d, se = mt - mc, np.sqrt(vt + vc)
    rel = mt / mc - 1
    se_rel = np.sqrt(vt / mc ** 2 + mt ** 2 * vc / mc ** 4)
    return dict(mean_c=mc, mean_t=mt, diff=float(d), se=float(se), lo=float(d - Z975 * se), hi=float(d + Z975 * se),
                rel=float(rel), rel_lo=float(rel - Z975 * se_rel), rel_hi=float(rel + Z975 * se_rel),
                p=float(2 * stats.norm.sf(abs(d / se))))
