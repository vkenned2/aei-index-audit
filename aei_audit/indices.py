"""
aei_audit.indices
=================
Concentration and representation statistics for share-based indices.

The Anthropic AI Usage Index is a representation ratio:

    AUI_i = (usage_i / sum_j usage_j) / (pop_i / sum_j pop_j)

which is algebraically identical to (usage_i / pop_i) / (sum usage / sum pop),
i.e. a per-capita rate normalised by the global per-capita rate. AUI = 1 means
proportional representation. This is the same object as a "parity ratio" in the
conservation-representation literature and a "location quotient" in regional
economics.

Two properties matter for everything downstream:

1. CLOSURE. Usage shares are constrained to sum to 1. A real increase in one
   unit mechanically decreases every other unit's share. Standard regression
   on shares therefore violates independence. Use clr() to move to an
   unconstrained log-ratio space before modelling.

2. DECOMPOSABILITY. Of the common concentration measures, only the Theil index
   decomposes exactly into between-group and within-group components. That is
   what makes it the right instrument for measuring how much reported
   inequality survives aggregation.

References
----------
Aitchison (1986) The Statistical Analysis of Compositional Data.
Shorrocks (1980) The class of additively decomposable inequality measures.
Theil (1967) Economics and Information Theory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "usage_share", "representation_ratio", "clr", "alr",
    "gini", "theil", "theil_decompose", "hhi", "cv", "atkinson",
    "lorenz_points", "concentration_table",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# basic constructions
# ---------------------------------------------------------------------------
def usage_share(counts) -> np.ndarray:
    x = np.asarray(counts, dtype=float)
    if np.any(x < 0):
        raise ValueError("negative counts")
    return x / x.sum()


def representation_ratio(counts, denom) -> np.ndarray:
    """
    The AUI. counts and denom are raw levels, not shares.
    Returns observed share divided by expected share.
    """
    s = usage_share(counts)
    e = usage_share(denom)
    if np.any(e <= 0):
        raise ValueError("zero denominator for at least one unit")
    return s / e


# ---------------------------------------------------------------------------
# compositional transforms  (the closure fix)
# ---------------------------------------------------------------------------
def clr(shares, zero_strategy: str = "raise"):
    """
    Centred log-ratio. clr(x)_i = log(x_i) - mean_j log(x_j).

    Maps the simplex to an unconstrained space where a regression coefficient
    reads as a change RELATIVE TO the geometric mean of all units. This is the
    honest estimand when the outcome is a share.

    zero_strategy:
      'raise'  -- fail loudly (default; you should know if you have zeros)
      'bayes'  -- Bayesian-multiplicative replacement, Martin-Fernandez (2015)
    """
    x = np.asarray(shares, dtype=float)
    if np.any(x <= 0):
        if zero_strategy == "raise":
            raise ValueError(
                "Zero or negative shares present; clr is undefined. Either "
                "aggregate away the zeros or pass zero_strategy='bayes' and "
                "document the replacement in your write-up."
            )
        elif zero_strategy == "bayes":
            n = len(x)
            delta = 0.5 / n                       # Jeffreys prior
            zeros = x <= 0
            x = x.copy()
            x[zeros] = delta
            x[~zeros] = x[~zeros] * (1 - zeros.sum() * delta)
        else:
            raise ValueError(f"unknown zero_strategy {zero_strategy!r}")
    lg = np.log(x)
    return lg - lg.mean()


def alr(shares, ref_index: int = -1):
    """Additive log-ratio against a chosen reference unit."""
    x = np.asarray(shares, dtype=float)
    return np.log(x) - np.log(x[ref_index])


# ---------------------------------------------------------------------------
# concentration measures
# ---------------------------------------------------------------------------
def gini(x, weights=None) -> float:
    """
    Population-weighted Gini of a per-unit rate.

    IMPORTANT: pass weights (e.g. working-age population). An unweighted Gini
    over states treats Wyoming and California as equally important, which is
    almost never the quantity anyone means. Report both and say which is which.
    """
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    if cxw[-1] <= _EPS:
        return 0.0
    # Brown / trapezoid formula on the Lorenz curve
    return float(
        1.0 - np.sum((cxw[1:] + cxw[:-1]) * np.diff(cw)) / (cxw[-1] * cw[-1])
    )


def theil(x, weights=None) -> float:
    """
    Theil T (GE(1)) of a per-unit rate. Additively decomposable.
    """
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    if np.any(x <= 0):
        raise ValueError("Theil undefined for non-positive rates")
    p = w / w.sum()
    mu = np.sum(p * x)
    return float(np.sum(p * (x / mu) * np.log(x / mu)))


def theil_decompose(x, groups, weights=None) -> dict:
    """
    Exact decomposition  T_total = T_between + T_within.

    THIS IS THE CENTREPIECE OF THE SCALE ANALYSIS.

    T_between is the inequality you still see after aggregating units into
    `groups`. T_within is the inequality that aggregation destroys. Running
    this over a hierarchy (state -> division -> region) gives an exact
    accounting of how much of the reported "AI divide" is scale-dependent.

    Returns dict with total, between, within, and share_between.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(groups)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)

    p = w / w.sum()
    mu = np.sum(p * x)
    total = theil(x, w)

    between, within = 0.0, 0.0
    for gv in pd.unique(g):
        m = g == gv
        pg = p[m].sum()
        mug = np.sum(p[m] * x[m]) / pg
        between += pg * (mug / mu) * np.log(mug / mu)
        within += pg * (mug / mu) * theil(x[m], w[m])

    return {
        "total": float(total),
        "between": float(between),
        "within": float(within),
        "share_between": float(between / total) if total > _EPS else np.nan,
        "n_groups": int(pd.unique(g).size),
        "check_residual": float(total - between - within),   # should be ~0
    }


def hhi(shares) -> float:
    s = np.asarray(shares, dtype=float)
    return float(np.sum(s ** 2))


def cv(x, weights=None) -> float:
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    p = w / w.sum()
    mu = np.sum(p * x)
    var = np.sum(p * (x - mu) ** 2)
    return float(np.sqrt(var) / mu)


def atkinson(x, epsilon: float = 0.5, weights=None) -> float:
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    p = w / w.sum()
    mu = np.sum(p * x)
    if np.isclose(epsilon, 1.0):
        return float(1 - np.exp(np.sum(p * np.log(x))) / mu)
    ede = (np.sum(p * x ** (1 - epsilon))) ** (1 / (1 - epsilon))
    return float(1 - ede / mu)


def lorenz_points(x, weights=None):
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_pop = np.insert(np.cumsum(w) / w.sum(), 0, 0.0)
    cum_use = np.insert(np.cumsum(x * w) / np.sum(x * w), 0, 0.0)
    return cum_pop, cum_use


# ---------------------------------------------------------------------------
def concentration_table(x, weights=None, groups=None) -> pd.Series:
    """Every measure at once, for a single scale. Convenience for notebooks."""
    out = {
        "gini": gini(x, weights),
        "theil": theil(x, weights),
        "cv": cv(x, weights),
        "atkinson_0.5": atkinson(x, 0.5, weights),
        "hhi_of_shares": hhi(usage_share(np.asarray(x) * (
            np.ones_like(x) if weights is None else np.asarray(weights)))),
        "n_units": len(np.asarray(x)),
    }
    if groups is not None:
        d = theil_decompose(x, groups, weights)
        out.update({f"theil_{k}": v for k, v in d.items()
                    if k in ("between", "within", "share_between")})
    return pd.Series(out)
