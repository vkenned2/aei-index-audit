"""
aei_audit.uncertainty
=====================
Sampling uncertainty for a published index that ships without any.

Each AEI geographic figure is estimated from ONE WEEK of conversations, and
Anthropic suppresses cells below 15 conversations or 5 unique accounts. Small
units therefore carry large sampling error, and none of it is reported.

Two questions this module answers:

  Q1. What is a defensible interval around each unit's AUI?
      Treat the vector of unit conversation counts as a single draw from a
      multinomial and resample. This propagates the sampling design that the
      index actually has, and it respects closure automatically, because a
      multinomial draw always sums back to N.

  Q2. How many of the published rankings are real?
      For each pair (i, j), compute P(rank_i < rank_j) across bootstrap draws.
      A pair is "distinguishable" at 95% if that probability exceeds 0.975 or
      falls below 0.025. Report the fraction of pairs that clear the bar, and
      the length of the longest chain of mutually distinguishable units.

The expected headline is that adjacent ranks are noise. That is a useful,
quotable, and immediately actionable finding for anyone citing this data, and
it is the kind of thing an index owner genuinely wants to be told.

NOTE ON A LIMITATION: the multinomial treats conversations as independent
draws. They are not; they cluster within accounts. Anthropic reports unique
account counts alongside conversations, so a cluster-bootstrap at the account
level is preferable where those counts are available. Both are implemented.
Report the account-clustered version as primary and say why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["multinomial_bootstrap", "clustered_bootstrap",
           "aui_intervals", "rank_stability", "summarise_rank_stability"]


def multinomial_bootstrap(counts: np.ndarray, n_boot: int = 2000,
                          seed: int = 20260822) -> np.ndarray:
    """Resample the unit-count vector from Multinomial(N, observed shares)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(counts, dtype=float)
    n = int(round(c.sum()))
    p = c / c.sum()
    return rng.multinomial(n, p, size=n_boot).astype(float)


def clustered_bootstrap(counts: np.ndarray, accounts: np.ndarray,
                        n_boot: int = 2000, seed: int = 20260822) -> np.ndarray:
    """
    Account-clustered bootstrap. Approximates the design effect from
    conversations clustering within accounts by inflating per-unit variance by
    the average conversations-per-account.

    This is an approximation, not a true cluster resample, because the raw
    account-level data is not public. State the approximation explicitly.
    """
    rng = np.random.default_rng(seed)
    c = np.asarray(counts, dtype=float)
    a = np.asarray(accounts, dtype=float)
    deff = np.where(a > 0, c / np.maximum(a, 1.0), 1.0)     # convos per account
    eff_n = c / np.maximum(deff, 1.0)                        # effective sample
    draws = rng.poisson(np.maximum(eff_n, 1e-9), size=(n_boot, len(c)))
    return draws * np.maximum(deff, 1.0)


def aui_intervals(counts, denom, units, n_boot: int = 2000,
                  accounts=None, alpha: float = 0.05,
                  seed: int = 20260822) -> pd.DataFrame:
    """Point estimate plus percentile interval for each unit's AUI."""
    c = np.asarray(counts, float)
    d = np.asarray(denom, float)
    e = d / d.sum()

    boots = (clustered_bootstrap(c, np.asarray(accounts, float), n_boot, seed)
             if accounts is not None
             else multinomial_bootstrap(c, n_boot, seed))

    shares = boots / boots.sum(axis=1, keepdims=True)
    aui_b = shares / e

    lo = np.percentile(aui_b, 100 * alpha / 2, axis=0)
    hi = np.percentile(aui_b, 100 * (1 - alpha / 2), axis=0)
    pt = (c / c.sum()) / e

    return pd.DataFrame({
        "unit": units, "aui": pt, "lo": lo, "hi": hi,
        "ci_width": hi - lo,
        "rel_width": (hi - lo) / pt,
        "conversations": c,
    }).sort_values("aui", ascending=False).reset_index(drop=True)


def rank_stability(counts, denom, units, n_boot: int = 2000,
                   accounts=None, seed: int = 20260822) -> pd.DataFrame:
    """
    Pairwise P(unit i ranks above unit j) across bootstrap draws.
    Returns a units x units DataFrame.
    """
    c = np.asarray(counts, float)
    d = np.asarray(denom, float)
    e = d / d.sum()
    boots = (clustered_bootstrap(c, np.asarray(accounts, float), n_boot, seed)
             if accounts is not None
             else multinomial_bootstrap(c, n_boot, seed))
    aui_b = (boots / boots.sum(axis=1, keepdims=True)) / e

    n = len(units)
    P = np.zeros((n, n))
    for i in range(n):
        P[i] = (aui_b[:, i][:, None] > aui_b).mean(axis=0)
    np.fill_diagonal(P, np.nan)
    return pd.DataFrame(P, index=units, columns=units)


def summarise_rank_stability(P: pd.DataFrame, thresh: float = 0.975) -> dict:
    """
    Headline numbers from the pairwise matrix.

    distinguishable_pairs : fraction of pairs where the ordering is resolved
    adjacent_resolved     : fraction of ADJACENT published ranks that are
                            resolved. This is the quotable one, and it is
                            usually far lower than people assume.
    """
    A = P.to_numpy()
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    vals = A[iu]
    resolved = np.mean((vals > thresh) | (vals < 1 - thresh))

    adj = [A[i, i + 1] for i in range(n - 1)]
    adj_res = np.mean([(v > thresh) or (v < 1 - thresh) for v in adj])

    return {
        "n_units": n,
        "distinguishable_pairs": float(resolved),
        "adjacent_ranks_resolved": float(adj_res),
        "interpretation": (
            f"{100*adj_res:.0f}% of adjacent published rankings are "
            f"statistically resolved at the {100*thresh:.1f}% level. The "
            f"remainder are ties that the sampling design cannot separate."
        ),
    }
