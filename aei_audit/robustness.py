"""
aei_audit.robustness
====================
Four checks that attack specific threats to the headline claims.

Each answers a question a referee will ask:

  weighting_sensitivity  "Does 45% inequality-lost depend on weighting by
                          population rather than counting states equally?"
  leave_one_out          "Is DC or California carrying the whole result?"
  multi_k_zoning_null    "Why nine regions? Is the 95th-percentile result
                          special to k=9?"
  temporal_stability     "Is the ordering stable even where the distribution
                          moves?"
  placebo_breaks         "Is the Nov->Feb distributional change unusual, or
                          does every wave boundary look like that?"

Nothing here is added for volume. A robustness check that cannot change a
conclusion is decoration and should be cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indices as ix
from . import aggregation as ag

__all__ = ["weighting_sensitivity", "leave_one_out", "multi_k_zoning_null",
           "temporal_stability", "placebo_breaks"]

MAPS = [("Census region", "census_region"), ("Time zone", "time_zone"),
        ("BEA region", "bea_region"), ("Census division", "census_division"),
        ("Federal region", "federal_region")]


def _roll(g, col):
    a = g.groupby(col)[["usage_share", "pop_share_fixed"]].sum()
    a["aui"] = a.usage_share / a.pop_share_fixed
    return a


# ---------------------------------------------------------------------------
def weighting_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Headline statistics under three weighting conventions.

    population  -- weight by working-age population (default; the quantity
                   most people mean by "inequality in AI use")
    equal       -- every state counts once (a state-level, not person-level,
                   claim; Wyoming = California)
    usage       -- weight by usage share (describes where the conversations
                   are, not where the people are)

    A conclusion that survives only one convention is not a conclusion.
    """
    rows = []
    for w, g in panel.groupby("wave"):
        a = g.aui_fixed.values
        for name, wt in [("population", g.pop_share_fixed.values),
                         ("equal", np.ones(len(g))),
                         ("usage", g.usage_share.values)]:
            d = ix.theil_decompose(a, g.census_division.values, weights=wt)
            rows.append({"wave": w, "weighting": name,
                         "gini": ix.gini(a, wt), "theil": ix.theil(a, wt),
                         "share_between_k9": d["share_between"],
                         "pct_lost_k9": 100 * (1 - d["share_between"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def leave_one_out(panel: pd.DataFrame, wave: str) -> pd.DataFrame:
    """
    Recompute the headline statistics with each state removed in turn.

    Shares are RENORMALISED after removal, which is the correct operation on
    compositional data: dropping a unit does not leave the rest summing to 100.
    """
    g0 = panel[panel.wave == wave]
    full_between = ix.theil_decompose(
        g0.aui_fixed.values, g0.census_division.values,
        weights=g0.pop_share_fixed.values)["share_between"]
    full_gini = ix.gini(g0.aui_fixed.values, g0.pop_share_fixed.values)

    rows = []
    for s in g0.state:
        g = g0[g0.state != s].copy()
        g["usage_share"] = 100 * g.usage_share / g.usage_share.sum()
        g["pop_share_fixed"] = 100 * g.pop_share_fixed / g.pop_share_fixed.sum()
        g["aui_fixed"] = g.usage_share / g.pop_share_fixed
        d = ix.theil_decompose(g.aui_fixed.values, g.census_division.values,
                               weights=g.pop_share_fixed.values)
        rows.append({"dropped": s, "share_between_k9": d["share_between"],
                     "gini": ix.gini(g.aui_fixed.values, g.pop_share_fixed.values)})
    out = pd.DataFrame(rows)
    out["d_share_between"] = out.share_between_k9 - full_between
    out["d_gini"] = out.gini - full_gini
    out.attrs["full_share_between"] = full_between
    out.attrs["full_gini"] = full_gini
    return out.sort_values("d_gini")


# ---------------------------------------------------------------------------
def multi_k_zoning_null(panel: pd.DataFrame, wave: str,
                        ks=(4, 6, 8, 9, 10), n_sims: int = 800,
                        seed: int = 20260822) -> pd.DataFrame:
    """
    Repeat the zoning null at several group counts.

    Answers "why nine?". For each k, the null is random contiguous
    k-partitions; the observed value is whichever REAL government map has that
    k. If real maps sit in the upper tail at every k, the finding is about
    administrative geography generally, not an artifact of one partition.
    """
    g = panel[panel.wave == wave].set_index("state")
    g = g[g.index.isin(ag.CONTIGUOUS_48)]

    real = {}
    for nm, col in MAPS:
        a = _roll(g, col)
        real.setdefault(len(a), []).append(
            (nm, ix.gini(a.aui.values, a.pop_share_fixed.values)))

    rows = []
    for k in ks:
        sim = ag.zoning_monte_carlo(g.aui_fixed, g.usage_share, g.pop_share_fixed,
                                    k=k, stat_fn=ix.gini, n_sims=n_sims, seed=seed)
        for nm, obs in real.get(k, [(None, np.nan)]):
            rows.append({
                "k": k, "real_map": nm, "observed": obs,
                "null_median": float(np.median(sim)),
                "null_p05": float(np.percentile(sim, 5)),
                "null_p95": float(np.percentile(sim, 95)),
                "percentile": float(100 * (sim < obs).mean()) if nm else np.nan,
                "null_spread_pct_med": float(100 * (sim.max() - sim.min())
                                             / np.median(sim)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def temporal_stability(panel: pd.DataFrame) -> dict:
    """
    Ordering versus distribution.

    A high Spearman rho alongside large share movements is NOT a contradiction:
    rank correlation is nearly blind to magnitude. Reporting both together is
    the finding, because it separates "who is ahead" (stable) from "by how
    much" (not stable).
    """
    A = panel.pivot(index="state", columns="wave", values="aui_fixed")
    S = panel.pivot(index="state", columns="wave", values="usage_share")
    W = list(A.columns)

    rho = A.corr(method="spearman")
    R = A.rank(ascending=False)

    top = int(np.ceil(0.1 * len(A)))
    persist = {}
    for i in range(len(W) - 1):
        a = set(A[W[i]].nlargest(top).index)
        b = set(A[W[i + 1]].nlargest(top).index)
        persist[f"{W[i]}->{W[i+1]}"] = len(a & b) / top

    return {
        "spearman": rho,
        "mean_abs_rank_move": {f"{W[i]}->{W[i+1]}":
                               float((R[W[i+1]] - R[W[i]]).abs().mean())
                               for i in range(len(W) - 1)},
        "max_abs_share_move_pp": {f"{W[i]}->{W[i+1]}":
                                  float((S[W[i+1]] - S[W[i]]).abs().max())
                                  for i in range(len(W) - 1)},
        "top_decile_persistence": persist,
        "note": ("High rho with large pp moves means the ORDER is stable while "
                 "the DISTRIBUTION is not. Report both."),
    }


# ---------------------------------------------------------------------------
def placebo_breaks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Characterise every wave boundary, not just the one of interest.

    For each adjacent pair, measure how much the cross-state distribution
    moved: total variation distance between usage-share vectors, change in
    Theil, and the largest single-state move. If the 2025-11 -> 2026-02
    boundary (which carries the platform change) is not an outlier among
    these, the "break" is not distinguishable from ordinary wave-to-wave
    churn. If it IS an outlier, that supports the non-identification claim.
    """
    S = panel.pivot(index="state", columns="wave", values="usage_share")
    A = panel.pivot(index="state", columns="wave", values="aui_fixed")
    P = panel.groupby("state").pop_share_fixed.first()
    W = list(S.columns)

    rows = []
    for i in range(len(W) - 1):
        a, b = W[i], W[i + 1]
        tvd = 0.5 * (S[b] - S[a]).abs().sum()          # total variation, in pp
        rows.append({
            "boundary": f"{a} -> {b}",
            "tvd_pp": float(tvd),
            "d_theil": float(ix.theil(A[b].values, P.values)
                             - ix.theil(A[a].values, P.values)),
            "max_state_move_pp": float((S[b] - S[a]).abs().max()),
            "which_state": (S[b] - S[a]).abs().idxmax(),
            "spearman": float(A[a].corr(A[b], method="spearman")),
        })
    return pd.DataFrame(rows)
