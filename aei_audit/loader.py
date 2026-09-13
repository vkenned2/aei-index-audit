"""
aei_audit.loader
================
Parser and share-space methods for the real AEI release format.

VERIFIED AGAINST: aei_claude_ai_2026-06-26.csv (June 2026 "Cadences" release)

FORMAT
------
Long: date_start, date_end, geo_id, geo_level, category_name, hierarchy_level,
metric_id, value, node_name, node_external_id.

  geo_level     : global | country | subregion   (US states are 'subregion')
  geo_id        : 'US-CA' style for US states; 'USA' is the national aggregate
  category_name : overall | onet | request | soc_occupation
  metric_id     : 53 metrics; the two that matter here are
                    usage_pct              -- share of all usage, sums to 100
                    usage_per_capita_index -- the AUI, 1.0 = proportional

THREE FACTS THAT DRIVE EVERY DESIGN CHOICE BELOW
------------------------------------------------
1. THERE ARE NO RAW COUNTS. Only shares and the index. Anything that needs to
   sum conversations must instead sum shares.

2. THE DENOMINATOR IS RECOVERABLE. AUI = usage_share / pop_share, so
       pop_share = usage_pct / usage_per_capita_index
   Verified to sum to 100.000 (Apr 2026) and 99.977 (May 2026). This is
   Anthropic's OWN implied denominator, which is strictly better than
   substituting an external population estimate and hoping it matches.

   Both usage_pct and the recovered pop_share are additive shares, so
   aggregation is exact: sum each over the group, then divide.

3. VALUES ARE ROUNDED TO 2 DECIMAL PLACES. For Wyoming (usage_pct = 0.07) that
   is +/-7.1% relative uncertainty before any sampling consideration; median
   across states is 0.47%. Rounding, not sampling, is the binding precision
   constraint for small units.

   This explains an artifact that looks like a finding: the recovered pop_share
   drifts 8.1% for SD and 7.0% for WY between April and May. Population does
   not move 8% in a month. The drift IS the rounding error for those states.
   Do not report it as denominator instability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_us_states", "load_occupation_shares", "aggregate_shares",
           "rounding_interval", "rounding_monte_carlo", "rank_resolution"]

ROUND_UNIT = 0.01          # values reported to 2dp
HALF_UNIT = ROUND_UNIT / 2

_USECOLS = ["date_start", "date_end", "geo_id", "geo_level", "category_name",
            "hierarchy_level", "metric_id", "value", "node_name"]


def load_us_states(path: str, drop_territories: bool = True) -> pd.DataFrame:
    """
    Return one row per state x month with usage share, AUI, and the recovered
    implied population share.

    drop_territories: PR is present but is not a state and has no ACS/OES
    comparability. Dropped by default; record the choice in the paper.
    """
    d = pd.read_csv(path, usecols=_USECOLS,
                    dtype={"geo_id": "category", "geo_level": "category",
                           "category_name": "category", "metric_id": "category"},
                    low_memory=False)

    m = (d["geo_id"].astype(str).str.match(r"^US-[A-Z]{2}$")
         & (d["category_name"] == "overall")
         & d["metric_id"].isin(["usage_pct", "usage_per_capita_index"]))

    w = (d[m].pivot_table(index=["geo_id", "date_start"], columns="metric_id",
                          values="value", observed=True)
             .reset_index()
             .rename(columns={"usage_per_capita_index": "aui"}))

    w["state"] = w["geo_id"].astype(str).str[3:]
    if drop_territories:
        w = w[~w["state"].isin(["PR", "VI", "GU", "AS", "MP"])]

    w = w.dropna(subset=["usage_pct", "aui"])
    if (w["aui"] <= 0).any():
        raise ValueError("non-positive AUI; cannot recover denominator")

    w["pop_share"] = w["usage_pct"] / w["aui"]
    w = w.rename(columns={"date_start": "wave"})
    return w[["state", "wave", "usage_pct", "aui", "pop_share"]].reset_index(drop=True)


def load_occupation_shares(path: str) -> pd.DataFrame:
    """
    State x occupation usage shares from category_name == 'soc_occupation'.

    491 occupations x 52 states in the June 2026 release. This is the input to
    the shift-share decomposition: how much of a state's AUI is its
    occupational MIX versus within-occupation INTENSITY.
    """
    d = pd.read_csv(path, usecols=_USECOLS,
                    dtype={"geo_id": "category", "category_name": "category",
                           "metric_id": "category", "node_name": "category"},
                    low_memory=False)
    m = (d["geo_id"].astype(str).str.match(r"^US-[A-Z]{2}$")
         & (d["category_name"] == "soc_occupation")
         & (d["metric_id"] == "pct"))
    out = d[m].copy()
    out["state"] = out["geo_id"].astype(str).str[3:]
    return (out.rename(columns={"node_name": "occupation", "date_start": "wave",
                                "value": "usage_pct_within_state"})
               [["state", "wave", "occupation", "usage_pct_within_state"]]
               .reset_index(drop=True))


# ---------------------------------------------------------------------------
def aggregate_shares(panel: pd.DataFrame, mapping: dict[str, str],
                     unit_col: str = "state") -> pd.DataFrame:
    """
    Exact aggregation in share space. Replaces aggregation.aggregate_panel()
    for count-free data.

    Both usage_pct and pop_share are additive over units, so:
        AUI_group = sum(usage_pct) / sum(pop_share)
    This is exact, not an approximation, and it never averages ratios.
    """
    df = panel.copy()
    df["_g"] = df[unit_col].map(mapping)
    missing = sorted(df.loc[df["_g"].isna(), unit_col].unique())
    if missing:
        raise ValueError(f"units absent from mapping: {missing}")

    keys = (["wave"] if "wave" in df.columns else []) + ["_g"]
    out = (df.groupby(keys, as_index=False)[["usage_pct", "pop_share"]].sum()
             .rename(columns={"_g": "unit"}))
    out["aui"] = out["usage_pct"] / out["pop_share"]
    return out


# ---------------------------------------------------------------------------
# uncertainty: rounding, not sampling
# ---------------------------------------------------------------------------
def rounding_interval(value: float) -> tuple[float, float]:
    """Interval consistent with a value reported to 2dp."""
    return value - HALF_UNIT, value + HALF_UNIT


def rounding_monte_carlo(panel: pd.DataFrame, stat_fn, n_sims: int = 4000,
                         seed: int = 20260822) -> np.ndarray:
    """
    Propagate 2dp rounding through any statistic.

    For each draw, replace every reported usage_pct and pop_share with a
    uniform draw from its rounding interval, renormalise the shares to sum to
    their reported total, recompute the statistic.

    This is the correct uncertainty model for this data. A multinomial
    bootstrap is NOT available: there are no counts to resample. Say so
    explicitly in the paper rather than importing an inapplicable method.
    """
    rng = np.random.default_rng(seed)
    u = panel["usage_pct"].to_numpy(float)
    p = panel["pop_share"].to_numpy(float)
    n = len(u)

    out = np.empty(n_sims)
    for i in range(n_sims):
        ud = np.clip(u + rng.uniform(-HALF_UNIT, HALF_UNIT, n), 1e-9, None)
        pd_ = np.clip(p + rng.uniform(-HALF_UNIT, HALF_UNIT, n), 1e-9, None)
        ud *= u.sum() / ud.sum()
        pd_ *= p.sum() / pd_.sum()
        out[i] = stat_fn(ud / pd_, pd_)
    return out


def rank_resolution(panel: pd.DataFrame, n_sims: int = 4000,
                    seed: int = 20260822, thresh: float = 0.975) -> dict:
    """
    How many published rankings survive rounding alone?

    Note this is a LOWER BOUND on total uncertainty: it includes rounding but
    not sampling error, which is unobservable without counts. If rankings are
    already unresolved under rounding alone, adding sampling can only make it
    worse. That is the honest framing and it is a strong one.
    """
    rng = np.random.default_rng(seed)
    u = panel["usage_pct"].to_numpy(float)
    p = panel["pop_share"].to_numpy(float)
    units = panel["state"].tolist() if "state" in panel else panel["unit"].tolist()
    n = len(u)

    draws = np.empty((n_sims, n))
    for i in range(n_sims):
        ud = np.clip(u + rng.uniform(-HALF_UNIT, HALF_UNIT, n), 1e-9, None)
        pd_ = np.clip(p + rng.uniform(-HALF_UNIT, HALF_UNIT, n), 1e-9, None)
        draws[i] = (ud * u.sum() / ud.sum()) / (pd_ * p.sum() / pd_.sum())

    order = np.argsort(-(u / p))
    draws, units = draws[:, order], [units[i] for i in order]

    P = np.zeros((n, n))
    for i in range(n):
        P[i] = (draws[:, i][:, None] > draws).mean(axis=0)
    np.fill_diagonal(P, np.nan)

    adj = np.array([P[i, i + 1] for i in range(n - 1)])
    iu = np.triu_indices(n, 1)
    allp = P[iu]
    return {
        "units_ranked": units,
        "pairwise": pd.DataFrame(P, index=units, columns=units),
        "adjacent_resolved": float(np.mean((adj > thresh) | (adj < 1 - thresh))),
        "all_pairs_resolved": float(np.mean((allp > thresh) | (allp < 1 - thresh))),
        "note": ("Lower bound: rounding only. Sampling error is unobservable "
                 "without conversation counts."),
    }
