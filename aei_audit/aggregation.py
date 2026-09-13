"""
aei_audit.aggregation
=====================
The modifiable areal unit problem (MAUP), applied to the AI Usage Index.

MAUP has two separable components (Openshaw 1984):

  SCALE EFFECT   Results change with the NUMBER of units. Fifty states, nine
                 divisions, four regions: same underlying data, different
                 measured inequality. Quantified exactly by the Theil
                 between/within decomposition in indices.py.

  ZONING EFFECT  Results change with WHERE the boundaries fall, holding the
                 number of units fixed. This is the one nobody checks, and it
                 is the more damaging of the two, because it means a headline
                 statistic can be moved without changing a single observation.

The zoning test here is a Monte Carlo: generate many random CONTIGUOUS
partitions of the lower-48 states into k regions, recompute the concentration
statistic on each, and compare the real administrative geographies (Census
divisions, BEA regions) against that null distribution.

If the observed Census-division Gini sits in the middle of the random-partition
distribution, then "the AI divide across US regions" is substantially a
property of the map rather than of behaviour, and any policy claim keyed to it
inherits that arbitrariness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "STATE_ADJACENCY", "CENSUS_DIVISION", "CENSUS_REGION", "BEA_REGION",
    "CONTIGUOUS_48", "aggregate_panel", "random_contiguous_partition",
    "FEDERAL_REGION", "TIME_ZONE", "OFFICIAL_ZONINGS",
    "official_zoning_spread", "zoning_monte_carlo", "scale_ladder",
]

# ---------------------------------------------------------------------------
# adjacency graph of the lower 48 + DC (rook contiguity, Four Corners included)
# ---------------------------------------------------------------------------
STATE_ADJACENCY: dict[str, list[str]] = {
    "AL": ["FL", "GA", "MS", "TN"],
    "AZ": ["CA", "CO", "NM", "NV", "UT"],
    "AR": ["LA", "MS", "MO", "OK", "TN", "TX"],
    "CA": ["AZ", "NV", "OR"],
    "CO": ["AZ", "KS", "NE", "NM", "OK", "UT", "WY"],
    "CT": ["MA", "NY", "RI"],
    "DE": ["MD", "NJ", "PA"],
    "DC": ["MD", "VA"],
    "FL": ["AL", "GA"],
    "GA": ["AL", "FL", "NC", "SC", "TN"],
    "ID": ["MT", "NV", "OR", "UT", "WA", "WY"],
    "IL": ["IN", "IA", "KY", "MO", "WI"],
    "IN": ["IL", "KY", "MI", "OH"],
    "IA": ["IL", "MN", "MO", "NE", "SD", "WI"],
    "KS": ["CO", "MO", "NE", "OK"],
    "KY": ["IL", "IN", "MO", "OH", "TN", "VA", "WV"],
    "LA": ["AR", "MS", "TX"],
    "ME": ["NH"],
    "MD": ["DC", "DE", "PA", "VA", "WV"],
    "MA": ["CT", "NH", "NY", "RI", "VT"],
    "MI": ["IN", "OH", "WI"],
    "MN": ["IA", "ND", "SD", "WI"],
    "MS": ["AL", "AR", "LA", "TN"],
    "MO": ["AR", "IL", "IA", "KS", "KY", "NE", "OK", "TN"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NE": ["CO", "IA", "KS", "MO", "SD", "WY"],
    "NV": ["AZ", "CA", "ID", "OR", "UT"],
    "NH": ["ME", "MA", "VT"],
    "NJ": ["DE", "NY", "PA"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NY": ["CT", "MA", "NJ", "PA", "VT"],
    "NC": ["GA", "SC", "TN", "VA"],
    "ND": ["MN", "MT", "SD"],
    "OH": ["IN", "KY", "MI", "PA", "WV"],
    "OK": ["AR", "CO", "KS", "MO", "NM", "TX"],
    "OR": ["CA", "ID", "NV", "WA"],
    "PA": ["DE", "MD", "NJ", "NY", "OH", "WV"],
    "RI": ["CT", "MA"],
    "SC": ["GA", "NC"],
    "SD": ["IA", "MN", "MT", "NE", "ND", "WY"],
    "TN": ["AL", "AR", "GA", "KY", "MS", "MO", "NC", "VA"],
    "TX": ["AR", "LA", "NM", "OK"],
    "UT": ["AZ", "CO", "ID", "NM", "NV", "WY"],
    "VT": ["MA", "NH", "NY"],
    "VA": ["DC", "KY", "MD", "NC", "TN", "WV"],
    "WA": ["ID", "OR"],
    "WV": ["KY", "MD", "OH", "PA", "VA"],
    "WI": ["IA", "IL", "MI", "MN"],
    "WY": ["CO", "ID", "MT", "NE", "SD", "UT"],
}
CONTIGUOUS_48 = sorted(STATE_ADJACENCY)          # AK and HI excluded by design

# ---------------------------------------------------------------------------
# real administrative geographies, as zoning comparators
# ---------------------------------------------------------------------------
_DIV = {
    "New England": "CT ME MA NH RI VT",
    "Middle Atlantic": "NJ NY PA",
    "East North Central": "IL IN MI OH WI",
    "West North Central": "IA KS MN MO NE ND SD",
    "South Atlantic": "DE DC FL GA MD NC SC VA WV",
    "East South Central": "AL KY MS TN",
    "West South Central": "AR LA OK TX",
    "Mountain": "AZ CO ID MT NV NM UT WY",
    "Pacific": "AK CA HI OR WA",
}
CENSUS_DIVISION = {s: d for d, ss in _DIV.items() for s in ss.split()}

_REG = {
    "Northeast": "New England Middle Atlantic",
    "Midwest": "East North Central West North Central",
    "South": "South Atlantic East South Central West South Central",
    "West": "Mountain Pacific",
}
_div2reg = {}
for r, ds in _REG.items():
    for d in _DIV:
        if d in ds:
            _div2reg[d] = r
CENSUS_REGION = {s: _div2reg[d] for s, d in CENSUS_DIVISION.items()}

_BEA = {
    "New England": "CT ME MA NH RI VT",
    "Mideast": "DE DC MD NJ NY PA",
    "Great Lakes": "IL IN MI OH WI",
    "Plains": "IA KS MN MO NE ND SD",
    "Southeast": "AL AR FL GA KY LA MS NC SC TN VA WV",
    "Southwest": "AZ NM OK TX",
    "Rocky Mountain": "CO ID MT UT WY",
    "Far West": "AK CA HI NV OR WA",
}
BEA_REGION = {s: r for r, ss in _BEA.items() for s in ss.split()}

_FED10 = {   # EPA / HHS / FEMA share these state groupings
    "Region 1":  "CT ME MA NH RI VT",
    "Region 2":  "NJ NY",
    "Region 3":  "DE DC MD PA VA WV",
    "Region 4":  "AL FL GA KY MS NC SC TN",
    "Region 5":  "IL IN MI MN OH WI",
    "Region 6":  "AR LA NM OK TX",
    "Region 7":  "IA KS MO NE",
    "Region 8":  "CO MT ND SD UT WY",
    "Region 9":  "AZ CA HI NV",
    "Region 10": "AK ID OR WA",
}
FEDERAL_REGION = {s: r for r, ss in _FED10.items() for s in ss.split()}

# Predominant time zone. Several states are split (ID, KS, NE, ND, SD, TX, FL,
# IN, KY, MI, OR, TN); each is assigned by where most of its population lives.
# DOCUMENT THIS in the write-up -- it is an analyst choice, which is itself the
# point of the exercise.
_TZ = {
    "Eastern":  "CT DC DE FL GA IN KY MA MD ME MI NC NH NJ NY OH PA RI SC VA VT WV",
    "Central":  "AL AR IA IL KS LA MN MO MS ND NE OK SD TN TX WI",
    "Mountain": "AZ CO ID MT NM UT WY",
    "Pacific":  "CA NV OR WA",
    "Alaska":   "AK",
    "Hawaii":   "HI",
}
TIME_ZONE = {s: z for z, ss in _TZ.items() for s in ss.split()}

# Every official map, keyed by k. The headline zoning result comes from these,
# NOT from the Monte Carlo. The Monte Carlo supplies the null that tells you
# whether the spread across real maps is large.
OFFICIAL_ZONINGS = {
    "Census region (k=4)":   (CENSUS_REGION, 4),
    "Time zone (k=6)":       (TIME_ZONE, 6),
    "BEA region (k=8)":      (BEA_REGION, 8),
    "Census division (k=9)": (CENSUS_DIVISION, 9),
    "Federal region (k=10)": (FEDERAL_REGION, 10),
}


def official_zoning_spread(panel, stat_fn, **kw):
    """
    The headline table: the same statistic under every official US map.

    Two federal agencies drew k=8 and k=9 maps of the same country; two more
    drew different k=4 maps. If the measured statistic moves across them, the
    quantity is a property of the map as much as of behaviour.
    """
    import pandas as _pd
    rows = []
    for label, (mp, k) in OFFICIAL_ZONINGS.items():
        agg = aggregate_panel(panel, mp, **kw)
        rate = (agg["conversations"] / agg["denominator"]).to_numpy()
        rows.append({"zoning": label, "k": k,
                     "stat": stat_fn(rate, agg["denominator"].to_numpy())})
    out = _pd.DataFrame(rows).sort_values("k")
    lo, hi = out["stat"].min(), out["stat"].max()
    out.attrs["spread_pct_of_median"] = 100 * (hi - lo) / out["stat"].median()
    return out



# ---------------------------------------------------------------------------
def aggregate_panel(panel: pd.DataFrame,
                    mapping: dict[str, str],
                    unit_col: str = "state",
                    count_col: str = "conversations",
                    denom_col: str = "denominator",
                    extra_sum: tuple[str, ...] = ()) -> pd.DataFrame:
    """
    Roll a unit-level panel up to groups.

    CRITICAL: aggregate the RAW COUNTS and RAW DENOMINATORS, then recompute the
    index. Averaging unit-level AUI values is wrong (it is an unweighted mean of
    ratios) and is the single most common error in this kind of exercise.
    """
    df = panel.copy()
    df["_group"] = df[unit_col].map(mapping)
    unmapped = df.loc[df["_group"].isna(), unit_col].unique()
    if len(unmapped):
        raise ValueError(f"units absent from mapping: {sorted(unmapped)}")

    keys = [c for c in ("wave", "wave_date") if c in df.columns] + ["_group"]
    agg = {count_col: "sum", denom_col: "sum", **{c: "sum" for c in extra_sum}}
    out = df.groupby(keys, as_index=False).agg(agg).rename(columns={"_group": "unit"})

    # recompute the representation ratio at the new scale, within each wave
    if "wave" in out.columns:
        ctot = out.groupby("wave")[count_col].transform("sum")
        dtot = out.groupby("wave")[denom_col].transform("sum")
    else:
        ctot = out[count_col].sum()
        dtot = out[denom_col].sum()
    out["aui"] = (out[count_col] / ctot) / (out[denom_col] / dtot)
    out["rate"] = out[count_col] / out[denom_col]
    return out


# ---------------------------------------------------------------------------
def random_contiguous_partition(k: int, rng: np.random.Generator,
                                units: list[str] | None = None,
                                adjacency: dict | None = None,
                                max_tries: int = 200) -> dict[str, int]:
    """
    Random contiguous partition by multi-source region growing.

    Seed k units at random, then repeatedly pick an unassigned unit adjacent to
    an already-claimed region and assign it. Produces contiguous, connected,
    size-varying regions. Retries if the growth stalls on a disconnected
    remainder.
    """
    units = units or CONTIGUOUS_48
    adjacency = adjacency or STATE_ADJACENCY
    uset = set(units)

    for _ in range(max_tries):
        seeds = list(rng.choice(units, size=k, replace=False))
        label = {s: i for i, s in enumerate(seeds)}
        frontier = {i: [n for n in adjacency[s] if n in uset and n not in label]
                    for i, s in enumerate(seeds)}

        stalled = False
        while len(label) < len(units):
            live = [i for i in range(k)
                    if any(n not in label for n in frontier[i])]
            if not live:
                stalled = True
                break
            i = int(rng.choice(live))
            cands = [n for n in frontier[i] if n not in label]
            pick = str(rng.choice(cands))
            label[pick] = i
            frontier[i].extend(n for n in adjacency[pick]
                               if n in uset and n not in label)
        if not stalled:
            return label
    raise RuntimeError(f"could not build a contiguous {k}-partition in "
                       f"{max_tries} tries")


def zoning_monte_carlo(values: pd.Series,
                       counts: pd.Series,
                       denom: pd.Series,
                       k: int,
                       stat_fn,
                       n_sims: int = 2000,
                       seed: int = 20260822) -> np.ndarray:
    """
    Distribution of a concentration statistic over random k-region partitions.

    values : per-unit rate (unused directly; kept for signature symmetry)
    counts : raw conversation counts, indexed by unit
    denom  : raw denominator, indexed by unit
    stat_fn: callable(rate_array, weight_array) -> float

    Returns the array of simulated statistics. Compare the real geographies
    (Census divisions k=9, BEA k=8, Census regions k=4) against this.
    """
    rng = np.random.default_rng(seed)
    units = [u for u in CONTIGUOUS_48 if u in counts.index]
    c = counts.reindex(units).to_numpy(float)
    d = denom.reindex(units).to_numpy(float)

    out = np.empty(n_sims)
    for i in range(n_sims):
        lab = random_contiguous_partition(k, rng, units=units)
        g = np.array([lab[u] for u in units])
        cg = np.bincount(g, weights=c, minlength=k)
        dg = np.bincount(g, weights=d, minlength=k)
        keep = dg > 0
        out[i] = stat_fn(cg[keep] / dg[keep], dg[keep])
    return out


def scale_ladder(panel: pd.DataFrame, **kw) -> dict[str, pd.DataFrame]:
    """Convenience: the same panel at every standard scale."""
    return {
        "state (n=51)": panel,
        "BEA region (n=8)": aggregate_panel(panel, BEA_REGION, **kw),
        "Census division (n=9)": aggregate_panel(panel, CENSUS_DIVISION, **kw),
        "Census region (n=4)": aggregate_panel(panel, CENSUS_REGION, **kw),
    }
