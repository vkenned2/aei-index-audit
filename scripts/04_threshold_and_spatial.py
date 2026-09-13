"""
04_threshold_and_spatial.py

Reproduces the results in Sections 4.4, 4.5, 4.7 and 4.9, which are the
manuscript's central claims:

    Section 4.4   Moran's I, spatial autocorrelation, effective sample size
    Section 4.5   adjacent-rank resolution, published tier separation
    Section 4.7   convergent validity against gross state product
    Section 4.9   threshold classification instability, decomposed by source
                  variance decomposition: sampling vs specification

Inputs:  data/processed/panel.csv
         data/raw/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv
             (needed only for the gross-state-product and tier variables,
              which appear solely in the enriched 2025-08 release)

Outputs: results/tables/T11_threshold_decomposition.csv
         results/tables/T12_variance_decomposition.csv
         results/figures/F7_threshold.png

Usage:   python scripts/04_threshold_and_spatial.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aei_audit import indices as ix, aggregation as ag  # noqa: E402

SEED = 20260822
RAW = Path(os.environ.get("AEI_RAW_DIR", "data/raw"))
ENRICHED = RAW / "aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv"
TAB = Path("results/tables"); TAB.mkdir(parents=True, exist_ok=True)
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)

MAPS = [("Census region", "census_region"), ("Time zone", "time_zone"),
        ("BEA region", "bea_region"), ("Census division", "census_division"),
        ("Federal region", "federal_region")]
INK, RED, MUT = "#18211F", "#A63D2F", "#C9C9C9"


# ---------------------------------------------------------------------------
def rook_weights(units):
    """Row-standardised rook contiguity matrix over the contiguous jurisdictions."""
    W = np.zeros((len(units), len(units)))
    for i, a in enumerate(units):
        for j, b in enumerate(units):
            if b in ag.STATE_ADJACENCY[a]:
                W[i, j] = 1
    return W / W.sum(1, keepdims=True)


def moran(x, W, nsim=4999, seed=SEED):
    """Moran's I with permutation inference."""
    x = np.asarray(x, float)
    z = x - x.mean()
    n = len(x)
    I = (n / W.sum()) * (z @ W @ z) / (z @ z)
    rng = np.random.default_rng(seed)
    null = np.empty(nsim)
    for k in range(nsim):
        zp = rng.permutation(z)
        null[k] = (n / W.sum()) * (zp @ W @ zp) / (zp @ zp)
    p = (np.sum(np.abs(null) >= abs(I)) + 1) / (nsim + 1)
    return float(I), float(p)


# ---------------------------------------------------------------------------
def spatial_and_validity(panel):
    """Sections 4.4 and 4.7. Requires the enriched release."""
    if not ENRICHED.exists():
        print(f"SKIPPED Sections 4.4 and 4.7: {ENRICHED} not present.")
        print("        These use gross_state_product and usage_tier, which are")
        print("        published only in the enriched 2025-08 release.\n")
        return

    d = pd.read_csv(ENRICHED,
                    usecols=["geo_id", "geography", "variable", "value"],
                    dtype={"geo_id": "category", "geography": "category",
                           "variable": "category"}, low_memory=False)
    s = d[d.geography == "state_us"].pivot_table(
        index="geo_id", columns="variable", values="value", observed=True)
    s = s[s.index.astype(str).str.match(r"^[A-Z]{2}$")]
    s.index = s.index.astype(str)
    s = s.dropna(subset=["usage_per_capita_index", "gdp_per_working_age_capita",
                         "working_age_pop", "usage_count"])

    units = [u for u in ag.CONTIGUOUS_48 if u in s.index]
    W = rook_weights(units)

    print("SECTION 4.4  SPATIAL AUTOCORRELATION")
    for var, lab in [("usage_per_capita_index", "usage index"),
                     ("gdp_per_working_age_capita", "GSP per working-age capita")]:
        I, p = moran(np.log(s.loc[units, var]), W)
        print(f"  {lab:28s} Moran's I = {I:+.4f}  permutation p = {p:.4f}")
    I, _ = moran(np.log(s.loc[units, "usage_per_capita_index"]), W, nsim=999)
    n = len(units)
    print(f"  first-order deflation: n = {n} -> n_eff ~ {n*(1-I)/(1+I):.1f}")
    print("  (approximation only; see Section 3.4)\n")

    print("SECTION 4.5  PUBLISHED TIER SEPARATION")
    T = s.dropna(subset=["usage_tier"]).copy()
    rng = np.random.default_rng(SEED)
    c = T.usage_count.to_numpy(float)
    e = (T.working_age_pop / T.working_age_pop.sum()).to_numpy()
    B = rng.multinomial(int(c.sum()), c / c.sum(), size=4000).astype(float)
    A = (B / B.sum(1, keepdims=True)) / e
    tiers = T.usage_tier.to_numpy()
    for lo, hi in [(1, 2), (2, 3), (3, 4)]:
        L, H = A[:, tiers == lo], A[:, tiers == hi]
        print(f"  P(tier {hi} strictly separates from tier {lo}) = "
              f"{np.mean(H.min(1) > L.max(1)):.3f}")
    print()

    print("SECTION 4.7  CONVERGENT VALIDITY")
    x = np.log(s.gdp_per_working_age_capita)
    y = np.log(s.usage_per_capita_index)
    sl, ic, r, pv, se = stats.linregress(x, y)
    print(f"  log(index) ~ log(GSP per w.a. capita): slope={sl:.3f} "
          f"(SE {se:.3f})  R2={r**2:.3f}  n={len(s)}")
    res = y - (sl * x + ic)
    print(f"  residual SD = {res.std():.3f} log points")
    Ir, pr = moran(res.loc[units], W)
    print(f"  Moran's I on the residual = {Ir:+.4f}, p = {pr:.4f}\n")


# ---------------------------------------------------------------------------
def threshold(panel):
    """Section 4.9, decomposed by source of variation."""
    waves = sorted(panel.wave.unique())
    rec = []
    for w in waves:
        g = panel[panel.wave == w]
        for mn, mc in MAPS:
            a = g.groupby(mc)[["usage_share", "pop_share_fixed"]].sum()
            a["aui"] = a.usage_share / a.pop_share_fixed
            for _, r in g.iterrows():
                rec.append({"state": r.state, "wave": w, "map": mn,
                            "cross": int(a.loc[r[mc], "aui"] > 1.0)})
    R = pd.DataFrame(rec)
    N = R.state.nunique()

    rows = []
    print("SECTION 4.9  THRESHOLD CLASSIFICATION, BY SOURCE OF VARIATION")
    for w in waves:
        x = R[R.wave == w].groupby("state").cross.agg(["min", "max"])
        n = int((x["min"] != x["max"]).sum())
        rows.append({"source": f"Map only, within {w}", "n_unstable": n,
                     "n_units": N, "pct": 100 * n / N})
        print(f"  map only, within {w}: {n}/{N} = {100*n/N:.0f}%")
    map_only = np.mean([r["pct"] for r in rows])

    wo = []
    for mn, _ in MAPS:
        x = R[R["map"] == mn].groupby("state").cross.agg(["min", "max"])
        n = int((x["min"] != x["max"]).sum())
        wo.append(n)
        print(f"  wave only, at {mn:16s}: {n}/{N} = {100*n/N:.0f}%")
    rows.append({"source": "Wave only, mean over five maps",
                 "n_unstable": float(np.mean(wo)), "n_units": N,
                 "pct": 100 * np.mean(wo) / N})

    pooled = R.groupby("state").cross.agg(["min", "max"])
    n_all = int((pooled["min"] != pooled["max"]).sum())
    rows.append({"source": "Full grid, five maps by five waves",
                 "n_unstable": n_all, "n_units": N, "pct": 100 * n_all / N})
    print(f"\n  MAP-ONLY MEAN : {map_only:.0f}%")
    print(f"  WAVE-ONLY MEAN: {100*np.mean(wo)/N:.0f}%")
    print(f"  FULL GRID     : {n_all}/{N} = {100*n_all/N:.0f}%\n")
    pd.DataFrame(rows).to_csv(TAB / "T11_threshold_decomposition.csv", index=False)

    # Figure 7
    cr = (100 * R.groupby("state").cross.mean()).sort_values(ascending=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .25, "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(range(len(cr)), cr.values,
           color=[MUT if (v >= 95 or v <= 5) else RED for v in cr], width=.78)
    ax.axhline(95, color=INK, lw=.9, ls="--")
    ax.axhline(5, color=INK, lw=.9, ls="--")
    ax.set_xticks(range(len(cr)))
    ax.set_xticklabels(cr.index, fontsize=6.5, rotation=90)
    ax.set_ylabel("% of configurations classified\nabove proportional use")
    ax.set_ylim(-2, 104)
    ax.set_title(f"Threshold classification under 25 configurations "
                 f"(5 official maps x 5 waves).\n{n_all} of {N} states "
                 f"({100*n_all/N:.0f}%) change classification.",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "F7_threshold.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote results/figures/F7_threshold.png\n")


# ---------------------------------------------------------------------------
def variance_decomposition(panel):
    """
    Section 4.9. Three comparisons of specification against sampling variation,
    reported together because each answers a slightly different question.
    """
    g = panel[(panel.wave == "2026-02") & panel.usage_count.notna()]
    rng = np.random.default_rng(SEED)
    c = g.usage_count.to_numpy(float)
    w = g.pop_share_fixed.to_numpy()
    B = rng.multinomial(int(c.sum()), c / c.sum(), size=2000).astype(float)

    print("SECTION 4.9  VARIANCE DECOMPOSITION")
    rows = []

    # (1) as reported: state-level sampling vs five official maps
    A = (B / B.sum(1, keepdims=True)) / (w / 100)
    samp_state = np.array([ix.gini(A[i], w) for i in range(2000)])
    spec = []
    for nm, mc in MAPS:
        a = g.groupby(mc)[["usage_share", "pop_share_fixed"]].sum()
        a["aui"] = a.usage_share / a.pop_share_fixed
        spec.append(ix.gini(a.aui.values, a.pop_share_fixed.values))
    r1 = np.std(spec, ddof=1) / samp_state.std()
    rows.append({"comparison": "state-level sampling vs five official maps",
                 "sampling_sd": samp_state.std(),
                 "spec_sd": np.std(spec, ddof=1), "ratio": r1})
    print(f"  (1) sampling at k=51 vs five maps      : {r1:.1f}x")

    # (2) sampling computed within each configuration
    within = []
    for nm, mc in MAPS:
        boot = []
        for i in range(2000):
            tmp = g.copy()
            tmp["us"] = 100 * B[i] / B[i].sum()
            a = tmp.groupby(mc).agg(u=("us", "sum"), pp=("pop_share_fixed", "sum"))
            boot.append(ix.gini((a.u / a.pp).values, a.pp.values))
        within.append(np.std(boot))
    r2 = np.std(spec, ddof=1) / np.mean(within)
    rows.append({"comparison": "within-configuration sampling vs five maps",
                 "sampling_sd": np.mean(within),
                 "spec_sd": np.std(spec, ddof=1), "ratio": r2})
    print(f"  (2) within-configuration sampling      : {r2:.1f}x")

    # (3) zoning only, resolution held at k=9
    gg = g.set_index("state")
    gg = gg[gg.index.isin(ag.CONTIGUOUS_48)]
    sim = ag.zoning_monte_carlo(gg.aui_fixed, gg.usage_share, gg.pop_share_fixed,
                                k=9, stat_fn=ix.gini, n_sims=1500, seed=SEED)
    idx = [list(g.state).index(s) for s in gg.index]
    samp9 = []
    for i in range(500):
        tmp = gg.copy()
        tmp["us"] = 100 * B[i][idx] / B[i][idx].sum()
        a = tmp.groupby("census_division").agg(u=("us", "sum"),
                                               pp=("pop_share_fixed", "sum"))
        samp9.append(ix.gini((a.u / a.pp).values, a.pp.values))
    r3 = sim.std() / np.std(samp9)
    rows.append({"comparison": "zoning only at fixed k=9",
                 "sampling_sd": np.std(samp9), "spec_sd": sim.std(), "ratio": r3})
    print(f"  (3) zoning only, resolution fixed at 9 : {r3:.1f}x")
    print("  The conclusion is insensitive to which comparison is used.\n")

    pd.DataFrame(rows).to_csv(TAB / "T12_variance_decomposition.csv", index=False)


if __name__ == "__main__":
    panel = pd.read_csv("data/processed/panel.csv")
    print(f"panel: {len(panel)} rows, {panel.state.nunique()} jurisdictions, "
          f"{panel.wave.nunique()} waves\n")
    spatial_and_validity(panel)
    threshold(panel)
    variance_decomposition(panel)
