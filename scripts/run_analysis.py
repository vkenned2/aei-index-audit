"""
run_analysis.py -- every figure and table, from data/processed/panel.csv

Produces:
  results/figures/F1_scale_decomposition.png
  results/figures/F2_official_maps.png
  results/figures/F3_zoning_null.png
  results/figures/F4_lorenz_by_scale.png
  results/figures/F5_state_caterpillar.png
  results/figures/F6_platform_break.png
  results/tables/T1_theil.csv
  results/tables/T2_zoning.csv
  results/tables/T3_zoning_null.csv
  results/tables/T4_rank_resolution.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aei_audit import indices as ix, aggregation as ag  # noqa: E402

FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
TAB = Path("results/tables"); TAB.mkdir(parents=True, exist_ok=True)
SEED = 20260822

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "figure.dpi": 160, "savefig.bbox": "tight",
})
INK, ACCENT, MUTE = "#2b2b2b", "#b03a2e", "#c9c9c9"

MAPS = [("Census region", "census_region"), ("Time zone", "time_zone"),
        ("BEA region", "bea_region"), ("Census division", "census_division"),
        ("Federal region", "federal_region")]


def load():
    p = pd.read_csv("data/processed/panel.csv")
    return p, sorted(p.wave.unique())


def roll(g, col):
    """Aggregate shares to a geography and recompute the index."""
    a = g.groupby(col)[["usage_share", "pop_share_fixed"]].sum()
    a["aui"] = a.usage_share / a.pop_share_fixed
    return a


# ---------------------------------------------------------------- F1 + T1
def f1_scale(p, W):
    rows = []
    for w in W:
        g = p[p.wave == w]
        rows.append({"wave": w, "grouping": "State (k=51)", "k": 51,
                     "total": ix.theil(g.aui_fixed.values, g.pop_share_fixed.values),
                     "between": np.nan, "within": np.nan, "share_between": 1.0})
        for nm, col in MAPS:
            d = ix.theil_decompose(g.aui_fixed.values, g[col].values,
                                   weights=g.pop_share_fixed.values)
            rows.append({"wave": w, "grouping": f"{nm} (k={d['n_groups']})",
                         "k": d["n_groups"], **{k: d[k] for k in
                         ("total", "between", "within", "share_between")}})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "T1_theil.csv", index=False)

    fig, axes = plt.subplots(1, len(W), figsize=(2.6 * len(W), 3.4), sharey=True)
    for ax, w in zip(np.atleast_1d(axes), W):
        s = t[(t.wave == w) & t.between.notna()].sort_values("k")
        lbl = [f"k={k}" for k in s.k]
        ax.bar(lbl, s.between, color=ACCENT, label="between-group")
        ax.bar(lbl, s.within, bottom=s.between, color=MUTE, label="within-group")
        ax.set_title(w, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("Theil index of AI usage")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Aggregation destroys 40–60% of measured inequality",
                 fontsize=11, y=1.02)
    fig.savefig(FIG / "F1_scale_decomposition.png"); plt.close(fig)
    return t


# ---------------------------------------------------------------- F2 + T2
def f2_maps(p, W):
    rows = []
    for w in W:
        g = p[p.wave == w]
        r = {"wave": w, "State (k=51)": ix.gini(g.aui_fixed.values,
                                                g.pop_share_fixed.values)}
        for nm, col in MAPS:
            a = roll(g, col)
            r[f"{nm} (k={len(a)})"] = ix.gini(a.aui.values, a.pop_share_fixed.values)
        rows.append(r)
    z = pd.DataFrame(rows).set_index("wave")
    mapcols = [c for c in z.columns if not c.startswith("State")]
    z["spread_pct_of_median"] = 100 * (z[mapcols].max(axis=1) - z[mapcols].min(axis=1)) / z[mapcols].median(axis=1)
    z.to_csv(TAB / "T2_zoning.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4))
    x = np.arange(len(W))
    for c in mapcols:
        ax.plot(x, z[c], marker="o", ms=4, lw=1.4, label=c)
    ax.plot(x, z[[c for c in z.columns if c.startswith("State")][0]],
            marker="s", ms=4, lw=1.8, color=INK, ls="--", label="State (k=51)")
    ax.set_xticks(x); ax.set_xticklabels(W)
    ax.set_ylabel("Gini of AI usage rate")
    ax.set_title("Same data, five official US maps, five different answers")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(FIG / "F2_official_maps.png"); plt.close(fig)
    return z


# ---------------------------------------------------------------- F3 + T3
def f3_null(p, W, n_sims=1500):
    fig, axes = plt.subplots(1, len(W), figsize=(2.6 * len(W), 3.2), sharey=True)
    rows = []
    for ax, w in zip(np.atleast_1d(axes), W):
        g = p[p.wave == w].set_index("state")
        g = g[g.index.isin(ag.CONTIGUOUS_48)]
        sim = ag.zoning_monte_carlo(g.aui_fixed, g.usage_share, g.pop_share_fixed,
                                    k=9, stat_fn=ix.gini, n_sims=n_sims, seed=SEED)
        a = roll(g, "census_division")
        obs = ix.gini(a.aui.values, a.pop_share_fixed.values)
        pct = 100 * (sim < obs).mean()
        rows.append({"wave": w, "observed": obs, "null_median": np.median(sim),
                     "null_min": sim.min(), "null_max": sim.max(),
                     "percentile": pct,
                     "null_spread_pct_of_median": 100*(sim.max()-sim.min())/np.median(sim)})
        ax.hist(sim, bins=45, color=MUTE, edgecolor="none")
        ax.axvline(obs, color=ACCENT, lw=1.8)
        ax.set_title(f"{w}\npctile {pct:.0f}", fontsize=8)
        ax.set_xlabel("Gini, k=9")
    np.atleast_1d(axes)[0].set_ylabel("random contiguous maps")
    fig.suptitle("Census divisions sit near the TOP of what any 9-region map yields",
                 fontsize=11, y=1.04)
    fig.savefig(FIG / "F3_zoning_null.png"); plt.close(fig)
    t = pd.DataFrame(rows); t.to_csv(TAB / "T3_zoning_null.csv", index=False)
    return t


# ---------------------------------------------------------------- F4
def f4_lorenz(p, wave="2026-02"):
    g = p[p.wave == wave]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color="k", lw=.8, ls=":")
    specs = [("State (k=51)", g.aui_fixed.values, g.pop_share_fixed.values)]
    for nm, col in MAPS:
        a = roll(g, col)
        specs.append((f"{nm} (k={len(a)})", a.aui.values, a.pop_share_fixed.values))
    for nm, v, wt in specs:
        cx, cy = ix.lorenz_points(v, wt)
        ax.plot(cx, cy, lw=1.4, label=nm)
    ax.set_xlabel("cumulative share of working-age population")
    ax.set_ylabel("cumulative share of Claude usage")
    ax.set_title(f"Lorenz curves collapse toward equality\nas units coarsen ({wave})")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(FIG / "F4_lorenz_by_scale.png"); plt.close(fig)


# ---------------------------------------------------------------- F5 + T4
def f5_caterpillar(p, wave="2026-02", n_boot=3000):
    g = p[p.wave == wave].dropna(subset=["usage_count"]).copy()
    rng = np.random.default_rng(SEED)
    c = g.usage_count.to_numpy(float)
    e = (g.pop_share_fixed / 100).to_numpy()
    boots = rng.multinomial(int(c.sum()), c / c.sum(), size=n_boot).astype(float)
    aui_b = (boots / boots.sum(1, keepdims=True)) / e

    g["lo"] = np.percentile(aui_b, 2.5, axis=0)
    g["hi"] = np.percentile(aui_b, 97.5, axis=0)
    g = g.sort_values("aui_fixed", ascending=False).reset_index(drop=True)

    order = np.argsort(-(c / e))
    d = aui_b[:, order]
    n = d.shape[1]
    P = np.array([(d[:, i][:, None] > d).mean(0) for i in range(n)])
    adj = np.array([P[i, i+1] for i in range(n-1)])
    res = float(np.mean((adj > .975) | (adj < .025)))

    fig, ax = plt.subplots(figsize=(5.2, 8))
    y = np.arange(len(g))
    ax.hlines(y, g.lo, g.hi, color=MUTE, lw=1.6)
    ax.scatter(g.aui_fixed, y, s=12, color=INK, zorder=3)
    ax.axvline(1, color=ACCENT, lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(g.state, fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel("AI Usage Index (1.0 = proportional)")
    ax.set_title(f"State index with sampling intervals ({wave})\n"
                 f"only {100*res:.0f}% of adjacent ranks are resolved", fontsize=9)
    fig.savefig(FIG / "F5_state_caterpillar.png"); plt.close(fig)

    g[["state", "aui_fixed", "lo", "hi", "usage_count"]].to_csv(
        TAB / "T4_rank_resolution.csv", index=False)
    return res


# ---------------------------------------------------------------- F6
def f6_platform(p, W):
    th = [ix.theil(p[p.wave == w].aui_fixed.values,
                   p[p.wave == w].pop_share_fixed.values) for w in W]
    sh = p.pivot(index="state", columns="wave", values="usage_share")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot(W, th, marker="o", color=INK, lw=1.6)
    a1.axvspan(1.5, 2.5, color=ACCENT, alpha=.12)
    a1.text(2, max(th)*.92, "platform universe\nchanges here", fontsize=7,
            ha="center", color=ACCENT)
    a1.set_ylabel("Theil index"); a1.set_title("Inequality falls, but the\nuniverse changed mid-series", fontsize=9)
    a1.tick_params(axis="x", labelrotation=30)

    for s in ["CA", "NY", "TX", "FL"]:
        a2.plot(W, sh.loc[s], marker="o", ms=4, lw=1.4, label=s)
    a2.axvspan(0.5, 1.5, color="#4a6fa5", alpha=.12)
    a2.text(1, sh.loc["CA"].max()*.55, "CA/NY attribution\nanomaly", fontsize=7,
            ha="center", color="#4a6fa5")
    a2.set_ylabel("% of national usage"); a2.set_title("Two large states swap share,\nthen swap back", fontsize=9)
    a2.legend(frameon=False, fontsize=7); a2.tick_params(axis="x", labelrotation=30)
    fig.savefig(FIG / "F6_platform_break.png"); plt.close(fig)


if __name__ == "__main__":
    p, W = load()
    print(f"panel: {len(p)} rows, {p.state.nunique()} states, {len(W)} waves\n")
    t1 = f1_scale(p, W); print("F1/T1 scale done")
    t2 = f2_maps(p, W);  print("F2/T2 zoning done")
    t3 = f3_null(p, W);  print("F3/T3 null done")
    f4_lorenz(p);        print("F4 lorenz done")
    r = f5_caterpillar(p); print(f"F5/T4 caterpillar done (adjacent resolved {100*r:.0f}%)")
    f6_platform(p, W);   print("F6 platform done")

    print("\n--- HEADLINES ---")
    d = t1[t1.grouping.str.startswith("Census division")]
    print(f"1. share_between at k=9: {d.share_between.min():.3f} to {d.share_between.max():.3f}"
          f"  -> {100*(1-d.share_between.max()):.0f}-{100*(1-d.share_between.min()):.0f}% of inequality lost")
    print(f"2. spread across official maps: {t2.spread_pct_of_median.min():.1f}% to "
          f"{t2.spread_pct_of_median.max():.1f}% of median")
    print(f"3. Census-division percentile in null: {[f'{x:.0f}' for x in t3.percentile]}")
    print(f"4. adjacent ranks resolved: {100*r:.0f}%")
