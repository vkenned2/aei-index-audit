"""
aei_audit.speccurve
===================
Specification curve analysis for the DiD payload.

Simonsohn, Simmons & Nelson (2020), Nature Human Behaviour 4:1208-1214.

WHY THIS AND NOT A SINGLE DID
-----------------------------
The claim of Analysis 6 is not "state AI statutes had effect X". It is that any
such estimate is not identified independently of the analyst's choice of
geography and denominator. Demonstrating that requires estimating the same
target under every defensible specification and showing the estimate moves.

THE OBJECTION THIS MODULE EXISTS TO ANSWER
------------------------------------------
"Your coefficients span -0.05 to +0.12, but each has a standard error of 0.15,
so they are all the same number and you have shown nothing."

That objection is correct unless you compare the two sources of variation
directly. decompose_uncertainty() does that:

    sampling SD      : median standard error of a single specification
    specification SD : standard deviation of the point estimates across specs
    ratio            : specification SD / sampling SD

A ratio near or above 1 means choosing the map matters as much as collecting
the data. That is the result. A ratio well below 1 means the specifications
agree and the index is robust on this dimension, which is also a result and
must be reported as one.

SPECIFICATIONS MUST BE FIXED BEFORE ESTIMATION
----------------------------------------------
Enumerate the grid in docs/PRE_ANALYSIS.md before running anything. A curve
assembled after seeing results is a garden of forking paths with a plot
attached. Every specification in the grid must be one you would have defended
in advance, and none may be dropped afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

__all__ = ["Spec", "build_grid", "run_curve", "decompose_uncertainty",
           "plot_curve"]


@dataclass(frozen=True)
class Spec:
    scale: str          # 'state' | 'census_division' | 'census_region' | ...
    denominator: str    # 'working_age_pop' | 'knowledge_occ' | 'labor_force'
    weighted: bool      # denominator-weighted estimation
    include_dc: bool

    @property
    def label(self) -> str:
        return (f"{self.scale}|{self.denominator}"
                f"|{'wt' if self.weighted else 'unwt'}"
                f"|{'+DC' if self.include_dc else '-DC'}")


def build_grid(scales, denominators, weighted=(True, False),
               include_dc=(True, False)) -> list[Spec]:
    """
    Full factorial. Fix this in the pre-analysis note BEFORE estimating.

    3 scales x 3 denominators x 2 x 2 = 36 specifications.
    """
    return [Spec(*c) for c in product(scales, denominators, weighted, include_dc)]


def run_curve(panel: pd.DataFrame, grid: list[Spec], estimate_fn) -> pd.DataFrame:
    """
    estimate_fn(panel, spec) -> (coef, se, n_units)

    Supply your own estimator so this module stays independent of pyfixest.
    A minimal TWFE version:

        def estimate_fn(panel, spec):
            df = prepare(panel, spec)          # aggregate, swap denominator, filter DC
            w  = df.denominator if spec.weighted else None
            m  = pf.feols("log_rate ~ treated_post | unit + wave",
                          data=df, weights=w, vcov={"CRV1": "unit"})
            return (float(m.coef()["treated_post"]),
                    float(m.se()["treated_post"]),
                    df.unit.nunique())

    EVERY specification in the grid must be reported. Dropping one that
    failed to converge is a result about the data and belongs in the table
    with a note, not in the bin.
    """
    rows = []
    for spec in grid:
        try:
            coef, se, n = estimate_fn(panel, spec)
            ok, note = True, ""
        except Exception as e:                      # noqa: BLE001
            coef = se = np.nan
            n, ok, note = 0, False, f"{type(e).__name__}: {e}"
        rows.append({**spec.__dict__, "label": spec.label,
                     "coef": coef, "se": se, "n_units": n,
                     "converged": ok, "note": note})

    out = pd.DataFrame(rows).sort_values("coef").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["ci_lo"] = out["coef"] - 1.96 * out["se"]
    out["ci_hi"] = out["coef"] + 1.96 * out["se"]
    out["sig"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    return out


def decompose_uncertainty(curve: pd.DataFrame) -> dict:
    """
    Split total uncertainty about the policy effect into its two sources.

    THIS IS THE HEADLINE NUMBER OF ANALYSIS 6.
    """
    c = curve.loc[curve["converged"]]
    if c.empty:
        raise ValueError("no specification converged")

    samp = float(np.median(c["se"]))
    spec = float(np.std(c["coef"], ddof=1))
    lo, hi = float(c["coef"].min()), float(c["coef"].max())

    sign_stable = bool((c["coef"] > 0).all() or (c["coef"] < 0).all())
    ratio = spec / samp if samp > 0 else np.nan

    if not np.isfinite(ratio):
        verdict = "cannot compute"
    elif ratio >= 1.0:
        verdict = ("Specification uncertainty EXCEEDS sampling uncertainty. "
                   "The analyst's choice of geography and denominator matters "
                   "at least as much as the data collection. Any single "
                   "published estimate understates its own uncertainty.")
    elif ratio >= 0.5:
        verdict = ("Specification uncertainty is a substantial fraction of "
                   "sampling uncertainty and should be reported alongside it.")
    else:
        verdict = ("Specifications broadly agree. On this dimension the "
                   "estimate is robust, and the paper reports that.")

    return {
        "n_specs": int(len(c)),
        "n_failed": int((~curve["converged"]).sum()),
        "sampling_sd": samp,
        "specification_sd": spec,
        "ratio_spec_to_sampling": ratio,
        "coef_min": lo, "coef_max": hi, "coef_range": hi - lo,
        "range_in_sampling_sds": (hi - lo) / samp if samp > 0 else np.nan,
        "sign_stable": sign_stable,
        "pct_significant": float(c["sig"].mean()),
        "verdict": verdict,
    }


def plot_curve(curve: pd.DataFrame, path: str, ylabel: str = "DiD coefficient"):
    """
    Standard two-panel specification curve: estimates on top, the specification
    grid below, aligned by rank.
    """
    import matplotlib.pyplot as plt

    c = curve.loc[curve["converged"]].reset_index(drop=True)
    dims = ["scale", "denominator", "weighted", "include_dc"]
    levels = [(d, v) for d in dims for v in sorted(c[d].astype(str).unique())]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4], "hspace": 0.05})

    ax.vlines(c["rank"], c["ci_lo"], c["ci_hi"], color="#c9c9c9", lw=1)
    ax.scatter(c["rank"], c["coef"], s=16,
               c=np.where(c["sig"], "#b03a2e", "#4a4a4a"), zorder=3)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)

    for y, (d, v) in enumerate(levels):
        on = c[d].astype(str) == v
        ax2.scatter(c["rank"][on], np.full(on.sum(), y), s=6, c="#2c2c2c")
    ax2.set_yticks(range(len(levels)))
    ax2.set_yticklabels([f"{d}: {v}" for d, v in levels], fontsize=7)
    ax2.set_xlabel("specification, ordered by estimate")
    ax2.invert_yaxis()
    ax2.spines[["top", "right"]].set_visible(False)

    fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
