"""
aei_audit.did
=============
Difference-in-differences for the US state panel, built on numpy alone so it
runs in a bare Colab with no compilation and no version drift.

DESIGN
------
  units     51 US states + DC
  waves     4  (Aug-2025, Nov-2025 | Feb-2026, May-2026)
  treatment state AI statutes effective 2026-01-01
  pre/post  waves 1-2 pre, waves 3-4 post

WHY NOT CLUSTER-ROBUST SEs
--------------------------
With roughly three treated clusters, cluster-robust standard errors over-reject
badly and the wild cluster bootstrap is also unreliable when the number of
TREATED clusters is small (MacKinnon & Webb 2017, 2018). The correct tool is
randomization inference: permute treatment across states, rebuild the null
empirically, read the p-value off it. No asymptotics in the number of treated
units are required.

WHY NOT SYNTHETIC CONTROL
-------------------------
Two pre-periods. A synthetic control fits two points trivially and carries no
information. Say this in the write-up rather than attempting it.

THE PAYLOAD
-----------
did_across_scales() re-runs the identical specification at state (k=51),
division (k=9) and region (k=4) level. If the estimate moves materially, the
policy conclusion is scale-dependent, and that is the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["twfe", "event_study", "randomization_inference",
           "minimum_detectable_effect", "did_across_scales"]


# ---------------------------------------------------------------------------
def _design(df: pd.DataFrame, y: str, treat_col: str):
    """Two-way fixed effects via explicit dummies. 204 x ~56, trivially small."""
    d = df.dropna(subset=[y]).copy()
    u = pd.get_dummies(d["unit"], prefix="u", drop_first=True, dtype=float)
    w = pd.get_dummies(d["wave"], prefix="w", drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d[treat_col].to_numpy(float),
                         u.to_numpy(), w.to_numpy()])
    return X, d[y].to_numpy(float)


def twfe(df: pd.DataFrame, y: str, treat_col: str = "D") -> float:
    """Point estimate on the treated-post interaction."""
    X, yy = _design(df, y, treat_col)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return float(beta[1])


def event_study(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """
    Coefficients by event time, reference = -1 (last pre-treatment wave).

    With two pre-waves you get exactly ONE testable pre-coefficient (t = -2).
    That is weak evidence on parallel trends. Present it as weak; do not call
    a single insignificant lead a validated assumption.
    """
    d = df.dropna(subset=[y]).copy()
    ets = sorted(e for e in d["event_time"].unique() if e != -1)
    for e in ets:
        d[f"et_{e}"] = ((d["event_time"] == e) & (d["treated"] == 1)).astype(float)

    cols = [f"et_{e}" for e in ets]
    u = pd.get_dummies(d["unit"], prefix="u", drop_first=True, dtype=float)
    w = pd.get_dummies(d["wave"], prefix="w", drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d[cols].to_numpy(float),
                         u.to_numpy(), w.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)

    return pd.DataFrame({"event_time": ets,
                         "coef": beta[1:1 + len(ets)]}).sort_values("event_time")


# ---------------------------------------------------------------------------
def randomization_inference(df: pd.DataFrame, y: str, n_perm: int = 10_000,
                            seed: int = 20260824, eligible=None) -> dict:
    """
    Fisher exact test. Reassign treatment to a random set of the same size,
    recompute the DiD, repeat. p = share of |placebo| >= |observed|.

    eligible : optional list of units allowed to receive placebo treatment.
        Restricting to units in the same pre-period size tercile as the real
        treated states is a sharper test. Report both, unrestricted first.
    """
    rng = np.random.default_rng(seed)
    observed = twfe(df, y)

    units = sorted(df["unit"].unique())
    treated_units = sorted(df.loc[df["treated"] == 1, "unit"].unique())
    k = len(treated_units)
    pool = list(eligible) if eligible is not None else units

    post = df.set_index(["unit", "wave"])["post"]
    placebos = np.empty(n_perm)
    for i in range(n_perm):
        fake = set(rng.choice(pool, size=k, replace=False))
        tmp = df.copy()
        tmp["treated"] = tmp["unit"].isin(fake).astype(int)
        tmp["D"] = tmp["treated"] * tmp["post"]
        placebos[i] = twfe(tmp, y)

    p = float(np.mean(np.abs(placebos) >= abs(observed)))
    return {
        "outcome": y, "observed": observed, "p_ri": p, "n_treated": k,
        "null_sd": float(placebos.std()),
        "null_ci": (float(np.percentile(placebos, 2.5)),
                    float(np.percentile(placebos, 97.5))),
        "placebos": placebos,
    }


def minimum_detectable_effect(df: pd.DataFrame, y: str,
                              taus=None, n_sim: int = 300,
                              n_perm: int = 400, alpha: float = 0.05,
                              seed: int = 20260824) -> dict:
    """
    Simulation-based MDE consistent with the RI test.

    Inject a known effect tau into the post-period of a random treated set,
    run the RI test, record rejection. MDE = smallest tau rejected >= 80% of
    the time.

    THIS IS THE HEADLINE WHEN THE RESULT IS NULL. "No significant effect" is
    not a finding. "We could not have detected anything smaller than an X%
    change" is.
    """
    rng = np.random.default_rng(seed)
    taus = np.arange(0.02, 0.61, 0.03) if taus is None else np.asarray(taus)
    units = sorted(df["unit"].unique())
    k = int(df.loc[df["treated"] == 1, "unit"].nunique())

    power = []
    for tau in taus:
        rej = 0
        for _ in range(n_sim):
            fake = set(rng.choice(units, size=k, replace=False))
            tmp = df.copy()
            tmp["treated"] = tmp["unit"].isin(fake).astype(int)
            tmp["D"] = tmp["treated"] * tmp["post"]
            tmp.loc[tmp["D"] == 1, y] = tmp.loc[tmp["D"] == 1, y] + tau
            r = randomization_inference(tmp, y, n_perm=n_perm,
                                        seed=int(rng.integers(1e9)))
            rej += int(r["p_ri"] < alpha)
        power.append(rej / n_sim)

    power = np.asarray(power)
    hit = np.where(power >= 0.80)[0]
    mde = float(taus[hit[0]]) if len(hit) else float("nan")
    return {"taus": taus, "power": power, "mde_80_log": mde,
            "mde_80_pct": float(100 * (np.exp(mde) - 1)) if mde == mde else np.nan}


# ---------------------------------------------------------------------------
def did_across_scales(panel: pd.DataFrame, mappings: dict[str, dict],
                      count_col: str = "conversations",
                      denom_col: str = "denominator",
                      n_perm: int = 2000) -> pd.DataFrame:
    """
    THE PAYLOAD. The identical specification at every geographic scale.

    A state-level statute aggregated to divisions produces a partially treated
    division. Two codings are reported:
      'any'   -- a group is treated if ANY member state is treated
      'share' -- continuous exposure = treated share of the group's denominator

    If the estimate changes sign or magnitude across scales, the policy
    conclusion is a property of the map. That is the result.
    """
    from .aggregation import aggregate_panel

    treated_states = set(panel.loc[panel["treated"] == 1, "unit"])
    rows = []

    for label, mp in mappings.items():
        if mp is None:
            g = panel.copy()
        else:
            src = panel.rename(columns={"unit": "state"})
            g = aggregate_panel(src, mp, count_col=count_col, denom_col=denom_col)
            # exposure of each group
            src["_g"] = src["state"].map(mp)
            expo = (src.assign(t=src["state"].isin(treated_states).astype(float))
                       .groupby("_g")
                       .apply(lambda d: np.average(d["t"], weights=d[denom_col]),
                              include_groups=False))
            g["exposure"] = g["unit"].map(expo)
            g["treated"] = (g["exposure"] > 0).astype(int)
            g = g.merge(panel[["wave", "post", "event_time"]].drop_duplicates(),
                        on="wave", how="left")

        g["y"] = np.log(g[count_col] / g[denom_col])
        for coding in ("any", "share"):
            gg = g.copy()
            gg["D"] = (gg["treated"] * gg["post"] if coding == "any"
                       else gg.get("exposure", gg["treated"]) * gg["post"])
            est = twfe(gg, "y")
            ri = (randomization_inference(gg, "y", n_perm=n_perm)
                  if gg["treated"].sum() and coding == "any" else None)
            rows.append({"scale": label, "n_units": gg["unit"].nunique(),
                         "coding": coding, "estimate": est,
                         "pct": 100 * (np.exp(est) - 1),
                         "p_ri": ri["p_ri"] if ri else np.nan})
    return pd.DataFrame(rows)
