"""
aei_audit.shiftshare
====================
Why is a state's AI Usage Index high?

Two candidate answers with opposite policy implications:

  COMPOSITION  The state employs more people in occupations that use Claude
               heavily. DC ranks high because DC is full of analysts and
               lawyers. Nothing unusual is happening per worker.

  INTENSITY    Within the same occupations, this state's workers use Claude
               more than their counterparts elsewhere. DC's analysts use it
               harder than Ohio's analysts.

The first says adoption follows the industrial map. The second says something
local is driving it, and is the one that would justify a place-based
intervention.

DECOMPOSITION
-------------
For state s and occupation o:

    e_so  employment share of occupation o within state s        (BLS OES)
    r_so  usage per worker in occupation o, state s              (AEI x BLS)
    r_o   usage per worker in occupation o, nationally           (AEI x BLS)
    e_o   national employment share of occupation o              (BLS OES)

    actual          R_s = sum_o  e_so * r_so
    counterfactual  C_s = sum_o  e_so * r_o      <- local mix, national rates
    national        R_n = sum_o  e_o  * r_o

    composition = C_s - R_n      (mix differs from the nation)
    intensity   = R_s - C_s      (rates differ from the nation)
    total       = R_s - R_n      = composition + intensity   [exact]

The identity is exact by construction, and test_identity() enforces it. If it
does not hold to floating point, the inputs are misaligned.

INPUTS YOU MUST SUPPLY
----------------------
AEI gives usage; it does not give employment. You need BLS OES state-level
employment by SOC major group:

    https://www.bls.gov/oes/tables.htm  ->  "All data" state file (oesmYYst)
    Filter to O_GROUP == 'major', keep AREA_TITLE, OCC_TITLE, TOT_EMP.

The 22 AEI soc_occupation categories are the SOC major groups verbatim, so the
join is on title with no crosswalk beyond dropping 'not_classified'.

CAVEATS TO STATE IN THE PAPER
-----------------------------
1. AEI reports usage BY the occupation inferred from conversation content, not
   by the user's actual job. A student writing code is counted under Computer
   and Mathematical. This is a real limitation and it biases toward
   composition looking larger than it is.
2. 'not_classified' usage is dropped, and its share varies by state.
3. Only 47 states carry the soc layer in the 2025-08 release. Report which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["load_aei_occupation_shares", "load_bls_oes", "check_join",
           "decompose", "test_identity", "summarise"]

DROP = {"not_classified"}


def load_aei_occupation_shares(path: str, geo_level: str = "state_us") -> pd.DataFrame:
    """
    Return state x occupation usage shares plus each state's total usage count.

    Columns: state, occupation, soc_pct, usage_count, usage_in_occ
    """
    d = pd.read_csv(path, usecols=["geo_id", "geography", "facet", "variable",
                                   "value", "cluster_name"],
                    dtype={"geo_id": "category", "geography": "category",
                           "facet": "category", "variable": "category",
                           "cluster_name": "category"}, low_memory=False)

    occ = d[(d.geography == geo_level) & (d.facet == "soc_occupation")
            & (d.variable == "soc_pct")].copy()
    occ["state"] = occ.geo_id.astype(str)
    occ["occupation"] = occ.cluster_name.astype(str)
    occ = occ[~occ.occupation.isin(DROP) & occ.state.str.match(r"^[A-Z]{2}$")]

    tot = d[(d.geography == geo_level) & (d.variable == "usage_count")].copy()
    tot["state"] = tot.geo_id.astype(str)
    tot = tot[tot.state.str.match(r"^[A-Z]{2}$")].set_index("state").value

    out = occ[["state", "occupation", "value"]].rename(columns={"value": "soc_pct"})
    out["usage_count"] = out.state.map(tot)

    # DO NOT RENORMALISE. 'not_classified' is not a rounding remainder: it is
    # 72.6% of usage in the median state (3.5% classified in Vermont). Rescaling
    # the classified share to 100% reassigns unclassified conversations to
    # whichever occupations happen to clear the disclosure threshold, which for
    # sparse states means assigning a state's entire volume to one occupation.
    # usage_in_occ therefore counts CLASSIFIED conversations only, and the
    # decomposition is a statement about classified usage per worker.
    out["usage_in_occ"] = out.usage_count * out.soc_pct / 100

    cov = out.groupby("state").agg(n_occ=("occupation", "nunique"),
                                   classified_pct=("soc_pct", "sum"))
    out = out.merge(cov, on="state")
    return out.dropna().reset_index(drop=True)


def coverage_report(aei: pd.DataFrame) -> pd.DataFrame:
    """
    Occupational coverage by state. RUN THIS BEFORE DECOMPOSING.

    A shift-share needs the occupational distribution of a state's usage. Where
    most usage is unclassified, that distribution is unobserved and the
    decomposition is not identified no matter how clean the arithmetic looks.
    """
    return (aei.groupby("state")
               .agg(n_occ=("n_occ", "first"), classified_pct=("classified_pct", "first"),
                    usage_count=("usage_count", "first"))
               .sort_values("classified_pct", ascending=False))


def decompose(aei: pd.DataFrame, emp: pd.DataFrame) -> pd.DataFrame:
    """
    aei : state, occupation, usage_in_occ    (from load_aei_occupation_shares)
    emp : state, occupation, employment      (BLS OES, SOC major groups)

    Returns one row per state with the exact composition/intensity split.
    """
    df = aei.merge(emp, on=["state", "occupation"], how="inner")
    if df.empty:
        raise ValueError("empty join: check that occupation titles match exactly")

    # national usage rate per worker, by occupation
    nat = (df.groupby("occupation")
             .agg(u=("usage_in_occ", "sum"), e=("employment", "sum")))
    nat["r_o"] = nat.u / nat.e
    E = nat.e.sum()
    nat["e_o"] = nat.e / E
    R_n = float((nat.e_o * nat.r_o).sum())

    df = df.join(nat[["r_o"]], on="occupation")
    df["e_so"] = df.employment / df.groupby("state").employment.transform("sum")
    df["r_so"] = df.usage_in_occ / df.employment

    g = df.groupby("state")
    out = pd.DataFrame({
        "R_s": g.apply(lambda x: (x.e_so * x.r_so).sum(), include_groups=False),
        "C_s": g.apply(lambda x: (x.e_so * x.r_o).sum(), include_groups=False),
        "n_occ": g.size(),
    })
    out["R_n"] = R_n
    out["composition"] = out.C_s - R_n
    out["intensity"] = out.R_s - out.C_s
    out["total"] = out.R_s - R_n
    out["residual"] = out.total - (out.composition + out.intensity)
    out["index"] = out.R_s / R_n
    out["pct_composition"] = 100 * out.composition / out.total.abs().where(out.total.abs() > 0)
    return out.sort_values("index", ascending=False)


def test_identity(res: pd.DataFrame, tol: float = 1e-12) -> None:
    """composition + intensity must equal total, exactly."""
    bad = res.residual.abs().max()
    assert bad < tol, f"decomposition identity violated: max residual {bad:.2e}"


def summarise(res: pd.DataFrame, top: int = 8) -> str:
    t = res.head(top)
    lines = [f"national rate R_n = {res.R_n.iloc[0]:.6f} conversations per worker",
             f"identity residual (max) = {res.residual.abs().max():.2e}", "",
             f"{'state':>6} {'index':>7} {'composition':>12} {'intensity':>10} {'% comp':>8}"]
    for s, r in t.iterrows():
        lines.append(f"{s:>6} {r['index']:>7.2f} {r['composition']:>12.5f} "
                     f"{r['intensity']:>10.5f} {r['pct_composition']:>7.0f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BLS OES loader
# ---------------------------------------------------------------------------
_STATE_ABBR = {
 'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
 'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC',
 'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL',
 'Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA',
 'Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
 'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
 'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
 'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
 'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
 'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN',
 'Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
 'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}


def load_bls_oes(path: str, sheet: int | str = 0) -> pd.DataFrame:
    """
    Parse the raw BLS OEWS state file into the shape decompose() expects.

    GET THE FILE
        https://www.bls.gov/oes/tables.htm
        -> "All data" -> state file, e.g. oesm24st.zip
        -> unzip, it contains state_M2024_dl.xlsx

    Then:
        emp = load_bls_oes("state_M2024_dl.xlsx")

    Handles the format quirks that bite people:
      * TOT_EMP carries thousands separators and '**' for suppressed cells
      * O_GROUP must be filtered to 'major' or you double-count every worker
        (majors, minors, broads and detailed occupations all sit in one file)
      * BLS major-group titles end in ' Occupations'; the AEI soc_occupation
        labels do not, so the suffix is stripped before joining
      * territories (PR, GU, VI) and 'U.S.' rollups are dropped

    Returns: state, occupation, employment
    """
    if str(path).lower().endswith((".xlsx", ".xls")):
        d = pd.read_excel(path, sheet_name=sheet)
    else:
        d = pd.read_csv(path, low_memory=False, encoding="latin-1")
    d.columns = [c.strip().upper() for c in d.columns]

    need = {"AREA_TITLE", "OCC_TITLE", "TOT_EMP", "O_GROUP"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(
            f"missing columns {sorted(missing)}; found {list(d.columns)[:15]}. "
            f"Check you opened the STATE file (state_MYYYY_dl.xlsx), not the "
            f"national or metro one."
        )

    d = d[d.O_GROUP.astype(str).str.lower().str.strip() == "major"].copy()
    if d.empty:
        raise ValueError("no rows with O_GROUP=='major'; inspect d.O_GROUP.unique()")

    d["state"] = d.AREA_TITLE.astype(str).str.strip().map(_STATE_ABBR)
    d = d[d.state.notna()]

    d["employment"] = pd.to_numeric(
        d.TOT_EMP.astype(str).str.replace(",", "", regex=False)
                 .replace({"**": np.nan, "*": np.nan, "#": np.nan}),
        errors="coerce")
    d = d[d.employment.notna() & (d.employment > 0)]

    d["occupation"] = (d.OCC_TITLE.astype(str).str.strip()
                        .str.replace(r"\s+Occupations$", "", regex=True).str.strip())

    out = (d.groupby(["state", "occupation"], as_index=False).employment.sum())
    return out


def check_join(aei: pd.DataFrame, emp: pd.DataFrame) -> dict:
    """
    Report join coverage BEFORE decomposing. Run this every time.

    An unmatched occupation title silently drops that occupation from one side
    of the decomposition, which biases the composition term without raising
    anything.
    """
    a_occ, e_occ = set(aei.occupation), set(emp.occupation)
    a_st, e_st = set(aei.state), set(emp.state)
    return {
        "occupations_matched": len(a_occ & e_occ),
        "in_aei_not_bls": sorted(a_occ - e_occ),
        "in_bls_not_aei": sorted(e_occ - a_occ),
        "states_matched": len(a_st & e_st),
        "states_dropped": sorted(a_st - e_st),
        "verdict": ("OK" if not (a_occ - e_occ)
                    else "FIX TITLES BEFORE PROCEEDING: unmatched AEI occupations "
                         "will be silently dropped and the composition term will be wrong"),
    }
