"""
compile_panel.py
================
Harmonise every AEI release into ONE analysis-ready US state panel.

FOUR RELEASES, FOUR DIFFERENT SCHEMAS
-------------------------------------
| release  | geography level | US id  | counts | denominator      |
|----------|-----------------|--------|--------|------------------|
| 2025-08  | state_us        | 'CA'   | yes    | working_age_pop  |
| 2025-11  | country-state   | 'US-CA'| yes    | none             |
| 2026-02  | country-state   | 'US-CA'| yes    | none             |
| 2026-06  | subregion       | 'US-CA'| NO     | recoverable      |

The 2026-06 "Cadences" release also carries TWO monthly waves in one file, so
five waves come out of four files.

HARMONISATION DECISIONS (each is a judgement call; all are logged)
------------------------------------------------------------------
D1. DENOMINATOR. working_age_pop is published only in the Aug-2025 enriched
    release. It is held FIXED across all waves. This is deliberate, not a
    workaround: a fixed denominator isolates changes in usage from changes in
    population re-estimation, which is what you want for panel work. The
    consequence is that aui_fixed is NOT identical to a published AUI computed
    on a contemporaneous denominator, and both are carried so the difference
    is auditable.

D2. UNITS. Restricted to the 51 units present in every wave (50 states + DC).
    Territories (PR, GU, VI) appear only from 2026-02; 'not_classified' is
    dropped. Both exclusions are counted in the build log.

D3. PLATFORM. Aug-2025 and Nov-2025 are "Claude AI (Free and Pro)". From
    2026-02 the universe becomes "Claude AI (Free, Pro, and Max)". This is
    carried as a column, NOT silently pooled. It is confounded with any
    1 Jan 2026 treatment date and that fact must survive into the analysis.

D4. CADENCE. The first three waves are one-week windows; the 2026-06 waves are
    monthly. Carried as a column. A single week and a full month are not
    interchangeable and pooling them without a cadence control is an error.

D5. COUNTS. Absent for the two 2026-06 waves. Downstream code must branch:
    multinomial bootstrap where counts exist, rounding-interval Monte Carlo
    where they do not.

OUTPUT
------
    data/processed/panel.csv         tidy, one row per state x wave
    data/processed/DICTIONARY.md     column definitions
    data/processed/build_log.txt     every decision + every validation result
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aei_audit import aggregation as ag  # noqa: E402

UPLOADS = Path(os.environ.get("AEI_RAW_DIR", "data/raw"))
OUT = Path("data/processed")
LOG: list[str] = []


def log(m=""):
    print(m)
    LOG.append(str(m))


# ---------------------------------------------------------------------------
# per-release parsers
# ---------------------------------------------------------------------------
OLD_COLS = ["geo_id", "geography", "date_start", "date_end",
            "platform_and_product", "variable", "value"]


def parse_old(path: Path, geo_level: str, prefixed: bool) -> pd.DataFrame:
    """Aug-2025 / Nov-2025 / Feb-2026 long format."""
    d = pd.read_csv(path, usecols=OLD_COLS,
                    dtype={"geo_id": "category", "geography": "category",
                           "variable": "category",
                           "platform_and_product": "category"},
                    low_memory=False)
    s = d[d["geography"] == geo_level].copy()
    s["gid"] = s["geo_id"].astype(str)

    if prefixed:
        s = s[s["gid"].str.match(r"^US-[A-Z]{2}$")]
        s["state"] = s["gid"].str[3:]
    else:
        s = s[s["gid"].str.match(r"^[A-Z]{2}$")]      # drops 'not_classified'
        s["state"] = s["gid"]

    meta = s.iloc[0]
    w = s.pivot_table(index="state", columns="variable", values="value",
                      observed=True).reset_index()
    w["date_start"] = meta["date_start"]
    w["date_end"] = meta["date_end"]
    w["platform_raw"] = str(meta["platform_and_product"])
    return w


NEW_COLS = ["date_start", "date_end", "geo_id", "geo_level",
            "category_name", "metric_id", "value"]


def parse_cadences(path: Path) -> pd.DataFrame:
    """2026-06 Cadences format. Two monthly waves in one file, no counts."""
    d = pd.read_csv(path, usecols=NEW_COLS,
                    dtype={"geo_id": "category", "geo_level": "category",
                           "category_name": "category", "metric_id": "category"},
                    low_memory=False)
    m = (d["geo_id"].astype(str).str.match(r"^US-[A-Z]{2}$")
         & (d["category_name"] == "overall")
         & d["metric_id"].isin(["usage_pct", "usage_per_capita_index"]))
    s = d[m].copy()
    s["state"] = s["geo_id"].astype(str).str[3:]
    w = (s.pivot_table(index=["state", "date_start", "date_end"],
                       columns="metric_id", values="value", observed=True)
           .reset_index())
    w["platform_raw"] = "Claude AI (Cadences release; platform label not in file)"
    return w


# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    log("=" * 74)
    log("AEI US STATE PANEL -- BUILD LOG")
    log("=" * 74)

    aug = parse_old(UPLOADS / "aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv",
                    "state_us", prefixed=False)
    nov = parse_old(UPLOADS / "aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv",
                    "country-state", prefixed=True)
    feb = parse_old(UPLOADS / "aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv",
                    "country-state", prefixed=True)
    jun = parse_cadences(UPLOADS / "aei_claude_ai_2026-06-26.csv")

    log("\n-- parsed --")
    for nm, df in [("2025-08", aug), ("2025-11", nov), ("2026-02", feb),
                   ("2026-06 (2 waves)", jun)]:
        log(f"   {nm:20s} {len(df):4d} rows, {df['state'].nunique()} states")

    # ---- D1: fixed denominator ---------------------------------------------
    pop = (aug.set_index("state")["working_age_pop"].dropna()
           .rename("working_age_pop"))
    log(f"\nD1 fixed denominator: working_age_pop from 2025-08, "
        f"{len(pop)} states, total {pop.sum():,.0f}")

    frames = []
    for wave_id, df, cadence in [("2025-08", aug, "week"),
                                 ("2025-11", nov, "week"),
                                 ("2026-02", feb, "week")]:
        t = df[["state", "date_start", "date_end", "platform_raw",
                "usage_count", "usage_pct"]].copy()
        t["wave"] = wave_id
        t["cadence"] = cadence
        t["aui_published"] = (df["usage_per_capita_index"]
                              if "usage_per_capita_index" in df else np.nan)
        frames.append(t)

    for (ds, de), g in jun.groupby(["date_start", "date_end"], observed=True):
        t = g[["state", "date_start", "date_end", "platform_raw",
               "usage_pct"]].copy()
        t["wave"] = str(ds)[:7]
        t["cadence"] = "month"
        t["usage_count"] = np.nan
        t["aui_published"] = g["usage_per_capita_index"].values
        frames.append(t)

    panel = pd.concat(frames, ignore_index=True)

    # ---- D2: balanced units -------------------------------------------------
    n_waves = panel["wave"].nunique()
    cnt = panel.groupby("state")["wave"].nunique()
    keep = sorted(set(cnt[cnt == n_waves].index) & set(pop.index))
    dropped = sorted(set(panel["state"]) - set(keep))
    log(f"\nD2 balance: {n_waves} waves; keeping {len(keep)} units present in all")
    log(f"   dropped: {dropped}")
    panel = panel[panel["state"].isin(keep)].copy()

    # ---- D3/D4 flags --------------------------------------------------------
    panel["platform"] = np.where(
        panel["platform_raw"].str.contains("Max", na=False), "free_pro_max",
        np.where(panel["platform_raw"].str.contains("Free and Pro", na=False),
                 "free_pro", "unlabelled"))
    log("\nD3 platform by wave:")
    for w, g in panel.groupby("wave"):
        log(f"   {w}  {g['platform'].iloc[0]:14s} cadence={g['cadence'].iloc[0]}")
    log("   NOTE: universe changes between 2025-11 and 2026-02, which straddles")
    log("         2026-01-01. Any treatment dated then is CONFOUNDED.")

    # ---- derived quantities -------------------------------------------------
    panel = panel.merge(pop, left_on="state", right_index=True, how="left")
    panel["pop_share_fixed"] = 100 * panel["working_age_pop"] / pop.sum()

    tot = panel.groupby("wave")["usage_pct"].transform("sum")
    panel["usage_share"] = 100 * panel["usage_pct"] / tot          # renormalised
    panel["aui_fixed"] = panel["usage_share"] / panel["pop_share_fixed"]
    panel["rate"] = panel["usage_count"] / panel["working_age_pop"]
    panel["log_aui_fixed"] = np.log(panel["aui_fixed"])

    # implied denominator, recoverable wherever a published AUI exists
    panel["pop_share_implied"] = np.where(
        panel["aui_published"].notna() & (panel["aui_published"] > 0),
        panel["usage_pct"] / panel["aui_published"], np.nan)

    # ---- geography crosswalks ----------------------------------------------
    for col, mp in [("census_division", ag.CENSUS_DIVISION),
                    ("census_region", ag.CENSUS_REGION),
                    ("bea_region", ag.BEA_REGION),
                    ("federal_region", ag.FEDERAL_REGION),
                    ("time_zone", ag.TIME_ZONE)]:
        panel[col] = panel["state"].map(mp)

    order = ["state", "wave", "date_start", "date_end", "cadence", "platform",
             "usage_count", "usage_pct", "usage_share", "aui_published",
             "working_age_pop", "pop_share_fixed", "pop_share_implied",
             "aui_fixed", "log_aui_fixed", "rate",
             "census_division", "census_region", "bea_region",
             "federal_region", "time_zone", "platform_raw"]
    return panel[order].sort_values(["wave", "state"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def validate(panel: pd.DataFrame):
    log("\n" + "=" * 74)
    log("VALIDATION")
    log("=" * 74)
    hard = []

    def chk(name, ok, detail="", severity="hard"):
        log(f"[{'PASS' if ok else ('FAIL' if severity=='hard' else 'WARN')}] {name}"
            + (f"\n         {detail}" if detail else ""))
        if not ok and severity == "hard":
            hard.append(name)

    chk("no duplicate state-wave rows",
        not panel.duplicated(["state", "wave"]).any())
    chk("panel balanced",
        panel.groupby("wave")["state"].nunique().nunique() == 1,
        f"{panel.groupby('wave')['state'].nunique().to_dict()}")

    for w, g in panel.groupby("wave"):
        chk(f"usage_share sums to 100 in {w}",
            abs(g["usage_share"].sum() - 100) < 1e-6,
            f"{g['usage_share'].sum():.8f}")

    chk("aui_fixed weighted mean is 1",
        all(abs((g["aui_fixed"] * g["pop_share_fixed"] / 100).sum() - 1) < 1e-9
            for _, g in panel.groupby("wave")))

    # cross-method: two algebraically identical routes to the index
    g = panel[panel["wave"] == "2025-08"]
    a = g["usage_share"] / g["pop_share_fixed"]
    b = ((g["usage_count"] / g["working_age_pop"])
         / (g["usage_count"].sum() / g["working_age_pop"].sum()))
    chk("AUI reproduces two independent ways",
        np.allclose(a, b, rtol=1e-9), f"max |A-B| = {np.max(np.abs(a-b)):.2e}")

    # recomputed vs published, where a published figure exists
    m = panel["aui_published"].notna()
    dev = (panel.loc[m, "aui_fixed"] - panel.loc[m, "aui_published"]).abs()
    chk("recomputed AUI tracks published AUI", dev.max() < 0.75,
        f"max |diff| = {dev.max():.3f}, median = {dev.median():.3f}. "
        f"Non-zero is EXPECTED: fixed vs contemporaneous denominator (D1).",
        severity="soft")

    # is the fixed denominator defensible?
    m2 = panel["pop_share_implied"].notna()
    rel = ((panel.loc[m2, "pop_share_implied"] - panel.loc[m2, "pop_share_fixed"]).abs()
           / panel.loc[m2, "pop_share_fixed"])
    chk("fixed vs implied population share agree", rel.median() < 0.10,
        f"median rel. diff = {100*rel.median():.1f}%, max = {100*rel.max():.1f}%",
        severity="soft")

    geo_cols = ["census_division","census_region","bea_region",
                "federal_region","time_zone"]
    unmapped = {c: sorted(panel.loc[panel[c].isna(),"state"].unique())
                for c in geo_cols if panel[c].isna().any()}
    chk("geography crosswalks cover every unit", not unmapped,
        f"unmapped: {unmapped}. groupby drops NaN keys SILENTLY, so an "
        f"incomplete crosswalk invalidates the cross-map comparison." if unmapped else "")

    chk("aggregation conserves usage_share",
        all(abs(ag.__dict__ and
                panel[panel.wave == w].groupby("census_division")["usage_share"]
                .sum().sum() - 100) < 1e-6 for w in panel["wave"].unique()))

    # sampling precision where counts exist
    c = panel["usage_count"].dropna()
    log(f"\n[INFO] Poisson CV where counts exist: median "
        f"{100*(1/np.sqrt(c)).median():.1f}%, worst {100*(1/np.sqrt(c)).max():.1f}%")
    log(f"[INFO] waves without counts (rounding-interval model required): "
        f"{sorted(panel.loc[panel['usage_count'].isna(),'wave'].unique())}")

    if hard:
        raise AssertionError(f"HARD FAILURES: {hard}")
    log("\nAll hard checks passed.")


DICT = """# panel.csv data dictionary

One row per US state x wave. 51 units (50 states + DC), 5 waves.

| column | meaning |
|---|---|
| `state` | two-letter postal code |
| `wave` | YYYY-MM label |
| `date_start`, `date_end` | sample window as published |
| `cadence` | `week` (2025-08, 2025-11, 2026-02) or `month` (2026-04, 2026-05) |
| `platform` | `free_pro` or `free_pro_max`. **Changes between 2025-11 and 2026-02.** |
| `usage_count` | raw conversations. NaN for the two 2026-06 waves. |
| `usage_pct` | published share of national usage |
| `usage_share` | `usage_pct` renormalised to sum to exactly 100 within wave |
| `aui_published` | published `usage_per_capita_index`, where available |
| `working_age_pop` | fixed 2025-08 denominator, constant across waves |
| `pop_share_fixed` | `working_age_pop` as % of national |
| `pop_share_implied` | `usage_pct / aui_published`; Anthropic's own implied denominator |
| `aui_fixed` | `usage_share / pop_share_fixed`. **Primary outcome.** |
| `log_aui_fixed` | log of the above |
| `rate` | `usage_count / working_age_pop`. NaN where counts absent. |
| `census_division`, `census_region`, `bea_region`, `federal_region`, `time_zone` | official US geographies for the zoning analysis |

## Read this before analysing

1. **Platform redefinition.** Max users enter at 2026-02. That boundary
   straddles 2026-01-01, so a DiD keyed to that date cannot separate policy
   from universe change. Do not run it as if it can.
2. **Cadence mismatch.** One week versus one month. Control for it or restrict
   to same-cadence waves.
3. **Missing counts.** Branch the uncertainty model: multinomial bootstrap
   where `usage_count` is present, rounding-interval Monte Carlo where it is not.
4. **`aui_fixed` != `aui_published`.** By construction (fixed denominator).
   Use `aui_fixed` for anything across waves; cite `aui_published` when
   quoting Anthropic's own figures.
5. **2025-11 carries a CA/NY attribution anomaly.** CA loses 7.69pp and NY
   gains 6.48pp relative to 2025-08 while the other 47 states move a combined
   +1.21pp; 2026-02 reverses it. Report any result that depends on that wave
   with and without it.
"""

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p = build()
    validate(p)
    p.to_csv(OUT / "panel.csv", index=False)
    (OUT / "DICTIONARY.md").write_text(DICT)
    (OUT / "build_log.txt").write_text("\n".join(LOG))
    log(f"\nWrote {OUT/'panel.csv'}  ({len(p)} rows x {p.shape[1]} cols)")
