"""
aei_audit.validation
====================
Checks that the DATA is what you think it is.

tests/test_identities.py verifies the mathematics. This module verifies the
inputs. Both are necessary and they fail for different reasons: the maths can
be perfect while the panel silently lost twelve states to a bad join.

The distinction that matters:

    "the code ran"        -> no exception was raised
    "the analysis is correct" -> the numbers entering the analysis are the
                                 numbers the source actually published, the
                                 joins preserved what they should preserve,
                                 and every quantity can be recomputed a second
                                 independent way and agrees

Every function returns a Check. Run validate_panel() before any analysis and
put its output in the appendix of the paper. For an audit paper this is not
optional: publishing a critique of someone else's data handling while your own
pipeline has a join error ends the paper.

Usage
-----
    from aei_audit import validation as va
    report = va.validate_panel(panel)
    print(report)
    report.raise_if_failed()          # stop the notebook on a hard failure
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Check", "Report", "validate_panel", "cross_check_aui",
           "spot_check_unit", "check_aggregation_invariance",
           "check_suppression_accounting", "check_crosswalk_complete",
           "flag_outliers"]


@dataclass
class Check:
    name: str
    passed: bool
    severity: str          # "hard" stops the analysis, "soft" is a warning
    detail: str = ""

    def __str__(self):
        mark = "PASS" if self.passed else ("FAIL" if self.severity == "hard" else "WARN")
        return f"[{mark}] {self.name}" + (f"\n         {self.detail}" if self.detail else "")


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, c: Check):
        self.checks.append(c)
        return self

    @property
    def hard_failures(self):
        return [c for c in self.checks if not c.passed and c.severity == "hard"]

    def raise_if_failed(self):
        if self.hard_failures:
            raise AssertionError(
                "Hard validation failures; do not proceed:\n"
                + "\n".join(str(c) for c in self.hard_failures)
            )

    def __str__(self):
        n_pass = sum(c.passed for c in self.checks)
        head = f"VALIDATION: {n_pass}/{len(self.checks)} passed"
        return head + "\n" + "\n".join(str(c) for c in self.checks)


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------
def _no_duplicate_keys(panel, unit_col, wave_col) -> Check:
    keys = [unit_col] + ([wave_col] if wave_col in panel.columns else [])
    dupes = panel.duplicated(subset=keys).sum()
    return Check("no duplicate unit-wave rows", dupes == 0, "hard",
                 f"{dupes} duplicated keys" if dupes else "")


def _balanced(panel, unit_col, wave_col) -> Check:
    if wave_col not in panel.columns:
        return Check("panel balance", True, "soft", "single wave; not applicable")
    n = panel.groupby(unit_col)[wave_col].nunique()
    k = panel[wave_col].nunique()
    bad = sorted(n[n < k].index)
    return Check("panel is balanced across waves", not bad, "soft",
                 f"units missing waves: {bad}. Suppression correlates with unit "
                 f"size, so this is NON-RANDOM attrition. Restrict to the "
                 f"balanced intersection and report the drop." if bad else "")


def _positive_and_finite(panel, cols) -> Check:
    bad = {}
    for c in cols:
        if c not in panel.columns:
            continue
        s = panel[c]
        n = int((~np.isfinite(s)).sum() + (s <= 0).sum())
        if n:
            bad[c] = n
    return Check("counts and denominators positive and finite", not bad, "hard",
                 f"non-positive or non-finite: {bad}" if bad else "")


def _no_missing(panel, cols) -> Check:
    miss = {c: int(panel[c].isna().sum()) for c in cols
            if c in panel.columns and panel[c].isna().any()}
    return Check("no missing values in analysis columns", not miss, "hard",
                 f"missing: {miss}" if miss else "")


# ---------------------------------------------------------------------------
# cross-method verification  (Section 4 of the brief, done properly)
# ---------------------------------------------------------------------------
def cross_check_aui(counts, denom, published_aui=None, tol=1e-9) -> Check:
    """
    Compute the index two algebraically independent ways and require agreement.

      route A (share ratio) : (c_i / sum c) / (d_i / sum d)
      route B (rate ratio)  : (c_i / d_i) / (sum c / sum d)

    These are identical on paper. If they disagree numerically you have a
    dtype, alignment, or NaN-propagation bug. If a published_aui column is
    supplied, also check your reconstruction against Anthropic's own figure.
    """
    c = np.asarray(counts, float)
    d = np.asarray(denom, float)
    a = (c / c.sum()) / (d / d.sum())
    b = (c / d) / (c.sum() / d.sum())
    ok = np.allclose(a, b, rtol=tol)
    detail = f"max |A-B| = {np.max(np.abs(a-b)):.2e}"

    if published_aui is not None:
        p = np.asarray(published_aui, float)
        m = np.isfinite(p)
        dev = np.max(np.abs(a[m] - p[m])) if m.any() else np.nan
        agree = bool(np.isfinite(dev) and dev < 1e-3)
        ok = ok and agree
        detail += f"; max deviation from published = {dev:.2e}"
        if not agree:
            detail += (". YOUR RECONSTRUCTION DOES NOT MATCH THE PUBLISHED "
                       "INDEX. Resolve before doing anything else: you are "
                       "either using a different denominator vintage or a "
                       "different universe of conversations.")
    return Check("AUI reproduces two independent ways", ok, "hard", detail)


def spot_check_unit(panel, unit, unit_col="state", count_col="conversations",
                    denom_col="denominator") -> str:
    """
    Print the arithmetic for ONE unit so you can redo it on paper.

    Do this for two or three units and actually check them by hand. Automated
    checks catch systematic errors; hand-checking catches the assumptions the
    automated checks share with the buggy code.
    """
    row = panel.loc[panel[unit_col] == unit].iloc[0]
    C, D = panel[count_col].sum(), panel[denom_col].sum()
    c, d = row[count_col], row[denom_col]
    return (
        f"HAND CHECK: {unit}\n"
        f"  conversations       c = {c:,.0f}\n"
        f"  denominator         d = {d:,.0f}\n"
        f"  national totals     C = {C:,.0f}   D = {D:,.0f}\n"
        f"  usage share   c/C     = {c/C:.6f}\n"
        f"  expected share d/D    = {d/D:.6f}\n"
        f"  AUI = (c/C)/(d/D)     = {(c/C)/(d/D):.4f}\n"
        f"  cross-check (c/d)/(C/D) = {(c/d)/(C/D):.4f}\n"
        f"  -> redo this on a calculator. If it disagrees, stop."
    )


def check_aggregation_invariance(panel, mapping, count_col="conversations",
                                 denom_col="denominator",
                                 unit_col="state") -> Check:
    """
    Aggregation must conserve mass. Summing states into regions cannot create
    or destroy conversations. This catches the most common join bug, where a
    crosswalk silently drops or duplicates units.
    """
    from .aggregation import aggregate_panel
    before_c, before_d = panel[count_col].sum(), panel[denom_col].sum()
    agg = aggregate_panel(panel, mapping, unit_col=unit_col,
                          count_col=count_col, denom_col=denom_col)
    after_c, after_d = agg[count_col].sum(), agg[denom_col].sum()
    ok = np.isclose(before_c, after_c) and np.isclose(before_d, after_d)
    return Check("aggregation conserves totals", ok, "hard",
                 f"counts {before_c:,.0f} -> {after_c:,.0f}; "
                 f"denominators {before_d:,.0f} -> {after_d:,.0f}")


def check_suppression_accounting(panel, published_total=None,
                                 count_col="conversations") -> Check:
    """
    Does the sum of published unit counts equal the published national total?

    THIS IS THE CHECK THE WHOLE SCALE ANALYSIS RESTS ON. If suppressed cells
    are dropped rather than pooled into a remainder, the sum of units falls
    short of the national total, and every aggregate you build is computed on
    a censored base. Report the shortfall as a percentage in the paper.
    """
    if published_total is None:
        return Check("suppression accounting", True, "soft",
                     "No published national total supplied. Find one, or ask "
                     "econ-research@anthropic.com whether suppressed mass is "
                     "dropped or pooled. Do not assume it is pooled.")
    s = panel[count_col].sum()
    gap = (published_total - s) / published_total
    return Check("unit counts sum to the published total", abs(gap) < 0.01, "soft",
                 f"units sum to {s:,.0f} vs published {published_total:,.0f} "
                 f"({100*gap:+.2f}%). Any shortfall is suppressed mass and must "
                 f"be disclosed.")


def check_crosswalk_complete(panel, cols, unit_col="state") -> Check:
    """
    Every unit must map to a group under EVERY geography.

    pandas groupby drops NaN keys silently, so an incomplete crosswalk means
    one map is computed on a different unit set than the others and the
    comparison across maps is invalid. This is a HARD failure: it produced a
    real bug (AK and HI absent from TIME_ZONE, so the time-zone column was
    computed on 49 of 51 states).
    """
    bad = {}
    for c in cols:
        if c not in panel.columns:
            bad[c] = "column absent"
            continue
        miss = sorted(panel.loc[panel[c].isna(), unit_col].unique())
        if miss:
            bad[c] = miss
    return Check("geography crosswalks cover every unit", not bad, "hard",
                 f"unmapped: {bad}" if bad else "")


def flag_outliers(panel, unit_col="state", count_col="conversations",
                  denom_col="denominator", z=3.0) -> Check:
    """
    Flag, do not drop. DC will almost certainly appear here because of its
    workforce composition. That is real, not an error. Run every headline
    with and without flagged units; if a finding depends on one unit, say so.
    """
    r = np.log(panel[count_col] / panel[denom_col])
    med, mad = np.median(r), np.median(np.abs(r - np.median(r)))
    rob = 0.6745 * (r - med) / mad if mad > 0 else r * 0
    out = panel.loc[np.abs(rob) > z, unit_col].tolist()
    return Check("outlier scan (flag, never drop)", True, "soft",
                 f"robust-z > {z}: {out}. Re-run headlines with and without "
                 f"these; report if a result depends on them." if out
                 else "none flagged")


# ---------------------------------------------------------------------------
def validate_panel(panel: pd.DataFrame, unit_col="state", wave_col="wave",
                   count_col="conversations", denom_col="denominator",
                   published_aui_col=None, published_total=None,
                   mapping=None) -> Report:
    """Run every check. Call before any analysis; print the result into the appendix."""
    cols = [count_col, denom_col]
    rep = Report()
    rep.add(_no_duplicate_keys(panel, unit_col, wave_col))
    rep.add(_no_missing(panel, cols))
    rep.add(_positive_and_finite(panel, cols))
    rep.add(_balanced(panel, unit_col, wave_col))

    first = (panel[panel[wave_col] == panel[wave_col].iloc[0]]
             if wave_col in panel.columns else panel)
    rep.add(cross_check_aui(
        first[count_col], first[denom_col],
        first[published_aui_col] if published_aui_col else None))

    if mapping is not None:
        rep.add(check_aggregation_invariance(first, mapping,
                                             count_col, denom_col, unit_col))
    rep.add(check_suppression_accounting(first, published_total, count_col))
    rep.add(flag_outliers(first, unit_col, count_col, denom_col))
    return rep
