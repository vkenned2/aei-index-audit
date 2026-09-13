"""
03_robustness.py

Reproduces the robustness results reported in the manuscript:

    Section 4.1   weighting sensitivity (41 to 74% range)
    Section 4.3   multi-k zoning null
    Section 4.6   placebo breakpoints, temporal stability
    leave-one-state-out on the headline statistics

Input:   data/processed/panel.csv
Outputs: results/tables/T5_weighting.csv
         results/tables/T6_leave_one_out.csv
         results/tables/T9_multi_k_null.csv
         results/tables/T10_placebo_breaks.csv

Usage:   python scripts/03_robustness.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aei_audit import robustness as rb  # noqa: E402

SEED = 20260822
TAB = Path("results/tables")
TAB.mkdir(parents=True, exist_ok=True)


def main():
    panel = pd.read_csv("data/processed/panel.csv")

    # --- Section 4.1: does the headline survive weighting convention? -------
    W = rb.weighting_sensitivity(panel)
    W.to_csv(TAB / "T5_weighting.csv", index=False)
    print("WEIGHTING SENSITIVITY, % of inequality lost at k=9")
    print(W.pivot(index="wave", columns="weighting", values="pct_lost_k9").round(1).to_string())
    print(f"  full range across all wave x convention cells: "
          f"{W.pct_lost_k9.min():.1f}% to {W.pct_lost_k9.max():.1f}%\n")

    # --- leave-one-state-out -------------------------------------------------
    L = rb.leave_one_out(panel, "2026-02")
    L.to_csv(TAB / "T6_leave_one_out.csv", index=False)
    print("LEAVE-ONE-STATE-OUT (2026-02)")
    print(f"  full sample: gini={L.attrs['full_gini']:.4f} "
          f"share_between={L.attrs['full_share_between']:.4f}")
    print(f"  gini range across 51 drops:          "
          f"{L.gini.min():.4f} to {L.gini.max():.4f}")
    print(f"  share_between range across 51 drops: "
          f"{L.share_between_k9.min():.4f} to {L.share_between_k9.max():.4f}\n")

    # --- Section 4.3: is the k=9 result specific to k=9? ---------------------
    M = rb.multi_k_zoning_null(panel, "2026-02", ks=(4, 6, 8, 9, 10),
                               n_sims=1000, seed=SEED)
    M.to_csv(TAB / "T9_multi_k_null.csv", index=False)
    print("MULTI-k ZONING NULL (2026-02, 1,000 partitions per k)")
    print(M.dropna(subset=["real_map"])[
        ["k", "real_map", "observed", "null_median", "percentile",
         "null_spread_pct_med"]].round(3).to_string(index=False))
    print()

    # --- Section 4.6: temporal stability and placebo breakpoints -------------
    T = rb.temporal_stability(panel)
    print("TEMPORAL STABILITY, Spearman between waves")
    print(T["spearman"].round(3).to_string())
    print("\n  boundary               mean|rank move|  max|share move| pp  top-decile persistence")
    for k in T["mean_abs_rank_move"]:
        print(f"  {k:22s} {T['mean_abs_rank_move'][k]:8.2f} "
              f"{T['max_abs_share_move_pp'][k]:15.2f} "
              f"{T['top_decile_persistence'][k]:22.2f}")

    B = rb.placebo_breaks(panel)
    B.to_csv(TAB / "T10_placebo_breaks.csv", index=False)
    print("\nPLACEBO BREAKPOINTS, every wave boundary")
    print(B[["boundary", "tvd_pp", "d_theil", "max_state_move_pp",
             "which_state", "spearman"]].round(4).to_string(index=False))
    print("\n  The 2025-11 to 2026-02 boundary carries the product-universe change.")
    print("  It is not the largest distributional break in the series.")


if __name__ == "__main__":
    main()
