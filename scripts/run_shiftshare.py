"""
run_shiftshare.py -- the one command, once you have the BLS file.

    python scripts/run_shiftshare.py data/raw/state_M2024_dl.xlsx

Writes results/tables/T7_shiftshare.csv and results/figures/F7_shiftshare.png
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aei_audit import shiftshare as ss

AEI = os.environ.get("AEI_RAW_DIR", "data/raw") + "/aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv"

def main(bls_path):
    aei = ss.load_aei_occupation_shares(AEI)
    emp = ss.load_bls_oes(bls_path)

    chk = ss.check_join(aei, emp)
    print("JOIN CHECK")
    for k, v in chk.items(): print(f"  {k}: {v}")
    if chk["in_aei_not_bls"]:
        sys.exit("\nUnmatched occupation titles. Fix the mapping, then re-run. "
                 "Do not proceed: unmatched rows are dropped silently and the "
                 "composition term will be biased.")

    res = ss.decompose(aei, emp)
    ss.test_identity(res)
    print("\n" + ss.summarise(res, top=12))

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    res.to_csv("results/tables/T7_shiftshare.csv")

    t = res.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(t))
    ax.barh(y, t.composition, color="#b03a2e", label="occupational composition")
    ax.barh(y, t.intensity, left=t.composition, color="#c9c9c9",
            label="within-occupation intensity")
    ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("deviation from national usage rate per worker")
    ax.set_title("Why are high-index states high?", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig("results/figures/F7_shiftshare.png", dpi=160, bbox_inches="tight")

    print(f"\ncomposition explains a median of {res.pct_composition.median():.0f}% "
          f"of the gap across {len(res)} states")
    print("wrote results/tables/T7_shiftshare.csv and results/figures/F7_shiftshare.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/run_shiftshare.py <path to state_MYYYY_dl.xlsx>")
    main(sys.argv[1])
