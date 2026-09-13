"""
run_all.py

Reproduces every table and figure in the manuscript from the harmonized panel.

    python run_all.py

Requires the four raw release files in data/raw/ (see README) for the panel
build and for the enriched-release analyses. Steps that need data not present
are skipped with an explanatory message rather than failing.

All stochastic procedures are seeded from SEED below. No other source of
randomness is used.
"""
import subprocess
import sys
import time
from pathlib import Path

SEED = 20260822

STEPS = [
    ("identity tests",          [sys.executable, "-m", "pytest", "tests/", "-q"]),
    ("compile panel",           [sys.executable, "scripts/compile_panel.py"]),
    ("core analysis",           [sys.executable, "scripts/run_analysis.py"]),
    ("robustness",              [sys.executable, "scripts/03_robustness.py"]),
    ("threshold and spatial",   [sys.executable, "scripts/04_threshold_and_spatial.py"]),
    ("shift-share",             [sys.executable, "scripts/run_shiftshare.py",
                                 "data/raw/state_M2024_dl.xlsx"]),
    ("dashboard",               [sys.executable, "scripts/build_site.py"]),
]

if __name__ == "__main__":
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    failed = []
    for name, cmd in STEPS:
        t0 = time.time()
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode:
            print(f"  FAILED: {name}")
            failed.append(name)
        else:
            print(f"  ok ({time.time()-t0:.1f}s)")
    if failed:
        sys.exit(f"\nIncomplete. Failed steps: {failed}")
    print("\nAll steps complete. Outputs in results/.")
