# Measurement configuration as a policy variable

Replication materials for *Measurement configuration as a policy variable: threshold instability in a
geographic AI adoption index.*

**Vishal Kennedy**, University of Tennessee, Knoxville
**Jeevanantham Sankaran**, Independent researcher, Tamil Nadu, India

---

## Overview

Indices of AI usage built on geographic shares are increasingly used to support threshold decisions: a
region is above or below proportional use, a jurisdiction is in a top tier, a gap is or is not
closing. This repository contains the code behind a measurement audit asking whether such decisions
survive defensible variation in how the measurement is configured.

The case study is the Anthropic Economic Index, examined across five waves covering 51 United States
jurisdictions. The analysis does not construct a new index. It reconstructs an existing one and tests
what that index supports.

## Requirements

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies are numpy, pandas, matplotlib, scipy, openpyxl, huggingface_hub and pytest. The
`pyfixest` entry in `requirements.txt` is needed only by `aei_audit/speccurve.py`, which supports the
specification-curve design described in Section 3.6 and is not exercised by the main pipeline.

---

## Data

**No raw data is committed.** Two external sources are required.

### 1. Anthropic Economic Index

Released under CC-BY-4.0 at
[huggingface.co/datasets/Anthropic/EconomicIndex](https://huggingface.co/datasets/Anthropic/EconomicIndex).

Four release files are needed. Place them in `data/raw/`:

| File | Release | Provides |
|---|---|---|
| `aei_enriched_claude_ai_2025-08-04_to_2025-08-11.csv` | 2025-09-15 | counts, working-age population, gross state product, `usage_tier`, occupational layer |
| `aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv` | 2026-01-15 | counts |
| `aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv` | 2026-03-24 | counts |
| `aei_claude_ai_2026-06-26.csv` | 2026-06-26 | two monthly waves, shares only |

The four files use four incompatible schemas. `scripts/compile_panel.py` documents each and
normalises them into a single panel.

If your files live elsewhere, set `AEI_RAW_DIR` rather than editing the scripts:

```bash
AEI_RAW_DIR=/path/to/files python scripts/compile_panel.py
```

### 2. Bureau of Labor Statistics OEWS

State employment by SOC major group, required only for the shift-share in Section 4.8.

Download the state file from [bls.gov/oes/tables.htm](https://www.bls.gov/oes/tables.htm) under "All
data", extract it, and place `state_M2024_dl.xlsx` in `data/raw/`.

### Included Data

`data/processed/panel.csv` is the harmonized 255-row panel derived from the AEI releases. It is
redistributed here under CC-BY-4.0 with attribution to Anthropic, so the analysis scripts can run
without first obtaining the raw releases. `data/processed/DICTIONARY.md` defines every column and
records the five caveats that govern interpretation.

---

## Reproducing the Analysis

```bash
python run_all.py
```

Runs the whole pipeline in order. Steps whose inputs are absent are skipped with an explanatory
message rather than failing.

To run individual stages:

| Order | Command | Produces |
|---|---|---|
| 1 | `python -m pytest tests/ -q` | verifies the decomposition identities before anything else |
| 2 | `python scripts/compile_panel.py` | `data/processed/panel.csv`, `build_log.txt` (needs raw releases) |
| 3 | `python scripts/run_analysis.py` | Tables 1 to 4, Figures 1 to 6 |
| 4 | `python scripts/03_robustness.py` | Tables 5, 6, 9, 10 |
| 5 | `python scripts/04_threshold_and_spatial.py` | Tables 11, 12, Figure 7 |
| 6 | `python scripts/run_shiftshare.py data/raw/state_M2024_dl.xlsx` | Tables 7, 8, shift-share figure |
| 7 | `python scripts/build_site.py` | `docs/index.html`, a self-contained interactive release |

Without the raw releases, skip step 2. The committed panel covers steps 3 to 5 and 7. Sections 4.4,
4.5 and 4.7 additionally need the enriched 2025-08 file, because gross state product, `usage_tier` and
the occupational layer appear only there; step 5 detects its absence and says so.

---

## Reproducibility

**Seed.** Every stochastic procedure is seeded from `SEED = 20260822`. No bare `numpy.random` calls
appear in `aei_audit/` or `scripts/`.

**Iteration counts.** 4,999 permutations for Moran's I inference; 4,000 resamples for tier separation
and rank resolution; 2,000 for the variance decomposition; 1,500 for the zoning null; 1,000 per level
for the multi-k null.

**Identity tests.** `tests/test_identities.py` checks the Theil decomposition identity, CLR closure,
scale invariance of the representation ratio, contiguity of generated partitions, and that
aggregation recomputes the index rather than averaging ratios. Maximum observed decomposition residual
is 2.9e-16; the shift-share identity reconciles to exactly zero.

**Validation gates.** `aei_audit/validation.py` runs before analysis and fails hard on duplicate keys,
missing values, non-positive counts, cross-method disagreement in the index, non-conservation under
aggregation, or an incomplete geographic crosswalk. The last was added after an incomplete time-zone
crosswalk caused one geography to be computed on 49 units while being compared against others computed
on 51.

**Pre-analysis note.** `docs/PRE_ANALYSIS.md` records predictions and decision rules committed before
any real data was loaded. Two of six predictions were falsified; the manuscript reports this.

---

## License

Code is MIT licensed (see `LICENSE`). Analysis text and figures are CC-BY-4.0. The Anthropic Economic
Index is CC-BY-4.0 by Anthropic. BLS OEWS data is in the public domain.

## Disclosure

This repository contains an independent analysis. It was not commissioned, funded, reviewed, or endorsed by Anthropic.
The corresponding author has applied for employment at Anthropic; this is disclosed in the
manuscript's competing interests statement.
