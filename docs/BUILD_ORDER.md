# Build order

Scripts to write, in order. Each reads from `data/`, writes to `results/`, and
is called by `run_all.py`. Modules in `aei_audit/` already exist and are tested.

| Script | Reads | Writes | Est. |
|---|---|---|---|
| `00_discover.py` | HF listing | `data/_schema_report.txt` | 1h |
| `01_build_panel.py` | HF releases | `data/processed/panel.csv` | 3h |
| `02_validate.py` | panel | `results/tables/validation.txt` | 0.5h |
| `03_scale.py` | panel | `T1_theil.csv`, `F1_scale_ladder.png` | 2h |
| `04_zoning.py` | panel | `T2_official_maps.csv`, `F2_zoning_null.png` | 3h |
| `05_denominators.py` | panel + ACS/OES | `T3_denominators.csv`, `F3_rank_churn.png` | 6h |
| `06_uncertainty.py` | panel | `T4_intervals.csv`, `F4_rank_matrix.png` | 3h |
| `07_vintage.py` | HF revisions | `T5_revisions.csv` | 2h |
| `08_did_sensitivity.py` | panel + treatment | `T6_did_grid.csv`, `F5_coef_spread.png` | 4h |

Cut order if time is short: 07, then 05's international section, then 06.
Never cut 02.
