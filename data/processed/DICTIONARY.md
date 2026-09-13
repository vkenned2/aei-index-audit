# panel.csv data dictionary

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
