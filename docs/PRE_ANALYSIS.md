# Pre-analysis note

**Committed before any real data was loaded. Do not edit after that point.**
Corrections go in a dated addendum at the bottom, never by rewriting above.

The purpose is narrow. This is an audit paper: it argues that measured
quantities move with analyst choices. That argument is only credible if the
author did not search over choices to find the largest movement. The git
timestamp on this file is the evidence that I did not.

---

## 1. Question

How much of the geographic variation reported by the Anthropic AI Usage Index
is attributable to analyst choices in constructing the index, rather than to
variation in behaviour?

## 2. Null

The null for every analysis below is **invariance**: the reported quantity does
not change when the analyst choice changes. Concretely:

| Analysis | Null (invariance) | Statistic |
|---|---|---|
| Scale | All inequality is between-group; aggregation loses nothing | Theil `share_between` = 1 |
| Zoning | The statistic is the same under any map at fixed k | spread across official maps = 0; observed sits at the median of the random-partition null |
| Denominator | Rankings do not depend on what you divide by | Spearman rho = 1 |
| Sampling | Published rankings are fully resolved | fraction of adjacent ranks resolved = 1 |
| DiD | A policy estimate does not depend on geography or denominator | spread across the nine specifications = 0 |

This is not a hypothesis-testing paper and no p-value is the object of
interest. The estimands are magnitudes of departure from invariance.

**If the null holds, that is a publishable result and I will report it as one.**
"The AI Usage Index is robust to scale, zoning, and denominator choice" is
useful to everyone using the data. I am not required to find fragility and I
will not go looking for it.

## 3. Predictions

Made before seeing any real output. Falsifiable.

- **P1 (scale).** At k=9 Census divisions, more than 40% of total Theil
  inequality is *within*-group, i.e. `share_between < 0.60`.
- **P2 (zoning).** The Gini across the five official US maps spans more than
  15% of its median.
- **P3 (zoning null).** The observed Census-division value falls inside the
  central 90% of the random contiguous k=9 partition distribution, i.e. the
  administrative map is not special.
- **P4 (denominator).** Spearman rho between the working-age-population AUI and
  the BLS knowledge-occupation AUI is below 0.90, and at least five states move
  more than ten rank positions.
- **P5 (sampling).** Fewer than 30% of adjacent published state rankings are
  resolved at the 95% level.
- **P6 (DiD).** Specification SD exceeds 0.5x sampling SD, and at least one
  Census-region specification fails or is degenerate because aggregation has
  destroyed the treated/control contrast.

## 3b. Specification grid for Analysis 6 (fixed before estimation)

Full factorial, 36 cells. Every cell is reported, including any that fail to
converge; a failure is a result about the data, not a cell to delete.

| Dimension | Levels |
|---|---|
| Geographic scale | state (k=51), Census division (k=9), Census region (k=4) |
| Denominator | working-age population, BLS knowledge-occupation employment, labour force |
| Weighting | denominator-weighted, unweighted |
| DC | included, excluded |

**Treatment coding rule.** Treatment intensity for a unit is the share of its
population living under a statute effective 2026-01-01. At state level this
collapses to binary, so the rule nests the clean case. At coarser scales it
becomes continuous-intensity DiD.

This rule exists because aggregation destroys the treated/control contrast:
CA, IL and TX fall in three different Census regions, so a binary "contains any
treated state" coding would leave one control unit out of four. The degradation
of the design under aggregation is itself a reported finding, not a nuisance.

**Bound.** A simulation-based minimum detectable effect is computed for ONE
specification only (state scale, working-age population, weighted, DC included)
and reported as the bound. A null claim without a bound is not a finding.

## 3c. What each outcome licenses

Decided before seeing results.

| Outcome | Signature | Claim permitted |
|---|---|---|
| A. Fragile | spec SD >= sampling SD; sign flips across grid | The estimate is not identified independently of geography. No single number should be published. |
| B. Robust | spec SD << sampling SD; all cells near zero; modest MDE | Under every defensible specification, no detectable effect, bounded at the MDE. Robustness strengthens the null. |
| C. Uninformative | both SDs large; MDE enormous | This data cannot answer the question. Report the bound; the paper rests on Analyses 1-5. |

Outcome C is the most likely and is survivable, because Analysis 6 was never
carrying the paper.

## 4. Analyses fixed in advance

Exactly six, in this order: scale, zoning, denominator, sampling, vintage, DiD
sensitivity. No analysis will be added after seeing results without being
labelled *post hoc* in the paper.

Primary concentration measure for the SCALE analysis is **Theil
`share_between`**, because it is exact at any k. Gini is secondary and is not
reported as a headline at k<8, where four to seven observations make it
unstable. Atkinson, HHI and CV are appendix only.

**Reconstruction gate.** Before any analysis, the published AUI must be
reproduced from published counts and the stated denominator. If reconstruction
fails, that is recorded as a finding about documentation and every subsequent
denominator comparison is labelled as reconstruction-to-reconstruction rather
than against the published index.

Primary geography: **US states.** One international section, denominator
substitution only.

## 5. Fixed decision rules

- **Outliers.** DC is expected to be extreme and will be flagged, never
  dropped. Every headline is reported with and without it. If a result depends
  on DC, that is stated in the result, not buried.
- **Suppression.** If unit counts do not sum to the published national total,
  the shortfall is reported as a percentage in the main text.
- **Unbalanced panel.** Restrict to the balanced intersection of units present
  in all waves and report which units were dropped and their size.
- **Weighting.** All concentration measures are denominator-weighted. The
  unweighted versions go in an appendix, labelled as such.
- **Seed.** 20260822 throughout. No analysis is re-run with a different seed to
  obtain a different answer.

## 6. What would falsify the paper's thesis

If `share_between` at k=9 exceeds 0.80, the official-map spread is under 5% of
median, Spearman rho across denominators exceeds 0.95, and more than 60% of
adjacent ranks resolve, then the index is robust on every dimension tested and
the paper reports that. Title becomes an affirmative one.

---

*Addenda (dated, append-only):*
