# Cover letter, Data & Policy

To the Editors
*Data & Policy*
Cambridge University Press

Dear Editors,

We submit "Measurement configuration as a policy variable: threshold instability in a geographic AI
adoption index" for consideration as a Research Article.

**The problem.** Statistics describing the geographic spread of artificial intelligence are entering
policy discussion, and increasingly they carry decision rules: a region is above or below
proportional use, a jurisdiction is in the top tier, a gap is or is not closing. Whether such a rule
returns a stable answer depends on analytical choices that are not currently disclosed anywhere. We
ask whether the decision, rather than merely the estimate, survives defensible variation in how the
measurement is configured.

**What we did.** We audit a publicly released measure of AI usage by geography across five waves and
51 United States jurisdictions. We do not construct a new index. The dataset is released under an
open licence, which is what makes external reconstruction possible; the properties we describe follow
from the construction and are common to representation indices generally.

**Principal findings.** Applying a simple threshold rule across the five official United States
government geographies, 65% of states change classification depending on which agency's map is used.
Decomposing uncertainty in the headline inequality statistic, the standard deviation across those
maps is 14.4 times the sampling standard deviation, and the two intervals are disjoint; the result
holds under an alternative inequality measure at a ratio of 7.5. Three construction properties
account for this: aggregation to standard geographies removes 41 to 74% of measured inequality
depending on weighting convention; the reporting geography sits at the 78th to 95th percentile of a
null distribution of random contiguous partitions; and the index is estimated from finite samples
without published uncertainty, with only 14% of adjacent rankings resolved.

**Why this belongs in Data & Policy.** The contribution is not the audit but what follows from it: a
four-item disclosure standard for measurements that carry decision rules, covering the configuration
grid considered, the threshold-crossing rate, the specification-to-sampling variance ratio, and the
configurations that flip the conclusion. We argue this generalises beyond geographic indices to any
threshold-triggered governance rule, including the capability thresholds now appearing in frontier AI
regulation, where the evaluation literature has independently established that scores are
substantially configuration-dependent. The paper is therefore addressed to the question of what
evidence standards should apply when data products become policy triggers, which we understand to be
central to the journal's remit.

**What we are careful not to claim.** The paper reports no finding of error in the index we examine.
We also decline to report a policy estimate the data cannot identify, and say so rather than
presenting a coefficient. The manuscript documents four occasions on which our own analysis was
incorrect and was caught by validation before release, including one in which the arithmetic was
exact and only an implausible output revealed the problem.

**Reproducibility.** The harmonized panel, all analysis code, an automated validation suite, and a
pre-analysis note committed to version control before any data were loaded are openly available at
https://github.com/vkenned2/aei-index-audit. Every numeric claim traces to logged output.
Decomposition identities reconcile to 2.9 x 10^-16 and to zero respectively.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and has not been
published previously. Both authors have approved the submission. The work was neither commissioned,
funded, reviewed, nor endorsed by the organisation whose data we examine. The corresponding author
has applied for employment at that organisation and this is disclosed in the competing interests
statement. Use of generative AI in preparing the analysis and manuscript is disclosed in a dedicated
section.

We would welcome the opportunity to revise in light of reviewer comments.

Yours sincerely,

Vishal Kennedy (corresponding author)
Department of Ecology and Evolutionary Biology, University of Tennessee, Knoxville
vkenned2@vols.utk.edu

Jeevanantham Sankaran
Independent researcher, Tamil Nadu, India

---

## NOTE, remove before sending

Do NOT mention the APC, a waiver request, or funding in this letter. Cambridge states that editors,
editorial board members and reviewers have no involvement with open access funding and cannot grant
discounts or waivers. Raising it here achieves nothing and puts a commercial question in front of
people whose only job is to assess the science.

The APC is handled separately, and only after acceptance. See the two emails below.
