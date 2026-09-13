# Cover letter

To the Editors
*Social Indicators Research*

Dear Editors,

We submit "Measurement configuration as a policy variable: threshold instability in a geographic AI
adoption index" for consideration as an Original Research article.

**What the paper does.** We do not construct a new index. We conduct an uncertainty and sensitivity
audit of an existing one, in the tradition your journal has developed over many years. Our case is
the Anthropic Economic Index, a publicly released measure of AI usage by geography that is beginning
to be cited in policy discussion. Because the data is released under an open licence, the index can
be reconstructed externally, which is what makes the analysis possible.

**The question.** Composite and representation indices increasingly carry decision rules: a region is
above or below proportional use, a jurisdiction is in the top tier, a gap is or is not closing. We ask
whether the decision, rather than merely the estimate, survives defensible variation in analytical
configuration.

**Principal findings.** Applying a simple threshold rule across five official United States government
geographies and five data waves, 65% of states change classification depending on which agency's map
is used. Decomposing uncertainty in the headline inequality statistic, the standard deviation across
those maps is 14.4 times the sampling standard deviation, and the two intervals are disjoint; the
result holds under the Theil index as well, at a ratio of 7.5. Three construction properties account
for this: aggregation to standard geographies removes 41 to 74% of measured inequality depending on
weighting convention; the reporting geography sits at the 78th to 95th percentile of a null of random
contiguous partitions; and the index is estimated from finite samples without published uncertainty,
with only 14% of adjacent rankings resolved. We close with a four-item disclosure standard for
measurements that carry decision rules.

**Fit with the journal.** The closest precedent we know of is Kuc-Czarnecka, Lo Piano and Saltelli
(2020), published in this journal, which reconstructs the World Bank's Doing Business Index to show
that a composite can support more than one narrative without any change to the underlying data. Our
contribution extends that logic in two directions: to a geographic representation index, where the
modifiable areal unit problem supplies an additional and unexamined source of variation; and from the
estimate to the classification, by reporting threshold-crossing rates rather than only ranking
instability. The methodological lineage runs directly through Saisana, Saltelli and Tarantola (2005)
and Paruolo, Saisana and Saltelli (2013).

**What we are careful not to claim.** The paper reports no finding of error in the index we examine.
The properties we quantify follow from the construction and are common to representation indices
generally; characterising them is a normal part of establishing what a measure supports. We also
decline to report a policy estimate the data cannot identify, and we say so rather than presenting a
coefficient. The manuscript documents four occasions on which our own analysis was incorrect and was
caught by validation before release, including one in which the arithmetic was exact and only an
implausible output revealed the problem.

**Reproducibility.** The harmonized panel, all analysis code, an automated validation suite, and a
pre-analysis note committed to version control before any data were loaded are openly available at
https://github.com/vkenned2/aei-index-audit. Every numeric claim traces to logged output.
Decomposition identities reconcile to 2.9 x 10^-16 and to zero respectively.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and has not been
published previously. Both authors have approved the submission. The work was neither commissioned,
funded, reviewed, nor endorsed by Anthropic. The corresponding author has applied for employment at
Anthropic and this is disclosed in the competing interests statement. Use of generative AI in
preparing the analysis and manuscript is disclosed in a dedicated section.

We hope the manuscript is of interest and would welcome the opportunity to revise in light of
reviewer comments.

Yours sincerely,

Vishal Kennedy (corresponding author)
Department of Ecology and Evolutionary Biology, University of Tennessee, Knoxville
vkenned2@vols.utk.edu

Jeevanantham Sankaran
Independent researcher, Tamil Nadu, India

---

## Suggested reviewers

Optional at this journal, but worth offering. Candidates should have published on composite indicator
robustness, the modifiable areal unit problem, or measurement in technology policy. Do not suggest
anyone at Anthropic, and do not suggest anyone you have collaborated with.

## Note on the second-choice submission

If *Social Indicators Research* declines, the same letter adapts to *Data & Policy* (Cambridge) with
two changes: replace the Kuc-Czarnecka paragraph with one framing the disclosure standard as the
contribution, and request an APC waiver at submission, since that journal is fully open access and
waivers are supported through its Alan Turing Institute, Office for National Statistics and Gates
Foundation agreements.
