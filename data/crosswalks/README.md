# Denominators to source (Analysis 3)

Anthropic's AUI divides by working-age population. That is one choice among
several, and the choice determines the conclusion.

| Denominator | Source | Why it matters |
|---|---|---|
| Working-age population (15-64) | UN WPP / Census ACS S0101 | Anthropic's own choice. Baseline. |
| Internet users | ITU DataHub | Reframes the gap as ACCESS. The most policy-relevant swap. |
| Labour force | ILOSTAT / BLS LAUS | Ties usage to work rather than residence |
| Knowledge-occupation employment | BLS OES (SOC 11,13,15,17,19,23,27) | Anthropic's own occupational analysis implies this is the exposed base |
| Broadband subscriptions | ITU / FCC Form 477 | Infrastructure ceiling |
| GDP (PPP) | World Bank | Strauss's angle. For comparability, not primary. |

METHOD: recompute AUI under each, then report (a) Spearman rank correlation
between rankings, (b) how far named units move, (c) which choice would change
a policy conclusion.

Expected headline: India at 0.27 on working-age population should move
substantially on an internet-user denominator. If it does, the "AI divide"
is measuring connectivity as much as adoption.
