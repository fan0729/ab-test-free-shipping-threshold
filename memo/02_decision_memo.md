# Decision Memo: Free-Shipping Threshold $100 → $75

**To:** Product Lead, VP Growth  **From:** Data Science  **Data:** simulated 14-day experiment, 394,724 users logged

## The decision
| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: $100 → $75 |
| **Decision** | Roll out to everyone, roll out to a segment, or keep $100 (VP Growth decides, Finance signs off on margin) |
| **Decision metric** | Margin per visitor (conversion × margin per order). Guardrail: return rate |

## Recommendation
**Do not roll the $75 threshold out to all customers yet.** It reliably increases conversion (+5.7%, 95% CI +2.1% to +9.4%) and reliably raises shipping cost per order (+28%), but this test **cannot show that profit per visitor goes up or even holds**: the margin estimate is -0.7% with a CI of -5.5% to +4.2%. Fix the exposure-logging bug, then run a follow-up aimed at new customers (below).

## What we found
| Metric | Control | Treatment | Effect (95% CI) | Adj. p |
|---|---|---|---|---|
| **Margin per visitor (primary)** | $1.11 | $1.10 | -0.7% (-5.5% to +4.2%) | p = 0.79 |
| Conversion rate | 3.36% | 3.55% | +5.7% (+2.1% to +9.4%) | 0.007 |
| Net revenue per visitor | $3.07 | $3.19 | +3.9% (-0.9% to +8.6%) | 0.41 |
| *Mechanism:* AOV | $104.23 | $103.27 | -0.9% (-3.5% to +1.7%) | 0.98 |
| *Mechanism:* net shipping cost per order | $4.71 | $6.04 | +28.3% (+25.1% to +31.5%) | <0.001 |
| *Guardrail:* return rate per order | 12.3% | 12.2% | -0.8% (-10.1% to +8.5%) | 0.98 |

*Analysis population: 364,213 users (see data-quality note). Adjusted p = Holm across the six non-primary metrics.*

## Decision rule check (rule in the design doc, section 1)
| Criterion | Result | Met? |
|---|---|---|
| Conversion significantly up (Holm) | +5.7%, p = 0.007 | Yes |
| Margin per visitor CI lower bound above −2% | lower bound -5.5% | **No** |
| Return rate not significantly up | -0.8%, p = 0.98 | Yes |
| Keep-$100 trigger (margin CI upper bound below 0, conversion not up, or returns up) | margin upper bound +4.2% | Not triggered |

**Outcome:** margin is inconclusive, so the rule says do not ship to everyone and run a targeted follow-up.

**Dollar impact (assumes the analysed population stays at about 9.5M users/year; old-app users excluded):** margin per visitor changes by −$0.008 (CI −$0.062 to +$0.047), i.e. **−$71K per year, with a range from −$586K to +$443K**. The interval straddles zero, so the honest reading is "unknown sign, plausible range roughly ±$0.5M".

## Why I do not trust the topline conversion number without a fix
- **Sample ratio mismatch:** the treatment arm has 194,711 users vs 200,013 in control (χ² p = 3e-17). The shortfall sits entirely in mobile users on an old app version, where treatment exposure logging failed mostly for people who did **not** buy.
- If ignored, conversion lift reads +8.7% instead of +5.7%: the bug **overstates the win by about 3 points**.
- I excluded old-app users from both arms (a pre-treatment definition, so randomisation is preserved). Results therefore describe users who are not on the old app (about 92% of logged users).
- After exclusion the ratio is clean (p = 0.80) and covariates are balanced (all |SMD| < 0.01).

## Segments (treat as hypotheses, not findings)
| Segment | Conversion | Margin per visitor | Note |
|---|---|---|---|
| New customers (pre-registered) | +14.6% (+6.4% to +22.9%) | +7.7% (-3.5% to +18.9%) | Conversion lift survives multiple-testing adjustment |
| Returning customers (pre-registered) | +3.4% (-0.7% to +7.4%) | -2.8% (-8.2% to +2.6%) | Little extra conversion, pays the shipping cost |

The pattern makes economic sense (new customers are the ones a lower threshold can win; returning customers mostly just stop paying a fee), but the formal new-vs-returning interaction test is **not significant (p = 0.10)**. No segment's **margin** effect is significant after correction. Conversion lift is significant after correction for new customers, desktop and tablet; traffic-source cells are exploratory and none is significant.

![Segments](../figures/segments_forest.png)

## Other checks
- **Novelty:** week-1 conversion lift +4.4% vs week-2 +7.2%; difference p = 0.48 and trend test p = 0.41. At this volume a decay is not detectable either way, so week-2 numbers carry the same uncertainty as the full window. Week-2 margin is -2.7% (-9.6% to +4.2%).
- **Estimator robustness:** raw, winsorised, bootstrap, CUPED and regression-adjusted margin estimates agree within noise; CUPED shrinks variance by only 2.2%.
- **Peeking:** the daily p-value for conversion first dips under 0.05 on day 10. Stopping at the first significant look would have inflated false positives to 22%; we ran the full 14 days.

## Risks if we shipped anyway
1. Margin could be anywhere from -5.5% to +4.2%, i.e. −$586K to +$443K per year on the analysed population.
2. Shipping cost per order rises 28% for certain, and returns data is too sparse to rule out a rise.
3. Part of the conversion lift is likely novelty; we cannot size it from this test.
4. Results exclude old-app users, so they need a separate decision.
5. **Launch-and-watch is not a safe fallback.** The threshold is easy to reverse, but without a holdout the same noise applies and a before/after comparison is confounded by seasonality, so a margin loss of this size could go unnoticed.

## Proposed next steps
1. **Fix exposure logging** for old-app users and re-validate SRM before any new test.
2. **Follow-up test on new customers only** (the hypothesis the economics supports). To detect a 5% margin lift needs about 706,420 users per arm (about 124 days at current new-customer traffic), which is impractical; a **10% MDE needs about 176,605 per arm (about 31 days)**. Agree with Finance first on the margin lift that would justify shipping, because that sets the sample size.
3. **Cheaper variants** to test in parallel: a $85 threshold, or a first-order-only free-shipping offer, which keep more of the margin.
4. Add a **long-run holdout** (5%) after any launch to measure the post-novelty effect.
