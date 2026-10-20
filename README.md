# A/B Testing Case Study: Free-Shipping Threshold ($100 → $75)

## Product brief
| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: $100 → $75, for a fashion e-commerce store |
| **Decision** | Roll out to everyone, roll out to a segment, or keep $100 (VP Growth decides; Finance signs off on margin) |
| **Decision metric** | Margin per visitor = conversion × margin per order. Guardrail: return rate |
| **Decision rule** | Ship only if conversion is significantly up, the margin CI's lower bound is above −2%, and returns are not significantly up (`memo/01_experiment_design.md`) |
| **Outcome** | Conversion +5.7%, margin per visitor inconclusive (-5.5% to +4.2%), so do not ship to everyone; test new customers |

An end-to-end experimentation project on **simulated** data with a **known ground truth**: experiment design, power analysis, validity checks, layered analysis, sequential-testing simulation, 100-replication validation, and a decision memo. Because the true effect is known, every method is checked against the right answer.

> **Everything here is simulated, and the code was built with Claude Code.** Numbers are from `data_gen/generate_experiment.py` (fixed seed). A Criteo replication script is included (`src/05_criteo_replication.py`) but has **not been run**; no real-data result is claimed.

## Headline result
| Question | Answer |
|---|---|
| Does $75 raise conversion? | Yes: **+5.7%** (95% CI +2.1% to +9.4%) |
| Does it raise profit per visitor? | **Unknown**: -0.7% (CI -5.5% to +4.2%); the test can only detect ≥6.7% |
| What does it cost? | Net shipping cost per order **+28%** |
| Recommendation | Do not ship globally; fix logging; run a follow-up on new customers (`memo/02_decision_memo.md`) |

## What the project demonstrates
| Skill | Where | Result |
|---|---|---|
| Experiment design, metric hierarchy, MDE | `memo/01_experiment_design.md` | 14 days is under-powered for margin (6.7% MDE); the design says so up front |
| Power analysis (formula + simulation) | `src/01_power.py` | Formula 80% vs bootstrap simulation 76% at the same effect (heavy tails) |
| **SRM diagnosis** | `src/02_validity.py` | χ² p = 3e-17, localised to old-app mobile; naive lift +8.7% vs +5.7% after exclusion |
| Covariate balance | `src/02_validity.py` | SMD missed the bug (|SMD| = 0.09 < 0.1); SRM caught it |
| A/A validation | `src/02_validity.py` | False-positive rate over 1,000 splits: margin/Welch 4.6%, conversion 5.4%, margin/CUPED 4.7% |
| Heavy-tail robustness | `src/03_analysis.py` | Welch, winsorised, bootstrap agree |
| **CUPED / regression adjustment** | `src/03_analysis.py` | Only 2.2% variance reduction: documented as a limit, not a win |
| Ratio metrics (delta method) | `src/abtools.py` | AOV CI matches the user-level bootstrap |
| Multiple testing | `src/03_analysis.py` | Holm (secondary metrics), BH (segments) |
| Heterogeneity | `src/03_analysis.py` | New customers look best, but interaction p = 0.10: a hypothesis, not a finding |
| Novelty | `src/03_analysis.py` | Week-1 vs week-2 p = 0.48: not resolvable at this volume |
| **Peeking / sequential testing** | `src/04_peeking.py` | Daily looks: 21.9% false positives; Pocock 5.0%, OBF-style 4.9% |
| Ground-truth validation | `src/06`, `src/07` | 100 replications: margin CI coverage 94%; naive analysis covers the truth 69% |

## What the ground truth revealed
- True conversion lift +6.6% (window) / +5.0% (steady state); estimate +5.7%.
- True margin effect +1.5% (window) / +0.1% (steady state), so the inconclusive verdict was right.
- New customers truly gain (+6.0%), returning customers truly lose (-1.5%), but the test lacked the power to confirm it.

## Data-generating process (what is built in)
Zero-inflated heavy-tailed baskets with rare giant orders; pre-period spend; new-vs-returning heterogeneity; basket-padding toward the threshold; day-of-week seasonality; novelty effect on purchase odds; return rates; and an **injected exposure-logging bug** (treatment-arm users on old mobile app versions go missing, mostly non-buyers). Cost model: 42% gross margin, $9.00 carrier cost, $7.50 customer fee under the threshold, $6 per return.

## Repository layout
```
data_gen/generate_experiment.py   simulator (potential outcomes for both arms, hidden truth)
src/abtools.py                    reusable toolkit: SRM, SMD, Welch, bootstrap, CUPED, delta method, sequential boundaries
src/01_power.py ... 08_build_docs.py
notebooks/ab_test_walkthrough.ipynb   executed narrative walkthrough
memo/                             design doc and decision memo (md + PDF), resume/interview notes
figures/  results/                every chart and every number cited
run_all.sh                        reproduces everything (--skip-montecarlo skips the 10-minute run)
```
Run: `pip install -r requirements.txt && bash run_all.sh`

## Limitations (stated plainly)
- Simulated users are independent: no interference, network effects or long-run behaviour change.
- The cost model and effect sizes are assumptions chosen to be plausible, not estimates from a real business.
- Pre-period covariates are weak by construction of a sparse purchase outcome; a real site with richer history would see larger CUPED gains.
- Sequential boundaries assume equal information per day; a production design would use alpha-spending on actual information fractions.
