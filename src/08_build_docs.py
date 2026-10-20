"""Build the written deliverables from results/*.json so every number in prose is traceable to code output.

  memo/01_experiment_design.md      (uses power.json only -- written BEFORE looking at results)
  memo/02_decision_memo.md          (uses analysis/validity/peeking; NOT the ground truth)
  memo/03_resume_and_interview.md   (bullets + prep; uses everything incl. Monte Carlo + truth)
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
L = lambda n: json.load(open(f"{ROOT}/results/{n}.json"))
P, A, V, K, M, R = L("power"), L("analysis"), L("validity"), L("peeking"), L("montecarlo"), L("reveal")


def pct(x, d=1, sign=False):
    return f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%"


def usd(x, d=2, sign=False):
    s = f"{abs(x):,.{d}f}"
    return (("+" if x >= 0 else "−") if sign else ("−" if x < 0 else "")) + "$" + s


def k(x):
    return f"${abs(x)/1000:,.0f}K"


def ksign(x):
    return ("+" if x >= 0 else "−") + k(x)


met = {r["metric"]: r for r in A["metrics"]}
nai = {r["metric"]: r for r in A["naive_metrics"]}
seg = {r["segment"]: r for r in A["segments"]}
base, ach, ideal, trf = P["baseline"], P["achieved"], P["ideal"], P["traffic"]
mp, cv, ship, ret, aov = met["Margin per visitor"], met["Conversion rate"], met["Net shipping cost per order"], met["Return rate (per order)"], met["AOV (basket $ per order)"]
an = A["annualisation"]
nov = A["novelty"]
srm = V["srm_all"]
new_days = 2 * P["new_only"]["n_per_arm_5pct"] / (trf["per_day"] * P["new_only"]["share_new"])
n10 = P["new_only"]["n_per_arm_5pct"] / 4
new_days10 = 2 * n10 / (trf["per_day"] * P["new_only"]["share_new"])
users_yr_plan = trf["per_day"] * 365
margin_yr_plan = users_yr_plan * base["margin_per_visitor"]

# pre-committed decision rule (design doc section 1), evaluated on the clean-population results
crit_conv = cv["p_holm"] < 0.05 and cv["diff"] > 0
crit_margin = mp["rel_lo"] > -0.02
crit_return = not (ret["p_holm"] < 0.05 and ret["diff"] > 0)
crit_kill = (mp["rel_hi"] < 0) or (not crit_conv) or (not crit_return)
OUTCOME = "ship" if (crit_conv and crit_margin and crit_return) else ("keep" if crit_kill else "follow-up")
assert OUTCOME == "follow-up", "results changed: the memo recommendation text must be rewritten"
yes = lambda b: "Yes" if b else "**No**"

# =============================================================== 1. DESIGN DOC
design = f"""# Experiment Design: Free-Shipping Threshold $100 → $75

*Status: drafted before analysing the experiment (see the process note in section 1). Planning inputs come from a separate historical sample of {base['pilot_n']:,} control-experience users. All data in this project is **simulated**.*

## 1. Product brief
| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: lower from $100 to $75, for every order on web and app |
| **Who decides** | Checkout / Growth PM proposes, VP Growth decides, Finance signs off on the margin definition |
| **User problem (assumed)** | Carts under the threshold see a $7.50 shipping fee late in checkout. Shoppers with a $75–$100 cart either leave or pad the cart to reach free shipping. |
| **Hypothesis** | Removing the fee for $75–$100 carts raises conversion enough to pay for the shipping cost we absorb on those orders |
| **Why it is not obvious** | Every order in the $75–$100 band becomes less profitable even if more people buy, so conversion alone would score the feature as a win |
| **Decision** | Roll out at $75 to everyone, roll out to a segment, or keep $100 |
| **Reversibility** | A threshold is a config change, so a wrong launch can be rolled back. The cost of being wrong is the margin given away until someone notices |
| **What the decision is worth (planning figure, all traffic)** | About {users_yr_plan/1e6:.1f}M visitors a year × {usd(base['margin_per_visitor'])} margin per visitor ≈ ${margin_yr_plan/1e6:.1f}M margin a year. 1% of margin per visitor ≈ ${margin_yr_plan*0.01/1e3:,.0f}K a year; the {pct(ach['mde_margin_raw'])} MDE ≈ ${margin_yr_plan*ach['mde_margin_raw']/1e3:,.0f}K a year |

**Metric tree**
```
Margin per visitor  =  conversion rate  ×  margin per order
margin per order    =  42% × basket × (1 − return)  −  net shipping cost  −  $6 × return
```

**Decision rule** (evaluated on the primary population after validity checks):

| Outcome | Condition | Action |
|---|---|---|
| **Ship to everyone** | Conversion significantly up (Holm p < 0.05) **and** margin per visitor 95% CI lower bound above −2% **and** return rate not significantly up | Roll out; keep a 5% holdout |
| **Do not ship broadly; targeted follow-up** | Anything else that is not a kill, i.e. margin is inconclusive | Test the pre-registered segment (new customers) or a cheaper variant |
| **Keep $100** | Margin CI upper bound below 0, **or** conversion not significantly up, **or** return rate significantly up | Stop |

The −2% is the margin loss Growth would tolerate in exchange for new-customer growth. In a real project I would agree it with Finance before the test.

*Process note: in this portfolio build, the first draft already contained the margin criterion (CI lower bound above −2%) and the rule "inconclusive margin means do not ship broadly". The Keep-$100 row, the return-rate wording and the table layout were added afterwards for clarity. The recommendation does not depend on them.*

## 2. Design
| Item | Choice | Why |
|---|---|---|
| Randomisation unit | User (assigned on first visit in the window) | Avoids one person seeing both prices |
| Allocation | 50 / 50 | Maximises power |
| Window | 14 days | Covers two full weekly cycles; business cap on run time |
| Traffic | about {trf['per_day']:,.0f} users/day, {trf['total']:,} total, {trf['per_arm']:,} per arm | Planning assumption |
| Exposure logging | Client-side tag | **Known risk:** tag failures would cause SRM, so SRM is checked first |

## 3. Metrics
| Role | Metric | Definition |
|---|---|---|
| **Primary** | Margin per visitor | 42% × merchandise value after returns − net shipping cost − $6 per return, summed per assigned user (zeros included) |
| Driver (secondary) | Conversion rate; net revenue per visitor | Did the feature do what it was built to do? |
| Mechanism (diagnostic) | AOV; items per order; net shipping cost per order | Explain *why* margin moved. Shipping cost rises by construction, so it cannot be a guardrail |
| Guardrail | Return rate per order | Extra buyers may return more. Page-load time is not simulated |
| Pre-registered segments | New vs returning; device | Chosen *before* data, because new customers are expected to respond most |
| Exploratory segments | Traffic source | Reported but flagged exploratory |

**Assumptions (cost model):** carrier cost $9.00 per order; customers under the threshold pay a $7.50 fee, so net shipping cost is $1.50 (paid) or $9.00 (free).

## 4. Baseline and variance (from the historical sample)
- Conversion: **{pct(base['conv'],2)}**
- Margin per visitor: **{usd(base['margin_per_visitor'])}**, standard deviation **{usd(base['margin_sd'])}** (coefficient of variation {base['cv']:.1f}: zero-inflated and heavy-tailed)
- Outlier rule: winsorise per-user margin at the pilot 99.95th percentile (**{usd(base['winsor_cap'])}**); report raw and winsorised results

## 5. Power and sample size (α = 0.05 two-sided, power 80%)
| Question | Answer |
|---|---|
| Users per arm needed for a **5%** relative lift in margin per visitor | **{ideal['n_per_arm_5pct']:,.0f}** → about **{ideal['days_needed_5pct']:.0f} days** at planned traffic |
| Users per arm available in 14 days | **{trf['per_arm']:,}** |
| Min. detectable effect on margin per visitor in 14 days | **{pct(ach['mde_margin_raw'])}** relative ({pct(ach['mde_margin_adj'])} after covariate adjustment) |
| Min. detectable effect on conversion in 14 days | **{pct(ach['mde_conv_rel'])}** relative |
| Simulation check of power at the margin MDE | formula says {P['sim_power'][2]['power_formula']:.0%}, bootstrap simulation says **{P['sim_power'][2]['power_sim']:.0%}** at a {pct(P['sim_power'][2]['rel_mde'],1)} effect (heavy tails make the formula optimistic) |

![MDE vs sample size](../figures/power_curve.png)

**Honest planning conclusion:** a 14-day test is well powered for conversion but **under-powered for margin per visitor**. We accept this because 14 days is the business cap, and we record it up front: if margin per visitor comes back inconclusive, that is the expected outcome, not a failure of the test.
Covariate adjustment helps little ({pct(base['cuped_var_reduction_prespend_only'])} variance reduction in the pilot) because a purchase is a rare event and 40% of users are new with no history.

## 6. Analysis plan (fixed in advance)
1. **Validity first:** SRM chi-square at α = 0.001; covariate balance (|SMD| > 0.1 flagged); 1,000-split A/A test of the pipeline.
2. **Primary:** difference in mean margin per visitor with 95% CI (Welch), cross-checked by bootstrap, winsorised, CUPED and regression-adjusted versions.
3. **Driver, mechanism and guardrail metrics:** Holm-adjusted p-values across the six non-primary metrics; ratio metrics by delta method (users are the randomisation unit).
4. **Segments:** pre-registered segments get a formal interaction test; all segment cells are BH-adjusted.
5. **Time effects:** week-1 vs week-2 comparison and a linear trend test on the treatment effect.
6. **Decision rule:** the decision table in section 1.
7. **Looks:** no early stopping. A sequential design is evaluated separately (see peeking analysis).
"""
os.makedirs(f"{ROOT}/memo", exist_ok=True)
open(f"{ROOT}/memo/01_experiment_design.md", "w").write(design)

# =============================================================== 2. DECISION MEMO
n_clean, n_all = A["population"]["n_clean"], A["population"]["n_all"]
w2m, w2c = nov["week2"]["margin"], nov["week2"]["conv"]
memo = f"""# Decision Memo: Free-Shipping Threshold $100 → $75

**To:** Product Lead, VP Growth  **From:** Data Science  **Data:** simulated 14-day experiment, {n_all:,} users logged

## The decision
| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: $100 → $75 |
| **Decision** | Roll out to everyone, roll out to a segment, or keep $100 (VP Growth decides, Finance signs off on margin) |
| **Decision metric** | Margin per visitor (conversion × margin per order). Guardrail: return rate |

## Recommendation
**Do not roll the $75 threshold out to all customers yet.** It reliably increases conversion ({pct(cv['rel'],1,True)}, 95% CI {pct(cv['rel_lo'],1,True)} to {pct(cv['rel_hi'],1,True)}) and reliably raises shipping cost per order ({pct(ship['rel'],0,True)}), but this test **cannot show that profit per visitor goes up or even holds**: the margin estimate is {pct(mp['rel'],1,True)} with a CI of {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}. Fix the exposure-logging bug, then run a follow-up aimed at new customers (below).

## What we found
| Metric | Control | Treatment | Effect (95% CI) | Adj. p |
|---|---|---|---|---|
| **Margin per visitor (primary)** | {usd(mp['mean_c'])} | {usd(mp['mean_t'])} | {pct(mp['rel'],1,True)} ({pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}) | p = {mp['p']:.2f} |
| Conversion rate | {pct(cv['mean_c'],2)} | {pct(cv['mean_t'],2)} | {pct(cv['rel'],1,True)} ({pct(cv['rel_lo'],1,True)} to {pct(cv['rel_hi'],1,True)}) | {cv['p_holm']:.3f} |
| Net revenue per visitor | {usd(met['Net revenue per visitor']['mean_c'])} | {usd(met['Net revenue per visitor']['mean_t'])} | {pct(met['Net revenue per visitor']['rel'],1,True)} ({pct(met['Net revenue per visitor']['rel_lo'],1,True)} to {pct(met['Net revenue per visitor']['rel_hi'],1,True)}) | {met['Net revenue per visitor']['p_holm']:.2f} |
| *Mechanism:* AOV | {usd(aov['mean_c'])} | {usd(aov['mean_t'])} | {pct(aov['rel'],1,True)} ({pct(aov['rel_lo'],1,True)} to {pct(aov['rel_hi'],1,True)}) | {aov['p_holm']:.2f} |
| *Mechanism:* net shipping cost per order | {usd(ship['mean_c'])} | {usd(ship['mean_t'])} | {pct(ship['rel'],1,True)} ({pct(ship['rel_lo'],1,True)} to {pct(ship['rel_hi'],1,True)}) | <0.001 |
| *Guardrail:* return rate per order | {pct(ret['mean_c'],1)} | {pct(ret['mean_t'],1)} | {pct(ret['rel'],1,True)} ({pct(ret['rel_lo'],1,True)} to {pct(ret['rel_hi'],1,True)}) | {ret['p_holm']:.2f} |

*Analysis population: {n_clean:,} users (see data-quality note). Adjusted p = Holm across the six non-primary metrics.*

## Decision rule check (rule in the design doc, section 1)
| Criterion | Result | Met? |
|---|---|---|
| Conversion significantly up (Holm) | {pct(cv['rel'],1,True)}, p = {cv['p_holm']:.3f} | {yes(crit_conv)} |
| Margin per visitor CI lower bound above −2% | lower bound {pct(mp['rel_lo'],1,True)} | {yes(crit_margin)} |
| Return rate not significantly up | {pct(ret['rel'],1,True)}, p = {ret['p_holm']:.2f} | {yes(crit_return)} |
| Keep-$100 trigger (margin CI upper bound below 0, conversion not up, or returns up) | margin upper bound {pct(mp['rel_hi'],1,True)} | {"Triggered" if crit_kill else "Not triggered"} |

**Outcome:** margin is inconclusive, so the rule says do not ship to everyone and run a targeted follow-up.

**Dollar impact (assumes the analysed population stays at about {an['users_per_year']/1e6:.1f}M users/year; old-app users excluded):** margin per visitor changes by {usd(an['margin_pv_diff'],3,True)} (CI {usd(an['margin_pv_lo'],3,True)} to {usd(an['margin_pv_hi'],3,True)}), i.e. **{ksign(an['annual_point'])} per year, with a range from {ksign(an['annual_lo'])} to {ksign(an['annual_hi'])}**. The interval straddles zero, so the honest reading is "unknown sign, plausible range roughly ±$0.5M".

## Why I do not trust the topline conversion number without a fix
- **Sample ratio mismatch:** the treatment arm has {srm['n_treat']:,} users vs {srm['n_ctrl']:,} in control (χ² p = {srm['p']:.0e}). The shortfall sits entirely in mobile users on an old app version, where treatment exposure logging failed mostly for people who did **not** buy.
- If ignored, conversion lift reads {pct(nai['Conversion rate']['rel'],1,True)} instead of {pct(cv['rel'],1,True)}: the bug **overstates the win by about {(nai['Conversion rate']['rel']-cv['rel'])*100:.0f} points**.
- I excluded old-app users from both arms (a pre-treatment definition, so randomisation is preserved). Results therefore describe users who are not on the old app (about {n_clean/n_all:.0%} of logged users).
- After exclusion the ratio is clean (p = {V['srm_clean']['p']:.2f}) and covariates are balanced (all |SMD| < 0.01).

## Segments (treat as hypotheses, not findings)
| Segment | Conversion | Margin per visitor | Note |
|---|---|---|---|
| New customers (pre-registered) | {pct(seg['New']['conv_rel'],1,True)} ({pct(seg['New']['conv_lo'],1,True)} to {pct(seg['New']['conv_hi'],1,True)}) | {pct(seg['New']['margin_rel'],1,True)} ({pct(seg['New']['margin_lo'],1,True)} to {pct(seg['New']['margin_hi'],1,True)}) | Conversion lift survives multiple-testing adjustment |
| Returning customers (pre-registered) | {pct(seg['Returning']['conv_rel'],1,True)} ({pct(seg['Returning']['conv_lo'],1,True)} to {pct(seg['Returning']['conv_hi'],1,True)}) | {pct(seg['Returning']['margin_rel'],1,True)} ({pct(seg['Returning']['margin_lo'],1,True)} to {pct(seg['Returning']['margin_hi'],1,True)}) | Little extra conversion, pays the shipping cost |

The pattern makes economic sense (new customers are the ones a lower threshold can win; returning customers mostly just stop paying a fee), but the formal new-vs-returning interaction test is **not significant (p = {A['interaction_tests_margin']['is_new']['p']:.2f})**. No segment's **margin** effect is significant after correction. Conversion lift is significant after correction for new customers, desktop and tablet; traffic-source cells are exploratory and none is significant.

![Segments](../figures/segments_forest.png)

## Other checks
- **Novelty:** week-1 conversion lift {pct(nov['week1']['conv']['rel'],1,True)} vs week-2 {pct(w2c['rel'],1,True)}; difference p = {nov['week1_minus_week2_conv']['p']:.2f} and trend test p = {nov['trend_test']['p']:.2f}. At this volume a decay is not detectable either way, so week-2 numbers carry the same uncertainty as the full window. Week-2 margin is {pct(w2m['rel'],1,True)} ({pct(w2m['rel_lo'],1,True)} to {pct(w2m['rel_hi'],1,True)}).
- **Estimator robustness:** raw, winsorised, bootstrap, CUPED and regression-adjusted margin estimates agree within noise; CUPED shrinks variance by only {pct(A['cuped']['var_reduction'])}.
- **Peeking:** the daily p-value for conversion first dips under 0.05 on day {int(next(r['day'] for r in K['real_path'] if r['p'] < 0.05))}. Stopping at the first significant look would have inflated false positives to {pct(K['null']['peek_daily_naive'],0)}; we ran the full 14 days.

## Risks if we shipped anyway
1. Margin could be anywhere from {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}, i.e. {ksign(an['annual_lo'])} to {ksign(an['annual_hi'])} per year on the analysed population.
2. Shipping cost per order rises {pct(ship['rel'],0)} for certain, and returns data is too sparse to rule out a rise.
3. Part of the conversion lift is likely novelty; we cannot size it from this test.
4. Results exclude old-app users, so they need a separate decision.
5. **Launch-and-watch is not a safe fallback.** The threshold is easy to reverse, but without a holdout the same noise applies and a before/after comparison is confounded by seasonality, so a margin loss of this size could go unnoticed.

## Proposed next steps
1. **Fix exposure logging** for old-app users and re-validate SRM before any new test.
2. **Follow-up test on new customers only** (the hypothesis the economics supports). To detect a 5% margin lift needs about {P['new_only']['n_per_arm_5pct']:,.0f} users per arm (about {new_days:.0f} days at current new-customer traffic), which is impractical; a **10% MDE needs about {n10:,.0f} per arm (about {new_days10:.0f} days)**. Agree with Finance first on the margin lift that would justify shipping, because that sets the sample size.
3. **Cheaper variants** to test in parallel: a $85 threshold, or a first-order-only free-shipping offer, which keep more of the margin.
4. Add a **long-run holdout** (5%) after any launch to measure the post-novelty effect.
"""
open(f"{ROOT}/memo/02_decision_memo.md", "w").write(memo)

# =============================================================== 3. RESUME + INTERVIEW
mm = M["margin"]
cm = M["conversion"]
aa = V["aa"]
tw = R["true_effects"]
rn = K["null"]
ra = K["alt_5pct"]
doc = f"""# Resume Bullets & Interview Prep (A/B Testing Project)

> **Read this first.** Everything below is from a **simulated** experiment. The word "simulated" must appear in the bullet and on the README. Do not list this project until you have run `run_all.sh` yourself and can explain every function in `src/abtools.py` without looking.
> The Criteo replication script exists but **has not been run** (network blocked in the build environment). Do not claim any Criteo result until you run it and fill in your own numbers.
> Label it "simulated" in every bullet and "built with Claude Code" in the README (the commit history shows it anyway). Whether the resume project header also carries "built with Claude Code" is your call: the project instructions say include it, your latest note says you dropped it.

## Resume bullets (every number below comes from the files in `results/`)
**Project header (suggested):** Free-Shipping Threshold A/B Test: checkout feature decision *(simulated; built with Claude Code)*

**Option A: full version (project section)**
- Framed a checkout change (free-shipping threshold $100 → $75) as a **ship / segment / keep** decision with **margin per visitor** as the decision metric and return rate as the guardrail; planned a 14-day test on {trf['total']:,} simulated users ({pct(ach['mde_margin_raw'])} MDE on margin per visitor).
- Recommended **not shipping to everyone**: conversion rose {pct(cv['rel'],1,True)} but margin per visitor was inconclusive (95% CI {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}, {ksign(an['annual_lo'])} to {ksign(an['annual_hi'])} per year); proposed a new-customer follow-up under a pre-committed decision rule.
- Caught an injected sample-ratio mismatch (χ² p = {srm['p']:.0e}) and traced it to logging failure in old-app mobile users; showed that ignoring it overstated conversion lift ({pct(nai['Conversion rate']['rel'],1)} vs {pct(cv['rel'],1)} after exclusion).
- Validated the pipeline with 1,000 A/A tests (false-positive rate {pct(aa['margin_welch']['fpr'])}–{pct(aa['conv_z']['fpr'])}) and 100 full-experiment replications ({mm['coverage_window']:.0%} CI coverage on margin per visitor); showed daily peeking inflates false positives to {pct(rn['peek_daily_naive'],1)} versus {pct(rn['obf'],1)} with calibrated sequential boundaries.

**Option B: one line (for a crowded resume)**
- Evaluated a **simulated** checkout experiment (free-shipping threshold $100 → $75; built with Claude Code): conversion {pct(cv['rel'],1,True)} but margin per visitor inconclusive, so recommended a targeted new-customer follow-up over a full rollout; validated with SRM diagnosis, 1,000 A/A tests and 100 full replications.

## 30-second pitch (open with this)
"I simulated a checkout experiment for a fashion e-commerce store: should we lower the free-shipping threshold from $100 to $75? The feature gives away margin to buy conversion, so the metric that decides it is margin per visitor, not conversion. I wrote the ship / segment / keep rule up front. Conversion rose {pct(cv['rel'],1)}, but margin per visitor was inconclusive. A 14-day test could only detect a {pct(ach['mde_margin_raw'])} change, so I recommended not shipping to everyone and testing new customers, where the economics look best. It is simulated, which let me check every method against the true effect. The code was built with Claude Code and I can walk through it."

**Do not write:** "improved conversion by X%" (nothing was shipped), or any CUPED headline number (it only reduced variance by {pct(A['cuped']['var_reduction'])}; that is a finding about CUPED's limits, not a win).

## What the ground truth said (the part no real project can show)
- True conversion lift in the clean population: **{pct(tw['clean_window']['conv_rel'],1,True)}** during the window, **{pct(tw['clean_steady']['conv_rel'],1,True)}** at steady state. Estimate: {pct(cv['rel'],1,True)} (CI covers both).
- True margin-per-visitor effect: **{pct(tw['clean_window']['margin_rel'],1,True)}** in the window, **{pct(tw['clean_steady']['margin_rel'],1,True)}** at steady state, so the "inconclusive" verdict was correct: the effect is smaller than the test could resolve.
- New customers truly gain ({pct(tw['new_steady']['margin_rel'],1,True)} steady state) and returning customers truly lose ({pct(tw['returning_steady']['margin_rel'],1,True)}), matching the memo's hypothesis. The test could not confirm it.
- The daily novelty decay was real but invisible in daily data (see `figures/novelty_vs_truth.png`).
- Monte Carlo (100 replications): raw margin estimator bias {usd(mm['bias_raw'],4,True)} per visitor against a standard deviation of {usd(mm['sd_est_raw'],3)}; the naive conversion estimate overstated the relative lift by {cm['bias_naive_rel']*100:.1f} points on average and its CI covered the truth {cm['coverage_naive']:.0%} of the time. Empirical power on conversion {cm['power_clean']:.0%}; the margin CI excluded zero in only {mm['share_ci_excludes_zero']:.0%} of replications.

## Interview questions this project prepares you for
**Why is margin per visitor the decision metric, not conversion?** The feature gives away margin to buy conversion. Conversion alone would count a $9 shipping subsidy as a win. Margin per visitor nets both sides: conversion × margin per order.

**Who decides, and what would change their mind?** VP Growth decides and Finance owns the margin definition. A result that clears the pre-committed bar (margin CI lower bound above −2%, conversion up, returns not up) would change the answer. The −2% is a tolerance I set; in a real project I would agree it with Finance before the test.

**What if margin had come back +3% with a CI of −1% to +7%?** The rule says ship: conversion is up and even the pessimistic end loses less than 2%. I would keep a 5% holdout to measure the post-novelty effect.

**Where else does this decision shape show up?** Any feature that trades unit margin for volume: delivery-fee minimums, discount depth, a longer free trial. The same three things carry over: a metric that nets both sides, a guardrail on the quality of the extra volume, and a rule set before the data.

**Why CUPED, and when does it not help?** I used pre-period spend as the covariate. It reduced variance by only {pct(A['cuped']['var_reduction'])} (corr {A['cuped']['corr']:.2f}) and CI width by {pct(A['cuped']['ci_width_reduction'])}. A purchase is a rare binary event, 40% of users are new with no history, so a user-level pre-period spend cannot explain much. CUPED pays off when the pre-period metric strongly predicts the outcome (engagement, repeat-purchase metrics).

**What did you do about SRM?** Chi-square on arm counts (p = {srm['p']:.0e}); split by pre-treatment segments to localise it (old-app mobile: {pct(V['srm_mobile_by_appold']['1']['share_treat'],1)} treated vs 50% expected, p ≈ 1e-191); showed the bias it caused; excluded the segment in both arms; re-checked SRM (p = {V['srm_clean']['p']:.2f}). Note the SMD balance check missed it (|SMD| = {abs([r for r in V['balance'] if r['covariate']=='app_old'][0]['all users']):.2f}, under the 0.1 flag), which is why SRM is checked first.

**Why not just a t-test on revenue?** Revenue per visitor is zero-inflated and heavy-tailed (CV {base['cv']:.1f}). With 200K per arm the t-test is calibrated (A/A false-positive rate {pct(aa['margin_welch']['fpr'])}), but I cross-checked with a bootstrap and a winsorised version, and the simulation-based power came in at {P['sim_power'][2]['power_sim']:.0%} where the formula said {P['sim_power'][2]['power_formula']:.0%}.

**Conversion is up but margin is flat. What do you recommend?** Do not ship globally: net shipping cost per order rose {pct(ship['rel'],0,True)} (the price of the conversion lift) and the margin CI's lower bound ({pct(mp['rel_lo'],1,True)}) is below my pre-committed −2% tolerance. Test the segment where the economics work (new customers), sized from a margin lift Finance agrees would justify the change.

**Peeking?** {pct(rn['peek_daily_naive'],0)} false positives with daily looks under the null. Fix: pre-specified looks with alpha spending. I calibrated Pocock and O'Brien–Fleming-style boundaries by simulation; OBF kept {pct(ra['obf'],0)} power vs {pct(ra['fixed_day14'],0)} fixed and it stops on average at day {ra['avg_stop_day_obf']:.1f} of 14 when the true lift is 5% (day {rn['avg_stop_day_obf']:.1f} under H0, i.e. it rarely stops early when there is no effect).

**Network effects or interference?** Not modelled here: users are independent by construction. In a marketplace (shared inventory, delivery capacity) this would break SUTVA, and I would look at cluster randomisation or switchback designs.

**What would you do differently?** Extend to 25 days (or 4 weeks) for the margin metric, log pre-period margin and visits for stronger covariates, and add a long-run holdout to measure novelty decay.

## Before you list this
- [ ] Run `bash run_all.sh` yourself and confirm the numbers match.
- [ ] Be able to explain, unprompted: why the cap is from pilot data, why theta is estimated on pooled data, why Holm for secondary but not primary.
- [ ] Say feature, metric and decision in one breath (see the 30-second pitch).
- [ ] Decide whether the resume project header carries "built with Claude Code" (the README keeps it either way).
- [ ] Put "simulated" in the bullet. If an interviewer asks "is this real?", say no in the first sentence, then explain why simulation lets you verify your methods against known truth.
- [ ] Run `src/05_criteo_replication.py` locally; add a Criteo bullet only with your own output.
"""
open(f"{ROOT}/memo/03_resume_and_interview.md", "w").write(doc)
print("docs written")

# =============================================================== 4. README
readme = f"""# A/B Testing Case Study: Free-Shipping Threshold ($100 → $75)

## Product brief
| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: $100 → $75, for a fashion e-commerce store |
| **Decision** | Roll out to everyone, roll out to a segment, or keep $100 (VP Growth decides; Finance signs off on margin) |
| **Decision metric** | Margin per visitor = conversion × margin per order. Guardrail: return rate |
| **Decision rule** | Ship only if conversion is significantly up, the margin CI's lower bound is above −2%, and returns are not significantly up (`memo/01_experiment_design.md`) |
| **Outcome** | Conversion {pct(cv['rel'],1,True)}, margin per visitor inconclusive ({pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}), so do not ship to everyone; test new customers |

An end-to-end experimentation project on **simulated** data with a **known ground truth**: experiment design, power analysis, validity checks, layered analysis, sequential-testing simulation, 100-replication validation, and a decision memo. Because the true effect is known, every method is checked against the right answer.

> **Everything here is simulated, and the code was built with Claude Code.** Numbers are from `data_gen/generate_experiment.py` (fixed seed). A Criteo replication script is included (`src/05_criteo_replication.py`) but has **not been run**; no real-data result is claimed.

## Headline result
| Question | Answer |
|---|---|
| Does $75 raise conversion? | Yes: **{pct(cv['rel'],1,True)}** (95% CI {pct(cv['rel_lo'],1,True)} to {pct(cv['rel_hi'],1,True)}) |
| Does it raise profit per visitor? | **Unknown**: {pct(mp['rel'],1,True)} (CI {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}); the test can only detect ≥{pct(ach['mde_margin_raw'])} |
| What does it cost? | Net shipping cost per order **{pct(ship['rel'],0,True)}** |
| Recommendation | Do not ship globally; fix logging; run a follow-up on new customers (`memo/02_decision_memo.md`) |

## What the project demonstrates
| Skill | Where | Result |
|---|---|---|
| Experiment design, metric hierarchy, MDE | `memo/01_experiment_design.md` | 14 days is under-powered for margin ({pct(ach['mde_margin_raw'])} MDE); the design says so up front |
| Power analysis (formula + simulation) | `src/01_power.py` | Formula {P['sim_power'][2]['power_formula']:.0%} vs bootstrap simulation {P['sim_power'][2]['power_sim']:.0%} at the same effect (heavy tails) |
| **SRM diagnosis** | `src/02_validity.py` | χ² p = {srm['p']:.0e}, localised to old-app mobile; naive lift {pct(nai['Conversion rate']['rel'],1,True)} vs {pct(cv['rel'],1,True)} after exclusion |
| Covariate balance | `src/02_validity.py` | SMD missed the bug (|SMD| = 0.09 < 0.1); SRM caught it |
| A/A validation | `src/02_validity.py` | False-positive rate over 1,000 splits: margin/Welch {pct(aa['margin_welch']['fpr'])}, conversion {pct(aa['conv_z']['fpr'])}, margin/CUPED {pct(aa['margin_cuped']['fpr'])} |
| Heavy-tail robustness | `src/03_analysis.py` | Welch, winsorised, bootstrap agree |
| **CUPED / regression adjustment** | `src/03_analysis.py` | Only {pct(A['cuped']['var_reduction'])} variance reduction: documented as a limit, not a win |
| Ratio metrics (delta method) | `src/abtools.py` | AOV CI matches the user-level bootstrap |
| Multiple testing | `src/03_analysis.py` | Holm (secondary metrics), BH (segments) |
| Heterogeneity | `src/03_analysis.py` | New customers look best, but interaction p = {A['interaction_tests_margin']['is_new']['p']:.2f}: a hypothesis, not a finding |
| Novelty | `src/03_analysis.py` | Week-1 vs week-2 p = {nov['week1_minus_week2_conv']['p']:.2f}: not resolvable at this volume |
| **Peeking / sequential testing** | `src/04_peeking.py` | Daily looks: {pct(rn['peek_daily_naive'],1)} false positives; Pocock {pct(rn['pocock'],1)}, OBF-style {pct(rn['obf'],1)} |
| Ground-truth validation | `src/06`, `src/07` | {M['n_reps']} replications: margin CI coverage {mm['coverage_window']:.0%}; naive analysis covers the truth {cm['coverage_naive']:.0%} |

## What the ground truth revealed
- True conversion lift {pct(tw['clean_window']['conv_rel'],1,True)} (window) / {pct(tw['clean_steady']['conv_rel'],1,True)} (steady state); estimate {pct(cv['rel'],1,True)}.
- True margin effect {pct(tw['clean_window']['margin_rel'],1,True)} (window) / {pct(tw['clean_steady']['margin_rel'],1,True)} (steady state), so the inconclusive verdict was right.
- New customers truly gain ({pct(tw['new_steady']['margin_rel'],1,True)}), returning customers truly lose ({pct(tw['returning_steady']['margin_rel'],1,True)}), but the test lacked the power to confirm it.

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
"""
open(f"{ROOT}/README.md", "w").write(readme)
print("README written")
