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
BROAD = "ship" if (crit_conv and crit_margin and crit_return) else ("keep" if crit_kill else "follow-up")
# tier 2: pre-registered segments that earn a targeted follow-up (conversion up after BH correction, net-positive margin not ruled out)
TARGETS = [r["segment"] for r in A["segments"] if r["status"] == "pre-registered" and r["conv_p_bh"] < 0.05 and r["conv_rel"] > 0 and r["margin_hi"] > 0]
assert BROAD == "keep" and TARGETS == ["New"], "results changed: the memo recommendation text must be rewritten"
share_new = P["new_only"]["share_new"]
max_smd = max(abs(r["excl. old-app"]) for r in V["balance"])
srm_app = V["srm_by_appold"]["1"]
yes = lambda b: "Yes" if b else "**No**"

# =============================================================== 1. DESIGN DOC
design = f"""# Experiment Design: Free-Shipping Threshold $100 → $75

*Status: version 2 (semi-synthetic). The customers, order history and control outcomes are **real H&M data**; the treatment effect, returns and the logging fault are **simulated** (see the README). Planning inputs come from a separate real sample of {base['pilot_n']:,} customers in the 14 days before the test window.*

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
| **Keep $100 for everyone** | Margin CI upper bound below 0, **or** conversion not significantly up (Holm), **or** return rate significantly up | Do not roll out broadly |
| **Targeted follow-up (segment tier)** | Applied on top of the rows above: a pre-registered segment whose conversion lift is significant after Benjamini–Hochberg correction across segments **and** whose margin CI upper bound is above 0 | Run a follow-up test in that segment only, sized for the margin lift Finance requires. A segment is never launched on this result alone |

The −2% is the margin loss Growth would tolerate in exchange for new-customer growth. In a real project I would agree it with Finance before the test.

*Process note: version 1 used fully simulated data and ended "margin inconclusive; follow up with new customers". A reviewer asked for real data and for a segment-level decision. In version 2 the segment tier above was added in response to that feedback, **after** I had seen the overall version-2 results, so it is a response to feedback and not a pre-registration. "New vs returning" was fixed as a pre-registered segment in version 1, before any data; typical order size and age were added in version 2 before the segment analysis was run. Because I also wrote the simulator's assumed responders, the segment analysis tests whether the methods separate real from null segments; it does not discover who responds in the real world.*

## 2. Design
| Item | Choice | Why |
|---|---|---|
| Randomisation unit | User (assigned on first visit in the window) | Avoids one person seeing both prices |
| Allocation | 50 / 50 | Maximises power |
| Window | 14 days | Covers two full weekly cycles; business cap on run time |
| Traffic | about {trf['per_day']:,.0f} users/day, {trf['total']:,} total, {trf['per_arm']:,} per arm | Planning assumption |
| Exposure logging | Client-side tag | **Known risk:** tag failures would cause SRM, so SRM is checked first. The fault in this project is injected (simulated) to test that check |
| Real vs simulated | Real: customers, history, control orders. Simulated: treatment effect, returns | No public dataset contains a shipping-threshold experiment |

## 3. Metrics
| Role | Metric | Definition |
|---|---|---|
| **Primary** | Margin per visitor | 42% × merchandise value after returns − net shipping cost − $6 per return, summed per assigned user (zeros included) |
| Driver (secondary) | Conversion rate; net revenue per visitor | Did the feature do what it was built to do? |
| Mechanism (diagnostic) | AOV; items per order; net shipping cost per order | Explain *why* margin moved. Shipping cost rises by construction, so it cannot be a guardrail |
| Guardrail | Return rate per order | Extra buyers may return more. Page-load time is not simulated |
| Pre-registered segments | New vs returning; typical order size (pre-period average order); age band | New customers and shoppers whose usual order sits just under the threshold are the ones a lower threshold can plausibly win |
| Exploratory segments | Club status; fashion-news subscription | Reported but flagged exploratory |

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
Covariate adjustment helps little ({pct(base['cuped_var_reduction_prespend_only'])} variance reduction in the pilot) because a purchase is a rare event and {share_new:.0%} of users are new with no history.

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


def segrow(k_):
    r_ = seg[k_]
    return f"| {k_} | {pct(r_['conv_rel'],1,True)} ({pct(r_['conv_lo'],1,True)} to {pct(r_['conv_hi'],1,True)}) | {pct(r_['margin_rel'],1,True)} ({pct(r_['margin_lo'],1,True)} to {pct(r_['margin_hi'],1,True)}) | {('<0.001' if r_['conv_p_bh'] < 0.001 else format(r_['conv_p_bh'], '.3f'))} |"


share_new = seg["New"]["n"] / A["population"]["n_clean"]   # share of the analysed sample (the design doc above used the pilot share)
segrows = "\n".join(segrow(k_) for k_, r_ in seg.items() if r_["status"] == "pre-registered")
sig_raw = [r_ for r_ in seg.values() if r_["margin_p"] < 0.05]
n_cells = len(seg)
seg_text = (
    f"The pattern makes economic sense: customers with no history are the ones a lower threshold can win, while returning customers mostly stop paying a fee. "
    f"But the formal new-vs-returning interaction test for margin is **not significant (p = {A['interaction_tests_margin']['is_new']['p']:.2f})**, and no segment's **margin** effect is significant after correction. "
    + (f"One cell looks significant before correction ({sig_raw[0]['segment']}: margin {pct(sig_raw[0]['margin_rel'],1,True)}, raw p = {sig_raw[0]['margin_p']:.3f}) and does not survive Benjamini–Hochberg (p = {sig_raw[0]['margin_p_bh']:.2f}). "
       f"With {n_cells} segment cells, about {n_cells*0.05:.1f} raw p-values below 0.05 are expected by chance alone, so a single raw hit is weak evidence and segment margin results are hypotheses to validate, not findings. " if sig_raw else "")
    + "Club status and fashion-news cells are exploratory."
)

# =============================================================== 2. DECISION MEMO
n_clean, n_all = A["population"]["n_clean"], A["population"]["n_all"]
w2m, w2c = nov["week2"]["margin"], nov["week2"]["conv"]
memo = f"""# Decision Memo: Free-Shipping Threshold $100 → $75

**To:** Product Lead, VP Growth  **From:** Data Science  **Data:** 14-day experiment on real H&M customers and orders with a **simulated** treatment effect, {n_all:,} customers logged

## The decision
| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: $100 → $75 |
| **Decision** | Roll out to everyone, roll out to a segment, or keep $100 (VP Growth decides, Finance signs off on margin) |
| **Decision metric** | Margin per visitor (conversion × margin per order). Guardrail: return rate |

## Recommendation
**Keep $100 for the general customer base, and run a targeted follow-up test for new customers.** Across all customers the $75 threshold raises shipping cost per order by {pct(ship['rel'],0)} for certain, but conversion rises only {pct(cv['rel'],1,True)} (95% CI {pct(cv['rel_lo'],1,True)} to {pct(cv['rel_hi'],1,True)}; p = {cv['p']:.3f} unadjusted, {cv['p_holm']:.2f} after Holm correction) and **margin per visitor is not shown to hold**: {pct(mp['rel'],1,True)} with a CI of {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}. The exception is **new customers** (no online purchase in the prior 13 months, {share_new:.0%} of customers): conversion is {pct(seg['New']['conv_rel'],1,True)} there and survives multiple-testing correction, and the margin estimate is {pct(seg['New']['margin_rel'],1,True)} (CI {pct(seg['New']['margin_lo'],1,True)} to {pct(seg['New']['margin_hi'],1,True)}), so a net-positive margin is plausible but unproven. Fix the exposure-logging bug first, then validate in that segment (below).

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

**Outcome (all customers):** conversion does not survive the Holm correction and margin is inconclusive, so the rule says **keep $100 for everyone**. **Segment tier:** only {", ".join(TARGETS).lower()} customers qualify for a targeted follow-up (conversion lift significant after BH correction and a margin gain not ruled out).

**Dollar impact (assumes the analysed population stays at about {an['users_per_year']/1e6:.1f}M users/year; old-app users excluded):** margin per visitor changes by {usd(an['margin_pv_diff'],3,True)} (CI {usd(an['margin_pv_lo'],3,True)} to {usd(an['margin_pv_hi'],3,True)}), i.e. **{ksign(an['annual_point'])} per year, with a range from {ksign(an['annual_lo'])} to {ksign(an['annual_hi'])}**. The interval straddles zero, so the honest reading is "unknown sign, plausible range from {ksign(an['annual_lo'])} to {ksign(an['annual_hi'])}".

## Why I do not trust the topline conversion number without a fix
- **Sample ratio mismatch:** the treatment arm has {srm['n_treat']:,} users vs {srm['n_ctrl']:,} in control (χ² p = {srm['p']:.0e}). The shortfall sits entirely in users flagged as on an old app version ({pct(srm_app['share_treat'],1)} of that group landed in treatment vs 50% expected), where treatment exposure logging failed mostly for people who did **not** buy. This fault is injected (simulated) on purpose, to test that the pipeline catches it.
- If ignored, conversion lift reads {pct(nai['Conversion rate']['rel'],1,True)} instead of {pct(cv['rel'],1,True)}: the bug **overstates the win by about {(nai['Conversion rate']['rel']-cv['rel'])*100:.0f} points**.
- I excluded old-app users from both arms (a pre-treatment definition, so randomisation is preserved). Results therefore describe users who are not on the old app (about {n_clean/n_all:.0%} of logged users).
- After exclusion the ratio is clean (p = {V['srm_clean']['p']:.2f}) and covariates are balanced (all |SMD| < {max_smd:.2f}).

## Segments (treat as hypotheses, not findings)
| Segment (pre-registered) | Conversion | Margin per visitor | Conversion p (BH) |
|---|---|---|---|
{segrows}

{seg_text}

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
2. **Follow-up test on new customers only** (the one segment that passed the segment tier). To detect a 5% margin lift needs about {P['new_only']['n_per_arm_5pct']:,.0f} users per arm (about {new_days:.0f} days at current new-customer traffic), which is impractical; a **10% MDE needs about {n10:,.0f} per arm (about {new_days10:.0f} days)**. The {pct(seg['New']['margin_rel'],0,True)} estimate comes from the best-looking segment, so expect it to shrink in the follow-up (winner's curse). Agree with Finance first on the margin lift that would justify launching, because that sets the sample size.
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

> **Read this first.** Everything below uses **real H&M customers and orders with a simulated treatment effect** (semi-synthetic). Say "semi-synthetic" or "simulated treatment effect" in every bullet and on the README. Do not list this project until you have run the pipeline yourself and can explain every function in `src/abtools.py` without looking.
> Never describe a result as a measured business effect: nothing was shipped and the treatment effect is an assumption. Whether the resume project header also carries "built with Claude Code" is your call; the README keeps it either way.

## Resume bullets (every number below comes from the files in `results/`)
**Project header (suggested):** Free-Shipping Threshold A/B Test: checkout feature decision *(semi-synthetic: real H&M customers, simulated treatment effect; built with Claude Code)*

**Option A: full version (project section)**
- Framed a checkout change (free-shipping threshold $100 → $75) as a **ship / segment / keep** decision with **margin per visitor** as the decision metric and return rate as the guardrail; planned a 14-day test on {trf['total']:,} real H&M customers with a simulated treatment effect ({pct(ach['mde_margin_raw'])} MDE on margin per visitor).
- Recommended **keeping $100 for the general base** (conversion {pct(cv['rel'],1,True)}, not significant after correction; margin per visitor inconclusive, 95% CI {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}) and a **targeted follow-up test for new customers** (conversion {pct(seg['New']['conv_rel'],1,True)} after multiple-testing correction), using a two-tier decision rule.
- Caught an injected sample-ratio mismatch (χ² p = {srm['p']:.0e}) and traced it to the injected logging fault in old-app users; showed that ignoring it overstated conversion lift ({pct(nai['Conversion rate']['rel'],1)} vs {pct(cv['rel'],1)} after exclusion).
- Validated the pipeline with 1,000 A/A tests (false-positive rate {pct(aa['margin_welch']['fpr'])}–{pct(aa['conv_z']['fpr'])}) and 100 full-experiment replications ({mm['coverage_window']:.0%} CI coverage on margin per visitor); showed daily peeking inflates false positives to {pct(rn['peek_daily_naive'],1)} versus {pct(rn['obf'],1)} with calibrated sequential boundaries.

**Option B: one line (for a crowded resume)**
- Evaluated a **semi-synthetic** checkout experiment on real H&M customers (free-shipping threshold $100 → $75; simulated treatment effect; built with Claude Code): margin per visitor inconclusive overall, so recommended a targeted new-customer follow-up over a full rollout; validated with SRM diagnosis, 1,000 A/A tests and 100 full replications.

## 30-second pitch (open with this)
"I built a semi-synthetic checkout experiment for a fashion e-commerce store: should we lower the free-shipping threshold from $100 to $75? The customers, order history and baseline purchases are real H&M data; the treatment effect is simulated because no public dataset has a shipping-threshold test. The feature gives away margin to buy conversion, so the metric that decides it is margin per visitor, not conversion. Overall, margin was inconclusive and conversion did not survive correction, so the rule says keep $100 for everyone, but new customers pass the segment tier, so I recommended a targeted follow-up test there. Because the effect is simulated I could check every method against the true effect. The code was built with Claude Code and I can walk through it."

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

**Why CUPED, and when does it not help?** I used pre-period spend as the covariate. It reduced variance by only {pct(A['cuped']['var_reduction'])} (corr {A['cuped']['corr']:.2f}) and CI width by {pct(A['cuped']['ci_width_reduction'])}. A purchase is a rare binary event, {share_new:.0%} of users are new with no history, so a user-level pre-period spend cannot explain much. CUPED pays off when the pre-period metric strongly predicts the outcome (engagement, repeat-purchase metrics).

**What did you do about SRM?** Chi-square on arm counts (p = {srm['p']:.0e}); split by pre-treatment segments to localise it (old-app users: {pct(srm_app['share_treat'],1)} treated vs 50% expected, p = {srm_app['p']:.0e}); showed the bias it caused; excluded the segment in both arms; re-checked SRM (p = {V['srm_clean']['p']:.2f}). Note the SMD balance check missed it (|SMD| = {abs([r for r in V['balance'] if r['covariate']=='app_old'][0]['all users']):.2f}, under the 0.1 flag), which is why SRM is checked first.

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
- [ ] Put "semi-synthetic" or "simulated treatment effect" in the bullet. If an interviewer asks "is this real?", say: the customers and baseline orders are real H&M data, the treatment effect is simulated, and that is why I can check every method against known truth.
"""
open(f"{ROOT}/memo/03_resume_and_interview.md", "w").write(doc)
print("docs written")

# =============================================================== 4. README
readme = f"""# Free-Shipping Threshold A/B Test

**Should a fashion e-commerce store lower its free-shipping threshold from $100 to $75?** A semi-synthetic experiment on **real H&M customers and orders**, analysed end to end, finishing with a ship / segment / keep decision.

> **Semi-synthetic, built with Claude Code.** The customers, their order history and what they actually bought in a 14-day window are **real** ([H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) on Kaggle). The **treatment effect, returns and the logging fault are simulated**, because no public dataset contains a shipping-threshold experiment. That choice has a purpose: the true effect is known, so every method below is checked against the right answer. No business result is claimed.

## The decision

| | |
|---|---|
| **Feature** | Free-shipping threshold at checkout: $100 → $75, for every order |
| **Decision** | Roll out to everyone, run a targeted follow-up in a segment, or keep $100 (VP Growth decides; Finance signs off on margin) |
| **Decision metric** | Margin per visitor = conversion × margin per order. Guardrail: return rate |
| **Decision rule** | Tier 1 (everyone): ship only if conversion is significantly up, the margin CI's lower bound is above −2%, and returns are not significantly up. Tier 2 (segments): a pre-registered segment earns a targeted follow-up if its conversion lift survives multiple-testing correction and a margin gain is not ruled out |

The feature gives away margin to buy conversion, so conversion alone would count a shipping subsidy as a win. Margin per visitor nets both sides.

## What I found

| | Result |
|---|---|
| Conversion, all customers | **{pct(cv['rel'],1,True)}** (95% CI {pct(cv['rel_lo'],1,True)} to {pct(cv['rel_hi'],1,True)}); not significant after Holm correction (p = {cv['p_holm']:.2f}) |
| Margin per visitor (decision metric) | **Inconclusive:** {pct(mp['rel'],1,True)} (CI {pct(mp['rel_lo'],1,True)} to {pct(mp['rel_hi'],1,True)}), or {ksign(an['annual_lo'])} to {ksign(an['annual_hi'])} per year |
| Net shipping cost per order | {pct(ship['rel'],0,True)} |
| New customers (no online purchase in the prior 13 months) | Conversion **{pct(seg['New']['conv_rel'],1,True)}** (survives correction); margin {pct(seg['New']['margin_rel'],1,True)} (CI {pct(seg['New']['margin_lo'],1,True)} to {pct(seg['New']['margin_hi'],1,True)}) |
| Recommendation | **Keep $100 for the general customer base. Run a targeted follow-up test for new customers**, sized for the margin lift Finance requires |

A 14-day test can only detect a {pct(ach['mde_margin_raw'])} change in margin per visitor, so an inconclusive overall margin result was the expected outcome, and the design doc says so up front. The ground truth agrees: the true overall margin effect is {pct(tw['clean_window']['margin_rel'],1,True)} during the test and {pct(tw['clean_steady']['margin_rel'],1,True)} at steady state, smaller than the test could resolve, while new customers truly gain ({pct(tw['new_steady']['margin_rel'],1,True)} at steady state) and returning customers truly lose ({pct(tw['returning_steady']['margin_rel'],1,True)}).

![Segment effects](figures/segments_forest.png)

{seg_text} The new-vs-returning margin interaction test is not significant (p = {A['interaction_tests_margin']['is_new']['p']:.2f}), so the new-customer result is a hypothesis for the follow-up, not a finding.

## What is in the analysis

| Step | What it shows | Result |
|---|---|---|
| Design and power | Metric tree, decision rule, minimum detectable effect | {pct(ach['mde_margin_raw'])} MDE on margin per visitor; the bootstrap power check gives {P['sim_power'][2]['power_sim']:.0%} where the formula says {P['sim_power'][2]['power_formula']:.0%} (heavy tails) |
| **Sample ratio mismatch** | An injected logging fault, found before looking at effects | χ² p = {srm['p']:.0e}; traced to the old-app group. Ignoring it gives a {pct(nai['Conversion rate']['rel'],1,True)} conversion lift instead of {pct(cv['rel'],1,True)} |
| Pipeline validation | 1,000 A/A tests on control-only customers | False-positive rate {pct(aa['margin_welch']['fpr'])} to {pct(aa['conv_z']['fpr'])} |
| Heavy-tailed outcome | Welch, winsorised and bootstrap estimates | Agree within noise |
| **Variance reduction** | CUPED and regression adjustment | Only {pct(A['cuped']['var_reduction'])} variance reduction: purchases are rare and {share_new:.0%} of customers are new with no history |
| Multiple testing | Holm for the six non-primary metrics, BH for segments | One segment margin cell looks significant before correction and does not survive it |
| **Peeking** | Daily looks vs corrected sequential boundaries | {pct(rn['peek_daily_naive'],1)} false positives with daily looks vs {pct(rn['obf'],1)} with O'Brien–Fleming-style boundaries ({pct(ra['obf'],0)} power vs {pct(ra['fixed_day14'],0)} fixed) |
| **Ground-truth check** | 100 full replications of the experiment | {mm['coverage_window']:.0%} CI coverage on margin per visitor; the naive analysis covers the truth {cm['coverage_naive']:.0%} of the time |

![Peeking](figures/peeking.png)

![Replications](figures/montecarlo_coverage.png)

## Read this next

- [`memo/02_decision_memo.pdf`](memo/02_decision_memo.pdf): the recommendation, the decision-rule check, dollar impact and next steps
- [`memo/01_experiment_design.pdf`](memo/01_experiment_design.pdf): the product brief, metrics, power analysis and analysis plan
- [`notebooks/ab_test_walkthrough.ipynb`](notebooks/ab_test_walkthrough.ipynb): an executed walkthrough of the key results

## Reproduce

Tested on Python 3.13. The H&M files are not in this repository (Kaggle competition data, not mine to redistribute), so there is a one-time preparation step.

```
pip install -r requirements.txt

# 1. Download customers.csv and transactions_train.csv from the Kaggle competition page
#    (free account; accept the competition rules), then build the small per-customer tables (~1 minute):
unzip -p transactions_train.csv.zip | python data_gen/prepare_hm_base.py --customers customers.csv --out data/hm_base

# 2. Run everything
bash run_all.sh --skip-montecarlo     # about a minute
bash run_all.sh                       # adds the 100-replication run, about 10 minutes
```

Seeds are fixed, so the numbers reproduce exactly. The PDFs need `pandoc` and `wkhtmltopdf`; without them that step is skipped.

## How the data is built

`data_gen/prepare_hm_base.py` turns the raw H&M files into one row per customer: age, club-member status, fashion-news subscription, 90-day pre-period online spend and order count, whether the customer has ever bought online, and what they actually bought online in a 14-day window (basket value, items, day). An order is one customer's online purchases on one day (`sales_channel_id` = 2, the usual reading of the dataset).

`data_gen/generate_experiment.py` then samples customers, randomises them, and simulates the treatment on top of the real control outcome, generating both potential outcomes for every customer so the true effect is known:

| Real (H&M) | Simulated (assumptions) |
|---|---|
| Customers, age, club status, news subscription | Who responds: no-history customers most; among returning customers, those whose usual order is $60–$100; none among $100+. Age, club and news have no true effect, so they act as negative controls |
| Order history, pre-period spend, new vs returning | Extra buyers, basket padding up to $75, a novelty effect that decays over about 3 days |
| Who bought in the window, basket value, items, day | Returns, the cost model (42% margin, $9.00 carrier cost, $7.50 fee under the threshold, $6 per return) |
| Heavy-tailed real baskets and real daily traffic | An injected exposure-logging fault (about 9% of customers flagged old-app, treatment arm only, mostly non-buyers) |

The pilot sample used for planning is the real 14-day period before the test window, control experience only.

## Repository layout

```
data_gen/prepare_hm_base.py       stage 1: real H&M files -> small per-customer tables (run once)
data_gen/generate_experiment.py   stage 2: randomise, simulate the treatment, write the hidden truth
data_gen/legacy_fully_synthetic.py  version 1 generator (fully simulated), kept for reference
src/abtools.py                    toolkit: SRM, SMD, Welch, bootstrap, CUPED, delta method, sequential boundaries
src/01_power.py ... 08_build_docs.py   one script per step of the analysis
extras/                           optional Criteo replication script (not run, not part of the results)
notebooks/                        executed walkthrough
memo/                             design doc and decision memo (md + PDF)
figures/  results/                every chart and every number cited
run_all.sh                        reproduces everything
```

## Limitations

- **The treatment effect is an assumption.** The control data is real, but no real shipping-threshold experiment sits underneath it. The segment analysis therefore tests whether the methods separate responsive from null segments that I defined in the simulator; it does not discover who responds in the real world.
- H&M prices are anonymised, so order values are rescaled so the median order is $62 (the $75 and $100 thresholds land at fixed points of the real basket distribution). I assumed the store ran a $100 threshold during the window and that real baskets already contain any padding it caused.
- "Visitors" are all customers in the file, including people who did not buy in the window; true site traffic is not in the data. Postal codes are hashed, so there is no regional segment. "New" means no online purchase since the data starts (about 13 months), which mixes first-time and lapsed customers.
- Returns are not in the dataset and are simulated. Customers are independent: no interference, network effects or long-run behaviour change.
- The cost model and the decision-rule thresholds (−2% margin tolerance, return-rate wording) are my choices; in a real project they would be agreed with Finance before the test.
- Sequential boundaries assume equal information per day; a production design would use alpha-spending on actual information fractions.
"""
open(f"{ROOT}/README.md", "w").write(readme)
print("README written")
