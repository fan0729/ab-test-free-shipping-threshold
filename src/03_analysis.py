"""Phase 5: analysis (layered).  Run AFTER 02_validity.py.

Population decision (from validity checks): the logging bug makes old-app mobile users unusable,
so the primary analysis excludes app_old == 1 in BOTH arms (a pre-treatment definition).
The naive all-user result is reported only to show the size of the bias.
"""
import json, os, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from plotting import BLUE, GRAY, ORANGE, INK, MUTED, AQUA
from abtools import (compare, welch, bootstrap_diff, cuped_adjust, regression_adjust, adjust_pvalues, two_prop_z)

ROOT = os.path.join(os.path.dirname(__file__), "..")
power = json.load(open(f"{ROOT}/results/power.json"))
CAP = power["baseline"]["winsor_cap"]           # pre-specified from pilot data
df_all = pd.read_csv(f"{ROOT}/data/experiment.csv.gz")
df = df_all[df_all.app_old == 0].reset_index(drop=True)
df["margin_w"] = df.margin.clip(upper=CAP)       # winsorised primary (sensitivity)
out = {"population": dict(n_all=len(df_all), n_clean=len(df), cap=CAP)}


def split(d):
    return d[d.variant == 1], d[d.variant == 0]


def metric_table(d):
    t, c = split(d)
    spec = {  # name: (num col, den col, role)
        "Margin per visitor": ("margin", None, "primary"),
        "Conversion rate": ("purchased", None, "secondary"),
        "Net revenue per visitor": ("net_rev", None, "secondary"),
        "AOV (basket $ per order)": ("basket", "purchased", "mechanism"),
        "Items per order": ("items", "purchased", "mechanism"),
        "Return rate (per order)": ("returned", "purchased", "guardrail"),
        "Net shipping cost per order": ("ship_cost", "purchased", "mechanism"),
    }
    rows = []
    for name, (num, den, role) in spec.items():
        r = compare(t[num].values, c[num].values, t[den].values if den else None, c[den].values if den else None)
        rows.append(dict(metric=name, role=role, **r))
    tab = pd.DataFrame(rows)
    fam = tab.role != "primary"           # primary tested alone at alpha=0.05; the rest form one family
    tab["p_holm"] = np.nan
    tab["p_bh"] = np.nan
    tab.loc[fam, "p_holm"] = adjust_pvalues(tab.loc[fam, "p"].values, "holm")
    tab.loc[fam, "p_bh"] = adjust_pvalues(tab.loc[fam, "p"].values, "fdr_bh")
    return tab


# ---- A. naive (all logged users, bug ignored) vs clean population
naive = metric_table(df_all)
final = metric_table(df)
out["naive_metrics"] = naive.to_dict("records")
out["metrics"] = final.to_dict("records")

# ---- B. margin per visitor: five ways to compute the same effect
t, c = split(df)
base = c.margin.mean()
est = {}
est["Welch t-test (raw)"] = welch(t.margin.values, c.margin.values)
est["Welch, winsorised at pilot P99.95"] = welch(t.margin_w.values, c.margin_w.values)
bs = bootstrap_diff(t.margin.values, c.margin.values, B=2000, seed=3)
est["Bootstrap percentile CI (2,000 reps)"] = bs
yadj, cdiag = cuped_adjust(df.margin.values, df.pre_spend_90d.values)
df["margin_cuped"] = yadj
tt, cc = split(df)
est["CUPED (pre-period spend)"] = welch(tt.margin_cuped.values, cc.margin_cuped.values)
est["Regression adj. (pre-spend, is_new, device; interacted)"] = regression_adjust(
    df.assign(mobile=(df.device == "mobile").astype(int), desktop=(df.device == "desktop").astype(int)),
    "margin", covars=("pre_spend_90d", "is_new", "mobile", "desktop"))
for k, v in est.items():
    v["rel"], v["rel_lo"], v["rel_hi"] = v["diff"] / base, v["lo"] / base, v["hi"] / base
    v["width_vs_raw"] = (v["hi"] - v["lo"]) / (est["Welch t-test (raw)"]["hi"] - est["Welch t-test (raw)"]["lo"])
out["margin_methods"] = est
out["cuped"] = dict(**cdiag, ci_width_reduction=float(1 - est["CUPED (pre-period spend)"]["width_vs_raw"]))
# where CUPED does help: returning customers only (new customers have no history)
ret = df[df.is_new == 0]
_, rdiag = cuped_adjust(ret.margin.values, ret.pre_spend_90d.values)
out["cuped_returning_only"] = rdiag
out["cuped_new_only_corr"] = float(np.corrcoef(df[df.is_new == 1].margin, df[df.is_new == 1].pre_spend_90d)[0, 1]) if df[df.is_new == 1].pre_spend_90d.std() > 0 else 0.0

fig, ax = plt.subplots(figsize=(8, 3.8))
names = list(est.keys())[::-1]
for i, k in enumerate(names):
    v = est[k]
    ax.plot([v["rel_lo"] * 100, v["rel_hi"] * 100], [i, i], color=BLUE, lw=2)
    ax.scatter([v["rel"] * 100], [i], color=BLUE, s=28, zorder=3)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Effect on margin per visitor (% of control mean), 95% CI")
ax.set_title("Five estimators, same answer: intervals barely shrink")
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/estimator_comparison.png"); plt.close(fig)

# ---- C. delta method vs bootstrap on a ratio metric (AOV)
rng = np.random.default_rng(5)
def aov_boot(B=1000):
    tb, cb = t[t.purchased == 1], c[c.purchased == 1]
    # bootstrap USERS (not orders): resample all users, recompute ratio; sparse trick on user table
    nt, nc = len(t), len(c)
    out_d = []
    for _ in range(B):
        it = rng.integers(0, nt, nt); ic = rng.integers(0, nc, nc)
        a1 = t.basket.values[it].sum() / t.purchased.values[it].sum()
        a0 = c.basket.values[ic].sum() / c.purchased.values[ic].sum()
        out_d.append(a1 - a0)
    return np.percentile(out_d, [2.5, 97.5]).tolist()
aov_row = final[final.metric.str.startswith("AOV")].iloc[0]
out["aov_delta_vs_boot"] = dict(delta_ci=[aov_row.lo, aov_row.hi], boot_ci=aov_boot(500))

# ---- D. heterogeneity
segs = [("New vs returning", "is_new", {1: "New", 0: "Returning"}, "pre-registered"),
        ("Device", "device", {k: k.title() for k in ["mobile", "desktop", "tablet"]}, "pre-registered"),
        ("Traffic source", "source", {k: k.title() for k in ["organic", "paid", "email", "direct"]}, "exploratory")]
rows = []
for fam, col, labels, status in segs:
    for v, lab in labels.items():
        g = df[df[col] == v]
        gt, gc = split(g)
        cm = compare(gt.margin.values, gc.margin.values)
        cv = compare(gt.purchased.values, gc.purchased.values)
        rows.append(dict(family=fam, segment=lab, status=status, n=len(g),
                         margin_rel=cm["rel"], margin_lo=cm["rel_lo"], margin_hi=cm["rel_hi"], margin_p=cm["p"], margin_diff=cm["diff"],
                         conv_rel=cv["rel"], conv_lo=cv["rel_lo"], conv_hi=cv["rel_hi"], conv_p=cv["p"]))
seg = pd.DataFrame(rows)
seg["margin_p_bh"] = adjust_pvalues(seg.margin_p.values, "fdr_bh")
seg["conv_p_bh"] = adjust_pvalues(seg.conv_p.values, "fdr_bh")
out["segments"] = seg.to_dict("records")
# formal interaction tests (is the effect DIFFERENT across segments?)
inter = {}
for col in ["is_new", "device", "source"]:
    full = smf.ols(f"margin ~ variant * C({col})", df).fit(cov_type="HC1")
    red = smf.ols(f"margin ~ variant + C({col})", df).fit(cov_type="HC1")
    names_i = [n for n in full.params.index if n.startswith("variant:")]
    w = full.wald_test(" , ".join(f"{n} = 0" for n in names_i), scalar=True)
    inter[col] = dict(p=float(w.pvalue))
out["interaction_tests_margin"] = inter

fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
ys = np.arange(len(seg))[::-1]
for ax, (m, ttl) in zip(axs, [("conv", "Conversion"), ("margin", "Margin per visitor")]):
    for y_, (_, r) in zip(ys, seg.iterrows()):
        col_ = BLUE if r.status == "pre-registered" else GRAY
        ax.plot([r[f"{m}_lo"] * 100, r[f"{m}_hi"] * 100], [y_, y_], color=col_, lw=2)
        ax.scatter([r[f"{m}_rel"] * 100], [y_], color=col_, s=26, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_title(f"{ttl}: relative effect (%), 95% CI", fontsize=10)
axs[0].set_yticks(ys); axs[0].set_yticklabels([f"{r.segment}" for _, r in seg.iterrows()])
axs[0].text(0.0, -0.13, "Blue = pre-registered segment, gray = exploratory", transform=axs[0].transAxes, fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/segments_forest.png"); plt.close(fig)

# ---- E. novelty effect
rows = []
for d in range(14):
    g = df[df.arrival_day == d]; gt, gc = split(g)
    cv = compare(gt.purchased.values, gc.purchased.values)
    rows.append(dict(day=d, rel=cv["rel"], lo=cv["rel_lo"], hi=cv["rel_hi"]))
daily = pd.DataFrame(rows)
wk = {}
for name, mask in [("week1", df.arrival_day < 7), ("week2", df.arrival_day >= 7)]:
    g = df[mask]; gt, gc = split(g)
    wk[name] = dict(conv=compare(gt.purchased.values, gc.purchased.values), margin=compare(gt.margin.values, gc.margin.values))
d_conv = wk["week1"]["conv"]["diff"] - wk["week2"]["conv"]["diff"]
se_conv = np.sqrt(wk["week1"]["conv"]["se"] ** 2 + wk["week2"]["conv"]["se"] ** 2)
wk["week1_minus_week2_conv"] = dict(diff=float(d_conv), se=float(se_conv), p=float(2 * stats.norm.sf(abs(d_conv / se_conv))))
# one-degree-of-freedom test for a time trend in the treatment effect (more power than week1-vs-week2)
tr = smf.ols("purchased ~ variant * day_c", df.assign(day_c=df.arrival_day - 6.5)).fit(cov_type="HC1")
wk["trend_test"] = dict(slope_per_day=float(tr.params["variant:day_c"]), se=float(tr.bse["variant:day_c"]),
                        p=float(tr.pvalues["variant:day_c"]))
out["novelty"] = wk
out["daily_conv"] = daily.to_dict("records")

fig, ax = plt.subplots(figsize=(7.4, 4))
ax.errorbar(daily.day, daily.rel * 100, yerr=[(daily.rel - daily.lo) * 100, (daily.hi - daily.rel) * 100],
            fmt="o", color=BLUE, ms=4, lw=1.2, capsize=0)
for lo_d, hi_d, key in [(0, 6, "week1"), (7, 13, "week2")]:
    v = wk[key]["conv"]["rel"] * 100
    ax.plot([lo_d - 0.3, hi_d + 0.3], [v, v], color=ORANGE, lw=2.5)
    ax.text((lo_d + hi_d) / 2, -17, f"{key[:-1].title()} {key[-1]} average: {v:+.1f}%", color=ORANGE, fontsize=9, ha="center")
ax.axhline(0, color=MUTED, lw=1)
ax.axvline(6.5, color=MUTED, lw=0.8, ls="--")
ax.set_ylim(-20, 30)
ax.set_xlabel("Arrival day (0 = Monday of week 1)"); ax.set_ylabel("Conversion lift (relative, %)")
ax.set_title("Daily conversion lift: too noisy to see a decay at this volume")
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/novelty_effect.png"); plt.close(fig)

# ---- F. annualised impact (assumption: traffic stays at the experiment's daily rate)
users_per_year = len(df) / 14 * 365   # analysed (clean) population only
m_all = final[final.metric == "Margin per visitor"].iloc[0]
out["annualisation"] = dict(users_per_year=float(users_per_year),
                            margin_pv_diff=float(m_all["diff"]), margin_pv_lo=float(m_all["lo"]), margin_pv_hi=float(m_all["hi"]),
                            annual_point=float(m_all["diff"] * users_per_year), annual_lo=float(m_all["lo"] * users_per_year),
                            annual_hi=float(m_all["hi"] * users_per_year),
                            week2_annual_point=float(wk["week2"]["margin"]["diff"] * users_per_year),
                            week2_annual_lo=float(wk["week2"]["margin"]["lo"] * users_per_year),
                            week2_annual_hi=float(wk["week2"]["margin"]["hi"] * users_per_year))

json.dump(out, open(f"{ROOT}/results/analysis.json", "w"), indent=2, default=float)

pd.set_option("display.width", 200)
print("NAIVE (bug ignored):"); print(naive[["metric", "mean_c", "mean_t", "rel", "rel_lo", "rel_hi", "p"]].round(4).to_string())
print("\nCLEAN:"); print(final[["metric", "role", "mean_c", "mean_t", "diff", "rel", "rel_lo", "rel_hi", "p", "p_holm", "p_bh"]].round(4).to_string())
print("\nMethods:"); [print(f"{k:55s} rel={v['rel']:+.3%} [{v['rel_lo']:+.3%},{v['rel_hi']:+.3%}] width_vs_raw={v['width_vs_raw']:.3f}") for k, v in est.items()]
print("\nCUPED", out["cuped"], "returning-only", out["cuped_returning_only"], "new-only corr", out["cuped_new_only_corr"])
print("\nAOV delta vs boot", out["aov_delta_vs_boot"])
print("\nSegments:"); print(seg.round(4).to_string())
print("interaction", inter)
print("\nNovelty", json.dumps(wk, indent=1, default=float)[:1800])
print("\nAnnual", out["annualisation"])
