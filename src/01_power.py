"""Phase 1-2: pre-experiment planning from historical (pilot) data only.

Reads data/pilot_history.csv.gz (control experience, no treatment). Never touches the experiment.
Outputs results/power.json, figures/power_curve.png and memo/01_experiment_design.md.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from plotting import BLUE, GRAY, INK, MUTED
from abtools import welch, regression_adjust

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(f"{ROOT}/results", exist_ok=True)
pilot = pd.read_csv(f"{ROOT}/data/pilot_history.csv.gz")  # plan on ALL traffic (we don't know about the bug yet)

ZA, ZB = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
N_USERS_TOTAL, DAYS = 400_000, 14           # planned traffic over the 14-day window (business cap)
PER_DAY = N_USERS_TOTAL / DAYS

mu, sd = pilot.margin.mean(), pilot.margin.std()
conv = pilot.purchased.mean()
cap = float(np.quantile(pilot.margin, 0.9995))   # pre-specified winsorisation cap

# variance reduction available from pre-period covariates (estimated on pilot, no treatment involved)
ra = regression_adjust(pilot.assign(variant=np.random.default_rng(0).integers(0, 2, len(pilot))),
                       "margin", covars=("pre_spend_90d", "is_new"))
r2 = ra["r2"]
vr = float(1 - np.var(pilot.margin - np.polyfit(pilot.pre_spend_90d, pilot.margin, 1)[0] * pilot.pre_spend_90d, ddof=1) / pilot.margin.var(ddof=1))


def n_per_arm_mean(rel_mde, var_factor=1.0):
    return 2 * (ZA + ZB) ** 2 * (sd ** 2 * var_factor) / (mu * rel_mde) ** 2


def mde_mean(n, var_factor=1.0):
    return (ZA + ZB) * np.sqrt(2 * sd ** 2 * var_factor / n) / mu


def n_per_arm_prop(rel_mde):
    p1, p2 = conv, conv * (1 + rel_mde)
    pbar = (p1 + p2) / 2
    return (ZA * np.sqrt(2 * pbar * (1 - pbar)) + ZB * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2


def mde_prop(n):
    lo, hi = 1e-4, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if n_per_arm_prop(mid) > n else (lo, mid)
    return hi


n_arm = N_USERS_TOTAL // 2
vf = 1 - r2

# simulation-based power check for margin per visitor (bootstrap from the pilot, effect injected multiplicatively)
rng = np.random.default_rng(7)
y = pilot.margin.values


def sim_power(rel, n, sims=400):
    hits = 0
    for _ in range(sims):
        c = y[rng.integers(0, len(y), n)]
        t = y[rng.integers(0, len(y), n)] * (1 + rel)
        hits += welch(t, c)["p"] < 0.05
    return hits / sims


sim_rows = []
for rel in [0.03, 0.05, 0.0667, 0.08]:
    sim_rows.append(dict(rel_mde=rel, power_sim=sim_power(rel, n_arm), power_formula=float(
        stats.norm.cdf(rel * mu / np.sqrt(2 * sd ** 2 / n_arm) - ZA))))

res = dict(
    baseline=dict(conv=float(conv), margin_per_visitor=float(mu), margin_sd=float(sd), cv=float(sd / mu),
                  winsor_cap=cap, pilot_n=int(len(pilot)), r2_prespend_isnew=float(r2),
                  cuped_var_reduction_prespend_only=vr),
    traffic=dict(total=N_USERS_TOTAL, per_arm=n_arm, days=DAYS, per_day=PER_DAY),
    ideal=dict(n_per_arm_5pct=float(n_per_arm_mean(0.05)), days_needed_5pct=float(2 * n_per_arm_mean(0.05) / PER_DAY),
               n_per_arm_5pct_adj=float(n_per_arm_mean(0.05, vf)), days_needed_5pct_adj=float(2 * n_per_arm_mean(0.05, vf) / PER_DAY)),
    achieved=dict(mde_margin_raw=float(mde_mean(n_arm)), mde_margin_adj=float(mde_mean(n_arm, vf)),
                  mde_conv_rel=float(mde_prop(n_arm)), n_per_arm_conv_5pct=float(n_per_arm_prop(0.05)),
                  n_per_arm_conv_3pct=float(n_per_arm_prop(0.03))),
    sim_power=sim_rows,
)
# follow-up test for new customers only
new = pilot[pilot.is_new == 1]
mu_n, sd_n = new.margin.mean(), new.margin.std()
res["new_only"] = dict(share_new=float((pilot.is_new == 1).mean()), margin_mean=float(mu_n), margin_sd=float(sd_n),
                       n_per_arm_5pct=float(2 * (ZA + ZB) ** 2 * sd_n ** 2 / (mu_n * 0.05) ** 2))
json.dump(res, open(f"{ROOT}/results/power.json", "w"), indent=2)

# figure: relative MDE vs per-arm sample size
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ns = np.linspace(40_000, 700_000, 200)
ax.plot(ns / 1e3, [mde_mean(n) * 100 for n in ns], color=GRAY, label="Raw")
ax.plot(ns / 1e3, [mde_mean(n, vf) * 100 for n in ns], color=BLUE, lw=1.2, label="With covariate adjustment")
ax.axvline(n_arm / 1e3, color=MUTED, lw=1, ls="--")
ax.text(n_arm / 1e3 + 8, ax.get_ylim()[1] * 0.92, "14-day traffic cap\n(200K / arm)", color=MUTED, fontsize=9, va="top")
ax.text(560, 9.5, "Raw and covariate-adjusted curves overlap:\nadjustment removes only ~2% of variance", color=MUTED, fontsize=9, ha="center")
ax.set_xlabel("Users per arm (thousands)")
ax.set_ylabel("Min. detectable effect (relative, %)")
ax.set_title("Margin per visitor: what the test can detect (80% power, α = 0.05)")
fig.tight_layout()
fig.savefig(f"{ROOT}/figures/power_curve.png")
print(json.dumps(res, indent=2))
