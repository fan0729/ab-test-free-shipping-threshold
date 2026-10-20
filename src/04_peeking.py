"""Phase 5.8: the peeking problem.  Monte-Carlo with binomial data, plus the real experiment's daily p-value path."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from scipy import stats
from plotting import BLUE, GRAY, ORANGE, INK, MUTED
from abtools import calibrate_boundaries, two_prop_z

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_csv(f"{ROOT}/data/experiment.csv.gz")
df = df[df.app_old == 0]
K = 14
per_day_arm = int(len(df) / K / 2)                    # users per arm per day (clean population)
p0 = float(df[df.variant == 0].purchased.mean())
rng = np.random.default_rng(2026)

# ---- boundaries calibrated on one seed, verified on independent binomial simulations
b_obf = calibrate_boundaries(K, shape="obf", seed=101)
b_poc = calibrate_boundaries(K, shape="pocock", seed=102)


def sim_paths(rel_lift, n_sims):
    pc = p0
    pt = p0 * (1 + rel_lift)
    xc = rng.binomial(per_day_arm, pc, (n_sims, K)).cumsum(1)
    xt = rng.binomial(per_day_arm, pt, (n_sims, K)).cumsum(1)
    n = per_day_arm * np.arange(1, K + 1)
    p1, p2 = xt / n, xc / n
    pp = (xt + xc) / (2 * n)
    z = (p1 - p2) / np.sqrt(pp * (1 - pp) * 2 / n)
    return z


def summarise(z):
    crit = 1.96
    out = {}
    out["fixed_day14"] = np.mean(np.abs(z[:, -1]) > crit)
    out["peek_daily_naive"] = np.mean((np.abs(z) > crit).any(1))
    out["peek_weekly_naive"] = np.mean((np.abs(z[:, [6, 13]]) > crit).any(1))
    out["pocock"] = np.mean((np.abs(z) > b_poc).any(1))
    out["obf"] = np.mean((np.abs(z) > b_obf).any(1))
    # average stopping day (1-indexed) for rules that stop at first crossing
    for name, bound in [("pocock", b_poc), ("obf", b_obf), ("peek_daily_naive", np.full(K, crit))]:
        cross = np.abs(z) > bound
        first = np.where(cross.any(1), cross.argmax(1) + 1, K)
        out[f"avg_stop_day_{name}"] = float(first.mean())
    return {k: float(v) for k, v in out.items()}


z0 = sim_paths(0.0, 20_000)
res = {"null": summarise(z0), "boundaries": dict(obf=b_obf.round(3).tolist(), pocock=b_poc.round(3).tolist()),
       "setup": dict(per_day_per_arm=per_day_arm, base_conv=p0, looks=K, null_sims=20_000)}
# power at the effect we think is plausible (relative +5% on conversion)
z1 = sim_paths(0.05, 10_000)
res["alt_5pct"] = summarise(z1)
res["alt_5pct"]["true_rel_lift"] = 0.05

# ---- the real experiment's daily path
rows = []
for d in range(K):
    g = df[df.arrival_day <= d]
    r = two_prop_z(g[g.variant == 1].purchased.values, g[g.variant == 0].purchased.values)
    rows.append(dict(day=d + 1, z=r["diff"] / r["se"], p=r["p"], rel=r["diff"] / r["mean_c"]))
real = pd.DataFrame(rows)
res["real_path"] = real.round(5).to_dict("records")
json.dump(res, open(f"{ROOT}/results/peeking.json", "w"), indent=2)

# ---- figure
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
days = np.arange(1, K + 1)
for i in range(40):
    ax.plot(days, np.abs(z0[i]), color=GRAY, lw=0.8, alpha=0.5)
ax.hlines(1.96, 1, 14, color=ORANGE, lw=2)
ax.plot(days, b_obf, color=BLUE, lw=2)
ax.text(14.2, 1.7, "Naive 1.96", color=ORANGE, va="center", fontsize=9)
ax.text(14.2, b_obf[-1] + 0.25, "OBF-style\n(starts at 7.9)", color=BLUE, va="center", fontsize=9)
ax.set_xlim(1, 16); ax.set_ylim(0, 5)
ax.set_xlabel("Days of data"); ax.set_ylabel("|z| statistic")
ax.set_title("40 A/A experiments: many touch 1.96 by chance", fontsize=10)
ax = axs[1]
labs = ["Fixed\n(day 14)", "Peek daily\n(naive)", "Peek weekly\n(naive)", "Pocock", "OBF-style"]
vals = [res["null"][k] * 100 for k in ["fixed_day14", "peek_daily_naive", "peek_weekly_naive", "pocock", "obf"]]
cols = [GRAY, ORANGE, ORANGE, BLUE, BLUE]
bars = ax.bar(labs, vals, color=cols, width=0.62)
ax.axhline(5, color=MUTED, lw=1, ls="--")
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("False-positive rate under H0 (%)"); ax.tick_params(axis="x", labelsize=8)
ax.set_title("Type I error by stopping rule", fontsize=10)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/peeking.png"); plt.close(fig)

print(json.dumps({k: res[k] for k in ["null", "alt_5pct", "boundaries", "setup"]}, indent=1))
print(real.round(4).to_string())
