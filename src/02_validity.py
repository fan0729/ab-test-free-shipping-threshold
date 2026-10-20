"""Phase 4: validity checks BEFORE looking at any treatment effect.

1. Sample Ratio Mismatch  2. SRM diagnosis by segment  3. Covariate balance (SMD)
4. A/A simulation (1,000 random splits of control-only users) to validate the pipeline.
Outputs results/validity.json and figures.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from plotting import BLUE, GRAY, ORANGE, INK, MUTED
from abtools import srm_test, smd, welch, two_prop_z, cuped_adjust

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_csv(f"{ROOT}/data/experiment.csv.gz")
t, c = df[df.variant == 1], df[df.variant == 0]
out = {}

# ---- 1. SRM on the full logged data
out["srm_all"] = srm_test(len(t), len(c))

# ---- 2. Where does the mismatch live?  (chi-square within each pre-treatment segment)
rows = []
for col in ["device", "source", "is_new", "app_old"]:
    for v, g in df.groupby(col):
        r = srm_test((g.variant == 1).sum(), (g.variant == 0).sum())
        rows.append(dict(segment=f"{col}={v}", n=len(g), share_treat=r["share_treat"], p=r["p"]))
seg = pd.DataFrame(rows).sort_values("p")
out["srm_by_segment"] = seg.to_dict("records")
g = df[(df.device == "mobile")]
out["srm_mobile_by_appold"] = {str(k): srm_test((x.variant == 1).sum(), (x.variant == 0).sum())
                               for k, x in g.groupby("app_old")}
# SRM after excluding the affected segment (pre-treatment definition, available in both arms)
clean = df[df.app_old == 0]
out["srm_clean"] = srm_test((clean.variant == 1).sum(), (clean.variant == 0).sum())

# conversion of surviving old-app users, treatment vs control: shows the selection bias directly
ao = df[df.app_old == 1].groupby("variant").agg(n=("user_id", "size"), conv=("purchased", "mean"))
out["old_app_conv"] = ao.reset_index().to_dict("records")

# bias if the bug is ignored: naive vs clean-population conversion lift
out["naive_conv_lift_all"] = two_prop_z(t.purchased.values, c.purchased.values)
cc, ct = clean[clean.variant == 0], clean[clean.variant == 1]
out["clean_conv_lift"] = two_prop_z(ct.purchased.values, cc.purchased.values)

# ---- 3. Covariate balance (SMD) on pre-treatment covariates
def balance(a, b, label):
    rows = []
    for col in ["is_new", "pre_spend_90d", "pre_orders_90d", "app_old", "arrival_day"]:
        rows.append((col, smd(a[col].values.astype(float), b[col].values.astype(float))))
    for d in ["mobile", "desktop", "tablet"]:
        rows.append((f"device={d}", smd((a.device == d).values.astype(float), (b.device == d).values.astype(float))))
    for s in ["organic", "paid", "email", "direct"]:
        rows.append((f"source={s}", smd((a.source == s).values.astype(float), (b.source == s).values.astype(float))))
    return pd.DataFrame(rows, columns=["covariate", label]).set_index("covariate")

bal = balance(t, c, "all users").join(balance(ct, cc, "excl. old-app"))
out["balance"] = bal.round(4).reset_index().to_dict("records")

fig, ax = plt.subplots(figsize=(7, 4.6))
b = bal.sort_values("all users")
y = np.arange(len(b))
ax.axvspan(-0.1, 0.1, color="#eef0f2", zorder=0)
ax.scatter(b["all users"], y, color=ORANGE, s=36, label="All logged users", zorder=3)
ax.scatter(b["excl. old-app"], y, color=BLUE, s=36, label="Excluding old-app mobile", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(b.index, fontsize=8)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Standardised mean difference (treatment − control); band = ±0.1")
ax.set_title("Covariate balance before vs after removing the affected segment")
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/covariate_balance.png"); plt.close(fig)

# ---- 4. A/A simulation: 1,000 random splits of CONTROL-only users (clean population)
rng = np.random.default_rng(11)
pool = cc.reset_index(drop=True)
ym, yc, xpre = pool.margin.values, pool.purchased.values, pool.pre_spend_90d.values
yadj, _ = cuped_adjust(ym, xpre)
n = len(pool)
res = {"conv_z": [], "margin_welch": [], "margin_cuped": []}
for _ in range(1000):
    m = rng.random(n) < 0.5
    res["conv_z"].append(two_prop_z(yc[m], yc[~m])["p"])
    res["margin_welch"].append(welch(ym[m], ym[~m])["p"])
    res["margin_cuped"].append(welch(yadj[m], yadj[~m])["p"])
aa = {k: dict(fpr=float(np.mean(np.array(v) < 0.05)), n_sims=1000) for k, v in res.items()}
out["aa"] = aa
out["aa_n"] = int(n)

fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, (k, ttl) in zip(axs, [("conv_z", "Conversion (z-test)"), ("margin_welch", "Margin / visitor (Welch)"),
                                ("margin_cuped", "Margin / visitor (CUPED)")]):
    ax.hist(res[k], bins=20, range=(0, 1), color=BLUE, edgecolor="#fcfcfb")
    ax.axhline(1000 / 20, color=MUTED, lw=1, ls="--")
    ax.set_title(f"{ttl}\nFPR = {aa[k]['fpr']:.1%}", fontsize=10)
    ax.set_xlabel("p-value")
axs[0].set_ylabel("Count of 1,000 A/A tests")
fig.suptitle("A/A test: p-values should be uniform; ~5% below 0.05", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/aa_pvalues.png"); plt.close(fig)

json.dump(out, open(f"{ROOT}/results/validity.json", "w"), indent=2)
print(json.dumps({k: out[k] for k in ["srm_all", "srm_clean", "srm_mobile_by_appold", "old_app_conv", "naive_conv_lift_all", "clean_conv_lift", "aa"]}, indent=1))
print(seg.head(6).to_string())
print(bal.round(3).to_string())
