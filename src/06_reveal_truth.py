"""Final step: open data/truth.json and compare every estimate against the known true effect."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from plotting import BLUE, GRAY, ORANGE, MUTED

ROOT = os.path.join(os.path.dirname(__file__), "..")
truth = json.load(open(f"{ROOT}/data/truth.json"))
A = json.load(open(f"{ROOT}/results/analysis.json"))
V = json.load(open(f"{ROOT}/results/validity.json"))
tw, ts = truth["experiment_window"], truth["steady_state"]
CL = "clean (excl. old-app mobile)"
rows = []


def add(name, est, lo, hi, tr_w, tr_s):
    rows.append(dict(item=name, estimate=est, lo=lo, hi=hi, truth_window=tr_w, truth_steady=tr_s,
                     covers_window=bool(lo <= tr_w <= hi), covers_steady=bool(lo <= tr_s <= hi)))


m = {r["metric"]: r for r in A["metrics"]}
add("Conversion lift, clean pop. (relative)", m["Conversion rate"]["rel"], m["Conversion rate"]["rel_lo"], m["Conversion rate"]["rel_hi"],
    tw[CL]["conv_rel"], ts[CL]["conv_rel"])
nv = {r["metric"]: r for r in A["naive_metrics"]}["Conversion rate"]
add("Conversion lift, NAIVE all users (relative)", nv["rel"], nv["rel_lo"], nv["rel_hi"], tw["all"]["conv_rel"], ts["all"]["conv_rel"])
for k, v in A["margin_methods"].items():
    add(f"Margin/visitor $: {k}", v["diff"], v["lo"], v["hi"], tw[CL]["margin_ate"], ts[CL]["margin_ate"])
res = pd.DataFrame(rows)
res_main = res.dropna(subset=["lo"]).copy()

# segment CIs on $ scale: rebuild from rel CI and control mean
seg_rows = []
for r in A["segments"]:
    key = {"New": "new", "Returning": "returning", "Mobile": "mobile", "Desktop": "desktop", "Tablet": "tablet",
           "Organic": "src_organic", "Paid": "src_paid", "Email": "src_email", "Direct": "src_direct"}[r["segment"]]
    # relative-effect CI on rel scale vs the TRUE relative effect in the same segment
    seg_rows.append(dict(segment=r["segment"], status=r["status"], est_rel=r["margin_rel"], lo=r["margin_lo"], hi=r["margin_hi"],
                         truth_window_rel=tw[key]["margin_rel"], truth_steady_rel=ts[key]["margin_rel"],
                         covers_window=bool(r["margin_lo"] <= tw[key]["margin_rel"] <= r["margin_hi"]),
                         sig_raw=bool(r["margin_p"] < 0.05)))
seg = pd.DataFrame(seg_rows)

# winsorisation bias
w = A["margin_methods"]["Welch, winsorised at pilot P99.95"]["diff"]
raw = A["margin_methods"]["Welch t-test (raw)"]["diff"]
out = dict(
    table=res_main.round(5).to_dict("records"),
    segments=seg.round(5).to_dict("records"),
    winsor_shift_vs_raw=float(w - raw),
    bug=truth["bug"],
    true_effects=dict(
        clean_window=dict(conv_rel=tw[CL]["conv_rel"], margin_rel=tw[CL]["margin_rel"], margin_ate=tw[CL]["margin_ate"]),
        clean_steady=dict(conv_rel=ts[CL]["conv_rel"], margin_rel=ts[CL]["margin_rel"], margin_ate=ts[CL]["margin_ate"]),
        new_steady=dict(conv_rel=ts["new"]["conv_rel"], margin_rel=ts["new"]["margin_rel"], margin_ate=ts["new"]["margin_ate"]),
        returning_steady=dict(conv_rel=ts["returning"]["conv_rel"], margin_rel=ts["returning"]["margin_rel"], margin_ate=ts["returning"]["margin_ate"]),
        new_window=dict(margin_rel=tw["new"]["margin_rel"]), returning_window=dict(margin_rel=tw["returning"]["margin_rel"]),
        ship_cost_per_buyer_ctrl=tw[CL]["ship_cost_ctrl"] / tw[CL]["purchased_ctrl"], ),
    by_day_conv_rel_all_users=truth["by_day_conv_rel"],
)
json.dump(out, open(f"{ROOT}/results/reveal.json", "w"), indent=2, default=float)

# ---- figure: daily lift estimates vs true daily lift
daily = pd.DataFrame(A["daily_conv"])
fig, ax = plt.subplots(figsize=(7.4, 4))
ax.errorbar(daily.day, daily.rel * 100, yerr=[(daily.rel - daily.lo) * 100, (daily.hi - daily.rel) * 100], fmt="o", color=BLUE, ms=4, lw=1.2,
            label="Estimate ± 95% CI")
ax.plot(range(14), np.array(truth["by_day_conv_rel"]) * 100, color=ORANGE, lw=2, label="True lift (known from simulation)")
ax.axhline(ts[CL]["conv_rel"] * 100, color=MUTED, lw=1, ls="--")
ax.text(0.1, ts[CL]["conv_rel"] * 100 + 2.2, "True steady state", color=MUTED, ha="left", fontsize=9)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Arrival day"); ax.set_ylabel("Conversion lift (relative, %)")
ax.set_title("The decay was real, but daily data could not resolve it")
ax.set_ylim(-18, 36)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/novelty_vs_truth.png"); plt.close(fig)

pd.set_option("display.width", 220)
print(res_main[["item", "estimate", "lo", "hi", "truth_window", "truth_steady", "covers_window", "covers_steady"]].round(4).to_string())
print(seg.round(4).to_string())
print("winsor shift vs raw:", out["winsor_shift_vs_raw"], truth["bug"])
