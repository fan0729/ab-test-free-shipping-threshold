"""Part B: apply the same toolkit to a REAL randomised experiment (Criteo Uplift Prediction dataset).

NOT RUN in the build environment (network access to Criteo was blocked). Run it yourself:

    1. Download criteo-research-uplift-v2.1.csv.gz from the Criteo AI Lab uplift dataset page
       (https://ailab.criteo.com/criteo-uplift-prediction-dataset/) into data/criteo/
    2. python src/05_criteo_replication.py --path data/criteo/criteo-research-uplift-v2.1.csv.gz

Dataset facts to VERIFY against the dataset's own documentation before trusting the defaults:
  * columns: f0..f11 (anonymised features), treatment (0/1), conversion, visit, exposure
  * treatment is deliberately NOT 50/50 (about 85% treated) -> pass --expected-treat-share
  * 'exposure' happens AFTER assignment: never filter or adjust on it (that would break randomisation).
    The headline analysis is intention-to-treat on 'treatment'.

What this script reproduces from the simulated project: SRM check against the expected ratio,
covariate balance (SMD), ITT effects with CIs on visit and conversion, regression adjustment
with pre-treatment features (variance reduction), and a 200-split A/A test inside the control group.
All numbers are printed and saved to results/criteo.json -- cite only what this produces.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from abtools import srm_test, smd, compare, regression_adjust, two_prop_z

ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--expected-treat-share", type=float, default=0.85, help="verify in dataset docs")
ap.add_argument("--adjust-rows", type=int, default=3_000_000, help="subsample size for the OLS adjustment (memory)")
ap.add_argument("--nrows", type=int, default=None, help="read only the first N rows (debugging)")
a = ap.parse_args()

df = pd.read_csv(a.path, nrows=a.nrows)
feats = [f"f{i}" for i in range(12)]
need = feats + ["treatment", "conversion", "visit"]
missing = [c for c in need if c not in df.columns]
assert not missing, f"columns not found: {missing}"
t, c = df[df.treatment == 1], df[df.treatment == 0]
out = {"n": int(len(df)), "n_treat": int(len(t)), "n_ctrl": int(len(c))}

# 1. SRM against the DOCUMENTED (unequal) allocation
out["srm"] = srm_test(len(t), len(c), expected_ratio=a.expected_treat_share)

# 2. covariate balance
out["balance_smd"] = {f: smd(t[f].values.astype(float), c[f].values.astype(float)) for f in feats}
out["max_abs_smd"] = float(max(abs(v) for v in out["balance_smd"].values()))

# 3. ITT effects (visit, conversion) -- binary outcomes, user-level
for m in ["visit", "conversion"]:
    out[f"itt_{m}"] = compare(t[m].values.astype(float), c[m].values.astype(float))
    out[f"itt_{m}"]["z_test_p"] = two_prop_z(t[m].values, c[m].values)["p"]

# 4. regression adjustment on a subsample (full 14M x 25 design matrix does not fit in most laptops)
sub = df.sample(min(a.adjust_rows, len(df)), random_state=0)
for m in ["visit", "conversion"]:
    raw = compare(sub[sub.treatment == 1][m].values.astype(float), sub[sub.treatment == 0][m].values.astype(float))
    adj = regression_adjust(sub, m, treat="treatment", covars=feats)
    out[f"adjust_{m}"] = dict(raw_se=raw["se"], adj_se=adj["se"], se_reduction=float(1 - adj["se"] / raw["se"]),
                              var_reduction=float(1 - (adj["se"] / raw["se"]) ** 2), adj_diff=adj["diff"], raw_diff=raw["diff"])

# 5. A/A inside the control group
rng = np.random.default_rng(3)
y = c.conversion.values.astype(float)
fp = 0
for _ in range(200):
    m = rng.random(len(y)) < 0.5
    fp += two_prop_z(y[m], y[~m])["p"] < 0.05
out["aa_conversion_fpr"] = dict(fpr=fp / 200, n_sims=200)

os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(out, open(f"{ROOT}/results/criteo.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
