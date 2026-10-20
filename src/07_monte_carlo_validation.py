"""Repeat the WHOLE experiment 100 times (new seed each time) and check that the pipeline behaves.

Because the generator knows the true effect in every replication, we can measure:
  * bias and 95%-CI coverage of each estimator,
  * how often SRM fires (it should, every time: the logging bug is always injected),
  * how wrong the naive all-user analysis is vs the clean-population analysis,
  * how often the test detects each effect (empirical power).
Run time ~10 min on 2 cores.   Output: results/montecarlo.json, figures/montecarlo_coverage.png
"""
import json, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from generate_experiment import generate
from abtools import compare, welch, cuped_adjust, srm_test

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAP = json.load(open(f"{ROOT}/results/power.json"))["baseline"]["winsor_cap"]
N_REPS = 100


def one(rep):
    df, truth = generate(seed=5000 + rep)
    tw, ts = truth["experiment_window"], truth["steady_state"]
    clean = df[df.app_old == 0]
    t, c = clean[clean.variant == 1], clean[clean.variant == 0]
    r = {"rep": rep}
    srm_all = srm_test((df.variant == 1).sum(), (df.variant == 0).sum())
    srm_cl = srm_test((clean.variant == 1).sum(), (clean.variant == 0).sum())
    r["srm_all_p"], r["srm_clean_p"] = srm_all["p"], srm_cl["p"]

    tr_w, tr_s = tw["clean (excl. old-app mobile)"], ts["clean (excl. old-app mobile)"]
    m = welch(t.margin.values, c.margin.values)
    r.update(m_est=m["diff"], m_lo=m["lo"], m_hi=m["hi"], m_truth_w=tr_w["margin_ate"], m_truth_s=tr_s["margin_ate"])
    mw = welch(t.margin.clip(upper=CAP).values, c.margin.clip(upper=CAP).values)
    r["mw_est"] = mw["diff"]
    ya, _ = cuped_adjust(clean.margin.values, clean.pre_spend_90d.values)
    mask = (clean.variant == 1).values
    mc = welch(ya[mask], ya[~mask])
    r.update(mc_est=mc["diff"], mc_lo=mc["lo"], mc_hi=mc["hi"])

    cv = compare(t.purchased.values.astype(float), c.purchased.values.astype(float))
    r.update(cv_rel=cv["rel"], cv_lo=cv["rel_lo"], cv_hi=cv["rel_hi"], cv_p=cv["p"],
             cv_truth_w=tr_w["conv_rel"], cv_truth_s=tr_s["conv_rel"])
    ta, ca = df[df.variant == 1], df[df.variant == 0]
    cvn = compare(ta.purchased.values.astype(float), ca.purchased.values.astype(float))
    r.update(cvn_rel=cvn["rel"], cvn_lo=cvn["rel_lo"], cvn_hi=cvn["rel_hi"], cvn_truth_w=tw["all"]["conv_rel"])
    return r


if __name__ == "__main__":
    with Pool(2) as p:
        rows = p.map(one, range(N_REPS))
    R = pd.DataFrame(rows)
    cover = lambda lo, hi, tr: float(((lo <= tr) & (tr <= hi)).mean())
    out = dict(
        n_reps=N_REPS,
        srm_flagged_all=float((R.srm_all_p < 0.001).mean()),
        srm_flagged_clean_at_5pct=float((R.srm_clean_p < 0.05).mean()),
        margin=dict(coverage_window=cover(R.m_lo, R.m_hi, R.m_truth_w), coverage_steady=cover(R.m_lo, R.m_hi, R.m_truth_s),
                    bias_raw=float((R.m_est - R.m_truth_w).mean()), bias_winsor=float((R.mw_est - R.m_truth_w).mean()),
                    bias_cuped=float((R.mc_est - R.m_truth_w).mean()),
                    coverage_cuped_window=cover(R.mc_lo, R.mc_hi, R.m_truth_w),
                    mean_ci_width_raw=float((R.m_hi - R.m_lo).mean()), mean_ci_width_cuped=float((R.mc_hi - R.mc_lo).mean()),
                    sd_est_raw=float(R.m_est.std()), sd_est_cuped=float(R.mc_est.std()),
                    truth_window_mean=float(R.m_truth_w.mean()), truth_steady_mean=float(R.m_truth_s.mean()),
                    share_ci_excludes_zero=float(((R.m_lo > 0) | (R.m_hi < 0)).mean())),
        conversion=dict(bias_clean_rel=float((R.cv_rel - R.cv_truth_w).mean()), coverage_clean_window=cover(R.cv_lo, R.cv_hi, R.cv_truth_w),
                        coverage_clean_steady=cover(R.cv_lo, R.cv_hi, R.cv_truth_s),
                        bias_naive_rel=float((R.cvn_rel - R.cvn_truth_w).mean()), coverage_naive=cover(R.cvn_lo, R.cvn_hi, R.cvn_truth_w),
                        power_clean=float((R.cv_p < 0.05).mean()),
                        truth_window_mean=float(R.cv_truth_w.mean()), truth_steady_mean=float(R.cv_truth_s.mean())),
    )
    json.dump(out, open(f"{ROOT}/results/montecarlo.json", "w"), indent=2)
    R.to_csv(f"{ROOT}/results/montecarlo_reps.csv", index=False)

    import importlib
    importlib.import_module('07b_plot_montecarlo').plot()
    print(json.dumps(out, indent=2))
