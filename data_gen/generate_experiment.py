"""Semi-synthetic A/B test: free-shipping threshold $100 -> $75 on REAL H&M customers and orders.

WHAT IS REAL (from the H&M Personalized Fashion Recommendations dataset on Kaggle):
  * the customers: age, club-member status, fashion-news subscription
  * their online order history (pre-period spend, number of orders, whether they have ever bought)
  * what each customer actually did in a 14-day window: whether they bought, the basket value,
    the number of items and the day of the order. This is the CONTROL experience (status quo).

WHAT IS SIMULATED (no public dataset contains a shipping-threshold experiment, so these are assumptions):
  * the treatment effect: extra buyers, basket padding up to the new threshold, and who responds most
  * returns (the dataset has none), the cost model, randomisation, and the injected logging bug

Because the control outcome is real and the treatment effect is simulated on top of it, we generate BOTH
potential outcomes for every customer, so the true effect is known and each estimator can be checked
against it. Results from this data are semi-synthetic: they validate the METHOD on realistic customers
and baskets, they are not a measured business result.

Stage 1 (real data -> small table, run once; needs the Kaggle files):  python data_gen/prepare_hm_base.py
Stage 2 (this file):                                                    python data_gen/generate_experiment.py

Ground truth is written to data/truth.json. Analysis code must not read it until src/06_reveal_truth.py.

Assumptions (all adjustable in CFG):
  * H&M prices are anonymised, so order values are rescaled so the MEDIAN real order is $62. The $75 / $100
    thresholds therefore sit at fixed points of the real basket distribution.
  * Order = one customer's online purchases on one day. Online = sales_channel_id 2 (the usual reading of
    the dataset; the documentation does not say so explicitly).
  * The store ran a $100 threshold during the window; real baskets already contain any padding it caused.
  * Gross margin 42%; carrier cost $9.00; customer pays $7.50 under the threshold (net shipping cost $1.50
    paid, $9.00 free); a returned order refunds the goods, costs $6 and the shipping is not refunded.
  * Who responds (assumed): customers with no history respond most; among returning customers, those whose
    typical order sits between $60 and $100 respond most; those who already spend over $100 do not respond.
    Age, club status and news subscription have NO true effect, so they act as negative controls.
  * Novelty effect: lift in purchase odds starts ~2.2x steady state and decays over ~3 days.
  * INJECTED BUG (a simulated fault, by design, to test the SRM check): treatment-arm exposure logging fails
    for ~9% of users flagged old_app, mostly non-purchasers -> sample ratio mismatch and an inflated lift.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = dict(
    n_users=400_000,
    n_days=14,
    thr_control=100.0,
    thr_treat=75.0,
    median_order_usd=62.0,
    gm_rate=0.42,
    ship_cost_paid=1.5,
    ship_cost_free=9.0,
    return_cost=6.0,
    # assumed steady-state effect on the log-odds of purchase, by customer type
    delta_new=0.115,
    delta_aov_low=0.025,    # returning, typical order < $60
    delta_aov_mid=0.070,    # returning, typical order $60-$100 (shipping threshold is the binding constraint)
    delta_aov_high=0.000,   # returning, typical order >= $100 (already gets free shipping)
    novelty_amp=1.2,
    novelty_tau=3.0,
    bump_prob=0.30,         # shoppers whose basket is just under the new threshold pad up to it
    incr_lo=75.0,           # incremental buyers' baskets are drawn from real baskets in [incr_lo, incr_hi)
    incr_hi=110.0,
    app_old_share=0.09,
    bug_drop_nonbuyer=0.30,
    bug_drop_buyer=0.02,
)
CLEAN = "clean (excl. old-app)"
_CACHE = {}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _shipping(v, thr, cfg):
    return np.where(v >= thr, cfg["ship_cost_free"], cfg["ship_cost_paid"])


def _outcomes(purchase, v, ret, thr, cfg):
    ship = np.where(purchase, _shipping(v, thr, cfg), 0.0)
    gross = np.where(purchase, v, 0.0)
    net_rev = gross * (1 - ret)
    margin = np.where(purchase, cfg["gm_rate"] * v * (1 - ret) - ship - cfg["return_cost"] * ret, 0.0)
    return dict(purchased=purchase.astype(int), basket=gross, net_rev=net_rev, ship_cost=ship,
                returned=(purchase & (ret == 1)).astype(int), margin=margin)


def _features(b):
    """Pre-treatment segment columns derived from the real customer fields."""
    age = b.age.values
    b["age_band"] = np.select([np.isnan(age), age < 25, age < 40], ["unknown", "u25", "25-39"], "40+")
    club = b.club_member_status.fillna("NONE").values
    b["club"] = np.select([club == "ACTIVE", club == "PRE-CREATE"], ["active", "pre_create"], "other")
    news = b.fashion_news_frequency.fillna("NONE").values
    b["news_sub"] = np.isin(news, ["Regularly", "Monthly"]).astype(int)
    b["is_new"] = 1 - b.has_history
    b["young"] = (b.age_band == "u25").astype(int)
    b["club_active"] = (b.club == "active").astype(int)
    aov = np.where(b.pre_orders_90d > 0, b.pre_spend_90d / b.pre_orders_90d.clip(lower=1), np.nan)
    b["aov_band"] = np.select([np.isnan(aov), aov < 60, aov < 100], ["none", "lt60", "60to100"], "ge100")
    return b


def _load(kind, cfg):
    """Real customer table for the experiment ('experiment') or the historical pilot ('pilot')."""
    if kind in _CACHE:
        return _CACHE[kind]
    exp_path = os.path.join(HERE, "..", "data", "hm_base_experiment.csv.gz")
    path = os.path.join(HERE, "..", "data", f"hm_base_{kind}.csv.gz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run data_gen/prepare_hm_base.py on the Kaggle H&M files first (see README).")
    # one price scale for everything: median real order in the EXPERIMENT window = $62
    if "scale" not in _CACHE:
        e = pd.read_csv(exp_path, usecols=["purchased", "basket"])
        _CACHE["scale"] = cfg["median_order_usd"] / float(e.loc[e.purchased == 1, "basket"].median())
    k = _CACHE["scale"]
    b = pd.read_csv(path)
    b["pre_spend_90d"] = b.pre_spend_90d * k
    b["basket"] = b.basket * k
    b = _features(b)
    # baseline purchase propensity from REAL pre-period behaviour and attributes (used only to size the
    # simulated treatment effect for people who did not buy in the real window)
    X = pd.get_dummies(b[["age_band", "club"]], drop_first=True).astype(float)
    X["is_new"] = b.is_new.values
    X["log_spend"] = np.log1p(b.pre_spend_90d.values)
    X["log_orders"] = np.log1p(b.pre_orders_90d.values)
    X["news_sub"] = b.news_sub.values
    fit = sm.GLM(b.purchased.values, sm.add_constant(X), family=sm.families.Binomial()).fit()
    b["p0"] = np.clip(fit.predict(sm.add_constant(X)).values, 1e-5, 0.95)
    _CACHE[kind] = b
    return b


def generate(cfg=CFG, seed=20261016, control_only=False, n_users=None, kind=None):
    rng = np.random.default_rng(seed)
    kind = kind or ("pilot" if control_only else "experiment")
    base = _load(kind, cfg)
    n = n_users or cfg["n_users"]
    d = base.iloc[rng.choice(len(base), n, replace=False)].reset_index(drop=True)

    # ---------- real pre-treatment covariates and real control outcome ----------
    is_new = d.is_new.values.astype(bool)
    buy0 = d.purchased.values == 1
    v_real = d.basket.values
    day_pool = d.loc[buy0, "order_day"].values.astype(int)
    arrival_day = np.where(buy0, np.nan_to_num(d.order_day.values, nan=0).astype(int), rng.choice(day_pool, n))
    arrival_day = np.clip(arrival_day, 0, cfg["n_days"] - 1)
    dow = arrival_day % 7                      # window starts on a Monday
    app_old = rng.random(n) < cfg["app_old_share"]   # simulated fault flag, independent of behaviour

    # ---------- simulated treatment effect (assumptions in the docstring) ----------
    u_conv, u_bump, u_pad, u_ret, u_drop, u_inc = (rng.random(n) for _ in range(6))
    delta = np.select([is_new, d.aov_band.values == "lt60", d.aov_band.values == "60to100"],
                      [cfg["delta_new"], cfg["delta_aov_low"], cfg["delta_aov_mid"]], cfg["delta_aov_high"])
    novelty = 1.0 + cfg["novelty_amp"] * np.exp(-arrival_day / cfg["novelty_tau"])
    p0 = d.p0.values
    lg = np.log(p0 / (1 - p0))
    q_win = (_sigmoid(lg + delta * novelty) - p0) / (1 - p0)     # chance a non-buyer converts because of the policy
    q_ss = (_sigmoid(lg + delta) - p0) / (1 - p0)
    buy1 = buy0 | (u_conv < q_win)
    buy1_ss = buy0 | (u_conv < q_ss)
    incr = buy1 & ~buy0
    incr_ss = buy1_ss & ~buy0

    # baskets: control = real order. Treatment = real order, padded up to $75 w.p. bump_prob if just under it;
    # incremental buyers draw a real basket from [75, 110)
    real_b = v_real[buy0]
    pool = real_b[(real_b >= cfg["incr_lo"]) & (real_b < cfg["incr_hi"])]
    v_inc = pool[(u_inc * len(pool)).astype(int)]
    bump = buy0 & (v_real >= 65.0) & (v_real < cfg["thr_treat"]) & (u_bump < cfg["bump_prob"])
    v1_buyer = np.where(bump, cfg["thr_treat"] + 6.0 * u_pad, v_real)
    v0 = np.where(buy0, v_real, 0.0)
    v1 = np.where(buy0, v1_buyer, np.where(incr, v_inc, 0.0))
    v1_ss = np.where(buy0, v1_buyer, np.where(incr_ss, v_inc, 0.0))

    # returns (not in the dataset): incremental buyers return more often
    ret0 = (u_ret < 0.12).astype(int)
    ret1 = np.where(buy0, (u_ret < 0.12), (u_ret < 0.16)).astype(int)

    Y0 = _outcomes(buy0, v0, ret0, cfg["thr_control"], cfg)
    Y1 = _outcomes(buy1, v1, ret1, cfg["thr_treat"], cfg)
    Y1ss = _outcomes(buy1_ss, v1_ss, ret1, cfg["thr_treat"], cfg)

    # ---------- randomisation ----------
    variant = (rng.random(n) < 0.5).astype(int)
    if control_only:
        variant[:] = 0
    obs = {k: np.where(variant == 1, Y1[k], Y0[k]) for k in Y0}

    # ---------- injected logging bug ----------
    affected = app_old & (variant == 1)
    p_drop = np.where(obs["purchased"] == 1, cfg["bug_drop_buyer"], cfg["bug_drop_nonbuyer"])
    dropped = affected & (u_drop < p_drop)

    items_real = np.where(buy0, d.n_items.values, 0).astype(int)
    items_inc = 1 + rng.poisson(np.where(incr, v1, 0.0) / 45.0)
    items = np.where(variant == 1, np.where(buy0, items_real, np.where(incr, items_inc, 0)), items_real)
    df = pd.DataFrame(dict(
        user_id=np.arange(n), arrival_day=arrival_day, dow=dow, age_band=d.age_band.values, club=d.club.values,
        news_sub=d.news_sub.values, aov_band=d.aov_band.values, young=d.young.values, club_active=d.club_active.values,
        is_new=is_new.astype(int), app_old=app_old.astype(int),
        pre_spend_90d=np.round(d.pre_spend_90d.values, 2), pre_orders_90d=d.pre_orders_90d.values, variant=variant,
        **{k: (np.round(val, 2) if val.dtype.kind == "f" else val) for k, val in obs.items()},
    ))
    df["items"] = items
    df = df.loc[~dropped].reset_index(drop=True)

    # ---------- ground truth (from potential outcomes of ALL sampled users, incl. dropped) ----------
    clean = ~app_old
    ab, cb, nw = d.age_band.values, d.club.values, d.news_sub.values
    segs = {"all": np.ones(n, bool), CLEAN: clean, "new": is_new & clean, "returning": ~is_new & clean}
    for s in ["lt60", "60to100", "ge100"]:
        segs[f"aov_{s}"] = (d.aov_band.values == s) & clean
    for s in ["u25", "25-39", "40+", "unknown"]:
        segs[f"age_{s}"] = (ab == s) & clean
    for s in ["active", "pre_create", "other"]:
        segs[f"club_{s}"] = (cb == s) & clean
    segs["news_yes"], segs["news_no"] = (nw == 1) & clean, (nw == 0) & clean

    def summarise(mask, Yt):
        out = {}
        for m in ["margin", "net_rev", "purchased", "basket", "ship_cost", "returned"]:
            out[m + "_ctrl"] = float(Y0[m][mask].mean())
            out[m + "_treat"] = float(Yt[m][mask].mean())
        out["margin_ate"] = out["margin_treat"] - out["margin_ctrl"]
        out["margin_rel"] = out["margin_ate"] / out["margin_ctrl"] if out["margin_ctrl"] else float("nan")
        out["conv_ate"] = out["purchased_treat"] - out["purchased_ctrl"]
        out["conv_rel"] = out["conv_ate"] / out["purchased_ctrl"] if out["purchased_ctrl"] else float("nan")
        out["rev_ate"] = out["net_rev_treat"] - out["net_rev_ctrl"]
        out["share_of_users"] = float(mask.mean())
        return out

    truth = {"experiment_window": {k: summarise(m, Y1) for k, m in segs.items()},
             "steady_state": {k: summarise(m, Y1ss) for k, m in segs.items()},
             "by_day_conv_rel": [
                 float((Y1["purchased"][arrival_day == dd].mean() - Y0["purchased"][arrival_day == dd].mean())
                       / Y0["purchased"][arrival_day == dd].mean()) for dd in range(cfg["n_days"])],
             "bug": {"n_dropped": int(dropped.sum()), "share_of_treat_arm": float(dropped.sum() / max(1, (variant == 1).sum()))},
             "config": cfg, "seed": seed, "price_scale_usd_per_unit": _CACHE["scale"]}
    return df, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=20261016)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, truth = generate(seed=a.seed)
    df.to_csv(os.path.join(a.out, "experiment.csv.gz"), index=False)
    json.dump(truth, open(os.path.join(a.out, "truth.json"), "w"), indent=2)

    # historical pilot: REAL customers in the 14 days before the experiment, control experience only
    pilot, _ = generate(seed=a.seed + 1, control_only=True, n_users=150_000)
    pilot.to_csv(os.path.join(a.out, "pilot_history.csv.gz"), index=False)
    print(f"experiment rows: {len(df):,}  pilot rows: {len(pilot):,}")
    print(df.groupby("variant").size())
