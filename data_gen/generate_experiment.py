"""
Synthetic A/B test generator: free-shipping threshold $100 -> $75 (fashion e-commerce).

WHY SYNTHETIC: because we generate BOTH potential outcomes for every user, we know the
exact true effect and can check whether each analysis method recovers it. All results
from this data must be labelled "simulated" on a resume / README.

Ground truth is written to data/truth.json. Analysis code must not read it until the
final "reveal" step (src/06_reveal_truth.py).

Business assumptions (all stated, all adjustable in CFG):
  * Gross margin on merchandise: 42%
  * Carrier cost per order: $9.00; customer pays a $7.50 fee when under the threshold,
    so the company's NET shipping cost is $1.50 (paid shipping) or $9.00 (free shipping)
  * Returned order: merchandise refunded, $6 processing cost, shipping not refunded

Built-in features (each one is a lesson the analysis must handle):
  1. Zero-inflated, heavy-tailed outcome (log-normal baskets + rare giant orders)
  2. Pre-experiment spend correlated with outcome (CUPED has something to work with)
  3. Heterogeneous effect: new customers respond much more than returning customers
  4. Novelty effect: lift in purchase odds starts ~2.2x steady state and decays over ~3 days
  5. Basket-bumping behaviour: shoppers near a threshold pad their cart up to it
  6. Day-of-week seasonality
  7. INJECTED BUG: treatment-arm exposure logging fails for old-app mobile users, mostly
     for non-purchasers (purchases are logged server-side) -> Sample Ratio Mismatch and
     inflated conversion lift if the bug is ignored.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CFG = dict(
    n_users=400_000,
    n_days=14,
    thr_control=100.0,
    thr_treat=75.0,
    gm_rate=0.42,
    ship_cost_paid=1.5,
    ship_cost_free=9.0,
    return_cost=6.0,
    # true treatment effects on log-odds of purchase (steady state)
    delta_new=0.115,
    delta_ret=0.035,
    novelty_amp=1.2,
    novelty_tau=3.0,
    # basket bumping
    bump_prob=0.30,
    # data-quality bug
    bug_drop_nonbuyer=0.30,
    bug_drop_buyer=0.02,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _shipping(v, thr, cfg):
    return np.where(v >= thr, cfg["ship_cost_free"], cfg["ship_cost_paid"])


def _basket_after_bump(v, u_bump, u_pad, lo, hi_thr, cfg):
    """Shoppers whose natural basket is in [lo, hi_thr) pad it up to the threshold w.p. bump_prob."""
    bump = (v >= lo) & (v < hi_thr) & (u_bump < cfg["bump_prob"])
    return np.where(bump, hi_thr + 6.0 * u_pad, v)


def _outcomes(purchase, v, ret, thr, cfg):
    """Per-user outcomes given purchase flag, basket value, return flag, threshold."""
    ship = np.where(purchase, _shipping(v, thr, cfg), 0.0)
    gross = np.where(purchase, v, 0.0)
    net_rev = gross * (1 - ret)
    margin = np.where(
        purchase,
        cfg["gm_rate"] * v * (1 - ret) - ship - cfg["return_cost"] * ret,
        0.0,
    )
    return dict(purchased=purchase.astype(int), basket=gross, net_rev=net_rev,
                ship_cost=ship, returned=(purchase & (ret == 1)).astype(int), margin=margin)


def generate(cfg=CFG, seed=20261016, control_only=False, n_users=None):
    rng = np.random.default_rng(seed)
    n = n_users or cfg["n_users"]

    # ---------- pre-treatment covariates ----------
    is_new = rng.random(n) < 0.40
    device = rng.choice(["mobile", "desktop", "tablet"], n, p=[0.60, 0.32, 0.08])
    source = rng.choice(["organic", "paid", "email", "direct"], n, p=[0.35, 0.30, 0.15, 0.20])
    app_old = (device == "mobile") & (rng.random(n) < 0.15)
    zv = rng.standard_normal(n)  # latent customer value (never observed)
    # arrival day, mild weekday weighting
    w = np.array([1.0, 0.95, 0.95, 1.0, 1.05, 1.15, 1.10])
    day_w = np.tile(w, cfg["n_days"] // 7 + 1)[: cfg["n_days"]]
    arrival_day = rng.choice(cfg["n_days"], n, p=day_w / day_w.sum())
    dow = arrival_day % 7  # day 0 = Monday
    weekend = (dow >= 5).astype(float)

    # pre-experiment spend (observed): only returning customers have history
    has_hist = (~is_new) & (rng.random(n) < _sigmoid(0.4 + 1.2 * zv))
    pre_spend = np.where(has_hist, np.exp(rng.normal(4.2 + 0.7 * zv, 0.6)), 0.0)
    pre_orders = np.where(has_hist, np.maximum(1, np.round(pre_spend / 75.0)), 0).astype(int)

    # ---------- potential outcomes (common random numbers) ----------
    u_conv, u_bump, u_pad, u_ret, u_drop = (rng.random(n) for _ in range(5))
    z_basket = rng.standard_normal(n)
    u_big, u_bigv = rng.random(n), rng.random(n)

    src_eff = pd.Series(source).map({"organic": 0.0, "paid": -0.25, "email": 0.35, "direct": 0.15}).values
    dev_eff = pd.Series(device).map({"mobile": 0.0, "desktop": 0.25, "tablet": 0.0}).values
    logit0 = -4.60 + 1.00 * zv + 0.90 * (~is_new) + dev_eff + src_eff + 0.10 * weekend
    p0 = _sigmoid(logit0)

    delta = np.where(is_new, cfg["delta_new"], cfg["delta_ret"])
    novelty = 1.0 + cfg["novelty_amp"] * np.exp(-arrival_day / cfg["novelty_tau"])
    p1 = _sigmoid(logit0 + delta * novelty)          # effect during the experiment window
    p1_ss = _sigmoid(logit0 + delta)                 # steady-state (no novelty) counterfactual

    buy0 = u_conv < p0
    buy1 = u_conv < p1
    buy1_ss = u_conv < p1_ss

    # natural basket value, with rare giant (wholesale-style) orders
    mu = np.log(62.0) + 0.30 * zv + 0.10 * (device == "desktop")
    v_nat = np.exp(mu + 0.55 * z_basket)
    v_nat = np.where(u_big < 0.0002, 800.0 + 1700.0 * u_bigv, v_nat)

    v0 = _basket_after_bump(v_nat, u_bump, u_pad, 85.0, cfg["thr_control"], cfg)
    v1 = _basket_after_bump(v_nat, u_bump, u_pad, 65.0, cfg["thr_treat"], cfg)

    # returns: incremental buyers (only buy because of the policy) return more often
    ret0 = (u_ret < 0.12).astype(int)
    ret1 = np.where(buy0, (u_ret < 0.12), (u_ret < 0.16)).astype(int)
    ret1_ss = ret1

    Y0 = _outcomes(buy0, v0, ret0, cfg["thr_control"], cfg)
    Y1 = _outcomes(buy1, v1, ret1, cfg["thr_treat"], cfg)
    Y1ss = _outcomes(buy1_ss, v1, ret1_ss, cfg["thr_treat"], cfg)

    # ---------- randomisation ----------
    variant = (rng.random(n) < 0.5).astype(int)
    if control_only:
        variant[:] = 0

    obs = {k: np.where(variant == 1, Y1[k], Y0[k]) for k in Y0}

    # ---------- injected logging bug ----------
    affected = app_old & (variant == 1)
    p_drop = np.where(obs["purchased"] == 1, cfg["bug_drop_buyer"], cfg["bug_drop_nonbuyer"])
    dropped = affected & (u_drop < p_drop)

    df = pd.DataFrame(dict(
        user_id=np.arange(n), arrival_day=arrival_day, dow=dow, device=device,
        is_new=is_new.astype(int), source=source, app_old=app_old.astype(int),
        pre_spend_90d=np.round(pre_spend, 2), pre_orders_90d=pre_orders, variant=variant,
        **{k: (np.round(val, 2) if val.dtype.kind == "f" else val) for k, val in obs.items()},
    ))
    df["items"] = np.where(df.purchased == 1, 1 + rng.poisson(df.basket.values / 45.0), 0)
    df = df.loc[~dropped].reset_index(drop=True)

    # ---------- ground truth (from potential outcomes of ALL users, incl. dropped) ----------
    clean = ~app_old  # population unaffected by the bug (pre-treatment definition)
    segs = {
        "all": np.ones(n, bool), "clean (excl. old-app mobile)": clean,
        # segment truths are computed inside the clean population (the one the final analysis uses)
        "new": is_new & clean, "returning": ~is_new & clean,
        "mobile": (device == "mobile") & clean, "desktop": (device == "desktop") & clean,
        "tablet": (device == "tablet") & clean,
    }
    for s in ["organic", "paid", "email", "direct"]:
        segs[f"src_{s}"] = (source == s) & clean

    def summarise(mask, Yt):
        out = {}
        for m in ["margin", "net_rev", "purchased", "basket", "ship_cost", "returned"]:
            out[m + "_ctrl"] = float(Y0[m][mask].mean())
            out[m + "_treat"] = float(Yt[m][mask].mean())
        out["margin_ate"] = out["margin_treat"] - out["margin_ctrl"]
        out["margin_rel"] = out["margin_ate"] / out["margin_ctrl"]
        out["conv_ate"] = out["purchased_treat"] - out["purchased_ctrl"]
        out["conv_rel"] = out["conv_ate"] / out["purchased_ctrl"]
        out["rev_ate"] = out["net_rev_treat"] - out["net_rev_ctrl"]
        return out

    truth = {"experiment_window": {k: summarise(m, Y1) for k, m in segs.items()},
             "steady_state": {k: summarise(m, Y1ss) for k, m in segs.items()},
             "by_day_conv_rel": [
                 float((Y1["purchased"][arrival_day == d].mean() - Y0["purchased"][arrival_day == d].mean())
                       / Y0["purchased"][arrival_day == d].mean()) for d in range(cfg["n_days"])],
             "bug": {"n_dropped": int(dropped.sum()), "share_of_treat_arm": float(dropped.sum() / max(1, (variant == 1).sum()))},
             "config": cfg, "seed": seed}
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

    # historical "pilot" sample (control experience only) used for pre-experiment planning
    pilot, _ = generate(seed=a.seed + 1, control_only=True, n_users=150_000)
    pilot.to_csv(os.path.join(a.out, "pilot_history.csv.gz"), index=False)
    print(f"experiment rows: {len(df):,}  pilot rows: {len(pilot):,}")
    print(df.groupby('variant').size())
