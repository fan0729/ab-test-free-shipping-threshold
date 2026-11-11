"""Stage 1 of the semi-synthetic pipeline: turn the REAL H&M Kaggle data into a small per-customer table.

Usage (streams the zip, no need to unzip the 3.5GB file; rows are date-sorted so we stop early):
    unzip -p transactions_train.csv.zip | python prepare_hm_base.py --customers customers.csv --out hm_base

Definitions (all from real data):
  * online order = one customer's online (sales_channel_id == 2) purchases on one day, summed
  * experiment window = 2019-10-14..2019-10-27 (14 days); pilot window = 2019-09-30..2019-10-13
  * pre_spend_90d / pre_orders_90d = online spend / order-days in the 90 days before the window
  * has_history = any online purchase between 2018-09-20 and the window start (is_new = 1 - has_history)
  * outcome = FIRST online order in the window (value in H&M's anonymised price units, item count, day)
"""
import argparse, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--customers", required=True)
ap.add_argument("--out", default="hm_base")
ap.add_argument("--stop-date", default="2019-10-28")
ap.add_argument("--max-rows", type=int, default=None)
a = ap.parse_args()
T0 = pd.Timestamp("2018-09-20")
t_start = time.time()

def chash(s):
    return pd.util.hash_pandas_object(s, index=False).values

parts, nrows = [], 0
reader = pd.read_csv(sys.stdin, chunksize=2_000_000, usecols=["t_dat", "customer_id", "price", "sales_channel_id"],
                     dtype={"t_dat": str, "customer_id": str, "price": "float64", "sales_channel_id": "int8"})
for ch in reader:
    nrows += len(ch)
    stop = ch.t_dat.values[-1] >= a.stop_date
    ch = ch[(ch.sales_channel_id == 2) & (ch.t_dat < a.stop_date)]
    if len(ch):
        day = (pd.to_datetime(ch.t_dat, format="%Y-%m-%d") - T0).dt.days.values.astype("int16")
        d = pd.DataFrame({"h": chash(ch.customer_id), "day": day, "value": ch.price.values})
        parts.append(d.groupby(["h", "day"], sort=False).agg(value=("value", "sum"), items=("value", "size")).reset_index())
    print(f"rows read {nrows:,}  last date {ch.t_dat.values[-1] if len(ch) else '-'}  {time.time()-t_start:.0f}s", file=sys.stderr, flush=True)
    if stop or (a.max_rows and nrows >= a.max_rows):
        break
o = pd.concat(parts).groupby(["h", "day"], sort=False).agg(value=("value", "sum"), items=("items", "sum")).reset_index()
print(f"online customer-day orders: {len(o):,}", file=sys.stderr)

cu = pd.read_csv(a.customers, dtype={"customer_id": str})
cu["h"] = chash(cu.customer_id)
assert cu.h.is_unique, "hash collision"
cu = cu.drop(columns=["customer_id", "postal_code"]).set_index("h")  # postal code is hashed/anonymised: no region available

def build(ws_s, we_s, label):
    ws, we = (pd.Timestamp(ws_s) - T0).days, (pd.Timestamp(we_s) - T0).days
    pre = o[(o.day >= ws - 90) & (o.day < ws)].groupby("h").agg(pre_spend_90d=("value", "sum"), pre_orders_90d=("day", "size"))
    hist = pd.Series(1, index=o.loc[o.day < ws, "h"].unique(), name="has_history")
    w = o[(o.day >= ws) & (o.day <= we)].sort_values("day")
    first = w.drop_duplicates("h").set_index("h").rename(columns={"value": "basket", "day": "order_day", "items": "n_items"})
    first["n_orders"] = w.groupby("h").size()
    first["order_day"] = first.order_day - ws
    base = cu.join(pre).join(hist).join(first)
    base["purchased"] = base.basket.notna().astype(int)
    for c in ["pre_spend_90d", "pre_orders_90d", "has_history", "n_items", "n_orders"]:
        base[c] = base[c].fillna(0)
    base["pre_orders_90d"] = base.pre_orders_90d.astype(int)
    base["has_history"] = base.has_history.astype(int)
    base["basket"] = base.basket.fillna(0.0)
    base = base.reset_index(drop=True)
    base.to_csv(f"{a.out}_{label}.csv.gz", index=False)
    print(label, base.shape, "purchase rate", round(base.purchased.mean(), 4), "no-history share", round(1 - base.has_history.mean(), 3),
          "median basket (units)", round(base.loc[base.purchased == 1, "basket"].median(), 5), file=sys.stderr)

build("2019-10-14", "2019-10-27", "experiment")
build("2019-09-30", "2019-10-13", "pilot")
print(f"done in {time.time()-t_start:.0f}s", file=sys.stderr)
