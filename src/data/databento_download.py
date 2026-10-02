r"""
Databento DOWNLOAD: CME futures ka har trade (asli orderflow), mahine-mahine ki files mein.

Kya milta hai har trade mein:
    ts_event : trade kab hua (UTC, nanosecond tak)
    price    : kis price pe
    size     : kitne contracts
    side     : AGGRESSOR kaun tha (yahi asli orderflow hai):
                 B = buyer ne market order se khareeda (buyer aggressive)
                 A = seller ne market order se becha (seller aggressive)
                 N = pata nahi (bahut kam)

SAFETY (paisa):
  1. Pehle har mahine ki COST poochta hai (free), total dikhata hai.
  2. Total --max-cost se zyada -> kuch download nahi.
  3. Bina --yes ke sirf PLAN dikhata hai, download nahi karta.
  4. Jo mahina pehle se download hai (file maujood), use skip (dobara paisa nahi lagega).
  5. Beech mein ruk gaya toh dobara chalao: wahin se aage chalega.

Output: data/orderflow/raw/{6E|6J|6B}_{YYYY-MM}.parquet

Chalane ka tareeka (venv active + DATABENTO_API_KEY set):
    python src\data\databento_download.py                 (sirf plan + cost)
    python src\data\databento_download.py --yes           (asli download)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "orderflow" / "raw"
DATASET = "GLBX.MDP3"
FUTURES = {"6E": "6E.v.0", "6J": "6J.v.0", "6B": "6B.v.0"}


def months(start: str, end: str):
    import pandas as pd
    s = pd.Timestamp(start).to_period("M")
    e = pd.Timestamp(end).to_period("M")
    for p in pd.period_range(s, e, freq="M"):
        a = max(p.start_time, pd.Timestamp(start))
        b = min(p.end_time.normalize() + pd.Timedelta(days=1), pd.Timestamp(end))
        if a < b:
            yield str(p), a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download CME futures trades (orderflow) from Databento")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="default = aaj se 2 din pehle")
    ap.add_argument("--futures", nargs="+", default=list(FUTURES), choices=list(FUTURES))
    ap.add_argument("--max-cost", type=float, default=95.0, help="isse zyada total cost -> download nahi")
    ap.add_argument("--yes", action="store_true", help="asli download karo (bina iske sirf plan)")
    a = ap.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit('DATABENTO_API_KEY nahi mila. Is terminal mein: set DATABENTO_API_KEY=db-...')
    import databento as db
    import pandas as pd

    client = db.Historical()
    end = a.end or (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    OUT.mkdir(parents=True, exist_ok=True)

    def cost_of(fut, s, e):
        for attempt in range(3):                      # internet hichki -> 3 baar try
            try:
                return client.metadata.get_cost(dataset=DATASET, symbols=[FUTURES[fut]], stype_in="continuous",
                                                schema="trades", start=s, end=e)
            except Exception as ex:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"   network issue ({type(ex).__name__}), retrying...", flush=True)
                time.sleep(3)

    # 1) Plan: kaunse mahine baaki hain + unki cost
    todo, total, done = [], 0.0, 0
    for fut in a.futures:
        all_months = list(months(a.start, end))
        missing = [(lb, s, e) for lb, s, e in all_months if not (OUT / f"{fut}_{lb}.parquet").exists()]
        done += len(all_months) - len(missing)
        if not missing:
            continue
        print(f"{fut}: checking cost of {len(missing)} months...", flush=True)
        if len(missing) == len(all_months):
            # kuch download nahi hua abhi tak -> poore period ki cost EK sawal mein, mahine ki cost baad mein
            whole = cost_of(fut, all_months[0][1], all_months[-1][2])
            per = whole / len(missing)
            for lb, s, e in missing:
                todo.append((fut, lb, s, e, OUT / f"{fut}_{lb}.parquet", per))
            total += whole
            print(f"   {fut}: ${whole:,.2f}", flush=True)
        else:
            for lb, s, e in missing:
                c = cost_of(fut, s, e)
                todo.append((fut, lb, s, e, OUT / f"{fut}_{lb}.parquet", c))
                total += c
            print(f"   {fut}: ${sum(t[5] for t in todo if t[0] == fut):,.2f}", flush=True)
    print(f"Period {a.start} -> {end}, futures {', '.join(a.futures)}")
    print(f"Already downloaded: {done} files. To download: {len(todo)} files, cost ${total:,.2f} (limit ${a.max_cost:g})")
    if not todo:
        print("Sab download ho chuka hai.")
        return
    if total > a.max_cost:
        sys.exit(f"STOP: ${total:,.2f} > limit ${a.max_cost:g}. Kuch download nahi kiya.")
    if not a.yes:
        print("\nYe sirf PLAN tha. Download ke liye dobara chalao: --yes ke saath")
        print("(Databento Billing -> Manage mein monthly limit pehle itni badha lo ki ye cost aa jaaye.)")
        return

    # 2) Download, ek mahina ek file
    spent = 0.0
    for i, (fut, label, s, e, f, cost) in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {fut} {label}: downloading...", flush=True)
        data = client.timeseries.get_range(dataset=DATASET, symbols=[FUTURES[fut]], stype_in="continuous",
                                           schema="trades", start=s, end=e)
        df = data.to_df()
        df = df.reset_index()[["ts_event", "price", "size", "side", "instrument_id"]]
        df["size"] = df["size"].astype("int32")
        df["side"] = df["side"].astype("category")
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(f)                       # poori file likhi gayi tabhi naam final (aadhi file nahi bachegi)
        spent += cost
        buy = (df["side"] == "B").mean() * 100
        print(f"[{i}/{len(todo)}] {fut} {label}: {len(df):,} trades, buyer-aggressor {buy:.0f}%, "
              f"${cost:.2f} (total ${spent:.2f}), {time.time() - t0:.0f}s")
    print(f"\nDone. Files in {OUT}")


if __name__ == "__main__":
    main()
