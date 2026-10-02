r"""
Databento COST CHECK (kuch download NAHI karta, koi credit NAHI katta).

Kya poochta hai: CME futures ke har TRADE ka data (price, size, aur SIDE = buyer aggressive tha ya
seller) kitne GB ka hai aur kitne $ ka padega, har saal ke hisaab se.

Futures:
    6E = Euro FX        (EURUSD jaisa)
    6J = Japanese Yen   (USDJPY ka ULTA: 6J = JPY/USD, isliye 6J upar = USDJPY neeche)
    6B = British Pound  (GBPUSD jaisa)
"6E.v.0" = continuous contract: jis mahine ke contract mein sabse zyada volume hai, wahi
(roll apne aap ho jaata hai; har 3 mahine contract badalta hai).

API key: Windows environment variable DATABENTO_API_KEY se (setx se save kiya tha).
Kabhi bhi key code mein mat likho.

Chalane ka tareeka (venv active karke, project folder se):
    python src\data\databento_cost.py
    python src\data\databento_cost.py --start 2022-01-01      (chhota period)
    python src\data\databento_cost.py --schema tbbo            (trades + us waqt ka best bid/ask)
"""
from __future__ import annotations

import argparse
import os
import sys

DATASET = "GLBX.MDP3"                       # CME Globex (Databento ka naam)
SYMBOLS = {"6E.v.0": "EURUSD", "6J.v.0": "USDJPY (inverse)", "6B.v.0": "GBPUSD"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Databento cost check (free, downloads nothing)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None, help="default = aaj se 2 din pehle")
    ap.add_argument("--schema", default="trades", help="trades (default) ya tbbo")
    a = ap.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("DATABENTO_API_KEY nahi mila. CMD mein: setx DATABENTO_API_KEY \"db-...\"  phir NAYA terminal kholo.")
    try:
        import databento as db
        import pandas as pd
    except ImportError:
        sys.exit("databento package nahi hai. Chalao: pip install databento")

    client = db.Historical()                 # key environment variable se khud padhta hai
    end = a.end or (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    years = list(range(int(a.start[:4]), int(end[:4]) + 1))

    print(f"Dataset {DATASET}, schema '{a.schema}', {a.start} -> {end}\n")
    print(f"{'symbol':8s} {'year':>5s} {'trades (records)':>17s} {'size GB':>9s} {'cost $':>8s}")
    total_cost, total_gb = 0.0, 0.0
    per_symbol = {}
    for sym, name in SYMBOLS.items():
        s_cost = 0.0
        for y in years:
            s = max(a.start, f"{y}-01-01")
            e = min(end, f"{y + 1}-01-01")
            if s >= e:
                continue
            kw = dict(dataset=DATASET, symbols=[sym], stype_in="continuous", schema=a.schema, start=s, end=e)
            n = client.metadata.get_record_count(**kw)
            gb = client.metadata.get_billable_size(**kw) / 1e9
            cost = client.metadata.get_cost(**kw)
            s_cost += cost
            total_gb += gb
            print(f"{sym:8s} {y:5d} {n:17,d} {gb:9.2f} {cost:8.2f}")
        per_symbol[f"{sym} ({name})"] = s_cost
        total_cost += s_cost
        print()

    print("Per future:")
    for k, v in per_symbol.items():
        print(f"  {k:28s} ${v:,.2f}")
    print(f"\nTOTAL: {total_gb:,.1f} GB  ->  ${total_cost:,.2f}   (free credit: $125)")
    if total_cost <= 100:
        print("OK: free credit mein aa jayega (humari safety limit $100).")
    else:
        print("ZYADA hai: chhota period try karo, jaise --start 2022-01-01")


if __name__ == "__main__":
    main()
