r"""
Dukascopy se historical candles download karke data/raw mein save karta hai.

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\data\download_dukascopy.py            -> default M5
    python src\data\download_dukascopy.py --tf H1    -> 1 ghante ki candles

Har saal ka data alag file mein save hota hai. Beech mein band ho gaya toh
dobara chalao - jo saal already download ho chuke hain wo skip ho jayenge.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import dukascopy_python as dk
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

# ----------------------------------------------------------------------
# Settings - naya pair chahiye toh sirf yahan badlo
# ----------------------------------------------------------------------
SYMBOL = "XAUUSD"
INSTRUMENT = INSTRUMENT_FX_METALS_XAU_USD
START_YEAR = 2016

# Humara naam (file ke liye) -> Dukascopy ka naam (download ke liye)
TIMEFRAMES = {
    "M1": dk.INTERVAL_MIN_1,
    "M5": dk.INTERVAL_MIN_5,
    "M15": dk.INTERVAL_MIN_15,
    "H1": dk.INTERVAL_HOUR_1,
}

# BID = sell price, ASK = buy price. Dono ka farak = spread (trading cost).
SIDES = {
    "BID": dk.OFFER_SIDE_BID,
    "ASK": dk.OFFER_SIDE_ASK,
}

# Ye file src/data/ mein hai -> 2 folder upar jaake project root milta hai
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_year(year: int, timeframe: str, side_name: str, side: str) -> None:
    """Ek saal + ek timeframe + ek side (BID/ASK) ka data download karke Parquet mein save karta hai."""
    now = datetime.now(timezone.utc)
    out_file = RAW_DIR / f"{SYMBOL}_{timeframe}_{side_name}_{year}.parquet"

    # Purane saal ki file already hai toh dobara download mat karo.
    # Current saal ko hamesha dobara download karte hain kyunki usme naya data aata rehta hai.
    if out_file.exists() and year < now.year:
        print(f"  {out_file.name} pehle se hai, skip")
        return

    # Timezone UTC dena zaroori hai. Bina timezone ke Python laptop ka time (IST)
    # maan leta hai aur data 5.5 ghante shift ho jata.
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = min(datetime(year + 1, 1, 1, tzinfo=timezone.utc), now)

    print(f"  Downloading {SYMBOL} {timeframe} {side_name} {year} ...", flush=True)
    df = dk.fetch(INSTRUMENT, TIMEFRAMES[timeframe], side, start, end, max_retries=5)

    if df.empty:
        print(f"  {year} ka koi data nahi mila")
        return

    df.to_parquet(out_file)
    print(f"  Saved {len(df):,} candles -> {out_file.name}")


def main() -> None:
    # Command line se timeframe lena: --tf M5 / --tf H1 etc.
    parser = argparse.ArgumentParser(description="Dukascopy se candles download karo")
    parser.add_argument("--tf", default="M5", choices=TIMEFRAMES.keys(),
                        help="Timeframe (default: M5)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current_year = datetime.now(timezone.utc).year

    for year in range(START_YEAR, current_year + 1):
        print(f"\n=== {year} ===")
        for side_name, side in SIDES.items():
            download_year(year, args.tf, side_name, side)
            time.sleep(1)  # Dukascopy server pe load na pade isliye thoda rukna

    print("\nDone! Files yahan hain:", RAW_DIR)


# Ye file seedha chalayi gayi ho (import nahi hui) tabhi main() chalega
if __name__ == "__main__":
    main()
