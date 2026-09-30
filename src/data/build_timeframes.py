r"""
Raw M5 data (data/raw) se saare timeframes banata hai aur data/processed mein save karta hai.

Chalane ka tareeka (ai-forex-trader folder se, venv active karke):
    python src\data\build_timeframes.py                    -> XAUUSD
    python src\data\build_timeframes.py --symbol USDJPY    -> USDJPY

Steps:
  1. Saare saal ki BID aur ASK files jodna
  2. Data saaf karna (duplicates, market band wali flat candles, galat prices)
  3. BID + ASK se spread nikalna
  4. M5 se M15, H1, H4, D1, W1 banana (resampling)
  5. Har timeframe ki ek Parquet file save karna

Output ki har candle mein do time hote hain (dono UTC):
  time        -> candle kab KHULI
  close_time  -> candle kab BAND hui. Higher timeframe ki candle tabhi use kar sakte hain
                 jab uska close_time aa chuka ho (lookahead se bachne ke liye).
  trade_date  -> (sirf D1/W1) TradingView/MT5 wali date. D1 ki candle pichli shaam khulti hai,
                 isliye `time` ek din pehle ka hota hai; trade_date asli trading din batata hai.
"""

import argparse
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
BASE_TF = "M5"
BASE_MINUTES = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Forex ka "din" New York 5 PM pe khatam hota hai (MT5 brokers bhi yahi use karte hain).
# NY time mein 7 ghante jodo toh 5 PM -> midnight ban jata hai. Is "trading clock" pe
# candles banayenge toh D1, H4, W1 sab broker charts se match karenge, DST ke saath bhi.
NY_TZ = "America/New_York"
NY_CLOSE_SHIFT = pd.Timedelta(hours=7)

# Humara naam -> pandas ka rule
TARGET_TFS = {
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "D",
    "W1": "W",
}


# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
def load_side(symbol: str, side: str) -> pd.DataFrame:
    """Ek pair ki ek side (BID ya ASK) ki saari saal wali files padh ke ek table banata hai."""
    files = sorted(RAW_DIR.glob(f"{symbol}_{BASE_TF}_{side}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{RAW_DIR} mein {symbol}_{BASE_TF}_{side}_*.parquet nahi mili")

    print(f"  {side}: {len(files)} files")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time"
    return df.sort_index()


# ----------------------------------------------------------------------
# 2 + 3. Clean aur spread
# ----------------------------------------------------------------------
def clean(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    """BID aur ASK ko jodta hai, kharab candles hatata hai, spread nikalta hai."""
    before = len(bid)

    # Same time ki candle do baar aa gayi (saal ki files overlap) toh ek hi rakho
    bid = bid[~bid.index.duplicated(keep="last")]
    ask = ask[~ask.index.duplicated(keep="last")]
    dupes = before - len(bid)

    # Chart BID price pe banta hai (MT5 bhi BID dikhata hai). ASK se sirf spread chahiye.
    df = bid[["open", "high", "low", "close", "volume"]].copy()
    df["spread"] = ask["close"].reindex(df.index) - df["close"]

    # Market band ho tab bhi Dukascopy flat candles deta hai (volume 0, O=H=L=C). Asli trading nahi hai.
    flat = (df["volume"] <= 0) & (df["high"] == df["low"])
    df = df[~flat]

    # Galat prices: 0 ya negative, ya high < low
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
    df = df[~bad]

    # Spread missing (ASK candle nahi thi) toh aas-paas ke spread se bhar do
    missing_spread = int(df["spread"].isna().sum())
    df["spread"] = df["spread"].ffill().bfill()

    print(f"  Duplicates hataye:            {dupes:,}")
    print(f"  Market-band flat candles:     {int(flat.sum()):,}")
    print(f"  Galat price wali candles:     {int(bad.sum()):,}")
    print(f"  Spread missing (bhara gaya):  {missing_spread:,}")
    print(f"  Negative spread (check karo): {int((df['spread'] < 0).sum()):,}")
    return df


def report_gaps(df: pd.DataFrame) -> None:
    """Weekend ke alawa bade gaps dhoondta hai (data missing ho sakta hai)."""
    gaps = df.index.to_series().diff()
    big = gaps[gaps > pd.Timedelta(hours=2)]
    # Weekend gap normal hai (~48+ ghante). Uske alawa 2 ghante se bade gaps dikhao.
    unusual = big[big < pd.Timedelta(hours=40)]
    print(f"  2h se bade gaps (weekend chhod ke): {len(unusual)}")
    for t, g in unusual.head(10).items():
        print(f"    {t}  gap = {g}")


# ----------------------------------------------------------------------
# 4. Resample
# ----------------------------------------------------------------------
def trading_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC time -> 'trading clock' (NY time + 7h, bina timezone ke). Iska midnight = NY 5 PM."""
    return (index.tz_convert(NY_TZ) + NY_CLOSE_SHIFT).tz_localize(None)


def resample(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """M5 candles ko bade timeframe mein jodta hai."""
    clock = trading_clock(m5.index)

    # Har M5 candle kis bade candle ka hissa hai -> uski "bucket" key
    if rule == "W":
        # Hafta Sunday 5 PM NY se shuru = trading clock pe Monday 00:00
        key = clock.to_period("W-SUN").start_time
    else:
        key = clock.floor(rule)

    df = m5.reset_index()
    df["bucket"] = key
    out = df.groupby("bucket").agg(
        time=("time", "first"),          # pehli M5 candle ka time = bade candle ka open time
        last_m5=("time", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        spread=("spread", "mean"),
        m5_count=("open", "size"),       # kitni M5 candles se bani (quality check)
    )
    out["close_time"] = out["last_m5"] + pd.Timedelta(minutes=BASE_MINUTES)

    # D1/W1 ke liye trading date (TradingView/MT5 wali). Candle pichli shaam 6 PM NY pe
    # khulti hai, isliye `time` ek din pehle ka dikhta hai. Bucket (trading clock) ki date
    # hi asli trading date hai. W1 ke liye ye hafte ka Monday hai.
    if rule in ("D", "W"):
        out["trade_date"] = out.index.normalize()

    out = out.drop(columns="last_m5").set_index("time")
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def build(symbol: str) -> None:
    """Ek pair ke saare timeframes banata hai."""
    print(f"\n########## {symbol} ##########")
    print("\n[1] Loading raw data")
    bid = load_side(symbol, "BID")
    ask = load_side(symbol, "ASK")

    print("\n[2] Cleaning")
    m5 = clean(bid, ask)
    report_gaps(m5)
    m5["close_time"] = m5.index + pd.Timedelta(minutes=BASE_MINUTES)
    m5["m5_count"] = 1

    print("\n[3] Saving timeframes")
    out_file = PROCESSED_DIR / f"{symbol}_{BASE_TF}.parquet"
    m5.to_parquet(out_file)
    print(f"  {BASE_TF:>3}: {len(m5):>9,} candles -> {out_file.name}")

    for name, rule in TARGET_TFS.items():
        df = resample(m5, rule)
        out_file = PROCESSED_DIR / f"{symbol}_{name}.parquet"
        df.to_parquet(out_file)
        print(f"  {name:>3}: {len(df):>9,} candles -> {out_file.name}")

    print(f"\nData range: {m5.index[0]}  ->  {m5.index[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw M5 se saare timeframes banao")
    parser.add_argument("--symbol", nargs="+", default=["XAUUSD"],
                        help="Ek ya zyada pairs (default: XAUUSD)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbol:
        build(symbol)
    print("\nDone!")


if __name__ == "__main__":
    main()
