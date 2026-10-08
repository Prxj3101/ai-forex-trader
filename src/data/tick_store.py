r"""
TICK STORE: JForex tick CSV (EET time) -> fast compact files per pair per month.

data/ticks/store/{SYMBOL}/{YYYY-MM}.npz  with arrays:
  t   int64  UTC time in milliseconds
  ask float64, bid float64
Converted once; the tick backtest reads only the months it needs (much faster than 43 GB of CSV).

Chalana:  python src\data\tick_store.py EURUSD 2019      (one yearly CSV -> 12 monthly files)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "ticks" / "jforex"
STORE = ROOT / "data" / "ticks" / "store"
EET = "Europe/Helsinki"


def source_files(sym, year):
    """Yearly CSV if present, else the 12 monthly CSVs of that year."""
    f = SRC / f"{sym}_Ticks_{year}.01.01_{year + 1}.01.01.csv"
    if f.exists():
        return [f]
    if year == 2026:
        c = sorted(SRC.glob(f"{sym}_Ticks_2026.01.01_*.csv"), key=lambda p: p.stat().st_size)
        return c[-1:]
    return sorted(SRC.glob(f"{sym}_Ticks_{year}.??.01_*.csv"))


def _eet_to_utc_ms(ts: np.ndarray) -> np.ndarray:
    """ts: bytes array 'YYYY.MM.DD HH:MM:SS.fff' (EET/EEST local). Vectorised parse.
    DST switches happen on Sunday early morning (FX market closed), so no ambiguous ticks."""
    b = np.frombuffer(ts.astype("S23").tobytes(), dtype=np.uint8).reshape(-1, 23).astype(np.int64) - 48
    num = lambda i, j: (b[:, i:j] * (10 ** np.arange(j - i - 1, -1, -1))).sum(1)
    y, mo, d = num(0, 4), num(5, 7), num(8, 10)
    hh, mm, ss, ms = num(11, 13), num(14, 16), num(17, 19), num(20, 23)
    days = (pd.to_datetime(pd.DataFrame({"year": y, "month": mo, "day": d})).to_numpy()
            .astype("datetime64[D]").astype(np.int64))
    naive = days * 86_400_000 + ((hh * 60 + mm) * 60 + ss) * 1000 + ms
    off = np.full(len(naive), 2 * 3_600_000, dtype=np.int64)
    for yr in np.unique(y):
        # EU DST: last Sunday of March 03:00 local -> last Sunday of October 04:00 local
        mar = pd.Timestamp(f"{yr}-03-31"); mar -= pd.Timedelta(days=(mar.weekday() + 1) % 7)
        octo = pd.Timestamp(f"{yr}-10-31"); octo -= pd.Timedelta(days=(octo.weekday() + 1) % 7)
        a = (mar + pd.Timedelta(hours=3)).value // 1_000_000
        z = (octo + pd.Timedelta(hours=4)).value // 1_000_000
        off[(y == yr) & (naive >= a) & (naive < z)] = 3 * 3_600_000
    return naive - off


def convert(sym, year):
    import pyarrow.csv as pc
    out = STORE / sym
    out.mkdir(parents=True, exist_ok=True)
    buf, counts = {}, {}
    for f in source_files(sym, year):
        _convert_file(pc, f, out, buf, counts, year)
    for k in list(buf):
        _flush(out, buf, counts, k)
    (out / f"_done_{year}").write_text(str(counts))
    print(sym, year, counts, flush=True)


def _flush(out, buf, counts, k):
    parts = buf.pop(k)
    t = np.concatenate([p[0] for p in parts]); o = np.argsort(t, kind="stable")
    np.savez(out / f"{k}.npz", t=t[o], ask=np.concatenate([p[1] for p in parts])[o],
             bid=np.concatenate([p[2] for p in parts])[o])
    counts[k] = int(len(t))


def _convert_file(pc, f, out, buf, counts, year):
    rd = pc.open_csv(f, read_options=pc.ReadOptions(block_size=64 << 20), convert_options=pc.ConvertOptions(
        include_columns=["Time (EET)", "Ask", "Bid"],
        column_types={"Time (EET)": "string", "Ask": "float64", "Bid": "float64"}))
    for batch in rd:
        t = _eet_to_utc_ms(batch.column(0).to_numpy(zero_copy_only=False).astype("S23"))
        a = batch.column(1).to_numpy(); b = batch.column(2).to_numpy()
        ym = t.astype("datetime64[ms]").astype("datetime64[M]").astype(str)
        keep = np.char.startswith(ym, str(year))     # boundary ticks of another UTC year are dropped
        t, a, b, ym = t[keep], a[keep], b[keep], ym[keep]
        if len(t) == 0:
            continue
        for k in np.unique(ym):
            m = ym == k
            buf.setdefault(k, []).append((t[m], a[m], b[m]))
        for k in [k for k in buf if k < ym[-1]]:
            _flush(out, buf, counts, k)

if __name__ == "__main__":
    convert(sys.argv[1], int(sys.argv[2]))
