r"""
VISION BOT ("Human Brain with Eyes") - step 1: pictures + tick labels for every zone.
Plan + safety rules: CLAUDE.md -> "VISION BOT PLAN".

Picture of ONE zone at the moment of the fill (limit entry at the zone edge):
  panels  H4 | H1 | M15 (last NB CLOSED bars, close_time <= fill)  +  ZOOM: 1-min bars built from BID ticks,
          last 60 min strictly before the fill tick (the last minute may be unfinished = what a human sees live).
  model image  uint8 [2 channels, 64, 4*3*NB]: ch0 = OHLC bars (3 px per bar: open tick | high-low | close tick),
               ch1 = zone box (128) + SL line (255).  No prices, no dates, no pair name. Each panel scaled on its own.
Label from TICKS (tick_engine rules, 3-day hold): best R reached before the SL is hit
  0 = SL before 1R | 1 = 1R..2R | 2 = 2R..3R | 3 = 3R+ (or best R at timeout, floored).
Check: label>=1 must equal r_tp1 == TP and label>=2 must equal r_tp2 == TP in data/research/<SYM>_v2.parquet.
Chalana: python src\research\vision\vision_data.py samples      (6 sample pictures from TRAIN years 2016-2020 only)
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.backtest.tick_engine import ticks, HOLD_MS  # noqa: E402

OUT = ROOT / "data/research/vision"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
NB, H = 60, 64
TFS = ["H4", "H1", "M15"]


@lru_cache(maxsize=12)
def bars(sym, tf):
    b = pd.read_parquet(ROOT / f"data/processed/{sym}_{tf}.parquet", columns=["open", "high", "low", "close", "close_time"])
    return b


def zones(years=None):
    rows = []
    for s in PAIRS:
        v = pd.read_parquet(ROOT / f"data/research/{s}_v2.parquet")
        v = v[(v.zone_tf.astype(str) == "M15") & (v.touch_number == 1) & v.r_tp1.notna()].copy()
        v["symbol"] = s
        rows.append(v)
    d = pd.concat(rows, ignore_index=True)
    d = d.sort_values(["symbol", "fill_ms", "zone_candle_time"]).drop_duplicates(["symbol", "fill_ms"], keep="last")  # 2 zones, same fill tick -> keep newest
    if years:
        d = d[d.fill_time.dt.year.isin(years)]
    return d.reset_index(drop=True)


def fill_ms(r):
    return int(r.fill_ms) if pd.notna(r.get("fill_ms", np.nan)) else int(r.fill_time.value // 10**6)


# ---------------------------------------------------------------- label (ticks only)
def label(r):
    bull = str(r.direction) == "bull"
    f = fill_ms(r)
    z = ticks(r.symbol, f, f + HOLD_MS)
    if z is None or len(z[0]) == 0:
        return dict(label=np.nan, best_r=np.nan, why="missing")
    t, a, b = z
    risk = abs(r.entry - r.sl)
    px = b if bull else a
    fav = (px - r.entry) / risk if bull else (r.entry - px) / risk
    sl = fav <= -1.0
    i = int(np.argmax(sl)) if sl.any() else len(fav)
    best = float(fav[:i].max()) if i > 0 else -1.0          # best BEFORE the SL tick
    return dict(label=int(np.clip(np.floor(max(best, 0)), 0, 3)), best_r=best, why="sl" if i < len(fav) else "timeout")


# ---------------------------------------------------------------- picture
def panel_bars(r, tf):
    b = bars(r.symbol, tf)
    j = int(b.close_time.searchsorted(r.fill_time, side="right"))   # close_time is sorted (checked)
    return b.iloc[max(0, j - NB):j]                           # ONLY closed bars (close_time <= fill bar start)


def zoom_bars(r):
    f = fill_ms(r)
    z = ticks(r.symbol, f - 60 * 60_000, f)                   # strictly before the fill tick
    if z is None or len(z[0]) < 5:
        return None
    t, a, bid = z
    assert t.max() < f, "future tick in zoom"                 # safety: only ticks before the fill
    s = pd.Series(bid, index=pd.to_datetime(t, unit="ms", utc=True))
    o = s.resample("1min").ohlc().dropna()
    return o.iloc[-NB:]


def _scale(lo, hi, p):
    return np.clip(np.round((H - 1) - (p - lo) / (hi - lo) * (H - 1)), 0, H - 1).astype(int)


def render_panel(df, r):
    img = np.zeros((2, H, 3 * NB), np.uint8)
    if df is None or len(df) == 0:
        return img
    off = NB - len(df)                                        # right-aligned like a live chart
    has_zone = pd.notna(r.get("top", np.nan))                 # pretraining windows have no zone -> ch1 stays empty
    lo = min(df.low.min(), r.bottom, r.sl) if has_zone else df.low.min()
    hi = max(df.high.max(), r.top, r.sl) if has_zone else df.high.max()
    if hi <= lo:
        return img
    o, h, l, c = (_scale(lo, hi, df[k].to_numpy()) for k in ("open", "high", "low", "close"))
    for i in range(len(df)):
        x = 3 * (i + off)
        img[0, o[i], x] = 255
        img[0, h[i]:l[i] + 1, x + 1] = 255
        img[0, c[i], x + 2] = 255
    if not has_zone:
        return img
    zt, zb = _scale(lo, hi, np.array([r.top, r.bottom]))
    start = df.index.searchsorted(r.zone_candle_time)       # zone box starts at the zone candle (or panel left edge)
    xs = 3 * (min(int(start), len(df) - 1) + off)
    img[1, zt:zb + 1, xs:] = 128
    img[1, _scale(lo, hi, np.array([r.sl]))[0], :] = 255
    return img


def picture(r):
    panels = {tf: panel_bars(r, tf) for tf in TFS}
    panels["ZOOM"] = zoom_bars(r)
    # safety: nothing at or after the fill
    for tf in TFS:
        assert (panels[tf].close_time <= r.fill_time).all(), "future bar in picture"   # fill_time = start of the fill M5 bar
    img = np.concatenate([render_panel(panels[k], r) for k in TFS + ["ZOOM"]], axis=2)
    return img, panels


# ---------------------------------------------------------------- human version (for Prajwal's eye check)
def human_png(r, panels, lab, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), facecolor="#0e1117")
    for ax, k in zip(axs, TFS + ["ZOOM"]):
        df = panels[k]
        ax.set_facecolor("#0e1117"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.set_title("M1 zoom (last 60 min)" if k == "ZOOM" else k, color="#ccc", fontsize=10)
        if df is None or len(df) == 0:
            continue
        x = np.arange(len(df))
        col = np.where(df.close >= df.open, "#2fd39a", "#ff5c7a")
        ax.vlines(x, df.low, df.high, color=col, lw=0.8)
        ax.bar(x, np.maximum((df.close - df.open).abs(), (df.high.max() - df.low.min()) * 0.002),
               bottom=np.minimum(df.open, df.close), color=col, width=0.6)
        s = min(int(df.index.searchsorted(r.zone_candle_time)), len(df) - 1)
        ax.add_patch(Rectangle((s - 0.5, r.bottom), len(df) - s, r.top - r.bottom, color="#4aa3ff", alpha=0.25))
        ax.axhline(r.entry, color="#00e5ff", lw=0.8); ax.axhline(r.sl, color="#ff5c7a", lw=0.8, ls="--")
        lo = min(df.low.min(), r.sl); hi = max(df.high.max(), r.top)
        ax.set_ylim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05)
    names = {0: "SL first (loss)", 1: "reached 1R (not 2R)", 2: "reached 2R (not 3R)", 3: "3R+ runner"}
    fig.suptitle(f"{r.vid}  {r.direction.upper()}   ->  label {lab['label']}: {names.get(lab['label'], '?')}   (best {lab['best_r']:.2f}R before SL)",
                 color="white", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=90, facecolor=fig.get_facecolor()); plt.close(fig)


def samples(n=6, seed=20261010):
    OUT.mkdir(parents=True, exist_ok=True)
    d = zones(years=range(2016, 2021))                       # TRAIN years only - exam data stays unseen
    print("zones (M15 first touch, all years):", len(zones()), "| train 2016-20:", len(d))
    pick = d.sample(n, random_state=seed).reset_index(drop=True)
    rows = []
    for i, r in pick.iterrows():
        r = r.copy(); r["vid"] = f"s{i + 1}"
        lab = label(r)
        img, panels = picture(r)
        np.save(OUT / f"sample_{r.vid}.npy", img)
        human_png(r, panels, lab, OUT / f"sample_{r.vid}.png")
        # model image, upscaled so a human can look at it too
        import matplotlib.pyplot as plt
        show = np.stack([img[0], img[1] // 2, np.zeros_like(img[0])], -1)
        plt.imsave(OUT / f"sample_{r.vid}_model.png", np.kron(show, np.ones((3, 3, 1), np.uint8)))
        ok1 = (lab["label"] >= 1) == (r.r_tp1_exit == "tp"); ok2 = (lab["label"] >= 2) == (r.r_tp2_exit == "tp")
        rows.append(dict(vid=r.vid, sym=r.symbol, dir=r.direction, fill=str(r.fill_time)[:16], label=lab["label"], best_r=round(lab["best_r"], 2),
                         v2_tp1=r.r_tp1_exit, v2_tp2=r.r_tp2_exit, match=ok1 and ok2))
    print(pd.DataFrame(rows).to_string(index=False))


def check_labels(n=400, seed=1):
    """label vs v2 tick results on many random train-year zones."""
    d = zones(years=range(2016, 2021)).sample(n, random_state=seed)
    bad = 0
    for _, r in d.iterrows():
        lab = label(r)
        ok = ((lab["label"] >= 1) == (r.r_tp1_exit == "tp")) and ((lab["label"] >= 2) == (r.r_tp2_exit == "tp"))
        bad += not ok
    print(f"label check: {n - bad}/{n} match v2 tick results")


if __name__ == "__main__":
    {"samples": samples, "check": check_labels}[sys.argv[1]]()
