r"""
JUDGEMENT TEST (9 Oct 2026): blind check of HIS eye on the bot's zones.
250 random M15 zone first-touches 2019-2026 (EUR/GBP/USDJPY, wick entry, tick-exact results), shown WITHOUT result,
date or ML score: H1 context + M15 chart only up to the moment price reaches the zone. He answers TAKE / SKIP
(+ confidence, tags) on a private artifact page; answers are read back and compared with the hidden results.
Key (results + ML score) stays here: data/research/judge/key.parquet. Images: data/research/judge/img/<id>.png
Pass rule (fixed before): >= ~150 TAKE answers, TAKE group net PF > 1.10 and clearly above the SKIP group (bootstrap).
Chalana: python src\research\judge_make.py sample      (once)
         python src\research\judge_make.py images 1    (batch 1 = first 50)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/research/judge"
KEY = OUT / "key.parquet"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
NY = "America/New_York"
N, BATCH, SEED = 250, 50, 20261009


def sample():
    if KEY.exists():
        print("key exists, not resampling"); return
    rows = []
    for s in PAIRS:
        v = pd.read_parquet(ROOT / f"data/research/{s}_v2.parquet")
        v = v[(v.zone_tf.astype(str) == "M15") & (v.touch_number == 1) & v.r_tp1.notna()]
        v = v[v.fill_time.dt.year >= 2019]
        v["symbol"] = s
        rows.append(v)
    d = pd.concat(rows, ignore_index=True)
    rng = np.random.default_rng(SEED)
    per = N // len(PAIRS) + 1
    pick = pd.concat([g.sample(per, random_state=int(rng.integers(1e9))) for _, g in d.groupby("symbol")]).head(N)
    pick = pick.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sc = pd.read_parquet(ROOT / "data/research/ml_scores.parquet")
    sc["symbol"] = sc.symbol.astype(str); sc["direction"] = sc.direction.astype(str)
    pick["direction"] = pick.direction.astype(str)
    pick = pick.merge(sc[["symbol", "direction", "zone_candle_time", "touch_number", "fill_time", "score", "keep_0.3", "va_ok", "act_ok"]],
                      on=["symbol", "direction", "zone_candle_time", "touch_number", "fill_time"], how="left")
    pick["id"] = [f"z{i + 1:03d}" for i in range(len(pick))]
    pick["batch"] = pick.index // BATCH + 1
    keep = ["id", "batch", "symbol", "direction", "zone_candle_time", "created_at", "fill_time", "entry", "sl", "top", "bottom",
            "body_top", "body_bottom", "disp_atr", "killzone", "session_fill", "r_tp1", "r_tp2", "r_trail", "score", "keep_0.3",
            "va_ok", "act_ok"]
    pick[keep].to_parquet(KEY)
    print(len(pick), "zones; per pair", pick.symbol.value_counts().to_dict(), "| win at 1:1", round(float((pick.r_tp1 > 0).mean()) * 100, 1),
          "| ML keep", int(pick["keep_0.3"].fillna(False).sum()))


def _candles(ax, df, up="#2fd39a", dn="#ff5c7a", w=0.62):
    x = np.arange(len(df))
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, up, dn)
    ax.vlines(x, l, h, color=col, linewidth=0.8)
    ax.bar(x, np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0015), bottom=np.minimum(o, c), width=w, color=col, linewidth=0)
    return x


def _ticks(ax, idx, n=6):
    ny = idx.tz_convert(NY)
    pos = np.linspace(0, len(idx) - 1, n).astype(int)
    ax.set_xticks(pos)
    ax.set_xticklabels([ny[p].strftime("%a %H:%M") for p in pos], fontsize=8)


from functools import lru_cache


@lru_cache(maxsize=12)
def _bars(s, tf):
    return pd.read_parquet(ROOT / f"data/processed/{s}_{tf}.parquet", columns=["open", "high", "low", "close", "close_time"])


def image(r, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, fg, grid, muted = "#0f1318", "#dfe6ee", "#1d242c", "#8a97a6"
    s, bull = r.symbol, r.direction == "bull"
    ft = pd.Timestamp(r.fill_time)
    frames = {}
    for tf, n in (("H1", 120), ("M15", 190)):
        df = _bars(s, tf)
        df = df[df.close_time <= ft].tail(n)
        frames[tf] = df
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=100, gridspec_kw=dict(width_ratios=[1, 1.75]))
    fig.patch.set_facecolor(bg)
    zc = "#2fd39a" if bull else "#ff5c7a"
    for ax, tf in zip(axes, ("H1", "M15")):
        df = frames[tf]
        ax.set_facecolor(bg)
        x = _candles(ax, df)
        ax.grid(color=grid, linewidth=0.6); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        ax.yaxis.tick_right()
        _ticks(ax, df.index)
        ci = int(np.searchsorted(df.index.values, pd.Timestamp(r.created_at).tz_convert("UTC").to_datetime64()))
        ci = min(max(ci - 1, 0), len(df) - 1)
        x0, x1 = ci, len(df) + (22 if tf == "M15" else 10)
        ax.add_patch(plt.Rectangle((x0 - 0.5, r.bottom), x1 - x0, r.top - r.bottom, facecolor=zc, alpha=0.16, edgecolor=zc, linewidth=1.2))
        ax.hlines(r.entry, x0 - 0.5, x1, color="#38d9f5", linewidth=1.6)
        ax.hlines(r.sl, x0 - 0.5, x1, color="#ff5c7a", linewidth=1.1, linestyles="dashed")
        lo = min(df.low.min(), r.sl, r.bottom); hi = max(df.high.max(), r.top, r.entry)
        pad = (hi - lo) * 0.06
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-1, x1 + 1)
        ax.axvline(len(df) - 0.5, color=muted, linewidth=0.8, linestyle=":")
        dg = 3 if s.endswith("JPY") else 5
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f"%.{dg}f"))
        ax.set_title(f"{s}  {tf}", color=fg, fontsize=11, loc="left", fontweight="bold")
        if tf == "M15":
            ax.text(len(df) - 1, hi + pad * 0.4, "NOW: price touches the entry next  ", color=muted, fontsize=8, va="top", ha="right")
            slp = abs(r.entry - r.sl) / PIP[s]
            ax.text(x1 + 0.6, r.entry, f"ENTRY {'BUY' if bull else 'SELL'}", color="#38d9f5", fontsize=9, fontweight="bold",
                    va="top" if bull else "bottom", ha="right")
            ax.text(x1 + 0.6, r.sl, f"SL {slp:.1f} pips", color="#ff5c7a", fontsize=9, va="top" if bull else "bottom", ha="right")
    fig.tight_layout(pad=1.2)
    fig.savefig(out, facecolor=bg)
    plt.close(fig)


def images(batch: int):
    k = pd.read_parquet(KEY)
    k = k[k.batch == batch]
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    meta = []
    for r in k.itertuples(index=False):
        image(r, OUT / "img" / f"{r.id}.png")
        ft = pd.Timestamp(r.fill_time).tz_convert(NY)
        meta.append(dict(id=r.id, pair=r.symbol, side="BUY" if r.direction == "bull" else "SELL",
                         zone="demand" if r.direction == "bull" else "supply", when=ft.strftime("%a %H:%M") + " NY",
                         session=str(r.session_fill), killzone=str(r.killzone) if str(r.killzone) != "none" else "",
                         sl_pips=round(abs(r.entry - r.sl) / PIP[r.symbol], 1), disp_atr=round(float(r.disp_atr), 1)))
    import json
    (OUT / f"batch{batch}.json").write_text(json.dumps(meta))
    print("batch", batch, len(meta), "images")


CHART_BARS = {"M5": 600, "M15": 500, "H1": 400, "H4": 250}


def chartdata(batch: int):
    """Interactive page data: closed bars of each timeframe up to the fill (nothing after), zone levels. No dates leak
    except bar times, which the page shows only as weekday + NY time."""
    import json
    k = pd.read_parquet(KEY)
    k = k[k.batch == batch]
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for r in k.itertuples(index=False):
        ft = pd.Timestamp(r.fill_time)
        dg = 3 if r.symbol.endswith("JPY") else 5
        out = dict(id=r.id, digits=dg, top=round(float(r.top), dg), bottom=round(float(r.bottom), dg),
                   entry=round(float(r.entry), dg), sl=round(float(r.sl), dg),
                   created=int(pd.Timestamp(r.created_at).timestamp()), side="BUY" if r.direction == "bull" else "SELL", tf={})
        for tf, n in CHART_BARS.items():
            df = _bars(r.symbol, tf)
            df = df[df.close_time <= ft].tail(n)
            t = (df.index.as_unit("s").asi8).tolist()
            out["tf"][tf] = [[int(a), round(float(o), dg), round(float(h), dg), round(float(l_), dg), round(float(c), dg)]
                             for a, o, h, l_, c in zip(t, df.open, df.high, df.low, df.close)]
        (OUT / "data" / f"{r.id}.json").write_text(json.dumps(out, separators=(",", ":")))
    print("batch", batch, "chart data", len(k))


def meta(batch: int):
    """Card info for the page (no result / date / ML): data/research/judge/batch<N>.json"""
    import json
    k = pd.read_parquet(KEY)
    k = k[k.batch == batch]
    out = []
    for r in k.itertuples(index=False):
        ft = pd.Timestamp(r.fill_time).tz_convert(NY)
        out.append(dict(id=r.id, pair=r.symbol, side="BUY" if r.direction == "bull" else "SELL",
                        zone="demand" if r.direction == "bull" else "supply", when=ft.strftime("%a %H:%M") + " NY",
                        session=str(r.session_fill), killzone=str(r.killzone) if str(r.killzone) != "none" else "",
                        sl_pips=round(abs(r.entry - r.sl) / PIP[r.symbol], 1), disp_atr=round(float(r.disp_atr), 1)))
    (OUT / f"batch{batch}.json").write_text(json.dumps(out))
    print("meta", batch, len(out))


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample()
    elif sys.argv[1] == "data":
        chartdata(int(sys.argv[2]))
        meta(int(sys.argv[2]))
    else:
        images(int(sys.argv[2]))
