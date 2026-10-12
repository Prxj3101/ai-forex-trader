r"""
VISION V3 = last big zone try (#1 size/cost-aware HYBRID + #2 PRETRAINING), on the WATCH-5 dataset only.
Rules fixed BEFORE results (CLAUDE.md "VISION V3"):
  PRETRAIN  windows every 30 min, 2016-2019 only (fit years; last 3 days of 2019 purged), same picture (no zone).
            question: from the bid at time T, does price go +1 H1-ATR(14) first (1), -1 ATR first (0) or neither in 24 h (2)?
            pretrain-val = Oct-Dec 2019 (early stop). Images on disk as a memmap (too big for RAM).
  MIN SL    smallest SL (pips) rule chosen on FIT 2016-19 only (keeps >= 50% of fit trades, best fit take-all PF tp2).
  NUMBERS   log SL pips, cost R, spread R, NY hour sin/cos, weekday, pair, direction, zone height ATR, risk ATR.
  MODELS    cnn_hyb (picture+numbers, CNN from zero) | cnn_pre (same, CNN eye from PRETRAIN), each 3 seeds + ensemble
            | lgbm_v3 (numbers + old features) | shuffle (cnn_pre, shuffled labels). NO ViT (Prajwal's choice).
  PASS      candidate ensemble cnn_pre_ens only. Metric = bot's own exit (pf_smart), MOCK, net:
            top-30% PF >= 1.20 AND top-50% PF >= 1.05 AND above shuffle AND above cnn_hyb_ens (for cnn_pre) -> exam opened once.
Chalana:  python src\research\vision\vision_v3.py build        (pretrain pictures, ~30-45 min)
          python src\research\vision\vision_v3.py pretrain     (GPU)
          python src\research\vision\vision_v3.py train        (all models + PASS check)
          python src\research\vision\vision_v3.py all          (build + pretrain + train)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import vision_data as vd  # noqa: E402
import vision_train as vt  # noqa: E402
from src.backtest.tick_engine import ticks  # noqa: E402

PRE = vd.OUT / "pretrain"
MODELS = ROOT / "models" / "vision" / "v3"
vt.BUILD = vd.OUT / "build_w5"
vt.RES = vd.OUT / "results_v3"
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2}
PRE_YEARS = range(2016, 2020)
PURGE_END = pd.Timestamp("2019-12-29", tz="UTC")
PRE_VAL = pd.Timestamp("2019-10-01", tz="UTC")
SMOKE = 0


# ============================================================ PRETRAIN DATA
def h1_atr(sym):
    b = vd.bars(sym, "H1")
    pc = b.close.shift()
    tr = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs()))
    return b.close_time.dt.tz_convert(None).to_numpy(), tr.rolling(14).mean().to_numpy()   # naive UTC


def pre_chunk(job):
    sym, year, limit = job
    out = PRE / (f"_smoke_{sym}_{year}" if limit else f"{sym}_{year}")
    if out.with_suffix(".parquet").exists() and not limit:
        return f"{sym} {year}: already done"
    b = vd.bars(sym, "M15")
    ts = b.close_time[(b.close_time.dt.year == year) & b.close_time.dt.minute.isin([0, 30]) & (b.close_time < PURGE_END)]
    ts = ts.iloc[200:] if year == PRE_YEARS[0] else ts                    # enough history for the first panels
    if limit:
        ts = ts.iloc[::max(1, len(ts) // limit)].head(limit)
    act, aval = h1_atr(sym)
    imgs = np.zeros((len(ts), 2, vd.H, 4 * 3 * vd.NB), np.uint8)
    rows, t0 = [], time.time()
    for i, T in enumerate(ts):
        t_ms = int(T.value // 10**6)
        r = pd.Series(dict(symbol=sym, fill_time=T, fill_ms=t_ms, top=np.nan, bottom=np.nan, sl=np.nan, zone_candle_time=T))
        rec = dict(symbol=sym, t=T, y=-1, err="")
        try:
            j = int(np.searchsorted(act, np.datetime64(T.tz_convert(None)), side="right")) - 1   # last H1 bar CLOSED by T
            atr = aval[j] if j >= 0 else np.nan
            z = ticks(sym, t_ms, t_ms + 24 * 3600_000)
            if z is None or len(z[0]) < 10 or not np.isfinite(atr):
                rec["err"] = "no_data"
            else:
                bid = z[2]; p0 = bid[0]
                up, dn = bid >= p0 + atr, bid <= p0 - atr
                iu = int(np.argmax(up)) if up.any() else len(bid)
                idn = int(np.argmax(dn)) if dn.any() else len(bid)
                rec["y"] = 1 if iu < idn else (0 if idn < iu else 2)
                rec["atr_pips"] = atr / PIP[sym]
                img, panels = vd.picture(r)                                 # asserts: nothing at/after T
                imgs[i] = img
        except Exception as e:
            rec["err"] = f"{type(e).__name__}: {e}"[:150]
        rows.append(rec)
    meta = pd.DataFrame(rows)
    PRE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), img=imgs)
    meta.to_parquet(out.with_suffix(".parquet"))
    return f"{sym} {year}: {len(ts)} windows, ok {int((meta.y >= 0).sum())}, {time.time() - t0:.0f}s"


def pre_build(procs=4, limit=0):
    jobs = [(s, y, limit) for s in vd.PAIRS for y in PRE_YEARS]
    if procs == 1:
        for j in jobs:
            print(pre_chunk(j), flush=True)
        return
    from multiprocessing import Pool
    with Pool(procs) as p:
        for msg in p.imap_unordered(pre_chunk, jobs):
            print(msg, flush=True)


def pre_memmap(smoke=False):
    """All pretrain pictures in ONE file on disk (memmap) + labels. Built once."""
    tag = "_smoke" if smoke else ""
    mf, lf = PRE / f"all_imgs{tag}.npy", PRE / f"all_meta{tag}.parquet"
    if mf.exists() and lf.exists():
        return np.load(mf, mmap_mode="r"), pd.read_parquet(lf)
    fs = sorted(f for f in PRE.glob("*.parquet") if f.name.startswith("_smoke") == smoke and not f.name.startswith("all_"))
    metas = [pd.read_parquet(f).assign(src=str(f), row=lambda d: np.arange(len(d))) for f in fs]
    m = pd.concat(metas, ignore_index=True)
    m = m[m.y >= 0].reset_index(drop=True)
    mm = np.lib.format.open_memmap(mf, mode="w+", dtype=np.uint8, shape=(len(m), 2, vd.H, 4 * 3 * vd.NB))
    for f, g in m.groupby("src"):
        mm[g.index.to_numpy()] = np.load(Path(f).with_suffix(".npz"))["img"][g.row.to_numpy()]
    mm.flush(); del mm
    m.drop(columns=["src", "row"]).to_parquet(lf)
    return np.load(mf, mmap_mode="r"), pd.read_parquet(lf)


# ============================================================ MODELS
def make_eye():
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(2, 32), blk(32, 64), blk(64, 96), blk(96, 128), nn.AdaptiveAvgPool2d((2, 4)))


def make_net(n_num, n_out, kind="cnn"):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "cnn":
                self.eye, feat = make_eye(), 4 * 128 * 8
            else:                                                         # ViT ImageNet (SWAG 384) as the eye
                w = vt.make_vit(); w.net.heads.head = nn.Identity(); self.eye, feat = w, 768
            self.num = nn.Sequential(nn.Linear(n_num, 64), nn.ReLU()) if n_num else None
            self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat + (64 if n_num else 0), 256), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(256, n_out))

        def see(self, x):
            if kind != "cnn":
                return self.eye(x)
            x = x.float() / 255.0
            b = x.shape[0]
            p = x.unfold(3, 180, 180).permute(0, 3, 1, 2, 4).reshape(b * 4, 2, 64, 180)
            return self.eye(p).reshape(b, -1)

        def forward(self, x, num=None):
            f = self.see(x)
            if self.num is not None:
                f = torch.cat([f, self.num(num)], 1)
            return self.head(f)
    return Net()


def fit_loop(name, net, X, y, num, tr, va, lr_groups, ep=30, pat=4, bs=128, seed=0):
    """Generic training loop (X may be a disk memmap). Early stop on VAL loss, LR x0.3 when VAL stalls."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)
    opt = torch.optim.AdamW(lr_groups(net), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    yt = torch.from_numpy(np.asarray(y)).long()
    nt = None if num is None else torch.from_numpy(num.astype(np.float32))
    if SMOKE:
        ep, bs = 2, 16

    def batch(j):
        j = np.sort(j)
        xb = torch.from_numpy(np.ascontiguousarray(X[j])).to(dev)
        return xb, (None if nt is None else nt[j].to(dev)), j

    def predict(idx):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast(**amp):
            for i in range(0, len(idx), 16 if SMOKE else 256):
                xb, nb, j = batch(idx[i:i + (16 if SMOKE else 256)])
                out.append((j, torch.softmax(net(xb, nb).float(), 1).cpu().numpy()))
        jj = np.concatenate([a for a, _ in out]); pp = np.concatenate([b for _, b in out])
        res = np.zeros((len(X) if len(idx) == len(X) else jj.max() + 1, pp.shape[1]), np.float32)
        res[jj] = pp
        return res

    best, state, bad = 9e9, None, 0
    for e in range(ep):
        net.train(); t0 = time.time(); perm = rng.permutation(tr); tl = 0.0
        for i in range(0, len(perm), bs):
            xb, nb, j = batch(perm[i:i + bs])
            with torch.autocast(**amp):
                loss = F.cross_entropy(net(xb, nb).float(), yt[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(j)
        pv = predict(va)[va]
        vl = float(F.nll_loss(torch.log(torch.from_numpy(pv) + 1e-9), yt[va]))
        sched.step(vl)
        acc = float((pv.argmax(1) == np.asarray(y)[va]).mean())
        print(f"  {name} epoch {e + 1}: fit loss {tl / len(tr):.4f} | VAL loss {vl:.4f} acc {acc * 100:.1f}% | "
              f"lr {opt.param_groups[-1]['lr']:.1e} | {time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                print(f"  {name}: VAL stopped improving -> stop, keep best epoch", flush=True); break
    net.load_state_dict(state)
    return net, predict


# ============================================================ PRETRAIN
def pretrain():
    X, m = pre_memmap(smoke=bool(SMOKE))
    va = np.flatnonzero(m.t >= PRE_VAL); tr = np.flatnonzero(m.t < PRE_VAL)
    print(f"PRETRAIN windows: {len(m)} | train {len(tr)} | val (Oct-Dec 2019) {len(va)} | label mix %",
          (m.y.value_counts(normalize=True).sort_index() * 100).round(1).to_dict(), flush=True)
    net = make_net(0, 3)
    net, _ = fit_loop("pretrain", net, X, m.y.to_numpy(), None, tr, va, lambda n: n.parameters(), ep=15, pat=3)
    base = m.y.iloc[va].value_counts(normalize=True).max()
    print(f"PRETRAIN guessing-the-most-common answer on val = {base * 100:.1f}%  (compare with VAL acc above)")
    if not SMOKE:
        MODELS.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(net.eye.state_dict(), MODELS / "eye_pretrained.pt")
    return net


# ============================================================ ZONES (watch 5) + numbers + min SL
def zone_data():
    vt.SMOKE = bool(SMOKE)
    m, x = vt.load_data()
    m["sl_pips"] = m.risk / m.symbol.map(PIP)
    fit = m.period == "fit"
    rows = []
    for thr in [0, 2, 3, 4, 5, 6, 8, 10]:
        k = fit & (m.sl_pips >= thr)
        rows.append(dict(min_sl_pips=thr, keep=round(float(k.sum() / fit.sum()), 3), fit_pf_tp2=round(vt.pf(m.n_tp2[k]), 3)))
    t = pd.DataFrame(rows); print("MIN SL rule (FIT 2016-19 only):\n" + t.to_string(index=False))
    ok = t[t.keep >= 0.5]
    thr = int(ok.loc[ok.fit_pf_tp2.idxmax(), "min_sl_pips"])
    print(f"-> chosen min SL = {thr} pips (applied to every period)", flush=True)
    keep = (m.sl_pips >= thr).to_numpy()
    m, x = m[keep].reset_index(drop=True), x[keep]
    ft = m.fill_time.dt.tz_convert("America/New_York")
    h = ft.dt.hour + ft.dt.minute / 60
    wd = ft.dt.weekday.replace(6, 0)
    num = pd.DataFrame({"log_sl": np.log(m.sl_pips), "cost_r": m.cost_r, "spread_r": m.spread_r,
                        "h_sin": np.sin(2 * np.pi * h / 24), "h_cos": np.cos(2 * np.pi * h / 24),
                        "dir": np.where(m.direction == "bull", 1.0, -1.0),
                        "zone_h_atr": m.zone_height_atr, "risk_atr": m.risk_atr * m.risk / m.orig_risk})
    for d in range(5):
        num[f"wd{d}"] = (wd == d).astype(float)
    for s in vd.PAIRS:
        num[s] = (m.symbol == s).astype(float)
    f = (m.period == "fit").to_numpy()
    mu, sd = num[f].mean(), num[f].std().replace(0, 1)
    num = ((num - mu) / sd).fillna(0.0).to_numpy(np.float32)
    return m, x, num, thr


def run_models(m, x, num, seeds=(0, 1, 2)):
    import torch
    y = m.label.to_numpy()
    fit = np.flatnonzero(m.period == "fit"); val = np.flatnonzero(m.period == "val")
    all_idx = np.arange(len(m))
    eye_file = MODELS / "eye_pretrained.pt"

    def one(name, kind, seed, pre=False, shuffle=False):
        yy = y.copy()
        if shuffle:
            yy[fit] = np.random.default_rng(seed).permutation(yy[fit])
        net = make_net(num.shape[1], 4, kind)
        if pre:
            net.eye.load_state_dict(torch.load(eye_file, map_location="cpu"))
        if kind == "cnn":
            groups = (lambda n: [{"params": n.eye.parameters(), "lr": 1e-4 if pre else 1e-3},
                                 {"params": list(n.num.parameters()) + list(n.head.parameters()), "lr": 1e-3}])
            _, pred = fit_loop(name, net, x, yy, num, fit, val, groups, seed=seed)
        else:
            groups = (lambda n: [{"params": n.eye.parameters(), "lr": 2e-5},
                                 {"params": list(n.num.parameters()) + list(n.head.parameters()), "lr": 1e-3}])
            _, pred = fit_loop(name, net, x, yy, num, fit, val, groups, ep=12, pat=3, bs=32, seed=seed)
        return pred(all_idx)

    def family(base, kind, pre=False):
        ps = []
        for s in seeds:
            p = one(f"{base}_s{s}", kind, s, pre=pre); vt.evaluate(f"{base}_s{s}", m, p); ps.append(p)
        if len(ps) > 1:
            vt.evaluate(f"{base}_ens", m, np.mean(ps, 0))

    # numbers baseline
    import lightgbm as lgb
    from src.ml import walkforward as wf
    X2 = pd.concat([pd.DataFrame(num, columns=[f"n{i}" for i in range(num.shape[1])]), m[wf.FEATURES].reset_index(drop=True)], axis=1)
    for c in wf.CATEGORICAL:
        X2[c] = X2[c].astype("category")
    params = dict(wf.PARAMS); params.update(objective="multiclass", num_class=4)
    vt.evaluate("lgbm_v3", m, lgb.LGBMClassifier(**params).fit(X2.iloc[fit], y[fit]).predict_proba(X2))
    family("cnn_hyb", "cnn")
    if eye_file.exists():
        family("cnn_pre", "cnn", pre=True)
        vt.evaluate("shuffle", m, one("shuffle", "cnn", 7, pre=True, shuffle=True))
    # ViT dropped (Prajwal, 11 Oct): CNN only


def pass_check():
    def row(name, keep):
        f = vt.RES / f"eval_{name}.json"
        if not f.exists():
            return None
        return next(r for r in json.load(open(f))["rows"] if r["keep"] == keep)["pf_smart"]
    sh, base = row("shuffle", 0.3), row("cnn_hyb_ens", 0.3)
    print("\n===== PASS CHECK (MOCK 2021-22, net, bot's own exit) =====")
    for cand in ["cnn_pre_ens"]:
        t30, t50 = row(cand, 0.3), row(cand, 0.5)
        if t30 is None:
            print(cand, "missing"); continue
        ok = t30 >= 1.20 and t50 >= 1.05 and (sh is None or t30 > sh) and (cand != "cnn_pre_ens" or base is None or t30 > base)
        print(f"{cand}: top30 PF {t30:.3f} (need >= 1.20) | top50 PF {t50:.3f} (need >= 1.05) | shuffle {sh} | cnn_hyb_ens {base} -> {'PASS' if ok else 'FAIL'}")


def train():
    m, x, num, thr = zone_data()
    vt.RES.mkdir(parents=True, exist_ok=True)
    json.dump(dict(min_sl_pips=thr), open(vt.RES / "min_sl.json", "w"))
    run_models(m, x, num, seeds=(0,) if SMOKE else (0, 1, 2))
    vt.report(); pass_check()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "--smoke" in sys.argv:
        SMOKE = 1; vt.RES = vd.OUT / "results_v3_smoke"; MODELS = MODELS / "smoke"
    if cmd in ("build", "all"):
        pre_build(procs=1 if SMOKE else 4, limit=60 if SMOKE else 0)
    if cmd in ("pretrain", "all"):
        pretrain()
    if cmd in ("train", "all"):
        train()
