r"""
VISION BOT - Step 3: training + MOCK test (2021-22). The EXAM (2023-26) is NEVER loaded here.

Periods: FIT 2016-2019 (learning) | VAL 2020 (early stop: training stops when VAL loss stops improving)
         | MOCK 2021-2022 (comparison of models, threshold choice). Purged zones are never used.
Models (all predict the same 4 classes: 0 SL | 1 1R | 2 2R | 3 3R+):
  cnn      own small CNN, one shared "eye" per panel (H4 | H1 | M15 | M1-zoom), trained from zero (3 seeds)
  vit      ViT-B/16 pretrained (ImageNet SWAG, 384x384), fine-tuned, panels as 2x2 grid (3 seeds + ensemble)
  lgbm     LightGBM on the old numeric features (same 4 classes) = "numbers" baseline
  shuffle  CNN trained on SHUFFLED labels = leak test, must find NOTHING
Decision (same for every model): for each class k and exit (tp1 / tp2 / trail) the average NET R on FIT
  -> expected net R of a zone = sum_k P(k) * avgR[k, exit]; score = best exit's expected R; that exit is used.
Report on MOCK: AUC, and keep top 100/50/30/20/10 % by score -> trades, PF, total R (net: $5/lot + 0.2 pip stop slip).

Chalana (venv, laptop):  python src\research\vision\vision_train.py all        (sab: lgbm, cnn x3, shuffle, vit, report)
                         python src\research\vision\vision_train.py cnn | vit | lgbm | shuffle | report
                         python src\research\vision\vision_train.py smoke      (tiny CPU/GPU run to test the code)
                         python src\research\vision\vision_train.py eyetest    (aankh test: can the CNN see things that ARE in the picture?)
  Phase 2 watch mode: add the dataset name:  python src\research\vision\vision_train.py all w5   (or w15)
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
import vision_build as vb  # noqa: E402
import vision_data as vd  # noqa: E402

DATASET = sys.argv[2] if len(sys.argv) > 2 and __name__ == "__main__" else "touch"   # touch | w5 | w15 (Phase 2 watch mode)
BUILD = vb.BUILD if DATASET == "touch" else vd.OUT / f"build_{DATASET}"
RES = vd.OUT / ("results" if DATASET == "touch" else f"results_{DATASET}")
MODELS = ROOT / "models" / "vision" / ("" if DATASET == "touch" else DATASET)
POL = ["tp1", "tp2", "trail"]
KEEP = [1.0, 0.5, 0.3, 0.2, 0.1]
SMOKE = False


# ---------------------------------------------------------------- data (exam can not be loaded)
def load_data():
    """Memory-lean: read the small label files first, then fill ONE preallocated image array chunk by chunk."""
    metas = []
    for f in sorted(BUILD.glob("*.parquet")):
        if f.name.startswith("_"):
            continue
        m = pd.read_parquet(f)
        m["row_in_chunk"] = np.arange(len(m)); m["chunk_file"] = str(f)
        m = m[m.split.isin(["train", "mock"])]                  # exam + purge rows dropped here
        if "status" in m:                                       # watch mode: only real trades (not skipped touches)
            m = m[m.status == "ok"]
        assert not (m.split == "exam").any()
        if len(m):
            metas.append(m)
    m = pd.concat(metas, ignore_index=True)
    m["period"] = np.where(m.split == "mock", "mock", np.where(m.fill_time.dt.year <= 2019, "fit", "val"))
    if SMOKE:
        m = pd.concat([g.head(400) for _, g in m.groupby("period")], ignore_index=True)
    x = np.zeros((len(m), 2, vd.H, 4 * 3 * vd.NB), np.uint8)
    for f, g in m.groupby("chunk_file"):                       # exam chunk files are never opened
        x[g.index.to_numpy()] = np.load(Path(f).with_suffix(".npz"))["img"][g.row_in_chunk.to_numpy()]
    # net R (same cost rules as the old ML) + numeric features for the LightGBM baseline
    from src.ml import walkforward as wf
    z = wf.load(vd.PAIRS)
    z = z[(z.zone_tf.astype(str) == "M15") & (z.touch_number == 1)].copy()
    z["symbol_s"] = z.symbol.astype(str)
    z = z.drop_duplicates(["symbol_s", "fill_ms"], keep="last")
    watch = "n_tp1" in m                                        # watch mode: net R already from tick_engine for the NEW entry
    cols = ["symbol_s", "fill_ms", "r_tp1", "r_tp2", "r_trail"] + [c for c in wf.FEATURES if not (watch and c == "cost_r")]
    z = z[cols].rename(columns={"r_tp1": "n_tp1", "r_tp2": "n_tp2", "r_trail": "n_trail", "fill_ms": "key_ms"})
    if watch:
        z = z.drop(columns=["n_tp1", "n_tp2", "n_trail"])
    n0 = len(m)
    m["key_ms"] = m["touch_ms"] if watch else m["fill_ms"]
    m = m.merge(z, left_on=["symbol", "key_ms"], right_on=["symbol_s", "key_ms"], how="left", suffixes=("", "_f"))
    assert len(m) == n0 and m.n_tp1.notna().all(), "net R merge failed"
    m["label"] = m.label.astype(int)
    print(f"data: {len(m)} zones | fit {int((m.period == 'fit').sum())} | val {int((m.period == 'val').sum())} "
          f"| mock {int((m.period == 'mock').sum())} | images {x.shape} {x.nbytes / 1e9:.1f} GB", flush=True)
    return m, x


# ---------------------------------------------------------------- decision + report
def pf(r):
    r = np.asarray(r, float)
    neg = -r[r < 0].sum()
    return float(r[r > 0].sum() / neg) if neg > 0 else float("nan")


def class_avg(m):
    f = m[m.period == "fit"]
    return np.array([[f.loc[f.label == k, "n_" + p].mean() for p in POL] for k in range(4)])   # [4 classes, 3 exits]


def decide(m, prob, avg):
    e = prob @ avg                                              # expected net R per exit
    return e.max(1), e.argmax(1)


def evaluate(name, m, prob):
    from sklearn.metrics import roc_auc_score
    avg = class_avg(m)
    score, pol = decide(m, prob, avg)
    mk = (m.period == "mock").to_numpy()
    y = m.label.to_numpy()
    out = dict(model=name, auc_1r=float(roc_auc_score(y[mk] >= 1, prob[mk, 1:].sum(1))),
               auc_2r=float(roc_auc_score(y[mk] >= 2, prob[mk, 2:].sum(1))), rows=[])
    # thresholds come from FIT scores (not mock) -> top q% means "score above the fit-set q quantile"
    fit_score = score[(m.period == "fit").to_numpy()]
    for q in KEEP:
        thr = -np.inf if q >= 1 else np.quantile(fit_score, 1 - q)
        sel = mk & (score >= thr)
        smart = m["n_tp1"].to_numpy().copy()
        for j, p in enumerate(POL):
            smart = np.where(pol == j, m["n_" + p].to_numpy(), smart)
        row = dict(keep=q, n=int(sel.sum()))
        for p in POL:
            r = m.loc[sel, "n_" + p].to_numpy()
            row[f"pf_{p}"], row[f"R_{p}"] = round(pf(r), 3), round(float(r.sum()), 1)
        r = smart[sel]
        row["pf_smart"], row["R_smart"] = round(pf(r), 3), round(float(r.sum()), 1)
        out["rows"].append(row)
    RES.mkdir(parents=True, exist_ok=True)
    p = m.loc[mk, ["symbol", "direction", "fill_time", "fill_ms", "label", "n_tp1", "n_tp2", "n_trail"]].copy()
    p[["p0", "p1", "p2", "p3"]] = prob[mk]; p["score"] = score[mk]; p["exit"] = np.array(POL)[pol[mk]]
    p.to_parquet(RES / f"mock_{name}.parquet")
    json.dump(out, open(RES / f"eval_{name}.json", "w"), indent=1)
    print_eval(out)
    return out


def print_eval(o):
    print(f"\n== {o['model']}  | MOCK AUC: reach 1R {o['auc_1r']:.3f}  reach 2R {o['auc_2r']:.3f}  (0.5 = coin toss)")
    print(pd.DataFrame(o["rows"]).to_string(index=False), flush=True)


# ---------------------------------------------------------------- LightGBM baseline
def run_lgbm(m, x):
    import lightgbm as lgb
    from src.ml import walkforward as wf
    tr = m.period.isin(["fit"]).to_numpy()
    X = m[wf.FEATURES].copy()
    for c in wf.CATEGORICAL:
        X[c] = X[c].astype("category")
    params = dict(wf.PARAMS); params.update(objective="multiclass", num_class=4)
    mdl = lgb.LGBMClassifier(**params).fit(X[tr], m.label[tr])
    return evaluate("lgbm", m, mdl.predict_proba(X))


# ---------------------------------------------------------------- neural nets
def torch_setup(seed):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch, dev


def make_cnn():
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    class PanelCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.eye = nn.Sequential(blk(2, 32), blk(32, 64), blk(64, 96), blk(96, 128), nn.AdaptiveAvgPool2d((2, 4)))
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(4 * 128 * 8, 256), nn.ReLU(),
                                      nn.Dropout(0.3), nn.Linear(256, 4))

        def forward(self, x):                                   # x [B, 2, 64, 720] uint8 -> 4 panels of 180 px
            x = x.float() / 255.0
            b = x.shape[0]
            p = x.unfold(3, 180, 180).permute(0, 3, 1, 2, 4).reshape(b * 4, 2, 64, 180)   # [B*4, 2, 64, 180]
            f = self.eye(p).reshape(b, 4, -1).flatten(1)        # same eye for every panel, order kept = which panel
            return self.head(f)
    return PanelCNN()


def make_vit():
    import torch
    import torch.nn as nn
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)   # 384x384 version = more detail per candle
    size = net.image_size
    net.heads.head = nn.Linear(net.heads.head.in_features, 4)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", mean); self.register_buffer("std", std)

        def forward(self, x):                                   # [B,2,64,720] -> 2x2 grid [B,3,128,360] -> 384x384
            x = x.float() / 255.0
            p = x.unfold(3, 180, 180)                           # [B,2,64,4,180]
            top = torch.cat([p[:, :, :, 0], p[:, :, :, 1]], 3); bot = torch.cat([p[:, :, :, 2], p[:, :, :, 3]], 3)
            g = torch.cat([top, bot], 2)                        # [B,2,128,360]
            g = torch.cat([g[:, :1], g[:, 1:], g[:, :1]], 1)    # 3 channels: bars | zone | bars
            g = torch.nn.functional.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
            return self.net((g - self.mean) / self.std)
    return Wrap()


def train_net(name, m, x, kind="cnn", seed=0, shuffle=False):
    torch, dev = torch_setup(seed)
    import torch.nn.functional as F
    net = (make_cnn() if kind == "cnn" else make_vit()).to(dev)
    cfg = dict(cnn=dict(lr=1e-3, bs=128, ep=30, pat=4), vit=dict(lr=2e-5, bs=32, ep=12, pat=3))[kind]
    if SMOKE:
        cfg.update(ep=2, bs=16)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4 if kind == "cnn" else 0.05)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1)   # VAL stalls -> smaller steps
    xt = torch.from_numpy(x)
    y = m.label.to_numpy().copy()
    fit = np.flatnonzero(m.period == "fit"); val = np.flatnonzero(m.period == "val")
    if shuffle:
        y[fit] = np.random.default_rng(seed).permutation(y[fit])      # leak test: answers mixed up
    yt = torch.from_numpy(y).long()
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    def predict(idx):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast(**amp):
            pb = 16 if SMOKE else 256
            for i in range(0, len(idx), pb):
                j = idx[i:i + pb]
                out.append(torch.softmax(net(xt[j].to(dev)).float(), 1).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad = 9e9, None, 0
    for ep in range(cfg["ep"]):
        net.train(); t0 = time.time(); perm = np.random.permutation(fit); tl = 0.0
        for i in range(0, len(perm), cfg["bs"]):
            j = perm[i:i + cfg["bs"]]
            with torch.autocast(**amp):
                loss = F.cross_entropy(net(xt[j].to(dev)).float(), yt[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(j)
        pv = predict(val)
        vl = float(F.nll_loss(torch.log(torch.from_numpy(pv) + 1e-9), yt[val]))
        sched.step(vl)
        print(f"  {name} epoch {ep + 1}: fit loss {tl / len(fit):.4f} | VAL loss {vl:.4f} | lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["pat"]:
                print(f"  {name}: VAL stopped improving -> stop, keep best epoch", flush=True); break
    net.load_state_dict(best_state)
    MODELS.mkdir(parents=True, exist_ok=True)
    if not SMOKE:
        torch.save(best_state, MODELS / f"{name}.pt")
    return predict(np.arange(len(m)))


def run_cnn(m, x, seeds=(0, 1, 2)):
    probs = []
    for s in seeds:
        p = train_net(f"cnn_s{s}", m, x, "cnn", s)
        evaluate(f"cnn_s{s}", m, p); probs.append(p)
    if len(probs) > 1:
        evaluate("cnn_ens", m, np.mean(probs, 0))             # average of the seeds = steadier


def run_shuffle(m, x):
    evaluate("shuffle", m, train_net("shuffle", m, x, "cnn", 7, shuffle=True))


def run_vit(m, x, seeds=(0, 1, 2)):
    probs = []
    for s in seeds:
        p = train_net(f"vit_s{s}", m, x, "vit", s)
        evaluate(f"vit_s{s}", m, p); probs.append(p)
    if len(probs) > 1:
        evaluate("vit_ens", m, np.mean(probs, 0))


def eyetest(m, x):
    """AANKH TEST: questions whose answer IS in the picture. If the CNN can not learn these, the training is broken."""
    from sklearn.metrics import roc_auc_score
    h1 = np.array([vd.panel_bars(r, "H1").close.iloc[[0, -1]].to_numpy() for _, r in m.iterrows()])
    m15 = np.array([vd.panel_bars(r, "M15")[["open", "close"]].iloc[-1].to_numpy() for _, r in m.iterrows()])
    tasks = {"buy_or_sell": (m.direction == "bull").to_numpy().astype(int),
             "h1_panel_up": (h1[:, 1] > h1[:, 0]).astype(int),
             "last_m15_green": (m15[:, 1] > m15[:, 0]).astype(int)}
    mk = (m.period == "mock").to_numpy()
    for name, y in tasks.items():
        mm = m.copy(); mm["label"] = y
        p = train_net(f"eye_{name}", mm, x, "cnn", 0)
        acc = float(((p[mk, 1] > p[mk, 0]).astype(int) == y[mk]).mean())
        print(f"EYE TEST {name:15s} MOCK accuracy {acc * 100:.1f}%  AUC {roc_auc_score(y[mk], p[mk, 1]):.3f}  (guessing = {max(y[mk].mean(), 1 - y[mk].mean()) * 100:.0f}%)", flush=True)


def report():
    fs = sorted(RES.glob("eval_*.json"))
    rows = []
    for f in fs:
        o = json.load(open(f))
        r30 = next(r for r in o["rows"] if r["keep"] == 0.3); r100 = next(r for r in o["rows"] if r["keep"] == 1.0)
        rows.append(dict(model=o["model"], auc_1r=round(o["auc_1r"], 3), auc_2r=round(o["auc_2r"], 3),
                         all_pf_tp2=r100["pf_tp2"], top30_n=r30["n"], top30_pf_tp1=r30["pf_tp1"], top30_pf_tp2=r30["pf_tp2"],
                         top30_pf_trail=r30["pf_trail"], top30_pf_smart=r30["pf_smart"], top30_R_smart=r30["R_smart"]))
    print("\nMOCK 2021-22 SUMMARY (net of costs)\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        report(); sys.exit()
    if cmd == "smoke":
        SMOKE = True
        RES = vd.OUT / f"results_smoke_{DATASET}"
    m, x = load_data()
    if cmd == "eyetest":
        eyetest(m, x); sys.exit()
    if cmd in ("lgbm", "all", "smoke"):
        run_lgbm(m, x)
    if cmd in ("cnn", "all", "smoke"):
        run_cnn(m, x, seeds=(0,) if SMOKE else (0, 1, 2))
    if cmd in ("shuffle", "all"):
        run_shuffle(m, x)
    if cmd in ("vit", "all", "smoke"):
        run_vit(m, x, seeds=(0,) if SMOKE else (0, 1, 2))
    report() if not SMOKE else None
