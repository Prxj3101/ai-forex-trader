r"""
MONTHLY TRAINING (15 Oct 2026, his plan): har mahine ke end pe zone bot ka ML naye live data se dobara seekhe -
lekin naya model SIRF tab lage jab woh naye data pe purane se behtar ho (warna purana hi chalta rahe).

  1. VPS se live data lao (/api/live/learning): har chhua hua zone = ML features + dono exits ka R (1:1 aur trail,
     kharche ke baad) + tera decision (AVOID / TAKE / EXIT) -> data/learning/live_rows.parquet (jamaa hota rehta hai).
  2. CANDIDATE = purana historical data (data/research/*_v2.parquet) + live zones (wazan LIVE_WEIGHT = 3, naya market
     zyada gine) - SIRF pichhle EVAL_DAYS (30) din chhod ke. Entry model (jeetega?) + exit model (trail - 1:1).
  3. GATE: pichhle 30 din ke live zones (jo candidate ne nahi dekhe) pe purana vs naya model: dono apne top-30% zones
     lete hain, apna exit chunte hain -> R aur PF. Naya tabhi lagega jab R aur PF dono >= purana, aur kam se kam
     MIN_EVAL (30) zones hon. Warna "KEEP OLD".
  4. ACCEPT -> purane models models/archive/<date>/ mein, naya final model SAARE data pe (eval bhi) -> models/zone_brain.txt,
     models/exit_brain.txt, ml_walkforward.json ka threshold. Phir: git push, VPS pull, FOREX BOT restart.
  5. TERE FAISLE: AVOID / TAKE / EXIT kitne sahi the (R mein). 100+ faisle hone pe "teri aankh" ka model bhi
     train hota hai (sirf report, live mein tab lagayenge jab woh sach mein madad kare).
Chalana (LAPTOP pe, jahan historical datasets hain):
    python src\ml\monthly_retrain.py            (data lao, compare, behtar ho toh models badlo)
    python src\ml\monthly_retrain.py --dry-run  (sirf compare + report, kuch nahi badalta)
Report: data/learning/monthly_<date>.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.ml import walkforward as wf  # noqa: E402

LEARN = ROOT / "data" / "learning"
LIVE = ROOT / "data" / "live"
MODELS = ROOT / "models"
EVAL_DAYS = 30
MIN_EVAL = 30
LIVE_WEIGHT = 3.0
KEY = ["symbol", "zone", "direction", "fill_time"]
RJ = ROOT / "data" / "research" / "ml_walkforward.json"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- 1. live data
def fetch_live(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.api.server import _remote_cfg, _remote_get
    remote = _remote_cfg()
    if remote is not None:
        r = _remote_get(remote, "/api/live/learning", {}, timeout=120)
        if r.get("status") == "unreachable":
            raise SystemExit(f"VPS se data nahi mila: {r.get('error')}  (VPS pe git pull + LIVE API restart kiya?)")
        rows, acts = r.get("rows", []), r.get("actions", [])
        log(f"VPS: {len(rows)} touched zones, {len(acts)} of your actions")
    else:
        f = LIVE / "whatif_features.jsonl"
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        a = LIVE / "user_actions.csv"
        acts = pd.read_csv(a).to_dict("records") if a.exists() else []
        log(f"local data/live: {len(rows)} touched zones, {len(acts)} of your actions")
    LEARN.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    old_f = LEARN / "live_rows.parquet"
    if old_f.exists():
        old = pd.read_parquet(old_f)
        old["features"] = old["features"].map(lambda s: json.loads(s) if isinstance(s, str) else s)
        new = pd.concat([old, new], ignore_index=True) if len(new) else old
    if len(new):
        new = new.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
        save = new.copy()
        save["features"] = save["features"].map(json.dumps)
        save.astype({c: str for c in save.columns if save[c].dtype == object and c != "features"}).to_parquet(old_f)
    acts = pd.DataFrame(acts)
    if len(acts):
        af = LEARN / "user_actions.csv"
        if af.exists():
            acts = pd.concat([pd.read_csv(af), acts], ignore_index=True).drop_duplicates()
        acts.to_csv(af, index=False)
    return new, acts


def live_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """live rows -> same columns as the historical training data (FEATURES + labels)."""
    if rows.empty:
        return pd.DataFrame()
    f = pd.DataFrame(list(rows["features"]))
    d = pd.concat([rows.drop(columns=["features"]).reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    d = d.loc[:, ~d.columns.duplicated(keep="last")]
    d["fill_time"] = pd.to_datetime(d["fill_time"], utc=True, errors="coerce", format="mixed")
    for c in ("net_tp1", "net_trail", "net_brain", "net_bot"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d.dropna(subset=["fill_time", "net_tp1", "net_trail"])
    d["r_trail"], d["r_tp1"] = d["net_trail"], d["net_tp1"]
    d["y"] = (d["r_trail"] > 0).astype(int)
    for c in wf.FEATURES:
        if c not in d:
            d[c] = np.nan
    for c in wf.NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in wf.BOOL:
        d[c] = d[c].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(v in (True, 1, 1.0, "True", "true")))
    return d.sort_values("fill_time").reset_index(drop=True)


# ---------------------------------------------------------------- 2-3. train + gate
def _cat(df: pd.DataFrame, cats: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    for c in wf.CATEGORICAL:
        df[c] = df[c].astype(str)
        df[c] = pd.Categorical(df[c], categories=cats[c]) if cats else df[c].astype("category")
    return df


def _pf(r):
    r = pd.Series(r, dtype=float)
    gl = -r[r < 0].sum()
    return round(float(r[r > 0].sum() / gl), 3) if gl > 0 else (float("inf") if len(r) and r.sum() > 0 else None)


def judge(score, thr, exit_edge, d) -> dict:
    take = score >= thr
    r = np.where(exit_edge > 0, d["net_trail"], d["net_tp1"])[take]
    return dict(taken=int(take.sum()), R=round(float(np.sum(r)), 2), pf=_pf(r),
                win_pct=round(100 * float((r > 0).mean()), 1) if len(r) else None)


def fit_models(train: pd.DataFrame, weights: np.ndarray):
    m = lgb.LGBMClassifier(**wf.PARAMS).fit(train[wf.FEATURES], train["y"], sample_weight=weights)
    e = lgb.LGBMRegressor(**wf.EXIT_PARAMS).fit(train[wf.FEATURES], (train["r_trail"] - train["r_tp1"]).clip(-3, 3),
                                              sample_weight=weights)
    thr = float(np.quantile(m.predict_proba(train[wf.FEATURES])[:, 1], 0.7))     # top 30%, same as walkforward.train
    return m, e, thr


def decisions_report(live: pd.DataFrame, acts: pd.DataFrame) -> dict:
    """Tere faisle (R mein): AVOID ne kitna bachaya, TAKE ne kitna diya, EXIT badlav sahi tha ya galat."""
    out = {}
    if live.empty:
        return out
    u = live.get("user", pd.Series("", index=live.index)).fillna("")
    av, tk = live[u == "avoid"], live[u == "take"]
    out["avoid"] = dict(n=len(av), bot_R=round(float(av["net_bot"].sum()), 2), right=int((av["net_bot"] < 0).sum()))
    out["take"] = dict(n=len(tk), R=round(float(tk["net_brain"].sum()), 2), right=int((tk["net_brain"] > 0).sum()))
    ex = live[live.get("exit_bot").notna() & (live["exit_brain"] != live["exit_bot"])] if "exit_bot" in live else live.iloc[:0]
    g = ex["net_brain"] - ex["net_bot"]
    out["exit_before_trade"] = dict(n=len(ex), R_gained=round(float(g.sum()), 2), right=int((g > 0).sum()))
    out["n_decisions"] = int(len(av) + len(tk) + len(ex))
    return out


def eye_model(live: pd.DataFrame) -> dict | None:
    """100+ faisle: model jo tera AVOID / TAKE predict kare (sirf report)."""
    u = live.get("user", pd.Series("", index=live.index)).fillna("")
    d = live[u.isin(["avoid", "take"])]
    if len(d) < 100 or d["user"].nunique() < 2:
        return dict(status=f"need 100 AVOID/TAKE decisions, have {len(d)}")
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    X = _cat(d)[wf.FEATURES]
    y = (d["user"] == "take").astype(int)
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=10, verbose=-1),
                          X, y, cv=5, method="predict_proba")[:, 1]
    return dict(status="trained (report only)", n=len(d), auc=round(float(roc_auc_score(y, p)), 3))


def run(dry: bool = False, log=print) -> dict:
    rows, acts = fetch_live(log)
    live = live_frame(rows)
    rep = dict(made_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), live_rows=int(len(live)),
               eval_days=EVAL_DAYS, live_weight=LIVE_WEIGHT)
    rep["your_decisions"] = decisions_report(live, acts)
    rep["eye_model"] = eye_model(live) if len(live) else None
    if live.empty:
        rep["decision"] = "KEEP OLD (no live data yet)"
        return _save(rep, log)
    cut = live["fill_time"].max() - pd.Timedelta(days=EVAL_DAYS)
    ev, lv_train = live[live.fill_time > cut], live[live.fill_time <= cut]
    if len(ev) < MIN_EVAL:                                   # thoda data -> pichhle 60 din
        cut = live["fill_time"].max() - pd.Timedelta(days=2 * EVAL_DAYS)
        ev, lv_train = live[live.fill_time > cut], live[live.fill_time <= cut]
    rep["eval"] = dict(n=int(len(ev)), from_=str(ev.fill_time.min()) if len(ev) else None, to=str(ev.fill_time.max()) if len(ev) else None)
    log(f"live zones: {len(live)} (train {len(lv_train)}, test {len(ev)})")
    if len(ev) < MIN_EVAL:
        rep["decision"] = f"KEEP OLD (only {len(ev)} touched zones to test on, need {MIN_EVAL})"
        return _save(rep, log)

    hist = wf.load(config.CORE_SYMBOLS)
    log(f"historical: {len(hist):,} zones up to {hist.fill_time.max():%Y-%m-%d}")
    cols = wf.FEATURES + ["y", "r_trail", "r_tp1", "fill_time"]
    train = pd.concat([hist[cols].assign(w=1.0), lv_train[cols].assign(w=LIVE_WEIGHT)], ignore_index=True)
    train = _cat(train)
    cats = {c: list(train[c].cat.categories) for c in wf.CATEGORICAL}
    m, e, thr = fit_models(train, train["w"].to_numpy())
    evc = _cat(ev, cats)

    from src.live.brain import Brain
    old = Brain()
    old_d = ev.copy()
    for c in wf.CATEGORICAL:
        old_d[c] = old_d[c].astype(str).astype("category")
    o = judge(old.entry.predict(old_d[old.features]), old.threshold, old.exit.predict(old_d[old.features]), ev)
    n = judge(m.predict_proba(evc[wf.FEATURES])[:, 1], thr, e.predict(evc[wf.FEATURES]), ev)
    allr = dict(taken=len(ev), R=round(float(ev["net_bot"].sum()), 2), pf=_pf(ev["net_bot"]))
    rep.update(old_model=o, new_model=n, every_zone=allr, old_threshold=round(old.threshold, 4), new_threshold=round(thr, 4))
    log(f"TEST on last {EVAL_DAYS}d live zones: OLD {o} | NEW {n} | every zone {allr}")
    better = (n["taken"] >= 5 and n["R"] >= o["R"] and (n["pf"] or 0) >= (o["pf"] or 0))
    if not better:
        rep["decision"] = "KEEP OLD (new model not better on the unseen live weeks)"
        return _save(rep, log)
    if dry:
        rep["decision"] = "NEW IS BETTER (dry run - nothing changed)"
        return _save(rep, log)
    # ACCEPT: backup + final model on everything (incl. the test weeks)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    arch = MODELS / "archive" / stamp
    arch.mkdir(parents=True, exist_ok=True)
    rj = RJ
    for f in (MODELS / "zone_brain.txt", MODELS / "exit_brain.txt", rj):
        if f.exists():
            shutil.copy2(f, arch / f.name)
    full = _cat(pd.concat([hist[cols].assign(w=1.0), live[cols].assign(w=LIVE_WEIGHT)], ignore_index=True))
    m, e, thr = fit_models(full, full["w"].to_numpy())
    m.booster_.save_model(str(MODELS / "zone_brain.txt"))
    e.booster_.save_model(str(MODELS / "exit_brain.txt"))
    res = json.loads(rj.read_text()) if rj.exists() else {}
    tr = m.predict_proba(full[wf.FEATURES])[:, 1]
    res["final_thresholds"] = {str(k): float(np.quantile(tr, 1 - k)) for k in wf.KEEP_TOP if k < 1}
    res["trained_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    res["monthly_retrain"] = dict(at=res["trained_at"], live_rows=int(len(live)), live_weight=LIVE_WEIGHT, test=n, old_test=o,
                                  backup=_rel(arch))
    rj.write_text(json.dumps(res, indent=2, default=str))
    rep["decision"] = f"NEW MODEL SAVED (old ones in {_rel(arch)}). Now: git push, VPS git pull, restart FOREX BOT."
    return _save(rep, log)


def _save(rep: dict, log=print) -> dict:
    LEARN.mkdir(parents=True, exist_ok=True)
    f = LEARN / f"monthly_{pd.Timestamp.now():%Y-%m-%d}.json"
    f.write_text(json.dumps(rep, indent=1, default=str))
    log(f"\nDECISION: {rep['decision']}")
    yd = rep.get("your_decisions") or {}
    if yd:
        log(f"YOUR DECISIONS: AVOID {yd.get('avoid')} | TAKE {yd.get('take')} | EXIT before trade {yd.get('exit_before_trade')}")
    log(f"eye model: {rep.get('eye_model')}")
    log(f"report -> {_rel(f)}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monthly zone-bot ML training with a safety gate")
    ap.add_argument("--dry-run", action="store_true", help="compare only, do not replace the models")
    a = ap.parse_args()
    run(dry=a.dry_run)
