# Robustness check of the h4_zone confluence (16 Oct 2026); run from repo root: python src/research/confluence_h4_check.py
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import src.research.confluence_test as ct
from src.ml import walkforward as wf
from src import config
from src.features.smc import find_orderflow_zones, track_zones
from src.research.zone_dataset import load
def pf(r):
    gl=-r[r<0].sum(); return round(float(r[r>0].sum()/gl),3) if gl else np.nan
conf=pd.concat([pd.read_parquet(ct.OUT/f"{s}_conf.parquet") for s in ct.SYMS],ignore_index=True)
key=["symbol","zone_id","touch_number"]
# h4 overlap with swing 3 too
def h4flag(sub, sym, n):
    h4=load(sym,"H4")
    z4=track_zones(find_orderflow_zones(h4,swing_n=n,min_disp_atr=config.disp_for(sym)),h4,max_bars=360)
    zc=z4.created_at.to_numpy(dtype="datetime64[ns]")
    zi=z4.invalidated_time.fillna(pd.Timestamp("2100-01-01",tz="UTC")).to_numpy(dtype="datetime64[ns]")
    ze=np.minimum(zi,zc+np.timedelta64(84,"D")); zb=(z4.direction=="bull").to_numpy(); zt=z4.top.to_numpy(); zbo=z4.bottom.to_numpy()
    tt=sub.fill_time.to_numpy(dtype="datetime64[ns]"); b=(sub.direction=="bull").to_numpy(); top=sub.top.to_numpy(); bot=sub.bottom.to_numpy()
    return np.array([((zc<=tt[i])&(tt[i]<ze)&(zb==b[i])&(zbo<=top[i])&(zt>=bot[i])).any() for i in range(len(sub))])
conf["h4_n3"]=False
for s in ct.SYMS:
    m=conf.symbol==s; conf.loc[m,"h4_n3"]=h4flag(conf[m],s,3)
d=wf.load(ct.SYMS,folder=ct.BASE).merge(conf[key+["h4_zone","h4_n3"]],on=key,how="left")
for c in wf.CATEGORICAL: d[c]=d[c].astype("category")
d["h4_zone"]=d.h4_zone.astype("boolean").fillna(False).astype(bool); d["h4_n3"]=d.h4_n3.astype("boolean").fillna(False).astype(bool)
w=wf.walk_forward(d,2019,log=lambda *a:None)
w.to_parquet(ct.OUT/"wf_base_h4flag.parquet")
for k in (0.3,0.5,0.7,1.0):
    b=w[(w.zone_tf=="M15")&(w.touch_number==1)&w[f"keep_{k}"]&w.va_ok&w.act_ok]
    for col in ("h4_zone","h4_n3"):
        g=b[b[col]]
        print(f"keep {k} {col}: {len(g)} pf {pf(g.r_ml)} 19-21 {pf(g[g.test_year<=2021].r_ml)} 22-26 {pf(g[g.test_year>=2022].r_ml)} tp1 {pf(g.r_tp1)} trail {pf(g.r_trail)} R {g.r_ml.sum():.1f}")
b=w[(w.zone_tf=="M15")&(w.touch_number==1)&w["keep_0.3"]&w.va_ok&w.act_ok]; g=b[b.h4_zone]
print("by pair",{s:(len(x),pf(x.r_ml)) for s,x in g.groupby("symbol",observed=True)})
print("by year R",g.groupby("test_year").r_ml.sum().round(1).to_dict())
print("by dir",{s:pf(x.r_ml) for s,x in g.groupby("direction",observed=True)})
rng=np.random.default_rng(1); obs=pf(g.r_ml)
sims=np.array([pf(b.r_ml.sample(len(g),random_state=int(rng.integers(1e9)))) for _ in range(5000)])
print("random p",(sims>=obs).mean())
# raw no ML
r=w[(w.zone_tf=="M15")&(w.touch_number==1)&w.h4_zone]
print("no ML, h4 yes, OOS 2019-26:",len(r),pf(r.r_ml),pf(r.r_tp1),pf(r.r_trail))
