import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np; pd.set_option("display.width",250)
from src.research import orb_ml_test as o, orb_nr_test as nr
d,res=o.wf()
z=nr.load(); q30={r: z[(z.when.dt.year<=2020)&(z.range==r)].w_ratio.quantile(0.3) for r in ["ASIA","LON30","LON60"]}
d["tight"]=d.apply(lambda r: r.w_ratio<=q30[r["range"]],axis=1); comp=(d.nr4==1)|d.tight
col="BRK_2R_net"; pf=o.pf
d["hour_ny"]=pd.to_datetime(d.break_ms,unit="ms",utc=True).dt.tz_convert("America/New_York").dt.hour
out=[]
for s in (1,2,3):
    x=d[res[("2R",s,0.3)]&comp&(d.year>=2019)].copy()
    one=x.sort_values("break_ms").groupby(["symbol","day"]).head(1)
    for name,y in (("all ranges (backtest)",x),("max 1 per pair per day",one)):
        t=y[y.year>=2021]; e=y.sort_values("break_ms")[col].cumsum()
        out.append(dict(seed=s,version=name,pf_21_26=pf(t[col]),pf_19_20=pf(y[y.year<=2020][col]),
                        pf_21_23=pf(t[t.year<=2023][col]),pf_24_26=pf(t[t.year>=2024][col]),n_yr=round(len(t)/5.75),
                        R_21_26=round(t[col].sum()),maxDD=round((e.cummax()-e).max(),1)))
    if s==1: keep1=one
print(pd.DataFrame(out).to_string(index=False))
# breakdown, seed 1, max-1 version, 2019-26
x=keep1
print("\nPF by RANGE x PAIR (2019-26, seed 1, max 1/day) [n trades]:")
tab=x.groupby(["range","symbol"])[col].agg(lambda r: f"{pf(r)} [{len(r)}]").unstack(); print(tab.to_string())
print("\nPF by range:", x.groupby("range")[col].apply(pf).to_dict())
print("PF by pair:", x.groupby("symbol")[col].apply(pf).to_dict())
print("\nPF by NY hour of break:", x.groupby("hour_ny")[col].agg(lambda r: f"{pf(r)} [{len(r)}]").to_dict())
# same breakdown with NO ML/compression (plain ORB 2R), to see session/pair nature
p=d[d.year>=2019]
print("\nPLAIN ORB 2R (no filter) PF by RANGE x PAIR 2019-26:")
print(p.groupby(["range","symbol"])[col].apply(pf).unstack().to_string())
