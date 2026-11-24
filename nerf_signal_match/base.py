import pandas as pd, numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
tw=pd.read_csv('tw.csv'); hw=pd.read_csv('hw.csv')
K=1545; lat=np.arange(K,len(hw))
tvt=hw.TVT.values; gr=hw.GR.values; z=hw.Z.values; md=hw.MD.values
print("last known TVT %.2f at MD %.0f; lateral TVT: min %.2f max %.2f mean %.2f std %.2f"%(tvt[K-1],md[K-1],tvt[lat].min(),tvt[lat].max(),tvt[lat].mean(),tvt[lat].std()))
rm=lambda p: np.sqrt(np.mean((p-tvt[lat])**2))
print("BASELINE const last-known TVT: RMSE %.2f  MAE %.2f"%(rm(np.full(len(lat),tvt[K-1])),np.mean(np.abs(tvt[K-1]-tvt[lat]))))
# constant-g baseline: TVT = -Z + g with g fixed at last-known (i.e. flat structure / apparent dip 0 in Z)
g0=tvt[K-1]+z[K-1]; print("BASELINE flat-structure (TVT=-Z+g0): RMSE %.2f"%rm(-z[lat]+g0))
# linear extrapolation of g from last 300 known rows
gk=tvt+z; p=np.polyfit(md[K-300:K],gk[K-300:K],1); print("linear-g slope fit",p[0]," RMSE %.2f"%rm(-z[lat]+np.polyval(p,md[lat])))
# ceiling: GR_hw vs typewell GR at TRUE TVT
ttv=tw.TVT.values; tgr=tw.GR.values
f=lambda w: np.interp(w,ttv,tgr)
for name,idx in [('prefix',np.arange(K)),('lateral',lat)]:
    ok=~np.isnan(gr[idx]); a=gr[idx][ok]; b=f(tvt[idx][ok])
    print(name,"corr raw %.3f"%np.corrcoef(a,b)[0,1], end=' | ')
    for s in [5,20,50]:
        gs=gaussian_filter1d(np.where(np.isnan(gr),np.nanmean(gr),gr),s)[idx]; 
        # type-well smoothed in w with same effective sigma in TVT (md step ~1ft, dTVT/dMD small in lateral)
        ts=gaussian_filter1d(tgr,s/0.5)[:]; bs=np.interp(tvt[idx],ttv,ts)
        print("sigma %d corr %.3f"%(s,np.corrcoef(gs,bs)[0,1]),end='; ')
    print()
print("GR stats hw lateral mean/std %.1f/%.1f ; typewell in lateral TVT range mean/std %.1f/%.1f"%(np.nanmean(gr[lat]),np.nanstd(gr[lat]),f(tvt[lat]).mean(),f(tvt[lat]).std()))
print("typewell GR 12190-12240 mean %.1f std %.1f"%(tgr[(ttv>12190)&(ttv<12240)].mean(),tgr[(ttv>12190)&(ttv<12240)].std()))
