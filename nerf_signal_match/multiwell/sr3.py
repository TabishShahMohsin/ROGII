import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from scipy.signal import butter,filtfilt
W=pickle.load(open('wells.pkl','rb'));N=sorted(W)
def lp(g,c):
    g=pd.Series(g).interpolate().bfill().ffill().values;return filtfilt(*butter(1,c/0.5),g)
perm=np.random.RandomState(1).permutation(len(N))[:100]
D=[-12,-8,-5,-3,-2,-1,0,1,2,3,5,8,12];E_t={x:[] for x in D};E_s={x:[] for x in D};E_c={x:[] for x in D}
L=300;B=0.5
for wi in perm:
    d=W[N[wi]];K=d['K'];n=len(d['MD'])
    if n-K<600: continue
    tvt=d['TVT'].astype(float).copy();tvt[:K]=d['TVT_input'][:K]
    g=lp(d['GR'].astype(float),0.4);ok=~np.isnan(tvt)&~np.isnan(g)
    # reference profile built from rows<i-L, but for speed: build reference from rows < K + (n-K)/2 and evaluate rows >= K+(n-K)/2+L  (truth TVT)
    h=K+(n-K)//2;lo=np.floor(np.nanmin(tvt[:h])/B).astype(int);hi=np.ceil(np.nanmax(tvt[:h])/B).astype(int)
    idx=np.arange(lo,hi+1);s=np.zeros(len(idx));c=np.zeros(len(idx))
    for i in range(h):
        if ok[i]:
            b=int(round(tvt[i]/B))-lo;s[b]+=g[i];c[b]+=1
    prof=np.where(c>0,s/np.maximum(c,1),np.nan)
    rr=np.arange(h+L,n);rr=rr[ok[rr]]
    if len(rr)<100:continue
    for x in D:
        t=tvt[rr]+x;tw=np.interp(t,d['tw_tvt'],d['tw_gr']);b=np.round(t/B).astype(int)-lo;m=(b>=0)&(b<len(idx))
        pv=np.full(len(rr),np.nan);pv[m]=prof[b[m]];m2=m&~np.isnan(pv)
        E_t[x].append(np.mean((g[rr][m2]-tw[m2])**2));E_s[x].append(np.mean((g[rr][m2]-pv[m2])**2))
        E_c[x].append(np.mean((g[rr][m2]-0.5*pv[m2]-0.5*tw[m2])**2))
print('offset  typelog  selfref  combo(avg of refs)  (mean squared GR residual, true TVT +offset)')
for x in D:print('%4d  %7.1f %7.1f %7.1f'%(x,np.nanmean(E_t[x]),np.nanmean(E_s[x]),np.nanmean(E_c[x])))
