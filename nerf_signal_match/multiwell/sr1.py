import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from scipy.signal import butter,filtfilt
W=pickle.load(open('wells.pkl','rb'));N=sorted(W)
def lp(g,c):
    g=pd.Series(g).interpolate().bfill().ffill().values
    return g if c>=0.5 else filtfilt(*butter(1,c/0.5),g)
rows=[]
perm=np.random.RandomState(1).permutation(len(N))[:150]
for wi in perm:
    d=W[N[wi]];K=d['K'];n=len(d['MD'])
    if n-K<300: continue
    tvt=d['TVT'].astype(float).copy(); tvt[:K]=d['TVT_input'][:K]
    gr=d['GR'].astype(float)
    ok=~np.isnan(tvt)&~np.isnan(gr)
    tw=np.interp(tvt,d['tw_tvt'],d['tw_gr'])
    res={}
    for c in [0.5,0.4,0.2,0.1,0.05,0.02]:
        g=lp(gr,c);res[c]=np.mean((g[K:]-tw[K:])[ok[K:]]**2)
    # self reference: for lateral rows, reference = mean GR of earlier rows (index<i-200) within 0.5ft truth TVT
    g=lp(gr,0.4);bins=np.round(tvt/0.5).astype(int)
    from collections import defaultdict
    acc=defaultdict(lambda:[0.,0]);err_s=[];err_t=[];err_tp=[]
    # prefix-only reference
    pref=defaultdict(lambda:[0.,0])
    for i in range(K):
        if ok[i]: a=pref[bins[i]];a[0]+=g[i];a[1]+=1
    for i in range(n):
        if i>=K and ok[i]:
            a=acc[bins[i]]; b=pref[bins[i]]
            if a[1]>0: err_s.append(((g[i]-a[0]/a[1])**2,(g[i]-tw[i])**2))
            if b[1]>0: err_tp.append(((g[i]-b[0]/b[1])**2,(g[i]-tw[i])**2))
        if ok[i]: a=acc[bins[i]];a[0]+=g[i];a[1]+=1
    nl=ok[K:].sum()
    rows.append(dict(w=wi,n=n-K,cov_self=len(err_s)/nl,cov_pref=len(err_tp)/nl,
        s_self=np.mean([a for a,b in err_s]) if err_s else np.nan,s_tw_on_self=np.mean([b for a,b in err_s]) if err_s else np.nan,
        s_pref=np.mean([a for a,b in err_tp]) if err_tp else np.nan,s_tw_on_pref=np.mean([b for a,b in err_tp]) if err_tp else np.nan,
        **{f'lp{c}':v for c,v in res.items()}))
df=pd.DataFrame(rows);pd.set_option('display.width',200)
print(len(df),'wells')
print('coverage of lateral rows with earlier same-TVT data: self %.2f  prefix-only %.2f'%(df.cov_self.mean(),df.cov_pref.mean()))
w=df.n
def wm(c):return np.sqrt(np.nansum(df[c]*w)/w.sum())
print('GR residual RMS vs typelog at TRUE tvt, by lowpass cutoff:',{c:round(wm(f'lp{c}'),1) for c in [0.5,0.4,0.2,0.1,0.05,0.02]})
m=df.dropna(subset=['s_self']);ww=m.n*m.cov_self
print('on rows with self-ref: self-ref resid RMS %.1f vs typelog %.1f'%(np.sqrt((m.s_self*ww).sum()/ww.sum()),np.sqrt((m.s_tw_on_self*ww).sum()/ww.sum())))
m=df.dropna(subset=['s_pref']);ww=m.n*m.cov_pref
print('prefix-ref: resid RMS %.1f vs typelog %.1f'%(np.sqrt((m.s_pref*ww).sum()/ww.sum()),np.sqrt((m.s_tw_on_pref*ww).sum()/ww.sum())))
print(df[['cov_self','cov_pref']].describe().T)
