import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from scipy.signal import butter,filtfilt
from collections import defaultdict
W=pickle.load(open('wells.pkl','rb'));N=sorted(W)
def lp(g,c):
    g=pd.Series(g).interpolate().bfill().ffill().values
    return filtfilt(*butter(1,c/0.5),g)
perm=np.random.RandomState(1).permutation(len(N))[:150]
out={L:[] for L in [100,300,1000]}
for wi in perm:
    d=W[N[wi]];K=d['K'];n=len(d['MD'])
    if n-K<300: continue
    tvt=d['TVT'].astype(float).copy();tvt[:K]=d['TVT_input'][:K]
    g=lp(d['GR'].astype(float),0.4);ok=~np.isnan(tvt)&~np.isnan(g)
    tw=np.interp(tvt,d['tw_tvt'],d['tw_gr']);bins=np.round(tvt/0.5).astype(int)
    for L in out:
        acc=defaultdict(lambda:[0.,0]);ptr=0
        for i in range(n):
            while ptr<i-L:
                if ok[ptr]: a=acc[bins[ptr]];a[0]+=g[ptr];a[1]+=1
                ptr+=1
            if i>=K and ok[i]:
                a=acc.get(bins[i])
                if a and a[1]>0: out[L].append((wi,(g[i]-a[0]/a[1])**2,(g[i]-tw[i])**2,a[1]))
        
for L,v in out.items():
    v=np.array(v);ntot=sum(max(len(W[N[w]]['MD'])-W[N[w]]['K'],0) for w in set(v[:,0].astype(int)))
    print('lag>=%d ft: covered rows %d (%.0f%%), selfref RMS %.1f vs typelog RMS %.1f, median #samples %d'%(L,len(v),100*len(v)/ntot,np.sqrt(v[:,1].mean()),np.sqrt(v[:,2].mean()),np.median(v[:,3])))
