import sys,pickle,time,numpy as np,pandas as pd
from vit3 import *
cfg=sys.argv[1]; df=pd.read_pickle('fail2.pkl'); bad=list(df[df.rm>15].index)
perm=np.random.RandomState(0).permutation(len(NAMES)); sub=[int(w) for w in perm[500:540]]; ws=list(dict.fromkeys(bad+sub)); out=[]; t0=time.time()
def tbounds(d,K,tol=1.5):
    md=d['MD'][K:]; s=(d['TVT'][K:]+d['Z'][K:]); n=len(md); i=0; bs=[0]
    while i<n-1:
        j=min(i+20,n-1)
        while j<n-1:
            k=min(j+20,n-1); c=np.polyfit(md[i:k+1],s[i:k+1],1)
            if np.max(np.abs(np.polyval(c,md[i:k+1])-s[i:k+1]))>tol: break
            j=k
        bs.append(j); i=j
    bs[-1]=n; return sorted(set(bs))
for wi in ws:
    d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K; z,gr,tvt0,twg,m0,y=prep(d)
    if cfg=='oracleB': b=tbounds(d,K); kw=dict(max_m=0.12,mr=0.12,dens=1,gamma=60000)
    elif cfg=='L500': b=np.unique(np.linspace(0,n,max(n//500,1)+1).astype(int)).tolist(); kw=dict(max_m=0.1,mr=0.1,dens=1,gamma=60000)
    elif cfg=='L300': b=np.unique(np.linspace(0,n,max(n//300,1)+1).astype(int)).tolist(); kw=dict(max_m=0.1,mr=0.1,dens=1,gamma=60000)
    q=fm(b,z,gr,tvt0,twg,d['tw_tvt'],**kw)
    if q: out.append(dict(w=wi,y=y,p=q[0][0],P=q[0],n=n,nb=len(b)))
pickle.dump(out,open(f'vm_geo_{cfg}.pkl','wb')); print(cfg,len(out),'%.0fs'%(time.time()-t0))
