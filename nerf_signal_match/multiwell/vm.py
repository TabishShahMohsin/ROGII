import sys,pickle,time,numpy as np,pandas as pd
from vit3 import *
cfg=sys.argv[1]; mm=float(sys.argv[2]); gam=float(sys.argv[3]); df=pd.read_pickle('fail2.pkl'); bad=list(df[df.rm>15].index)
perm=np.random.RandomState(0).permutation(len(NAMES)); sub=[int(w) for w in perm[500:540]]; ws=list(dict.fromkeys(bad+sub)); out=[]; t0=time.time()
for wi in ws:
    d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K
    z,gr,tvt0,twg,m0,y=prep(d); b=bounds(n,'unif'); r=fm(b,z,gr,tvt0,twg,d['tw_tvt'],max_m=mm,mr=mm,gamma=gam)
    if r: out.append(dict(w=wi,y=y,p=r[0][0],P=r[0],n=n))
pickle.dump(out,open(f'vm_{cfg}.pkl','wb')); print(cfg,len(out),'%.0fs'%(time.time()-t0))
