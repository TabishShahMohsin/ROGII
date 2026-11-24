import sys,pickle,time,numpy as np,pandas as pd
from vit4 import *
cfg=sys.argv[1]; lam=float(sys.argv[2]); gam=float(sys.argv[3]); df=pd.read_pickle('fail2.pkl'); bad=list(df[df.rm>15].index)
perm=np.random.RandomState(0).permutation(len(NAMES)); sub=[int(w) for w in perm[500:540]]; ws=[int(w) for w in perm[500:773]][int(sys.argv[4]):int(sys.argv[5])]; out=[]; t0=time.time()
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
for wi in ws:
    d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K; r=R[wi]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[wi]:off[wi+1]])
    z,gr,tvt0,twg,m0,y=prep(d); b=bounds(n,'unif'); q=fm(b,z,gr,tvt0,twg,d['tw_tvt'],Sp=S,lam=lam,gamma=gam)
    if q: out.append(dict(w=wi,y=y,p=q[0][0],P=q[0],n=n))
pickle.dump(out,open(f'vmF_{cfg}.pkl','wb')); print(cfg,len(out),'%.0fs'%(time.time()-t0))
