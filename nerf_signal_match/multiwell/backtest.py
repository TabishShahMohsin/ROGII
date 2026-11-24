# Per-well backtest of Viterbi configs on the PREFIX (where the geologist's TVT_input is known): does prefix error predict lateral error / best config?
import sys,pickle,time,numpy as np
from vit import W,NAMES,lp,bounds,find_ms
cfg=sys.argv[1]; a,b=int(sys.argv[2]),int(sys.argv[3]); perm=np.random.RandomState(0).permutation(len(NAMES)); out=[]; t0=time.time(); SPAN=2500
def bnd_fn(n,cl): return np.unique(np.linspace(0,n,max(n//cl,1)+1).astype(int)).tolist()
for wi in perm[a:b]:
    wi=int(wi); d=W[NAMES[wi]]; K=d['K']; s0=max(0,K-SPAN)
    while K-s0>=600:
        mdq=d['MD'][s0:K]; Zq=d['Z'][s0:K]; c=np.polyfit(mdq,Zq,1)
        if np.max(np.abs(np.polyval(c,mdq)-Zq))<40: break
        s0+=100
    if K-s0<600: continue
    md=d['MD'][s0:K]; Z=d['Z'][s0:K]; mz,cz=np.polyfit(md,Z,1); z=mz*md+cz-Z; gr=lp(d['GR'][s0:K]); y=d['TVT_input'][s0:K]; tvt0=y[0]; n=len(y)
    kw={}
    if cfg=='base': bb=bounds(n,'unif')
    elif cfg=='c250': bb=bnd_fn(n,250)
    elif cfg=='g30k': bb=bounds(n,'unif'); kw=dict(gamma=30000)
    elif cfg=='g300k': bb=bounds(n,'unif'); kw=dict(gamma=300000)
    try: r=find_ms(bb,z,gr,tvt0,d['tw_gr'],d['tw_tvt'],**kw)
    except Exception as e: r=None
    if r is None: continue
    out.append(dict(w=wi,p=r[0][0],y=y,n=n,hold=np.sqrt(np.mean((tvt0-y)**2)),rm=np.sqrt(np.mean((r[0][0]-y)**2))))
pickle.dump(out,open(f'bt_{cfg}_{a}_{b}.pkl','wb')); print(cfg,len(out),time.time()-t0)
