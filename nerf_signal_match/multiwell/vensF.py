import sys,pickle,numpy as np,time
import vit
from vit import *
cfg=sys.argv[1]; perm=np.random.RandomState(0).permutation(len(NAMES)); out=[]; t0=time.time()
def bnd_fn(n,gr,cl):
    return np.unique(np.linspace(0,n,max(n//cl,1)+1).astype(int)).tolist()
for wi in perm[500:773]:
    wi=int(wi); d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K
    if n<50: continue
    md=d['MD'][K:]; Z=d['Z'][K:]; mz,cz=np.polyfit(md,Z,1); z=mz*md+cz-Z; gr=lp(d['GR'][K:]); tvt0=d['TVT_input'][K-1]; y=d['TVT'][K:]
    kw={}; 
    if cfg=='peaks': b=bounds(n,'peaks',gr)
    elif cfg=='c100': b=bnd_fn(n,gr,100)
    elif cfg=='c250': b=bnd_fn(n,gr,250)
    elif cfg=='shift': b=sorted(set([0]+[x+85 for x in bounds(n,'unif')[1:-1] if x+85<n]+[n]))
    elif cfg=='g30k': b=bounds(n,'unif'); kw=dict(gamma=30000)
    elif cfg=='g300k': b=bounds(n,'unif'); kw=dict(gamma=300000)
    else: b=bounds(n,'unif')
    if len(b)<2: continue
    try: r=find_ms(b,z,gr,tvt0,d['tw_gr'],d['tw_tvt'],**kw)
    except Exception as e: r=None
    if r is None: continue
    out.append(dict(w=wi,y=y,p=r[0][0],n=n))
pickle.dump(out,open(f'vensF_{cfg}.pkl','wb')); print(cfg,len(out),time.time()-t0)
