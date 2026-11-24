import sys, numpy as np, time, json
from common import *
from decoder import *
import prior
P0=prior.fit(W,NAMES)
SUBSET=[NAMES[i] for i in np.random.RandomState(1).permutation(len(NAMES))[:int(sys.argv[2]) if len(sys.argv)>2 else 60]]
def zsl(d):
    K=d['K']; md=d['MD'][K:].astype(float); Z=d['Z'][K:].astype(float)
    return float(np.polyfit(md,Z,1)[0]) if len(md)>5 else 0.0
def evalcfg(cfg, ws=SUBSET):
    pk=cfg.get('prior',{}); S=S_TIGHT if cfg.get('grid')=='tight' else S_GRID
    lpb,lps=prior.logpriors(P0,DURS,S,**pk); res={}; res_s={}
    for w in ws:
        d=W[w]; dip=cfg.get('dipc',0.0)*zsl(d)
        pr=prep(d,dip=dip,tw_blend=cfg.get('tw_blend',0.5),gr_mask=cfg.get('gr_mask',False))
        top,soft,_=decode(pr,lpb,lps,S=S,**cfg.get('dk',{}))
        res[w]=top; res_s[w]=soft
    return pooled(res),pooled(res_s),res
if __name__=='__main__':
    cfgs=json.loads(sys.argv[1])
    for c in cfgs:
        t=time.time(); a,b,_=evalcfg(c); print(json.dumps(c),round(a,3),round(b,3),round(time.time()-t),flush=True)
