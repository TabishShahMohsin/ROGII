# batch-2 beam variants around the gradient-heavy emission; family typewell with region-held-out members
import numpy as np
from common import W, NAMES, REG, CEN
from decoder import *
import prior, family
P0=prior.fit(W,NAMES)
LP={'base':prior.logpriors(P0,DURS,S_GRID)+(S_GRID,),'loose':prior.logpriors(P0,DURS,S_GRID,rho_clip=0.90,sig_scale=1.0)+(S_GRID,),
    'tg':prior.logpriors(P0,DURS,S_TIGHT)+(S_TIGHT,)}
G=dict(wl=0.6,wg=2.5,temp=3.0)
VAR={'g25':('base',{},{}),'g25_tg':('tg',{},{}),'g25_nf':('base',{},dict(follow=0.0)),'g25_lo':('loose',{},{}),
     'g25_w14':('base',{},dict(wdrift=14.0)),'g25_f85':('base',{},dict(follow=0.85)),'g15':('base',{},dict(wg=1.5))}
def heel_fit(d,ref):
    K=d['K']; kt=d['TVT_input'][:K].astype(float); kg=d['GR'][:K].astype(float); m=np.isfinite(kt)&np.isfinite(kg)
    if m.sum()<10: return np.inf
    r=(kg[m]-np.interp(kt[m],ref[0],ref[1]))/25.0; return float(np.mean(0.5*np.log1p(r*r)))
def job(w):
    d=W[w]; i=NAMES.index(w); out={}
    for name,(pn,pk,dk) in VAR.items():
        lpb,lps,S=LP[pn]; pr=prep(d,**pk); kw=dict(G); kw.update(dk)
        top,soft,_=decode(pr,lpb,lps,S=S,**kw)
        out[name]=top.astype(np.float32); out[name+'_s']=soft.astype(np.float32); out[name+'_em']=emission(pr,top)
    allowed=[v for j,v in enumerate(NAMES) if REG[j]!=REG[i]]
    ref,nm=family.family_tw(d,allowed,CEN[i]); out['fam_n_em']=float(nm)
    ok=ref is not None and heel_fit(d,ref)<heel_fit(d,(d['tw_TVT'].astype(float),d['tw_GR'].astype(float)))
    out['fam_ok_em']=float(ok)
    if ok:
        lpb,lps,S=LP['base']; pr=prep(d,tw_override=ref); pr0=prep(d); pr['gs']=pr0['gs']
        top,soft,_=decode(pr,lpb,lps,S=S,**G); out['fam']=top.astype(np.float32); out['fam_s']=soft.astype(np.float32)
    else:
        out['fam']=out['g25']; out['fam_s']=out['g25_s']
    return w,out
