# family-typewell coverage study: wider radius / lower corr thresholds, region-held-out members, heel gate
import numpy as np
from common import W, NAMES, REG, CEN
from decoder import *
import prior, family
from job2 import heel_fit, LP, G
CFG={'fam12tg':(12000,0.9,'tg',1),'fam25':(25000,0.9,'base',1),'fam25tg':(25000,0.9,'tg',1),'fam25ng':(25000,0.9,'base',0)}
def job(w):
    d=W[w]; i=NAMES.index(w); out={}
    allowed=[v for j,v in enumerate(NAMES) if REG[j]!=REG[i]]
    h0=heel_fit(d,(d['tw_TVT'].astype(float),d['tw_GR'].astype(float)))
    cache={}
    for name,(md_,c_,pn,gate) in CFG.items():
        key=(md_,c_)
        if key not in cache: cache[key]=family.family_tw(d,allowed,CEN[i],maxdist=md_,corr=c_)
        ref,nm=cache[key]; ok=ref is not None and (heel_fit(d,ref)<h0 or not gate)
        out[name+'_ok_em']=float(ok); out[name+'_n_em']=float(nm)
        if ok:
            lpb,lps,S=LP[pn]; pr=prep(d,tw_override=ref); pr['gs']=prep(d)['gs']
            top,soft,_=decode(pr,lpb,lps,S=S,**G); out[name+'_s']=soft.astype(np.float32)
    return w,out
