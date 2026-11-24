import sys, os, time, pickle, numpy as np
from multiprocessing import Pool
os.environ.setdefault('OMP_NUM_THREADS','1')
from common import W, NAMES
from decoder import *
import prior
P0=prior.fit(W,NAMES)
PRI={}
def pri(name):
    if name not in PRI:
        S=S_TIGHT if name=='tightgrid' else S_GRID
        kw=dict(base={},tight=dict(rho_clip=0.995,sig_scale=0.35),vtight=dict(rho_clip=0.999,sig_scale=0.2),
                loose=dict(rho_clip=0.90,sig_scale=1.0),tightgrid={})[name]
        PRI[name]=prior.logpriors(P0,DURS,S,**kw)+(S,)
    return PRI[name]
# variant: (prior name, prep kwargs, decode kwargs)
VAR={
 'base':('base',{},{}),
 'nofollow':('base',{},dict(follow=0.0)),
 'tight':('base',{},{}),
}
VAR={
 'base':('base',{},dict(temp=3.0)),
 'nofollow':('base',{},dict(follow=0.0,temp=3.0)),
 'tightp':('tight',{},dict(temp=3.0)),
 'vtightp':('vtight',{},dict(temp=3.0)),
 'loosep':('loose',{},dict(temp=3.0)),
 'tgrid':('tightgrid',{},dict(temp=3.0)),
 'znorm':('base',dict(znorm=True),dict(temp=3.0)),
 'grad':('base',{},dict(wl=0.6,wg=1.5,temp=3.0)),
 'ncc':('base',{},dict(ncc_w=5.0,temp=3.0)),
 'gmask':('base',dict(gr_mask=True),dict(temp=3.0)),
}
def job(w):
    out={}; d=W[w]
    for name,(pn,pk,dk) in VAR.items():
        lpb,lps,S=pri(pn); pr=prep(d,**pk)
        if pk.get('znorm'): pr['gs']=float(np.std(pr['gr'][:max(10,len(pr['gr'])//10)]))*0+25.0
        top,soft,(paths,fs)=decode(pr,lpb,lps,S=S,**dk)
        out[name]=top.astype(np.float32); out[name+'_s']=soft.astype(np.float32)
        out[name+'_em']=emission(pr,top)
    return w,out
if __name__=='__main__':
    a,b,outf,nproc=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
    ws=NAMES[a:b]; t=time.time(); res={}
    with Pool(nproc) as pool:
        for i,(w,o) in enumerate(pool.imap_unordered(job,ws,chunksize=1)):
            res[w]=o
            if i%20==0: print(i,len(ws),round(time.time()-t),flush=True)
    pickle.dump(res,open(outf,'wb')); print('done',time.time()-t)
