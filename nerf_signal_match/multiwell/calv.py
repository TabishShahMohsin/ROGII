# heel-calibrated type log: fit GR_prefix ~ a*tw(TVT)+b on last N prefix rows, apply to type log before Viterbi
import pickle,sys,time,numpy as np
from vit import W,NAMES,lp,find_ms,bounds
def calib(d,N,mode):
    K=d['K'];i0=max(K-N,0);tv=d['TVT_input'][i0:K].astype(float);g=lp(d['GR'][:K])[i0:K];ok=~np.isnan(tv);tv=tv[ok];g=g[ok]
    if len(g)<100:return 1.,0.
    tw=np.interp(tv,d['tw_tvt'],d['tw_gr'])
    if mode=='gain_off':
        A=np.c_[tw,np.ones_like(tw)];a,b=np.linalg.lstsq(A,g,rcond=None)[0];a=float(np.clip(a,0.7,1.4));b=float(np.mean(g)-a*np.mean(tw));return a,b
    if mode=='off':return 1.,float(np.mean(g-tw))
    if mode=='gain':
        a=float(np.clip(np.std(g)/np.std(tw),0.7,1.4));return a,0.
def run(wi,cfgs):
    d=W[NAMES[wi]];K=d['K'];n=len(d['MD'])-K
    if n<50:return None
    md=d['MD'][K:];Z=d['Z'][K:];mz,cz=np.polyfit(md,Z,1);z=mz*md+cz-Z;gr=lp(d['GR'][K:]);tvt0=d['TVT_input'][K-1];y=d['TVT'][K:];bnd=bounds(n,'unif',gr);out=dict(w=wi,n=n,hold=np.mean((tvt0-y)**2))
    for nm,(N,mode) in cfgs.items():
        a,b=(1.,0.) if mode is None else calib(d,N,mode);twg=a*d['tw_gr']+b
        r=find_ms(bnd,z,gr,tvt0,twg,d['tw_tvt']);P,w,tc=r;out[nm+'_top1']=np.mean((P[0]-y)**2);out[nm+'_ab']=(a,b);out[nm+'_p']=P[0]
    return out
cfgs={'base':(0,None),'go1500':(1500,'gain_off'),'go600':(600,'gain_off'),'off1500':(1500,'off'),'gain1500':(1500,'gain')}
if __name__=="__main__":
    a,b=int(sys.argv[1]),int(sys.argv[2]);perm=np.random.RandomState(0).permutation(len(NAMES));res=[];t0=time.time()
    for wi in perm[a:b]:
        try:r=run(int(wi),cfgs)
        except Exception as e:print('err',e);r=None
        if r:res.append(r)
        print(len(res),'%.0fs'%(time.time()-t0),flush=True)
    pickle.dump(res,open(f'calv_{a}_{b}.pkl','wb'))
    tot=lambda k:np.sqrt(sum(r[k]*r['n'] for r in res)/sum(r['n'] for r in res))
    print({k:round(float(tot(k)),2) for k in ['hold']+[c+'_top1' for c in cfgs]})
