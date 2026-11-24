import pickle, numpy as np, time
from common import W,NAMES
from hmm import field_fn, calib
R=pickle.load(open('ds.pkl','rb'))
P3=dict(ST=5,D=40.,dstep=1.0,q=0.02,beta=0.3,win=200,sigf=1.0,P=1500,minvalid=0.3,corr_min=0.0,hp=0.0,form='fisher')
def corr_cube(g,ok,fv,win,ST):
    """g [L], ok [L] bool, fv [L,S]; windowed (centered, `win` rows) normalized correlation at stride ST -> rho [T,S], n [T]"""
    L,S=fv.shape; o=ok.astype(float); g0=np.where(ok,g,0.)
    def cs(a): 
        c=np.zeros((L+1,)+a.shape[1:]); c[1:]=np.cumsum(a,0); return c
    cn,cg,cgg=cs(o),cs(g0),cs(g0*g0); cf=cs(fv*o[:,None]); cff=cs(fv*fv*o[:,None]); cfg=cs(fv*(g0*o)[:,None])
    cen=np.arange(0,L,ST); lo=np.clip(cen-win//2,0,L); hi=np.clip(cen+win//2+1,0,L)
    sw=lambda c: c[hi]-c[lo]
    n=sw(cn); sg,sgg=sw(cg),sw(cgg); sf,sff,sfg=sw(cf),sw(cff),sw(cfg)
    nn=np.maximum(n,1); cov=sfg-sf*sg[:,None]/nn[:,None]; vf=sff-sf**2/nn[:,None]; vg=sgg-sg**2/nn
    rho=cov/np.sqrt(np.maximum(vf,1e-9)*np.maximum(vg,1e-9)[:,None]); return rho,n,cen
def hmm_corr3(r,p=P3):
    w=NAMES[r['w']]; d=W[w]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md)); out={}
    pr_all=r['tvt_end']+np.interp(lat,r['idx'],r['yA']); out['prior']=pr_all
    f=field_fn(d['tw_tvt'],d['tw_gr'],p['sigf']); a,b,sg,c=calib(d,K,f,p['P']); out['corr']=c
    gr=d['GR'][lat].astype(np.float64); ok=~np.isnan(gr)
    dg=np.arange(-p['D'],p['D']+1e-9,p['dstep']); S=len(dg)
    if ok.sum()<50: out['pred']=pr_all; out['dmean']=np.zeros(1); out['dsd']=np.full(len(lat),99.); out['ls']=lat[::p['ST']]-K; return out
    fv=f(pr_all[:,None]+dg[None,:]); rho,n,cen=corr_cube(gr,ok,fv,p['win'],p['ST']); T=len(cen)
    nok=(n>=p['minvalid']*p['win'])
    neff=np.maximum(n/ (p['win']/p['ST']),1.)      # effective independent-ish samples per stride
    ll=p['beta']*np.where(nok[:,None],rho,0.)*(1.0 if p['form']=='lin' else 1.0)
    step=p['ST']; sd=np.sqrt(p['q']*step); Tm=np.exp(-0.5*((dg[None,:]-dg[:,None])/sd)**2); Tm/=Tm.sum(1,keepdims=True)
    sd0=np.sqrt(p['q']*max(md[lat[0]]-md[K-1],1.)); a0=np.exp(-0.5*(dg/sd0)**2); a0/=a0.sum()
    al=np.zeros((T,S)); al[0]=a0*np.exp(ll[0]-ll[0].max()); al[0]/=al[0].sum()
    for t in range(1,T): v=(al[t-1]@Tm)*np.exp(ll[t]-ll[t].max()); al[t]=v/v.sum()
    be=np.ones((T,S))
    for t in range(T-2,-1,-1): v=Tm@(be[t+1]*np.exp(ll[t+1]-ll[t+1].max())); be[t]=v/v.sum()
    post=al*be; post/=post.sum(1,keepdims=True); mean=post@dg; sdp=np.sqrt(np.maximum(post@(dg**2)-mean**2,0))
    out['pred']=pr_all+np.interp(np.arange(len(lat)),cen,mean); out['dmean']=mean; out['dsd']=np.interp(np.arange(len(lat)),cen,sdp); out['ls']=cen; return out
def evaluate3(rs,p=P3):
    ep=[];eb=[]
    for r in rs:
        d=W[NAMES[r['w']]]; K=d['K']; tv=d['TVT'][K:].astype(float); o=hmm_corr3(r,p); ep.append(o['pred']-tv); eb.append(o['prior']-tv)
    f=lambda e: float(np.sqrt(np.mean(np.concatenate(e)**2))); return f(eb),f(ep)
if __name__=="__main__":
    rng=np.random.RandomState(1); idx=rng.choice(len(R),120,replace=False); rs=[R[i] for i in idx]
    t0=time.time(); print("default:",evaluate3(rs),f"[{time.time()-t0:.0f}s]",flush=True)
    for name,vals in [('beta',[0.03,0.1,1.0,3.0]),('win',[50,100,400]),('q',[0.005,0.1]),('sigf',[0,2])]:
        print(name,' '.join(f"{v}:{evaluate3(rs,{**P3,name:v})[1]:.2f}" for v in vals),flush=True)
