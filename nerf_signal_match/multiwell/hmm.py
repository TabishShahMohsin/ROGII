import pickle, numpy as np, sys, time
from scipy.ndimage import gaussian_filter1d
from common import W,NAMES
R=pickle.load(open('ds.pkl','rb'))
P=dict(ST=5,D=40.,dstep=1.0,q=0.05,rho=1.0,tau=0.15,nu=3.0,sigf=2.0,P=1500,corr_min=0.2,sig_floor=0.15,usegr=1)
def field_fn(tw_tvt,tw_gr,sig):
    dw=np.median(np.diff(tw_tvt)); g=gaussian_filter1d(tw_gr.astype(np.float64),sig/dw,mode='nearest') if sig>0 else tw_gr.astype(np.float64)
    return lambda w: np.interp(w,tw_tvt,g)
def calib(d,K,f,Pn):
    i0=max(0,K-Pn); gr=d['GR'][i0:K].astype(np.float64); tv=d['TVT'][i0:K].astype(np.float64); ok=~np.isnan(gr)
    if ok.sum()<50: return 0.,0.,1.,0.
    fp=f(tv[ok]); A=np.c_[fp,np.ones(ok.sum())]; (a,b),*_=np.linalg.lstsq(A,gr[ok],rcond=None); res=gr[ok]-(a*fp+b)
    sig=1.4826*np.median(np.abs(res-np.median(res))); c=np.corrcoef(fp,gr[ok])[0,1]; return a,b,max(sig,1e-3),c
def hmm_well(r,p=P,ret_extra=False):
    w=NAMES[r['w']]; d=W[w]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md))
    prior_lat=r['tvt_end']+np.interp(lat,r['idx'],r['yA']); out=dict(prior=prior_lat)
    f=field_fn(d['tw_tvt'],d['tw_gr'],p['sigf']); a,b,sg,c=calib(d,K,f,p['P']); out['corr']=c
    if not p['usegr'] or c<p['corr_min'] or abs(a)<1e-6: out['pred']=prior_lat; return out
    ls=lat[::p['ST']]; pr=r['tvt_end']+np.interp(ls,r['idx'],r['yA']); gr=d['GR'][ls].astype(np.float64); ok=~np.isnan(gr); sg=max(sg,p['sig_floor']*gr[ok].std() if ok.any() else 1)
    dg=np.arange(-p['D'],p['D']+1e-9,p['dstep']); S=len(dg); T=len(ls)
    ds=md[ls[0]]-md[K-1]; step=p['ST']
    # emission log-lik [T,S] (Student-t, tempered)
    wmat=pr[:,None]+dg[None,:]; mu=a*f(wmat)+b; res=(gr[:,None]-mu)/sg
    ll=-(p['nu']+1)/2*np.log1p(res**2/p['nu'])*p['tau']; ll[~ok]=0
    # transition: d_{t+1}=rho d_t + N(0,q*step)
    sd=np.sqrt(p['q']*step); Tm=np.exp(-0.5*((dg[None,:]-p['rho']*dg[:,None])/sd)**2); Tm/=Tm.sum(1,keepdims=True)
    sd0=np.sqrt(p['q']*max(ds,1.)); a0=np.exp(-0.5*(dg/sd0)**2); a0/=a0.sum()
    al=np.zeros((T,S)); al[0]=a0*np.exp(ll[0]-ll[0].max()); al[0]/=al[0].sum()
    for t in range(1,T): v=(al[t-1]@Tm)*np.exp(ll[t]-ll[t].max()); al[t]=v/v.sum()
    be=np.ones((T,S))
    for t in range(T-2,-1,-1): v=Tm@(be[t+1]*np.exp(ll[t+1]-ll[t+1].max())); be[t]=v/v.sum()
    post=al*be; post/=post.sum(1,keepdims=True); mean=post@dg; sd_post=np.sqrt(np.maximum(post@(dg**2)-mean**2,0)); out['dsd']=np.interp(lat,ls,sd_post)
    out['pred']=np.interp(lat,ls,pr+mean); out['dmean']=mean; return out
def evaluate(rs,p=P,verbose=False):
    ep=[];eb=[];per=[]
    for r in rs:
        d=W[NAMES[r['w']]]; K=d['K']; tv=d['TVT'][K:].astype(np.float64); o=hmm_well(r,p); ep.append(o['pred']-tv); eb.append(o['prior']-tv)
    f=lambda e: np.sqrt(np.mean(np.concatenate(e)**2)); return f(eb),f(ep),ep,eb
if __name__=="__main__":
    rng=np.random.RandomState(1); idx=rng.choice(len(R),120,replace=False); rs=[R[i] for i in idx]
    t0=time.time(); pb,pp,ep,eb=evaluate(rs); print(f"dev 120 wells: prior-only {pb:.2f} -> prior+GR(HMM) {pp:.2f}  [{time.time()-t0:.0f}s]")
    pw=np.array([[np.sqrt(np.mean(a**2)),np.sqrt(np.mean(b**2))] for a,b in zip(ep,eb)]); print("per-well median: prior %.2f -> GR %.2f ; wells improved %.0f%%"%(np.median(pw[:,1]),np.median(pw[:,0]),100*(pw[:,0]<pw[:,1]).mean()))
