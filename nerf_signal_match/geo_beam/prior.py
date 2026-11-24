# Segment priors in geo = TVT+Z space, fit from train laterals (own implementation).
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
def rdp(x,y,tol):
    keep=np.zeros(len(x),bool); keep[[0,-1]]=True; st=[(0,len(x)-1)]
    while st:
        i,j=st.pop()
        if j-i<2: continue
        yh=y[i]+(x[i+1:j]-x[i])*(y[j]-y[i])/(x[j]-x[i]+1e-12); d=np.abs(y[i+1:j]-yh); k=int(np.argmax(d))
        if d[k]>tol: k+=i+1; keep[k]=True; st+=[(i,k),(k,j)]
    return np.flatnonzero(keep)
def segments(W,names,tol=0.75,win=151):
    L=[];S=[];pairs=[]
    for w in names:
        d=W[w]; K=d['K']; md=d['MD'][K-1:].astype(float); g=(d['TVT'][K-1:]+d['Z'][K-1:]).astype(float)
        if len(md)<win+5: continue
        gs=savgol_filter(g,win,3,mode='interp'); ks=rdp(md,gs,tol)
        l=np.diff(md[ks]); s=np.diff(gs[ks])/np.maximum(l,1e-9)
        L+=list(l); S+=list(s); pairs+=list(zip(s[:-1],s[1:]))
    return np.array(L),np.array(S),np.array(pairs)
def fit(W,names):
    L,S,P=segments(W,names); lh=np.log(np.clip(L,1,None))
    rho=float(np.sum(P[:,0]*P[:,1])/np.sum(P[:,0]**2))
    # sigma(len) = c*len^-gamma
    qs=np.quantile(L,np.linspace(0,1,9)); xs=[];ys=[]
    for a,b in zip(qs[:-1],qs[1:]):
        m=(L>=a)&(L<b)
        if m.sum()>30: xs.append(np.log(np.median(L[m]))); ys.append(np.log(S[m].std()))
    gneg,logc=np.polyfit(xs,ys,1)
    return dict(mu=float(lh.mean()),sd=float(lh.std()),rho=rho,gamma=float(-gneg),logc=float(logc),n=len(L))
def logpriors(p,durs,sc,rho_clip=0.96,rho_scale=2.0,sig_scale=1.3,w_smag=1.5,smag=0.09):
    """log p(duration class) and log p(s | prev s, duration) as bin masses on the slope grid."""
    durs=np.asarray(durs,float); sc=np.asarray(sc,float)
    de=np.r_[1,np.sqrt(durs[:-1]*durs[1:]),1e5]; se=np.r_[-10,(sc[:-1]+sc[1:])/2,10]
    cdf=lambda e: stats.norm.cdf((np.log(e)-p['mu'])/p['sd'])
    lpb=np.log(np.clip(np.diff(cdf(de)),1e-12,None))
    rho=min(p['rho']*rho_scale,rho_clip); lps=np.zeros((len(durs),len(sc),len(sc)))
    for bi,dl in enumerate(durs):
        sig=np.exp(p['logc'])*dl**(-p['gamma'])*sig_scale*np.sqrt(max(1-rho**2,1e-3))
        for j in range(len(sc)):
            lps[bi,j]=np.log(np.clip(np.diff(stats.norm.cdf((se-rho*sc[j])/sig)),1e-12,None))
    if w_smag>0: lps+=w_smag*(-0.5*(sc/smag)**2)[None,None,:]
    return lpb,lps
if __name__=='__main__':
    from common import W,NAMES
    p=fit(W,NAMES); print(p)
