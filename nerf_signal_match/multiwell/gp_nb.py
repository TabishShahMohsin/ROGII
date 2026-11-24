import numpy as np,time,sys
from scipy.linalg import cho_factor,cho_solve
from common import *
import common
perm=np.random.RandomState(0).permutation(len(NAMES)); TE=[int(w) for w in perm[500:773]]
def m52(r): s=np.sqrt(5)*r; return (1+s+s*s/3)*np.exp(-s)
def kern(A,B,ell,s2): 
    d=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1))/ell; return s2*m52(d)
def gp_incr(wi,ell=1500.,tau2=25.,s2=None,rad=4000.,maxn=900,excl=None,grid=50):
    d=W[NAMES[wi]]; K=d['K']; n=len(d['MD']); lat=np.arange(K,n,grid); lat=np.r_[K-1,lat]; xq=np.c_[d['X'][lat],d['Y'][lat]]
    ids=common.TREE.query_ball_point(xq[::max(1,len(xq)//40)],rad); sel=np.unique(np.concatenate([np.array(i,int) for i in ids])) if len(ids) else np.array([],int)
    ex=np.zeros(len(NAMES),bool); ex[list(excl if excl is not None else {wi})]=True; sel=sel[~ex[PW[sel]]]
    if len(sel)<10: return None
    if len(sel)>maxn:
        c=xq.mean(0); dd=np.linalg.norm(np.c_[PX[sel],PY[sel]]-c,axis=1); sel=sel[np.argsort(dd)[:maxn]]
    X=np.c_[PX[sel],PY[sel]]; mu0=common._poly(PX[sel],PY[sel],2)@common._BETA; r=PA[sel]-mu0
    s2_=s2 if s2 is not None else max(np.var(r),1.)
    Kxx=kern(X,X,ell,s2_)+tau2*np.eye(len(X)); cf=cho_factor(Kxx,lower=True); al=cho_solve(cf,r)
    Ks=kern(xq,X,ell,s2_); mean=common._poly(xq[:,0],xq[:,1],2)@common._BETA+Ks@al
    V=cho_solve(cf,Ks.T); cov=kern(xq,xq,ell,s2_)-Ks@V
    inc=mean[1:]-mean[0]; var=np.diag(cov)[1:]+cov[0,0]-2*cov[1:,0]
    return lat[1:],inc,np.sqrt(np.maximum(var,1e-6)),len(sel)
def tvt_hat(wi,lat,inc):
    d=W[NAMES[wi]]; K=d['K']; return d['TVT_input'][K-1]+inc-(d['Z'][lat]-d['Z'][K-1])
if __name__=="__main__":
    ws=TE[:70]; t0=time.time()
    for h in [1000,2000,3000]:
        errs=[]
        for wi in ws:
            d=W[NAMES[wi]]; K=d['K']; n=len(d['MD']); lat=np.r_[np.arange(K,n,50)]; xq=np.c_[d['X'][lat],d['Y'][lat]]
            sf=common.nb2(np.r_[d['X'][K-1],d['X'][lat]],np.r_[d['Y'][K-1],d['Y'][lat]],h,{wi}); inc=sf[1:]-sf[0]; th=d['TVT_input'][K-1]+inc-(d['Z'][lat]-d['Z'][K-1]); errs.append(th-d['TVT'][lat])
        print('nb2 h=%d: pooled RMSE %.2f  [%.0fs]'%(h,pooled(errs),time.time()-t0),flush=True)
    for ell in [800,1500,3000]:
        for tau2 in [25,100]:
            errs=[]
            for wi in ws:
                r=gp_incr(wi,ell=ell,tau2=tau2)
                if r is None: continue
                lat,inc,sd,nn=r; errs.append(tvt_hat(wi,lat,inc)-W[NAMES[wi]]['TVT'][lat])
            print('GP matern52 ell=%d tau2=%d: pooled RMSE %.2f [%.0fs]'%(ell,tau2,pooled(errs),time.time()-t0),flush=True)
