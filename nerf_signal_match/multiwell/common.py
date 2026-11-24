import pickle, numpy as np, time
from scipy.spatial import cKDTree
W=pickle.load(open('wells.pkl','rb')); NAMES=sorted(W)
SUB=25
# point cloud of structure surface (ANCC) from ALL wells (prefix+lateral), subsampled
PX=[];PY=[];PA=[];PW=[]
for i,w in enumerate(NAMES):
    d=W[w]; a=d['ANCC']; m=~np.isnan(a); idx=np.where(m)[0][::SUB]
    PX.append(d['X'][idx]);PY.append(d['Y'][idx]);PA.append(a[idx].astype(np.float64));PW.append(np.full(len(idx),i))
PX=np.concatenate(PX);PY=np.concatenate(PY);PA=np.concatenate(PA);PW=np.concatenate(PW)
TREE=cKDTree(np.c_[PX,PY])
def lateral(w):
    d=W[w]; K=d['K']; return d,K,np.arange(K,len(d['MD']))
def pooled(errs): 
    e=np.concatenate(errs); return float(np.sqrt(np.mean(e**2)))
def excluded_wells(wi,r):
    """indices of wells having any (sub-sampled) point within r ft of target's whole path (includes self)"""
    d=W[NAMES[wi]]; q=np.c_[d['X'][::SUB],d['Y'][::SUB]]
    ids=TREE.query_ball_point(q,r); s=set([wi])
    for l in ids: s.update(PW[l].tolist())
    return s
def nb_surface(wi,xq,yq,h=3000.,excl=None,degree=1,minw=1e-3,ret_w=False):
    """Gaussian-kernel local-linear (or NW if degree=0) estimate of ANCC(x,y) from all wells except `excl`"""
    excl=excl if excl is not None else {wi}; ex=np.zeros(len(NAMES),bool); ex[list(excl)]=True
    r=3.5*h; m=(~ex[PW])&(PX>xq.min()-r)&(PX<xq.max()+r)&(PY>yq.min()-r)&(PY<yq.max()+r)
    cx,cy,ca=PX[m],PY[m],PA[m]; out=np.full(len(xq),np.nan)
    wsum=np.zeros(len(xq))
    if len(cx)<8: return (out,wsum) if ret_w else out
    for j in range(len(xq)):
        dx=cx-xq[j]; dy=cy-yq[j]; d2=dx*dx+dy*dy; k=d2<r*r
        if k.sum()<8: continue
        dxk,dyk,ak=dx[k],dy[k],ca[k]; ww=np.exp(-0.5*d2[k]/h**2)
        if ww.sum()<minw: continue
        wsum[j]=ww.sum()
        if degree==0: out[j]=(ww*ak).sum()/ww.sum()
        else:
            A=np.c_[np.ones(k.sum()),dxk,dyk]; M=A.T@(A*ww[:,None])+1e-9*np.eye(3)*ww.sum(); out[j]=np.linalg.solve(M,(A*ww[:,None]).T@ak)[0]
    return (out,wsum) if ret_w else out

# ---- v2 surface: global quadratic trend + Nadaraya-Watson smoothing of residuals, shrunk to trend where data sparse
_x0,_y0=PX.mean(),PY.mean(); _sx,_sy=PX.std(),PY.std()
def _poly(x,y,deg=2):
    u=(x-_x0)/_sx; v=(y-_y0)/_sy; return np.c_[tuple(u**i*v**j for i in range(deg+1) for j in range(deg+1-i))]
_A=_poly(PX,PY,2); _BETA=np.linalg.lstsq(_A,PA,rcond=None)[0]; _RES=PA-_A@_BETA
def nb2(xq,yq,h,excl,lam=0.5,ret_w=False):
    ex=np.zeros(len(NAMES),bool); ex[list(excl)]=True; rr=3.5*h
    m=(~ex[PW])&(PX>xq.min()-rr)&(PX<xq.max()+rr)&(PY>yq.min()-rr)&(PY<yq.max()+rr); cx,cy,cr=PX[m],PY[m],_RES[m]; out=np.zeros(len(xq)); wsum=np.zeros(len(xq))
    for j in range(len(xq)):
        d2=(cx-xq[j])**2+(cy-yq[j])**2; k=d2<rr*rr
        if k.sum()<5: continue
        ww=np.exp(-0.5*d2[k]/h**2); s=ww.sum(); wsum[j]=s; out[j]=(ww*cr[k]).sum()/(s+1e-9)*(s/(s+lam))
    res=_poly(xq,yq,2)@_BETA+out
    return (res,wsum) if ret_w else res
