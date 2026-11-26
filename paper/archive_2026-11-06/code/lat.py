# Segment-Viterbi lattice in S = TVT - z_det - tvt0 space (user's formulation, Viterbi Physics Backbone notebook)
import numpy as np, pickle, sys

from common import W, NAMES, FOLD
from scipy.signal import butter, filtfilt
b_,a_=butter(1,0.4/0.5,btype='low')
def lowpass(x):
    i=np.arange(len(x)); m=np.isfinite(x)
    x=np.interp(i,i[m],x[m]) if m.sum()>1 else np.zeros(len(x))
    return filtfilt(b_,a_,x)
def prep(w,chunk=170,short=70):
    d=W[w]; K=d['K']; md=d['MD'][K:].astype(float); Z=d['Z'][K:].astype(float)
    n=len(md); mz,cz=np.polyfit(md,Z,1); z=(mz*md+cz)-Z
    tvt0=float(d['TVT_input'][K-1]); gr=lowpass(d['GR'][K:].astype(float))
    tw_t=d['tw_TVT'].astype(float); tw_g=d['tw_GR'].astype(float); o=np.argsort(tw_t); tw_t,tw_g=tw_t[o],tw_g[o]
    sel=(tw_t>tvt0-200)&(tw_t<tvt0+200); mu=np.nanmean(tw_g[sel]) if sel.sum()>10 else np.nanmean(tw_g); sd=(np.nanstd(tw_g[sel]) if sel.sum()>10 else np.nanstd(tw_g))+5
    nb=max(n//(short if n<1000 else chunk),1); bnd=np.linspace(0,n,nb+1).astype(int)
    out=dict(w=w,n=n,z=z,tvt0=tvt0,mu=mu,sd=sd,gr=(gr-mu)/sd,gr_raw=gr,tw_t=tw_t,tw_g=(tw_g-mu)/sd,bnd=bnd,mz=mz,md=md)
    if 'TVT' in d: out['tvt']=d['TVT'][K:].astype(float); out['S']=out['tvt']-z-tvt0
    return out
def anchors(p,step=1.0,mmax=0.05,cap=150.):
    S0=-p['z'][0]; A=[]
    for i,r in enumerate(p['bnd']):
        if i==0: A.append(np.array([S0])); continue
        R=min(mmax*r,cap); A.append(S0+np.arange(-np.floor(R/step),np.floor(R/step)+1)*step)
    return A
def path_tvt(p,anc):  # anc: S value at each boundary -> per-row TVT
    bnd=p['bnd']; S=np.zeros(p['n'])
    for i in range(len(bnd)-1):
        s0,s1=bnd[i],bnd[i+1]; L=s1-s0; S[s0:s1]=anc[i]+(anc[i+1]-anc[i])*np.arange(L)/L
    return S+p['z']+p['tvt0']
def true_anchors(p,A):
    bnd=p['bnd']; Sb=np.r_[p['S'][bnd[:-1]], p['S'][-1]]
    return np.array([a[np.argmin(np.abs(a-s))] for a,s in zip(A,Sb)]), Sb
if __name__=='__main__':
    for chunk,step in [(170,1/3),(170,1.0),(100,1.0),(300,1.0)]:
        se=n=0; se_c=0
        for w in NAMES:
            p=prep(w,chunk); A=anchors(p,step)
            ta,Sb=true_anchors(p,A); t=path_tvt(p,ta); e=t-p['tvt']; se+=np.sum(e*e); n+=len(e)
            e2=path_tvt(p,Sb)-p['tvt']; se_c+=np.sum(e2*e2)
        print(f'chunk {chunk} step {step:.2f}: lattice ceiling {np.sqrt(se/n):.3f}  (exact knots {np.sqrt(se_c/n):.3f})',flush=True)
