from common import *
import sys
STEP=50; HS=[3000,6000]
def build(wi):
    w=NAMES[wi]; d,K,lat=lateral(w); md=d['MD']; z=d['Z'].astype(np.float64); tvt=d['TVT'].astype(np.float64); g=tvt+z
    qi=np.unique(np.r_[lat[::STEP],lat[-1]]); pre=np.arange(max(0,K-300),K,30); allq=np.r_[pre,qi]
    ex={wi}; row={'w':wi,'K':K,'idx':qi,'md':md[qi],'dist':md[qi]-md[K-1],'z':z[qi],'dz':z[qi]-z[K-1],'y':tvt[qi]-tvt[K-1],'tvt_end':tvt[K-1]}
    for h in HS:
        a,ws=nb_surface(wi,d['X'][allq],d['Y'][allq],h=h,excl=ex,degree=1,ret_w=True)
        ok=~np.isnan(a)
        if ok.sum()<2: a=np.zeros(len(a)); ws=np.zeros(len(a)); ok=np.ones(len(a),bool)
        a=np.interp(np.arange(len(a)),np.where(ok)[0],a[ok]); ap,aq=a[:len(pre)],a[len(pre):]
        c_end=g[K-1]-ap[-1]; c_win=np.mean(g[pre]-ap)          # anchor at end point vs mean over last 300 prefix rows
        row[f'a{h}']=aq; row[f'Inb_end{h}']=aq-ap[-1]-(z[qi]-z[K-1]); row[f'Inb_win{h}']=(c_win+aq-z[qi])-tvt[K-1]; row[f'ws{h}']=ws[len(pre):]
    # local dip extrapolation (g-slope over last 300 / 1000 prefix rows)
    for M in [300,1000]:
        m=slice(max(0,K-M),K); s=np.polyfit(md[m],g[m],1)[0]; row[f'Ilin{M}']=s*(md[qi]-md[K-1])-(z[qi]-z[K-1])
    return row
if __name__=="__main__":
    t0=time.time(); R=[]
    for wi in range(len(NAMES)):
        R.append(build(wi))
        if wi%100==0: print(wi,round(time.time()-t0),flush=True)
    pickle.dump(R,open('ds.pkl','wb')); print("done",round(time.time()-t0))
