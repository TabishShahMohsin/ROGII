from common import *
STEP=50; HS=[800,1200,2000,4000]
def build(wi):
    w=NAMES[wi]; d,K,lat=lateral(w); md=d['MD']; z=d['Z'].astype(np.float64); tvt=d['TVT'].astype(np.float64); g=tvt+z
    qi=np.unique(np.r_[lat[::STEP],lat[-1]]); pre=np.arange(max(0,K-300),K,30); allq=np.r_[pre,qi]; ex={wi}; row={}
    for h in HS:
        a,ws=nb2(d['X'][allq],d['Y'][allq],h,ex,ret_w=True); ap,aq=a[:len(pre)],a[len(pre):]
        c_win=np.mean(g[pre]-ap); row[f'V_end{h}']=aq-ap[-1]-(z[qi]-z[K-1]); row[f'V_win{h}']=(c_win+aq-z[qi])-tvt[K-1]; row[f'vw{h}']=ws[len(pre):]
        row[f'vgrad{h}']=np.gradient(aq,md[qi]) if len(qi)>1 else np.zeros(len(qi))
    return row
if __name__=="__main__":
    t0=time.time(); R=pickle.load(open('ds.pkl','rb'))
    for wi in range(len(NAMES)):
        assert R[wi]['w']==wi; R[wi].update(build(wi))
        if wi%100==0: print(wi,round(time.time()-t0),flush=True)
    pickle.dump(R,open('ds3.pkl','wb')); print("done",round(time.time()-t0))
