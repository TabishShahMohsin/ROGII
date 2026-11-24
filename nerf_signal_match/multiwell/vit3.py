import sys,pickle,time,numpy as np
from vit import W,NAMES,lp,bounds
def fm(bnd,z,gr,tvt0,tw_gr,tw_tvt,m0=None,w0=0.0,cap=1e12,gamma=110000,max_m=0.05,mr=0.05,dens=3,minsep=7.0,nsel=5):
    nch=len(bnd)-1; a0=-z[0]; anchors=[np.array([a0])]
    for i in range(1,len(bnd)):
        lo=a0-mr*bnd[i]; hi=a0+mr*bnd[i]; anchors.append(np.linspace(lo,hi,max(int((hi-lo)*dens),1)))
    costs=[np.full(len(a),np.inf) for a in anchors]; bp=[np.zeros(len(a),dtype=int) for a in anchors]; costs[0][0]=0
    for i in range(nch):
        s,e=bnd[i],bnd[i+1]; L=e-s; st=np.arange(L); cz=z[s:e]; cg=gr[s:e]; ca=anchors[i]; na=anchors[i+1]
        Mm=(na[None,:]-ca[:,None])/L; ok=(costs[i][:,None]!=np.inf)&(np.abs(Mm)<=max_m); ci,ni=np.where(ok)
        if len(ci)==0: return None
        cost=np.empty(len(ci)); BL=max(1,4000000//L)
        for q0 in range(0,len(ci),BL):
            cq=ci[q0:q0+BL]; nq=ni[q0:q0+BL]; m=Mm[cq,nq][:,None]; hyp=ca[cq][:,None]+m*st[None,:]+cz[None,:]+tvt0
            pg=np.interp(hyp.ravel(),tw_tvt,tw_gr).reshape(len(cq),L); cost[q0:q0+BL]=np.minimum((cg[None,:]-pg)**2,cap).sum(1)+gamma*L*m.ravel()**2+costs[i][cq]
        m=Mm[ci,ni][:,None]
        if i==0 and m0 is not None: cost+=w0*L*(m.ravel()-m0)**2
        T=np.full((len(ca),len(na)),np.inf); T[ci,ni]=cost; mi=T.argmin(0); mc=T.min(0); u=mc<costs[i+1]; costs[i+1][u]=mc[u]; bp[i+1][u]=mi[u]
    fin=np.where(costs[-1]!=np.inf)[0]; fin=fin[np.argsort(costs[-1][fin])]; sa=[]; sel=[]
    for node in fin:
        t=np.zeros(nch+1); k=node; t[-1]=anchors[-1][k]
        for i in range(nch-1,-1,-1): k=bp[i+1][k]; t[i]=anchors[i][k]
        if all(np.mean(np.abs(t-q))>=minsep for q in sa): sa.append(t); sel.append(node)
        if len(sa)==nsel: break
    P=[]
    for t in sa:
        p=np.zeros(len(z))
        for i in range(nch): s,e=bnd[i],bnd[i+1]; p[s:e]=t[i]+(t[i+1]-t[i])/(e-s)*np.arange(e-s)+z[s:e]+tvt0
        P.append(p)
    return np.array(P),costs[-1][sel]
def prep(d,cal=False,span=1500):
    K=d['K']; md=d['MD'][K:]; Z=d['Z'][K:]; mz,cz=np.polyfit(md,Z,1); z=mz*md+cz-Z; gr=lp(d['GR'][K:]); tvt0=d['TVT_input'][K-1]; twg=d['tw_gr'].astype(float)
    if cal:
        i0=max(0,K-span); gp=d['GR'][i0:K]; tv=d['TVT_input'][i0:K]; ok=~np.isnan(gp)&~np.isnan(tv)
        if ok.sum()>100:
            f=np.interp(tv[ok],d['tw_tvt'],twg); a,b=np.polyfit(f,gp[ok],1)
            if 0.5<a<2: twg=a*twg+b
    s=slice(max(0,K-500),K); sl=np.polyfit(d['MD'][s],d['TVT_input'][s]+d['Z'][s],1)[0]; m0=np.clip(sl-mz,-0.05,0.05)
    return z,gr,tvt0,twg,m0,d['TVT'][K:]
if __name__=="__main__":
    cfg=sys.argv[1]; kw=dict(base={},cal={},cap=dict(cap=900.0),m0=dict(w0=110000.0),calcap=dict(cap=900.0),all=dict(cap=900.0,w0=110000.0))[cfg]
    perm=np.random.RandomState(0).permutation(len(NAMES)); out=[]; t0=time.time()
    for wi in perm[500:540]:
        wi=int(wi); d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K
        if n<50: continue
        z,gr,tvt0,twg,m0,y=prep(d,cal=cfg in('cal','calcap','all')); b=bounds(n,'unif')
        r=fm(b,z,gr,tvt0,twg,d['tw_tvt'],m0=m0,**kw)
        if r: out.append(dict(w=wi,y=y,p=r[0][0],P=r[0],n=n))
    se=sum(((o['p']-o['y'])**2).sum() for o in out); nn=sum(o['n'] for o in out); pickle.dump(out,open(f'v3_{cfg}.pkl','wb')); print(cfg,len(out),'%.2f'%np.sqrt(se/nn),'%.0fs'%(time.time()-t0))
