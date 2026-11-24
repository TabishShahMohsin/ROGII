# Viterbi with internals: min-marginals (fwd+bwd), posterior over anchors, soft decode, physics diagnostics per lateral row
import pickle,sys,time,numpy as np
from vit import W,NAMES,lp,bounds
def vint(wi,anchor_density=3,max_m_range=0.05,max_m=0.05,gamma=110000,minsep=7.0,nsel=5,pool=1000):
    d=W[NAMES[wi]];K=d['K'];n=len(d['MD'])-K
    if n<50:return None
    md=d['MD'][K:];Z=d['Z'][K:];mz,cz=np.polyfit(md,Z,1);z=mz*md+cz-Z;gr=lp(d['GR'][K:]);tvt0=float(d['TVT_input'][K-1]);tw_gr=d['tw_gr'];tw_tvt=d['tw_tvt']
    bnd=bounds(n,'unif',gr);nch=len(bnd)-1
    a0=-z[0];anchors=[np.array([a0])]
    for i in range(1,len(bnd)):
        lo=a0-max_m_range*bnd[i];hi=a0+max_m_range*bnd[i];anchors.append(np.linspace(lo,hi,max(int((hi-lo)*anchor_density),1)))
    costs=[np.full(len(a),np.inf) for a in anchors];bp=[np.zeros(len(a),dtype=int) for a in anchors];costs[0][0]=0;E=[None]*nch
    for i in range(nch):
        s,e=bnd[i],bnd[i+1];L=e-s;st=np.arange(L);cz_=z[s:e];cg=gr[s:e];ca=anchors[i];na=anchors[i+1]
        Mm=(na[None,:]-ca[:,None])/L;ok=(costs[i][:,None]!=np.inf)&(np.abs(Mm)<=max_m);ci,ni=np.where(ok)
        if len(ci)==0:return None
        m=Mm[ci,ni][:,None];hyp=ca[ci][:,None]+m*st[None,:]+cz_[None,:]+tvt0
        pg=np.interp(hyp.ravel(),tw_tvt,tw_gr).reshape(len(ci),L);edge=((cg[None,:]-pg)**2).sum(1)+gamma*L*m.ravel()**2
        Em=np.full((len(ca),len(na)),np.inf,dtype=np.float64);Em[ci,ni]=edge;E[i]=Em.astype(np.float32)
        T=Em+costs[i][:,None];mi=T.argmin(0);mc=T.min(0);u=mc<costs[i+1];costs[i+1][u]=mc[u];bp[i+1][u]=mi[u]
    # backward
    bwd=[None]*(nch+1);bwd[nch]=np.zeros(len(anchors[nch]))
    for i in range(nch-1,-1,-1):bwd[i]=(E[i].astype(np.float64)+bwd[i+1][None,:]).min(1)
    best=costs[-1].min();sig2=max(best/n,1.0)
    fin=np.where(costs[-1]!=np.inf)[0];fin=fin[np.argsort(costs[-1][fin])][:pool];sel=[];sa=[]
    for node in fin:
        t=np.zeros(nch+1);k=node;t[-1]=anchors[-1][k]
        for i in range(nch-1,-1,-1):k=bp[i+1][k];t[i]=anchors[i][k]
        if all(np.mean(np.abs(t-q))>=minsep for q in sa):sel.append(node);sa.append(t)
        if len(sel)==nsel:break
    tc=costs[-1][sel]
    # per-boundary posterior stats (TVT shift is constant per boundary => use anchor values)
    mapA=sa[0];ST={}
    for tau in (30,300):
        pm=np.zeros(nch+1);psd=np.zeros(nch+1);pent=np.zeros(nch+1);far=np.zeros(nch+1);mode2=np.zeros(nch+1);nmode=np.zeros(nch+1);curv=np.zeros(nch+1)
        for i in range(nch+1):
            mm=costs[i]+bwd[i];ok=np.isfinite(mm);a=anchors[i]
            if ok.sum()<2:pm[i]=mapA[i];continue
            l=-(mm[ok]-mm[ok].min())/(2*sig2*tau);p=np.exp(l-l.max());p/=p.sum();aa=a[ok];mu=(p*aa).sum();pm[i]=mu;psd[i]=np.sqrt((p*(aa-mu)**2).sum());pent[i]=-(p*np.log(p+1e-12)).sum()
            far[i]=p[np.abs(aa-mapA[i])>8].sum()
            pk=np.where((p[1:-1]>p[:-2])&(p[1:-1]>=p[2:])&(p[1:-1]>0.02*p.max()))[0]+1;nmode[i]=len(pk)
            if far[i]>0:
                mk_=np.abs(aa-mapA[i])>8;j=np.argmax(np.where(mk_,p,0));mode2[i]=aa[j]-mapA[i]
            j0=np.argmin(np.abs(aa-mapA[i]));lo_,hi_=max(j0-9,0),min(j0+10,len(aa)-1);curv[i]=(mm[ok][lo_]+mm[ok][hi_]-2*mm[ok][j0])/(2*sig2)/9.0**2 if hi_>lo_ else 0
        ST[tau]=(pm,psd,pent,far,mode2,nmode,curv)
    pm=ST[30][0]
    idx=np.arange(n);bx=np.array(bnd)
    I=lambda v:np.interp(idx,bx,v)
    # paths
    def mk(t):
        p=np.zeros(n)
        for i in range(nch):
            s,e=bnd[i],bnd[i+1];p[s:e]=t[i]+(t[i+1]-t[i])/(e-s)*np.arange(e-s)+z[s:e]+tvt0
        return p
    Pm=mk(mapA);Ppm=mk(pm);P=np.array([mk(t) for t in sa])
    slope=np.zeros(n);kink=np.zeros(n)
    sl=[(mapA[i+1]-mapA[i])/(bnd[i+1]-bnd[i]) for i in range(nch)]
    for i in range(nch):slope[bnd[i]:bnd[i+1]]=sl[i];kink[bnd[i]:bnd[i+1]]=abs(sl[i]-sl[i-1]) if i>0 else 0
    # physics: local fit along MAP path
    twp=np.interp(Pm,tw_tvt,tw_gr);res=gr-twp;k=np.ones(101)/101
    fit_rms=np.sqrt(np.convolve(res**2,k,'same'));gc=np.convolve(gr,k,'same');tc_=np.convolve(twp,k,'same')
    cov=np.convolve((gr-gc)*(twp-tc_),k,'same');vg=np.convolve((gr-gc)**2,k,'same');vt=np.convolve((twp-tc_)**2,k,'same');lcorr=cov/np.sqrt(vg*vt+1e-9)
    twg=np.abs(np.gradient(tw_gr,tw_tvt));tw_sens=np.convolve(np.interp(Pm,tw_tvt,twg),k,'same')
    tw_loc_sd=np.sqrt(np.convolve((twp-tc_)**2,k,'same'))
    edge=np.minimum(Pm-tw_tvt.min(),tw_tvt.max()-Pm)
    # fit rms of the second-best path (alias competitor) and its gap
    twp2=np.interp(P[1],tw_tvt,tw_gr) if len(P)>1 else twp;fit2=np.sqrt(np.convolve((gr-twp2)**2,k,'same'))
    out=dict(w=wi,n=n,K=K,tvt0=tvt0,map=Pm,pmean=Ppm,P=P,tc=tc,
        **{f'{nm}{tau}':I(ST[tau][j]) for tau in (30,300) for j,nm in enumerate(['pm_','post_sd','post_ent','far','mode2','nmode','curv']) if nm!='pm_'},pmean300=mk(ST[300][0]),slope=slope,kink=kink,
        fit_rms=fit_rms,lcorr=lcorr,tw_sens=tw_sens,tw_loc_sd=tw_loc_sd,edge=edge,fit2_minus_fit1=fit2-fit_rms,
        spread=P.std(0) if len(P)>1 else np.zeros(n),rng5=P.max(0)-P.min(0),
        gap12=(tc[1]-tc[0])/(best+1e-9) if len(tc)>1 else 1.0,cost_per_row=best/n,nch=nch,sig2=sig2,
        pmean_minus_map=Ppm-Pm,y=d['TVT'][K:])
    return out
if __name__=="__main__":
    k=int(sys.argv[1]);perm=np.random.RandomState(0).permutation(len(NAMES));order=list(perm[500:773])+list(perm[:500]);mine=order[k::2];res=[];t0=time.time();cnt=0
    for wi in mine:
        try:r=vint(int(wi))
        except Exception as ex:print('err',wi,ex,flush=True);r=None
        if r:res.append(r)
        cnt+=1
        if cnt%20==0:
            pickle.dump(res,open(f'vint_{k}.pkl','wb'));print(cnt,'%.0fs'%(time.time()-t0),flush=True)
    pickle.dump(res,open(f'vint_{k}.pkl','wb'));print('done',flush=True)
