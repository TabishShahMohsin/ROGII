# self-reference Viterbi: pass1 = user's anchor-Viterbi; pass2 = same but reference = blend(typelog, profile of GR vs TVT built from prefix + pass1 decode excluding +-L ft around chunk)
import pickle,sys,time,numpy as np
from vit import W,NAMES,lp,find_ms,bounds
B=0.5
def build_prof(tv,g,lo,nb):
    b=np.round(tv/B).astype(int)-lo;m=(b>=0)&(b<nb)
    s=np.bincount(b[m],g[m],nb);c=np.bincount(b[m],minlength=nb);return s,c
def find_sr(bnd,z,gr,tvt0,tw_gr,tw_tvt,pre_tv,pre_g,p1,alpha=0.5,L=300,minc=3,anchor_density=3,max_m_range=0.05,max_m=0.05,gamma=110000,nsel=5,minsep=7.0,c=2.0,pool=1000):
    allt=np.concatenate([pre_tv,p1]);lo=int(np.floor(np.nanmin(allt)/B))-40;hi=int(np.ceil(np.nanmax(allt)/B))+40;nb=hi-lo+1
    Sp,Cp=build_prof(pre_tv,pre_g,lo,nb);Sl,Cl=build_prof(p1,gr,lo,nb)
    nch=len(bnd)-1;a0=-z[0];anchors=[np.array([a0])]
    for i in range(1,len(bnd)):
        l_=a0-max_m_range*bnd[i];h_=a0+max_m_range*bnd[i];anchors.append(np.linspace(l_,h_,max(int((h_-l_)*anchor_density),1)))
    costs=[np.full(len(a),np.inf) for a in anchors];bp=[np.zeros(len(a),dtype=int) for a in anchors];costs[0][0]=0
    grid=(np.arange(nb)+lo)*B
    for i in range(nch):
        s,e=bnd[i],bnd[i+1];Lc=e-s;st=np.arange(Lc);cz=z[s:e];cg=gr[s:e];ca=anchors[i];na=anchors[i+1]
        ex=np.arange(max(s-L,0),min(e+L,len(p1)))
        Se,Ce=build_prof(p1[ex],gr[ex],lo,nb);S=Sp+Sl-Se;C=Cp+Cl-Ce
        prof=np.where(C>=minc,S/np.maximum(C,1),np.nan)
        Mm=(na[None,:]-ca[:,None])/Lc;ok=(costs[i][:,None]!=np.inf)&(np.abs(Mm)<=max_m);ci,ni=np.where(ok)
        if len(ci)==0:return None
        m=Mm[ci,ni][:,None];hyp=ca[ci][:,None]+m*st[None,:]+cz[None,:]+tvt0
        pg=np.interp(hyp.ravel(),tw_tvt,tw_gr)
        pr=np.interp(hyp.ravel(),grid,prof,left=np.nan,right=np.nan)  # nan-propagating interp: fix below
        # np.interp with nans in fp yields nan near them; fine -> fall back to typelog there
        pm=np.where(np.isnan(pr),pg,(1-alpha)*pg+alpha*pr).reshape(len(ci),Lc)
        cost=((cg[None,:]-pm)**2).sum(1)+gamma*Lc*m.ravel()**2+costs[i][ci]
        T=np.full((len(ca),len(na)),np.inf);T[ci,ni]=cost;mi=T.argmin(0);mc=T.min(0);u=mc<costs[i+1]
        costs[i+1][u]=mc[u];bp[i+1][u]=mi[u]
    fin=np.where(costs[-1]!=np.inf)[0];fin=fin[np.argsort(costs[-1][fin])][:pool];sel=[];sa=[]
    for node in fin:
        t=np.zeros(nch+1);k=node;t[-1]=anchors[-1][k]
        for i in range(nch-1,-1,-1):k=bp[i+1][k];t[i]=anchors[i][k]
        if all(np.mean(np.abs(t-q))>=minsep for q in sa):sel.append(node);sa.append(t)
        if len(sel)==nsel:break
    paths=[]
    for t in sa:
        p=np.zeros(len(z))
        for i in range(nch):
            s,e=bnd[i],bnd[i+1];mm=(t[i+1]-t[i])/(e-s);p[s:e]=t[i]+mm*np.arange(e-s)+z[s:e]+tvt0
        paths.append(p)
    tc=costs[-1][sel];w=np.exp(-c*(tc-tc.min())/(tc.max()-tc.min()+1e-8));w/=w.sum()
    return np.array(paths),w,tc
def runsr(wi,alphas=(0.5,),Ls=(300,)):
    d=W[NAMES[wi]];K=d['K'];n=len(d['MD'])-K
    if n<50:return None
    md=d['MD'][K:];Z=d['Z'][K:];mz,cz=np.polyfit(md,Z,1);z=mz*md+cz-Z;gr=lp(d['GR'][K:]);tvt0=d['TVT_input'][K-1];y=d['TVT'][K:]
    bnd=bounds(n,'unif',gr)
    r=find_ms(bnd,z,gr,tvt0,d['tw_gr'],d['tw_tvt'])
    if r is None:return None
    P,w,tc=r;p1=P[0];out=dict(n=n,hold=np.mean((tvt0-y)**2),w=wi);se=lambda p:np.mean((p-y)**2)
    out['p1_top1']=se(p1);out['p1_wavg']=se((w[:,None]*P).sum(0))
    pre_tv=d['TVT_input'][:K].astype(float);pre_g=lp(d['GR'][:K]);ok=~np.isnan(pre_tv);pre_tv=pre_tv[ok];pre_g=pre_g[ok]
    for a in alphas:
        for L in Ls:
            r2=find_sr(bnd,z,gr,tvt0,d['tw_gr'],d['tw_tvt'],pre_tv,pre_g,p1,alpha=a,L=L)
            if r2 is None:continue
            P2,w2,_=r2;out[f'a{a}_L{L}_top1']=se(P2[0]);out[f'a{a}_L{L}_wavg']=se((w2[:,None]*P2).sum(0));out[f'a{a}_L{L}_P']=P2;out[f'a{a}_L{L}_w']=w2
    out['P1']=P;out['w1']=w
    return out
if __name__=="__main__":
    a,b=int(sys.argv[1]),int(sys.argv[2]);perm=np.random.RandomState(0).permutation(len(NAMES));res=[];t0=time.time()
    for wi in perm[a:b]:
        try:r=runsr(int(wi),alphas=(0.5,0.8),Ls=(300,))
        except Exception as e:print('err',NAMES[wi],e);r=None
        if r:res.append(r)
        print(len(res),'%.0fs'%(time.time()-t0),flush=True)
    pickle.dump(res,open(f'vsr_{a}_{b}.pkl','wb'))
    ks=[k for k in res[0] if k.endswith('top1') or k.endswith('wavg') or k=='hold'];tot=lambda k:np.sqrt(sum(r[k]*r['n'] for r in res if k in r)/sum(r['n'] for r in res if k in r))
    print({k:round(tot(k),2) for k in ks})
