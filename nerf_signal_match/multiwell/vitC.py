# Corridor Viterbi: boundary anchors restricted to centre-path +- R on a fine grid (needle-width resolution), data cost = plain SSE (best localiser of the geologist pick)
import sys,pickle,time,numpy as np,pandas as pd
from vit3 import W,NAMES,lp,bounds,prep
def LOSS(g,f,kind):
    r=g-f
    if kind=='sse': return r**2
    if kind=='huber': a=np.abs(r); return np.where(a<10,0.5*r**2,10*(a-5))
    if kind=='hetero': return r**2/(0.7*np.maximum(f,5)+20)
    if kind=='log': return (np.log(np.maximum(g,5))-np.log(np.maximum(f,5)))**2*1000
    if kind=='abs': return np.abs(r)*10
    if kind=='cap': return np.minimum(r**2,625)
    if kind=='sq4': return np.sqrt(np.abs(r))**4*0+ (np.abs(r)**1.5)*5
def fmc(bnd,z,gr,tvt0,tw_gr,tw_tvt,center,R=20.,step=0.25,gamma=0.,max_m=0.15,nsel=1,loss='sse'):
    nch=len(bnd)-1; cA=center-z-tvt0   # centre path in anchor (A) space
    anchors=[np.array([-z[0]])]
    for i in range(1,len(bnd)):
        c=cA[min(bnd[i],len(cA)-1)]; anchors.append(np.arange(c-R,c+R+1e-9,step))
    costs=[np.full(len(a),np.inf) for a in anchors]; bp=[np.zeros(len(a),dtype=int) for a in anchors]; costs[0][0]=0
    for i in range(nch):
        s,e=bnd[i],bnd[i+1]; L=e-s; st=np.arange(L); cz=z[s:e]; cg=gr[s:e]; ca=anchors[i]; na=anchors[i+1]
        Mm=(na[None,:]-ca[:,None])/L; ok=(costs[i][:,None]!=np.inf)&(np.abs(Mm)<=max_m); ci,ni=np.where(ok)
        if len(ci)==0: return None
        cost=np.empty(len(ci)); BL=max(1,3000000//L)
        for q0 in range(0,len(ci),BL):
            cq=ci[q0:q0+BL]; nq=ni[q0:q0+BL]; m=Mm[cq,nq][:,None]; hyp=ca[cq][:,None]+m*st[None,:]+cz[None,:]+tvt0
            pg=np.interp(hyp.ravel(),tw_tvt,tw_gr).reshape(len(cq),L); cost[q0:q0+BL]=LOSS(cg[None,:],pg,loss).sum(1)+gamma*L*m.ravel()**2+costs[i][cq]
        T=np.full((len(ca),len(na)),np.inf); T[ci,ni]=cost; mi=T.argmin(0); mc=T.min(0); u=mc<costs[i+1]; costs[i+1][u]=mc[u]; bp[i+1][u]=mi[u]
    k=int(np.argmin(costs[-1])); t=np.zeros(nch+1); t[-1]=anchors[-1][k]
    for i in range(nch-1,-1,-1): k=bp[i+1][k]; t[i]=anchors[i][k]
    p=np.zeros(len(z))
    for i in range(nch): s,e=bnd[i],bnd[i+1]; p[s:e]=t[i]+(t[i+1]-t[i])/(e-s)*np.arange(e-s)+z[s:e]+tvt0
    return p
def tbounds(d,K,tol=1.5):
    md=d['MD'][K:]; s=(d['TVT'][K:]+d['Z'][K:]); n=len(md); i=0; bs=[0]
    while i<n-1:
        j=min(i+20,n-1)
        while j<n-1:
            k=min(j+20,n-1); c=np.polyfit(md[i:k+1],s[i:k+1],1)
            if np.max(np.abs(np.polyval(c,md[i:k+1])-s[i:k+1]))>tol: break
            j=k
        bs.append(j); i=j
    bs[-1]=n; return sorted(set(bs))
if __name__=="__main__":
    cfg=sys.argv[1]; df=pd.read_pickle('fail2.pkl'); bad=list(df[df.rm>15].index); perm=np.random.RandomState(0).permutation(len(NAMES)); sub=[int(w) for w in perm[500:540]]; ws=[int(w) for w in perm[500:773]][int(sys.argv[2]):int(sys.argv[3])]
    R_=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R_]); out=[]; t0=time.time()
    for wi in ws:
        d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K; r=R_[wi]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[wi]:off[wi+1]])
        z,gr,tvt0,twg,m0,y=prep(d)
        if cfg.startswith('o:'):
            _,L_,los,gam,stp=cfg.split(':'); L_=int(L_); b=np.unique(np.linspace(0,n,max(n//L_,1)+1).astype(int)).tolist() if L_>0 else tbounds(d,K)
            p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],y,R=15,gamma=float(gam),step=float(stp),loss=los)
        elif cfg=='oracle': b=tbounds(d,K); p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],y,R=15,gamma=0)
        elif cfg=='oracle_unif': b=bounds(n,'unif'); p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],y,R=15,gamma=0)
        elif cfg=='S25_170': b=bounds(n,'unif'); p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],S,R=25,gamma=20000)
        elif cfg=='S25_170g': b=bounds(n,'unif'); p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],S,R=25,gamma=110000)
        elif cfg=='S25_300abs': b=np.unique(np.linspace(0,n,max(n//300,1)+1).astype(int)).tolist(); p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],S,R=25,gamma=20000,loss='abs')
        elif cfg=='S25_300': b=np.unique(np.linspace(0,n,max(n//300,1)+1).astype(int)).tolist(); p=fmc(b,z,gr,tvt0,twg,d['tw_tvt'],S,R=25,gamma=20000)
        if p is not None: out.append(dict(w=wi,y=y,p=p,n=n))
    pickle.dump(out,open(f'vmC_{cfg}_{sys.argv[2]}.pkl','wb')); print(cfg,len(out),'%.0fs'%(time.time()-t0))
