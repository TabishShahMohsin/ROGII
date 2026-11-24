# Port of the user's anchor-Viterbi (Viterbi Physics Backbone.ipynb) evaluated on train wells with the K-prefix split.
import pickle, sys, time, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
W=pickle.load(open('wells.pkl','rb')); NAMES=sorted(W)
def lp(g,cutoff=0.4):
    g=pd.Series(g).interpolate(method='linear').bfill().ffill().values; b,a=butter(1,cutoff/0.5); return filtfilt(b,a,g)
def find_ms(bnd,z,gr,tvt0,tw_gr,tw_tvt,anchor_density=3,max_m_range=0.05,max_m=0.05,gamma=110000,pool=1000,nsel=5,minsep=7.0,c=2.0):
    nch=len(bnd)-1; a0=-z[0]; anchors=[np.array([a0])]
    for i in range(1,len(bnd)):
        lo=a0-max_m_range*bnd[i]; hi=a0+max_m_range*bnd[i]; anchors.append(np.linspace(lo,hi,max(int((hi-lo)*anchor_density),1)))
    costs=[np.full(len(a),np.inf) for a in anchors]; bp=[np.zeros(len(a),dtype=int) for a in anchors]; costs[0][0]=0
    for i in range(nch):
        s,e=bnd[i],bnd[i+1]; L=e-s; st=np.arange(L); cz=z[s:e]; cg=gr[s:e]; ca=anchors[i]; na=anchors[i+1]
        Mm=(na[None,:]-ca[:,None])/L; ok=(costs[i][:,None]!=np.inf)&(np.abs(Mm)<=max_m); ci,ni=np.where(ok)
        if len(ci)==0: return None
        m=Mm[ci,ni][:,None]; hyp=ca[ci][:,None]+m*st[None,:]+cz[None,:]+tvt0
        pg=np.interp(hyp.ravel(),tw_tvt,tw_gr).reshape(len(ci),L); cost=((cg[None,:]-pg)**2).sum(1)+gamma*L*m.ravel()**2+costs[i][ci]
        T=np.full((len(ca),len(na)),np.inf); T[ci,ni]=cost; mi=T.argmin(0); mc=T.min(0); u=mc<costs[i+1]
        costs[i+1][u]=mc[u]; bp[i+1][u]=mi[u]
    fin=np.where(costs[-1]!=np.inf)[0]; fin=fin[np.argsort(costs[-1][fin])][:pool]; sel=[];sa=[]
    for node in fin:
        t=np.zeros(nch+1); k=node; t[-1]=anchors[-1][k]
        for i in range(nch-1,-1,-1): k=bp[i+1][k]; t[i]=anchors[i][k]
        if all(np.mean(np.abs(t-q))>=minsep for q in sa): sel.append(node); sa.append(t)
        if len(sel)==nsel: break
    paths=[]
    for t in sa:
        p=np.zeros(len(z))
        for i in range(nch):
            s,e=bnd[i],bnd[i+1]; mm=(t[i+1]-t[i])/(e-s); p[s:e]=t[i]+mm*np.arange(e-s)+z[s:e]+tvt0
        paths.append(p)
    tc=costs[-1][sel]; w=np.exp(-c*(tc-tc.min())/(tc.max()-tc.min()+1e-8)); w/=w.sum()
    return np.array(paths),w,tc
def bounds(n,mode,gr=None):
    if mode=='unif': return np.unique(np.linspace(0,n,max(n//(70 if n<1000 else 170),1)+1).astype(int)).tolist()
    pk,_=find_peaks(gr,prominence=5.0,distance=50); mid=((pk[:-1]+pk[1:])//2).tolist(); return sorted([0]+mid+[n])
def run(wi,mode='unif',**kw):
    d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K
    if n<50: return None
    md=d['MD'][K:]; Z=d['Z'][K:]; mz,cz=np.polyfit(md,Z,1); z=mz*md+cz-Z; gr=lp(d['GR'][K:]); tvt0=d['TVT_input'][K-1]; y=d['TVT'][K:]
    bnd=bounds(n,mode,gr)
    if len(bnd)<2: return None
    r=find_ms(bnd,z,gr,tvt0,d['tw_gr'],d['tw_tvt'],**kw)
    if r is None: return None
    P,w,tc=r; se=lambda p:np.mean((p-y)**2)
    return dict(P=P,w_=w,tc=tc,y=y,tvt0=tvt0,n=n,hold=np.mean((tvt0-y)**2),top1=se(P[0]),wavg=se((w[:,None]*P).sum(0)),oracle=min(se(p) for p in P),mean5=se(P.mean(0)))
if __name__=="__main__":
    mode=sys.argv[1]; a,b=int(sys.argv[2]),int(sys.argv[3]); perm=np.random.RandomState(0).permutation(len(NAMES)); out=[]; t0=time.time()
    for wi in perm[a:b]:
        try: r=run(int(wi),mode)
        except Exception as e: print('err',NAMES[wi],e); r=None
        if r: r['w']=int(wi); out.append(r)
    pickle.dump(out,open(f'vitP_{mode}_{a}_{b}.pkl','wb'))
    tot=lambda k:np.sqrt(sum(r[k]*r['n'] for r in out)/sum(r['n'] for r in out))
    print(mode,len(out),'wells %.0fs'%(time.time()-t0),{k:round(tot(k),2) for k in ['hold','top1','wavg','mean5','oracle']})
