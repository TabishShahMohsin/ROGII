# Out-of-fold learned-comparator posteriors on the 50-ft grid of every well (folds = the same GroupKFold used by the stacks).
import pickle, numpy as np, torch, time
from sklearn.model_selection import GroupKFold
import learned_match as L
from common import W,NAMES
torch.set_num_threads(2)
R=pickle.load(open('ds6.pkl','rb'))
y=np.concatenate([r['y'] for r in R]); g=np.concatenate([np.full(len(r['y']),r['w']) for r in R]); dummy=np.zeros(len(y))
folds=[np.unique(g[te]) for tr,te in GroupKFold(5).split(dummy,y,g)]
def grid_windows(wi):
    """one window centred on every 50-ft grid point (lateral rows), same construction as training windows"""
    r=R[wi]; d=W[NAMES[wi]]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md)); tv=d['TVT'][lat].astype(np.float64); gr=d['GR'][lat].astype(np.float64)
    pr=r['tvt_end']+np.interp(lat,r['idx'],r['yA']); f=L.field_fn(d['tw_tvt'],d['tw_gr'],1.0); a,b,sg,c=L.calib(d,K,f,1500)
    n=len(r['idx']); G=np.zeros((n,L.NS)); M=np.zeros((n,L.NS),bool); P=np.zeros((n,L.NS)); T=np.zeros(n)
    for k,gi in enumerate(r['idx']-K):
        s0=int(np.clip(gi-L.WIN//2,0,max(len(lat)-L.WIN-1,0))); sl=slice(s0,s0+L.WIN,L.SUB); gg=gr[sl]; m=~np.isnan(gg)
        gg=np.pad(gg,(0,L.NS-len(gg)),constant_values=np.nan) if len(gg)<L.NS else gg; m=~np.isnan(gg); pp=pr[sl]; pp=np.pad(pp,(0,L.NS-len(pp)),mode='edge') if len(pp)<L.NS else pp
        G[k]=np.nan_to_num(gg); M[k]=m; P[k]=pp; T[k]=np.mean(tv[sl]-pr[sl]) if len(tv[sl]) else 0
    if abs(a)<1e-6: return None
    Fc=f(P[:,None,:]+L.DG[None,:,None]); return dict(G=G,M=M,Fc=Fc,T=T,a=a,b=b,sg=sg,wi=wi)
if __name__=="__main__":
    t0=time.time(); post={}
    for fi,te_w in enumerate(folds):
        te_set=set(te_w.tolist()); tr=[];
        for wi in range(len(NAMES)):
            if wi in te_set: continue
            w=L.windows(wi,60)
            if w is not None: tr.append(w)
        Xt=torch.tensor(np.concatenate([L.feats(w) for w in tr])); Ttr=np.concatenate([w['T'] for w in tr])
        soft=np.exp(-0.5*((L.DG[None,:]-Ttr[:,None])/2.0)**2); soft/=soft.sum(1,keepdims=True); Yt=torch.tensor(soft,dtype=torch.float32)
        torch.manual_seed(fi); m=L.Comparator(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2); N=len(Xt); steps=2500; sch=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=steps)
        for it in range(steps):
            i=torch.randint(0,N,(128,)); loss=-(Yt[i]*torch.log_softmax(m(Xt[i]),1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        for wi in te_w:
            w=grid_windows(int(wi))
            if w is None: post[int(wi)]=None; continue
            X=torch.tensor(L.feats(w))
            with torch.no_grad(): post[int(wi)]=torch.softmax(m(X),1).numpy()
        print(f"fold {fi} done, train windows {N}, [{time.time()-t0:.0f}s]",flush=True)
    pickle.dump(post,open('comp_post.pkl','wb')); print("saved")
