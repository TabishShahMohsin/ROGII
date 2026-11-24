# Test-time training of the comparator on each well's OWN prefix (true TVT known): build windows from the prefix tail,
# draw synthetic prior offsets (known by construction), fine-tune a copy of the pre-trained comparator for a few steps, then score the lateral.
import pickle, copy, numpy as np, torch, time
import learned_match as L
from common import W,NAMES
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
R=L.R
def prefix_windows(wi,span=2500,stride=40):
    d=W[NAMES[wi]]; K=d['K']; md=d['MD']; gr=d['GR'].astype(np.float64); tv=d['TVT'].astype(np.float64)
    f=L.field_fn(d['tw_tvt'],d['tw_gr'],1.0); a,b,sg,c=L.calib(d,K,f,1500)
    i0=max(0,K-span); out=[]
    for s0 in range(i0,K-L.WIN,stride):
        sl=slice(s0,s0+L.WIN,L.SUB); g=gr[sl]; ok=~np.isnan(g)
        if ok.mean()<0.5: continue
        out.append((g,ok,tv[sl]))
    if len(out)<8: return None
    return dict(G=np.array([o[0] for o in out]),M=np.array([o[1] for o in out]),TV=np.array([o[2] for o in out]),a=a,b=b,sg=sg,f=f)
def synth(pw,rng,n):
    idx=rng.randint(0,len(pw['G']),n); T=rng.uniform(-30,30,n)*(rng.rand(n)<0.9)+rng.randn(n)*2    # synthetic offsets (some near 0 like the real prior)
    P=pw['TV'][idx]-T[:,None]; Fc=pw['f'](P[:,None,:]+L.DG[None,:,None])
    win=dict(G=pw['G'][idx],M=pw['M'][idx],Fc=Fc,a=pw['a'],b=pw['b'],sg=pw['sg']); return torch.tensor(L.feats(win)),T
def score(m,win):
    X=torch.tensor(L.feats(win));
    with torch.no_grad(): return torch.cat([m(X[k:k+256]) for k in range(0,len(X),256)]).numpy()
if __name__=="__main__":
    base=L.Comparator(); base.load_state_dict(torch.load('comparator.pt')); base.eval()
    wells=[wi for wi in range(len(NAMES)) if wi not in L.TRS][:80]; rng=np.random.RandomState(0); t0=time.time()
    H0=[];H1=[];T_all=[];LG0=[];LG1=[]; CE=[[],[]]
    for wi in wells:
        lw=L.windows(wi,120); pw=prefix_windows(wi)
        if lw is None or pw is None: continue
        m=copy.deepcopy(base); m.train(); opt=torch.optim.AdamW(m.parameters(),5e-4,weight_decay=1e-2)
        for it in range(60):
            X,T=synth(pw,rng,48); soft=np.exp(-0.5*((L.DG[None,:]-T[:,None])/2.0)**2); soft/=soft.sum(1,keepdims=True)
            loss=-(torch.tensor(soft,dtype=torch.float32)*torch.log_softmax(m(X),1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); l0=score(base,lw); l1=score(m,lw); T_all.append(lw['T']); LG0.append(l0); LG1.append(l1)
    T=np.concatenate(T_all); l0=np.concatenate(LG0); l1=np.concatenate(LG1)
    soft=np.exp(-0.5*((L.DG[None,:]-T[:,None])/2.0)**2); soft/=soft.sum(1,keepdims=True)
    ce=lambda lg: -(soft*torch.log_softmax(torch.tensor(lg),1).numpy()).sum(1).mean()
    print(f"{len(wells)} held-out wells, {len(T)} windows [{time.time()-t0:.0f}s]")
    print(f"pre-trained comparator       : hit<=2ft {L.hit(l0,T,2):.3f} <=4ft {L.hit(l0,T,4):.3f}  CE {ce(l0):.3f}")
    print(f"+ per-well test-time training: hit<=2ft {L.hit(l1,T,2):.3f} <=4ft {L.hit(l1,T,4):.3f}  CE {ce(l1):.3f}   (prior-only CE 2.980)")
