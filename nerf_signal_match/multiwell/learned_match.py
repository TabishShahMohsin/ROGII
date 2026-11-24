# Learned pairwise sequence comparator ("neural cost function") for GR-vs-type-log matching.
# For every lateral window and every candidate offset delta (relative to the structural prior path) a CNN sees the aligned pair
# (hw GR window, type-log values along prior+delta) and scores it; trained with InfoNCE/softmax over candidates against the TRUE offset.
# Compared by "hit-rate" (argmax within 2 ft of the true offset; chance ~ 5/33) against handcrafted costs: windowed correlation, SSE, 1-D Wasserstein.
import pickle, numpy as np, torch, torch.nn as nn, time
from scipy.ndimage import gaussian_filter1d
from common import W,NAMES
from hmm import field_fn, calib
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
R=pickle.load(open('ds6.pkl','rb'))
rs=np.random.RandomState(0); perm=rs.permutation(len(NAMES)); TRS=set(perm[:500].tolist())     # same split as the NeRF experiments (well index in NAMES == R index)
DG=np.arange(-32,32.01,2.0); NC=len(DG); WIN=240; SUB=4; NS=WIN//SUB
def windows(wi,stride):
    r=R[wi]; d=W[NAMES[wi]]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md)); tv=d['TVT'][lat].astype(np.float64); gr=d['GR'][lat].astype(np.float64)
    pr=r['tvt_end']+np.interp(lat,r['idx'],r['yA']); f=field_fn(d['tw_tvt'],d['tw_gr'],1.0); a,b,sg,c=calib(d,K,f,1500)
    if abs(a)<1e-6: return None
    out=[]
    for s0 in range(0,len(lat)-WIN,stride):
        sl=slice(s0,s0+WIN,SUB); g=gr[sl]; ok=~np.isnan(g)
        if ok.mean()<0.5: continue
        td=np.mean(tv[sl]-pr[sl])
        if abs(td)>31: continue
        out.append((g,ok,pr[sl],td,s0))
    if not out: return None
    G=np.array([o[0] for o in out]); M=np.array([o[1] for o in out]); P=np.array([o[2] for o in out]); T=np.array([o[3] for o in out])
    Fc=f(P[:,None,:]+DG[None,:,None])                                  # [n,NC,NS]
    return dict(G=G,M=M,Fc=Fc,T=T,a=a,b=b,sg=sg,wi=wi)
def feats(win):
    G,M,Fc,a,b,sg=win['G'],win['M'].astype(np.float32),win['Fc'],win['a'],win['b'],win['sg']; Gz=np.where(M>0,G,0)
    n=M.sum(1,keepdims=True)+1e-6; mg=(Gz*M).sum(1,keepdims=True)/n; sgz=np.sqrt((((Gz-mg)*M)**2).sum(1,keepdims=True)/n)+1e-3
    c0=((Gz-mg)/sgz*M)[:,None,:].repeat(NC,1)
    mf=(Fc*M[:,None,:]).sum(2,keepdims=True)/n[:,:,None]; sf=np.sqrt((((Fc-mf)*M[:,None,:])**2).sum(2,keepdims=True)/n[:,:,None])+1e-3
    c1=(Fc-mf)/sf*M[:,None,:]
    c2=np.clip((Gz[:,None,:]-(a*Fc+b))/sg,-5,5)*M[:,None,:]
    c3=((Fc-90)/30)*M[:,None,:]
    return np.stack([c0,c1,c2,c3],2).astype(np.float32)                 # [n,NC,4,NS]
def handcrafted(win):
    G,M,Fc,a,b,sg=win['G'],win['M'],win['Fc'],win['a'],win['b'],win['sg']; out={}
    corr=np.zeros((len(G),NC)); sse=np.zeros((len(G),NC)); w1=np.zeros((len(G),NC))
    for i in range(len(G)):
        m=M[i]; g=G[i][m]
        for j in range(NC):
            f=Fc[i,j][m]; corr[i,j]=np.corrcoef(g,f)[0,1] if f.std()>1e-6 and g.std()>1e-6 else 0
            sse[i,j]=-np.mean(np.minimum(((g-(a*f+b))/sg)**2,25))
        # band-distribution cost (permutation invariant): compare quantiles of calibrated hw GR with quantiles of type-log values over the path's candidate band
        qg=np.quantile(g,np.linspace(0.05,0.95,19))
        for j in range(NC):
            f=Fc[i,j][m]; w1[i,j]=-np.mean(np.abs(qg-(a*np.quantile(f,np.linspace(0.05,0.95,19))+b)))
    return dict(corr=corr,sse=sse,w1=w1)
class Comparator(nn.Module):
    def __init__(s,h=32):
        super().__init__()
        s.cnn=nn.Sequential(nn.Conv1d(4,h,5,padding=2),nn.GELU(),nn.Conv1d(h,h,5,stride=2,padding=2),nn.GELU(),nn.Conv1d(h,2*h,5,stride=2,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        s.head=nn.Sequential(nn.Linear(2*h+1,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x):                                                    # x [B,NC,4,NS]
        B=x.shape[0]; z=s.cnn(x.reshape(B*NC,4,NS)).squeeze(-1).reshape(B,NC,-1); dl=torch.tensor(DG/32,dtype=torch.float32)[None,:,None].expand(B,-1,-1)
        return s.head(torch.cat([z,dl],-1)).squeeze(-1)                  # logits [B,NC]
def hit(logits,T,tol):
    am=DG[np.argmax(logits,1)]; return np.mean(np.abs(am-T)<=tol)
if __name__=="__main__":
    t0=time.time(); tr=[];te=[]
    for wi in range(len(NAMES)):
        w=windows(wi,60 if wi in TRS else 120)
        if w is not None: (tr if wi in TRS else te).append(w)
    print(f"windows: train wells {len(tr)} test wells {len(te)}  [{time.time()-t0:.0f}s]",flush=True)
    cat=lambda L,k: np.concatenate([w[k] for w in L]); Ttr,Tte=cat(tr,'T'),cat(te,'T'); print("train windows",len(Ttr),"test windows",len(Tte),"| true-offset std %.1f ft"%Ttr.std())
    Xtr=np.concatenate([feats(w) for w in tr]); Xte=np.concatenate([feats(w) for w in te]); print("features",Xtr.shape,flush=True)
    # handcrafted baselines on the test windows (subsample for speed)
    sub=[w for w in te][:60]; H={k:np.concatenate([handcrafted(w)[k] for w in sub]) for k in ['corr','sse','w1']}; Ts=np.concatenate([w['T'] for w in sub])
    print("chance hit within 2ft: %.3f, 4ft: %.3f"%(5/NC,9/NC))
    for k,v in H.items(): print(f"  handcrafted {k:<5}: hit<=2ft {hit(v,Ts,2):.3f}  <=4ft {hit(v,Ts,4):.3f}",flush=True)
    tgt=lambda T: torch.tensor(np.exp(-0.5*((DG[None,:]-T[:,None])/2.0)**2)/np.exp(-0.5*((DG[None,:]-T[:,None])/2.0)**2).sum(1,keepdims=True),dtype=torch.float32)
    Xt=torch.tensor(Xtr); Yt=tgt(Ttr); m=Comparator(); opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=1e-2); N=len(Xt); steps=3000; sch=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=steps)
    for it in range(steps):
        i=torch.randint(0,N,(128,)); lg=m(Xt[i]); loss=-(Yt[i]*torch.log_softmax(lg,1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it%500==0:
            m.eval()
            with torch.no_grad(): lt=m(torch.tensor(Xte[:3000])).numpy()
            m.train(); print(f"  it{it} loss {loss.item():.3f} | test hit<=2ft {hit(lt,Tte[:3000],2):.3f} <=4ft {hit(lt,Tte[:3000],4):.3f}",flush=True)
    m.eval()
    with torch.no_grad(): lt=np.concatenate([m(torch.tensor(Xte[k:k+512])).numpy() for k in range(0,len(Xte),512)])
    print(f"LEARNED comparator, held-out wells: hit<=2ft {hit(lt,Tte,2):.3f}  <=4ft {hit(lt,Tte,4):.3f}  [{time.time()-t0:.0f}s]")
    # same windows as handcrafted subset for apples-to-apples
    nsub=len(Ts); print(f"   (first {nsub} test windows: learned {hit(lt[:nsub],Ts,2):.3f}/{hit(lt[:nsub],Ts,4):.3f})")
    torch.save(m.state_dict(),'comparator.pt')
