# Cross-attention matcher: hw GR window tokens attend over the TYPE-LOG segment around the prior band (no candidate enumeration).
# Tokens carry coordinates (position along the window, prior-path TVT relative to band centre; type-log coordinate relative to band centre),
# so attention can learn "where in the type log does this stretch of the lateral live", conditioned on distinctive marker beds.
import pickle, numpy as np, torch, torch.nn as nn, time
import learned_match as L
from common import W,NAMES
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
R=L.R; SEG=np.arange(-48,48.01,1.0); NSEG=len(SEG)
def build(wi,stride):
    r=R[wi]; d=W[NAMES[wi]]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md)); tv=d['TVT'][lat].astype(np.float64); gr=d['GR'][lat].astype(np.float64)
    pr=r['tvt_end']+np.interp(lat,r['idx'],r['yA']); f=L.field_fn(d['tw_tvt'],d['tw_gr'],1.0); a,b,sg,c=L.calib(d,K,f,1500)
    if abs(a)<1e-6: return None
    out=[]
    for s0 in range(0,len(lat)-L.WIN,stride):
        sl=slice(s0,s0+L.WIN,L.SUB); g=gr[sl]; ok=~np.isnan(g)
        if ok.mean()<0.5: continue
        td=np.mean(tv[sl]-pr[sl])
        if abs(td)>31: continue
        out.append((g,ok,pr[sl],td))
    if not out: return None
    G=np.array([o[0] for o in out]); M=np.array([o[1] for o in out]); P=np.array([o[2] for o in out]); T=np.array([o[3] for o in out])
    gc=np.where(M,(G-b)/a,0.0); pm=P.mean(1,keepdims=True); seg=f(pm+SEG[None,:])
    return dict(gc=((gc-90)/30*M).astype(np.float32),M=M.astype(np.float32),prel=((P-pm)/20).astype(np.float32),seg=((seg-90)/30).astype(np.float32),T=T)
class XMatch(nn.Module):
    def __init__(s,d=48,heads=4,layers=2):
        super().__init__()
        s.hw=nn.Linear(4,d); s.tw=nn.Sequential(nn.Conv1d(2,d,5,padding=2),nn.GELU(),nn.Conv1d(d,d,5,padding=2)); s.pos=nn.Parameter(torch.zeros(L.NS,d))
        s.att=nn.ModuleList([nn.MultiheadAttention(d,heads,batch_first=True) for _ in range(layers)]); s.ffn=nn.ModuleList([nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d)) for _ in range(layers)])
        s.n1=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]); s.n2=nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]); s.out=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,L.NC))
    def forward(s,gc,M,prel,seg):
        B=gc.shape[0]; t=torch.linspace(-1,1,L.NS)[None,:].expand(B,-1); h=s.hw(torch.stack([gc,M,prel,t],-1))+s.pos[None]
        coord=torch.tensor(SEG/48,dtype=torch.float32)[None,None,:].expand(B,1,-1); kv=s.tw(torch.cat([seg[:,None,:],coord],1)).transpose(1,2)
        for a,f,n1,n2 in zip(s.att,s.ffn,s.n1,s.n2):
            h=n1(h+a(h,kv,kv)[0]); h=n2(h+f(h))
        return s.out((h*M[...,None]).sum(1)/(M.sum(1,keepdim=True)+1e-6))
def tens(L_,keys): return [torch.tensor(np.concatenate([w[k] for w in L_])) for k in keys]
if __name__=="__main__":
    t0=time.time(); tr=[];te=[]
    for wi in range(len(NAMES)):
        w=build(wi,60 if wi in L.TRS else 120)
        if w is not None: (tr if wi in L.TRS else te).append(w)
    gc,M,prel,seg,T=tens(tr,['gc','M','prel','seg','T']); tgc,tM,tprel,tseg,tT=tens(te,['gc','M','prel','seg','T']); print("windows",len(T),len(tT),flush=True)
    soft=lambda T_: (lambda e: e/e.sum(1,keepdims=True))(np.exp(-0.5*((L.DG[None,:]-T_.numpy()[:,None])/2.0)**2)); Y=torch.tensor(soft(T),dtype=torch.float32)
    m=XMatch(); opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-2); steps=3000; sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=steps); N=len(T)
    def ev():
        m.eval()
        with torch.no_grad(): lg=torch.cat([m(tgc[k:k+512],tM[k:k+512],tprel[k:k+512],tseg[k:k+512]) for k in range(0,len(tT),512)]).numpy()
        m.train(); return L.hit(lg,tT.numpy(),2),L.hit(lg,tT.numpy(),4),lg
    for it in range(steps):
        i=torch.randint(0,N,(128,)); loss=-(Y[i]*torch.log_softmax(m(gc[i],M[i],prel[i],seg[i]),1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it%500==0: h2,h4,_=ev(); print(f"  it{it} loss {loss.item():.3f} | held-out hit<=2ft {h2:.3f} <=4ft {h4:.3f}  [{time.time()-t0:.0f}s]",flush=True)
    h2,h4,lg=ev(); Yt=torch.tensor(soft(tT),dtype=torch.float32); ce=-(Yt*torch.log_softmax(torch.tensor(lg),1)).sum(1).mean().item()
    print(f"CROSS-ATTENTION matcher held-out: hit<=2ft {h2:.3f} <=4ft {h4:.3f}  CE {ce:.3f} (prior-only 2.980, CNN comparator 2.880)")
