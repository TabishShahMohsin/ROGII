# NeRF-style SCENE across wells: F(x, y, u) -> GR, u = stratigraphic depth below the ANCC top (TVT - tw_top). Rays = well logs with known pose (true TVT).
# Variants (all evaluated on held-out wells at TRUE TVT with per-well prefix calibration):
#   tw   : typewell lookup (baseline)
#   z    : F(u)               shared 1D field + per-well gain/offset latents (generalizable reference curve)
#   xyz  : F(x,y,u)           3D neural field (hash-free Fourier encodings)
#   cond : F(x,y,u | tw(u))   image-conditioned (PixelNeRF-style): typewell value/derivative fed in, network learns the correction
import pickle, numpy as np, torch, torch.nn as nn, time
from scipy.ndimage import gaussian_filter1d
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
W=pickle.load(open('wells.pkl','rb')); NAMES=sorted(W)
rs=np.random.RandomState(0); perm=rs.permutation(len(NAMES)); TR=[NAMES[i] for i in perm[:500]]; TE=[NAMES[i] for i in perm[500:]]
def top(d):
    for k in ['ANCC','ASTNU','ASTNL','EGFDU','EGFDL','BUDA']:
        if k in d['tw_geo']: return d['tw_geo'][k]
    return None
TOP={w:top(W[w]) for w in NAMES}
allx=np.concatenate([W[w]['X'][::50] for w in NAMES]); ally=np.concatenate([W[w]['Y'][::50] for w in NAMES]); XM,XS,YM,YS=allx.mean(),allx.std(),ally.mean(),ally.std()
def tw_val(d,tvt):
    tv=d['tw_tvt']; g=d['tw_gr'].astype(np.float64); gs=gaussian_filter1d(g,4.0/np.median(np.diff(tv)),mode='nearest')
    return np.interp(tvt,tv,g),np.interp(tvt,tv,gs),np.gradient(gs,tv)[np.clip(np.searchsorted(tv,tvt),0,len(tv)-1)]
def build(names,stride=5,ds=1):
    U=[];X=[];Y=[];G=[];Wd=[];T0=[];T1=[];T2=[]
    for w in names:
        d=W[w]; t0=TOP[w]
        if t0 is None: continue
        gr=d['GR'].astype(np.float64); idx=np.arange(0,len(gr),stride); idx=idx[~np.isnan(gr[idx])]
        tvt=d['TVT'][idx].astype(np.float64); a,b,c=tw_val(d,tvt)
        U.append(tvt-t0);X.append((d['X'][idx]-XM)/XS);Y.append((d['Y'][idx]-YM)/YS);G.append(gr[idx]);Wd.append(np.full(len(idx),NAMES.index(w)));T0.append(a);T1.append(b);T2.append(c)
    f=lambda a: torch.tensor(np.concatenate(a),dtype=torch.float32)
    return f(U),f(X),f(Y),f(G),torch.tensor(np.concatenate(Wd)).long(),f(T0),f(T1),f(T2)
def pe(x,L,scale):
    out=[x[:,None]]
    for l in range(L): out+=[torch.sin((2**l)*np.pi*x[:,None]/scale),torch.cos((2**l)*np.pi*x[:,None]/scale)]
    return torch.cat(out,1)
class Scene(nn.Module):
    def __init__(s,kind,hid=128):
        super().__init__(); s.kind=kind; Lu=9; Lx=3; din=(1+2*Lu)+(2*(1+2*Lx) if kind in('xyz','cond') else 0)+(3 if kind=='cond' else 0); s.Lu=Lu; s.Lx=Lx
        s.net=nn.Sequential(nn.Linear(din,hid),nn.SiLU(),nn.Linear(hid,hid),nn.SiLU(),nn.Linear(hid,hid),nn.SiLU(),nn.Linear(hid,1))
    def forward(s,u,x,y,t0,t1,t2):
        f=[pe(u/600.0,s.Lu,1.0)]
        if s.kind in('xyz','cond'): f+=[pe(x,s.Lx,3.0),pe(y,s.Lx,3.0)]
        if s.kind=='cond': f+=[((t0-90)/30)[:,None],((t1-90)/30)[:,None],(t2*10)[:,None]]
        o=s.net(torch.cat(f,1)).squeeze(-1)*30+90
        return o+(t0-90)*0 if s.kind!='cond' else o+t1*0
def evaluate(kind,m,names):
    res=[]
    with torch.no_grad():
        for w in names:
            d=W[w]; t0=TOP[w]; K=d['K']
            if t0 is None: continue
            gr=d['GR'].astype(np.float64); ok=~np.isnan(gr); ar=np.arange(len(gr)); ip=np.where(ok&(ar<K))[0][::2]; il=np.where(ok&(ar>=K))[0][::2]
            if len(ip)<50 or len(il)<100: continue
            def pr(i):
                tvt=d['TVT'][i].astype(np.float64); a,b,c=tw_val(d,tvt); T=lambda z: torch.tensor(z,dtype=torch.float32)
                if kind=='tw': return a
                return m(T(tvt-t0),T((d['X'][i]-XM)/XS),T((d['Y'][i]-YM)/YS),T(a),T(b),T(c)).numpy()
            fp=pr(ip); (a_,b_),*_=np.linalg.lstsq(np.c_[fp,np.ones(len(fp))],gr[ip],rcond=None); fl=a_*pr(il)+b_
            res.append((np.sqrt(np.mean((fl-gr[il])**2)),np.corrcoef(fl,gr[il])[0,1]))
    r=np.array(res); return np.sqrt(np.mean(r[:,0]**2)),np.nanmedian(r[:,1]),np.nanmean(r[:,1]),len(r)
if __name__=="__main__":
    U,X,Y,G,Wd,T0,T1,T2=build(TR); N=len(G); print("train samples",N,flush=True); nw=len(NAMES)
    r=evaluate('tw',None,TE); print(f"[tw lookup baseline] held-out {r[3]} wells: lateral GR RMSE {r[0]:.2f} corr median {r[1]:.3f} mean {r[2]:.3f}",flush=True)
    for kind in ['z','xyz','cond']:
        torch.manual_seed(0); m=Scene(kind); ga=nn.Parameter(torch.ones(nw)); gb=nn.Parameter(torch.zeros(nw))
        opt=torch.optim.Adam(list(m.parameters())+[ga,gb],1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,1e-5); t0_=time.time()
        for it in range(4000):
            i=torch.randint(0,N,(4096,)); pred=ga[Wd[i]]*m(U[i],X[i],Y[i],T0[i],T1[i],T2[i])+gb[Wd[i]]
            loss=nn.functional.huber_loss(pred,G[i],delta=10.0); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if it%1000==0: print(f"  {kind} it{it} loss {loss.item():.2f}",flush=True)
        r=evaluate(kind,m,TE); print(f"[{kind}] held-out {r[3]} wells: lateral GR RMSE at TRUE TVT {r[0]:.2f}  corr median {r[1]:.3f} mean {r[2]:.3f}  [{time.time()-t0_:.0f}s]",flush=True)
        torch.save(m.state_dict(),f'scene_{kind}.pt')
