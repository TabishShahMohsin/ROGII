# NeRF-style forward model for GR: observation = volume-rendering integral of a formation field along the tool's sensing axis.
# Ground-level NeRF changes: (1) ray samples -> offsets o_k around bit TVT; (2) sigma(o,dip) from MLP with positional encoding (view-dependent branch = apparent dip);
# (3) alpha-compositing weights w_k = T_k*(1-exp(-sigma_k*delta)); (4) field F(w)= typewell lookup -> rendered GR = sum_k w_k F(TVT+o_k).
import pickle, numpy as np, torch, torch.nn as nn, time, sys
from scipy.ndimage import gaussian_filter1d
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
W=pickle.load(open('wells.pkl','rb')); NAMES=sorted(W)
rs=np.random.RandomState(0); perm=rs.permutation(len(NAMES)); TR=[NAMES[i] for i in perm[:500]]; TE=[NAMES[i] for i in perm[500:]]
H=12.0; OFF=np.arange(-H,H+1e-9,1.0); NO=len(OFF)
tw_all=[]; base={}; w0={}; dw={}; n_={}; pos=0
for w in NAMES:
    d=W[w]; tv=d['tw_tvt']; g=d['tw_gr'].astype(np.float32); tw_all.append(g); base[w]=pos; w0[w]=tv[0]; dw[w]=float(np.median(np.diff(tv))); n_[w]=len(tv); pos+=len(tv)
TW=torch.tensor(np.concatenate(tw_all))
BASE=torch.tensor([base[w] for w in NAMES]); W0=torch.tensor([w0[w] for w in NAMES],dtype=torch.float32)
DW=torch.tensor([dw[w] for w in NAMES],dtype=torch.float32); NN=torch.tensor([n_[w] for w in NAMES],dtype=torch.float32)

def samples(names,stride=5):
    T=[];D=[];Y=[];Wid=[]
    for w in names:
        d=W[w]; md=d['MD']; tvt=d['TVT'].astype(np.float64); gr=d['GR'].astype(np.float64)
        dip=np.gradient(gaussian_filter1d(tvt,15),md)
        idx=np.arange(0,len(md),stride); idx=idx[~np.isnan(gr[idx])]
        T.append(tvt[idx]); D.append(dip[idx]); Y.append(gr[idx]); Wid.append(np.full(len(idx),NAMES.index(w)))
    f=lambda a: torch.tensor(np.concatenate(a),dtype=torch.float32)
    return f(T),f(D),f(Y),torch.tensor(np.concatenate(Wid)).long()

def lookup(wid,tvt):          # wid [N], tvt [N,K] -> typewell GR at those TVTs (linear interpolation)
    b=BASE[wid][:,None]; x=(tvt-W0[wid][:,None])/DW[wid][:,None]; n=NN[wid][:,None]
    x=torch.minimum(torch.clamp(x,min=0),n-1.001); i0=x.floor().long(); fr=x-i0
    return TW[b+i0]*(1-fr)+TW[b+i0+1]*fr

class Render(nn.Module):
    def __init__(s,mode='nerf',hid=32,L=4,dip_cond=True):
        super().__init__(); s.mode=mode; s.dip_cond=dip_cond; s.L=L
        din=1+2*L+((1+2*L) if dip_cond else 0)
        s.mlp=nn.Sequential(nn.Linear(din,hid),nn.SiLU(),nn.Linear(hid,hid),nn.SiLU(),nn.Linear(hid,1))
        s.free=nn.Parameter(torch.zeros(NO)); s.register_buffer('o',torch.tensor(OFF,dtype=torch.float32))
    def weights(s,dip):
        N=len(dip)
        if s.mode=='delta':
            w=torch.zeros(N,NO); w[:,NO//2]=1; return w
        if s.mode=='free': return torch.softmax(s.free,0)[None].expand(N,-1)
        o=(s.o/H)[None,:].expand(N,-1); fe=[o[...,None]]
        for l in range(s.L): fe+=[torch.sin((2**l)*np.pi*o)[...,None],torch.cos((2**l)*np.pi*o)[...,None]]
        if s.dip_cond:
            dp=(dip.clamp(-1,1)*5)[:,None].expand(-1,NO)[...,None]; fe.append(dp)
            for l in range(s.L): fe+=[torch.sin((2**l)*dp),torch.cos((2**l)*dp)]
        sig=nn.functional.softplus(s.mlp(torch.cat(fe,-1)).squeeze(-1)-2.0)       # density per sample along the sensing axis
        a=1-torch.exp(-sig*1.0)                                                    # alpha (delta = 1 ft)
        Tm=torch.exp(-torch.cumsum(torch.cat([torch.zeros(N,1),sig[:,:-1]],1),1))  # transmittance front-to-back
        w=Tm*a; return w/(w.sum(1,keepdim=True)+1e-6)
    def forward(s,wid,tvt,dip):
        w=s.weights(dip); f=lookup(wid,tvt[:,None]+s.o[None,:]); return (w*f).sum(1)

def calib_eval(model,names):
    """per-well prefix calibration (a,b LSQ at TRUE TVT), then predict lateral GR at TRUE TVT"""
    res=[]
    with torch.no_grad():
        for w in names:
            d=W[w]; K=d['K']; gr=d['GR'].astype(np.float64); md=d['MD']; tvt=d['TVT'].astype(np.float64); dip=np.gradient(gaussian_filter1d(tvt,15),md)
            ok=~np.isnan(gr); ar=np.arange(len(gr)); ip=np.where(ok&(ar<K))[0][::2]; il=np.where(ok&(ar>=K))[0][::2]
            if len(ip)<50 or len(il)<100: continue
            wid=torch.full((1,),NAMES.index(w),dtype=torch.long)
            def pr(i): n=len(i); return model(wid.expand(n),torch.tensor(tvt[i],dtype=torch.float32),torch.tensor(dip[i],dtype=torch.float32)).numpy()
            fp=pr(ip); (a,b),*_=np.linalg.lstsq(np.c_[fp,np.ones(len(fp))],gr[ip],rcond=None); fl=a*pr(il)+b
            res.append((np.sqrt(np.mean((fl-gr[il])**2)),np.corrcoef(fl,gr[il])[0,1]))
    r=np.array(res); return np.sqrt(np.mean(r[:,0]**2)),np.nanmedian(r[:,1]),np.nanmean(r[:,1]),len(r)

if __name__=="__main__":
    tv,dp,y,wid=samples(TR); N=len(y); print("train samples",N,"held-out wells",len(TE),flush=True)
    nw=len(NAMES)
    for mode,dc in [('delta',False),('free',False),('nerf',False),('nerf',True)]:
        torch.manual_seed(0); m=Render(mode,dip_cond=dc); ga=nn.Parameter(torch.ones(nw)); gb=nn.Parameter(torch.zeros(nw))   # per-well gain/offset latents
        opt=torch.optim.Adam(list(m.parameters())+[ga,gb],3e-3); t0=time.time(); steps=0 if mode=='delta' else 1500
        for it in range(steps):
            i=torch.randint(0,N,(4096,)); pred=ga[wid[i]]*m(wid[i],tv[i],dp[i])+gb[wid[i]]
            loss=nn.functional.huber_loss(pred,y[i],delta=10.0); opt.zero_grad(); loss.backward(); opt.step()
            if it%500==0: print(f"  {mode}/dip={dc} it{it} loss {loss.item():.2f}",flush=True)
        r=calib_eval(m,TE); print(f"[{mode}, dip-cond={dc}] held-out wells {r[3]}: lateral GR RMSE at TRUE TVT {r[0]:.2f}  corr median {r[1]:.3f} mean {r[2]:.3f}  [{time.time()-t0:.0f}s]",flush=True)
        if mode=='nerf':
            with torch.no_grad(): w=m.weights(torch.tensor([0.0,0.05,0.3])).numpy(); print("   kernel weights offsets -3..+3 ft at dip 0/0.05/0.3:",[np.round(w[k][NO//2-3:NO//2+4],3).tolist() for k in range(3)])
        torch.save(m.state_dict(),f'rend_{mode}_{int(dc)}.pt')
