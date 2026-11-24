import numpy as np, pandas as pd, torch, torch.nn as nn, time
from scipy.ndimage import gaussian_filter1d
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
tw=pd.read_csv('tw.csv'); ttv=tw.TVT.values; tgr=tw.GR.values
W0,W1=ttv.min(),ttv.max(); GM,GS=tgr.mean(),tgr.std()
print(f"typewell: n={len(ttv)} TVT [{W0},{W1}] step 0.5 | GR mean {GM:.2f} std {GS:.2f} NaN {np.isnan(tgr).sum()}")
SIG=[0,1,2,4,8,16,32]   # blur scales in feet (mip-style scale conditioning)
def blur(s): return tgr if s==0 else gaussian_filter1d(tgr,s/0.5,mode='nearest')
TG=np.stack([(blur(s)-GM)/GS for s in SIG])           # [S,N]
def u_of(w): return (w-(W0+W1)/2)/((W1-W0)/2)           # -> [-1,1]
L=10
class Field(nn.Module):
    def __init__(s,h=128):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(2*L+2,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
    def forward(s,u,sc):           # u [B], sc [B] = scale index normalized 0..1
        k=(2.0**torch.arange(L))*np.pi
        ang=u[:,None]*k[None]
        # mip-style: attenuate high freqs for coarse scales (freq k beyond 1/sigma contributes nothing)
        return s.net(torch.cat([u[:,None],sc[:,None],torch.sin(ang),torch.cos(ang)],1)).squeeze(-1)
def scale_feat(sig): return torch.tensor(np.log2(1+np.asarray(sig,dtype=np.float64))/np.log2(33),dtype=torch.float32)
if __name__=="__main__":
    f=Field(); opt=torch.optim.Adam(f.parameters(),1e-3)
    U=torch.tensor(u_of(ttv),dtype=torch.float32); T=torch.tensor(TG,dtype=torch.float32)
    sf=scale_feat(SIG); N=len(ttv); t0=time.time()
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,6000,1e-5)
    for it in range(6001):
        i=torch.randint(0,N,(2048,)); j=torch.randint(0,len(SIG),(2048,))
        loss=((f(U[i],sf[j])-T[j,i])**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it%1000==0:
            with torch.no_grad():
                msg=[]
                for j,s in enumerate(SIG):
                    p=f(U,sf[j].expand(N)); r2=1-((p-T[j])**2).mean()/T[j].var(); msg.append(f"σ={s}:R2={r2:.3f}")
            print(it,f"loss {loss.item():.4f}",' '.join(msg),f"[{time.time()-t0:.0f}s]",flush=True)
    torch.save(f.state_dict(),'field.pt')
    # sanity: range of outputs in physical units vs data
    with torch.no_grad(): p=f(U,sf[0].expand(N))*GS+GM
    print("field GR pred range",p.min().item(),p.max().item(),"data range",tgr.min(),tgr.max(),"RMSE(GR units) %.2f"%np.sqrt(((p.numpy()-tgr)**2).mean()))
