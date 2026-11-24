import numpy as np, pandas as pd, torch, sys, time
from field import *
torch.manual_seed(0)
hw=pd.read_csv('hw.csv'); K=1545; N=len(hw)
md=hw.MD.values; z=hw.Z.values; gr=hw.GR.values; tvt=hw.TVT.values
f=Field(); f.load_state_dict(torch.load('field.pt')); f.eval()
for p in f.parameters(): p.requires_grad_(False)
sf=scale_feat(SIG)
def F(w,j):     # field in standardized GR units at scale index j, w torch [n]
    u=torch.clamp(torch.tensor(1.0)*((w-(W0+W1)/2)/((W1-W0)/2)),-1,1)
    return f(u,sf[j].expand(len(u)))
# ---- calibration on the prefix (true TVT given): GR_hw ~ a*field + b
gr_n=(gr-GM)/GS; ok_p=~np.isnan(gr_n[:K])
with torch.no_grad(): fp=F(torch.tensor(tvt[:K],dtype=torch.float32),0).numpy()
A=np.vstack([fp[ok_p],np.ones(ok_p.sum())]).T; (a,b),*_=np.linalg.lstsq(A,gr_n[:K][ok_p],rcond=None)
res=gr_n[:K][ok_p]-(a*fp[ok_p]+b); sig_res=1.4826*np.median(np.abs(res-np.median(res)))
print(f"[calib] prefix valid={ok_p.sum()}/{K}  gain a={a:.3f} offset b={b:.3f}  corr={np.corrcoef(fp[ok_p],gr_n[:K][ok_p])[0,1]:.3f}  robust resid sigma={sig_res:.3f} (std units; hw GR std {np.nanstd(gr_n[:K]):.2f})")
# ---- lateral problem
lat=np.arange(K,N); n=len(lat); okl=~np.isnan(gr_n[lat]); print(f"[lateral] n={n} valid GR={okl.sum()} MD {md[K]:.0f}-{md[-1]:.0f}")
g_end=tvt[K-1]+z[K-1]; gk=tvt+z; p=np.polyfit(md[K-300:K],gk[K-300:K],1); slope0=p[0]
print(f"[prior] g_end={g_end:.2f} slope0 from last 300 known rows = {slope0:.4f} ft/ft")
dmd=np.diff(md[K-1:]); KN=150.0; kn=np.arange(md[K],md[-1]+KN,KN); Kn=len(kn)
Bm=np.zeros((n,Kn)); 
for i,m in enumerate(md[lat]): 
    j=min(int((m-kn[0])//KN),Kn-2); t=(m-kn[j])/KN; Bm[i,j]=1-t; Bm[i,j+1]=t
Bm=torch.tensor(Bm,dtype=torch.float32); zl=torch.tensor(z[lat],dtype=torch.float32); dm=torch.tensor(np.diff(md[K-1:]),dtype=torch.float32)
yl=torch.tensor(np.nan_to_num(gr_n[lat]),dtype=torch.float32); mk=torch.tensor(okl,dtype=torch.float32)
def tvt_of(sp):         # sp = slope in % (ft/ft*100)
    g=g_end+torch.cumsum((Bm@sp)/100*dm,0); return g-zl
def rmse(t): return float(np.sqrt(np.mean((t.detach().numpy()-tvt[lat])**2)))
def run(init,lam_s,lam_0,tag,steps=(150,150,150,150,200,200,200),verbose=True,fit_gain=False):
    sp=torch.full((Kn,),float(init),requires_grad=True); opt=torch.optim.Adam([sp],0.05)
    s0=torch.tensor(slope0*100); t0=time.time(); hist=[]
    for stage,(j,ns) in enumerate(zip(range(len(SIG)-1,-1,-1),steps)):   # coarse -> fine
        for it in range(ns):
            w=tvt_of(sp); pred=a*F(w,j)+b
            r=(yl-pred)/sig_res; hub=torch.nn.functional.huber_loss(r,torch.zeros_like(r),reduction='none',delta=2.0)
            data=(hub*mk).sum()/mk.sum()
            prior=lam_s*((sp[1:]-sp[:-1])**2).mean()+lam_0*((sp-s0)**2).mean()
            loss=data+prior; opt.zero_grad(); loss.backward(); opt.step()
        hist.append(rmse(tvt_of(sp)))
        if verbose: print(f"  [{tag}] stage σ={SIG[j]:>2}ft  data={data.item():.3f} prior={prior.item():.3f}  RMSE_vs_truth={hist[-1]:.2f}  slope%[min,max]=[{sp.min().item():.2f},{sp.max().item():.2f}]  [{time.time()-t0:.0f}s]",flush=True)
    return tvt_of(sp),sp.detach()
if __name__=="__main__":
    rm0=lambda t:rmse(torch.tensor(t,dtype=torch.float32))
    print("reference: const-last RMSE %.2f | linear-g(slope0) RMSE %.2f"%(rm0(np.full(n,tvt[K-1])), rmse(tvt_of(torch.full((Kn,),slope0*100)))))
    print("\n== A) GR only: init slope=0, no prior (how much does the neural field alone recover?)")
    tA,_=run(0.0,0.0,0.0,"A")
    print("\n== B) init slope0, weak smooth prior")
    tB,_=run(slope0*100,1.0,0.1,"B")
