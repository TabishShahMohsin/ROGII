# Generic version of the neural-field pipeline: python3 pipeline.py tw.csv hw.csv   (hyper-params FIXED from well 1)
import numpy as np, pandas as pd, torch, torch.nn as nn, sys, time
from scipy.ndimage import gaussian_filter1d
torch.set_num_threads(2)
SIG=[0,1,2,4,8,16,32]; L=10; LAM_S,LAM_0,KN=1.0,0.1,150.0
class Field(nn.Module):
    def __init__(s,h=128):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2*L+2,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
    def forward(s,u,sc):
        ang=u[:,None]*((2.0**torch.arange(L))*np.pi)[None]; return s.net(torch.cat([u[:,None],sc[:,None],torch.sin(ang),torch.cos(ang)],1)).squeeze(-1)
sfeat=torch.tensor(np.log2(1+np.asarray(SIG,dtype=np.float64))/np.log2(33),dtype=torch.float32)
def train_field(tw,seed=0,steps=6000):
    torch.manual_seed(seed); ttv=tw.TVT.values; tgr=tw.GR.values; dw=np.median(np.diff(ttv))
    W0,W1=ttv.min(),ttv.max(); GM,GS=tgr.mean(),tgr.std()
    TG=torch.tensor(np.stack([((tgr if s==0 else gaussian_filter1d(tgr,s/dw,mode='nearest'))-GM)/GS for s in SIG]),dtype=torch.float32)
    U=torch.tensor((ttv-(W0+W1)/2)/((W1-W0)/2),dtype=torch.float32); N=len(ttv)
    f=Field(); opt=torch.optim.Adam(f.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,1e-5)
    for it in range(steps+1):
        i=torch.randint(0,N,(2048,)); j=torch.randint(0,len(SIG),(2048,)); loss=((f(U[i],sfeat[j])-TG[j,i])**2).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad(): r2=[float(1-((f(U,sfeat[j].expand(N))-TG[j])**2).mean()/TG[j].var()) for j in range(len(SIG))]
    print(f"[field] typewell n={N} step={dw} TVT[{W0},{W1}] GR mean/std {GM:.1f}/{GS:.1f}  R2 per scale {[round(x,3) for x in r2]}",flush=True)
    f.eval(); [p.requires_grad_(False) for p in f.parameters()]
    return f,(W0,W1,GM,GS)
def main(twf,hwf):
    tw=pd.read_csv(twf); hw=pd.read_csv(hwf); f,(W0,W1,GM,GS)=train_field(tw)
    N=len(hw); K=int(hw.TVT_input.notna().sum()); md=hw.MD.values; z=hw.Z.values; tvt=hw.TVT.values; gr_n=(hw.GR.values-GM)/GS
    F=lambda w,j: f(torch.clamp((w-(W0+W1)/2)/((W1-W0)/2),-1,1),sfeat[j].expand(len(w)))
    ok_p=~np.isnan(gr_n[:K])
    with torch.no_grad(): fp=F(torch.tensor(tvt[:K],dtype=torch.float32),0).numpy()
    (a,b),*_=np.linalg.lstsq(np.vstack([fp[ok_p],np.ones(ok_p.sum())]).T,gr_n[:K][ok_p],rcond=None)
    res=gr_n[:K][ok_p]-(a*fp[ok_p]+b); sig=1.4826*np.median(np.abs(res-np.median(res)))
    print(f"[calib] prefix K={K} valid GR {ok_p.sum()}  a={a:.3f} b={b:.3f} corr={np.corrcoef(fp[ok_p],gr_n[:K][ok_p])[0,1]:.3f} resid sigma={sig:.3f}")
    lat=np.arange(K,N); n=len(lat); okl=~np.isnan(gr_n[lat]); print(f"[lateral] n={n} valid GR {okl.sum()} ({okl.mean():.0%})  lateral TVT range {tvt[lat].min():.1f}-{tvt[lat].max():.1f}")
    g_end=tvt[K-1]+z[K-1]; slope0=np.polyfit(md[K-300:K],(tvt+z)[K-300:K],1)[0]
    kn=np.arange(md[K],md[-1]+KN,KN); Kn=len(kn); Bm=np.zeros((n,Kn))
    for i,m in enumerate(md[lat]): j=min(int((m-kn[0])//KN),Kn-2); t=(m-kn[j])/KN; Bm[i,j]=1-t; Bm[i,j+1]=t
    Bm=torch.tensor(Bm,dtype=torch.float32); zl=torch.tensor(z[lat],dtype=torch.float32); dm=torch.tensor(np.diff(md[K-1:]),dtype=torch.float32)
    yl=torch.tensor(np.nan_to_num(gr_n[lat]),dtype=torch.float32); mk=torch.tensor(okl,dtype=torch.float32)
    tvt_of=lambda sp: g_end+torch.cumsum((Bm@sp)/100*dm,0)-zl
    rmse=lambda t: float(np.sqrt(np.mean((np.asarray(t)-tvt[lat])**2)))
    def run(init,ls,l0,yl_,mk_,steps=(150,150,150,150,200,200,200),verbose=False):
        sp=torch.full((Kn,),float(init),requires_grad=True); opt=torch.optim.Adam([sp],0.05); s0=torch.tensor(slope0*100)
        for j,ns in zip(range(len(SIG)-1,-1,-1),steps):
            for _ in range(ns):
                r=(yl_-(a*F(tvt_of(sp),j)+b))/sig; hub=torch.nn.functional.huber_loss(r,torch.zeros_like(r),reduction='none',delta=2.0); data=(hub*mk_).sum()/mk_.sum()
                loss=data+ls*((sp[1:]-sp[:-1])**2).mean()+l0*((sp-s0)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if verbose: print(f"  σ={SIG[j]:>2} data={data.item():.3f} RMSE={rmse(tvt_of(sp).detach()):.2f} slope%[{sp.min().item():.2f},{sp.max().item():.2f}]",flush=True)
        return tvt_of(sp).detach().numpy()
    print(f"[prior] g_end={g_end:.2f} slope0={slope0*100:.2f}%  knots={Kn}")
    out={}
    out['const last TVT']=rmse(np.full(n,tvt[K-1]))
    out['linear-g (slope0)']=rmse(tvt_of(torch.full((Kn,),slope0*100)).detach().numpy())
    print("-- NF smooth prior (fixed hyper-params from well 1: lam_s=1, lam_0=0.1)")
    tB=run(slope0*100,LAM_S,LAM_0,yl,mk,verbose=True); out['NF + smooth prior']=rmse(tB)
    print("-- NF GR-only (init 0, no prior)"); out['NF GR-only']=rmse(run(0.0,0,0,yl,mk))
    for nm,perm in [('placebo GR reversed',torch.arange(n-1,-1,-1)),('placebo GR shifted n/4',(torch.arange(n)+n//4)%n)]:
        out[nm]=rmse(run(slope0*100,LAM_S,LAM_0,yl[perm],mk[perm]))
    print("\nRESULTS (lateral TVT RMSE, ft):"); [print(f"  {k:<26}{v:7.2f}") for k,v in out.items()]
    print("MAE NF %.2f ; linear %.2f ; const %.2f"%(np.mean(abs(tB-tvt[lat])),np.mean(abs(tvt_of(torch.full((Kn,),slope0*100)).detach().numpy()-tvt[lat])),np.mean(abs(tvt[K-1]-tvt[lat]))))
    np.save(f"pred_{hwf.split('.')[0]}.npy",tB)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    lin=tvt_of(torch.full((Kn,),slope0*100)).detach().numpy(); fig,ax=plt.subplots(2,1,figsize=(13,8))
    ax[0].plot(md[lat],tvt[lat],'k',label='truth');ax[0].plot(md[lat],lin,'b--',label='linear-g');ax[0].plot(md[lat],tB,'r',label='neural field');ax[0].legend();ax[0].set_title(f'{hwf}: lateral TVT')
    ax[1].plot(md[lat],lin-tvt[lat],'b--');ax[1].plot(md[lat],tB-tvt[lat],'r');ax[1].axhline(0,c='k',lw=.5);ax[1].set_title('error (pred-truth) ft');plt.tight_layout();plt.savefig(f"result_{hwf.split('.')[0]}.png",dpi=65)
if __name__=="__main__": main(sys.argv[1],sys.argv[2])
