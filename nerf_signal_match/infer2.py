import numpy as np, torch, time
import infer as I
from infer import *
Jm=torch.tensor((md[lat][:,None]>=kn[None,:]).astype(np.float32))      # step at each knot
def tvt_j(sp,dj): return tvt_of(sp)+Jm@dj
def run2(init,lam_s,lam_0,lam_j,tag="J",steps=(150,150,150,150,200,200,200),verbose=False,yl=None,mk=None):
    yl_=I.yl if yl is None else yl; mk_=I.mk if mk is None else mk
    sp=torch.full((Kn,),float(init),requires_grad=True); dj=torch.zeros(Kn,requires_grad=True)
    opt=torch.optim.Adam([{'params':[sp],'lr':0.05},{'params':[dj],'lr':0.3}]); s0=torch.tensor(I.slope0*100)
    for j,ns in zip(range(len(SIG)-1,-1,-1),steps):
        for it in range(ns):
            w=tvt_j(sp,dj); r=(yl_-(a*F(w,j)+b))/sig_res
            hub=torch.nn.functional.huber_loss(r,torch.zeros_like(r),reduction='none',delta=2.0); data=(hub*mk_).sum()/mk_.sum()
            prior=lam_s*((sp[1:]-sp[:-1])**2).mean()+lam_0*((sp-s0)**2).mean()+lam_j*torch.sqrt(dj**2+0.01).sum()
            loss=data+prior; opt.zero_grad(); loss.backward(); opt.step()
        if verbose: print(f"  [{tag}] σ={SIG[j]:>2} data={data.item():.3f} RMSE={rmse(tvt_j(sp,dj)):.2f}")
    return tvt_j(sp,dj).detach(),sp.detach(),dj.detach()
if __name__=="__main__":
    s0=I.slope0*100; i_true=4555; print(f"true fault: between MD {md[i_true]:.0f}-{md[i_true+1]:.0f}, TVT step {tvt[i_true+1]-tvt[i_true]:.2f} ft; nearest knots at MD {kn[np.argmin(abs(kn-md[i_true+1]))]:.0f}")
    for lj in [0.0,0.003,0.01,0.03,0.1]:
        t,sp,dj=run2(s0,1.0,0.1,lj)
        big=[(int(kn[k]),round(float(dj[k]),1)) for k in range(Kn) if abs(dj[k])>1.5]
        print(f"lam_j={lj:<6} RMSE={rmse(t):.2f}  jumps(|d|>1.5ft) at (MD,ft): {big}",flush=True)
