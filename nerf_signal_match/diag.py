from scipy.ndimage import gaussian_filter1d
import infer as I, numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from infer import *
s0=I.slope0*100
# truth slope (%) per knot: fit by regressing g_true on knots
gt=(tvt+z)[lat]-g_end                       # true g - g_end
def truth_slopes():
    # dg/dmd averaged inside each knot interval
    out=[]
    for k in range(Kn-1):
        m=(md[lat]>=kn[k])&(md[lat]<kn[k+1])
        if m.sum()>5: x=md[lat][m]; out.append(np.polyfit(x,((tvt+z)[lat])[m],1)[0]*100)
        else: out.append(np.nan)
    return np.array(out)
ts=truth_slopes(); print("truth slope pct per 150ft knot: min %.2f max %.2f mean %.2f std %.2f ; prior slope0 pct = %.2f"%(np.nanmin(ts),np.nanmax(ts),np.nanmean(ts),np.nanstd(ts),s0))
# 1) lambda sweep (all use same schedule; true RMSE printed only for reporting)
print("\n== sweep (lam_s, lam_0) -> final RMSE vs truth")
res={}
for ls,l0 in [(0.1,0.01),(1,0.1),(1,1),(10,1),(10,10),(100,10)]:
    t,sp=run(s0,ls,l0,"sw",verbose=False); res[(ls,l0)]=t; print(f"  lam_s={ls:<5} lam_0={l0:<5} RMSE={rmse(t):.2f}",flush=True)
# 2) placebo: same optimization with GR scrambled (reversed in MD) -> does the GR signal matter?
print("\n== placebo controls (B settings): is the improvement from real GR or just the optimizer/prior?")
y_save=I.yl.clone()
for name,perm in [("GR reversed in MD",torch.arange(n-1,-1,-1)),("GR circularly shifted +1500",(torch.arange(n)+1500)%n)]:
    I.yl=y_save[perm]; I.mk=torch.tensor(okl,dtype=torch.float32)[perm]
    t,_=run(s0,1.0,0.1,"pl",verbose=False); print(f"  {name}: RMSE={rmse(t):.2f}",flush=True)
I.yl=y_save; I.mk=torch.tensor(okl,dtype=torch.float32)
# 3) plot best-config
tB,spB=run(s0,1.0,0.1,"B",verbose=False)
lin=tvt_of(torch.full((Kn,),s0))
fig,ax=plt.subplots(3,1,figsize=(14,11))
ax[0].plot(md[lat],tvt[lat],'k',label='truth');ax[0].plot(md[lat],lin.numpy(),'b--',label='linear-g baseline');ax[0].plot(md[lat],tB.detach().numpy(),'r',label='neural-field fit');ax[0].legend();ax[0].set_title('Lateral TVT')
ax[1].plot(md[lat],(lin.numpy()-tvt[lat]),'b--',label='linear err');ax[1].plot(md[lat],(tB.detach().numpy()-tvt[lat]),'r',label='NF err');ax[1].axhline(0,c='k',lw=.5);ax[1].legend();ax[1].set_title('Error (pred - truth), ft')
ax[2].plot(md[lat],gr[lat],lw=.4,c='gray',label='hw GR');
with torch.no_grad(): pr=(a*F(tB.detach(),0)+b).numpy()*GS+GM; pt=(a*F(torch.tensor(tvt[lat],dtype=torch.float32),0)+b).numpy()*GS+GM
ax[2].plot(md[lat],gaussian_filter1d(np.nan_to_num(gr[lat],nan=np.nanmean(gr)),15),'k',label='hw GR smoothed 15ft');ax[2].plot(md[lat],gaussian_filter1d(pr,15),'r',label='field @ NF TVT');ax[2].plot(md[lat],gaussian_filter1d(pt,15),'g',label='field @ true TVT');ax[2].legend();ax[2].set_title('GR match')
plt.tight_layout();plt.savefig('result.png',dpi=65)
