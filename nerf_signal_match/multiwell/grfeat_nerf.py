# GR cost-profile features, but with the NeRF-rendered forward model (learned compositing kernel) instead of a point lookup of the type log
import pickle, numpy as np, torch, time
import grfeat
from grfeat import *
import nerf_fwd as NF
m=NF.Render('nerf',dip_cond=True); m.load_state_dict(torch.load('rend_nerf_1.pt')); m.eval()
with torch.no_grad(): KER=m.weights(torch.tensor([0.02])).numpy()[0]; KER=KER/KER.sum()
print("kernel (offset -12..12 ft):",np.round(KER,3))
from scipy.ndimage import convolve1d, gaussian_filter1d
def field_nerf(tw_tvt,tw_gr,sig):
    dw=np.median(np.diff(tw_tvt)); k_ft=NF.OFF; kk=np.interp(np.arange(-12,12.01,dw),k_ft,KER); kk/=kk.sum(); g=convolve1d(tw_gr.astype(np.float64),kk,mode='nearest')
    if sig>0: g=gaussian_filter1d(g,sig/dw,mode='nearest')
    return lambda w: np.interp(w,tw_tvt,g)
grfeat.field_fn=field_nerf
R=pickle.load(open('ds6.pkl','rb')); t0=time.time()
for r in R:
    o=grfeat.feats(r)
    for k,v in o.items(): r['n_'+k]=v
pickle.dump(R,open('ds7.pkl','wb')); print("done",round(time.time()-t0))
