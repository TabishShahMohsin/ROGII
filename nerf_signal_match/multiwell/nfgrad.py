import pickle, numpy as np, torch, time, sys
from scipy.ndimage import gaussian_filter1d
from common import W,NAMES
from hmm import calib, field_fn
torch.set_num_threads(2)
R=pickle.load(open('ds.pkl','rb'))
SIGS=[24.,12.,6.,3.,1.5,0.75]
PN=dict(ST=3,Kd=200.,s0=15.,s1=5.,wd=1.0,neff_len=20.,steps=(80,80,100,100,120,120),lr=0.5,P=1500,sigf=0.0,nu=3.0,corr_min=0.0,huber=2.0)
class TWField:
    """differentiable multi-scale lookup of typewell GR (stand-in for neural field; see ablation)"""
    def __init__(s,tvt,gr,sigs):
        dw=float(np.median(np.diff(tvt))); s.w0=float(tvt[0]); s.dw=dw; s.n=len(tvt)
        s.G=torch.tensor(np.stack([gaussian_filter1d(gr.astype(np.float64),max(sg,0.0)/dw,mode='nearest') if sg>0 else gr.astype(np.float64) for sg in sigs]),dtype=torch.float32)
    def __call__(s,w,j):
        x=torch.clamp((w-s.w0)/s.dw,0,s.n-1.001); i0=x.floor().long(); fr=x-i0; g=s.G[j]; return g[i0]*(1-fr)+g[i0+1]*fr
def refine(r,p=PN,prior_key='yA',verbose=False):
    w=NAMES[r['w']]; d=W[w]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md)); out={}
    pr_all=r['tvt_end']+np.interp(lat,r['idx'],r[prior_key]); out['prior']=pr_all
    f0=field_fn(d['tw_tvt'],d['tw_gr'],p['sigf']); a,b,sg,c=calib(d,K,f0,p['P']); out['corr']=c
    ls=np.arange(0,len(lat),p['ST']); gr=d['GR'][lat][ls].astype(np.float64); ok=~np.isnan(gr)
    if c<p['corr_min'] or ok.sum()<50 or abs(a)<1e-6: out['pred']=pr_all; return out
    fld=TWField(d['tw_tvt'],d['tw_gr'],SIGS); mdl=md[lat][ls]; dist=mdl-md[K-1]
    Kd=p['Kd']; kn=np.arange(0,dist.max()+Kd,Kd); nk=len(kn)
    B=np.zeros((len(ls),nk)); 
    for i,x in enumerate(dist): j=min(int(x//Kd),nk-2); t=(x-kn[j])/Kd; B[i,j]=1-t; B[i,j+1]=t
    B=torch.tensor(B,dtype=torch.float32); pr=torch.tensor(pr_all[ls],dtype=torch.float32); y=torch.tensor(np.nan_to_num(gr),dtype=torch.float32); m=torch.tensor(ok,dtype=torch.float32)
    delta=torch.zeros(nk,requires_grad=True); opt=torch.optim.Adam([delta],p['lr'])
    nef=max(m.sum().item()/(p['neff_len']/p['ST']),1.0)    # effective # independent samples
    for st,(ns) in zip(range(len(SIGS)),p['steps']):
        for _ in range(ns):
            dl=B@delta; mu=a*fld(pr+dl,st)+b; res=(y-mu)/sg
            hub=torch.nn.functional.huber_loss(res,torch.zeros_like(res),reduction='none',delta=p['huber']); data=nef*(hub*m).sum()/m.sum()*p['wd']
            dd=delta[1:]-delta[:-1]; prior=0.5*((delta[1:]/p['s0'])**2).sum()+0.5*((dd/p['s1'])**2).sum()
            loss=data+prior; opt.zero_grad(); loss.backward(); opt.step()
    dl=(B@delta).detach().numpy(); out['pred']=pr_all+np.interp(np.arange(len(lat)),ls,dl); out['delta']=dl; return out
def evaluate(rs,p=PN,**kw):
    ep=[];eb=[]
    for r in rs:
        d=W[NAMES[r['w']]]; K=d['K']; tv=d['TVT'][K:].astype(float); o=refine(r,p,**kw); ep.append(o['pred']-tv); eb.append(o['prior']-tv)
    f=lambda e: float(np.sqrt(np.mean(np.concatenate(e)**2))); return f(eb),f(ep)
if __name__=="__main__":
    rng=np.random.RandomState(1); idx=rng.choice(len(R),40,replace=False); rs=[R[i] for i in idx]
    t0=time.time(); print("NF-grad on stage-A prior, 40 dev wells: prior -> refined",evaluate(rs),f"[{time.time()-t0:.0f}s]")
