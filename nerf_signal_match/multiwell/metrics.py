# Which matching metric best "localizes" the geologist's own pick?  Shift the TRUE TVT path by delta (ft), score GR(hw) vs type-log(path+delta) per window,
# and ask: is the best delta 0?  (hit = argmax within +-2 / +-5 ft).  Chance for +-2: 5/61.
import pickle,sys,numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d as gf
from scipy.stats import rankdata
from common import W,NAMES
from vit import lp
DEL=np.arange(-30,30.1,1.0); STEP=2
perm=np.random.RandomState(0).permutation(len(NAMES)); WS=[int(w) for w in perm[500:773]]
def zs(a): return (a-a.mean(-1,keepdims=True))/(a.std(-1,keepdims=True)+1e-6)
def corr(g,f): return (zs(g)*zs(f)).mean(-1)
def huber(r,d=10.): a=np.abs(r); return np.where(a<d,0.5*r**2,d*(a-0.5*d))
def metrics(g,f,wf,calg):
    """g [nw,L], f [D,nw,L], wf = |dF/dpath| weights [D,nw,L]; higher=better"""
    gg=g[None]; out={}
    out['sse']=-((gg-f)**2).mean(-1)
    out['sse_prefixcal']=-((calg[None]-f)**2).mean(-1)
    out['huber']=-huber(gg-f).mean(-1)
    out['corr']=corr(np.broadcast_to(gg,f.shape),f)
    gd=np.diff(g,axis=-1)[None]; fd=np.diff(f,axis=-1); out['dcorr']=corr(np.broadcast_to(gd,fd.shape),fd)
    zg=zs(np.broadcast_to(gg,f.shape)); zf=zs(f); w=wf/(wf.mean(-1,keepdims=True)+1e-9); out['wcorr']=(w*zg*zf).mean(-1)
    out['spearman']=corr(rankdata(np.broadcast_to(gg,f.shape),axis=-1),rankdata(f,axis=-1))
    return out
def run(Lw,sig,wells=WS):
    res={}; cnt=0
    for wi in wells:
        d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K
        if n<Lw*1.2: continue
        gr=lp(d['GR'])[K::STEP]; y=d['TVT'][K::STEP].astype(float); ref=d['TVT_input'][K-1]
        # prefix calibration (user's viewer): standardise hw to type-log scale using last 1000 prefix rows
        pre=slice(max(0,K-1000),K); gp=lp(d['GR'])[pre]; tp=np.interp(d['TVT_input'][pre],d['tw_tvt'],d['tw_gr'])
        a=tp.std()/(gp.std()+1e-6); b=tp.mean()-a*gp.mean()
        F=np.stack([np.interp(y+dl,d['tw_tvt'],d['tw_gr']) for dl in DEL])          # [D,n]
        g=gr.copy()
        if sig>0: g=gf(g,sig/STEP,mode='nearest'); F=gf(F,sig/STEP,axis=1,mode='nearest')
        wF=np.abs(np.gradient(F,axis=1))
        L=Lw//STEP; nw=len(g)//L
        if nw<1: continue
        gw=g[:nw*L].reshape(nw,L); Fw=F[:,:nw*L].reshape(len(DEL),nw,L); Ww=wF[:,:nw*L].reshape(len(DEL),nw,L)
        m=metrics(gw,Fw,Ww,a*gw+b)
        # drop windows with NaN/flat
        ok=(gw.std(1)>1e-3)&(Fw[len(DEL)//2].std(1)>1e-3)
        for k,v in m.items():
            v=v[:,ok]; am=DEL[np.nanargmax(v,0)]; res.setdefault(k,[]).append(am)
        cnt+=ok.sum()
    return {k:np.concatenate(v) for k,v in res.items()},cnt
if __name__=="__main__":
    rows=[]
    for Lw in [300,800,2000]:
        for sig in [0,10,40]:
            r,c=run(Lw,sig)
            for k,am in r.items(): rows.append(dict(window=Lw,smooth=sig,metric=k,n=c,hit2=np.mean(np.abs(am)<=2),hit5=np.mean(np.abs(am)<=5),mad=np.median(np.abs(am))))
            print('done',Lw,sig,c,flush=True)
    df=pd.DataFrame(rows); df.to_pickle('metrics_res.pkl')
    pd.set_option('display.width',200); print(df.pivot_table(index=['window','smooth'],columns='metric',values='hit2').round(3).to_string()); print('chance hit2 = %.3f hit5 = %.3f'%(5/61,11/61))
    print(df.pivot_table(index=['window','smooth'],columns='metric',values='hit5').round(3).to_string())
