import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d as gf
from common import W,NAMES
from vit import lp
from metrics import DEL,STEP,WS
def huber(r,d=10.): a=np.abs(r); return np.where(a<d,0.5*r**2,d*(a-0.5*d))
def run(Lw,wells=WS):
    res={}; cnt=0
    for wi in wells:
        d=W[NAMES[wi]]; K=d['K']; n=len(d['MD'])-K
        if n<Lw*1.2: continue
        gr=lp(d['GR'])[K::STEP]; y=d['TVT'][K::STEP].astype(float); tv=d['tw_tvt']; tg=d['tw_gr'].astype(float); dt=np.median(np.diff(tv))
        L=Lw//STEP; nw=len(gr)//L; gw=gr[:nw*L].reshape(nw,L); ok=gw.std(1)>1e-3
        def lookup(tgv,off=0.0): return np.stack([np.interp(y+dl+off,tv,tgv) for dl in DEL])[:,:nw*L].reshape(len(DEL),nw,L)
        def sc(v): return v[:,ok]
        def add(k,v): res.setdefault(k,[]).append(DEL[np.nanargmax(sc(v),0)])
        F=lookup(tg); add('sse',-((gw[None]-F)**2).mean(-1))
        for st in [1,2,4,8]:                                   # tool-response: smooth the TYPE LOG in TVT (vertical) domain
            Fs=lookup(gf(tg,st/dt,mode='nearest')); add(f'sse_TVTsmooth{st}',-((gw[None]-Fs)**2).mean(-1))
        add('sse_log',-((np.log(np.maximum(gw[None],5))-np.log(np.maximum(F,5)))**2).mean(-1))
        for c0 in [0.5,1.0]:                                   # heteroscedastic (Poisson-like) noise: var ~ c0*f+eps
            add(f'sse_hetero{c0}',-(((gw[None]-F)**2)/(c0*np.maximum(F,5)+20)).mean(-1))
        for e0 in [1,2,4]:                                     # envelope / interval loss: free to slide the type log by +-e0 ft
            E=np.stack([lookup(tg,e) for e in np.linspace(-e0,e0,5)]); lo=E.min(0); hi=E.max(0)
            r=np.maximum(0,np.maximum(lo-gw[None],gw[None]-hi)); add(f'band{e0}',-(r**2).mean(-1))
        cnt+=ok.sum()
    return {k:np.concatenate(v) for k,v in res.items()},cnt
rows=[]
for Lw in [300,800]:
    r,c=run(Lw); 
    for k,am in r.items(): rows.append(dict(window=Lw,metric=k,n=c,hit2=np.mean(np.abs(am)<=2),hit5=np.mean(np.abs(am)<=5)))
df=pd.DataFrame(rows); df.to_pickle('metrics2_res.pkl'); print(df.pivot_table(index='metric',columns='window',values=['hit2','hit5']).round(3).sort_values(('hit2',300),ascending=False).to_string())
