# What does the "geologist" (label author) do? T1: piecewise-linear structure of truth in Z+TVT space. T3: is the type log stretched/shifted per well?
import pickle,numpy as np,pandas as pd
from common import W,NAMES
from vit import lp
df=pd.read_pickle('fail2.pkl'); bad=list(df[df.rm>15].index)
# ---------- T1
def segs(x,y,tol):
    n=len(x); i=0; out=[]
    while i<n-1:
        j=min(i+20,n-1)
        while j<n-1:
            k=min(j+20,n-1); c=np.polyfit(x[i:k+1],y[i:k+1],1)
            if np.max(np.abs(np.polyval(c,x[i:k+1])-y[i:k+1]))>tol: break
            j=k
        c=np.polyfit(x[i:j+1],y[i:j+1],1); out.append((i,j,c[0])); i=j
    return out
L=[];SL=[];NS=[];rows=[]
rng=np.random.RandomState(1); sample=rng.choice(773,150,replace=False)
for wi in sample:
    d=W[NAMES[wi]]; K=d['K']; md=d['MD'][K:][::5]; s=(d['TVT'][K:]+d['Z'][K:])[::5]; sg=segs(md,s,1.5)
    L+= [md[j]-md[i] for i,j,_ in sg]; SL+=[abs(m) for *_,m in sg]; NS.append(len(sg)/((md[-1]-md[0])/1000)); 
L=np.array(L);SL=np.array(SL)
print('T1 truth Z+TVT piecewise-linear fit (tol 1.5 ft, 150 wells): segments per 1000 ft: median %.2f mean %.2f'%(np.median(NS),np.mean(NS)))
print('   segment length ft: p10 %.0f p50 %.0f p90 %.0f | |slope| p50 %.3f p90 %.3f p99 %.3f | frac segs |slope|>0.05: %.3f'%(*np.percentile(L,[10,50,90]),*np.percentile(SL,[50,90,99]),(SL>0.05).mean()))
# ---------- T3
def corr(a,b): 
    m=~np.isnan(a)&~np.isnan(b); return np.corrcoef(a[m],b[m])[0,1] if m.sum()>20 and a[m].std()>0 and b[m].std()>0 else np.nan
def warp_tw(d,tvt,s,b,ref): return np.interp(ref+s*(tvt-ref)+b,d['tw_tvt'],d['tw_gr'])
S_=np.linspace(0.8,1.2,17); B_=np.arange(-20,20.1,2.0); res=[]
for wi in list(bad)+[int(w) for w in sample[:60] if w not in bad]:
    d=W[NAMES[wi]]; K=d['K']; gr=lp(d['GR']); y=d['TVT']; ref=d['TVT_input'][K-1]
    lat=slice(K,None,4); pre=slice(max(0,K-3000),K,4)
    c0=corr(gr[lat],warp_tw(d,y[lat],1,0,ref))
    best=(-9,1,0)
    for s in S_:
        for b in B_:
            c=corr(gr[pre],warp_tw(d,d['TVT_input'][pre],s,b,ref)); 
            if c>best[0]: best=(c,s,b)
    cp=corr(gr[lat],warp_tw(d,y[lat],best[1],best[2],ref)); c0p=corr(gr[pre],warp_tw(d,d['TVT_input'][pre],1,0,ref))
    bo=(-9,1,0)
    for s in S_:
        for b in B_:
            c=corr(gr[lat],warp_tw(d,y[lat],s,b,ref)); 
            if c>bo[0]: bo=(c,s,b)
    res.append(dict(w=wi,bad=wi in bad,prefix_corr_unwarped=c0p,prefix_corr_warped=best[0],ps=best[1],pb=best[2],lat_corr_unwarped=c0,lat_corr_prefixwarp=cp,lat_corr_oracle=bo[0],os=bo[1],ob=bo[2]))
r=pd.DataFrame(res); r.to_pickle('geo1.pkl')
pd.set_option('display.width',200)
for nm,g in [('BAD',r[r.bad]),('OTHER',r[~r.bad])]:
    print(nm,len(g),g[['prefix_corr_unwarped','prefix_corr_warped','lat_corr_unwarped','lat_corr_prefixwarp','lat_corr_oracle']].mean().round(3).to_dict())
    print('   |oracle scale-1| mean %.3f, |oracle shift| mean %.1f ; prefix-fit scale-1 mean %.3f shift %.1f ; corr(prefix s, oracle s)=%.2f corr(prefix b,oracle b)=%.2f'%((g.os-1).abs().mean(),g.ob.abs().mean(),(g.ps-1).abs().mean(),g.pb.abs().mean(),g.ps.corr(g.os),g.pb.corr(g.ob)))
print(r[r.bad].round(2)[['w','prefix_corr_unwarped','prefix_corr_warped','lat_corr_unwarped','lat_corr_prefixwarp','lat_corr_oracle','ps','pb','os','ob']].to_string())
