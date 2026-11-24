# Is lateral GR spatially coherent across wells? (pairs of lateral samples at nearby XY AND nearby true TVT, from different wells)
import numpy as np
from scipy.spatial import cKDTree
from common import W,NAMES
from vit import lp
P=[];T=[];G=[];I=[];F=[]
for i,n in enumerate(NAMES):
    d=W[n]; K=d['K']; idx=np.arange(K,len(d["MD"]),30); g=lp(d['GR'])[idx]; ok=~np.isnan(g)&~np.isnan(d['TVT'][idx])
    P.append(np.c_[d['X'][idx],d['Y'][idx]][ok]); T.append(d['TVT'][idx][ok]); G.append(g[ok]); I.append(np.full(ok.sum(),i)); F.append(np.interp(d['TVT'][idx][ok],d['tw_tvt'],d['tw_gr']))
P=np.concatenate(P);T=np.concatenate(T);G=np.concatenate(G);I=np.concatenate(I);F=np.concatenate(F); print(len(P),'lateral samples')
kd=cKDTree(P); pr=kd.query_pairs(600,output_type='ndarray'); pr=pr[I[pr[:,0]]!=I[pr[:,1]]]
dist=np.linalg.norm(P[pr[:,0]]-P[pr[:,1]],axis=1); dT=np.abs(T[pr[:,0]]-T[pr[:,1]])   # NOTE TVT is in each well's own type-log frame; ANCC-based stratigraphic frame is comparable only if tw tops align
print('pairs',len(pr))
ga,gb=G[pr[:,0]],G[pr[:,1]]
# use ANCC-based stratigraphic depth instead (frame-free): u = TVT - tw_top.  need tops: recompute via ANCC+Z?  approximate using TVT directly first
for dmax in [2,5]:
  print('--- |dTVT|<%d ft (TVT in each well own frame)'%dmax)
  for lo,hi in [(0,50),(50,150),(150,400),(400,600)]:
    m=(dist>=lo)&(dist<hi)&(dT<dmax)
    if m.sum()>200: print('  sep %4d-%4d ft  n=%8d  corr(GR_a,GR_b)=%.2f  RMS(GR_a-GR_b)=%.1f API'%(lo,hi,m.sum(),np.corrcoef(ga[m],gb[m])[0,1],np.sqrt(np.mean((ga[m]-gb[m])**2))))
print('reference: RMS(GR - typewell(TVT)) along own truth: %.1f API, corr %.2f'%(np.sqrt(np.mean((G-F)**2)),np.corrcoef(G,F)[0,1]))
