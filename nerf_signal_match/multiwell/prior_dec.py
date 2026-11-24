import numpy as np,common
from common import W,NAMES,pooled
perm=np.random.RandomState(0).permutation(len(NAMES)); TE=[int(w) for w in perm[500:773]]
res={k:[] for k in ['hold(TVT const)','ANCC const','nb2 h=1000','nb2 h=1000 excl<=300ft','0.5hold+0.5nb2','prefix-slope(ANCC) 1000','prefix-slope TVT 1000']}
for wi in TE:
    d=W[NAMES[wi]]; K=d['K']; n=len(d['MD']); lat=np.arange(K,n,50); y=d['TVT'][lat]; t0=d['TVT_input'][K-1]; dz=d['Z'][lat]-d['Z'][K-1]; md=d['MD'][lat]-d['MD'][K-1]
    P=lambda inc: t0+inc-dz
    res['hold(TVT const)'].append(P(dz)-y); res['ANCC const'].append(P(0*dz)-y)
    xq=np.r_[d['X'][K-1],d['X'][lat]]; yq=np.r_[d['Y'][K-1],d['Y'][lat]]
    sf=common.nb2(xq,yq,1000,{wi}); inc=sf[1:]-sf[0]; res['nb2 h=1000'].append(P(inc)-y)
    ex=common.excluded_wells(wi,300); sf=common.nb2(xq,yq,1000,ex); res['nb2 h=1000 excl<=300ft'].append(P(sf[1:]-sf[0])-y)
    res['0.5hold+0.5nb2'].append(P(0.5*dz+0.5*inc)-y)
    s=slice(max(0,K-1000),K); a=d['TVT_input'][s]+d['Z'][s]; sl=np.polyfit(d['MD'][s],a,1)[0]; res['prefix-slope(ANCC) 1000'].append(P(sl*md)-y)
    sl2=np.polyfit(d['MD'][s],d['TVT_input'][s],1)[0]; res['prefix-slope TVT 1000'].append(t0+sl2*md-y)
for k,v in res.items(): print('%-28s pooled RMSE %.2f'%(k,pooled(v)))
