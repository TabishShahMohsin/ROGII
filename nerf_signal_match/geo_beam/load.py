import numpy as np, pickle
z=np.load('pack.npz'); W={}; T={}
for k in z.files:
    sp,w,kind,c=k.split('/')
    D=(W if sp=='train' else T).setdefault(w,{})
    D[('tw_' if kind=='tw' else '')+c]=z[k]
for D in list(W.values())+list(T.values()):
    ti=D['TVT_input']; kn=np.isfinite(ti); D['K']=int(np.flatnonzero(kn)[-1])+1 if kn.any() else 0
    D['contig']=bool(kn[:D['K']].all())
pickle.dump(W,open('wells.pkl','wb')); pickle.dump(T,open('test.pkl','wb'))
Ks=np.array([D['K'] for D in W.values()]); N=np.array([len(D['MD']) for D in W.values()])
print(len(W),len(T),'contig',all(D['contig'] for D in W.values()), 'eval rows',(N-Ks).sum(), 'mean K frac',(Ks/N).mean())
md=W[next(iter(W))]['MD']; print('md step',np.unique(np.round(np.diff(md),3))[:5])
print('nan GR frac', np.mean([np.isnan(D['GR'][D['K']:]).mean() for D in W.values()]))
