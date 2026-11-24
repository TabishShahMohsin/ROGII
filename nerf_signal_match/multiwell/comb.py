import pickle,numpy as np
from common import W,NAMES
V=pickle.load(open('vitP_unif_500_637.pkl','rb'))+pickle.load(open('vitP_unif_637_773.pkl','rb'))
R=pickle.load(open('ds6.pkl','rb')); pa=pickle.load(open('oof_stackA.pkl','rb')); pb=pickle.load(open('oof_seqnn.pkl','rb')); pp=(pa+pb)/2
off=np.cumsum([0]+[len(r['y']) for r in R]); res={}
def add(k,se,n):
    res.setdefault(k,[0,0]); res[k][0]+=se; res[k][1]+=n
for v in V:
    wi=v['w']; r=R[wi]; K=W[NAMES[wi]]['K']; n=v['n']; lat=np.arange(K,K+n); S=r['tvt_end']+np.interp(lat,r['idx'],pp[off[wi]:off[wi+1]]); y=v['y']; P=v['P']
    add('stack',((S-y)**2).sum(),n); add('top1',((P[0]-y)**2).sum(),n)
    dist=np.array([np.mean(np.abs(q-S)) for q in P]); add('nearest-to-stack',((P[dist.argmin()]-y)**2).sum(),n)
    for lam in [0.5,1,2]:
        sc=(v['tc']-v['tc'].min())/(v['tc'].max()-v['tc'].min()+1e-8)*10+lam*dist/5; add(f'cost+{lam}dist',((P[sc.argmin()]-y)**2).sum(),n)
    for a in [0.3,0.5,0.7]: add(f'blend{a}',((a*P[0]+(1-a)*S-y)**2).sum(),n)
    g=1/(1+(dist[0]/15)**2); add('gated',((g*P[0]+(1-g)*S-y)**2).sum(),n)
    add('oracle(stack,top1)',min(((S-y)**2).sum(),((P[0]-y)**2).sum()),n)
for k,(s,n) in res.items(): print(f'{k:20s} {np.sqrt(s/n):.2f}')
