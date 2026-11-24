import pickle,numpy as np
from common import W,NAMES
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
res={}
def add(k,se,n): res.setdefault(k,[0,0]); res[k][0]+=se; res[k][1]+=n
for w in D['base']:
    d=W[NAMES[w]]; K=d['K']; y=D['base'][w]['y']; n=len(y); r=R[w]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]]); bag=np.mean([D[c][w]['p'] for c in cf],0)
    x=np.arange(n)/max(n,1)
    for deg in [0,1]:
        c=np.polyfit(x,S-bag,deg); corr=np.polyval(c,x)
        for lam in [0.2,0.35,0.5]: add(f'bag+{lam}*linfit(S-bag) deg{deg}',((bag+lam*corr-y)**2).sum(),n)
    # with weights ramp: stack gets more say late
    for lam in [0.3,0.5]: add(f'ramp blend {lam}',(( (1-lam*x)*bag+lam*x*S-y)**2).sum(),n)
    add('0.8bag+0.2S',((0.8*bag+0.2*S-y)**2).sum(),n)
for k,(s,n) in res.items(): print(f'{k:40s} {np.sqrt(s/n):.2f}')
