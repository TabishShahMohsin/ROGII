import pickle,numpy as np,glob
from common import W,NAMES
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
C={}
for k in ['S25_300','S25_300abs']:
    C[k]={o['w']:o for f in glob.glob(f'vmC_{k}_*.pkl') for o in pickle.load(open(f,'rb'))}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
ws=sorted(set(D['base'])&set.intersection(*[set(v) for v in C.values()])); print(len(ws),'wells; corridor sets',{k:len(v) for k,v in C.items()})
def S(w):
    K=W[NAMES[w]]['K']; n=D['base'][w]['n']; r=R[w]; return r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]])
def rm(f):
    se=n=0
    for w in ws: y=D['base'][w]['y']; se+=((f(w)-y)**2).sum(); n+=len(y)
    return np.sqrt(se/n)
bag=lambda w:np.mean([D[c][w]['p'] for c in cf],0)
print('base %.2f bag4 %.2f stack %.2f bag4*.8+S*.2 %.2f'%(rm(lambda w:D['base'][w]['p']),rm(bag),rm(S),rm(lambda w:.8*bag(w)+.2*S(w))))
for k,v in C.items():
    print('%s alone %.2f | mean(bag4,%s) %.2f | 0.3*%s+0.7*bag4 %.2f | +0.1 stack %.2f'%(k,rm(lambda w:v[w]['p']),k,rm(lambda w:(bag(w)+v[w]['p'])/2),k,rm(lambda w:.3*v[w]['p']+.7*bag(w)),rm(lambda w:.9*(.3*v[w]['p']+.7*bag(w))+.1*S(w))))
both=lambda w:(C['S25_300'][w]['p']+C['S25_300abs'][w]['p'])/2
print('mean(bag4, both corridors) %.2f ; 0.4*both+0.6*bag4 %.2f'%(rm(lambda w:(bag(w)+both(w))/2),rm(lambda w:.4*both(w)+.6*bag(w))))
# error floor structure: fraction of wells where corridor beats bag
import numpy as np
b=np.array([np.mean((bag(w)-D['base'][w]['y'])**2) for w in ws]); c=np.array([np.mean((C['S25_300'][w]['p']-D['base'][w]['y'])**2) for w in ws]); print('corridor better than bag on %.0f%% of wells'%(100*np.mean(c<b)))
