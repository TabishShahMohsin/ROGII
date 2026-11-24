import pickle,numpy as np,os
from common import W,NAMES
cf=[c for c in ['base','peaks','c100','c250','g30k','g300k'] if os.path.exists(f'vens_{c}.pkl')]
D={c:{o['w']:o for o in pickle.load(open(f'vens_{c}.pkl','rb'))} for c in cf}; common=sorted(set.intersection(*[set(D[c]) for c in cf]))
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2
off=np.cumsum([0]+[len(r['y']) for r in R])
def S(w):
    K=W[NAMES[w]]['K']; n=D['base'][w]['n']; r=R[w]; return r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]])
def rm(f):
    se=n=0
    for w in common: y=D['base'][w]['y']; se+=((f(w)-y)**2).sum(); n+=len(y)
    return np.sqrt(se/n)
use=[c for c in cf if c not in('peaks','c100')]; print(use)
bag=lambda w:np.mean([D[c][w]['p'] for c in use],0)
print('stack %.2f bag %.2f'%(rm(S),rm(bag)))
for a in [.5,.6,.7,.8,.9]: print(a,'%.2f'%rm(lambda w:a*bag(w)+(1-a)*S(w)))
# robust: median over members incl all
allm=lambda w:np.median([D[c][w]['p'] for c in cf]+[S(w)],0); print('median all+stack %.2f'%rm(allm))
