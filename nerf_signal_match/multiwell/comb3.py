import pickle,numpy as np
from common import W,NAMES
cf=['base','g30k','g300k','shift']
D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}; common=sorted(set.intersection(*[set(D[c]) for c in cf]))
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
def S(w):
    K=W[NAMES[w]]['K']; n=D['base'][w]['n']; r=R[w]; return r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]])
def rm(f):
    se=n=0
    for w in common: y=D['base'][w]['y']; se+=((f(w)-y)**2).sum(); n+=len(y)
    return np.sqrt(se/n)
print(len(common))
for c in cf: print(c,'%.2f'%rm(lambda w:D[c][w]['p']))
bag=lambda w:np.mean([D[c][w]['p'] for c in cf],0); print('bag4 %.2f'%rm(bag),'stack %.2f'%rm(S))
for a in [.7,.8,.9]: print('bag4+stack',a,'%.2f'%rm(lambda w:a*bag(w)+(1-a)*S(w)))
med=lambda w:np.median([D[c][w]['p'] for c in cf],0); print('median4 %.2f'%rm(med))
