import pickle,numpy as np
from common import W,NAMES
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
Sv={o['w']:o for f in ['vmF_S05a.pkl','vmF_S05b.pkl'] for o in pickle.load(open(f,'rb'))}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
ws=sorted(set(D['base'])&set(Sv)); print(len(ws))
def S(w):
    K=W[NAMES[w]]['K']; n=D['base'][w]['n']; r=R[w]; return r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]])
def rm(f):
    se=n=0
    for w in ws: y=D['base'][w]['y']; se+=((f(w)-y)**2).sum(); n+=len(y)
    return np.sqrt(se/n)
bag=lambda w:np.mean([D[c][w]['p'] for c in cf],0); s05=lambda w:Sv[w]['p']
print('base %.2f | S05 alone %.2f | bag4 %.2f | stack %.2f'%(rm(lambda w:D['base'][w]['p']),rm(s05),rm(bag),rm(S)))
print('bag4*.8+S*.2 %.2f'%rm(lambda w:.8*bag(w)+.2*S(w)))
for a in [.3,.4,.5,.6]: print('mix a*S05+(1-a)*bag4  a=%.1f: %.2f | +0.1 stack: %.2f'%(a,rm(lambda w:a*s05(w)+(1-a)*bag(w)),rm(lambda w:.9*(a*s05(w)+(1-a)*bag(w))+.1*S(w))))
