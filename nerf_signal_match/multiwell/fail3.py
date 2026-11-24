import pickle,numpy as np
from scipy.ndimage import uniform_filter1d as uf
from common import W,NAMES
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
res={}
def add(k,se,n): res.setdefault(k,[0,0]); res[k][0]+=se; res[k][1]+=n
for w in D['base']:
    d=W[NAMES[w]]; K=d['K']; y=D['base'][w]['y']; n=len(y); r=R[w]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]]); bag=np.mean([D[c][w]['p'] for c in cf],0); t0=d['TVT_input'][K-1]
    add('bag',((bag-y)**2).sum(),n); add('0.8bag+0.2S',((0.8*bag+0.2*S-y)**2).sum(),n)
    add('ORACLE const offset removed',((bag-bag.mean()+y.mean()-y)**2).sum(),n)
    x=np.arange(n); c=np.polyfit(x,y-bag,1); add('ORACLE linear trend removed',((bag+np.polyval(c,x)-y)**2).sum(),n)
    add('ORACLE first-1000 mean offset fixed',((bag+(y[:1000]-bag[:1000]).mean()-y)**2).sum(),n)
    for tau in [300,1000,2000]:
        if n>tau: 
            hf=bag-uf(bag,tau,mode='nearest'); lf=uf(S,tau,mode='nearest'); add(f'S_lowfreq{tau}+bag_hf',((lf+hf-y)**2).sum(),n)
            lf2=uf(0.5*S+0.5*bag,tau,mode='nearest'); add(f'mix_lf{tau}+bag_hf',((lf2+hf-y)**2).sum(),n)
        else: add(f'S_lowfreq{tau}+bag_hf',((bag-y)**2).sum(),n); add(f'mix_lf{tau}+bag_hf',((bag-y)**2).sum(),n)
    # error as function of position
    for a,b in [(0,500),(500,1500),(1500,3000),(3000,10**6)]:
        e=(bag[a:b]-y[a:b]); add(f'pos{a}-{b}',(e**2).sum(),len(e))
for k,(s,n) in res.items(): print(f'{k:40s} {np.sqrt(s/n):.2f}')
