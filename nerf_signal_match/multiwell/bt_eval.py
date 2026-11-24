import pickle,numpy as np,pandas as pd
from scipy.stats import spearmanr
from common import W,NAMES
cfgs=['base','g30k','g300k','c250']
BT={c:{o['w']:o for o in pickle.load(open(f'bt_{c}_500_580.pkl','rb'))} for c in cfgs}
L={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in ['base','g30k','g300k']}; L['c250']={o['w']:o for o in pickle.load(open('vens_c250.pkl','rb'))}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
def S(w):
    K=W[NAMES[w]]['K']; n=L['base'][w]['n']; r=R[w]; return r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]])
ws=sorted(set.intersection(*[set(BT[c]) for c in cfgs])&set.intersection(*[set(L[c]) for c in cfgs])); print(len(ws),'wells')
pre=pd.DataFrame({c:[BT[c][w]['rm'] for w in ws] for c in cfgs},index=ws); lat=pd.DataFrame({c:[np.sqrt(np.mean((L[c][w]['p']-L[c][w]['y'])**2)) for w in ws] for c in cfgs},index=ws)
print('Spearman(prefix RMSE, lateral RMSE) per config:',{c:round(spearmanr(pre[c],lat[c])[0],2) for c in cfgs})
print('median prefix RMSE %s | lateral %s'%(pre.median().round(1).to_dict(),lat.median().round(1).to_dict()))
def pooled(f):
    se=n=0
    for w in ws: y=L['base'][w]['y']; se+=((f(w)-y)**2).sum(); n+=len(y)
    return np.sqrt(se/n)
print('fixed: ',{c:round(pooled(lambda w:L[c][w]['p']),2) for c in cfgs}, ' mean4: %.2f'%pooled(lambda w:np.mean([L[c][w]['p'] for c in cfgs],0)),' stack: %.2f'%pooled(S))
best=lambda w:min(cfgs,key=lambda c:BT[c][w]['rm']); print('select config by prefix backtest: %.2f'%pooled(lambda w:L[best(w)][w]['p']))
orac=lambda w:min(cfgs,key=lambda c:np.mean((L[c][w]['p']-L[c][w]['y'])**2)); print('oracle config per well: %.2f'%pooled(lambda w:L[orac(w)][w]['p']))
for tau in [2,4,8]:
    def wavg(w):
        e=np.array([BT[c][w]['rm'] for c in cfgs]); k=np.exp(-(e-e.min())/tau); k/=k.sum(); return sum(ki*L[c][w]['p'] for ki,c in zip(k,cfgs))
    print('softmin weights over configs tau=%d: %.2f'%(tau,pooled(wavg)))
# gating Viterbi vs stack by prefix reliability
bag=lambda w:np.mean([L[c][w]['p'] for c in cfgs],0); pr=lambda w:np.mean([BT[c][w]['rm'] for c in cfgs]); 
ratio={w:pr(w)/max(BT['base'][w]['hold'],1e-6) for w in ws}
print('Spearman(prefix mean RMSE, lateral bag RMSE) = %.2f ; Spearman(prefix RMSE/hold, lateral RMSE)=%.2f'%(spearmanr([pr(w) for w in ws],[np.sqrt(np.mean((bag(w)-L['base'][w]['y'])**2)) for w in ws])[0],spearmanr([ratio[w] for w in ws],[np.sqrt(np.mean((bag(w)-L['base'][w]['y'])**2)) for w in ws])[0]))
for t in [4,6,8,12]:
    f=lambda w:(lambda g:g*bag(w)+(1-g)*S(w))(1/(1+(pr(w)/t)**3)); print('gate vit-vs-stack by prefix RMSE (t=%d): %.2f'%(t,pooled(f)))
print('fixed 0.8bag+0.2S: %.2f'%pooled(lambda w:0.8*bag(w)+0.2*S(w)))
