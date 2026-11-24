import pickle,numpy as np
from common import W,NAMES
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
V=pickle.load(open('vitP_unif_500_637.pkl','rb'))+pickle.load(open('vitP_unif_637_773.pkl','rb'))
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2
off=np.cumsum([0]+[len(r['y']) for r in R])
X=[];Y=[];G=[];Sm=[]
for v in V:
    wi=v['w']; r=R[wi]; d=W[NAMES[wi]]; K=d['K']; n=v['n']; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[wi]:off[wi+1]]); P=v['P']; y=v['y']; tc=v['tc']; t0=v['tvt0']
    for j,p in enumerate(P):
        o=[q for k,q in enumerate(P) if k!=j]
        X.append([j,tc[j]-tc[0],(tc[j]-tc[0])/(abs(tc[0])+1),np.log(tc[0]/n),np.mean(np.abs(p-S)),p[-1]-S[-1],np.mean(np.abs(p-t0)),p[-1]-t0,np.abs(p-t0).max(),np.mean([np.mean(np.abs(p-q)) for q in o]) if o else 0,np.mean(np.abs(np.diff(p))),n,len(P),np.mean(np.abs(P[0]-S))])
        Y.append(np.mean((p-y)**2)); G.append(len(Sm)); 
    Sm.append((S,y,P))
X=np.array(X,float);Y=np.array(Y);G=np.array(G)
pred=np.zeros(len(Y))
for tr,te in GroupKFold(5).split(X,Y,G):
    m=H(max_iter=150,learning_rate=0.05,max_depth=3,min_samples_leaf=20).fit(X[tr],np.log1p(Y[tr])); pred[te]=m.predict(X[te])
se=dict(top1=0,sel=0,selb=0,stack=0,orac=0,blend=0); nn=0
for wi,(S,y,P) in enumerate(Sm):
    ix=np.where(G==wi)[0]; j=pred[ix].argmin(); p=P[j]; nn+=len(y)
    se['top1']+=((P[0]-y)**2).sum(); se['sel']+=((p-y)**2).sum(); se['selb']+=((0.8*p+0.2*S-y)**2).sum(); se['stack']+=((S-y)**2).sum(); se['orac']+=min(((q-y)**2).sum() for q in P); se['blend']+=((0.8*P[0]+0.2*S-y)**2).sum()
print({k:round(float(np.sqrt(v/nn)),2) for k,v in se.items()})
