import pickle,numpy as np
from common import W,NAMES
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.model_selection import KFold
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
cen=np.array([[W[n]['X'].mean(),W[n]['Y'].mean()] for n in NAMES]); kd=cKDTree(cen)
ws=sorted(D['base']); F=[];T=[];C=[]
for w in ws:
    d=W[NAMES[w]]; K=d['K']; y=D['base'][w]['y']; n=len(y); r=R[w]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]]); P=np.array([D[c][w]['p'] for c in cf]); bag=P.mean(0); t0=d['TVT_input'][K-1]
    x=np.arange(n)/n; cS=np.polyfit(x,S-bag,1); cB=np.polyfit(x,bag-t0,1); tgt=np.polyfit(x,y-bag,1)
    s=slice(max(0,K-1000),K); pre=np.polyfit(d['MD'][s],d['TVT_input'][s],1)[0]
    F.append([cS[0],cS[1],cB[0],cB[1],bag[-1]-S[-1],bag[-1]-t0,S[-1]-t0,P.std(0).mean(),np.abs(P[0]-S).mean(),np.log(kd.query(cen[w],2)[0][1]+1),n/1000,pre,np.abs(bag-t0).max()])
    T.append(tgt); C.append((bag,y,x))
F=np.array(F);T=np.array(T)
def ev(pred):
    se=nn=0
    for (bag,y,x),t in zip(C,pred): se+=((bag+t[0]*x*0+np.polyval(t,x)-y)**2).sum(); nn+=len(y)
    return np.sqrt(se/nn)
print('bag %.2f'%ev(np.zeros_like(T)),' oracle %.2f'%ev(T))
for name,mk in [('ridge',lambda:RidgeCV(alphas=np.logspace(-1,4,12))),('gbm',lambda:GB(n_estimators=80,max_depth=2,learning_rate=0.05,subsample=0.7,loss='huber',random_state=0))]:
    out=[]
    for seed in range(3):
        pr=np.zeros_like(T)
        for tr,te in KFold(5,shuffle=True,random_state=seed).split(F):
            mu,sd=F[tr].mean(0),F[tr].std(0)+1e-9
            for j in range(2):
                m=mk().fit((F[tr]-mu)/sd,T[tr,j]); pr[te,j]=m.predict((F[te]-mu)/sd)
        out.append([ev(pr),ev(0.5*pr)])
    print(name,'full %.2f  half-shrunk %.2f'%tuple(np.mean(out,0)))
