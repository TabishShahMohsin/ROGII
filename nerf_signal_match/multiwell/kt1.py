import pickle,glob,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HC,HistGradientBoostingRegressor as HR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
A=pd.read_pickle('comb3_out.pkl');y=A.y.values;blk=A.blk.values;rm=lambda p:np.sqrt(np.mean((p-y)**2))
print('reference: bag mean_v %.2f | stack %.2f | 0.8/0.2 %.2f | combiner %.2f'%(rm(A.mean_v.values),rm(A.stk.values),rm(.8*A.mean_v.values+.2*A.stk.values),rm(A.pred.values)))
# E1: signed sum-to-one ridge vs simplex over members (fold-safe, spatial folds)
M=['base','g30k','g300k','shift','stk'];X=A[M].values
def signed(Xtr,ytr,lam):
    # min ||y - X w||^2 + lam||w-1/m||^2 s.t. sum w=1  -> solve via substitution
    m=X.shape[1];w0=np.ones(m)/m;Z=Xtr-Xtr[:,[-1]];t=ytr-Xtr[:,-1]-(Z[:, :-1]@np.zeros(m-1))  # reparam w_last=1-sum others
    Zs=Xtr[:,:-1]-Xtr[:,[-1]];ts=ytr-Xtr[:,-1];c=np.linalg.solve(Zs.T@Zs+lam*np.eye(m-1)*len(ytr)*1e-3,Zs.T@ts+lam*np.eye(m-1)@w0[:-1]*len(ytr)*1e-3);return np.r_[c,1-c.sum()]
from scipy.optimize import nnls
for nm,fn in (('simplex-ish (nnls, renorm)',lambda a,b:(lambda w:w/w.sum())(nnls(a,b)[0])),('signed ridge lam=1',lambda a,b:signed(a,b,1)),('signed ridge lam=0.01',lambda a,b:signed(a,b,0.01))):
    p=np.zeros(len(y));ws=[]
    for b in np.unique(blk):
        tr=blk!=b;w=fn(X[tr],y[tr]);p[~tr]=X[~tr]@w;ws.append(w)
    print('E1 %-28s %.2f  weights %s'%(nm,rm(p),np.round(np.mean(ws,0),2)))
# E2: per-well hard gate between combiner pred and bag/stack, target-free well aggregates
V={}
for f in glob.glob('vint_*.pkl'):
    for o in pickle.load(open(f,'rb')):V[o['w']]={k:(float(np.mean(o[k])) if k in('post_sd300','far300','post_sd30','far30','fit_rms','lcorr','tw_sens','spread','curv300','nmode300','kink') else o[k]) for k in ('post_sd300','far300','post_sd30','far30','fit_rms','lcorr','tw_sens','spread','curv300','nmode300','kink','gap12','cost_per_row','nch')}
rows=[]
for w,g in A.groupby('w'):
    r=dict(w=w,blk=g.blk.iloc[0],n=len(g),dn=g.dn.iloc[0],**V[w])
    for c in ['sd_v','gap_S','rng_v']:r[c+'_m']=g[c].mean();r[c+'_x']=g[c].max()
    r['stk_m']=g.stk.abs().mean();r['mv_m']=g.mean_v.abs().mean();r['mv_end']=g.mean_v.iloc[-20:].mean();r['stk_end']=g.stk.iloc[-20:].mean()
    r['sse_comb']=np.sum((g.pred-g.y)**2);r['sse_stk']=np.sum((g.stk-g.y)**2);r['sse_bag']=np.sum((g.mean_v-g.y)**2);r['sse_mix']=np.sum((.8*g.mean_v+.2*g.stk-g.y)**2);rows.append(r)
Wd=pd.DataFrame(rows);feat=[c for c in Wd.columns if c not in('w','blk','n','sse_comb','sse_stk','sse_bag','sse_mix')]
tot=Wd.n.sum()
print('\nE2 oracle per-well best-of {comb,stk,bag}: %.2f  (headroom) | best-of {comb,stk} %.2f'%(np.sqrt(Wd[['sse_comb','sse_stk','sse_bag']].min(1).sum()/tot),np.sqrt(Wd[['sse_comb','sse_stk']].min(1).sum()/tot)))
# gate between combiner (A) and bag (B) : predict p(B better) ; also soft-gate
for a,b in (('comb','bag'),('comb','stk'),('mix','stk')):
    lab=(Wd['sse_'+b]<Wd['sse_'+a]).astype(int).values;pr=np.zeros(len(Wd))
    for tr,te in GroupKFold(5).split(Wd,lab,Wd.blk):
        m=HC(max_iter=150,learning_rate=0.05,max_depth=3,min_samples_leaf=15,l2_regularization=5.0,random_state=0).fit(Wd.iloc[tr][feat],lab[tr],sample_weight=np.sqrt(Wd.n.values[tr]));pr[te]=m.predict_proba(Wd.iloc[te][feat])[:,1]
    hard=np.where(pr>0.5,Wd['sse_'+b],Wd['sse_'+a]);soft=pr*Wd['sse_'+b]*0+0  # placeholder
    print('E2 gate %s vs %s: base %.2f  alt %.2f  hard-gate %.2f   (alt better on %.0f%% wells, gate acc %.0f%%)'%(a,b,np.sqrt(Wd['sse_'+a].sum()/tot),np.sqrt(Wd['sse_'+b].sum()/tot),np.sqrt(hard.sum()/tot),100*lab.mean(),100*((pr>0.5)==lab).mean()))
# E3: direction coefficient: pred = comb + c*(bag - stk), c fit fold-safe
d=(A.mean_v-A.stk).values;resid=y-A.pred.values;p=np.zeros(len(y));cs=[]
for b in np.unique(blk):
    tr=blk!=b;c=(d[tr]*resid[tr]).sum()/(d[tr]**2).sum();p[~tr]=A.pred.values[~tr]+c*d[~tr];cs.append(c)
print('\nE3 combiner + c*(bag-stk): %.2f (c per fold %s)'%(rm(p),np.round(cs,3)))
