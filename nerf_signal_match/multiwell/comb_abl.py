import pickle,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from common import W,NAMES
A=pd.read_pickle('combiner_out.pkl'); y=A.y.values; cf=['base','g30k','g300k','shift']
base=.8*A.mean_v.values+.2*A.stk.values; rm=lambda p:np.sqrt(np.mean((p-y)**2))
cen=np.array([[W[NAMES[w]]['X'].mean(),W[NAMES[w]]['Y'].mean()] for w in A.w.unique()]); km=KMeans(8,n_init=5,random_state=0).fit(cen); blk=dict(zip(A.w.unique(),km.labels_)); A['blk']=A.w.map(blk)
feat_all=cf+['S05','C300','C300abs','stk','mean_v','sd_v','sd_all','gap_S','mean_all','t','frac','n','dn']
mk=lambda:H(max_iter=250,learning_rate=0.04,max_depth=4,min_samples_leaf=100,l2_regularization=5.0,random_state=0)
def cv(cols,groups,resid=True,seeds=(0,)):
    pr=np.zeros(len(A))
    for tr,te in GroupKFold(5).split(A,y,groups):
        m=mk().fit(A.iloc[tr][cols].values,(y-base)[tr]); pr[te]=m.predict(A.iloc[te][cols].values)+base[te]
    return pr
print('baseline 0.8bag+0.2S %.2f'%rm(base))
print('well-grouped CV, all features: %.2f'%rm(cv(feat_all,A.w.values)))
print('SPATIAL-BLOCK CV (8 regions), all features: %.2f'%rm(cv(feat_all,A.blk.values)))
for nm,drop in [('no position feats (t,frac,n,dn)',['t','frac','n','dn']),('no dn',['dn']),('no individual estimators',cf+['S05','C300','C300abs','stk']),('no disagreement',['sd_v','sd_all','gap_S']),('no corridor/S05 members',['S05','C300','C300abs']),('no stack',['stk','gap_S'])]:
    cols=[c for c in feat_all if c not in drop]; print('spatial CV, %-34s %.2f'%(nm,rm(cv(cols,A.blk.values))))
# per-well effect
pr=cv(feat_all,A.blk.values); A['pr']=pr; g=A.groupby('w'); bw=g.apply(lambda d:np.sqrt(np.mean((base[d.index]-d.y)**2))); pw=g.apply(lambda d:np.sqrt(np.mean((pr[d.index]-d.y)**2)))
print('wells improved: %.0f%%; worst-19 wells RMSE %.1f -> %.1f'%(100*(pw<bw).mean(),np.sqrt((bw.sort_values().tail(19)**2).mean()),np.sqrt((pw[bw.sort_values().tail(19).index]**2).mean())))
print('by region RMSE base->comb:',[(int(b),round(float(np.sqrt(np.mean((base[A.blk==b]-y[A.blk==b])**2))),1),round(float(np.sqrt(np.mean((pr[A.blk==b]-y[A.blk==b])**2))),1)) for b in range(8)])
