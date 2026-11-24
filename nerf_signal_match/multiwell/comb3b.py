import numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
A=pd.read_pickle('comb3_out.pkl'); y=A.y.values; cf=['base','g30k','g300k','shift']; rm=lambda p,m=slice(None):np.sqrt(np.mean((p-y)[m]**2))
perm=np.random.RandomState(0).permutation(773); m273=A.w.isin(set(int(w) for w in perm[500:773])).values
def cv(cols,base,seeds=(0,1,2)):
    pr=np.zeros(len(A))
    for tr,te in GroupKFold(5).split(A,y,A.blk.values):
        for s in seeds:
            m=H(max_iter=250,learning_rate=0.04,max_depth=4,min_samples_leaf=150,l2_regularization=5.0,max_features=0.6,random_state=s).fit(A.iloc[tr][cols].values,(y-base)[tr]); pr[te]+=(m.predict(A.iloc[te][cols].values)+base[te])/len(seeds)
    return pr
b0=A.mean_v.values; print('bag4 baseline: all %.2f 273 %.2f'%(rm(b0),rm(b0,m273)))
f_nostk=cf+['mean_v','sd_v','rng_v','t','frac','n','dn']
p=cv(f_nostk,b0); print('Viterbi-only combiner (no stack): all %.2f | 273 %.2f'%(rm(p),rm(p,m273)))
f_nodn=[c for c in f_nostk if c!='dn']; p2=cv(f_nodn,b0); print('  ... without isolation feature dn: all %.2f | 273 %.2f'%(rm(p2),rm(p2,m273)))
p3=cv(cf+['stk','mean_v','sd_v','gap_S','rng_v','t','frac','n','dn'],b0); print('with stack feature, base=bag4: all %.2f | 273 %.2f'%(rm(p3),rm(p3,m273)))
pr=A.pred.values; print('stack-based full (earlier): all %.2f'%rm(pr))
A['pred_nostk']=p; A.to_pickle('comb3_out.pkl')
