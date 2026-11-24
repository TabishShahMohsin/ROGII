import pickle,glob,numpy as np,pandas as pd,warnings,sys;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
A=pd.read_pickle('comb3_out.pkl');y=A.y.values
V={}
for f in glob.glob('vint_*.pkl'):
    for o in pickle.load(open(f,'rb')):V[o['w']]=o
ws=[w for w in A.w.unique() if w in V];print(len(ws),'of',A.w.nunique(),'wells have internals')
A=A[A.w.isin(ws)].copy();y=A.y.values
sc=['post_sd30','post_ent30','far30','mode230','nmode30','curv30','post_sd300','post_ent300','far300','mode2300','nmode300','curv300','slope','kink','fit_rms','lcorr','tw_sens','tw_loc_sd','edge','fit2_minus_fit1','spread','rng5']
cols={c:np.zeros(len(A)) for c in sc+['map','pmean','pmean300','gap12','cpr','nch']}
pos=0
for w,g in A.groupby('w',sort=False):
    o=V[w];li=np.clip(g.t.values.astype(int),0,o['n']-1);ix=A.index.get_indexer(g.index);t0=o['tvt0']
    for c in sc:cols[c][ix]=o[c][li]
    for c in ('map','pmean','pmean300'):cols[c][ix]=o[c][li]-t0
    cols['gap12'][ix]=o['gap12'];cols['cpr'][ix]=o['cost_per_row'];cols['nch'][ix]=o['nch']
for c,v in cols.items():A[c]=v
A['abs_slope']=A.slope.abs();A['d_pm30']=A['pmean']-A['map'];A['d_pm300']=A['pmean300']-A['map']
rm=lambda p:np.sqrt(np.mean((p-y)**2));base=.8*A.mean_v.values+.2*A.stk.values;blk=A.blk.values
def cv(fs,seeds=(0,1,2)):
    pr=np.zeros(len(A));X=A[fs].values
    for tr,te in GroupKFold(5).split(X,y,blk):
        for s in seeds:
            m=H(max_iter=250,learning_rate=0.04,max_depth=4,min_samples_leaf=150,l2_regularization=5.0,max_features=0.6,random_state=s).fit(X[tr],(y-base)[tr]);pr[te]+=(m.predict(X[te])+base[te])/len(seeds)
    return pr
cf=['base','g30k','g300k','shift'];F0=cf+['stk','mean_v','sd_v','gap_S','rng_v','t','frac','n','dn']
est=['map','pmean','pmean300','d_pm30','d_pm300']
grp={'uncertainty':['post_sd30','post_ent30','post_sd300','post_ent300'],'alias':['far30','far300','mode230','mode2300','nmode30','nmode300','curv30','curv300','gap12','spread','rng5'],
 'physics_fit':['fit_rms','lcorr','tw_sens','tw_loc_sd','edge','fit2_minus_fit1','cpr'],'structure':['slope','abs_slope','kink','nch']}
allg=sum(grp.values(),[])
print('baseline 0.8bag+0.2stk %.2f'%rm(base),flush=True)
R={}
def run(nm,fs):
    p=cv(fs);R[nm]=p;print('%-34s %.2f'%(nm,rm(p)),flush=True)
run('C0 current features',F0)
run('C1 + soft-decode estimators',F0+est)
run('C2 + all diagnostics',F0+allg)
run('C3 + estimators + diagnostics',F0+est+allg)
for g,fs in grp.items():run('  only +'+g,F0+fs)
for g,fs in grp.items():run('  all minus '+g,F0+est+[c for c in allg if c not in fs])
pickle.dump(R,open('comb4_R.pkl','wb'))
