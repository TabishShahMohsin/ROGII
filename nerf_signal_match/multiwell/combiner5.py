import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
A=pd.read_pickle('comb3_out.pkl');y=A.y.values;R0={r['w']:r for r in pickle.load(open('nbpath.pkl','rb'))};from common import NAMES;R={i:R0[n] for i,n in enumerate(NAMES)}
names=[]
for mode in ('md','xyz'):
    for k in (1,2,3):names.append(f'{mode}{k}')
    names+= [f'{mode}_mean',f'{mode}_sd',f'{mode}_dw',f'{mode}_d1',f'{mode}_par1',f'{mode}_dm']
C={c:np.zeros(len(A)) for c in names}
for w,g in A.groupby('w',sort=False):
    r=R[w];n=len(r['y']);li=np.clip(g.t.values.astype(int),0,n-1);ix=A.index.get_indexer(g.index);t0=r['tvt0'];wt=1/(np.array(r['dist'])+50);wt/=wt.sum()
    for mode in ('md','xyz'):
        P=r[mode][:,li]-t0
        for k in range(3):C[f'{mode}{k+1}'][ix]=P[k]
        C[f'{mode}_mean'][ix]=P.mean(0);C[f'{mode}_sd'][ix]=P.std(0);C[f'{mode}_dw'][ix]=(wt[:,None]*P).sum(0)
        C[f'{mode}_d1'][ix]=r['dist'][0];C[f'{mode}_par1'][ix]=r['par'][0];C[f'{mode}_dm'][ix]=r[mode+'_d'][0][li]
for c,v in C.items():A[c]=v
for m in ('md','xyz'):A[f'{m}_gap']=A[f'{m}_dw']-A.mean_v
rm=lambda p:np.sqrt(np.mean((p-y)**2));base=.8*A.mean_v.values+.2*A.stk.values;blk=A.blk.values
print('nb-path alone (md_dw, TVT-t0 space) %.2f | xyz_dw %.2f | hold %.2f'%(rm(A.md_dw.values),rm(A.xyz_dw.values),rm(0*y)))
def cv(fs,seeds=(0,1,2)):
    pr=np.zeros(len(A));X=A[fs].values
    for tr,te in GroupKFold(5).split(X,y,blk):
        for s in seeds:
            m=H(max_iter=250,learning_rate=0.04,max_depth=4,min_samples_leaf=150,l2_regularization=5.0,max_features=0.6,random_state=s).fit(X[tr],(y-base)[tr]);pr[te]+=(m.predict(X[te])+base[te])/len(seeds)
    return pr
cf=['base','g30k','g300k','shift'];F0=cf+['stk','mean_v','sd_v','gap_S','rng_v','t','frac','n','dn']
ext=lambda m:[f'{m}{k}' for k in (1,2,3)]+[f'{m}_mean',f'{m}_sd',f'{m}_dw',f'{m}_d1',f'{m}_par1',f'{m}_dm',f'{m}_gap']
for nm,fs in (('C0 current',F0),('+ md-projection (notebook, by MD)',F0+ext('md')),('+ xyz-projection (notebook, by 3D nearest)',F0+ext('xyz')),('+ both',F0+ext('md')+ext('xyz')),('+ means/gaps only (md,xyz)',F0+['md_dw','xyz_dw','md_gap','xyz_gap','md_sd','xyz_sd','md_d1'])):
    p=cv(fs);print('%-44s %.2f'%(nm,rm(p)),flush=True)
