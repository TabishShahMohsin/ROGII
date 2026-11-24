import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
A=pd.read_pickle('comb3_out.pkl');F=pd.read_pickle('ladder_F.pkl');y=A.y.values;dZ=F.dZ.values;dm=F.dMD.values
rm=lambda e:np.sqrt(np.mean(e**2));blk=A.blk.values
zb=F.zsl.values*dm-dZ          # TVT-space prediction of 'S flat with Z-slope': TVT=t0+zsl*dMD-dZ
print('hold TVT %.2f | S0+zsl*dMD %.2f'%(rm(y),rm(y-zb)))
# linear: zsl term + nbd
def cvfit(X,t):
    p=np.zeros(len(t))
    for b in np.unique(blk):
        tr=blk!=b;c=np.linalg.lstsq(X[tr],t[tr],rcond=None)[0];p[~tr]=X[~tr]@c
    return p
for h in (1500,4000):
    nd=F[f'nbd{h}'].values;nw=F[f'nbw{h}'].values;gS=y+dZ
    p=cvfit(np.c_[F.zsl*dm,nd],gS);print('zsl + nbd%d: %.2f'%(h,rm(gS-p)),end=' | ')
    p=cvfit(np.c_[F.zsl*dm,nd*nw/(nw+.3),nd],gS);print('zsl + nbd*w, nbd: %.2f'%rm(gS-p))
base_feats=['dZ','dMD','dX','dY','sl100','sl300','sl1000','sl3000','zsl','nbd1500','nbw1500','nbd4000','nbw4000']
X0=pd.concat([F[base_feats],A[['t','frac','n','dn']]],axis=1)
def cv(X,target,base,seeds=(0,1)):
    pr=np.zeros(len(X))
    for tr,te in GroupKFold(5).split(X,y,blk):
        for s in seeds:
            m=H(max_iter=250,learning_rate=0.04,max_depth=4,min_samples_leaf=150,l2_regularization=5.0,max_features=0.6,random_state=s).fit(X.iloc[tr].values,(target-base)[tr]);pr[te]+=(m.predict(X.iloc[te].values)+base[te])/len(seeds)
    return pr
print('\nHGB, NO Viterbi/stack features (pure geometry/prefix/neighbour), spatial CV:')
z=np.zeros(len(y))
for nm,base in (('residual of hold-TVT (TVT space)',z),('residual of S0+zsl*dMD (S space)',zb)):
    p=cv(X0,y,base);print('  %-36s %.2f'%(nm,rm(y-p)),flush=True)
print('\nHGB with Viterbi ensemble+stack features:')
vf=['base','g30k','g300k','shift','stk','mean_v','sd_v','gap_S','rng_v']
X1=pd.concat([A[vf],X0],axis=1);bv=.8*A.mean_v.values+.2*A.stk.values
for nm,base in (('base=0.8mv+0.2stk (current, TVT)',bv),('base=S0+zsl*dMD',zb),('base=0',z)):
    p=cv(X1,y,base);print('  %-36s %.2f'%(nm,rm(y-p)),flush=True)
# add S-space versions of the estimators as features (mean_v+dZ etc) -- trees see differences to the S-baseline
X2=X1.copy();X2['mv_minus_zb']=A.mean_v.values-zb;X2['stk_minus_zb']=A.stk.values-zb
p=cv(X2,y,bv);print('  + estimator-minus-zbase feats, base=bv  %.2f'%rm(y-p))
