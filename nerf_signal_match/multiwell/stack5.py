import pickle, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
R=pickle.load(open('ds7.pkl','rb'))
y=np.concatenate([r['y'] for r in R]); g=np.concatenate([np.full(len(r['y']),r['w']) for r in R]); rm=lambda e: np.sqrt(np.mean(e**2))
def mat(keys): return np.concatenate([np.c_[tuple(r[k] for k in keys)] for r in R])
old=['Inb_end3000','Inb_win3000','Inb_end6000','Inb_win6000','Ilin300','Ilin1000','dist','dz','ws3000','ws6000']
new=[f'V_{a}{h}' for h in [800,1200,2000,4000] for a in ('end','win')]+[f'vw{h}' for h in [800,1200,2000,4000]]+[f'vgrad{h}' for h in [800,1200,2000,4000]]
GRF=[f'cost{w}_{k}' for w in [150,500,1500] for k in range(9)]+[f'dstar{w}' for w in [150,500,1500]]+[f'vf{w}' for w in [150,500,1500]]+['calc','cala']
NGR=['n_'+k for k in GRF]
Z=[f'inc{s}' for s in [50,200,600]]+[f'curv{s}' for s in [50,200,600]]+['z_minus_smooth','z_minus_smooth600','dz_to_end','md_to_end','dz_since_K','z_from_runmax','z_from_runmin','az_dev','inc_fwd','inc_bwd','inc_change']
def cv(keys,name):
    F=mat(keys); pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(F,y,g):
        m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=200,random_state=0); m.fit(F[tr],y[tr]); pred[te]=m.predict(F[te])
    print(f"{name:<46} CV pooled RMSE {rm(pred-y):6.3f}",flush=True)
cv(old+new+Z,"A + Z (no GR)"); cv(old+new+GRF+Z,"A + Z + GR cost (type-log lookup)"); cv(old+new+NGR+Z,"A + Z + GR cost (NeRF-rendered kernel)"); cv(old+new+GRF+NGR+Z,"A + Z + both")
