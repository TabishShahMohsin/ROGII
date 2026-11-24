import pickle, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.ndimage import gaussian_filter1d
import learned_match as L
R=pickle.load(open('ds6.pkl','rb')); post=pickle.load(open('comp_post.pkl','rb'))
DG=L.DG
for r in R:
    p=post.get(r['w']); n=len(r['y'])
    if p is None: 
        for k in ['cp_mean','cp_sd','cp_max','cp_near','cp_neg','cp_pos','cp_ent','cp_mean_s','cp_pos_s','cp_neg_s']: r[k]=np.zeros(n)
        continue
    pm=p@DG; sd=np.sqrt(np.maximum(p@(DG**2)-pm**2,0)); r['cp_mean']=pm; r['cp_sd']=sd; r['cp_max']=p.max(1); r['cp_near']=p[:,np.abs(DG)<=4].sum(1); r['cp_neg']=p[:,DG<-6].sum(1); r['cp_pos']=p[:,DG>6].sum(1)
    r['cp_ent']=-(p*np.log(p+1e-9)).sum(1); r['cp_mean_s']=gaussian_filter1d(pm,3,mode='nearest'); r['cp_pos_s']=gaussian_filter1d(r['cp_pos'],3,mode='nearest'); r['cp_neg_s']=gaussian_filter1d(r['cp_neg'],3,mode='nearest')
y=np.concatenate([r['y'] for r in R]); g=np.concatenate([np.full(len(r['y']),r['w']) for r in R]); rm=lambda e: np.sqrt(np.mean(e**2))
def mat(keys): return np.concatenate([np.c_[tuple(r[k] for k in keys)] for r in R])
old=['Inb_end3000','Inb_win3000','Inb_end6000','Inb_win6000','Ilin300','Ilin1000','dist','dz','ws3000','ws6000']
new=[f'V_{a}{h}' for h in [800,1200,2000,4000] for a in ('end','win')]+[f'vw{h}' for h in [800,1200,2000,4000]]+[f'vgrad{h}' for h in [800,1200,2000,4000]]
GRF=[f'cost{w}_{k}' for w in [150,500,1500] for k in range(9)]+[f'dstar{w}' for w in [150,500,1500]]+[f'vf{w}' for w in [150,500,1500]]+['calc','cala']
Z=[f'inc{s}' for s in [50,200,600]]+[f'curv{s}' for s in [50,200,600]]+['z_minus_smooth','z_minus_smooth600','dz_to_end','md_to_end','dz_since_K','z_from_runmax','z_from_runmin','az_dev','inc_fwd','inc_bwd','inc_change']
CP=['cp_mean','cp_sd','cp_max','cp_near','cp_neg','cp_pos','cp_ent','cp_mean_s','cp_pos_s','cp_neg_s']
def cv(keys,name):
    F=mat(keys); pred=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(F,y,g):
        m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=200,random_state=0); m.fit(F[tr],y[tr]); pred[te]=m.predict(F[te])
    print(f"{name:<52} CV pooled RMSE {rm(pred-y):6.3f}",flush=True); return pred
cv(old+new+Z,"A + Z (no GR)"); p0=cv(old+new+GRF+Z,"A + Z + handcrafted GR cost (10.87 ref)"); p1=cv(old+new+Z+CP,"A + Z + LEARNED comparator posteriors"); p2=cv(old+new+GRF+Z+CP,"A + Z + handcrafted + learned")
wi=np.unique(g); sa=np.array([np.sum((p0-y)[g==w]**2) for w in wi]); sb=np.array([np.sum((p2-y)[g==w]**2) for w in wi]); nw=np.array([np.sum(g==w) for w in wi]); rs=np.random.RandomState(0); d=[]
for _ in range(500): i=rs.choice(len(wi),len(wi)); d.append(np.sqrt(sa[i].sum()/nw[i].sum())-np.sqrt(sb[i].sum()/nw[i].sum()))
print("paired bootstrap gain of learned posteriors over handcrafted-GR stack: %.2f ft, 95%% CI [%.2f, %.2f]"%(np.mean(d),*np.percentile(d,[2.5,97.5])))
pickle.dump(p2,open('oof_stack_cp.pkl','wb'))
