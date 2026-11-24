import pickle,glob,sys,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from common import W,NAMES,TREE,PW
cf=['base','g30k','g300k','shift']
D={c:{} for c in cf}
for c in cf:
    for f in glob.glob(f'vensAll_{c}_*.pkl')+[f'vensF_{c}.pkl']:
        for o in pickle.load(open(f,'rb')): D[c][o['w']]=o
ws=sorted(set.intersection(*[set(D[c]) for c in cf])); print(len(ws),'wells with all 4 variants')
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
rows=[]
for w in ws:
    d=W[NAMES[w]]; K=d['K']; yv=D['base'][w]['y']; n=len(yv); t0=d['TVT_input'][K-1]; r=R[w]; li=np.clip(r['idx']-K,0,n-1)
    df=pd.DataFrame({c:D[c][w]['p'][li]-t0 for c in cf}); df['stk']=r['tvt_end']+pp[off[w]:off[w+1]]-t0; df['w']=w; df['y']=yv[li]-t0; df['t']=li; df['frac']=li/n; df['n']=n
    q=np.c_[d['X'][K+li],d['Y'][K+li]]; dist,ii=TREE.query(q,k=300,distance_upper_bound=3000); ii=np.minimum(ii,len(PW)-1); dist=np.where((PW[ii]==w)|~np.isfinite(dist),np.inf,dist); df['dn']=np.minimum(dist.min(1),3000)
    df['mean_v']=df[cf].mean(axis=1); df['sd_v']=df[cf].std(axis=1); df['gap_S']=df['mean_v']-df['stk']; df['rng_v']=df[cf].max(axis=1)-df[cf].min(axis=1); rows.append(df)
A=pd.concat(rows,ignore_index=True); y=A.y.values; print(A.shape)
cen=np.array([[W[NAMES[w]]['X'].mean(),W[NAMES[w]]['Y'].mean()] for w in ws]); nb=12 if len(ws)>400 else 8; km=KMeans(nb,n_init=5,random_state=0).fit(cen); A['blk']=A.w.map(dict(zip(ws,km.labels_)))
base=.8*A.mean_v.values+.2*A.stk.values; rm=lambda p:np.sqrt(np.mean((p-y)**2))
perm=np.random.RandomState(0).permutation(len(NAMES)); te273=set(int(w) for w in perm[500:773]); m273=A.w.isin(te273).values
feats=cf+['stk','mean_v','sd_v','gap_S','rng_v','t','frac','n','dn']
def cv(cols,groups,seeds=(0,1,2),depth=4,leaf=150,it=250,lr=0.04):
    pr=np.zeros(len(A))
    for tr,te in GroupKFold(5).split(A,y,groups):
        for s in seeds:
            m=H(max_iter=it,learning_rate=lr,max_depth=depth,min_samples_leaf=leaf,l2_regularization=5.0,max_features=0.6,random_state=s).fit(A.iloc[tr][cols].values,(y-base)[tr]); pr[te]+=(m.predict(A.iloc[te][cols].values)+base[te])/len(seeds)
    return pr
print('baseline 0.8bag+0.2S: all %.2f | 273-subset %.2f'%(rm(base),np.sqrt(np.mean((base-y)[m273]**2))))
pr=cv(feats,A.blk.values); print('combiner, SPATIAL CV (%d regions): all %.2f | 273-subset %.2f'%(nb,rm(pr),np.sqrt(np.mean((pr-y)[m273]**2))),flush=True)
for dp,lf,it in [(3,300,300),(5,100,250),(6,200,200)]:
    p=cv(feats,A.blk.values,seeds=(0,),depth=dp,leaf=lf,it=it); print('  depth %d leaf %d it %d: all %.2f | 273 %.2f'%(dp,lf,it,rm(p),np.sqrt(np.mean((p-y)[m273]**2))),flush=True)
A['pred']=pr; A.to_pickle('comb3_out.pkl')
