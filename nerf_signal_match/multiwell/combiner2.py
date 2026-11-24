import pickle,glob,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
from common import W,NAMES,TREE,PW
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
C={k:{o['w']:o for f in glob.glob(f'vmC_{k}_*.pkl') for o in pickle.load(open(f,'rb'))} for k in ['S25_300','S25_300abs']}; S05={o['w']:o for f in glob.glob('vmF_S05*.pkl') for o in pickle.load(open(f,'rb'))}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
rows=[]; skip={'w','K','idx','y','tvt_end','yA'}
for w in sorted(D['base']):
    d=W[NAMES[w]]; K=d['K']; yv=D['base'][w]['y']; n=len(yv); t0=d['TVT_input'][K-1]; r=R[w]; li=np.clip(r['idx']-K,0,n-1); m=len(li)
    P={c:D[c][w]['p'][li] for c in cf}; P['S05']=S05[w]['p'][li]; P['C300']=C['S25_300'][w]['p'][li]; P['C300abs']=C['S25_300abs'][w]['p'][li]; P['stk']=(r['tvt_end']+pp[off[w]:off[w+1]])
    df=pd.DataFrame({k:v-t0 for k,v in P.items()}); df['w']=w; df['y']=yv[li]-t0; df['t']=li; df['frac']=li/n; df['n']=n
    q=np.c_[d['X'][K+li],d['Y'][K+li]]; dist,ii=TREE.query(q,k=300,distance_upper_bound=3000); ii=np.minimum(ii,len(PW)-1); dist=np.where((PW[ii]==w)|~np.isfinite(dist),np.inf,dist); df['dn']=np.minimum(dist.min(1),3000)
    V=df[cf+['S05','C300','C300abs']].values; df['mean_v']=df[cf].mean(axis=1); df['sd_v']=df[cf].std(axis=1); df['sd_all']=V.std(1); df['gap_S']=df['mean_v']-df['stk']; df['mean_all']=V.mean(1)
    ex={}
    for k,v in r.items():
        if k in skip: continue
        a=np.asarray(v)
        if a.ndim==1 and len(a)==m and a.dtype.kind in 'fi': ex['ds_'+k]=a.astype(float)
    df=pd.concat([df,pd.DataFrame(ex)],axis=1); df['ds_yA']=r['yA']; rows.append(df)
A=pd.concat(rows,ignore_index=True); y=A.y.values; print(A.shape,'extra ds6 features:',sum(c.startswith('ds_') for c in A.columns))
cen=np.array([[W[NAMES[w]]['X'].mean(),W[NAMES[w]]['Y'].mean()] for w in A.w.unique()]); km=KMeans(8,n_init=5,random_state=0).fit(cen); A['blk']=A.w.map(dict(zip(A.w.unique(),km.labels_)))
base=.8*A.mean_v.values+.2*A.stk.values; rm=lambda p:np.sqrt(np.mean((p-y)**2)); A.to_pickle('comb2_frame.pkl')
f1=cf+['S05','C300','C300abs','stk','mean_v','sd_v','sd_all','gap_S','mean_all','t','frac','n','dn']; f2=f1+[c for c in A.columns if c.startswith('ds_')]
def cv(cols,groups,seeds=(0,1,2),depth=4,leaf=100,it=250):
    pr=np.zeros(len(A))
    for tr,te in GroupKFold(5).split(A,y,groups):
        for s in seeds:
            m=H(max_iter=it,learning_rate=0.04,max_depth=depth,min_samples_leaf=leaf,l2_regularization=5.0,max_features=0.5,random_state=s).fit(A.iloc[tr][cols].values,(y-base)[tr]); pr[te]+=(m.predict(A.iloc[te][cols].values)+base[te])/len(seeds)
    return pr
print('baseline %.2f'%rm(base))
for nm,cols in [('original features',f1),('+ ds6 stack features',f2)]:
    print('%-22s well-grouped %.2f | SPATIAL %.2f'%(nm,rm(cv(cols,A.w.values)),rm(cv(cols,A.blk.values))),flush=True)
p=cv(f2,A.blk.values,depth=3,leaf=200,it=300); print('+ds6, shallower/more regularised, spatial: %.2f'%rm(p))
A['pred_sp']=cv(f2,A.blk.values); A.to_pickle('comb2_out.pkl')
