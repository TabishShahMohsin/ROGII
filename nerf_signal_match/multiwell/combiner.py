import pickle,glob,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.linear_model import RidgeCV,LinearRegression
from sklearn.model_selection import GroupKFold
from scipy.optimize import nnls
from common import W,NAMES,TREE,PW
cf=['base','g30k','g300k','shift']; D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
C={k:{o['w']:o for f in glob.glob(f'vmC_{k}_*.pkl') for o in pickle.load(open(f,'rb'))} for k in ['S25_300','S25_300abs']}; S05={o['w']:o for f in glob.glob('vmF_S05*.pkl') for o in pickle.load(open(f,'rb'))}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
GR=25; rows=[]
for w in sorted(D['base']):
    d=W[NAMES[w]]; K=d['K']; y=D['base'][w]['y']; n=len(y); t0=d['TVT_input'][K-1]; r=R[w]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]])
    P={c:D[c][w]['p'] for c in cf}; P['S05']=S05[w]['p']; P['C300']=C['S25_300'][w]['p']; P['C300abs']=C['S25_300abs'][w]['p']; P['stk']=S
    ix=np.arange(0,n,GR); q=np.c_[d['X'][K+ix],d['Y'][K+ix]]
    dist,ii=TREE.query(q,k=300,distance_upper_bound=3000); ii=np.minimum(ii,len(PW)-1); dist=np.where((PW[ii]==w)|~np.isfinite(dist),np.inf,dist); dn=np.minimum(dist.min(1),3000)
    df=pd.DataFrame({k:v[ix]-t0 for k,v in P.items()}); df['w']=w; df['y']=y[ix]-t0; df['t']=ix; df['frac']=ix/n; df['n']=n; df['dn']=dn
    V=df[cf+['S05','C300','C300abs']].values; df['mean_v']=df[cf].mean(1); df['sd_v']=df[cf].std(1); df['sd_all']=V.std(1); df['gap_S']=df['mean_v']-df['stk']; df['mean_all']=V.mean(1)
    df['sd_prefixstep']=np.abs(df['mean_v']).values  # magnitude of predicted displacement
    rows.append(df)
A=pd.concat(rows,ignore_index=True); print(A.shape,A.w.nunique())
y=A.y.values; g=A.w.values; rm=lambda p:np.sqrt(np.mean((p-y)**2))
print('bag4 %.2f | 0.8bag+0.2S %.2f | x1.1 %.2f'%(rm(A.mean_v),rm(.8*A.mean_v+.2*A['stk']),rm(1.1*(.8*A.mean_v+.2*A['stk']))))
base_cols=cf+['S05','C300','C300abs','stk']; feat_all=base_cols+['mean_v','sd_v','sd_all','gap_S','mean_all','t','frac','n','dn']
def cv(make,cols,seeds=1,resid=None):
    pr=np.zeros(len(A))
    for tr,te in GroupKFold(5).split(A,y,g):
        m=make().fit(A.iloc[tr][cols].values,(y-(resid if resid is not None else 0))[tr] if resid is None else (y-resid)[tr]); pr[te]=m.predict(A.iloc[te][cols].values)+(0 if resid is None else resid[te])
    return pr
def nnls_cv(cols):
    pr=np.zeros(len(A))
    for tr,te in GroupKFold(5).split(A,y,g):
        Xt=A.iloc[tr][cols].values; w_,_=nnls(Xt,y[tr]); pr[te]=A.iloc[te][cols].values@w_
    return pr
p=nnls_cv(base_cols); print('NNLS linear stack of the 8 estimators (grouped CV): %.2f'%rm(p))
base=.8*A.mean_v.values+.2*A['stk'].values
for nm,mk in [('HGB shallow',lambda:H(max_iter=150,learning_rate=0.04,max_depth=3,min_samples_leaf=200,l2_regularization=5.0,random_state=0)),('HGB deeper',lambda:H(max_iter=250,learning_rate=0.04,max_depth=4,min_samples_leaf=100,l2_regularization=5.0,random_state=0))]:
    p1=cv(mk,feat_all); p2=cv(mk,feat_all,resid=base); print('%s on y: %.2f | on residual of 0.8bag+0.2S: %.2f'%(nm,rm(p1),rm(p2)))
    pr=.5*(p1+p2); print('   avg of the two: %.2f'%rm(pr))
p3=cv(lambda:H(max_iter=150,learning_rate=0.04,max_depth=3,min_samples_leaf=200,l2_regularization=5.0,random_state=0),['mean_v','stk','sd_v','sd_all','gap_S','t','frac','dn'],resid=base); print('HGB small feature set on residual: %.2f'%rm(p3))
A['pred']=cv(lambda:H(max_iter=150,learning_rate=0.04,max_depth=3,min_samples_leaf=200,l2_regularization=5.0,random_state=0),feat_all,resid=base); A.to_pickle('combiner_out.pkl')
# disagreement quintiles
A['q']=pd.qcut(A.sd_v,5,labels=False); print('RMSE of 0.8bag+0.2S by disagreement quintile:',A.assign(e=(base-y)**2).groupby('q').e.mean().pow(.5).round(1).tolist())
