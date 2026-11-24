import pickle,glob,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HR
from sklearn.model_selection import GroupKFold
from vit import W,NAMES
CV=pickle.load(open('cvol.pkl','rb'));A=pd.read_pickle('comb3_out.pkl')
V={}
for f in glob.glob('vint_*.pkl'):
    for o in pickle.load(open(f,'rb')):V[o['w']]=o
L=250;rows=[];cache={}
for w,g in A.groupby('w'):
    o=V[w];d=W[NAMES[w]];K=d['K'];n=o['n'];t0=o['tvt0'];ref=CV[w]['ref'];y=o['y']
    li=g.t.values.astype(int);od=np.argsort(li);lu,ix=np.unique(li[od],return_index=True);stk=np.interp(np.arange(n),lu,g.stk.values[od][ix])+t0;bag=np.interp(np.arange(n),lu,g.mean_v.values[od][ix])+t0
    P=o['P'];nc=len(P);tw=np.interp(P,d['tw_tvt'],d['tw_gr']);gr=d['GR'][K:].astype(float);gr=np.where(np.isnan(gr),np.nanmean(gr),gr)
    tc=o['tc'];gap=(tc-tc[0])/(tc[0]+1e-9)
    for s in range(0,n,L):
        sl=slice(s,min(s+L,n));fr=[np.sqrt(np.mean((gr[sl]-tw[k][sl])**2)) for k in range(nc)];cr=[np.corrcoef(gr[sl],tw[k][sl])[0,1] if np.std(tw[k][sl])>1e-6 else 0 for k in range(nc)]
        dr=[np.mean(P[k][sl]-ref[sl]) for k in range(nc)];ds=[np.mean(P[k][sl]-stk[sl]) for k in range(nc)];db=[np.mean(P[k][sl]-bag[sl]) for k in range(nc)]
        for k in range(nc):
            rows.append(dict(w=w,blk=int(g.blk.iloc[0]),s=s,k=k,gap=gap[k],fit=fr[k],fit_rel=fr[k]-min(fr),fit_rank=int(np.argsort(np.argsort(fr))[k]),corr=cr[k],corr_rel=cr[k]-max(cr),
                d_ref=dr[k],ad_ref=abs(dr[k]),ref_rank=int(np.argsort(np.argsort(np.abs(dr)))[k]),d_stk=ds[k],ad_stk=abs(ds[k]),d_bag=db[k],ad_bag=abs(db[k]),
                rms_ref=np.sqrt(np.mean((P[k][sl]-ref[sl])**2)),slope=(P[k][sl][-1]-P[k][sl][0])/max(sl.stop-sl.start,1),off0=P[k][sl][0]-t0,frac=s/n,n=n,
                far=float(o['far300'][sl].mean()),psd=float(o['post_sd300'][sl].mean()),mse=float(np.mean((P[k][sl]-y[sl])**2)),L=sl.stop-sl.start))
        
D=pd.DataFrame(rows);D['lm']=np.log1p(D.mse);print(D.shape)
feats=['k','gap','fit','fit_rel','fit_rank','corr','corr_rel','d_ref','ad_ref','ref_rank','d_stk','ad_stk','d_bag','ad_bag','rms_ref','slope','off0','frac','n','far','psd']
pr=np.zeros(len(D))
for tr,te in GroupKFold(5).split(D,groups=D.blk):
    m=HR(max_iter=300,learning_rate=0.05,max_depth=5,min_samples_leaf=40,l2_regularization=3.0,random_state=0).fit(D.iloc[tr][feats],D.lm.values[tr]);pr[te]=m.predict(D.iloc[te][feats])
D['pr']=pr
# choose per chunk
def pooled(sel):
    s=D.loc[sel];return np.sqrt((s.mse*s.L).sum()/s.L.sum())
tot=D.groupby(['w','s']).L.first().sum()
print('MAP (k=0)             %.2f'%pooled(D.k==0))
sel=D.groupby(['w','s']).pr.idxmin();print('selector argmin       %.2f'%pooled(sel.values))
print('oracle                %.2f'%pooled(D.groupby(['w','s']).mse.idxmin().values))
print('closest-to-ref (comb) %.2f'%pooled(D.groupby(['w','s']).ad_ref.idxmin().values))
# softmin-weight average (MSE-weighted): needs actual paths; approximate via expected MSE = weights on candidate errors not additive -> skip
# how often selector picks non-MAP and is it right
pick=D.loc[sel.values];print('selector picks k>0 in %.0f%% of chunks; when it does, mean mse %.1f vs MAP mse on those chunks %.1f'%(100*(pick.k>0).mean(),pick[pick.k>0].mse.mean(),D[(D.k==0)&(D.set_index(['w','s']).index.isin(pick[pick.k>0].set_index(['w','s']).index))].mse.mean()))
pickle.dump((D,sel.values),open('sel2.pkl','wb'))
