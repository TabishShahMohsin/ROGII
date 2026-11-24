import pickle,numpy as np,pandas as pd
from common import W,NAMES
from scipy.spatial import cKDTree
cf=['base','g30k','g300k','shift']
D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
V={v['w']:v for v in pickle.load(open('vitP_unif_500_637.pkl','rb'))+pickle.load(open('vitP_unif_637_773.pkl','rb'))}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
# well centroid for isolation
cen=np.array([[W[n]['X'].mean(),W[n]['Y'].mean()] for n in NAMES]); kd=cKDTree(cen)
rows=[]
for w in sorted(D['base']):
    d=W[NAMES[w]]; K=d['K']; o=D['base'][w]; y=o['y']; n=o['n']; r=R[w]
    S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]]); bag=np.mean([D[c][w]['p'] for c in cf],0); t0=d['TVT_input'][K-1]
    gr=d['GR']; Z=d['Z'][K:]; md=d['MD'][K:]; mz=np.polyfit(md,Z,1)[0]
    P=V[w]['P']; 
    rows.append(dict(w=w,name=NAMES[w],n=n,K=K,se_bag=np.mean((bag-y)**2),se_top1=np.mean((o['p']-y)**2),se_stack=np.mean((S-y)**2),se_hold=np.mean((t0-y)**2),
      se_orac=min(np.mean((q-y)**2) for q in P),drift=np.abs(y-t0).max(),yend=y[-1]-t0,ystd=y.std(),grnan=np.isnan(gr[K:]).mean(),grstd=np.nanstd(gr[K:]),
      iso=kd.query(cen[w],2)[0][1],twlen=len(d['tw_tvt']),zrange=np.ptp(Z),slope=abs(mz),
      bagbias=np.mean(bag-y),endsl=(y[-1]-y[-200])/200 if n>300 else 0,
      tvtsat=np.mean(np.abs(np.diff(o['p'][::100]))/100>0.045) if n>300 else 0))
df=pd.DataFrame(rows); df['tot']=df.se_bag*df.n; df=df.sort_values('tot',ascending=False); df.to_pickle('fail.pkl')
T=df.tot.sum(); c=df.tot.cumsum()/T
print('wells',len(df),'pooled RMSE %.2f'%np.sqrt(T/df.n.sum()))
for q in [5,10,20,30]: print('top %d wells = %.0f%% of SE, %.0f%% of rows'%(q,100*c.iloc[q-1],100*df.n.iloc[:q].sum()/df.n.sum()))
print('median per-well RMSE %.2f, mean %.2f'%(np.median(np.sqrt(df.se_bag)),np.sqrt(df.se_bag).mean()))
print(df.head(15)[['name','n','se_bag','se_top1','se_stack','se_hold','se_orac','drift','yend','iso','grnan','twlen']].round(1).to_string())
for c_ in ['n','drift','ystd','grnan','grstd','iso','twlen','zrange','slope','endsl','tvtsat']: print(c_,'spearman vs RMSE: %.2f'%df[c_].corr(np.sqrt(df.se_bag),method='spearman'))
# bucketed
df['rm']=np.sqrt(df.se_bag)
for c_,bins in [('n',[0,1500,3000,5000,8000,1e5]),('drift',[0,10,25,50,100,1e4]),('iso',[0,500,1500,3000,6000,1e6])]:
    g=df.groupby(pd.cut(df[c_],bins)); print(c_); print(pd.DataFrame(dict(wells=g.size(),rmse=g.apply(lambda x:np.sqrt((x.se_bag*x.n).sum()/x.n.sum())),stack=g.apply(lambda x:np.sqrt((x.se_stack*x.n).sum()/x.n.sum())),orac=g.apply(lambda x:np.sqrt((x.se_orac*x.n).sum()/x.n.sum())),share=g.apply(lambda x:x.tot.sum()/T))).round(2).to_string())
# failure types for bad wells (rmse>15)
b=df[df.rm>15]; print('bad wells',len(b),'share of SE %.0f%%'%(100*b.tot.sum()/T),' oracle-fixable (orac<8):',(np.sqrt(b.se_orac)<8).sum(),' stack better than bag:',(b.se_stack<b.se_bag).sum(),' bias frac of MSE: %.2f'%np.mean(b.bagbias**2/b.se_bag))
