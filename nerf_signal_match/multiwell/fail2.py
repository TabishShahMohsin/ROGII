import pickle,numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import W,NAMES
df=pd.read_pickle('fail.pkl'); cf=['base','g30k','g300k','shift']
D={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}
R=pickle.load(open('ds6.pkl','rb')); pp=(pickle.load(open('oof_stackA.pkl','rb'))+pickle.load(open('oof_seqnn.pkl','rb')))/2; off=np.cumsum([0]+[len(r['y']) for r in R])
def info(w):
    d=W[NAMES[w]]; K=d['K']; md=d['MD'][K:]; Z=d['Z'][K:]; mz,cz=np.polyfit(md,Z,1); z=mz*md+cz-Z; y=d['TVT'][K:]; t0=d['TVT_input'][K-1]; A=y-z-t0
    n=len(y); L=170; sl=[(A[min(i+L,n)-1]-A[i])/(min(i+L,n)-1-i+1e-9) for i in range(0,n-L,L)]
    # also how well does the true path match GR vs the type log (min achievable GR SSE) vs chosen path
    from vit import lp
    gr=lp(d['GR'][K:]); tw=np.interp(y,d['tw_tvt'],d['tw_gr']); 
    return dict(maxslope=np.max(np.abs(sl)) if sl else 0,fracinf=np.mean(np.abs(sl)>0.05) if sl else 0,corr_true=np.corrcoef(gr,tw)[0,1],A=A,z=z,y=y,t0=t0,d=d,K=K)
rows=[]
for w in df.w:
    i=info(w); rows.append((w,i['maxslope'],i['fracinf'],i['corr_true']))
x=pd.DataFrame(rows,columns=['w','maxslope','fracinf','corr_true']).set_index('w'); df=df.set_index('w').join(x); df['rm']=np.sqrt(df.se_bag)
bad=df[df.rm>15]; ok=df[df.rm<=15]
print('bad vs ok: maxslope %.3f / %.3f | frac chunks |true slope|>0.05 %.3f / %.3f | true-path GR corr %.2f / %.2f'%(bad.maxslope.mean(),ok.maxslope.mean(),bad.fracinf.mean(),ok.fracinf.mean(),bad.corr_true.mean(),ok.corr_true.mean()))
print('bad: drift %.0f vs %.0f ; ystd %.0f vs %.0f'%(bad.drift.mean(),ok.drift.mean(),bad.ystd.mean(),ok.ystd.mean()))
print(bad[['name','n','rm','maxslope','fracinf','corr_true','drift']].round(2).to_string())
# how Viterbi vs stack vs oracle choose: stack better?
print('stack<bag in bad:',(bad.se_stack<bad.se_bag).sum(),'/',len(bad),' ; in ok:',(ok.se_stack<ok.se_bag).sum(),'/',len(ok))
fig,ax=plt.subplots(3,4,figsize=(22,12))
for a,w in zip(ax.ravel(),bad.index[:12]):
    i=info(w); d=i['d']; K=i['K']; n=len(i['y']); r=R[w]; S=r['tvt_end']+np.interp(np.arange(K,K+n),r['idx'],pp[off[w]:off[w+1]]); bag=np.mean([D[c][w]['p'] for c in cf],0)
    a.plot(i['y'],'k',lw=2,label='truth'); a.plot(bag,'r',label='Viterbi bag'); a.plot(S,'b',label='stack'); a.axhline(i['t0'],c='gray',ls=':'); a.set_title(f"{NAMES[w]} rmse bag {np.sqrt(df.se_bag[w]):.0f} stack {np.sqrt(df.se_stack[w]):.0f} corr {df.corr_true[w]:.2f}",fontsize=9)
ax[0,0].legend(); plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/failures.png',dpi=70)
df.to_pickle('fail2.pkl')
