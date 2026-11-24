# cost volumes around the combiner reference: for each 16-row block and candidate offset d (ft), mean squared GR misfit vs type log at TVT=ref+d
import pickle,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from vit import W,NAMES,lp
from scipy.signal import butter,filtfilt
A=pd.read_pickle('comb3_out.pkl');BS=16;D=np.arange(-48,49,1.0)
def smooth(g,c):
    return filtfilt(*butter(1,c/0.5),g)
out={}
for w,g in A.groupby('w'):
    d=W[NAMES[w]];K=d['K'];n=len(d['MD'])-K;t0=float(d['TVT_input'][K-1]);y=d['TVT'][K:].astype(float)
    li=np.clip(g.t.values.astype(int),0,n-1);o=np.argsort(li);li=li[o];pr=g.pred.values[o]+t0
    # unique li
    li_u,ix=np.unique(li,return_index=True);ref=np.interp(np.arange(n),li_u,pr[ix])
    gr=lp(d['GR'][K:]);gs=smooth(gr,0.05);twg=d['tw_gr'];twt=d['tw_tvt'];tws=smooth(np.interp(np.arange(len(twt)),np.arange(len(twt)),twg),0.05) if False else twg
    B=(n+BS-1)//BS;C1=np.zeros((B,len(D)),np.float32);C2=np.zeros((B,len(D)),np.float32)
    tw_s=filtfilt(*butter(1,0.05/0.5),twg)
    for b in range(B):
        s,e=b*BS,min((b+1)*BS,n);hyp=ref[s:e,None]+D[None,:]
        C1[b]=((gr[s:e,None]-np.interp(hyp.ravel(),twt,twg).reshape(hyp.shape))**2).mean(0)
        C2[b]=((gs[s:e,None]-np.interp(hyp.ravel(),twt,tw_s).reshape(hyp.shape))**2).mean(0)
    off=np.array([np.mean(y[b*BS:(b+1)*BS]-ref[b*BS:(b+1)*BS]) for b in range(B)],np.float32)
    # reference slope & other per-block context
    ctx=np.stack([np.array([ref[b*BS:(b+1)*BS].mean()-t0 for b in range(B)]),np.arange(B)/B,np.full(B,n/5000.),np.array([gr[b*BS:(b+1)*BS].mean() for b in range(B)])/100.],1).astype(np.float32)
    out[w]=dict(C1=C1,C2=C2,off=off,ctx=ctx,n=n,ref=ref.astype(np.float32),y=y.astype(np.float32),blk=int(g.blk.iloc[0]))
pickle.dump(out,open('cvol.pkl','wb'))
e=np.concatenate([(r['y']-r['ref']) for r in out.values()]);print('reference RMSE over full-res rows %.2f'%np.sqrt(np.mean(e**2)),'| frac |err|>48: %.3f'%np.mean(np.abs(e)>48))
# quick signal check: block argmin cost offset vs truth
num=den=0;rows=[]
for r in out.values():
    am=D[r['C1'].argmin(1)];soft=(lambda c:(np.exp(-(c-c.min(1,keepdims=True))/20.)*D).sum(1)/np.exp(-(c-c.min(1,keepdims=True))/20.).sum(1))(r['C1'])
    rows.append((r['off'],am,soft))
o=np.concatenate([a[0] for a in rows]);am=np.concatenate([a[1] for a in rows]);so=np.concatenate([a[2] for a in rows])
print('block-level offset truth rms %.2f | argmin-cost err rms %.2f | softmin err %.2f | corr argmin/truth %.2f'%(np.sqrt(np.mean(o**2)),np.sqrt(np.mean((o-am)**2)),np.sqrt(np.mean((o-so)**2)),np.corrcoef(o,am)[0,1]))
