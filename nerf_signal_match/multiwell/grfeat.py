import pickle, numpy as np, time
from common import W,NAMES
from hmm import field_fn, calib
R=pickle.load(open('ds3.pkl','rb'))
DEL=np.array([-20,-12,-6,-3,0,3,6,12,20.]); WINS=[150,500,1500]
def wmean(x,ok,idx,win):
    c=np.r_[0,np.cumsum(np.where(ok,x,0.))]; n=np.r_[0,np.cumsum(ok.astype(float))]; L=len(x)
    lo=np.clip(idx-win//2,0,L); hi=np.clip(idx+win//2+1,0,L); nn=n[hi]-n[lo]; return (c[hi]-c[lo])/np.maximum(nn,1),nn/np.maximum(hi-lo,1)
def feats(r):
    d=W[NAMES[r['w']]]; K=d['K']; md=d['MD']; lat=np.arange(K,len(md)); gi=r['idx']-K     # grid positions within lateral
    pr=r['tvt_end']+np.interp(lat,r['idx'],r['yA']); gr=d['GR'][lat].astype(np.float64); ok=~np.isnan(gr)
    out={}; f=field_fn(d['tw_tvt'],d['tw_gr'],2.0); a,b,sg,c=calib(d,K,f,1500); out['calc']=np.full(len(gi),c); out['cala']=np.full(len(gi),a)
    if ok.sum()<50 or abs(a)<1e-6:
        for W_ in WINS:
            for k in range(len(DEL)): out[f'cost{W_}_{k}']=np.zeros(len(gi))
            out[f'dstar{W_}']=np.zeros(len(gi)); out[f'vf{W_}']=np.zeros(len(gi))
        return out
    cost=[]; 
    for dl in DEL:
        res=(gr-(a*f(pr+dl)+b))/sg; cost.append(np.where(ok,np.minimum(res**2,25.),0.))
    cost=np.array(cost)                                          # [ND,L]
    for W_ in WINS:
        cw=np.array([wmean(cost[k],ok,gi,W_)[0] for k in range(len(DEL))]); vf=wmean(np.zeros(len(gr)),ok,gi,W_)[1]
        base=cw[DEL.tolist().index(0.0)]
        for k in range(len(DEL)): out[f'cost{W_}_{k}']=cw[k]-base
        out[f'dstar{W_}']=DEL[np.argmin(cw,0)]; out[f'vf{W_}']=vf
    return out
if __name__=="__main__":
    t0=time.time()
    for i,r in enumerate(R):
        r.update(feats(r))
        if i%200==0: print(i,round(time.time()-t0),flush=True)
    pickle.dump(R,open('ds4.pkl','wb')); print("done",round(time.time()-t0))
