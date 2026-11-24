import numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
A=pd.read_pickle('comb3_out.pkl');F=pd.read_pickle('ladder_F.pkl');y=A.y.values;dZ=F.dZ.values;yS=y+dZ
rm=lambda e:np.sqrt(np.mean(e**2))
print('rows',len(y),'wells',A.w.nunique())
print('hold TVT   (pred TVT=t0)      : %.2f'%rm(y))
print('hold S=Z+TVT (pred S=S0)      : %.2f   <- baseline in S space'%rm(yS))
print('  S0 error is lateral-drift of structure; TVT-hold error includes the dZ term (std dZ %.1f)'%dZ.std())
# well-level breakdown
for nm,e in (('holdTVT',y),('holdS',yS)):
    s=pd.Series(e**2).groupby(A.w.values).sum();print(nm,'worst19 wells share of SSE %.0f%%'%(100*s.nlargest(19).sum()/s.sum()))
blk=A.blk.values
def cvfit(X,t,l2=0.0):
    p=np.zeros(len(t));coefs=[]
    for b in np.unique(blk):
        tr=blk!=b;c=np.linalg.lstsq(X[tr].T@X[tr]+l2*np.eye(X.shape[1]),X[tr].T@t[tr],rcond=None)[0];p[~tr]=X[~tr]@c;coefs.append(c)
    return p,np.mean(coefs,0)
dm=F.dMD.values
print('\n-- Step 1: add prefix-slope extrapolation  S = S0 + b*slope_W*dMD  (b by spatial-CV least squares)')
for Wn in (100,300,1000,3000):
    x=(F[f'sl{Wn}'].values*dm)[:,None];p,c=cvfit(x,yS);print(' W=%4d ft: b=%.2f  RMSE %.2f'%(Wn,c[0],rm(yS-p)))
print('-- Step 1b: slope of lateral Z (known geometry) as predictor')
x=(F.zsl.values*dm)[:,None];p,c=cvfit(x,yS);print(' b=%.2f  RMSE %.2f'%(c[0],rm(yS-p)))
x=np.c_[F.sl300*dm,F.sl3000*dm,F.zsl*dm];p,c=cvfit(x,yS);print(' sl300+sl3000+zsl: coefs',np.round(c,2),'RMSE %.2f'%rm(yS-p))
print('\n-- Step 2: add neighbour structure delta (nbd) ; shrink by weight')
for h in (1500,4000):
    nd=F[f'nbd{h}'].values;nw=F[f'nbw{h}'].values
    x=np.c_[nd];p,c=cvfit(x,yS);print(' h=%d nbd only: g=%.2f RMSE %.2f'%(h,c[0],rm(yS-p)))
    x=np.c_[nd,nd*(nw/(nw+0.3)),F.sl300*dm];p,c=cvfit(x,yS);print(' h=%d nbd, nbd*w, sl300: coefs %s RMSE %.2f'%(h,np.round(c,2),rm(yS-p)))
print('\n-- Step 3: add Viterbi ensemble (S-space pred = mean_v + dZ) as an estimate; blend with step-2 base')
mv=A.mean_v.values+dZ;stk=A.stk.values+dZ
print(' mean_v alone %.2f, stack alone %.2f, 0.8/0.2 %.2f'%(rm(yS-mv),rm(yS-stk),rm(yS-(.8*mv+.2*stk))))
nd=F['nbd4000'].values
for nm,x in (('mv',np.c_[mv]),('mv+stk',np.c_[mv,stk]),('mv+stk+nbd',np.c_[mv,stk,nd]),('mv+stk+nbd+sl300',np.c_[mv,stk,nd,F.sl300*dm]),('mv+stk+nbd+sl300+zsl',np.c_[mv,stk,nd,F.sl300*dm,F.zsl*dm])):
    p,c=cvfit(x,yS);print(' linear blend %-22s coefs %s RMSE %.2f'%(nm,np.round(c,2),rm(yS-p)))
print(' (combiner HGB from before: %.2f)'%rm(y-A.pred.values))
