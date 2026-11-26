# Kaggle-side data loader (same fold protocol as local: sorted names, perm seed 42, %5)
import numpy as np, pandas as pd, glob, os, pickle
BASE='/kaggle/input/competitions/rogii-wellbore-geology-prediction'
if not os.path.exists(BASE): BASE=os.environ.get('ROGII_DATA','../data')
def _load(split):
    D={}
    for f in sorted(glob.glob(f'{BASE}/{split}/*__horizontal_well.csv')):
        w=os.path.basename(f).split('__')[0]; hw=pd.read_csv(f); tw=pd.read_csv(f'{BASE}/{split}/{w}__typewell.csv')
        d={c:hw[c].values.astype(float) for c in ['MD','X','Y','Z','GR','TVT_input']}
        if 'TVT' in hw: d['TVT']=hw['TVT'].values.astype(float)
        d['tw_TVT']=tw['TVT'].values.astype(float); d['tw_GR']=tw['GR'].values.astype(float)
        d['K']=int(np.flatnonzero(np.isfinite(d['TVT_input']))[-1])+1; D[w]=d
    return D
W=_load('train'); NAMES=sorted(W)
_perm=np.random.RandomState(42).permutation(len(NAMES))
FOLD=np.zeros(len(NAMES),int); FOLD[_perm]=np.arange(len(NAMES))%5
NBP=os.environ.get('NBP','/kaggle/working/nbf.pkl')
if not os.path.exists(NBP):
    import field3
    out={}
    for f in range(5):
        sup=[w for i,w in enumerate(NAMES) if FOLD[i]!=f]; F=field3.Field(W,sup)
        for i,w in enumerate(NAMES):
            if FOLD[i]==f: out[w]=F.path(W[w],robust=0.05)[0].astype(np.float32)
        print('field fold',f,flush=True)
    pickle.dump(out,open(NBP,'wb'))
