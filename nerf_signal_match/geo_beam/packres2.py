# pack per-well pickles: TVT-like keys (in TV) as float16 delta vs tvt0, other arrays raw float16, scalars float32
import sys, os, glob, pickle, numpy as np
from common import W
TV={'v_map','v_pmean','v_pmean300','v_mean5'}
src,dst=sys.argv[1],sys.argv[2]; out={}
for f in sorted(glob.glob(f'{src}/*.pkl')):
    w=os.path.basename(f)[:-4]; o=pickle.load(open(f,'rb')); d=W[w]; t0=float(d['TVT_input'][d['K']-1])
    if o is None: continue
    for k,v in o.items():
        if isinstance(v,np.ndarray):
            out[f'{w}/{k}']=(np.clip(v.astype(np.float64)-t0,-6e4,6e4) if k in TV else np.clip(np.nan_to_num(v.astype(np.float64)),-6e4,6e4)).astype(np.float16)
        else: out[f'{w}/{k}']=np.array([v],np.float32)
np.savez_compressed(dst,**out); print(len(out),os.path.getsize(dst)/1e6)
