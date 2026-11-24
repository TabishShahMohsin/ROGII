# run the user's Viterbi-with-internals (vint) on one well, keep compact outputs
import numpy as np, time
from vint import vint
from vit import NAMES
KEEP=['map','pmean','pmean300','post_sd30','post_sd300','far30','far300','nmode30','spread','rng5','fit_rms','lcorr']
def job(w):
    wi=NAMES.index(w)
    try: r=vint(wi)
    except Exception as e: r=None
    if r is None: return w,None
    o={('v_'+k):np.asarray(r[k],np.float32) for k in KEEP}
    o['v_mean5']=r['P'].mean(0).astype(np.float32)
    o['v_gap12_em']=float(r['gap12']); o['v_cpr_em']=float(r['cost_per_row'])
    return w,o
if __name__=='__main__':
    import sys; from vit import W
    t=time.time(); w,o=job(NAMES[3]); d=W[w]; y=d['TVT'][d['K']:]
    print(time.time()-t,{k:float(np.sqrt(np.mean((o[k]-y)**2))) for k in ['v_map','v_pmean','v_pmean300','v_mean5']})
