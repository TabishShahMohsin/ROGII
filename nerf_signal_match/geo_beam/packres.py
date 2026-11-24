# pack per-well result pickles -> compact npz (float16 deltas vs each well's tvt0), for transfer
import sys, os, glob, pickle, numpy as np
from common import W
src, dst = sys.argv[1], sys.argv[2]
out = {}
for f in sorted(glob.glob(f'{src}/*.pkl')):
    w = os.path.basename(f)[:-4]; o = pickle.load(open(f, 'rb')); d = W[w]
    t0 = float(d['TVT_input'][d['K'] - 1])
    for k, v in o.items():
        if isinstance(v, np.ndarray):
            out[f'{w}/{k}'] = np.clip(v.astype(np.float64) - t0, -6e4, 6e4).astype(np.float16)
        else:
            out[f'{w}/{k}'] = np.array([v], np.float32)
np.savez_compressed(dst, **out)
print(len(out), os.path.getsize(dst) / 1e6)
