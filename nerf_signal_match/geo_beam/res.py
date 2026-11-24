# load decoder results from cloud pickles and/or packed device npz into R[w][variant] -> float64 TVT array
import glob, os, pickle, numpy as np
from common import W, NAMES


def load(pkls=(), npzs=(), dirs=()):
    R = {}
    for p in pkls:
        for w, o in pickle.load(open(p, 'rb')).items():
            R.setdefault(w, {}).update({k: (v.astype(np.float64) if isinstance(v, np.ndarray) else float(v)) for k, v in o.items()})
    for dd in dirs:
        for f in glob.glob(f'{dd}/*.pkl'):
            w = os.path.basename(f)[:-4]; o = pickle.load(open(f, 'rb'))
            R.setdefault(w, {}).update({k: (v.astype(np.float64) if isinstance(v, np.ndarray) else float(v)) for k, v in o.items()})
    for p in npzs:
        z = np.load(p)
        for key in z.files:
            w, k = key.split('/', 1); d = W[w]; t0 = float(d['TVT_input'][d['K'] - 1])
            v = z[key]
            R.setdefault(w, {})[k] = float(v[0]) if k.endswith('_em') else (v.astype(np.float64) + t0)
    return R
