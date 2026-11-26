"""Discussion: can the neighbour field's failure on a well be predicted at test time (a confidence gate)?   python -m analysis.field_confidence"""
# Can the neighbour field's error on a well be predicted at test time? Hindcast: start the field 1,000 rows before the
# end of the known section and compare it with the known TVT there; correlate with the field's error in the eval zone.
import sys, pickle, numpy as np

from scipy.stats import spearmanr
from ssv.data import load_split, folds
from ssv.field import Field
W = load_split('train'); N = sorted(W); F = folds(N)
from ssv import neighbours; from analysis.common import RESULTS; import os; NB = neighbours.train_cache(W, os.path.join(RESULTS, 'nb_train.pkl'))
rows = []
for k in range(5):
    sup = [w for w, f in zip(N, F) if f != k]; fld = Field(W, sup)
    for w, f in zip(N, F):
        if f != k: continue
        d = W[w]; K = d['K']; L = min(1000, K - 300)
        if L < 300: continue
        d2 = dict(d); d2['K'] = K - L; ti = d['TVT_input'].copy(); ti[K - L:] = np.nan; d2['TVT_input'] = ti
        hp = fld.path(d2, robust=0.05)[0][:L]                       # field over the last L known rows
        hind = np.sqrt(np.mean((hp - d['TVT_input'][K - L:K]) ** 2))
        y = d['TVT'][K:]; ev = np.sqrt(np.mean((NB[w]['path'] - y) ** 2))
        dist = float(np.median(NB[w]['dmed'][K:]))
        rows.append((w, hind, ev, dist))
    print('fold', k, flush=True)
a = np.array([r[1:] for r in rows])
print(len(a), 'wells')
print('Spearman(hindcast error, eval-zone field error) = %.2f' % spearmanr(a[:, 0], a[:, 1])[0])
print('Spearman(distance to support wells, eval-zone field error) = %.2f' % spearmanr(a[:, 2], a[:, 1])[0])
q = np.quantile(a[:, 0], [0.25, 0.5, 0.75])
for lo, hi in [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)]:
    m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    print(f'hindcast error in [{lo:.1f},{hi:.1f}): wells {m.sum()}, field eval RMSE median {np.median(a[m,1]):.1f}, share > 20 ft {np.mean(a[m,1] > 20):.2f}')
pickle.dump(rows, open(os.path.join(RESULTS, 'field_hindcast.pkl'), 'wb'))
