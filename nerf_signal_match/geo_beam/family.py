# "Family" typewell: pool the lateral GR(true TVT) of training wells whose typewell matches ours.
# Idea from Shrey Gandhi's write-up (trick 14); own implementation. Leak-free under spatial CV:
# members are drawn only from the `allowed` well set (training folds).
import numpy as np
from common import W, NAMES, CEN

BW = 0.5


def binned(t, g):
    m = np.isfinite(t) & np.isfinite(g)
    if m.sum() < 5:
        return None
    k = np.round(t[m] / BW).astype(np.int64)
    o = np.argsort(k, kind='stable'); k = k[o]; v = g[m][o]
    u, st = np.unique(k, return_index=True)
    med = np.array([np.median(a) for a in np.split(v, st[1:])])
    return u * BW, med


FAM = {}
for w in NAMES:
    d = W[w]
    FAM[w] = (binned(d['tw_TVT'].astype(float), d['tw_GR'].astype(float)),
              binned(d['TVT'].astype(float), d['GR'].astype(float)))


def _corr(at, ag, bt, bg):
    lo, hi = max(at.min(), bt.min()), min(at.max(), bt.max())
    if hi - lo < 10:
        return -1.0
    g = np.arange(lo, hi + BW, BW)
    a = np.interp(g, at, ag); b = np.interp(g, bt, bg)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def _shift(mt, mg, rt, rg, maxlag=20):
    lo, hi = min(mt.min(), rt.min()), max(mt.max(), rt.max())
    g = np.arange(lo, hi + BW, BW)
    a0 = np.interp(g, mt, mg, left=np.nan, right=np.nan); r = np.interp(g, rt, rg, left=np.nan, right=np.nan)
    best, bl = -2, 0
    for lag in range(-maxlag, maxlag + 1):
        a = np.roll(a0, lag); m = np.isfinite(a) & np.isfinite(r)
        if m.sum() < 20:
            continue
        if a[m].std() < 1e-6 or r[m].std() < 1e-6:
            continue
        c = float(np.corrcoef(a[m], r[m])[0, 1])
        if c > best:
            best, bl = c, lag
    return bl * BW


def family_tw(d, allowed, xy, maxdist=12000., corr=0.9, minmem=3, blend=0.0, maxmem=40):
    """Return (tvt_grid, gr) family reference on the provided typewell's grid, or None."""
    st = d['tw_TVT'].astype(float); sg = d['tw_GR'].astype(float)
    mem = []
    for w in allowed:
        if (CEN[NAMES.index(w)] - xy) @ (CEN[NAMES.index(w)] - xy) > maxdist ** 2:
            continue
        mtw, mfull = FAM[w]
        if mtw is None or mfull is None:
            continue
        c = _corr(st, sg, mtw[0], mtw[1])
        if c > corr:
            mem.append((float((CEN[NAMES.index(w)] - xy) @ (CEN[NAMES.index(w)] - xy)), w))
    mem = [w for _, w in sorted(mem)[:maxmem]]
    if len(mem) < minmem:
        return None, 0
    Ts = []; Gs = []
    for w in mem:
        mt, mg = FAM[w][1]
        sh = _shift(mt, mg, st, sg)
        Ts.append(mt - sh); Gs.append(mg)
    cur = binned(np.concatenate(Ts), np.concatenate(Gs))
    if cur is None:
        return None, len(mem)
    ct, cg = cur
    out = sg.copy()
    have = np.isin(np.round(st / BW).astype(np.int64), np.round(ct / BW).astype(np.int64))
    out[have] = np.interp(st[have], ct, cg)
    if blend > 0:
        out = blend * sg + (1 - blend) * out
    return (st, out), len(mem)
