"""Section 3.5: between parallel neighbouring wells, shapes transfer and levels do not.
For every pair of wells overlapping >= 1.5 kft within 400 ft, the difference of their surfaces S = TVT + Z along the
target's prediction zone is split into a constant offset, a linear drift and a residual.   python -m analysis.levels"""
import numpy as np
from scipy.spatial import cKDTree
from .common import train_wells
from .wells import families


def main():
    W = train_wells()
    N = sorted(W)
    TR = {w: cKDTree(np.c_[W[w]["X"], W[w]["Y"]]) for w in N}
    MID = np.array([[np.median(W[w]["X"]), np.median(W[w]["Y"])] for w in N])
    mt = cKDTree(MID)
    fam = families(W, N)
    acc = {True: np.zeros(4), False: np.zeros(4)}     # total, constant, drift, residual
    per = {True: [], False: []}
    for i, w in enumerate(N):
        d = W[w]; K = d["K"]; ev = np.arange(K, len(d["MD"]), 10)
        xy = np.c_[d["X"][ev], d["Y"][ev]]; S = (d["TVT"] + d["Z"])[ev]; s = d["MD"][ev] - d["MD"][K]
        for j in mt.query_ball_point(MID[i], 6000):
            if j == i:
                continue
            v = N[j]; dd, ii = TR[v].query(xy); m = dd < 400
            if m.sum() < 100 or np.ptp(s[m]) < 1500:
                continue
            delta = S[m] - (W[v]["TVT"] + W[v]["Z"])[ii[m]]
            A = np.c_[np.ones(m.sum()), s[m] / 1000]
            c = np.linalg.lstsq(A, delta, rcond=None)[0]
            r = delta - A @ c
            sib = fam[w] == fam[v]
            t, k, e = (delta ** 2).sum(), len(delta) * delta.mean() ** 2, (r ** 2).sum()
            acc[sib] += [t, k, t - k - e, e]
            per[sib].append((abs(delta.mean()), abs(c[1]), np.sqrt(np.mean(r ** 2))))
    for sib in (True, False):
        a, p = acc[sib], np.array(per[sib])
        print(f"{'same typewell family' if sib else 'different families'}: {len(p)} ordered pairs; share of the squared "
              f"surface difference: constant {a[1] / a[0]:.3f}, linear drift {a[2] / a[0]:.3f}, residual {a[3] / a[0]:.3f}; "
              f"median |offset| {np.median(p[:, 0]):.1f} ft, |drift| {np.median(p[:, 1]):.2f} ft/kft, residual rms {np.median(p[:, 2]):.2f} ft")


if __name__ == "__main__":
    main()
