"""Section 3.2: the interpreted surface S = TVT + Z over the evaluation zone (rows without TVT_input), its straight
pieces, their slopes with and without detrending by Z, and faults (Table 2, Fig. 3).   python -m analysis.structure

Method. TVT and Z are stored to 0.01 ft, so S carries a rounding error of at most +-0.01 ft and consecutive
differences of S stagger by up to 0.02 ft; slopes and faults are therefore never taken from consecutive rows.
The evaluation zone is cut into the fewest pieces such that a least-squares line fits every row of a piece within
TOL = 0.015 ft (just above the rounding bound; greedy longest-feasible pieces with a galloping search). Each piece
gets its least-squares slope. At every junction the two neighbouring lines are evaluated midway between the last row
of one piece and the first row of the next: their difference is the offset at the junction (a fault when |offset| >
FAULT) and the difference of their slopes is the change of dip. Each junction is counted once.
"""
import numpy as np
from multiprocessing import Pool
from .common import train_wells, save

TOL = 0.015     # ft
FAULT = 0.1     # ft; continuous junctions show |offset| <= ~0.03 ft (rounding), see the printed distribution


def _fit(x, S, i, j):
    """Least-squares line on rows i..j (inclusive): slope, intercept, max |residual|."""
    xx, yy = x[i:j + 1] - x[i], S[i:j + 1]
    if j == i:
        return 0.0, yy[0], 0.0
    b, a = np.polyfit(xx, yy, 1)
    return b, a - b * x[i], float(np.abs(yy - (a + b * xx)).max())


def pieces(x, S, tol=TOL):
    """Partition rows into the longest consecutive pieces that a line fits within tol (rows i..j inclusive)."""
    n, i, out = len(S), 0, []
    while i < n:
        if i == n - 1:
            out.append((i, i))
            break
        step, j = 1, i + 1
        while j + step < n and _fit(x, S, i, j + step)[2] <= tol:
            j += step
            step *= 2
        lo, hi = j, min(n - 1, j + step)
        if _fit(x, S, i, hi)[2] <= tol:
            lo = hi
        else:
            while hi - lo > 1:
                m = (lo + hi) // 2
                if _fit(x, S, i, m)[2] <= tol:
                    lo = m
                else:
                    hi = m
        out.append((i, lo))
        i = lo + 1
    return out


def one(item):
    w, d = item
    K = d["K"]
    x = d["MD"][K:] - d["MD"][K]
    S = (d["TVT"] + d["Z"])[K:]
    mz = np.polyfit(x, d["Z"][K:], 1)[0]
    P = []
    for i, j in pieces(x, S):
        b, a, r = _fit(x, S, i, j)
        P.append((i, j, x[j] - x[i] + 1, b, a, r))       # length in ft = rows covered (1 ft per row)
    P = np.array(P)
    J = []
    for p, q in zip(P[:-1], P[1:]):
        xm = (x[int(p[1])] + x[int(q[0])]) / 2
        J.append(((q[4] + q[3] * xm) - (p[4] + p[3] * xm), q[3] - p[3], xm))
    return w, dict(pieces=P, junctions=np.array(J).reshape(-1, 3), mz=mz, K=K, n=len(S))


def main():
    W = train_wells()
    with Pool(2) as p:
        R = dict(p.map(one, sorted(W.items()), chunksize=8))
    save(R, "structure.pkl")
    N = sorted(R)
    P = np.concatenate([np.c_[R[w]["pieces"], np.full(len(R[w]["pieces"]), R[w]["mz"])] for w in N])
    L, m = P[:, 2], P[:, 3]
    md_ = m - P[:, 6]
    print(f"evaluation zones: {sum(R[w]['n'] for w in N)} rows; {len(P)} pieces (tolerance {TOL} ft); "
          f"pieces per well median {np.median([len(R[w]['pieces']) for w in N]):.0f}; length q10/50/90 {np.quantile(L, [.1, .5, .9]).round(0)} ft")
    for lo, hi in [(0, 10), (10, 40), (40, 80), (80, 200), (200, 500), (500, np.inf)]:
        k = (L >= lo) & (L < hi)
        print(f"  length [{lo},{hi}): n={k.sum()} share of length={L[k].sum() / L.sum():.3f} |slope| q50/90/99 raw "
              f"{np.quantile(np.abs(m[k]), [.5, .9, .99]).round(4)} detrended {np.quantile(np.abs(md_[k]), [.5, .9, .99]).round(4)}")
    for nm, x in [("raw", m), ("detrended", md_)]:
        print(f"  {nm}: share of length with |slope| <= 0.02 / 0.03 / 0.05 / 0.10:",
              [round(float((L * (np.abs(x) <= t)).sum() / L.sum()), 4) for t in (0.02, 0.03, 0.05, 0.10)])
    print("  Z-trend slope q05/q50/q95:", np.quantile([R[w]["mz"] for w in N], [.05, .5, .95]).round(4))
    J = np.concatenate([R[w]["junctions"] for w in N])
    off, dslope = np.abs(J[:, 0]), np.abs(J[:, 1])
    print(f"  junctions {len(J)}: |offset| q50/90/99 {np.quantile(off, [.5, .9, .99]).round(3)} ft; "
          f"|offset| histogram (0,.02,.05,.1,.2,.5,1,2,5,inf): {np.histogram(off, [0, .02, .05, .1, .2, .5, 1, 2, 5, np.inf])[0].tolist()}")
    print(f"  change of slope at junctions |dslope| q50/90/99 {np.quantile(dslope, [.5, .9, .99]).round(4)}")
    f = off > FAULT
    nf = [int((np.abs(R[w]["junctions"][:, 0]) > FAULT).sum()) for w in N]
    print(f"  faults (|offset| > {FAULT} ft): {f.sum()} junctions in {np.mean(np.array(nf) > 0):.3f} of the wells; "
          f"size q50/90/99 {np.quantile(off[f], [.5, .9, .99]).round(2)} ft; per 1000 ft {f.sum() / L.sum() * 1000:.3f}")
    for th in (0.5, 1, 2, 5, 10):
        print(f"  wells with a fault > {th} ft: {np.mean([np.any(np.abs(R[w]['junctions'][:, 0]) > th) for w in N]):.3f}")


if __name__ == "__main__":
    main()
