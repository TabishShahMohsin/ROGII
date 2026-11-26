"""Section 3.1 and 3.4: well lengths, missing GR, typewell families, spatial layout.   python -m analysis.wells"""
import collections
import numpy as np
from scipy.spatial import cKDTree
from .common import train_wells, fold_of, save


def families(W, N):
    """Wells whose typewells share more than half of their (TVT, GR) samples belong to one family (union-find)."""
    keys = {w: set(zip(np.round(W[w]["tw_TVT"], 2).tolist(), np.round(W[w]["tw_GR"], 2).tolist())) for w in N}
    index = collections.defaultdict(list)
    for w in N:
        for k in keys[w]:
            index[k].append(w)
    parent = {w: w for w in N}

    def find(w):
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w
    for w in N:
        cnt = collections.Counter(v for k in keys[w] for v in index[k] if v != w)
        for v, c in cnt.items():
            if c > 0.5 * min(len(keys[w]), len(keys[v])):
                parent[find(w)] = find(v)
    return {w: find(w) for w in N}


def main():
    W = train_wells()
    N = sorted(W)
    F = fold_of(N)
    fo = np.array([F[w] for w in N])
    n = np.array([len(W[w]["MD"]) for w in N]); K = np.array([W[w]["K"] for w in N]); npred = n - K
    print("rows per well q10/50/90", np.quantile(n, [.1, .5, .9]), "known", np.quantile(K, [.1, .5, .9]),
          "predicted", np.quantile(npred, [.1, .5, .9]), "min/max predicted", npred.min(), npred.max())
    print("predicted share of the well q10/50/90", np.quantile(npred / n, [.1, .5, .9]).round(3), "total predicted rows", npred.sum())
    print("share of predicted rows in the longest 20% of wells", round(np.sort(npred)[::-1][:len(N) // 5].sum() / npred.sum(), 3))
    mp = [np.isnan(W[w]["GR"][W[w]["K"]:]).mean() for w in N]; mk = [np.isnan(W[w]["GR"][:W[w]["K"]]).mean() for w in N]
    print("missing lateral GR: prediction zone mean", round(np.mean(mp), 3), "q50/90", np.quantile(mp, [.5, .9]).round(3),
          "known section mean", round(np.mean(mk), 3), "; typewell GR missing", round(np.mean([np.isnan(W[w]["tw_GR"]).mean() for w in N]), 4))
    fam = families(W, N)
    c = collections.Counter(fam.values())
    print("typewell families", len(c), "single-well", sum(v == 1 for v in c.values()), "largest", max(c.values()))
    pts = {w: np.c_[W[w]["X"][::20], W[w]["Y"][::20]] for w in N}
    allp = np.vstack([pts[w] for w in N]); wid = np.concatenate([np.full(len(pts[w]), i) for i, w in enumerate(N)])
    tree = cKDTree(allp)

    def nearest(i, allowed):
        dd, ii = tree.query(pts[N[i]], k=200)
        out = []
        for r in range(len(dd)):
            ok = allowed[wid[ii[r]]] & (wid[ii[r]] != i)
            out.append(dd[r][ok][0] if ok.any() else 2e4)
        return np.median(out)
    nn_any = np.array([nearest(i, np.ones(len(N), bool)) for i in range(len(N))])
    nn_oth = np.array([nearest(i, fo != fo[i]) for i in range(len(N))])
    print("median lateral-to-nearest-well distance (ft) q10/50/90: any", np.quantile(nn_any, [.1, .5, .9]).round(0),
          "other fold", np.quantile(nn_oth, [.1, .5, .9]).round(0))
    xy = np.array([[np.median(W[w]["X"]), np.median(W[w]["Y"])] for w in N])
    print("field extent (miles)", (np.ptp(xy, 0) / 5280).round(1))
    save(dict(names=N, fold=fo, n=n, K=K, npred=npred, nn_any=nn_any, nn_oth=nn_oth, xy=xy), "wells.pkl")


if __name__ == "__main__":
    main()
