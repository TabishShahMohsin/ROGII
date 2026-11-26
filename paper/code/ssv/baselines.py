"""Reference predictors evaluated with the same protocol as the model (prediction zone of every training well).

    python -m ssv.baselines --out baselines.pkl [--hand]   (--hand adds the hand-crafted-cost Viterbi; slower)
"""
import argparse
import math
import pickle
import numpy as np
import torch
from .config import Config
from .data import load_split, folds
from .features import prep, to_torch, gr_stats, lattice_start, Lattice
from .model import SSVNet, fit_stats, decode, prediction_rows
from . import neighbours


def hold_tvt(d):
    """Last known TVT, held constant."""
    return np.full(len(d["MD"]) - d["K"], d["TVT_input"][d["K"] - 1])


def hold_s_trend(d):
    """Layers parallel to the trajectory trend: TVT = TVT_K + z_det - z_det(K), z_det = Z_lin - Z."""
    K, md, Z = d["K"], d["MD"], d["Z"]
    mz, cz = np.polyfit(md[K:], Z[K:], 1)
    zdet = (mz * md + cz) - Z
    return d["TVT_input"][K - 1] + zdet[K:] - zdet[K - 1]


def hold_s_flat(d):
    """Flat layers: S = TVT + Z constant."""
    K = d["K"]
    return d["TVT_input"][K - 1] - (d["Z"][K:] - d["Z"][K - 1])


def lattice_oracle(d, cfg):
    """Best lattice path given the truth (min squared error): the resolution limit of the lattice."""
    D, KC, MV = cfg.chunk, cfg.kc, cfg.mv
    K, N = d["K"], len(d["MD"])
    s, r = lattice_start(K, N, D)
    md, Z = d["MD"], d["Z"]
    mz, cz = np.polyfit(md[s:], Z[s:], 1)
    zd = (mz * md + cz) - Z
    off = d["TVT_input"][s] + zd[s:] - zd[s]
    y = np.r_[d["TVT_input"][s:K], d["TVT"][K:]] - off
    n = len(y)
    C = math.ceil((n - 1) / D)
    y = np.r_[y, np.full(C * D + 1 - n, y[-1])]
    wt = np.zeros(len(y)); wt[r:n] = 1
    A = np.arange(-KC, KC + 1, dtype=float)[:, None]
    M = np.arange(-MV, MV + 1, dtype=float)[None, :]
    INF = 1e18
    al = np.full(2 * KC + 1, INF); al[KC] = 0
    bps = []
    for c in range(C):
        t = np.arange(D + (c == C - 1)); rows = c * D + t; tau = t / D; w = wt[rows]; yy = y[rows]
        s0, s1, s2 = w.sum(), (w * tau).sum(), (w * tau * tau).sum()
        y0, y1, y2 = (w * yy).sum(), (w * tau * yy).sum(), (w * yy * yy).sum()
        cost = A * A * s0 + 2 * A * M * s1 + M * M * s2 - 2 * A * y0 - 2 * M * y1 + y2
        tot = al[:, None] + cost
        new = np.full(2 * KC + 1, INF); arg = np.zeros(2 * KC + 1, int)
        for k in range(2 * MV + 1):
            src = np.arange(2 * KC + 1); dst = src + k - MV; ok = (dst >= 0) & (dst <= 2 * KC)
            v = tot[src[ok], k]; b = v < new[dst[ok]]
            new[dst[ok][b]] = v[b]; arg[dst[ok][b]] = src[ok][b]
        al = new; bps.append(arg)
    k = int(al.argmin()); anc = [k]
    for c in range(C - 1, -1, -1):
        k = bps[c][k]; anc.append(k)
    anc = np.array(anc[::-1], float) - KC
    f = np.arange(D) / D
    path = np.r_[np.concatenate([anc[i] + (anc[i + 1] - anc[i]) * f for i in range(C)]), anc[-1]][:n]
    return (path + off)[r:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="baselines.pkl")
    ap.add_argument("--nb-cache", default="nb_train.pkl")
    ap.add_argument("--data", default=None)
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--hand-folds", default="0,1,2,3,4", help="folds for the hand-cost Viterbi (saved one file per fold)")
    a = ap.parse_args(argv)
    cfg = Config()
    W = load_split("train", a.data)
    NB = neighbours.train_cache(W, a.nb_cache)
    names = sorted(W)
    f = folds(names)
    res = {}
    for w in names:
        d = W[w]
        res[w] = dict(hold_tvt=hold_tvt(d), hold_s_flat=hold_s_flat(d), hold_s_trend=hold_s_trend(d),
                      field=NB[w]["path"].astype(float), oracle=lattice_oracle(d, cfg))
    if a.hand:   # Viterbi with the clipped squared GR misfit as the only cost (first order, untrained network)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        hc = Config(momentum=False)
        import os
        for k in [int(v) for v in a.hand_folds.split(",")]:
            part = a.out.replace(".pkl", f"_hand_f{k}.pkl")
            if os.path.exists(part):
                for w, v in pickle.load(open(part, "rb")).items():
                    res[w]["hand_viterbi"] = v
                print("hand-cost Viterbi fold", k, "loaded", flush=True)
                continue
            tr = [w for w, fk in zip(names, f) if fk != k]
            va = [w for w, fk in zip(names, f) if fk == k]
            gmu, gsd = gr_stats({w: W[w] for w in tr})
            lat = Lattice(hc, dev)
            Ptr = [to_torch(prep(w, W[w], NB[w], hc, gmu, gsd), dev) for w in tr[::4]]
            st = fit_stats(Ptr, lat, hc)
            net = SSVNet(hc, st).to(dev).eval()
            with torch.no_grad():
                for i in range(0, len(va), 4):
                    ps = [to_torch(prep(w, W[w], NB[w], hc, gmu, gsd), dev) for w in va[i:i + 4]]
                    for t, p in zip(decode(net, ps, lat, hard=True), ps):
                        res[p["w"]]["hand_viterbi"] = prediction_rows(t, p).cpu().numpy()
            pickle.dump({w: res[w]["hand_viterbi"] for w in va}, open(part, "wb"))
            print("hand-cost Viterbi fold", k, flush=True)
    pickle.dump(res, open(a.out, "wb"))
    from .evaluate import pooled_rmse
    for key in res[names[0]]:
        have = [w for w in names if key in res[w]]
        print(f"{key:14s} {pooled_rmse({w: res[w] for w in have}, W, key):.3f}  ({len(have)} wells)")


if __name__ == "__main__":
    main()
