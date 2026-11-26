"""Metrics: row-pooled RMSE (the competition metric), per-well RMSE, error concentration, error along the lateral.

    python -m ssv.evaluate runs/f*s0/val.pkl            # out-of-fold summary of one seed
"""
import argparse
import glob
import math
import pickle
import numpy as np
from .data import load_split


def pooled_rmse(pred: dict, wells: dict, key=None):
    se = n = 0.0
    for w, t in pred.items():
        t = t[key] if key else t
        y = wells[w]["TVT"][wells[w]["K"]:]
        m = np.isfinite(y)
        se += float(((t - y)[m] ** 2).sum())
        n += m.sum()
    return math.sqrt(se / n)


def per_well(pred: dict, wells: dict, key=None):
    out = {}
    for w, t in pred.items():
        t = t[key] if key else t
        y = wells[w]["TVT"][wells[w]["K"]:]
        m = np.isfinite(y)
        out[w] = dict(rmse=float(np.sqrt(np.mean((t - y)[m] ** 2))), sse=float(((t - y)[m] ** 2).sum()), n=int(m.sum()))
    return out


def concentration(pw: dict, drops=(0, 1, 3, 5, 10, 20, 50)):
    """RMSE of the remaining wells after removing the k worst (by squared error), and the share removed."""
    sse = np.array(sorted((v["sse"] for v in pw.values()), reverse=True))
    order = sorted(pw.values(), key=lambda v: -v["sse"])
    n = np.array([v["n"] for v in order])
    tot = sse.sum()
    return [(k, math.sqrt(sse[k:].sum() / n[k:].sum()), sse[:k].sum() / tot) for k in drops]


def by_position(pred: dict, wells: dict, key=None, bins=10):
    """RMSE by decile of relative position in the prediction zone."""
    se, n = np.zeros(bins), np.zeros(bins)
    for w, t in pred.items():
        t = t[key] if key else t
        y = wells[w]["TVT"][wells[w]["K"]:]
        b = np.minimum((np.arange(len(y)) * bins) // len(y), bins - 1)
        m = np.isfinite(y)
        np.add.at(se, b[m], (t - y)[m] ** 2)
        np.add.at(n, b[m], 1)
    return np.sqrt(se / n)


def load_oof(paths):
    out = {}
    for p in paths:
        out.update(pickle.load(open(p, "rb")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("val", nargs="+", help="val.pkl files (one per fold)")
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    W = load_split("train", a.data)
    paths = [p for g in a.val for p in glob.glob(g)]
    oof = load_oof(paths)
    print(f"{len(oof)} wells from {len(paths)} files")
    for k in ("soft", "hard"):
        pw = per_well(oof, W, k)
        print(f"{k}: pooled RMSE {pooled_rmse(oof, W, k):.3f}")
        print("   worst removed / RMSE of rest / share of squared error:",
              [(c, round(r, 2), round(s, 2)) for c, r, s in concentration(pw)])
        print("   RMSE by position decile:", by_position(oof, W, k).round(2))


if __name__ == "__main__":
    main()
