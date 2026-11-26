"""Fold-safe neighbour inputs: the structure-field path and the distance to the nearest labelled wells.

For a training well the support set is the wells of the other folds; for a test well it is all training wells.
"""
import os
import pickle
import time
import numpy as np
from scipy.spatial import cKDTree
from .field import Field
from .data import folds


def build(targets: dict, support_wells: dict, every=20, k_near=16) -> dict:
    """targets/support_wells: {name: well dict}. Returns {name: {'path': TVT along the well, 'dmed': ft}}."""
    names = sorted(support_wells)
    fld = Field(support_wells, names)
    pts = np.vstack([np.c_[support_wells[w]["X"][::every], support_wells[w]["Y"][::every]] for w in names])
    tree = cKDTree(pts)
    out = {}
    for w, d in targets.items():
        dd, _ = tree.query(np.c_[d["X"], d["Y"]], k=k_near)
        out[w] = dict(path=fld.path(d, robust=0.05)[0].astype(np.float32),
                      dmed=np.median(dd, 1).astype(np.float32))
    return out


def train_cache(wells: dict, path: str, n_folds=5, seed=42) -> dict:
    """Neighbour inputs for every training well, each built from the other folds only (cached on disk)."""
    if path and os.path.exists(path):
        return pickle.load(open(path, "rb"))
    names = sorted(wells)
    f = folds(names, n_folds, seed)
    out, t0 = {}, time.time()
    for k in range(n_folds):
        sup = {w: wells[w] for w, fk in zip(names, f) if fk != k}
        tgt = {w: wells[w] for w, fk in zip(names, f) if fk == k}
        out.update(build(tgt, sup))
        print(f"neighbour cache fold {k}: {time.time() - t0:.0f}s", flush=True)
    if path:
        pickle.dump(out, open(path, "wb"))
    return out
