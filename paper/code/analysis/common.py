"""Shared loading for the analysis scripts (run from the code root: python -m analysis.<script>)."""
import os
import pickle
import numpy as np
from ssv.data import load_split, folds

RESULTS = os.environ.get("RESULTS", "results")
os.makedirs(RESULTS, exist_ok=True)
_cache = {}


def train_wells():
    if "W" not in _cache:
        p = os.path.join(RESULTS, "train_wells.pkl")
        if os.path.exists(p):
            _cache["W"] = pickle.load(open(p, "rb"))
        else:
            _cache["W"] = load_split("train")
            pickle.dump(_cache["W"], open(p, "wb"), protocol=4)
    return _cache["W"]


def fold_of(names):
    return dict(zip(sorted(names), folds(sorted(names))))


def save(obj, name):
    pickle.dump(obj, open(os.path.join(RESULTS, name), "wb"), protocol=4)


def load(name):
    return pickle.load(open(os.path.join(RESULTS, name), "rb"))
