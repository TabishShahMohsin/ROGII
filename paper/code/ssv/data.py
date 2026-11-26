"""Reading the competition files and the cross-validation split."""
import glob
import os
import numpy as np
import pandas as pd

KAGGLE_DIR = "/kaggle/input/competitions/rogii-wellbore-geology-prediction"


def data_dir() -> str:
    """Competition folder (holds train/, test/, sample_submission.csv)."""
    if os.environ.get("ROGII_DATA"):
        return os.environ["ROGII_DATA"]
    if os.path.exists(KAGGLE_DIR):
        return KAGGLE_DIR
    for c in ["../data", "../../data", "data"]:
        if os.path.exists(os.path.join(c, "train")):
            return c
    raise FileNotFoundError("set ROGII_DATA to the competition folder")


def load_split(split: str, base: str = None) -> dict:
    """Return {well: dict of numpy arrays}. K = number of rows with known TVT_input (the known section)."""
    base = base or data_dir()
    wells = {}
    for f in sorted(glob.glob(os.path.join(base, split, "*__horizontal_well.csv"))):
        w = os.path.basename(f).split("__")[0]
        hw = pd.read_csv(f)
        tw = pd.read_csv(os.path.join(base, split, f"{w}__typewell.csv"))
        d = {c: hw[c].to_numpy(float) for c in ["MD", "X", "Y", "Z", "GR", "TVT_input"]}
        if "TVT" in hw:
            d["TVT"] = hw["TVT"].to_numpy(float)
        d["tw_TVT"] = tw["TVT"].to_numpy(float)
        d["tw_GR"] = tw["GR"].to_numpy(float)
        known = np.flatnonzero(np.isfinite(d["TVT_input"]))
        d["K"] = int(known[-1]) + 1
        wells[w] = d
    return wells


def folds(names, n_folds=5, seed=42) -> np.ndarray:
    """Random split by well: sorted names, fixed permutation, fold = position mod n_folds."""
    names = sorted(names)
    perm = np.random.RandomState(seed).permutation(len(names))
    f = np.zeros(len(names), int)
    f[perm] = np.arange(len(names)) % n_folds
    return f


def split_names(names, fold, n_folds=5, seed=42):
    names = sorted(names)
    if fold < 0:
        return names, []
    f = folds(names, n_folds, seed)
    return [w for w, k in zip(names, f) if k != fold], [w for w, k in zip(names, f) if k == fold]
