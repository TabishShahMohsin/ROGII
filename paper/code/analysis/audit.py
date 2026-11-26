"""Section 5: distribution of every model input over all candidate segments (fold-0 training wells), and the scale
used by the earlier version.   python -m analysis.audit"""
import json
import os
import numpy as np
from ssv.config import Config
from ssv.data import split_names
from ssv.features import prep, to_torch, gr_stats, Lattice
from ssv.model import fit_stats
from ssv import neighbours
from .common import train_wells, RESULTS

SEQ = ["g", "t", "g-t", "dnb [ft]", "inc", "nan", "dnear [kft]"]
SC = ["slope", "slope-field", "offset [ft]", "h", "h_window", "nan fraction", "u [kft]"]


def main():
    W = train_wells()
    NB = neighbours.train_cache(W, os.path.join(RESULTS, "nb_train.pkl"))
    cfg = Config()
    tr, _ = split_names(W, 0)
    gmu, gsd = gr_stats({w: W[w] for w in tr})
    lat = Lattice(cfg, "cpu")
    P = {w: to_torch(prep(w, W[w], NB[w], cfg, gmu, gsd), "cpu") for w in tr}
    st = fit_stats([P[w] for w in tr], lat, cfg)
    print(f"clip {st['clip']:.3f} (std units^2) = |g-t| <= {np.sqrt(st['clip']) * gsd:.1f} API; earlier clip 25 = {5 * gsd:.0f} API")
    for nm, m, s in zip(SEQ, st["seq_mu"], st["seq_sd"]):
        print(f"  sequence {nm:12s} mean {m:9.3f}  std {s:9.3f}")
    for nm, m, s in zip(SC, st["sc_mu"], st["sc_sd"]):
        print(f"  scalar   {nm:12s} mean {m:9.3f}  std {s:9.3f}")
    e = np.concatenate([((P[w]["y"] - P[w]["off"]) - P[w]["snb"])[P[w]["lm"] > 0].numpy() for w in tr])
    print(f"along the true path: S - S_field mean {e.mean():.2f} std {e.std():.2f} ft; earlier scaling /10 -> std {e.std() / 10:.2f} "
          f"(candidates {st['seq_sd'][3] / 10:.2f})")
    json.dump(dict(st, gr_mu=gmu, gr_sd=gsd), open(os.path.join(RESULTS, "stats_fold0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
