"""Section 3.3: GR misfit along the true path and the ambiguity of constant shifts.   python -m analysis.gr_signal"""
import numpy as np
import torch
from ssv.config import Config
from ssv.features import prep, to_torch, gr_stats, Lattice, true_path_misfit
from .common import train_wells, save


def main():
    W = train_wells()
    names = sorted(W)
    cfg = Config()
    gmu, gsd = gr_stats(W)
    print(f"GR global mean / std (API): {gmu:.2f} / {gsd:.2f}")
    lat = Lattice(cfg, "cpu")
    out = {}
    for seen in (True, False):
        cfg.seen = seen
        R, per = [], {}
        for w in names:
            d = W[w]
            nb = dict(path=np.zeros(len(d["MD"]) - d["K"], np.float32), dmed=np.zeros(len(d["MD"]), np.float32))
            p = to_torch(prep(w, d, nb, cfg, gmu, gsd), "cpu")
            R.append(true_path_misfit(p, lat, cfg.tw_step).numpy())
            lm = p["lm"] > 0
            g, y = p["g"][lm], p["y"][lm]

            def misfit(delta):
                fi = ((y + delta - p["tw0"]) / cfg.tw_step).clamp(0, len(p["twg"]) - 1.001)
                i0 = fi.floor().long()
                a = fi - i0
                return float(((g - (p["twg"][i0] * (1 - a) + p["twg"][i0 + 1] * a)) ** 2).mean())
            per[w] = dict(true=misfit(0.0), alt=min(misfit(x) for x in list(range(-30, -4)) + list(range(5, 31))),
                          var=float(g.var()), n=int(lm.sum()))
        R = np.concatenate(R) * gsd
        a = np.array([[v["true"], v["alt"], v["var"], v["n"]] for v in per.values()])
        ratio = a[:, 1] / np.maximum(a[:, 0], 1e-9)
        print(f"reference = typewell{' + own known section' if seen else ''}:")
        print(f"  |g - t| at the truth (API) q50 {np.quantile(np.abs(R), .5):.2f} q90 {np.quantile(np.abs(R), .9):.2f} "
              f"q99 {np.quantile(np.abs(R), .99):.2f} rms {np.sqrt(np.mean(R ** 2)):.2f}")
        print(f"  share of GR variance unexplained at the truth: {(a[:, 0] * a[:, 3]).sum() / (a[:, 2] * a[:, 3]).sum():.3f}")
        print(f"  wells where a shift of 5-30 ft fits within 30% of the truth: {np.mean(ratio < 1.3):.3f}; better than the truth: {np.mean(ratio < 1):.3f}")
        out[seen] = per
    save(out, "gr_signal.pkl")


if __name__ == "__main__":
    main()
