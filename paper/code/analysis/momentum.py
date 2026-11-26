"""Section 4.3: how the interpreter's 80-ft moves relate to the neighbour field and to the previous move.
python -m analysis.momentum"""
import os
import numpy as np
from ssv import neighbours
from .common import train_wells, RESULTS

D = 80


def main():
    W = train_wells()
    NB = neighbours.train_cache(W, os.path.join(RESULTS, "nb_train.pkl"))
    X = []
    for w in sorted(W):
        d = W[w]; K = d["K"]; md = d["MD"][K - 1:]; Z = d["Z"][K - 1:]
        mz, cz = np.polyfit(md, Z, 1)
        zdet = (mz * md + cz) - Z
        y = d["TVT"][K - 1:] - zdet                                   # detrended (lattice) coordinate, truth
        f = np.r_[d["TVT_input"][K - 1], NB[w]["path"]] - zdet         # same for the neighbour field
        n = (len(y) - 1) // D
        mv, fm = np.diff(y[::D][:n + 1]), np.diff(f[::D][:n + 1])
        for k in range(1, len(mv)):
            X.append((mv[k], fm[k], mv[k - 1], fm[k - 1]))
    X = np.array(X)
    m, f, p, fp = X.T
    print(f"{len(m)} chunk moves; sd of the move {m.std():.2f} ft")

    def fit(A):
        A = np.c_[A, np.ones(len(m))]
        c = np.linalg.lstsq(A, m, rcond=None)[0]
        return np.sqrt(np.mean((m - A @ c) ** 2)), c
    for nm, A in [("zero move", np.zeros((len(m), 0))), ("field move", f[:, None]), ("previous move", p[:, None]),
                  ("field + previous", np.c_[f, p]), ("field + previous + previous field", np.c_[f, p, fp])]:
        r, c = fit(A)
        print(f"  predict the move from {nm:36s} RMSE {r:.2f} ft   coefficients {np.round(c, 3)}")
    r, c = fit(np.c_[f, p, fp])
    print(f"  => m_k ~ {c[0]:.2f} f_k + {c[1]:.2f} (m_k-1 - {-c[2] / max(c[1], 1e-9):.2f} f_k-1)")


if __name__ == "__main__":
    main()
