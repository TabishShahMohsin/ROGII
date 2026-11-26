"""Neighbour structure field: a map-view estimate of the gradient of the structural surface S = TVT + Z.

Observations are along-track dips dS/ds of labelled support wells, measured over SEG-ft pieces every SUB ft,
plus the query well's own known section (extra weight). At each query point the local gradient is a Gaussian-weighted
least-squares fit with an anisotropic distance (rotation `angle`, along-axis compression `ratio`) and a ridge pull
(`lam`) towards the regional gradient, followed by one Tukey reweighting pass (`robust`, ft/ft). Integrating the
gradient along the query trajectory gives the neighbour prediction of S, hence of TVT.
"""
import numpy as np
from scipy.spatial import cKDTree

SEG, SUB = 40, 20


def obs_from(x, y, s, rows, wid):
    i0 = rows[rows + SEG < len(x)]; i1 = i0 + SEG
    dx = x[i1] - x[i0]; dy = y[i1] - y[i0]; ds = np.hypot(dx, dy); da = s[i1] - s[i0]
    ok = (ds > 0.8 * SEG) & np.isfinite(da)
    return np.c_[(x[i0] + x[i1]) / 2, (y[i0] + y[i1]) / 2, dx / np.maximum(ds, 1e-9), dy / np.maximum(ds, 1e-9),
                 da / np.maximum(ds, 1e-9), np.full(len(i0), wid)][ok]


class Field:
    def __init__(self, W, names, angle=50.0, ratio=1.3):
        self.th = np.radians(angle); self.ratio = ratio; obs = []
        for i, w in enumerate(names):
            d = W[w]; x = d['X'].astype(float); y = d['Y'].astype(float); s = (d['TVT'] + d['Z']).astype(float)
            obs.append(obs_from(x, y, s, np.arange(0, len(x), SUB), i))
        self.O = np.vstack(obs); self.names = list(names)
        self.tree = cKDTree(self._m(self.O[:, :2]))
        self.g0 = np.linalg.lstsq(self.O[:, 2:4], self.O[:, 4], rcond=None)[0]

    def _m(self, xy):
        c, s = np.cos(self.th), np.sin(self.th)
        a = xy[:, 0] * c + xy[:, 1] * s; b = -xy[:, 0] * s + xy[:, 1] * c
        return np.c_[a / self.ratio, b]

    def path(self, d, h=900., k=128, lam=0.5, sib=None, sibw=0.2, pref=512, pref_w=3.0, idw=False, robust=0.0, step=20):
        K = d['K']; n = len(d['MD']); q = np.r_[np.arange(K - 1, n, step), n - 1]
        x = d['X'].astype(float); y = d['Y'].astype(float)
        xq = x[q]; yq = y[q]; mid = np.c_[(xq[1:] + xq[:-1]) / 2, (yq[1:] + yq[:-1]) / 2]
        kk = min(k, len(self.O)); dd, ii = self.tree.query(self._m(mid), k=kk)
        T = self.O[ii, 2:4]; g = self.O[ii, 4]
        wg = 1.0 / np.maximum(dd, 25.0) ** 2 if idw else np.exp(-0.5 * (dd / h) ** 2)
        if sib is not None and sibw != 1.0:
            wg = wg * np.where(np.isin(self.O[ii, 5].astype(int), sib), 1.0, sibw)
        if pref > 0:
            # own build-section dip observations of the query well (known TVT_input)
            s = (d['TVT_input'] + d['Z']).astype(float); r0 = max(0, K - pref)
            P = obs_from(x, y, s, np.arange(r0, K - SEG, SUB), -1)
            if len(P):
                dp = np.sqrt(((self._m(mid)[:, None, :] - self._m(P[:, :2])[None, :, :]) ** 2).sum(-1))
                wp = (1.0 / np.maximum(dp, 25.0) ** 2 if idw else np.exp(-0.5 * (dp / h) ** 2)) * pref_w
                T = np.concatenate([T, np.broadcast_to(P[None, :, 2:4], (len(mid), len(P), 2))], 1)
                g = np.concatenate([g, np.broadcast_to(P[None, :, 4], (len(mid), len(P)))], 1)
                wg = np.concatenate([wg, wp], 1)

        def solve(wg):
            Mx = np.einsum('qk,qki,qkj->qij', wg, T, T) + lam * np.eye(2)[None] * (wg.sum(1)[:, None, None] if idw else 1.0)
            b = np.einsum('qk,qki,qk->qi', wg, T, g) + lam * self.g0[None] * (wg.sum(1)[:, None] if idw else 1.0)
            return np.linalg.solve(Mx, b[..., None])[..., 0]
        Gr = solve(wg)
        if robust > 0:
            r = g - np.einsum('qki,qi->qk', T, Gr); u = np.clip(np.abs(r) / robust, 0, 1)
            Gr = solve(wg * (1 - u ** 2) ** 2 + 1e-12)
        dS = np.r_[0, np.cumsum(Gr[:, 0] * np.diff(xq) + Gr[:, 1] * np.diff(yq))]; z = d['Z'][q].astype(float)
        tv = float(d['TVT_input'][K - 1]) + dS - (z - z[0])
        return np.interp(np.arange(K, n), q, tv), float(np.median(dd[:, 0]))
