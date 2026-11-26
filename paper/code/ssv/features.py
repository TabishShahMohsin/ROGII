"""Per-well preprocessing and the inputs of every candidate segment.

Coordinates. Let s be the first lattice row (chosen so that the chunk grid ends exactly at the last row) and
Z_lin the least-squares line of Z(MD) over rows >= s. The lattice coordinate of a TVT value is

    S = TVT - off,   off(row) = TVT(s) + z_det(row) - z_det(s),   z_det = Z_lin - Z.

S = 0 at row s; S constant means "layers parallel to the trajectory trend". Every candidate segment is a straight
line in S between two anchors (integers in [-KC, KC]) at consecutive chunk boundaries.

Raw inputs (before standardization, Sec. 4.2):
  sequence channels at NPE points along the extended segment line
    0 g      lateral GR (low-pass + box smoothed), global standard score
    1 t      reference GR at the TVT implied by the line, same standard score
    2 g - t  misfit
    3 dnb    S(line) - S(neighbour field)   [ft]
    4 inc    1 on the current chunk, 0 on the context chunks
    5 nan    1 where the lateral GR was missing (interpolated)
    6 dnear  median distance to the nearest labelled wells [kft]
  scalars per segment
    0 slope           move / chunk  [ft/ft]
    1 slope - field   minus the neighbour-field slope over the chunk
    2 offset          mean of dnb over the current chunk [ft]
    3 h               clipped squared misfit, mean over the current chunk (the hand-crafted cost)
    4 h_window        same over the whole window
    5 nan fraction    over the current chunk
    6 u               MD distance from the last known row [kft]
"""
import math
import numpy as np
import torch
from scipy.signal import butter, filtfilt

NCH, NSC = 7, 7
SEQ_STD = [0, 1, 2, 3, 6]          # standardized sequence channels (4, 5 are indicators)
SC_STD = [0, 1, 2, 3, 4, 6]        # standardized scalars (5 is a fraction)
NB_SEQ, NB_SC = [3, 6], [1, 2]     # neighbour-derived inputs (zeroed when the neighbour inputs are off)
_b, _a = butter(1, 0.4 / 0.5, btype="low")


def _interp_nan(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    i = np.arange(len(x))
    if m.sum() > 1:
        return np.interp(i, i[m], x[m]), ~m
    return np.zeros(len(x)), ~m


def lowpass(x):
    return filtfilt(_b, _a, _interp_nan(x)[0])


def box(x, k):
    c = np.r_[0, np.cumsum(x)]
    i = np.arange(len(x))
    lo, hi = np.clip(i - k // 2, 0, len(x)), np.clip(i + k - k // 2, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def gr_stats(wells: dict) -> tuple:
    """Global mean and std of the low-pass lateral GR over the given (training) wells."""
    a = np.concatenate([lowpass(d["GR"]) for d in wells.values()])
    return float(a.mean()), float(a.std())


def seen_reference(d, tw_t, tw_g, blend, window):
    """Reference GR(TVT): typewell, blended with the well's own known-section GR binned by its known TVT."""
    K = d["K"]
    tvi, gr = d["TVT_input"][:K], d["GR"][:K]
    m = np.isfinite(tvi) & np.isfinite(gr) & (np.abs(tvi - tvi[K - 1]) <= window)
    if m.sum() < 20:
        return tw_g
    b = np.round(tvi[m] * 4) / 4
    u, inv = np.unique(b, return_inverse=True)
    s = np.bincount(inv, gr[m]) / np.bincount(inv)
    out = tw_g.copy()
    c = (tw_t >= u.min()) & (tw_t <= u.max())
    out[c] = blend * np.interp(tw_t[c], u, s) + (1 - blend) * tw_g[c]
    return out


def lattice_start(K, N, D):
    """First lattice row s and r = K - s known rows inside the lattice (chunk grid ends at the last row)."""
    npred = N - K
    r = (D - npred % D) % D or D
    r = min(r, K - 1)
    return K - r, r


def prep(name, d, nb, cfg, gmu, gsd):
    """Numpy arrays of one well on its lattice (rows s .. s + C*D, padded with the last value)."""
    D, MV = cfg.chunk, cfg.mv
    K, N = d["K"], len(d["MD"])
    s, r = lattice_start(K, N, D)
    C = int(math.ceil((N - s) / D))
    Lr = C * D + 1
    md, Z = d["MD"], d["Z"]
    mz, cz = np.polyfit(md[s:], Z[s:], 1)
    zdet = (mz * md + cz) - Z
    tvi = d["TVT_input"][:K].astype(float)
    off_all = tvi[s] + zdet - zdet[s]
    off = off_all[s:]
    # move of the known section over the last chunk before s (lattice units) -> conditions the first move
    j0 = max(0, s - D)
    p0 = ((tvi[s] - off_all[s]) - (tvi[j0] - off_all[j0])) * D / max(s - j0, 1) if s > 0 else 0.0
    p0 = float(np.clip(p0, -3 * MV, 3 * MV))
    g, isnan = _interp_nan(d["GR"])
    g = box(filtfilt(_b, _a, g), cfg.gr_box)[s:]
    isnan = isnan[s:].astype(np.float32)
    tw_t, tw_g = d["tw_TVT"].astype(float), d["tw_GR"].astype(float)
    mt = np.isfinite(tw_t)
    tw_t, tw_g = tw_t[mt], tw_g[mt]
    o = np.argsort(tw_t)
    tw_t, tw_g = tw_t[o], tw_g[o]
    tw_g = _interp_nan(tw_g)[0] if np.isfinite(tw_g).sum() > 1 else np.full(len(tw_t), gmu)
    if cfg.seen:
        tw_g = seen_reference(d, tw_t, tw_g, cfg.seen_blend, cfg.seen_window)
    grid = np.arange(tw_t[0] - 50, tw_t[-1] + 50, cfg.tw_step)
    twg = np.interp(grid, tw_t, tw_g)

    def pad(a, val=None):
        a = np.asarray(a, np.float32)
        return np.r_[a, np.full(Lr - len(a), a[-1] if val is None else val, np.float32)]

    # neighbour field path in lattice units (known rows use the known TVT)
    nbp = np.r_[tvi[s:K], np.asarray(nb["path"], float)] - off
    nbp = _interp_nan(nbp)[0]
    dnear = np.clip(np.asarray(nb["dmed"], float)[s:] / 1000.0, 0, 3)
    y = d["TVT"][s:].astype(np.float32) if "TVT" in d else np.full(N - s, np.nan, np.float32)
    lm = np.zeros(N - s, np.float32)
    lm[r:] = 1
    lm[~np.isfinite(y)] = 0
    cs = np.arange(C)
    i0 = np.minimum(s + cs * D, N - 1)
    u = ((md[i0] - md[K - 1]) / 1000.0).astype(np.float32)
    b0, b1 = np.minimum(cs * D, len(nbp) - 1), np.minimum(cs * D + D, len(nbp) - 1)
    mnb = (nbp[b1] - nbp[b0]).astype(np.float32)                      # field move per chunk (ft)
    dnc = dnear[np.minimum(cs * D, len(dnear) - 1)].astype(np.float32)
    return dict(w=name, C=C, r=r, s=s, K=K, N=N, npred=N - K, p0=np.float32(p0),
                g=pad((g - gmu) / gsd), nan=pad(isnan, 1.0), off=pad(off), snb=pad(nbp),
                y=pad(np.nan_to_num(y), 0.0), lm=pad(lm, 0.0), twg=((twg - gmu) / gsd).astype(np.float32),
                tw0=float(grid[0]), dnear=pad(dnear, 0.0), u=u, mnb=mnb, dnc=dnc)


def to_torch(p, device):
    return {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in p.items()}


class Lattice:
    """Constant tensors of the lattice for a config."""

    def __init__(self, cfg, device):
        self.D, self.KC, self.MV = cfg.chunk, cfg.kc, cfg.mv
        self.NA, self.NM = 2 * self.KC + 1, 2 * self.MV + 1
        self.NPE = cfg.pts_per_chunk * (cfg.ext_back + 1 + cfg.ext_fwd)
        self.AV = torch.arange(-self.KC, self.KC + 1, dtype=torch.float32, device=device)
        self.DV = torch.arange(-self.MV, self.MV + 1, dtype=torch.float32, device=device)
        self.REL = torch.linspace(-cfg.ext_back * self.D, (cfg.ext_fwd + 1) * self.D - 1, self.NPE, device=device)
        self.INC = ((self.REL >= 0) & (self.REL < self.D)).float()
        self.VALID = (self.AV[:, None] + self.DV[None, :]).abs() <= self.KC
        self.device = device


def raw_inputs(p, cs, lat: Lattice, clip: float, tw_step: float):
    """Raw inputs of all candidate segments of chunks cs: seq [n, NA, NM, NPE, NCH], scal [n, NA, NM, NSC]."""
    D, NA, NM = lat.D, lat.NA, lat.NM
    n = len(cs)
    Lr = len(p["g"])
    ridx = (cs[:, None] * D + lat.REL[None, :]).round().long().clamp(0, Lr - 1)
    S = (lat.AV[None, :, None, None] + lat.DV[None, None, :, None] * (lat.REL / D)[None, None, None, :])
    S = S.expand(n, -1, -1, -1)
    tv = S + p["off"][ridx][:, None, None, :]
    fi = torch.nan_to_num((tv - p["tw0"]) / tw_step, nan=0.0).clamp(0, len(p["twg"]) - 1.001)
    i0 = fi.floor().long()
    a = fi - i0
    t = p["twg"][i0] * (1 - a) + p["twg"][i0 + 1] * a

    def E(x):
        return x[ridx][:, None, None, :].expand_as(t)

    g, nn_ = E(p["g"]), E(p["nan"])
    dnb = S - p["snb"][ridx][:, None, None, :]
    inc = lat.INC[None, None, None, :].expand_as(t)
    seq = torch.stack([g, t, g - t, dnb, inc, nn_, E(p["dnear"])], -1)
    e2 = ((g - t) ** 2).clamp(max=clip)
    ninc = lat.INC.sum()
    h = (e2 * inc).sum(-1) / ninc
    hw = e2.mean(-1)
    m = (lat.DV / D)[None, None, :].expand(n, NA, -1)
    mf = (p["mnb"][cs] / D)[:, None, None].expand(-1, NA, NM)
    offn = (dnb * inc).sum(-1) / ninc
    nanf = (nn_ * inc).sum(-1) / ninc
    uc = p["u"][cs][:, None, None].expand(-1, NA, NM)
    scal = torch.stack([m, m - mf, offn, h, hw, nanf, uc], -1)
    return seq, scal


def true_path_misfit(p, lat: Lattice, tw_step: float):
    """GR misfit g - t along the TRUE path (prediction rows), for choosing the clip and for the GR analyses."""
    y, lm = p["y"], p["lm"] > 0
    fi = ((y - p["tw0"]) / tw_step).clamp(0, len(p["twg"]) - 1.001)
    i0 = fi.floor().long()
    a = fi - i0
    t = p["twg"][i0] * (1 - a) + p["twg"][i0 + 1] * a
    return (p["g"] - t)[lm]
