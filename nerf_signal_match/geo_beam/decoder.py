# Geo-space explicit-duration segmental beam decoder for ROGII.
# Own implementation of the ideas described in Shrey Gandhi's public write-ups
# ("Where is the drill bit in the rock?", Kaggle 2026): geo = TVT + Z latent, piecewise-linear
# segments with log-normal durations and AR(1) slopes, Cauchy GR-level + GR-gradient emission,
# heel GR splice into the typewell, drill-follow expanding funnel. No code copied.
import numpy as np
from numba import njit

S_GRID = np.array([-0.22, -0.15, -0.10, -0.07, -0.05, -0.035, -0.02, -0.008, 0.0,
                   0.008, 0.02, 0.035, 0.05, 0.07, 0.10, 0.15, 0.22])
S_TIGHT = np.array([-0.08, -0.06, -0.045, -0.035, -0.025, -0.018, -0.012, -0.006, 0.0,
                    0.006, 0.012, 0.018, 0.025, 0.035, 0.045, 0.06, 0.08])
DURS = np.array([75, 105, 165, 225, 330, 495])
STRIDE = 15


@njit(cache=True)
def _twv(t, t0, inv, g):
    # uniform-grid linear interpolation of typewell GR at TVT t (clamped)
    x = (t - t0) * inv
    n = g.shape[0]
    if x <= 0.0:
        return g[0]
    if x >= n - 1:
        return g[n - 1]
    i = int(x)
    f = x - i
    return g[i] * (1.0 - f) + g[i + 1] * f


@njit(cache=True)
def _decode(md, z, gr, w, t0, dtw, twg, lpb, lps, S, durs, stride, beam,
            geo0, s0idx, gs, gsg, wl, wg, gwin, wprior, wdrift, dbase, dgrow, cen, ncc_w):
    N = md.shape[0]
    ns = S.shape[0]
    nd = durs.shape[0]
    P = (N + stride - 1) // stride + 1          # positions 0, stride, ..., N
    posr = np.empty(P, np.int64)
    for p in range(P - 1):
        posr[p] = p * stride
    posr[P - 1] = N
    sc = np.full((P, beam), -1e300)
    gg = np.zeros((P, beam))
    si = np.zeros((P, beam), np.int64)
    cnt = np.zeros(P, np.int64)
    bpp = np.full((P, beam), -1, np.int64)
    bsl = np.full((P, beam), -1, np.int64)
    sc[0, 0] = 0.0; gg[0, 0] = geo0; si[0, 0] = s0idx; cnt[0] = 1
    inv = 1.0 / dtw
    maxL = durs[nd - 1]
    ebuf = np.zeros((maxL + 1, ns))
    cum = np.zeros(ns)
    cumd = np.zeros((nd, ns))
    oo = np.zeros(ns); ee = np.zeros(ns); oe = np.zeros(ns); e1 = np.zeros(ns); o1 = 0.0
    for pp in range(P - 1):
        R = posr[pp]
        if cnt[pp] == 0:
            continue
        Tmax = min(R + maxL, N)
        # end positions per duration
        eps = np.empty(nd, np.int64)
        for d in range(nd):
            T = R + durs[d]
            if T >= N:
                eps[d] = P - 1
            else:
                eps[d] = (T + stride - 1) // stride  # snap up to grid
                if eps[d] >= P - 1:
                    eps[d] = P - 1
        for b in range(cnt[pp]):
            g0 = gg[pp, b]; ps = si[pp, b]; s_ = sc[pp, b]
            for j in range(ns):
                cum[j] = 0.0; oo[j] = 0.0; ee[j] = 0.0; oe[j] = 0.0; e1[j] = 0.0
            o1 = 0.0
            d = 0
            for r in range(R, Tmax + 1):
                # record cumulative cost at segment ends (end row exclusive = r)
                while d < nd and (posr[eps[d]] == r):
                    for j in range(ns):
                        c = cum[j]
                        if ncc_w > 0.0:
                            n_ = r - R
                            if n_ >= 40:
                                mo = o1 / n_; me = e1[j] / n_
                                cv = oe[j] - n_ * mo * me
                                vo = oo[j] - n_ * mo * mo; ve = ee[j] - n_ * me * me
                                rr = cv / np.sqrt(max(vo, 1e-9) * max(ve, 1e-9))
                                if rr < 0.3:
                                    c += ncc_w * (0.3 - rr) * np.sqrt(n_)
                        cumd[d, j] = c
                    d += 1
                if d >= nd or r >= N:
                    break
                dm = md[r] - md[R]
                for j in range(ns):
                    t = g0 + S[j] * dm - z[r]
                    e = _twv(t, t0, inv, twg)
                    ebuf[r - R, j] = e
                    res = (gr[r] - e) / gs
                    c = wl * 0.5 * np.log1p(res * res) * w[r]
                    k = r - R - gwin
                    if wg > 0.0 and k >= 0:
                        rg = ((gr[r] - gr[r - gwin]) - (e - ebuf[k, j])) / gsg
                        c += wg * 0.5 * np.log1p(rg * rg) * min(w[r], w[r - gwin])
                    cum[j] += c
                    if ncc_w > 0.0:
                        oe[j] += gr[r] * e; ee[j] += e * e; e1[j] += e
                if ncc_w > 0.0:
                    o1 += gr[r]
                    for j in range(ns):
                        oo[j] += gr[r] * gr[r]
            # extend
            seen_last = False
            for d in range(nd):
                ep = eps[d]
                if ep == P - 1:
                    if seen_last:
                        continue
                    seen_last = True
                elif d > 0 and eps[d] == eps[d - 1]:
                    continue
                Te = posr[ep]
                mde = (md[Te] if Te < N else md[N - 1] + (md[N - 1] - md[N - 2])) - md[R]
                dist = (md[Te - 1] - md[0]) if Te >= 1 else 0.0
                allowed = dbase + dgrow * dist
                cz = cen[Te - 1]
                for j in range(ns):
                    gn = g0 + S[j] * mde
                    tot = s_ - cumd[d, j] + wprior * (lpb[d] + lps[d, ps, j])
                    if wdrift > 0.0:
                        ex = abs(gn - cz) - allowed
                        if ex > 0.0:
                            tot -= wdrift * 0.5 * (ex / allowed) ** 2
                    # insert into beam at ep
                    n = cnt[ep]
                    if n == beam and tot <= sc[ep, beam - 1]:
                        continue
                    if n < beam:
                        q = n; cnt[ep] = n + 1
                    else:
                        q = beam - 1
                    while q > 0 and tot > sc[ep, q - 1]:
                        sc[ep, q] = sc[ep, q - 1]; gg[ep, q] = gg[ep, q - 1]; si[ep, q] = si[ep, q - 1]
                        bpp[ep, q] = bpp[ep, q - 1]; bsl[ep, q] = bsl[ep, q - 1]
                        q -= 1
                    sc[ep, q] = tot; gg[ep, q] = gn; si[ep, q] = j; bpp[ep, q] = pp; bsl[ep, q] = b
    # backtrack all final hypotheses
    nf = cnt[P - 1]
    paths = np.zeros((nf, N))
    fs = np.zeros(nf)
    for f in range(nf):
        fs[f] = sc[P - 1, f]
        cp = P - 1; cb = f
        while cp > 0:
            pp = bpp[cp, cb]; pb = bsl[cp, cb]
            R = posr[pp]; Te = posr[cp]
            ga = gg[pp, pb]; gb = gg[cp, cb]
            mde = (md[Te] if Te < N else md[N - 1] + (md[N - 1] - md[N - 2])) - md[R]
            for r in range(R, min(Te, N)):
                paths[f, r] = ga + (gb - ga) * (md[r] - md[R]) / max(mde, 1e-9) - z[r]
            cp = pp; cb = pb
    return paths, fs


def prep(d, tw_blend=0.5, znorm=False, tw_override=None, gr_mask=False, dip=0.0):
    """Build decoder inputs for well dict d (keys MD,Z,GR,TVT_input,K,tw_TVT,tw_GR)."""
    K = d['K']
    md = d['MD'].astype(np.float64); z = d['Z'].astype(np.float64)
    gr_all = d['GR'].astype(np.float64)
    ti = d['TVT_input'].astype(np.float64)
    if tw_override is not None:
        twt, twg = tw_override
    else:
        twt = d['tw_TVT'].astype(np.float64); twg = d['tw_GR'].astype(np.float64)
    twt = np.asarray(twt, np.float64); twg = np.asarray(twg, np.float64).copy()
    kt = ti[:K]; kg = gr_all[:K]; mk = np.isfinite(kt) & np.isfinite(kg)
    dtw = float(np.median(np.diff(twt)))
    # emission scale from heel agreement (before splice)
    if mk.sum() >= 10:
        gs = float(np.std(kg[mk] - np.interp(kt[mk], twt, twg)))
    else:
        gs = 30.0
    # heel splice: blend the well's own heel GR into the typewell bins it covers
    if tw_blend > 0 and mk.sum() >= 10:
        kb = np.round((kt[mk] - twt[0]) / dtw).astype(np.int64)
        ok = (kb >= 0) & (kb < len(twt))
        if ok.any():
            kb = kb[ok]; vv = kg[mk][ok]
            order = np.argsort(kb, kind='stable'); kb = kb[order]; vv = vv[order]
            ub, st = np.unique(kb, return_index=True)
            med = np.array([np.median(a) for a in np.split(vv, st[1:])])
            twg[ub] = tw_blend * med + (1 - tw_blend) * twg[ub]
    gr = gr_all[K:].copy()
    w = np.isfinite(gr).astype(np.float64) if gr_mask else np.ones(len(gr))
    # interpolate missing GR along MD
    idx = np.arange(len(gr_all)); fin = np.isfinite(gr_all)
    gfill = np.interp(idx, idx[fin], gr_all[fin]) if fin.any() else np.full(len(gr_all), np.mean(twg))
    gr = gfill[K:]
    if znorm:
        gr = (gr - gr.mean()) / (gr.std() + 1e-9) * 18 + 90
        twg = (twg - twg.mean()) / (twg.std() + 1e-9) * 18 + 90
        gs = None  # caller sets
    tvt0 = float(ti[K - 1])
    zz = z[K:].copy()
    if dip != 0.0:
        zz = zz - dip * (md[K:] - md[K - 1])
    geo0 = tvt0 + zz[0]
    # known geo trend slope (tail 40% of heel)
    kk = max(5, int(0.4 * K))
    gk = ti[K - kk:K] + z[K - kk:K]
    mkk = np.isfinite(gk)
    a_k = float(np.polyfit(md[K - kk:K][mkk], gk[mkk], 1)[0]) if mkk.sum() >= 5 else 0.0
    return dict(md=md[K:], z=zz, gr=gr, w=w, t0=float(twt[0]), dtw=dtw, twg=twg, gs=gs,
                geo0=geo0, a_k=a_k, tvt0=tvt0, dipadj=(dip * (md[K:] - md[K - 1])))


DEF = dict(wl=1.8, wg=0.45, gwin=4, wprior=2.955, wdrift=7.0, dbase=5.0, dgrow=0.008, follow=0.7,
           gs_min=19.785, gs_max=47.3, gsg_min=2.0, gsg_max=60.0, beam=20, stride=STRIDE, ncc_w=0.0,
           s0='zero', temp=0.0)


def decode(pr, lpb, lps, S=S_GRID, **kw):
    o = dict(DEF); o.update(kw)
    md, z, gr = pr['md'], pr['z'], pr['gr']
    N = len(md)
    gs = pr['gs'] if pr['gs'] is not None else 25.0
    gs = float(np.clip(gs, o['gs_min'], o['gs_max']))
    gw = o['gwin']
    gsg = float(np.clip(np.std(gr[gw:] - gr[:-gw]), o['gsg_min'], o['gsg_max'])) if N > gw + 1 else 1.0
    dist = md - md[0]
    cen = pr['geo0'] + (1 - o['follow']) * pr['a_k'] * dist + o['follow'] * (z - z[0])
    s0 = int(np.argmin(np.abs(S - (pr['a_k'] if o['s0'] == 'ak' else 0.0))))
    if N < 3:
        return np.full(N, pr['tvt0']), None, None
    paths, fs = _decode(md, z, gr, pr['w'], pr['t0'], pr['dtw'], pr['twg'], lpb, lps, np.asarray(S, float),
                        DURS.astype(np.int64), int(o['stride']), int(o['beam']), pr['geo0'], s0, gs, gsg,
                        o['wl'], o['wg'], gw, o['wprior'], o['wdrift'], o['dbase'], o['dgrow'], cen, o['ncc_w'])
    top = paths[0]
    if o['temp'] > 0:
        a = (fs - fs.max()) / o['temp']; wt = np.exp(a); wt /= wt.sum()
        soft = (wt[:, None] * paths).sum(0)
    else:
        soft = top
    return top, soft, (paths, fs)


def emission(pr, tvt, gs=None):
    """mean Cauchy GR misfit of a TVT path (used by follow-gate)."""
    e = np.interp(tvt, pr['t0'] + pr['dtw'] * np.arange(len(pr['twg'])), pr['twg'])
    g = gs if gs is not None else float(np.clip(pr['gs'] if pr['gs'] is not None else 25, 19.785, 47.3))
    return float(np.mean(0.5 * np.log1p(((pr['gr'] - e) / g) ** 2)))
