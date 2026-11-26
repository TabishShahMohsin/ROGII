"""Second-order (momentum) segment-Viterbi on the anchor lattice: soft (forward-backward with soft-min) and hard.

State at chunk boundary c: (anchor a in [-KC..KC], move m in [-MV..MV] of the chunk that ends at c).
Path cost  = sum_c cost[c, a_c, m_c] + tau[c, m_{c-1}, m_c],   with tau[0] replaced by tau0[m_0]
             (tau0 conditions the first move on the move of the known section just before the lattice).
cost[b, c, a, m] is indexed by the START anchor a of chunk c and its move m; the segment ends at a + m.
Disallowed transitions carry the cost BIG.
"""
import torch
import torch.nn.functional as F

BIG = 1e4


def softmin(x, T, dim):
    return -T * torch.logsumexp(-x / T, dim)


def _shift_in(x, MV, NA):
    """x[B, NA, NM] indexed by (start anchor, move) -> y[B, NA, NM] indexed by (end anchor a+m, move)."""
    B, _, NM = x.shape
    xp = F.pad(x, (0, 0, MV, MV), value=BIG)
    return torch.stack([xp[:, 2 * MV - k:2 * MV - k + NA, k] for k in range(NM)], 2)


def _shift_out(y, MV, NA):
    """Inverse of _shift_in: (end anchor, move) -> (start anchor, move)."""
    B, _, NM = y.shape
    yp = F.pad(y, (0, 0, MV, MV), value=BIG)
    return torch.stack([yp[:, k:k + NA, k] for k in range(NM)], 2)


def forward_backward(cost, tau, tau0, T, KC, MV, return_marginals=False):
    """Expected anchor (posterior mean under exp(-cost/T)) at every boundary 0..C.

    cost [B, C, NA, NM], tau [B, C, NM, NM] (entry [c, p, m] used for c >= 1), tau0 [B, NM], T > 0.
    Returns S [B, C+1] (S[:, 0] = 0) and, optionally, the anchor marginals [B, C, NA].
    """
    B, C, NA, NM = cost.shape
    dev, dt = cost.device, cost.dtype
    AV = torch.arange(-KC, KC + 1, dtype=dt, device=dev)
    a0 = torch.full((B, NA), BIG, device=dev, dtype=dt)
    a0[:, KC] = 0
    al = [_shift_in(a0[:, :, None] + tau0[:, None, :] + cost[:, 0], MV, NA)]
    for c in range(1, C):
        bm = softmin(al[-1][:, :, :, None] + tau[:, c][:, None], T, 2)
        al.append(_shift_in(bm + cost[:, c], MV, NA))
    be = [torch.zeros(B, NA, NM, device=dev, dtype=dt)]
    for c in range(C - 1, 0, -1):
        g = _shift_out(be[0], MV, NA) + cost[:, c]
        be.insert(0, softmin(tau[:, c][:, None] + g[:, :, None, :], T, 3))
    S = [torch.zeros(B, device=dev, dtype=dt)]
    marg = []
    for c in range(C):
        p = torch.softmax((-(al[c] + be[c]) / T).reshape(B, -1), 1).reshape(B, NA, NM).sum(2)
        marg.append(p)
        S.append((p * AV).sum(1))
    S = torch.stack(S, 1)
    return (S, torch.stack(marg, 1)) if return_marginals else S


@torch.no_grad()
def viterbi(cost, tau, tau0, KC, MV):
    """Minimum-cost anchor path [B, C+1] (anchor values, S[:, 0] = 0)."""
    B, C, NA, NM = cost.shape
    dev = cost.device
    a0 = torch.full((B, NA), BIG, device=dev, dtype=cost.dtype)
    a0[:, KC] = 0
    al = _shift_in(a0[:, :, None] + tau0[:, None, :] + cost[:, 0], MV, NA)
    bps = []
    for c in range(1, C):
        v, ix = (al[:, :, :, None] + tau[:, c][:, None]).min(2)
        bps.append(ix)
        al = _shift_in(v + cost[:, c], MV, NA)
    flat = al.reshape(B, -1).argmin(1)
    a, m = flat // NM, flat % NM
    anc = [a]
    for c in range(C - 1, 0, -1):
        a = a - (m - MV)
        anc.append(a)
        m = bps[c - 1].gather(1, a[:, None, None].expand(-1, 1, NM))[:, 0].gather(1, m[:, None])[:, 0]
    anc.append(torch.full_like(a, KC))
    return torch.stack(anc[::-1], 1).to(cost.dtype) - KC


def brute_force(cost, tau, tau0, T, KC, MV):
    """Enumerate all move sequences (tiny lattices only). Returns (soft expectation, best path) per batch item."""
    import itertools
    import math
    B, C, NA, NM = cost.shape
    out = []
    for b in range(B):
        Z, E, best = 0.0, torch.zeros(C + 1, dtype=torch.float64), (float("inf"), None)
        for ms in itertools.product(range(NM), repeat=C):
            a, tot, path, ok = 0, float(tau0[b, ms[0]]), [0], True
            for c, m in enumerate(ms):
                if c > 0:
                    tot += float(tau[b, c, ms[c - 1], m])
                tot += float(cost[b, c, a + KC, m])
                a = a + m - MV
                path.append(a)
                ok &= abs(a) <= KC
            if not ok or tot >= BIG / 2:
                continue
            w = math.exp(-tot / T)
            Z += w
            E += w * torch.tensor(path, dtype=torch.float64)
            if tot < best[0]:
                best = (tot, path)
        out.append(((E / Z).float(), torch.tensor(best[1], dtype=torch.float32)))
    return out
