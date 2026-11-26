"""Segment-cost network, transition (momentum) network, standardization, lattice cost assembly and decoding."""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .dp import BIG, forward_backward, viterbi
from .features import NCH, NSC, SEQ_STD, SC_STD, NB_SEQ, NB_SC, raw_inputs, true_path_misfit, Lattice

H_SCALAR = 3   # index of the hand-crafted misfit among the scalars


@torch.no_grad()
def fit_stats(ps, lat: Lattice, cfg, max_chunks=6, seed=0):
    """Standardization constants from training wells.

    clip: squared GR misfit is clipped at (99th percentile of |g - t| along the TRUE paths)^2.
    mean / std of every input over all valid candidate segments of up to `max_chunks` random chunks per well.
    """
    rs = np.random.RandomState(seed)
    r = torch.cat([true_path_misfit(p, lat, cfg.tw_step) for p in ps]).abs()
    q = torch.quantile(r[torch.randperm(len(r))[:2_000_000]], torch.tensor([0.5, 0.9, 0.99], device=r.device))
    clip = float(q[2]) ** 2 if cfg.misfit_clip < 0 else cfg.misfit_clip
    s1 = torch.zeros(NCH, dtype=torch.float64, device=lat.device)
    s2 = torch.zeros_like(s1)
    c1 = torch.zeros(NSC, dtype=torch.float64, device=lat.device)
    c2 = torch.zeros_like(c1)
    n_seq = n_sc = 0
    for p in ps:
        cs = torch.from_numpy(rs.choice(p["C"], min(max_chunks, p["C"]), replace=False)).to(lat.device)
        seq, scal = raw_inputs(p, cs, lat, clip, cfg.tw_step)
        v = lat.VALID[None].expand(len(cs), -1, -1)
        sq, sc = seq[v].double(), scal[v].double()
        s1 += sq.sum((0, 1)); s2 += (sq ** 2).sum((0, 1)); n_seq += sq.shape[0] * sq.shape[1]
        c1 += sc.sum(0); c2 += (sc ** 2).sum(0); n_sc += sc.shape[0]
    mu, sd = s1 / n_seq, (s2 / n_seq - (s1 / n_seq) ** 2).clamp_min(1e-12).sqrt()
    cmu, csd = c1 / n_sc, (c2 / n_sc - (c1 / n_sc) ** 2).clamp_min(1e-12).sqrt()
    if cfg.scaling == "legacy":     # earlier version: fixed scales, offset /10 ft, misfit clipped at 25 (5 GR std)
        mu, cmu = torch.zeros_like(mu), torch.zeros_like(cmu)
        sd = torch.tensor([1, 1, 1, 10, 1, 1, 1], dtype=torch.float64)
        csd = torch.tensor([1 / 20, 1 / 20, 20, 1, 1, 1, 5], dtype=torch.float64)
        clip = 25.0
    return dict(clip=clip, misfit_q50_q90_q99=[float(x) for x in q], seq_mu=mu.tolist(), seq_sd=sd.tolist(),
                sc_mu=cmu.tolist(), sc_sd=csd.tolist())


class SSVNet(nn.Module):
    def __init__(self, cfg, stats):
        super().__init__()
        self.cfg, self.clip = cfg, stats["clip"]
        self.MV = cfg.mv
        mu, sd = torch.zeros(NCH), torch.ones(NCH)
        mu[SEQ_STD] = torch.tensor(stats["seq_mu"], dtype=torch.float32)[SEQ_STD]
        sd[SEQ_STD] = torch.tensor(stats["seq_sd"], dtype=torch.float32)[SEQ_STD]
        cmu, csd = torch.zeros(NSC), torch.ones(NSC)
        cmu[SC_STD] = torch.tensor(stats["sc_mu"], dtype=torch.float32)[SC_STD]
        csd[SC_STD] = torch.tensor(stats["sc_sd"], dtype=torch.float32)[SC_STD]
        for k, v in dict(seq_mu=mu, seq_sd=sd, sc_mu=cmu, sc_sd=csd).items():
            self.register_buffer(k, v)
        keep_seq, keep_sc = torch.ones(NCH), torch.ones(NSC)
        if not cfg.position:
            keep_sc[6] = 0
        if not cfg.gr:                       # hide every GR-derived input (g, t, g - t, h, h_window)
            keep_seq[[0, 1, 2]] = 0
            keep_sc[[3, 4]] = 0
        self.register_buffer("seq_keep_nb", keep_seq.clone())
        self.register_buffer("sc_keep_nb", keep_sc.clone())
        keep_seq[NB_SEQ] = 0
        keep_sc[NB_SC] = 0
        self.register_buffer("seq_keep_no", keep_seq)
        self.register_buffer("sc_keep_no", keep_sc)
        h, M = cfg.hidden, cfg.mlp
        self.conv = nn.Sequential(nn.Conv1d(NCH, h, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(2 * h + NSC, M), nn.ReLU())
        self.head = nn.Linear(M + NSC, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():                       # start exactly at the (standardized) hand-crafted misfit cost
            self.head.weight[0, M + H_SCALAR] = 1.0
        self.tau = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.tau[2].weight)
        nn.init.zeros_(self.tau[2].bias)
        self.logT = nn.Parameter(torch.tensor(float(cfg.logT0)))

    def segment_cost(self, seq, scal, nbon):
        """seq [N, NPE, NCH], scal [N, NSC] raw inputs; nbon scalar 0/1 (neighbour inputs on)."""
        seq = (seq - self.seq_mu) / self.seq_sd
        scal = (scal - self.sc_mu) / self.sc_sd
        on = float(self.cfg.neighbour and nbon)
        seq = seq * (self.seq_keep_nb if on else self.seq_keep_no)
        scal = scal * (self.sc_keep_nb if on else self.sc_keep_no)
        z = self.conv(seq.transpose(1, 2))
        z = torch.cat([z.mean(2), z.amax(2), scal], 1)
        return self.head(torch.cat([self.mlp(z), scal], 1))[:, 0]

    def transition(self, p, m, u, mnb, dn, mnbp):
        """Momentum cost of move m after move p (lattice units); mnb, mnbp: field moves of this / previous chunk."""
        MV = self.MV
        x = torch.broadcast_tensors(p / MV, m / MV, (m - p) / MV, u / 5, mnb / MV, (m - mnb) / MV, dn,
                                    (p - mnbp) / MV, (m - mnb - 0.6 * (p - mnbp)) / MV)
        return self.tau(torch.stack(x, -1))[..., 0]


def well_costs(net, p, lat, block=24):
    """Segment costs [C, NA, NM] of one well (BIG for moves leaving the lattice)."""
    out = []
    for c0 in range(0, p["C"], block):
        cs = torch.arange(c0, min(p["C"], c0 + block), device=lat.device)

        def f(cs=cs):
            seq, scal = raw_inputs(p, cs, lat, net.clip, net.cfg.tw_step)
            n = len(cs)
            v = net.segment_cost(seq.reshape(-1, lat.NPE, NCH), scal.reshape(-1, NSC), p.get("nbon", 1))
            v = v.reshape(n, lat.NA, lat.NM)
            return torch.where(lat.VALID[None], v, torch.full_like(v, BIG))
        out.append(checkpoint(f, use_reentrant=False) if torch.is_grad_enabled() else f())
    return torch.cat(out, 0)


def batch_costs(net, ps, lat):
    """Stack the lattice costs of several wells (padded with zero-cost straight moves)."""
    Cm = max(p["C"] for p in ps)
    dev = lat.device
    pad = torch.full((lat.NA, lat.NM), BIG, device=dev)
    pad[:, lat.MV] = 0.0
    cost = torch.stack([torch.cat([well_costs(net, p, lat), pad[None].expand(Cm - p["C"], -1, -1)], 0) for p in ps])
    if not net.cfg.momentum:
        return cost, torch.zeros(len(ps), Cm, lat.NM, lat.NM, device=dev), torch.zeros(len(ps), lat.NM, device=dev)
    P_, M_ = lat.DV[:, None], lat.DV[None, :]
    taus, tau0 = [], []
    for p in ps:
        on = float(net.cfg.neighbour and p.get("nbon", 1))
        ext = Cm - p["C"]
        u = torch.cat([p["u"], p["u"][-1:].expand(ext)]) * float(net.cfg.position)
        mnb = torch.cat([p["mnb"], torch.zeros(ext, device=dev)]) * on
        dn = torch.cat([p["dnc"], p["dnc"][-1:].expand(ext)]) * on
        mnbp = torch.cat([mnb[:1], mnb[:-1]])
        tt = net.transition(P_[None], M_[None], u[:, None, None], mnb[:, None, None], dn[:, None, None],
                            mnbp[:, None, None])
        live = torch.arange(Cm, device=dev) < p["C"]
        taus.append(torch.where(live[:, None, None], tt, torch.zeros_like(tt)))
        tau0.append(net.transition(torch.tensor(float(p["p0"]), device=dev), lat.DV, u[0], mnb[0], dn[0], mnb[0]))
    return cost, torch.stack(taus), torch.stack(tau0)


def rows_tvt(Sb, ps, lat):
    """Anchor values at the boundaries [B, C+1] -> per-row TVT on each well's lattice rows."""
    B, Cp1 = Sb.shape
    f = torch.arange(lat.D, device=Sb.device, dtype=Sb.dtype) / lat.D
    Srow = torch.cat([(Sb[:, :-1, None] + (Sb[:, 1:] - Sb[:, :-1])[:, :, None] * f).reshape(B, -1), Sb[:, -1:]], 1)
    return [Srow[i, :len(p["off"])] + p["off"] for i, p in enumerate(ps)]


def decode(net, ps, lat, hard=False):
    cost, tau, tau0 = batch_costs(net, ps, lat)
    if hard:
        Sb = viterbi(cost, tau, tau0, lat.KC, lat.MV)
    else:
        Sb = forward_backward(cost, tau, tau0, net.logT.exp(), lat.KC, lat.MV)
    return rows_tvt(Sb, ps, lat)


def prediction_rows(t, p):
    """Model rows of the prediction zone (rows K .. N-1 of the well)."""
    return t[p["r"]:p["r"] + p["npred"]]
