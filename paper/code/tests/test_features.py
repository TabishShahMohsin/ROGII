"""Alignment and scaling checks on real wells (skipped when the competition data are not available)."""
import os
import numpy as np
import pytest
import torch
from ssv.config import Config
from ssv.data import load_split, data_dir
from ssv.features import prep, to_torch, gr_stats, Lattice, raw_inputs, lattice_start
from ssv.model import fit_stats, SSVNet, decode
from ssv.train import batch_loss
from ssv import neighbours

try:
    data_dir()
    HAVE = True
except FileNotFoundError:
    HAVE = False
pytestmark = pytest.mark.skipif(not HAVE, reason="set ROGII_DATA to run")


@pytest.fixture(scope="module")
def setup():
    W = load_split("train")
    names = sorted(W)[:12]
    sub = {w: W[w] for w in names}
    nb = neighbours.build(sub, {w: W[w] for w in sorted(W)[12:60]})
    cfg = Config()
    gmu, gsd = gr_stats(sub)
    P = {w: to_torch(prep(w, W[w], nb[w], cfg, gmu, gsd), "cpu") for w in names}
    return W, names, cfg, P, Lattice(cfg, "cpu")


def test_lattice_grid_ends_at_last_row(setup):
    W, names, cfg, P, lat = setup
    for w in names:
        d = W[w]
        s, r = lattice_start(d["K"], len(d["MD"]), cfg.chunk)
        assert 1 <= r <= cfg.chunk and s + r == d["K"]
        assert (len(d["MD"]) - s) % cfg.chunk == 0 or r == d["K"] - 1


def test_known_rows_map_exactly(setup):
    """S = 0 at the lattice start, and the known TVT is recovered exactly from the lattice coordinate."""
    W, names, cfg, P, lat = setup
    for w in names:
        p, d = P[w], W[w]
        s, K = p["s"], d["K"]
        assert abs(float(p["off"][0]) - d["TVT_input"][s]) < 1e-3          # S(s) = TVT(s) - off(s) = 0
        y = p["y"][: p["N"] - s].numpy()
        assert np.allclose(y[: K - s], d["TVT"][s:K], atol=1e-3)
        assert float(p["lm"][: p["r"]].sum()) == 0 and float(p["lm"][p["r"]: p["r"] + p["npred"]].min()) == 1


def test_true_path_inside_lattice(setup):
    """The true surface stays inside the anchor range for these wells."""
    W, names, cfg, P, lat = setup
    for w in names:
        p = P[w]
        S = (p["y"] - p["off"])[p["lm"] > 0]
        assert S.abs().max() < cfg.kc


def test_inputs_and_stats_are_finite(setup):
    W, names, cfg, P, lat = setup
    st = fit_stats([P[w] for w in names], lat, cfg)
    assert st["clip"] > 0 and np.isfinite(st["seq_sd"]).all() and min(st["seq_sd"]) > 0
    seq, scal = raw_inputs(P[names[0]], torch.arange(3), lat, st["clip"], cfg.tw_step)
    assert torch.isfinite(seq).all() and torch.isfinite(scal).all()
    st.update(gr_mu=0.0, gr_sd=1.0)
    net = SSVNet(cfg, st)
    with torch.no_grad():
        out = decode(net, [P[names[0]]], lat, hard=False)[0]
    assert torch.isfinite(out).all()


def test_global_loss_weights_every_row_equally():
    """Summing the 'global' batch losses over an epoch gives total squared error / (B * mean rows)."""
    rs = np.random.RandomState(0)
    ps, preds = [], []
    for n in [100, 3000, 250, 5000, 800, 1200, 60, 4000]:
        y = torch.tensor(rs.randn(n), dtype=torch.float32)
        ps.append(dict(y=y, lm=torch.ones(n)))
        preds.append(y + torch.tensor(rs.randn(n), dtype=torch.float32))
    nbar = float(np.mean([len(p["y"]) for p in ps]))
    tot = sum(batch_loss(preds[i:i + 4], ps[i:i + 4], "global", nbar)[0] for i in range(0, 8, 4))
    se = sum(((t - p["y"]) ** 2).sum() for t, p in zip(preds, ps))
    assert torch.allclose(tot, se / (4 * nbar))
