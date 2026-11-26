"""Train one model on one fold (or on all wells with fold = -1).

    python -m ssv.train --config '{"fold": 0, "seed": 0}' --out runs/f0s0 [--nb-cache nb.pkl]

Writes <out>/model.pt (EMA weights + config + standardization constants), <out>/log.jsonl (one line per epoch),
and <out>/val.pkl (prediction-zone soft and hard TVT of every validation well).
"""
import argparse
import copy
import json
import math
import os
import pickle
import time
import numpy as np
import torch
from .config import Config
from .data import load_split, split_names
from .features import prep, to_torch, gr_stats, Lattice
from .model import SSVNet, fit_stats, batch_costs, rows_tvt, decode, prediction_rows
from .dp import forward_backward
from . import neighbours


def batch_loss(pred, ps, norm, rows_per_well):
    se = sum(((t - p["y"]) ** 2 * p["lm"]).sum() for t, p in zip(pred, ps))
    n = sum(p["lm"].sum() for p in ps)
    if norm == "global":          # each predicted row has the same weight in every batch
        return se / (len(ps) * rows_per_well), se, n
    if norm == "batch":           # rows of short-well batches weigh more
        return se / n, se, n
    if norm == "well":            # each well has the same weight (rows of short wells weigh more)
        return sum(((t - p["y"]) ** 2 * p["lm"]).sum() / p["lm"].sum() for t, p in zip(pred, ps)) / len(ps), se, n
    raise ValueError(norm)


@torch.no_grad()
def evaluate(net, P, names, lat, B, keep=False):
    net.eval()
    se = dict(soft=0.0, hard=0.0)
    n = 0.0
    out = {}
    for i in range(0, len(names), B):
        ps = [P[w] for w in names[i:i + B]]
        for k, hard in (("soft", False), ("hard", True)):
            pred = decode(net, ps, lat, hard)
            for t, p in zip(pred, ps):
                se[k] += float(((t - p["y"]) ** 2 * p["lm"]).sum())
                if keep:
                    out.setdefault(p["w"], {})[k] = prediction_rows(t, p).cpu().numpy()
        n += float(sum(p["lm"].sum() for p in ps))
    r = {k: round(math.sqrt(v / n), 4) for k, v in se.items()}
    return (r, out) if keep else r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}", help="JSON string or path to a JSON file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nb-cache", default="nb_train.pkl")
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many train and val wells")
    a = ap.parse_args(argv)
    cfg = Config.from_json(open(a.config).read() if os.path.exists(a.config) else a.config)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = load_split("train", a.data)
    tr, va = split_names(W, cfg.fold, cfg.n_folds, cfg.fold_seed)
    if a.limit:
        tr, va = tr[:a.limit], va[:max(1, a.limit // 4)]
    NB = neighbours.train_cache(W, a.nb_cache, cfg.n_folds, cfg.fold_seed)
    gmu, gsd = gr_stats({w: W[w] for w in tr})
    t0 = time.time()
    P = {w: to_torch(prep(w, W[w], NB[w], cfg, gmu, gsd), dev) for w in tr + va}
    lat = Lattice(cfg, dev)
    stats = fit_stats([P[w] for w in tr], lat, cfg, seed=cfg.seed)
    stats.update(gr_mu=gmu, gr_sd=gsd)
    net = SSVNet(cfg, stats).to(dev)
    print(json.dumps(dict(event="start", wells_train=len(tr), wells_val=len(va), device=dev,
                          prep_s=round(time.time() - t0, 1), params=sum(q.numel() for q in net.parameters()),
                          config=json.loads(cfg.to_json()), stats=stats)), flush=True)
    ema = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    B = cfg.batch_wells
    steps = cfg.epochs * math.ceil(len(tr) / B)
    rows_per_well = float(np.mean([float(P[w]["lm"].sum()) for w in tr]))
    rs = np.random.RandomState(cfg.seed)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    if va:
        r0 = evaluate(net, P, va, lat, B)
        log.write(json.dumps(dict(epoch=-1, val=r0)) + "\n"); log.flush()
        print(json.dumps(dict(epoch=-1, val=r0)), flush=True)
    st = 0
    for ep in range(cfg.epochs):
        net.train()
        t1 = time.time()
        se_sum = n_sum = 0.0
        order = list(rs.permutation(tr))
        for i in range(0, len(order), B):
            ps = []
            for w in order[i:i + B]:
                p = dict(P[w])
                p["nbon"] = 0 if (cfg.nb_dropout > 0 and rs.rand() < cfg.nb_dropout) else 1
                ps.append(p)
            for g_ in opt.param_groups:
                g_["lr"] = cfg.lr * (cfg.lr_min_frac + (1 - cfg.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * st / steps)))
            cost, tau, tau0 = batch_costs(net, ps, lat)
            Sb = forward_backward(cost, tau, tau0, net.logT.exp(), lat.KC, lat.MV)
            loss, se, n = batch_loss(rows_tvt(Sb, ps, lat), ps, cfg.loss_norm, rows_per_well)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            st += 1
            with torch.no_grad():
                for e, q in zip(ema.parameters(), net.parameters()):
                    e.mul_(cfg.ema).add_(q, alpha=1 - cfg.ema)
            se_sum += float(se.detach())
            n_sum += float(n)
        rec = dict(epoch=ep, train_rmse=round(math.sqrt(se_sum / n_sum), 4), T=round(float(net.logT.exp()), 5),
                   lr=opt.param_groups[0]["lr"], sec=round(time.time() - t1, 1))
        if va:
            rec["val"] = evaluate(ema, P, va, lat, B)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        torch.save(dict(state=ema.state_dict(), config=json.loads(cfg.to_json()), stats=stats),
                   os.path.join(a.out, "model.pt"))
    if va:
        r, kp = evaluate(ema, P, va, lat, B, keep=True)
        pickle.dump(kp, open(os.path.join(a.out, "val.pkl"), "wb"))
        print(json.dumps(dict(event="final", val=r)), flush=True)
    log.close()


if __name__ == "__main__":
    main()
