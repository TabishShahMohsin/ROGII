"""Predict the test wells with one or more trained models (average of their soft paths) and write submission.csv.

    python -m ssv.predict --models runs/f0s0/model.pt runs/f1s0/model.pt ... --out submission.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from .config import Config
from .data import load_split, data_dir
from .features import prep, to_torch, Lattice
from .model import SSVNet, decode, prediction_rows
from . import neighbours


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    net = SSVNet(cfg, ck["stats"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, cfg, ck["stats"]


@torch.no_grad()
def predict_wells(net, cfg, stats, wells, nb, device, batch=4, hard=False):
    lat = Lattice(cfg, device)
    out, names = {}, sorted(wells)
    for i in range(0, len(names), batch):
        ps = [to_torch(prep(w, wells[w], nb[w], cfg, stats["gr_mu"], stats["gr_sd"]), device) for w in names[i:i + batch]]
        for t, p in zip(decode(net, ps, lat, hard), ps):
            out[p["w"]] = prediction_rows(t, p).cpu().numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--data", default=None)
    ap.add_argument("--hard", action="store_true", help="Viterbi path instead of the posterior mean")
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = load_split("train", a.data), load_split("test", a.data)
    nb = neighbours.build(test, train)              # test wells: support = all training wells
    preds = []
    for m in a.models:
        net, cfg, stats = load_model(m, dev)
        preds.append(predict_wells(net, cfg, stats, test, nb, dev, hard=a.hard))
        print("predicted with", m, flush=True)
    rows = {}
    for w, d in test.items():
        t = np.mean([p[w] for p in preds], 0)
        assert len(t) == len(d["MD"]) - d["K"]
        for j, v in enumerate(t):
            rows[f"{w}_{d['K'] + j}"] = float(v)
    sub = pd.read_csv(os.path.join(a.data or data_dir(), "sample_submission.csv"))
    col = [c for c in sub.columns if c != "id"][0]
    sub[col] = sub["id"].map(rows)
    missing = int(sub[col].isna().sum())
    print(f"rows {len(sub)}, missing {missing}", flush=True)
    assert missing == 0, "every id of the sample submission must be predicted"
    sub.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
