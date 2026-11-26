"""All tunable settings in one place. Every value that is not a pure implementation detail is listed here,
with the analysis that motivates it (see the paper, Sections 3-4)."""
from dataclasses import dataclass, asdict, fields
import json


@dataclass
class Config:
    # ---------------- lattice (Sec. 4.1) ----------------
    chunk: int = 80            # rows (= ft of MD) per straight segment; 88% of the interpreted evaluation-zone path lies in pieces >= 80 ft
    kc: int = 100              # anchors on each side of the trend line, 1 ft apart (lattice spans +-100 ft)
    max_slope: float = 0.05    # |dS/dMD| relative to the trajectory trend; covers 98.6% of the interpreted path length
    # ---------------- inputs (Sec. 4.2) ----------------
    pts_per_chunk: int = 16    # samples of the segment line per chunk
    ext_back: int = 2          # chunks of context before the segment (line extended backwards)
    ext_fwd: int = 2           # chunks of context after the segment (line extended forwards)
    gr_box: int = 5            # rows of box smoothing of the lateral GR after the low-pass filter
    tw_step: float = 0.25      # ft, resampling step of the typewell
    seen_blend: float = 0.7    # weight of the well's own known section in the reference GR(TVT) curve
    seen_window: float = 100.0 # ft of TVT around the last known TVT used for the own-section reference
    misfit_clip: float = -1.0  # clip of the squared GR misfit in standardized units; <0 -> value from the stats file
    scaling: str = "data"      # 'data': standardize with training statistics; 'legacy': the earlier fixed scales (ablation)
    # ---------------- model (Sec. 4.3) ----------------
    hidden: int = 32           # conv channels of the segment-cost network
    mlp: int = 64              # hidden units of the segment-cost head
    momentum: bool = True      # second-order lattice with a learned transition cost
    position: bool = True      # distance from the last known row as an input
    neighbour: bool = True     # neighbour structure-field inputs
    gr: bool = True            # GR inputs (False: structure prior and momentum only, ablation)
    seen: bool = True          # blend the own known section into the reference curve
    logT0: float = -3.0        # initial log temperature of the soft-min
    # ---------------- training (Sec. 4.4) ----------------
    epochs: int = 15
    batch_wells: int = 4
    lr: float = 1e-3
    lr_min_frac: float = 0.05  # cosine schedule floor (fraction of lr)
    weight_decay: float = 1e-4
    ema: float = 0.99
    grad_clip: float = 5.0
    loss_norm: str = "global"  # 'global': every predicted row has the same weight in every batch (unbiased for the
                               #            row-pooled RMSE); 'batch': divide by the rows of the batch; 'well': mean per well
    nb_dropout: float = 0.0    # probability of hiding the neighbour inputs of a training well
    seed: int = 0
    # ---------------- cross-validation ----------------
    n_folds: int = 5
    fold: int = 0              # validation fold; -1 = train on all wells
    fold_seed: int = 42
    tag: str = "ssv"

    @property
    def mv(self) -> int:       # largest anchor move per chunk
        return int(round(self.max_slope * self.chunk))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls.from_dict(json.loads(s))
