"""Write the job lists of all training notebooks of the paper (kaggle/jobs_*.json)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
F5 = range(5)


def jobs(tag, cfg, folds=F5, seed=0):
    return [{"tag": f"{tag}_f{f}_s{seed}", "config": dict(cfg, fold=f, seed=seed, tag=tag)} for f in folds]


SETS = {
    # main result: 5 folds x 3 seeds (one notebook per seed)
    "final_s0": jobs("final", {}, seed=0),
    "final_s1": jobs("final", {}, seed=1),
    "final_s2": jobs("final", {}, seed=2),
    # ablations on all five folds
    "ablate_a": jobs("nomom", {"momentum": False}) + jobs("nonb", {"neighbour": False}),
    "ablate_b": jobs("legacy", {"scaling": "legacy"}) + jobs("batchloss", {"loss_norm": "batch"}),
    # cheaper ablations on folds 0 and 1
    "ablate_c": (jobs("noseen", {"seen": False}, folds=(0, 1)) + jobs("nopos", {"position": False}, folds=(0, 1))
                 + jobs("wellloss", {"loss_norm": "well"}, folds=(0, 1)) + jobs("nbdrop", {"nb_dropout": 0.3}, folds=(0, 1))
                 + jobs("nogr", {"gr": False}, folds=(0, 1))),
}

if __name__ == "__main__":
    for k, v in SETS.items():
        json.dump(v, open(os.path.join(HERE, f"jobs_{k}.json"), "w"), indent=1)
        print(k, len(v), "jobs")
