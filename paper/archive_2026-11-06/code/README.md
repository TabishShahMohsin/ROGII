# Soft segment-Viterbi for TVT prediction: code

Author: Tabish Shah Mohsin. Data: Kaggle "ROGII – Wellbore Geology Prediction".

## Files

| File | Role |
|---|---|
| `common.py` | Loads the train wells, defines the 5-fold split (sorted well names, permutation seed 42, index mod 5), and builds the fold-safe neighbour field for every training well (`nbf.pkl`). |
| `field3.py` | Neighbour structure field. Takes along-track dip observations of S = TVT + Z from labelled support wells, plus the query well's own build section. Solves a weighted least-squares fit for the local map-view gradient, with one robust reweighting pass, then integrates the gradient along the query trajectory. |
| `lat.py` | Lattice construction. Detrends the lateral into S space, places chunk boundaries every 80 rows (1 row = 1 ft MD), and puts anchors on a 1 ft grid within ±0.05·distance (cap ±100 ft) of the straight baseline. |
| `vsoft.py` | Model and training: segment features, cost network, soft Viterbi (forward–backward with softmin), MSE loss on TVT, hard Viterbi decode, training loop. Also contains the variants used in the ablations (`big`, `map`). |
| `vana.py` | Error analysis: per-well RMSE, share of squared error, error by position along the lateral, and plots. |
| `kbest.py` | Distinct alternative Viterbi paths (min-marginal paths more than 4 ft apart) and the best-of-k oracle. |
| `vpred.py` | Test-time prediction. Builds the neighbour field from all training wells, runs the hard Viterbi path, and writes `submission.csv`. |

## Environment

Python 3.11+, with numpy, pandas, scipy, torch (CPU is enough) and matplotlib.

## Data location

Point `ROGII_DATA` at the competition folder, which holds `train/`, `test/` and `sample_submission.csv`. On Kaggle the path `/kaggle/input/competitions/rogii-wellbore-geology-prediction` is detected automatically.

The scripts also read these environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `NBP` | Path of the neighbour-field cache | `/kaggle/working/nbf.pkl`; built on first import of `common.py` (about 1 minute) |
| `CKDIR` | Checkpoint directory | |
| `NT` | Number of torch threads | |

## Reproduce the released model (fold 0)

```bash
export ROGII_DATA=/path/to/data NBP=./nbf.pkl CKDIR=./ NT=4
python vsoft.py '{"tag":"smallnb","h":12,"ep":10,"ckpt":0,"eb":1,"ef":1,"lr":1e-3,"lrmin":0.05,"wd":1e-4,"ema":0.995,"logT0":-3,"fold":0,"skiphand":1}'
```

- **Output:** `vs_smallnb_best.pt` holds the EMA weights of the best validation epoch, and `vs_smallnb.pt` is the resumable checkpoint.
- **Run time:** about 20 minutes per epoch on 4 CPU cores.
- **Validation output:** each epoch prints both the soft-path and hard-path RMSE.

## Evaluate, analyse, predict with the released weights (no training)

```bash
CFG='{"h":12,"ckpt":0,"eb":1,"ef":1,"logT0":-3,"fold":0}'
python vana.py  "$CFG" ../weights/vs_smallnb_best.pt ./ana     # fold-0 per-well analysis (hard 6.575)
PLOT=2d35f86d,896d15b9 python kbest.py "$CFG" ../weights/vs_smallnb_best.pt ./kb   # top-4 distinct paths
python vpred.py "$CFG" ../weights/vs_smallnb_best.pt submission.csv                  # test prediction
```

## Ablation configs (fold 0, `vsoft.py`)

| Run | Config |
|---|---|
| hand cost (no learning) | Printed when `skiphand` is not set |
| sym1 | `{"tag":"sym1","eb":1,"ef":1,"ep":6,"logT0":-3,"lr":1e-3,"fold":0,"nbch":0,"ckpt":1}` (Adam, flat learning rate) |
| ahead8 | Same as sym1 with `"eb":0,"ef":8` |
| smallnb | Config above |
| big24 | `{"tag":"big24","big":1,"h":24,"ep":6,"ckpt":0,"eb":1,"ef":1,"lr":1e-3,"lrmin":0.05,"wd":1e-4,"ema":0.995,"logT0":-3,"fold":0,"skiphand":1}` |
| map24 | `{"tag":"map24","map":1,"nl":6,"ds":4,"h":24,"ep":10,"lr":1e-3,"lrmin":0.05,"wd":1e-4,"ema":0.995,"logT0":-3,"fold":0,"skiphand":1}` |

## Weights (`../weights`)

- **`vs_smallnb_best.pt`:** PyTorch `state_dict` of the cost network, 1,271 parameters, fold-0 model, best epoch (9th of 10).
  - Learned hand-cost weight w = 0.877 and temperature T = 0.049.
  - SHA-256 is verified on export from Kaggle.
- **`vs_smallnb.pt`:** final checkpoint (epoch 10) with optimizer state and EMA weights.
