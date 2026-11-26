# Soft segment-Viterbi (SSV) for TVT prediction

Code for *Learning the segment cost of a differentiable segment-Viterbi for true vertical thickness prediction in
horizontal wells* (Al-Fakih and Shah Mohsin). Data: Kaggle competition
[ROGII – Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).

## Layout

| Path | Content |
|---|---|
| `ssv/config.py` | every setting, with the measurement that motivates it |
| `ssv/data.py` | reading the competition files; the 5-fold split (sorted names, permutation seed 42, position mod 5) |
| `ssv/field.py`, `ssv/neighbours.py` | neighbour structure field and the fold-safe neighbour inputs |
| `ssv/features.py` | detrended lattice coordinate, reference GR (typewell + own known section), raw inputs of every candidate segment |
| `ssv/model.py` | standardization constants, segment-cost and transition networks, lattice cost assembly, decoding |
| `ssv/dp.py` | second-order soft (forward-backward with soft-min) and hard segment-Viterbi; brute-force reference |
| `ssv/train.py` | training on one fold (`fold=-1`: all wells) |
| `ssv/predict.py` | test prediction with one or several models, `submission.csv` |
| `ssv/evaluate.py` | pooled RMSE, per-well errors, error concentration, error along the lateral, out-of-fold summary |
| `ssv/baselines.py` | hold-TVT, trend, neighbour field, lattice oracle, hand-crafted-cost Viterbi |
| `analysis/` | data analyses and figures of Section 3 of the paper |
| `kaggle/build_notebook.py` | builds self-contained Kaggle notebooks (training, evaluation, submission) |
| `tests/` | DP vs brute force; lattice alignment; input scaling; loss weighting |

## Install and test

```bash
pip install -r requirements.txt
export ROGII_DATA=/path/to/rogii-wellbore-geology-prediction   # holds train/, test/, sample_submission.csv
python -m pytest -q tests
```

## Train, evaluate, predict

```bash
python -m ssv.train --config '{"fold": 0, "seed": 0}' --out runs/f0s0 --nb-cache nb_train.pkl
python -m ssv.evaluate runs/f*s0/val.pkl                       # out-of-fold RMSE once all five folds are trained
python -m ssv.predict --models runs/f*s0/model.pt --out submission.csv
python -m ssv.baselines --out baselines.pkl --hand
```

Any `Config` field can be set in the JSON (`ssv/config.py`). The ablations of the paper are
`{"momentum": false}`, `{"neighbour": false}`, `{"seen": false}`, `{"position": false}`, `{"loss_norm": "batch"}`,
`{"loss_norm": "well"}`, `{"nb_dropout": 0.3}`, `{"scaling": "legacy"}`, `{"gr": false}` (see `kaggle/make_jobs.py`).

Training needs a GPU (about 5 minutes per epoch on one T4 for 618 wells); inference runs on a CPU.

## Kaggle

```bash
python kaggle/make_jobs.py                                     # writes kaggle/jobs_*.json (all runs of the paper)
python kaggle/build_notebook.py train    kaggle/jobs_final_s0.json  train_final_s0.ipynb
python kaggle/build_notebook.py evaluate kaggle/eval_final.json     evaluate.ipynb
python kaggle/build_notebook.py submit   kaggle/submit_final.json   submit_final.ipynb
```

Import the notebook, add the competition data (and, for evaluation/submission, the training notebooks' outputs) as
inputs, choose *GPU T4 x2* for training (two folds run in parallel) or no accelerator for submission, and save a version.
