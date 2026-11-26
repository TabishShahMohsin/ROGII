"""Build self-contained Kaggle notebooks (the package source is embedded with %%writefile cells).

    python kaggle/build_notebook.py train  jobs.json  out.ipynb   # jobs: list of {"tag":..., "config":{...}}
    python kaggle/build_notebook.py submit models.json out.ipynb  # models: list of glob patterns under /kaggle/input
    python kaggle/build_notebook.py evaluate patterns.json out.ipynb

Training notebooks run the jobs two at a time, one per GPU of a "GPU T4 x2" session.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "ssv")


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def package_cells():
    cells = [code("import os\nos.makedirs('/kaggle/working/ssv', exist_ok=True)\n")]
    for f in sorted(glob.glob(os.path.join(PKG, "*.py"))):
        cells.append(code(f"%%writefile /kaggle/working/ssv/{os.path.basename(f)}\n" + open(f).read()))
    return cells


TRAIN = r'''import subprocess, os, json, time, sys
os.chdir('/kaggle/working')
JOBS = %s
import torch
ngpu = max(1, torch.cuda.device_count())
# fold-safe neighbour inputs for every training well (CPU, ~1 min)
subprocess.run([sys.executable, '-c', 'from ssv.data import load_split; from ssv import neighbours; '
                'neighbours.train_cache(load_split("train"), "/kaggle/working/nb_train.pkl")'], check=True)
running, queue = {}, list(JOBS)
logs = {}
while queue or running:
    for g in range(ngpu):
        if g not in running and queue:
            job = queue.pop(0)
            out = f"/kaggle/working/runs/{job['tag']}"; os.makedirs(out, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS='2')
            f = open(out + '/stdout.txt', 'w')
            running[g] = (job, subprocess.Popen([sys.executable, '-m', 'ssv.train', '--config', json.dumps(job['config']),
                                                 '--out', out, '--nb-cache', '/kaggle/working/nb_train.pkl'],
                                                env=env, stdout=f, stderr=subprocess.STDOUT), time.time(), f)
            print('started', job['tag'], 'on GPU', g, flush=True)
    time.sleep(30)
    for g, (job, p, t0, f) in list(running.items()):
        if p.poll() is not None:
            f.close(); del running[g]
            tail = open(f"/kaggle/working/runs/{job['tag']}/stdout.txt").read().splitlines()[-3:]
            print('finished', job['tag'], 'exit', p.returncode, f'{(time.time()-t0)/3600:.2f} h', flush=True)
            for l in tail: print('   ', l[:400], flush=True)
    for g, (job, p, t0, f) in running.items():
        lp = f"/kaggle/working/runs/{job['tag']}/log.jsonl"
        if os.path.exists(lp):
            ls = open(lp).read().splitlines()
            if ls and logs.get(job['tag']) != ls[-1]:
                logs[job['tag']] = ls[-1]; print(job['tag'], ls[-1][:300], flush=True)
'''

SUBMIT = r'''import glob, subprocess, sys, os
os.chdir('/kaggle/working')
PATTERNS = %s
models = sorted({m for pat in PATTERNS for m in glob.glob(pat, recursive=True)})
print(len(models), 'models'); [print('  ', m) for m in models]
assert models, 'no model.pt found under the inputs'
subprocess.run([sys.executable, '-m', 'ssv.predict', '--models', *models, '--out', '/kaggle/working/submission.csv'], check=True)
'''

EVALUATE = r'''import glob, subprocess, sys, os
os.chdir('/kaggle/working')
PATTERNS = %s
for name, pats in PATTERNS.items():
    files = sorted({m for pat in pats for m in glob.glob(pat, recursive=True)})
    print('=====', name, len(files), 'files', flush=True)
    subprocess.run([sys.executable, '-m', 'ssv.evaluate', *files], check=True)
'''


def main():
    kind, spec, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
    title = {"train": "Soft segment-Viterbi: training", "submit": "Soft segment-Viterbi: submission",
             "evaluate": "Soft segment-Viterbi: out-of-fold evaluation"}[kind]
    tmpl = {"train": TRAIN, "submit": SUBMIT, "evaluate": EVALUATE}[kind]
    cells = [md(f"# {title}\nSource: the `ssv` package of the paper's code release (embedded below).")]
    cells += package_cells() + [code(tmpl % json.dumps(spec, indent=1))]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(out, "w"), indent=1)
    print(out, len(cells), "cells")


if __name__ == "__main__":
    main()
