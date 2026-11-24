# generic resumable runner: python3 runj.py MODULE START END OUTDIR NPROC BUDGET_S [order]
import sys, os, time, pickle, importlib
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),'old'))
mod = importlib.import_module(sys.argv[1])
from common import NAMES
if __name__ == '__main__':
    a, b, outd, nproc, budget = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), float(sys.argv[6])
    rev = len(sys.argv) > 7 and sys.argv[7] == 'rev'
    os.makedirs(outd, exist_ok=True); t = time.time()
    todo = [w for w in NAMES[a:b] if not os.path.exists(f'{outd}/{w}.pkl')]
    if rev: todo = todo[::-1]
    done = 0
    with Pool(nproc) as pool:
        for w, o in pool.imap_unordered(mod.job, todo, chunksize=1):
            pickle.dump(o, open(f'{outd}/{w}.pkl', 'wb')); done += 1
            if time.time() - t > budget: pool.terminate(); break
    print('done', done, 'remaining', len(todo) - done, round(time.time() - t), flush=True)
