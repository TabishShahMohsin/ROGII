# resumable: per-well pickles in outdir; stops submitting after `budget` seconds
import sys, os, time, pickle
from multiprocessing import Pool
import run
if __name__=='__main__':
    a,b,outd,nproc,budget=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),float(sys.argv[5])
    os.makedirs(outd,exist_ok=True); t=time.time()
    todo=[w for w in run.NAMES[a:b] if not os.path.exists(f'{outd}/{w}.pkl')]
    done=0
    with Pool(nproc) as pool:
        it=pool.imap_unordered(run.job,todo,chunksize=1)
        for w,o in it:
            pickle.dump(o,open(f'{outd}/{w}.pkl','wb')); done+=1
            if time.time()-t>budget: pool.terminate(); break
    print('done',done,'remaining',len(todo)-done,round(time.time()-t))
