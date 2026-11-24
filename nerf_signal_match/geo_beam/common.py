import numpy as np, pickle
from sklearn.cluster import KMeans
W=pickle.load(open('/home/claude/rg/wells.pkl','rb')); NAMES=sorted(W)
CEN=np.array([[np.nanmean(W[w]['X']),np.nanmean(W[w]['Y'])] for w in NAMES])
REG=KMeans(12,n_init=10,random_state=0).fit_predict(CEN)
def ev(w): d=W[w]; return d['TVT'][d['K']:].astype(np.float64)
def pooled(pred):  # pred: dict w->array over eval rows
    se=0.;n=0
    for w,p in pred.items(): e=p-ev(w); se+=np.sum(e*e); n+=len(e)
    return float(np.sqrt(se/n))
def perwell(pred): return {w:float(np.sqrt(np.mean((p-ev(w))**2))) for w,p in pred.items()}
