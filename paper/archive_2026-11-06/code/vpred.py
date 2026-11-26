# Predict test wells with a trained soft-Viterbi cost net (hard path) and write submission.csv.
import sys, os, json, pickle, numpy as np, pandas as pd, torch
CFGS=sys.argv[1]; WTS=sys.argv[2]; OUT=sys.argv[3]
import common, field3
TE=common._load('test'); TN=sorted(TE); print('test wells',len(TN),flush=True)
F=field3.Field(common.W,common.NAMES)                      # neighbour field from all training wells
nbf=pickle.load(open(common.NBP,'rb'))
for w in TN:
    nbf[w]=F.path(TE[w],robust=0.05)[0].astype(np.float32)
    TE[w]['TVT']=np.full(len(TE[w]['MD']),np.nan)           # placeholder: build() expects the key; never used for prediction
pickle.dump(nbf,open(common.NBP,'wb')); common.W.update(TE)
sys.argv=['vsoft.py',json.dumps(dict(json.loads(CFGS),ep=0,skiphand=1,tag='pred_tmp'))]
exec(open('vsoft.py').read())
net.load_state_dict(torch.load(WTS,weights_only=False)); net.eval()
out={}
with torch.no_grad():
    for k,w in enumerate(TN):
        d=build(w); Cs=lattice_costs(net,d); p=to_tvt(hard_path(Cs,d['A']),d).numpy(); K=TE[w]['K']
        assert len(p)==len(TE[w]['MD'])-K
        for j,v in enumerate(p): out[f'{w}_{K+j}']=float(v)
        if k%20==0: print(k,w,flush=True)
sub=pd.read_csv(f'{common.BASE}/sample_submission.csv'); sub['tvt']=sub['id'].map(out)
print('missing ids',sub['tvt'].isna().mean(),flush=True); sub['tvt']=sub['tvt'].fillna(0); sub.to_csv(OUT,index=False); print(sub.head(),flush=True)
