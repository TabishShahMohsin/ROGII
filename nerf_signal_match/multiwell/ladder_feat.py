import pickle,numpy as np,pandas as pd,warnings,time;warnings.filterwarnings('ignore')
from common import W,NAMES,nb2
A=pd.read_pickle('comb3_out.pkl');rows=[];t0=time.time()
for w,g in A.groupby('w'):
    d=W[NAMES[w]];K=d['K'];li=g.t.values.astype(int);ab=K+li
    Z=d['Z'].astype(float);MD=d['MD'].astype(float);S=d['TVT_input'].astype(float)+Z
    z0=Z[K-1];md0=MD[K-1];tv0=d['TVT_input'][K-1]
    f=pd.DataFrame({'dZ':Z[ab]-z0,'dMD':MD[ab]-md0,'dX':d['X'][ab]-d['X'][K-1],'dY':d['Y'][ab]-d['Y'][K-1]},index=g.index)
    for Wn in (100,300,1000,3000):
        a=max(K-1-Wn,0);b=K-1;f[f'sl{Wn}']=(S[b]-S[a])/max(MD[b]-MD[a],1e-6)
    # linear slope of Z in lateral (known: Z path is known for the whole lateral)
    f['zsl']=np.polyfit(MD[K:],Z[K:],1)[0]
    # neighbour surface delta (ANCC field), sampled every ~10th row then interpolated
    sub=np.unique(np.r_[np.arange(0,len(li),10),len(li)-1]);xq=np.r_[d['X'][K-1],d['X'][ab[sub]]];yq=np.r_[d['Y'][K-1],d['Y'][ab[sub]]]
    for h in (1500,4000):
        v,ws=nb2(xq,yq,h,{w},ret_w=True);dv=v[1:]-v[0]
        f[f'nbd{h}']=np.interp(li,li[sub],dv);f[f'nbw{h}']=np.interp(li,li[sub],np.minimum(ws[1:],ws[0]))
    f['w']=w;rows.append(f)
    if len(rows)%100==0:print(len(rows),'%.0fs'%(time.time()-t0),flush=True)
F=pd.concat(rows).loc[A.index];F.to_pickle('ladder_F.pkl');print(F.shape)
