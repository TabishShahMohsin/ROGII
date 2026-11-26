# Top-k distinct paths of the trained segment Viterbi (list Viterbi + diversity filter), plotted in S space.
import sys, os, json, pickle, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
CFGS=sys.argv[1]; WTS=sys.argv[2]; OUT=sys.argv[3]; WELLS=sys.argv[4].split(',') if len(sys.argv)>4 else None
sys.argv=['vsoft.py',json.dumps(dict(json.loads(CFGS),ep=0,skiphand=1,tag='kb_tmp'))]
exec(open('vsoft.py').read())
if WTS!='none': net.load_state_dict(torch.load(WTS,weights_only=False))
net.eval(); os.makedirs(OUT,exist_ok=True)
PLOT=set(os.environ.get('PLOT','').split(',')) - {''}; NSHOW=4; MINSEP=float(os.environ.get('MINSEP',4.0))
def min_marginal_paths(Cs):
    # best path forced through each node (i,a): alpha(i,a)+beta(i,a); returns candidates sorted by cost
    Ms=[M.numpy() for M in Cs]; n=len(Ms)
    al=[np.zeros(1)]; bpa=[None]
    for M in Ms: t=al[-1][:,None]+M; bpa.append(t.argmin(0)); al.append(t.min(0))
    be=[None]*(n+1); be[n]=np.zeros(Ms[-1].shape[1]); fpa=[None]*(n+1)
    for i in range(n-1,-1,-1): t=Ms[i]+be[i+1][None,:]; fpa[i]=t.argmin(1); be[i]=t.min(1)
    cand=[]
    for i in range(1,n+1):
        mm=al[i]+be[i]
        for a in np.argsort(mm)[:60]: cand.append((mm[a],i,a))
    cand.sort(key=lambda x:x[0]); out=[]
    for cost,i,a in cand:
        nodes=[0]*(n+1); nodes[i]=a; j=a
        for k in range(i,0,-1): j=bpa[k][j]; nodes[k-1]=j
        j=a
        for k in range(i,n): j=fpa[k][j]; nodes[k+1]=j
        out.append((cost,nodes))
    return out
rows=[]
with torch.no_grad():
    T=net.logT.exp()
    for w in (WELLS or va):
        d=D[w]; Cs=lattice_costs(net,d); A=d['A']; y=d['tvt']; off=d['z']+d['tvt0']; Sy=y-off
        paths=min_marginal_paths(Cs); soft=to_tvt(soft_path(Cs,A,T),d).numpy()-off
        sel=[]
        for cost,nodes in paths:
            S=to_tvt(torch.tensor([A[i][n] for i,n in enumerate(nodes)],dtype=torch.float32),d).numpy()-off
            if all(np.sqrt(np.mean((S-s2)**2))>MINSEP for _,s2,_ in sel): sel.append((cost,S,nodes))
            if len(sel)>=NSHOW: break
        best=sel[0][0]; r=lambda S:float(np.sqrt(np.mean((S-Sy)**2)))
        rows.append(dict(w=w,n=len(Sy),soft=r(soft),**{f'p{k+1}_rmse':r(S) for k,(c,S,_) in enumerate(sel)},**{f'p{k+1}_dcost':float(c-best) for k,(c,S,_) in enumerate(sel)},oracle_topk=min(r(S) for _,S,_ in sel)))
        if PLOT and w not in PLOT: continue
        md=np.asarray(W[w]['MD'][W[w]['K']:],float)
        fig,ax=plt.subplots(figsize=(14,6)); cols=['r','tab:orange','tab:green','tab:purple']
        for i,b in enumerate(d['bnd']):
            rr=min(b,len(md)-1); ax.scatter(np.full(len(A[i]),md[rr]),A[i],s=0.3,c='0.85')
        ax.plot(md,Sy,'k',lw=2.2,label='truth')
        for k,(c,S,_) in enumerate(sel): ax.plot(md,S,c=cols[k],lw=1.3,label=f'path {k+1}: +{c-best:.1f} cost, rmse {r(S):.1f}')
        ax.plot(md,soft,'m--',lw=0.9,label=f'soft (mean) rmse {r(soft):.1f}')
        yy=np.r_[Sy,np.concatenate([s for _,s,_ in sel])]; ax.set_ylim(yy.max()+8,yy.min()-8)
        ax.set_ylabel('S = TVT - z_det - tvt0 (ft)'); ax.set_xlabel('MD'); ax.legend(fontsize=8); ax.set_title(f'{w}: top {len(sel)} distinct Viterbi paths (>{MINSEP:.0f} ft apart)')
        fig.tight_layout(); f=f'{OUT}/K_{w}.png'; fig.savefig(f,dpi=80); plt.close(fig); print(f,flush=True)
import pandas as pd
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/topk.csv',index=False)
pool=lambda c: np.sqrt((df[c]**2*df.n).sum()/df.n.sum())
print('pooled RMSE  hard(p1)',pool('p1_rmse').round(3),' soft',pool('soft').round(3),' best-of-top-k oracle',pool('oracle_topk').round(3))
for k in [2,3,4]:
    cols=[f'p{j}_rmse' for j in range(1,k+1) if f'p{j}_rmse' in df]; df[f'or{k}']=df[cols].min(1); print(f'best of top-{k}:',pool(f'or{k}').round(3))
print('median cost gap p2/p3/p4:',df[[c for c in df if c.endswith('_dcost')]].median().round(3).to_dict())
df['sse']=df.p1_rmse**2*df.n; print(df.sort_values('sse',ascending=False).head(20).round(2).to_string())
