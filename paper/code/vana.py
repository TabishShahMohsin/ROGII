# Error analysis for a trained soft-Viterbi cost net: per-well RMSE, SSE share, error vs position, plots of grid/oracle/pred vs truth.
import sys, os, json, pickle, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
CFGS=sys.argv[1]; WTS=sys.argv[2]; OUT=sys.argv[3] if len(sys.argv)>3 else '/kaggle/working/ana'
sys.argv=['vsoft.py',json.dumps(dict(json.loads(CFGS),ep=0,skiphand=1,tag='ana_tmp'))]
exec(open('vsoft.py').read())          # defines net, build/lattice fns, va, D; ep=0 so no training
net.load_state_dict(torch.load(WTS,weights_only=False)); net.eval(); os.makedirs(OUT,exist_ok=True)
rows=[]; ERR={}
with torch.no_grad():
    T=net.logT.exp()
    for w in va:
        d=D[w]; y=d['tvt']; Cs=lattice_costs(net,d)
        hard=to_tvt(hard_path(Cs,d['A']),d).numpy(); soft=to_tvt(soft_path(Cs,d['A'],T),d).numpy()
        hand=to_tvt(hard_path(lattice_costs(net,d,False),d['A']),d).numpy()
        p=prep(w,CH,CH); ta,Sb=true_anchors(p,d['A']); orc=path_tvt(p,ta)       # best path the grid allows
        base=d['z']+d['tvt0']+(-p['z'][0])                                        # straight baseline (S=S0)
        e=hard-y; ERR[w]=dict(y=y,hard=hard,soft=soft,hand=hand,orc=orc,base=base,md=p['md'],gr=p['gr_raw'],A=d['A'],bnd=d['bnd'],z=d['z'],tvt0=d['tvt0'])
        r=lambda a:float(np.sqrt(np.mean((a-y)**2)))
        rows.append(dict(w=w,n=len(y),hard=r(hard),soft=r(soft),hand=r(hand),oracle=r(orc),base=r(base),sse=float(np.sum(e*e)),
                         maxabs=float(np.abs(e).max()),drift=float(e[-1]-e[0]),truth_range=float(y.max()-y.min()),out_of_grid=float(np.mean(np.abs(Sb-np.r_[ta]) > 1.0))))
import pandas as pd
df=pd.DataFrame(rows).sort_values('sse',ascending=False); tot=df.sse.sum(); df['sse_share']=df.sse/tot; df['cum_share']=df.sse_share.cumsum()
df.to_csv(f'{OUT}/per_well.csv',index=False); pickle.dump(ERR,open(f'{OUT}/paths.pkl','wb'))
N=df.n.sum(); print('overall hard',np.sqrt(tot/N).round(3))
for k in [k for k in [1,3,5,10,20] if k<len(df)]: print(f'drop worst {k}: rmse {np.sqrt(df.sse.iloc[k:].sum()/df.n.iloc[k:].sum()):.3f}  (worst {k} = {df.sse_share.iloc[:k].sum()*100:.1f}% of SSE)')
print(df.head(20).round(2).to_string())
# error vs relative position along lateral
bins=np.linspace(0,1,11); se=np.zeros(10); cnt=np.zeros(10)
for w,v in ERR.items():
    e=(v['hard']-v['y'])**2; pos=np.linspace(0,1,len(e),endpoint=False); b=np.minimum((pos*10).astype(int),9)
    np.add.at(se,b,e); np.add.at(cnt,b,1)
print('rmse by position decile:',np.sqrt(se/cnt).round(2))
# plots: worst 8 + 4 median wells
pick=list(df.w.iloc[:8])+list(df.w.iloc[len(df)//2-2:len(df)//2+2])
for w in pick:
    v=ERR[w]; md=v['md']; fig,ax=plt.subplots(2,1,figsize=(14,8),sharex=True,gridspec_kw=dict(height_ratios=[3,1]))
    for i,b in enumerate(v['bnd']):                    # anchor grid at each boundary (in TVT)
        r=min(b,len(md)-1); a=v['A'][i]+v['z'][r]+v['tvt0']; ax[0].scatter(np.full(len(a),md[r]),a,s=0.3,c='0.85')
    ax[0].plot(md,v['y'],'k',lw=2,label='truth'); ax[0].plot(md,v['orc'],'g--',lw=1,label='grid best fit')
    ax[0].plot(md,v['hard'],'r',lw=1.2,label='pred (hard)'); ax[0].plot(md,v['hand'],'b:',lw=1,label='hand cost'); ax[0].plot(md,v['base'],'0.5',lw=0.8,label='straight baseline')
    lo,hi=v['y'].min()-30,v['y'].max()+30; ax[0].set_ylim(hi,lo); ax[0].set_ylabel('TVT'); ax[0].legend(fontsize=8)
    rr=df.set_index('w').loc[w]; ax[0].set_title(f"{w}  hard {rr.hard:.1f}  hand {rr.hand:.1f}  grid-best {rr.oracle:.2f}  share {rr.sse_share*100:.1f}%")
    ax[1].plot(md,v['hard']-v['y'],'r'); ax[1].axhline(0,c='k',lw=0.5); ax[1].set_ylabel('pred - truth'); ax[1].set_xlabel('MD')
    fig.tight_layout(); fig.savefig(f'{OUT}/well_{w}.png',dpi=80); plt.close(fig)
print('saved',OUT)
