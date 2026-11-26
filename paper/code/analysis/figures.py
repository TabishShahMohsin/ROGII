"""Figures 1-5 of the paper (run after analysis.structure and analysis.wells).   python -m analysis.figures"""
# Data figures for the paper.
import sys, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'figure.dpi':150,'savefig.bbox':'tight'})
from analysis.common import train_wells, load
W=train_wells(); N=sorted(W)
R=load('structure.pkl'); WS=load('wells.pkl')
import os; OUT=os.environ.get('FIGURES','figures')+'/'; os.makedirs(OUT,exist_ok=True)
# ---------- Fig: one well ----------
w='01869cd4'; d=W[w]; K=d['K']; md=d['MD']; Z=d['Z']; tv=d['TVT']; S=tv+Z
mz,cz=np.polyfit(md[K:],Z[K:],1)
fig,ax=plt.subplots(4,1,figsize=(6.8,6.6),sharex=True)
ax[0].plot(md,Z,'k',lw=.8,label='Z (vertical coordinate, < 0 below sea level)'); ax[0].plot(md[K:],mz*md[K:]+cz,'C1--',lw=.8,label='linear trend of Z (prediction zone)')
ax[0].set_ylabel('Z (ft)'); ax[0].legend(loc='lower left',frameon=False)
ax[1].plot(md[:K],tv[:K],'C0',lw=.9,label='TVT, known section'); ax[1].plot(md[K:],tv[K:],'k',lw=.9,label='TVT, to predict')
ax[1].axhline(tv[K-1],color='C3',ls=':',lw=.8,label='last known TVT held'); ax[1].set_ylabel('TVT (ft)'); ax[1].legend(loc='lower right',frameon=False,ncol=1)
P=R[w]['pieces']; b=(K+P[:,0]).astype(int)
ax[2].plot(md,S,'k',lw=.8,label='S = TVT + Z (elevation of the TVT = 0 surface)'); ax[2].plot(md[b],S[b],'o',ms=2.2,mfc='C3',mec='none',label='starts of straight pieces (evaluation zone)')
ax[2].set_ylabel('S (ft)'); ax[2].legend(loc='upper right',frameon=False)
g=d['GR']; ax[3].plot(md,g,color='0.3',lw=.5,label='lateral GR')
tw=np.interp(tv,np.sort(d['tw_TVT']),d['tw_GR'][np.argsort(d['tw_TVT'])]); ax[3].plot(md,tw,'C2',lw=.6,alpha=.8,label='typewell GR at the true TVT')
ax[3].set_ylabel('GR (API)'); ax[3].set_xlabel('MD (ft)'); ax[3].legend(loc='upper right',frameon=False,ncol=2)
for a in ax: a.axvline(md[K-1],color='0.5',lw=.6,ls='--')
fig.align_ylabels(ax); plt.savefig(OUT+'well_example.pdf'); plt.close()
# ---------- Fig: segments ----------
P=np.concatenate([np.c_[R[x]['pieces'],np.full(len(R[x]['pieces']),R[x]['mz'])] for x in N]); L=P[:,2]; m=P[:,3]; mdt=m-P[:,6]
J=np.concatenate([R[x]['junctions'] for x in N])
fig,ax=plt.subplots(1,4,figsize=(9,2.3))
bins=np.logspace(0,np.log10(L.max()+1),40)
ax[0].hist(L,bins=bins,weights=L/L.sum(),color='0.4'); ax[0].set_xscale('log'); ax[0].set_xlabel('piece length (ft)'); ax[0].set_ylabel('share of path length')
ax[0].axvline(80,color='C3',ls='--',lw=.8)
ax[1].scatter(L,np.abs(mdt)+1e-4,s=1,alpha=.15,color='k',rasterized=True); ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].axhline(0.05,color='C3',ls='--',lw=.8); ax[1].set_xlabel('piece length (ft)'); ax[1].set_ylabel('|detrended slope| (ft/ft)')
for arr,lab,c in [(np.abs(m),'raw dS/dMD','C0'),(np.abs(mdt),'relative to Z trend','C1')]:
    o=np.argsort(arr); cum=np.cumsum(L[o])/L.sum(); ax[2].plot(arr[o],cum,color=c,label=lab)
ax[2].axvline(0.05,color='C3',ls='--',lw=.8); ax[2].set_xlim(0,0.12); ax[2].set_xlabel('|slope| (ft/ft)'); ax[2].set_ylabel('cumulative share of length'); ax[2].legend(frameon=False,loc='lower right')
ax[3].hist(np.abs(J[:,0])+1e-4,bins=np.logspace(-3,1.5,50),color='0.4'); ax[3].set_xscale('log'); ax[3].set_yscale('log')
ax[3].axvline(0.1,color='C3',ls='--',lw=.8); ax[3].set_xlabel('|offset| at junction (ft)'); ax[3].set_ylabel('junctions')
plt.tight_layout(); plt.savefig(OUT+'segments.pdf'); plt.close()
# ---------- Fig: map ----------
xy=WS['xy']/5280; fo=WS['fold']
fig,ax=plt.subplots(1,2,figsize=(7,3))
for k in range(5): ax[0].scatter(xy[fo==k,0]-xy[:,0].min(),xy[fo==k,1]-xy[:,1].min(),s=4,label=f'fold {k}')
ax[0].set_aspect('equal'); ax[0].set_xlabel('easting (mi)'); ax[0].set_ylabel('northing (mi)'); ax[0].legend(frameon=False,markerscale=2,fontsize=6,loc='upper left')
ax[1].hist(WS['nn_any'],bins=np.logspace(1.5,4.3,40),alpha=.6,label='nearest other well',color='0.3')
ax[1].hist(WS['nn_oth'],bins=np.logspace(1.5,4.3,40),alpha=.6,label='nearest other-fold well',color='C1')
ax[1].set_xscale('log'); ax[1].set_xlabel('median lateral-to-lateral distance (ft)'); ax[1].set_ylabel('wells'); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(OUT+'map.pdf'); plt.close()
# ---------- Fig: lattice schematic ----------
fig,ax=plt.subplots(figsize=(6.5,2.6)); D=80; KC=8; MV=4
for c in range(6):
    R_=min(KC,int(0.05*D*c)) if c>0 else 0
    ys=np.arange(-KC,KC+1); ax.plot(np.full(len(ys),c*D),ys,'o',ms=2,color='0.75')
rs=np.random.RandomState(3); a0=0
for k in [-4,-2,0,2,4]: ax.plot([0,D],[0,k],color='C0',lw=.6,alpha=.6)
for k in range(-3,4): ax.plot([D,2*D],[2,2+k],color='C0',lw=.4,alpha=.4)
path=[0,2,3,1,1,4]; ax.plot(np.arange(6)*D,path,'C3',lw=1.6,label='chosen path (piecewise linear)')
ax.plot([-160,0],[1.5,0],'k',lw=1.5,label='known section'); ax.axvline(0,color='0.5',ls='--',lw=.6)
ax.set_xlabel('rows (ft of MD) from the lattice start'); ax.set_ylabel(r'anchor of $\tilde{S}$ (ft)'); ax.legend(frameon=False,loc='lower right')
ax.text(D*0.45,-6.5,'moves |m| <= 4 ft per 80 ft',fontsize=7)
plt.tight_layout(); plt.savefig(OUT+'lattice.pdf'); plt.close()
print('ok')
