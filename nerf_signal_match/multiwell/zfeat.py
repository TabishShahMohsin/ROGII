import pickle, numpy as np
from common import W,NAMES
from scipy.ndimage import gaussian_filter1d
R=pickle.load(open('ds5.pkl','rb'))
def zf(r):
    d=W[NAMES[r['w']]]; K=d['K']; md=d['MD']; z=d['Z'].astype(np.float64); x=d['X']; y=d['Y']; gi=r['idx']; n=len(gi); out={}
    inc=np.gradient(z,md)              # dZ/dMD  (-1 = vertical down; 0 horizontal)
    for s in [50,200,600]:
        sm=gaussian_filter1d(inc,s); out[f'inc{s}']=sm[gi]; 
        out[f'curv{s}']=np.gradient(sm,md)[gi]*1000
    zs=gaussian_filter1d(z,300); out['z_minus_smooth']=(z-zs)[gi]; out['z_minus_smooth600']=(z-gaussian_filter1d(z,900))[gi]
    L=len(md); out['dz_to_end']=z[-1]-z[gi]; out['md_to_end']=md[-1]-md[gi]; out['dz_since_K']=z[gi]-z[K-1]
    # running extrema of Z since anchor (driller turning points)
    zk=z[K-1:]; cmax=np.maximum.accumulate(zk); cmin=np.minimum.accumulate(zk); ii=gi-(K-1)
    out['z_from_runmax']=(z[gi]-cmax[ii]); out['z_from_runmin']=(z[gi]-cmin[ii])
    # azimuth change (turns)
    az=np.unwrap(np.arctan2(np.gradient(y,md),np.gradient(x,md))); out['az_dev']=(gaussian_filter1d(az,100)-az[K-1])[gi]
    # forward-looking inclination change: mean inc over next 300 minus previous 300
    c=np.r_[0,np.cumsum(inc)]; fw=np.array([(c[min(i+300,L)]-c[i])/max(min(i+300,L)-i,1) for i in gi]); bw=np.array([(c[i]-c[max(i-300,0)])/max(i-max(i-300,0),1) for i in gi]); out['inc_fwd']=fw; out['inc_bwd']=bw; out['inc_change']=fw-bw
    return out
for r in R: r.update(zf(r))
pickle.dump(R,open('ds6.pkl','wb')); print("ok",len(R))
