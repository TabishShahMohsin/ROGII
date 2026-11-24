# Port of the user's 'ZTVT Near Wells' notebook: neighbour lateral Z+TVT projected onto the target lateral, aligned at the heel
import pickle,numpy as np,time
from scipy.spatial import cKDTree
W=pickle.load(open('wells.pkl','rb'));N=sorted(W)
LAT={};ST=[]
for w in N:
    d=W[w];K=d['K'];LAT[w]=dict(MD=d['MD'][K:].astype(float),XYZ=np.c_[d['X'][K:],d['Y'][K:],d['Z'][K:]].astype(float),S=(d['TVT'][K:]+d['Z'][K:]).astype(float),K=K)
    ST.append([d['X'][K],d['Y'][K]])
ST=np.array(ST);STREE=cKDTree(ST)
TREE3={}
def t3(w):
    if w not in TREE3:TREE3[w]=cKDTree(LAT[w]['XYZ'])
    return TREE3[w]
def similarity(wa,wb):   # neighbour a vs target b, as in calculate_well_similarity(e2,e1): mean over A-points of min dist to B
    A=LAT[wa]['XYZ'];B=LAT[wb]['XYZ'];dist,_=t3(wb).query(A);va=A[-1]-A[0];vb=B[-1]-B[0]
    return float(dist.mean()),float(np.dot(va/np.linalg.norm(va),vb/np.linalg.norm(vb)))
def project(wt,wn,mode):
    T=LAT[wt];Nn=LAT[wn]
    if mode=='md':
        tr=cKDTree(Nn['MD'][:,None]);dist,ix=tr.query(T['MD'][:,None])
    else:
        dist,ix=t3(wn).query(T['XYZ'])
    return Nn['S'][ix],dist
def run(w,kc=10,kn=3,nal=30):
    d=W[w];K=d['K'];T=LAT[w];_,cand=STREE.query(ST[N.index(w)],k=kc+1);cand=[N[i] for i in cand if N[i]!=w][:kc]
    res=[]
    for c in cand:
        dist,par=similarity(c,w);res.append((c,dist,par))
    pos=sorted([r for r in res if r[2]>0],key=lambda r:r[1]);neg=sorted([r for r in res if r[2]<=0],key=lambda r:r[1]);top=(pos+neg)[:kn]
    S0=float(d['TVT_input'][K-1]+d['Z'][K-1]);Zl=T['XYZ'][:,2];out=dict(w=w,nb=[r[0] for r in top],dist=[r[1] for r in top],par=[r[2] for r in top])
    m1,c1=np.polyfit(T['MD'],Zl,1)
    for mode in ('md','xyz'):
        P=[];D=[]
        for c,_,_ in top:
            Sp,dd=project(w,c,mode);Sp=Sp-Sp[:nal].mean()+S0     # align projected Z+TVT at heel with known Z+TVT at K-1
            P.append(Sp-Zl);D.append(dd)                              # -> TVT prediction
        out[mode]=np.array(P);out[mode+'_d']=np.array(D)
    out['y']=d['TVT'][K:];out['tvt0']=float(d['TVT_input'][K-1]);return out
if __name__=="__main__":
    t0=time.time();res=[]
    for i,w in enumerate(N):
        try:res.append(run(w))
        except Exception as e:print('err',w,e)
        if i%100==0:print(i,'%.0fs'%(time.time()-t0),flush=True)
    pickle.dump(res,open('nbpath.pkl','wb'));print('done',len(res))
