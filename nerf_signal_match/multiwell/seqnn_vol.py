import pickle, numpy as np, torch, torch.nn as nn, time
from sklearn.model_selection import GroupKFold
torch.set_num_threads(2)
exec(open('stack4.py').read().split("def cv")[0])
keys=old+new+GRF+Z
import pickle as _p
_post=_p.load(open('comp_post.pkl','rb'))
def _vol(r):
    p=_post.get(r['w']); n=len(r['y'])
    return np.zeros((n,33),np.float32) if p is None else (np.log(p+1e-3)+np.log(33)).astype(np.float32)
Xs=[np.c_[np.c_[tuple(r[k] for k in keys)].astype(np.float32),_vol(r)] for r in R]; Ys=[r['y'].astype(np.float32) for r in R]
allX=np.concatenate(Xs); mu=np.median(allX,0); sd=np.percentile(allX,90,0)-np.percentile(allX,10,0)+1e-6
def prep(X): return np.clip((X-mu)/sd,-8,8).astype(np.float32)
Xs=[prep(x) for x in Xs]; D=Xs[0].shape[1]
class Net(nn.Module):
    def __init__(s,h=48,p=0.2):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D,h),nn.GELU(),nn.Dropout(p)); s.gru=nn.GRU(h,h,batch_first=True,bidirectional=True,num_layers=1); s.out=nn.Sequential(nn.Linear(2*h+h,h),nn.GELU(),nn.Dropout(p),nn.Linear(h,1))
    def forward(s,x):
        h=s.inp(x); g,_=s.gru(h); return s.out(torch.cat([g,h],-1)).squeeze(-1)*10.0
def batchify(idx):
    L=max(len(Xs[i]) for i in idx); X=np.zeros((len(idx),L,D),np.float32); Y=np.zeros((len(idx),L),np.float32); M=np.zeros((len(idx),L),np.float32)
    for j,i in enumerate(idx): n=len(Xs[i]); X[j,:n]=Xs[i]; Y[j,:n]=Ys[i]; M[j,:n]=1
    return torch.tensor(X),torch.tensor(Y),torch.tensor(M)
def train(tr,epochs=40,seed=0,bs=16):
    torch.manual_seed(seed); np.random.seed(seed); net=Net(); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-2); sch=torch.optim.lr_scheduler.OneCycleLR(opt,4e-3,total_steps=epochs*((len(tr)+bs-1)//bs))
    for ep in range(epochs):
        net.train(); perm=np.random.permutation(tr)
        for b in range(0,len(perm),bs):
            X,Y,M=batchify(perm[b:b+bs]); loss=(((net(X)-Y)**2)*M).sum()/M.sum(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
    return net
def predict(net,idx):
    net.eval(); out={}
    with torch.no_grad():
        for b in range(0,len(idx),32):
            X,Y,M=batchify(idx[b:b+32]); P=net(X).numpy()
            for j,i in enumerate(idx[b:b+32]): out[i]=P[j,:len(Xs[i])]
    return out
t0=time.time(); oof={}; wells=np.arange(len(R)); gk=GroupKFold(5)
for f,(tr,te) in enumerate(gk.split(wells,groups=wells)):
    ps=[]
    for s in range(2): net=train(tr,seed=s); ps.append(predict(net,te))
    for i in te: oof[i]=np.mean([p[i] for p in ps],0)
    print("fold",f,round(time.time()-t0),"s",flush=True)
pred=np.concatenate([oof[i] for i in range(len(R))]); yy=np.concatenate(Ys)
print("seqNN (bi-GRU) OOF pooled RMSE %.3f"%np.sqrt(np.mean((pred-yy)**2)))
hgb=pickle.load(open('oof_stackA.pkl','rb')); print("HGB %.3f | avg(HGB,NN) %.3f"%(np.sqrt(np.mean((hgb-yy)**2)),np.sqrt(np.mean((0.5*hgb+0.5*pred-yy)**2))))
pickle.dump(pred,open('oof_seqnn_vol.pkl','wb'))
