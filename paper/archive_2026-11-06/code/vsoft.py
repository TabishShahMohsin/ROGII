# Soft (differentiable) segment Viterbi: forward-backward over the anchor lattice, expected S path -> TVT -> MSE.
import sys, os, time, json, pickle, numpy as np, torch, torch.nn as nn
from lat import *
torch.set_num_threads(int(os.environ.get('NT',2)))
CFG=json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
CH=CFG.get('chunk',80); NP=CFG.get('np',16); EXTB=CFG.get('eb',1); EXTF=CFG.get('ef',1); MM=0.05; CAP=CFG.get('cap',100.)
from common import NBP
NBF=pickle.load(open(NBP,'rb'))
def build(w):
    """precompute per-chunk segment features (numpy) for one well; cached to disk"""
    p=prep(w,CH,CH); A=anchors(p,1.0,MM,CAP); n=p['n']; b=p['bnd']; grc=np.r_[0,np.cumsum(p['gr'])]
    nb=np.asarray(NBF[w],float); i_=np.arange(len(nb)); m_=np.isfinite(nb); nb=np.interp(i_,i_[m_],nb[m_]); Snb=nb-p['z']-p['tvt0']
    chunks=[]
    for i in range(len(b)-1):
        s0,s1=b[i],b[i+1]; L=s1-s0
        a=A[i][:,None]; bb=A[i+1][None,:]; ok=np.abs(bb-a)/L<=MM+1e-9; ia,ib=np.nonzero(ok)
        av=A[i][ia]; bv=A[i+1][ib]; m=(bv-av)/L
        NPE=NP*(EXTB+1+EXTF); te=s0+np.linspace(-EXTB*L,(EXTF+1)*L-1,NPE); te=np.clip(te,0,n-1); lo,hi=s0-EXTB*L,s1+EXTF*L; tei=np.round(te).astype(int)
        hw=max(1,int((hi-lo)//(2*NPE))); l_=np.clip(tei-hw,0,n); r_=np.clip(tei+hw+1,0,n); ge=(grc[r_]-grc[l_])/np.maximum(r_-l_,1)
        Se=av[:,None]+m[:,None]*(te[None,:]-s0); tv=Se+p['z'][tei][None,:]+p['tvt0']
        tw=np.interp(tv.ravel(),p['tw_t'],p['tw_g']).reshape(tv.shape); g=np.broadcast_to(ge[None,:],tw.shape)
        inc=np.zeros(NPE,np.float32); inc[NP*EXTB:NP*(EXTB+1)]=1
        hand=(((g-tw)**2)*inc[None,:]).sum(1)/max(inc.sum(),1)
        mnb=(Snb[s1-1]-Snb[s0])/max(L-1,1); offnb=(av+bv)/2-(Snb[s0]+Snb[s1-1])/2
        scal=np.stack([m*20,(m-mnb)*20,np.full_like(m,s0/4000),offnb/20,hand,((g-tw)**2).mean(1)],1).astype(np.float32)
        dnb=(Se-Snb[tei][None,:])/10.
        chunks.append(dict(ia=ia,ib=ib,seq=np.stack([g,tw,g-tw,dnb],1).astype(np.float16),scal=scal,L=L))
    return dict(A=A,chunks=chunks,bnd=b,z=p['z'],tvt0=p['tvt0'],tvt=p['tvt'],n=n)
H=CFG.get('h',12); BIG=CFG.get('big',0); NBC=CFG.get('nbch',1)
class Cost(nn.Module):
    def __init__(s,h=H):
        super().__init__()
        cin=4+NBC
        if BIG:
            s.c1=nn.Conv1d(cin,h,5,padding=2); s.c2=nn.Conv1d(h,h,3,padding=2,dilation=2); s.c3=nn.Conv1d(h,h,3,padding=4,dilation=4)
            s.mlp=nn.Sequential(nn.Linear(3*h+6,64),nn.GELU(),nn.Dropout(CFG.get('drop',0.1)),nn.Linear(64,1))
        else:
            s.conv=nn.Sequential(nn.Conv1d(cin,h,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(h,h,3,padding=1),nn.ReLU())
            s.mlp=nn.Sequential(nn.Linear(2*h+6,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(s.mlp[-1].weight); nn.init.zeros_(s.mlp[-1].bias)
        s.w=nn.Parameter(torch.tensor([1.0])); s.logT=nn.Parameter(torch.tensor(float(CFG.get('logT0',0.0))))
    def forward(s,seq,scal,inc):
        if not NBC: seq=seq[:,:3]
        x=torch.cat([seq,inc[None,None,:].expand(len(seq),1,-1)],1)
        if BIG:
            f=nn.functional.gelu; h=f(s.c1(x)); h=h+f(s.c2(h)); h=h+f(s.c3(h))
            cur=(h*inc).sum(2)/inc.sum()
            z=torch.cat([h.mean(2),h.amax(2),cur,scal],1)
        else:
            h=s.conv(x); z=torch.cat([h.mean(2),h.amax(2),scal],1)
        return s.w[0]*scal[:,4]+s.mlp(z)[:,0]
def lattice_costs(net,d,use_net=True):
    Cs=[]
    for c in d['chunks']:
        seq=torch.from_numpy(c['seq'].astype(np.float32)); scal=torch.from_numpy(c['scal'])
        NPE=seq.shape[2]; inc=torch.zeros(NPE); inc[NP*EXTB:NP*(EXTB+1)]=1
        if use_net and torch.is_grad_enabled() and CFG.get('ckpt',1):
            from torch.utils.checkpoint import checkpoint
            v=checkpoint(net,seq,scal,inc,use_reentrant=False)
        else: v=net(seq,scal,inc) if use_net else scal[:,4]
        M=torch.full((len(d['A'][len(Cs)]),len(d['A'][len(Cs)+1])),1e4)
        M[c['ia'],c['ib']]=v*(c['L']/CH); Cs.append(M)
    return Cs

MAP=CFG.get('map',0); DS=CFG.get('ds',4); KC=int(CAP)
def build_map(w):
    p=prep(w,CH,CH); A=anchors(p,1.0,MM,CAP); n=p['n']; b=p['bnd']; S0=-p['z'][0]
    nb=np.asarray(NBF[w],float); i_=np.arange(len(nb)); m_=np.isfinite(nb); nb=np.interp(i_,i_[m_],nb[m_]); Snb=nb-p['z']-p['tvt0']
    rd=np.arange(DS//2,n,DS); off=np.arange(-KC,KC+1).astype(float); Sg=S0+off
    grc=np.r_[0,np.cumsum(p['gr'])]; l_=np.clip(rd-DS//2,0,n); r_=np.clip(rd+DS-DS//2,0,n); g=(grc[r_]-grc[l_])/np.maximum(r_-l_,1)
    tv=Sg[:,None]+p['z'][rd][None,:]+p['tvt0']; tw=np.interp(tv.ravel(),p['tw_t'],p['tw_g']).reshape(tv.shape)
    G=np.broadcast_to(g[None,:],tw.shape); dnb=(Sg[:,None]-Snb[rd][None,:])/10.
    img=np.stack([G,tw,G-tw,dnb]).astype(np.float32)
    chunks=[]
    for i in range(len(b)-1):
        s0,s1=b[i],b[i+1]; L=s1-s0
        a=A[i][:,None]; bb=A[i+1][None,:]; ok=np.abs(bb-a)/L<=MM+1e-9; ia,ib=np.nonzero(ok)
        av=A[i][ia]; bv=A[i+1][ib]; m=(bv-av)/L
        J=np.nonzero((rd>=s0)&(rd<s1))[0]; t=rd[J]-s0
        kk=(av[:,None]+m[:,None]*t[None,:])-S0+KC; kk=np.clip(kk,0,2*KC-1e-4); k0=np.floor(kk).astype(np.int32); fr=(kk-k0).astype(np.float32)
        mnb=(Snb[s1-1]-Snb[s0])/max(L-1,1); offnb=(av+bv)/2-(Snb[s0]+Snb[s1-1])/2
        scal=np.stack([m*20,(m-mnb)*20,offnb/20],1).astype(np.float32)
        chunks.append(dict(ia=ia,ib=ib,J=J,k0=k0,fr=fr,scal=scal,L=L))
    return dict(A=A,chunks=chunks,bnd=b,z=p['z'],tvt0=p['tvt0'],tvt=p['tvt'],n=n,img=img)
class MapCost(nn.Module):
    def __init__(s,h=H):
        super().__init__()
        s.c0=nn.Conv2d(4,h,3,padding=1)
        s.blk=nn.ModuleList([nn.Conv2d(h,h,3,padding=(dk,dr),dilation=(dk,dr)) for dk,dr in [(1,1),(2,2),(4,4),(8,8),(1,16),(2,32)][:CFG.get('nl',5)]])
        s.out=nn.Conv2d(h,1,1); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
        s.lin=nn.Linear(3,1); nn.init.zeros_(s.lin.weight); nn.init.zeros_(s.lin.bias)
        s.w=nn.Parameter(torch.tensor([1.0])); s.logT=nn.Parameter(torch.tensor(float(CFG.get('logT0',0.0))))
    def emap(s,img):
        f=nn.functional.gelu; x=img[None]; h=f(s.c0(x))
        for c in s.blk: h=h+f(c(h))
        return s.out(h)[0,0]
def map_costs(net,d,use_net=True):
    img=torch.from_numpy(d['img']); hand=img[2]**2
    e=net.emap(img) if use_net else None
    Cs=[]
    for i,c in enumerate(d['chunks']):
        J=torch.from_numpy(c['J']).long(); k0=torch.from_numpy(c['k0']).long(); fr=torch.from_numpy(c['fr'])
        def gat(M):
            Mj=M[:,J]; return (Mj.gather(0,k0)*(1-fr)+Mj.gather(0,k0+1)*fr).mean(1) if len(J) else torch.zeros(len(k0))
        if use_net: v=net.w[0]*gat(hand)+gat(e)+net.lin(torch.from_numpy(c['scal']))[:,0]
        else: v=gat(hand)
        M=torch.full((len(d['A'][i]),len(d['A'][i+1])),1e4); M[c['ia'],c['ib']]=v*(c['L']/CH); Cs.append(M)
    return Cs
if MAP:
    build=build_map; Cost=MapCost; lattice_costs=map_costs
def soft_path(Cs,A,T):
    al=[torch.zeros(1)]
    for M in Cs: al.append(-T*torch.logsumexp(-(al[-1][:,None]+M)/T,0))
    be=[torch.zeros(len(A[-1]))]
    for M in reversed(Cs): be.insert(0,-T*torch.logsumexp(-(M+be[0][None,:])/T,1))
    S=[(torch.softmax(-(a+b)/T,0)*torch.from_numpy(Av).float()).sum() for a,b,Av in zip(al,be,A)]
    return torch.stack(S)
def hard_path(Cs,A):
    C=[torch.zeros(1)]; BP=[None]
    for M in Cs: t=C[-1][:,None]+M; v,ix=t.min(0); C.append(v); BP.append(ix)
    k=int(C[-1].argmin()); idx=[k]
    for i in range(len(Cs),0,-1): k=int(BP[i][k]); idx.append(k)
    idx=idx[::-1]; return torch.tensor([A[i][j] for i,j in enumerate(idx)],dtype=torch.float32)
def to_tvt(S,d):
    b=d['bnd']; n=d['n']; rows=torch.arange(n,dtype=torch.float32); out=torch.zeros(n)
    for i in range(len(b)-1):
        s0,s1=b[i],b[i+1]; t=(rows[s0:s1]-s0)/(s1-s0); out[s0:s1]=S[i]+(S[i+1]-S[i])*t
    return out+torch.from_numpy(d['z']).float()+d['tvt0']
net=Cost(); opt=torch.optim.AdamW(net.parameters(),lr=CFG.get('lr',3e-3),weight_decay=CFG.get('wd',0.0))
import copy; ema=copy.deepcopy(net); ema.eval(); EMA=CFG.get('ema',0.0)
def ema_upd():
    with torch.no_grad():
        for a,b in zip(ema.parameters(),net.parameters()): a.mul_(EMA).add_(b,alpha=1-EMA)
rs=np.random.RandomState(0)
FV=CFG.get('fold',0)
tr=[w for i,w in enumerate(NAMES) if FOLD[i]!=FV]; va=[w for i,w in enumerate(NAMES) if FOLD[i]==FV]
if CFG.get('ntr'): tr=list(rs.choice(tr,CFG['ntr'],replace=False))
if CFG.get('nva'): va=list(rs.choice(va,CFG['nva'],replace=False))
class _L(dict):
    def __missing__(s,w): return build(w)
D=_L()
def evaluate(ws,use_net=True,m=None):
    m=m or net; m.eval(); se=dict(soft=0,hard=0); n=0
    with torch.no_grad():
        T=m.logT.exp()
        for w in ws:
            d=D[w]; Cs=lattice_costs(m,d,use_net); y=torch.from_numpy(d['tvt']).float()
            se['soft']+=((to_tvt(soft_path(Cs,d['A'],T),d)-y)**2).sum().item(); se['hard']+=((to_tvt(hard_path(Cs,d['A']),d)-y)**2).sum().item(); n+=len(y)
    return {k:round(np.sqrt(v/n),3) for k,v in se.items()}
CK=os.environ.get("CKDIR","/kaggle/working/")+f"vs_{CFG.get('tag','a')}.pt"; EP0=0
if os.path.exists(CK):
    ck=torch.load(CK,weights_only=False); net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); ema.load_state_dict(ck['ema']); EP0=ck['ep']+1; rs=ck['rs']; BEST=ck['best']; print('resume',EP0,flush=True)
else:
    BEST=1e9
    if not CFG.get('skiphand'): print('hand cost (T=1):',evaluate(va,False),flush=True)
NEP=CFG.get('ep',6); LR=CFG.get('lr',3e-3); NTR=len(tr)
for ep in range(EP0,NEP):
    t=time.time(); L_=[]; net.train()
    for k,w in enumerate(rs.permutation(tr)):
        fr=(ep+k/NTR)/NEP
        for g_ in opt.param_groups: g_['lr']=LR*(CFG.get('lrmin',1.0)+(1-CFG.get('lrmin',1.0))*0.5*(1+np.cos(np.pi*fr)))
        d=D[w]; Cs=lattice_costs(net,d); S=soft_path(Cs,d['A'],net.logT.exp())
        loss=((to_tvt(S,d)-torch.from_numpy(d['tvt']).float())**2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step(); L_.append(loss.item())
        if EMA: ema_upd()
    ev=evaluate(va,m=ema if EMA else net)
    if ev['hard']<BEST: BEST=ev['hard']; torch.save((ema if EMA else net).state_dict(),CK.replace('.pt','_best.pt'))
    torch.save(dict(net=net.state_dict(),opt=opt.state_dict(),ema=ema.state_dict(),ep=ep,rs=rs,best=BEST),CK)
    print(f'ep {ep} train rmse {np.sqrt(np.mean(L_)):.3f} val {ev} best {BEST:.3f} T {net.logT.exp().item():.3f} {time.time()-t:.0f}s',flush=True)
