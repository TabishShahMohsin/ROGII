import pickle,sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from sklearn.model_selection import GroupKFold
torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
R=pickle.load(open('cvol.pkl','rb'));ws=sorted(R);D=np.arange(-48,49,1.0);Dt=torch.tensor(D,dtype=torch.float32);BS=16
def prep(r):
    c1=np.log1p(r['C1']);c2=np.log1p(r['C2']);c1=c1-c1.min(1,keepdims=True);c2=c2-c2.min(1,keepdims=True)
    B=len(c1);dd=np.broadcast_to(D/48.,(B,len(D)));ctx=np.broadcast_to(r['ctx'][:,None,:],(B,len(D),4))
    x=np.concatenate([c1[...,None],c2[...,None],dd[...,None],(dd**2)[...,None],ctx],-1)  # B,d,8
    return torch.tensor(x.transpose(2,1,0).copy(),dtype=torch.float32),torch.tensor(r['off'])   # C,d,B ; B
X={w:prep(R[w]) for w in ws}
class Res(nn.Module):
    def __init__(s,c,dil):
        super().__init__();s.a=nn.Conv2d(c,c,3,padding=(1,dil),dilation=(1,dil));s.b=nn.Conv2d(c,c,3,padding=1)
        s.n1=nn.GroupNorm(4,c);s.n2=nn.GroupNorm(4,c)
    def forward(s,x):return x+s.b(F.relu(s.n2(s.a(F.relu(s.n1(x))))))
class Net(nn.Module):
    def __init__(s,cin=8,c=24):
        super().__init__();s.i=nn.Conv2d(cin,c,3,padding=1);s.r=nn.Sequential(*[Res(c,d) for d in (1,2,4,8,16,1)]);s.o=nn.Conv2d(c,1,1)
    def forward(s,x):
        h=s.r(s.i(x));return s.o(F.relu(h))[:,0]      # N,d,B
def decode(lg):
    p=F.softmax(lg,1);return (p*Dt[None,:,None]).sum(1),p
def step(net,x,off):
    lg=net(x[None]);mu,p=decode(lg);mu=mu[0];tgt=torch.exp(-0.5*((Dt[:,None]-off[None,:])/3.)**2);tgt=tgt/(tgt.sum(0,keepdim=True)+1e-9)
    ce=-(tgt*torch.log(p[0]+1e-9)).sum(0).mean();hub=F.huber_loss(mu,off.clamp(-48,48),delta=5.)
    return hub+0.3*ce
def predict(net,w):
    net.eval()
    with torch.no_grad():mu,_=decode(net(X[w][0][None]));
    net.train();return mu[0].numpy()
if __name__=="__main__":
    EP=int(sys.argv[1]) if len(sys.argv)>1 else 12;CROP=160
    blk=np.array([R[w]['blk'] for w in ws]);oof={};t0=time.time()
    for f,(tr,te) in enumerate(GroupKFold(5).split(ws,groups=blk)):
        net=Net();opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-3);sched=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=EP*len(tr))
        for ep in range(EP):
            tot=0
            for i in np.random.permutation(tr):
                x,off=X[ws[i]];B=x.shape[-1]
                if B>CROP:s=np.random.randint(0,B-CROP);x=x[...,s:s+CROP];off=off[s:s+CROP]
                loss=step(net,x,off);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();sched.step();tot+=loss.item()
            print(f'fold {f} ep {ep} loss {tot/len(tr):.3f} {time.time()-t0:.0f}s',flush=True)
        for i in te:oof[ws[i]]=predict(net,ws[i])
        e=np.concatenate([np.repeat(oof[ws[i]],BS)[:R[ws[i]]['n']]-(R[ws[i]]['y']-R[ws[i]]['ref']) for i in te]);r0=np.concatenate([(R[ws[i]]['y']-R[ws[i]]['ref']) for i in te])
        print(f'== fold {f}: ref RMSE {np.sqrt(np.mean(r0**2)):.2f} -> refined {np.sqrt(np.mean(e**2)):.2f}',flush=True)
        pickle.dump(oof,open('rnet_oof.pkl','wb'))
    e=np.concatenate([np.repeat(oof[w],BS)[:R[w]['n']]-(R[w]['y']-R[w]['ref']) for w in ws]);r0=np.concatenate([(R[w]['y']-R[w]['ref']) for w in ws])
    print('ALL: ref %.2f refined %.2f'%(np.sqrt(np.mean(r0**2)),np.sqrt(np.mean(e**2))))
    for a in (0.25,0.5,0.75):
        e=np.concatenate([a*np.repeat(oof[w],BS)[:R[w]['n']]-(R[w]['y']-R[w]['ref']) for w in ws]);print(' shrink',a,'%.2f'%np.sqrt(np.mean(e**2)))
