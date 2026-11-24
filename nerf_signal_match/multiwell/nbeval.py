import pickle,numpy as np
R=pickle.load(open('nbpath.pkl','rb'))
def pooled(f):
    s=n=0
    for r in R:
        e=f(r)-r['y'];s+=np.sum(e**2);n+=len(e)
    return np.sqrt(s/n)
print(len(R),'wells')
print('hold TVT %.2f'%pooled(lambda r:np.full_like(r['y'],r['tvt0'])))
for mode in ('md','xyz'):
    for k in range(3):print(mode,'rank',k+1,'%.2f'%pooled(lambda r:r[mode][k]))
    print(mode,'mean of 3 %.2f'%pooled(lambda r:r[mode].mean(0)),'| median of 3 %.2f'%pooled(lambda r:np.median(r[mode],0)))
    def wd(r):
        w=1/(np.array(r['dist'])+50);w/=w.sum();return (w[:,None]*r[mode]).sum(0)
    print(mode,'dist-weighted %.2f'%pooled(wd))
    def orc(r):
        e=[np.mean((p-r['y'])**2) for p in r[mode]];return r[mode][int(np.argmin(e))]
    print(mode,'ORACLE best-of-3 (notebook-style, uses truth) %.2f'%pooled(orc))
# by neighbour distance of rank1
d1=np.array([r['dist'][0] for r in R]);n=np.array([len(r['y']) for r in R])
for lo,hi in ((0,150),(150,400),(400,800),(800,1e9)):
    m=(d1>=lo)&(d1<hi);sub=[r for r,k in zip(R,m) if k]
    s=sum(np.sum((r['xyz'].mean(0)-r['y'])**2) for r in sub);h=sum(np.sum((r['tvt0']-r['y'])**2) for r in sub);nn=sum(len(r['y']) for r in sub)
    print('mean dist to rank1 neighbour %4d-%4s: wells %3d, xyz-mean RMSE %.1f vs hold %.1f'%(lo,hi if hi<1e8 else 'inf',m.sum(),np.sqrt(s/nn),np.sqrt(h/nn)))
