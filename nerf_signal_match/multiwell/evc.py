import pickle,numpy as np,pandas as pd,glob,os
df=pd.read_pickle('fail2.pkl'); bad=set(df[df.rm>15].index); perm=np.random.RandomState(0).permutation(773); sub=set(int(w) for w in perm[500:540])
base={o['w']:o for o in pickle.load(open('vensF_base.pkl','rb'))}
cf=['base','g30k','g300k','shift']; B={c:{o['w']:o for o in pickle.load(open(f'vensF_{c}.pkl','rb'))} for c in cf}; bag=lambda w:np.mean([B[c][w]['p'] for c in cf],0)
def rm(f,ws):
    se=n=0
    for w in ws: y=base[w]['y']; se+=((f(w)-y)**2).sum(); n+=len(y)
    return np.sqrt(se/n)
print('%-26s bad %.2f sub40 %.2f'%('base viterbi',rm(lambda w:base[w]['p'],bad),rm(lambda w:base[w]['p'],sub)))
print('%-26s bad %.2f sub40 %.2f'%('bag4',rm(bag,bad),rm(bag,sub)))
for f in sorted(glob.glob('vm_c_*.pkl')):
    D={o['w']:o for o in pickle.load(open(f,'rb'))}; nm=f[5:-4]
    print('%-26s bad %.2f sub40 %.2f'%(nm,rm(lambda w:D[w]['p'],bad),rm(lambda w:D[w]['p'],sub)),' | mean w/ bag4: bad %.2f sub %.2f'%(rm(lambda w:(D[w]['p']+bag(w))/2,bad),rm(lambda w:(D[w]['p']+bag(w))/2,sub)))
