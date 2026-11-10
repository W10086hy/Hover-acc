"""NumPy inference with PyTorch's GRU gate order r,z,n. No simulation imports."""
import numpy as np

def rotation(q):
    q=np.asarray(q);q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=q.dtype)

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-80,80)))
def linear(x,w,p): return x@w[p+'.weight'].T+w[p+'.bias']

def gru(x,h,w,p):
    a=x@w[p+'.weight_ih_l0'].T+w[p+'.bias_ih_l0']
    b=h@w[p+'.weight_hh_l0'].T+w[p+'.bias_hh_l0']
    ar,az,an=np.split(a,3);br,bz,bn=np.split(b,3)
    z=sigmoid(az+bz);n=np.tanh(an+sigmoid(ar+br)*bn)
    return (1-z)*n+z*h

def current_features(o,gravity,proxy_scale):
    indices=(1,2,3,4,5,6,18,19,20,21,22,23,24,25,*range(26,35))
    q=o[:4]/np.linalg.norm(o[:4]);w,x,y,z=q;r=rotation(q)
    features=o[list(indices)].copy();features[:2]=[w*x+y*z,w*y-x*z]
    a=o[7:10]*20+gravity*r[2]
    v=r.T@np.r_[o[28:30]*3,-o[19]*3]
    p=r.T@np.r_[o[30:32]*5,o[18]*2]
    features=np.r_[features[:14],a[:2]/3,v[:2]/3,p[:2]/5,features[20:]]
    features[8:]*=proxy_scale
    return features

class Actor:
    def __init__(self,weights,model):
        self.w={k[6:]:v for k,v in weights.items() if k.startswith('actor.')}
        self.model=model;self.hidden=np.zeros(model['gru_hidden'],np.float32)
    def __call__(self,o,context):
        self.hidden=gru(o,self.hidden,self.w,'rnn.rnn')
        h=self.hidden+np.tanh(context@self.w['conditioning.input.weight'].T)@self.w['conditioning.output.weight'].T
        for i in [0,2]:
            h=linear(h,self.w,f'mlp.{i}');h=np.where(h>0,h,np.expm1(np.minimum(h,0)))
        h=linear(h,self.w,'mlp.4')
        return h+linear(current_features(o,self.model['gravity_mps2'],self.model['proxy_scale']),self.w,'fast_head')

class Residual:
    def __init__(self,weights):
        self.w={k[9:]:v for k,v in weights.items() if k.startswith('residual.')}
        self.hidden=np.zeros(128,np.float32)
    def __call__(self,obs,context,motors):
        self.hidden=gru(obs,self.hidden,self.w,'history')
        h=np.r_[obs,self.hidden,context,2*motors-1]
        for i in [0,2]:h=np.tanh(linear(h,self.w,f'encoder.{i}'))
        return self.w['scale']*np.tanh(linear(h,self.w,'output'))
