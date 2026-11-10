import torch
from torch import nn
from .dynamics.torch_model import rotations
def body_tilt_current_features(observation, indices):
    """Select sensor features with heading-independent roll/pitch tilt entries.

    Quaternion-vector x/y components rotate with half the world yaw angle; they
    are not body roll/pitch errors at arbitrary headings. Half the body vertical
    components give sin(roll)*cos(pitch)/2 and sin(pitch)/2 instead. This changes
    the feature semantics and must be enabled by an explicit new configuration,
    with fresh training or a separately measured transfer. It prescribes no
    target tilt: all downstream gains remain learned/trainable.
    """
    if observation.shape[-1] != 35 or tuple(indices[:3]) != (1,2,3):
        raise ValueError('Expected35onboard inputs and quaternion-vector selection')
    quaternion=observation[...,:4]
    quaternion=quaternion/quaternion.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    w,x,y,z=quaternion.unbind(-1)
    selected=observation[...,indices]
    return torch.cat((torch.stack((w*x+y*z,w*y-x*z),-1),selected[...,2:]),-1)

def body_frame_current_features(observation,indices,gravity_mps2):
    """Body tilt plus causal IMU/leaky-history proxies in the same body frame.

    Horizontal histories remain biased, leaky integrals, not navigation truth.
    Vertical displacement/velocity use only the existing barometric estimates.
    The acceleration transformation uses nominal gravity, not simulated loads.
    """
    expected=(1,2,3,4,5,6,18,19,20,21,22,23,24,25,*range(26,35))
    if tuple(indices)!=expected:raise ValueError('Explicit23column body-frame branch required')
    features=body_tilt_current_features(observation,indices)
    q=observation[...,:4]
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    r=rotations(q.reshape(-1,4)).reshape(*q.shape[:-1],3,3)
    body_acceleration=observation[...,7:10]*20+gravity_mps2*r[...,2,:]
    launch_velocity=torch.cat((observation[...,28:30]*3,-observation[...,19:20]*3),-1)
    launch_position=torch.cat((observation[...,30:32]*5,observation[...,18:19]*2),-1)
    body_velocity=torch.matmul(r.transpose(-1,-2),launch_velocity.unsqueeze(-1)).squeeze(-1)
    body_position=torch.matmul(r.transpose(-1,-2),launch_position.unsqueeze(-1)).squeeze(-1)
    return torch.cat((features[...,:14],body_acceleration[...,:2]/3,
        body_velocity[...,:2]/3,body_position[...,:2]/5,features[...,20:]),-1)

class RecurrentLayer(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.rnn=nn.GRU(35,hidden)

class Actor(nn.Module):
    indices=(1,2,3,4,5,6,18,19,20,21,22,23,24,25,*range(26,35))
    def __init__(self,model):
        super().__init__();hidden=model['gru_hidden'];actions=model.get('action_dim',4)
        if actions not in (3,4):raise ValueError('Expected three acceleration or four rate/thrust outputs')
        widths=[hidden,*model['hidden_dims'],actions];layers=[]
        for i in range(len(widths)-1):
            layers.append(nn.Linear(widths[i],widths[i+1]))
            if i<len(widths)-2:layers.append(nn.ELU())
        self.mlp=nn.Sequential(*layers);self.rnn=RecurrentLayer(hidden);self.fast_head=nn.Linear(23,actions)
        self.obs_normalizer=nn.Identity();self.gravity=model['gravity_mps2'];self.proxy_scale=model['proxy_scale']
        self.conditioning=None
        if model.get('conditioning',{}).get('enabled',False):
            from .conditioning import ParameterEncoder
            self.conditioning=ParameterEncoder(hidden,model['conditioning'].get('width',32))
    def motor_mean(self,features,recurrent):
        current=body_frame_current_features(features,self.indices,self.gravity)
        current=torch.cat((current[:,:8],current[:,8:]*self.proxy_scale),-1)
        return self.mlp(recurrent)+self.fast_head(current)
    def forward(self,features,hidden=None,parameter_context=None):
        if features.shape[-1]!=35:raise ValueError('Actor accepts only35 onboard features')
        latent,hidden=self.rnn.rnn(features[None],hidden)
        recurrent=latent[0]
        if self.conditioning is not None:
            if parameter_context is None or parameter_context.shape[0]!=features.shape[0]:
                raise ValueError('Conditioned Actor requires explicit calibrated vehicle context')
            recurrent=recurrent+self.conditioning(parameter_context)
        return self.motor_mean(features,recurrent),hidden

def actor_mean(actor,features,hidden=None,parameter_context=None):return actor(features,hidden,parameter_context)
