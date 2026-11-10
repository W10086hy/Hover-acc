"""Analytic prescribed cart motion, restricted to simulation and evaluation."""
import math
import numpy as np


class CartMotion:
    def __init__(self,speed=4.,profile=None):
        self.speed=float(speed);self.profile=dict(profile or {'kind':'constant'})
        self.kind=self.profile['kind'];self.hold=float(self.profile.get('hold_s',0.));self.ramp=float(self.profile.get('ramp_s',0.))
        if self.kind not in ['constant','half_cosine_ramp']:raise ValueError('Unknown cart motion')
        if not all(math.isfinite(v) and v>=0 for v in [self.speed,self.hold,self.ramp]):raise ValueError('Finite nonnegative motion parameters required')
        if self.kind=='half_cosine_ramp' and self.ramp<=0:raise ValueError('A smooth start needs positive ramp duration')

    def kinematics(self,time):
        t=np.asarray(time,dtype=float)
        if not np.isfinite(t).all() or (t<0).any():raise ValueError('Nonnegative finite simulation time required')
        if self.kind=='constant':return self.speed*t,np.full_like(t,self.speed),np.zeros_like(t)
        r=np.clip(t-self.hold,0.,self.ramp);angle=np.pi*r/self.ramp
        position=.5*self.speed*(r-self.ramp/np.pi*np.sin(angle))+self.speed*np.maximum(t-self.hold-self.ramp,0.)
        velocity=.5*self.speed*(1.-np.cos(angle))
        acceleration=np.where((t>self.hold)&(t<self.hold+self.ramp),.5*self.speed*np.pi/self.ramp*np.sin(angle),0.)
        return position,velocity,acceleration

    def phases(self,duration):
        if self.kind=='constant':return {'hover' if self.speed==0 else 'cruise':[0.,float(duration)]}
        return {name:[a,min(b,float(duration))] for name,a,b in
            [('hover',0.,self.hold),('acceleration',self.hold,self.hold+self.ramp),('cruise',self.hold+self.ramp,float(duration))]
            if a<float(duration) and b>a}
