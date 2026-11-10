"""Differentiable translating-frame cable dynamics for prescribed cart motion.

State positions/velocities are relative to the translating cart; orientation
is still in the inertial launch axes. Physical relative air flow uses w-vcart.
Both aircraft and each massive cable node receive the fictitious force -m*a.
Sensor loads remain physical inertial specific force, without that fictitious
force. No cart quantity is exposed to the policy or nominal onboard model.
"""
import torch
from P2.hover_bptt.flexible_rope import FlexibleRopeModel,qdot
from P2.hover_bptt.model import ExactWingOffBodyDragModel


class FrameRopeModel(FlexibleRopeModel):
    def loads(self,x,external):
        # The plant's external input has six columns: effective ambient wind
        # (world wind minus cart velocity), then prescribed frame acceleration.
        wind=external[:,:3]
        return super().loads(x,wind)

    def forward(self,x,u,external):
        wind=external[:,:3];acceleration=external[:,3:6];h=self.dt;q,v=self.nodes(x)
        mid,dm=self.actuator_at(x,u,h/2);end,de=self.actuator_at(x,u,h)
        stage=torch.cat((x[:,:13],mid,dm,x[:,19:]),-1)
        force,torque,_,_=ExactWingOffBodyDragModel.loads(self,stage,wind)
        cable=self.rope_loads(x,wind)
        body_force=force-self.mass*acceleration
        node_force=cable['node_external_force']-self.node_mass[:,:,None]*acceleration[:,None]
        forces=torch.cat((body_force,torque-torch.linalg.cross(x[:,10:13],x[:,10:13]*self.inertia),node_force.flatten(1)),-1)
        velocity,tension,_=self.velocity_step(x,forces,wind,load_data=cable)
        attitude=x[:,6:10]+h*qdot(x[:,6:10],velocity[:,3:6]);attitude=attitude/attitude.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        nodes=q+h*velocity[:,6:].reshape_as(q)
        body=torch.cat((x[:,:3]+h*velocity[:,:3],velocity[:,:3],attitude,velocity[:,3:6],end,de),-1)
        y=torch.cat((body,nodes.flatten(1),velocity[:,6:],x[:,-1:]),-1)
        with torch.no_grad():
            peak=torch.maximum(x[:,-1],torch.maximum(tension.amax(-1),torch.maximum(cable['tension'].amax(-1),self.rope_loads(y,wind)['tension'].amax(-1))))
        return torch.cat((y[:,:-1],peak[:,None]),-1)


class FramePlantGraph:
    """The existing recurrent VJP implementation with an explicit6D input ABI."""
    def __new__(cls,model,state,commands,external):
        from .frame_graph import SixChannelPlantGraph
        return SixChannelPlantGraph(model,state,commands,external)
