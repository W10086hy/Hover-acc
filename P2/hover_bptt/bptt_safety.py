"""Offline differentiable cost near declared simulation safety boundaries.

No desired pose: attitude penalty is zero below70%of the hard tilt bound.
These truth quantities are used only by the training loss, never by Actor.
"""
import math
import torch
from .dynamics.torch_model import rotations


def safety_penalty(env):
    x=env.state;limits=env.cfg['offline_limits'];soft=.7
    height=-x[:,2];target=-env.model.target[2]
    squares=[x[:,10:13].square().sum(-1)/limits['rate_radps']**2,
        x[:,3:6].square().sum(-1)/limits['speed_mps']**2,
        x[:,:2].square().sum(-1)/limits['horizontal_radius_m']**2,
        ((height-target).clamp_min(0)/(limits['max_height_m']-target)).square(),
        ((target-height).clamp_min(0)/(target-limits['min_height_m'])).square()]
    tension=env.model.loads(x,env.wind())[3]
    squares.append((tension/limits['tension_n']).square())
    penalty=torch.relu(torch.stack(squares,-1)-soft**2).square().sum(-1)
    angle=math.radians(limits['tilt_deg']);csoft=math.cos(soft*angle);chard=math.cos(angle)
    tilt=torch.relu((csoft-rotations(x[:,6:10])[:,2,2])/(csoft-chard)).square()
    return penalty+tilt
