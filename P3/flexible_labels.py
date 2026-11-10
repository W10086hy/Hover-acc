"""Structural body-equation error at common external cable traction.

Cable nodes belong to each model's material geometry. Reusing actual nodes in
a different nominal rest geometry creates an artificial spring force. Compare
vehicle parameters and attachment lever arms at the same external endpoint
force instead. This force is an offline label quantity, never a neural input.
This label does not identify unknown rope material or distributed shape errors.
"""
import torch
from P2.hover_bptt.dynamics.torch_model import rotations,mv


def structural_wrench_target(actual,nominal,state,wind):
    observed=actual.mechanical_rhs(state,wind)
    predicted=nominal.mechanical_rhs(state,wind)
    physical=actual.rope_loads(state,wind)
    reference=nominal.rope_loads(state,wind)
    R=rotations(state[:,6:10]);body_force=mv(R.transpose(1,2),physical['endpoint_force'])
    common_nominal_torque=torch.linalg.cross(nominal.arms.expand(len(state),3),body_force)
    nominal_acceleration=predicted[:,3:6]+(physical['endpoint_force']-reference['endpoint_force'])/nominal.mass
    nominal_angular=predicted[:,10:13]+(common_nominal_torque-reference['endpoint_torque'])/nominal.inertia
    return torch.cat((nominal.mass*mv(R.transpose(1,2),observed[:,3:6]-nominal_acceleration),
                      nominal.inertia*(observed[:,10:13]-nominal_angular)),-1)
