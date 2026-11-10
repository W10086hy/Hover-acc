"""Differentiable onboard loop in a translating cart frame; no truth feedback."""
import copy,hashlib,math
from types import SimpleNamespace
import numpy as np
import torch
from P2.hover_bptt.flexible_environment import FlexibleRopeEnvironment
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec,identity
from P2.hover_bptt.functional import path_cost
from P2.hover_bptt.io import read
from P3.compensation import ResidualFeedforwardInterface
from .frame_dynamics import FrameRopeModel,FramePlantGraph
from .motion import CartMotion
from .environment import wind_trace

ROOT=__import__('pathlib').Path(__file__).resolve().parents[1]


class DifferentiableResidual(ResidualFeedforwardInterface):
    # Keep the exact original feedforward arithmetic and fixed network; enable
    # derivatives through its causal inputs and the Actor's raw commands.
    apply=ResidualFeedforwardInterface.apply.__wrapped__


class WorldWindRecord:
    def __init__(self,trace,seed):self.trace=trace;self.seed=seed
    def at(self,tick):return self.trace[tick,torch.arange(len(tick),device=tick.device)]
    def statistics(self):
        speed=self.trace.norm(dim=-1)
        return [dict(actual_mean_wind_mps=float(speed[:,i].mean()),actual_rms_wind_mps=float(speed[:,i].square().mean().sqrt()),actual_peak_wind_mps=float(speed[:,i].max())) for i in range(speed.shape[1])]


class FrameEnvironment(FlexibleRopeEnvironment):
    def __init__(self,n,*args,protocol,residual_net,compensation,**kwargs):
        self.protocol=copy.deepcopy(protocol);device=kwargs['device'];world=wind_trace(protocol,n)
        if 'training_wind_seeds_per_case' in protocol:
            world=torch.cat([wind_trace({**protocol,'wind_seed':seed},1) for seed in protocol['training_wind_seeds_per_case']],1)
        pc=kwargs['plant_configuration'];control_dt=pc['simulation']['control_dt_s'];physics_dt=pc['simulation']['physics_dt_s'];substeps=round(control_dt/physics_dt)
        steps=len(world);motions=[CartMotion(c['cart_speed_mps'],c.get('cart_motion')) for c in protocol['case_specs']]
        theta=math.radians(protocol['cart_direction_engineering_deg']);direction=np.array([math.cos(theta),math.sin(theta),0.])
        time=np.arange(steps)*control_dt;subtime=time[:,None]+np.arange(substeps)*physics_dt
        velocity=np.stack([m.kinematics(time)[1] for m in motions],1)[:,:,None]*direction
        subvelocity=np.stack([m.kinematics(subtime)[1] for m in motions],1)[:,:,:,None]*direction
        subacceleration=np.stack([m.kinematics(subtime)[2] for m in motions],1)[:,:,:,None]*direction
        acceleration=np.stack([m.kinematics(time)[2] for m in motions],1)[:,:,None]*direction
        # Static tables live in an object, not in the recurrent tensor boundary.
        # Only elapsed indexes them during forward and exact segment replay.
        self._frame=SimpleNamespace(world=world.to(device),velocity=torch.tensor(velocity,dtype=torch.float32,device=device),
            instant=torch.tensor(np.concatenate((world.numpy()-velocity,acceleration),-1),dtype=torch.float32,device=device),
            controls=torch.tensor(np.concatenate((world.numpy()[:,:,None,:]-subvelocity,subacceleration),-1),dtype=torch.float32,device=device),
            limits=torch.tensor([protocol.get('offline_world_speed_limit_mps',max(5.,c['cart_speed_mps']+1.)) for c in protocol['case_specs']],device=device))
        super().__init__(n,*args,**kwargs)
        self.controller=DifferentiableResidual(self.controller,residual_net,compensation)
        # cuDNN needs a training-mode GRU to differentiate its inputs. This
        # one-layer, zero-dropout encoder stays weight-frozen and deterministic.
        self.controller.residual_net.history.train()
        self.training_peak_speed=self.state.new_zeros(n);self.training_peak_radius=self.state.new_zeros(n)
        self.wind_field=WorldWindRecord(self._frame.world,protocol['wind_seed'])
        h=hashlib.sha256(self._frame.controls.cpu().numpy().tobytes()).hexdigest()
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,moving_frame_input_sha256=h,protocol=protocol))

    def wind(self):
        tick=self.elapsed.clamp_max(len(self._frame.instant)-1)
        return self._frame.instant[tick,torch.arange(self.n,device=self.device)]

    def _observe(self,active=None):
        if not isinstance(self.model,FrameRopeModel):
            self.model=FrameRopeModel(self.pc,device=self.device,arms=self.model.arms,lengths=self.model.lengths,rope=self.flexible_rope_options)
            self.model.body_drag=self.model.body_drag[None,:]*self.model.body_drag.new_tensor([c['actual_body_drag_scale'] for c in self.specifications])[:,None]
        return super()._observe(active)

    def metrics(self,wind=None):
        m=super().metrics(wind);velocity=self._frame.velocity[self.elapsed.clamp_max(len(self._frame.velocity)-1),torch.arange(self.n,device=self.device)]
        speed=(self.state[:,3:6]+velocity).norm(dim=-1)
        m['failure_code']=(m['failure_code']&~(1<<3))|((speed>self._frame.limits).long()<<3)
        m['speed']=speed
        return m

    def step(self,raw):
        previous=self.controller.previous
        self.controller.set_observation(self.obs['policy'],self.parameter_context)
        motors,supervision=self.controller.apply(self.packet,raw,self.dt)
        commands=torch.cat((motors,torch.zeros_like(motors[:,:2])),-1)
        external=self._frame.controls[self.elapsed,torch.arange(self.n,device=self.device)]
        self.state=self.model.control(self.state,commands,external);self.elapsed=self.elapsed+1
        self.obs=self._observe();cost=path_cost(self,motors,previous)
        with torch.no_grad():
            metrics=self.metrics();codes=metrics['failure_code']
            failure=((codes&~(8|64))!=0) if self.protocol.get('training_continue_after_performance_boundary',False) else codes!=0
            invalid=~torch.isfinite(raw).all(-1)|~torch.isfinite(self.packet.gyro).all(-1)|~torch.isfinite(self.packet.motor_speed).all(-1)
            done=failure|(self.elapsed>=self.episode_steps)
        # Diagnostic logs are unnecessary in recomputed training segments.
        for name in ['trace_bounded','trace_applied','trace_nominal_motors','trace_motors','trace_fallback']:getattr(self.controller,name).clear()
        return self.obs,cost,done,dict(failure=failure,invalid_policy_action=invalid,fallback=invalid,**metrics)

    def install_cuda_graph(self):
        if not str(self.device).startswith('cuda'):raise ValueError('CUDA qualification requested')
        commands=torch.cat((self.controller.previous,torch.zeros_like(self.controller.previous[:,:2])),-1)
        external=self._frame.controls[self.elapsed,torch.arange(self.n,device=self.device)]
        self.model.control=FramePlantGraph(self.model,self.state,commands,external)


def build_training(protocol,saved,residual_net,device='cuda'):
    stage=copy.deepcopy(saved['configuration']);height=protocol['target_aircraft_height_m']
    stage['environment'].update(initial_xy_m=0.,initial_z_m=0.,initial_tilt_deg=0.,initial_speed_mps=0.,initial_rate_radps=0.)
    stage['controller']['target_height_m']=height
    stage['offline_limits'].update(min_height_m=max(1.,height-4.),max_height_m=height+4.,speed_mps=max(5.,protocol['cart_speed_mps']+1.))
    if 'offline_world_speed_limit_mps' in protocol:stage['offline_limits']['speed_mps']=protocol['offline_world_speed_limit_mps']
    if 'training_recovery_radius_m' in protocol:stage['offline_limits']['horizontal_radius_m']=protocol['training_recovery_radius_m']
    stage['training']['report_credit_progress']=True
    pc=plant_config();rp=read(ROOT/protocol['residual_protocol']);nominal=VehicleSpec.from_config(pc);actual=nominal.scaled(**rp['control_actual_scales'])
    pc['simulation']['target_w_m']=[0.,0.,-height];pc['tether']['anchor_w_m']=[0.,0.,-protocol['cart_anchor_height_m']]
    slack=protocol.get('tether_slack_m')
    if slack is not None and (not math.isfinite(slack) or slack<0):raise ValueError('Finite nonnegative extra tether length required')
    length_offset=-.05 if slack is None else slack-protocol['cart_anchor_height_m']
    cases=[dict(c,initial_condition='hover_capture',attachment=2 if c['attachment_cm']==5 else 1,
        tether_rest_length_m=height+length_offset-c['attachment_cm']/100,preflight_heading_rad=0.,
        wind_speed=c.get('wind_peak_mps',1.),direction_rad=math.radians(protocol['wind_mean_direction_deg']),
        gust_fraction=0.,wind_profile='steady',actual_body_drag_scale=rp['control_actual_drag_scale']) for c in protocol['case_specs']]
    return FrameEnvironment(len(cases),protocol['state_seed'],wind_seed=protocol['wind_seed'],device=device,stage=stage,episode_s=protocol['duration_s'],
        specifications=cases,vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=pc,rope=rp['rope'],
        protocol=protocol,residual_net=residual_net,compensation=rp['compensation'])
