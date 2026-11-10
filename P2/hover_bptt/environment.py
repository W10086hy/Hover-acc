import json,math,copy,hashlib
from pathlib import Path
import torch
from tensordict import TensorDict
from .dynamics.torch_model import CapturedControl,rotations
from .model import WingOffBodyDragModel,ExactWingOffBodyDragModel
from .math3d import quaternion
from .controller import HeadingHoldInterface
from .acceleration import AccelerationInterface
from .onboard import Sensors,NominalVehicle,ObservationHistory
from .aerodynamics import NominalAerodynamics
from .sensors import YawCalibratedSensors,YawCalibratedFunctionalSensors
from .functional import FunctionalHistory,FunctionalHeight,FunctionalMotion,DifferentiableAerodynamics,path_cost
from .rewards import origin_reward
from .wind import WindField
from .vehicle_spec import VehicleSpec,identity

def plant_config():return json.loads((Path(__file__).parent/'configs/vehicle.json').read_text(encoding='utf-8'))
def settings():return json.loads((Path(__file__).parents[1]/'configuration.json').read_text(encoding='utf-8'))
class RateEnvironment:
    """Actor/supervisor consume Sensors; all world quantities stay on simulator side."""
    def __init__(self, n=128, seed=20261020, device="cuda", stage=None,
                 episode_s=None, auto_reset=True, specifications=None, capture=True,wind_seed=None,functional=False,
                 vehicle_spec=None,nominal_vehicle_spec=None,plant_configuration=None,model_factory=None,random_device=None):
        self.n, self.device = n, device
        self.random_device=str(torch.device(random_device or device))
        if self.random_device not in ('cpu',str(torch.device(device))):
            raise ValueError('Random draws must use CPU or the physical simulation device')
        self.wind_field=None
        self.functional=functional
        self.cfg = settings() if stage is None else stage
        base=plant_config() if plant_configuration is None else copy.deepcopy(plant_configuration)
        if (vehicle_spec is None)!=(nominal_vehicle_spec is None):
            raise ValueError('Actual and calibrated nominal vehicle specifications must be explicit together')
        self.actual_spec=VehicleSpec.from_config(base) if vehicle_spec is None else vehicle_spec
        self.nominal_spec=VehicleSpec.from_config(base) if nominal_vehicle_spec is None else nominal_vehicle_spec
        self.pc=base if vehicle_spec is None else self.actual_spec.apply(base)
        self.nominal_pc=base if nominal_vehicle_spec is None else self.nominal_spec.apply(base)
        self.vehicle_identity=identity(dict(plant=self.pc,nominal=self.nominal_pc,cases=specifications))
        self.parameter_context=None
        conditioning=self.cfg['model'].get('conditioning',{})
        if conditioning.get('enabled',False):
            reference=VehicleSpec.from_dict(conditioning['reference_vehicle'])
            self.parameter_context=torch.tensor(self.nominal_spec.context(reference),device=device).expand(n,-1).clone()
        self.dt = self.pc["simulation"]["control_dt_s"]
        self.episode_steps = round((episode_s or self.cfg["environment"]["training_episode_s"])/self.dt)
        self.auto_reset = auto_reset
        self.specifications = specifications
        if specifications is None or len(specifications)!=n:raise ValueError("Explicit ordered5cm/10cm cases required")
        self.gen = torch.Generator(device=self.random_device).manual_seed(seed)
        self.sensor_gen = torch.Generator(device=self.random_device).manual_seed(seed+1000003)
        attachments = torch.arange(n, device=device) % 2
        if specifications is not None:
            attachments = torch.tensor([s["attachment"] for s in specifications], device=device)
        arms = torch.zeros(n, 3, device=device)
        arms[:, 2] = torch.tensor([.05 if int(a)==2 else .1 for a in attachments],device=device)
        lengths = torch.tensor([s.get('tether_rest_length_m',4.90 if int(a)==2 else 4.85) for s,a in zip(specifications,attachments)],device=device)
        if not bool(torch.isfinite(lengths).all()) or bool((lengths<=0).any()):raise ValueError('Explicit finite positive tether lengths required')
        if any(int(a)not in [1,2]for a in attachments):raise ValueError("Only5cm/10cm tether geometry is retained")
        factory=model_factory or (ExactWingOffBodyDragModel if functional else WingOffBodyDragModel)
        self.model = factory(self.pc,device=device,arms=arms,lengths=lengths)
        scales=[float(c.get("actual_body_drag_scale",1.))for c in specifications]
        if functional or any(v!=1. for v in scales):self.model.body_drag=self.model.body_drag[None,:]*self.model.body_drag.new_tensor(scales)[:,None]
        self.vehicle = NominalVehicle(self.nominal_pc, device)
        control = self.cfg['controller']
        acceleration = control.get('action_mode') == 'launch_acceleration'
        interface = AccelerationInterface if acceleration else HeadingHoldInterface
        if self.cfg['model'].get('action_dim', 4) != interface.action_dim:
            raise ValueError('Actor output width does not match the configured flight interface')
        self.controller = interface(self.vehicle, n, control)
        aerodynamic=NominalAerodynamics(self.nominal_pc["airframe"],self.nominal_pc["aero"],device)
        self.history = ObservationHistory(self.vehicle, n, self.dt,self.cfg["sensors"].get("motion_features"),aerodynamic,
            self.cfg['sensors'].get('angular_prediction_mode','previous_rpm'),target_height_m=self.cfg['controller'].get('target_height_m',5.))
        self.sensors = YawCalibratedSensors(n,device,self.cfg["sensors"],self.sensor_gen)
        self.state = self.model.initial(n)
        self.elapsed = torch.zeros(n, device=device, dtype=torch.long)
        self.wind_speed = torch.zeros(n, device=device)
        self.wind_direction = torch.zeros_like(self.wind_speed)
        self.wind_frequency = torch.zeros_like(self.wind_speed)
        self.wind_phase = torch.zeros_like(self.wind_speed)
        self.gust_fraction = torch.zeros_like(self.wind_speed)
        self.all_mask = torch.ones(n, device=device, dtype=torch.bool)
        self._reset(self.all_mask)
        wind = self.wind()
        command = torch.cat((self.controller.previous, torch.zeros(n, 2, device=device)), -1)
        self.graph = (CapturedControl(self.model, self.state, command,
                      wind[:, None, :].expand(-1, self.model.substeps, -1).contiguous())
                      if device.startswith("cuda") and capture else None)
        self.obs = self._observe()
        self.wind_field=WindField(specifications,wind_seed if wind_seed is not None else seed+1,duration_s=self.episode_steps*self.dt,dt=self.dt,device=self.random_device)
        if self.random_device!=str(torch.device(device)):
            self.wind_field.trace=self.wind_field.trace.to(device)
        # Configuration and exogenous wind remain fixed for an environment.
        # Rebuild the environment when changing them; state/RNG alone cannot
        # reproduce a trajectory under a different wind trace or sensor model.
        self.trajectory_schema_version=2
        self.wind_trace_sha256=hashlib.sha256(self.wind_field.trace.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        self.trajectory_identity=identity(dict(version=self.trajectory_schema_version,
            vehicle_identity=self.vehicle_identity,configuration=self.cfg,n=self.n,
            wind_seed=self.wind_field.seed,wind_trace_sha256=self.wind_trace_sha256,
            wind_dtype=str(self.wind_field.trace.dtype),device=str(torch.device(device)),
            dt_s=self.dt,episode_steps=self.episode_steps,functional=self.functional,
            auto_reset=self.auto_reset))
        if self.random_device!=str(torch.device(device)):
            self.trajectory_identity=identity(dict(base=self.trajectory_identity,random_device=self.random_device))

    def random(self, *shape):
        return torch.rand(*shape, device=self.random_device, generator=self.gen).to(self.device)

    def _base_reset(self, mask):
        c, n = self.cfg["environment"], self.n
        state = self.model.initial(n)
        state[:, :2] += (2*self.random(n, 2)-1)*c["initial_xy_m"]
        state[:, 2] += (2*self.random(n)-1)*c["initial_z_m"]
        euler = (2*self.random(n, 3)-1)*math.radians(c["initial_tilt_deg"])
        euler[:, 2] = (2*self.random(n)-1)*math.pi
        state[:, 6:10] = quaternion(euler)
        state[:, 3:6] = (2*self.random(n, 3)-1)*c["initial_speed_mps"]
        state[:, 10:13] = (2*self.random(n, 3)-1)*c["initial_rate_radps"]
        speed = math.sqrt((self.vehicle.mass*self.vehicle.gravity+self.cfg["controller"]["trim_excess_n"])/(4*self.vehicle.kf))
        state[:, 13:17] = speed
        self.state[mask] = state[mask]
        self.elapsed[mask] = 0
        self.controller.reset(mask)
        self.history.reset(mask)
        self.sensors.reset(mask)
        values = {"wind_speed": self.random(n)*c["max_training_wind_mps"],
                  "wind_direction": self.random(n)*2*math.pi,
                  "wind_phase": self.random(n)*2*math.pi,
                  "wind_frequency": .2+.6*self.random(n),
                  "gust_fraction": torch.full((n,), c["wind_gust_fraction"], device=self.device)}
        if self.specifications is not None:
            values["wind_speed"] = self.wind_speed.new_tensor([s["wind_speed"] for s in self.specifications])
            values["wind_direction"] = self.wind_speed.new_tensor([s["direction_rad"] for s in self.specifications])
            values["gust_fraction"] = self.wind_speed.new_tensor([s["gust_fraction"] for s in self.specifications])
        for key, value in values.items():
            getattr(self, key)[mask] = value[mask]

    def _reset(self,mask):
        self._base_reset(mask)
        if self.specifications is None or any('preflight_heading_rad'not in c for c in self.specifications):raise ValueError('Operator preflight heading must be explicit for every scenario')
        heading=torch.tensor([c['preflight_heading_rad']for c in self.specifications],device=self.device)
        if not bool(torch.isfinite(heading).all()):raise ValueError('Finite preflight heading required')
        rotation=rotations(self.state[:,6:10]);roll=torch.atan2(rotation[:,2,1],rotation[:,2,2]);pitch=torch.asin((-rotation[:,2,0]).clamp(-1,1))
        q=quaternion(torch.stack((roll,pitch,heading),-1))
        self.state[:,6:10]=torch.where(mask[:,None],q,self.state[:,6:10])

    def wind(self):
        if self.wind_field is not None:return self.wind_field.at(self.elapsed)
        t = self.elapsed*self.dt
        ramp = ((t-2)/2).clamp(0, 1)
        speed = self.wind_speed*(1+self.gust_fraction*torch.sin(2*math.pi*self.wind_frequency*t+self.wind_phase))*ramp
        return torch.stack((speed*self.wind_direction.cos(), speed*self.wind_direction.sin(), torch.zeros_like(speed)), -1)

    def _physical_loads(self,wind):
        return self.model.loads(self.state,wind)

    def _observe(self, active=None):
        wind = self.wind()
        _, _, force, tension = self._physical_loads(wind)
        packet = self.sensors.sample(self.state[:, 6:10], self.state[:, 10:13], force,
                                     self.state[:, 13:17], -self.state[:, 2], active)
        policy = self.history.update(packet, self.controller.previous, active)
        # Simulation diagnostics only;the Actor consumes the35D policy tensor.
        privileged = torch.cat((policy, (self.state[:, :3]-self.model.target)/5,
                                self.state[:, 3:6]/3, wind/3, tension[:, None]/10, self.state[:, 6:10], self.model.arms[:, 2:3]), -1)
        self.packet = packet
        return TensorDict({"policy": policy, "privileged": privileged}, [self.n])

    def metrics(self, wind=None):
        if wind is None: wind = self.wind()
        r = rotations(self.state[:, 6:10])
        tilt = torch.acos(r[:, 2, 2].clamp(-1, 1))
        rate = self.state[:, 10:13].square().sum(-1).sqrt()
        speed = self.state[:, 3:6].square().sum(-1).sqrt()
        height = -self.state[:, 2]
        radius = self.state[:, :2].square().sum(-1).sqrt()
        tension = self._physical_loads(wind)[3]
        lim = self.cfg["offline_limits"]
        # These are offline/environment checks, not inputs to the onboard fallback.
        conditions = [~torch.isfinite(self.state).all(-1), tilt > math.radians(lim["tilt_deg"]),
                      rate > lim["rate_radps"], speed > lim["speed_mps"],
                      height < lim["min_height_m"], height > lim["max_height_m"],
                      radius > lim["horizontal_radius_m"], tension > lim["tension_n"]]
        codes = torch.zeros(self.n, device=self.device, dtype=torch.long)
        for i, condition in enumerate(conditions): codes |= condition.long() << i
        return {"tilt": tilt, "rate": rate, "speed": speed, "height": height, "radius": radius,
                "tension": tension, "failure_code": codes}

    def step(self, raw_action):
        previous = self.controller.previous.clone()
        motors, supervision = self.controller.apply(self.packet, raw_action, self.dt)
        commands = torch.cat((motors, torch.zeros(self.n, 2, device=self.device)), -1)
        winds = self.wind()[:, None, :].expand(-1, self.model.substeps, -1).contiguous()
        if self.graph:
            self.state.copy_(self.graph(self.state, commands, winds))
        else:
            self.state.copy_(self.model.control(self.state, commands, winds))
        self.elapsed += 1
        observed = self._observe()
        m = self.metrics()
        failure = m["failure_code"] != 0
        timeout = (self.elapsed >= self.episode_steps) & ~failure
        done = failure | timeout
        r = self.cfg["reward"]
        if r.get("kind") in ("origin_proximity", "height_priority"):
            reward, state_cost = origin_reward(self.state[:, :3]-self.model.target, m["tilt"], m["rate"], m["speed"],
                supervision["normalized_action"], motors-previous, failure, r, self.dt)
        else:
            yaw = torch.atan2(rotations(self.packet.attitude)[:, 1, 0], rotations(self.packet.attitude)[:, 0, 0])
            state_cost = (r["tilt_squared"]*m["tilt"].square()+r["relative_yaw_squared"]*yaw.square()+
                          r["rate_squared"]*m["rate"].square()+
                          r["horizontal_speed_squared"]*self.state[:, 3:5].square().sum(-1)+
                          r["vertical_speed_squared"]*self.state[:, 5].square())
            action_cost = (r["normalized_residual_squared"]*(supervision["residual"]/self.cfg["controller"]["residual_throttle_limit"]).square().mean(-1)+
                           r["command_change_squared"]*((motors-previous)/.03).square().mean(-1))
            reward = torch.nan_to_num(r["alive"]-state_cost-action_cost-r["failure"]*failure, nan=-r["failure"], neginf=-r["failure"])
        info={**m,**supervision,"failure":failure,"timeout":timeout,"state_cost":state_cost,"terminal_observation":observed}
        if self.auto_reset and bool(done.any()):
            self._reset(done)
            observed = self._observe(done)
        self.obs = observed
        return observed, reward, done, info
class DifferentiableRateEnvironment(RateEnvironment):
    def __init__(self,*args,**kwargs):
        kwargs.update(capture=False,auto_reset=False,functional=True)
        super().__init__(*args,**kwargs)
        # Reclass existing state so RNG streams, delayed sensor histories and
        # initialization are identical, including the first observed packet.
        self.sensors.__class__=YawCalibratedFunctionalSensors
        self.history.__class__=FunctionalHistory
        self.history.height.__class__=FunctionalHeight
        if self.history.motion:
            self.history.motion.__class__=FunctionalMotion
            self.history.motion.aerodynamics.__class__=DifferentiableAerodynamics

    def step(self,raw_action):
        invalid=(~torch.isfinite(raw_action).all(-1)|~torch.isfinite(self.packet.gyro).all(-1)
                 |~torch.isfinite(self.packet.motor_speed).all(-1)|~torch.isfinite(self.packet.attitude).all(-1))
        previous=self.controller.previous
        motors,_=self.controller.apply(self.packet,raw_action,self.dt)
        commands=torch.cat((motors,torch.zeros_like(motors[:,:2])),-1)
        winds=self.wind()[:,None,:].expand(-1,self.model.substeps,-1).contiguous()
        self.state=self.model.control(self.state,commands,winds)
        self.elapsed=self.elapsed+1
        self.obs=self._observe()
        cost=path_cost(self,motors,previous)
        with torch.no_grad():
            metrics=self.metrics();failure=metrics['failure_code']!=0
            done=failure | (self.elapsed>=self.episode_steps)
        return self.obs,cost,done,{'failure':failure,'invalid_policy_action':invalid,'fallback':invalid,**metrics}

    def detach_graph(self):
        # Detach every time-dependent buffer, then reconstruct the packet with
        # those detached buffers. RNG state, biases and physical values persist.
        objects=[self,self.controller,self.sensors,self.history,self.history.height]
        if self.history.motion:objects.append(self.history.motion)
        for obj in objects:
            for name,value in list(vars(obj).items()):
                if isinstance(value,torch.Tensor):setattr(obj,name,value.detach())
        self.packet=Sensors(self.sensors.q,self.sensors.rate,self.sensors.force,self.sensors.rpm,self.sensors.height)
        self.obs=self.obs.detach()


def __getattr__(name):
    if name in ('FlexibleRopeEnvironment','FlexibleRopeEvaluationEnvironment'):
        from . import flexible_environment
        return getattr(flexible_environment,name)
    raise AttributeError(name)
