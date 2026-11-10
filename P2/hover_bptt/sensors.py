import math
import torch
from .dynamics.torch_model import rotations
from .math3d import quaternion,multiply
from .onboard import Sensors
BARO_RECORDER=None
YAW_RECORDER=None
class SensorSimulation:
    """Truth enters here to simulate measurements, never into onboard controller functions."""
    def __init__(self, n, device, cfg, generator):
        self.n, self.device, self.cfg, self.gen = n, device, cfg, generator
        self.valid = torch.zeros(n, device=device, dtype=torch.bool)
        self.gyro_bias = torch.zeros(n, 3, device=device)
        self.accel_bias = torch.zeros_like(self.gyro_bias)
        self.heading = torch.zeros(n, device=device)
        self.yaw_error = torch.zeros(n, device=device)
        self.yaw_drift = torch.zeros(n, device=device)
        self.baro_bias = torch.zeros(n, device=device)
        self.baro_drift = torch.zeros(n, device=device)
        self.tick = torch.zeros(n, device=device, dtype=torch.long)
        self.baro_period = round(100/cfg["baro_rate_hz"])
        self.baro_delay = cfg["baro_delay_samples"]
        self.baro_queue = torch.zeros(n, self.baro_delay+1, device=device)
        self.q = torch.zeros(n, 4, device=device)
        self.q[:, 0] = 1
        self.rate = torch.zeros(n, 3, device=device)
        self.force = torch.zeros_like(self.rate)
        self.rpm = torch.zeros(n, 4, device=device)
        self.height = torch.zeros(n, device=device)

    def normal(self, *shape):
        return torch.randn(*shape, device=self.gen.device, generator=self.gen).to(self.device)

    def reset(self, mask):
        c, n = self.cfg, self.n
        self.gyro_bias[mask] = (c["gyro_bias_radps"]*self.normal(n, 3))[mask]
        self.accel_bias[mask] = (c["accel_bias_mps2"]*self.normal(n, 3))[mask]
        self.yaw_drift[mask] = (math.radians(c["yaw_drift_degps"])*self.normal(n))[mask]
        self.baro_bias[mask] = (c["baro_bias_m"]*self.normal(n))[mask]
        self.baro_drift[mask] = (c["baro_drift_mps"]*self.normal(n))[mask]
        self.yaw_error[mask] = 0
        self.tick[mask] = 0
        self.valid[mask] = False

    def sample(self, truth_q, truth_rate, truth_force, truth_rpm, truth_height, active=None):
        if active is None:
            active = torch.ones_like(self.valid)
        c, n = self.cfg, self.n
        new = ~self.valid
        angles = math.radians(c["attitude_noise_deg"])*self.normal(n, 3)
        angles[:, 2] += self.yaw_error
        q = multiply(quaternion(angles), truth_q)
        r = rotations(q)
        heading = torch.atan2(r[:, 1, 0], r[:, 0, 0])
        self.heading.copy_(torch.where(new & active, heading, self.heading))
        reference = torch.zeros(n, 3, device=self.device)
        reference[:, 2] = -self.heading
        q = multiply(quaternion(reference), q)
        q = q*torch.where(q[:, :1] < 0, -1., 1.)
        gyro = truth_rate+self.gyro_bias+c["gyro_noise_radps"]*self.normal(n, 3)
        force = truth_force+self.accel_bias+c["accel_noise_mps2"]*self.normal(n, 3)
        force = torch.where(new[:, None], force, self.force+c["specific_force_filter_alpha"]*(force-self.force))
        rpm = (truth_rpm+c["rpm_noise_radps"]*self.normal(n, 4)).clamp_min(0)
        baro = truth_height+self.baro_bias+self.tick*.01*self.baro_drift+c["baro_noise_m"]*self.normal(n)
        queue = torch.cat((baro[:, None], self.baro_queue[:, :-1]), -1)
        fresh = active & ((self.tick % self.baro_period) == 0)
        queue = torch.where(new[:, None], baro[:, None].expand_as(queue), queue)
        self.baro_queue.copy_(torch.where(fresh[:, None], queue, self.baro_queue))
        delayed = self.baro_queue[:, -1]
        self.height.copy_(torch.where(fresh, delayed, self.height))
        for name, value in (("q", q), ("rate", gyro), ("force", force), ("rpm", rpm)):
            old = getattr(self, name)
            old.copy_(torch.where(active[:, None], value, old))
        self.yaw_error += active*.01*self.yaw_drift
        self.tick += active.long()
        self.valid |= active
        return Sensors(self.q, self.rate, self.force, self.rpm, self.height)
class FunctionalSensors(SensorSimulation):
    def sample(self,truth_q,truth_rate,truth_force,truth_rpm,truth_height,active=None):
        if active is None:active=torch.ones_like(self.valid)
        c,n=self.cfg,self.n;new=~self.valid
        angles=math.radians(c['attitude_noise_deg'])*self.normal(n,3)
        angles=torch.cat((angles[:,:2],(angles[:,2]+self.yaw_error)[:,None]),-1)
        q=multiply(quaternion(angles),truth_q);r=rotations(q)
        heading=torch.atan2(r[:,1,0],r[:,0,0])
        self.heading=torch.where(new & active,heading,self.heading)
        z=torch.zeros_like(self.heading)
        reference=torch.stack((z,z,-self.heading),-1)
        q=multiply(quaternion(reference),q);q=q*torch.where(q[:,:1]<0,-1.,1.)
        gyro=truth_rate+self.gyro_bias+c['gyro_noise_radps']*self.normal(n,3)
        force=truth_force+self.accel_bias+c['accel_noise_mps2']*self.normal(n,3)
        force=torch.where(new[:,None],force,self.force+c['specific_force_filter_alpha']*(force-self.force))
        rpm=(truth_rpm+c['rpm_noise_radps']*self.normal(n,4)).clamp_min(0)
        baro=truth_height+self.baro_bias+self.tick*.01*self.baro_drift+c['baro_noise_m']*self.normal(n)
        queue=torch.cat((baro[:,None],self.baro_queue[:,:-1]),-1)
        fresh=active & ((self.tick % self.baro_period)==0)
        queue=torch.where(new[:,None],baro[:,None].expand_as(queue),queue)
        self.baro_queue=torch.where(fresh[:,None],queue,self.baro_queue)
        self.height=torch.where(fresh,self.baro_queue[:,-1],self.height)
        for name,value in [('q',q),('rate',gyro),('force',force),('rpm',rpm)]:
            setattr(self,name,torch.where(active[:,None],value,getattr(self,name)))
        self.yaw_error=self.yaw_error+active*.01*self.yaw_drift
        self.tick=self.tick+active.long();self.valid=self.valid|active
        return Sensors(self.q,self.rate,self.force,self.rpm,self.height)
def fit_ground_baro(measurements,times):
    if measurements.ndim!=2 or times.ndim!=1 or measurements.shape[1]!=len(times)or len(times)<4:
        raise ValueError('Past stationary measurement matrix and timestamps required')
    if not bool(torch.isfinite(measurements).all())or not bool(torch.isfinite(times).all())or bool((times>0).any())or not bool((times[1:]>times[:-1]).all()):
        raise ValueError('Finite strictly increasing past timestamps required')
    t=times.double();y=measurements.double();center=t-t.mean();denominator=center.square().sum()
    drift=((y-y.mean(-1,keepdim=True))*center).sum(-1)/denominator
    bias=y.mean(-1)-drift*t.mean()
    return bias,drift

class CalibrationMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        settings=self.cfg.get('preflight_baro_calibration',{})
        self.calibration_enabled=bool(settings.get('enabled',False))
        self.calibration_seconds=float(settings.get('stationary_s',20.))
        if self.calibration_enabled and self.calibration_seconds!=20.:raise ValueError('This ablation uses explicit20s stationary preflight')
        self.calibration_seed=self.gen.initial_seed()+94711
        self.calibration_bias=torch.zeros_like(self.baro_bias)
        self.calibration_drift=torch.zeros_like(self.baro_drift)
        self.calibration_epochs=torch.zeros(self.n,dtype=torch.long,device=self.device)

    def reset(self,mask):
        super().reset(mask)
        if not self.calibration_enabled or not bool(mask.any()):return
        indices=torch.nonzero(mask).flatten();rate=self.cfg['baro_rate_hz'];count=round(self.calibration_seconds*rate)
        times=torch.arange(-count,0,device=self.device,dtype=self.baro_bias.dtype)/rate
        epochs=self.calibration_epochs.clone();epochs[mask]+=1;self.calibration_epochs=epochs
        noises=[]
        for index in indices.tolist():
            seed=(self.calibration_seed+100003*index+10000019*int(epochs[index]))%(2**63-1)
            generator=torch.Generator(device=self.gen.device).manual_seed(seed)
            noises.append(torch.randn(count,generator=generator,device=self.gen.device,dtype=self.baro_bias.dtype).to(self.device))
        # Simulator-only generation of past stationary sensor readings.
        readings=self.baro_bias[indices,None]+self.baro_drift[indices,None]*times+self.cfg['baro_noise_m']*torch.stack(noises)
        # Deployable estimator receives only measured heights and past times.
        bias,drift=fit_ground_baro(readings,times)
        self.calibration_bias=self.calibration_bias.clone();self.calibration_drift=self.calibration_drift.clone()
        self.calibration_bias[indices]=bias.to(self.baro_bias.dtype)
        self.calibration_drift[indices]=drift.to(self.baro_drift.dtype)
        if BARO_RECORDER is not None:
            BARO_RECORDER(dict(indices=indices.detach().cpu(),times_s=times.detach().cpu(),measurements_m=readings.detach().cpu(),
                estimated_bias_m=bias.detach().cpu(),estimated_drift_mps=drift.detach().cpu(),
                epoch=epochs[indices].detach().cpu(),sensor_initial_seed=self.gen.initial_seed(),
                estimator_inputs='Only past stationary ground-relative barometer measurements and timestamps'))

    def sample(self,*args,**kwargs):
        if not self.calibration_enabled:return super().sample(*args,**kwargs)
        active=kwargs.get('active',args[5]if len(args)>5 else None)
        if active is None:active=torch.ones_like(self.valid)
        tick=self.tick.clone();fresh=active & ((tick%self.baro_period)==0)
        packet=super().sample(*args,**kwargs)
        source_time=(tick-self.baro_delay*self.baro_period).clamp_min(0).to(self.height.dtype)*.01
        correction=self.calibration_bias+source_time*self.calibration_drift
        # Correct only fresh reports;the held height is already corrected.
        # Keeping public self.height canonical preserves existing BPTT boundaries.
        self.height=torch.where(fresh,self.height-correction,self.height)
        return Sensors(packet.attitude,packet.gyro,packet.specific_force,packet.motor_speed,self.height)
def fit_stationary_yaw(measurements,times):
    if (measurements.ndim!=2 or times.ndim!=1 or measurements.shape[1]!=len(times)
        or len(times)<4 or not bool(torch.isfinite(measurements).all())
        or not bool(torch.isfinite(times).all()) or bool((times>0).any())
        or not bool((times[1:]>times[:-1]).all())):
        raise ValueError('Finite past stationary yaw samples and increasing timestamps required')
    readings=measurements.double();time=times.double()
    difference=readings[:,1:]-readings[:,:-1]
    delta=torch.atan2(difference.sin(),difference.cos())
    unwrapped=torch.cat((readings[:,:1],readings[:,:1]+delta.cumsum(-1)),-1)
    centered=time-time.mean()
    return ((unwrapped-unwrapped.mean(-1,keepdim=True))*centered).sum(-1)/centered.square().sum()

class YawCalibrationMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.estimated_yaw_drift=torch.zeros_like(self.yaw_drift)
        self.yaw_calibration_epochs=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.yaw_calibration_seed=self.gen.initial_seed()+371719

    def reset(self,mask):
        super().reset(mask)
        if not bool(mask.any()):return
        indices=torch.nonzero(mask).flatten();count=2000
        time=torch.arange(-count,0,device=self.device,dtype=self.yaw_drift.dtype)*.01
        epochs=self.yaw_calibration_epochs.clone();epochs[mask]+=1;self.yaw_calibration_epochs=epochs
        noises=[]
        for index in indices.tolist():
            seed=(self.yaw_calibration_seed+100003*index+10000019*int(epochs[index]))%(2**63-1)
            generator=torch.Generator(device=self.gen.device).manual_seed(seed)
            noises.append(torch.randn(count,generator=generator,device=self.gen.device,dtype=self.yaw_drift.dtype).to(self.device))
        # Simulator boundary only: create measured past yaw while physically stationary.
        measured=self.yaw_drift[indices,None]*time+math.radians(self.cfg['attitude_noise_deg'])*torch.stack(noises)
        measured=torch.atan2(measured.sin(),measured.cos())
        # No hidden bias/drift or true aircraft yaw is an estimator argument.
        estimated=fit_stationary_yaw(measured,time)
        value=self.estimated_yaw_drift.clone();value[indices]=estimated.to(value.dtype);self.estimated_yaw_drift=value
        if YAW_RECORDER is not None:
            YAW_RECORDER(dict(indices=indices.detach().cpu(),times_s=time.detach().cpu(),
                measured_yaw_rad=measured.detach().cpu(),estimated_drift_radps=estimated.detach().cpu(),
                stationary_preflight_s=20.,estimator_inputs='Past measured attitude yaw and timestamps only'))

    def sample(self,*args,**kwargs):
        active=kwargs.get('active',args[5]if len(args)>5 else None)
        if active is None:active=torch.ones_like(self.valid)
        tick=self.tick.clone();packet=super().sample(*args,**kwargs)
        correction=-self.estimated_yaw_drift*tick.to(self.estimated_yaw_drift.dtype)*.01
        zero=torch.zeros_like(correction)
        corrected=multiply(quaternion(torch.stack((zero,zero,correction),-1)),packet.attitude)
        self.q=torch.where(active[:,None],corrected,packet.attitude)
        return Sensors(self.q,packet.gyro,packet.specific_force,packet.motor_speed,packet.baro_height)

class CalibratedSensors(CalibrationMixin,SensorSimulation):pass
class CalibratedFunctionalSensors(CalibrationMixin,FunctionalSensors):pass
class YawCalibratedSensors(YawCalibrationMixin,CalibratedSensors):pass
class YawCalibratedFunctionalSensors(YawCalibrationMixin,CalibratedFunctionalSensors):pass
