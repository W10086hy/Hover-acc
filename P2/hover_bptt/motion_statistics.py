from contextlib import contextmanager
import math
import torch

class TailStatistics:
    def __init__(self,n,device,dt=.01,start_step=9000,end_step=12000):
        if dt<=0 or not 0<=start_step<end_step:raise ValueError('Valid fullrate observation window required')
        self.dt=dt;self.start=start_step;self.end=end_step;self.expected=end_step-start_step
        self.count=torch.zeros(n,device=device,dtype=torch.float64)
        self.sum_error=torch.zeros(n,3,device=device,dtype=torch.float64);self.error2=self.sum_error.clone();self.time_error=self.sum_error.clone()
        self.time=self.count.clone();self.time2=self.count.clone();self.speed2=self.count.clone();self.alpha2=self.count.clone();self.motor2=self.count.clone()
        self.horizontal_speed2=self.count.clone()
    def add(self,key,previous_rate,previous_motor,state,motor,alive,target):
        if not self.start<=key<self.end:return
        with torch.no_grad():
            active=alive.double();error=torch.where(alive[:,None],(state[:,:3]-target).double(),torch.zeros_like(self.sum_error));time=(key-self.start+1)*self.dt
            self.count+=active;self.sum_error+=error*active[:,None];self.error2+=error.square()*active[:,None]
            self.time_error+=time*error*active[:,None];self.time+=time*active;self.time2+=time*time*active
            self.speed2+=torch.where(alive,state[:,3:6].double().square().sum(-1),torch.zeros_like(active))
            self.horizontal_speed2+=torch.where(alive,state[:,3:5].double().square().sum(-1),torch.zeros_like(active))
            self.alpha2+=torch.where(alive,((state[:,10:13]-previous_rate).double()/self.dt).square().sum(-1),torch.zeros_like(active))
            self.motor2+=torch.where(alive,(motor-previous_motor).double().square().mean(-1),torch.zeros_like(active))
    def finish(self):
        n=self.count.clamp_min(1);centered=(self.error2-self.sum_error.square()/n[:,None]).clamp_min(0)
        stt=(self.time2-self.time.square()/n).clamp_min(1e-20)
        covariance=self.time_error-self.time[:,None]*self.sum_error/n[:,None]
        residual=(centered-covariance.square()/stt[:,None]).clamp_min(0)
        values=dict(samples=self.count,late_origin_rmse_m=(self.error2.sum(-1)/n).sqrt(),
            late_horizontal_std_m=(centered[:,:2].sum(-1)/n).sqrt(),
            late_horizontal_bias_m=(self.sum_error[:,:2]/n[:,None]).norm(dim=-1),
            late_horizontal_drift_mps=(covariance[:,:2]/stt[:,None]).norm(dim=-1),
            late_horizontal_speed_rms_mps=(self.horizontal_speed2/n).sqrt(),
            late_detrended_position_std_m=(residual.sum(-1)/n).sqrt(),late_detrended_height_std_m=(residual[:,2]/n).sqrt(),
            late_speed_rms_mps=(self.speed2/n).sqrt(),late_angular_acceleration_rms_radps2=(self.alpha2/n).sqrt(),
            late_motor_step_rms_normalized=(self.motor2/n).sqrt())
        if any(not bool(torch.isfinite(v).all())for v in values.values()):raise ValueError('Nonfinite offline motion statistics')
        return dict(sample_hz=1/self.dt,start_step=self.start,end_step=self.end,expected_samples=self.expected,
            **{key:value.detach().cpu().tolist()for key,value in values.items()})


@contextmanager
def collect(stream,statistics):
    native=stream.step;existed='step'in vars(stream);seen=set()
    def step(actor):
        key=stream.steps;fresh=key not in seen
        if fresh:seen.add(key)
        included=fresh and statistics.start<=key<statistics.end
        if included:
            previous_rate=stream.env.state[:,10:13].detach().clone();previous_motor=stream.env.controller.previous.detach().clone()
        value=native(actor)
        if included:statistics.add(key,previous_rate,previous_motor,stream.env.state.detach(),stream.env.controller.previous.detach(),stream.alive,stream.env.model.target)
        return value
    try:stream.step=step;yield
    finally:
        if existed:stream.step=native
        else:del stream.step

