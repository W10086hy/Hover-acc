import math
import torch
PROFILES=("steady","gust","turbulence","time_varying","combined")
class WindField:
    """Exogenous wind sampled at the existing100Hz control interval.

    Compute once using a dedicated generator. Queries by episode elapsed time
    have no side effects; RK4 substeps retain the legacy zero-order wind hold.
    """
    def __init__(self,specifications,seed,duration_s=120.,dt=.01,device='cpu'):
        self.dt=dt;self.seed=seed
        steps=round(duration_s/dt)+1;n=len(specifications)
        if n<1 or steps<2 or dt<=0:raise ValueError('Nonempty positive-duration wind field required')
        # Subsets containing only integer-valued speeds must still retain the
        # fractional turbulence coefficients below. Inferred int64 previously
        # made base.new_tensor([.1,.1,.04]) zero and silently removed turbulence.
        base=torch.tensor([r['wind_speed'] for r in specifications],device=device,dtype=torch.float32)
        direction=torch.tensor([r['direction_rad'] for r in specifications],device=device,dtype=torch.float32)
        kinds=[r['wind_profile'] for r in specifications]
        if any(k not in PROFILES for k in kinds):raise ValueError('Unknown wind profile')
        gen=torch.Generator(device=device).manual_seed(seed+2000003)
        # Draw initial OU state from its stationary distribution. Multiplying
        # by the launch ramp below preserves the existing calm2s startup.
        # Fixed-shape draws make every shared time prefix independent of the
        # requested duration, including the final sample. A variable-size
        # randn call can redraw its CPU vector tail differently when extended.
        block=1024
        noise=torch.cat([torch.randn(block,n,3,device=device,generator=gen)
            for _ in range((steps+block-1)//block)],0)[:steps]
        colored=torch.empty_like(noise);colored[0]=noise[0]
        decay=math.exp(-dt/1.5);innovation=math.sqrt(1-decay*decay)
        for t in range(1,steps):colored[t]=decay*colored[t-1]+innovation*noise[t]
        time=torch.arange(steps,device=device,dtype=base.dtype)*dt
        ramp=((time-2)/2).clamp(0,1)
        gust=torch.tensor([k in ['gust','combined'] for k in kinds],device=device)
        turbulent=torch.tensor([k in ['turbulence','combined'] for k in kinds],device=device)
        varying=torch.tensor([k in ['time_varying','combined'] for k in kinds],device=device)
        phase=(time-10).remainder(20)
        pulse=torch.where((time>=10)&(phase<=4),.5*(1-torch.cos(2*math.pi*phase/4)),0.)
        magnitude=base[None,:]*(1+.2*torch.sin(2*math.pi*time[:,None]/30)*varying[None,:]
            +.2*pulse[:,None]*gust[None,:])
        angle=direction[None,:]+math.radians(30)*torch.sin(2*math.pi*time[:,None]/45)*varying[None,:]
        mean=torch.stack((magnitude*angle.cos(),magnitude*angle.sin(),torch.zeros_like(magnitude)),-1)
        sigma=base[:,None]*base.new_tensor([.1,.1,.04])[None,:]
        self.trace=(mean+colored*sigma[None,:,:]*turbulent[None,:,None])*ramp[:,None,None]

    def at(self,elapsed):
        if bool((elapsed<0).any()) or bool((elapsed>=len(self.trace)).any()):
            raise ValueError('Wind query outside the declared horizon')
        return self.trace[elapsed,torch.arange(elapsed.shape[0],device=elapsed.device)]

    def statistics(self):
        speed=self.trace.norm(dim=-1)
        return [dict(actual_mean_wind_mps=float(speed[:,i].mean()),
            actual_rms_wind_mps=float(speed[:,i].square().mean().sqrt()),
            actual_peak_wind_mps=float(speed[:,i].max())) for i in range(speed.shape[1])]
