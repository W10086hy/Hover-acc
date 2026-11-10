"""Torch batch port of the validated legacy plant (not the paper surrogate)."""
import json,math
from pathlib import Path
from typing import Tuple
import torch
from torch import Tensor,nn


def rotations(q:Tensor)->Tensor:
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)


def mv(r:Tensor,v:Tensor)->Tensor:
    return torch.bmm(r,v.unsqueeze(-1)).squeeze(-1)


def qdot(q:Tensor,w:Tensor)->Tensor:
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-12)
    s=q[:,:1];v=q[:,1:]
    return .5*torch.cat((-(v*w).sum(-1,keepdim=True),s*w+torch.linalg.cross(v,w)),dim=-1)


class TorchModel(nn.Module):
    def __init__(self,cfg,device="cuda",dtype=torch.float32,arms=None,lengths=None):
        super().__init__()
        if cfg.get("model_family")!="c1_hover_noaileron" or cfg["airframe"]["rotor_tilt_deg"]!=0:
            raise ValueError("Only the validated zero-tilt legacy family is supported")
        a,u,c,t,p= (cfg[k] for k in ["airframe","actuators","aero","tether","power"])
        def buf(name,value):self.register_buffer(name,torch.as_tensor(value,device=device,dtype=dtype))
        self.mass=float(a["mass_kg"]);self.gravity=float(a["gravity_mps2"])
        self.rho=float(a["air_density_kgm3"]);self.area=float(a["area_m2"]);self.kappa=math.radians(a["wing_install_deg"])
        self.span=float(a["span_m"]);self.chord=float(a["chord_m"])
        k=math.radians(a["wing_install_deg"]);co,si=math.cos(k),math.sin(k)
        buf("wing_to_body",[[co,0,si],[0,1,0],[-si,0,co]])
        buf("inertia",a["inertia_body_kgm2"]);buf("target",cfg["simulation"]["target_w_m"])
        buf("gravity_force",[0.,0.,self.mass*self.gravity]);buf("moment_scales",[self.span,self.chord,self.span])
        self.kf=float(u["kf"]);self.km=float(u["km"]);self.motor_tau=float(u["motor_tau_s"])
        self.servo_tau=float(u["servo_tau_s"]);self.servo_rate=float(u["servo_rate_radps"])
        self.delta_limit=math.radians(u["servo_limit_deg"]);self.deadzone=float(u["motor_deadzone"])
        self.pa=float(u["motor_speed_polynomial"][0]);self.pb=float(u["motor_speed_polynomial"][1]);self.pc=float(u["motor_speed_polynomial"][2])
        self.max_speed=max(0.,self.pa+self.pb+self.pc);self.max_thrust=self.kf*self.max_speed**2
        dx,dy=a["rotor_dx_m"],a["rotor_dy_m"];km=self.km/self.kf
        buf("rotor_map",[[0,0,0,0],[0,0,0,0],[-1,-1,-1,-1],[-dy,-dy,dy,dy],[dx,-dx,dx,-dx],[km,-km,-km,km]])
        self.aero_enabled=bool(c["enabled"]);self.gyro=bool(u.get("include_gyro",False))
        self.tether_enabled=bool(t["enabled"]);self.sigma=float(c.get("angle_regularization_mps",.3))
        self.c0=float(c["c0"]);self.c1=float(c["c1"])
        self.cy_beta=float(c["cy_beta"]);self.cl_beta=float(c["cl_beta"]);self.cn_beta=float(c["cn_beta"])
        self.cl_p=float(c["cl_p"]);self.cm_q=float(c["cm_q"]);self.cn_r=float(c["cn_r"])
        self.cl_delta_a=float(c["cl_delta_a"]);self.cm_delta_e=float(c["cm_delta_e"]);self.cn_delta_a=float(c["cn_delta_a"])
        if c["cl_delta_e"]!=0 or c["cd_delta_e"]!=0:raise ValueError("Unsupported lift/drag control derivatives")
        table=json.loads((Path(__file__).parents[1]/"configs/cm_table.json").read_text())
        buf("breaks",table["breaks"]);buf("coefs",table["coefs"])
        buf("arms",t["attachment_b_m"] if arms is None else arms)
        buf("lengths",t["rest_length_m"] if lengths is None else lengths)
        buf("anchor",t["anchor_w_m"])
        self.stiffness=float(t["stiffness_npm"]);self.damping=float(t["damping_nspm"])
        self.jm=float(p["rotor_inertia_kgm2"]);self.efficiency=float(p["efficiency"])
        self.idle=float(p["motor_idle_w"]);self.servo_hold=float(p["servo_hold_w"]);self.servo_power=float(p["servo_rate_w_per_radps"])
        self.dt=float(cfg["simulation"]["physics_dt_s"])
        self.control_dt=float(cfg["simulation"]["control_dt_s"]);self.substeps=round(self.control_dt/self.dt)

    def speed_target(self,u:Tensor)->Tensor:
        u=u.clamp(0.,1.)
        return torch.where(u>=self.deadzone,(self.pa*u*u+self.pb*u+self.pc).clamp_min(0.),torch.zeros_like(u))

    def throttle_from_speed(self,speed:Tensor)->Tensor:
        speed=speed.clamp(0.,self.max_speed);d=speed-self.pc
        u=2*d/(self.pb+torch.sqrt((self.pb*self.pb+4*self.pa*d).clamp_min(0.)))
        return torch.where(speed<=1e-12,torch.zeros_like(u),u.clamp(0.,1.))

    def actuator_at(self,x:Tensor,u:Tensor,t:float)->Tuple[Tensor,Tensor]:
        target=self.speed_target(u[:,:4])
        motor=target+(x[:,13:17]-target)*math.exp(-t/self.motor_tau)
        desired=u[:,4:].clamp(-self.delta_limit,self.delta_limit)
        error=desired-x[:,17:19];ae=error.abs()
        switch=((ae-self.servo_rate*self.servo_tau)/self.servo_rate).clamp_min(0.)
        residual=torch.where(t<switch,ae-self.servo_rate*t,
            torch.minimum(ae,torch.full_like(ae,self.servo_rate*self.servo_tau))*torch.exp(-(t-switch).clamp_min(0.)/self.servo_tau))
        return motor,desired-torch.sign(error)*residual

    def loads(self,x:Tensor,wind:Tensor)->Tuple[Tensor,Tensor,Tensor,Tensor]:
        r=rotations(x[:,6:10]);rt=r.transpose(1,2)
        va_b=mv(rt,x[:,3:6]-wind);va_l=va_b@self.wing_to_body
        speed=torch.linalg.vector_norm(va_l,dim=-1)
        smooth=torch.tanh(speed/self.sigma)
        alpha=torch.where(torch.sqrt(va_l[:,0].square()+va_l[:,2].square())>1e-9,torch.atan2(va_l[:,2],va_l[:,0]),torch.zeros_like(speed))*smooth
        beta=torch.asin((va_l[:,1]/(speed+self.sigma)).clamp(-1.,1.))*smooth
        qbar=.5*self.rho*speed.square()
        idx=(torch.searchsorted(self.breaks,alpha.contiguous(),right=True)-1).clamp(0,self.coefs.shape[0]-1)
        d=alpha-self.breaks[idx];co=self.coefs[idx]
        cm=((co[:,0]*d+co[:,1])*d+co[:,2])*d+co[:,3]
        rates=x[:,10:13]@self.wing_to_body;den=2*speed.clamp_min(1.)
        de=x[:,17]+x[:,18];da=x[:,18]-x[:,17]
        lift=qbar*self.area*(self.c1*torch.sin(alpha)*torch.cos(alpha))
        drag=qbar*self.area*(self.c0+self.c1*torch.sin(alpha).square())
        lam=self.kappa-alpha
        # The installation angle is a module constant; no tensor-to-host conversion in captured calls.
        force=torch.stack((-drag*torch.cos(lam)-lift*torch.sin(lam),qbar*self.area*self.cy_beta*beta,
                           drag*torch.sin(lam)-lift*torch.cos(lam)),-1)
        coeff=torch.stack((self.cl_beta*beta+self.cl_p*self.span*rates[:,0]/den+self.cl_delta_a*da,
                           cm+self.cm_q*self.chord*rates[:,1]/den+self.cm_delta_e*de,
                           self.cn_beta*beta+self.cn_r*self.span*rates[:,2]/den+self.cn_delta_a*da),-1)
        scaled=coeff*self.moment_scales
        torque=(qbar*self.area).unsqueeze(-1)*scaled@self.wing_to_body.T
        if not self.aero_enabled:force=torch.zeros_like(force);torque=torch.zeros_like(torque)
        arms=self.arms.expand(x.shape[0],3)
        displacement=x[:,:3]+mv(r,arms)-self.anchor
        length=torch.linalg.vector_norm(displacement,dim=-1)
        direction=displacement/length.clamp_min(1e-12).unsqueeze(-1)
        va=x[:,3:6]+mv(r,torch.linalg.cross(x[:,10:13],arms))
        length_rate=(direction*va).sum(-1)
        tension=torch.where((length>=self.lengths)&(length>1e-12),
            (self.stiffness*(length-self.lengths)+self.damping*length_rate).clamp_min(0.),torch.zeros_like(length))
        if not self.tether_enabled:tension=torch.zeros_like(tension)
        tether=-tension.unsqueeze(-1)*direction
        torque=torque+torch.linalg.cross(arms,mv(rt,tether))
        omega=x[:,13:17].clamp_min(0.)
        rotor=self.kf*omega.square()@self.rotor_map.T
        if self.gyro:
            spin=omega[:,0]-omega[:,1]-omega[:,2]+omega[:,3]
            torque=torque+self.jm*torch.stack((spin*x[:,11],-spin*x[:,10],torch.zeros_like(spin)),-1)
        ng=mv(r,rotor[:,:3]+force)+tether
        return ng+self.gravity_force,torque+rotor[:,3:],mv(rt,ng)/self.mass,tension

    def power(self,x:Tensor,u:Tensor)->Tensor:
        omega=x[:,13:17].clamp_min(0.)
        dw=(self.speed_target(u[:,:4])-omega)/self.motor_tau
        dd=((u[:,4:].clamp(-self.delta_limit,self.delta_limit)-x[:,17:19])/self.servo_tau).clamp(-self.servo_rate,self.servo_rate)
        return ((self.km*omega.pow(3)+self.jm*(omega*dw).clamp_min(0.))/self.efficiency+self.idle*(omega>1.)).sum(-1)+2*self.servo_hold+self.servo_power*dd.abs().sum(-1)

    def mechanical_rhs(self,x:Tensor,wind:Tensor)->Tensor:
        force,torque,_,_=self.loads(x,wind)
        w=x[:,10:13]
        dw=(torque-torch.linalg.cross(w,w*self.inertia))/self.inertia
        return torch.cat((x[:,3:6],force/self.mass,qdot(x[:,6:10],w),dw),-1)

    def rhs(self,x:Tensor,u:Tensor,wind:Tensor)->Tensor:
        dw=(self.speed_target(u[:,:4])-x[:,13:17])/self.motor_tau
        dd=((u[:,4:].clamp(-self.delta_limit,self.delta_limit)-x[:,17:19])/self.servo_tau).clamp(-self.servo_rate,self.servo_rate)
        return torch.cat((self.mechanical_rhs(x,wind),dw,dd),-1)

    def forward(self,x:Tensor,u:Tensor,wind:Tensor)->Tensor:
        h=self.dt;wh,dh=self.actuator_at(x,u,h/2);we,de=self.actuator_at(x,u,h)
        k1=self.mechanical_rhs(x,wind)
        k2=self.mechanical_rhs(torch.cat((x[:,:13]+h*k1/2,wh,dh),-1),wind)
        k3=self.mechanical_rhs(torch.cat((x[:,:13]+h*k2/2,wh,dh),-1),wind)
        k4=self.mechanical_rhs(torch.cat((x[:,:13]+h*k3,we,de),-1),wind)
        y=x[:,:13]+h*(k1+2*k2+2*k3+k4)/6
        q=y[:,6:10]/torch.linalg.vector_norm(y[:,6:10],dim=-1,keepdim=True).clamp_min(1e-12)
        return torch.cat((y[:,:6],q,y[:,10:13],we,de),-1)

    def control(self,x:Tensor,u:Tensor,winds:Tensor)->Tensor:
        for i in range(self.substeps):x=self.forward(x,u,winds[:,i,:])
        return x

    def initial(self,n:int)->Tensor:
        x=torch.zeros((n,19),device=self.inertia.device,dtype=self.inertia.dtype)
        x[:,:3]=self.target;x[:,6]=1.
        x[:,13:17]=math.sqrt(self.mass*self.gravity/(4*self.kf))
        return x


class CapturedControl:
    """Fixed-shape CUDA graph; output storage is reused, so clone retained samples."""
    def __init__(self,model,x,u,winds):
        self.model=model;self.x=x.clone();self.u=u.clone();self.winds=winds.clone()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):model.control(self.x,self.u,self.winds)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=model.control(self.x,self.u,self.winds)
    def __call__(self,x,u,winds):
        self.x.copy_(x);self.u.copy_(u);self.winds.copy_(winds)
        self.graph.replay()
        return self.output

