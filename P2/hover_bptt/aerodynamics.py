"""Onboard nominal force model: fixed vehicle coefficients and estimated airflow only."""
import math
import torch


class NominalAerodynamics:
    def __init__(self, airframe, aero, device):
        self.rho,self.area=airframe["air_density_kgm3"],airframe["area_m2"]
        self.kappa=math.radians(airframe["wing_install_deg"])
        c,s=math.cos(self.kappa),math.sin(self.kappa)
        self.wing_to_body=torch.tensor([[c,0,s],[0,1,0],[-s,0,c]],device=device)
        self.sigma=aero["angle_regularization_mps"]
        self.c0,self.c1,self.cy=aero["c0"],aero["c1"],aero["cy_beta"]

    def force(self, estimated_body_airflow):
        v=estimated_body_airflow@self.wing_to_body
        speed=torch.linalg.vector_norm(v,dim=-1)
        smooth=torch.tanh(speed/self.sigma)
        alpha=torch.where(torch.sqrt(v[:,0].square()+v[:,2].square())>1e-9,
                          torch.atan2(v[:,2],v[:,0]),torch.zeros_like(speed))*smooth
        beta=torch.asin((v[:,1]/(speed+self.sigma)).clamp(-1,1))*smooth
        q=.5*self.rho*speed.square()
        lift=q*self.area*self.c1*alpha.sin()*alpha.cos()
        drag=q*self.area*(self.c0+self.c1*alpha.sin().square())
        lam=self.kappa-alpha
        return torch.stack((-drag*lam.cos()-lift*lam.sin(),q*self.area*self.cy*beta,
                            drag*lam.sin()-lift*lam.cos()),-1)
