"""P3 sensor/reference standardization for the frozen 35-input acceleration Actor.

No position, wind or tether input. Physical wrench feedforward is not sent.
"""
import json
import math
from pathlib import Path
from dataclasses import replace
import numpy as np
from .network import rotation
from .control import Observer


def multiply(a,b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


def physical_reference(cfg):
    a,u=cfg['airframe'],cfg['actuators']
    return dict(mass_kg=a['mass_kg'],inertia=a['inertia_body_kgm2'],rotor_dx_m=a['rotor_dx_m'],rotor_dy_m=a['rotor_dy_m'],
        gravity=a['gravity_mps2'],kf=u['kf'],km=u['km'],poly=u['motor_speed_polynomial'],motor_tau=u['motor_tau_s'],
        rotor_inertia=cfg['power']['rotor_inertia_kgm2'],include_gyro=u['include_gyro'])


class ObserverCalibration:
    def __init__(self,parameters):
        self.parameters=parameters
        self.C=np.asarray(parameters['body_from_sensor'],float);self.V=np.asarray(parameters['body_from_virtual'],float)
        self.bias=np.asarray(parameters['bias_body'],float);self.gyro_bias=np.asarray(parameters['gyro_bias_sensor'],float)
        self.r=self.V.T@np.asarray(parameters['com_geometry'],float)
        self.mass=float(parameters['mass_kg']);self.gain=float(parameters['motor_speed_gain']);self.tau=float(parameters['motor_tau_s'])
        for R in (self.C,self.V):
            if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-7) or np.linalg.det(R)<.999:
                raise ValueError('Proper finite P3 rotation required')
        for v in (self.bias,self.gyro_bias,self.r):
            if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Finite P3 three-vector required')
        if not .7<self.gain<1.3 or not 0<self.tau<1 or not 0<self.mass<20 or np.linalg.norm(self.r)>.05:
            raise ValueError('P3 parameters outside supported bounds')
        self.sensor_to_virtual_q=self.quaternion(self.C.T@self.V)
        self.body_to_virtual_q=self.quaternion(self.V)

    @staticmethod
    def quaternion(R):
        # Supported installation rotations are small, with positive trace.
        w=math.sqrt(max(0.,1+np.trace(R)))/2
        if w<.5:raise ValueError('Small installation rotation required')
        q=np.r_[w,np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(4*w)]
        return q/np.linalg.norm(q)

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        if data.get('schema')=='P3_aligned_residual_v2':
            from .aligned_p3 import AlignedCalibration
            return AlignedCalibration(data)
        if data.get('schema')=='P3_unified_residual_v1':
            from .unified_p3 import UnifiedCalibration
            return UnifiedCalibration(data)
        if data.get('schema')!='P3_observer_standardization_v1' or not data.get('quality_passed') or not data.get('alignment_audit_passed'):
            raise ValueError('Validated high-rate Position calibration required')
        return cls(data)

    def correct_packet(self,packet,*,attitude_is_sensor_frame=True):
        C,V=self.C,self.V
        q=multiply(packet.attitude,self.sensor_to_virtual_q if attitude_is_sensor_frame else self.body_to_virtual_q)
        return replace(packet,attitude=q/np.linalg.norm(q),gyro=V.T@(C@(packet.gyro-self.gyro_bias)),
            specific_force=V.T@(C@packet.specific_force-self.bias))

    def validate_reference(self,cfg):
        if 'reference_physics' in self.parameters and self.parameters['reference_physics']!=physical_reference(cfg):
            raise ValueError('P3 standard physical reference differs from P2; height and Actor are not compatibility keys')

    def rates_to_reported(self,rates):
        return self.C.T@self.V@np.asarray(rates,float)


class CalibratedObserver(Observer):
    def __init__(self,cfg,calibration,dt=.01,force_filter_alpha=.5):
        calibration.validate_reference(cfg['nominal'])
        super().__init__(cfg,dt);self.calibration=calibration;self.force_filter_alpha=force_filter_alpha
        a=self.pc['airframe'];pos=np.array([[a['rotor_dx_m'],a['rotor_dy_m'],0],[-a['rotor_dx_m'],a['rotor_dy_m'],0],
            [a['rotor_dx_m'],-a['rotor_dy_m'],0],[-a['rotor_dx_m'],-a['rotor_dy_m'],0]])
        axis=-calibration.V[:,2];spin=np.array([1,-1,-1,1])
        torque=self.kf*np.cross(pos,axis)-self.km*spin[:,None]*axis
        self.prediction_map=(torque@calibration.V).T

    def reset(self):
        super().reset();self.lever_correction=np.zeros(3);self.transport_torque=np.zeros(3)
        self.sample_dt_s=self.dt
        self.innovation_before=np.zeros(3);self.innovation_after=np.zeros(3);self.last_standardized_packet=None

    def update(self,packet,applied):
        c=self.calibration;valid=self.valid;w=packet.gyro
        dt=float(self.sample_dt_s)
        if not np.isfinite(dt) or not 0<dt<=.05:raise ValueError('Fresh near100Hz sensor interval required')
        alpha=(w-self.rate)/dt if valid else np.zeros(3)
        lever=np.cross(alpha,c.r)+np.cross(w,np.cross(w,c.r))
        self.lever_correction=self.lever_correction+self.force_filter_alpha*(lever-self.lever_correction) if valid else lever
        standardized=replace(packet,specific_force=packet.specific_force+self.lever_correction)
        transport=np.cross(c.r,c.mass*standardized.specific_force)
        average_transport=.5*(self.transport_torque+transport) if valid else transport
        target=c.gain*np.maximum(np.polyval(self.poly,np.clip(applied,0,1)),0.)
        delta=self.rpm-target;f=c.tau/dt*(-math.expm1(-dt/c.tau));sf=c.tau/(2*dt)*(-math.expm1(-2*dt/c.tau))
        mean=target+delta*f;square=target**2+2*target*delta*f+delta**2*sf
        torque=self.prediction_map@np.maximum(square,0)
        spin=mean@[1.,-1.,-1.,1.];torque+=self.jm*np.array([spin*self.rate[1],-spin*self.rate[0],0.])
        predicted=self.rate+dt*(torque-np.cross(self.rate,self.rate*self.J))/self.J
        self.innovation_before=(w-predicted)*self.dt/dt if valid else np.zeros(3)
        # Self disturbance is -transport. Remove its integrated contribution,
        # in rad/s, rather than subtracting an Nm quantity from Actor features.
        predicted-=dt*average_transport/self.J
        self.innovation_after=(w-predicted)*self.dt/dt if valid else np.zeros(3)
        result=super().update(standardized,applied)
        result[23:26]=np.clip(self.innovation_after/.1,-5,5)
        self.transport_torque=transport;self.last_standardized_packet=standardized
        return result
