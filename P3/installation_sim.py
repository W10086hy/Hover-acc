"""Native physical installation/payload perturbations, outside onboard boundary."""
import copy
import xml.etree.ElementTree as ET
import numpy as np
import torch

from mujoco_sim.physics import S, rotation
from P2.hover_bptt.onboard import NominalVehicle, Sensors
from .body_calibration_sim import OffsetRigidPlant, OffsetFlexiblePlant, BodyCalibrationEnvironment
from .installation_calibration import rotor_wrench


def euler_matrix(degrees):
    x,y,z=np.deg2rad(degrees); cx,cy,cz=np.cos([x,y,z]); sx,sy,sz=np.sin([x,y,z])
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])


def payload_properties(mass, inertia, com, payloads):
    masses=np.array([mass]+[p['mass_kg'] for p in payloads],float)
    points=np.array([com]+[p['position_m'] for p in payloads],float)
    if (masses<=0).any() or not np.isfinite(points).all(): raise ValueError('Physical point payloads required')
    total=masses.sum(); center=np.sum(masses[:,None]*points,axis=0)/total
    J=np.array(inertia,float).copy()
    for m,p in zip(masses,points):
        d=p-center; J+=m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    return float(total),center,J


def full_xml(text, inertia):
    tree=ET.fromstring(text); J=np.asarray(inertia)*S[:,None]*S[None,:]
    for body in tree.findall('./worldbody/body'):
        item=body.find('inertial'); item.attrib.pop('quat',None); item.attrib.pop('diaginertia',None)
        item.set('fullinertia',' '.join(map(str,[J[0,0],J[1,1],J[2,2],J[0,1],J[0,2],J[1,2]])))
    return ET.tostring(tree,encoding='unicode')


class InstallationRigidPlant(OffsetRigidPlant):
    def __init__(self,*args,full_inertia,thrust_axis,**kwargs):
        self.full_inertia=np.asarray(full_inertia,float); self.thrust_axis=np.asarray(thrust_axis,float)
        super().__init__(*args,**kwargs)

    def _xml(self): return full_xml(super()._xml(),self.full_inertia)

    def wrenches(self,x,wind,**averages):
        force,torque,specific,tension=super().wrenches(x,wind,**averages)
        speed=np.maximum(x[:,13:17],0); sq=averages.get('motor_squared',speed**2); mean=averages.get('motor_mean',speed)
        oldF,oldT=rotor_wrench(self.cfg,sq,mean,x[:,10:13],[0,0,-1.])
        newF,newT=rotor_wrench(self.cfg,sq,mean,x[:,10:13],self.thrust_axis)
        delta=newF-oldF
        force+=np.einsum('nij,nj->ni',rotation(x[:,6:10]),delta)
        torque+=newT-oldT-np.cross(self.com_body,delta)
        return force,torque,specific+delta/self.mass,tension


class InstallationFlexiblePlant(OffsetFlexiblePlant):
    def __init__(self,*args,full_inertia,thrust_axis,**kwargs):
        self.full_inertia=np.asarray(full_inertia,float); self.thrust_axis=np.asarray(thrust_axis,float)
        super().__init__(*args,**kwargs)
        base=copy.deepcopy(self.rigid.cfg); base['tether']['enabled']=False
        self.rigid=InstallationRigidPlant(base,self._arms,self._lengths,com_body=self.com_body,
            full_inertia=self.full_inertia,thrust_axis=self.thrust_axis,physics_dt_s=self.dt,
            drag_scales=kwargs.get('drag_scales'))

    def _xml(self): return full_xml(super()._xml(),self.full_inertia)


def virtual_vehicle(vehicle, calibration, speed_gain, *, allocation):
    v=copy.copy(vehicle)
    nominal=dict(airframe=dict(rotor_dx_m=vehicle.rotor_map[4,0].item(),rotor_dy_m=-vehicle.rotor_map[3,0].item()),
                 actuators=dict(kf=vehicle.kf,km=vehicle.km,include_gyro=False),aero=dict(body_rate_drag=[0,0,0]))
    d=-calibration.body_from_virtual[:,2].detach().numpy()
    _,torque=rotor_wrench(nominal,np.eye(4),np.zeros((4,4)),np.zeros((4,3)),d)
    V=calibration.body_from_virtual.detach().numpy()
    v.rotor_map=vehicle.rotor_map.clone()
    v.rotor_map[3:]=torch.tensor((torque@V/vehicle.kf).T).to(v.rotor_map)
    v.inv_allocation=torch.linalg.inv(torch.cat((-v.rotor_map[2:3],v.rotor_map[3:])))
    if allocation:
        v.pa*=speed_gain; v.pb*=speed_gain; v.pc*=speed_gain
        v.max_speed=vehicle.max_speed*speed_gain
        v.max_thrust=v.kf*v.max_speed**2
    return v


class InstallationEnvironment(BodyCalibrationEnvironment):
    def __init__(self,*args,installation,**kwargs):
        self.installation=installation
        self.mount=euler_matrix(installation['imu_mount_deg'])
        self.motor_axis=euler_matrix(installation['motor_mount_deg'])@np.array([0.,0.,-1.])
        super().__init__(*args,**kwargs)

    def make_native_plant(self):
        return InstallationFlexiblePlant(self.pc,self.model.arms,self.model.lengths,self.flexible_rope_options,
            com_body=self.actual_com,full_inertia=self.installation['full_inertia'],thrust_axis=self.motor_axis,
            device=self.device,physics_dt_s=self.physics_dt_s,
            drag_scales=[float(c.get('actual_body_drag_scale',1)) for c in self.specifications])

    def configure_raw_sensors(self):
        body_bias=self.mount@np.asarray(self.installation['gyro_bias_sensor'])
        self.sensors.gyro_bias.copy_(torch.tensor(body_bias).to(self.sensors.gyro_bias))

    def raw_sensor_packet(self,packet):
        C=torch.tensor(self.mount).to(packet.gyro)
        return Sensors(packet.attitude,packet.gyro@C,packet.specific_force@C,packet.motor_speed,packet.baro_height)

    def calibrated_vehicles(self):
        return (virtual_vehicle(self.vehicle,self.body_calibration,self.calibration_speed_gain,allocation=True),
                virtual_vehicle(self.vehicle,self.body_calibration,self.calibration_speed_gain,allocation=False),1.)
