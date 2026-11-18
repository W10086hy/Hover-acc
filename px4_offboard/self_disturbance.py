"""Explicit simulated sensor biases and physically relocated mass; no Actor inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from .physics import S


def simulation_feedback(plant,clock,noise=None):
    rpm=plant.x[13:17].copy()
    if noise is not None:rpm+=noise
    return np.r_[np.maximum(rpm,0),plant.last_applied_throttle,float(clock)]


def make_assets(source,target):
    source,target=Path(source),Path(target)
    if target.exists():raise FileExistsError('Keep prior perturbation assets')
    cfg=json.loads((source/'config.json').read_text())
    tree=ET.fromstring((source/'model.xml').read_text())
    body=tree.find("./worldbody/body[@name='vehicle0']");item=body.find('inertial')
    mass=float(item.get('mass'));original_com=np.fromstring(item.get('pos'),sep=' ')*S
    if np.linalg.norm(original_com)>1e-12 or item.get('fullinertia') is not None:
        raise ValueError('Expected the explicitly indexed centered baseline XML')
    J=np.diag(np.fromstring(item.get('diaginertia'),sep=' '))
    if item.get('quat')!='1 0 0 0':raise ValueError('Expected baseline body inertia axes')
    com=np.array([.009,-.007,0.]);relocated=.2;point=com*mass/relocated
    # Move an existing central component, conserving total mass. Parallel-axis
    # transport first to the geometry origin, then to the new true center.
    axis=lambda r:np.dot(r,r)*np.eye(3)-np.outer(r,r)
    J=J+relocated*axis(point)-mass*axis(com)
    if np.linalg.eigvalsh(J).min()<=0:raise ValueError('Nonphysical inertia')
    item.set('pos',' '.join(map(str,com*S)))
    item.attrib.pop('diaginertia');item.attrib.pop('quat')
    xmlJ=J*S[:,None]*S[None,:]
    item.set('fullinertia',' '.join(map(str,[xmlJ[0,0],xmlJ[1,1],xmlJ[2,2],xmlJ[0,1],xmlJ[0,2],xmlJ[1,2]])))
    cfg=copy.deepcopy(cfg)
    cfg['actual']['airframe']['mass_kg']=mass
    cfg['actual']['airframe']['inertia_body_kgm2']=np.diag(J).tolist()
    cfg['simulation_self_disturbance']=dict(gyro_bias_radps=[.008,-.006,.004],accel_bias_mps2=[.15,-.12,.2],
        true_cg_body_m=com.tolist(),true_full_inertia_FRD=J.tolist(),actual_mass_kg=mass,
        relocated_component_kg=relocated,component_from_m=[0,0,0],component_to_m=point.tolist(),
        sensor_installation_deg=[0,0,0],motor_installation_deg=[0,0,0],
        scope='Synthetic additive sensor biases and real CG/inertia relocation; not controller inputs')
    target.mkdir(parents=True)
    for name in ('weights.npz','initial.npz'):shutil.copyfile(source/name,target/name)
    (target/'model.xml').write_text(ET.tostring(tree,encoding='unicode'),encoding='utf-8',newline='\n')
    (target/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8',newline='\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (target/'export.json').write_text(json.dumps(dict(parent_assets=str(source),
        parent_sha256={n:sha(source/n) for n in ('config.json','model.xml','weights.npz','initial.npz')},
        artifact_sha256={n:sha(target/n) for n in ('config.json','model.xml','weights.npz','initial.npz')},
        nominal_unchanged=True,P2_updates=0,disturbance=cfg['simulation_self_disturbance']),indent=2)+'\n',encoding='utf-8')
    return cfg


def wrap_sensor_sender(sender,cfg):
    disturbance=cfg.get('simulation_self_disturbance')
    if disturbance is None:return sender
    accel=np.asarray(disturbance['accel_bias_mps2']);gyro=np.asarray(disturbance['gyro_bias_radps'])
    class Mav:
        def __init__(self,original):self.original=original
        def __getattr__(self,name):return getattr(self.original,name)
        def hil_sensor_send(self,*args):
            values=list(args);values[1:4]=np.asarray(values[1:4])+accel;values[4:7]=np.asarray(values[4:7])+gyro
            return self.original.hil_sensor_send(*values)
    class Connection:
        def __init__(self,original):self.original=original;self.mav=Mav(original.mav)
        def __getattr__(self,name):return getattr(self.original,name)
    def wrapped(conn,*args,**kwargs):
        result=sender(Connection(conn),*args,**kwargs)
        samples=kwargs.get('imu_sample')
        if samples is not None and len(samples)==2:
            samples[0]=np.asarray(samples[0])+accel;samples[1]=np.asarray(samples[1])+gyro
        return result
    return wrapped


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('px4_offboard/acceleration_dual/assets'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();make_assets(a.source,a.output)
