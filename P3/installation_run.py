"""Train P3 on installation/payload perturbations and compare a frozen P2."""
import argparse
import copy
import shutil
from functools import partial
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch

from P2.hover_bptt.io import load_checkpoint,read,write,save,digest
from P2.hover_bptt.vehicle_spec import VehicleSpec,identity
from P2.hover_bptt.flexible_training import source_manifest
from P2.hover_bptt.evaluation import evaluate_one
from P2.hover_bptt.generalization import cases_at
from P2.evaluate_acceleration import hover_gate,STATIONARITY
from P2.verify_stationarity import verify_evaluation
from .body_calibration_sim import collect_position
from .body_calibration_run import CHECKPOINT,REPO,calibrate_speed_gain
from .installation_calibration import train_installed
from .installation_sim import euler_matrix,payload_properties,InstallationRigidPlant,InstallationEnvironment
from .train_body_calibration import load_calibration

PAYLOAD=[dict(mass_kg=.04,position_m=[.12,.06,.03]),dict(mass_kg=.04,position_m=[-.04,-.1,-.03])]
PROFILES=[dict(name='nominal_install',mass=1.,inertia=[1.,1.,1.],motor_tau=1.,motor_gain=1.,
    imu_mount_deg=[2.,-1.,1.],motor_mount_deg=[1.,-1.5,0.],bias_sensor=[.16,-.11,.22],base_com=[.008,-.006,0.],height=5.),
    dict(name='light_install',mass=.9,inertia=[.95,1.05,.95],motor_tau=.9,motor_gain=1.1,
    imu_mount_deg=[-3.,2.,-1.5],motor_mount_deg=[-1.5,1.,0.],bias_sensor=[-.18,.14,-.16],base_com=[-.007,.009,0.],height=8.),
    dict(name='heavy_install',mass=1.1,inertia=[1.05,.95,1.05],motor_tau=1.1,motor_gain=.9,
    imu_mount_deg=[3.,-3.,2.],motor_mount_deg=[1.5,1.5,0.],bias_sensor=[.12,.2,-.18],base_com=[.01,.006,0.],height=12.)]
ROOT=REPO/'P3/results/installation_20261115'


def physical(nominal,profile):
    base=VehicleSpec.from_config(nominal).scaled(**{k:profile[k] for k in ('mass','inertia','motor_tau','motor_gain')})
    mass,com,J=payload_properties(base.mass_kg,np.diag(base.inertia_body_kgm2),profile['base_com'],PAYLOAD)
    cfg=base.apply(nominal)
    cfg['airframe']['mass_kg']=mass; cfg['airframe']['inertia_body_kgm2']=np.diag(J).tolist()
    install=dict(imu_mount_deg=profile['imu_mount_deg'],motor_mount_deg=profile['motor_mount_deg'],
        full_inertia=J.tolist(),gyro_bias_sensor=[.001,-.002,.0015],payloads=PAYLOAD,
        total_mass_kg=mass,com_geometry_m=com.tolist())
    return cfg,install


def train(root):
    if root.exists(): raise FileExistsError('Preserve previous installation run')
    root.mkdir(parents=True)
    saved,sha=load_checkpoint(CHECKPOINT); nominal=saved['plant_configuration']
    sources=source_manifest()
    for name in ('P3/installation_run.py','P3/installation_sim.py','P3/installation_calibration.py',
        'P3/body_calibration.py','P3/body_calibration_sim.py','P3/body_transport.py','P3/actuator_calibration.py',
        'P3/train_body_calibration.py','P3/position_identification.py','P3/body_calibration_run.py',
        'mujoco_sim/physics.py','mujoco_sim/flexible.py','P2/evaluate_acceleration.py','P2/verify_stationarity.py'):
        sources[name]=digest(REPO/name)
    manifest=dict(profiles=PROFILES,payloads=PAYLOAD,source_sha256=sources,p2_checkpoint=str(CHECKPOINT.relative_to(REPO)).replace('\\','/'),
        p2_sha256=sha,capture_seed=2026112501,evaluation_seed=2026113501,
        capture_s=42.,duration_s=120.,steps_per_iteration=1600,formal_final_acceptance=False,
        independent_body_attitude_reference=True,imu_and_motor_mounting='unknown to P3 until fitted',
        position_controller='local surrogate, not real PX4',motor_axes='common cant; fixed individual rotor positions')
    write(root/'manifest.json',manifest)
    with ZipFile(root/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for name in sources:z.write(REPO/name,name)
    for i,p in enumerate(PROFILES):
        print(dict(phase='capture',profile=p['name']),flush=True)
        folder=root/p['name']; folder.mkdir()
        actual,install=physical(nominal,p)
        previous=REPO/'P3/results/installation_20261115'/p['name']
        if i==0 and (previous/'calibration/training.json').exists():
            old=read(previous/'capture.json'); trained=read(previous/'calibration/training.json')
            if old['installation']!=install or trained['input_sha256']!=digest(previous/'position.npz'):
                raise RuntimeError('Cannot reuse a changed nominal installation')
            for name in ('position.npz','capture.json','actuator_gain.json'):
                shutil.copy2(previous/name,folder/name)
            shutil.copytree(previous/'calibration',folder/'calibration')
            shutil.copy2(previous.parent/'frozen_source.zip',root/'reused_nominal_source.zip')
            write(folder/'reuse.json',dict(source=previous.relative_to(REPO).as_posix(),
                input_sha256=digest(folder/'position.npz'),checkpoint_sha256=digest(folder/'calibration/calibration.pt'),
                reason='Same physical perturbation and acquisition seed; only other aircraft IMU angles changed'))
            print(dict(phase='reuse_nominal_calibration',profile=p['name']),flush=True)
            continue
        C=euler_matrix(p['imu_mount_deg']); axis=euler_matrix(p['motor_mount_deg'])@np.array([0.,0.,-1.])
        factory=partial(InstallationRigidPlant,full_inertia=install['full_inertia'],thrust_axis=axis)
        data,metadata=collect_position(nominal,actual,saved['configuration']['controller'],
            bias=C@p['bias_sensor'],com_body=install['com_geometry_m'],seconds=manifest['capture_s'],
            height=p['height'],seed=manifest['capture_seed']+100*i,plant_factory=factory,
            sensor_transform=lambda g,f:(g@C,f@C),gyro_bias_body=C@install['gyro_bias_sensor'])
        np.savez_compressed(folder/'position.npz',**data)
        write(folder/'capture.json',dict(metadata,installation=install,position_npz_sha256=digest(folder/'position.npz')))
        model,opt,report=train_installed(data,nominal,steps=manifest['steps_per_iteration'],controlled_external_loads=True)
        Cfit=model.body_from_sensor.numpy(); dfit=-model.body_from_virtual[:,2].numpy()
        report['physical_error']=dict(imu_rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(Cfit.T@C)-1)/2,-1,1)))),
            motor_axis_deg=float(np.degrees(np.arccos(np.clip(dfit@axis,-1,1)))),
            com_error_mm=((model.com_geometry.numpy()-install['com_geometry_m'])*1000).tolist(),
            full_inertia_relative_frobenius=float(np.linalg.norm(model.full_inertia.numpy()-install['full_inertia'])/np.linalg.norm(install['full_inertia'])))
        report['input_sha256']=digest(folder/'position.npz')
        record=dict(schema_version=2,model_kind='P3_installed_IMU_motor_axis_full_inertia',model=model.state_dict(),
            optimizer=opt.state_dict(),nominal=nominal,report=report)
        (folder/'calibration').mkdir(); save(folder/'calibration/calibration.pt',record)
        report['checkpoint_sha256']=digest(folder/'calibration/calibration.pt'); write(folder/'calibration/training.json',report)
        calibrate_speed_gain(folder,nominal)
        print(dict(phase='trained',profile=p['name'],quality=report['development_quality_gate_passed'],errors=report['physical_error']),flush=True)
        if not report['development_quality_gate_passed']: raise RuntimeError('Installation calibration failed holdout gate')


def validate(root,profile_name,seconds):
    m=read(root/'manifest.json')
    if any(digest(REPO/k)!=v for k,v in m['source_sha256'].items()):raise RuntimeError('Source changed since snapshot')
    saved,sha=load_checkpoint(CHECKPOINT)
    if sha!=m['p2_sha256']:raise RuntimeError('P2 changed')
    nominal=saved['plant_configuration']; rope=read(REPO/'P2/acceleration_hover_resume.json')['rope']
    for i,p in enumerate(m['profiles']):
        if p['name']!=profile_name:continue
        folder=root/p['name']; actual,install=physical(nominal,p)
        model,record=load_calibration(folder/'calibration/calibration.pt')
        mass=record['report']['axis_mass']['mass_kg']; gain=read(folder/'actuator_gain.json')['speed_gain']
        C=euler_matrix(p['imu_mount_deg'])
        for enabled in (False,True):
            mode='on' if enabled else 'off'; out=folder/(('smoke_' if seconds<120 else 'flight_')+mode)
            if out.exists():
                if (out/'audit.json').exists():continue
                raise RuntimeError('Preserve incomplete output '+str(out))
            records=[]; built=[]
            def factory(*args,**kwargs):
                env=InstallationEnvironment(*args,installation=install,rope=rope,physics_dt_s=.0005,
                    actual_bias=C@p['bias_sensor'],actual_com=install['com_geometry_m'],
                    calibration=model if enabled else None,calibration_mass=mass,calibration_speed_gain=gain,
                    compensation_mode='reference_transport',**kwargs)
                step=env.step
                def logged(raw):
                    result=step(raw); info=result[-1]
                    records.append((env.state[:,:13].numpy().copy(),info['failure_code'].numpy().copy(),
                        info['invalid_policy_action'].numpy().copy(),info['flexible_interval_peak_tension_n'].numpy().copy()))
                    return result
                env.step=logged; built.append(env); return env
            print(dict(phase='validate',profile=p['name'],mode=mode,seconds=seconds),flush=True)
            result=evaluate_one(CHECKPOINT,out,m['evaluation_seed']+100*i,cases_at([67.5],[(1,'steady'),(5,'steady')]),
                duration_s=seconds,environment_factory=factory,vehicle_spec=VehicleSpec.from_config(actual),
                nominal_vehicle_spec=VehicleSpec.from_config(nominal),plant_configuration=nominal,
                checkpoint_snapshot=(saved,sha),stage_label=f'安装倾斜/偏载 P2'+('+P3' if enabled else ''))
            env=built[0]
            body,codes,invalid,tension=map(np.stack,zip(*records))
            np.savez_compressed(out/'fullrate.npz',body_state_ned=body,failure_code=codes,
                invalid_policy_action=invalid,interval_peak_tension_n=tension,target_w_m=env.model.target.numpy(),dt_s=env.dt)
            for c in result['cases']:
                c['hover_gate']=hover_gate(c);c['stationarity_working_criteria_passed']=c['hover_gate']['settled']
            result.update(engine='native_MuJoCo_installation_and_payload',p3_enabled=enabled,installation=install,
                installation_sha256=identity(install),rope=rope,policy_updates=0,optimizer_created=False,
                p3_checkpoint_sha256=digest(folder/'calibration/calibration.pt'),actuator_gain_sha256=digest(folder/'actuator_gain.json'),
                fullrate_evidence_sha256=digest(out/'fullrate.npz'),offline_limits=env.cfg['offline_limits'],
                stationarity_working_criteria=STATIONARITY,physics_dt_s=.0005,mj_step_calls=env.model.mj_step_calls)
            write(out/'result.json',result);write(out/'audit.json',verify_evaluation(out))
            (out/'scene.xml').write_text(env.model.xml,encoding='utf-8')
    if any(digest(REPO/k)!=v for k,v in m['source_sha256'].items()):raise RuntimeError('Source changed during run')


def refresh_sources(root, reason):
    m=read(root/'manifest.json')
    if m['profiles']!=PROFILES:raise ValueError('Physical profiles changed; use a separate run')
    revisions=m.get('source_revisions',[dict(archive='frozen_source.zip',source_sha256=m['source_sha256'])])
    current={k:digest(REPO/k) for k in m['source_sha256']}
    name=f'frozen_source_{len(revisions):02d}.zip'
    with ZipFile(root/name,'x',ZIP_DEFLATED) as z:
        for path in current:z.write(REPO/path,path)
    revisions.append(dict(archive=name,source_sha256=current,reason=reason))
    m.update(source_sha256=current,source_revisions=revisions)
    write(root/'manifest.json',m)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT);parser.add_argument('--phase',choices=['train','smoke','validate','snapshot'],required=True)
    parser.add_argument('--profile',choices=[p['name'] for p in PROFILES],default=PROFILES[0]['name'])
    parser.add_argument('--snapshot-reason',default='Maintenance snapshot; earlier experiment source revisions retained')
    args=parser.parse_args(); root=args.output.resolve()
    if not root.is_relative_to(REPO/'P3/results'):raise ValueError('Keep outputs in P3/results')
    torch.set_num_threads(1)
    if args.phase=='train':train(root)
    elif args.phase=='snapshot':refresh_sources(root,args.snapshot_reason)
    else:validate(root,args.profile,2. if args.phase=='smoke' else 120.)


if __name__=='__main__':main()
