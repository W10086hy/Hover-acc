"""Train a separate nine-dimensional P3 candidate with deployment-matched timing."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch

from P2.hover_bptt.io import digest, read, save, write
from px4_offboard.control import Sensors
from px4_offboard.policy_timing import PolicySampler, filter_forces
from px4_offboard.aligned_p3 import PROCESSING, aligned_features, support_confidence, angular_prediction
from px4_offboard.unified_p3 import sensor_features, SENSOR_SCALE, nominal_alpha
from px4_offboard.p3_observer import physical_reference
from .unified_train import partition_data, fit_joint, validate_capture_conditions
from .train_body_calibration import split_position
from .px4_position_train import align_motor_clock, excitation_window


def position_partitions(samples):
    return split_position(excitation_window(samples))


def resample_position(samples):
    """Use exactly the online bracketed grid, independently inside each split."""
    source=np.asarray(samples['time_s'],float)
    active=np.asarray(samples['position_armed_airborne'])
    if len(source)<2 or active.dtype!=np.bool_ or not active.all():
        raise ValueError('Continuous airborne Position required; split mode changes first')
    sampler=PolicySampler();ticks=[]
    for i,t in enumerate(source):
        packet=Sensors(np.array([1.,0,0,0]),samples['gyro_radps'][i],samples['specific_force_mps2'][i],
            samples['motor_radps'][i],0.)
        ticks.extend(sampler.push(float(t),packet,samples['previous_throttle'][i]))
    out={k:np.asarray(v).copy() for k,v in samples.items() if k.startswith('position_') and k!='position_armed_airborne'}
    out.update(time_s=np.array([row[0] for row in ticks]),
        gyro_radps=np.array([row[1].gyro for row in ticks]),
        specific_force_mps2=np.array([row[1].specific_force for row in ticks]),
        motor_radps=np.array([row[1].motor_speed for row in ticks]),
        previous_throttle=np.array([row[2] for row in ticks]),
        position_armed_airborne=np.ones(len(ticks),dtype=bool))
    out['filtered_force']=filter_forces(out['specific_force_mps2'])
    return out


def runtime_arrays(data,sensor,nominal):
    previous=np.c_[data['previous_observed_gyro'],data['previous_filtered_force']]
    previous=previous+sensor_features(previous[:,:3],previous[:,3:])@sensor.T*SENSOR_SCALE
    current=data['filtered_force']+(sensor_features(data['previous_observed_gyro'],data['filtered_force'])@sensor.T*SENSOR_SCALE)[:,3:]
    features,square,mean=aligned_features(previous[:,:3],.5*(previous[:,3:]+current),data['previous_rpm'],data['rpm'])
    target=data['reference_alpha']-nominal_alpha(previous[:,:3],square,mean,nominal)
    return features,target,square,mean


def score(data,model,nominal,limits,feature_mean,feature_scale,parameters=None):
    sensor=model['sensor'].detach().numpy();angular=model['angular'].detach().numpy()
    corrected=data['observed']+data['sensor_x']@sensor.T*SENSOR_SCALE
    features,target,_,_=runtime_arrays(data,sensor,nominal)
    prediction=np.clip(features@angular.T,-limits,limits)
    if parameters is not None:prediction=angular_prediction(features,angular,limits,parameters)
    inertia=np.asarray(nominal['airframe']['inertia_body_kgm2'])
    rms=lambda v:np.sqrt(np.mean(v*v,axis=0)).tolist()
    before=np.asarray(rms(target*inertia));after=np.asarray(rms((target-prediction)*inertia))
    return dict(samples=len(target),gyro_rmse_radps=rms((corrected-data['target'])[:,:3]),
        force_rmse_mps2=rms((corrected-data['target'])[:,3:]),
        torque_before_rmse_nm=before.tolist(),torque_after_rmse_nm=after.tolist(),
        torque_improvement_fraction=(1-after/np.maximum(before,1e-12)).tolist(),
        feature_ood_sample_fraction=float(np.mean((np.abs((features-feature_mean)/feature_scale)>8).any(1))),
        source_time_range_s=data['time_range'])


def train(samples_path,output,nominal,*,capture=None,steps=400):
    samples_path=Path(samples_path);output=Path(output)
    if output.exists():raise FileExistsError('Use a new P3 output directory')
    metadata_path=(Path(capture) if capture else samples_path.parent)/'capture.json'
    metadata=read(metadata_path);conditions=validate_capture_conditions(metadata)
    samples=dict(np.load(samples_path,allow_pickle=False))
    if capture:
        capture=Path(capture)
        if samples_path.resolve()!=(capture/'position.npz').resolve():raise ValueError('Position identity mismatch')
        summary=read(capture/'flight/summary.json')
        if summary.get('error') is not None or summary.get('policy_ticks'):
            raise ValueError('Completed Position capture without P2 takeover required')
    first,last,boundary=position_partitions(samples)
    timeline_hash=None
    if capture:
        timeline_path=capture/'raw_capture.npz'
        timeline=np.load(timeline_path)['independent_body']
        first,last=align_motor_clock(first,timeline),align_motor_clock(last,timeline)
        timeline_hash=digest(timeline_path)
    first,last=resample_position(first),resample_position(last)
    training,held=partition_data(first,nominal),partition_data(last,nominal)
    torch.set_num_threads(1)
    model,optimizer=fit_joint(training,steps,nominal,array_builder=runtime_arrays,rate_drag=[0.,0.,0.])
    features,target,_,_=runtime_arrays(training,model['sensor'].detach().numpy(),nominal)
    inertia=np.asarray(nominal['airframe']['inertia_body_kgm2'])
    limits=np.maximum(1.25*np.quantile(np.abs(target*inertia),.999,axis=0),.0001)/inertia
    mean=features.mean(0);scale=np.maximum(features.std(0),.01)
    scores={key:score(value,model,nominal,limits,mean,scale) for key,value in [('train',training),('holdout',held)]}
    h=scores['holdout'];baseline=np.array(h['torque_before_rmse_nm']);after=np.array(h['torque_after_rmse_nm'])
    per_axis=(after<=np.maximum(.95*baseline,.0003))
    quality=(max(h['gyro_rmse_radps'])<.015 and max(h['force_rmse_mps2'])<.08
        and max(h['torque_after_rmse_nm'])<.015 and per_axis.all() and h['feature_ood_sample_fraction']<=.05)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/aligned_train.py','P3/unified_train.py','P3/position_identification.py',
        'P3/installation_calibration.py','P3/train_body_calibration.py','P3/px4_position_train.py',
        'px4_offboard/aligned_p3.py','px4_offboard/unified_p3.py','px4_offboard/policy_timing.py',
        'px4_offboard/acceleration_runtime.py','px4_offboard/control.py','px4_offboard/network.py']
    report=dict(**scores,processing=PROCESSING,input_sha256=digest(samples_path),
        capture_metadata_sha256=digest(metadata_path),motor_timeline_sha256=timeline_hash,
        split_time_s=boundary,raw_time_holdout_independent=True,capture_conditions=conditions,
        joint_adam_steps=steps,P2_updates=0,learned_output_dim=9,fitted_physical_source_parameters=[],
        quality_passed=bool(quality),per_axis_quality_passed=per_axis.tolist(),
        status='candidate_only',independent_capture_validated=False,closed_loop_validated=False,
        source_sha256={n:digest(repo/n) for n in sources})
    output.mkdir(parents=True)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    save(output/'model.pt',dict(model_kind='P3_aligned_observation_residual',model=model.state_dict(),
        optimizer=optimizer.state_dict(),nominal=nominal,report=report))
    report['checkpoint_sha256']=digest(output/'model.pt')
    write(output/'training.json',report)
    parameters=dict(schema='P3_aligned_residual_v2',quality_passed=bool(quality),processing=PROCESSING,
        checkpoint_sha256=report['checkpoint_sha256'],reference_physics=physical_reference(nominal),
        attitude_reference='PX4_body_FRD',sensor_weights=model['sensor'].detach().tolist(),
        angular_weights=model['angular'].detach().tolist(),angular_limit_radps2=limits.tolist(),
        feature_mean=mean.tolist(),feature_scale=scale.tolist(),status='candidate_only',
        closed_loop_validated=False,scope='Nine joint observation residuals; matched100Hz/filter; measuredRPM moments')
    write(output/'observer.json',parameters)
    print(dict(output=str(output),quality_passed=bool(quality),holdout=h),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--capture',type=Path);p.add_argument('--steps',type=int,default=400)
    p.add_argument('--controlled-external-loads',required=True,action='store_true')
    p.add_argument('--assets',type=Path,default=Path('px4_offboard/acceleration_dual/assets'))
    a=p.parse_args();train(a.samples,a.output,read(a.assets/'config.json')['nominal'],capture=a.capture,steps=a.steps)
