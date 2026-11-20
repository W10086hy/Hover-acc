"""Position-only causal angular increment fit, with frozen sensor calibration.

Independent pose still validates the sensor calibration and angular prediction.
Current gyro is a training target, never a current-step angular feature.
"""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch

from P2.hover_bptt.io import read,write,digest,save
from px4_offboard.unified_p3 import sensor_features,SENSOR_SCALE,nominal_alpha
from px4_offboard.aligned_p3 import AlignedCalibration,angular_prediction
from .aligned_train import position_partitions,resample_position,partition_data,runtime_arrays,score
from .unified_train import ridge,validate_capture_conditions
from .px4_position_train import align_motor_clock


def runtime_target(data,sensor,nominal):
    features,_,square,mean=runtime_arrays(data,sensor,nominal)
    previous=np.c_[data['previous_observed_gyro'],data['previous_filtered_force']]
    previous+=sensor_features(previous[:,:3],previous[:,3:])@sensor.T*SENSOR_SCALE
    raw=data['observed']
    current=raw[:,:3]+(sensor_features(raw[:,:3],raw[:,3:])@sensor[:3].T)*SENSOR_SCALE[:3]
    target=(current-previous[:,:3])/data['dt'][:,None]-nominal_alpha(previous[:,:3],square,mean,nominal)
    return features,target


def fit_features(features,*,balanced=False):
    transformed=np.array(features,copy=True)
    if balanced:transformed[:,13:17]-=transformed[:,13:17].mean(1,keepdims=True)
    indices=np.r_[np.arange(17),np.arange(25,28)]
    return transformed[:,indices],indices


def export_weights(compact,indices,*,balanced=False):
    angular=np.zeros((3,28));angular[:,indices]=compact
    if balanced:angular[:,13:17]-=angular[:,13:17].mean(1,keepdims=True)
    return angular


def train(capture,parent,output,nominal,steps=400,*,balanced_rpm=False):
    capture,parent,output=Path(capture),Path(parent),Path(output)
    if output.exists():raise FileExistsError('Preserve earlier increment candidate')
    validate_capture_conditions(read(capture/'capture.json'))
    summary=read(capture/'flight/summary.json')
    if summary['error'] is not None or summary.get('policy_ticks'):raise ValueError('Complete Position without P2 required')
    parameters=read(parent);AlignedCalibration(parameters).validate_reference(nominal)
    sensor=np.asarray(parameters['sensor_weights'])
    first,last,_=position_partitions(dict(np.load(capture/'position.npz')))
    timeline=np.load(capture/'raw_capture.npz')['independent_body']
    data=[partition_data(resample_position(align_motor_clock(part,timeline)),nominal) for part in (first,last)]
    features,target=runtime_target(data[0],sensor,nominal)
    # Remove duplicate, highly collinear previous/current RPM squared bases.
    compact,indices=fit_features(features,balanced=balanced_rpm)
    initial=ridge(compact,target)
    torch.set_num_threads(1)
    weights=torch.nn.Parameter(torch.tensor(initial))
    optimizer=torch.optim.Adam([weights],lr=1e-5)
    x,y=torch.tensor(compact),torch.tensor(target)
    inertia=np.asarray(nominal['airframe']['inertia_body_kgm2'])
    scale=torch.tensor(inertia/.01)
    for _ in range(steps):
        optimizer.zero_grad();loss=((x@weights.T-y)*scale).square().mean();loss.backward();optimizer.step()
    angular=export_weights(weights.detach().numpy(),indices,balanced=balanced_rpm)
    model={'sensor':torch.tensor(sensor),'angular':torch.tensor(angular)}
    limits=np.maximum(1.25*np.quantile(np.abs(target*inertia),.999,axis=0),.0001)/inertia
    mean,spread=features.mean(0),np.maximum(features.std(0),.01)
    parameters.update(angular_weights=angular.tolist(),angular_limit_radps2=limits.tolist(),
        feature_mean=mean.tolist(),feature_scale=spread.tolist(),feature_support_fallback='training_mean',
        status='candidate_only',closed_loop_validated=False)
    parameters['scope']+='; Position measured-increment angular target; compact RPM basis; frozen sensor calibration'
    if balanced_rpm:parameters['scope']+='; collective-free differential RPM basis'
    metrics={}
    for name,part in zip(('train','holdout'),data):
        pose=score(part,model,nominal,limits,mean,spread,parameters)
        f,t=runtime_target(part,sensor,nominal)
        error=(t-angular_prediction(f,angular,limits,parameters))*inertia
        pose['measured_increment_torque_rmse_nm']=np.sqrt(np.mean(error**2,axis=0)).tolist()
        pose['measured_increment_torque_mean_error_nm']=error.mean(0).tolist()
        metrics[name]=pose
    held=metrics['holdout']
    passed=(max(held['gyro_rmse_radps'])<.015 and max(held['force_rmse_mps2'])<.08
        and max(held['torque_after_rmse_nm'])<.015 and max(held['measured_increment_torque_rmse_nm'])<.015
        and (np.asarray(held['torque_after_rmse_nm'])<=np.maximum(.95*np.asarray(held['torque_before_rmse_nm']),.0003)).all())
    parameters['quality_passed']=bool(passed)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/impulse_train.py','P3/aligned_train.py','P3/unified_train.py','P3/px4_position_train.py',
        'P3/position_identification.py','P3/installation_calibration.py','P3/train_body_calibration.py',
        'px4_offboard/aligned_p3.py','px4_offboard/unified_p3.py','px4_offboard/policy_timing.py']
    report=dict(**metrics,parent_sha256=digest(parent),capture_sha256={n:digest(capture/n) for n in ('position.npz','raw_capture.npz','capture.json')},
        source_sha256={n:digest(repo/n) for n in sources},P2_updates=0,sensor_optimizer_updates=0,
        angular_adam_steps=steps,learned_output_dim=9,angular_feature_indices=indices.tolist(),
        collective_free_rpm=balanced_rpm,
        fitted_physical_source_parameters=[],raw_time_holdout_independent=True,quality_passed=bool(passed),
        angular_target='corrected measured gyro increment / 10ms minus nominal angular acceleration',
        status='candidate_only',formal_final_acceptance=False)
    output.mkdir(parents=True)
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    save(output/'model.pt',dict(model_kind='P3_aligned_observation_residual_increment',model=model,
        optimizer=optimizer.state_dict(),compact_angular=weights.detach(),nominal=nominal,report=report))
    report['checkpoint_sha256']=digest(output/'model.pt');parameters['checkpoint_sha256']=report['checkpoint_sha256']
    AlignedCalibration(parameters) if passed else None
    write(output/'training.json',report);write(output/'observer_accel_consistent.json',parameters)
    print(dict(output=str(output),quality_passed=bool(passed),holdout=held),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--balanced-rpm',action='store_true')
    args=parser.parse_args();train(args.capture,args.parent,args.output,read(args.assets/'config.json')['nominal'],balanced_rpm=args.balanced_rpm)
