"""Fit existing installation/bias/CG P3 from a completed real-PX4 Position capture."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from P2.hover_bptt.io import read,write,save,digest
from .installation_calibration import estimate_mount,body_samples,fit_axis_mass,fit_full_dynamics,score
from .body_calibration import prepare_position
from .actuator_calibration import fit_speed_gain
from .train_body_calibration import split_position


def excitation_window(samples):
    from .position_identification import POSITION_FIELDS,ODOMETRY_FIELDS
    active=np.asarray(samples['position_armed_airborne'],bool);indices=np.flatnonzero(active)
    if len(indices)<100 or not active[indices[0]:].all():raise ValueError('Need one continuous excited Position interval')
    start=float(samples['time_s'][indices[0]]);result={}
    for fields,clock,lower in ((POSITION_FIELDS,'time_s',start),(ODOMETRY_FIELDS,'position_time_s',start-.25)):
        mask=np.asarray(samples[clock])>=lower
        for key in fields:result[key]=np.asarray(samples[key])[mask].copy()
    return result


def align_motor_clock(samples,timeline):
    """Offline IMU interpolation onto measured ESC/command clock, within one split.

    Each 4ms command is retained so the first-order interval motor model sees
    its true zero-order hold. No actual motor time constant is supplied.
    """
    timeline=np.asarray(timeline,float);raw=np.asarray(samples['time_s'],float)
    if np.any(np.diff(raw)<=0) or np.any(np.diff(timeline[:,0])<=0):raise ValueError('Monotonic clocks required')
    rows=timeline[(timeline[:,0]>=raw[0]-1e-8)&(timeline[:,0]<=raw[-1]+1e-8)]
    if len(rows)<2:raise ValueError('No high-rate motor coverage in this raw partition')
    t=rows[:,0];out={k:np.asarray(v).copy() for k,v in samples.items() if k.startswith('position_') and k!='position_armed_airborne'}
    out['time_s']=t
    for name in ('specific_force_mps2','gyro_radps'):
        out[name]=np.column_stack([np.interp(t,raw,v) for v in np.asarray(samples[name]).T])
    out['motor_radps']=rows[:,14:18].copy();out['previous_throttle']=rows[:,18:22].copy()
    index=np.searchsorted(raw,t,side='right').clip(1,len(raw))-1
    out['position_armed_airborne']=np.asarray(samples['position_armed_airborne'])[index].copy()
    return out


def fit_aligned(samples,timeline,nominal,*,steps):
    raw,held,boundary=split_position(samples)
    # Split original sensor records BEFORE interpolation, pose derivatives or lag fits.
    training=align_motor_clock(raw,timeline);validation=align_motor_clock(held,timeline)
    mount=estimate_mount(training);body=body_samples(training,mount);held_body=body_samples(validation,mount)
    r=np.zeros(3)
    for _ in range(3):
        force_fit=fit_axis_mass(body,nominal,r);axis=force_fit['thrust_axis_body']
        data=prepare_position(body,nominal,controlled_external_loads=True,rotor_axes_body=axis)
        model,optimizer,report=fit_full_dynamics(data,steps=steps,mounting=mount,axis=axis)
        r=model.com_geometry.detach().numpy()
    val=prepare_position(held_body,nominal,controlled_external_loads=True,motor_response=data['motor_response'],rotor_axes_body=axis)
    report.update(mounting=mount,axis_mass=force_fit,train=score(model,data),holdout=score(model,val),
        axis_mass_holdout=fit_axis_mass(held_body,nominal,r,axis=axis,mass=force_fit['mass_kg']),
        raw_time_holdout_independent=True,split_time_s=boundary,total_adam_steps=3*steps,
        com_z_identified=False,motor_response=data['motor_response'],
        sampling='Original PX4 IMU split first, interpolated within partitions to actual 250Hz ESC/command clock',
        angular_fitting_reference='independent Position body pose derivative')
    report['development_quality_gate_passed']=(max(report['holdout']['torque_rmse_nm'])<.015
        and max(report['holdout']['accel_rmse_mps2'])<.08 and max(report['axis_mass_holdout']['force_rmse_n'])<.25)
    return model.requires_grad_(False),optimizer,report,training,validation


def train(capture,output,nominal,*,steps=1600):
    capture=Path(capture);output=Path(output);metadata=read(capture/'capture.json')
    if not metadata.get('real_PX4_Position') or not metadata.get('tether_force_disabled') or metadata.get('wind_mps')!=0:
        raise ValueError('Real Position and controlled zero wind/cable loads required')
    if metadata.get('Actor_used') or metadata.get('excitation_start_s') is None:
        raise ValueError('Need Position-only excitation before P3 fitting')
    summary=read(capture/'flight/summary.json')
    if summary['error'] is not None or summary['policy_ticks']:
        raise ValueError('Capture must complete safely without P2 takeover')
    samples=excitation_window(dict(np.load(capture/'position.npz')))
    timeline=np.load(capture/'raw_capture.npz')['independent_body']
    output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/px4_position_train.py','P3/installation_calibration.py','P3/body_calibration.py',
        'P3/position_identification.py','P3/train_body_calibration.py','P3/actuator_calibration.py']
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    input_sha=digest(capture/'position.npz')
    try:
        # No truth_only_physical values are passed to either fitter.
        model,optimizer,report,training,held=fit_aligned(samples,timeline,nominal,steps=steps)
        gain=fit_speed_gain(training,nominal)
        gain['holdout']=fit_speed_gain(held,nominal,speed_gain=gain['speed_gain'])
        report.update(input_sha256=input_sha,capture_metadata_sha256=digest(capture/'capture.json'),
            motor_timeline_input_sha256=digest(capture/'raw_capture.npz'),
            real_PX4_Position=True,independent_reference=metadata['independent_body_pose_reference'],
            P2_updates=0,source_sha256={n:digest(repo/n) for n in sources},
            px4_param_overrides=metadata['px4_param_overrides'])
        write(output/'actuator_gain.json',gain)
        record=dict(schema_version=2,model_kind='P3_installed_IMU_motor_axis_full_inertia',model=model.state_dict(),
            optimizer=optimizer.state_dict(),nominal=nominal,report=report)
        save(output/'calibration.pt',record)
        report['checkpoint_sha256']=digest(output/'calibration.pt');write(output/'training.json',report)
        print(dict(quality_passed=report['development_quality_gate_passed'],holdout=report['holdout'],
            axis_mass=report['axis_mass'],accel_bias=report['accel_bias_body_mps2'],com=report['com_geometry_m']),flush=True)
        if not report['development_quality_gate_passed']:raise RuntimeError('P3 holdout failed; do not enable compensation')
        return model,report
    except Exception as error:
        write(output/'failure.json',dict(error=f'{type(error).__name__}: {error}',input_sha256=input_sha,
            compensation_approved=False,truth_used_by_fitter=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--assets',type=Path,default=Path('px4_offboard/acceleration_dual/assets'))
    a=p.parse_args();train(a.capture,a.output,read(a.assets/'config.json')['nominal'],steps=a.steps)
