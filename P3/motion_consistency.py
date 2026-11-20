"""Position-only velocity-increment consistency screen, never an Actor input."""
import argparse
from pathlib import Path
import numpy as np
from P2.hover_bptt.io import read,write,digest
from px4_offboard.aligned_p3 import AlignedCalibration
from px4_offboard.control import Sensors
from px4_offboard.network import rotation
from .aligned_train import position_partitions,resample_position
from .px4_position_train import align_motor_clock
from .unified_train import validate_capture_conditions


def window_errors(time,rotations,force,velocity,gravity,window_s=2.):
    time,rotations,force,velocity=map(np.asarray,(time,rotations,force,velocity))
    if len(time)<3 or (np.diff(time)<=0).any():raise ValueError('Monotonic Position validation clock required')
    acceleration=np.einsum('nij,nj->ni',rotations,force)+[0.,0.,gravity]
    design=[];errors=[];durations=[];start=0
    while start<len(time)-1:
        end=int(np.searchsorted(time,time[start]+window_s))
        if end>=len(time):break
        if np.max(np.diff(time[start:end+1]))>.05:raise ValueError('Gapped Position validation interval')
        part=slice(start,end+1)
        design.append(np.trapezoid(rotations[part],time[part],axis=0))
        errors.append(np.trapezoid(acceleration[part],time[part],axis=0)-(velocity[end]-velocity[start]))
        durations.append(time[end]-time[start]);start=end
    if len(errors)<4:raise ValueError('At least four complete independent Position windows required')
    x,y=np.concatenate(design),np.concatenate(errors)
    bias,_,rank,singular=np.linalg.lstsq(x,y,rcond=None)
    if rank<3:raise ValueError('Unresolved body acceleration consistency')
    errors=np.asarray(errors)/np.asarray(durations)[:,None]
    return dict(windows=len(errors),window_s=window_s,equivalent_body_acceleration_residual_mps2=bias.tolist(),
        world_window_error_rmse_mps2=np.sqrt(np.mean(errors**2,axis=0)).tolist(),
        design_condition_number=float(singular[0]/singular[-1]),
        development_bias_screen_passed=bool(np.linalg.norm(bias[:2])<=.03 and abs(bias[2])<=.1),
        development_screen_limits_mps2=dict(body_xy_norm=.03,body_z=.1),
        screen_limits_are_development_only_not_formal_acceptance=True)


def audit(capture,model,output,offset=(0.,0.,0.),*,partition='holdout'):
    capture,model,output=Path(capture),Path(model),Path(output)
    if output.exists():raise FileExistsError('Preserve earlier motion consistency evidence')
    validate_capture_conditions(read(capture/'capture.json'))
    if read(capture/'flight/summary.json')['error'] is not None:raise ValueError('Completed Position required')
    calibration=AlignedCalibration(read(model))
    if partition not in ('train','holdout'):raise ValueError('Explicit raw time partition required')
    samples=dict(np.load(capture/'position.npz'));first,last,_=position_partitions(samples)
    held=first if partition=='train' else last
    body=np.load(capture/'raw_capture.npz')['independent_body']
    held=resample_position(align_motor_clock(held,body));time=held['time_s']
    with np.load(capture/'flight/position_p3.npz') as z:
        stamp=z['position_time_s'];q=z['position_attitude_wxyz'].copy()
        resets=z['position_reset_counter']
    mask=(stamp>=time[0])&(stamp<=time[-1])
    if len(np.unique(resets[mask]))!=1:raise ValueError('Attitude reset in Position validation')
    for k in range(1,len(q)):
        if q[k]@q[k-1]<0:q[k]*=-1
    q=np.column_stack([np.interp(time,stamp,q[:,k]) for k in range(4)])
    rotations=[];forces=[]
    for i in range(len(time)):
        packet=Sensors(q[i],held['gyro_radps'][i],held['filtered_force'][i],held['motor_radps'][i],0.)
        packet=calibration.correct_packet(packet)
        rotations.append(rotation(packet.attitude));forces.append(packet.specific_force-np.asarray(offset))
    rotations=np.asarray(rotations);forces=np.asarray(forces)
    velocity=np.column_stack([np.interp(time,body[:,0],body[:,k]) for k in (4,5,6)])
    # Exclude each raw partition's independent filter startup.
    valid=time>=time[0]+.25;time,rotations,forces,velocity=[v[valid] for v in (time,rotations,forces,velocity)]
    result=dict(baseline=window_errors(time,rotations,forces,velocity,9.8),
        injected_right015=window_errors(time,rotations,forces+[0.,.15,0.],velocity,9.8),
        reference='simulated independent body velocity; Position-only offline validation',
        raw_time_holdout_only=partition=='holdout',partition=partition,Actor_inputs_changed=False,optimizer_updates=0,
        preflight_force_offset_mps2=list(offset),model_sha256=digest(model),
        capture_sha256={n:digest(capture/n) for n in ('position.npz','raw_capture.npz','flight/position_p3.npz')},
        source_sha256=digest(__file__),formal_final_acceptance=False,
        interpretation='Equivalent residual includes attitude, force, timing and reference errors; not a unique physical accelerometer bias')
    write(output,result);print(result['baseline']);print('Injected:',result['injected_right015'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.capture,a.model,a.output)
