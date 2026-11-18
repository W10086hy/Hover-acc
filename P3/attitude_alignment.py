"""Position-only attitude-reference consistency; no navigation input at runtime."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from P2.hover_bptt.io import read,write,digest
from px4_offboard.network import rotation
from px4_offboard.aligned_p3 import AlignedCalibration


def tilt_quaternion(tilt):
    vector=np.r_[tilt,0.];angle=np.linalg.norm(vector)
    return np.r_[np.cos(angle/2),vector*(np.sin(angle/2)/angle if angle else .5)]


def gravity(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=q.T
    return np.c_[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]


def fit_tilt(reported,reference):
    a,b=gravity(reported),gravity(reference)
    tilt=np.zeros(2)
    for _ in range(12):
        residual=a@rotation(tilt_quaternion(tilt))-b
        columns=[]
        for k in range(2):
            trial=tilt.copy();trial[k]+=1e-6
            columns.append(((a@rotation(tilt_quaternion(trial))-b-residual)/1e-6).ravel())
        change=np.linalg.lstsq(np.column_stack(columns),-residual.ravel(),rcond=None)[0]
        tilt+=change
        if np.linalg.norm(tilt)>np.deg2rad(3):raise ValueError('Attitude discrepancy beyond supported3degree tilt')
        if np.linalg.norm(change)<1e-12:break
    return tilt_quaternion(tilt)


def pose_mask(times,reported,counter,start,end):
    valid=(times>=start)&(times<=end)&np.isfinite(reported).all(1)&(np.linalg.norm(reported,axis=1)>.5)
    if len(np.unique(counter[valid]))!=1:raise ValueError('Attitude reference reset within calibration window')
    return valid


def export(folder,capture):
    folder,capture=Path(folder),Path(capture)
    destination=folder/'observer_consistent.json'
    if destination.exists():raise FileExistsError('Preserve previous reference-alignment export')
    parameters=read(folder/'observer_supported_mean.json');training=read(folder/'training.json')
    if digest(capture/'position.npz')!=training['input_sha256']:raise ValueError('Original Position source required')
    meta=read(capture/'capture.json')
    if not meta['real_PX4_Position'] or meta['wind_mps']!=0 or not meta['tether_force_disabled'] or meta['Actor_used']:
        raise ValueError('Controlled Position and independent reference required')
    raw=np.load(capture/'raw_capture.npz')['independent_body']
    with np.load(capture/'flight/position_p3.npz') as z:
        times=z['position_time_s'];reported=z['position_attitude_wxyz'];counter=z['position_reset_counter']
    with np.load(capture/'position.npz') as z:
        active=z['time_s'][z['position_armed_airborne']]
    truth_q=raw[:,7:11].copy()
    for i in range(1,len(truth_q)):
        if truth_q[i]@truth_q[i-1]<0:truth_q[i]*=-1
    reference=np.column_stack([np.interp(times,raw[:,0],truth_q[:,k]) for k in range(4)])
    valid=pose_mask(times,reported,counter,active[0]+.25,active[-1]-.25)
    before=valid&(times<training['split_time_s']);held=valid&~before
    if before.sum()<100 or held.sum()<100:raise ValueError('Separate raw pose partitions required')
    q=fit_tilt(reported[before],reference[before]);R=rotation(q)
    metrics={}
    for name,mask in [('train',before),('holdout',held)]:
        a,b=gravity(reported[mask]),gravity(reference[mask])
        metrics[name]=dict(samples=int(mask.sum()),
            gravity_before_rmse_mps2=np.sqrt(np.mean((9.8*(a-b))**2,axis=0)).tolist(),
            gravity_after_rmse_mps2=np.sqrt(np.mean((9.8*(a@R-b))**2,axis=0)).tolist(),
            time_range_s=[float(times[mask][0]),float(times[mask][-1])])
    after=np.array(metrics['holdout']['gravity_after_rmse_mps2'])
    if max(after)>.08:raise ValueError('Independent pose holdout does not support static tilt correction')
    parameters['attitude_right_quaternion']=q.tolist()
    parameters['scope']+='; Position-trained attitude tilt consistency; launch yaw retained'
    AlignedCalibration(parameters)
    write(destination,parameters)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/attitude_alignment.py','px4_offboard/aligned_p3.py']
    with ZipFile(folder/'attitude_alignment_source.zip','x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    report=dict(**metrics,attitude_right_quaternion=q.tolist(),P2_updates=0,P3_weight_updates=0,
        fit_method='two-axis right-tilt Gauss Newton; yaw fixed',
        independent_reference_used_only_in_Position=True,raw_time_holdout_independent=True,
        input_sha256={n:digest(capture/n) for n in ['position.npz','raw_capture.npz','flight/position_p3.npz']},
        observer_sha256=digest(destination),checkpoint_sha256=digest(folder/'model.pt'),
        source_sha256={n:digest(repo/n) for n in sources},status='candidate_only')
    write(folder/'attitude_alignment.json',report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('capture',type=Path)
    a=p.parse_args();export(a.folder,a.capture)
