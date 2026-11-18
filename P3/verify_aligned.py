"""Recompute candidate holdouts without fitting, from immutable Position sources."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from P2.hover_bptt.io import read,write,digest
from px4_offboard.aligned_p3 import AlignedCalibration
from .aligned_train import position_partitions,resample_position,partition_data,score
from .px4_position_train import align_motor_clock,excitation_window

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'P3/results/aligned_20261117'


def source(name):
    if name=='perturbed_px4_position':
        capture=BASE/'perturbed_position'
        return capture/'position.npz',capture
    capture=ROOT/'P3/results/px4_position_20261116/capture' if name=='px4_position' else None
    path=capture/'position.npz' if capture else ROOT/f'P3/results/installation_20261115/{name}/position.npz'
    return path,capture


def verify():
    torch.set_num_threads(1);results=[]
    names=['nominal_install','light_install','heavy_install','px4_position']
    if (BASE/'perturbed_px4_position/model.pt').exists():names.append('perturbed_px4_position')
    for name in names:
        folder=BASE/name;report=read(folder/'training.json');parameters=read(folder/'observer.json')
        model=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        assert model['optimizer']['state'] and model['report']['P2_updates']==0
        assert digest(folder/'model.pt')==report['checkpoint_sha256']==parameters['checkpoint_sha256']
        AlignedCalibration(parameters)
        with ZipFile(folder/'frozen_source.zip') as z:
            assert z.testzip() is None
            for path,sha in report['source_sha256'].items():assert hashlib.sha256(z.read(path)).hexdigest()==sha
        path,capture=source(name);assert digest(path)==report['input_sha256']
        first,last,boundary=position_partitions(dict(np.load(path)))
        assert first['time_s'][-1]<last['time_s'][0] and boundary==report['split_time_s']
        if capture:
            timeline=np.load(capture/'raw_capture.npz')['independent_body']
            assert digest(capture/'raw_capture.npz')==report['motor_timeline_sha256']
            first,last=align_motor_clock(first,timeline),align_motor_clock(last,timeline)
        held=partition_data(resample_position(last),model['nominal'])
        actual=score(held,model['model'],model['nominal'],np.array(parameters['angular_limit_radps2']),
            np.array(parameters['feature_mean']),np.array(parameters['feature_scale']))
        for key in ('gyro_rmse_radps','force_rmse_mps2','torque_after_rmse_nm','torque_before_rmse_nm'):
            np.testing.assert_allclose(actual[key],report['holdout'][key],rtol=1e-10,atol=1e-12)
        results.append(dict(aircraft=name,sha256=report['checkpoint_sha256'],holdout=actual))
        for observer,tag in [('observer_supported.json','support_export'),
                             ('observer_supported_mean.json','support_mean_export')]:
            if not (folder/observer).exists():continue
            p=read(folder/observer);AlignedCalibration(p)
            export=read(folder/(tag+'.json'))
            assert digest(folder/observer)==export['observer_sha256']
            assert export['checkpoint_sha256']==digest(folder/'model.pt')
            assert export['optimizer_updates']==export['P2_updates']==0
            np.testing.assert_array_equal(p['sensor_weights'],parameters['sensor_weights'])
            np.testing.assert_array_equal(p['angular_weights'],parameters['angular_weights'])
            gated=score(held,model['model'],model['nominal'],np.array(p['angular_limit_radps2']),
                np.array(p['feature_mean']),np.array(p['feature_scale']),p)
            np.testing.assert_allclose(gated['torque_after_rmse_nm'],export['holdout']['torque_after_rmse_nm'],rtol=1e-10)
            with ZipFile(folder/(tag+'_source.zip')) as z:
                for path,sha in export['source_sha256'].items():assert hashlib.sha256(z.read(path)).hexdigest()==sha
        if (folder/'observer_consistent.json').exists():
            verify_attitude_export(folder,capture)
        for observer,tag,key,value in (
            ('observer_virtual.json','actuation_export','actuation_observation','standard_vehicle_equivalent'),
            ('observer_accel_consistent.json','acceleration_export','acceleration_actuation','attitude_reference_consistent'),
        ):
            if not (folder/observer).exists():continue
            p=read(folder/observer);AlignedCalibration(p)
            base=read(folder/'observer_consistent.json');proof=read(folder/(tag+'.json'))
            assert proof['parent_sha256']==digest(folder/'observer_consistent.json')
            assert proof['observer_sha256']==digest(folder/observer)
            assert proof['checkpoint_sha256']==digest(folder/'model.pt')
            assert proof['P2_updates']==proof['P3_weight_updates']==0
            assert p[key]==value
            assert set(p)==set(base)|{key}
            for field in base:
                if field!='scope':assert p[field]==base[field],field
            with ZipFile(folder/(tag+'_source.zip')) as z:
                for path,sha in proof['source_sha256'].items():assert hashlib.sha256(z.read(path)).hexdigest()==sha
    actor=read(ROOT/'P2/models.json')['current']
    assert digest(ROOT/actor['checkpoint'])==actor['sha256']=='4743e4e7d40f26c2e3baea8ad8a6ee94bc14c2710838c9fd94235211abdb1ab5'
    return dict(passed=True,P2_updates=0,candidates=results,default_model_replaced=False)


def pose_validation(capture,parameters,start,end):
    from .attitude_alignment import gravity,pose_mask
    from px4_offboard.network import rotation
    capture=Path(capture)
    raw=np.load(capture/'raw_capture.npz')['independent_body'];truth=raw[:,7:11].copy()
    for i in range(1,len(truth)):
        if truth[i]@truth[i-1]<0:truth[i]*=-1
    with np.load(capture/'flight/position_p3.npz') as z:
        t=z['position_time_s'];q=z['position_attitude_wxyz'];counter=z['position_reset_counter']
    reference=np.column_stack([np.interp(t,raw[:,0],truth[:,k]) for k in range(4)])
    valid=pose_mask(t,q,counter,start,end)
    assert valid.sum()>=100
    a,b=gravity(q[valid]),gravity(reference[valid])
    R=rotation(np.array(parameters['attitude_right_quaternion']))
    return dict(samples=int(valid.sum()),
        gravity_before_rmse_mps2=np.sqrt(np.mean((9.8*(a-b))**2,axis=0)).tolist(),
        gravity_after_rmse_mps2=np.sqrt(np.mean((9.8*(a@R-b))**2,axis=0)).tolist(),
        time_range_s=[float(t[valid][0]),float(t[valid][-1])])


def verify_attitude_export(folder,capture):
    p=read(folder/'observer_consistent.json');AlignedCalibration(p)
    base=read(folder/'observer_supported_mean.json');proof=read(folder/'attitude_alignment.json')
    assert digest(folder/'observer_consistent.json')==proof['observer_sha256']
    assert proof['checkpoint_sha256']==digest(folder/'model.pt')
    assert proof['P2_updates']==proof['P3_weight_updates']==0
    for key in base:
        if key!='scope':assert p[key]==base[key],key
    for path,sha in proof['input_sha256'].items():assert digest(capture/path)==sha
    with ZipFile(folder/'attitude_alignment_source.zip') as z:
        for path,sha in proof['source_sha256'].items():assert hashlib.sha256(z.read(path)).hexdigest()==sha
    for name in ['train','holdout']:
        actual=pose_validation(capture,p,*proof[name]['time_range_s'])
        assert actual['samples']==proof[name]['samples']
        for key in ['gravity_before_rmse_mps2','gravity_after_rmse_mps2']:
            np.testing.assert_allclose(actual[key],proof[name][key],rtol=1e-10,atol=1e-12)


def independent(capture,model_folder=None,observer='observer_supported.json',output=None):
    capture=Path(capture);metadata=read(capture/'capture.json');flight=read(capture/'flight/summary.json')
    assert metadata['real_PX4_Position'] and metadata['wind_mps']==0 and metadata['tether_force_disabled']
    assert not metadata['Actor_used'] and flight['error'] is None
    folder=Path(model_folder) if model_folder else BASE/'px4_position';parameters=read(folder/observer)
    AlignedCalibration(parameters)
    model=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    samples=excitation_window(dict(np.load(capture/'position.npz')))
    samples=align_motor_clock(samples,np.load(capture/'raw_capture.npz')['independent_body'])
    data=partition_data(resample_position(samples),model['nominal'])
    actual=score(data,model['model'],model['nominal'],np.array(parameters['angular_limit_radps2']),
        np.array(parameters['feature_mean']),np.array(parameters['feature_scale']),parameters)
    trace=np.load(capture/'flight/trace.npy');start=flight['switch_heartbeat_s'];duration=flight['requested_post_duration_s']
    post=trace[(trace[:,0]>=start-1e-7)&(trace[:,0]<start+duration-1e-7)]
    assert len(post)==round(duration/.004) and (post[:,1]==3).all() and (post[:,2]==1).all() and (post[:,33]==0).all()
    before=np.array(actual['torque_before_rmse_nm']);after=np.array(actual['torque_after_rmse_nm'])
    passed=(max(actual['gyro_rmse_radps'])<.015 and max(actual['force_rmse_mps2'])<.08
        and max(after)<.015 and (after<=np.maximum(.95*before,.0003)).all() and actual['feature_ood_sample_fraction']<=.05)
    result=dict(independent_capture=True,quality_passed=bool(passed),complete_Position_s=duration,optimizer_updates=0,P2_updates=0,
        checkpoint_sha256=digest(folder/'model.pt'),position_sha256=digest(capture/'position.npz'),
        summary_sha256=digest(capture/'flight/summary.json'),score=actual)
    if 'attitude_right_quaternion' in parameters:
        result['attitude_reference']=pose_validation(capture,parameters,*data['time_range'])
        result['attitude_reference_passed']=max(result['attitude_reference']['gravity_after_rmse_mps2'])<.08
        result['quality_passed']=bool(passed and result['attitude_reference_passed'])
    target=Path(output) if output else BASE/'independent_validation.json'
    if target.exists():raise FileExistsError('Preserve previous independent validation; choose a new output')
    write(target,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--independent-capture',type=Path)
    p.add_argument('--model-folder',type=Path);p.add_argument('--observer',default='observer_supported.json')
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=independent(a.independent_capture,a.model_folder,a.observer,a.output) if a.independent_capture else verify()
    print(json.dumps(result,indent=2))
