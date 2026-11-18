"""Export an optional training-support gate; never update P2 or P3 weights."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from P2.hover_bptt.io import read,write,digest
from px4_offboard.aligned_p3 import AlignedCalibration
from .aligned_train import position_partitions,resample_position,partition_data,score
from .px4_position_train import align_motor_clock


def export(folder,samples_path,capture=None,*,fallback="zero"):
    folder,samples_path=Path(folder),Path(samples_path)
    tag='support_mean_export' if fallback=='training_mean' else 'support_export'
    destination=folder/('observer_supported_mean.json' if fallback=='training_mean' else 'observer_supported.json')
    if destination.exists():raise FileExistsError('Preserve earlier deployment candidate')
    report=read(folder/'training.json');parameters=read(folder/'observer.json')
    if not report['quality_passed'] or digest(samples_path)!=report['input_sha256']:
        raise ValueError('Passed checkpoint and original Position source required')
    checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    if digest(folder/'model.pt')!=report['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    nominal,model=checkpoint['nominal'],checkpoint['model']
    parameters['feature_support_gate']=dict(full_below=4.,zero_above=8.)
    parameters['feature_support_fallback']=fallback
    parameters['scope']+='; angular residual faded outside Position training support'
    AlignedCalibration(parameters)
    raw=dict(np.load(samples_path));first,last,boundary=position_partitions(raw)
    if capture:
        capture=Path(capture)
        if samples_path.resolve()!=(capture/'position.npz').resolve():raise ValueError('Capture identity mismatch')
        timeline=np.load(capture/'raw_capture.npz')['independent_body']
        first,last=align_motor_clock(first,timeline),align_motor_clock(last,timeline)
    scores={}
    for name,data in [('train',first),('holdout',last)]:
        data=partition_data(resample_position(data),nominal)
        scores[name]=score(data,model,nominal,np.array(parameters['angular_limit_radps2']),
            np.array(parameters['feature_mean']),np.array(parameters['feature_scale']),parameters)
    h=scores['holdout'];before=np.array(h['torque_before_rmse_nm']);after=np.array(h['torque_after_rmse_nm'])
    passed=(max(h['gyro_rmse_radps'])<.015 and max(h['force_rmse_mps2'])<.08
        and max(after)<.015 and (after<=np.maximum(.95*before,.0003)).all())
    parameters['quality_passed']=bool(passed)
    write(destination,parameters)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/aligned_export.py','P3/aligned_train.py','px4_offboard/aligned_p3.py','px4_offboard/policy_timing.py']
    with ZipFile(folder/(tag+'_source.zip'),'x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    write(folder/(tag+'.json'),dict(**scores,quality_passed=bool(passed),
        optimizer_updates=0,P2_updates=0,checkpoint_sha256=digest(folder/'model.pt'),
        observer_sha256=digest(destination),source_sha256={n:digest(repo/n) for n in sources},
        status='candidate_only',flight_validated=False,independent_capture_validated=False))
    print(dict(output=str(destination),quality_passed=bool(passed),holdout=h),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('samples',type=Path)
    p.add_argument('--capture',type=Path);p.add_argument('--fallback',choices=('zero','training_mean'),default='zero')
    a=p.parse_args();export(a.folder,a.samples,a.capture,fallback=a.fallback)
