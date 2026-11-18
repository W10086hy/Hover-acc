"""Replay onboard source events through the deployed fixed pipeline; no truth input."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .control import Sensors
from .acceleration_runtime import AccelerationController
from .policy_timing import FixedPolicyRunner,atomic_json
from .offboard import relative_attitude
from .p3_observer import ObserverCalibration


def verify_records(records,cfg,weights,*,calibration=None,heading=0.,offset=None):
    controller=AccelerationController(copy.deepcopy(cfg),weights,calibration=calibration)
    controller.accel_offset=np.zeros(3) if offset is None else np.asarray(offset)
    runner=FixedPolicyRunner(controller);previous_entry=None;expected=[];actual=[];entries=set()
    for row in records:
        if row.get('event')=='source_packet':
            entry=row['entry_id'];entries.add(entry)
            if entry!=previous_entry:
                runner.reset();controller.observer.target_height_m=row['target_baro_height_m'];previous_entry=entry
            p=Sensors(np.array(row['attitude_wxyz']),np.array(row['gyro_radps']),np.array(row['force_mps2']),
                np.array(row['rpm_radps']),float(row['baro_height_m']))
            if calibration is not None:p=calibration.correct_packet(p)
            p=Sensors(relative_attitude(p.attitude,heading),p.gyro,p.specific_force,p.motor_speed,p.baro_height)
            expected.extend(runner.advance(p,np.array(row['applied_throttle']),row['source_stamp_s']))
        elif row.get('event')=='policy_tick':actual.append(row)
    assert expected and len(expected)==len(actual),(len(expected),len(actual))
    stats=dict(entries=len(entries),policy_ticks=len(actual),source_events=sum(r.get('event')=='source_packet' for r in records))
    fields=[('grid_stamp_s','time'),('observation','observation'),('actor_raw','raw_action'),('acceleration','acceleration')]
    if any('wire_acceleration' in row for row in actual):
        fields.append(('wire_acceleration','wire_acceleration'))
    for field,label in fields:
        a=np.asarray([r[field] for r in expected]);b=np.asarray([r[field] for r in actual])
        error=float(np.max(np.abs(a-b)));stats['max_'+label+'_error']=error
        assert error<1e-5,(field,error)
    stats.update(passed=True,P2_updates=0,horizontal_navigation_truth_used=False)
    return stats


def verify(folder,assets,model=None):
    folder,assets=Path(folder),Path(assets)
    runtime=json.loads((folder/'policy_runtime.json').read_text())
    cfg=json.loads((assets/'config.json').read_text())
    interface=json.loads((folder/'interface.json').read_text())
    cfg['deployment']=dict(acceleration_filter_s=interface['acceleration_filter_s'],
        innovation_filter_s=interface.get('innovation_filter_s',0.))
    records=[json.loads(line) for line in (folder/'policy_runtime_samples.jsonl').open(encoding='utf-8')]
    calibration=ObserverCalibration.load(model) if model else None
    result=verify_records(records,cfg,dict(np.load(assets/'weights.npz')),calibration=calibration,
        heading=runtime['launch_heading_rad'],offset=runtime['accel_offset'])
    atomic_json(folder/'source_replay.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--p3-model',type=Path);a=p.parse_args();print(verify(a.folder,a.assets,a.p3_model))
