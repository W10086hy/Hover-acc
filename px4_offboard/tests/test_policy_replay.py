import copy
import json
from pathlib import Path
import unittest
import numpy as np
from px4_offboard.control import Sensors
from px4_offboard.acceleration_runtime import AccelerationController
from px4_offboard.policy_timing import FixedPolicyRunner
from px4_offboard.offboard import relative_attitude

ROOT=Path(__file__).resolve().parents[2]


class PolicyReplayTests(unittest.TestCase):
    def test_source_events_replay_same_grid_observation_and_action_across_mode_entries(self):
        from px4_offboard.replay_policy import verify_records
        cfg=json.loads((ROOT/'px4_offboard/acceleration_dual/assets/config.json').read_text())
        weights=dict(np.load(ROOT/'px4_offboard/acceleration_dual/assets/weights.npz'))
        ctrl=AccelerationController(copy.deepcopy(cfg),weights);ctrl.accel_offset=np.array([.1,0,0])
        runner=FixedPolicyRunner(ctrl);records=[];heading=.2
        for entry,height in [(1,7.),(2,9.)]:
            runner.reset();ctrl.observer.target_height_m=height
            for t in entry+np.arange(8)/70:
                q=np.array([np.cos(heading/2),0,0,np.sin(heading/2)])
                raw=Sensors(q,np.array([.01*t,0,0]),np.array([.1,0,-9.8]),np.ones(4)*530,height)
                event=dict(event='source_packet',entry_id=entry,source_stamp_s=float(t),
                    attitude_wxyz=q.tolist(),gyro_radps=raw.gyro.tolist(),force_mps2=raw.specific_force.tolist(),
                    rpm_radps=raw.motor_speed.tolist(),applied_throttle=[.54]*4,baro_height_m=height,target_baro_height_m=height)
                records.append(event)
                packet=Sensors(relative_attitude(q,heading),raw.gyro,raw.specific_force,raw.motor_speed,height)
                for row in runner.advance(packet,np.ones(4)*.54,float(t)):
                    records.append(dict(row,event='policy_tick',entry_id=entry))
        result=verify_records(records,cfg,weights,heading=heading,offset=ctrl.accel_offset)
        self.assertEqual(result['entries'],2)
        self.assertLess(result['max_observation_error'],1e-7)
        self.assertLess(result['max_acceleration_error'],1e-7)
        broken=copy.deepcopy(records)
        next(row for row in broken if row['event']=='policy_tick')['observation'][0]+=.1
        with self.assertRaises(AssertionError):verify_records(broken,cfg,weights,heading=heading,offset=ctrl.accel_offset)


if __name__=='__main__':unittest.main()
