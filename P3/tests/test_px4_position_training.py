import unittest
import numpy as np


class PX4PositionTrainingTests(unittest.TestCase):
    def test_motor_alignment_preserves_commands_inside_each_sparse_imu_interval(self):
        from P3.px4_position_train import align_motor_clock
        from P3.position_identification import _motor_fit
        time=np.arange(0,4,.004);tau=.002;poly=np.array([-387.3,1269.,-32.5])
        throttle=.5+.03*np.sin(20*time);target=np.polyval(poly,throttle);rpm=np.empty(len(time));rpm[0]=target[0]
        for k in range(1,len(time)):rpm[k]=target[k]+(rpm[k-1]-target[k])*np.exp(-.004/tau)
        truth=np.zeros((len(time),23));truth[:,0]=time;truth[:,14:18]=rpm[:,None];truth[:,18:22]=throttle[:,None]
        ix=np.unique(np.r_[np.arange(0,len(time),5),np.arange(2,len(time),5),len(time)-1]);t=time[ix]
        raw=dict(time_s=t,specific_force_mps2=np.zeros((len(t),3)),gyro_radps=np.zeros((len(t),3)),
            motor_radps=np.tile(rpm[ix,None],(1,4)),previous_throttle=np.tile(throttle[ix,None],(1,4)),position_armed_airborne=np.ones(len(t),bool),
            position_time_s=time,position_ned_m=np.zeros((len(time),3)),position_attitude_wxyz=np.tile([1,0,0,0],(len(time),1)),position_reset_counter=np.zeros(len(time),int))
        out=align_motor_clock(raw,truth)
        np.testing.assert_allclose(out['time_s'],time)
        np.testing.assert_allclose(out['previous_throttle'][:,0],throttle)
        dt=np.diff(time)[:,None];estimated,_,_,_=_motor_fit(out['motor_radps'][:-1],out['motor_radps'][1:],np.polyval(poly,out['previous_throttle'][1:]),dt,.002)
        self.assertAlmostEqual(estimated,tau,places=6)

    def test_holdout_is_split_within_excited_position_not_ground_warmup(self):
        from P3.px4_position_train import excitation_window
        from P3.train_body_calibration import split_position
        t=np.arange(0,10.001,.01);s=dict(time_s=t,position_armed_airborne=t>=8,
            specific_force_mps2=np.zeros((len(t),3)),gyro_radps=np.zeros((len(t),3)),
            motor_radps=np.ones((len(t),4)),previous_throttle=np.full((len(t),4),.5),
            position_time_s=t,position_ned_m=np.zeros((len(t),3)),
            position_attitude_wxyz=np.tile([1,0,0,0],(len(t),1)),position_reset_counter=np.zeros(len(t),int))
        w=excitation_window(s);train,held,cut=split_position(w)
        self.assertAlmostEqual(w['time_s'][0],8.)
        self.assertTrue(train['position_armed_airborne'].all());self.assertTrue(held['position_armed_airborne'].all())
        self.assertGreater(cut,8.);self.assertLess(cut,10.)
        self.assertLess(w['position_time_s'][0],8.)


if __name__=='__main__':unittest.main()
