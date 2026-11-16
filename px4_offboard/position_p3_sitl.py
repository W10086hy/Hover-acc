"""Real PX4 Position acquisition for P3; no Actor control and no cable/wind loads."""
import argparse
import json
import math
from pathlib import Path
import sys
import subprocess
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from .physics import Plant,S
from .network import rotation


class PositionPlant(Plant):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Keep the accepted body XML unchanged. Free cable nodes are unloaded
        # and gravity-compensated; zero tendon stiffness/damping transmits no load.
        self.k[:]=0.;self.c[:]=0.
        self.model.tendon_stiffness[self.tendons]=0.;self.model.tendon_damping[self.tendons]=0.
        self.model.body_gravcomp[self.nodes]=1.
        self.peak=0.
        self.mass=float(self.model.body_mass[self.body])
        self.last_throttle=np.zeros(4);self.clock=0.;self.position_active=False
    def advance(self,output,wind,period=.004):
        if np.linalg.norm(wind)>0:raise ValueError('Position calibration requires zero external wind')
        self.last_throttle=super().advance(output,wind,period)
        return self.last_throttle


class ExcitationGate:
    def __init__(self):self.steady_since=None;self.stable_completed=None;self.start=None
    def update(self,clock,*,steady,active):
        if not active:self.__init__();return
        self.steady_since=(self.steady_since if self.steady_since is not None else clock) if steady else None
        if self.stable_completed is None and self.steady_since is not None and clock-self.steady_since>=3:
            self.stable_completed=clock
        if self.start is None and self.stable_completed is not None and clock-self.stable_completed>=4:self.start=clock


def position_excitation(t):
    # PX4 applies manual-stick deadzones; tiny sinusoidal sticks are ineffective.
    return (round(160*math.sin(.91*t)),round(160*math.sin(1.13*t+.3)),
        500+round(100*math.sin(.73*t)),round(180*math.sin(1.47*t)))


def build_capture(measurements,truth,*,excitation_start):
    m=np.asarray(measurements,float);truth=np.asarray(truth,float)
    if m.ndim!=2 or m.shape[1]!=7 or truth.ndim!=2 or truth.shape[1]!=23:
        raise ValueError('Expected timestamp/IMU7 and timestamp/state17/throttle4/Position23')
    m=m[(m[:,0]>=truth[0,0])&(m[:,0]<=truth[-1,0])]
    m=m[np.r_[True,np.diff(m[:,0])>0]]
    t=m[:,0];clock=truth[:,0]
    rpm=np.column_stack([np.interp(t,clock,truth[:,k]) for k in range(14,18)])
    index=np.searchsorted(clock,t,side='left').clip(max=len(clock)-1)
    active=(truth[index,22]>0)&(t>=excitation_start)
    return dict(time_s=t,specific_force_mps2=m[:,1:4],gyro_radps=m[:,4:7],motor_radps=rpm.clip(min=0),
        previous_throttle=truth[index,18:22],position_armed_airborne=active,
        position_time_s=clock,position_ned_m=truth[:,1:4],position_attitude_wxyz=truth[:,7:11],
        position_reset_counter=np.zeros(len(clock),dtype=int))


def main():
    from .stock_transfer import stock_check_hold as legacy
    from .offboard import Offboard
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('px4_offboard/acceleration_dual/assets'))
    p.add_argument('--seconds',type=float,choices=(15.,30.,120.),default=120.)
    p.add_argument('--pitch-rate-k',type=float,help='Run-local MC_PITCHRATE_K override for inner-loop diagnosis')
    p.add_argument('--seed',type=int,default=2026114601);a=p.parse_args()
    if a.pitch_rate_k is not None and not 0<a.pitch_rate_k<=1:p.error('Pitch rate gain scale must be in (0,1]')
    root=a.output.resolve()
    if not root.is_relative_to(Path.cwd().resolve()) or root==Path.cwd().resolve():p.error('Use a new output subdirectory')
    root.mkdir(parents=True,exist_ok=False)
    plants=[];endpoints=[];measurements=[];truth=[];gate=ExcitationGate()
    def factory(*args,**kwargs):
        plant=PositionPlant(*args,**kwargs);plants.append(plant);return plant
    original_sensor=legacy.control_message
    def sensor(conn,clock,plant,fixture,rng,step,site_pressure_altitude,**kwargs):
        plant.clock=clock
        active=bool(endpoints and endpoints[0].last_heartbeat and endpoints[0].last_heartbeat.base_mode&128
            and ((endpoints[0].last_heartbeat.custom_mode>>16)&255)==3 and not fixture)
        plant.position_active=active
        truth.append(np.r_[clock,plant.x[:17],plant.last_throttle,int(active)])
        tilt=math.acos(np.clip(rotation(plant.x[6:10])[2,2],-1,1))
        steady=active and np.linalg.norm(plant.x[3:6])<.18 and abs(-plant.x[2]-plant.target_height)<.20 and tilt<math.radians(8)
        gate.update(clock,steady=steady,active=active)
        return original_sensor(conn,clock,plant,fixture,rng,step,site_pressure_altitude,**kwargs)
    class Endpoint(Offboard):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);endpoints.append(self)
            original=self.link.mav.manual_control_send
            def manual(target,x,y,z,r,buttons,*extra):
                if gate.start is not None and plants[0].position_active:
                    t=plants[0].clock-gate.start
                    # Gentle Position velocity/yaw/vertical stick excitation.
                    x,y,z,r=position_excitation(t)
                return original(target,x,y,z,r,buttons,*extra)
            self.link.mav.manual_control_send=manual
        def ingest(self,msg,clock):
            super().ingest(msg,clock)
            if msg.get_type()=='HIGHRES_IMU':
                stamp=msg.time_usec/1e6
                if not measurements or stamp>measurements[-1][0]:
                    measurements.append([stamp,msg.xacc,msg.yacc,msg.zacc,msg.xgyro,msg.ygyro,msg.zgyro])
    legacy.Plant=factory;legacy.Offboard=Endpoint;legacy.control_message=sensor
    with ZipFile(root/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for name in ('position_p3_sitl.py','physics.py','offboard.py','control.py','network.py','motor_mapping.py','stock_transfer/stock_check_hold.py'):
            z.write(Path(__file__).parent/name,'px4_offboard/'+name)
    original_popen=subprocess.Popen
    def px4_popen(args,*rest,**kwargs):
        proc=original_popen(args,*rest,**kwargs)
        if isinstance(args,list) and str(args[0]).endswith('/bin/px4') and a.pitch_rate_k is not None:
            proc.stdin.write(f'param set MC_PITCHRATE_K {a.pitch_rate_k}\nparam show MC_PITCHRATE_K\n');proc.stdin.flush()
        return proc
    subprocess.Popen=px4_popen
    sys.argv=[str(legacy.__file__),str(root/'flight'),'--assets',str(a.assets.resolve()),'--seconds',str(a.seconds),
        '--sensor-seed',str(a.seed),'--wind','0','--keep-position','--no-residual','--record-position-p3']
    try:legacy.main()
    finally:
        subprocess.Popen=original_popen
        np.savez_compressed(root/'raw_capture.npz',px4_imu=np.asarray(measurements),independent_body=np.asarray(truth))
        if measurements and truth and gate.start is not None:
            np.savez_compressed(root/'position.npz',**build_capture(measurements,truth,excitation_start=gate.start))
        if plants:
            plant=plants[0];R=rotation(plant.model.body_iquat[plant.body]);J=R@np.diag(plant.model.body_inertia[plant.body])@R.T
            metadata=dict(real_PX4_Position=True,Actor_used=False,wind_mps=0.,tether_force_disabled=True,
                independent_body_pose_reference='simulated independent body tracking, Position-only; not P2 input',
                IMU_source='PX4 HIGHRES_IMU; not merely injected HIL samples',ESC_source='simulated actual rotor speed and applied throttle',
                excitation_start_s=gate.start,truth_only_physical=dict(mass_kg=plant.mass,full_inertia_FRD=(J*S[:,None]*S[None,:]).tolist()),
                px4_param_overrides={'MC_PITCHRATE_K':a.pitch_rate_k} if a.pitch_rate_k is not None else {},
                sensor_seed=a.seed,planned_capture_s=a.seconds,structural_fit_must_pass_holdout=True)
            (root/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
            summary_path=root/'flight/summary.json';summary=json.loads(summary_path.read_text())
            summary.update(parameters_overridden=bool(metadata['px4_param_overrides']),px4_param_overrides=metadata['px4_param_overrides'])
            summary_path.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
