"""Run the existing stock-PX4 fixture with a frozen Actor and actual MAVROS output."""
import argparse
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from .offboard import Offboard
from .acceleration_runtime import AccelerationController
from .mavros_acceleration import MavrosSink
from .wire_proxy import WireProxy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interface',choices=('acceleration','rate_thrust'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--seconds',type=float,choices=(15.,30.,120.),default=120.);p.add_argument('--wind',type=float,default=1.)
    p.add_argument('--wind-ramp-s',type=float,default=0.)
    p.add_argument('--seed',type=int,default=2026114501)
    p.add_argument('--policy-source',choices=('mavlink','mavros'),default='mavros')
    p.add_argument('--pitch-rate-k',type=float,help='Explicit run-local MC_PITCHRATE_K, without changing frozen baselines')
    p.add_argument('--p3-calibration',type=Path)
    p.add_argument('--acceleration-filter-s',type=float,default=.065);a=p.parse_args()
    if not np.isfinite(a.acceleration_filter_s) or a.acceleration_filter_s<0:p.error('Nonnegative finite command filter required')
    if a.pitch_rate_k is not None and not 0<a.pitch_rate_k<=1:p.error('Pitch rate gain scale must be in (0,1]')
    if a.p3_calibration and a.policy_source!='mavros':p.error('P3 standardization currently requires the MAVROS sensor node')
    p3_data=json.loads(a.p3_calibration.read_text()) if a.p3_calibration else None
    p3_hash=hashlib.sha256(a.p3_calibration.read_bytes()).hexdigest() if a.p3_calibration else None
    root=a.output.resolve()
    if not root.is_relative_to(Path.cwd().resolve()) or root==Path.cwd().resolve():p.error('Output must be a new subdirectory of this project')
    root.mkdir(parents=True,exist_ok=False)
    asset_hashes={n:hashlib.sha256((a.assets/n).read_bytes()).hexdigest() for n in ('config.json','weights.npz','initial.npz','model.xml')}
    env=os.environ.copy();env.update(ROS_MASTER_URI='http://127.0.0.1:11315',ROS_HOME=str(root/'ros_home'),ROS_LOG_DIR=str(root/'ros_logs'))
    os.environ.update(env)
    from .ros_compat import load_rospy
    rospy=load_rospy()
    processes=[];handles=[];endpoint=[];original_popen=subprocess.Popen
    proxy=WireProxy();atexit.register(proxy.close)
    service_env=dict(env,PATH='/opt/ros/noetic/bin:/usr/bin:/bin:/usr/sbin:/sbin',PYTHONNOUSERSITE='1')
    def stopped(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    def terminate(proc):
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    def launch(args):
        log=(root/(str(len(processes))+'_service.log')).open('w');handles.append(log)
        proc=original_popen(args,env=service_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);atexit.register(terminate,proc);return proc
    master=launch(['/opt/ros/noetic/bin/roscore','-p','11315'])
    deadline=time.monotonic()+20.
    while True:
        try:
            with socket.create_connection(('127.0.0.1',11315),timeout=.25):break
        except OSError:
            if master.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Task-local ROS master failed; see service log')
            time.sleep(.1)
    rospy.set_param('/use_sim_time',a.policy_source=='mavros')
    mavros_launch=str(Path(__file__).with_name('dual_mavros.launch')) if a.policy_source=='mavros' else 'px4.launch'
    launch(['roslaunch',mavros_launch,'fcu_url:=udp://:14541@127.0.0.1:14582'] if a.policy_source=='mavros'
        else ['roslaunch','mavros',mavros_launch,'fcu_url:=udp://:14541@127.0.0.1:14582','gcs_url:='])
    rospy.init_node('hover_dual_interface',anonymous=False,disable_signals=True)
    sink=MavrosSink(interface=a.interface)
    from std_msgs.msg import Bool,Float64MultiArray
    ready=[False];rospy.Subscriber('/hover/policy_ready',Bool,lambda msg:ready.__setitem__(0,bool(msg.data)),queue_size=1)
    feedback_pub=rospy.Publisher('/hover/esc_feedback',Float64MultiArray,queue_size=1)
    from rosgraph_msgs.msg import Clock
    clock_pub=rospy.Publisher('/clock',Clock,queue_size=1)
    if a.policy_source=='mavros':
        fixture_height=json.loads((a.assets/'config.json').read_text())['stage']['controller']['target_height_m']
        launch([sys.executable,'-u','-B','-m','px4_offboard.mavros_policy','--assets',str(a.assets.resolve()),
            '--interface',a.interface,'--reference-height-m',str(fixture_height),'--record',str(root/'policy_runtime.json'),
            '--acceleration-filter-s',str(a.acceleration_filter_s)]+(['--p3-calibration',str(a.p3_calibration.resolve())] if a.p3_calibration else []))
    from .stock_transfer import stock_check_hold as legacy
    original_sensor=legacy.control_message
    def sensor_clock(conn,clock,*args,**kwargs):
        if a.policy_source=='mavros':clock_pub.publish(Clock(clock=rospy.Time.from_sec(clock)))
        return original_sensor(conn,clock,*args,**kwargs)
    legacy.control_message=sensor_clock

    class Endpoint(Offboard):
        def __init__(self,link,cfg,weights,**kwargs):
            super().__init__(link,cfg,weights,residual=False,
                reference_sample_height_m=kwargs.get('reference_sample_height_m',0.))
            deployment=copy.deepcopy(cfg);deployment['deployment']={'acceleration_filter_s':a.acceleration_filter_s}
            self.controller=AccelerationController(deployment,weights);self.wire_echo=[];endpoint.append(self)
            self.calibration_samples=[];self.calibration_complete=False;self.calibration_stamp=None
        def command(self,command,*args):
            if command==400 and (not self.calibration_complete or (a.policy_source=='mavros' and not ready[0])):return
            return super().command(command,*args)
        def request_height_telemetry(self):
            super().request_height_telemetry()
            for msg in (83,85):self.command(511,msg,10_000)
        def ingest(self,msg,clock):
            super().ingest(msg,clock)
            if msg.get_type()=='HIGHRES_IMU' and not self.calibration_complete and not (self.last_heartbeat and self.last_heartbeat.base_mode&128):
                attitude=self.messages.get('ATTITUDE_QUATERNION')
                if attitude is not None and msg.time_usec!=self.calibration_stamp:
                    from .network import rotation
                    q=np.array([attitude.q1,attitude.q2,attitude.q3,attitude.q4])
                    if np.linalg.norm([msg.xgyro,msg.ygyro,msg.zgyro])<.03:
                        force=np.array([msg.xacc,msg.yacc,msg.zacc])
                        self.calibration_samples.append(force+self.controller.observer.g*rotation(q)[2])
                        self.calibration_stamp=msg.time_usec
                    if len(self.calibration_samples)>=2000:
                        self.controller.accel_offset=np.mean(self.calibration_samples,axis=0)
                        self.calibration_complete=True
            if msg.get_type() in ('ATTITUDE_TARGET','POSITION_TARGET_LOCAL_NED'):
                row=msg.to_dict();row['observed_s']=clock;self.wire_echo.append(row)
        def setpoint(self,rates,thrust,clock):
            if a.policy_source=='mavros':return
            acc=self.controller.latest_acceleration
            if acc is None:acc=np.zeros(3)
            if sink.send(acc,rates,thrust,clock,active=self.is_offboard_armed()):self.commands+=1
        def update(self,rpm,applied,clock):
            if a.policy_source=='mavros':
                feedback_pub.publish(Float64MultiArray(data=np.r_[rpm,applied].tolist()));return None
            return super().update(rpm,applied,clock)

    def px4_popen(args,*rest,**kwargs):
        proc=original_popen(args,*rest,**kwargs)
        if isinstance(args,list) and str(args[0]).endswith('/bin/px4'):
            if a.pitch_rate_k is not None:
                proc.stdin.write(f'param set MC_PITCHRATE_K {a.pitch_rate_k}\nparam show MC_PITCHRATE_K\n');proc.stdin.flush()
            proc.stdin.write('mavlink start -u 14581 -o 14582 -t 127.0.0.1 -m onboard -r 4000000\n');proc.stdin.flush()
            if a.policy_source=='mavros':
                for stream,hz in (('HEARTBEAT',10),('LOCAL_POSITION_NED',50),('ATTITUDE_QUATERNION',100),('HIGHRES_IMU',100),('ALTITUDE',20)):
                    proc.stdin.write(f'mavlink stream -u 14581 -s {stream} -r {hz}\n')
                proc.stdin.flush()
        return proc
    sources=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+list(Path(__file__).parent.glob('*.launch'))
    with ZipFile(root/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for f in sources:z.write(f,'px4_offboard/'+f.name)
        z.write(Path(legacy.__file__),'px4_offboard/stock_transfer/stock_check_hold.py')
    legacy.Offboard=Endpoint
    subprocess.Popen=px4_popen
    sys.argv=[str(legacy.__file__),str(root/'flight'),'--assets',str(a.assets.resolve()),'--seconds',str(a.seconds),
        '--wind',str(a.wind),'--wind-ramp-s',str(a.wind_ramp_s),'--sensor-seed',str(a.seed),'--no-residual']
    try:legacy.main()
    finally:
        subprocess.Popen=original_popen
        summary_path=root/'flight/summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            summary.update(parameters_overridden=a.pitch_rate_k is not None,
                px4_param_overrides={'MC_PITCHRATE_K':a.pitch_rate_k} if a.pitch_rate_k is not None else {})
            summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        (root/'mavros_commands.json').write_text(json.dumps(sink.sent,indent=2)+'\n')
        (root/'wire_echo.json').write_text(json.dumps(endpoint[0].wire_echo if endpoint else [],indent=2)+'\n')
        proxy.close()
        (root/'wire_commands.json').write_text(json.dumps(dict(commands=proxy.commands,error=proxy.error),indent=2)+'\n')
        if endpoint:
            (root/'preflight_calibration.json').write_text(json.dumps(dict(samples=len(endpoint[0].calibration_samples),
                complete=endpoint[0].calibration_complete,accel_offset=endpoint[0].controller.accel_offset.tolist()),indent=2)+'\n')
        (root/'interface.json').write_text(json.dumps(dict(interface=a.interface,transport='actual ROS Noetic MAVROS',
            pitch_rate_k=a.pitch_rate_k,px4_param_overrides={'MC_PITCHRATE_K':a.pitch_rate_k} if a.pitch_rate_k is not None else {},
            policy_source=a.policy_source,
            asset_sha256=asset_hashes,wind_ramp_s=a.wind_ramp_s,
            ROS_simulated_clock=a.policy_source=='mavros',
            P3_enabled=p3_data is not None,p3_config_sha256=p3_hash,p3_checkpoint_sha256=p3_data['checkpoint_sha256'] if p3_data else None,
            P3_scope=p3_data['scope'] if p3_data else None,physical_torque_feedforward_sent=False,
            full_P3_preserved_in='P3/results/installation_20261115',assets=str(a.assets),seed=a.seed,
            controller_sha256=json.loads((a.assets/'config.json').read_text())['controller_sha256'],
            acceleration_filter_s=a.acceleration_filter_s,
            rate_adapter='same acceleration Actor with attitude PI rate reference and local vertical PID; not native torque equivalence',
            GPS_for_PX4_Position=True,GPS_policy_input=False,requested_duration_s=a.seconds,wind_mps=a.wind),indent=2)+'\n')
        rospy.signal_shutdown('run completed')
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for f in handles:f.close()


if __name__=='__main__':main()
