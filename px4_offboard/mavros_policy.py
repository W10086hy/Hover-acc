"""NX entry: onboard MAVROS telemetry -> frozen P2 -> MAVROS acceleration setpoint.

RPM and applied throttle are mandatory. No horizontal position/velocity enters
the Actor. Full native P3 torque/allocator compensation is a separate route.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from .control import Sensors
from .network import rotation
from .offboard import relative_attitude
from .acceleration_runtime import AccelerationController
from .mavros_acceleration import MavrosSink
from .readiness import PreflightCalibration,PolicyClock,ModeEntryGuard
from .policy_timing import FixedPolicyRunner, FeedbackHistory, atomic_json, shutdown_error
from .unified_p3 import dynamics_features, nominal_alpha


def multiply(a,b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


def imu_to_packet(quaternion_xyzw,gyro_flu,force_flu,rpm,baro_height,heading):
    q=np.asarray(quaternion_xyzw,float)[[3,0,1,2]]
    if not np.isfinite(q).all() or np.linalg.norm(q)<1e-6:raise ValueError('Valid ENU/FLU IMU orientation required')
    q/=np.linalg.norm(q)
    # ENU->NED on the left, FRD->FLU on the right.
    q=multiply(multiply(np.array([0.,np.sqrt(.5),np.sqrt(.5),0.]),q),np.array([0.,1.,0.,0.]))
    if q[0]<0:q=-q
    return Sensors(relative_attitude(q,heading),np.asarray(gyro_flu)*[1,-1,-1],
        np.asarray(force_flu)*[1,-1,-1],np.asarray(rpm),float(baro_height))


def main():
    from .ros_compat import load_rospy
    rospy=load_rospy()
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64MultiArray,Bool
    from mavros_msgs.msg import Altitude,ESCStatus,RCOut
    from mavros_msgs.srv import CommandLong
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('px4_offboard/acceleration_dual/assets'))
    p.add_argument('--namespace',default='/mavros');p.add_argument('--launch-heading-deg',type=float)
    p.add_argument('--interface',choices=('acceleration','rate_thrust'),default='acceleration')
    p.add_argument('--record',type=Path)
    p.add_argument('--timing',choices=('auto','fixed','legacy'),default='auto',help='auto: fixed for noP3/v2, legacy for historical P3')
    p.add_argument('--p3-calibration',type=Path,help='Validated Position P3 JSON; standardizes existing35 Actor inputs')
    p.add_argument('--reference-height-m',type=float,default=0.)
    p.add_argument('--acceleration-filter-s',type=float,default=.065)
    p.add_argument('--sim-observation-bias-mps2',type=float,nargs=3,default=[0.,0.,0.],
        help='SITL-only post-calibration residual force fault in reported body axes')
    p.add_argument('--innovation-filter-s',type=float,default=0.,
        help='Experimental low-pass on angular innovation after P3 and before clipping; fixed acceleration only')
    p.add_argument('--feedback-topic',default='/hover/esc_feedback',help='Optional 8-vector: P2-order four rad/s and four actual throttles')
    a=p.parse_args(rospy.myargv()[1:]);rospy.init_node('hover_acceleration_policy',anonymous=False)
    cfg=json.loads((a.assets/'config.json').read_text())
    if np.any(a.sim_observation_bias_mps2) and (not rospy.get_param('/use_sim_time',False) or a.interface!='acceleration'):
        p.error('Simulation observation bias requires simulated ROS time and acceleration mode')
    cfg['deployment']={'acceleration_filter_s':a.acceleration_filter_s,'innovation_filter_s':a.innovation_filter_s,
        'sim_observation_bias_mps2':a.sim_observation_bias_mps2}
    from .p3_observer import ObserverCalibration
    calibration_model=ObserverCalibration.load(a.p3_calibration) if a.p3_calibration else None
    controller=AccelerationController(cfg,np.load(a.assets/'weights.npz'),calibration=calibration_model)
    schema=calibration_model.parameters.get('schema') if calibration_model else None
    timing=('fixed' if schema in (None,'P3_aligned_residual_v2') else 'legacy') if a.timing=='auto' else a.timing
    if np.any(a.sim_observation_bias_mps2) and timing!='fixed':
        p.error('Simulation observation bias requires fixed timing')
    if schema=='P3_aligned_residual_v2' and timing!='fixed':raise ValueError('Aligned P3 requires fixed timing')
    if a.innovation_filter_s and (timing!='fixed' or a.interface!='acceleration'):
        p.error('Innovation filtering requires fixed timing and the acceleration interface')
    runner=FixedPolicyRunner(controller,a.interface) if timing=='fixed' else None
    rpm_history=FeedbackHistory();throttle_history=FeedbackHistory()
    sink=MavrosSink(namespace=a.namespace,interface=a.interface);imu=[None,None,None];baro=[None,None];rpm=[None,None];throttle=[None,None]
    ready_pub=rospy.Publisher('/hover/policy_ready',Bool,queue_size=1,latch=True);ready_pub.publish(False)
    heading=math.radians(a.launch_heading_deg) if a.launch_heading_deg is not None else 0.
    input_clock=sink.clock
    def on_imu(msg):imu[:]=[msg,input_clock(),time.monotonic()]
    def on_baro(msg):baro[:]=[float(msg.monotonic),input_clock()]
    def on_esc(msg):
        if len(msg.esc_status)>=4:
            rpm[:]=[np.array([msg.esc_status[i].rpm for i in (0,3,2,1)])*(2*math.pi/60),input_clock()]
            rpm_history.add(msg.header.stamp.to_sec(),rpm[0])
    def on_rc(msg):
        if len(msg.channels)>=4:
            throttle[:]=[(np.array(msg.channels)[[0,3,2,1]]-1000)/1000.,input_clock()]
            throttle_history.add(msg.header.stamp.to_sec(),throttle[0])
    def feedback(msg):
        values=np.asarray(msg.data)
        if values.shape in ((8,),(9,)):
            rpm[:]=[values[:4],input_clock()];throttle[:]=[values[4:8],input_clock()]
            if values.shape==(9,):
                rpm_history.add(float(values[8]),values[:4],source='custom');throttle_history.add(float(values[8]),values[4:8],source='custom')
    for topic,kind,callback in (('/imu/data',Imu,on_imu),('/altitude',Altitude,on_baro),('/esc_status',ESCStatus,on_esc),('/rc/out',RCOut,on_rc)):
        rospy.Subscriber(a.namespace+topic,kind,callback,queue_size=1)
    rospy.Subscriber(a.feedback_topic,Float64MultiArray,feedback,queue_size=1)
    calibration=PreflightCalibration(controller.observer.g);zero=None;filtered=None;was_active=False;requested=False
    sample_clock=PolicyClock();mode_guard=ModeEntryGuard()
    def reset_history(active,reason='mode exit'):
        nonlocal filtered,was_active
        if runner is not None:runner.reset()
        else:controller.reset()
        filtered=None;was_active=False;sample_clock.invalidate(active,reason)
    steps=0;p3_diagnostics=[];run_error=None;samples_file=None;feedback_token=None;entry_id=0
    if a.record is not None:
        if a.record.exists():raise FileExistsError('Preserve earlier runtime record')
        a.record.parent.mkdir(parents=True,exist_ok=True)
        samples_file=a.record.with_name(a.record.stem+'_samples.jsonl').open('x',encoding='utf-8',buffering=1)
    def record(final=True):
        if a.record is not None:
            data=dict(controller_sha256=cfg['controller_sha256'],interface=a.interface,
                timing=timing,policy_dt_s=controller.dt,error=run_error,innovation_filter_s=a.innovation_filter_s,
                sim_observation_bias_mps2=a.sim_observation_bias_mps2,
                freshness_clock='ROS_sim_time' if rospy.get_param('/use_sim_time',False) else 'host_monotonic',
                input_source='MAVROS IMU/altitude/state; measured ESC RPM and executed throttle',policy_steps=steps,
                calibration_samples=len(calibration.forces),accel_offset=controller.accel_offset.tolist(),
                takeoff_baro_zero_m=zero,launch_heading_rad=heading,
                mode_entry_blocked=mode_guard.blocked,IMU_stream_blocked=sample_clock.blocked,IMU_stream_last_rejection=sample_clock.last_rejection,
                IMU_max_gap_s=sample_clock.max_gap_s,controller_target_baro_height_m=controller.observer.target_height_m,
                preflight_noise_rejection=calibration.last_rejection,
                preflight_force_std_mps2=calibration.force_std.tolist() if calibration.force_std is not None else None,
                target_local_z_enu_m=sink.latch.height,commands=list(sink.sent) if final else [],
                command_count=len(sink.sent),summary_complete=bool(final))
            if calibration_model is not None:
                data.update(P3_enabled=True,P3_scope=calibration_model.parameters['scope'],
                    p3_checkpoint_sha256=calibration_model.parameters['checkpoint_sha256'],physical_torque_feedforward_sent=False,
                    P3_diagnostics_10Hz=p3_diagnostics if final else p3_diagnostics[-10:])
            atomic_json(a.record,data)
            if samples_file is not None and not samples_file.closed:
                samples_file.flush()
                os.fsync(samples_file.fileno())
    rospy.on_shutdown(record)
    rate=rospy.Rate(100)
    try:
        while not rospy.is_shutdown():
            now=input_clock();state=sink.state;armed=bool(state and state.armed)
            if state and state.connected and not requested:
                try:
                    request=rospy.ServiceProxy(a.namespace+'/cmd/command',CommandLong)
                    for number,interval in ((31,10000),(105,10000),(141,50000)):
                        request(False,511,0,float(number),float(interval),0.,0.,0.,0.,0.)
                    requested=True
                except rospy.ServiceException:pass
            valid_state=sink.state_valid
            mode=state.mode if state else None
            mode_guard.update(valid=valid_state,armed=armed,mode=mode)
            confirmed_exit=mode_guard.confirmed_exit
            if mode_guard.blocked:
                if was_active:reset_history(True,'state telemetry stale during Offboard entry')
                rate.sleep();continue
            if confirmed_exit:
                v=controller.observer;hover=float(v.throttle(np.array([math.sqrt(v.mass*v.g/(4*v.kf))]))[0])
                sink.latch.heading=heading;sink.send([0,0],[0,0,0],hover,now,active=False)
                if was_active or sample_clock.blocked:reset_history(False,'confirmed mode exit')
                was_active=False
            if not valid_state:rate.sleep();continue
            # First entry may still await ESC feedback. A missing sample cannot
            # corrupt a history that has not started; a blocked entry stays blocked.
            clock_active=sink.active and (was_active or sample_clock.blocked)
            if imu[0] is None or baro[0] is None or now-imu[1]>.25 or now-baro[1]>.25:
                if clock_active:reset_history(True,'IMU or barometer missing/stale')
                rate.sleep();continue
            if sample_clock.received is not None and not sample_clock.fresh(now,.25) and clock_active:
                reset_history(True,'IMU receive clock stale');rate.sleep();continue
            msg,imu_received,imu_host_received=imu;stamp=msg.header.stamp.to_sec()
            if zero is None and sample_clock.stamp is not None and (stamp<sample_clock.stamp or stamp-sample_clock.stamp>sample_clock.max_gap_s):
                calibration=PreflightCalibration(controller.observer.g)
            if not sample_clock.accept(stamp,imu_received,active=clock_active):
                if sample_clock.blocked:reset_history(True,'IMU source clock rejected')
                rate.sleep();continue
            q=msg.orientation;w=msg.angular_velocity;f=msg.linear_acceleration
            try:packet=imu_to_packet([q.x,q.y,q.z,q.w],[w.x,w.y,w.z],[f.x,f.y,f.z],np.zeros(4),baro[0],0.)
            except ValueError:
                if clock_active:reset_history(True)
                rate.sleep();continue
            if calibration_model is not None:packet=calibration_model.correct_packet(packet)
            if not np.isfinite(packet.gyro).all() or not np.isfinite(packet.specific_force).all() or not math.isfinite(packet.baro_height):
                if clock_active:reset_history(True)
                rate.sleep();continue
            if not armed:
                if a.launch_heading_deg is None:
                    r=rotation(packet.attitude);heading=math.atan2(r[1,0],r[0,0])
                calibration.add(packet);fit=calibration.result()
                if fit is not None and zero is None:
                    controller.accel_offset=(calibration_model.select_preflight_offset(fit[0])
                        if hasattr(calibration_model,'select_preflight_offset') else fit[0])
                    zero=fit[1]-a.reference_height_m
                    rospy.loginfo('P2 preflight calibration ready; Actor %s',cfg['controller_sha256'])
                    ready_pub.publish(True)
                rate.sleep();continue
            if not sink.active or zero is None:rate.sleep();continue
            if any(v[0] is None or now-v[1]>.25 for v in (rpm,throttle)):
                if clock_active:reset_history(True,'motor feedback reception stale')
                rate.sleep();continue
            rpm_value,applied=rpm[0],throttle[0];rpm_age=None;throttle_age=None
            if runner is not None:
                token=(rpm_history.rollbacks,throttle_history.rollbacks)
                if clock_active and feedback_token is not None and token!=feedback_token:
                    reset_history(True,'motor feedback source clock rollback');rate.sleep();continue
                past_rpm,past_throttle=rpm_history.at(stamp),throttle_history.at(stamp)
                if past_rpm is None or past_throttle is None:
                    if clock_active:reset_history(True,dict(reason='motor/IMU alignment missing',imu_stamp_s=stamp,
                        rpm_last_source_s=rpm_history.rows[-1][0] if rpm_history.rows else None,
                        throttle_last_source_s=throttle_history.rows[-1][0] if throttle_history.rows else None))
                    rate.sleep();continue
                rpm_value,rpm_age=past_rpm;applied,throttle_age=past_throttle
            if not np.isfinite(rpm_value).all() or (rpm_value<0).any() or not np.isfinite(applied).all() or (applied<0).any() or (applied>1).any():
                if clock_active:reset_history(True)
                rate.sleep();continue
            packet=imu_to_packet([q.x,q.y,q.z,q.w],[w.x,w.y,w.z],[f.x,f.y,f.z],rpm_value,baro[0]-zero,0.)
            raw_packet=packet
            if calibration_model is not None:packet=calibration_model.correct_packet(packet)
            packet=Sensors(relative_attitude(packet.attitude,heading),packet.gyro,packet.specific_force,packet.motor_speed,packet.baro_height)
            if not was_active:
                if runner is not None:runner.reset()
                else:controller.reset()
                controller.observer.target_height_m=packet.baro_height;filtered=None;was_active=True
                entry_id+=1
                feedback_token=(rpm_history.rollbacks,throttle_history.rollbacks)
            tick_diagnostics=[]
            if runner is not None and samples_file is not None:
                event=dict(event='source_packet',entry_id=entry_id,source_stamp_s=stamp,
                    received_monotonic_s=imu_host_received,received_clock_s=imu_received,
                    attitude_wxyz=raw_packet.attitude.tolist(),gyro_radps=raw_packet.gyro.tolist(),
                    force_mps2=raw_packet.specific_force.tolist(),rpm_radps=raw_packet.motor_speed.tolist(),
                    applied_throttle=applied.tolist(),baro_height_m=float(raw_packet.baro_height),
                    target_baro_height_m=controller.observer.target_height_m)
                samples_file.write(json.dumps(event,allow_nan=False)+'\n')
            if runner is not None:
                try:tick_diagnostics=runner.advance(packet,applied,stamp)
                except ValueError as exc:
                    reset_history(True,'policy grid: '+str(exc));rate.sleep();continue
                if not tick_diagnostics:rate.sleep();continue
                latest=tick_diagnostics[-1]
                acceleration=np.asarray(latest['wire_acceleration']);rates=np.asarray(latest['rates']);thrust=latest['thrust']
            else:
                if calibration_model is not None:controller.observer.sample_dt_s=sample_clock.elapsed_s or controller.dt
                filtered=packet.specific_force.copy() if filtered is None else filtered+.5*(packet.specific_force-filtered)
                packet=Sensors(packet.attitude,packet.gyro,filtered,packet.motor_speed,packet.baro_height)
                v=controller.observer
                _,sq,mean=dynamics_features(v.rate,packet.specific_force,v.rpm,packet.motor_speed,applied,controller.dt,v.pc)
                raw=packet.gyro-(v.rate+controller.dt*nominal_alpha(v.rate,sq,mean,v.pc)) if v.valid else np.zeros(3)
                if a.interface=='acceleration':
                    acceleration=controller.acceleration(packet,applied);rates=np.zeros(3);thrust=0.
                else:
                    rates,thrust,_=controller(packet,applied);acceleration=controller.latest_acceleration
                    if calibration_model is not None:rates=calibration_model.rates_to_reported(rates)
            if runner is None:
                v=controller.observer
                tick_diagnostics=[dict(grid_stamp_s=stamp,source_stamp_s=stamp,source_interval_s=sample_clock.elapsed_s,
                    timing='legacy',innovation_before_radps=np.asarray(getattr(v,'innovation_before',raw)).tolist(),
                    innovation_after_radps=np.asarray(getattr(v,'innovation_after',raw)).tolist(),
                    observation=controller.last_observation.tolist(),actor_raw=controller.last_raw_action.tolist(),
                    acceleration=acceleration.tolist(),rates=rates.tolist(),thrust=float(thrust))]
            sink.latch.heading=heading;sent=sink.send(acceleration,rates,thrust,now)
            if samples_file is not None:
                for i,d in enumerate(tick_diagnostics):
                    d.update(event="policy_tick",entry_id=entry_id,received_monotonic_s=imu_host_received,received_clock_s=imu_received,rpm_source_age_s=rpm_age,throttle_source_age_s=throttle_age,
                        source_attitude_wxyz=raw_packet.attitude.tolist(),source_gyro_radps=raw_packet.gyro.tolist(),
                        source_force_mps2=raw_packet.specific_force.tolist(),source_rpm_radps=raw_packet.motor_speed.tolist(),
                        source_applied_throttle=applied.tolist(),source_baro_height_m=float(raw_packet.baro_height),
                        wire_sent=bool(sent and i==len(tick_diagnostics)-1))
                    samples_file.write(json.dumps(d,allow_nan=False)+'\n')
            if calibration_model is not None and a.record is not None and steps%10==0:
                v=controller.observer
                diagnostic=dict(stamp_s=stamp,innovation_before_radps=v.innovation_before.tolist(),innovation_after_radps=v.innovation_after.tolist())
                if hasattr(v,'unified_compensation'):
                    diagnostic.update(unified_compensation=v.unified_compensation.tolist(),equivalent_self_torque_nm=v.equivalent_self_torque.tolist())
                else:diagnostic['transport_torque_nm']=v.transport_torque.tolist()
                p3_diagnostics.append(diagnostic)
            steps+=len(tick_diagnostics) if runner is not None else 1
            if steps%100==0:record(final=False)
            rate.sleep()
    except BaseException as exc:
        run_error=shutdown_error(exc,rospy.is_shutdown(),rospy.ROSInterruptException)
        if run_error is not None:raise
    finally:
        record()
        if samples_file is not None:samples_file.close()



if __name__=='__main__':main()
