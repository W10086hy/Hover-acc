"""NX entry: onboard MAVROS telemetry -> frozen P2 -> MAVROS acceleration setpoint.

RPM and applied throttle are mandatory. No horizontal position/velocity enters
the Actor. Full native P3 torque/allocator compensation is a separate route.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from .control import Sensors
from .network import rotation
from .offboard import relative_attitude
from .acceleration_runtime import AccelerationController
from .mavros_acceleration import MavrosSink
from .readiness import PreflightCalibration,PolicyClock


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
    p.add_argument('--p3-calibration',type=Path,help='Validated Position P3 JSON; standardizes existing35 Actor inputs')
    p.add_argument('--reference-height-m',type=float,default=0.)
    p.add_argument('--acceleration-filter-s',type=float,default=.065)
    p.add_argument('--feedback-topic',default='/hover/esc_feedback',help='Optional 8-vector: P2-order four rad/s and four actual throttles')
    a=p.parse_args(rospy.myargv()[1:]);rospy.init_node('hover_acceleration_policy',anonymous=False)
    cfg=json.loads((a.assets/'config.json').read_text());cfg['deployment']={'acceleration_filter_s':a.acceleration_filter_s}
    from .p3_observer import ObserverCalibration
    calibration_model=ObserverCalibration.load(a.p3_calibration) if a.p3_calibration else None
    controller=AccelerationController(cfg,np.load(a.assets/'weights.npz'),calibration=calibration_model)
    sink=MavrosSink(namespace=a.namespace,interface=a.interface);imu=[None,None];baro=[None,None];rpm=[None,None];throttle=[None,None]
    ready_pub=rospy.Publisher('/hover/policy_ready',Bool,queue_size=1,latch=True);ready_pub.publish(False)
    heading=math.radians(a.launch_heading_deg) if a.launch_heading_deg is not None else 0.
    def on_imu(msg):imu[:]=[msg,time.monotonic()]
    def on_baro(msg):baro[:]=[float(msg.monotonic),time.monotonic()]
    def on_esc(msg):
        if len(msg.esc_status)>=4:rpm[:]=[np.array([msg.esc_status[i].rpm for i in (0,3,2,1)])*(2*math.pi/60),time.monotonic()]
    def on_rc(msg):
        if len(msg.channels)>=4:throttle[:]=[(np.array(msg.channels)[[0,3,2,1]]-1000)/1000.,time.monotonic()]
    def feedback(msg):
        values=np.asarray(msg.data)
        if values.shape==(8,):rpm[:]=[values[:4],time.monotonic()];throttle[:]=[values[4:],time.monotonic()]
    for topic,kind,callback in (('/imu/data',Imu,on_imu),('/altitude',Altitude,on_baro),('/esc_status',ESCStatus,on_esc),('/rc/out',RCOut,on_rc)):
        rospy.Subscriber(a.namespace+topic,kind,callback,queue_size=1)
    rospy.Subscriber(a.feedback_topic,Float64MultiArray,feedback,queue_size=1)
    calibration=PreflightCalibration(controller.observer.g);zero=None;filtered=None;was_active=False;requested=False
    sample_clock=PolicyClock()
    def reset_history(active):
        nonlocal filtered,was_active
        controller.reset();filtered=None;was_active=False;sample_clock.invalidate(active,'missing/stale/invalid input')
    steps=0;p3_diagnostics=[]
    def record():
        if a.record is not None:
            a.record.write_text(json.dumps(dict(controller_sha256=cfg['controller_sha256'],interface=a.interface,
                input_source='MAVROS IMU/altitude/state; ESC rad/s and applied throttle',policy_steps=steps,
                calibration_samples=len(calibration.forces),accel_offset=controller.accel_offset.tolist(),
                takeoff_baro_zero_m=zero,launch_heading_rad=heading,
                IMU_stream_blocked=sample_clock.blocked,IMU_stream_last_rejection=sample_clock.last_rejection,IMU_max_gap_s=sample_clock.max_gap_s,
                controller_target_baro_height_m=controller.observer.target_height_m,
                target_local_z_enu_m=sink.latch.height,commands=list(sink.sent)),indent=2)+'\n')
            if calibration_model is not None:
                data=json.loads(a.record.read_text());data.update(P3_enabled=True,P3_scope=calibration_model.parameters['scope'],
                    p3_checkpoint_sha256=calibration_model.parameters['checkpoint_sha256'],physical_torque_feedforward_sent=False,
                    P3_diagnostics_10Hz=p3_diagnostics)
                a.record.write_text(json.dumps(data,indent=2)+'\n')
    rospy.on_shutdown(record)
    rate=rospy.Rate(100)
    while not rospy.is_shutdown():
        now=time.monotonic();state=sink.state;armed=bool(state and state.armed)
        if state and state.connected and not requested:
            try:
                request=rospy.ServiceProxy(a.namespace+'/cmd/command',CommandLong)
                for number,interval in ((31,10000),(105,10000),(141,50000)):
                    request(False,511,0,float(number),float(interval),0.,0.,0.,0.,0.)
                requested=True
            except rospy.ServiceException:pass
        if not sink.active:
            v=controller.observer;hover=float(v.throttle(np.array([math.sqrt(v.mass*v.g/(4*v.kf))]))[0])
            sink.latch.heading=heading;sink.send([0,0],[0,0,0],hover,now,active=False)
            if was_active:reset_history(False)
            was_active=False
        if not sink.state_valid:rate.sleep();continue
        # First entry may still await ESC feedback. A missing sample cannot
        # corrupt a history that has not started; a blocked entry stays blocked.
        clock_active=sink.active and (was_active or sample_clock.blocked)
        if imu[0] is None or baro[0] is None or now-imu[1]>.25 or now-baro[1]>.25:
            if clock_active:reset_history(True)
            rate.sleep();continue
        if sample_clock.received is not None and not sample_clock.fresh(now,.25) and clock_active:
            reset_history(True);rate.sleep();continue
        msg=imu[0];stamp=msg.header.stamp.to_sec()
        if not sample_clock.accept(stamp,now,active=clock_active):
            if sample_clock.blocked:reset_history(True)
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
                controller.accel_offset=fit[0];zero=fit[1]-a.reference_height_m
                rospy.loginfo('P2 preflight calibration ready; Actor %s',cfg['controller_sha256'])
                ready_pub.publish(True)
            rate.sleep();continue
        if not sink.active or zero is None:rate.sleep();continue
        if any(v[0] is None or now-v[1]>.25 for v in (rpm,throttle)):
            if clock_active:reset_history(True)
            rate.sleep();continue
        if not np.isfinite(rpm[0]).all() or (rpm[0]<0).any() or not np.isfinite(throttle[0]).all() or (throttle[0]<0).any() or (throttle[0]>1).any():
            if clock_active:reset_history(True)
            rate.sleep();continue
        packet=imu_to_packet([q.x,q.y,q.z,q.w],[w.x,w.y,w.z],[f.x,f.y,f.z],rpm[0],baro[0]-zero,0.)
        if calibration_model is not None:packet=calibration_model.correct_packet(packet)
        packet=Sensors(relative_attitude(packet.attitude,heading),packet.gyro,packet.specific_force,packet.motor_speed,packet.baro_height)
        if not was_active:
            controller.reset();controller.observer.target_height_m=packet.baro_height;filtered=None;was_active=True
        if calibration_model is not None:controller.observer.sample_dt_s=sample_clock.elapsed_s or controller.dt
        filtered=packet.specific_force.copy() if filtered is None else filtered+.5*(packet.specific_force-filtered)
        packet=Sensors(packet.attitude,packet.gyro,filtered,packet.motor_speed,packet.baro_height)
        if a.interface=='acceleration':
            acceleration=controller.acceleration(packet,throttle[0]);rates=np.zeros(3);thrust=0.
        else:
            rates,thrust,_=controller(packet,throttle[0]);acceleration=controller.latest_acceleration
            if calibration_model is not None:rates=calibration_model.rates_to_reported(rates)
        sink.latch.heading=heading;sink.send(acceleration,rates,thrust,now)
        if calibration_model is not None and a.record is not None and steps%10==0:
            v=controller.observer;p3_diagnostics.append(dict(stamp_s=stamp,transport_torque_nm=v.transport_torque.tolist(),
                innovation_before_radps=v.innovation_before.tolist(),innovation_after_radps=v.innovation_after.tolist()))
        steps+=1;rate.sleep()


if __name__=='__main__':main()
