"""MAVROS horizontal-acceleration + latched-altitude sender. No ROS import at module load."""
import argparse
import math
import time
import sys
from pathlib import Path
import numpy as np
from .acceleration_setpoint import HORIZONTAL_ACCELERATION_ALTITUDE_MASK
from .readiness import SampleClock,StampedAcceleration


class AccelerationLatch:
    def __init__(self,*,launch_heading_rad):
        if not math.isfinite(launch_heading_rad):raise ValueError('Finite launch heading required')
        self.heading=launch_heading_rad;self.height=None;self.reset_counter=None;self.blocked=False

    def reset(self):self.height=None;self.reset_counter=None;self.blocked=False

    def command(self,clock,*,active,local_z_enu,received_s,acceleration,reset_counter=0):
        if not active:self.reset()
        if not all(math.isfinite(v) for v in (clock,local_z_enu,received_s)) or not 0<=clock-received_s<=.25:return None
        a=np.asarray(acceleration,float)
        if a.shape!=(2,) or not np.isfinite(a).all():return None
        if active:
            if self.reset_counter is not None and reset_counter!=self.reset_counter:self.blocked=True
            if self.blocked:return None
            self.reset_counter=reset_counter
            if self.height is None:self.height=float(local_z_enu)
            f,right=a;c,s=math.cos(self.heading),math.sin(self.heading)
            north,east=c*f-s*right,s*f+c*right
        else:north=east=0.
        return dict(type_mask=HORIZONTAL_ACCELERATION_ALTITUDE_MASK,height_enu_m=self.height if active else float(local_z_enu),
            acceleration_enu=[float(east),float(north),0.],yaw_enu_rad=math.pi/2-self.heading)


class MavrosSink:
    def __init__(self,*,interface='acceleration',namespace='/mavros',launch_heading_rad=0.):
        from .ros_compat import load_rospy
        rospy=load_rospy()
        self.clock=(lambda:rospy.Time.now().to_sec()) if rospy.get_param('/use_sim_time',False) else time.monotonic
        from geometry_msgs.msg import PoseStamped
        from mavros_msgs.msg import PositionTarget,AttitudeTarget,State
        from std_msgs.msg import UInt32
        self.rospy=rospy;self.interface=interface;self.PositionTarget=PositionTarget;self.AttitudeTarget=AttitudeTarget
        self.latch=AccelerationLatch(launch_heading_rad=launch_heading_rad)
        self.z=None;self.z_received=None;self.sent=[];self.state=None
        self.pose_clock=SampleClock();self.state_clock=SampleClock()
        self.frame_counter=0;self.pose_discontinuities=0
        self.raw=rospy.Publisher(namespace+'/setpoint_raw/local',PositionTarget,queue_size=1)
        self.attitude=rospy.Publisher(namespace+'/setpoint_raw/attitude',AttitudeTarget,queue_size=1)
        rospy.Subscriber(namespace+'/local_position/pose',PoseStamped,self._pose,queue_size=1)
        rospy.Subscriber(namespace+'/state',State,self._state,queue_size=1)
        rospy.Subscriber('/hover/local_frame_reset_counter',UInt32,self._reset,queue_size=1)

    def _now(self):return self.clock() if hasattr(self,'clock') else time.monotonic()

    def _pose(self,msg):
        z=float(msg.pose.position.z);now=self._now()
        if not math.isfinite(z) or not self.pose_clock.update(msg.header.stamp.to_sec(),now):return
        if self.z is not None and 0<now-self.z_received<.1 and abs(z-self.z)>.5:self.pose_discontinuities+=1
        self.z=z;self.z_received=now
    def _reset(self,msg):self.frame_counter=int(msg.data)
    def _state(self,msg):
        if not msg.connected:
            self.state=msg;self.state_clock=SampleClock();self.latch.reset();return
        if self.state_clock.update(msg.header.stamp.to_sec(),self._now()):self.state=msg

    @property
    def state_valid(self):return bool(self.state and self.state.connected and self.state_clock.fresh(self._now(),1.5))

    @property
    def active(self):return bool(self.state_valid and self.state.armed and self.state.mode=='OFFBOARD')

    def send(self,acceleration,rates,thrust,clock,*,active=None):
        if not self.state_valid:
            self.latch.reset();return False
        active=self.active if active is None else active
        if self.interface=='acceleration':
            if self.z is None:return False
            fields=self.latch.command(self._now(),active=active,local_z_enu=self.z,
                received_s=self.z_received,acceleration=np.asarray(acceleration)[:2],reset_counter=(self.frame_counter,self.pose_discontinuities))
            if fields is None:return False
            msg=self.PositionTarget();msg.header.stamp=self.rospy.Time.now();msg.coordinate_frame=1;msg.type_mask=fields['type_mask']
            msg.position.z=fields['height_enu_m'];msg.yaw=fields['yaw_enu_rad']
            msg.acceleration_or_force.x,msg.acceleration_or_force.y,msg.acceleration_or_force.z=fields['acceleration_enu']
            self.raw.publish(msg)
            record=dict(clock=clock,message='SET_POSITION_TARGET_LOCAL_NED',**fields)
        else:
            rates=np.asarray(rates,float)
            if rates.shape!=(3,) or not np.isfinite(rates).all() or not math.isfinite(thrust) or not 0<=thrust<=1:return False
            msg=self.AttitudeTarget();msg.header.stamp=self.rospy.Time.now();msg.type_mask=128;msg.orientation.w=1.
            msg.body_rate.x,msg.body_rate.y,msg.body_rate.z=rates*[1,-1,-1];msg.thrust=float(thrust)
            self.attitude.publish(msg);record=dict(clock=clock,message='SET_ATTITUDE_TARGET',body_rate_frd=rates.tolist(),thrust=thrust)
        self.sent.append(record);return True


def main():
    from .ros_compat import load_rospy
    rospy=load_rospy()
    from geometry_msgs.msg import Vector3Stamped
    from mavros_msgs.msg import State
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',default='/mavros');parser.add_argument('--command-topic',default='/hover/acceleration_command')
    parser.add_argument('--launch-heading-deg',type=float,required=True)
    args=parser.parse_args(rospy.myargv()[1:]);rospy.init_node('hover_acceleration_sender',anonymous=False)
    sink=MavrosSink(namespace=args.namespace,launch_heading_rad=math.radians(args.launch_heading_deg));latest=StampedAcceleration()
    def command(msg):
        if msg.header.frame_id not in ('','launch_heading_ned'):return
        latest.update(msg.header.stamp.to_sec(),time.monotonic(),[msg.vector.x,msg.vector.y])
    rospy.Subscriber(args.command_topic,Vector3Stamped,command,queue_size=1)
    rate=rospy.Rate(100)
    while not rospy.is_shutdown():
        now=time.monotonic();value=latest.get(now)
        if sink.active and value is not None:sink.send(value,[0,0,0],0.,now)
        elif not sink.active:sink.send([0,0],[0,0,0],0.,now)
        rate.sleep()


if __name__=='__main__':main()
