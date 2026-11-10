"""MAVLink body-rate/thrust endpoint, also usable with an external sensor packet.

Only ATTITUDE_QUATERNION, HIGHRES_IMU, takeoff-relative barometric height and
measured ESC speed enter the policy. All output uses
SET_ATTITUDE_TARGET with ATTITUDE_IGNORE=128, body rates rad/s, thrust[0,1].
"""
import collections
import numpy as np
from .control import Controller,Sensors

def relative_attitude(q,heading):
    c,s=np.cos(-heading/2),np.sin(-heading/2);w,x,y,z=q
    result=np.array([c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w],np.float32)
    return result if result[0]>=0 else -result

class Offboard:
    def __init__(self,link,cfg,weights,residual=True,*,reference_sample_height_m=0.):
        self.link=link;self.controller=Controller(cfg,weights,residual=residual)
        self.messages={};self.received={};self.events=[];self.last_heartbeat=None;self.acks=[]
        self.heading=None;self.filtered_force=None;self.commands=0;self.latest=None
        self.rpm_history=collections.deque(maxlen=160);self.previous_gyro_time=None
        # Simulation fixtures may be held above the takeoff plane. Real ground samples are at 0 m.
        if not np.isfinite(reference_sample_height_m):
            raise ValueError('Reference sample height must be finite')
        self.reference_sample_height_m = float(reference_sample_height_m)
        self.takeoff_baro_zero_m = None
        self.takeoff_baro_locked_s = None
        self._preflight_altitudes = []
        self._altitude_timestamp = 0
    def record_motor_sample(self,rpm,clock):
        values=np.asarray(rpm,float)
        if values.shape!=(4,) or not np.isfinite(values).all():raise ValueError('Four measured motor speeds required')
        self.rpm_history.append((float(clock),values.copy()))
    def motor_moments(self,end):
        if self.previous_gyro_time is None:return None
        start=self.previous_gyro_time
        if end<=start:return None
        samples=list(self.rpm_history);times=np.array([v[0] for v in samples]);rpm=np.array([v[1] for v in samples])
        if len(times)<2 or times[0]>start or times[-1]<end:raise RuntimeError('RPM history does not cover gyro interval')
        grid=np.r_[start,times[(times>start)&(times<end)],end]
        speed=np.stack([np.interp(grid,times,rpm[:,i]) for i in range(4)],axis=-1)
        elapsed=end-start
        return np.trapezoid(speed**2,grid,axis=0)/elapsed,np.trapezoid(speed,grid,axis=0)/elapsed,elapsed
    def ingest(self,msg,clock):
        kind=msg.get_type()
        if kind in ('HEARTBEAT', 'EXTENDED_SYS_STATE', 'ALTITUDE') and (
                msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1):
            return
        if kind=='HEARTBEAT':
            self.last_heartbeat=msg
            self.received[kind]=clock
            if msg.base_mode & 128:
                self._preflight_altitudes.clear()
        if kind=='EXTENDED_SYS_STATE' and msg.landed_state != 1:
            self._preflight_altitudes.clear()
        if kind in ('ATTITUDE_QUATERNION','HIGHRES_IMU','EXTENDED_SYS_STATE'):
            self.messages[kind]=msg;self.received[kind]=clock
        if kind=='ALTITUDE':
            self.record_barometric_altitude(msg, clock)
        if kind=='COMMAND_ACK':self.acks.append(dict(command=msg.command,result=msg.result,time_s=clock))
        if kind=='STATUSTEXT':self.events.append(dict(time_s=clock,severity=msg.severity,text=msg.text))
    def record_barometric_altitude(self, msg, clock):
        altitude = msg.altitude_monotonic
        if not np.isfinite(altitude):
            self._preflight_altitudes.clear()
            return
        if msg.time_usec <= self._altitude_timestamp:
            return
        self._altitude_timestamp = msg.time_usec
        self.messages['ALTITUDE'] = msg
        self.received['ALTITUDE'] = clock
        if self.takeoff_baro_zero_m is not None:
            return
        ground = self.messages.get('EXTENDED_SYS_STATE')
        if (self.last_heartbeat is None or self.last_heartbeat.base_mode & 128
                or clock-self.received['HEARTBEAT'] > 1.5 or ground is None
                or ground.landed_state != 1 or clock-self.received['EXTENDED_SYS_STATE'] > 1.):
            self._preflight_altitudes.clear()
            return
        samples = self._preflight_altitudes
        if samples and clock-samples[-1][0] > .25:
            samples.clear()
        samples.append((clock, altitude))
        if len(samples) >= 5 and clock-samples[0][0] >= 1.:
            self.takeoff_baro_zero_m = float(np.median([value for _, value in samples])) - self.reference_sample_height_m
            self.takeoff_baro_locked_s = clock
            samples.clear()
    def pump(self,clock):
        while True:
            msg=self.link.recv_match(blocking=False)
            if msg is None:break
            self.ingest(msg,clock)
    def heartbeat(self):
        from pymavlink import mavutil
        self.link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,mavutil.mavlink.MAV_STATE_ACTIVE)
    def command(self,command,*args):
        params=[*args]+[0.]*(7-len(args));self.link.mav.command_long_send(1,1,command,0,*params)
    def request_height_telemetry(self):
        self.command(511, 141, 50_000)   # ALTITUDE, 20 Hz
        self.command(511, 245, 200_000)  # EXTENDED_SYS_STATE, 5 Hz
    def setpoint(self,rates,thrust,clock):
        self.link.mav.set_attitude_target_send(int(clock*1000)&0xffffffff,1,1,128,[1,0,0,0],*map(float,rates),float(thrust))
        self.commands+=1
    def mode_arm(self):
        from pymavlink import mavutil
        if self.takeoff_baro_zero_m is None:
            raise RuntimeError('Takeoff barometric reference unavailable')
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,1.,6.,0.)
        self.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1.,0.)
    def is_offboard_armed(self):
        h=self.last_heartbeat
        return h is not None and bool(h.base_mode&128) and ((h.custom_mode>>16)&255)==6
    def update(self,rpm,applied,clock):
        required = ('ATTITUDE_QUATERNION', 'HIGHRES_IMU', 'ALTITUDE')
        if self.takeoff_baro_zero_m is None or not all(k in self.messages for k in required):return None
        if any(clock-self.received[k]>.25 for k in required):
            raise RuntimeError('Stale PX4 onboard telemetry: refuse frozen sensor reuse')
        a=self.messages['ATTITUDE_QUATERNION'];imu=self.messages['HIGHRES_IMU']
        height = self.messages['ALTITUDE'].altitude_monotonic - self.takeoff_baro_zero_m
        q=np.array([a.q1,a.q2,a.q3,a.q4]);q/=np.linalg.norm(q)
        if self.heading is None:
            from .network import rotation
            r=rotation(q);self.heading=np.arctan2(r[1,0],r[0,0])
        force=np.array([imu.xacc,imu.yacc,imu.zacc])
        self.filtered_force=force if self.filtered_force is None else self.filtered_force+.5*(force-self.filtered_force)
        gyro_time=imu.time_usec/1e6
        if self.previous_gyro_time is not None and gyro_time<=self.previous_gyro_time:
            raise RuntimeError('PX4 gyro timestamp did not advance at policy tick')
        moments=self.motor_moments(gyro_time)
        packet=Sensors(relative_attitude(q,self.heading),np.array([imu.xgyro,imu.ygyro,imu.zgyro]),self.filtered_force,
            np.array(rpm),float(height),moments)
        self.previous_gyro_time=gyro_time
        rates,thrust,info=self.controller(packet,np.array(applied))
        self.setpoint(rates,thrust,clock);self.latest=(packet,rates,thrust,info)
        return self.latest
