"""Causal sample freshness and finite preflight fitting, independent of ROS."""
import math
import numpy as np
from .network import rotation


class SampleClock:
    def __init__(self):self.stamp=None;self.received=None
    def update(self,stamp,received):
        if not math.isfinite(stamp) or not math.isfinite(received) or stamp<=0 or (self.stamp is not None and stamp<=self.stamp):return False
        self.stamp=float(stamp);self.received=float(received);return True
    def fresh(self,now,limit):return self.received is not None and 0<=now-self.received<=limit


class PolicyClock(SampleClock):
    """Stop the current Offboard entry after a gap/rollback, never reuse old history."""
    def __init__(self,max_gap_s=.05):
        super().__init__();self.max_gap_s=max_gap_s;self.blocked=False;self.elapsed_s=None;self.last_rejection=None
    def invalidate(self,active,reason=None):
        if active and not self.blocked and reason is not None:self.last_rejection=reason
        self.blocked=self.blocked or active;self.stamp=None;self.received=None;self.elapsed_s=None
    def accept(self,stamp,received,*,active):
        if not active:self.blocked=False
        if self.blocked:return False
        if not math.isfinite(stamp) or stamp<=0:self.invalidate(active,'invalid IMU timestamp');return False
        delta=None if self.stamp is None else stamp-self.stamp
        if delta is not None and (delta<0 or delta>self.max_gap_s):
            self.invalidate(active,dict(IMU_gap_or_rollback_s=delta))
            if active:return False
        if self.update(stamp,received):self.elapsed_s=delta if delta is not None and 0<delta<=self.max_gap_s else None;return True
        return False


class StampedAcceleration:
    def __init__(self):self.clock=SampleClock();self.value=None
    def update(self,stamp,received,value):
        value=np.asarray(value,float)
        if value.shape!=(2,) or not np.isfinite(value).all() or not self.clock.update(stamp,received):return False
        self.value=value.copy();return True
    def get(self,now):return self.value.copy() if self.value is not None and self.clock.fresh(now,.15) else None


class PreflightCalibration:
    def __init__(self,gravity,samples=2000):self.gravity=gravity;self.required=samples;self.forces=[];self.barometers=[]
    def add(self,packet):
        if len(self.forces)>=self.required:return False
        q,w,f=np.asarray(packet.attitude),np.asarray(packet.gyro),np.asarray(packet.specific_force)
        if q.shape!=(4,) or w.shape!=(3,) or f.shape!=(3,) or not all(np.isfinite(v).all() for v in (q,w,f)) or not math.isfinite(packet.baro_height):return False
        if np.linalg.norm(q)<1e-6 or np.linalg.norm(w)>.03:return False
        self.forces.append((f+self.gravity*rotation(q)[2]).copy());self.barometers.append(float(packet.baro_height));return True
    def result(self):
        if len(self.forces)<self.required:return None
        bias=np.mean(self.forces,axis=0);zero=float(np.median(self.barometers))
        return (bias,zero) if np.isfinite(bias).all() and math.isfinite(zero) else None
