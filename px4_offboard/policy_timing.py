"""Source-time grid shared by Position preprocessing and onboard inference."""
from dataclasses import replace
from collections import deque
import json
import math
import os
from pathlib import Path
import tempfile
from threading import RLock

import numpy as np

POLICY_DT = .01
FORCE_ALPHA = .5


class FeedbackHistory:
    """Never attach future ESC/output feedback to an older IMU sample."""
    def __init__(self):
        self.lock=RLock()
        self.rows=deque(maxlen=128)
        self.source='default'
        self.rollbacks=0
    def add(self,stamp,value,source='default'):
        with self.lock:
            if not math.isfinite(stamp) or stamp<=0:return
            if self.source=='custom' and source!='custom':return
            if source=='custom' and self.source!='custom':
                self.rows.clear();self.source='custom'
            if self.rows and stamp<self.rows[-1][0]:
                self.rows.clear();self.rollbacks+=1
            if self.rows and stamp==self.rows[-1][0]:return
            self.rows.append((float(stamp),np.asarray(value,float).copy()))
    def at(self,stamp,max_age=.03):
        with self.lock:
            for time,value in reversed(tuple(self.rows)):
                age=stamp-time
                if age>=-1e-9:return (value.copy(),float(max(age,0))) if age<=max_age else None
            return None


class PolicySampler:
    """Interpolate only bracketed samples; no extrapolation or repeated GRU ticks."""
    def __init__(self, dt=POLICY_DT, max_gap=.05):
        if not math.isfinite(dt) or not 0 < dt <= max_gap: raise ValueError('Invalid policy grid')
        self.dt, self.max_gap = dt, max_gap
        self.reset()

    def reset(self):
        self.previous = None
        self.origin = None
        self.tick = 0
        self.blocked = False

    def push(self, stamp, packet, applied):
        if self.blocked: raise ValueError('Policy grid blocked until mode exit/reset')
        values = np.r_[packet.attitude, packet.gyro, packet.specific_force,
                       packet.motor_speed, packet.baro_height, applied]
        if not math.isfinite(stamp) or stamp <= 0 or not np.isfinite(values).all():
            self.blocked = True
            raise ValueError('Finite positive source timestamp and sensors required')
        applied = np.asarray(applied, float).copy()
        if applied.shape != (4,) or (applied < 0).any() or (applied > 1).any() or (packet.motor_speed < 0).any():
            self.blocked = True
            raise ValueError('Measured motor feedback outside bounds')
        if self.previous is None:
            self.origin = stamp
            self.previous = (stamp, packet, applied)
            self.tick = 1
            return [(stamp, packet, applied)]
        before, old, old_applied = self.previous
        gap = stamp - before
        if gap == 0: return []
        if not 0 < gap <= self.max_gap + 1e-10:
            self.blocked = True
            self.previous = None
            raise ValueError('Source clock rollback or gap')
        result = []
        while self.origin + self.tick * self.dt <= stamp + 1e-9:
            at = self.origin + self.tick * self.dt
            weight = float(np.clip((at - before) / gap, 0, 1))
            q = packet.attitude if old.attitude @ packet.attitude >= 0 else -packet.attitude
            q = old.attitude + weight * (q - old.attitude)
            q = q / np.linalg.norm(q)
            blend = lambda a, b: a + weight * (b - a)
            item = replace(packet, attitude=q, gyro=blend(old.gyro, packet.gyro),
                specific_force=blend(old.specific_force, packet.specific_force),
                motor_speed=blend(old.motor_speed, packet.motor_speed),
                baro_height=blend(old.baro_height, packet.baro_height), interval_motor_moments=None)
            result.append((at, item, (applied if abs(at-stamp) < 1e-9 else old_applied).copy()))
            self.tick += 1
        self.previous = (stamp, packet, applied)
        return result


class ForceFilter:
    def __init__(self): self.reset()
    def reset(self): self.value = None
    def update(self, force):
        force = np.asarray(force, float)
        self.value = force.copy() if self.value is None else self.value + FORCE_ALPHA * (force-self.value)
        return self.value.copy()


def filter_forces(values):
    f = ForceFilter()
    return np.array([f.update(row) for row in values])


def measured_motor_moments(previous, current):
    """Exact moments of the piecewise-linear measured RPM path, not command lag."""
    a, b = np.asarray(previous), np.asarray(current)
    return (a*a + a*b + b*b) / 3., (a+b) / 2.


class FixedPolicyRunner:
    """Shared SIM/onboard grid, filter, observer and frozen Actor execution."""
    def __init__(self, controller, interface='acceleration'):
        self.controller=controller
        self.interface=interface
        if controller.calibration is not None and controller.calibration.parameters.get('schema')!='P3_aligned_residual_v2':
            raise ValueError('Use a retrained aligned P3 with the fixed processing chain')
        if (controller.calibration is not None and controller.calibration.parameters.get('acceleration_actuation')
            and interface!='acceleration'):
            raise ValueError('Attitude reference actuation applies to the acceleration interface only')
        self.sampler=PolicySampler(controller.dt)
        self.filter=ForceFilter()

    def reset(self):
        self.sampler.reset();self.filter.reset();self.controller.reset()

    def advance(self, packet, applied, stamp):
        from .unified_p3 import nominal_alpha
        result=[]
        try:ticks=self.sampler.push(stamp,packet,applied)
        except ValueError:
            self.controller.reset();self.filter.reset()
            raise
        for tick,p,u in ticks:
            c=self.controller;v=c.observer
            source_force=p.specific_force.copy()
            square,mean=measured_motor_moments(v.rpm,p.motor_speed)
            raw=p.gyro-(v.rate+c.dt*nominal_alpha(v.rate,square,mean,v.pc)) if v.valid else np.zeros(3)
            p=replace(p,specific_force=self.filter.update(p.specific_force),
                      interval_motor_moments=(square,mean,c.dt))
            v.sample_dt_s=c.dt
            if self.interface=='acceleration':
                acceleration=c.acceleration(p,u);rates=np.zeros(3);thrust=0.
            else:
                rates,thrust,_=c(p,u);acceleration=c.latest_acceleration
                if c.calibration is not None:rates=c.calibration.rates_to_reported(rates)
            wire_acceleration=acceleration
            if self.interface=='acceleration' and c.calibration is not None:
                wire_acceleration=c.calibration.adapt_acceleration(acceleration,p.attitude,v.g,c.limits)
            diagnostic=dict(grid_stamp_s=float(tick),source_stamp_s=float(stamp),
                interpolation_latency_s=float(max(0,stamp-tick)),
                innovation_before_radps=np.asarray(getattr(v,'innovation_before',raw)).tolist(),
                innovation_after_radps=np.asarray(getattr(v,'innovation_after',raw)).tolist(),
                innovation_to_actor_radps=(c.last_observation[23:26]*.1).tolist(),
                observation=c.last_observation.tolist(),actor_raw=c.last_raw_action.tolist(),
                unfiltered_acceleration=c.last_unfiltered_acceleration.tolist(),
                acceleration=acceleration.tolist(),wire_acceleration=wire_acceleration.tolist(),rates=rates.tolist(),thrust=float(thrust),
                feature_ood_fraction=float(getattr(v,'feature_ood_fraction',0.)))
            diagnostic.update(grid_attitude_wxyz=p.attitude.tolist(),grid_gyro_radps=p.gyro.tolist(),
                grid_force_unfiltered_mps2=source_force.tolist(),grid_rpm_radps=p.motor_speed.tolist(),
                grid_applied_throttle=u.tolist(),baro_height_m=float(p.baro_height))
            diagnostic['P3_support_confidence']=float(getattr(v,'support_confidence',1.))
            diagnostic['virtual_actuation_scale']=float(getattr(v,'virtual_actuation_scale',1.))
            if hasattr(v,'unified_compensation'):
                diagnostic.update(unified_compensation=v.unified_compensation.tolist(),
                    equivalent_self_torque_nm=v.equivalent_self_torque.tolist())
            diagnostic['innovation_clipped_before']=(np.abs(diagnostic['innovation_before_radps'])>.5).tolist()
            diagnostic['innovation_clipped_after']=(np.abs(diagnostic['innovation_after_radps'])>.5).tolist()
            result.append(diagnostic)
        return result


def atomic_json(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as out:
            temporary = out.name
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None: Path(temporary).unlink(missing_ok=True)


def shutdown_error(exc, requested, interrupt_type):
    return None if requested and isinstance(exc, interrupt_type) else repr(exc)
