"""Nine residuals on a fixed policy grid, with measured interval RPM moments."""
from dataclasses import replace
import numpy as np
from .control import Observer
from .network import rotation
from .unified_p3 import UnifiedCalibration, nominal_alpha
from .policy_timing import POLICY_DT, FORCE_ALPHA, measured_motor_moments

PROCESSING = dict(policy_dt_s=POLICY_DT, force_filter_alpha=FORCE_ALPHA,
                  motor_moments='measured_rpm_linear')


def support_confidence(features,parameters):
    gate=parameters.get('feature_support_gate')
    if gate is None:return np.ones(np.asarray(features).shape[:-1])
    z=np.abs((features-np.asarray(parameters['feature_mean']))/np.asarray(parameters['feature_scale']))
    return np.clip((gate['zero_above']-z.max(axis=-1))/(gate['zero_above']-gate['full_below']),0,1)


def angular_prediction(features,weights,limits,parameters):
    confidence=support_confidence(features,parameters)
    if parameters.get('feature_support_fallback')=='training_mean':
        mean=np.asarray(parameters['feature_mean'])
        supported=mean+confidence[...,None]*(features-mean)
        return np.clip(supported@np.asarray(weights).T,-limits,limits)
    return np.clip(features@np.asarray(weights).T,-limits,limits)*confidence[...,None]


def virtual_actuation(rpm,own_alpha,reference):
    """Equivalent standard-body rotor observation, preserving collective thrust.

    This changes only the Actor's observation. Physical RPM/commands remain the
    inputs to the P3 causal model and are never replaced in the feedback store.
    Infeasible corrections are reduced along the zero-collective direction.
    """
    square=np.asarray(rpm,float)**2
    change=np.linalg.solve(reference.allocation,np.r_[0.,reference.J*own_alpha])/reference.kf
    factors=np.ones(4)
    positive=change>0;negative=change<0
    factors[positive]=(reference.vmax**2-square[positive])/change[positive]
    factors[negative]=-square[negative]/change[negative]
    scale=float(np.clip(factors.min(),0,1))
    return np.sqrt(np.maximum(square+scale*change,0)),scale


def aligned_features(previous_gyro, force, previous_rpm, rpm):
    w = np.asarray(previous_gyro)
    a, b = np.asarray(previous_rpm), np.asarray(rpm)
    square, mean = measured_motor_moments(a, b)
    spin = mean @ np.array([1., -1., -1., 1.])
    quadratic = np.stack([w[..., i]*w[..., j] for i,j in
        ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))], axis=-1)
    features = np.concatenate((np.ones(w.shape[:-1]+(1,)),w/.5,quadratic/.25,
        np.asarray(force)/10.,square/360000.,a*a/360000.,b*b/360000.,spin[...,None]*w/300.),axis=-1)
    return features, square, mean


class AlignedCalibration(UnifiedCalibration):
    def __init__(self, parameters):
        processing=dict(parameters.get('processing',{}))
        force_contract=processing.pop('force_trim_contract',None)
        if (parameters.get('schema')!='P3_aligned_residual_v2' or processing!=PROCESSING
            or force_contract not in (None,'position_static_blend_v1')
            or (force_contract is not None and parameters.get('preflight_force_mode')!='position_consistency')):
            raise ValueError('Aligned P3 needs its exact 100Hz/filter/measured-RPM contract')
        super().__init__({**parameters,'schema':'P3_unified_residual_v1'})
        self.parameters = parameters
        self.force_mode=parameters.get('preflight_force_mode','static_residual')
        self.static_blend=float(parameters.get('static_residual_blend',0.))
        self.position_force_offset=np.asarray(parameters.get('position_force_offset_mps2',[0.,0.,0.]),float)
        if (self.force_mode not in ('static_residual','position_consistency')
            or self.position_force_offset.shape!=(3,) or not np.isfinite(self.position_force_offset).all()
            or np.linalg.norm(self.position_force_offset)>.1
            or not np.isfinite(self.static_blend) or not 0<=self.static_blend<=1
            or (self.force_mode=='static_residual' and self.static_blend!=0)
            or (self.force_mode=='static_residual' and np.any(self.position_force_offset))):
            raise ValueError('Bounded, explicit Position force-refinement contract required')
        self.angular_innovation_map=np.asarray(parameters.get('angular_innovation_map',np.eye(3)),float)
        if (self.angular_innovation_map.shape!=(3,3) or not np.isfinite(self.angular_innovation_map).all()
            or np.linalg.det(self.angular_innovation_map)<=0
            or np.linalg.cond(self.angular_innovation_map)>10
            or np.linalg.svd(self.angular_innovation_map,compute_uv=False).min()<.25
            or np.linalg.svd(self.angular_innovation_map,compute_uv=False).max()>4):
            raise ValueError('Finite orientation-preserving, conditioned angular response map required')
        if 'feature_mean' in parameters:
            mean,scale=np.asarray(parameters['feature_mean']),np.asarray(parameters['feature_scale'])
            if mean.shape!=(28,) or scale.shape!=(28,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
                raise ValueError('Finite training-only feature normalization required')
        if 'feature_support_gate' in parameters:
            gate=parameters['feature_support_gate']
            if 'feature_mean' not in parameters or gate!={'full_below':4.,'zero_above':8.}:
                raise ValueError('Supported Position feature gate required')
        if parameters.get('feature_support_fallback','zero') not in ('zero','training_mean'):
            raise ValueError('Unknown P3 support fallback')
        if parameters.get('feature_support_fallback')=='training_mean' and 'feature_mean' not in parameters:
            raise ValueError('Training mean required for supported residual fallback')
        if parameters.get('actuation_observation','measured') not in ('measured','standard_vehicle_equivalent'):
            raise ValueError('Unknown actuation observation contract')
        self.attitude_right=np.asarray(parameters.get('attitude_right_quaternion',[1.,0,0,0]),float)
        if (self.attitude_right.shape!=(4,) or not np.isfinite(self.attitude_right).all()
            or abs(np.linalg.norm(self.attitude_right)-1)>1e-8 or abs(self.attitude_right[3])>1e-10
            or self.attitude_right[0]<np.cos(np.deg2rad(3)/2)):
            raise ValueError('Validated small right-tilt attitude correction required')
        actuation=parameters.get('acceleration_actuation')
        if actuation is not None and (actuation!='attitude_reference_consistent'
                                    or 'attitude_right_quaternion' not in parameters):
            raise ValueError('Acceleration actuation requires a validated attitude reference')

    def select_preflight_offset(self,static_offset):
        static=np.asarray(static_offset,float)
        return ((1-self.static_blend)*self.position_force_offset+self.static_blend*static
            if self.force_mode=='position_consistency' else static.copy())

    def adapt_acceleration(self,acceleration,corrected_attitude,gravity,limits):
        if not self.parameters.get('acceleration_actuation'):
            return acceleration
        corrected=rotation(corrected_attitude)
        corrected_down=corrected[:,2]
        reported_down=(corrected@rotation(self.attitude_right).T)[:,2]
        if min(corrected_down[2],reported_down[2])<.5:
            return acceleration
        correction=gravity*(corrected_down[:2]/corrected_down[2]-reported_down[:2]/reported_down[2])
        wire=np.array(acceleration,float,copy=True)
        wire[:2]=np.clip(wire[:2]+correction,-np.asarray(limits)[:2],np.asarray(limits)[:2])
        return wire

    def correct_packet(self,packet):
        from .p3_observer import multiply
        corrected=super().correct_packet(packet)
        q=multiply(corrected.attitude,self.attitude_right)
        return replace(corrected,attitude=q/np.linalg.norm(q))


class AlignedObserver(Observer):
    def __init__(self,cfg,calibration,dt=POLICY_DT):
        if not np.isclose(dt,POLICY_DT,rtol=0,atol=1e-12):raise ValueError('Frozen P2 requires 10ms grid')
        calibration.validate_reference(cfg['nominal'])
        self.calibration=calibration
        super().__init__(cfg,dt)

    def reset(self):
        super().reset()
        self.sample_dt_s=self.dt
        self.previous_force=np.zeros(3)
        self.reference_force_offset=self.calibration.position_force_offset.copy()
        self.unified_compensation=np.zeros(9)
        self.equivalent_self_torque=np.zeros(3)
        self.innovation_before=np.zeros(3)
        self.innovation_residual=np.zeros(3)
        self.innovation_after=np.zeros(3)
        self.last_standardized_packet=None
        self.feature_ood_fraction=0.
        self.support_confidence=1.
        self.virtual_actuation_scale=1.
        self.innovation_clipped_before=np.zeros(3,dtype=bool)
        self.innovation_clipped_after=np.zeros(3,dtype=bool)

    def update(self,packet,applied):
        if not np.isclose(self.sample_dt_s,self.dt,rtol=0,atol=1e-9):
            raise ValueError('Resample source clock before the frozen Actor')
        valid=self.valid
        force=.5*(self.previous_force+packet.specific_force) if valid else packet.specific_force
        # Keep the frozen own model in the force reference used during its fit.
        features,square,mean=aligned_features(self.rate,force+self.reference_force_offset,self.rpm,packet.motor_speed)
        self.support_confidence=float(support_confidence(features,self.calibration.parameters))
        own=angular_prediction(features,self.calibration.angular_weights,
            self.calibration.angular_limit,self.calibration.parameters)
        predicted=self.rate+self.dt*nominal_alpha(self.rate,square,mean,self.pc)
        self.innovation_before=packet.gyro-predicted if valid else np.zeros(3)
        self.innovation_residual=self.innovation_before-self.dt*own if valid else np.zeros(3)
        self.innovation_after=self.calibration.angular_innovation_map@self.innovation_residual
        result=super().update(replace(packet,interval_motor_moments=(square,mean,self.dt)),applied)
        result[23:26]=np.clip(self.innovation_after/.1,-5,5)
        if self.calibration.parameters.get('actuation_observation')=='standard_vehicle_equivalent':
            virtual,scale=virtual_actuation(packet.motor_speed,own,self)
            target=np.maximum(np.polyval(self.poly,np.clip(applied,0,1)),0)
            virtual_target,command_scale=virtual_actuation(target,own,self)
            result[10:14]=virtual/self.vmax
            result[14:18]=2*self.throttle(virtual_target)-1
            self.virtual_actuation_scale=min(scale,command_scale)
        self.innovation_clipped_before=np.abs(self.innovation_before)>.5
        self.innovation_clipped_after=np.abs(self.innovation_after)>.5
        self.equivalent_self_torque=self.J*own
        sensor_correction=self.calibration.last_sensor_correction.copy()
        sensor_correction[3:]-=self.reference_force_offset
        self.unified_compensation=np.r_[sensor_correction,own]
        self.previous_force=packet.specific_force.copy()
        self.last_standardized_packet=packet
        p=self.calibration.parameters
        if 'feature_mean' in p:
            z=(features-np.asarray(p['feature_mean']))/np.asarray(p['feature_scale'])
            self.feature_ood_fraction=float(np.mean(np.abs(z)>8))
        return result
