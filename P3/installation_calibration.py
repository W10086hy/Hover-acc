"""Position identification of IMU mounting, motor cant, CG and full inertia."""
import numpy as np
import torch

from P2.hover_bptt.dynamics.torch_model import rotations
from P2.hover_bptt.math3d import multiply
from P2.hover_bptt.onboard import Sensors
from .body_calibration import BodyCalibration, prepare_position
from .position_identification import _position_samples, _position_kinematics, SPIN_SIGNS


def rotor_wrench(nominal, squared, mean, gyro, axis):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-6 or axis[2] > -.98:
        raise ValueError('Small common upward thrust axis required')
    a, m = nominal['airframe'], nominal['actuators']
    positions = np.array([[a['rotor_dx_m'],a['rotor_dy_m'],0],[-a['rotor_dx_m'],a['rotor_dy_m'],0],
                          [a['rotor_dx_m'],-a['rotor_dy_m'],0],[-a['rotor_dx_m'],-a['rotor_dy_m'],0]])
    force = m['kf']*squared.sum(1)[:,None]*axis
    mapping = m['kf']*np.cross(positions,axis)-m['km']*SPIN_SIGNS[:,None]*axis
    torque = squared@mapping
    if m.get('include_gyro'):
        torque -= nominal['power']['rotor_inertia_kgm2']*(mean@SPIN_SIGNS)[:,None]*np.cross(gyro,axis)
    torque -= gyro*np.asarray(nominal['aero']['body_rate_drag'])
    return force, torque


def pose_rates(samples):
    time, _, gyro, _, _, before = _position_samples(samples)
    stamp = np.asarray(samples['position_time_s'], float)
    q = np.asarray(samples['position_attitude_wxyz'], float)
    reset = np.asarray(samples['position_reset_counter'])
    if q.shape != (len(stamp),4) or reset.shape != stamp.shape or np.any(np.diff(stamp)<=0):
        raise ValueError('Invalid independent body pose reference')
    dt = np.diff(stamp)
    valid = np.isfinite(q).all(1) & (np.linalg.norm(q,axis=1)>.5)
    good = valid[:-1]&valid[1:]&(np.diff(reset)==0)&(dt <= min(.1,2.5*np.median(dt)))
    covered = np.zeros_like(good)
    for run in np.split(before,np.flatnonzero(np.diff(before)!=1)+1):
        covered |= (stamp[:-1]>=time[run[0]])&(stamp[1:]<=time[run[-1]+1])
    indices = np.flatnonzero(good&covered)
    chunks=[]
    for run in np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1):
        if len(run)<3: continue
        R0=rotations(torch.tensor(q[run])).numpy(); R1=rotations(torch.tensor(q[run+1])).numpy()
        delta=R0.transpose(0,2,1)@R1
        angle=np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1))
        if (angle>.25).any(): raise ValueError('Pose reference angular jump')
        coeff=np.full_like(angle,.5); mask=angle>1e-6
        coeff[mask]=angle[mask]/(2*np.sin(angle[mask]))
        vec=np.column_stack((delta[:,2,1]-delta[:,1,2],delta[:,0,2]-delta[:,2,0],delta[:,1,0]-delta[:,0,1]))
        rate=vec*coeff[:,None]/dt[run,None]
        mid=(stamp[run]+stamp[run+1])/2
        measured=np.column_stack([np.interp(mid,time,axis) for axis in gyro.T])
        chunks.append((mid,rate,measured))
    if sum(len(c[0]) for c in chunks)<50: raise ValueError('Insufficient contiguous Position pose excitation')
    return chunks


def estimate_mount(samples):
    chunks=pose_rates(samples)
    body=np.concatenate([c[1] for c in chunks]); sensor=np.concatenate([c[2] for c in chunks])
    x=sensor-sensor.mean(0); y=body-body.mean(0)
    U,s,Vt=np.linalg.svd(x.T@y)
    if not np.isfinite(s).all() or s[0]<=1e-12 or s[-1]<.001*s[0]:
        raise ValueError('Three-axis IMU mounting excitation unresolved')
    Q=U@np.diag([1.,1.,np.linalg.det(U@Vt)])@Vt
    C=Q.T
    bias=sensor.mean(0)-body.mean(0)@C
    error=(sensor-bias)@C.T-body
    return dict(body_from_sensor=C.tolist(), gyro_bias_sensor_radps=bias.tolist(),
                gyro_fit_rmse_radps=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                relative_min_singular=float(s[-1]/s[0]), reference='independent_Position_body_pose')


def body_samples(samples, mounting):
    C=np.asarray(mounting['body_from_sensor']); b=np.asarray(mounting['gyro_bias_sensor_radps'])
    output=dict(samples)
    output['specific_force_mps2']=np.asarray(samples['specific_force_mps2'])@C.T
    gyro=(np.asarray(samples['gyro_radps'])-b)@C.T
    time=np.asarray(samples['time_s']); active=np.zeros(len(time),bool)
    # Use the allowed independent pose derivative for inertial fitting, not a
    # noisy gyro derivative. Raw IMU remains the runtime source after calibration.
    for t,w,_ in pose_rates(samples):
        mask=(time>=t[0])&(time<=t[-1])
        gyro[mask]=np.column_stack([np.interp(time[mask],t,axis) for axis in w.T])
        active |= mask
    output['gyro_radps']=gyro
    output['position_armed_airborne']=np.asarray(samples['position_armed_airborne'])&active
    return output


def fit_axis_mass(samples, nominal, r, *, axis=None, mass=None):
    time,_,gyro,rpm,_,before=_position_samples(samples)
    drag, reference=_position_kinematics(samples,nominal,time,before)
    after=before+1
    alpha=(gyro[after]-gyro[before])/(time[after]-time[before])[:,None]
    f=reference+np.cross(alpha,r)+np.cross(gyro[after],np.cross(gyro[after],r))
    drag=drag*nominal['airframe']['mass_kg']
    T=nominal['actuators']['kf']*np.sum(rpm[after]**2,axis=1)
    valid=np.isfinite(f).all(1)&np.isfinite(drag).all(1)
    f,drag,T=f[valid],drag[valid],T[valid]
    if len(T)<100: raise ValueError('Insufficient Position force reference')
    if axis is None:
        a=np.sum(T[:,None]*f,axis=0)/np.sum(T*T)
        b=np.sum(T[:,None]*drag,axis=0)/np.sum(T*T)
        k=1-b@b
        beta=(-a@b+np.sqrt((a@b)**2+k*(a@a)))/k
        axis=(a-beta*b)/beta; mass=1/beta
    axis=np.asarray(axis); mass=float(mass)
    if not .7*nominal['airframe']['mass_kg']<mass<1.3*nominal['airframe']['mass_kg'] or axis[2]>-.98:
        raise ValueError('Motor axis/mass outside supported calibration bounds')
    error=(T[:,None]*axis+drag)-mass*f
    return dict(thrust_axis_body=axis.tolist(),mass_kg=mass,samples=len(T),force_rmse_n=np.sqrt(np.mean(error**2,axis=0)).tolist())


class InstalledCalibration(BodyCalibration):
    def __init__(self, nominal_inertia, body_from_sensor=None, gyro_bias_sensor=None, thrust_axis=None):
        super().__init__(nominal_inertia)
        del self.log_inertia_scale
        self.inertia_coeff=torch.nn.Parameter(torch.tensor([1.,1.,1.,0.,0.,0.],dtype=torch.float64))
        C=np.eye(3) if body_from_sensor is None else np.asarray(body_from_sensor)
        d=np.array([0.,0.,-1.]) if thrust_axis is None else np.asarray(thrust_axis)
        z=-d; qw=np.sqrt((1+z[2])/2); q=np.r_[qw,np.cross([0.,0.,1.],z)/(2*qw)]
        self.register_buffer('body_from_sensor',torch.tensor(C,dtype=torch.float64))
        self.register_buffer('gyro_bias_sensor',torch.tensor(np.zeros(3) if gyro_bias_sensor is None else gyro_bias_sensor,dtype=torch.float64))
        self.register_buffer('virtual_quaternion',torch.tensor(q,dtype=torch.float64))
        self.register_buffer('body_from_virtual',rotations(self.virtual_quaternion[None])[0])
        basis=torch.zeros(6,3,3,dtype=torch.float64)
        for i in range(3): basis[i,i,i]=self.nominal_inertia[i]
        for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):
            basis[k,i,j]=basis[k,j,i]=torch.sqrt(self.nominal_inertia[i]*self.nominal_inertia[j])
        self.register_buffer('inertia_basis',basis)

    @property
    def full_inertia(self): return torch.einsum('k,kij->ij',self.inertia_coeff,self.inertia_basis)
    @property
    def inertia(self): return self.full_inertia.diag()
    @property
    def com_geometry(self): return torch.cat((self.com_xy,self.com_xy.new_zeros(1)))
    @property
    def com_body(self): return self.body_from_virtual.T@self.com_geometry

    def momentum_torque(self, gyro, alpha, origin_force, gyro_outer=None):
        J=self.full_inertia.to(gyro)
        if gyro_outer is None: quadratic=torch.linalg.cross(gyro,gyro@J.T)
        else:
            M=gyro_outer@J.T
            quadratic=torch.stack((M[:,1,2]-M[:,2,1],M[:,2,0]-M[:,0,2],M[:,0,1]-M[:,1,0]),-1)
        return alpha@J.T+quadratic+torch.linalg.cross(self.com_geometry.to(gyro).expand_as(gyro),origin_force)

    def correct_packet(self, packet, *, preapplied_offset=None):
        force=packet.specific_force
        if preapplied_offset is not None: force=force+torch.as_tensor(preapplied_offset).to(force)
        C=self.body_from_sensor.to(force); V=self.body_from_virtual.to(force)
        f=(force@C.T-self.bias.to(force))@V
        w=((packet.gyro-self.gyro_bias_sensor.to(force))@C.T)@V
        q=multiply(packet.attitude,self.virtual_quaternion.to(packet.attitude).expand_as(packet.attitude))
        return Sensors(q,w,f,packet.motor_speed,packet.baro_height)


def fit_full_dynamics(data, *, steps=1600, mounting=None, axis=None):
    mounting=mounting or dict(body_from_sensor=np.eye(3),gyro_bias_sensor_radps=np.zeros(3))
    model=InstalledCalibration(data['nominal_inertia'],mounting['body_from_sensor'],mounting['gyro_bias_sensor_radps'],axis)
    values={k:torch.tensor(np.asarray(data[k]),dtype=torch.float64) for k in ('gyro','alpha','origin_force','torque','bias_residual')}
    w,a,f,t,b=(values[k] for k in ('gyro','alpha','origin_force','torque','bias_residual'))
    outer=torch.tensor(data.get('gyro_outer',np.asarray(data['gyro'])[:,:,None]*np.asarray(data['gyro'])[:,None,:]),dtype=w.dtype)
    columns=[]
    for J in model.inertia_basis:
        M=outer@J.T
        cross=torch.stack((M[:,1,2]-M[:,2,1],M[:,2,0]-M[:,0,2],M[:,0,1]-M[:,1,0]),-1)
        columns.append((a@J.T+cross)/model.nominal_inertia)
    for r in torch.eye(3,dtype=w.dtype)[:2]*.01:
        columns.append(torch.linalg.cross(r.expand_as(f),f)/model.nominal_inertia)
    D=torch.stack(columns,-1).reshape(-1,8)
    singular=torch.linalg.svdvals(D); relative=float(singular[-1]/singular[0])
    if relative<1e-5 or not torch.isfinite(D).all(): raise ValueError('Full inertia/CG excitation unresolved')
    inverse=torch.linalg.pinv(D)
    optimizer=torch.optim.Adam([dict(params=[model.bias],lr=.015),dict(params=[model.com_xy],lr=.0005),dict(params=[model.inertia_coeff],lr=.015)])
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        error=(model.momentum_torque(w,a,f,outer)-t)/model.nominal_inertia
        loss=(inverse@error.reshape(-1)).square().mean()+((model.bias-b)/.1).square().mean()
        loss.backward(); optimizer.step()
    eigen=torch.linalg.eigvalsh(model.full_inertia.detach())
    if (eigen<=0).any() or 2*eigen.max()>eigen.sum() or (model.com_xy.abs()>.03).any() or (model.bias.abs()>1).any():
        raise ValueError('Nonphysical installed calibration')
    return model,optimizer,dict(steps=steps,loss=float(loss.detach()),relative_min_singular=relative,
        accel_bias_body_mps2=model.bias.detach().tolist(),com_geometry_m=model.com_geometry.detach().tolist(),
        full_inertia_body_kgm2=model.full_inertia.detach().tolist())


def score(model, data):
    with torch.no_grad():
        w,a,f,t,b=[torch.tensor(data[k],dtype=torch.float64) for k in ('gyro','alpha','origin_force','torque','bias_residual')]
        outer=torch.tensor(data['gyro_outer'],dtype=w.dtype)
        error=model.momentum_torque(w,a,f,outer)-t
        return dict(samples=len(w),torque_rmse_nm=error.square().mean(0).sqrt().tolist(),
                    accel_rmse_mps2=(b-model.bias).square().mean(0).sqrt().tolist())


def train_installed(samples, nominal, *, steps=1600, controlled_external_loads=False):
    if not controlled_external_loads:
        raise ValueError('Installed calibration requires controlled external loads')
    from .train_body_calibration import split_position
    raw,held,boundary=split_position(samples)
    mount=estimate_mount(raw)
    body=body_samples(raw,mount); validation=body_samples(held,mount)
    r=np.zeros(3)
    for iteration in range(3):
        force_fit=fit_axis_mass(body,nominal,r)
        axis=force_fit['thrust_axis_body']
        data=prepare_position(body,nominal,controlled_external_loads=True,rotor_axes_body=axis)
        model,optimizer,report=fit_full_dynamics(data,steps=steps,mounting=mount,axis=axis)
        r=model.com_geometry.detach().numpy()
    val=prepare_position(validation,nominal,controlled_external_loads=True,
        motor_response=data['motor_response'],rotor_axes_body=axis)
    report.update(mounting=mount,axis_mass=force_fit,train=score(model,data),holdout=score(model,val),
        axis_mass_holdout=fit_axis_mass(validation,nominal,r,axis=axis,mass=force_fit['mass_kg']),
        raw_time_holdout_independent=True,split_time_s=boundary,total_adam_steps=3*steps,
        mounting_holdout_reference='same independent pose definition; fixed training rotation',
        com_z_identified=False,angular_fitting_reference='Position body attitude derivative',
        block_gyro_covariance_included=True)
    report['development_quality_gate_passed']=(max(report['holdout']['torque_rmse_nm'])<.015
        and max(report['holdout']['accel_rmse_mps2'])<.08 and max(report['axis_mass_holdout']['force_rmse_n'])<.25)
    return model.requires_grad_(False),optimizer,report
