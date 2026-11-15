"""Position-only calibration in FRD. No Actor, height or tether is an input.

r points from the geometric/IMU origin to the CG. Rotor forces act at the
geometric rotor locations, so their extra moment about the CG is -r x F.
Accelerometer bias is additive measurement error, not an external torque.
"""
import copy
import numpy as np
import torch

from P2.hover_bptt.onboard import Sensors
from .position_identification import (_position_samples, _position_kinematics,
                                      _motor_fit, _rotor_wrench)


class BodyCalibration(torch.nn.Module):
    def __init__(self, nominal_inertia):
        super().__init__()
        inertia = torch.as_tensor(nominal_inertia, dtype=torch.float64)
        if inertia.shape != (3,) or not torch.isfinite(inertia).all() or (inertia <= 0).any():
            raise ValueError('Positive three-axis nominal inertia required')
        self.register_buffer('nominal_inertia', inertia.clone())
        self.bias = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.com_xy = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.log_inertia_scale = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    @property
    def accel_bias(self):
        return self.bias

    @property
    def com_body(self):
        return torch.cat((self.com_xy, self.com_xy.new_zeros(1)))

    @property
    def inertia(self):
        return self.nominal_inertia * self.log_inertia_scale.exp()

    def momentum_torque(self, gyro, alpha, rotor_force):
        inertia = self.inertia.to(gyro)
        r = self.com_body.to(gyro).expand_as(gyro)
        return inertia * alpha + torch.linalg.cross(gyro, inertia * gyro) + torch.linalg.cross(r, rotor_force)

    def correct_packet(self, packet, *, preapplied_offset=None):
        """Correct raw IMU, or subtract only the remainder after preflight calibration."""
        offset = self.bias.to(packet.specific_force)
        if preapplied_offset is not None:
            offset = offset - torch.as_tensor(preapplied_offset).to(offset)
        return Sensors(packet.attitude, packet.gyro, packet.specific_force-offset,
                       packet.motor_speed, packet.baro_height)

    def compensation_torque(self, rotor_force):
        """Translate a geometric-origin force to the learned CG: +r x F.

        Rotor-only diagnostics pass RPM-derived force. Reference transport
        passes the identified mass times bias/lever-corrected onboard IMU.
        """
        return torch.linalg.cross(self.com_body.to(rotor_force).expand_as(rotor_force), rotor_force)

    def observer_vehicle(self, vehicle):
        """Use identified CG for RPM-based angular prediction, keep P2 allocation.

        A compensated craft still uses asymmetric RPM to counter -r x F.
        Predicting those RPM with zero CG creates a false gyro innovation.
        Only the observer's torque map changes; P2 inertia and context stay fixed.
        """
        observer = copy.copy(vehicle)
        observer.rotor_map = vehicle.rotor_map.clone()
        force_map = observer.rotor_map[:3].T
        r = self.com_body.detach().to(force_map).expand_as(force_map)
        observer.rotor_map[3:] -= torch.linalg.cross(r, force_map).T
        return observer


def prepare_position(samples, nominal, *, controlled_external_loads, motor_response=None, rotor_axes_body=None):
    """Build offline body-frame equations; position and IMU refer to the same point.

    The position reference uses a noncausal .4 s local polynomial and is ONLY
    used during identification. Caller must attest negligible/controlled
    external loads. Wind or changing cable loads invalidate the CG equation.
    """
    if controlled_external_loads is not True:
        raise ValueError('CG fitting requires explicitly controlled external loads')
    time, force, gyro, rpm, throttle, before = _position_samples(samples)
    kinematics = _position_kinematics(samples, nominal, time, before)
    if kinematics is None:
        raise ValueError('Position aiding is required for accelerometer bias')
    nominal_drag_accel, reference_force = kinematics
    after = before + 1
    dt = (time[after]-time[before])[:, None]
    motor = nominal['actuators']
    target = np.where(throttle[after] >= motor['motor_deadzone'],
                      np.maximum(np.polyval(motor['motor_speed_polynomial'], throttle[after]), 0), 0)
    start, end = rpm[before], rpm[after]
    response_source = 'this_record' if motor_response is None else 'training_record'
    if motor_response is None:
        try:
            tau, gain, _, _ = _motor_fit(start, end, target, dt, motor['motor_tau_s'])
        except ValueError:
            motor_response = dict(identified=False)
        else:
            motor_response = dict(identified=True, tau=tau, gain=gain)
    if not motor_response['identified']:
        mean = (start+end)/2
        squared = (start**2+start*end+end**2)/3
        motor_model = 'measured_linear_interpolation'
    else:
        tau, gain = float(motor_response['tau']), float(motor_response['gain'])
        if not np.isfinite([tau, gain]).all() or min(tau, gain) <= 0:
            raise ValueError('Positive finite training motor response required')
        target = gain*target
        delta = start-target
        factor = -np.expm1(-dt/tau)/(dt/tau)
        mean = target+delta*factor
        squared = target**2+2*target*delta*factor+delta**2*(-np.expm1(-2*dt/tau)/(2*dt/tau))
        motor_model = 'measured_start_fitted_first_order_interval'
    rate = (gyro[before]+gyro[after])/2
    alpha = (gyro[after]-gyro[before])/dt
    torque = _rotor_wrench(nominal, squared, mean, rate)
    rotor_force = np.zeros_like(rate)
    rotor_force[:, 2] = -motor['kf']*squared.sum(axis=1)
    if rotor_axes_body is not None:
        from .installation_calibration import rotor_wrench
        rotor_force, torque = rotor_wrench(nominal, squared, mean, rate, rotor_axes_body)
    residual = force[after]-reference_force
    valid = np.isfinite(residual).all(axis=1)
    rows = np.flatnonzero(valid)
    # Integrate short blocks to reduce gyro differentiation noise. Never bridge
    # mode, timestamp, odometry-reset or invalid-position boundaries.
    runs = np.split(rows, np.flatnonzero(np.diff(before[rows]) != 1)+1)
    origin_force = rotor_force+nominal_drag_accel*nominal['airframe']['mass_kg']
    output = {name: [] for name in ('gyro', 'alpha', 'torque', 'rotor_force', 'bias_residual', 'origin_force', 'gyro_outer')}
    for run in runs:
        for first in range(0, len(run), 8):
            group = run[first:first+8]
            if len(group) < 4:
                continue
            weights = dt[group, 0]
            for key, value in zip(output, (rate, alpha, torque, rotor_force, residual, origin_force,
                                           rate[:, :, None]*rate[:, None, :])):
                output[key].append(np.average(value[group], axis=0, weights=weights))
    if len(output['gyro']) < 16:
        raise ValueError('Insufficient valid contiguous Position aiding')
    output = {key: np.asarray(value) for key, value in output.items()}
    output['nominal_inertia'] = np.asarray(nominal['airframe']['inertia_body_kgm2'])
    output['motor_response'] = motor_response
    output['metadata'] = dict(source='armed_airborne_position_body_FRD', motor_interval_model=motor_model,
                              raw_valid_intervals=int(valid.sum()), integrated_blocks=len(output['gyro']),
                              controlled_external_loads=True, position_reference='same_point_as_IMU',
                              motor_response_source=response_source)
    return output


def _tensors(data):
    names = ('gyro', 'alpha', 'rotor_force', 'torque', 'bias_residual')
    arrays = [np.asarray(data[key], dtype=float) for key in names]
    n = len(arrays[0])
    if n < 16 or any(x.shape != (n, 3) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Need finite body-frame calibration rows [N,3]')
    if not np.allclose(arrays[2][:, :2], 0) or (arrays[2][:, 2] >= 0).any():
        raise ValueError('Only parallel upward rotor force in FRD is supported')
    result = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in zip(names, arrays)}
    origin = np.asarray(data.get('origin_force', arrays[2]), dtype=float)
    if origin.shape != (n, 3) or not np.isfinite(origin).all():
        raise ValueError('Finite controlled-origin forces [N,3] required')
    result['origin_force'] = torch.as_tensor(origin, dtype=torch.float64)
    return result


def evaluate_calibration(model, data):
    values = _tensors(data)
    with torch.no_grad():
        error = model.momentum_torque(values['gyro'], values['alpha'], values['origin_force'])-values['torque']
        base = (model.nominal_inertia*values['alpha'] + torch.linalg.cross(
            values['gyro'], model.nominal_inertia*values['gyro'])-values['torque'])
        bias = values['bias_residual']
        rms = lambda x: x.square().mean(0).sqrt().tolist()
        return dict(samples=len(error), torque_before_rmse_nm=rms(base), torque_after_rmse_nm=rms(error),
                    accel_before_rmse_mps2=rms(bias), accel_after_rmse_mps2=rms(bias-model.bias))


def fit_calibration(data, *, steps=1600):
    """Adam trains five physical calibration weights and three nuisance inertias.

    No true bias, CG, actual inertia, target height or Actor is consumed.
    Matrix rank checks use parameter-scaled, inertia-normalized body equations.
    """
    values = _tensors(data)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Positive integer training steps required')
    model = BodyCalibration(data['nominal_inertia'])
    w, a, f = (values[key] for key in ('gyro', 'alpha', 'origin_force'))
    columns = []
    for axis in torch.eye(3, dtype=w.dtype):
        j = axis*model.nominal_inertia
        columns.append((j*a+torch.linalg.cross(w, j*w))/model.nominal_inertia)
    for axis in torch.eye(3, dtype=w.dtype)[:2]:
        columns.append(torch.linalg.cross((axis*.01).expand_as(f), f)/model.nominal_inertia)
    matrix = torch.stack(columns, dim=-1).reshape(-1, 5)
    singular = torch.linalg.svdvals(matrix)
    relative = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
    if relative < .001:
        raise ValueError('Insufficient body angular/collective excitation to separate CG and inertia')
    # Whitening the five physical directions avoids letting the large, nearly
    # constant CG moment drown out the small inertia response during Adam.
    inverse_design = torch.linalg.pinv(matrix)
    parameter_scale = w.new_tensor([.1, .1, .1, 1., 1.])
    optimizer = torch.optim.Adam([dict(params=[model.bias], lr=.015),
                                  dict(params=[model.com_xy], lr=.0005),
                                  dict(params=[model.log_inertia_scale], lr=.015)])
    initial = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torque_error = (model.momentum_torque(w, a, f)-values['torque'])/model.nominal_inertia
        parameter_error = inverse_design @ torque_error.reshape(-1)
        loss = (parameter_error/parameter_scale).square().mean() + ((model.bias-values['bias_residual'])/.1).square().mean()
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    if (not all(torch.isfinite(p).all() for p in model.parameters())
            or (model.bias.abs() > 1).any() or (model.com_xy.abs() > .03).any()
            or (model.log_inertia_scale.abs() > .4).any()
            or 2*model.inertia.max() > model.inertia.sum()):
        raise ValueError('Calibration outside validated physical bounds')
    model.eval()
    report = dict(schema_version=1, method='Position_body_FRD_physical_parameter_Adam',
                  steps=steps, initial_loss=initial, final_loss=float(loss.detach()),
                  accel_bias_body_mps2=model.bias.detach().tolist(),
                  com_body_m=model.com_body.detach().tolist(), com_z_identified=False,
                  inertia_body_kgm2=model.inertia.detach().tolist(),
                  design_relative_min_singular=relative, metrics=evaluate_calibration(model, data),
                  torque_sign='disturbance=-r_cross_F; compensation=+r_cross_F',
                  controlled_external_loads_required=True)
    return model, optimizer, report
