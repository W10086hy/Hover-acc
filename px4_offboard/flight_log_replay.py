"""Read-only real-flight analysis and recorded-input replay. Never sends commands."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .acceleration_runtime import AccelerationController
from .control import Sensors
from .network import Actor, current_features
from .policy_timing import FixedPolicyRunner
from .unified_p3 import dynamics_features, nominal_alpha


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rms(values, axis=0):
    return np.sqrt(np.mean(values**2, axis=axis))


def matrix(data, name, size):
    return np.column_stack([data[f'{name}[{i}]'] for i in range(size)])


def window(data, start, end):
    stamp = data['timestamp'] * 1e-6
    return (stamp >= start) & (stamp <= end)


def load(log):
    files = {p.name: p for folder in ('companion', 'flight_controller') for p in (log/folder).iterdir()}
    hashes = {}
    for entry in read(log/'download_manifest.json'):
        name = entry['local'].replace('\\', '/').rsplit('/', 1)[-1]
        path = files[name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Original log differs from download manifest: {name}')
        hashes[name] = digest
    lines = files['samples.jsonl'].read_text(encoding='utf-8').splitlines()
    rows, bad = [], []
    for number, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines):
                raise ValueError(f'Malformed JSON inside the sample stream at line {number}')
            bad.append(number)
    return rows, bad, hashes


def band_summary(time, values):
    dt = .01
    grid = np.arange(time[0], time[-1], dt)
    series = np.column_stack([np.interp(grid, time, column) for column in values.T])
    taper = np.hanning(len(grid))
    power = np.abs(np.fft.rfft((series-series.mean(axis=0))*taper[:, None], axis=0))**2
    freq = np.fft.rfftfreq(len(grid), dt)
    band = (freq >= 2.5) & (freq <= 4.5)
    peak = (freq >= .5) & (freq <= 15.)
    return dict(std_mps2=values.std(axis=0).tolist(), mean_mps2=values.mean(axis=0).tolist(),
        peak_0p5_to_15_hz=freq[peak][np.argmax(power[peak], axis=0)].tolist(),
        band_2p5_to_4p5_hz_rms_mps2=np.sqrt(2*power[band].sum(axis=0)/(len(grid)*(taper@taper))).tolist())


def recorded_actor(stamp, observations, cfg, weights):
    actor = Actor(weights, cfg['stage']['model'])
    ctx = np.asarray(cfg['external_nominal_context'], np.float32)
    limits = np.asarray(cfg['stage']['controller']['acceleration']['limits_mps2'])
    command = np.zeros(3)
    alpha = -np.expm1(-.01/.065)
    output = np.empty((len(stamp), 3))
    for i, obs in enumerate(observations):
        if i and not 0 < stamp[i] - stamp[i-1] <= .05:
            actor.hidden.fill(0.)
            command.fill(0.)
        command += alpha*(np.tanh(actor(obs, ctx))*limits-command)
        output[i] = command
    return output


def reconstruct_packets(rows, stamp, obs, protocol, cfg):
    # NX logs post-filter features. Invert its documented IIRs within each
    # contiguous source interval; the first sample initializes each filter.
    filtered_force = obs[:, 7:10].astype(float)*20+protocol['accel_offset_mps2']
    target = np.array([r['inference_info']['hold_height_m'] for r in rows])
    filtered_height = target-2*obs[:, 18]
    force, height = filtered_force.copy(), filtered_height.copy()
    valid = (np.diff(stamp) > 0) & (np.diff(stamp) <= .05)
    i = np.flatnonzero(valid)+1
    force[i] = 2*filtered_force[i]-filtered_force[i-1]
    height[i] = (filtered_height[i]-.85*filtered_height[i-1])/.15
    vmax = sum(cfg['nominal']['actuators']['motor_speed_polynomial'])
    return [Sensors(q[:4].astype(float), q[4:7].astype(float)*4, f, q[10:14].astype(float)*vmax, h)
            for q, f, h in zip(obs, force, height)]


def candidate_replay(stamp, packets, applied, cfg, weights, protocol, tau):
    cfg = copy.deepcopy(cfg)
    cfg['deployment'] = dict(acceleration_filter_s=.065, innovation_filter_s=tau)
    ctl = AccelerationController(cfg, weights)
    ctl.accel_offset = np.asarray(protocol['accel_offset_mps2'])
    ctl.observer.target_height_m = packets[0].baro_height
    runner = FixedPolicyRunner(ctl)
    sent, ticks = [], []
    for t, packet, u in zip(stamp, packets, applied):
        fresh = runner.advance(packet, u, t)
        if fresh:
            ticks.extend(fresh)
            sent.append([t, *fresh[-1]['wire_acceleration'][:2]])
    sent = np.array(sent)
    before = np.array([r['innovation_before_radps'] for r in ticks])
    used = np.array([r['innovation_to_actor_radps'] for r in ticks])
    report = band_summary(sent[:, 0], sent[:, 1:])
    report.update(policy_ticks=len(ticks), emitted_updates=len(sent), innovation_filter_s=tau,
        pre_filter_innovation_rms_radps=rms(before).tolist(), actor_innovation_rms_radps=rms(used).tolist(),
        innovation_clip_fraction=np.mean(np.abs(used) >= .499999, axis=0).tolist())
    return report


def nav_transitions(data):
    nav = data['nav_state']
    return [dict(time_s=float(data['timestamp'][i]*1e-6), state=int(nav[i]), failsafe=bool(data['failsafe'][i]))
            for i in np.flatnonzero(np.r_[True, np.diff(nav) != 0])]


def wire_match(topic, stamp, commands, heading, start, end):
    fit = window(topic, start+.05, end-.05)
    recv = matrix(topic, 'acceleration', 3)[fit, :2]
    t = topic['timestamp'][fit]*1e-6
    c, s = np.cos(heading), np.sin(heading)
    target = commands[:, :2] @ np.array([[c, s], [-s, c]])
    def error(lag):
        predicted = np.column_stack([np.interp(t-lag, stamp, col) for col in target.T])
        return rms(predicted-recv, axis=None)
    rmse, offset = min((error(lag), lag) for lag in np.arange(-.05, .201, .005))
    return dict(samples=len(t), rmse_mps2=float(rmse), best_time_shift_s=float(offset))


def analyze(log, assets, tau):
    from pyulog import ULog

    rows, bad, hashes = load(log)
    protocol = read(log/'companion/protocol.json')
    cfg = read(assets/'config.json')
    if protocol['checkpoint_sha256'] != cfg['controller_sha256']:
        raise ValueError('Recorded checkpoint does not match current P2')
    weights_sha = hashlib.sha256((assets/'weights.npz').read_bytes()).hexdigest()
    export = read(assets/'export.json')
    if weights_sha != export['artifact_sha256']['weights.npz']:
        raise ValueError('Actor weights differ from the frozen NPZ export')
    with np.load(assets/'weights.npz') as source:
        weights = dict(source)
    ulog = ULog(str(log/'flight_controller/px4_log_007.ulg'))
    if ulog.file_corruption:
        raise ValueError('PX4 ULog corruption detected')
    data = {d.name: d.data for d in ulog.data_list if d.multi_id == 0}
    stamp = np.array([r['imu_us'] for r in rows])*1e-6
    obs = np.array([r['observation'] for r in rows], dtype=np.float32)
    logged = np.array([r['candidate_acceleration_mps2'] for r in rows])
    replayed = recorded_actor(stamp, obs, cfg, weights)
    packets = reconstruct_packets(rows, stamp, obs, protocol, cfg)
    applied = (obs[:, 14:18].astype(float)+1)/2
    rpm = np.stack([p.motor_speed for p in packets])
    gyro = np.stack([p.gyro for p in packets])
    force = np.stack([p.specific_force for p in packets])
    nominal = cfg['nominal']
    _, square, mean = dynamics_features(gyro[:-1], force[1:], rpm[:-1], rpm[1:], applied[1:], .01, nominal)
    delta = np.vstack((np.zeros(3), np.diff(gyro, axis=0)-.01*nominal_alpha(gyro[:-1], square, mean, nominal)))
    model = cfg['stage']['model']
    angular_features = np.stack([current_features(o, model['gravity_mps2'], model['proxy_scale'])[11:14] for o in obs])
    head_angular = angular_features@weights['actor.fast_head.weight'][:, 11:14].T
    active = np.array([r['network_execution'] for r in rows])
    edges = np.diff(np.r_[False, active, False].astype(int))
    status, pose, battery, traj = (data[key] for key in
        ('vehicle_status','vehicle_local_position','battery_status','trajectory_setpoint'))
    switches = data['manual_control_switches']
    warning = battery['warning']
    cuts = np.flatnonzero(np.r_[True, np.diff(warning) != 0])
    warnings = [dict(time_s=float(battery['timestamp'][i]*1e-6), warning=int(warning[i]),
        voltage_v=float(battery['voltage_v'][i]), remaining=float(battery['remaining'][i])) for i in cuts]
    intervals = []
    for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        t = stamp[i:j]
        start, end = t[0], t[-1]
        still = window(pose, start, end)
        z = rows[i]['target_local_z_ned_m']
        origin = np.array([np.interp(start, pose['timestamp']*1e-6, pose[k]) for k in ('x','y')])
        horizontal = np.column_stack([pose[k][still] for k in ('x','y')])-origin
        vertical = pose['z'][still]-z
        xy_sq = np.sum(horizontal**2, axis=1)
        mask_battery = window(battery, start, end)
        motor_lag = np.arange(0., .081, .002)
        targets = np.maximum(np.polyval(nominal['actuators']['motor_speed_polynomial'], applied[i:j]), 0.)
        def best_lag(controls, speeds):
            corr, lag = max((np.corrcoef(np.interp(t-d, t, controls), speeds)[0, 1], d) for d in motor_lag)
            return dict(lag_s=float(lag), correlation=float(corr))
        base = candidate_replay(t, packets[i:j], applied[i:j], cfg, weights, protocol, 0.)
        candidate = candidate_replay(t, packets[i:j], applied[i:j], cfg, weights, protocol, tau)
        old = band_summary(t, logged[i:j, :2])
        err = np.max(np.abs(replayed[i:j]-logged[i:j]))
        if err >= 1e-6:
            raise ValueError(f'Actor replay mismatch in active interval: {err}')
        residual = np.array([r['inference_info']['P3_aggregate_residual'] for r in rows[i:j]])
        rates = data['vehicle_angular_velocity']
        selected = window(rates, start, end)
        measured_rates = matrix(rates, 'xyz', 3)[selected]
        rate_stamp = rates['timestamp'][selected]*1e-6
        rate_targets = data['vehicle_rates_setpoint']
        rate_target_stamp = rate_targets['timestamp']*1e-6
        rate_lag = []
        for axis, channel in enumerate(('roll','pitch')):
            correlation, lag = max((np.corrcoef(
                np.interp(rate_stamp-lag, rate_target_stamp, rate_targets[channel]), measured_rates[:, axis]
            )[0, 1], lag) for lag in np.arange(0, .151, .005))
            rate_lag.append(dict(axis=channel, lag_s=float(lag), correlation=float(correlation)))
        sensor = data['sensor_combined']
        preposition = window(sensor, start-10, start-.1)
        flags = data['failsafe_flags']
        active_flags = window(flags, start+.1, end)
        intervals.append(dict(start_fc_s=float(start), end_fc_s=float(end), duration_s=float(end-start),
            samples=int(j-i), average_inference_hz=float((j-i-1)/(end-start)),
            largest_sample_gap_ms=float(1000*np.diff(t).max()), target_local_z_ned_m=z,
            local_position_estimated_rmse_3d_h_xy_m=np.sqrt([np.mean(xy_sq+vertical**2),
                np.mean(vertical**2),np.mean(xy_sq)]).tolist(),
            local_position_max_xy_m=float(np.sqrt(xy_sq.max())), last_below_target_m=float(vertical[-1]),
            target_speed_to_rpm_effective_lag=[best_lag(u, r) for u, r in zip(targets.T, rpm[i:j].T)],
            battery_min_v=float(battery['voltage_v'][mask_battery].min()),
            battery_max_warning=int(warning[mask_battery].max()),
            battery_estimated_remaining_range=[float(battery['remaining'][mask_battery].min()),
                float(battery['remaining'][mask_battery].max())],
            failsafe_flags_active={key:bool(flags[key][active_flags].any()) for key in
                ('offboard_control_signal_lost','local_position_invalid','local_altitude_invalid')},
            gyro_rms_radps=rms(measured_rates).tolist(),
            gyro_peak_0p5_to_15_hz=band_summary(rate_stamp,measured_rates)['peak_0p5_to_15_hz'],
            px4_rate_setpoint_to_gyro_effective_lag=rate_lag,
            pre_takeover_fc_accel_std_mps2=matrix(sensor,'accelerometer_m_s2',3)[preposition].std(0).tolist(),
            legacy_model_innovation_rms_radps=rms(delta[i:j]).tolist(),
            legacy_model_innovation_exceeds_clip_fraction=np.mean(np.abs(delta[i:j]) > .5, axis=0).tolist(),
            actor_angular_fast_head_std=head_angular[i:j].std(0).tolist(),
            logged_aggregate_angular_rms_radps=rms(residual[:, 3:]).tolist(),
            received_by_px4=wire_match(traj, t, logged[i:j], protocol['launch_heading_rad'], start, end),
            logged_command=old, recorded_actor_max_error_mps2=float(err), fixed_no_P3=base,
            fixed_no_P3_with_innovation_filter=candidate))
    return dict(source_hashes=hashes, logged_samples=len(rows), malformed_final_lines=bad,
        result_json_bytes=(log/'companion/result.json').stat().st_size, PX4_firmware=ulog.msg_info_dict['ver_sw'],
        P2_checkpoint_sha256=protocol['checkpoint_sha256'], weights_sha256=weights_sha,
        reported_real_mass_kg=protocol['actual_mass_kg'], mass_provenance=protocol['aircraft_mass_confirmation'],
        FC_initial_parameters={n:ulog.initial_parameters[n] for n in
            ('THR_MDL_FAC','MC_PITCHRATE_K','COM_LOW_BAT_ACT','BAT1_CAPACITY','BAT1_N_CELLS')},
        logged_P3_scope=protocol['P3_scope'], logged_actuator_mapping=protocol['motor_mapping']['policy_to_fc_motor_indices'],
        physical_wiring_verified=protocol['motor_mapping']['physical_wiring_verified'],
        mode_transitions=nav_transitions(status), battery_warning_transitions=warnings,
        manual_offboard_switches=[dict(time_s=float(switches['timestamp'][k]*1e-6),
            value=int(switches['offboard_switch'][k]))
            for k in np.flatnonzero(np.r_[True,np.diff(switches['offboard_switch']) != 0])],
        intervals=intervals, replay_uses_reconstructed_sources=True,
        replay_replaces_logged_P3_with_no_P3=True, replay_closed_loop=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--assets', type=Path, default=Path('px4_offboard/acceleration_dual/assets'))
    p.add_argument('--innovation-filter-s', type=float, default=.2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    log, output = a.log.resolve(), a.output.resolve()
    if not output.is_relative_to(Path.cwd().resolve()) or output.is_relative_to(log):
        p.error('Analysis output must be in the current project, separate from original logs')
    report = analyze(log, a.assets, a.innovation_filter_s)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
