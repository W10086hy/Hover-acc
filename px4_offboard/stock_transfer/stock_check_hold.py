import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from pymavlink import mavutil
from px4_offboard.motor_mapping import PX4_MOTORS_IN_P2_ORDER, px4_to_p2_motor_order
from px4_offboard.network import rotation
from px4_offboard.offboard import Offboard
from px4_offboard.physics import Plant

BASE = Path.home() / 'hover_bptt_px4/actor37_5m_20261109'
ASSETS = BASE / 'px4_offboard/actor37_5m/assets'
PX4 = Path.home() / 'PX4-Autopilot'
BUILD = PX4 / 'build/px4_sitl_default'
DT = .004


def mode(heartbeat):
    return (heartbeat.custom_mode >> 16) & 255 if heartbeat else 0


def control_message(conn, clock, plant, fixture, rng, step, site_pressure_altitude):
    q = plant.x[6:10]
    force = np.array([0., 0., -plant.gravity]) if fixture else plant.specific_force()
    gyro = np.zeros(3) if fixture else plant.x[10:13]
    height = -plant.x[2]
    accel = force + rng.normal(0, .01, 3)
    rate = gyro + rng.normal(0, .0005, 3)
    mag = rotation(q).T @ np.array([.215, 0., .427])
    time_usec = round(clock*1e6)
    if step % 25 == 0:
        x, y, z = plant.x[:3]
        vx, vy, vz = plant.x[3:6]
        conn.mav.hil_gps_send(time_usec, 3,
            round((47.397742 + x/111320.)*1e7),
            round((8.545594 + y/(111320.*math.cos(math.radians(47.397742))))*1e7),
            round(-z*1000), 30, 45, round(np.linalg.norm([vx, vy])*100),
            round(vx*100), round(vy*100), round(vz*100), 65535, 12)
    pressure = 1013.25*(1-(site_pressure_altitude + height + rng.normal(0.,.01))/44330.)**5.255
    conn.mav.hil_sensor_send(time_usec, *accel, *rate, *mag,
        pressure, 0., height, 20., 63 | (64|128|256|512|2048|4096 if step%5==0 else 0))
    return time_usec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--keep-position', action='store_true')
    residual = parser.add_mutually_exclusive_group()
    residual.add_argument('--no-residual', action='store_true')
    residual.add_argument('--residual', action='store_true')
    parser.add_argument('--assets', type=Path, default=ASSETS)
    parser.add_argument('--motor-tau', type=float)
    parser.add_argument('--seconds', type=float, choices=(15., 30., 120.), default=15.)
    parser.add_argument('--sensor-seed', type=int, default=21683134)
    parser.add_argument('--wind', type=float, default=0.)
    parser.add_argument('--wind-ramp-s', type=float, default=0.,
                        help='After Offboard entry, smoothly raise wind from 1 m/s to --wind over this duration')
    parser.add_argument('--site-pressure-hpa', type=float, default=1013.25,
                        help='Simulated barometer pressure at the takeoff plane')
    args = parser.parse_args()
    if not math.isfinite(args.wind) or args.wind < 0:
        parser.error('--wind must be finite and nonnegative')
    if not math.isfinite(args.wind_ramp_s) or not 0 <= args.wind_ramp_s < args.seconds:
        parser.error('--wind-ramp-s must be nonnegative and shorter than --seconds')
    if args.wind_ramp_s and (args.keep_position or args.wind <= 1):
        parser.error('A wind ramp requires Offboard and a target wind above 1 m/s')
    if not math.isfinite(args.site_pressure_hpa) or args.site_pressure_hpa <= 0:
        parser.error('--site-pressure-hpa must be finite and positive')
    site_pressure_altitude = 44330.*(1.-(args.site_pressure_hpa/1013.25)**(1./5.255))
    root = args.output.resolve()
    root.mkdir()
    log = (root/'px4.log').open('w')
    env = dict(os.environ, PX4_SIM_MODEL='none_iris', HEADLESS='1')
    env.pop('PX4_SYS_AUTOSTART', None)
    env.pop('PX4_SIMULATOR', None)
    conn = mavutil.mavlink_connection('tcpin:127.0.0.1:4560', source_system=1, source_component=200)
    link = mavutil.mavlink_connection('udpin:127.0.0.1:14540', source_system=42, source_component=191)
    proc = subprocess.Popen([str(BUILD/'bin/px4'), '-i', '0', '-s',
        str(BUILD/'etc/init.d-posix/rcS'), str(BUILD/'etc')], cwd=root,
        env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    plant = Plant(args.assets, 0)
    cfg = plant.cfg
    if args.motor_tau is not None:
        if not math.isfinite(args.motor_tau) or args.motor_tau <= 0:
            raise ValueError('Motor response time must be finite and positive')
        plant.u['motor_tau_s'] = args.motor_tau
    use_residual = args.residual or (not args.no_residual and cfg.get('residual_default_enabled', True))
    off = Offboard(link, cfg, np.load(args.assets/'weights.npz'), residual=use_residual,
                   reference_sample_height_m=plant.target_height)
    poly = np.array(cfg['nominal']['actuators']['motor_speed_polynomial'])
    vmax = poly.sum()
    rng = np.random.default_rng(args.sensor_seed)
    wind = np.array([0., 1. if args.wind_ramp_s else args.wind, 0.])
    applied = np.zeros(4)
    controls = np.zeros(4)
    physical = np.zeros(4)
    released = None
    position_stable_since = None
    requested = None
    switch = None
    error = None
    rows = []
    policy_ticks = []
    policy_heights = []
    wire = None
    stream_requested = False
    receive_counts = {}
    duration = None
    metadata = dict(PX4_source_commit=subprocess.check_output(
        ['git','-C',str(PX4),'rev-parse','HEAD'],text=True).strip(),
        stock_rcS=True, stock_airframe='10016_none_iris', simulation_motor_order=list(PX4_MOTORS_IN_P2_ORDER),
        GPS_for_PX4_position_reference=True,GPS_policy_input=False,
        parameters_overridden=False, policy_updates=0,residual_updates=0,
        simulated_motor_tau_s=plant.u['motor_tau_s'],
        site_pressure_hpa=args.site_pressure_hpa,
        stock_thrust_conversion=off.controller.settings.get('stock_px4') is not None,
        sensor_seed=args.sensor_seed, requested_post_duration_s=args.seconds,
        wind_mps=args.wind, wind_ramp_s=args.wind_ramp_s,
        hold_position=args.keep_position, residual_enabled=use_residual,
        controller_sha256=cfg['controller_sha256'],residual_sha256=cfg['residual_sha256'])
    try:
        for k in range(round((80 + args.seconds) / DT)):
            clock = 1+k*DT
            if proc.poll() is not None:
                raise RuntimeError(f'PX4 exited {proc.returncode}')
            off.pump(clock)
            h = off.last_heartbeat
            if h is not None and not stream_requested:
                for message, hz in [(31,100),(105,100),(30,50)]:
                    off.command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, message, 1e6/hz)
                off.request_height_telemetry()
                stream_requested = True
            armed = bool(h and h.base_mode & 128)
            current = mode(h)
            if k%250==0:
                off.heartbeat()
                print(f't={clock:.1f} mode={current} armed={armed} pwm={controls.mean():.3f} '
                      f'h={-plant.x[2]:.3f} vz={plant.x[5]:.3f} release={released} switch={switch}',flush=True)
                if h and clock >= 14 and current != 3 and requested is None and released is None:
                    off.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,1.,3.,0.)
                if h and current == 3 and not armed and released is None and clock>=15:
                    if off.takeoff_baro_zero_m is not None:
                        off.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1.,0.)
                    elif clock>=30:
                        raise RuntimeError('Takeoff barometric reference unavailable before arming')
            if k%5==0 and h:
                height = -plant.x[2]
                target_error = 5.0 - height
                if not armed:
                    manual_throttle = 0
                elif released is None:
                    manual_throttle = 720
                elif requested is not None or abs(target_error - .45*plant.x[5]*-1) < .10:
                    manual_throttle = 500
                else:
                    predicted_error = target_error + .45*plant.x[5]
                    manual_throttle = 500 + round(np.sign(predicted_error)*(65 + min(110, 75*abs(predicted_error))))
                link.mav.manual_control_send(1,0,0,manual_throttle,0,0)

            measured = plant.x[13:17] + rng.normal(0, 1., 4)
            off.record_motor_sample(measured, clock)
            if released is not None:
                speed = np.linalg.norm(plant.x[3:6])
                roll_pitch = math.acos(np.clip(rotation(plant.x[6:10])[2,2],-1,1))
                steady = current==3 and armed and speed<.18 and abs(-plant.x[2]-5.)<.20 and roll_pitch<math.radians(8)
                if requested is None:
                    position_stable_since = (position_stable_since or clock) if steady else None
                if requested is None and position_stable_since is not None and clock-position_stable_since>=3.:
                    requested = clock
                if requested is not None:
                    if args.keep_position:
                        switch = switch or requested+2.
                    elif current==6 and switch is None:
                        switch = clock
                    if not args.keep_position and switch is None and k%5==0:
                        off.setpoint(np.zeros(3),float(np.mean(physical)),clock)
                        if clock-requested>=2. and (k%125==0):
                            off.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,1.,6.,0.)
                    if not args.keep_position and switch is None and clock-requested>8.:
                        raise RuntimeError('stock PX4 refused Offboard mode')
                    if switch is not None and clock>=switch and not args.keep_position and (2*k)//5 != (2*(k-1))//5:
                        imu = off.messages.get('HIGHRES_IMU')
                        # At most one policy update per new IMU packet; do not reuse stale IMU timestamps.
                        if imu is not None and (off.previous_gyro_time is None or imu.time_usec/1e6>off.previous_gyro_time):
                            feedback = off.update(measured,applied,clock)
                            if feedback is not None:
                                packet, rate, thrust, detail = feedback
                                wire = [*rate,thrust]
                                policy_ticks.append(clock)
                                policy_heights.append(packet.baro_height)
                    if switch is not None and clock-switch>=args.seconds:
                        duration = clock-switch
                        break
            stamp = control_message(conn,clock,plant,released is None,rng,k,site_pressure_altitude)
            deadline = time.monotonic()+(1. if k>20 else .1)
            got = False
            while time.monotonic()<deadline:
                msg = conn.recv_match(blocking=False)
                off.pump(clock)
                if msg is None:
                    time.sleep(.0001)
                    continue
                kind = msg.get_type()
                receive_counts[kind] = receive_counts.get(kind,0)+1
                if kind=='HIL_ACTUATOR_CONTROLS' and msg.time_usec >= stamp-1:
                    physical = np.asarray(msg.controls[:4],dtype=float)
                    controls = px4_to_p2_motor_order(physical)
                    got = True
                    break
            if not got and k>50:
                raise RuntimeError(f'Lockstep actuator timeout at {clock:.3f}')
            nominal_speed = np.where(controls>0,np.maximum(0,np.polyval(poly,controls)),0)
            if released is None:
                actual_speed=np.maximum(0,np.polyval(cfg['actual']['actuators']['motor_speed_polynomial'],controls))
                if current==3 and armed and plant.u['kf']*np.sum(actual_speed**2)>=.98*plant.mass*plant.gravity:
                    plant.x[13:17] = actual_speed
                    released = clock
                    print('POSITION plant release',clock,flush=True)
            if released is not None:
                if args.wind_ramp_s and switch is not None:
                    fraction = min(1., (clock - switch) / args.wind_ramp_s)
                    wind[1] = 1. + (args.wind - 1.) * .5 * (1. - math.cos(math.pi * fraction))
                applied=plant.advance(nominal_speed/vmax,wind,period=DT)
                tilt=math.degrees(math.acos(np.clip(rotation(plant.x[6:10])[2,2],-1,1)))
                rope_nodes = plant.x[19:19 + 3*plant.free]
                rows.append([clock,current,int(armed),*plant.x[:17],*physical,*applied,
                             *(wire or [float('nan')]*4),tilt,plant.peak,*rope_nodes])
                if tilt>45. or not (3.<-plant.x[2]<7.) or plant.peak>30.:
                    raise RuntimeError(f'Physical guard at t={clock:.3f}: tilt={tilt:.2f}, height={-plant.x[2]:.3f}, tension={plant.peak:.2f}')
                if switch is not None and not args.keep_position and (current!=6 or not armed):
                    raise RuntimeError(f'Offboard or armed lost at t={clock:.3f}')
            if released is None and clock>48.:
                raise RuntimeError('stock Position did not reach support thrust for release')
            if released is not None and requested is None and clock-released>48.:
                raise RuntimeError('stock Position did not achieve stable flight')
        else:
            raise RuntimeError('Simulation time limit reached before the requested evaluation completed')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        metadata.update(release_s=released,stable_position_since_s=position_stable_since,
            prestream_s=requested,switch_heartbeat_s=switch,post_duration_s=duration,
            error=error,policy_ticks=policy_ticks,observed_wire_messages=receive_counts,acks=off.acks,events=off.events,
            last_received_times=off.received, last_gyro_timestamp=off.previous_gyro_time,
            takeoff_baro_zero_m=off.takeoff_baro_zero_m, takeoff_baro_locked_s=off.takeoff_baro_locked_s,
            policy_baro_heights_m=policy_heights)
        if rows:
            np.save(root/'trace.npy',np.asarray(rows))
        (root/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({key:value for key,value in metadata.items()
                          if key not in ('policy_ticks', 'policy_baro_heights_m')}),flush=True)
        try:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
        conn.close()
        link.close()
        log.close()


if __name__=='__main__':
    main()
