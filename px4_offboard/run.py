"""Real PX4 SITL <-> MAVLink <-> MuJoCo closed loop, running in the Linux VM."""
import argparse,collections,hashlib,json,math,os,signal,subprocess,time,traceback,zipfile
from pathlib import Path
import numpy as np
from pymavlink import mavutil
from .motor_mapping import px4_to_p2_motor_order
from .physics import Plant
from .network import rotation
from .offboard import Offboard
from .px4_startup import startup
from .wind import make_wind

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def run(args):
    if not all(math.isfinite(v) for v in [args.duration,args.preflight,args.cart_speed,args.wind_peak]):raise ValueError('Finite run settings required')
    if args.duration<=0 or not math.isclose(args.duration*10,round(args.duration*10)) or args.preflight<10 or not math.isclose(args.preflight*10,round(args.preflight*10)):
        raise ValueError('Positive duration and >=10s fixture, both multiples of0.1s required')
    if args.cart_speed<0 or args.wind_peak<0:raise ValueError('Nonnegative cart speed/wind required')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    assets=Path(args.assets).resolve();plant=Plant(assets,args.cart_speed);cfg=plant.cfg
    cfg['protocol']['cart_speed_mps']=args.cart_speed
    cfg['protocol']['duration_s']=args.duration
    cfg['protocol']['case_specs'][0]['cart_speed_mps']=args.cart_speed
    target_height=plant.target_height
    cfg['protocol']['wind_peak_mps']=args.wind_peak
    cfg['protocol']['wind_seed']=args.seed
    wind_profile=getattr(args,'wind_profile',None) or cfg['protocol'].get('wind_profile','varying')
    cfg['protocol']['wind_profile']=wind_profile
    wind_at=make_wind(args.wind_peak,args.seed,wind_profile,cfg['protocol'].get('wind_mean_direction_deg',90.))
    rc,settings=startup(cfg);(output/'rcS').write_text(rc);write(output/'px4_settings.json',settings)
    px4root=Path(args.px4).resolve();binary=px4root/'build/px4_sitl_default/bin/px4';etc=px4root/'build/px4_sitl_default/etc'
    metadata=dict(config=cfg,duration_s=args.duration,cart_speed_mps=args.cart_speed,wind_peak_mps=args.wind_peak,
        wind_seed=args.seed,sensor_seed=args.seed+1000003,physics_dt_s=plant.dt,sensor_period_s=.0025,
        offboard_period_s=.01,PX4_source_commit=subprocess.check_output(['git','-C',str(px4root),'rev-parse','HEAD'],text=True).strip(),
        PX4_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),MuJoCo=__import__('mujoco').__version__,
        numpy=np.__version__,pymavlink=__import__('pymavlink').__version__,
        GPS_sent=False,vision_sent=False,position_controller=False,truth_attitude_injected=False,
        policy_updates=0,residual_updates=0,residual_enabled=not args.no_residual,wind_profile=wind_profile,
        residual_transfer='Fz thrust correction and approximate equivalent P-rate torque offset; per-motor inverse-lag omitted',
        initialization='20s fixed-pose virtual hover fixture for PX4 sensor alignment; excluded from flight score; no takeoff claim',
        controller_inputs='PX4 estimated attitude/rates,HIGHRES_IMU,takeoff-relative ALTITUDE.altitude_monotonic,simulated measured RPM,previous applied ESC throttle,external nominal parameters')
    metadata['angular_prediction_mode']='Causal measured400Hz RPM moments over successive PX4 HIGHRES_IMU gyro timestamps'
    source_files=list(Path(__file__).parent.glob('*.py'))
    metadata['source_sha256']={'px4_offboard/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    metadata['asset_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}
    metadata['Python']=__import__('sys').version
    with zipfile.ZipFile(output/'frozen_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in source_files:archive.write(p,'px4_offboard/'+p.name)
    write(output/'protocol.json',metadata)
    log=(output/'px4.log').open('w');env=os.environ.copy();env['PX4_SIM_MODEL']='none_iris'
    process=subprocess.Popen([str(binary),'-i','0','-s',str(output/'rcS'),str(etc)],cwd=output,env=env,
        stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    connection=None;link=None;began=time.monotonic();last_progress=began
    records=[];truth=[];times=[];wind_rows=[];rpm_rows=[];action_rows=[];px4_rows=[];obs_rows=[];modes=[];ff_rows=[];truth_control=[]
    peaks=dict(tilt_deg=0.,body_rate_radps=0.,height_error_m=0.,tension_n=0.)
    failure=None;failure_s=None;counts=collections.Counter();latencies=[];controls=np.zeros(4);hil_armed=False;first_hil=None;previous_applied=np.zeros(4)
    try:
        connection=mavutil.mavlink_connection('tcpin:127.0.0.1:4560',source_system=1,source_component=200)
        link=mavutil.mavlink_connection('udpin:127.0.0.1:14540',source_system=42,source_component=191)
        # Register the client immediately instead of relying on default broadcast.
        link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,4)
        off=Offboard(link,cfg,np.load(assets/'weights.npz'),residual=not args.no_residual,
                     reference_sample_height_m=target_height,hold_height_on_entry=False)
        rng=np.random.default_rng(args.seed+1000003)
        dt=.0025;preflight_steps=round(args.preflight/dt);flight_steps=round(args.duration/dt)
        force_bias=rng.normal(0,.003,3);gyro_bias=rng.normal(0,.0003,3)
        last_pressure=1013.25*(1-target_height/44330.)**5.255
        for k in range(preflight_steps+flight_steps):
            fixture=k<preflight_steps;flight_t=max(0.,(k-preflight_steps)*dt);clock=1.+k*dt
            if process.poll() is not None:raise RuntimeError(f'PX4 exited {process.returncode}')
            off.pump(clock)
            measured_rpm=plant.x[13:17]+rng.normal(0,1.,4)
            off.record_motor_sample(measured_rpm,clock)
            if k%400==0:off.heartbeat()
            if fixture:
                if k%4==0:off.setpoint(np.zeros(3),off.controller.hover/off.controller.max_total,clock)
                if (k>=preflight_steps-1600 and k%400==0 and off.takeoff_baro_zero_m is not None
                        and not off.is_offboard_armed()):off.mode_arm()
                specific=np.array([0.,0.,-plant.gravity]);gyro=np.zeros(3);q=np.array([1.,0.,0.,0.]);height=target_height
            else:
                if k==preflight_steps:
                    if off.takeoff_baro_zero_m is None:raise RuntimeError('Takeoff barometric reference unavailable at release')
                    if not off.is_offboard_armed() or not hil_armed:raise RuntimeError('PX4 did not enter armed Offboard normally during fixture')
                    plant.x[13:17]=controls*sum(cfg['nominal']['actuators']['motor_speed_polynomial'])
                    previous_applied=off.controller.observer.throttle(plant.x[13:17])
                specific=plant.specific_force() if failure is None else np.array([0.,0.,-plant.gravity])
                gyro=plant.x[10:13];q=plant.x[6:10];height=-plant.x[2]
                if k%4==0 and failure is None:
                    result=off.update(measured_rpm,previous_applied,clock)
                    if result is None:raise RuntimeError('Required PX4 onboard telemetry unavailable at release')
            accel=specific+force_bias+rng.normal(0,.04,3);rate=gyro+gyro_bias+rng.normal(0,.003,3)
            mag=rotation(q).T@np.array([.215,0.,.427])+rng.normal(0,.0001,3)
            fields=63
            if k%20==0:
                noisy_height=height+rng.normal(0,.02);last_pressure=1013.25*(1-noisy_height/44330.)**5.255
                fields|=64|128|256|512|2048|4096
            sensor_us=round(clock*1e6)
            connection.mav.hil_sensor_send(sensor_us,*accel,*rate,*mag,last_pressure,0.,height,20.,fields)
            counts['HIL_SENSOR']+=1
            # HIL_SENSOR drives PX4 lockstep time. Wait for the corresponding
            # actual PX4 actuator output, while continuing to service MAVLink.
            deadline=time.monotonic()+(2. if first_hil is not None else .02);got=False;sent=time.monotonic()
            while time.monotonic()<deadline:
                msg=connection.recv_match(blocking=False)
                off.pump(clock)
                if msg is None:
                    time.sleep(.00015);continue
                kind=msg.get_type();counts[kind]+=1
                if kind=='HIL_ACTUATOR_CONTROLS':
                    controls = px4_to_p2_motor_order(msg.controls)
                    hil_armed = bool(msg.mode & 128)
                    if first_hil is None:first_hil=dict(time_usec=msg.time_usec,flags=msg.flags,mode=msg.mode)
                    if msg.time_usec>=sensor_us:
                        got=True;latencies.append(time.monotonic()-sent);break
            if not got and first_hil is not None:raise RuntimeError(f'PX4 lockstep actuator timeout at {clock:.4f}s')
            if not fixture:
                wind=wind_at(flight_t)
                if failure is None:
                    previous_applied=plant.advance(controls,wind,period=dt)
                    x=plant.x;anchor,_=plant.cart();error=x[:3]-np.r_[anchor[:2],-target_height]
                    r=rotation(x[6:10]);tilt=np.rad2deg(np.arccos(np.clip(r[2,2],-1,1)))
                    for name,val in [('tilt_deg',tilt),('body_rate_radps',np.linalg.norm(x[10:13])),('height_error_m',abs(-x[2]-target_height)),('tension_n',plant.peak)]:peaks[name]=max(peaks[name],float(val))
                    conditions=[not np.isfinite(x).all(),tilt>45,np.linalg.norm(x[10:13])>4,abs(-x[2]-target_height)>4,plant.peak>30,not hil_armed,not off.is_offboard_armed()]
                    if any(conditions):failure=int(sum(int(c)<<i for i,c in enumerate(conditions)));failure_s=flight_t+dt
                else:error=np.array([6.,0.,4.]);wind=wind_at(flight_t)
                if k%4==3:
                    records.append([*error,plant.peak,hil_armed,float(off.is_offboard_armed())]);wind_rows.append(wind.copy())
                    truth_control.append(plant.x[:13].copy())
                    if off.latest:
                        packet,rates,thrust,info=off.latest
                        action_rows.append([*rates,thrust,*controls]);px4_rows.append([*packet.attitude,*packet.gyro,*packet.specific_force,packet.baro_height])
                        rpm_rows.append(plant.x[13:17].copy());obs_rows.append(info['observation'].copy());ff_rows.append(info['correction'].copy())
                    modes.append([clock,int(off.last_heartbeat.base_mode) if off.last_heartbeat else 0,int(off.last_heartbeat.custom_mode) if off.last_heartbeat else 0])
                if (k-preflight_steps)%40==39:
                    truth.append(plant.x.copy());times.append(flight_t+dt)
            if time.monotonic()-last_progress>5:
                progress=dict(phase='fixture' if fixture else 'flight',time_s=flight_t,clock=clock,wall_s=time.monotonic()-began,
                    armed=hil_armed,offboard_armed=off.is_offboard_armed(),failure=failure,controls=controls.tolist(),telemetry=list(off.messages),last_events=off.events[-3:])
                write(output/'progress.json',progress);print(json.dumps(progress),flush=True);last_progress=time.monotonic()
        rows=np.array(records);rmse=np.sqrt(np.mean(rows[:,:3]**2,axis=0));xy=np.sqrt(np.mean(np.sum(rows[:,:2]**2,axis=1)))
        result=dict(duration_s=args.duration,flight_samples_100hz=len(rows),safe_completed=int(failure is None),count=1,
            physical_failure_code=failure,first_failure_s=failure_s,
            rmse_3d_h_xy_m=[float(np.linalg.norm(rmse)),float(rmse[2]),float(xy)],
            maximum_horizontal_error_m=float(np.linalg.norm(rows[:,:2],axis=1).max()),maximum_tension_n=peaks['tension_n'],
            maxima_checked_400hz=peaks,
            armed_sample_fraction=float(rows[:,4].mean()),offboard_armed_heartbeat_fraction=float(rows[:,5].mean()),
            wire_messages=dict(counts),first_hil=first_hil,commands_sent=off.commands,command_acks=off.acks,statustext=off.events,
            failure_bits=['nonfinite','tilt_over45deg','body_rate_over4radps','height_error_over4m','tension_over30N','PX4_disarmed','PX4_not_offboard'],
            failure_remainder_penalty_h_xy_m=[4.,6.],
            actuator_wait_mean_ms=float(np.mean(latencies)*1000),actuator_wait_max_ms=float(np.max(latencies)*1000),
            wall_s=time.monotonic()-began,physics_steps=plant.steps,physics_integrated_s=plant.steps*plant.dt,
            wind_peak_mps=float(np.linalg.norm(wind_rows,axis=1).max()),wind_seed=args.seed,wind_profile=wind_profile,
            target_height_m=target_height,cart_speed_mps=args.cart_speed,controller_sha256=cfg['controller_sha256'],residual_sha256=cfg['residual_sha256'],
            residual_enabled=not args.no_residual,policy_updated=False,residual_updated=False,formal_final_acceptance=False)
        np.savez_compressed(output/'trajectory.npz',time=np.r_[0.,times],state=np.array([plant.initial,*truth]),
            scoring=np.array(records),wind=np.array(wind_rows),commands=np.array(action_rows),px4_onboard=np.array(px4_rows),
            rpm=np.array(rpm_rows),observation=np.array(obs_rows),correction=np.array(ff_rows),heartbeat=np.array(modes),truth_control=np.array(truth_control))
        write(output/'result.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        write(output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),wall_s=time.monotonic()-began,
            first_hil=first_hil,wire_messages=dict(counts),last_flight_time_s=times[-1] if times else None,
            events=off.events[-25:] if 'off' in locals() else [],acks=off.acks if 'off' in locals() else []))
        raise
    finally:
        if link:
            try:link.mav.command_long_send(1,1,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,0,0,0,0,0,0,0)
            except Exception:pass
            link.close()
        if connection:connection.close()
        if process.poll() is None:
            try:process.stdin.write('shutdown\n');process.stdin.flush();process.wait(timeout=2)
            except Exception:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        log.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',default='stable_retest/slack5cm/assets');p.add_argument('--px4',default='/home/s/PX4-Autopilot')
    p.add_argument('--output',required=True);p.add_argument('--duration',type=float,default=120.);p.add_argument('--preflight',type=float,default=20.)
    p.add_argument('--cart-speed',type=float,default=5.);p.add_argument('--wind-peak',type=float,default=1.);p.add_argument('--seed',type=int,default=20272982)
    p.add_argument('--wind-profile',choices=('steady','varying','narrow_varying'))
    p.add_argument('--no-residual',action='store_true');run(p.parse_args())

if __name__=='__main__':main()
