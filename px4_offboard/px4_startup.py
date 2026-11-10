"""Generate run-local PX4 startup; never modify the installed firmware tree."""
import math
import numpy as np
from .motor_mapping import PX4_MOTORS_IN_P2_ORDER

def startup(cfg):
    pc=cfg['nominal'];a=pc['airframe'];u=pc['actuators'];s=cfg['stage']['controller']
    vmax=sum(u['motor_speed_polynomial']);fmax=u['kf']*vmax**2
    scales=np.array([2*math.sqrt(2)*a['rotor_dy_m']*fmax,2*math.sqrt(2)*a['rotor_dx_m']*fmax,4*u['km']*vmax**2])
    params=dict(SYS_AUTOSTART=10016,SYS_MC_EST_GROUP=2,MAV_SYS_ID=1,MAV_COMP_ID=1,MAV_TYPE=2,MAV_PROTO_VER=2,
        CAL_ACC0_ID=1310988,CAL_GYRO0_ID=1310988,CAL_MAG0_ID=197388,CAL_MAG0_PRIO=50,
        SENS_BOARD_X_OFF=0.000001,SENS_DPRES_OFF=0.001,SENS_IMU_MODE=1,EKF2_MULTI_IMU=0,
        IMU_INTEG_RATE=400,IMU_GYRO_RATEMAX=400,IMU_GYRO_CUTOFF=60,IMU_DGYRO_CUTOFF=30,
        EKF2_GPS_CTRL=0,EKF2_EV_CTRL=0,EKF2_MAG_TYPE=0,EKF2_HGT_REF=0,
        COM_ARM_WO_GPS=1,COM_RC_IN_MODE=4,COM_DISARM_PRFLT=0,COM_DISARM_LAND=0,
        COM_OF_LOSS_T=1.,COM_RC_LOSS_T=1.,COM_DL_LOSS_T=10.,COM_CPU_MAX=-1,
        CBRK_SUPPLY_CHK=894281,CBRK_AIRSPD_CHK=0,BAT1_N_CELLS=4,
        CA_AIRFRAME=0,CA_METHOD=0,CA_ROTOR_COUNT=4,THR_MDL_FAC=1.,MC_AIRMODE=0,
        MC_BAT_SCALE_EN=0,MOT_SLEW_MAX=0.,MPC_THR_HOVER=a['mass_kg']*a['gravity_mps2']/(4*fmax))
    for axis,i in [('ROLL',0),('PITCH',1),('YAW',2)]:
        params[f'MC_{axis}RATE_K']=1.
        for key,val in [('P',s['rate_kp_per_s'][i]),('I',s['rate_ki_per_s2'][i]),('D',0.),('FF',0.)]:
            params[f'MC_{axis}RATE_{key}']=a['inertia_body_kgm2'][i]*val/scales[i]
    for short,i in [('RR',0),('PR',1),('YR',2)]:
        params[f'MC_{short}_INT_LIM']=min(1.,a['inertia_body_kgm2'][i]*s['rate_ki_per_s2'][i]*s['rate_integral_limit']/scales[i])
    dx,dy=a['rotor_dx_m'],a['rotor_dy_m'];ratio=u['km']/u['kf']
    # v1.14 ControlAllocator clears an axis when every coefficient is<=0.05.
    # ratio≈0.0285 with CT=1 accidentally removed all yaw authority. Uniform
    # CT scaling cancels in normalized allocation;physical kf/km stay unchanged.
    effectiveness_scale=max(4.,.1/abs(ratio))
    p2_rotors = [(dx, dy, ratio), (-dx, dy, -ratio), (dx, -dy, -ratio), (-dx, -dy, ratio)]
    for motor, (x, y, km) in zip(PX4_MOTORS_IN_P2_ORDER, p2_rotors):
        rotor = motor - 1
        for key, value in [('PX', x), ('PY', y), ('PZ', 0), ('KM', km),
                           ('CT', effectiveness_scale), ('AX', 0), ('AY', 0), ('AZ', -1)]:
            params[f'CA_ROTOR{rotor}_{key}'] = value
        params[f'PWM_MAIN_FUNC{motor}'] = 100 + motor
    lines=['#!/bin/sh','. px4-alias.sh','param select parameters.bson']
    lines += [f'param set {k} {v:.12g}' for k,v in params.items()]
    lines += ['dataman start','simulator_mavlink start -c 4560','battery_simulator start','sensors start',
        'commander start','pwm_out_sim start -m sim','ekf2 start','control_allocator start',
        'mc_rate_control start','mc_att_control start','land_detector start multicopter',
        'mavlink start -x -u 14580 -o 14540 -t 127.0.0.1 -r 4000000 -m custom']
    for name,rate in [('HEARTBEAT',10),('ATTITUDE_QUATERNION',100),('HIGHRES_IMU',100),('ALTITUDE',20),('SYS_STATUS',1),('ATTITUDE_TARGET',100),('EXTENDED_SYS_STATE',5)]:
        lines.append(f'mavlink stream -u 14580 -s {name} -r {rate}')
    # Preserve actual runtime effectiveness evidence in every new PX4 log.
    lines += ['mavlink boot_complete','param save','ver all','control_allocator status']
    return '\n'.join(lines)+'\n',dict(parameters=params,torque_normalization_nm=scales.tolist(),
        simulation_motor_order=list(PX4_MOTORS_IN_P2_ORDER),
        effectiveness_coefficient_scale=effectiveness_scale,yaw_effectiveness_min_abs=effectiveness_scale*abs(ratio),
        weak_axis_threshold=0.05,physical_motor_parameters_changed=False,
        GPS_messages_sent=False,position_controllers_started=False,attitude_control_bypassed_in_body_rate_offboard=True)
