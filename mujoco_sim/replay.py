"""Render an archived MuJoCo flight with MuJoCo's native offscreen renderer."""
import argparse
import math
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.vehicle_spec import VehicleSpec
from .physics import MujocoPlant,to_mujoco


def frame_indices(times,speed):
    if not math.isfinite(speed) or not .1<=speed<=30: raise ValueError('Playback speed must be within0.1..30')
    values=np.arange(0,times[-1]+1e-9,speed/10)
    indices=np.unique(np.clip(np.searchsorted(times,values),0,len(times)-1))
    return np.unique(np.append(indices,len(times)-1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True);p.add_argument('--case',type=int,default=5)
    p.add_argument('--output',required=True);p.add_argument('--speed',type=float,default=6.)
    args=p.parse_args();directory=Path(args.evaluation);out=Path(args.output)
    result=read(directory/'result.json')
    if result.get('engine')!='mujoco': raise ValueError('Native MuJoCo replay requires a MuJoCo evaluation archive')
    if not 0<=args.case<len(result['cases']): raise ValueError('Case index outside archive')
    targets=[out,out.with_suffix('.png'),out.with_suffix('.json')]
    if any(f.exists() for f in targets): raise FileExistsError('Preserve previous replay')
    with np.load(directory/'trajectory.npz') as archive:
        times=archive['times'].copy();states=archive['state'][:,args.case].copy()
    indices=frame_indices(times,args.speed);case=result['cases'][args.case]
    arm=.05 if case['attachment']==2 else .1
    cfg=VehicleSpec.from_dict(result['actual_vehicle_spec']).apply(plant_config())
    native_rope=result.get('rope_model')=='native_spatial_tendon_upper_limit'
    plant_type=MujocoPlant;rope_options={}
    if native_rope:
        from .native_tendon import NativeTendonPlant
        plant_type=NativeTendonPlant;rope_options['tendon']=result['tendon_options']
    plant=plant_type(cfg,np.array([[0.,0.,arm]]),np.array([4.90 if arm==.05 else 4.85]),
        physics_dt_s=result['physics_dt_s'],drag_scales=[case.get('actual_body_drag_scale',1.)],**rope_options)
    model,data=plant.mj_model,plant.data
    camera=mujoco.MjvCamera();camera.lookat[:]=[0.,0.,2.8]
    camera.distance=8.;camera.azimuth=130.;camera.elevation=-17.
    fonts=[Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/simhei.ttf')]
    font_path=next((f for f in fonts if f.exists()),None)
    font=ImageFont.truetype(str(font_path),19) if font_path else ImageFont.load_default()
    small=ImageFont.truetype(str(font_path),16) if font_path else font
    frames=[];height=-states[:,2]
    height_low=min(4.5,math.floor(float(height.min())*10)/10)
    height_high=max(5.5,math.ceil(float(height.max())*10)/10)
    renderer=mujoco.Renderer(model,height=720,width=960)
    try:
        for i in indices:
            q,v=to_mujoco(states[i:i+1]);data.qpos[:]=q.ravel();data.qvel[:]=v.ravel()
            mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
            image=Image.new('RGB',(1280,800),'white');image.paste(Image.fromarray(renderer.render()),(0,80))
            draw=ImageDraw.Draw(image)
            backend_label='MuJoCo原生直绳' if native_rope else 'MuJoCo验证'
            draw.text((15,8),f'技术路线：BPTT｜{backend_label}｜{result["cohort"]}｜{args.speed:g}×回放',font=font,fill='black')
            draw.text((15,40),f'冻结21次策略；{arm*100:.0f}cm；基础风{case["wind_speed"]:g}m/s {math.degrees(case["direction_rad"]):g}° {case["wind_profile"]}；t={times[i]:.1f}/{result["duration_s"]:g}s',font=font,fill='black')
            lines=[f'单例完整{result["duration_s"]:g}秒',f'安全完成：{int(case["completed"])}/1',
                f'三维RMSE {case["position_rmse_m"]:.4f}m',f'高度RMSE {case["height_rmse_m"]:.4f}m',
                f'水平RMSE {case["horizontal_rmse_m"]:.4f}m',f'机头目标：0°',
                f'状态种子 {result["seed_pair"][0]}',f'风场种子 {result["seed_pair"][1]}',
                f'风峰值 {case["actual_peak_wind_mps"]:.3f}m/s',
                'MuJoCo刚体运动回放','原传感器/PI与权重冻结',
                '原生长度上限软约束' if native_rope else '直线弹性绳，无挠度','此为30秒开发验证']
            if not case['completed']:
                lines.append(f'触发界限：{case["safe_duration_s"]:.2f}s')
                if times[i]>=case['safe_duration_s']:
                    draw.rectangle((10,82,950,119),fill='#fff0c0')
                    draw.text((18,88),f'本例在{case["safe_duration_s"]:.2f}s触发离线安全界限；此后画面冻结，剩余时域计罚',font=small,fill='black')
            for k,line in enumerate(lines): draw.text((973,100+27*k),line,font=small,fill='black')
            left,top,right,bottom=1005,540,1250,750
            draw.text((left,top-25),'高度 h / m（目标5m）',font=small,fill='black')
            draw.rectangle((left,top,right,bottom),outline='gray')
            def height_y(h):return bottom-(bottom-top)*(float(h)-height_low)/(height_high-height_low)
            def points(vals):
                return [(left+(right-left)*float(t)/times[-1],height_y(h)) for t,h in zip(times,vals)]
            draw.line([(left,height_y(5)),(right,height_y(5))],fill='gray',width=1)
            draw.text((965,top),f'{height_high:.1f}',font=small,fill='black')
            draw.text((965,bottom-18),f'{height_low:.1f}',font=small,fill='black')
            draw.line(points(height),fill='#1968ae',width=2)
            marker=points(height)[i];draw.ellipse((marker[0]-4,marker[1]-4,marker[0]+4,marker[1]+4),fill='red')
            draw.text((left,760),'MuJoCo: +Y = 原工程 -Y',font=small,fill='black')
            frames.append(image)
    finally: renderer.close()
    out.parent.mkdir(parents=True,exist_ok=True)
    frames[len(frames)//2].save(out.with_suffix('.png'))
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=100,loop=0,disposal=2)
    write(out.with_suffix('.json'),dict(route='BPTT',engine='MuJoCo',cohort=result['cohort'],case_index=args.case,
        checkpoint_sha256=result['checkpoint_sha256'],source_trajectory_sha256=digest(directory/'trajectory.npz'),
        source_result_sha256=digest(directory/'result.json'),gif_sha256=digest(out),metric_horizon_s=result['duration_s'],
        playback_speed=args.speed,frames=len(frames),seed_pair=result['seed_pair'],case=case,new_simulation=False,
        rope_model=result.get('rope_model','custom_unilateral_spring'),
        render_source_sha256=digest(Path(__file__)),failure_freeze_banner=not case['completed']))
    print(out,flush=True)


if __name__=='__main__': main()
