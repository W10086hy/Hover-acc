"""Native MuJoCo offscreen replay of archived distributed-cable states."""
import argparse,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image,ImageDraw,ImageFont
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.io import read,write,digest
from .flexible import MujocoFlexiblePlant
from .physics import S,rotation
from .replay import frame_indices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',required=True)
    p.add_argument('--case',type=int,default=31);p.add_argument('--output',required=True)
    p.add_argument('--speed',type=float,default=6.)
    p.add_argument('--show-wind',action='store_true',help='Offline nose and wind-flow arrows, with upwind-angle diagnostics')
    a=p.parse_args();directory=Path(a.evaluation);out=Path(a.output);r=read(directory/'result.json')
    if r.get('engine')!='MuJoCo' or 'rope' not in r:raise ValueError('Use the native flexible-cable evaluation archive')
    if any(out.with_suffix(s).exists() for s in ['.gif','.png','.json']):raise FileExistsError('Preserve existing replay')
    with np.load(directory/'trajectory.npz') as data:t=data['times'].copy();state=data['state'][:,a.case].copy()
    indices=frame_indices(t,a.speed);case=r['cases'][a.case]
    archived_wind=None
    if a.show_wind:
        archived_wind=torch.load(directory/'wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'].numpy()[:,a.case]
    cfg=VehicleSpec.from_dict(r['actual_vehicle_spec']).apply(plant_config())
    plant=MujocoFlexiblePlant(cfg,torch.tensor([case['attachment_b_m']]),torch.tensor([case['tether_rest_length_m']]),
        r['rope'],physics_dt_s=r['physics_dt_s'],drag_scales=[case.get('actual_body_drag_scale',1.)])
    # Render-only lighting/line width; archived coordinates and physics stay fixed.
    xml=ET.fromstring(plant.xml);asset=ET.SubElement(xml,'asset')
    ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.96 .97 1',rgb2='.82 .86 .91',width='256',height='1536')
    ET.SubElement(xml.find('worldbody'),'light',pos='0 0 8',dir='0 0 -1',directional='true',ambient='.5 .5 .5',diffuse='.6 .6 .6')
    plant.mj_model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    plant.data=mujoco.MjData(plant.mj_model)
    model,data=plant.mj_model,plant.data;model.tendon_width[:]=.008
    model.vis.headlight.ambient[:]=.8;model.vis.headlight.diffuse[:]=.6
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0.,0.,2.7];camera.distance=max(8.,float(np.abs(state[:,:2]).max())+7)
    camera.azimuth=130.;camera.elevation=-30.
    renderer=mujoco.Renderer(model,height=720,width=960)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    frames=[];native_path=state[indices,:3]*S;heights=-state[:,2];hmax=max(6.,float(heights.max())+.4)
    def segment(scene,one,two,color):
        if np.linalg.norm(two-one)<1e-8:return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,1.5,one,two);scene.ngeom+=1
    def arrow(scene,origin,vector,color):
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.018,origin,origin+vector);scene.ngeom+=1
    try:
        for number,i in enumerate(indices):
            plant._sync(state[i:i+1].astype(np.float64));mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            for j in range(number):segment(renderer.scene,native_path[j],native_path[j+1],[.35,.35,.35,.6])
            wind_text=[]
            if a.show_wind:
                R=rotation(state[i:i+1,6:10])[0];nose=R[:,0]
                origin=state[i,:3]*S+np.array([0.,0.,.12])
                arrow(renderer.scene,origin,nose*S*.65,[1.,.1,.1,1.])
                wind=archived_wind[round(t[i]/.01)];norm=float(np.linalg.norm(wind[:2]))
                yaw=math.atan2(nose[1],nose[0])
                wind_text=['红：机头；青：风吹向']
                if norm>=.1:
                    flow=math.atan2(wind[1],wind[0]);coming=math.atan2(-wind[1],-wind[0])
                    angle=abs(math.atan2(math.sin(yaw-coming),math.cos(yaw-coming)))
                    arrow(renderer.scene,origin,wind*S/max(float(np.linalg.norm(wind)),1e-9)*.8,[0.,.7,1.,1.])
                    wind_text += [f'航向{math.degrees(yaw):.1f}°；来风{math.degrees(coming)%360:.1f}°',f'机头与来风夹角{math.degrees(angle):.1f}°']
                else:wind_text += [f'航向{math.degrees(yaw):.1f}°','静风/微风；来风方向不计']
            image=Image.new('RGB',(1280,800),'white');image.paste(Image.fromarray(renderer.render()),(0,80));draw=ImageDraw.Draw(image)
            draw.text((15,8),f'技术路线：BPTT｜MuJoCo原生有质量分段绳｜冻结26次策略｜{a.speed:g}倍回放',font=font,fill='black')
            draw.text((15,40),f'{case["attachment_b_m"][2]*100:g}cm｜基础风{case["wind_speed"]:g}m/s，峰值{case["actual_peak_wind_mps"]:.3f}m/s｜{case["wind_profile"]}｜t={t[i]:.1f}/{r["duration_s"]:g}s',font=font,fill='black')
            lines=[f'完整{r["duration_s"]:g}秒；安全{int(case["completed"])}/1',
                f'三维RMSE {case["position_rmse_m"]:.4f}m',f'高度RMSE {case["height_rmse_m"]:.4f}m',
                f'水平RMSE {case["horizontal_rmse_m"]:.4f}m',f'末30秒水平 {case["terminal30_horizontal_rmse_m"]:.4f}m',
                '12段原生弹簧/阻尼','11个有质量自由节点','灰线：已飞轨迹',
                '橙线：绳形，显示线宽加粗',f'状态种子 {r["seed_pair"][0]}',f'风场种子 {r["seed_pair"][1]}',
                '只回放归档状态，不重飞']
            lines+=wind_text
            for k,line in enumerate(lines):draw.text((975,105+27*k),line,font=small,fill='black')
            if not case['completed'] and t[i]>=case['safe_duration_s']:
                draw.rectangle((10,83,950,119),fill='#fff0c0')
                draw.text((18,88),f'{case["safe_duration_s"]:.2f}s触发安全界限；此后冻结，余时计罚',font=small,fill='black')
            left,top,right,bottom=1005,545,1255,750
            draw.text((left,top-26),'高度 / m（纵轴从0开始）',font=small,fill='black')
            draw.rectangle((left,top,right,bottom),outline='gray')
            points=[(left+(right-left)*float(tj)/t[-1],bottom-(bottom-top)*float(h)/hmax) for tj,h in zip(t,heights)]
            draw.line(points,fill='lightgray',width=1);draw.line(points[:i+1],fill='#1968ae',width=2)
            draw.text((974,top),f'{hmax:.0f}',font=small,fill='black');draw.text((982,bottom-18),'0',font=small,fill='black')
            draw.text((left,760),'MuJoCo +Y = 原工程 -Y',font=small,fill='black')
            frames.append(image)
    finally:renderer.close()
    assert plant.mj_step_calls==0
    out.parent.mkdir(parents=True,exist_ok=True)
    frames[-1].save(out.with_suffix('.png'))
    frames[0].save(out.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=100,loop=0,disposal=2)
    write(out.with_suffix('.json'),dict(engine='MuJoCo native offscreen rendering',case_index=a.case,case=case,
        checkpoint_sha256=r['checkpoint_sha256'],cumulative_physical_updates=26,metric_horizon_s=r['duration_s'],
        animation_start_s=float(t[indices[0]]),animation_end_s=float(t[indices[-1]]),frames=len(frames),playback_speed=a.speed,
        source_result_sha256=digest(directory/'result.json'),source_trajectory_sha256=digest(directory/'trajectory.npz'),
        gif_sha256=digest(out.with_suffix('.gif')),png_sha256=digest(out.with_suffix('.png')),
        renderer_source_sha256=digest(__file__),model_source_sha256=digest(Path(__file__).with_name('flexible.py')),
        new_simulation=False,mj_step_calls=0,height_axis_min_m=0,visual_only_rope_width_m=.008,
        wind_arrows=a.show_wind,angles_coordinate_system='P2 project XY,+X=0;wind vector pointsTO;upwind=-wind',
        source_wind_sha256=digest(directory/'wind_trace.pt') if a.show_wind else None))
    print(out,flush=True)


if __name__=='__main__':main()
