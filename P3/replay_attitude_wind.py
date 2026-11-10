"""Readable native-MuJoCo aircraft close-up and wind/attitude dashboard.

Only archived states are rendered. No mj_step, controller, fitting, pose-angle
exaggeration or resimulation. Project XY bearings use +X=0deg and +Y=90deg.
"""
import argparse,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image,ImageDraw,ImageFont
from P2.hover_bptt.io import read,write,digest
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec
from mujoco_sim.flexible import MujocoFlexiblePlant
from mujoco_sim.physics import S,rotation

RED='#d62832';GREEN='#14854c';BLUE='#245ac4';CYAN='#008eae';GRAY='#78828d';INK='#192838';ORANGE='#ce741a'
FONT='C:/Windows/Fonts/msyh.ttc'


def angle(v):return math.degrees(math.atan2(v[1],v[0]))%360


def screen_arrow(draw,one,two,color,width=4,head=12):
    a=np.asarray(one,dtype=float);b=np.asarray(two,dtype=float);d=b-a;n=np.linalg.norm(d)
    if n<1:return
    d/=n;perp=np.array([-d[1],d[0]])
    draw.line([tuple(a),tuple(b-d*head*.5)],fill=color,width=width)
    draw.polygon([tuple(b),tuple(b-d*head+perp*head*.45),tuple(b-d*head-perp*head*.45)],fill=color)


def scene_segment(scene,a,b,color,width=.002,arrow=False):
    if np.linalg.norm(b-a)<1e-9:return
    kind=mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
    mujoco.mjv_connector(g,kind,width,a,b);scene.ngeom+=1


class Dashboard:
    def __init__(self,directory,index):
        self.directory=Path(directory);self.result=read(self.directory/'result.json');self.case=self.result['cases'][index];self.index=index
        if self.result.get('engine')!='MuJoCo' or 'P3' not in self.result.get('phase',''):raise ValueError('Use current archived P3 MuJoCo states')
        with np.load(self.directory/'trajectory.npz') as data:self.t=data['times'].copy();self.s=data['state'][:,index].copy()
        self.wind_file=torch.load(self.directory/'wind_trace.pt',map_location='cpu',weights_only=False)
        self.wind=self.wind_file['wind_w_mps'].numpy()[:,index];self.wind_dt=self.wind_file['dt_s']
        self.R=rotation(self.s[:,6:10].astype(float))
        self.rpy=np.rad2deg(np.stack((np.arctan2(self.R[:,2,1],self.R[:,2,2]),np.arcsin(np.clip(-self.R[:,2,0],-1,1)),np.arctan2(self.R[:,1,0],self.R[:,0,0])),-1))
        heading=torch.load(self.directory/'heading_witness.pt',map_location='cpu',weights_only=False).numpy()[:,index]
        at=np.rint(self.t/.01).astype(int);self.yaw_check=float(np.abs(np.arctan2(np.sin(np.deg2rad(self.rpy[:,2])-heading[at]),np.cos(np.deg2rad(self.rpy[:,2])-heading[at]))).max())
        assert self.yaw_check<2e-6 and self.t[0]==0 and self.t[-1]==self.result['duration_s']
        config=VehicleSpec.from_dict(self.result['actual_vehicle_spec']).apply(plant_config())
        self.plant=MujocoFlexiblePlant(config,np.array([self.case['attachment_b_m']]),np.array([self.case['tether_rest_length_m']]),
            self.result['rope'],drag_scales=[self.case.get('actual_body_drag_scale',1.)])
        xml=ET.fromstring(self.plant.xml);asset=ET.SubElement(xml,'asset')
        ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.98 .99 1',rgb2='.9 .94 .98',width='256',height='1536')
        world=xml.find('worldbody');ET.SubElement(world,'light',pos='1 -2 8',dir='0 0 -1',directional='true',ambient='.25 .25 .25',diffuse='.45 .45 .45')
        world.find("site[@name='goal']").set('group','4')
        vehicle=world.find("body[@name='vehicle0']")
        for x in [-self.plant.rigid.dx,self.plant.rigid.dx]:
            for y in [-self.plant.rigid.dy,self.plant.rigid.dy]:
                ET.SubElement(vehicle,'geom',type='capsule',fromto=f'0 0 0 {x} {y} 0',size='.009',rgba='.14 .2 .27 1')
        for g in vehicle.findall('geom'):
            if g.get('type')=='cylinder':
                front=float(g.get('pos').split()[0])>0
                g.set('rgba','.85 .16 .2 .8' if front else '.18 .24 .3 .8')
        self.plant.mj_model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));self.plant.data=mujoco.MjData(self.plant.mj_model)
        model=self.plant.mj_model;model.tendon_width[:]=.006;model.vis.headlight.ambient[:]=.35;model.vis.headlight.diffuse[:]=.45
        self.renderer=mujoco.Renderer(model,height=500,width=860)
        self.close=mujoco.MjvCamera();self.close.distance=1.1;self.close.azimuth=120.;self.close.elevation=-23.
        self.close_option=mujoco.MjvOption();self.close_option.sitegroup[4]=0
        self.wide=mujoco.MjvCamera();self.wide.lookat[:]=[0,0,2.6];self.wide.distance=max(8.,float(np.abs(self.s[:,:2]).max())+7)
        self.wide.azimuth=130.;self.wide.elevation=-30.
        self.fonts={n:ImageFont.truetype(FONT,n) for n in [16,18,19,20,22,24,28,32]}
        self.native_path=self.s[:,:3]*S;self.span=max(.8,float(np.abs(self.s[:,:2]).max())+.18)
        self.provenance={name:digest(self.directory/name) for name in ['result.json','trajectory.npz','wind_trace.pt','heading_witness.pt']}

    def panel(self,draw,box,title):
        draw.rounded_rectangle(box,radius=10,fill='white',outline='#d5dfe9',width=2)
        draw.text((box[0]+12,box[1]+7),title,font=self.fonts[18],fill=INK)

    def render(self,i,speed):
        t=self.t[i];x=self.s[i];R=self.R[i];native=x[:3]*S
        wind=self.wind[round(t/self.wind_dt)];wind_speed=float(np.linalg.norm(wind));horizontal=float(np.linalg.norm(wind[:2]))
        self.plant._sync(x[None].astype(float));mujoco.mj_forward(self.plant.mj_model,self.plant.data)
        assert np.allclose(self.plant.data.xpos[self.plant.body_ids[0]],native,atol=1e-7)
        self.close.lookat[:]=native+np.array([0.,0.,.16])
        self.renderer.update_scene(self.plant.data,camera=self.close,scene_option=self.close_option);scene=self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        # Gray world-level reference ring translates with the vehicle, never tilts.
        theta=np.linspace(0,2*np.pi,49);ring=native+np.stack((.29*np.cos(theta),.29*np.sin(theta),np.full_like(theta,-.045)),-1)
        for a,b in zip(ring[:-1],ring[1:]):scene_segment(scene,a,b,[.2,.23,.28,.8],.0015)
        for vector,color,length in [(R[:,0],[.9,.05,.09,1],.43),(R[:,1],[.05,.6,.25,1],.29),(-R[:,2],[.1,.3,.95,1],.36)]:
            scene_segment(scene,native+np.array([0,0,.018]),native+np.array([0,0,.018])+vector*S*length,color,.014,True)
        if horizontal>=.1:
            origin=native+np.array([0,0,.46]);length=.25+.38*min(wind_speed/8,1)
            scene_segment(scene,origin-wind*S/max(wind_speed,1e-9)*length/2,origin+wind*S/max(wind_speed,1e-9)*length/2,[0,.65,.85,1],.025,True)
        close=Image.fromarray(self.renderer.render())
        self.renderer.update_scene(self.plant.data,camera=self.wide);scene=self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        for j in range(0,i,3):scene_segment(scene,self.native_path[j],self.native_path[min(j+3,i)],[.08,.37,.7,.8],.009)
        for z in range(-2,3):
            scene_segment(scene,np.array([z,-2,.008]),np.array([z,2,.008]),[.64,.7,.77,.5],.004)
            scene_segment(scene,np.array([-2,z,.008]),np.array([2,z,.008]),[.64,.7,.77,.5],.004)
        wide=Image.fromarray(self.renderer.render()).resize((434,244),Image.Resampling.LANCZOS)
        image=Image.new('RGB',(1400,930),'#edf3f8');draw=ImageDraw.Draw(image)
        draw.text((20,9),f'技术路线：BPTT｜P3残差补偿｜冻结26次策略｜MuJoCo原生回放｜完整{self.t[-1]:g}s｜{speed:g}倍速',font=self.fonts[24],fill=INK)
        draw.text((20,47),f'{self.case["attachment_b_m"][2]*100:g}cm连接点  ·  基础风{self.case["wind_speed"]:g}m/s / 单例峰值{self.case["actual_peak_wind_mps"]:.3f}m/s  ·  t={t:6.1f}/{self.t[-1]:g}s',font=self.fonts[22],fill=INK)
        draw.text((20,79),f'全程RMSE 三维/高度/水平：{self.case["position_rmse_m"]:.3f}/{self.case["height_rmse_m"]:.3f}/{self.case["horizontal_rmse_m"]:.3f}m   安全{int(self.case["completed"])}/1   状态/风种子 {self.result["seed_pair"][0]}/{self.result["seed_pair"][1]}   位姿与风仅用于离线显示',font=self.fonts[16],fill=INK)
        image.paste(close,(12,100));draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((22,110,840,144),radius=6,fill='white')
        draw.text((32,114),'机体特写：相机仅跟随平移；灰环为水平参考；姿态按真实角度显示',font=self.fonts[19],fill=INK)
        draw.rounded_rectangle((22,553,840,589),radius=6,fill='white')
        draw.text((33,559),'红：机头 +XB    绿：机体右侧 +YB    蓝：推力方向    青：风吹向',font=self.fonts[20],fill=INK)
        self.panel(draw,(884,100,1388,600),'姿态与风向（工程坐标；0°=+X，90°=+Y）')
        for row,(name,value,color) in enumerate(zip(['横滚 φ','俯仰 θ','航向 ψ'],self.rpy[i],[RED,GREEN,BLUE])):
            draw.text((903,143+45*row),name,font=self.fonts[24],fill=color)
            draw.text((1040,137+45*row),f'{value:+7.3f}°',font=self.fonts[32],fill=color)
        draw.text((904,286),f'当前风速 {wind_speed:.3f} m/s',font=self.fonts[28],fill=CYAN)
        draw.text((904,325),f'位置 x={x[0]:+.2f}  y={x[1]:+.2f}  h={-x[2]:.2f}m',font=self.fonts[18],fill=INK)
        center=np.array([1023.,465.]);radius=88
        draw.ellipse((*(center-radius),*(center+radius)),outline='#b6c5d4',width=2)
        draw.line((center[0]-radius-4,center[1],center[0]+radius+4,center[1]),fill='#c6d2de',width=1)
        draw.line((center[0],center[1]-radius-4,center[0],center[1]+radius+4),fill='#c6d2de',width=1)
        draw.text((1105,453),'+X',font=self.fonts[16],fill=GRAY);draw.text((1010,358),'+Y',font=self.fonts[16],fill=GRAY)
        nose=R[:2,0].copy();nose/=max(np.linalg.norm(nose),1e-9);screen_arrow(draw,center,center+nose*np.array([1,-1])*78,RED,5,14)
        draw.text((1151,373),'红：机头',font=self.fonts[20],fill=RED)
        draw.text((1151,407),'青：风吹向',font=self.fonts[20],fill=CYAN)
        draw.text((1151,441),'灰：风来向',font=self.fonts[20],fill=GRAY)
        if horizontal>=.1:
            v=wind[:2]/horizontal*np.array([1,-1]);screen_arrow(draw,center-v*62,center+v*82,CYAN,5,14)
            screen_arrow(draw,center,center-v*63,GRAY,2,9)
            flow=angle(wind);coming=(flow+180)%360;yaw=self.rpy[i,2]%360
            relative=abs((yaw-coming+180)%360-180)
            draw.text((1145,480),f'吹向 {flow:6.1f}°',font=self.fonts[20],fill=CYAN)
            draw.text((1145,514),f'来向 {coming:6.1f}°',font=self.fonts[20],fill=GRAY)
            draw.text((904,564),f'机头与来风夹角 {relative:.1f}°；0°表示迎风',font=self.fonts[19],fill=INK)
        else:draw.text((904,564),'静风/微风：不定义风向和迎风夹角',font=self.fonts[19],fill=GRAY)
        self.panel(draw,(12,612,462,916),'全局3D轨迹（地面高度=0m）');image.paste(wide,(20,649));draw=ImageDraw.Draw(image)
        self.panel(draw,(475,612,925,916),'水平XY轨迹：红机头 / 青风吹向')
        plot=(590,661,810,881);cx=(plot[0]+plot[2])/2;cy=(plot[1]+plot[3])/2;scale=min(plot[2]-plot[0],plot[3]-plot[1])/(2*self.span)
        xy=lambda q:(cx+float(q[0])*scale,cy-float(q[1])*scale)
        for value in np.linspace(-self.span,self.span,5):
            xx,yy=xy((value,value));draw.line((xx,plot[1],xx,plot[3]),fill='#e2e9f0');draw.line((plot[0],yy,plot[2],yy),fill='#e2e9f0')
            draw.text((xx-12,plot[3]+1),f'{value:.1f}',font=self.fonts[16],fill=GRAY)
            draw.text((plot[0]-39,yy-9),f'{value:.1f}',font=self.fonts[16],fill=GRAY)
        draw.line((plot[0],cy,plot[2],cy),fill='#a9b8c7');draw.line((cx,plot[1],cx,plot[3]),fill='#a9b8c7')
        draw.line([xy(q) for q in self.s[:,:2]],fill='#d0dae5',width=2)
        if i:draw.line([xy(q) for q in self.s[:i+1,:2]],fill='#326dac',width=3)
        point=np.array(xy(x[:2]));draw.ellipse((point[0]-4,point[1]-4,point[0]+4,point[1]+4),fill=RED)
        screen_arrow(draw,point,point+nose*np.array([1,-1])*28,RED,3,9)
        if horizontal>=.1:screen_arrow(draw,point-wind[:2]/horizontal*np.array([1,-1])*22,point+wind[:2]/horizontal*np.array([1,-1])*36,CYAN,3,10)
        draw.text((cx-5,cy-12),'*',font=self.fonts[24],fill='black')
        draw.text((813,876),'X/m',font=self.fonts[16],fill=GRAY);draw.text((546,642),'Y/m',font=self.fonts[16],fill=GRAY)
        self.panel(draw,(938,612,1388,916),'最近20秒姿态曲线（动态量程 / °）')
        window=(self.t>=max(0,t-20))&(self.t<=t);tw=self.t[window];values=self.rpy[window]
        limit=max(1.,float(np.abs(values).max())*1.2);box=(983,666,1365,865)
        lo=max(0,t-20);hi=max(20,t);map_point=lambda time,value:(box[0]+(time-lo)/(hi-lo)*(box[2]-box[0]),(box[1]+box[3])/2-value/limit*(box[3]-box[1])/2)
        for value in [-limit,0,limit]:
            yy=map_point(lo,value)[1];draw.line((box[0],yy,box[2],yy),fill='#cbd7e2');draw.text((943,yy-10),f'{value:.1f}',font=self.fonts[16],fill=GRAY)
        for k,color in enumerate([RED,GREEN,BLUE]):
            if len(tw)>1:draw.line([map_point(a,b) for a,b in zip(tw,values[:,k])],fill=color,width=2)
        draw.text((983,871),f'{lo:.0f}s',font=self.fonts[16],fill=GRAY);draw.text((1320,871),f'{hi:.0f}s',font=self.fonts[16],fill=GRAY)
        draw.text((985,892),'红=横滚  绿=俯仰  蓝=航向',font=self.fonts[16],fill=INK)
        return image

    def close_renderer(self):self.renderer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--case',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--speed',type=float,default=3.);p.add_argument('--preview-only',action='store_true')
    p.add_argument('--preview-time',type=float);a=p.parse_args();out=Path(a.output)
    if not 1<=a.speed<=10:raise ValueError('Playback within1..10 required')
    suffixes=['.preview.png','.preview.json'] if a.preview_only else ['.gif','.png','.json']
    if any(Path(str(out)+s).exists() for s in suffixes):raise FileExistsError('Preserve previous replay')
    out.parent.mkdir(parents=True,exist_ok=True);dash=Dashboard(a.evaluation,a.case)
    preview=np.searchsorted(dash.t,a.preview_time) if a.preview_time is not None else int(np.argmax(np.where(dash.t>=4,np.linalg.norm(dash.rpy[:,:2],axis=1),-1)))
    preview=min(preview,len(dash.t)-1);frames=[]
    try:
        if a.preview_only:
            dash.render(preview,a.speed).save(str(out)+'.preview.png')
            write(str(out)+'.preview.json',dict(case_index=a.case,time_s=float(dash.t[preview]),rpy_deg=dash.rpy[preview].tolist(),source_sha256=digest(__file__),new_simulation=False));return
        indices=np.unique(np.clip(np.rint(np.arange(0,dash.t[-1]+1e-6,a.speed/10)*10).astype(int),0,len(dash.t)-1))
        if indices[-1]!=len(dash.t)-1:indices=np.append(indices,len(dash.t)-1)
        for number,i in enumerate(indices):
            frame=dash.render(i,a.speed);frames.append(frame.quantize(colors=256,method=Image.Quantize.FASTOCTREE))
            if number%100==0:print(dict(frame=number,total=len(indices),simulation_s=float(dash.t[i])),flush=True)
        dash.render(preview,a.speed).save(out.with_suffix('.png'))
        frames[0].save(out.with_suffix('.gif'),save_all=True,append_images=frames[1:],duration=100,loop=0,disposal=2)
        assert dash.plant.mj_step_calls==0
        for name,sha in dash.provenance.items():assert digest(dash.directory/name)==sha
        parent=read(dash.directory.parent/'protocol.json')
        write(out.with_suffix('.json'),dict(case_index=a.case,case=dash.case,metric_horizon_s=dash.result['duration_s'],
            engine='MuJoCo native renderer;archived P3 state',stage='P3 frozen26 Actor and current residual',
            checkpoint_sha256=dash.result['checkpoint_sha256'],residual_checkpoint_sha256=parent['residual_checkpoint_sha256'],
            animation_start_s=float(dash.t[indices[0]]),animation_end_s=float(dash.t[indices[-1]]),frames=len(frames),fps=10,playback_speed=a.speed,
            poster_time_s=float(dash.t[preview]),source_files_sha256=dash.provenance,renderer_source_sha256=digest(__file__),
            model_source_sha256=digest(Path(__file__).parents[1]/'mujoco_sim/flexible.py'),
            gif_sha256=digest(out.with_suffix('.gif')),png_sha256=digest(out.with_suffix('.png')),
            mj_step_calls=0,new_simulation=False,policy_updates=0,residual_updates=0,attitude_angles_exaggerated=False,
            attitude_sample_hz=10,wind_sample_hz=100,yaw_vs_fullrate_witness_max_difference_rad=dash.yaw_check,
            camera='close-up follows translation;fixed120deg azimuth,-23deg elevation;global overview separate',
            arrows=dict(red='nose +XB',green='body right +YB',blue='thrust axis -ZB',cyan='world wind flowTO',gray='wind FROM'),
            coordinate_system='project XY bearings:+X=0,+Y=90;wind from=flow+180;3D native axes use S=diag(1,-1,-1)',
            pose_curve='last20s;dynamic y range;degrees',views=['native3d_closeup','global3d_trajectory','xy_trajectory','wind_compass','rpy_values','last20s_rpy']))
    finally:dash.close_renderer()
    print(dict(output=str(out),frames=len(frames),new_simulation=False),flush=True)


if __name__=='__main__':main()
