"""Offline native-MuJoCo cart/cable/attitude replay and scientific plots."""
import math
from pathlib import Path
import numpy as np
import torch,mujoco
from PIL import Image,ImageDraw,ImageFont
from P2.hover_bptt.io import read
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec
from mujoco_sim.physics import S,rotation
from P3.replay_attitude_wind import scene_segment,screen_arrow,FONT
from .physics import MovingAnchorPlant

ROOT=Path(__file__).resolve().parents[1]
BLUE='#245ac4';RED='#d62832';GREEN='#14854c';CYAN='#008eae';INK='#192838';GRAY='#78828d';ORANGE='#ce741a'


def attitude(state):
    R=rotation(state[:,6:10].astype(float))
    return R,np.rad2deg(np.stack((np.arctan2(R[:,2,1],R[:,2,2]),np.arcsin(np.clip(-R[:,2,0],-1,1)),np.arctan2(R[:,1,0],R[:,0,0])),-1))


class Dashboard:
    def __init__(self,base,index):
        self.base=Path(base);self.folder=self.base/'with_compensation';self.index=index
        self.protocol=read(self.base/'manifest.json')['protocol'];self.result=read(self.folder/'result.json');self.case=self.result['cases'][index]
        self.duration=float(self.result['duration_s'])
        self.continuation=bool(self.protocol.get('continue_after_performance_boundary',False))
        with np.load(self.folder/'trajectory.npz') as f:
            self.t=f['times'].copy();self.s=f['state'][:,index].copy();self.valid=f['valid'][:,index].copy()
            self.cart=f['cart_position_w_m'].copy();self.target=f['reference_position_w_m'].copy()
            self.cart_velocity=f['cart_velocity_w_m'].copy() if 'cart_velocity_w_m' in f else np.tile([4.,0.,0.],(len(self.t),1))
            if self.cart.ndim==3:self.cart=self.cart[:,index];self.target=self.target[:,index];self.cart_velocity=self.cart_velocity[:,index]
        self.error=self.s[:,:3]-self.target;self.R,self.rpy=attitude(self.s)
        self.wind=torch.load(self.folder/'wind_trace.pt',weights_only=False)['wind_w_mps'].numpy()[:,index]
        cfg=VehicleSpec.from_dict(self.result['actual_vehicle_spec']).apply(plant_config());cfg['tether']['anchor_w_m']=[0.,0.,-self.protocol['cart_anchor_height_m']]
        rope=read(ROOT/self.protocol['residual_protocol'])['rope'];arm=self.case['attachment_cm']/100
        cfg['simulation']['target_w_m']=[0.,0.,-self.protocol['target_aircraft_height_m']]
        self.plant=MovingAnchorPlant(cfg,np.array([[0.,0.,arm]]),np.array([self.case.get('tether_rest_length_m',self.protocol['target_aircraft_height_m']-.05-arm)]),rope,
            physics_dt_s=self.protocol['physics_dt_s'],cart_speed=self.case.get('cart_speed_mps',self.protocol['cart_speed_mps']),cart_direction_deg=self.protocol['cart_direction_engineering_deg'],
            cart_mass=self.protocol['cart_inertial_regularization_kg'],cart_motion=self.protocol.get('case_specs',[{}]*self.result['count'])[index].get('cart_motion',self.protocol.get('cart_motion')),drag_scales=[.9])
        self.plant.mj_model.tendon_width[:]=.012  # Display only; physical cable diameter stays3mm.
        self.plant.mj_model.tendon_rgba[:]=[1.,.48,.03,1.]
        self.renderer=mujoco.Renderer(self.plant.mj_model,height=460,width=860)
        self.option=mujoco.MjvOption();self.option.sitegroup[:]=0
        self.wide=mujoco.MjvCamera();self.wide.azimuth=130;self.wide.elevation=-22
        self.close=mujoco.MjvCamera();self.close.azimuth=120;self.close.elevation=-20;self.close.distance=1.25
        self.font={n:ImageFont.truetype(FONT,n) for n in [14,16,18,20,22,24,28,30]}
        self.span=max(.6,float(np.abs(self.error[self.valid,:2]).max())+.2)

    def panel(self,draw,box,title):
        draw.rounded_rectangle(box,radius=10,fill='white',outline='#d5dfe9',width=2)
        draw.text((box[0]+12,box[1]+7),title,font=self.font[18],fill=INK)

    def axes(self,draw,box,xlim,ylim):
        x0,y0,x1,y1=box
        draw.line((x0,y1,x1,y1),fill=GRAY,width=1);draw.line((x0,y0,x0,y1),fill=GRAY,width=1)
        def mapped(x,y):
            a=np.atleast_1d(x);b=np.atleast_1d(y)
            return np.stack((x0+(a-xlim[0])/(xlim[1]-xlim[0])*(x1-x0),y1-(b-ylim[0])/(ylim[1]-ylim[0])*(y1-y0)),-1)
        return mapped

    def series(self,draw,mapping,x,y,color,width=2):
        ok=np.isfinite(x)&np.isfinite(y)
        points=mapping(x[ok],y[ok]);points=[tuple(p) for p in points]
        if len(points)>1:draw.line(points,fill=color,width=width)

    def render(self,index,speed):
        t=self.t[index];shown=index if self.valid[index] else max(0,np.nonzero(self.valid)[0][-1])
        x=self.s[shown];R=self.R[shown];native=x[:3]*S;cart=self.cart[shown]*S
        wind=self.wind[round(self.t[shown]/.01)];ws=float(np.linalg.norm(wind));relative_wind=wind-x[3:6]
        p=self.plant;p.data.time=float(self.t[shown]);p._sync(x[None].astype(float));p._kinematics();mujoco.mj_forward(p.mj_model,p.data)
        self.wide.lookat[:]=(native+cart)/2;self.wide.distance=max(8.5,float(np.linalg.norm(native-cart))*1.75)
        self.renderer.update_scene(p.data,camera=self.wide,scene_option=self.option);scene=self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        for delta in range(-4,5):
            xx=math.floor(cart[0]/2)*2+delta*2
            scene_segment(scene,np.array([xx,-3,.005]),np.array([xx,3,.005]),[.62,.67,.72,.6],.007)
        # Plot the complete path in the translating cart frame at its current
        # origin, so hundreds of metres of cart travel do not hide tracking error.
        for j in range(0,shown,10):
            end=min(j+10,shown)
            a=(self.s[j,:3]-self.cart[j]+self.cart[shown])*S
            b=(self.s[end,:3]-self.cart[end]+self.cart[shown])*S
            scene_segment(scene,a,b,[.15,.38,.8,1],.015)
        target=self.target[shown]*S
        scene_segment(scene,target-[.18,0,0],target+[.18,0,0],[.05,.65,.2,1],.012)
        scene_segment(scene,target-[0,.18,0],target+[0,.18,0],[.05,.65,.2,1],.012)
        scene_segment(scene,cart+[-.6,0,.3],cart+[.5,0,.3],[.2,.3,.36,1],.04,True)
        origin=cart+np.array([.7,0.,2.4]);direction=wind*S/max(ws,1e-9)
        scene_segment(scene,origin-direction*.4,origin+direction*.65,[0.,.65,.85,1.],.04,True)
        wide=Image.fromarray(self.renderer.render())
        self.close.lookat[:]=native+[0,0,.08]
        self.renderer.update_scene(p.data,camera=self.close,scene_option=self.option);scene=self.renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        for vector,color,length in [(R[:,0],[.9,.05,.09,1],.45),(R[:,1],[.05,.6,.25,1],.30),(-R[:,2],[.1,.3,.95,1],.36)]:
            scene_segment(scene,native+[0,0,.02],native+[0,0,.02]+vector*S*length,color,.015,True)
        theta=np.linspace(0,2*np.pi,37);ring=native+np.stack((.30*np.cos(theta),.30*np.sin(theta),np.full_like(theta,-.05)),-1)
        for a,b in zip(ring[:-1],ring[1:]):scene_segment(scene,a,b,[.3,.3,.3,.7],.002)
        close=Image.fromarray(self.renderer.render()).resize((326,174),Image.Resampling.LANCZOS)
        image=Image.new('RGB',(1400,960),'#edf3f8');d=ImageDraw.Draw(image)
        updates=self.result.get('controller_cumulative_physical_updates',26)
        stage='track诊断续算' if self.continuation else 'track'
        d.text((18,9),f'技术路线：BPTT｜{stage}｜Actor累计{updates}次＋P3残差｜MuJoCo｜完整{self.duration:g}秒评分｜{speed:g}倍回放',font=self.font[24],fill=INK)
        label='初始随车4m/s' if self.case['initial_condition']=='co_moving_4mps' else '从静止悬停起步'
        d.text((18,47),f'{self.case["attachment_cm"]:g}cm连接点 · {label} · 小车当前{self.cart_velocity[shown,0]:.2f}m/s，沿+X · 风峰值1m/s · t={t:6.1f}/{self.duration:g}s',font=self.font[22],fill=INK)
        rmse='/'.join(f'{v:.3f}' for v in self.case['rmse_3d_h_xy_m'])
        metric='验收计罚RMSE(3D/h/xy)' if self.continuation else '三维/高度/水平RMSE'
        d.text((18,82),f'目标高度{self.protocol["target_aircraft_height_m"]:g}m：{metric}={rmse}m · 验收安全{int(self.case["completed"])}/1 · 种子{self.result["seed_pair"]} · 真值仅离线显示',font=self.font[16],fill=INK)
        self.panel(d,(12,116,884,626),'MuJoCo三维：蓝线为随车坐标完整轨迹；绳加粗、绿十字目标、青箭头风')
        image.paste(wide,(18,153));image.paste(close,(548,409));d=ImageDraw.Draw(image)
        d.rectangle((548,407,874,436),fill='white');d.text((556,411),'机体特写：红机头／绿机体右／蓝推力',font=self.font[16],fill=INK)
        self.panel(d,(894,116,1388,626),'真实姿态与风向：固定航向目标0°')
        for row,(name,val,color) in enumerate(zip(['横滚 φ','俯仰 θ','航向 ψ'],self.rpy[shown],[RED,GREEN,BLUE])):
            d.text((912,156+42*row),name,font=self.font[24],fill=color);d.text((1085,150+42*row),f'{val:+7.2f}°',font=self.font[30],fill=color)
        d.text((912,290),f'环境风 {ws:.3f}m/s；相对气流 {np.linalg.norm(relative_wind):.2f}m/s',font=self.font[20],fill=CYAN)
        d.text((912,324),f'飞机vx={x[3]:.2f}；小车vx={self.cart_velocity[shown,0]:.2f}m/s',font=self.font[20],fill=INK)
        d.text((912,350 if self.continuation else 358),f'x飞机/小车={x[0]:.1f}/{self.cart[shown,0]:.1f}m',font=self.font[18],fill=INK)
        if self.continuation:
            physical='/'.join(f'{v:.3f}' for v in self.case['physical_envelope_rmse_3d_h_xy_m'])
            label='物理轨迹RMSE' if self.case['physical_envelope_completed'] else '物理域计罚RMSE'
            d.text((912,376),f'{label}(3D/h/xy)={physical}m',font=self.font[14],fill=INK)
        center=np.array([1015.,493.]);radius=77
        d.ellipse((*(center-radius),*(center+radius)),outline='#b6c5d4',width=2)
        d.text((1091,482),'+X',font=self.font[16],fill=GRAY);d.text((1002,393),'+Y',font=self.font[16],fill=GRAY)
        nose=R[:2,0];nose=nose/max(np.linalg.norm(nose),1e-9)
        screen_arrow(d,center,center+nose*np.array([1,-1])*70,RED,4,12)
        w=wind[:2]/max(np.linalg.norm(wind[:2]),1e-9)*np.array([1,-1]);screen_arrow(d,center-w*55,center+w*68,CYAN,4,12)
        d.text((1131,430),'红：机头',font=self.font[20],fill=RED);d.text((1131,466),'青：风吹向',font=self.font[20],fill=CYAN)
        d.text((1131,504),'小车沿+X →',font=self.font[18],fill=INK)
        d.text((912,588),f'落后 {-self.error[shown,0]:.2f}m；侧偏 {self.error[shown,1]:+.2f}m；h={-x[2]:.2f}m',font=self.font[18],fill=INK)
        self._plots(d,index,shown)
        if self.continuation and not self.case['completed'] and t>=self.case['first_failure_s'] and self.valid[index]:
            d.rectangle((30,535,850,600),fill='#fff3dc')
            d.text((43,542),f'{self.case["first_failure_s"]:.2f}s已越界（代码{self.case["failure_code"]}），本画面继续记录物理运动',font=self.font[20],fill=ORANGE)
            d.text((43,574),'验收已按失效计罚；继续飞行不能当作验收通过',font=self.font[18],fill=INK)
        if not self.valid[index]:
            d.rectangle((30,515,850,600),fill='#fff0ec')
            failure_s=self.case.get('first_physical_failure_s') if self.continuation else self.case['first_failure_s']
            code=self.case.get('physical_failure_code') if self.continuation else self.case['failure_code']
            d.text((43,525),f'已失效：{failure_s:.2f}s，代码{code}',font=self.font[24],fill=RED)
            d.text((43,562),'后续只计罚；本画面冻结在失效前，不作为后续物理飞行',font=self.font[20],fill=RED)
        return image

    def _plots(self,d,index,shown):
        self.panel(d,(12,640,466,948),'水平轨迹：相对小车；绿十字为目标(0,0)')
        b=(66,693,431,901);span=self.span
        xy=self.axes(d,b,(-span,span),(-span,span));center=xy([0],[0])[0]
        d.line((center[0]-9,center[1],center[0]+9,center[1]),fill=GREEN,width=2)
        d.line((center[0],center[1]-9,center[0],center[1]+9),fill=GREEN,width=2)
        self.series(d,xy,self.error[:shown+1,0],self.error[:shown+1,1],BLUE,3)
        pt=xy([self.error[shown,0]],[self.error[shown,1]])[0];d.ellipse((pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4),fill=BLUE)
        d.text((70,911),f'相对X [−{span:.1f}, +{span:.1f}] m',font=self.font[16],fill=INK)
        d.text((61,671),f'相对Y [−{span:.1f}, +{span:.1f}] m',font=self.font[14],fill=GRAY)
        self.panel(d,(477,640,931,948),'高度与水平偏差：完整时间轴，纵轴从0开始')
        for y0,y1,values,color,upper,label in [(695,775,-self.s[:,2],GREEN,max(6.,self.protocol['target_aircraft_height_m']*1.2),'高度m'),(820,900,np.linalg.norm(self.error[:,:2],axis=-1),BLUE,max(1.,span*1.6),'水平m')]:
            chart=self.axes(d,(533,y0,904,y1),(0,self.duration),(0,upper))
            self.series(d,chart,self.t[:shown+1],values[:shown+1],color)
            d.text((490,y0+8),label,font=self.font[14],fill=color)
            xx=chart([self.t[index]],[0])[0,0];d.line((xx,y0,xx,y1),fill=GRAY,width=1)
            d.text((537,y0-21),f'0–{upper:.1f}',font=self.font[14],fill=GRAY)
        d.text((533,914),f'0                           时间s                        {self.duration:g}',font=self.font[14],fill=GRAY)
        self.panel(d,(942,640,1388,948),'最近20秒姿态曲线（°；真实角度）')
        begin=max(0,shown-200);tt=self.t[begin:shown+1];angles=self.rpy[begin:shown+1]
        bound=max(1.,float(np.abs(angles).max())*1.15)
        chart=self.axes(d,(995,693,1359,900),(max(0,self.t[shown]-20),max(20,self.t[shown])),(-bound,bound))
        for j,color in enumerate([RED,GREEN,BLUE]):self.series(d,chart,tt,angles[:,j],color,2)
        d.text((994,672),f'+/−{bound:.1f}°   红φ  绿θ  蓝ψ',font=self.font[14],fill=INK)
        d.text((1020,915),'时间s；未夸大机体姿态',font=self.font[16],fill=GRAY)



def main():
    from .replay_cases import main as replay_main
    replay_main()


if __name__=='__main__':main()
