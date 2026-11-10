"""Frozen policy residual feedforward, not policy training."""
import copy,tempfile,unittest,math
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from P3.compensation import ResidualFeedforwardInterface,CompensationEnvironment,compensate_motor_response
from P2.hover_bptt.controller import HeadingHoldInterface
from P2.hover_bptt.environment import RateEnvironment,plant_config
from P2.hover_bptt.onboard import NominalVehicle,Sensors
from P3.mismatch_dynamics import DynamicsResidual
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.io import read
ROOT=Path(__file__).resolve().parents[1]
P2_ROOT=ROOT.parent/'P2'
class ConstantResidual(nn.Module):
    def __init__(self,values):super().__init__();self.register_buffer('values',torch.tensor(values))
    def encode(self,obs,hidden):return hidden[0],hidden
    def forward(self,obs,history,context,commands):return self.values.expand(len(obs),-1)
class CompensationTests(unittest.TestCase):
    def setup_case(self):
        saved=torch.load(P2_ROOT/'tests/fixtures/conditioned21.pt',map_location='cpu',weights_only=False)
        vehicle=NominalVehicle(plant_config(),'cpu');settings=saved['configuration']['controller'];base=HeadingHoldInterface(vehicle,1,settings)
        speed=math.sqrt(base.hover_thrust/(4*vehicle.kf))
        packet=Sensors(torch.tensor([[1.,0,0,0]]),torch.tensor([[.2,0,0]]),torch.zeros(1,3),torch.full((1,4),speed),torch.tensor([5.]))
        return saved,vehicle,base,packet
    def options(self):
        return dict(warmup_s=0,ramp_s=0,filter_tau_s=0,thrust_limit_n=2,torque_limits_nm=[.03,.03,.02],log_motor_tau_limit=.7)
    def test_force_torque_sign_and_single_integrator_commit(self):
        saved,v,base,packet=self.setup_case();control=ResidualFeedforwardInterface(base,ConstantResidual([0,0,.5,.001,-.002,.003,0]),self.options())
        control.set_observation(torch.zeros(1,35),torch.zeros(1,14));motors,info=control.apply(packet,torch.zeros(1,4),.01)
        self.assertAlmostEqual(float(info['total_thrust_reference_n'][0]),base.hover_thrust+.5,places=5)
        self.assertTrue(torch.allclose(info['applied_wrench_feedforward'],torch.tensor([[.5,-.001,.002,-.003]])))
        self.assertAlmostEqual(float(control.integral[0,0]),-.002,places=7)
        self.assertAlmostEqual(float(control.thrust_integral[0]),.005,places=6)
        self.assertTrue(bool(((motors>=0)&(motors<=1)).all()))
    def test_zero_residual_preserves_closed_loop_bitwise(self):
        saved,_,_,_=self.setup_case();cfg=saved['configuration'];cases=cases_at([0])[:1]
        normal=RateEnvironment(1,20268901,wind_seed=20268902,device='cpu',stage=copy.deepcopy(cfg),episode_s=.3,specifications=cases,capture=False,auto_reset=False)
        comp=CompensationEnvironment(1,20268901,wind_seed=20268902,device='cpu',stage=copy.deepcopy(cfg),episode_s=.3,specifications=cases,capture=False,auto_reset=False,residual_net=DynamicsResidual(read(ROOT/'p3_mismatch_protocol.json')['model']),compensation=self.options())
        actor=Actor(cfg['model']);actor.load_state_dict(saved['actor']);actor.eval();h1=h2=None
        with torch.no_grad():
            for _ in range(30):
                a1,h1=actor(normal.obs['policy'],h1,normal.parameter_context);a2,h2=actor(comp.obs['policy'],h2,comp.parameter_context)
                normal.step(a1);comp.step(a2)
                self.assertTrue(torch.equal(normal.state,comp.state));self.assertTrue(torch.equal(normal.obs['policy'],comp.obs['policy']))
                self.assertTrue(torch.equal(normal.controller.integral,comp.controller.integral))
                self.assertTrue(torch.equal(normal.controller.thrust_integral,comp.controller.thrust_integral))
    def test_motor_inverse_matches_nominal_endpoint(self):
        _,v,_,_=self.setup_case();command=torch.full((1,4),.5);measured=torch.full((1,4),400.);log_tau=torch.tensor([.3]);dt=.01
        adjusted=compensate_motor_response(v,command,measured,log_tau,dt)
        speed=lambda u:v.pa*u.square()+v.pb*u+v.pc
        nominal=measured*math.exp(-dt/v.motor_tau)+speed(command)*(1-math.exp(-dt/v.motor_tau))
        decay=torch.exp(-dt/(v.motor_tau*log_tau.exp()))[:,None]
        actual=measured*decay+speed(adjusted)*(1-decay)
        self.assertTrue(torch.allclose(actual,nominal,atol=1e-4,rtol=1e-6))
        self.assertTrue(torch.equal(compensate_motor_response(v,command,measured,torch.zeros(1),dt),command))
    def test_nonfinite_network_falls_back_and_large_output_is_bounded(self):
        _,v,base,packet=self.setup_case();reference=copy.deepcopy(base);want,_=reference.apply(packet,torch.zeros(1,4),.01)
        c=ResidualFeedforwardInterface(base,ConstantResidual([0,0,float('nan'),1,2,3,0]),self.options());c.set_observation(torch.zeros(1,35),torch.zeros(1,14));got,info=c.apply(packet,torch.zeros(1,4),.01)
        self.assertTrue(torch.equal(got,want));self.assertTrue(bool(info['compensation_fallback'][0]))
        _,_,base,packet=self.setup_case();c=ResidualFeedforwardInterface(base,ConstantResidual([5,5,100,5,-5,5,5]),self.options());c.set_observation(torch.zeros(1,35),torch.zeros(1,14));_,info=c.apply(packet,torch.zeros(1,4),.01)
        self.assertTrue(torch.allclose(info['applied_wrench_feedforward'],torch.tensor([[2.,-.03,.03,-.02]])))
    def test_invalid_sensor_packet_disables_all_feedforward(self):
        _,_,base,packet=self.setup_case();bad=Sensors(packet.attitude,packet.gyro,packet.specific_force,torch.full_like(packet.motor_speed,float('nan')),packet.baro_height)
        reference=copy.deepcopy(base);want,_=reference.apply(bad,torch.zeros(1,4),.01)
        c=ResidualFeedforwardInterface(base,ConstantResidual([0,0,.5,.001,0,0,.2]),self.options());c.set_observation(torch.zeros(1,35),torch.zeros(1,14));got,info=c.apply(bad,torch.zeros(1,4),.01)
        self.assertTrue(torch.equal(got,want));self.assertTrue(bool(info['compensation_fallback'][0]))
    def test_frozen_checkpoint_comparison_entry(self):
        from P3 import evaluate_compensation as entry
        protocol=copy.deepcopy(read(ROOT/'p3_mismatch_protocol.json'));protocol['development_comparison_duration_s']=.02;protocol['control_evaluation_directions_deg']=[0]
        def read_protocol(file):return copy.deepcopy(protocol) if Path(file).name=='p3_mismatch_protocol.json' else read(file)
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as tmp:
            out=Path(tmp)/'comparison';residual=ROOT/'tests/fixtures/residual21.pt'
            with patch.object(entry,'read',side_effect=read_protocol),patch('sys.argv',['evaluate_compensation','--residual',str(residual),'--output',str(out),'--device','cpu']):entry.main()
            report=read(out/'summary.json');self.assertTrue(report['completed']);self.assertEqual(report['duration_s'],.02);self.assertEqual(report['policy_updates'],0)
            self.assertEqual(len({x['checkpoint_sha256'] for x in report['policies']}),1)
            self.assertTrue(report['checkpoint_and_adam_file_unchanged']);self.assertTrue(report['residual_unchanged'])
if __name__=='__main__':unittest.main()
