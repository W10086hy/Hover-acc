import copy,unittest
import torch
from P2.hover_bptt.io import read,load_checkpoint
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.flexible_rope import FlexibleRopeModel
from P2.hover_bptt.actor import Actor
from P3.flexible_domains import ROOT,load_protocol,build_domain
from P3.flexible_compensation import FlexibleCompensationEnvironment
from P3.mismatch_dynamics import wrench_target,DynamicsResidual,HybridDynamics
from P3.train_mismatch import fit_loss
from P3.flexible_labels import structural_wrench_target
from P2.hover_bptt.dynamics.torch_model import rotations,mv


class FlexibleResidualTests(unittest.TestCase):
    def test_common_traction_isolates_attachment_torque_without_false_vertical_force(self):
        p,_,_=load_protocol(ROOT/'flexible26_structural_protocol.json')
        *_,model=build_domain(p['splits']['train'][0],'cpu',p['rope'])
        x=model.initial(4);x[:,0]=.6;wind=torch.ones(4,3)
        nominal=FlexibleRopeModel(model.cfg if hasattr(model,'cfg') else plant_config(),device='cpu',
            arms=model.arms.new_tensor([[0.,0.,.05]]).expand(4,-1),lengths=model.lengths.new_full((4,),4.90),rope=p['rope'])
        target=structural_wrench_target(model,nominal,x,wind)
        physical_force=mv(rotations(x[:,6:10]).transpose(1,2),model.rope_loads(x,wind)['endpoint_force'])
        expected_torque=torch.linalg.cross(model.arms-nominal.arms,physical_force)
        self.assertTrue(torch.allclose(target[:,:3],torch.zeros(4,3),atol=3e-5,rtol=0))
        self.assertTrue(torch.allclose(target[:,3:],expected_torque,atol=1e-6,rtol=0))

    def test_matched_flexible_loads_cancel_and_true_parameter_changes_create_labels(self):
        p,saved,sha=load_protocol(ROOT/'flexible26_structural_protocol.json')
        base,actual,nominal,cases,pc,model=build_domain(p['splits']['train'][0],'cpu',p['rope'])
        x=model.initial(len(cases));wind=torch.full((len(cases),3),3.)
        matched=FlexibleRopeModel(pc,device='cpu',arms=model.arms.clone(),lengths=model.lengths.clone(),rope=p['rope'])
        self.assertTrue(torch.equal(wrench_target(matched,model,x,wind),torch.zeros(len(cases),6)))
        changed=actual.scaled(mass=1.05,inertia=.95,motor_tau=1.3).apply(base)
        wrong=FlexibleRopeModel(changed,device='cpu',arms=model.arms.clone(),lengths=model.lengths.clone(),rope=p['rope'])
        wrong.body_drag*=1.15
        self.assertGreater(float(wrench_target(wrong,model,x,wind).abs().max()),.1)
        self.assertEqual(x.shape[-1],86)

    def test_zero_residual_preserves86state_and_nominal_controller(self):
        p,saved,sha=load_protocol(ROOT/'flexible26_structural_protocol.json')
        base,actual,nominal,cases,pc,model=build_domain(p['splits']['train'][1],'cpu',p['rope']);cases=cases[:1]
        args=dict(n=1,seed=20689901,wind_seed=20689902,device='cpu',stage=copy.deepcopy(saved['configuration']),episode_s=.03,
            specifications=cases,vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,rope=p['rope'])
        normal=FlexibleCompensationEnvironment(**args);net=DynamicsResidual(p['model']);net.initialize_history(saved['actor'])
        options=dict(p['compensation']);options.update(warmup_s=0,ramp_s=0,filter_tau_s=0)
        comp=FlexibleCompensationEnvironment(**args,residual_net=net,compensation=options)
        actor=Actor(saved['configuration']['model']);actor.load_state_dict(saved['actor']);actor.eval();a=b=None
        with torch.no_grad():
            for _ in range(3):
                u,a=actor(normal.obs['policy'],a,normal.parameter_context);v,b=actor(comp.obs['policy'],b,comp.parameter_context)
                normal.step(u);comp.step(v)
                self.assertTrue(torch.equal(normal.state,comp.state));self.assertTrue(torch.equal(normal.obs['policy'],comp.obs['policy']))
        self.assertEqual(comp.state.shape[-1],86)

    def test_substep_motor_loss_uses_the_short_response_target(self):
        p,saved,sha=load_protocol(ROOT/'flexible26_structural_protocol.json')
        *_,model=build_domain(p['splits']['train'][1],'cpu',p['rope']);x=model.initial(8)[:2]
        net=DynamicsResidual(p['model']);commands=torch.full((2,1,4),.4)
        full=torch.cat((commands[:,0],torch.zeros(2,2)),-1);zero=torch.zeros(2,7)
        target=HybridDynamics(model).actuator_at(x,full,.002,zero)[0]
        batch=dict(obs=torch.zeros(2,35),hidden=torch.zeros(2,128),context=torch.zeros(2,14),commands=commands,
            targets=torch.zeros(2,6),initial_state=x,future_states=torch.zeros(2,1,86),motor_response_target=target)
        _,_,motor=fit_loss(net,model,batch,p['residual_training'])
        self.assertEqual(float(motor),0.)


if __name__=='__main__':unittest.main()
