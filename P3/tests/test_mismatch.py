"""Nominal/actual mismatch, physical units and hybrid transition contracts."""
import copy,importlib.util,math,unittest
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.model import ExactWingOffBodyDragModel

class MismatchDynamicsTests(unittest.TestCase):
    def module(self):
        name='P3.mismatch_dynamics'
        self.assertIsNotNone(importlib.util.find_spec(name),'Model-mismatch dynamics is missing')
        return __import__(name,fromlist=['HybridDynamics'])
    def model(self,cfg=None,arm=.05):
        return ExactWingOffBodyDragModel(cfg or plant_config(),device='cpu',arms=torch.tensor([[0.,0.,arm]]),lengths=torch.tensor([4.95-arm]))
    def test_matched_parameters_produce_zero_residual_not_total_load(self):
        m=self.module();model=self.model();state=model.initial(1);state[:,0]=.5;wind=torch.tensor([[5.,1.,0.]])
        self.assertTrue(torch.equal(m.wrench_target(model,model,state,wind),torch.zeros(1,6)))
        self.assertGreater(float(model.loads(state,wind)[3]),0.)
    def test_mass_difference_has_force_units(self):
        m=self.module();cfg=plant_config();cfg['airframe']['mass_kg']=2.;cfg['airframe']['gravity_mps2']=10.;cfg['actuators']['kf']=1.;cfg['tether']['enabled']=False
        actual=copy.deepcopy(cfg);actual['airframe']['mass_kg']=4.
        nominal=self.model(cfg);true=self.model(actual);state=nominal.initial(1);state[:,13:17]=1.
        target=m.wrench_target(true,nominal,state,torch.zeros(1,3))
        torch.testing.assert_close(target[:,:3],torch.tensor([[0.,0.,2.]]),rtol=0,atol=1e-6)
    def test_inertia_and_arm_residual_recover_actual_accelerations(self):
        m=self.module();cfg=plant_config();cfg['airframe']['inertia_body_kgm2']=[v*1.05 for v in cfg['airframe']['inertia_body_kgm2']]
        nominal=self.model();actual=self.model(cfg,arm=.1);state=nominal.initial(1);state[:,0]=.8;state[:,10:13]=torch.tensor([[.1,-.2,.05]])
        wind=torch.tensor([[3.,2.,.1]]);wrench=m.wrench_target(actual,nominal,state,wind)
        correction=torch.cat((wrench,torch.zeros(1,1)),-1)
        result=m.HybridDynamics(nominal).mechanical_rhs(state,wind,correction)
        torch.testing.assert_close(result,actual.mechanical_rhs(state,wind),rtol=1e-5,atol=2e-6)
        self.assertGreater(float(wrench[:,3:].norm()),0.)
    def test_drag_mismatch_is_state_dependent(self):
        m=self.module();nominal=self.model();cfg=plant_config();cfg['aero']['body_drag_n_per_mps2']=[.11]*3;actual=self.model(cfg);state=nominal.initial(1)
        target=m.wrench_target(actual,nominal,state,torch.tensor([[5.,0.,0.]]))
        self.assertAlmostEqual(float(target[0,0]),1.375,places=5)
        still=m.wrench_target(actual,nominal,state,torch.zeros(1,3))
        self.assertAlmostEqual(float(still[0,0]),0.,places=6)
    def test_motor_lag_correction_changes_dynamic_response(self):
        m=self.module();nominal=self.model();cfg=plant_config();cfg['actuators']['motor_tau_s']*=2;actual=self.model(cfg)
        state=nominal.initial(1);state[:,13:17]=0.;command=torch.tensor([[.5,.5,.5,.5,0.,0.]])
        correction=torch.zeros(1,7);correction[:,6]=math.log(2.)
        predicted=m.HybridDynamics(nominal).actuator_at(state,command,.002,correction)[0]
        expected=actual.actuator_at(state,command,.002)[0]
        torch.testing.assert_close(predicted,expected,rtol=1e-6,atol=1e-4)
        self.assertTrue((predicted<nominal.actuator_at(state,command,.002)[0]).all())
    def test_frozen_network_and_hybrid_input_gradients(self):
        m=self.module();net=m.DynamicsResidual({'width':32});obs=torch.randn(2,35,requires_grad=True);hidden=torch.zeros(2,128);context=torch.zeros(2,14);commands=torch.full((2,4),.5,requires_grad=True)
        self.assertTrue(torch.equal(net(obs,hidden,context,commands),torch.zeros(2,7)))
        with torch.no_grad():net.output.weight.fill_(.01)
        net.requires_grad_(False);net(obs,hidden,context,commands).sum().backward()
        self.assertGreater(float(commands.grad.norm()),0.);self.assertGreater(float(obs.grad.norm()),0.);self.assertTrue(all(p.grad is None for p in net.parameters()))
        model=self.model();correction=torch.zeros(1,7,requires_grad=True);state=model.initial(1);u=torch.tensor([[.5,.5,.5,.5,0.,0.]])
        result=m.HybridDynamics(model).control(state,u,torch.zeros(1,5,3),correction)
        result[:,3:17].sum().backward();self.assertTrue(torch.isfinite(correction.grad).all());self.assertGreater(float(correction.grad.norm()),0.)

class MismatchDataTests(unittest.TestCase):
    def module(self):
        name='P3.mismatch_data'
        self.assertIsNotNone(importlib.util.find_spec(name),'Mismatch data protocol is missing')
        return __import__(name,fromlist=['validate_protocol'])
    def test_seed_sets_and_final_seeds_cannot_overlap(self):
        m=self.module();p={'splits':{'train':[{'seed':10}],'development':[{'seed':20}],'test':[{'seed':30}]},'control_training_seed':40,'control_evaluation_seed':50}
        m.validate_protocol(p,{99,100})
        q=copy.deepcopy(p);q['splits']['test'][0]['seed']=10
        with self.assertRaises(ValueError):m.validate_protocol(q,{99,100})
        with self.assertRaises(ValueError):m.validate_protocol(p,{20,21})
    def test_nominal_arm_is_external_and_does_not_change_actual_cases(self):
        m=self.module();cases=[{'attachment':2},{'attachment':1}]
        arms,lengths=m.nominal_geometry(cases,5,'cpu')
        self.assertEqual([round(float(v),2) for v in arms[:,2]],[.05,.05])
        self.assertEqual([c['attachment'] for c in cases],[2,1])
        self.assertEqual(lengths.shape,(2,))
        with self.assertRaises(ValueError):m.nominal_geometry(cases,7.5,'cpu')
    def test_windows_do_not_cross_failure_or_end(self):
        m=self.module();r={'origins':torch.tensor([0,2,4]),'obs':torch.zeros(3,2,35),'hidden':torch.zeros(3,2,128),'context':torch.zeros(2,14),'states':torch.arange(6.).view(6,1,1).expand(6,2,19).clone(),'packets':torch.zeros(6,2,15),'commands':torch.arange(5.).view(5,1,1).expand(5,2,4).clone(),'alive':torch.ones(6,2,dtype=torch.bool),'targets':torch.zeros(3,2,6),'wind':torch.zeros(6,2,3)}
        r['alive'][3:,1]=False;b=m.windows(r,2)
        self.assertEqual(len(b['obs']),3);self.assertEqual(b['commands'][0,:,0].tolist(),[0.,1.]);self.assertEqual(b['future_states'][0,:,0].tolist(),[1.,2.])
        self.assertEqual(b['origin'].tolist(),[0,0,2])

if __name__=='__main__':unittest.main()
