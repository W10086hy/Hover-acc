import unittest
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt import model


class FlexibleRopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make(self, segments=6, **options):
        self.assertTrue(hasattr(model, 'FlexibleRopeModel'), 'Differentiable distributed rope required')
        cfg=plant_config()
        return model.FlexibleRopeModel(cfg, device='cpu', dtype=torch.float64,
            arms=[[0.,0.,.05]], lengths=[4.90], rope=dict(segments=segments, **options))

    def test_mass_and_unilateral_segment_force(self):
        m=self.make(); self.assertAlmostEqual(float(m.node_mass.sum()), .049, places=10)
        points=torch.tensor([[[0.,0.,0.],[.5,0.,0.],[1.,0.,0.],[1.5,0.,0.],
                              [2.,0.,0.],[2.5,0.,0.],[3.,0.,0.]]], dtype=torch.float64)
        load=m.segment_loads(points, torch.zeros_like(points))
        self.assertTrue(torch.equal(load['tension'], torch.zeros(1,6,dtype=torch.float64)))
        self.assertTrue(torch.isfinite(load['stiffness']).all())

    def test_endpoint_force_and_moment_follow_last_segment(self):
        m=self.make(); x=m.initial(1)
        q,v=m.nodes(x); q=q.clone(); q[:,-1,0]=.15
        x=m.with_nodes(x,q,v)
        detail=m.rope_loads(x, torch.zeros(1,3,dtype=x.dtype))
        F=detail['endpoint_force']; tau=detail['endpoint_torque']
        self.assertGreater(abs(float(F[0,0])), .01)
        torch.testing.assert_close(tau, torch.tensor([[-.05*float(F[0,1]), .05*float(F[0,0]),0.]],dtype=x.dtype))
        # All segment endpoint forces, including anchor/aircraft, sum to zero.
        torch.testing.assert_close(detail['nodal_internal_force'].sum(1), torch.zeros_like(F),atol=1e-12,rtol=0)

    def test_horizontal_held_rope_sags_under_gravity(self):
        m=self.make(); body=torch.zeros(1,19,dtype=torch.float64);body[:,0]=4.95;body[:,6]=1
        x=m.initialize_state(body)
        q,_=m.nodes(x)
        self.assertGreater(float(q[0,:,2].max()), .05)
        self.assertLess(float(m.static_residual(x).norm()), 1e-5)

    def test_coupled_gradient_matches_finite_difference(self):
        m=self.make(segments=3);x=m.initial(1).detach().requires_grad_()
        u=x.new_tensor([[.45,.45,.45,.45,0,0]]); w=x.new_tensor([[3.,1.,0.]])
        def f(a):
            y=m.control(a,u,w[:,None].expand(-1,m.substeps,-1))
            # Only aircraft outputs: node gradients must pass through coupling.
            return y[:,0].sum()+.03*y[:,11].sum()
        g=torch.autograd.grad(f(x),x)[0]
        for k in [0,7,10,19,22]:
            a=x.detach().clone();b=a.clone();a[0,k]+=1e-5;b[0,k]-=1e-5
            fd=(f(a)-f(b))/2e-5
            self.assertAlmostEqual(float(g[0,k]),float(fd),delta=2e-5+2e-3*abs(float(fd)))
        self.assertGreater(float(g[:,19:-1].abs().sum()),1e-5)

    def test_invalid_parameters_and_collapsed_segments(self):
        for opts in [dict(segments=1),dict(linear_density_kgpm=0),dict(axial_rigidity_n=-1),dict(air_drag_coefficient=float('nan'))]:
            with self.assertRaises(ValueError):self.make(**opts)
        m=self.make();p=torch.zeros(1,7,3,dtype=torch.float64)
        loads=m.segment_loads(p,p)
        self.assertTrue(all(torch.isfinite(value).all() for value in loads.values()))

    def test_implicit_step_cannot_hide_compressive_segment_impulses(self):
        m=self.make(segments=12);body=torch.zeros(1,19,dtype=torch.float64)
        body[:,2]=-4.9*(1+1e-6)-.05;body[:,6]=1
        body[:,13:17]=(m.mass*m.gravity/(4*m.kf))**.5
        end=body[:,:3]+body.new_tensor([[0.,0.,.05]])
        fractions=torch.arange(1,12,dtype=body.dtype)/12
        q=end[:,None]*fractions[None,:,None]
        x=torch.cat((body,q.flatten(1),torch.zeros_like(q).flatten(1),body.new_zeros(1,1)),1)
        u=torch.cat((m.throttle_from_speed(body[:,13:17]),body.new_zeros(1,2)),1)
        y=m.forward(x,u,body.new_zeros(1,3));_,v=m.nodes(y)
        # Infer actual applied tensions from momentum, independently of loads().
        tension=m.mass*y[:,5]/m.dt
        inferred=[tension]
        for i in reversed(range(m.free_nodes)):
            force=m.node_mass[:,i]*(v[:,i,2]/m.dt-m.gravity)
            tension=tension+force;inferred.append(tension)
        self.assertGreaterEqual(float(torch.stack(inferred).min()),-1e-8)

    def test_active_set_residual_with_independent_node_velocities(self):
        m=self.make(segments=12);x=m.initial(1).repeat(256,1)
        torch.manual_seed(1234);q,v=m.nodes(x);x=m.with_nodes(x,q,.15*torch.randn_like(v))
        x[:,3:6]=.15*torch.randn(256,3,dtype=x.dtype)
        wind=x.new_zeros(256,3);loads=m.rope_loads(x,wind)
        force=torch.cat((x.new_zeros(256,6),loads['node_external_force'].flatten(1)),1)
        _,tension,residual=m.velocity_step(x,force,wind)
        self.assertGreaterEqual(float(tension.min()),0.)
        self.assertLess(float(residual.max()),1e-8)

    def test_timestep_refinement_reduces_error(self):
        models=[]
        for dt in [.002,.001,.0005]:
            m=self.make(segments=6);m.dt=dt;m.substeps=round(.01/dt);models.append(m)
        initial=models[0].initial(1);states=[]
        for m in models:
            x=initial.clone();u=x.new_tensor([[.45,.45,.45,.45,0,0]])
            for _ in range(10):x=m.control(x,u,x.new_tensor([[[3.,1.,0.]]]).expand(-1,m.substeps,-1))
            states.append(x)
        coarse=(states[0][:,:3]-states[2][:,:3]).norm()
        fine=(states[1][:,:3]-states[2][:,:3]).norm()
        self.assertLess(float(fine),float(coarse)*.8)
        self.assertLess(float(coarse),.003)


if __name__=='__main__':unittest.main()
