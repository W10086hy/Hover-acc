"""Distributed tension-only cable and coupled, linearly implicit Torch plant.

World z points down. Endpoints are massless; endpoint half-cell masses are
assigned to the neighbouring free nodes. No bending, twist or contact model.
All default cable material values are unmeasured simulation assumptions.
"""
import math
import torch
from .model import ExactWingOffBodyDragModel
from .dynamics.torch_model import rotations, mv, qdot


DEFAULT_ROPE = dict(segments=12, linear_density_kgpm=.01, axial_rigidity_n=49.,
    axial_viscosity_ns=9.8, diameter_m=.003, air_drag_coefficient=1.2)


def skew(a):
    x,y,z=a.unbind(-1);o=torch.zeros_like(x)
    return torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(-1,3,3)


class FlexibleRopeModel(ExactWingOffBodyDragModel):
    def __init__(self,cfg,device='cuda',dtype=torch.float32,arms=None,lengths=None,rope=None):
        super().__init__(cfg,device=device,dtype=dtype,arms=arms,lengths=lengths)
        self.rope_options={**DEFAULT_ROPE,**(rope or {})}
        if set(self.rope_options)!=set(DEFAULT_ROPE):raise ValueError('Unknown flexible rope parameter')
        n=self.rope_options['segments']
        if isinstance(n,bool) or int(n)!=n or not 2<=n<=64:raise ValueError('Use 2..64 rope segments')
        for key,value in self.rope_options.items():
            if not math.isfinite(float(value)) or float(value)<=0:raise ValueError('Positive finite rope parameters required: '+key)
        if not self.tether_enabled:raise ValueError('Flexible rope requires enabled tether')
        if self.dt<=0 or self.control_dt<=0 or not math.isclose(self.dt*self.substeps,self.control_dt,abs_tol=1e-12):
            raise ValueError('Positive physics dt must divide control dt')
        self.tether_enabled=False  # Never apply the legacy straight spring as well.
        self.segments=int(n);self.free_nodes=int(n)-1;self.state_width=20+6*self.free_nodes
        def buf(name,value):self.register_buffer(name,torch.as_tensor(value,device=device,dtype=dtype))
        if self.lengths.ndim==0:self.lengths=self.lengths.reshape(1)
        if not bool(torch.isfinite(self.lengths).all()) or bool((self.lengths<=0).any()):raise ValueError('Positive rope lengths required')
        buf('segment_rest',self.lengths[:,None].expand(-1,n)/n)
        weights=torch.ones(self.free_nodes,device=device,dtype=dtype)
        weights[0]+=.5;weights[-1]+=.5
        buf('node_mass',self.rope_options['linear_density_kgpm']*self.lengths[:,None]*weights[None]/n)
        incidence=torch.zeros(n,self.free_nodes,device=device,dtype=dtype)
        for i in range(n):
            if i>0:incidence[i,i-1]-=1
            if i<self.free_nodes:incidence[i,i]+=1
        buf('internal_map',torch.kron(incidence,torch.eye(3,device=device,dtype=dtype)).reshape(n,3,-1))
        buf('eye3',torch.eye(3,device=device,dtype=dtype))
        buf('gravity_vector',[0.,0.,self.gravity])

    def nodes(self,x):
        cut=19+3*self.free_nodes
        return x[:,19:cut].reshape(-1,self.free_nodes,3),x[:,cut:cut+3*self.free_nodes].reshape(-1,self.free_nodes,3)

    def with_nodes(self,x,positions,velocities):
        return torch.cat((x[:,:19],positions.flatten(1),velocities.flatten(1),x[:,-1:]),-1)

    def points(self,x):
        q,v=self.nodes(x);r=rotations(x[:,6:10]);arm=self.arms.expand(len(x),3)
        endpoint=x[:,:3]+mv(r,arm)
        velocity=x[:,3:6]+mv(r,torch.linalg.cross(x[:,10:13],arm))
        return (torch.cat((self.anchor.expand(len(x),1,3),q,endpoint[:,None]),1),
                torch.cat((torch.zeros_like(velocity[:,None]),v,velocity[:,None]),1))

    def segment_loads(self,points,velocities):
        d=points[:,1:]-points[:,:-1];l=d.norm(dim=-1).clamp_min(1e-12);n=d/l[:,:,None]
        rate=((velocities[:,1:]-velocities[:,:-1])*n).sum(-1)
        k=self.rope_options['axial_rigidity_n']/self.segment_rest
        c=self.rope_options['axial_viscosity_ns']/self.segment_rest
        elastic=(k*(l-self.segment_rest)).clamp_min(0.)
        total=elastic+c*rate
        active=(l>=self.segment_rest)&(total>0)
        tension=torch.where(active,total,torch.zeros_like(total))
        nn=n[:,:,:,None]*n[:,:,None,:]
        stiffness=(k[:,:,None,None]*nn+(elastic/l)[:,:,None,None]*(self.eye3-nn))*active[:,:,None,None]
        damping=c[:,:,None,None]*nn*active[:,:,None,None]
        return dict(tension=tension,direction=n,stiffness=stiffness,damping=damping,
                    elastic=elastic,length=l)

    def rope_loads(self,x,wind):
        points,velocities=self.points(x);s=self.segment_loads(points,velocities)
        traction=s['tension'][:,:,None]*s['direction']
        nodal=torch.cat((traction[:,:1],traction[:,1:]-traction[:,:-1],-traction[:,-1:]),1)
        F=nodal[:,-1];r=rotations(x[:,6:10]);arm=self.arms.expand(len(x),3)
        torque=torch.linalg.cross(arm,mv(r.transpose(1,2),F))
        # Tributary lengths also include the redistributed endpoint half cells.
        tributary=self.node_mass/self.rope_options['linear_density_kgpm']
        tangent=s['direction'][:,:-1]+s['direction'][:,1:]
        tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        relative=velocities[:,1:-1]-wind[:,None]
        axial=(relative*tangent).sum(-1,keepdim=True)*tangent;normal=relative-axial
        drag=-.5*self.rho*self.rope_options['air_drag_coefficient']*self.rope_options['diameter_m']*tributary[:,:,None]*(
            normal.norm(dim=-1,keepdim=True)*normal+.02*axial.norm(dim=-1,keepdim=True)*axial)
        return dict(**s,endpoint_force=F,endpoint_torque=torque,nodal_internal_force=nodal,
                    node_external_force=self.node_mass[:,:,None]*self.gravity_vector+drag)

    def loads(self,x,wind):
        F,tau,specific,_=super().loads(x,wind);s=self.rope_loads(x,wind)
        return (F+s['endpoint_force'],tau+s['endpoint_torque'],
                specific+mv(rotations(x[:,6:10]).transpose(1,2),s['endpoint_force'])/self.mass,
                s['tension'].amax(-1))

    def mapping(self,x):
        batch=len(x);r=rotations(x[:,6:10]);H=-r@skew(self.arms.expand(batch,3))
        body=torch.cat((torch.zeros(batch,self.segments-1,3,6,device=x.device,dtype=x.dtype),
                        torch.cat((self.eye3.expand(batch,3,3),H),-1)[:,None]),1)
        return torch.cat((body,self.internal_map.expand(batch,-1,-1,-1)),-1)

    @staticmethod
    def assemble(B,tangent):
        return torch.einsum('bsai,bsaj->bij',B,torch.einsum('bsac,bscj->bsaj',tangent,B))

    def velocity_step(self,x,force,wind,*,load_data=None):
        """Backward axial constitutive law with nonnegative tensile impulses.

        A small cable-space complementarity problem replaces the full-space
        linear solve. Bounded primal/dual active-set refinements are differentiable
        within an active set. Final projection guarantees that every impulse
        applied to the coupled bodies is tensile, even at an active-set switch.
        Geometric direction is frozen over one substep (first-order method).
        """
        h=self.dt;_,v=self.nodes(x)
        s=self.rope_loads(x,wind) if load_data is None else load_data
        B=self.mapping(x)
        C=(B*s['direction'][:,:,:,None]).sum(2)
        velocity=torch.cat((x[:,3:6],x[:,10:13],v.flatten(1)),-1)
        mass=torch.cat((x.new_full((len(x),3),self.mass),self.inertia.expand(len(x),3),
                        self.node_mass.expand(len(x),-1).repeat_interleave(3,-1)),-1)
        free=velocity+h*force/mass
        k=self.rope_options['axial_rigidity_n']/self.segment_rest
        c=self.rope_options['axial_viscosity_ns']/self.segment_rest*(s['length']>=self.segment_rest)
        alpha=h*k+c
        A=torch.diag_embed(1/alpha)+h*torch.bmm(C/mass[:,None],C.transpose(1,2))
        b=k*(s['length']-self.segment_rest)/alpha+torch.bmm(C,free[:,:,None]).squeeze(-1)
        active=torch.ones_like(b,dtype=torch.bool);diagonal=A.diagonal(dim1=-2,dim2=-1)
        for _ in range(self.segments+2):
            mask=active.to(x.dtype)
            matrix=A*mask[:,:,None]*mask[:,None,:]+torch.diag_embed(1-mask)
            tension=torch.linalg.solve_ex(matrix,(b*mask)[:,:,None],check_errors=False)[0].squeeze(-1)
            residual=torch.bmm(A,tension[:,:,None]).squeeze(-1)-b
            active=(tension-residual/diagonal)>0
        tension=tension.clamp_min(0.)
        result=free-h*torch.bmm(C.transpose(1,2),tension[:,:,None]).squeeze(-1)/mass
        residual=torch.bmm(A,tension[:,:,None]).squeeze(-1)-b
        complementarity=torch.minimum(tension,residual).abs().amax(-1)
        return result,tension,complementarity

    def forward(self,x,u,wind):
        h=self.dt;q,v=self.nodes(x)
        mid,dm=self.actuator_at(x,u,h/2);end,de=self.actuator_at(x,u,h)
        stage=torch.cat((x[:,:13],mid,dm,x[:,19:]),-1)
        F,tau,_,_=super().loads(stage,wind)
        s=self.rope_loads(x,wind)
        forces=torch.cat((F,tau-torch.linalg.cross(x[:,10:13],x[:,10:13]*self.inertia),
                          s['node_external_force'].flatten(1)),-1)
        velocity,applied_tension,_=self.velocity_step(x,forces,wind,load_data=s)
        attitude=x[:,6:10]+h*qdot(x[:,6:10],velocity[:,3:6])
        attitude=attitude/attitude.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        q=q+h*velocity[:,6:].reshape_as(q)
        body=torch.cat((x[:,:3]+h*velocity[:,:3],velocity[:,:3],attitude,velocity[:,3:6],end,de),-1)
        y=torch.cat((body,q.flatten(1),velocity[:,6:],x[:,-1:]),-1)
        # Diagnostics are carried in state, never a mutable graph-replay cache.
        with torch.no_grad():
            peak=torch.maximum(x[:,-1],torch.maximum(applied_tension.amax(-1),torch.maximum(s['tension'].amax(-1),self.rope_loads(y,wind)['tension'].amax(-1))))
        return torch.cat((y[:,:-1],peak[:,None]),-1)

    def control(self,x,u,winds):
        x=torch.cat((x[:,:-1],torch.zeros_like(x[:,-1:])),1)
        for i in range(self.substeps):x=self.forward(x,u,winds[:,i])
        return x

    def static_residual(self,x):
        return self.rope_loads(x,torch.zeros_like(x[:,:3]))['nodal_internal_force'][:,1:-1]+self.node_mass[:,:,None]*self.gravity_vector

    def initialize_state(self,body):
        # Held-end static energy minimisation. This is an initial condition,
        # deliberately outside the time-dependent BPTT computation graph.
        with torch.no_grad():
            endpoint=body[:,:3]+mv(rotations(body[:,6:10]),self.arms.expand(len(body),3))
            d=endpoint-self.anchor;direction=d/d.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            down=body.new_tensor([0.,0.,1.]);normal=down-(direction*down).sum(-1,keepdim=True)*direction
            fraction=torch.arange(1,self.segments,device=body.device,dtype=body.dtype)/self.segments
            q=self.anchor+d[:,None]*fraction[None,:,None]+.2*normal[:,None]*fraction[None,:,None]*(1-fraction[None,:,None])
            x=torch.cat((body,q.flatten(1),torch.zeros_like(q).flatten(1),body.new_zeros(len(body),1)),1)
            B=self.internal_map.expand(len(body),-1,-1,-1)
            def energy(a):
                p,_=self.points(a);length=(p[:,1:]-p[:,:-1]).norm(dim=-1)
                strain=(length-self.segment_rest).clamp_min(0.)
                q,_=self.nodes(a)
                return (.5*self.rope_options['axial_rigidity_n']/self.segment_rest*strain.square()).sum(-1)-(q*self.gravity_vector*self.node_mass[:,:,None]).sum((1,2))
            for _ in range(60):
                residual=self.static_residual(x).flatten(1)
                if float(residual.norm(dim=-1).max())<(1e-8 if body.dtype==torch.float64 else 1e-4):break
                s=self.rope_loads(x,torch.zeros_like(body[:,:3]));K=self.assemble(B,s['stiffness'])
                K=K+torch.eye(3*self.free_nodes,device=body.device,dtype=body.dtype)*1e-5
                delta=torch.linalg.solve(K,residual[:,:,None]).squeeze(-1).reshape_as(q)
                delta=delta/(delta.norm(dim=-1).amax(-1)/.5).clamp_min(1.)[:,None,None]
                previous=energy(x);accepted=torch.zeros(len(body),dtype=torch.bool,device=body.device)
                original=x;q,v=self.nodes(x)
                for trial in range(12):
                    candidate=self.with_nodes(original,q+delta*(.5**trial),v)
                    okay=(energy(candidate)<=previous)&~accepted
                    x=torch.where(okay[:,None],candidate,x);accepted=accepted|okay
            return x

    def initial(self,n):
        return self.initialize_state(super().initial(n))
