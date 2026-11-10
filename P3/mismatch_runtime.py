"""Frozen mismatch dynamics in the BPTT plant; the P2 Actor is unchanged."""
import copy,hashlib
import torch
from torch.autograd.function import once_differentiable
from tensordict import TensorDict
from P2.hover_bptt.environment import DifferentiableRateEnvironment
from P2.hover_bptt.onboard import Sensors
from .mismatch_dynamics import HybridDynamics
from .mismatch_data import nominal_geometry
from P2.hover_bptt.vehicle_spec import identity
from P2.hover_bptt.cuda_bptt_kernel import RecurrentPlantGraph


class _HybridStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx,kernel,*inputs):
        ctx.kernel=kernel;ctx.save_for_backward(*inputs)
        return kernel.value(*inputs)
    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        kernel=ctx.kernel;kernel.check_stream()
        with torch.no_grad():
            for target,source in zip(kernel.inputs,ctx.saved_tensors):target.copy_(source)
            kernel.output_gradient.copy_(gradient);kernel.vjp_graph.replay()
            grads=tuple(g.clone() if needed else None for g,needed in zip(kernel.input_gradients,ctx.needs_input_grad[1:]))
        return (None,*grads)


class HybridTransitionGraph:
    def __init__(self,transition,samples,backward=True):
        if any(not x.is_cuda for x in samples):raise ValueError('CUDA samples required')
        self.transition=transition;self.inputs=tuple(x.detach().clone().requires_grad_() for x in samples)
        self.stream=torch.cuda.current_stream(samples[0].device);self.signature=tuple((x.shape,x.device,x.dtype) for x in samples)
        self.output_gradient=torch.ones_like(samples[0]);self.backward_enabled=backward
        warm=torch.cuda.Stream(device=samples[0].device);warm.wait_stream(self.stream)
        with torch.cuda.stream(warm),torch.enable_grad():
            for _ in range(3):
                y=transition(*self.inputs)
                if backward:torch.autograd.grad(y,self.inputs,self.output_gradient)
        self.stream.wait_stream(warm);self.forward_graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.forward_graph,stream=warm):self.output=transition(*self.inputs)
        self.stream.wait_stream(warm)
        if backward:
            self.vjp_graph=torch.cuda.CUDAGraph()
            with torch.enable_grad(),torch.cuda.graph(self.vjp_graph,stream=warm):
                y=transition(*self.inputs);self.input_gradients=torch.autograd.grad(y,self.inputs,self.output_gradient)
            self.stream.wait_stream(warm)
    def check_stream(self):
        if torch.cuda.current_stream(self.inputs[0].device)!=self.stream:raise ValueError('Serial construction CUDA stream required')
    def value(self,*inputs):
        self.check_stream()
        with torch.no_grad():
            for target,source in zip(self.inputs,inputs):target.copy_(source)
            self.forward_graph.replay();return self.output.clone()
    def __call__(self,*inputs):
        if len(inputs)!=len(self.signature) or any((x.shape,x.device,x.dtype)!=s for x,s in zip(inputs,self.signature)):raise ValueError('Fixed hybrid graph shape/device/dtype required')
        if torch.is_grad_enabled():
            if not self.backward_enabled:raise ValueError('Prediction-only graph cannot be used for gradients')
            return _HybridStep.apply(self,*inputs)
        return self.value(*inputs)


class HybridRateEnvironment(DifferentiableRateEnvironment):
    def __init__(self,*args,residual_net=None,nominal_arm_cm=5,**kwargs):
        n=args[0] if args else kwargs.get('n',128);device=kwargs.get('device','cuda')
        self.nominal_arm_cm=nominal_arm_cm;self._nominal_geometry_initialized=False
        self.fixed_history=torch.zeros(1,n,128,device=device);self.last_correction=torch.zeros(n,7,device=device)
        cases=copy.deepcopy(kwargs['specifications'])
        for case in cases:case['actual_body_drag_scale']=1.
        kwargs['specifications']=cases
        if kwargs.get('vehicle_spec')!=kwargs.get('nominal_vehicle_spec'):raise ValueError('Hybrid training plant receives calibrated parameters only')
        super().__init__(*args,**kwargs)
        self.residual_net=residual_net
        if residual_net is not None:
            residual_net.requires_grad_(False);residual_net.eval()
            # cuDNN needs a training-mode reserve for input/history gradients.
            # This one-layer GRU has no dropout; its weights stay frozen.
            residual_net.history.train()
        self.hybrid=HybridDynamics(self.model);plain=self.model.control
        commands=torch.cat((self.controller.previous,torch.zeros_like(self.controller.previous[:,:2])),-1)
        winds=self.wind()[:,None].expand(-1,self.model.substeps,-1).contiguous()
        if str(device).startswith('cuda'):
            self.transition=HybridTransitionGraph(self.hybrid.control,(self.state,commands,winds,self.last_correction)) if residual_net is not None else RecurrentPlantGraph(self.model,self.state,commands,winds)
        else:self.transition=self.hybrid.control if residual_net is not None else plain
        def control(state,command,wind):
            if self.residual_net is None:return self.transition(state,command,wind)
            features=self.obs['policy'];history,self.fixed_history=self.residual_net.encode(features,self.fixed_history)
            self.last_correction=self.residual_net(features,history,self.parameter_context,command[:,:4])
            return self.transition(state,command,wind,self.last_correction)
        self.model.control=control
        digest=hashlib.sha256()
        if residual_net is not None:
            for key,value in residual_net.state_dict().items():digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.trajectory_identity=identity(dict(base=self.trajectory_identity,nominal_arm_cm=nominal_arm_cm,residual_sha256=digest.hexdigest(),residual_enabled=residual_net is not None))
        self.vehicle_identity=identity(dict(base=self.vehicle_identity,nominal_arm_cm=nominal_arm_cm))
    def _observe(self,active=None):
        if not self._nominal_geometry_initialized:
            arms,lengths=nominal_geometry(self.specifications,self.nominal_arm_cm,self.device)
            self.model.arms=arms;self.model.lengths=lengths;self._nominal_geometry_initialized=True
        wind=self.wind();_,_,force,tension=self.model.loads(self.state,wind)
        force=force+self.last_correction[:,:3]/self.model.mass
        packet=self.sensors.sample(self.state[:,6:10],self.state[:,10:13],force,self.state[:,13:17],-self.state[:,2],active)
        policy=self.history.update(packet,self.controller.previous,active)
        privileged=torch.cat((policy,(self.state[:,:3]-self.model.target)/5,self.state[:,3:6]/3,wind/3,tension[:,None]/10,self.state[:,6:10],self.model.arms[:,2:3]),-1)
        self.packet=packet
        return TensorDict({'policy':policy,'privileged':privileged},[self.n])
