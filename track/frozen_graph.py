"""Exact serial CUDA forward/VJP for fixed onboard functions, with tensor state."""
import torch
from torch.autograd.function import once_differentiable


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx,kernel,*inputs):
        ctx.kernel=kernel;ctx.save_for_backward(*inputs);outputs=kernel.value(inputs)
        ctx.mark_non_differentiable(*(v for v in outputs if not v.is_floating_point()))
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx,*adjoints):
        k=ctx.kernel;k.copy(ctx.saved_tensors)
        with torch.no_grad():
            for index,buffer in k.adjoints.items():
                if adjoints[index] is None:buffer.zero_()
                else:buffer.copy_(adjoints[index])
            k.backward_graph.replay()
            gradients=tuple(k.gradients[j].clone() if v.is_floating_point() and ctx.needs_input_grad[j+1] else None for j,v in enumerate(ctx.saved_tensors))
        return (None,*gradients)


class FrozenFunctionGraph:
    def __init__(self,function,inputs):
        self.stream=torch.cuda.current_stream();self.signature=tuple((v.shape,v.dtype,v.device) for v in inputs)
        self.inputs=tuple(v.detach().clone().requires_grad_(v.is_floating_point()) for v in inputs)
        floats=[j for j,v in enumerate(self.inputs) if v.is_floating_point()]
        def derivatives():
            outputs=function(*self.inputs);targets=[(j,v) for j,v in enumerate(outputs) if v.requires_grad]
            if not hasattr(self,'adjoints'):self.adjoints={j:torch.ones_like(v) for j,v in targets}
            adjoints=self.adjoints
            gradients=torch.autograd.grad([v for j,v in targets],[self.inputs[j] for j in floats],[adjoints[j] for j,v in targets],allow_unused=True)
            return outputs,adjoints,{j:g if g is not None else torch.zeros_like(self.inputs[j]) for j,g in zip(floats,gradients)}
        warm=torch.cuda.Stream();warm.wait_stream(self.stream)
        with torch.cuda.stream(warm),torch.enable_grad():
            for _ in range(3):derivatives()
        self.stream.wait_stream(warm);self.forward_graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.forward_graph,stream=warm):self.outputs=function(*self.inputs)
        self.stream.wait_stream(warm);self.backward_graph=torch.cuda.CUDAGraph()
        with torch.enable_grad(),torch.cuda.graph(self.backward_graph,stream=warm):_,self.adjoints,self.gradients=derivatives()
        self.stream.wait_stream(warm)

    def copy(self,inputs):
        if torch.cuda.current_stream()!=self.stream:raise ValueError('Use the construction CUDA stream')
        with torch.no_grad():
            for destination,source,signature in zip(self.inputs,inputs,self.signature):
                if (source.shape,source.dtype,source.device)!=signature:raise ValueError('Fixed CUDA signature required')
                destination.copy_(source)

    def value(self,inputs):
        self.copy(inputs)
        with torch.no_grad():self.forward_graph.replay();return tuple(v.clone() for v in self.outputs)

    def __call__(self,*inputs):return _Step.apply(self,*inputs) if torch.is_grad_enabled() else self.value(inputs)


def controller_function(controller,dt):
    """Expose all state explicitly; no parameter or online sensor semantics change."""
    from P2.hover_bptt.onboard import Sensors
    names=[n for n,v in vars(controller).items() if isinstance(v,torch.Tensor)];native=controller.apply
    def function(*inputs):
        outer={n:getattr(controller,n) for n in names}
        try:
            for n,v in zip(names,inputs):setattr(controller,n,v)
            packet=Sensors(*inputs[len(names):len(names)+5]);motors,_=native(packet,inputs[-1],dt)
            return tuple(getattr(controller,n).clone() for n in names)+(motors.clone(),)
        finally:
            for n,v in outer.items():setattr(controller,n,v)
            for n in ['trace_bounded','trace_applied','trace_nominal_motors','trace_motors','trace_fallback']:getattr(controller,n).clear()
    return names,function


def install_controller(env):
    names,function=controller_function(env.controller,env.dt);packet=env.packet
    def arguments(packet,raw):return tuple(getattr(env.controller,n) for n in names)+(packet.attitude,packet.gyro,packet.specific_force,packet.motor_speed,packet.baro_height,raw)
    kernel=FrozenFunctionGraph(function,arguments(packet,env.state.new_zeros(env.n,4)))
    def apply(packet,raw,dt):
        if dt!=env.dt:raise ValueError('Fixed control period required')
        result=kernel(*arguments(packet,raw))
        for n,v in zip(names,result[:-1]):setattr(env.controller,n,v)
        return result[-1],{}
    env.controller.apply=apply
    return kernel


def history_function(history):
    from P2.hover_bptt.onboard import Sensors
    objects=[history,history.height]+([history.motion] if history.motion else [])
    names=[(i,n) for i,obj in enumerate(objects) for n,v in vars(obj).items() if isinstance(v,torch.Tensor)];native=history.update
    def function(*inputs):
        outer={(i,n):getattr(objects[i],n) for i,n in names}
        try:
            for (i,n),v in zip(names,inputs):setattr(objects[i],n,v)
            packet=Sensors(*inputs[len(names):len(names)+5]);policy=native(packet,inputs[-1])
            return tuple(getattr(objects[i],n).clone() for i,n in names)+(policy.clone(),)
        finally:
            for (i,n),v in outer.items():setattr(objects[i],n,v)
    return objects,names,function


def install_history(env):
    objects,names,function=history_function(env.history)
    def arguments(packet,motors):return tuple(getattr(objects[i],n) for i,n in names)+(packet.attitude,packet.gyro,packet.specific_force,packet.motor_speed,packet.baro_height,motors)
    kernel=FrozenFunctionGraph(function,arguments(env.packet,env.controller.previous))
    def update(packet,motors,active=None):
        if active is not None:raise ValueError('This fixed training graph advances all cases;failure holding is external')
        result=kernel(*arguments(packet,motors))
        for (i,n),v in zip(names,result[:-1]):setattr(objects[i],n,v)
        return result[-1]
    env.history.update=update
    return kernel


def install_loads(env):
    kernel=FrozenFunctionGraph(env.model.loads,(env.state,env.wind()));env.model.loads=kernel
    return kernel


def install(env):
    """Call after plant capture; functions contain no random draws or truth inputs."""
    return dict(controller=install_controller(env),history=install_history(env),loads=install_loads(env))
