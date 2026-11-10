"""CUDA forward/VJP capture for6-column exogenous plant inputs."""
import torch
from P2.hover_bptt.cuda_bptt_kernel import RecurrentPlantGraph


class SixChannelPlantGraph(RecurrentPlantGraph):
    def __init__(self,model,state,commands,winds):
        samples=(state,commands,winds)
        if any(not v.is_cuda or v.device!=state.device or v.dtype!=state.dtype for v in samples):raise ValueError('Matching CUDA tensors required')
        if state.dtype not in (torch.float32,torch.float64):raise ValueError('Validated dtype required')
        if state.shape!=(len(state),model.state_width) or commands.shape!=(len(state),6) or winds.shape!=(len(state),model.substeps,6):raise ValueError('Explicit frame-plant shapes required')
        if any(p.requires_grad for p in model.parameters()):raise ValueError('Fixed physics required')
        if any(b.requires_grad or b.device!=state.device or b.dtype!=state.dtype for b in model.buffers()):raise ValueError('Fixed matching buffers required')
        self.plain=model.control;self.signature=tuple((v.shape,v.device,v.dtype) for v in samples)
        self.inputs=tuple(v.detach().clone().requires_grad_() for v in samples);self.stream=torch.cuda.current_stream(state.device)
        self.output_gradient=torch.ones_like(state);warm=torch.cuda.Stream(device=state.device);warm.wait_stream(self.stream)
        with torch.cuda.stream(warm),torch.enable_grad():
            for _ in range(3):
                output=self.plain(*self.inputs);torch.autograd.grad(output,self.inputs,self.output_gradient)
        self.stream.wait_stream(warm);self.forward_graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.forward_graph,stream=warm):self.output=self.plain(*self.inputs)
        self.stream.wait_stream(warm);self.vjp_graph=torch.cuda.CUDAGraph()
        with torch.enable_grad(),torch.cuda.graph(self.vjp_graph,stream=warm):
            output=self.plain(*self.inputs);self.input_gradients=torch.autograd.grad(output,self.inputs,self.output_gradient)
        self.stream.wait_stream(warm)
