"""Pure legacy RK4 forward/VJP CUDA graphs safe for recurrent first-order BPTT.

Each invocation saves its own inputs. Backward restores those inputs and
recomputes the local VJP, rather than reusing the last time step's activations.
Only deterministic plant arithmetic is captured; sensors and RNG stay eager.
Fixed buffers/model and a single CUDA stream are required. No double backward.
"""
import torch
from torch.autograd.function import once_differentiable


class _RecurrentPlantStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, state, commands, winds):
        ctx.kernel = kernel
        ctx.save_for_backward(state, commands, winds)
        return kernel.forward_value(state, commands, winds)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        kernel = ctx.kernel
        kernel.check_stream()
        with torch.no_grad():
            for destination, source in zip(kernel.inputs, ctx.saved_tensors):
                destination.copy_(source)
            kernel.output_gradient.copy_(gradient)
            kernel.vjp_graph.replay()
            gradients = tuple(g.clone() if needed else None
                              for g, needed in zip(kernel.input_gradients,
                                                   ctx.needs_input_grad[1:]))
        return (None, *gradients)


class RecurrentPlantGraph:
    """Fixed-shape plant acceleration; output and gradient storage never aliases."""
    def __init__(self, model, state, commands, winds):
        samples = (state, commands, winds)
        if any(not value.is_cuda for value in samples):
            raise ValueError('CUDA tensors required')
        if any(value.device != state.device or value.dtype != state.dtype for value in samples):
            raise ValueError('Consistent device and dtype required')
        if state.dtype not in (torch.float32, torch.float64):
            raise ValueError('Validated dtype must be float32 or float64')
        if state.ndim != 2 or state.shape[1] != getattr(model,'state_width',19) or commands.shape != (state.shape[0], 6):
            raise ValueError('Expected fixed state/command shape')
        if winds.shape != (state.shape[0], model.substeps, 3):
            raise ValueError('Expected fixed wind shape')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Only fixed plant parameters supported')
        if any(b.requires_grad or b.device != state.device or b.dtype != state.dtype
               for b in model.buffers()):
            raise ValueError('Fixed matching plant buffers required')
        self.plain = model.control
        self.signature = tuple((v.shape, v.device, v.dtype) for v in samples)
        self.inputs = tuple(v.detach().clone().requires_grad_() for v in samples)
        self.stream = torch.cuda.current_stream(state.device)
        self.output_gradient = torch.ones_like(state)
        warm = torch.cuda.Stream(device=state.device)
        warm.wait_stream(self.stream)
        with torch.cuda.stream(warm), torch.enable_grad():
            for _ in range(3):
                output = self.plain(*self.inputs)
                torch.autograd.grad(output, self.inputs, self.output_gradient)
        self.stream.wait_stream(warm)
        self.forward_graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.forward_graph, stream=warm):
            self.output = self.plain(*self.inputs)
        self.stream.wait_stream(warm)
        self.vjp_graph = torch.cuda.CUDAGraph()
        with torch.enable_grad(), torch.cuda.graph(self.vjp_graph, stream=warm):
            local_output = self.plain(*self.inputs)
            self.input_gradients = torch.autograd.grad(local_output, self.inputs,
                                                       self.output_gradient)
        self.stream.wait_stream(warm)

    def check_stream(self):
        if torch.cuda.current_stream(self.inputs[0].device) != self.stream:
            raise ValueError('Use the construction CUDA stream for serial graph replay')

    def forward_value(self, state, commands, winds):
        self.check_stream()
        with torch.no_grad():
            for destination, source in zip(self.inputs, (state, commands, winds)):
                destination.copy_(source)
            self.forward_graph.replay()
            return self.output.clone()

    def __call__(self, state, commands, winds):
        for value, (shape, device, dtype) in zip((state, commands, winds), self.signature):
            if value.shape != shape:
                raise ValueError('Fixed CUDA graph input shape required')
            if value.device != device or value.dtype != dtype:
                raise ValueError('Fixed CUDA graph device and dtype required')
        if torch.is_grad_enabled():
            return _RecurrentPlantStep.apply(self, state, commands, winds)
        return self.forward_value(state, commands, winds)
