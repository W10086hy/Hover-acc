import json,math
import torch
from torch.autograd.function import once_differentiable
from tensordict import TensorDict
from .state import runtime_objects
from .onboard import Sensors
class BoundaryLayout:
    def __init__(self, stream):
        self.stream = stream
        self.buffers = [(key, name) for key, obj in runtime_objects(stream.env).items()
            for name, value in vars(obj).items() if isinstance(value, torch.Tensor)]
        self.observations = list(stream.env.obs.keys())
        self.counters = ['alive', 'counts', 'failure_codes', 'error_sums', 'cost_sum']

    def values(self):
        objects = runtime_objects(self.stream.env)
        return tuple([getattr(objects[key], name) for key, name in self.buffers]
            + [self.stream.env.obs[key] for key in self.observations]
            + [self.stream.hidden] + [getattr(self.stream, key) for key in self.counters])

    def assign(self, values, *, clone):
        values = iter(values); objects = runtime_objects(self.stream.env)
        take = lambda: next(values).clone() if clone else next(values)
        for key, name in self.buffers:
            setattr(objects[key], name, take())
        env = self.stream.env
        env.obs = TensorDict({key:take() for key in self.observations}, batch_size=[env.n])
        self.stream.hidden = take()
        for key in self.counters:
            setattr(self.stream, key, take())
        s = env.sensors
        env.packet = Sensors(s.q, s.rate, s.force, s.rpm, s.height)

class SegmentReplay:
    def __init__(self, stream, actor, layout, diagnostics):
        self.stream, self.actor, self.layout, self.diagnostics = stream, actor, layout, diagnostics

    def run(self, values, steps, start_step, rng):
        env = self.stream.env
        outer = self.layout.values()
        outer_step = self.stream.steps
        outer_rng = (env.gen.get_state(), env.sensor_gen.get_state())
        try:
            self.layout.assign(values, clone=True)
            self.stream.steps = start_step
            env.gen.set_state(rng[0]); env.sensor_gen.set_state(rng[1])
            costs = [self.stream.step(self.actor) for _ in range(steps)]
            result = tuple(value.clone() for value in self.layout.values())
            self.end_rng = (env.gen.get_state(), env.sensor_gen.get_state())
            return (*result, torch.stack(costs).mean())
        finally:
            self.layout.assign(outer, clone=False)
            self.stream.steps = outer_step
            env.gen.set_state(outer_rng[0]); env.sensor_gen.set_state(outer_rng[1])

class SegmentAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, replay, steps, start_step, rng, boundary_count, *inputs):
        ctx.replay, ctx.steps, ctx.start_step, ctx.rng = replay, steps, start_step, rng
        ctx.boundary_count = boundary_count
        ctx.save_for_backward(*inputs)
        ctx.set_materialize_grads(False)
        result = replay.run(inputs[:boundary_count], steps, start_step, rng)
        ctx.mark_non_differentiable(*(value for value in result if not value.is_floating_point()))
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_gradients):
        saved = ctx.saved_tensors; n = ctx.boundary_count
        with torch.enable_grad():
            values = tuple(value.detach().requires_grad_(True) if value.is_floating_point()
                else value.detach() for value in saved[:n])
            result = ctx.replay.run(values, ctx.steps, ctx.start_step, ctx.rng)
            targets = []; adjoints = []
            for value, gradient in zip(result, output_gradients):
                if value.requires_grad and gradient is not None:
                    targets.append(value); adjoints.append(gradient)
            variables = [value for value in values if value.requires_grad] + list(saved[n:])
            gradients = torch.autograd.grad(targets, variables, adjoints, allow_unused=True) if targets else (None,)*len(variables)
        pointer = iter(gradients); boundary_gradients = [next(pointer) if value.requires_grad else None for value in values]
        parameter_gradients = list(pointer)
        d = ctx.replay.diagnostics
        d['recomputed_segments'] += 1
        d['maximum_recomputed_steps'] = max(d['maximum_recomputed_steps'], ctx.steps)
        if d.get('emit_progress', False):
            print(json.dumps(dict(stage='checkpointed_full_credit_backward',
                recomputed_segments=d['recomputed_segments'], total_segments=d['checkpoint_segments'],
                segment_start_s=ctx.start_step*ctx.replay.stream.env.dt)), flush=True)
        return (None, None, None, None, None, *boundary_gradients, *parameter_gradients)
