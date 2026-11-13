import torch
from .adjoint import BoundaryLayout,SegmentReplay,SegmentAdjoint
def _checkpointed_loss(stream,actor,steps,segment_steps):
    layout=BoundaryLayout(stream);diagnostics=dict(recomputed_segments=0,maximum_recomputed_steps=0,emit_progress=False)
    replay=SegmentReplay(stream,actor,layout,diagnostics);parameters=tuple(p for p in actor.parameters() if p.requires_grad)
    costs=[]
    for offset in range(0,steps,segment_steps):
        count=min(segment_steps,steps-offset);values=layout.values();start=stream.steps
        rng=(stream.env.gen.get_state(),stream.env.sensor_gen.get_state())
        result=SegmentAdjoint.apply(replay,count,start,rng,len(values),*values,*parameters)
        layout.assign(result[:-1],clone=False);stream.steps=start+count
        stream.env.gen.set_state(replay.end_rng[0]);stream.env.sensor_gen.set_state(replay.end_rng[1])
        costs.append(result[-1]*(count/steps))
    return torch.stack(costs).sum(),diagnostics

def checkpointed_window(stream,actor,steps,segment_steps=100):
    loss,diagnostics=_checkpointed_loss(stream,actor,steps,segment_steps)
    loss.backward()
    stream.detach()
    return float(loss.detach()),diagnostics

def _terminal_objective_windows(stream, actor, chunks, credit_steps, recompute_steps):
    windows = []
    for _ in range(chunks):
        loss, diagnostics = _checkpointed_loss(stream, actor, credit_steps, recompute_steps)
        windows.append((loss, stream.objective_statistic, diagnostics))
        stream.detach()

    statistic = stream.objective_statistic.detach().requires_grad_()
    penalty = stream.objective_penalty(statistic)
    adjoint, = torch.autograd.grad(penalty, statistic)

    def backward_windows():
        for loss, statistic, diagnostics in windows:
            # The incoming statistic is detached at each credit boundary.
            # The final statistic's adjoint applies to every credit's contribution.
            # The caller divides all accumulated gradients by the credit count.
            torch.autograd.backward((loss, statistic), (torch.ones_like(loss), chunks * adjoint))
            yield float(loss.detach()), diagnostics

    return float(penalty.detach()), backward_windows()

def accumulated_window(stream,actor,*,total_steps=1000,credit_steps=200,recompute_steps=100):
    if (min(total_steps,credit_steps,recompute_steps)<1 or total_steps%credit_steps or credit_steps%recompute_steps):
        raise ValueError('Wholeequalshortcreditwindowsandrecomputesegmentsrequired')
    actor.zero_grad(set_to_none=True);chunks=total_steps//credit_steps;costs=[]
    diagnostics=dict(gradient_subwindows=chunks,gradient_credit_steps=credit_steps,optimizer_interval_steps=total_steps,recomputed_segments=0,maximum_recomputed_steps=0)
    start = stream.steps
    if hasattr(stream, 'objective_penalty'):
        terminal_cost, windows = _terminal_objective_windows(stream, actor, chunks, credit_steps, recompute_steps)
    else:
        terminal_cost = 0.
        windows = (checkpointed_window(stream, actor, credit_steps, recompute_steps) for _ in range(chunks))
    for index, (value, diag) in enumerate(windows, start=1):
        costs.append(value)
        diagnostics['recomputed_segments']+=diag['recomputed_segments'];diagnostics['maximum_recomputed_steps']=max(diagnostics['maximum_recomputed_steps'],diag['maximum_recomputed_steps'])
        if stream.env.cfg.get('training',{}).get('report_credit_progress',False):
            print(dict(stage='credit_window_completed',elapsed_simulation_s=(start+index*credit_steps)*stream.env.dt,
                       objective_s=total_steps*stream.env.dt,credit_s=credit_steps*stream.env.dt,cost=value),flush=True)
    for parameter in actor.parameters():
        if parameter.grad is not None:parameter.grad.div_(chunks)
    return sum(costs)/chunks + terminal_cost,diagnostics
def trainable_parameters(actor):
    parameters=[]
    for name,parameter in actor.named_parameters():
        parameter.requires_grad_(not name.startswith('distribution.'))
        if parameter.requires_grad:parameters.append(parameter)
    return parameters
