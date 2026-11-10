"""Measure30s truncated BPTT with no optimizer or checkpoint updates."""
import argparse,copy,time,zipfile,math
from pathlib import Path
import torch
from P2.hover_bptt.actor import Actor
from P2.hover_bptt.io import read,write,save,load_checkpoint,digest
from P2.hover_bptt.optimizer import initialize
from P2.hover_bptt.training import trajectory
from P2.hover_bptt.state import snapshot,restore_stream
from P3.mismatch_dynamics import DynamicsResidual
from P3.compensation import tensor_hash
from .height_speed import protocol_for,ROOT
from .frame_environment import build_training
from .frame_stream import FrameStream
from .frozen_graph import install


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--height',type=float,default=10.);p.add_argument('--seconds',type=float,default=30.);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--protocol');p.add_argument('--credit-s',type=float,choices=[10.,30.],default=10.)
    p.add_argument('--seed-offset',type=int,default=0);p.add_argument('--case-repeats',type=int,default=1);p.add_argument('--recovery-radius',type=float)
    p.add_argument('--onboard-graphs',action='store_true');p.add_argument('--cache-gradient',action='store_true')
    p.add_argument('--continuous-physical-training',action='store_true');p.add_argument('--speed-barrier-gain',type=float,default=20.);p.add_argument('--radius-barrier-gain',type=float,default=20.)
    a=p.parse_args();out=Path(a.output)
    if out.exists() or not out.resolve().is_relative_to(ROOT/'track'):raise ValueError('Use a new track qualification folder')
    torch.set_num_threads(2);suite=read(ROOT/'track/configs/height_speed_suite.json')
    protocol=read(ROOT/a.protocol) if a.protocol else protocol_for(suite,a.height)
    if a.protocol and (protocol['target_aircraft_height_m']!=a.height or protocol['duration_s']!=a.seconds or a.seed_offset or a.case_repeats!=1):
        raise ValueError('Explicit protocol defines height, duration, case batch and seeds')
    if a.seed_offset<0 or a.seed_offset%2 or not 1<=a.case_repeats<=8:raise ValueError('Use declared training cohorts')
    if any(not math.isfinite(v) or v<0 for v in [a.speed_barrier_gain,a.radius_barrier_gain]):raise ValueError('Finite nonnegative safety loss gains required')
    if not a.protocol:protocol.update(duration_s=a.seconds,state_seed=suite['training_seed_pair'][0]+a.seed_offset,wind_seed=suite['training_seed_pair'][1]+a.seed_offset)
    reserved={v for pair in read(ROOT/'P2/acceptance.json')['final_seed_pairs'] for v in pair.values()}
    if {protocol['state_seed'],protocol['wind_seed']}&(reserved|set(suite['development_seed_pair']+suite['independent_check_seed_pair'])):raise ValueError('Training/qualification seeds must be separate')
    if a.case_repeats>1:
        cases=protocol['case_specs'];protocol['case_specs']=[copy.deepcopy(c) for _ in range(a.case_repeats) for c in cases]
        protocol['training_wind_seeds_per_case']=[protocol['wind_seed']+2*j for j in range(a.case_repeats) for _ in cases]
    if a.recovery_radius is not None:protocol['training_recovery_radius_m']=a.recovery_radius
    if a.continuous_physical_training:protocol.update(training_continue_after_performance_boundary=True,training_speed_barrier_gain=a.speed_barrier_gain,training_radius_barrier_gain=a.radius_barrier_gain)
    original=ROOT/(protocol['controller_checkpoint'] if a.protocol else suite['baseline_actor'])
    residual_path=ROOT/(protocol['residual_checkpoint'] if a.protocol else suite['baseline_residual'])
    parent=Path(a.checkpoint).resolve() if a.checkpoint else original
    if not parent.is_relative_to(ROOT):raise ValueError('Use a project checkpoint')
    saved,actor_sha=load_checkpoint(parent);original_sha=digest(original);learned,net_sha=load_checkpoint(residual_path)
    if a.protocol and (original_sha!=protocol['controller_sha256'] or net_sha!=protocol['residual_sha256'] or learned['controller_checkpoint_sha256']!=original_sha):
        raise ValueError('Protocol, source Actor and frozen residual must have matching identities')
    torch.manual_seed(protocol['state_seed']);actor=Actor(saved['configuration']['model']).cuda();actor.load_state_dict(saved['actor']);actor.train()
    optimizer=initialize(actor,saved);residual=DynamicsResidual(learned['model_configuration']).cuda();residual.load_state_dict(learned['residual']);residual.eval();residual.requires_grad_(False)
    net_before=tensor_hash(residual.state_dict());out.mkdir(parents=True)
    source_paths=[ROOT/'track'/n for n in ['frame_dynamics.py','frame_graph.py','frame_environment.py','frame_stream.py','qualify_frame.py']]
    source_paths += [ROOT/'track'/n for n in ['frozen_graph.py','height_speed.py','environment.py','motion.py']]
    source_paths+=list((ROOT/'P2/hover_bptt').rglob('*.py'))+[ROOT/'P3/compensation.py',ROOT/'P3/mismatch_dynamics.py']
    source_paths+=list((ROOT/'P2/hover_bptt/configs').glob('*.json'))+[ROOT/protocol['residual_protocol'],ROOT/'track/configs/height_speed_suite.json',ROOT/'P2/acceptance.json']
    sources={f.relative_to(ROOT).as_posix():digest(f) for f in source_paths}
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,arcname=name)
    write(out/'manifest.json',dict(protocol=protocol,source_sha256=sources,actor_checkpoint_sha256=actor_sha,residual_checkpoint_sha256=net_sha,optimizer_updates=0,parent_checkpoint=parent.relative_to(ROOT).as_posix(),credit_s=min(a.credit_s,a.seconds)))
    started=time.perf_counter();env=build_training(protocol,saved,residual);env.install_cuda_graph()
    onboard=install(env) if a.onboard_graphs else None
    torch.cuda.synchronize();setup=time.perf_counter()-started
    stream=FrameStream(env);stream.hidden=env.state.new_zeros(1,env.n,saved['configuration']['model']['gru_hidden'])
    before=snapshot(env,stream,actor,optimizer,updates=0,episode=0,window=0)
    steps=round(a.seconds/env.dt);credit=min(round(a.credit_s/env.dt),steps)
    if steps%credit or credit%100:raise ValueError('Whole10s credit and1s segments required')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    loss,diagnostics,baseline,commands=trajectory(stream,actor,gradient=True,steps=steps,credit_steps=credit,recompute_steps=100)
    torch.cuda.synchronize();gradient_s=time.perf_counter()-start
    gradient_norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),float('inf'),error_if_nonfinite=True))
    baseline.update(maximum_world_speed_mps=env.training_peak_speed.cpu().tolist(),maximum_tracking_radius_m=env.training_peak_radius.cpu().tolist())
    if a.cache_gradient:
        save(out/'gradient.pt',dict(protocol=protocol,parent_checkpoint=parent.relative_to(ROOT).as_posix(),parent_checkpoint_sha256=actor_sha,residual_checkpoint_sha256=net_sha,
            initial=before,gradients={k:v.grad.detach().cpu() for k,v in actor.named_parameters() if v.grad is not None},
            baseline=baseline,baseline_commands=commands.cpu(),loss=loss,diagnostics=diagnostics,gradient_norm=gradient_norm,credit_s=credit*env.dt,source_sha256=sources,gradient_s=gradient_s,
            failure_codes=stream.failure_codes.cpu().tolist(),safe_duration_s=(stream.counts*env.dt).cpu().tolist()))
    restore_stream(env,stream,actor,optimizer,before);start=time.perf_counter()
    _,_,replay,_=trajectory(stream,actor,steps=steps);torch.cuda.synchronize();forward_s=time.perf_counter()-start
    replay.update(maximum_world_speed_mps=env.training_peak_speed.cpu().tolist(),maximum_tracking_radius_m=env.training_peak_radius.cpu().tolist())
    assert baseline==replay, 'Restored deterministic forward differs from gradient forward'
    assert gradient_norm>0, 'No Actor gradient'
    assert all(torch.equal(v.detach().cpu(),saved['actor'][k]) for k,v in actor.state_dict().items())
    assert tensor_hash(residual.state_dict())==net_before and digest(parent)==actor_sha and digest(original)==original_sha and digest(residual_path)==net_sha
    assert all(digest(ROOT/n)==sha for n,sha in sources.items())
    result=dict(passed=True,seconds=a.seconds,objective_s=a.seconds,credit_s=credit*env.dt,recompute_s=1.,full_untruncated_gradient=credit==steps,
        height_m=a.height,cases=env.n,setup_s=setup,gradient_s=gradient_s,paired_forward_s=forward_s,estimated_one_update_s=gradient_s+forward_s,
        peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/2**20,gradient_norm=gradient_norm,loss=loss,baseline=baseline,replay=replay,diagnostics=diagnostics,
        actor_weights_unchanged=True,residual_weights_unchanged=True,optimizer_updates=0,torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name())
    result['completion_definition']='physical hard constraints only;speed/radius penalized continuously' if a.continuous_physical_training else 'all configured offline limits'
    write(out/'qualification.json',result);print({k:v for k,v in result.items() if k not in ['baseline','replay','diagnostics']},flush=True)


if __name__=='__main__':main()
