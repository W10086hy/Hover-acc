"""GPU qualification and iteration estimate, without an optimizer update."""
import argparse,copy,time,math,zipfile
from pathlib import Path
import torch
import numpy as np
from .hover_bptt.actor import Actor
from .hover_bptt.io import read,write,load_checkpoint,digest
from .hover_bptt.generalization import cases_at
from .hover_bptt.flexible_environment import FlexibleRopeEnvironment
from .hover_bptt.flexible_training import FlexibleRopeStream,validate_protocol,source_manifest,progress,REPO
from .hover_bptt.plant import install_training_plant
from .hover_bptt.training import trajectory

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=1.);parser.add_argument('--device',default='cuda')
    parser.add_argument('--gradient-only',action='store_true')
    parser.add_argument('--protocol',default=str(ROOT/'flexible_rope_protocol.json'))
    a=parser.parse_args();torch.set_num_threads(2)
    if not 0<a.seconds<=30 or round(a.seconds*100)%10:raise ValueError('Whole0.1s qualification, within30s')
    if a.seconds>10 and not a.gradient_only:raise ValueError('30s pilot performs gradients only, no extra replay')
    p=read(a.protocol);validate_protocol(p)
    if p.get('domain_randomization',{}).get('enabled'):
        from .randomized_training import qualify
        return qualify(a,p)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/p['parent_checkpoint'];saved,sha=load_checkpoint(parent);cfg=copy.deepcopy(saved['configuration'])
    cfg['training']['report_credit_progress']=a.seconds>=10
    cases=cases_at(p['training_directions_deg'],wind_cases=p.get('training_winds'));steps=round(a.seconds*100)
    source=source_manifest(a.protocol)
    with zipfile.ZipFile(out/'frozen_source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source:archive.write(REPO/name,arcname=name)
    write(out/'run_manifest.json',dict(source_sha256=source,protocol=p,parent_checkpoint_sha256=sha,
        source_frozen_before_gradient=True,seconds=a.seconds,gradient_only=a.gradient_only,
        objective_credit_recompute_s=[a.seconds,min(10,a.seconds),min(1,a.seconds)],optimizer_created=False))
    def create():
        env=FlexibleRopeEnvironment(len(cases),20673101,wind_seed=20673102,device=a.device,stage=copy.deepcopy(cfg),
            episode_s=a.seconds,specifications=cases,rope=p['rope'])
        install_training_plant(env);stream=FlexibleRopeStream(env)
        stream.hidden=env.state.new_zeros(1,env.n,cfg['model']['gru_hidden'])
        return stream
    actor=Actor(cfg['model']).to(a.device);actor.load_state_dict(saved['actor']);actor.train()
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();stream=create();build=time.perf_counter()-started
    forward=None;forward_s=None
    if not a.gradient_only:
        started=time.perf_counter();_,_,forward,_=trajectory(stream,actor,steps=steps)
        if a.device.startswith('cuda'):torch.cuda.synchronize()
        forward_s=time.perf_counter()-started;del stream
        stream=create()
    samples=[stream.env.state.detach().cpu().numpy().copy()];times=[0.];seen=set();native=stream.step
    def traced_step(policy):
        key=stream.steps;value=native(policy)
        if key not in seen:
            seen.add(key)
            if stream.steps%10==0:
                samples.append(stream.env.state.detach().cpu().numpy().copy());times.append(stream.steps*stream.env.dt)
        return value
    stream.step=traced_step;progress(stream)
    print(dict(stage='flexible_gradient_start',seconds=a.seconds,count=len(cases),device=a.device,optimizer_created=False),flush=True)
    started=time.perf_counter()
    loss,diag,score,_=trajectory(stream,actor,gradient=True,steps=steps,credit_steps=min(1000,steps),recompute_steps=min(100,steps))
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    gradient_s=time.perf_counter()-started
    gradients={n:dict(norm=float(q.grad.norm()),finite=bool(torch.isfinite(q.grad).all())) for n,q in actor.named_parameters() if q.grad is not None}
    valid=bool(gradients) and all(v['finite'] for v in gradients.values()) and any(v['norm']>0 for v in gradients.values())
    unchanged=digest(parent)==sha and all(torch.equal(v.detach().cpu(),saved['actor'][k]) for k,v in actor.state_dict().items())
    if not valid or not unchanged:raise RuntimeError('Qualification gradient or protected weights invalid')
    result=dict(passed=True,qualification_not_training=True,optimizer_created=False,seconds=a.seconds,count=len(cases),
        rope=p['rope'],segments=p['rope']['segments'],state_width=stream.env.state.shape[1],construction_s=build,
        forward_wall_s=forward_s,gradient_recomputed_wall_s=gradient_s,
        estimated_full120_gradient_s=gradient_s*120/a.seconds,
        estimated_full120_forward_s=forward_s*120/a.seconds if forward_s is not None else None,
        estimated_iteration_with_one_candidate_s=(gradient_s+forward_s)*120/a.seconds if forward_s is not None else None,
        estimated_iteration_with_maximum_candidates_s=(gradient_s+p['maximum_paired_trials']*forward_s)*120/a.seconds if forward_s is not None else None,
        estimation='Linear extrapolation; actual10s credit memory, wind and failures can change throughput',
        gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,device=a.device,
        peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if a.device.startswith('cuda') else None,
        loss=loss,diagnostics=diag,gradient_parameters=gradients,score=score,forward_score=forward,
        parent_checkpoint_sha256=sha,parent_actor_and_adam_file_unchanged=unchanged,
        source_sha256={f.as_posix():digest(f) for f in [ROOT/'hover_bptt/flexible_rope.py',ROOT/'hover_bptt/flexible_environment.py',ROOT/'hover_bptt/flexible_training.py',Path(__file__)]},
        full_source_unchanged=source_manifest(a.protocol)==source)
    if not result['full_source_unchanged']:raise RuntimeError('Source changed during gradient pilot')
    np.savez_compressed(out/'trajectory.npz',times=np.asarray(times),state=np.stack(samples))
    rows=[]
    for i,c in enumerate(cases):
        rows.append(dict(c,**score['wind_statistics'][i],completed=score['completed'][i],
            position_rmse_m=score['position_rmse_m'][i],height_rmse_m=score['height_rmse_m'][i],
            horizontal_rmse_m=score['horizontal_rmse_m'][i],safe_duration_s=float(stream.counts[i]*stream.env.dt),
            failure_code=int(stream.failure_codes[i])))
    write(out/'result.json',dict(duration_s=a.seconds,cases=rows,seed_pair=[20673101,20673102],checkpoint_sha256=sha,
        cumulative_physical_updates=saved['cumulative_physical_updates'],stage_label='P2动态柔绳；梯度验证；策略未更新',
        rope=p['rope'],optimizer_created=False,accepted_updates=0,final_acceptance=False,
        safe_completed=sum(score['completed']),count=len(cases),
        rmse_3d_h_xy_m=[math.sqrt(sum(c[k]**2 for c in rows)/len(rows)) for k in ['position_rmse_m','height_rmse_m','horizontal_rmse_m']]))
    torch.save(dict(wind_w_mps=stream.env.wind_field.trace.cpu(),dt_s=stream.env.dt),out/'wind_trace.pt')
    write(out/'qualification.json',result)
    print({k:v for k,v in result.items() if k not in ['score','forward_score','gradient_parameters','source_sha256']},flush=True)


if __name__=='__main__':main()
