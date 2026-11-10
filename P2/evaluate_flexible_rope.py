"""Paired full-horizon development evaluation in the dynamic cable plant."""
import argparse
from functools import partial
from pathlib import Path
import torch
from .hover_bptt.environment import plant_config
from .hover_bptt.flexible_environment import FlexibleRopeEvaluationEnvironment
from .hover_bptt.generalization import cases_at
from .hover_bptt.evaluation import evaluate_one
from .hover_bptt.io import read,write
from .hover_bptt.dynamics.torch_model import CapturedControl

ROOT=Path(__file__).resolve().parent


def factory(*args,rope=None,**kwargs):
    env=FlexibleRopeEvaluationEnvironment(*args,rope=rope,**kwargs)
    if str(env.device).startswith('cuda'):
        commands=torch.cat((env.controller.previous,torch.zeros_like(env.controller.previous[:,:2])),-1)
        winds=env.wind()[:,None].expand(-1,env.model.substeps,-1).contiguous()
        env.model.control=CapturedControl(env.model,env.state,commands,winds)
    return env


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--device',default='cuda')
    p.add_argument('--protocol',default=str(ROOT/'flexible_rope_protocol.json'))
    p.add_argument('--baseline-checkpoint')
    a=p.parse_args();torch.set_num_threads(2)
    protocol=read(a.protocol)
    if protocol.get('domain_randomization',{}).get('enabled'):
        from .hover_bptt.flexible_training import validate_protocol
        from .randomized_training import evaluate
        validate_protocol(protocol)
        return evaluate(a,protocol)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cases=cases_at(protocol['development_directions_deg']);seed=protocol['development_seed']
    results=[]
    baseline=Path(a.baseline_checkpoint) if a.baseline_checkpoint else ROOT/protocol['parent_checkpoint']
    for label,checkpoint in [('before',baseline),('after',Path(a.checkpoint))]:
        result=evaluate_one(checkpoint,out/label,seed,cases,device=a.device,duration_s=a.seconds,
            plant_configuration=plant_config(),environment_factory=partial(factory,rope=protocol['rope']),
            stage_label=f'P2动态柔绳；{label}；开发评价')
        result.update(rope=protocol['rope'],rope_parameters_measured=False,
            trajectory_identity='extended state: body19 + free node positions + free node velocities + interval peak tension')
        write(out/label/'result.json',result);results.append(result)
    summary=dict(duration_s=a.seconds,seed_pair=[seed,seed+1],count=len(cases),
        before={k:results[0][k] for k in ['safe_completed','rmse_3d_h_xy_m','checkpoint_sha256']},
        after={k:results[1][k] for k in ['safe_completed','rmse_3d_h_xy_m','checkpoint_sha256']},
        relative_3d_improvement=1-results[1]['rmse_3d_h_xy_m'][0]/results[0]['rmse_3d_h_xy_m'][0],
        all_wind_traces_paired=True,rope=protocol['rope'],final_acceptance=False)
    # Verify the complete exogenous input, not just its seed.
    before=torch.load(out/'before/wind_trace.pt',weights_only=False);after=torch.load(out/'after/wind_trace.pt',weights_only=False)
    if not torch.equal(before['wind_w_mps'],after['wind_w_mps']):raise RuntimeError('Wind pairing differs')
    write(out/'comparison.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
