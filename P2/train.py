from pathlib import Path
import argparse,copy
import torch
from .hover_bptt.actor import Actor
from .hover_bptt.environment import DifferentiableRateEnvironment
from .hover_bptt.stream import ConvergenceStream
from .hover_bptt.cases import training_cases
from .hover_bptt.optimizer import initialize
from .hover_bptt.plant import install_training_plant
from .hover_bptt.training import guarded_update
from .hover_bptt.io import read,write,save,digest
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'tests/fixtures/baseline19.pt'))
    p.add_argument('--output',required=True);p.add_argument('--attempts',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--seed',type=int,default=20616201)
    a=p.parse_args();torch.set_num_threads(2)
    if not 1<=a.attempts<=8:raise ValueError('Bounded1–8 attempts required')
    reserved={v for pair in read(ROOT/'acceptance.json')['final_seed_pairs']for v in pair.values()}
    if {a.seed+100*i+j for i in range(a.attempts)for j in [0,1]}&reserved:raise ValueError('Final seeds cannot train')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve previous run')
    saved=torch.load(a.checkpoint,map_location=a.device,weights_only=False);cfg=copy.deepcopy(saved['configuration'])
    actor=Actor(cfg['model']).to(a.device);actor.load_state_dict(saved['actor']);actor.train();optimizer=initialize(actor,saved)
    out.mkdir(parents=True);specs=training_cases();rows=[];updates=saved['cumulative_physical_updates']
    write(out/'protocol.json',dict(source_sha256=digest(a.checkpoint),configuration=cfg,cases=specs,
        seed=a.seed,attempts=a.attempts,credit_s=10.,objective_s=120.,teacher_actions=False,final_seeds_used=False))
    for attempt in range(a.attempts):
        seed=a.seed+100*attempt;torch.manual_seed(seed)
        env=DifferentiableRateEnvironment(len(specs),seed,wind_seed=seed+1,device=a.device,stage=copy.deepcopy(cfg),episode_s=120.,specifications=specs)
        install_training_plant(env);stream=ConvergenceStream(env);stream.hidden=env.state.new_zeros(1,env.n,cfg['model']['gru_hidden'])
        result=guarded_update(stream,actor,optimizer);updates+=int(result['accepted']);rows.append(dict(attempt=attempt+1,seed_pair=[seed,seed+1],**result))
        exported=copy.deepcopy(saved);exported.update(actor={k:v.detach().cpu()for k,v in actor.state_dict().items()},optimizer=optimizer.state_dict(),
            cumulative_physical_updates=updates,optimizer_steps=updates,parent_checkpoint_sha256=digest(a.checkpoint),goal_achieved=False)
        save(out/'latest.pt',exported);write(out/'training_summary.json',dict(attempts=rows,cumulative_physical_updates=updates,goal_achieved=False))
        del stream,env
        if a.device=='cuda':torch.cuda.empty_cache()
    write(out/'terminal.json',dict(completed=True,cumulative_physical_updates=updates,checkpoint_sha256=digest(out/'latest.pt'),goal_achieved=False))
if __name__=='__main__':main()
