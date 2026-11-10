"""Fit equation-error wrench and actuator dynamics; P2 controller stays frozen."""
from pathlib import Path
import argparse,time,copy
import torch
from P2.hover_bptt.io import read,write,save,load_checkpoint,digest
from .mismatch_dynamics import DynamicsResidual,HybridDynamics
from .mismatch_data import load_split,model_from_record,exclusive_directory,source_manifest,unchanged,validate_protocol,reserved_seeds
ROOT=Path(__file__).resolve().parent
P2_ROOT=ROOT.parent/'P2'


def fit_loss(net,nominal,batch,configuration):
    correction=net(batch['obs'],batch['hidden'],batch['context'],batch['commands'][:,0])
    scale=correction.new_tensor(configuration['wrench_loss_scale'])
    wrench=((correction[:,:6]-batch['targets'])/scale).square().mean()
    commands=torch.cat((batch['commands'][:,0],torch.zeros_like(correction[:,:2])),-1)
    rpm=HybridDynamics(nominal).actuator_at(batch['initial_state'],commands,configuration.get('motor_prediction_dt_s',.01),correction)[0]
    motor_target=batch['motor_response_target'] if 'motor_response_target' in batch else batch['future_states'][:,0,13:17]
    motor=((rpm-motor_target)/configuration['rpm_loss_scale']).square().mean()
    return (wrench+motor)/2+configuration['regularization']*(correction/net.scale).square().mean(),wrench,motor


@torch.no_grad()
def score(net,groups,configuration,device):
    total=0;values=[0.,0.,0.]
    for record,data in groups:
        nominal=model_from_record(record,device)
        for start in range(0,len(data['obs']),1024):
            batch={k:v[start:start+1024].to(device) for k,v in data.items()};n=len(batch['obs'])
            loss,wrench,motor=fit_loss(net,nominal,batch,configuration);total+=n
            for i,x in enumerate((loss,wrench,motor)):values[i]+=float(x)*n
    return dict(loss=values[0]/total,normalized_wrench_mse=values[1]/total,rpm_mse_scaled=values[2]/total,windows=total)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--protocol',default=str(ROOT/'flexible26_structural_protocol.json'));a=p.parse_args()
    torch.set_num_threads(2);protocol=read(a.protocol);validate_protocol(protocol,reserved_seeds(read(P2_ROOT/'acceptance.json'),read(P2_ROOT/'p2_protocol.json')))
    if not read(Path(a.data)/'summary.json')['completed']:raise ValueError('Complete all frozen-policy collection first')
    cfg=protocol['residual_training'];torch.manual_seed(cfg['seed']);out=exclusive_directory(a.output)
    if protocol.get('plant_kind')=='dynamic_flexible_rope':
        from .flexible_domains import sources,load_protocol,freeze
        load_protocol(a.protocol);manifest=sources(a.protocol);freeze(out,manifest)
    else:manifest=source_manifest(ROOT)
    saved,parent_sha=load_checkpoint(ROOT/protocol['controller_checkpoint']);train=load_split(a.data,'train',1);development=load_split(a.data,'development',1)
    if protocol.get('plant_kind')=='dynamic_flexible_rope':
        data_protocol=read(Path(a.data)/'protocol.json')
        if data_protocol['controller_checkpoint_sha256']!=parent_sha or data_protocol['protocol']!=protocol:
            raise ValueError('Flexible training data must match the frozen-policy protocol exactly')
        for record,_ in train+development:
            if (record.get('plant_kind')!='dynamic_flexible_rope' or record['controller_checkpoint_sha256']!=parent_sha
                or record['rope']!=protocol['rope'] or record['states'].shape[-1]!=86
                or record['motor_response_dt_s']!=cfg['motor_prediction_dt_s']):
                raise ValueError('Wrong flexible plant, frozen policy, rope, or motor-response horizon')
    excluded=[]
    # This pilot calibrates one nominal tether geometry (5cm); 10cm is an
    # unknown actual geometry. Matched10cm labels remain a physics diagnostic,
    # not an ambiguous training target for a model calibrated to nominal5cm.
    for record,data in train:
        mask=(record['nominal_arms'][data['trajectory'],2]-.05).abs()<1e-6
        excluded.append(dict(group=record['group']['name'],outside_nominal5cm=int((~mask).sum())))
        for key in data:data[key]=data[key][mask]
    net=DynamicsResidual(protocol['model']).to(a.device);net.initialize_history(saved['actor'])
    optimizer=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    devices=[({k:v.to(a.device) for k,v in data.items()},model_from_record(record,a.device)) for record,data in train]
    generator=torch.Generator(device=a.device).manual_seed(cfg['seed']);history=[];best=float('inf');best_step=None;started=time.perf_counter()
    write(out/'protocol.json',dict(protocol=protocol,source_sha256=manifest,controller_checkpoint_sha256=parent_sha,data_protocol_sha256=digest(Path(a.data)/'protocol.json'),
        training_seed_pairs=[r['seed_pair'] for r,_ in train],development_seed_pairs=[r['seed_pair'] for r,_ in development],nominal_geometry_scope_cm=5,excluded_nonreference_geometry=excluded,test_used_for_selection=False))
    for step in range(cfg['steps']+1):
        if step%cfg['development_interval']==0 or step==cfg['steps']:
            net.eval();value=score(net,development,cfg,a.device);history.append(dict(step=step,**value,wall_s=time.perf_counter()-started))
            if value['loss']<best:
                best=value['loss'];best_step=step
                save(out/'best.pt',dict(schema_version=1,residual={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=copy.deepcopy(optimizer.state_dict()),model_configuration=protocol['model'],
                    selected_step=step,development_score=value,controller_checkpoint_sha256=parent_sha,source_sha256=manifest,protocol=protocol,nominal_geometry_scope_cm=5,test_used_for_selection=False))
            write(out/'training_summary.json',dict(completed=False,history=history,selected_step=best_step,selected_loss=best,gradient_steps=step))
            print(dict(stage='P3_mismatch_fit',step=step,development=value,best_step=best_step,wall_s=time.perf_counter()-started),flush=True)
        if step==cfg['steps']:break
        net.train();net.history.eval();data,nominal=devices[step%len(devices)]
        index=torch.randint(len(data['obs']),(cfg['batch_size'],),device=a.device,generator=generator);batch={k:v[index] for k,v in data.items()}
        loss,_,_=fit_loss(net,nominal,batch,cfg);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    write(out/'training_summary.json',dict(completed=True,history=history,selected_step=best_step,selected_loss=best,gradient_steps=cfg['steps'],batch_size=cfg['batch_size'],
        wall_s=time.perf_counter()-started,controller_unchanged=digest(ROOT/protocol['controller_checkpoint'])==parent_sha,source_unchanged=unchanged(ROOT,manifest),
        selected_checkpoint_sha256=digest(out/'best.pt'),improved_development=best<history[0]['loss'],test_used_for_selection=False,final_acceptance=False))


if __name__=='__main__':main()
