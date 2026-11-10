"""Independent test equation/2ms-response errors; never selects residual weights."""
import argparse,math
from pathlib import Path
import torch
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from .mismatch_data import load_split,model_from_record,exclusive_directory
from .mismatch_dynamics import DynamicsResidual,HybridDynamics
from .train_mismatch import fit_loss
from .compensation import tensor_hash
from .flexible_domains import ROOT,load_protocol,protected,check_unchanged


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',default=str(ROOT/'flexible26_structural_protocol.json'))
    p.add_argument('--data',required=True);p.add_argument('--residual',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(2)
    protocol,saved,sha=load_protocol(a.protocol);learned,residual_sha=load_checkpoint(a.residual)
    if learned['controller_checkpoint_sha256']!=sha or learned['test_used_for_selection']:raise ValueError('Wrong policy or test-selected residual')
    out=exclusive_directory(a.output);net=DynamicsResidual(learned['model_configuration']).to(a.device)
    net.load_state_dict(learned['residual']);net.requires_grad_(False);net.eval();identity=tensor_hash(net.state_dict())
    baseline=DynamicsResidual(learned['model_configuration']).to(a.device);baseline.initialize_history(saved['actor']);baseline.requires_grad_(False);baseline.eval()
    configuration=protocol['residual_training'];results=[];data_files={}
    for record,data in load_split(a.data,'test',1):
        assert record['plant_kind']=='dynamic_flexible_rope' and record['controller_checkpoint_sha256']==sha
        assert record['motor_response_dt_s']==configuration['motor_prediction_dt_s']
        nominal=model_from_record(record,a.device);count=0
        losses=torch.zeros(2,3,dtype=torch.double);squared=torch.zeros(2,6,dtype=torch.double);motor_squared=torch.zeros(2,dtype=torch.double)
        log_squared=torch.zeros(2,dtype=torch.double);expected_log=math.log(record['actual_vehicle_spec']['motor_tau_s']/record['nominal_vehicle_spec']['motor_tau_s'])
        for start in range(0,len(data['obs']),1024):
            batch={key:value[start:start+1024].to(a.device) for key,value in data.items()};n=len(batch['obs']);count+=n
            for i,model in enumerate([baseline,net]):
                loss,wrench,motor=fit_loss(model,nominal,batch,configuration)
                losses[i]+=torch.tensor([float(loss),float(wrench),float(motor)],dtype=torch.double)*n
                correction=model(batch['obs'],batch['hidden'],batch['context'],batch['commands'][:,0])
                squared[i]+=((correction[:,:6]-batch['targets']).double().square().sum(0)).cpu()
                command=torch.cat((batch['commands'][:,0],torch.zeros_like(correction[:,:2])),-1)
                predicted=HybridDynamics(nominal).actuator_at(batch['initial_state'],command,configuration['motor_prediction_dt_s'],correction)[0]
                motor_squared[i]+=float((predicted-batch['motor_response_target']).double().square().sum())/4
                log_squared[i]+=float((correction[:,6]-expected_log).double().square().sum())
        values=[]
        for i,name in enumerate(['nominal','learned_residual']):
            values.append(dict(model=name,loss=float(losses[i,0]/count),normalized_wrench_mse=float(losses[i,1]/count),
                motor_mse_scaled=float(losses[i,2]/count),wrench_rmse_n_nm=(squared[i]/count).sqrt().tolist(),
                motor_response_rmse_radps=math.sqrt(float(motor_squared[i]/count)),
                log_motor_tau_rmse=math.sqrt(float(log_squared[i]/count))))
        results.append(dict(group=record['group'],seed_pair=record['seed_pair'],windows=count,
            values=values,relative_loss_improvement=1-values[1]['loss']/values[0]['loss']))
    for f in sorted((Path(a.data)/'test').glob('*.pt')):data_files[str(f)]=digest(f)
    assert tensor_hash(net.state_dict())==identity and digest(a.residual)==residual_sha
    check_unchanged({},protected(protocol))
    report=dict(completed=True,controller_checkpoint_sha256=sha,residual_checkpoint_sha256=residual_sha,groups=results,
        actual_parameter_mismatches_applied=True,neural_inputs='obs35+frozenGRU128+nominal14+motor4',
        motor_response_dt_s=configuration['motor_prediction_dt_s'],motor_units='rad/s',force_torque_units=['N']*3+['Nm']*3,
        no_policy_or_residual_updates=True,test_used_for_selection=False,test_data_sha256=data_files,
        source_sha256=digest(__file__),formal_final_acceptance=False)
    write(out/'summary.json',report);print(report,flush=True)


if __name__=='__main__':main()
