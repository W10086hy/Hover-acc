"""Held-out model-error and multi-step sensor forecasts, separate from flight scores."""
from pathlib import Path
import argparse,math
import torch
from P2.hover_bptt.io import read,write,load_checkpoint,digest
from .mismatch_data import load_split,exclusive_directory
from .mismatch_dynamics import DynamicsResidual
from .mismatch_forecast import predict_window,sensor_errors
from P2.hover_bptt.model import ExactWingOffBodyDragModel
ROOT=Path(__file__).resolve().parent
P2_ROOT=ROOT.parent/'P2'


@torch.no_grad()
def evaluate(net,groups,protocol,device):
    horizons=protocol['forecast_horizons_steps'];sums={mode:{h:{} for h in horizons} for mode in ('nominal','hybrid')};counts={h:0 for h in horizons};total=0;wrench_sum=torch.zeros(6,dtype=torch.double);group_rows=[]
    for record,data in groups:
        stride=protocol['origin_stride_steps']*protocol['forecast_origin_stride']
        selected=(data['origin']-protocol['origin_start_steps'])%stride==0
        data={k:v[selected] for k,v in data.items()};windows_count=len(data['obs']);group_rows.append(dict(group=record['group'],seed_pair=record['seed_pair'],windows=windows_count,actual_attachments=sorted({c['attachment'] for c in record['cases']})))
        for start in range(0,windows_count,128):
            batch={k:v[start:start+128].to(device) for k,v in data.items()};n=len(batch['obs']);total+=n
            origins=batch['origin'].cpu();trajectories=batch['trajectory'].cpu()
            batch['end_wind']=record['wind'][origins[:,None]+torch.arange(max(horizons))[None]+1,trajectories[:,None]].to(device)
            model=ExactWingOffBodyDragModel(record['nominal_configuration'],device=device,arms=record['nominal_arms'][trajectories].to(device),lengths=record['nominal_lengths'][trajectories].to(device))
            initial=net(batch['obs'],batch['hidden'],batch['context'],batch['commands'][:,0])
            wrench_sum+=(initial[:,:6]-batch['targets']).double().square().sum(0).cpu()
            for h in horizons:counts[h]+=n
            for mode in ('nominal','hybrid'):
                result=predict_window(net,model,record['nominal_configuration'],batch,learned=mode=='hybrid')
                error=sensor_errors(result['packets'],batch['future_packets'])
                error['velocity']=result['states'][...,3:6]-batch['future_states'][...,3:6]
                error['position']=result['states'][...,:3]-batch['future_states'][...,:3]
                for h in horizons:
                    for key,values in error.items():
                        value=values[:,h-1].double().square().sum(0).cpu()
                        if not bool(torch.isfinite(value).all()):raise RuntimeError('Nonfinite model rollout')
                        sums[mode][h][key]=sums[mode][h].get(key,torch.zeros_like(value))+value
            print(dict(stage='P3_mismatch_forecast',group=record['group']['name'],completed_windows=min(start+128,windows_count),count=windows_count),flush=True)
            if device.startswith('cuda'):torch.cuda.empty_cache()
    metrics={};scales={'attitude':.02,'gyro':.1,'specific_force':.5,'rpm':5.,'height':.1}
    for mode in ('nominal','hybrid'):
        metrics[mode]=dict(horizons={});normalized=0.
        for h in horizons:
            channels={k:(v/counts[h]).sqrt().tolist() for k,v in sums[mode][h].items()}
            aggregate={k:float((v/counts[h]).mean().sqrt()) for k,v in sums[mode][h].items()}
            metrics[mode]['horizons'][str(h)]=dict(duration_s=.01*h,windows=counts[h],rmse_by_channel=channels,rmse_by_group=aggregate)
            normalized+=sum(float((sums[mode][h][k]/counts[h]).mean())/scale**2 for k,scale in scales.items())/len(scales)/len(horizons)
        metrics[mode]['normalized_sensor_prediction_mse']=normalized
    return dict(metrics=metrics,groups=group_rows,windows=total,wrench_prediction_rmse_N_Nm=(wrench_sum/total).sqrt().tolist(),
        relative_normalized_sensor_mse_reduction=1-metrics['hybrid']['normalized_sensor_prediction_mse']/metrics['nominal']['normalized_sensor_prediction_mse'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--split',choices=['development','test'],default='test');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    torch.set_num_threads(2);saved,identity=load_checkpoint(a.checkpoint);out=exclusive_directory(a.output);protocol=read(ROOT/'p3_mismatch_protocol.json')
    net=DynamicsResidual(saved['model_configuration']).to(a.device);net.load_state_dict(saved['residual']);net.requires_grad_(False);net.eval()
    groups=load_split(a.data,a.split,max(protocol['forecast_horizons_steps']));result=evaluate(net,groups,protocol,a.device)
    write(out/'summary.json',dict(completed=True,split=a.split,residual_checkpoint_sha256=identity,controller_checkpoint_sha256=saved['controller_checkpoint_sha256'],
        **result,checkpoint_unchanged=digest(a.checkpoint)==identity,forecast_condition='Offline matched simulator initial state and common future environmental input; no future measured output conditioning; not a deployable position/wind measurement or online identification claim',
        neural_inputs='Only measured/generated obs35, frozen onboard history128, nominal14 and applied motor4',
        height_channel='Kinematic barometric-height proxy; future delayed/noisy barometer clock is not reconstructed',final_acceptance=False))
    print(dict(stage='P3_mismatch_prediction_complete',normalized_sensor_mse_reduction=result['relative_normalized_sensor_mse_reduction']),flush=True)


if __name__=='__main__':main()
