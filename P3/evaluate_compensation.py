"""Same frozen BPTT/Adam, actual-plant comparison with residual compensation off/on."""
from pathlib import Path
import argparse,copy,time
import torch
from P2.hover_bptt.io import read,write,save,load_checkpoint,digest
from .mismatch_data import exclusive_directory,validate_protocol,reserved_seeds,source_manifest,unchanged
from .mismatch_dynamics import DynamicsResidual
from .compensation import CompensationEnvironment,tensor_hash
from P2.hover_bptt.environment import RateEnvironment,plant_config
from P2.hover_bptt.vehicle_spec import VehicleSpec
from P2.hover_bptt.generalization import cases_at
from P2.hover_bptt.evaluation import evaluate_one
ROOT=Path(__file__).resolve().parent
P2_ROOT=ROOT.parent/'P2'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--residual',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    torch.set_num_threads(2);protocol=read(ROOT/'p3_mismatch_protocol.json');validate_protocol(protocol,reserved_seeds(read(P2_ROOT/'acceptance.json'),read(P2_ROOT/'p2_protocol.json')))
    checkpoint=ROOT/protocol['controller_checkpoint'];saved,parent_sha=load_checkpoint(checkpoint);learned,residual_sha=load_checkpoint(args.residual)
    if learned['controller_checkpoint_sha256']!=parent_sha or learned['nominal_geometry_scope_cm']!=protocol['nominal_arm_cm']:raise ValueError('Residual was fitted with a different frozen controller or nominal geometry')
    out=exclusive_directory(args.output);manifest=source_manifest(ROOT);duration=protocol['development_comparison_duration_s'];base=plant_config();nominal=VehicleSpec.from_config(base);actual=nominal.scaled(**protocol['control_actual_scales']);cases=cases_at(protocol['control_evaluation_directions_deg'])
    for case in cases:case['actual_body_drag_scale']=protocol['control_actual_drag_scale']
    write(out/'protocol.json',dict(protocol=protocol,source_sha256=manifest,policy_checkpoint_sha256=parent_sha,residual_checkpoint_sha256=residual_sha,policy_updates=0,optimizer_created=False,actual_plant_adds_no_residual=True))
    net=DynamicsResidual(learned['model_configuration']).to(args.device);net.load_state_dict(learned['residual']);net.requires_grad_(False);net.eval();net_hash=tensor_hash(net.state_dict())
    policies=[];wind_sha=None;started=time.perf_counter()
    for enabled in (False,True):
        if not unchanged(ROOT,manifest):raise RuntimeError('Source changed during frozen compensation comparison')
        name='with_compensation' if enabled else 'without_compensation';built=[]
        def factory(*positional,**keywords):
            env=CompensationEnvironment(*positional,residual_net=net,compensation=protocol['compensation'],**keywords) if enabled else RateEnvironment(*positional,**keywords)
            built.append(env);return env
        result=evaluate_one(checkpoint,out/name,protocol['control_evaluation_seed'],cases,device=args.device,vehicle_spec=actual,nominal_vehicle_spec=nominal,plant_configuration=base,
            checkpoint_snapshot=(saved,parent_sha),duration_s=duration,environment_factory=factory,stage_label=f'P3冻结BPTT21次；'+('有残差前馈补偿' if enabled else '无残差补偿'))
        witness=digest(out/name/'wind_trace.pt')
        if wind_sha is None:wind_sha=witness
        if wind_sha!=witness or digest(checkpoint)!=parent_sha:raise RuntimeError('Fixed BPTT/Adam or paired actual wind changed')
        if enabled:
            evidence=built[0].controller.evidence();save(out/name/'compensation_trace.pt',evidence)
            write(out/name/'compensation_summary.json',dict(bounded_prediction_rms=evidence['bounded_prediction'].square().mean((0,1)).sqrt().tolist(),applied_output_rms=evidence['applied_correction'].square().mean((0,1)).sqrt().tolist(),
                fallback_count=int(evidence['fallback'].sum()),control_steps=len(evidence['bounded_prediction']),options=protocol['compensation'],actual_plant_unchanged=True,horizontal_force_directly_actuated=False,
                frozen_residual_memory_unchanged=tensor_hash(net.state_dict())==net_hash))
        policies.append(dict(policy=name,**{k:result[k] for k in ('count','safe_completed','rmse_3d_h_xy_m','weak_exceedances','strong_exceedances','duration_s','seed_pair','checkpoint_sha256','cumulative_physical_updates','policy_weights_unchanged','maximum_true_yaw_error_deg')},result_file=name+'/result.json'))
        write(out/'summary.json',dict(completed=False,policies=policies))
    if tensor_hash(net.state_dict())!=net_hash or digest(args.residual)!=residual_sha:raise RuntimeError('Frozen residual changed')
    write(out/'summary.json',dict(completed=True,policies=policies,count_per_policy=len(cases),duration_s=duration,seed_pair=[protocol['control_evaluation_seed'],protocol['control_evaluation_seed']+1],
        policy_updates=0,optimizer_created=False,checkpoint_and_adam_file_unchanged=digest(checkpoint)==parent_sha,residual_unchanged=True,residual_checkpoint_sha256=residual_sha,
        actual_wind_archive_sha256=wind_sha,independent_actual_plant=True,residual_force_added_to_actual_plant=False,source_unchanged=unchanged(ROOT,manifest),wall_s=time.perf_counter()-started,final_acceptance=False))
if __name__=='__main__':main()
