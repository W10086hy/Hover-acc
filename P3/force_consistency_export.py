"""Position-trained residual force reference; frozen P2 and P3 neural weights."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from P2.hover_bptt.io import read,write,digest
from px4_offboard.aligned_p3 import AlignedCalibration
from .motion_consistency import audit


def version_runtime(parent,output):
    parent,output=Path(parent),Path(output)
    if output.exists():raise FileExistsError('Preserve earlier versioned runtime export')
    parameters=read(parent);AlignedCalibration(parameters)
    if parameters.get('preflight_force_mode')!='position_consistency':raise ValueError('Position force reference required')
    parameters['processing']={**parameters['processing'],'force_trim_contract':'position_static_blend_v1'}
    AlignedCalibration(parameters);write(output,parameters)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/force_consistency_export.py','px4_offboard/aligned_p3.py','px4_offboard/acceleration_runtime.py',
        'px4_offboard/mavros_policy.py','px4_offboard/policy_timing.py','px4_offboard/replay_policy.py']
    with ZipFile(output.with_suffix('.source.zip'),'x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    proof=dict(parent_sha256=digest(parent),observer_sha256=digest(output),
        numerical_parameters_unchanged=True,P2_updates=0,P3_optimizer_updates=0,
        old_exact_processing_contract_rejects_this_file=True,
        source_sha256={n:digest(repo/n) for n in sources},
        purpose='Explicit runtime contract; use matched code so force blending cannot be silently ignored',
        formal_final_acceptance=False)
    write(output.with_suffix('.export.json'),proof)
    print(proof,flush=True);return proof


def export(capture,parent,output,*,retain_primary=False,static_blend=0.):
    capture,parent,output=Path(capture),Path(parent),Path(output)
    if output.exists():raise FileExistsError('Preserve previous force calibration experiment')
    parameters=read(parent);AlignedCalibration(parameters)
    if parameters.get('preflight_force_mode','static_residual')!='static_residual':
        raise ValueError('Refine the original Position sensor calibration only once')
    output.mkdir(parents=True)
    training=audit(capture,parent,output/'train_motion.json',partition='train')
    offset=np.array(training['baseline']['equivalent_body_acceleration_residual_mps2'])
    if np.linalg.norm(offset)>.1:raise ValueError('Large inconsistency requires primary calibration repair')
    raw_offset=offset.copy()
    if retain_primary:offset[:]=0.
    held=audit(capture,parent,output/'holdout_motion.json',offset=offset)
    passed=bool(held['baseline']['development_bias_screen_passed']
        and max(held['baseline']['world_window_error_rmse_mps2'])<=.1)
    parameters.update(preflight_force_mode='position_consistency',position_force_offset_mps2=offset.tolist(),
        static_residual_blend=static_blend,
        quality_passed=passed,status='candidate_only',closed_loop_validated=False)
    parameters['scope']+='; Position-trained residual force reference replaces static force trim; own model reference retained'
    if passed:AlignedCalibration(parameters)
    destination=output/'observer_position_force.json';write(destination,parameters)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/force_consistency_export.py','P3/motion_consistency.py','px4_offboard/aligned_p3.py',
        'px4_offboard/mavros_policy.py','px4_offboard/acceleration_runtime.py','px4_offboard/replay_policy.py']
    with ZipFile(output/'frozen_source.zip','x',ZIP_DEFLATED) as z:
        for n in sources:z.write(repo/n,n)
    report=dict(parent=str(parent),parent_sha256=digest(parent),checkpoint_sha256=parameters['checkpoint_sha256'],
        neural_weights_and_Adam_remain_at_parent=True,P2_updates=0,P3_optimizer_updates=0,
        learned_residual_force_offset_mps2=offset.tolist(),fit_uses_raw_training_partition_only=True,
        primary_Position_force_calibration_retained=retain_primary,raw_training_residual_mps2=raw_offset.tolist(),
        static_residual_blend=static_blend,
        quality_passed=passed,observer_sha256=digest(destination),
        source_sha256={n:digest(repo/n) for n in sources},
        train_proof_sha256=digest(output/'train_motion.json'),holdout_proof_sha256=digest(output/'holdout_motion.json'),
        semantics='Residual force is removed for P2, retained for the frozen own model reference; no Position input after takeover',
        status='candidate_only',formal_final_acceptance=False)
    write(output/'export.json',report);print(report,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retain-primary',action='store_true',help='Keep the primary Position force calibration without a second constant fit')
    p.add_argument('--static-blend',type=float,default=0.,help='Explicit development blend between Position and static residual force references')
    p.add_argument('--version-runtime',action='store_true',help='Export unchanged numerical parameters with an explicit matched-runtime contract')
    a=p.parse_args()
    if a.version_runtime:version_runtime(a.parent,a.output)
    else:
        if a.capture is None:p.error('--capture is required for Position calibration')
        export(a.capture,a.parent,a.output,retain_primary=a.retain_primary,static_blend=a.static_blend)
