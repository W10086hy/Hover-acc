"""Select an unchanged BPTT candidate only after native and PyTorch checks."""
import argparse
import copy
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib
import numpy as np
from project_files import resolve_artifact

from .hover_bptt.flexible_training import REPO
from .hover_bptt.io import load_checkpoint,read,write,save,digest
from .materialize_bptt_candidate import equivalent
from .verify_stationarity import verify_evaluation

# Reviewed frozen runner before the optional explicit-case CLI was added.
# Numerical runtime still must match the candidate and every ZIP byte is checked.
APPROVED_HISTORICAL_ENTRYPOINTS={
    'P2/evaluate_stationarity.py':{'c8134122ec707aa201e99a21589f36567c5e97e1c9de6a0d75b130f927586181'}}


def _frozen_runtime(manifest,archive_path,saved):
    required={name for name in saved['source_sha256'] if name.startswith('P2/hover_bptt/')
        or name in ['P2/configuration.json','track/frozen_graph.py']}
    entrypoints=({'P2/evaluate_stationarity.py','mujoco_sim/flexible.py','mujoco_sim/physics.py'}
        if 'domains' in manifest else {'P2/check_bptt_candidate.py'})
    assert required and required|entrypoints<=manifest['source_sha256'].keys(), 'Incomplete runtime source manifest'
    for name in entrypoints:
        known=APPROVED_HISTORICAL_ENTRYPOINTS.get(name,set())|{digest(REPO/name)}
        assert manifest['source_sha256'][name] in known, 'Unapproved evaluation entrypoint'
    with ZipFile(archive_path) as archive:
        for name,sha in manifest['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==sha, 'Frozen source bytes differ'
            if name.startswith('P2/hover_bptt/') or name in ['P2/configuration.json','track/frozen_graph.py']:
                assert saved['source_sha256'][name]==sha, 'Candidate/evaluation runtime differs'
            if name.startswith('mujoco_sim/'):
                assert digest(REPO/name)==sha, 'Native physics version differs'


def _native_evidence(comparison,saved,sha):
    from .compare_stationarity import compare
    from .randomized_training import domains_for
    from .hover_bptt.generalization import cases_at
    from .hover_bptt.vehicle_spec import identity
    protocol=saved['domain_randomization']['protocol']
    expected_domains=domains_for(protocol)
    expected_cases=cases_at(protocol['training_directions_deg'],protocol['training_winds'])
    evidence=comparison['evidence']
    assert len(evidence)==len(expected_domains)==3
    before=[resolve_artifact(row['before'],directory=True) for row in evidence]
    after=[resolve_artifact(row['after'],directory=True) for row in evidence]
    assert len({path.resolve() for path in after})==3, 'Repeated native domain'
    seen=set()
    for row,old,new in zip(evidence,before,after):
        assert digest(old/'result.json')==row['before_result_sha256']
        assert digest(new/'result.json')==row['after_result_sha256']
        assert digest(new/'fullrate.npz')==row['after_fullrate_evidence_sha256']
        result=read(new/'result.json');manifest=read(new.parent/'manifest.json')
        assert manifest['checkpoint_sha256']==result['checkpoint_sha256']==sha
        assert identity(manifest['protocol'])==identity(protocol), 'Native protocol differs'
        assert manifest['duration_s']==120 and manifest['device']=='cpu'
        domain=manifest['domains'];assert len(domain)==1 and domain[0] not in seen
        index=domain[0];seen.add(index)
        assert index in (0,1,2)
        expected=expected_domains[index]
        assert identity(result['actual_vehicle_spec'])==identity(expected['actual'].to_dict())
        assert identity(result['nominal_vehicle_spec'])==identity(expected['nominal'].to_dict())
        assert result['seed_pair']==[protocol['training_seed']+10*index+j for j in (0,1)]
        assert identity(result['offline_limits'])==identity(saved['configuration']['offline_limits'])
        assert len(result['cases'])==len(expected_cases)==8
        for case,wanted in zip(result['cases'],expected_cases):
            for field in ['attachment_b_m','wind_speed','direction_rad','wind_profile']:
                assert case[field]==wanted[field], 'Native case grid differs'
        with np.load(new/'fullrate.npz') as raw:
            np.testing.assert_array_equal(raw['target_w_m'],[0.,0.,-5.])
        _frozen_runtime(manifest,new.parent/'frozen_source.zip',saved)
    fresh=compare(before,after)
    parent_sha=saved['bptt_candidate_materialization']['parent_checkpoint_sha256']
    assert parent_sha==saved['parent_checkpoint_sha256']
    assert fresh['summary']['before']['checkpoint_sha256']==parent_sha, 'Comparison is not against the BPTT parent'
    assert fresh['summary']==comparison['summary'], 'Stale native summary'
    assert fresh['rows']==comparison['rows'], 'Stale native case rows'
    assert fresh['native_safe_and_worst_motion_improved']==comparison['native_safe_and_worst_motion_improved']
    return fresh['fullrate_audits']


def _guard_evidence(guard,guard_path,saved,sha):
    from .hover_bptt.flexible_training import paired_decision
    from .hover_bptt.stationarity import priority_decision
    from .hover_bptt.vehicle_spec import identity
    protocol=saved['domain_randomization']['protocol']
    manifest=read(Path(guard_path).parent/'manifest.json')
    assert manifest['checkpoint_sha256']==sha
    assert identity(manifest['protocol'])==identity(protocol)
    _frozen_runtime(manifest,Path(guard_path).parent/'frozen_source.zip',saved)
    reference=resolve_artifact(manifest['reference_run'],directory=True)
    original=read(reference/'training_summary.json')
    original_manifest=read(reference/'run_manifest.json')
    direction=saved['bptt_candidate_materialization']
    assert original_manifest['parent_checkpoint_sha256']==direction['parent_checkpoint_sha256']==saved['parent_checkpoint_sha256']
    assert digest(reference/'proposal.pt')==direction['proposal_checkpoint_sha256'], 'Wrong retained BPTT direction'
    assert identity(original_manifest['protocol'])==identity(protocol)
    assert guard['baseline']==original['baseline']
    assert len(guard['scores'])==len(guard['baseline'])==2
    reasons=[]
    for index,(base,score) in enumerate(zip(guard['baseline'],guard['scores'])):
        assert score['duration_s']==30 and score['objective_start_s']==90
        assert len(score['completed'])==8
        assert score['late_convergence']['expected_samples']==3000
        decision=paired_decision(base,score,protocol['objective_s'],protocol.get('height_absolute_tolerance_m',.01))
        secondary={'full_cost_not_lower','aggregate_position_regression'}
        cap=protocol.get('stationarity_height_rmse_cap_m')
        if cap is not None:
            secondary.add('safe_height_regression')
            if any(value>cap for value in score['height_rmse_m']):reasons.append(f'domain{index}:height_rmse_cap_exceeded')
        reasons.extend(f'domain{index}:{reason}' for reason in decision['reasons'] if reason not in secondary)
    priority=priority_decision(guard['baseline'],guard['scores'],protocol['stationarity_BPTT'])
    reasons.extend(priority['reasons'])
    assert guard['reasons']==reasons and guard['training_candidate_guard_passed']==(not reasons), 'Stale PyTorch guard'
    assert guard['stationarity_priority']==priority, 'Stale PyTorch motion decision'
    assert not reasons


def select(checkpoint,comparison_path,guard_path,output):
    output=Path(output).resolve()
    if not output.is_relative_to(REPO.resolve()):raise ValueError('Selection must stay in this project')
    if output.exists():raise FileExistsError(output)
    saved,sha=load_checkpoint(checkpoint)
    comparison=read(comparison_path);guard=read(guard_path)
    after=comparison['summary']['after']
    assert saved.get('unvalidated_bptt_proposal') and not saved.get('selected_for_use')
    assert after['checkpoint_sha256']==guard['checkpoint_sha256']==sha
    assert comparison['duration_s']==120 and after['count']==after['safe_completed']==24
    assert after['strong_count']==12 and comparison['native_safe_and_worst_motion_improved']
    assert guard['completed'] and guard['training_candidate_guard_passed'] and not guard['reasons']
    assert guard['count']==guard['safe_completed']==16 and guard['causal_forward_horizon_s']==120
    assert guard['motion_window_s']==30 and guard['objective_s']==30
    assert guard['policy_weights_unchanged'] and guard['initial_state_exactly_paired'] and guard['wind_exactly_paired']
    assert not guard['gradient_computed'] and guard['optimizer_updates']==0
    _guard_evidence(guard,guard_path,saved,sha)
    audits=_native_evidence(comparison,saved,sha)
    output.mkdir(parents=True)
    selected=copy.deepcopy(saved)
    selected.update(unvalidated_bptt_proposal=False,selected_for_use=True,
        bptt_selection=dict(candidate_checkpoint_sha256=sha,
            native_comparison=str(comparison_path),native_comparison_sha256=digest(comparison_path),
            pytorch_guard=str(guard_path),pytorch_guard_sha256=digest(guard_path),
            selection_source_sha256=digest(__file__),extra_optimizer_updates=0),
        goal_achieved=False,formal_final_acceptance=False)
    save(output/'latest.pt',selected)
    proof=equivalent(checkpoint,output/'latest.pt')
    with ZipFile(output/'selection_source.zip','x',ZIP_DEFLATED) as archive:
        archive.write(__file__,'P2/select_bptt_candidate.py')
        archive.write(Path(__file__).with_name('materialize_bptt_candidate.py'),'P2/materialize_bptt_candidate.py')
    report=dict(selected_checkpoint_sha256=digest(output/'latest.pt'),candidate_checkpoint_sha256=sha,
        checkpoint_equivalence=proof,cumulative_physical_updates=selected['cumulative_physical_updates'],
        optimizer_updates_in_selection=0,native_audits=audits,
        source_archive_sha256=digest(output/'selection_source.zip'),
        comparison_sha256=digest(comparison_path),pytorch_guard_sha256=digest(guard_path),
        strong_stationary=after['strong_stationary'],strong_count=12,
        goal_achieved=False,formal_final_acceptance=False)
    write(output/'selection_audit.json',report)
    return {k:v for k,v in report.items() if k not in ['native_audits','checkpoint_equivalence']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','comparison','pytorch-guard','output']:parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    print(select(args.checkpoint,args.comparison,args.pytorch_guard,args.output),flush=True)


if __name__=='__main__':main()
