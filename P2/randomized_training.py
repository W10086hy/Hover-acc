"""Domain-randomized continuation behind the standard flexible-rope CLIs."""
import copy
from functools import partial
from pathlib import Path
import math
import time
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from .hover_bptt.actor import Actor
from .hover_bptt.domain_randomization import sample_domains, describe_domains
from .hover_bptt.environment import plant_config
from .hover_bptt.flexible_environment import FlexibleRopeEnvironment
from .hover_bptt.flexible_training import (
    REPO, ROOT, FlexibleRopeStream, accumulate_domain_gradients, configure_learning_rates,
    paired_domain_update, source_manifest,
)
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, load_checkpoint, read, save, write
from .hover_bptt.optimizer import initialize, install_gradient_masks, parameter_names
from .hover_bptt.onboard_graph import install_onboard
from .hover_bptt.plant import install_training_plant
from .hover_bptt.vehicle_spec import VehicleSpec, identity


def domains_for(protocol, development=False):
    options = copy.deepcopy(protocol['domain_randomization'])
    if development:
        options['seed'] = protocol['development_parameter_seed']
    return sample_domains(VehicleSpec.from_config(plant_config()), options)


def create_streams(protocol, saved, domains, seconds, device):
    cfg = copy.deepcopy(saved['configuration'])
    cfg['training']['report_credit_progress'] = seconds >= 10
    if protocol.get('stationarity_BPTT'):
        cfg['training']['stationarity_BPTT']=copy.deepcopy(protocol['stationarity_BPTT'])
        cfg['training']['stationarity_metric_window_s']=protocol.get('stationarity_metric_window_s',10)
    cases = cases_at(protocol['training_directions_deg'], protocol.get('training_winds'))
    streams = []
    for index, domain in enumerate(domains):
        if index not in protocol.get('training_domain_indices',range(len(domains))):continue
        seed = protocol['training_seed'] + 10 * index
        warmup_s=protocol.get('warmup_s_by_domain',[0]*len(domains))[index]
        env = FlexibleRopeEnvironment(len(cases), seed, wind_seed=seed+1, device=device,
            stage=copy.deepcopy(cfg), episode_s=warmup_s+seconds, specifications=cases, rope=protocol['rope'],
            vehicle_spec=domain['actual'], nominal_vehicle_spec=domain['nominal'],
            random_device=protocol.get('training_random_device'))
        install_training_plant(env)
        install_onboard(env)
        stream = FlexibleRopeStream(env)
        stream.warmup_steps=round(warmup_s/env.dt)
        stream.hidden = env.state.new_zeros(1, env.n, cfg['model']['gru_hidden'])
        streams.append(stream)
    return streams, cfg, cases


def training_cohorts(protocol,saved,domains,seconds,device,actor):
    if protocol.get('cohort_threads',1)>1:
        from .parallel_cohorts import create_cohorts
        return create_cohorts(protocol,saved,domains,seconds,device,actor)
    streams,cfg,cases=create_streams(protocol,saved,domains,seconds,device)
    return streams,cfg,cases,None


def freeze_sources(output, protocol_path):
    sources = source_manifest(protocol_path)
    with ZipFile(output / 'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(REPO / name, name)
    return sources


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _same_saved_state(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(_same_saved_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)):
        return len(a)==len(b) and all(_same_saved_state(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def _check_numeric_factories(reference,manifest,sources):
    """Permit orchestration edits only when the sampled rollout factory is unchanged."""
    name='P2/randomized_training.py'
    if manifest['source_sha256'].get(name)==sources[name]:return
    import ast,hashlib
    with ZipFile(reference/'frozen_source.zip') as archive:old=archive.read(name)
    if hashlib.sha256(old).hexdigest()!=manifest['source_sha256'].get(name):
        raise ValueError('Timing reference frozen orchestration bytes differ')
    def signature(source):
        nodes=ast.parse(source).body
        selected={'domains_for','create_streams','training_cohorts'}
        return [ast.dump(node,include_attributes=False) for node in nodes if
            isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,ast.FunctionDef) and node.name in selected]
    if signature(old)!=signature((REPO/name).read_bytes()):
        raise ValueError('Timing reference numeric rollout factory differs')


def reuse_timing(reference,protocol,protocol_path):
    """Reuse measured runtime only; the new full gradient is still computed."""
    reference=Path(reference).resolve()
    if not reference.is_relative_to(REPO.resolve()):raise ValueError('Reference must stay in this project')
    manifest=read(reference/'run_manifest.json');summary=read(reference/'training_summary.json')
    parent=(ROOT/protocol['parent_checkpoint']).resolve()
    if digest(reference/'latest.pt')!=summary['checkpoint_sha256']:
        raise ValueError('Timing reference checkpoint bytes differ')
    accepted=summary['accepted_updates']
    if not summary['completed'] or accepted not in (0,1) or not math.isfinite(summary['gradient_norm']) or summary['gradient_norm']<=0:
        raise ValueError('A completed finite-gradient update is required')
    if accepted:
        if parent!=reference/'latest.pt' or digest(parent)!=summary['checkpoint_sha256']:
            raise ValueError('Reference checkpoint must be the actual next parent')
    else:
        original=(ROOT/manifest['protocol']['parent_checkpoint']).resolve()
        if parent!=original or digest(parent)!=manifest['parent_checkpoint_sha256']:
            raise ValueError('Rejected timing reference requires the unchanged original parent')
        baseline,_=load_checkpoint(parent);rejected,_=load_checkpoint(reference/'latest.pt')
        if any(not _same_saved_state(baseline[key],rejected[key]) for key in
                ['actor','optimizer','optimizer_parameter_names','cumulative_physical_updates','optimizer_steps']):
            raise ValueError('Rejected proposal did not exactly restore Actor and Adam')
    if not math.isfinite(summary['wall_s']) or summary['wall_s']<=0 or manifest['qualification']['device']!='cuda':
        raise ValueError('Measured CUDA iteration time required')
    old=manifest['protocol']
    keys=['objective_s','credit_s','recompute_s','rope','domain_randomization','training_domain_indices',
        'warmup_s_by_domain','training_directions_deg','training_winds','stationarity_BPTT','late_stationarity_window','training_random_device','cohort_threads','stationarity_metric_window_s']
    if any(identity(old.get(k))!=identity(protocol.get(k)) for k in keys):
        raise ValueError('Timing reuse requires identical physics, loss, windows and batch shape')
    sources=source_manifest(protocol_path)
    for name,sha in sources.items():
        if name.startswith('P2/hover_bptt/') or name in ['track/frozen_graph.py','P2/configuration.json','P2/parallel_cohorts.py']:
            if manifest['source_sha256'].get(name)!=sha:
                raise ValueError('Timing reference runtime source differs: '+name)
    _check_numeric_factories(reference,manifest,sources)
    domains=describe_domains(domains_for(protocol))
    if identity(domains)!=identity(manifest['domains']):raise ValueError('Timing reference parameters differ')
    return dict(passed=True,timing_only=True,gradient_precheck_performed=False,device='cuda',
        seconds=old['objective_s'],parent_checkpoint_sha256=digest(parent),protocol_identity=identity(protocol),
        domains=domains,source_sha256=sources,optimizer_created=False,
        reference_directory=reference.relative_to(REPO).as_posix(),
        reference_actual_iteration_s=summary['wall_s'],reference_trials=len(summary['trials']),
        reference_accepted_updates=accepted,rejected_parent_restore_verified=not bool(accepted),
        reference_summary_sha256=digest(reference/'training_summary.json'),
        estimation='Previous complete iteration with identical runtime/batch;not a gradient precheck of the new parent')


def preserve_optimizer_proposal(optimizer,actor,path,saved,*,configuration,parent_sha,sources,protocol,domains):
    """Save the exact post-Adam direction before any line-search rollback."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    native_step=optimizer.step;record=dict(optimizer_steps=0)
    def step(*args,**kwargs):
        if record['optimizer_steps']:raise RuntimeError('One Actor optimizer proposal per bounded update required')
        value=native_step(*args,**kwargs)
        proposal=copy.deepcopy(saved)
        proposal.update(configuration=copy.deepcopy(configuration),
            actor={name:p.detach().cpu().clone() for name,p in actor.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()),optimizer_parameter_names=parameter_names(actor,optimizer),
            cumulative_physical_updates=saved['cumulative_physical_updates']+1,
            optimizer_steps=saved['optimizer_steps']+1,parent_checkpoint_sha256=parent_sha,
            source_sha256=sources,domain_randomization=dict(protocol=protocol,domains=domains),
            unvalidated_bptt_proposal=True,selected_for_use=False,p3_enabled=False,goal_achieved=False)
        save(path,proposal)
        record.update(optimizer_steps=1,checkpoint_sha256=digest(path),checkpoint=path.name)
        return value
    optimizer.step=step
    return record


def qualify(args, protocol):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    saved, parent_sha = load_checkpoint(ROOT / protocol['parent_checkpoint'])
    domains = domains_for(protocol)
    sources = freeze_sources(output, args.protocol)
    actor = Actor(saved['configuration']['model']).to(args.device)
    actor.load_state_dict(saved['actor'])
    actor.train()
    install_gradient_masks(actor)
    started = time.perf_counter()
    streams, _, cases, executor = training_cohorts(protocol, saved, domains, args.seconds, args.device, actor)
    synchronize(args.device)
    setup_s = time.perf_counter() - started
    started = time.perf_counter()
    try:
        results = accumulate_domain_gradients(streams, actor, steps=round(args.seconds/.01),
            credit_steps=round(min(protocol['credit_s'], args.seconds)/.01),
            recompute_steps=round(min(protocol['recompute_s'], args.seconds)/.01),executor=executor)
    finally:
        if executor is not None:executor.close()
    synchronize(args.device)
    gradient_s = time.perf_counter() - started
    norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'), error_if_nonfinite=True))
    if norm <= 0 or not all(torch.equal(value.detach().cpu(), saved['actor'][name]) for name, value in actor.state_dict().items()):
        raise RuntimeError('Qualification must produce nonzero gradients with unchanged weights')
    if source_manifest(args.protocol) != sources:
        raise RuntimeError('Source changed during qualification')
    warmup_s=(results[0]['diagnostics']['parallel_warmup_wall_s'] if executor is not None else
        sum(result['diagnostics']['warmup_wall_s'] for result in results))
    estimate = (gradient_s-warmup_s) * protocol['objective_s'] / args.seconds
    report = dict(passed=True, seconds=args.seconds, optimizer_created=False, optimizer_updates=0,
        parent_checkpoint_sha256=parent_sha, protocol_identity=identity(protocol), rope=protocol['rope'],
        source_sha256=sources, domains=describe_domains(domains), setup_s=setup_s, gradient_s=gradient_s,
        domain_count=len(streams), available_domain_count=len(domains), cases_per_domain=len(cases), gradient_norm=norm,
        device=args.device, gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        measured_warmup_wall_s=warmup_s,warmup_s_by_domain=protocol.get('warmup_s_by_domain',[0]*len(domains)),
        training_random_device=protocol.get('training_random_device','physical_device'),
        cohort_threads=protocol.get('cohort_threads',1),
        estimated_objective_gradient_s=estimate+warmup_s,
        estimated_one_candidate_iteration_s=setup_s+2*warmup_s+estimate*1.5,
        estimation='Short-horizon gradient extrapolation plus measured full causal warmup for baseline and candidate',
        domain_scores=[result['score'] for result in results])
    write(output / 'qualification.json', report)
    print({key: value for key, value in report.items() if key not in ('source_sha256', 'domains', 'domain_scores')}, flush=True)


def train(args, protocol, qualification):
    if qualification['protocol_identity'] != identity(protocol) or qualification['device'] != 'cuda':
        raise ValueError('Use matching randomized GPU qualification')
    if identity(qualification['domains']) != identity(describe_domains(domains_for(protocol))):
        raise ValueError('Qualified parameter samples differ')
    saved, parent_sha = load_checkpoint(ROOT / protocol['parent_checkpoint'])
    if parent_sha != qualification['parent_checkpoint_sha256']:
        raise ValueError('Qualification parent differs')
    for name, sha in qualification['source_sha256'].items():
        if digest(REPO / name) != sha:
            raise ValueError('Qualification source differs: ' + name)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = freeze_sources(output, args.protocol)
    domains = domains_for(protocol)
    actor = Actor(saved['configuration']['model']).cuda()
    actor.load_state_dict(saved['actor'])
    actor.train()
    optimizer = initialize(actor, saved)
    old_rates, rates = configure_learning_rates(optimizer, protocol.get('optimizer_learning_rates'))
    protected = {str(ROOT / protocol['parent_checkpoint']): parent_sha,
        str(REPO / 'P3/retest8/residual/best.pt'): digest(REPO / 'P3/retest8/residual/best.pt')}
    write(output / 'run_manifest.json', dict(protocol=protocol, source_sha256=sources, domains=describe_domains(domains),
        parent_checkpoint_sha256=parent_sha, original_adam_restored=True,
        optimizer_lrs_before=old_rates, optimizer_lrs_after=rates, protected_files=protected,
        qualification=qualification, p3_enabled=False, formal_final_acceptance=False))
    state_path = REPO / 'project_state.json'
    state = read(state_path)
    state.update(training_running=True, domain_randomization_stage='P2_randomized_30s_update')
    write(state_path, state)
    started = time.perf_counter()
    executor=None
    try:
        streams, cfg, cases, executor = training_cohorts(protocol, saved, domains, protocol['objective_s'], 'cuda', actor)
        save(output / 'training_witness.pt', dict(domains=describe_domains(domains), dt_s=.01, cases=cases,
            active_domain_indices=protocol.get('training_domain_indices',list(range(len(domains)))),
            random_devices=[stream.env.random_device for stream in streams],
            initial_state=[stream.env.state.detach().cpu() for stream in streams],
            wind=[stream.env.wind_field.trace.detach().cpu() for stream in streams]))
        proposal_record=preserve_optimizer_proposal(optimizer,actor,output/'proposal.pt',saved,
            configuration=cfg,parent_sha=parent_sha,sources=sources,protocol=protocol,domains=describe_domains(domains))
        result = paired_domain_update(streams, actor, optimizer, protocol,executor=executor)
        if proposal_record['optimizer_steps']!=1:raise RuntimeError('Expected exactly one retained Adam proposal')
        if not result['accepted']:
            optimizer.load_state_dict(saved['optimizer'])
        if source_manifest(args.protocol) != sources or any(digest(path) != sha for path, sha in protected.items()):
            raise RuntimeError('Sources or protected weights changed')
        checkpoint = copy.deepcopy(saved)
        updates = int(result['accepted'])
        checkpoint.update(configuration=cfg, actor={name: value.detach().cpu() for name, value in actor.state_dict().items()},
            optimizer=optimizer.state_dict(), optimizer_parameter_names=parameter_names(actor, optimizer),
            cumulative_physical_updates=saved['cumulative_physical_updates']+updates,
            optimizer_steps=saved['optimizer_steps']+updates, parent_checkpoint_sha256=parent_sha,
            source_sha256=sources, domain_randomization=dict(protocol=protocol, domains=describe_domains(domains)),
            p3_enabled=False, goal_achieved=False)
        save(output / 'latest.pt', checkpoint)
        write(output / 'training_summary.json', dict(completed=True, accepted_updates=updates,
            checkpoint_sha256=digest(output / 'latest.pt'), wall_s=time.perf_counter()-started,
            retained_unvalidated_proposal=proposal_record,
            domain_count=len(streams), available_domain_count=len(domains), cases_per_domain=len(cases), **result))
    except BaseException as error:
        write(output / 'failure.json', dict(error=repr(error), wall_s=time.perf_counter()-started))
        raise
    finally:
        if executor is not None:executor.close()
        state = read(state_path)
        state.update(training_running=False, domain_randomization_stage='P2_update_finished',
            domain_randomization_output=output.resolve().relative_to(REPO).as_posix())
        write(state_path, state)


def evaluate(args, protocol):
    from .evaluate_flexible_rope import factory
    from .hover_bptt.evaluation import evaluate_one

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = freeze_sources(output, args.protocol)
    baseline = Path(args.baseline_checkpoint) if args.baseline_checkpoint else ROOT / protocol['parent_checkpoint']
    domains = domains_for(protocol, development=True)
    cases = cases_at(protocol['development_directions_deg'], protocol.get('development_winds'))
    rows = []
    for index, domain in enumerate(domains):
        pair = {}
        seed = protocol['development_seed'] + 10 * index
        for label, checkpoint in [('before', baseline), ('after', Path(args.checkpoint))]:
            folder = output / domain['name'] / label
            folder.parent.mkdir(exist_ok=True)
            score = evaluate_one(checkpoint, folder, seed, cases, device=args.device, duration_s=args.seconds,
                vehicle_spec=domain['actual'], nominal_vehicle_spec=domain['nominal'],
                plant_configuration=plant_config(), environment_factory=partial(factory, rope=protocol['rope']),
                stage_label='P2 randomized structural parameters; independent development')
            score.update(rope=protocol['rope'],rope_parameters_measured=False,engine='PyTorch')
            write(folder/'result.json',score)
            pair[label] = score
        before = torch.load(output/domain['name']/'before/wind_trace.pt', map_location='cpu', weights_only=False)
        after = torch.load(output/domain['name']/'after/wind_trace.pt', map_location='cpu', weights_only=False)
        if not torch.equal(before['wind_w_mps'], after['wind_w_mps']):
            raise RuntimeError('Development winds differ')
        rows.append(dict(domain=describe_domains([domain])[0], seed_pair=[seed, seed+1], **pair))
    aggregate = {}
    for label in ('before', 'after'):
        aggregate[label] = dict(safe=sum(row[label]['safe_completed'] for row in rows),
            count=len(domains)*len(cases),
            rmse_3d_h_xy_m=[math.sqrt(sum(row[label]['rmse_3d_h_xy_m'][i]**2 for row in rows)/len(rows)) for i in range(3)])
    if source_manifest(args.protocol) != sources:
        raise RuntimeError('Source changed during evaluation')
    write(output/'comparison.json', dict(duration_s=args.seconds, domains=rows, aggregate=aggregate,
        development_parameter_seed=protocol['development_parameter_seed'], formal_final_acceptance=False))
    print(aggregate, flush=True)
