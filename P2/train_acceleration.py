"""Train and validate the three-output acceleration policy on the flexible cable."""
import argparse
import copy
from itertools import product
import math
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from .hover_bptt.actor import Actor
from .hover_bptt.domain_randomization import describe_domains, sample_domains
from .hover_bptt.flexible_environment import FlexibleRopeEnvironment
from .hover_bptt.flexible_training import (
    FlexibleRopeStream, REPO, accumulate_domain_gradients, advance_warmup, source_manifest,
)
from .hover_bptt.functional import position_penalty
from .hover_bptt.generalization import cases_at
from .hover_bptt.io import digest, load_checkpoint, read, save, write
from .hover_bptt.onboard_graph import install_onboard
from .hover_bptt.optimizer import parameter_names
from .hover_bptt.plant import install_training_plant
from .hover_bptt.training import trajectory
from .hover_bptt.vehicle_spec import VehicleSpec
from .parallel_cohorts import CohortExecutor


STATIONARITY_FIELDS = {
    'horizontal_std_limit_m': 'late_horizontal_std_m',
    'horizontal_speed_limit_mps': 'late_horizontal_speed_rms_mps',
    'drift_limit_mps': 'late_horizontal_drift_mps',
}


class HoverStream(FlexibleRopeStream):
    def __init__(self, env, options, credit_steps):
        super().__init__(env)
        self.options = options
        self.credit_steps = credit_steps
        self.credit_drift_weight = options.get('credit_drift_weight', 0.)
        self.weak_credit_rmse_weight = options.get('weak_credit_rmse_weight', 0.)
        speed_filter_tau = options.get('weak_speed_filter_tau_s')
        self.weak_speed_filter_alpha = None
        if speed_filter_tau is not None:
            self.weak_speed_filter_alpha = -math.expm1(-env.dt / speed_filter_tau)
            env.training_filtered_velocity = env.state[:, 3:5].clone()
        if self.weak_credit_rmse_weight or speed_filter_tau is not None:
            self.weak_wind = torch.tensor([case['wind_speed'] == 1 for case in env.specifications],
                                          device=env.state.device)
        if self.credit_drift_weight:
            if credit_steps < 2:
                raise ValueError('Credit drift fitting requires at least two samples')
            # Environment tensors are restored by the adjoint at each recompute boundary.
            env.training_credit_slope = env.state.new_zeros((env.n, 2), dtype=torch.float64)
        if self.weak_credit_rmse_weight:
            env.training_credit_mse = env.state.new_zeros(env.n, dtype=torch.float64)
        weak_weight = options['weak_position_weight']
        weight_by_arm = {.05: options.get('weak_position_weight_5cm', weak_weight)}
        self.weak_position_weight = env.state.new_tensor([
            weight_by_arm.get(case['attachment_b_m'][2], weak_weight) if case['wind_speed'] == 1 else 0.
            for case in env.specifications])

    def step(self, actor):
        cost = super().step(actor)
        env = self.env
        error = env.state[:, :3] - env.model.target
        normalized = error[:, :2].square().sum(-1) / env.cfg['reward']['xy_scale_m']**2
        excess = self.weak_position_weight * position_penalty(normalized, 'pseudo_huber')
        ramp = min(1., max(0., (self.steps * env.dt - self.options['late_start_s']) / self.options['late_ramp_s']))
        velocity = env.state[:, 3:5]
        if self.weak_speed_filter_alpha is not None:
            filtered = torch.lerp(env.training_filtered_velocity, velocity, self.weak_speed_filter_alpha)
            env.training_filtered_velocity = torch.where(self.alive[:, None], filtered, env.training_filtered_velocity)
            velocity = torch.where(self.weak_wind[:, None], env.training_filtered_velocity, velocity)
        speed = self.options['late_speed_weight'] * ramp * velocity.square().sum(-1)
        cost = cost + torch.where(self.alive, excess + speed, 0.).mean()
        if self.steps <= self.warmup_steps or not (self.credit_drift_weight or self.weak_credit_rmse_weight):
            return cost

        count = self.credit_steps
        index = (self.steps - self.warmup_steps - 1) % count
        last_sample = index == count - 1
        # The step average divides these terminal costs by the credit length.
        if self.credit_drift_weight:
            coefficient = 12. * (index - (count - 1) / 2) / (env.dt * count * (count**2 - 1))
            slope = env.training_credit_slope + coefficient * error[:, :2].double()
            env.training_credit_slope = torch.zeros_like(slope) if last_sample else slope
            if last_sample:
                normalized_drift = slope.norm(dim=-1) / self.options['credit_drift_scale_mps']
                penalty = (normalized_drift - 1.).clamp_min(0.).square()
                cost = cost + count * self.credit_drift_weight * torch.where(self.alive, penalty, 0.).mean()
        if self.weak_credit_rmse_weight:
            squared_error = env.training_credit_mse + error.double().square().sum(-1)
            env.training_credit_mse = torch.zeros_like(squared_error) if last_sample else squared_error
            if last_sample:
                margin_m = self.options['weak_credit_rmse_margin_m']
                excess_mse = (squared_error / (count * margin_m**2) - 1.).clamp_min(0.)
                penalty = torch.where(self.alive & self.weak_wind, excess_mse.square(), 0.)
                cost = cost + count * self.weak_credit_rmse_weight * penalty.mean()
        return cost


class ObjectiveDriftStream(HoverStream):
    def __init__(self, env, options, credit_steps, objective_steps):
        super().__init__(env, options, credit_steps)
        if objective_steps < 2:
            raise ValueError('Objective drift fitting requires at least two samples')
        self.objective_steps = objective_steps
        self.use_displacements = options.get('objective_drift_use_displacements', False)
        env.training_objective_slope = env.state.new_zeros((env.n, 2), dtype=torch.float64)

    @property
    def objective_statistic(self):
        return self.env.training_objective_slope

    def objective_penalty(self, slope):
        excess = (slope.norm(dim=-1) / self.options['objective_drift_scale_mps'] - 1.).clamp_min(0.)
        return self.options['objective_drift_weight'] * torch.where(self.alive, excess.square(), 0.).mean()

    def step(self, actor):
        index = self.steps - self.warmup_steps
        previous = self.env.state[:, :2] if self.use_displacements and 0 < index < self.objective_steps else None
        cost = super().step(actor)
        if 0 <= index < self.objective_steps:
            count = self.objective_steps
            position = self.env.state[:, :2].double()
            denominator = self.env.dt * count * (count**2 - 1)
            if self.use_displacements:
                if previous is None:
                    return cost
                # Summation by parts gives the same full slope. Each credit
                # receives motion increments with its entering state detached.
                coefficient = 6. * index * (count - index) / denominator
                position = position - previous.double()
            else:
                coefficient = 12. * (index - (count - 1) / 2) / denominator
            self.env.training_objective_slope = (
                self.objective_statistic + coefficient * position)
        return cost


def validate_protocol(protocol):
    if (protocol['objective_s'], protocol['credit_s'], protocol['recompute_s']) != (30, 10, 1):
        raise ValueError('Expected 30 s objective, 10 s credit, 1 s recomputation')
    reserved = {v for pair in read(REPO/'P2/acceptance.json')['final_seed_pairs'] for v in pair.values()}
    reserved.update(v for pair in read(REPO/'P2/p2_protocol.json')['reserved_generalization_final_seed_pairs'] for v in pair)
    seeds = [protocol[f'{phase}_seed'] + offset for phase in ('training', 'selection', 'validation') for offset in (0, 1)]
    if len(seeds) != len(set(seeds)) or reserved.intersection(seeds):
        raise ValueError('Training, selection, validation and final seed pairs must be separate')
    directions = [angle % 360 for phase in ('training', 'selection', 'validation')
                  for angle in protocol[f'{phase}_directions_deg']]
    if len(directions) != len(set(directions)):
        raise ValueError('Keep development selection and validation directions separate')
    if any(type(v) is not int or not 0 <= v <= 90 for v in protocol.get('causal_prefix_s', {}).values()):
        raise ValueError('Causal prefix must be whole seconds in [0, 90]')
    hover = protocol.get('hover_training', {})
    if 'weak_speed_filter_tau_s' in hover:
        tau = hover['weak_speed_filter_tau_s']
        if type(tau) not in (int, float) or not math.isfinite(tau) or tau <= 0:
            raise ValueError('Weak-wind speed filter requires a positive finite time constant')
    if 'weak_position_weight_5cm' in hover:
        weight = hover['weak_position_weight_5cm']
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Finite nonnegative weak-wind position weight required for the 5 cm attachment')
    drift_fields = {'credit_drift_weight', 'credit_drift_scale_mps'}
    if drift_fields.intersection(hover) and any(
            type(hover.get(key)) not in (int, float) or not math.isfinite(hover[key]) or hover[key] <= 0
            for key in drift_fields):
        raise ValueError('Credit drift weight and scale must both be positive and finite')
    objective_drift_fields = {'objective_drift_weight', 'objective_drift_scale_mps'}
    if 'objective_drift_use_displacements' in hover and (
            type(hover['objective_drift_use_displacements']) is not bool
            or not objective_drift_fields.issubset(hover)):
        raise ValueError('Displacement formulation requires a configured objective drift penalty')
    if objective_drift_fields.intersection(hover):
        if any(type(hover.get(key)) not in (int, float) or not math.isfinite(hover[key]) or hover[key] <= 0
               for key in objective_drift_fields):
            raise ValueError('Objective drift weight and scale must both be positive and finite')
        if drift_fields.intersection(hover):
            raise ValueError('Choose either objective drift or credit drift')
    rmse_fields = {'weak_credit_rmse_weight', 'weak_credit_rmse_margin_m'}
    if rmse_fields.intersection(hover) and any(
            type(hover.get(key)) not in (int, float) or not math.isfinite(hover[key]) or hover[key] <= 0
            for key in rmse_fields):
        raise ValueError('Weak-wind credit RMSE weight and margin must both be positive and finite')
    if type(protocol.get('retain_safe_proposals', False)) is not bool:
        raise ValueError('retain_safe_proposals must be boolean')
    if protocol.get('retain_safe_proposals') and 'selection_stationarity' not in protocol:
        raise ValueError('Retaining safe proposals requires a separate stationarity gate')
    options = protocol.get('domain_randomization', {})
    grid = protocol.get('training_mass_gain_grid', False)
    selection_grid = protocol.get('selection_mass_gain_grid', False)
    if type(grid) is not bool or type(selection_grid) is not bool or (grid and not options.get('enabled')):
        raise ValueError('The mass/gain grid requires enabled domain randomization')
    if selection_grid and not grid:
        raise ValueError('Crossed selection requires a crossed training protocol')
    if 'selection_stationarity' in protocol:
        limits = protocol['selection_stationarity']
        if (not selection_grid or protocol.get('causal_prefix_s', {}).get('selection') != 90
                or not isinstance(limits, dict) or set(limits) != set(STATIONARITY_FIELDS)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in limits.values())):
            raise ValueError('Stationarity selection requires positive limits and a crossed 90–120 s evaluation')
    if options.get('enabled'):
        count = options['sample_count']
        rounds = protocol['maximum_updates']
        if grid:
            if count != 5 or any(lo >= hi for lo, hi in (options['mass_scale'], options['motor_gain_scale'])):
                raise ValueError('Mass/gain grid requires nominal plus four distinct corners')
        elif not 2 <= count <= 4:
            raise ValueError('Use 2–4 random cohorts including nominal')
        if not 1 <= rounds <= 8 or not options['include_nominal']:
            raise ValueError('Include nominal and use 1–8 bounded attempts')
        seeds = [protocol['training_seed'] + 1000 * iteration + 10 * cohort + offset
                 for iteration in range(rounds) for cohort in range(count) for offset in (0, 1)]
        seeds += [protocol['selection_seed'] + 10 * cohort + offset
                  for cohort in range(count) for offset in (0, 1)]
        seeds += [options['seed'] + iteration for iteration in range(rounds)]
        seeds += [protocol['selection_parameter_seed'], protocol['validation_parameter_seed']]
        seeds += [protocol['validation_seed'] + 10 * cohort + offset
                  for cohort in range(count) for offset in (0, 1)]
        if len(seeds) != len(set(seeds)) or reserved.intersection(seeds):
            raise ValueError('Parameter and rollout seeds must stay distinct across phases and attempts')


def acceleration_stage(parent, protocol):
    stage = copy.deepcopy(parent['configuration'])
    stage['model']['action_dim'] = 3
    stage['controller'].update(action_mode=protocol['action_mode'], acceleration=protocol['controller'])
    stage['training'].update(method='physical_BPTT_launch_acceleration', flight_s=protocol['objective_s'],
                             credit_s=protocol['credit_s'], recompute_s=protocol['recompute_s'])
    stage['training'].pop('stationarity_BPTT', None)
    return stage


@torch.no_grad()
def initialize_actor(parent, stage, protocol, device):
    actor = Actor(stage['model'])
    last = len(actor.mlp) - 1
    new_heads = {'fast_head.weight', 'fast_head.bias', f'mlp.{last}.weight', f'mlp.{last}.bias'}
    state = actor.state_dict()
    for name in state.keys() - new_heads:
        state[name] = parent['actor'][name]
    actor.load_state_dict(state)
    # Linearize the previous proportional body moments at hover, then express
    # that initial desired tilt and collective as a three-axis acceleration.
    old = parent['configuration']['controller']
    gravity = stage['model']['gravity_mps2']
    mass = stage['model']['conditioning']['reference_vehicle']['mass_kg']
    hover_acceleration = gravity + old['trim_excess_n'] / mass
    limits = protocol['controller']['limits_mps2']
    attitude_gains = protocol['controller']['attitude_kp_per_s2']
    rate_scale = [old['rate_kp_per_s'][i] * old['rate_limits_radps'][i] / attitude_gains[i] for i in (0, 1)]
    scale = actor.fast_head.weight.new_tensor([-hover_acceleration * rate_scale[1] / limits[0],
        hover_acceleration * rate_scale[0] / limits[1], -hover_acceleration * old['thrust_delta_fraction'] / limits[2]])
    for head, prefix in ((actor.fast_head, 'fast_head'), (actor.mlp[-1], f'mlp.{last}')):
        head.weight.copy_(scale[:, None] * parent['actor'][f'{prefix}.weight'][[1, 0, 3]])
        head.bias.copy_(scale * parent['actor'][f'{prefix}.bias'][[1, 0, 3]])
    # The first two fast features are approximately half roll and half pitch.
    actor.fast_head.weight[0, 1] -= 2 * hover_acceleration / limits[0]
    actor.fast_head.weight[1, 0] += 2 * hover_acceleration / limits[1]
    actor.fast_head.bias[2] += math.atanh(-old['trim_excess_n'] / (mass * limits[2]))
    return actor.to(device)


def build_stream(protocol, stage, plant, phase, seconds, device, *, domain=None, phase_seed=None):
    seed = protocol[f'{phase}_seed'] if phase_seed is None else phase_seed
    winds = protocol.get(f'{phase}_winds', [(1, 'steady'), (5, 'steady')])
    cases = cases_at(protocol[f'{phase}_directions_deg'], winds)
    prefix = protocol.get('causal_prefix_s', {}).get(phase, 0) if seconds == protocol['objective_s'] else 0
    specs = {} if domain is None else dict(vehicle_spec=domain['actual'], nominal_vehicle_spec=domain['nominal'])
    env = FlexibleRopeEnvironment(len(cases), seed, wind_seed=seed+1, device=device,
        stage=stage, plant_configuration=plant, episode_s=prefix+seconds, specifications=cases,
        rope=protocol['rope'], random_device='cpu', **specs)
    install_training_plant(env)
    install_onboard(env)
    hover = protocol.get('hover_training')
    credit_steps = round(min(seconds, protocol['credit_s']) / env.dt)
    if hover and hover.get('objective_drift_weight') and seconds == protocol['objective_s']:
        stream = ObjectiveDriftStream(env, hover, credit_steps, round(seconds / env.dt))
    else:
        stream = HoverStream(env, hover, credit_steps) if hover else FlexibleRopeStream(env)
    stream.warmup_steps = round(prefix/env.dt)
    stream.hidden = env.state.new_zeros(1, env.n, stage['model']['gru_hidden'])
    return stream


def acceleration_domains(protocol, plant, phase, iteration):
    options = dict(protocol['domain_randomization'])
    options['seed'] = (options['seed'] + iteration if phase == 'training' else protocol['selection_parameter_seed'])
    reference = VehicleSpec.from_config(plant)
    domains = sample_domains(reference, options)
    if protocol.get(f'{phase}_mass_gain_grid'):
        mass_levels = zip(('light', 'heavy'), options['mass_scale'])
        gain_levels = zip(('low_gain', 'high_gain'), options['motor_gain_scale'])
        for domain, ((mass_label, mass), (gain_label, gain)) in zip(
                domains[1:], product(mass_levels, gain_levels), strict=True):
            scales = dict(domain['scales'], mass=mass, motor_gain=gain)
            domain.update(name=f'{mass_label}_{gain_label}', scales=scales, actual=reference.scaled(**scales))
    return domains, options['seed']


def score_domains(actor, protocol, stage, plant, phase, seconds, device, gradient, iteration):
    domains, parameter_seed = acceleration_domains(protocol, plant, phase, iteration)
    seed = protocol[f'{phase}_seed'] + (1000 * iteration if phase == 'training' else 0)
    streams, contexts = [], []
    started = time.perf_counter()
    for index, domain in enumerate(domains):
        phase_seed = seed + 10 * index
        if device == 'cuda':
            context = torch.cuda.Stream()
            context.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(context):
                stream = build_stream(protocol, stage, plant, phase, seconds, device,
                                      domain=domain, phase_seed=phase_seed)
            context.synchronize()
            contexts.append(context)
        else:
            stream = build_stream(protocol, stage, plant, phase, seconds, device,
                                  domain=domain, phase_seed=phase_seed)
        streams.append(stream)
    dt = streams[0].env.dt
    steps = round(seconds/dt)
    executor = CohortExecutor(streams, actor, contexts, {'configuration': stage}, mask_rate_thrust=False) if contexts else None
    try:
        if gradient:
            records = accumulate_domain_gradients(streams, actor, steps=steps,
                credit_steps=round(min(seconds, protocol['credit_s'])/dt),
                recompute_steps=round(min(seconds, protocol['recompute_s'])/dt), executor=executor)
            scores = [record['score'] for record in records]
            diagnostics = [record['diagnostics'] for record in records]
        else:
            if executor is not None:
                scores = [result[2] for result in executor.forward(actor, steps=steps)]
            else:
                scores = []
                for stream in streams:
                    advance_warmup(stream, actor)
                    scores.append(trajectory(stream, actor, steps=steps)[2])
            diagnostics = []
    finally:
        if executor is not None:
            executor.close()
    result = dict(cost=math.fsum(row['cost'] for row in scores)/len(scores), duration_s=seconds,
        domains=describe_domains(domains), domain_scores=scores, gradient=diagnostics,
        parameter_seed=parameter_seed, wall_s=time.perf_counter()-started)
    for key in ('completed', 'position_rmse_m', 'height_rmse_m', 'horizontal_rmse_m', 'wind_speed'):
        result[key] = [value for row in scores for value in row[key]]
    print(dict(phase=phase, attempt=iteration+1, duration_s=seconds, wall_s=result['wall_s'],
               safe=sum(result['completed']), count=len(result['completed']), cost=result['cost']), flush=True)
    return result


def score(actor, protocol, stage, plant, phase, seconds, device, gradient=False, iteration=0):
    if protocol.get('domain_randomization', {}).get('enabled'):
        return score_domains(actor, protocol, stage, plant, phase, seconds, device, gradient, iteration)
    stream = build_stream(protocol, stage, plant, phase, seconds, device)
    step = stream.env.dt
    started = time.perf_counter()
    prefix_wall_s = advance_warmup(stream, actor)
    _, diagnostics, result, _ = trajectory(stream, actor, gradient=gradient,
        steps=round(seconds / step), credit_steps=round(min(seconds, protocol['credit_s']) / step),
        recompute_steps=round(min(seconds, protocol['recompute_s']) / step))
    result.update(wall_s=time.perf_counter()-started, gradient=diagnostics, prefix_wall_s=prefix_wall_s)
    print(dict(phase=phase, duration_s=seconds, wall_s=result['wall_s'],
               safe=sum(result['completed']), count=len(result['completed']), cost=result['cost']), flush=True)
    return result


def checkpoint(actor, optimizer, stage, plant, parent_sha, updates):
    return dict(schema_version=1, model_kind='launch_acceleration_BPTT', configuration=stage,
        plant_configuration=plant, actor={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()), optimizer_parameter_names=parameter_names(actor, optimizer),
        optimizer_steps=updates, cumulative_physical_updates=updates, source_checkpoint_sha256=parent_sha,
        initialization_provenance='Actor37 encoder and hidden layers; three-output head initialized by small-angle wrench conversion',
        p3_enabled=False, formal_final_acceptance=False)


def paired_improvement(before, after, *, allow_incumbent_failures=False):
    if not math.isfinite(after['cost']):
        return False
    if 'domain_scores' in before:
        def identities(result):
            return [(row['vehicle_identity'], row['wind_seed']) for row in result['domain_scores']]
        if identities(before) != identities(after):
            raise ValueError('Candidate must use the same ordered physical scenarios as the incumbent')
    if allow_incumbent_failures:
        paired = list(zip(before['completed'], after['completed'], before['height_rmse_m'],
                          after['height_rmse_m'], strict=True))
        if any(was_safe and (not is_safe or new_height > old_height + .03)
               for was_safe, is_safe, old_height, new_height in paired):
            return False
        return after['cost'] < before['cost']
    if not all(after['completed']):
        return False
    return (after['cost'] < before['cost'] and
            max(after['height_rmse_m']) <= max(before['height_rmse_m']) + .03)


def hover_selection(score):
    normalized = [rmse / (.5 if wind == 1 else 1.5)
                  for rmse, wind in zip(score['position_rmse_m'], score['wind_speed'])]
    gaps = [max(0., ratio - 1.) for ratio in normalized]
    return max(gaps), math.fsum(gaps), math.fsum(normalized)


def stationarity_selection(score, limits):
    gaps = []
    for group in score['domain_scores']:
        tail = group['late_convergence']
        values = [tail[field] for field in STATIONARITY_FIELDS.values()]
        for completed, count, *motion in zip(group['completed'], tail['samples'], *values, strict=True):
            if not completed or count != tail['expected_samples'] or count <= 0:
                raise ValueError('Stationarity selection requires a complete safe tail for every case')
            if any(not math.isfinite(value) or value < 0 for value in motion):
                raise ValueError('Invalid tail motion statistic')
            ratio = max(value / limits[limit] for value, limit in zip(motion, STATIONARITY_FIELDS))
            gaps.append(max(0., ratio - 1.))
    if not gaps:
        raise ValueError('Stationarity selection requires at least one case')
    return dict(count=len(gaps), exceedances=sum(gap > 0 for gap in gaps),
                mean_squared_excess=math.fsum(gap * gap for gap in gaps) / len(gaps))


def paired_stationarity(before, after, limits):
    try:
        old = stationarity_selection(before, limits)
        new = stationarity_selection(after, limits)
    except (KeyError, TypeError, ValueError):
        return dict(accepted=False, reason='invalid_or_incomplete_tail')
    accepted = (old['count'] == new['count'] and new['exceedances'] <= old['exceedances']
                and new['mean_squared_excess'] <= old['mean_squared_excess'])
    return dict(accepted=accepted, before=old, after=new)


def train(protocol, output, parent, parent_sha, qualification, device, resume=None):
    if resume is None:
        stage = acceleration_stage(parent, protocol)
        plant = parent['plant_configuration']
        actor = initialize_actor(parent, stage, protocol, device)
        prior_updates = 0
    else:
        stage = copy.deepcopy(resume['configuration'])
        plant = resume['plant_configuration']
        actor = Actor(stage['model']).to(device)
        actor.load_state_dict(resume['actor'])
        prior_updates = resume['optimizer_steps']
        if protocol.get('preflight_accel_calibration'):
            stage['sensors']['preflight_accel_calibration'] = {'gravity_mps2': stage['model']['gravity_mps2']}
        if 'horizontal_head_bias_delta' in protocol:
            bias = protocol['horizontal_head_bias_delta']
            if len(bias) != 2 or any(not math.isfinite(value) for value in bias):
                raise ValueError('Expected two finite horizontal output head initialization offsets')
            with torch.no_grad():
                actor.fast_head.bias[:2].add_(actor.fast_head.bias.new_tensor(bias))
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol['learning_rates'][0])
    if resume is not None:
        if parameter_names(actor, optimizer) != resume['optimizer_parameter_names']:
            raise ValueError('Resume optimizer parameter order differs')
        optimizer.load_state_dict(copy.deepcopy(resume['optimizer']))
    def materialize(updates):
        result = checkpoint(actor, optimizer, stage, plant, parent_sha, updates)
        if resume is not None:
            result['resume_checkpoint_sha256'] = protocol['resume_sha256']
            if 'horizontal_head_bias_delta' in protocol:
                result['horizontal_head_bias_delta'] = protocol['horizontal_head_bias_delta']
        return result

    grid_training = protocol.get('training_mass_gain_grid', False)
    stationarity_limits = protocol.get('selection_stationarity')
    if qualification is None:
        result = score(actor, protocol, stage, plant, 'training', 1, device, gradient=True)
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True))
        axes = actor.fast_head.weight.grad.norm(dim=1).detach().cpu().tolist()
        horizontal_only = stage['controller']['acceleration'].get('vertical_control') == 'px4_altitude_pid'
        active = axes[:2] if horizontal_only else axes
        if norm <= 0 or not all(value > 0 for value in active) or not all(result['completed']):
            raise RuntimeError('Acceleration qualification requires active gradients and safe short rollouts')
        if horizontal_only and axes[2] != 0:
            raise RuntimeError('Height-PID qualification received a vertical actor gradient')
        if protocol.get('preflight_accel_calibration'):
            save(output/'initialized.pt', materialize(prior_updates))
        selection = score(actor, protocol, stage, plant, 'selection', protocol['objective_s'], device)
        passed = any(selection['completed']) if grid_training else all(selection['completed'])
        if stationarity_limits is not None:
            passed = passed and paired_stationarity(selection, selection, stationarity_limits)['accepted']
        write(output/'qualification.json', dict(passed=passed, gradient_norm=norm, per_axis_gradient_norm=axes,
            score=result, initial_selection=selection, device=device, estimated_gradient_wall_s_30=result['wall_s']*30))
        if not passed:
            raise RuntimeError('Acceleration initialization did not meet the selection safety gate')
        return

    seconds = protocol['objective_s']
    initial = materialize(prior_updates)
    if resume is None:
        save(output/'initial.pt', initial)
    incumbent = initial
    best = qualification['initial_selection']
    report = dict(initial=best, candidates=[], selected_updates=0, final_acceptance=False)
    write(output/'training_summary.json', report)
    for update in range(1, protocol['maximum_updates']+1):
        fitted = score(actor, protocol, stage, plant, 'training', seconds, device, gradient=True, iteration=update-1)
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True))
        for trial_index, rate in enumerate(protocol['learning_rates'], start=1):
            actor.load_state_dict(incumbent['actor'])
            optimizer.load_state_dict(copy.deepcopy(incumbent['optimizer']))
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.step()
            validation = score(actor, protocol, stage, plant, 'selection', seconds, device)
            accepted = paired_improvement(best, validation, allow_incumbent_failures=grid_training)
            if grid_training:
                accepted = accepted and all(
                    new <= (.5 if wind == 1 else 1.5)
                    for old, new, wind in zip(best['position_rmse_m'], validation['position_rmse_m'],
                                              best['wind_speed'], strict=True)
                    if old <= (.5 if wind == 1 else 1.5))
            elif protocol.get('hover_training'):
                accepted = accepted and hover_selection(validation) < hover_selection(best)
            if protocol.get('domain_randomization', {}).get('enabled'):
                # A failed incumbent rollout still teaches through its live prefix.
                # Candidate safety is evaluated on the separate paired selection suite.
                accepted = accepted and norm > 0
            trial = dict(update=update, learning_rate=rate, gradient_norm=norm,
                         training=fitted, selection=validation)
            if stationarity_limits is not None:
                decision = paired_stationarity(best, validation, stationarity_limits)
                trial['stationarity_selection'] = decision
                if (protocol.get('retain_safe_proposals') and accepted and not decision['accepted']
                        and 'after' in decision and all(validation['completed'])
                        and all(math.isfinite(rmse) and 0. <= rmse <= (.5 if wind == 1 else 1.5)
                                for rmse, wind in zip(validation['position_rmse_m'],
                                                     validation['wind_speed'], strict=True))):
                    proposal = materialize(prior_updates + report['selected_updates'] + 1)
                    proposal.update(unvalidated_bptt_proposal=True, paired_selection_accepted=False,
                                    proposal_learning_rate=rate)
                    proposal_path = output/f'proposal{update:02}_trial{trial_index:02}.pt'
                    save(proposal_path, proposal)
                    trial['diagnostic_proposal'] = dict(file=proposal_path.name, sha256=digest(proposal_path))
                accepted = accepted and decision['accepted']
            report['candidates'].append(dict(trial, accepted=accepted))
            if accepted:
                selected_updates = report['selected_updates'] + 1
                incumbent = materialize(prior_updates + selected_updates)
                save(output/'latest.pt', incumbent)
                if resume is not None:
                    save(output/f'update{update:02}.pt', incumbent)
                best = validation
                report.update(selected_updates=selected_updates, checkpoint_sha256=digest(output/'latest.pt'), selected=best)
            write(output/'training_summary.json', report)
            if accepted:
                break
        if not accepted:
            actor.load_state_dict(incumbent['actor'])
            optimizer.load_state_dict(copy.deepcopy(incumbent['optimizer']))
            if not protocol.get('domain_randomization', {}).get('enabled'):
                break
    if not report['selected_updates']:
        raise RuntimeError('No trained candidate passed the paired development check')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['qualify', 'train'])
    parser.add_argument('--protocol', type=Path, default=REPO/'P2/acceleration_protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(21686001)
    protocol = read(args.protocol)
    validate_protocol(protocol)
    parent_path = REPO/protocol['parent_checkpoint']
    parent, parent_sha = load_checkpoint(parent_path)
    if parent_sha != protocol['parent_sha256']:
        raise ValueError('Protected Actor37 checkpoint changed')
    resume = None
    if 'resume_checkpoint' in protocol:
        resume_path = REPO/protocol['resume_checkpoint']
        resume, resume_sha = load_checkpoint(resume_path)
        if (resume_sha != protocol['resume_sha256'] or resume['model_kind'] != 'launch_acceleration_BPTT'
                or resume['source_checkpoint_sha256'] != parent_sha or not resume['optimizer']['state']):
            raise ValueError('Expected the declared acceleration Actor and Adam checkpoint')
    source = source_manifest(args.protocol) | {name: digest(REPO/name) for name in
        ['P2/train_acceleration.py', 'mujoco_sim/flexible.py', 'mujoco_sim/physics.py']}
    output = args.output.resolve()
    if not output.is_relative_to(REPO/'P2/results'):
        raise ValueError('Acceleration outputs must stay in P2/results')
    if args.mode == 'train':
        if args.qualification is None:
            parser.error('Training requires --qualification pointing at a passed preflight directory')
        qualification = read(args.qualification/'qualification.json')
        manifest = read(args.qualification/'manifest.json')
        if not qualification['passed'] or qualification['device'] != args.device or manifest['source_sha256'] != source:
            raise ValueError('Acceleration preflight must match the current protocol, source and device')
    else:
        qualification = None
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output/'frozen_source.zip', 'x', ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(REPO/name, name)
    write(output/'manifest.json', dict(protocol=protocol, source_sha256=source,
        parent_sha256=parent_sha, mode=args.mode, formal_final_acceptance=False))
    train(protocol, output, parent, parent_sha, qualification, args.device, resume)
    if digest(parent_path) != parent_sha or any(digest(REPO/name) != sha for name, sha in source.items()):
        raise RuntimeError('Frozen inputs changed during the run')


if __name__ == '__main__':
    main()
