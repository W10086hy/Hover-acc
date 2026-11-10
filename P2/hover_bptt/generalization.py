"""Small, explicit P2 domain protocol and paired development checks."""
import copy
import math
from .cases import evaluation_cases
from .vehicle_spec import VehicleSpec, PARAMETER_FIELDS, identity


def cases_at(directions,wind_cases=None):
    templates = evaluation_cases()
    output = []
    for attachment in (2, 1):
        for direction in directions:
            for speed, profile in (((1., 'steady'), (5., 'combined')) if wind_cases is None else wind_cases):
                case = copy.deepcopy(next(c for c in templates if c['attachment'] == attachment
                                          and c['wind_speed'] == speed and c['wind_profile'] == profile))
                case.update(wind_speed=float(speed),direction_rad=math.radians(direction), group=profile)
                output.append(case)
    return output


def cohort_specs(base, cohort):
    reference = VehicleSpec.from_config(base)
    return reference.scaled(**cohort['actual_scales']), reference.scaled(**cohort['nominal_scales'])


def cohort_cases(protocol, cohort, training=False):
    if training:
        return cases_at(protocol['training_directions_deg'])
    if cohort['case_set'] == 'original':
        return evaluation_cases()
    if cohort['case_set'] != 'new_directions':
        raise ValueError('Unknown development case set')
    return cases_at(protocol['development_directions_deg'])


def configure(saved, base, conditioned):
    cfg = copy.deepcopy(saved['configuration'])
    cfg['training']['report_credit_progress'] = True
    cfg['model']['conditioning'] = dict(enabled=conditioned, width=32,
        parameter_fields=list(PARAMETER_FIELDS), reference_vehicle=VehicleSpec.from_config(base).to_dict(),
        schema_version=1, context_source='calibrated airframe and actuator configuration only')
    return cfg


def validate_protocol(protocol, acceptance):
    if (protocol['objective_s'], protocol['credit_s'], protocol['recompute_s']) != (120, 10, 1):
        raise ValueError('The declared120/10/1 BPTT schedule must remain unchanged')
    reserved = {v for pair in acceptance['final_seed_pairs'] for v in pair.values()}
    reserved |= {v for pair in protocol['reserved_generalization_final_seed_pairs'] for v in pair}
    used = {protocol['training_seed'] + 100 * i + j
            for i in range(len(protocol['training_cohorts'])) for j in (0, 1)}
    used |= {v for pair in protocol['development_seed_pairs'] for v in pair}
    if used & reserved:
        raise ValueError('Reserved final seeds cannot be used for pilot training/selection')
    if set(protocol['training_directions_deg']) & set(protocol['development_directions_deg']):
        raise ValueError('Held-out direction combinations overlap pilot training')
    for pair in protocol['development_seed_pairs']:
        if len(pair) != 2 or pair[1] != pair[0] + 1:
            raise ValueError('Explicit state/wind seed pair with wind=state+1 required')


def paired_decision(before, after, guard):
    reasons = []
    metrics = ('position_rmse_m', 'height_rmse_m', 'horizontal_rmse_m', 'maximum_true_yaw_error_deg')
    try:
        n = len(before['completed'])
        if not n or not before.get('vehicle_identity') or before['vehicle_identity'] != after['vehicle_identity']:
            raise ValueError('Different physical/nominal course identities')
        for key in ('attachment', 'wind_speed', 'direction_rad', 'wind_profile'):
            if len(before[key]) != n or before[key] != after[key]:
                raise ValueError('Different ordered courses')
        for score in (before, after):
            if len(score['completed']) != n or not math.isfinite(score['cost']):
                raise ValueError('Incomplete or nonfinite course')
            if any(len(score[k]) != n or any(not math.isfinite(v) or v < 0 for v in score[k]) for k in metrics):
                raise ValueError('Invalid per-case metric')
        if not all(after['completed']):
            reasons.append('unsafe')
        if after['cost'] >= before['cost'] - max(1e-7, abs(before['cost']) * 1e-7):
            reasons.append('fullcost_not_lower')
        rms = lambda s: math.sqrt(math.fsum(v * v for v in s['position_rmse_m']) / n)
        if rms(after) > rms(before) * (1 + guard['aggregate_3d_relative_tolerance']) + guard['aggregate_3d_absolute_tolerance_m']:
            reasons.append('aggregate_position_regression')
        if any(a > b * (1 + guard['individual_height_relative_tolerance']) + guard['individual_height_absolute_tolerance_m']
               for a, b in zip(after['height_rmse_m'], before['height_rmse_m'])):
            reasons.append('individual_height_guard')
        if max(after['maximum_true_yaw_error_deg']) > guard['heading_max_deg']:
            reasons.append('heading_guard')
        return dict(accepted=not reasons, reasons=reasons, trajectory_count=n,
                    policy='P2 paired development, not final acceptance',
                    existing_rmse_exceedances_allowed=True)
    except (KeyError, ValueError, TypeError):
        return dict(accepted=False, reasons=['invalid_course_or_nonfinite_score'])
