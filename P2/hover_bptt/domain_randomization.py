"""Seeded, rollout-fixed structural variations with a calibrated nominal model."""
import math
import random


def sample_domains(reference, options):
    nominal = dict(name='nominal', actual=reference, nominal=reference,
        scales=dict(mass=1., inertia=[1., 1., 1.], motor_tau=1., motor_gain=1.))
    if not options or options.get('enabled') is False:
        return [nominal]
    keys = {'enabled', 'sample_count', 'seed', 'include_nominal', 'mass_scale',
        'inertia_scale', 'motor_tau_scale', 'motor_gain_scale'}
    if set(options) != keys or options['enabled'] is not True:
        raise ValueError('Explicit domain-randomization fields required')
    count, seed = options['sample_count'], options['seed']
    if type(count) is not int or not 1 <= count <= 32 or type(seed) is not int or seed < 0:
        raise ValueError('Use 1..32 domains and a nonnegative integer parameter seed')
    if type(options['include_nominal']) is not bool:
        raise ValueError('include_nominal must be boolean')
    for name in ('mass_scale', 'inertia_scale', 'motor_tau_scale', 'motor_gain_scale'):
        bounds = options[name]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError(f'Ordered finite positive bounds required for {name}')
    rng = random.Random(seed)
    domains = [nominal] if options['include_nominal'] else []
    for index in range(count - len(domains)):
        for _ in range(256):
            inertia = [rng.uniform(*options['inertia_scale']) for _ in range(3)]
            actual_inertia = [value * scale for value, scale in zip(reference.inertia_body_kgm2, inertia)]
            if 2 * max(actual_inertia) <= sum(actual_inertia) + 1e-12:
                break
        else:
            raise ValueError('Cannot sample physical principal inertias within the requested bounds')
        scales = dict(mass=rng.uniform(*options['mass_scale']), inertia=inertia,
            motor_tau=rng.uniform(*options['motor_tau_scale']), motor_gain=rng.uniform(*options['motor_gain_scale']))
        domains.append(dict(name=f'random_{index+1}', actual=reference.scaled(**scales), nominal=reference, scales=scales))
    return domains


def describe_domains(domains):
    return [dict(name=domain['name'], scales=domain['scales'],
        actual=domain['actual'].to_dict(), nominal=domain['nominal'].to_dict()) for domain in domains]
