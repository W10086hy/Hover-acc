"""Calibrated airframe/actuator parameters, without wind or tether information.

One immutable specification per environment/rollout keeps the established
scalar-parameter plant and fixed CUDA graphs valid. Actual and nominal specs
are passed separately; neither contains an anchor, attachment or drag sample.
"""
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math


PARAMETER_FIELDS = (
    'mass_kg', 'inertia_xx', 'inertia_yy', 'inertia_zz',
    'rotor_dx_m', 'rotor_dy_m', 'kf', 'km', 'motor_tau_s',
    'motor_poly_a', 'motor_poly_b', 'motor_poly_c',
    'motor_deadzone', 'rotor_inertia_kgm2',
)


@dataclass(frozen=True)
class VehicleSpec:
    mass_kg: float
    inertia_body_kgm2: tuple
    rotor_dx_m: float
    rotor_dy_m: float
    kf: float
    km: float
    motor_tau_s: float
    motor_speed_polynomial: tuple
    motor_deadzone: float
    rotor_inertia_kgm2: float
    include_gyro: bool

    def __post_init__(self):
        object.__setattr__(self, 'inertia_body_kgm2', tuple(self.inertia_body_kgm2))
        object.__setattr__(self, 'motor_speed_polynomial', tuple(self.motor_speed_polynomial))
        if len(self.inertia_body_kgm2) != 3 or len(self.motor_speed_polynomial) != 3:
            raise ValueError('Three principal inertias and motor polynomial coefficients required')
        positive = (self.mass_kg, *self.inertia_body_kgm2, self.rotor_dx_m,
                    self.rotor_dy_m, self.kf, self.km, self.motor_tau_s)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError('Finite positive airframe/actuator parameters required')
        if 2 * max(self.inertia_body_kgm2) > sum(self.inertia_body_kgm2) + 1e-12:
            raise ValueError('Principal inertias violate rigid-body triangle inequalities')
        if (not 0 <= self.motor_deadzone < 1 or not math.isfinite(self.rotor_inertia_kgm2)
                or self.rotor_inertia_kgm2 < 0 or not isinstance(self.include_gyro, bool)):
            raise ValueError('Invalid motor deadzone, rotor inertia or gyro flag')
        a, b, c = self.motor_speed_polynomial
        if (not all(math.isfinite(v) for v in (a, b, c)) or a + b + c <= 0
                or min(b, 2 * a + b) <= 0):
            raise ValueError('Positive, monotonic motor speed mapping required')

    @classmethod
    def from_config(cls, cfg):
        a, u = cfg['airframe'], cfg['actuators']
        return cls(a['mass_kg'], tuple(a['inertia_body_kgm2']), a['rotor_dx_m'],
                   a['rotor_dy_m'], u['kf'], u['km'], u['motor_tau_s'],
                   tuple(u['motor_speed_polynomial']), u['motor_deadzone'],
                   cfg['power']['rotor_inertia_kgm2'], bool(u.get('include_gyro', False)))

    @classmethod
    def from_dict(cls, value):
        # Dataclass constructor deliberately rejects unrecognized/privileged keys.
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    def apply(self, base):
        cfg = copy.deepcopy(base)
        a, u = cfg['airframe'], cfg['actuators']
        for key in ('mass_kg', 'rotor_dx_m', 'rotor_dy_m'):
            a[key] = getattr(self, key)
        a['inertia_body_kgm2'] = list(self.inertia_body_kgm2)
        for key in ('kf', 'km', 'motor_tau_s', 'motor_deadzone', 'include_gyro'):
            u[key] = getattr(self, key)
        u['motor_speed_polynomial'] = list(self.motor_speed_polynomial)
        cfg['power']['rotor_inertia_kgm2'] = self.rotor_inertia_kgm2
        thrust_ratio = 4 * self.kf * sum(self.motor_speed_polynomial)**2 / (self.mass_kg * a['gravity_mps2'])
        if thrust_ratio < 1.2:
            raise ValueError('Provisional simulation thrust-to-weight margin below 1.2')
        return cfg

    def values(self):
        return (self.mass_kg, *self.inertia_body_kgm2, self.rotor_dx_m,
                self.rotor_dy_m, self.kf, self.km, self.motor_tau_s,
                *self.motor_speed_polynomial, self.motor_deadzone, self.rotor_inertia_kgm2)

    def context(self, reference):
        if self.include_gyro != reference.include_gyro:
            raise ValueError('Changing the gyro model is outside the first P2 protocol')
        return [(v - r) / (abs(r) if abs(r) > 1e-12 else 1.)
                for v, r in zip(self.values(), reference.values())]

    def scaled(self, *, mass=1., inertia=1., motor_tau=1., motor_gain=1.):
        scales = tuple(inertia) if isinstance(inertia, (list, tuple)) else (inertia,) * 3
        if len(scales) != 3:
            raise ValueError('A scalar or three principal-inertia scales required')
        return replace(self, mass_kg=self.mass_kg * mass,
                       inertia_body_kgm2=tuple(v * scale for v, scale in zip(self.inertia_body_kgm2, scales)),
                       motor_tau_s=self.motor_tau_s * motor_tau,
                       motor_speed_polynomial=(self.motor_speed_polynomial if motor_gain == 1.
                                               else tuple(v * motor_gain for v in self.motor_speed_polynomial)))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()
