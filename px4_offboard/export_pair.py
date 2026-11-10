"""Export a frozen 5 m Actor and its paired flexible-rope P3 residual for SITL."""
import argparse
from pathlib import Path

import numpy as np
import torch

from P2.hover_bptt.actor import Actor as TorchActor
from P2.hover_bptt.io import digest, load_checkpoint, write
from P3.flexible_domains import ROOT as P3_ROOT, load_protocol
from P3.mismatch_dynamics import DynamicsResidual
from track.run import build
from .network import Actor, Residual


ROOT = P3_ROOT.parent


@torch.inference_mode()
def check_inference(saved, learned, weights, context):
    model = saved['configuration']['model']
    actor = TorchActor(model).eval()
    actor.load_state_dict(saved['actor'])
    residual = DynamicsResidual(learned['model_configuration']).eval()
    residual.load_state_dict(learned['residual'])
    actor_np, residual_np = Actor(weights, model), Residual(weights)
    actor_hidden = residual_hidden = None
    rng = np.random.default_rng(282)
    actor_error = residual_error = 0.
    for _ in range(100):
        observation = rng.normal(0., .05, 35).astype(np.float32)
        observation[0] = 1.
        command = rng.uniform(.4, .7, 4).astype(np.float32)
        obs = torch.from_numpy(observation[None])
        ctx = torch.from_numpy(context[None])
        expected, actor_hidden = actor(obs, actor_hidden, ctx)
        actual = actor_np(observation, context)
        np.testing.assert_allclose(actual, expected[0].numpy(), atol=5e-5, rtol=3e-5)
        actor_error = max(actor_error, float(np.max(np.abs(actual - expected[0].numpy()))))
        features, residual_hidden = residual.encode(obs, residual_hidden)
        expected = residual(obs, features, ctx, torch.from_numpy(command[None]))
        actual = residual_np(observation, context, command)
        np.testing.assert_allclose(actual, expected[0].numpy(), atol=4e-6, rtol=3e-5)
        residual_error = max(residual_error, float(np.max(np.abs(actual - expected[0].numpy()))))
    return dict(samples=100, actor_max_absolute_error=actor_error, residual_max_absolute_error=residual_error)


def export(protocol_path, residual_path, output):
    protocol_path, residual_path, output = (Path(p).resolve() for p in (protocol_path, residual_path, output))
    if not all(p.is_relative_to(ROOT) for p in (protocol_path, residual_path, output)):
        raise ValueError('Keep inputs and output in this project')
    if output.exists():
        raise FileExistsError(output)
    protocol, saved, actor_sha = load_protocol(protocol_path)
    learned, residual_sha = load_checkpoint(residual_path)
    if learned['controller_checkpoint_sha256'] != actor_sha or learned['protocol'] != protocol:
        raise ValueError('Residual was fitted against a different controller or protocol')
    if learned['nominal_geometry_scope_cm'] != 5 or protocol['collection_height_m'] != 5.:
        raise ValueError('This export uses the calibrated 5 m, 5 cm configuration')
    actor_path = (P3_ROOT / protocol['controller_checkpoint']).resolve()
    motion = dict(kind='half_cosine_ramp', hold_s=5., ramp_s=10.)
    course = dict(duration_s=120., cart_speed_mps=0., cart_direction_engineering_deg=0.,
        cart_anchor_height_m=0., target_aircraft_height_m=5., physics_dt_s=.0005,
        attachment_cm=[5], initial_conditions=['hover_capture'], cart_motion=motion,
        case_specs=[dict(attachment_cm=5, cart_speed_mps=0., cart_motion=motion)],
        cart_inertial_regularization_kg=1e6, state_seed=20683121, wind_seed=20683131,
        wind_peak_mps=1., wind_mean_direction_deg=90., wind_profile='varying',
        wind_normalization_horizon_s=120., controller_checkpoint=actor_path.relative_to(ROOT).as_posix(),
        controller_sha256=actor_sha, residual_checkpoint=residual_path.relative_to(ROOT).as_posix(),
        residual_sha256=residual_sha, residual_protocol=protocol_path.relative_to(ROOT).as_posix())
    env = build(course, saved)
    plant = env.model
    weights = {'actor.' + name: tensor.numpy() for name, tensor in saved['actor'].items()}
    weights.update({'residual.' + name: tensor.numpy() for name, tensor in learned['residual'].items()})
    context = env.parameter_context[0].numpy().copy()
    parity = check_inference(saved, learned, weights, context)
    if digest(actor_path) != actor_sha or digest(residual_path) != residual_sha:
        raise RuntimeError('Checkpoint changed during export')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'weights.npz', **weights)
    np.savez_compressed(output / 'initial.npz', state=env.state.numpy(), context=env.parameter_context.numpy(),
        node_mass=plant.node_mass, stiffness=plant.k, damping=plant.c, rest=plant.rest,
        arms=plant._arms, lengths=plant._lengths)
    (output / 'model.xml').write_text(plant.xml, encoding='utf-8')
    write(output / 'config.json', dict(schema_version=1, route='thrust_body_rate_BPTT_PX4_Offboard',
        protocol=course, stage=env.cfg, nominal=env.nominal_pc, actual=env.pc, rope=plant.rope_options,
        compensation=protocol['compensation'], controller_checkpoint=course['controller_checkpoint'],
        controller_sha256=actor_sha, residual_checkpoint=course['residual_checkpoint'], residual_sha256=residual_sha,
        external_nominal_context=context.tolist(), actual_drag_scale=protocol['control_actual_drag_scale'],
        cumulative_physical_updates=saved['cumulative_physical_updates'],
        validation_height_override=dict(height_new_m=5., rest_length_m=float(plant._lengths[0])),
        motor_order='P2 front-right,rear-right,front-left,rear-left;FRD', inference_parity=parity))
    print(dict(controller_sha256=actor_sha, residual_sha256=residual_sha, inference_parity=parity), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--residual', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    export(args.protocol, args.residual, args.output)
