"""Reuse the verified CUDA function graphs for the plain P2 onboard loop."""
import torch

from track.frozen_graph import FrozenFunctionGraph, install_history, install_loads
from .onboard import Sensors


def install_onboard(env):
    if not str(env.device).startswith('cuda'):
        return
    controller = env.controller
    names = [name for name, value in vars(controller).items() if isinstance(value, torch.Tensor)]
    native = controller.apply

    def arguments(packet, raw):
        return tuple(getattr(controller, name) for name in names) + (
            packet.attitude, packet.gyro, packet.specific_force, packet.motor_speed, packet.baro_height, raw)

    def function(*inputs):
        outer = {name: getattr(controller, name) for name in names}
        try:
            for name, value in zip(names, inputs):
                setattr(controller, name, value)
            packet = Sensors(*inputs[len(names):len(names)+5])
            motors, _ = native(packet, inputs[-1], env.dt)
            return tuple(getattr(controller, name).clone() for name in names) + (motors.clone(),)
        finally:
            for name, value in outer.items():
                setattr(controller, name, value)

    kernel = FrozenFunctionGraph(function, arguments(env.packet, env.state.new_zeros(env.n, controller.action_dim)))

    def apply(packet, raw, dt):
        if dt != env.dt:
            raise ValueError('Fixed control period required')
        result = kernel(*arguments(packet, raw))
        for name, value in zip(names, result[:-1]):
            setattr(controller, name, value)
        return result[-1], {}

    controller.apply = apply
    install_history(env)
    install_loads(env)
