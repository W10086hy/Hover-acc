"""A small calibrated-parameter adapter; zero initialization preserves behavior."""
import torch
from torch import nn
from .vehicle_spec import PARAMETER_FIELDS


class ParameterEncoder(nn.Module):
    def __init__(self, hidden, width=32):
        super().__init__()
        self.input = nn.Linear(len(PARAMETER_FIELDS), width, bias=False)
        self.output = nn.Linear(width, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, context):
        if context.ndim != 2 or context.shape[-1] != len(PARAMETER_FIELDS):
            raise ValueError('Expected calibrated vehicle context [N,14]')
        return self.output(torch.tanh(self.input(context)))


def load_actor(actor, saved):
    result = actor.load_state_dict(saved['actor'], strict=False)
    allowed = {'conditioning.input.weight', 'conditioning.output.weight'}
    if result.unexpected_keys or (set(result.missing_keys) - allowed):
        raise ValueError('Incompatible checkpoint: ' + str(result))
    if result.missing_keys and (set(result.missing_keys) != allowed or actor.conditioning is None):
        raise ValueError('Only a complete zero-output adapter may be added to a legacy checkpoint')
    return result
