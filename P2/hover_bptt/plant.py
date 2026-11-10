import torch
from torch.utils.checkpoint import checkpoint
from .cuda_bptt_kernel import RecurrentPlantGraph
def checkpoint_plant(model, *, reentrant=False):
    """Recompute only the pure RK4 function, never mutable sensors or RNG."""
    if getattr(model, '_physical_checkpoint_enabled', False):
        raise ValueError('Plant checkpoint already installed')
    plain = model.control

    def control(state, commands, winds):
        if not torch.is_grad_enabled():
            return plain(state, commands, winds)
        return checkpoint(plain, state, commands, winds, use_reentrant=reentrant,
                          preserve_rng_state=False)
    model.control = control
    model._physical_checkpoint_enabled = True
    return plain

def install_training_plant(env):
    if str(env.device).startswith('cuda'):
        commands=torch.cat((env.controller.previous,torch.zeros_like(env.controller.previous[:,:2])),-1)
        winds=env.wind()[:,None,:].expand(-1,env.model.substeps,-1).contiguous()
        env.model.control=RecurrentPlantGraph(env.model,env.state,commands,winds)
    else:checkpoint_plant(env.model,reentrant=False)
