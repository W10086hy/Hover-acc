import copy
import torch

def initialize(actor,checkpoint,learn_baro=False):
    projected=checkpoint.get('bptt_candidate_materialization',{}).get('parameter_subset','all')
    if projected!='all':
        raise ValueError('A projected Adam state requires explicit conversion before ordinary full-Actor resume')
    by_name=dict(actor.named_parameters());groups=[]
    for names,old in zip(checkpoint['optimizer_parameter_names'],checkpoint['optimizer']['param_groups']):
        groups.append(dict(params=[by_name[name]for name in names],lr=old['lr']))
    optimizer=torch.optim.Adam(groups);optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    restored={name for names in checkpoint['optimizer_parameter_names']for name in names}
    additional=[p for name,p in actor.named_parameters()if name not in restored]
    if additional:optimizer.add_param_group(dict(params=additional,lr=1e-4))
    install_gradient_masks(actor,learn_baro)
    return optimizer

def install_gradient_masks(actor,learn_baro=False):
    """Install the same masks on root/worker Actors without creating Adam."""
    mask=torch.ones_like(actor.fast_head.weight);mask[:,:8]=0
    if learn_baro:mask[3,6:8]=1
    mask[2]=0
    for i,j in [(0,12),(0,13),(1,11),(1,13)]:mask[i,j]=0
    actor.fast_head.weight.register_hook(lambda gradient:gradient*mask)
    bias_mask=torch.ones_like(actor.fast_head.bias);bias_mask[2]=0
    actor.fast_head.bias.register_hook(lambda gradient:gradient*bias_mask)
    weight_mask=torch.ones_like(actor.mlp[-1].weight);weight_mask[2]=0
    output_mask=torch.ones_like(actor.mlp[-1].bias);output_mask[2]=0
    actor.mlp[-1].weight.register_hook(lambda gradient:gradient*weight_mask)
    actor.mlp[-1].bias.register_hook(lambda gradient:gradient*output_mask)

def parameter_names(actor,optimizer):
    names={id(p):name for name,p in actor.named_parameters()}
    return [[names[id(p)]for p in group['params']]for group in optimizer.param_groups]
