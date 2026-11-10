import copy
import torch
from .onboard import Sensors
def runtime_objects(env):
    objects={'env':env,'controller':env.controller,'sensors':env.sensors,
        'history':env.history,'height':env.history.height}
    if env.history.motion:objects['motion']=env.history.motion
    return objects

def capture(env,hidden):
    return {'buffers':{key:{name:v.detach().clone() for name,v in vars(obj).items()
            if isinstance(v,torch.Tensor)} for key,obj in runtime_objects(env).items()},
        'obs':env.obs.detach().clone(),'hidden':None if hidden is None else hidden.detach().clone()}

def restore(env,record):
    for key,obj in runtime_objects(env).items():
        for name,value in record['buffers'][key].items():setattr(obj,name,value.clone())
    env.obs=record['obs'].clone()
    s=env.sensors;env.packet=Sensors(s.q,s.rate,s.force,s.rpm,s.height)
    return None if record['hidden'] is None else record['hidden'].clone()
def hold_streams(env,previous,mask,hidden):
    for key,obj in runtime_objects(env).items():
        for name,old in previous['buffers'][key].items():
            if old.ndim and old.shape[0]==env.n:
                current=getattr(obj,name)
                selected=mask.reshape((env.n,)+(1,)*(old.ndim-1))
                setattr(obj,name,torch.where(selected,old,current))
    for key in env.obs.keys():
        old=previous['obs'][key]
        env.obs[key]=torch.where(mask[:,None],old,env.obs[key])
    if hidden is not None:
        old=previous['hidden']
        if old is None:old=torch.zeros_like(hidden)
        hidden=torch.where(mask[None,:,None],old,hidden)
    s=env.sensors;env.packet=Sensors(s.q,s.rate,s.force,s.rpm,s.height)
    return hidden
COUNTERS=('alive','counts','failure_codes','error_sums','cost_sum')
REQUIRED={'actor','optimizer','updates','episode','window','stream','stream_steps',
          'env_rng','sensor_rng','torch_rng','cuda_rng','vehicle_identity',
          'trajectory_schema_version','trajectory_identity',*COUNTERS}
def snapshot(env,stream,actor,optimizer,*,updates,episode,window):
    result=dict(actor={k:v.detach().clone() for k,v in actor.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()),updates=updates,episode=episode,window=window,
        stream=capture(env,stream.hidden),stream_steps=stream.steps,
        env_rng=env.gen.get_state(),sensor_rng=env.sensor_gen.get_state(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(env.device) if str(env.device).startswith('cuda') else None,
        vehicle_identity=env.vehicle_identity,trajectory_schema_version=env.trajectory_schema_version,
        trajectory_identity=env.trajectory_identity)
    result.update({key:getattr(stream,key).detach().clone() for key in COUNTERS})
    return result

def restore_stream(env,stream,actor,optimizer,record):
    if REQUIRED-record.keys():raise ValueError('Incomplete saved trajectory: '+str(REQUIRED-record.keys()))
    if(record['trajectory_schema_version']!=env.trajectory_schema_version
            or record['trajectory_identity']!=env.trajectory_identity
            or record['vehicle_identity']!=env.vehicle_identity):
        raise ValueError('Trajectory identity differs: vehicle, wind, runtime configuration or horizon')
    if (record['cuda_rng'] is not None)!=(torch.device(env.device).type=='cuda'):
        raise ValueError('Exact continuation requires the original CPU/CUDA generator backend')
    if record['stream_steps']<0 or record['alive'].shape!=(env.n,):raise ValueError('Invalid stream counters')
    buffers=record['stream']['buffers']
    for key,obj in runtime_objects(env).items():
        if key not in buffers:raise ValueError('Missing physical/sensor state: '+key)
        expected={name for name,value in vars(obj).items() if isinstance(value,torch.Tensor)}
        if expected!=set(buffers[key]):raise ValueError('Physical/sensor state layout differs: '+key)
    if str(env.device).startswith('cuda') and record['cuda_rng'] is None:raise ValueError('Missing CUDA RNG')
    # Adam.load_state_dict can retain tensor aliases. A later optimizer step
    # must not mutate the saved rollback state used by another cohort/trial.
    actor.load_state_dict(record['actor']);optimizer.load_state_dict(copy.deepcopy(record['optimizer']))
    stream.hidden=restore(env,record['stream']);stream.steps=int(record['stream_steps'])
    for key in COUNTERS:setattr(stream,key,record[key].clone())
    env.gen.set_state(record['env_rng'].cpu());env.sensor_gen.set_state(record['sensor_rng'].cpu())
    torch.set_rng_state(record['torch_rng'].cpu())
    if record['cuda_rng'] is not None and str(env.device).startswith('cuda'):
        torch.cuda.set_rng_state(record['cuda_rng'].cpu(),env.device)
