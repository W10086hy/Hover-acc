"""One-second open-loop step refinement; not a full-flight convergence claim."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from P2.hover_bptt.environment import plant_config
from P2.hover_bptt.io import read,write,digest
from .flexible import MujocoFlexiblePlant


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--reference',default='P2/results/flexible_continue30/replay120')
    a=p.parse_args();torch.set_num_threads(2);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(a.reference);m=read(root/'manifest.json');indices=[26,30,31,58,62,63]
    with np.load(root/'evaluation/trajectory.npz') as data:
        index=int(np.argmin(np.abs(data['times']-90)));x=torch.from_numpy(data['state'][index,indices].copy()).double()
    wind=torch.load(root/'evaluation/wind_trace.pt',map_location='cpu',weights_only=False)['wind_w_mps'][9000:9100,indices].double()
    cfg=plant_config();poly=cfg['actuators']['motor_speed_polynomial'];rpm=x[:,13:17].numpy()
    throttle=(-poly[1]+np.sqrt(poly[1]**2-4*poly[0]*(poly[2]-rpm)))/(2*poly[0])
    u=torch.from_numpy(np.concatenate((np.clip(throttle,0,1),np.zeros((len(indices),2))),-1))
    arms=torch.tensor([m['cases'][i]['attachment_b_m'] for i in indices],dtype=torch.double)
    lengths=torch.tensor([m['cases'][i]['tether_rest_length_m'] for i in indices],dtype=torch.double)
    traces=[];started=time.perf_counter()
    for dt in [.0005,.00025,.000125]:
        model=MujocoFlexiblePlant(cfg,arms,lengths,m['rope'],physics_dt_s=dt)
        state=x.clone();states=[state.numpy().copy()]
        for step in range(100):
            state=model.control(state,u,wind[step,:,None].expand(-1,model.substeps,-1).contiguous())
            states.append(state.numpy().copy())
        trace=np.asarray(states);assert np.isfinite(trace).all();traces.append(trace)
        np.savez_compressed(out/f'trace_{dt:g}.npz',times=np.arange(101)/100,state=trace)
    differences=[]
    for coarse,fine in zip(traces[:-1],traces[1:]):
        differences.append(dict(maximum_body_position_difference_m=float(np.linalg.norm(coarse[:,:,:3]-fine[:,:,:3],axis=-1).max()),
            maximum_node_position_difference_m=float(np.linalg.norm((coarse[:,:,19:52]-fine[:,:,19:52]).reshape(101,6,11,3),axis=-1).max())))
    result=dict(mode='one-second fixed-RPM open-loop refinement from archived90s states;not120s closed-loop convergence',
        case_indices=indices,physics_steps_s=[.0005,.00025,.000125],differences=differences,
        all_finite=True,wall_s=time.perf_counter()-started,source_sha256=digest(__file__),
        reference_trajectory_sha256=digest(root/'evaluation/trajectory.npz'),model_source_sha256=digest(Path(__file__).with_name('flexible.py')))
    write(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':main()
