"""Short held-end cable check against an independent NumPy/Radau ODE.

This is a numerical material-model check, not a flight or training result.
The aircraft endpoint follows a prescribed smooth motion. No MuJoCo is used.
"""
import argparse,math,time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch
from .hover_bptt.environment import plant_config
from .hover_bptt.flexible_rope import FlexibleRopeModel
from .hover_bptt.io import read,write,digest

ROOT=Path(__file__).resolve().parent


def reference_forces(q,v,endpoint,endpoint_velocity,wind,*,length,EA,BA,density,diameter,Cd,rho,g):
    """Independent force law; no calls to Torch production equations."""
    n=len(q)+1;rest=length/n
    positions=np.concatenate((np.zeros((1,3)),q,np.asarray(endpoint)[None]))
    velocities=np.concatenate((np.zeros((1,3)),v,np.asarray(endpoint_velocity)[None]))
    segment=np.diff(positions,axis=0);distance=np.maximum(np.linalg.norm(segment,axis=1),1e-12)
    direction=segment/distance[:,None]
    rate=np.sum(np.diff(velocities,axis=0)*direction,axis=1)
    tension=np.where(distance>=rest,np.maximum(EA/rest*(distance-rest)+BA/rest*rate,0.),0.)
    tractions=tension[:,None]*direction
    force=tractions[1:]-tractions[:-1]
    weights=np.ones(n-1);weights[0]+=.5;weights[-1]+=.5
    tributary=rest*weights;mass=density*tributary
    tangent=direction[:-1]+direction[1:]
    tangent=tangent/np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-12)
    relative=v-wind
    axial=np.sum(relative*tangent,axis=1,keepdims=True)*tangent;normal=relative-axial
    drag=-.5*rho*Cd*diameter*tributary[:,None]*(np.linalg.norm(normal,axis=1,keepdims=True)*normal+
        .02*np.linalg.norm(axial,axis=1,keepdims=True)*axial)
    return force+drag+mass[:,None]*np.array([0.,0.,g]),-tractions[-1],tension


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=.3)
    args=parser.parse_args()
    if not .05<=args.seconds<=1 or not math.isclose(args.seconds/.002,round(args.seconds/.002)):
        raise ValueError('Short held-end check, integral2ms samples,0.05..1s')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    p=read(ROOT/'flexible_rope_protocol.json');base=plant_config();n=int(p['rope']['segments'])
    all_rows=[];traces={};started=time.perf_counter()
    for arm,length in [(.05,4.90),(.10,4.85)]:
        m=FlexibleRopeModel(base,device='cpu',dtype=torch.float64,arms=[[0.,0.,arm]],lengths=[length],rope=p['rope'])
        initial=m.initial(1).detach();q,v=m.nodes(initial)
        y0=np.concatenate((q.numpy().flatten(),v.numpy().flatten()))
        mass=m.node_mass.numpy().reshape(-1)
        law=dict(length=length,EA=p['rope']['axial_rigidity_n'],BA=p['rope']['axial_viscosity_ns'],
            density=p['rope']['linear_density_kgpm'],diameter=p['rope']['diameter_m'],
            Cd=p['rope']['air_drag_coefficient'],rho=m.rho,g=m.gravity)
        def boundary(t):
            w=2*math.pi*2.
            return np.array([.12*(1-math.cos(w*t)),0.,-5+arm]),np.array([.12*w*math.sin(w*t),0.,0.])
        for speed in [0.,5.]:
            wind=np.array([speed,0.,0.]);key=f'{round(arm*100)}cm_wind{speed:g}'
            samples=np.arange(round(args.seconds/.002)+1)*.002
            def rhs(t,y):
                q=y[:3*(n-1)].reshape(-1,3);v=y[3*(n-1):].reshape(-1,3)
                end,endv=boundary(t);force,_,_=reference_forces(q,v,end,endv,wind,**law)
                return np.concatenate((v.flatten(),(force/mass[:,None]).flatten()))
            tick=time.perf_counter()
            reference=solve_ivp(rhs,(0.,args.seconds),y0,method='Radau',rtol=1e-7,atol=1e-9,
                                t_eval=samples,max_step=.002,dense_output=True)
            if not reference.success:raise RuntimeError(reference.message)
            reference_s=time.perf_counter()-tick
            ref_nodes=reference.y[:3*(n-1)].T.reshape(-1,n-1,3)
            ref_force=[]
            for t,y in zip(reference.t,reference.y.T):
                end,endv=boundary(t)
                ref_force.append(reference_forces(y[:3*(n-1)].reshape(-1,3),y[3*(n-1):].reshape(-1,3),end,endv,wind,**law)[1])
            ref_force=np.asarray(ref_force)
            refinements=[]
            # Enforce prescribed endpoint velocity without changing cable mass or
            # material: body inverse mass/inertia approach zero in the constraint
            # solve; body positions are overwritten by the exact boundary path.
            m.mass=1e20;m.inertia.fill_(1e20)
            for dt in [.002,.001,.0005]:
                m.dt=dt;x=initial.clone();nodes=[q.numpy()[0].copy()];forces=[ref_force[0]];applied=[];applied_vectors=[];residuals=[]
                for step in range(round(args.seconds/dt)):
                    end,endv=boundary(step*dt);_,nextv=boundary((step+1)*dt)
                    x[:,:3]=x.new_tensor(end)-x.new_tensor([0.,0.,arm])
                    x[:,3:6]=x.new_tensor(nextv)
                    w=x.new_tensor(wind)[None];load=m.rope_loads(x,w);external=load['node_external_force']
                    force=torch.cat((x.new_zeros(1,6),external.flatten(1)),1)
                    with torch.no_grad():velocity,tension,residual=m.velocity_step(x,force,w)
                    position,oldv=m.nodes(x);newv=velocity[:,6:].reshape_as(oldv)
                    x=m.with_nodes(x,position+dt*newv,newv)
                    end,endv=boundary((step+1)*dt)
                    x[:,:3]=x.new_tensor(end)-x.new_tensor([0.,0.,arm]);x[:,3:6]=x.new_tensor(endv)
                    residuals.append(float(residual.max()));applied.append(float(tension[0,-1]))
                    applied_vectors.append((-tension[0,-1]*load['direction'][0,-1]).numpy().copy())
                    if (step+1)%round(.002/dt)==0:
                        nodes.append(m.nodes(x)[0].numpy()[0].copy())
                        forces.append(m.rope_loads(x,w)['endpoint_force'].numpy()[0].copy())
                values=np.stack(nodes);loads=np.stack(forces)
                node_gap=np.linalg.norm(values-ref_nodes,axis=-1)
                load_gap=np.linalg.norm(loads-ref_force,axis=-1)
                # Compare actual applied force impulses with reference interval
                # averages; instantaneous endpoint loads are a different metric.
                locations,weights=np.polynomial.legendre.leggauss(3)
                interval_force=[]
                for step in range(len(applied_vectors)):
                    average=np.zeros(3)
                    for location,weight in zip(locations,weights):
                        t=(step+.5+.5*location)*dt;y=reference.sol(t);end,endv=boundary(t)
                        average+=weight*.5*reference_forces(y[:3*(n-1)].reshape(-1,3),
                            y[3*(n-1):].reshape(-1,3),end,endv,wind,**law)[1]
                    interval_force.append(average)
                impulse_gap=np.linalg.norm((np.stack(applied_vectors)-np.stack(interval_force))*dt,axis=-1)
                refinements.append(dict(dt_s=dt,maximum_node_gap_m=float(node_gap.max()),
                    final_maximum_node_gap_m=float(node_gap[-1].max()),maximum_endpoint_force_gap_n=float(load_gap.max()),
                    rms_endpoint_force_gap_n=float(np.sqrt(np.mean(load_gap**2))),
                    maximum_complementarity_residual=float(max(residuals)),
                    maximum_terminal_force_impulse_gap_ns=float(impulse_gap.max()),
                    mean_terminal_force_impulse_gap_ns=float(impulse_gap.mean()),
                    peak_applied_terminal_tension_n=max(applied)))
                traces[f'{key}_dt{dt}_nodes']=values;traces[f'{key}_dt{dt}_forces']=loads
            traces[f'{key}_times']=samples;traces[f'{key}_reference_nodes']=ref_nodes;traces[f'{key}_reference_forces']=ref_force
            row=dict(attachment_m=arm,length_m=length,wind_mps=wind.tolist(),duration_s=args.seconds,
                reference=dict(method='Radau',rtol=1e-7,atol=1e-9,max_step_s=.002,nfev=reference.nfev,
                               nlu=reference.nlu,wall_s=reference_s,success=True),refinements=refinements)
            all_rows.append(row);print(row,flush=True)
    np.savez_compressed(out/'traces.npz',**traces)
    write(out/'result.json',dict(scope='Prescribed endpoint cable test; not flight, not training, not measured calibration',
        rope=p['rope'],boundary='x=.12*(1-cos(4*pi*t)),y=0,z=-5+arm',rows=all_rows,
        source_sha256={str(Path(__file__)):digest(__file__),str(ROOT/'hover_bptt/flexible_rope.py'):digest(ROOT/'hover_bptt/flexible_rope.py')},
        scipy_version=scipy.__version__,numpy_version=np.__version__,torch_version=torch.__version__,
        optimizer_created=False,wall_s=time.perf_counter()-started,final_acceptance=False))


if __name__=='__main__':main()
