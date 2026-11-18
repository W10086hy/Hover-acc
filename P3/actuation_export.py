"""Export an independent actuation compatibility candidate, with frozen weights."""
import argparse
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from P2.hover_bptt.io import read,write,digest
from px4_offboard.aligned_p3 import AlignedCalibration


def export(folder,*,acceleration=False):
    folder=Path(folder);parent=folder/'observer_consistent.json'
    destination=folder/('observer_accel_consistent.json' if acceleration else 'observer_virtual.json')
    tag='acceleration_export' if acceleration else 'actuation_export'
    if destination.exists():raise FileExistsError('Preserve previous actuation candidate')
    parameters=read(parent)
    assert parameters['checkpoint_sha256']==digest(folder/'model.pt')
    if acceleration:
        parameters['acceleration_actuation']='attitude_reference_consistent'
        parameters['scope']+='; horizontal acceleration transformed to the PX4 reported attitude reference'
        semantics='Only the horizontal acceleration setpoint is adapted using the frozen Position tilt correction'
    else:
        parameters['actuation_observation']='standard_vehicle_equivalent'
        parameters['scope']+='; equivalent standard-body RPM/throttle observations with unchanged collective'
        semantics='Only Actor RPM/throttle channels standardized using learned own angular residual; physical feedback and publication unchanged'
    AlignedCalibration(parameters)
    write(destination,parameters)
    repo=Path(__file__).resolve().parents[1]
    sources=['P3/actuation_export.py','px4_offboard/aligned_p3.py','px4_offboard/policy_timing.py']
    if acceleration:sources.append('px4_offboard/mavros_policy.py')
    with ZipFile(folder/(tag+'_source.zip'),'x',ZIP_DEFLATED) as z:
        for path in sources:z.write(repo/path,path)
    proof=dict(status='candidate_only',closed_loop_validated=False,P2_updates=0,P3_weight_updates=0,
        checkpoint_sha256=digest(folder/'model.pt'),parent_sha256=digest(parent),observer_sha256=digest(destination),
        source_sha256={path:digest(repo/path) for path in sources},
        semantics=semantics)
    write(folder/(tag+'.json'),proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--acceleration',action='store_true');a=p.parse_args()
    print(export(a.folder,acceleration=a.acceleration))
