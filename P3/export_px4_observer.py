"""Export frozen Position calibration for NumPy/MAVROS observer standardization."""
import argparse
from pathlib import Path
from P2.hover_bptt.io import read,write,digest
from .train_body_calibration import load_calibration
from px4_offboard.p3_observer import ObserverCalibration,physical_reference


def export(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Preserve earlier calibration export')
    audit=read(root/'alignment_audit.json');path=root/'calibration_aligned/calibration.pt'
    if not audit['passed'] or audit['checkpoint_sha256']!=digest(path):raise ValueError('Validated aligned Position checkpoint required')
    model,record=load_calibration(path);report=record['report']
    if not report['development_quality_gate_passed']:raise ValueError('P3 holdout gate did not pass')
    data=dict(schema='P3_observer_standardization_v1',quality_passed=True,alignment_audit_passed=True,
        checkpoint_sha256=digest(path),alignment_audit_sha256=digest(root/'alignment_audit.json'),
        body_from_sensor=model.body_from_sensor.tolist(),body_from_virtual=model.body_from_virtual.tolist(),
        gyro_bias_sensor=model.gyro_bias_sensor.tolist(),bias_body=model.bias.tolist(),com_geometry=model.com_geometry.tolist(),
        mass_kg=report['axis_mass']['mass_kg'],motor_speed_gain=read(root/'calibration_aligned/actuator_gain.json')['speed_gain'],
        motor_tau_s=report['motor_response']['tau'] if report['motor_response']['identified'] else record['nominal']['actuators']['motor_tau_s'],
        reference_physics=physical_reference(record['nominal']),scope='P3 sensor/reference/innovation standardization; PX4 owns disturbance rejection; no torque feedforward field',
        additional_Actor_inputs=0,PX4_firmware_modified=False)
    ObserverCalibration(data).validate_reference(record['nominal'])
    output.parent.mkdir(parents=True,exist_ok=True);write(output,data);print(dict(output=str(output),sha256=digest(output)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();export(a.root,a.output)
