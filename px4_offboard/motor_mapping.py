"""Map PX4 Quad X motor numbers to the frozen P2 per-motor channels."""

import numpy as np


# P2 channels: front-right, rear-right, front-left, rear-left (body FRD).
# PX4 v1.14.4 10016_none_iris numbers those motors 1, 4, 3, 2.
PX4_MOTORS_IN_P2_ORDER = (1, 4, 3, 2)
_PX4_TO_P2 = tuple(motor - 1 for motor in PX4_MOTORS_IN_P2_ORDER)


def px4_to_p2_motor_order(controls):
    """Read motors 1–4 from a PX4 output vector in P2 order."""
    return np.asarray(controls, dtype=float)[..., _PX4_TO_P2]
