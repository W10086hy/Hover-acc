"""MAVLink transport for horizontal acceleration and native PX4 altitude hold."""

import math

from .offboard import ONBOARD_SAMPLE_MAX_AGE_S


MAV_FRAME_LOCAL_NED = 1
# Ignore x/y positions, all velocities, vertical acceleration and yaw rate.
HORIZONTAL_ACCELERATION_ALTITUDE_MASK = (1 << 0) | (1 << 1) | (7 << 3) | (1 << 8) | (1 << 11)


class AltitudeSetpoint:
    """Caller supplies a fresh, valid local-NED estimator sample and mode state."""

    def __init__(self, link, *, launch_heading_rad):
        if not math.isfinite(launch_heading_rad):
            raise ValueError('Finite launch heading required in the PX4 local frame')
        self.link = link
        self.heading = math.atan2(math.sin(launch_heading_rad), math.cos(launch_heading_rad))
        self.target_z_m = None
        self._reset_counter = None
        self._frame_reset = False

    @staticmethod
    def _fresh(clock, z, received, counter):
        return (z is not None and received is not None and counter is not None
                and math.isfinite(z) and 0 <= clock-received <= ONBOARD_SAMPLE_MAX_AGE_S
                and 0 <= counter <= 255)

    def _send(self, clock, z, north, east):
        self.link.mav.set_position_target_local_ned_send(
            int(clock*1000) & 0xffffffff, 1, 1, MAV_FRAME_LOCAL_NED,
            HORIZONTAL_ACCELERATION_ALTITUDE_MASK, 0., 0., float(z), 0., 0., 0.,
            float(north), float(east), 0., self.heading, 0.)

    def prime(self, clock, *, local_z_m, received_s, frame_reset_counter):
        """Prestream neutral XY with current estimated Z before requesting Offboard."""
        if self.target_z_m is not None or not self._fresh(clock, local_z_m, received_s, frame_reset_counter):
            return False
        self._send(clock, local_z_m, 0., 0.)
        return True

    def update(self, clock, *, mode_active, local_z_m=None, received_s=None,
               frame_reset_counter=None, acceleration_xy=None):
        """Latch Z at entry and send XY from the Actor in the launch frame."""
        if not mode_active:
            self.target_z_m = None
            self._reset_counter = None
            self._frame_reset = False
            return False
        if self._frame_reset or not self._fresh(clock, local_z_m, received_s, frame_reset_counter):
            return False
        if self._reset_counter is not None and frame_reset_counter != self._reset_counter:
            self._frame_reset = True
            return False
        if self.target_z_m is None:
            self.target_z_m = float(local_z_m)
            self._reset_counter = frame_reset_counter
        if acceleration_xy is None:
            return False
        forward, right = acceleration_xy
        if not math.isfinite(forward) or not math.isfinite(right):
            return False
        cosine, sine = math.cos(self.heading), math.sin(self.heading)
        self._send(clock, self.target_z_m, cosine*forward - sine*right, sine*forward + cosine*right)
        return True
