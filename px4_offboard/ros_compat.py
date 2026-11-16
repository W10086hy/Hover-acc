"""Process-local compatibility for existing Noetic packages under Python 3.11."""
import logging
import sys


def load_rospy():
    path='/usr/lib/python3/dist-packages'
    if path not in sys.path:sys.path.append(path)
    import rosgraph.roslogging
    import rospy
    if sys.version_info>=(3,11):
        # Noetic's custom frame search can loop on its last stack frame when
        # Python 3.11 reports a different logging wrapper. Preserve standard
        # logging (including errors), using the interpreter's own caller lookup.
        rosgraph.roslogging.RospyLogger.findCaller=logging.Logger.findCaller
    return rospy
