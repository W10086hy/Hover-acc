"""Exogenous simulator wind;never an input to the policy."""
import math
import numpy as np

def make_wind(speed,seed,profile='varying',direction_deg=90.):
    if not math.isfinite(speed) or speed<0 or not math.isfinite(direction_deg):raise ValueError('Finite nonnegative wind speed/direction required')
    if profile=='steady':
        direction=math.radians(direction_deg);v=np.array([speed*math.cos(direction),speed*math.sin(direction),0.])
        return lambda t:v.copy()
    if profile=='narrow_varying':
        phases = np.random.default_rng(seed).uniform(0, 2*np.pi, 3)
        center_direction = math.radians(direction_deg)

        def narrow_wind(t):
            amplitude = speed*(.95+.04*math.sin(2*math.pi*t/7+phases[0])
                               +.01*math.sin(2*math.pi*t/1.7+phases[1]))
            direction = center_direction+math.radians(5)*math.sin(2*math.pi*t/17+phases[2])
            return np.array([amplitude*math.cos(direction), amplitude*math.sin(direction), 0.])

        return narrow_wind
    if profile!='varying':raise ValueError('Unknown wind profile')
    phases=np.random.default_rng(seed).uniform(0,2*np.pi,3);ref=np.arange(0,120.0001,.01)
    norm=(.60+.25*np.sin(2*np.pi*ref/7+phases[0])+.15*np.sin(2*np.pi*ref/1.7+phases[1])).max()
    def wind(t):
        amp=speed*(.60+.25*np.sin(2*np.pi*t/7+phases[0])+.15*np.sin(2*np.pi*t/1.7+phases[1]))/norm
        direction=np.deg2rad(direction_deg)+np.deg2rad(25)*np.sin(2*np.pi*t/17+phases[2])
        return np.array([amp*np.cos(direction),amp*np.sin(direction),0.])
    return wind
