"""Independent 3-D civilian motion fixtures; only observations reach filters."""
import numpy as np


FAMILIES = ['line', 'helix', 'figure8', 'stop_go', 'switching']


def make_case(family, seed, corrupt=False, samples=121, observation_times=None):
    rng = np.random.default_rng(seed)
    dt = .1*rng.uniform(.8, 1.2, samples-1)
    times = np.r_[0., np.cumsum(dt)]
    if observation_times is not None:
        times = np.asarray(observation_times, float)
        if (times.shape != (samples,) or not np.isfinite(times).all()
                or times[0] != 0 or np.any(np.diff(times) <= 0)):
            raise ValueError('Require samples increasing finite times beginning at zero')
        dt = np.diff(times)
    t = times
    frequency = rng.uniform(.35, .7)
    speed = rng.uniform(1.5, 3.5)
    phase = rng.uniform(-.5, .5)
    if family == 'line':
        local = np.c_[speed*t, .08*np.sin(.6*t), .05*np.sin(.9*t)]
    elif family == 'helix':
        radius = speed/frequency
        local = np.c_[radius*np.sin(frequency*t), radius*(1-np.cos(frequency*t)), .35*t]
    elif family == 'figure8':
        radius = speed/frequency
        local = np.c_[radius*np.sin(frequency*t+phase), .45*radius*np.sin(2*frequency*t),
                       .35*radius*np.sin(.7*frequency*t)]
    elif family == 'stop_go':
        # Smooth slowing, dwell, and restart on a curved route. Positions are
        # generated analytically, independently of the estimator's group flow.
        progression = t-1.5*np.tanh((t-4)/1.5)+1.5*np.tanh(-4/1.5)
        local = np.c_[speed*progression, 1.2*np.sin(.6*progression), .3*np.sin(progression)]
    elif family == 'switching':
        # A small civilian craft approaching changing waypoints. Future commands
        # are independently drawn and never made available to any estimator.
        points = rng.normal(size=(8, 3))*[5., 5., 2.]
        p, v = np.zeros(3), np.array([speed, 0., 0.])
        records = [p.copy()]
        for k, duration in enumerate(dt):
            subdivisions = 20
            h = duration/subdivisions
            for j in range(subdivisions):
                now = times[k]+j*h
                waypoint = points[min(int(now/1.8), len(points)-1)]
                desired = waypoint-p
                desired *= speed/max(np.linalg.norm(desired), 1e-9)
                acceleration = (desired-v)/.55
                acceleration *= min(1., 5/max(np.linalg.norm(acceleration), 1e-9))
                p += v*h+.5*acceleration*h*h
                v += acceleration*h
            records.append(p.copy())
        local = np.array(records)
    else:
        raise ValueError(family)
    # Every case has a random 3-D orientation. This is not observed attitude.
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    position = (local-local[0]) @ rotation.T+rng.uniform([-3., -3., 8.], [3., 3., 12.])
    bias = np.zeros_like(position)
    sigma = np.full(samples, .35)
    available = np.ones(samples, bool)
    if corrupt:
        for k, duration in enumerate(dt, 1):
            rho = np.exp(-duration/6.)
            bias[k] = rho*bias[k-1]+.06*np.sqrt(duration)*rng.normal(size=3)
            if 4 < times[k] < 6:
                bias[k] += np.array([.18, -.08, .05])*duration
        sigma[(t > 3)&(t < 5)] = 1.1
        available = rng.random(samples) > .07
        available[(t > 7)&(t < 7.8)] = False
        available[0] = True
    noise = rng.normal(size=position.shape)*sigma[:, None]
    if corrupt:
        spike = rng.random(samples) < .035
        spike[0] = False
        noise[spike] += rng.normal(size=(np.count_nonzero(spike), 3))*3.
    observations = position+bias+noise
    observations[~available] = np.nan
    # A civilian planar geofence; it is only queried after forecasts are formed.
    # No filter is told that a boundary is a physical barrier or intended route.
    return {'family': family, 'seed': seed, 'corrupt': corrupt, 'times': times,
            'truth': position, 'observations': observations, 'bias': bias,
            'noise': noise, 'sigma': sigma, 'boundary_normal': np.array([1., 0., 0.]),
            'boundary_offset': 0.}
