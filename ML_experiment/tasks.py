from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class Task:
    name: str
    kind: str
    output_dim: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    tail_x: list[torch.Tensor]
    tail_y: list[torch.Tensor]
    visual_limits: tuple[float, float, float, float] | None = None
    truth: object | None = None
    target_mean: torch.Tensor | None = None
    target_std: torch.Tensor | None = None

    @property
    def input_dim(self): return int(self.x_train.shape[1])


def _split(x, y, seed, train_fraction=.75):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    cut = int(train_fraction * len(x)); train, val = order[:cut], order[cut:]
    return x[train], y[train], x[val], y[val]


def _normalize_regression(task: Task):
    mean = task.y_train.mean(0, keepdim=True)
    std = task.y_train.std(0, keepdim=True).clamp_min(1e-6)
    task.y_train = (task.y_train - mean) / std; task.y_val = (task.y_val - mean) / std
    task.y_test = (task.y_test - mean) / std
    task.tail_y = [(value - mean) / std for value in task.tail_y]
    task.target_mean, task.target_std = mean, std
    return task


def _uniform(count, low, high, seed, dim=2):
    return torch.rand((count, dim), generator=torch.Generator().manual_seed(seed)) * (high - low) + low


def _spiral_points(count, lo, hi, seed, noise=.018):
    generator = torch.Generator().manual_seed(seed); u = torch.rand(count, generator=generator) * (hi - lo) + lo
    theta = .55 + 5.8 * math.pi * u; radius = .12 + .88 * u
    base = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), 1)
    x = torch.cat((base, -base)) + noise * torch.randn((2 * count, 2), generator=generator)
    y = torch.cat((torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long)))
    return x, y


def spiral(seed=0):
    x, y = _spiral_points(1800, .015, .5, 100 + seed); xtr, ytr, xva, yva = _split(x, y, 200 + seed)
    tail_x, tail_y = [], []
    for index in range(10):
        a, b = _spiral_points(300, .5 + .05 * index, .55 + .05 * index, 1000 + seed * 31 + index)
        tail_x.append(a); tail_y.append(b)
    xt, yt = _spiral_points(1000, .5, 1, 1900 + seed)
    def truth(p):
        angle = torch.atan2(p[:, 1], p[:, 0]); radius = torch.linalg.vector_norm(p, dim=1)
        expected = .55 + 5.8 * math.pi * (radius - .12) / .88
        return (torch.cos(angle - expected) < 0).long()
    return Task("spiral", "classification", 2, xtr, ytr, xva, yva, xt, yt, tail_x, tail_y,
                (-1.2, 1.2, -1.2, 1.2), truth)


def _orthogonal(dim, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    return torch.tensor(q, dtype=torch.float32)


def _nd_spiral_points(count, lo, hi, ambient, planes, seed, rotation):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(count, generator=generator) * (hi - lo) + lo
    theta = .45 + 5.4 * math.pi * u; radius = .12 + .88 * u; branches = []
    for label in (0, 1):
        features = []
        for plane in range(planes):
            frequency = plane + 1; phase = frequency * theta + .37 * plane + label * math.pi
            amplitude = radius * (1 + .08 * torch.sin((plane + 2) * theta)) / math.sqrt(planes)
            features.extend((amplitude * torch.cos(phase), amplitude * torch.sin(phase)))
        core = torch.stack(features, 1); padded = torch.zeros((count, ambient)); padded[:, :2 * planes] = core
        branches.append(padded @ rotation)
    x = torch.cat(branches) + .006 * torch.randn((2 * count, ambient), generator=generator)
    y = torch.cat((torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long)))
    return x, y


def nd_spiral(planes, seed=0, ambient=16):
    if 2 * planes > ambient: raise ValueError("each spiral plane needs two ambient dimensions")
    rotation = _orthogonal(ambient, 700 + seed)
    x, y = _nd_spiral_points(1800, .015, .5, ambient, planes, 800 + seed, rotation)
    xtr, ytr, xva, yva = _split(x, y, 900 + seed); tail_x, tail_y = [], []
    for index in range(10):
        lo, hi = .5 + .05 * index, .55 + .05 * index
        a, b = _nd_spiral_points(300, lo, hi, ambient, planes, 2100 + seed * 31 + index, rotation)
        tail_x.append(a); tail_y.append(b)
    xt, yt = _nd_spiral_points(1000, .5, 1, ambient, planes, 2900 + seed, rotation)
    name = "nd_spiral_low_rank" if planes == 1 else "nd_spiral_high_rank"
    return Task(name, "classification", 2, xtr, ytr, xva, yva, xt, yt, tail_x, tail_y)


def hypercube_checker(seed=0, dim=16):
    generator = torch.Generator().manual_seed(5600 + seed); rotation = _orthogonal(dim, 5700 + seed)
    def sample(count, lo, hi):
        chunks = []
        while sum(len(chunk) for chunk in chunks) < count:
            raw = (torch.rand((count * 2, dim), generator=generator) * 2 - 1) * hi
            radius = raw.abs().amax(1); chunks.append(raw[(radius >= lo) & (radius < hi)])
        raw = torch.cat(chunks)[:count]; label = (raw[:, :8] > 0).long().sum(1).remainder(2)
        return raw @ rotation, label
    x, y = sample(5000, 0, .62); xtr, ytr, xva, yva = _split(x, y, 5800 + seed)
    tail_x, tail_y = [], []
    for index in range(10):
        a, b = sample(600, .62 + .038 * index, .62 + .038 * (index + 1)); tail_x.append(a); tail_y.append(b)
    xt, yt = sample(3000, .62, 1.0)
    return Task("hypercube_checker", "classification", 2, xtr, ytr, xva, yva, xt, yt, tail_x, tail_y)


def _checker_truth(x):
    return (torch.floor((x[:, 0] + 1.2) / .30).long() + torch.floor((x[:, 1] + 1.2) / .30).long()).remainder(2)


def _balanced_region(count, seed, predicate):
    generator = torch.Generator().manual_seed(seed); parts = [[], []]
    while min(sum(map(len, parts[0])), sum(map(len, parts[1]))) < count // 2:
        x = _uniform(max(2000, count * 2), -1.2, 1.2, int(torch.randint(2**30, (), generator=generator)))
        x = x[predicate(x)]; y = _checker_truth(x)
        for label in (0, 1):
            need = count // 2 - sum(map(len, parts[label]))
            if need > 0: parts[label].append(x[y == label][:need])
    x = torch.cat((torch.cat(parts[0]), torch.cat(parts[1]))); y = _checker_truth(x)
    order = torch.randperm(len(x), generator=generator); return x[order], y[order]


def checkerboard(seed=0):
    x, y = _balanced_region(4200, 2100 + seed, lambda z: z.abs().amax(1) <= .58)
    xtr, ytr, xva, yva = _split(x, y, 2200 + seed); tail_x, tail_y = [], []
    for index in range(10):
        lo, hi = .58 + .062 * index, .58 + .062 * (index + 1)
        a, b = _balanced_region(600, 2300 + seed * 31 + index,
                                lambda z, lo=lo, hi=hi: (z.abs().amax(1) >= lo) & (z.abs().amax(1) < hi))
        tail_x.append(a); tail_y.append(b)
    xt, yt = _balanced_region(3000, 2600 + seed, lambda z: z.abs().amax(1) > .58)
    return Task("checkerboard", "classification", 2, xtr, ytr, xva, yva, xt, yt, tail_x, tail_y,
                (-1.2, 1.2, -1.2, 1.2), _checker_truth)


def two_moons(seed=0):
    generator = torch.Generator().manual_seed(3000 + seed); count = 2600
    t = torch.rand(count, generator=generator) * math.pi
    first = torch.stack((torch.cos(t), torch.sin(t)), 1)
    second = torch.stack((1 - torch.cos(t), .45 - torch.sin(t)), 1)
    x = torch.cat((first, second)) + .09 * torch.randn((2 * count, 2), generator=generator)
    y = torch.cat((torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long)))
    xtr, ytr, xva, yva = _split(x, y, 3100 + seed)
    return Task("two_moons", "classification", 2, xtr, ytr, xva, yva, xva, yva, [], [], (-1.3, 2.3, -1.3, 1.3))


def pinwheel(seed=0, classes=5):
    generator = torch.Generator().manual_seed(3200 + seed); per_class = 1100
    labels = torch.arange(classes).repeat_interleave(per_class)
    radial = torch.randn(classes * per_class, generator=generator) * .22 + 1
    base = 2 * math.pi * labels.float() / classes
    angle = base + 1.8 * (radial - 1) + .12 * torch.randn(len(labels), generator=generator)
    x = torch.stack((radial * torch.cos(angle), radial * torch.sin(angle)), 1)
    xtr, ytr, xva, yva = _split(x, labels, 3300 + seed)
    return Task("pinwheel", "classification", classes, xtr, ytr, xva, yva, xva, yva, [], [], (-1.6, 1.6, -1.6, 1.6))


def _functional_classification(name, truth, limits, seed, count=7000):
    xmin, xmax, ymin, ymax = limits; generator = torch.Generator().manual_seed(seed)
    pools = [[], []]
    while min(sum(map(len, pools[0])), sum(map(len, pools[1]))) < count // 2:
        x = torch.stack((torch.rand(count, generator=generator) * (xmax - xmin) + xmin,
                         torch.rand(count, generator=generator) * (ymax - ymin) + ymin), 1)
        y = truth(x)
        for label in (0, 1):
            need = count // 2 - sum(map(len, pools[label]))
            if need > 0: pools[label].append(x[y == label][:need])
    x = torch.cat((torch.cat(pools[0]), torch.cat(pools[1]))); y = truth(x)
    order = torch.randperm(len(x), generator=generator); x, y = x[order], y[order]
    xtr, ytr, xva, yva = _split(x, y, seed + 1)
    return Task(name, "classification", 2, xtr, ytr, xva, yva, xva, yva, [], [], limits, truth)


def xor_quads(seed=0):
    return _functional_classification("xor_quads", lambda x: ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long(),
                                      (-1, 1, -1, 1), 3400 + seed)


def sinusoid_bounds(seed=0):
    truth = lambda x: (x[:, 1].abs() < .42 + .22 * torch.sin(3.2 * x[:, 0])).long()
    return _functional_classification("sinusoid_bounds", truth, (-2, 2, -1.2, 1.2), 3500 + seed)


def radial_stripes(seed=0):
    truth = lambda x: torch.floor(4.2 * torch.linalg.vector_norm(x, dim=1)).long().remainder(2)
    return _functional_classification("radial_stripes", truth, (-1.5, 1.5, -1.5, 1.5), 3600 + seed)


def swiss_cheese(seed=0):
    centers = torch.tensor([[-1.1, -.8], [-.3, .6], [.65, -.55], [1.05, .75], [.15, -1.15]])
    radii = torch.tensor([.33, .28, .36, .31, .25])
    def truth(x):
        inside = ((x[:, None, :] - centers[None]).square().sum(2) < radii[None].square()).any(1)
        return (~inside).long()
    return _functional_classification("swiss_cheese", truth, (-1.6, 1.6, -1.6, 1.6), 3700 + seed)


def lorenz_lobes(seed=0):
    generator = torch.Generator().manual_seed(3800 + seed); state = torch.tensor([.1, .0, .0]); points = []
    for index in range(72000):
        x, y, z = state; derivative = torch.stack((10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z))
        state = state + .004 * derivative
        if index > 8000 and index % 8 == 0: points.append(state.clone())
    xyz = torch.stack(points); x = xyz[:, [0, 2]]; x = (x - x.mean(0)) / x.std(0)
    y = (xyz[:, 0] > 0).long(); order = torch.randperm(len(x), generator=generator); x, y = x[order], y[order]
    xtr, ytr, xva, yva = _split(x, y, 3900 + seed)
    return Task("lorenz_lobes", "classification", 2, xtr, ytr, xva, yva, xva, yva, [], [], (-2.5, 2.5, -2.5, 2.5))


def _regression_2d(name, function, limits, seed):
    xmin, xmax, ymin, ymax = limits
    x = torch.stack((torch.rand(8000, generator=torch.Generator().manual_seed(seed)) * (xmax - xmin) + xmin,
                     torch.rand(8000, generator=torch.Generator().manual_seed(seed + 1)) * (ymax - ymin) + ymin), 1)
    y = function(x).reshape(len(x), -1).float(); xtr, ytr, xva, yva = _split(x, y, seed + 2)
    task = Task(name, "regression", y.shape[1], xtr, ytr, xva, yva, xva, yva, [], [], limits, function)
    return _normalize_regression(task)


def periodic_wells(seed=0):
    return _regression_2d("periodic_wells", lambda x: (torch.cos(3 * x[:, 0]) + torch.cos(3 * x[:, 1]))[:, None],
                          (-math.pi, math.pi, -math.pi, math.pi), 4000 + seed)


def ripple(seed=0):
    return _regression_2d("ripple", lambda x: (torch.sin(3*x[:, 0])*torch.cos(3*x[:, 1]) + .5*torch.sin(5*x[:, 0]+x[:, 1]))[:, None],
                          (-math.pi, math.pi, -math.pi, math.pi), 4100 + seed)


def ring_sdf(seed=0):
    return _regression_2d("ring_sdf", lambda x: (torch.linalg.vector_norm(x, dim=1) - 1.2)[:, None],
                          (-2, 2, -2, 2), 4200 + seed)


def _complex_spiral(t):
    radius = .12*t + .015*t.square() + .07*torch.sin(3.1*t) + .015*torch.sin(.7*t.square())
    angle = t + .25*torch.sin(5*t) + .00035*t.pow(3)
    z = .055*t + .018*torch.sin(2.3*t) + .00055*t.square() - .000002*t.pow(3)
    return torch.cat((radius*torch.cos(angle), radius*torch.sin(angle), z), 1)


def complex_spiral_3d(seed=0):
    generator = torch.Generator().manual_seed(4300 + seed); total = 12 * math.pi
    t = torch.rand((6000, 1), generator=generator) * (.5 * total); y = _complex_spiral(t); x = 2 * t / total - 1
    xtr, ytr, xva, yva = _split(x, y, 4400 + seed); tail_x, tail_y = [], []
    for index in range(10):
        lo, hi = (.5 + .05 * index) * total, (.55 + .05 * index) * total
        tt = torch.linspace(lo, hi, 300)[:, None]; tail_x.append(2 * tt / total - 1); tail_y.append(_complex_spiral(tt))
    tt = torch.linspace(.5 * total, total, 2000)[:, None]
    task = Task("complex_spiral_3d", "regression", 3, xtr, ytr, xva, yva, 2*tt/total-1, _complex_spiral(tt), tail_x, tail_y)
    return _normalize_regression(task)


def periodic_nd(seed=0, dim=8):
    x = _uniform(10000, -math.pi, math.pi, 4500 + seed, dim); frequencies = torch.arange(dim).remainder(5) + 1
    y = (torch.cos(x * frequencies).sum(1) + .3 * torch.sin(2*x).sum(1))[:, None] / dim
    xtr, ytr, xva, yva = _split(x, y, 4600 + seed)
    return _normalize_regression(Task("periodic_nd", "regression", 1, xtr, ytr, xva, yva, xva, yva, [], []))


def hyperchecker(seed=0, dim=10):
    x = _uniform(12000, -math.pi, math.pi, 4700 + seed, dim)
    y = (torch.sign(torch.sin(3*x) + 1e-6).prod(1) > 0).long(); xtr, ytr, xva, yva = _split(x, y, 4800 + seed)
    return Task("hyperchecker", "classification", 2, xtr, ytr, xva, yva, xva, yva, [], [])


def _regression_1d(name, function, seed, train_limit=3.0, test_limit=5.0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((7000, 1), generator=generator) * (2 * train_limit) - train_limit
    y = function(x).reshape(len(x), -1); xtr, ytr, xva, yva = _split(x, y, seed + 1)
    tail_x, tail_y = [], []
    span = test_limit - train_limit
    for index in range(10):
        lo = train_limit + span * index / 10; hi = train_limit + span * (index + 1) / 10
        right = torch.linspace(lo, hi, 240)[:, None]; left = -torch.flip(right, (0,))
        points = torch.cat((left, right)); tail_x.append(points); tail_y.append(function(points).reshape(len(points), -1))
    test = torch.linspace(-test_limit, test_limit, 3000)[:, None]
    task = Task(name, "regression", y.shape[1], xtr, ytr, xva, yva,
                test, function(test).reshape(len(test), -1), tail_x, tail_y,
                truth=function)
    return _normalize_regression(task)


def multiscale_1d(seed=0):
    def target_fn(x):
        """Multi-scale signal supplied by the user."""
        smooth = 0.4 * torch.sin(1.5 * x)
        bumps = 0.6 * torch.exp(-30 * (x - 2.0) ** 2) - 0.4 * torch.exp(-40 * (x + 1.5) ** 2)
        medium = 0.3 * torch.sin(6 * x) * torch.exp(-0.15 * x**2)
        sharp = 0.2 * torch.sin(20 * x) * torch.exp(-0.3 * x**2)
        return smooth + bumps + medium + sharp
    return _regression_1d("multiscale_1d", target_fn, 5000 + seed)


def chirp_1d(seed=0):
    return _regression_1d("chirp_1d", lambda x: .55*torch.sin(.85*x.square()+.3*x) + .2*torch.sin(13*x)/(1+.12*x.square()), 5100 + seed)


def poly_drifted_chirp_1d(seed=0):
    """Amplitude- and phase-drifting chirp supplied by the user.

    The model observes normalized coordinates in [-1, 1], corresponding to
    physical x in [-pi, pi], and is evaluated through [-2, 2].
    """
    def function(normalized_x):
        x = math.pi * normalized_x
        return (1 + .2*x + .05*x.square()) * torch.sin(4*x + .4*x.square())
    return _regression_1d(
        "poly_drifted_chirp_1d", function, 5150 + seed,
        train_limit=1.0, test_limit=2.0,
    )


def localized_steps_1d(seed=0):
    def function(x):
        return .25*x + .55*torch.tanh(18*(x+1.1)) - .7*torch.tanh(24*(x-.45)) + .35*torch.exp(-45*(x-2.1).square())
    return _regression_1d("localized_steps_1d", function, 5200 + seed)


def fourier_mix_1d(seed=0):
    return _regression_1d("fourier_mix_1d", lambda x: .45*torch.sin(x)+.28*torch.cos(3*x-.2)+.18*torch.sin(9*x)+.09*torch.cos(21*x), 5300 + seed)


TASK_BUILDERS = {
    "spiral": spiral, "checkerboard": checkerboard, "two_moons": two_moons, "pinwheel": pinwheel,
    "nd_spiral_low_rank": lambda seed=0: nd_spiral(1, seed),
    "nd_spiral_high_rank": lambda seed=0: nd_spiral(8, seed), "hypercube_checker": hypercube_checker,
    "xor_quads": xor_quads, "sinusoid_bounds": sinusoid_bounds, "radial_stripes": radial_stripes,
    "swiss_cheese": swiss_cheese, "lorenz_lobes": lorenz_lobes, "periodic_wells": periodic_wells,
    "ripple": ripple, "ring_sdf": ring_sdf, "complex_spiral_3d": complex_spiral_3d,
    "periodic_nd": periodic_nd, "hyperchecker": hyperchecker,
    "multiscale_1d": multiscale_1d, "chirp_1d": chirp_1d,
    "poly_drifted_chirp_1d": poly_drifted_chirp_1d,
    "localized_steps_1d": localized_steps_1d, "fourier_mix_1d": fourier_mix_1d,
}
