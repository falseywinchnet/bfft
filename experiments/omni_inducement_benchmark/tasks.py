from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class Task:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    tail_x: list[torch.Tensor]
    tail_y: list[torch.Tensor]
    intrinsic_rank: int
    visual_limits: tuple[float, float, float, float] | None = None

    @property
    def input_dim(self) -> int:
        return int(self.x_train.shape[1])

    def truth(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "spiral_2d":
            theta = torch.atan2(x[:, 1], x[:, 0])
            radius = torch.linalg.vector_norm(x, dim=1)
            expected = .55 + 5.8 * math.pi * (radius - .12) / .88
            return (torch.cos(theta - expected) < 0).long()
        if self.name == "checkerboard_2d":
            ix = torch.floor((x[:, 0] + 1.2) / .30).long()
            iy = torch.floor((x[:, 1] + 1.2) / .30).long()
            return torch.remainder(ix + iy, 2)
        raise ValueError(f"{self.name} has no dense 2-D truth function")


def _orthogonal(dim: int, seed: int) -> torch.Tensor:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    return torch.tensor(q, dtype=torch.float32)


def _stratified_split(x: torch.Tensor, y: torch.Tensor, seed: int, val_fraction: float = .2):
    generator = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in (0, 1):
        indices = torch.where(y == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        cut = int(val_fraction * len(indices))
        val.append(indices[:cut]); train.append(indices[cut:])
    tr, va = torch.cat(train), torch.cat(val)
    return x[tr], y[tr], x[va], y[va]


def _spiral_samples(count: int, lo: float, hi: float, seed: int, noise: float = .018):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(count, generator=generator) * (hi - lo) + lo
    theta = .55 + 5.8 * math.pi * u
    radius = .12 + .88 * u
    base = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), 1)
    x = torch.cat((base, -base), 0)
    x = x + noise * torch.randn(x.shape, generator=generator)
    y = torch.cat((torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long)))
    return x, y


def spiral_2d(seed: int = 0, train_fraction: float = .5) -> Task:
    x, y = _spiral_samples(1400, .015, train_fraction, 100 + seed)
    xtr, ytr, xva, yva = _stratified_split(x, y, 200 + seed)
    tx, ty = [], []
    for index in range(10):
        lo = train_fraction + (1 - train_fraction) * index / 10
        hi = train_fraction + (1 - train_fraction) * (index + 1) / 10
        a, b = _spiral_samples(240, lo, hi, 1000 + 31 * seed + index)
        tx.append(a); ty.append(b)
    return Task("spiral_2d", xtr, ytr, xva, yva, tx, ty, 2, (-1.2, 1.2, -1.2, 1.2))


def _checker_label(x: torch.Tensor) -> torch.Tensor:
    ix = torch.floor((x[:, 0] + 1.2) / .30).long()
    iy = torch.floor((x[:, 1] + 1.2) / .30).long()
    return torch.remainder(ix + iy, 2)


def _balanced_uniform(count: int, low: float, high: float, seed: int, predicate=None):
    generator = torch.Generator().manual_seed(seed)
    parts = [[], []]
    total = lambda label: sum(len(tensor) for tensor in parts[label])
    while min(total(0), total(1)) < count // 2:
        candidate = torch.rand((max(2048, count * 2), 2), generator=generator) * (high - low) + low
        if predicate is not None:
            candidate = candidate[predicate(candidate)]
        labels = _checker_label(candidate)
        for label in (0, 1):
            needed = count // 2 - total(label)
            if needed > 0:
                parts[label].append(candidate[labels == label][:needed])
    x = torch.cat((torch.cat(parts[0]), torch.cat(parts[1])))
    y = torch.cat((torch.zeros(count // 2, dtype=torch.long), torch.ones(count // 2, dtype=torch.long)))
    order = torch.randperm(len(x), generator=generator)
    return x[order], y[order]


def checkerboard_2d(seed: int = 0) -> Task:
    inner = lambda x: x.abs().amax(1) <= .58
    x, y = _balanced_uniform(3200, -1.2, 1.2, 300 + seed, inner)
    xtr, ytr, xva, yva = _stratified_split(x, y, 400 + seed)
    tx, ty = [], []
    edges = torch.linspace(.58, 1.20, 11)
    for index in range(10):
        lo, hi = float(edges[index]), float(edges[index + 1])
        predicate = lambda z, a=lo, b=hi: (z.abs().amax(1) >= a) & (z.abs().amax(1) < b)
        a, b = _balanced_uniform(480, -1.2, 1.2, 1500 + 31 * seed + index, predicate)
        tx.append(a); ty.append(b)
    return Task("checkerboard_2d", xtr, ytr, xva, yva, tx, ty, 2, (-1.2, 1.2, -1.2, 1.2))


def _nd_spiral_points(count: int, lo: float, hi: float, ambient: int, planes: int,
                      seed: int, rotation: torch.Tensor):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(count, generator=generator) * (hi - lo) + lo
    theta = .45 + 5.4 * math.pi * u
    radius = .12 + .88 * u
    branches = []
    for label in (0, 1):
        features = []
        for plane in range(planes):
            frequency = plane + 1
            phase = frequency * theta + .37 * plane + label * math.pi
            amplitude = radius * (1 + .08 * torch.sin((plane + 2) * theta)) / math.sqrt(planes)
            features.extend((amplitude * torch.cos(phase), amplitude * torch.sin(phase)))
        core = torch.stack(features, 1)
        padded = torch.zeros((count, ambient))
        padded[:, :2 * planes] = core
        branches.append(padded @ rotation)
    x = torch.cat(branches)
    x = x + .006 * torch.randn(x.shape, generator=generator)
    y = torch.cat((torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long)))
    return x, y


def nd_spiral(planes: int, seed: int = 0, ambient: int = 16, train_fraction: float = .5) -> Task:
    if 2 * planes > ambient:
        raise ValueError("each spiral plane needs two ambient dimensions")
    rotation = _orthogonal(ambient, 700 + seed)
    x, y = _nd_spiral_points(1600, .015, train_fraction, ambient, planes, 800 + seed, rotation)
    xtr, ytr, xva, yva = _stratified_split(x, y, 900 + seed)
    tx, ty = [], []
    for index in range(10):
        lo = train_fraction + (1 - train_fraction) * index / 10
        hi = train_fraction + (1 - train_fraction) * (index + 1) / 10
        a, b = _nd_spiral_points(260, lo, hi, ambient, planes, 2100 + seed * 31 + index, rotation)
        tx.append(a); ty.append(b)
    name = "nd_spiral_low_rank" if planes == 1 else "nd_spiral_high_rank"
    return Task(name, xtr, ytr, xva, yva, tx, ty, 2 * planes)


def hypercube_checker(seed: int = 0, dim: int = 16) -> Task:
    generator = torch.Generator().manual_seed(2600 + seed)
    rotation = _orthogonal(dim, 2700 + seed)
    def sample(count: int, lo: float, hi: float):
        raw = (torch.rand((count * 2, dim), generator=generator) * 2 - 1) * hi
        radius = raw.abs().amax(1)
        raw = raw[(radius >= lo) & (radius < hi)][:count]
        while len(raw) < count:
            extra = (torch.rand((count * 2, dim), generator=generator) * 2 - 1) * hi
            rr = extra.abs().amax(1)
            raw = torch.cat((raw, extra[(rr >= lo) & (rr < hi)]))[:count]
        label = (raw[:, :8] > 0).long().sum(1).remainder(2)
        return raw @ rotation, label
    x, y = sample(4200, 0, .62)
    xtr, ytr, xva, yva = _stratified_split(x, y, 2800 + seed)
    tx, ty = [], []
    for index in range(10):
        a, b = sample(600, .62 + .038 * index, .62 + .038 * (index + 1))
        tx.append(a); ty.append(b)
    return Task("hypercube_checker", xtr, ytr, xva, yva, tx, ty, 8)


TASK_BUILDERS = {
    "spiral_2d": spiral_2d,
    "checkerboard_2d": checkerboard_2d,
    "nd_spiral_low_rank": lambda seed=0: nd_spiral(1, seed),
    "nd_spiral_high_rank": lambda seed=0: nd_spiral(8, seed),
    "hypercube_checker": hypercube_checker,
}
