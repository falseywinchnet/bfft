from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.pi / math.sqrt(3.0) #logistic CDF matched scale beats gelu and is less expensive

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)


def lelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid((math.pi / math.sqrt(3.0)) * x)


class BenchmarkModel(nn.Module):
    contextual = False

    def auxiliary_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(())


class LinearBaseline(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__(); self.linear = nn.Linear(input_dim, 2)

    def forward(self, x, context_x=None, context_y=None):
        return self.linear(x)


class DenseLELU(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)
        self.output = nn.Linear(width, 2)
        self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


def _circle_index(side: int):
    fy = torch.fft.fftfreq(side) * side
    fx = torch.fft.fftfreq(side) * side
    raw = torch.round(torch.sqrt(fy[:, None].square() + fx[None, :].square())).long()
    unique, inverse = torch.unique(raw.flatten(), return_inverse=True)
    return inverse.reshape(side, side), len(unique)


class StaticCircleLinear(nn.Module):
    def __init__(self, side: int, cin: int, cout: int):
        super().__init__(); self.side = side; self.cin = cin; self.cout = cout
        ring, rings = _circle_index(side); self.register_buffer("ring", ring); self.rings = rings
        self.delta = nn.Parameter(torch.zeros(rings, cout, cin))
        self.bias = nn.Parameter(torch.zeros(cout, 1, 1))

    def forward(self, flat):
        batch = len(flat); x = flat.view(batch, self.cin, self.side, self.side)
        spectrum = torch.fft.fft2(x, norm="ortho")
        base = x.new_zeros(self.cout, self.cin)
        for output in range(self.cout): base[output, output % self.cin] = 1
        coefficient = base[None] + .6 * torch.tanh(self.delta)
        gain = coefficient[self.ring.flatten()].view(self.side, self.side, self.cout, self.cin)
        gain = gain.permute(2, 3, 0, 1)
        transformed = (gain[None] * spectrum[:, None]).sum(2)
        return (torch.fft.ifft2(transformed, norm="ortho").real + self.bias).reshape(batch, -1)


class LivingCircleLinear(nn.Module):
    def __init__(self, side: int, cin: int, cout: int, response_width: int = 16):
        super().__init__(); self.side = side; self.cin = cin; self.cout = cout
        ring, rings = _circle_index(side); self.register_buffer("ring", ring); self.rings = rings
        self.response = nn.Sequential(nn.Linear(2, response_width), LELU(), nn.Linear(response_width, cout * cin))
        self.bias = nn.Parameter(torch.zeros(cout, 1, 1))
        nn.init.normal_(self.response[-1].weight, std=.025); nn.init.zeros_(self.response[-1].bias)

    def _ring_mean(self, value):
        batch, channels = value.shape[:2]; rid = self.ring.flatten()
        out = value.new_zeros(batch, channels, self.rings)
        out.scatter_add_(2, rid[None, None].expand(batch, channels, -1), value.reshape(batch, channels, -1))
        count = torch.bincount(rid, minlength=self.rings).to(value.dtype)
        return out / count[None, None]

    def forward(self, flat):
        batch = len(flat); x = flat.view(batch, self.cin, self.side, self.side)
        spectrum = torch.fft.fft2(x, norm="ortho")
        power = self._ring_mean(spectrum.abs().square()).mean(1)
        log_power = torch.log(power + 1e-6)
        log_power = (log_power - log_power.mean(1, keepdim=True)) / (log_power.std(1, keepdim=True) + 1e-5)
        # A pooled phase-coupling witness. It is relational but gives no direction its own parameters.
        shifted = torch.roll(spectrum.mean(1), shifts=(-1, -1), dims=(-2, -1))
        phase = spectrum.mean(1) * shifted.conj()
        coherence = self._ring_mean((phase / (phase.abs() + 1e-6))[:, None]).abs()[:, 0]
        stats = torch.stack((log_power, coherence), -1)
        coefficient = torch.tanh(self.response(stats)).view(batch, self.rings, self.cout, self.cin)
        base = x.new_zeros(self.cout, self.cin)
        for output in range(self.cout): base[output, output % self.cin] = 1
        coefficient = base[None, None] + .6 * coefficient
        gain = coefficient[:, self.ring.flatten()].view(batch, self.side, self.side, self.cout, self.cin)
        gain = gain.permute(0, 3, 4, 1, 2)
        transformed = (gain * spectrum[:, None]).sum(2)
        return (torch.fft.ifft2(transformed, norm="ortho").real + self.bias).reshape(batch, -1)


class CircleNet(BenchmarkModel):
    def __init__(self, input_dim: int, width: int, living: bool):
        super().__init__(); side = int(round(math.sqrt(width)))
        if side * side != width: raise ValueError("circle models require a square width")
        layer = LivingCircleLinear if living else StaticCircleLinear
        self.embed = nn.Linear(input_dim, width); self.up = layer(side, 1, 2)
        self.down = layer(side, 2, 1); self.output = nn.Linear(width, 2); self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class LivingGraphLinear(nn.Module):
    def __init__(self, nodes: int, cin: int, cout: int, order: int = 3):
        super().__init__(); self.nodes = nodes; self.cin = cin; self.cout = cout; self.order = order
        self.response = nn.Sequential(nn.Linear(3, 14), LELU(), nn.Linear(14, cout * cin))
        self.bias = nn.Parameter(torch.zeros(cout, 1)); nn.init.zeros_(self.response[-1].bias)

    def forward(self, flat):
        batch = len(flat); x = flat.view(batch, self.cin, self.nodes); points = x.transpose(1, 2)
        centered = points - points.mean(1, keepdim=True)
        delta = centered[:, :, None, :] - centered[:, None, :, :]
        covariance = torch.einsum("bnc,bnd->bcd", centered, centered) / max(1, self.nodes - 1)
        eye = torch.eye(self.cin, device=x.device, dtype=x.dtype)[None]
        covariance = covariance + (.05 * covariance.diagonal(dim1=-2, dim2=-1).mean(1) + 1e-3)[:, None, None] * eye
        inverse = torch.linalg.inv(covariance)
        distance = torch.einsum("bijn,bnm,bijm->bij", delta, inverse, delta)
        adjacency = torch.softmax(-distance / distance.detach().mean((1, 2), keepdim=True).clamp_min(1e-4), 2)
        states = [x]
        for _ in range(1, self.order): states.append(torch.einsum("bij,bcj->bci", adjacency, states[-1]))
        stack = torch.stack(states, 1)
        energy = stack.square().mean((2, 3)).clamp_min(1e-7)
        alignment = (stack * x[:, None]).mean((2, 3)) / torch.sqrt(energy * x.square().mean((1, 2))[:, None] + 1e-7)
        roughness = torch.stack([(s - torch.einsum("bij,bcj->bci", adjacency, s)).square().mean((1, 2)) for s in states], 1)
        stats = torch.stack((torch.log(energy), alignment, torch.log1p(roughness / energy)), -1)
        raw = torch.tanh(self.response(stats)).view(batch, self.order, self.cout, self.cin)
        base = x.new_zeros(self.cout, self.cin)
        for output in range(self.cout): base[output, output % self.cin] = 1
        scales = x.new_full((self.order,), .25); scales[0] = .45
        identity = x.new_zeros(self.order, self.cout, self.cin); identity[0] = base
        coefficient = raw * scales[None, :, None, None] + identity[None]
        return (torch.einsum("bmoi,bmin->bon", coefficient, stack) + self.bias[None]).reshape(batch, -1)


class LivingGraphNet(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__(); nodes = int(round(math.sqrt(width))); channels = width // nodes
        if nodes * channels != width: raise ValueError("graph width must factor through rounded sqrt")
        self.embed = nn.Linear(input_dim, width)
        self.up = LivingGraphLinear(nodes, channels, 2 * channels)
        self.down = LivingGraphLinear(nodes, 2 * channels, channels)
        self.output = nn.Linear(width, 2); self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class LearnedSubspaceLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, slices: int = 3, rank: int = 4):
        super().__init__(); self.slices = slices
        self.base = nn.Linear(n_in, n_out)
        self.down = nn.ModuleList([nn.Linear(n_in, rank, bias=False) for _ in range(slices)])
        self.up = nn.ModuleList([nn.Linear(rank, n_out, bias=False) for _ in range(slices)])
        self.response = nn.Sequential(nn.Linear(slices * (slices + 1) // 2, 16), LELU(), nn.Linear(16, slices))
        nn.init.zeros_(self.response[-1].weight); nn.init.zeros_(self.response[-1].bias)

    def forward(self, x):
        q = torch.stack([layer(x) for layer in self.down], 1); qn = F.normalize(q, dim=-1, eps=1e-7)
        gram = qn @ qn.transpose(1, 2); tri = torch.triu_indices(self.slices, self.slices, device=x.device)
        alpha = torch.tanh(self.response(gram[:, tri[0], tri[1]]))
        branches = torch.stack([layer(q[:, index]) for index, layer in enumerate(self.up)], 1)
        return self.base(x) + (alpha[:, :, None] * branches).sum(1) / math.sqrt(self.slices)


class SubspaceNet(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__(); self.embed = nn.Linear(input_dim, width)
        self.up = LearnedSubspaceLinear(width, 2 * width); self.down = LearnedSubspaceLinear(2 * width, width)
        self.output = nn.Linear(width, 2); self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class AtlasLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, charts: int = 6, rank: int = 3):
        super().__init__(); self.charts = charts; self.rank = rank
        self.base = nn.Linear(n_in, n_out); self.coordinates = nn.Linear(n_in, charts * rank, bias=False)
        self.basis = nn.Parameter(torch.randn(charts, rank, n_out) * .08); self.coefficients = nn.Linear(n_in, charts)
        self.temperature = nn.Parameter(torch.tensor(-.5)); self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        q = self.coordinates(x).view(len(x), self.charts, self.rank)
        directions = F.normalize(torch.einsum("bcr,cro->bco", q, self.basis), dim=-1, eps=1e-7)
        coefficient = torch.tanh(self.coefficients(x)); distance = (1 - directions @ directions.transpose(1, 2)).clamp_min(0)
        flow = torch.softmax(-distance / self.temperature.exp().clamp(.05, 5), -1)
        transported = torch.einsum("bij,bj->bi", flow, coefficient)
        correction = torch.einsum("bc,bco->bo", transported, directions) / math.sqrt(self.charts)
        return self.base(x) + torch.tanh(self.scale) * correction


class AtlasNet(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__(); self.embed = nn.Linear(input_dim, width); self.up = AtlasLinear(width, 2 * width)
        self.down = AtlasLinear(2 * width, width); self.output = nn.Linear(width, 2); self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class SoftEikonalLinear(nn.Module):
    """Dense map plus continuous metric allocation over a fixed direction pool."""
    def __init__(self, n_in: int, n_out: int, directions: int = 12, rank: int = 4):
        super().__init__(); self.directions = directions; self.rank = rank
        self.base = nn.Linear(n_in, n_out); self.metric = nn.Linear(n_in, rank * rank)
        generator = torch.Generator().manual_seed(9157 + n_in + n_out)
        primitive = torch.randn(directions, rank, n_in, generator=generator)
        self.register_buffer("primitive", F.normalize(primitive, dim=-1))
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.response = nn.Sequential(nn.Linear(4, 12), LELU(), nn.Linear(12, 1))
        self.scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x):
        batch = len(x); factor = self.metric(x).view(batch, self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.primitive, x)
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        norm = projected.square().mean(-1); alignment = projected.mean(-1)
        entropy_proxy = torch.log1p(projected.abs().mean(-1))
        stats = torch.stack((torch.log1p(cost), torch.log1p(norm), alignment, entropy_proxy), -1)
        response = self.response(stats).squeeze(-1); weight = torch.softmax(response - cost / (cost.mean(1, keepdim=True) + 1e-5), 1)
        pooled = torch.einsum("bd,bdr->br", weight, projected)
        return self.base(x) + F.softplus(self.scale) * (pooled @ self.shared)


class SoftEikonalNet(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__(); self.embed = nn.Linear(input_dim, width); self.up = SoftEikonalLinear(width, 2 * width)
        self.down = SoftEikonalLinear(2 * width, width); self.output = nn.Linear(width, 2); self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class JetLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, latent: int = 6, charts: int = 6, branch_rank: int = 2):
        super().__init__(); self.n_in = n_in; self.latent = latent; self.charts = charts
        self.base = nn.Linear(n_in, n_out)
        self.representation = nn.Sequential(nn.Linear(n_in, latent), LELU(), nn.Linear(latent, latent))
        self.coefficients = nn.Sequential(nn.Linear(latent, 2 * latent), LELU(), nn.Linear(2 * latent, charts))
        self.connection = nn.Sequential(nn.Linear(n_in, 2 * latent), LELU(), nn.Linear(2 * latent, latent * n_in))
        self.left = nn.Parameter(torch.randn(charts, n_out, branch_rank) * .08)
        self.right = nn.Parameter(torch.randn(charts, branch_rank, n_in) * .08)
        self.scale = nn.Parameter(torch.tensor(-1.5)); self.last_state = None

    def forward(self, x):
        z = self.representation(x); coefficient = self.coefficients(z)
        branch = torch.einsum("cri,bi->bcr", self.right, x)
        correction = torch.einsum("bc,cor,bcr->bo", coefficient, self.left, branch)
        connection = self.connection(x).view(-1, self.latent, self.n_in)
        self.last_state = (x, z, connection)
        return self.base(x) + F.softplus(self.scale) * correction / math.sqrt(self.charts)


class JetNet(BenchmarkModel):
    def __init__(self, input_dim: int, width: int):
        super().__init__(); self.embed = nn.Linear(input_dim, width); self.up = JetLinear(width, 2 * width)
        self.down = JetLinear(2 * width, width); self.output = nn.Linear(width, 2); self.activation = LELU()

    def forward(self, x, context_x=None, context_y=None):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))

    def auxiliary_loss(self, x):
        if len(x) < 4: return x.new_zeros(())
        embedded = self.embed(x); _ = self.up(embedded); source, latent, connection = self.up.last_state
        neighbor = torch.roll(torch.arange(len(x), device=x.device), 1)
        dx = source[neighbor] - source; dz = latent[neighbor] - latent
        prediction = torch.einsum("bli,bi->bl", connection, dx)
        scale = dz.detach().square().mean().sqrt().clamp_min(.05)
        return F.smooth_l1_loss(prediction / scale, dz / scale)


class MatrixExponentialNet(BenchmarkModel):
    """Small real matrix-exponential flow baseline, independently implemented."""
    def __init__(self, input_dim: int, width: int):
        super().__init__(); groups = max(2, width // 4); self.groups = groups; self.order = 3
        hidden = groups * self.order
        self.embed = nn.Linear(input_dim, hidden)
        self.generator = nn.Linear(hidden, groups * self.order * self.order)
        self.output = nn.Linear(hidden, 2); self.scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, context_x=None, context_y=None):
        h = self.embed(x).view(len(x), self.groups, self.order)
        generator = self.generator(h.flatten(1)).view(len(x), self.groups, self.order, self.order)
        generator = torch.tanh(generator) / math.sqrt(self.order)
        flowed = torch.einsum("bgij,bgj->bgi", torch.matrix_exp(generator), h)
        return self.output(lelu(F.softplus(self.scale) * flowed.flatten(1)))


class AssociativeShellNet(BenchmarkModel):
    contextual = True
    def __init__(self, input_dim: int, width: int, shells: int = 2):
        super().__init__(); self.shells = shells
        self.key = nn.Sequential(nn.Linear(input_dim, width), LELU(), nn.Linear(width, width))
        self.value = nn.Sequential(nn.Linear(input_dim + 1, width), LELU(), nn.Linear(width, width))
        self.query = nn.Sequential(nn.Linear(input_dim, width), LELU(), nn.Linear(width, width))
        self.output = nn.Linear(width, 2); self.shell_scale = nn.Parameter(torch.full((shells,), -1.0))

    def forward(self, x, context_x=None, context_y=None):
        if context_x is None or context_y is None: raise ValueError("associative shells require context")
        signed = 2 * context_y.to(x.dtype) - 1
        key = self.key(context_x); value = self.value(torch.cat((context_x, signed[:, None]), 1))
        memory = torch.einsum("ni,nj->ij", value, key) / len(context_x)
        h = self.query(x)
        for index in range(self.shells): h = h + F.softplus(self.shell_scale[index]) * lelu(h @ memory.T)
        return self.output(h)


def circular_basis(theta: torch.Tensor, harmonics: int):
    values = [torch.ones_like(theta)]
    for order in range(1, harmonics + 1): values.extend((torch.cos(order * theta), torch.sin(order * theta)))
    return torch.stack(values, -1)


class BanachSieveNet(BenchmarkModel):
    contextual = True
    def __init__(self, input_dim: int, width: int, views: int = 12, harmonics: int = 3, layers: int = 2):
        super().__init__(); self.views = views; self.harmonics = harmonics; self.layers = layers
        modes = 1 + 2 * harmonics; context_width = max(12, width // 2)
        self.embed = nn.Linear(input_dim, width)
        self.context_point = nn.Sequential(nn.Linear(input_dim + 1, context_width), LELU(), nn.Linear(context_width, context_width))
        self.potential = nn.Linear(2 * context_width, layers * modes)
        self.operator = nn.Parameter(torch.randn(layers, modes, width, width + 1) / math.sqrt(width * modes))
        self.layer_scale = nn.Parameter(torch.full((layers,), -1.0)); self.output = nn.Linear(width, 2)
        phase = 2 * math.pi * (torch.arange(views, dtype=torch.float32) + .5) / views
        self.register_buffer("phase", phase)

    def forward(self, x, context_x=None, context_y=None):
        if context_x is None or context_y is None: raise ValueError("Banach sieve requires context")
        signed = 2 * context_y.to(x.dtype) - 1
        points = self.context_point(torch.cat((context_x, signed[:, None]), 1))
        summary = torch.cat((points.mean(0), points.std(0, unbiased=False)))
        coefficient = self.potential(summary).view(self.layers, -1); basis = circular_basis(self.phase, self.harmonics)
        h = self.embed(x)
        for layer in range(self.layers):
            weight = torch.softmax(3 * torch.tanh(basis @ coefficient[layer]), 0)
            operators = torch.einsum("vm,mij->vij", basis, self.operator[layer])
            induced = torch.einsum("v,vij->ij", weight, operators)
            homogeneous = torch.cat((h, torch.ones_like(h[:, :1])), 1)
            h = h + F.softplus(self.layer_scale[layer]) * lelu(homogeneous @ induced.T)
        return self.output(h)


class ProjectiveRoleNet(BenchmarkModel):
    contextual = True
    def __init__(self, input_dim: int, width: int, roles: int = 8, rank: int = 3):
        super().__init__(); self.roles = roles; self.rank = rank
        self.encode = nn.Sequential(nn.Linear(input_dim, width), LELU(), nn.Linear(width, width))
        self.generators = nn.Parameter(torch.randn(roles, width, rank) / math.sqrt(width))
        self.clock = nn.Linear(width, 1); self.temperature = nn.Parameter(torch.tensor(-.7))
        self.steps = nn.Sequential(nn.Linear(2, 12), LELU(), nn.Linear(12, 3))

    def _roles(self, z):
        projection = torch.einsum("bw,rwk->brk", z, self.generators).square().sum(-1)
        return torch.softmax(projection / self.temperature.exp().clamp(.08, 5), 1)

    def forward(self, x, context_x=None, context_y=None):
        if context_x is None or context_y is None: raise ValueError("projective roles require context")
        context_z = self.encode(context_x); query_z = self.encode(x)
        context_roles = self._roles(context_z); query_roles = self._roles(query_z)
        signed = 2 * context_y.to(x.dtype) - 1
        role_value = (context_roles * signed[:, None]).sum(0) / context_roles.sum(0).clamp_min(1e-5)
        # A learned projective clock continuously mixes zero, one, and two role-transition steps.
        transition = context_roles.T @ context_roles
        transition = transition / transition.sum(1, keepdim=True).clamp_min(1e-5)
        nearest = torch.cdist(query_z, context_z).min(1).values
        clock_delta = self.clock(query_z).squeeze(1) - self.clock(context_z).mean()
        mixture = torch.softmax(self.steps(torch.stack((torch.log1p(nearest), clock_delta.abs()), 1)), 1)
        paths = torch.stack((query_roles, query_roles @ transition, query_roles @ transition @ transition), 1)
        transported = (mixture[:, :, None] * paths).sum(1)
        score = transported @ role_value
        return torch.stack((-score, score), 1)


MODEL_BUILDERS = {
    "linear": LinearBaseline,
    "dense_lelu": DenseLELU,
    "static_fourier_circle": lambda d, w: CircleNet(d, w, False),
    "living_fourier_circle": lambda d, w: CircleNet(d, w, True),
    "living_metric_graph": LivingGraphNet,
    "learned_subspace_gram": SubspaceNet,
    "hypersphere_atlas": AtlasNet,
    "soft_eikonal_pool": SoftEikonalNet,
    "jet_transport": JetNet,
    "matrix_exponential": MatrixExponentialNet,
    "associative_shells": AssociativeShellNet,
    "banach_eikonal_sieve": BanachSieveNet,
    "projective_roles": ProjectiveRoleNet,
}


def make_model(name: str, input_dim: int, width: int) -> BenchmarkModel:
    return MODEL_BUILDERS[name](input_dim, width)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
