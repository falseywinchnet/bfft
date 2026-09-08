"""Native residual march over geometric retained-rank transport blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .native_retained import NativeApplyStats, NativeRetainedPlan
from .transport import Array, HierarchicalTransportField


@dataclass(frozen=True)
class RetainedDepthWork:
    depth: int
    active_nodes: int
    block_applications: int
    gathered_source_coefficients: int
    scattered_receiver_coefficients: int


@dataclass(frozen=True)
class NativeDiffuseSolveResult:
    outgoing: Array
    incident: Array
    absorbed: Array
    escaped: Array
    residual: Array
    depth_count: int
    depth_work: tuple[RetainedDepthWork, ...]
    conservation_error: Array
    backend: str

    @property
    def block_applications(self) -> int:
        return sum(work.block_applications for work in self.depth_work)

    @property
    def gathered_source_coefficients(self) -> int:
        return sum(work.gathered_source_coefficients for work in self.depth_work)

    @property
    def scattered_receiver_coefficients(self) -> int:
        return sum(work.scattered_receiver_coefficients for work in self.depth_work)


class NativeDiffuseTransportPlan:
    """A fixed geometric field whose diminishing residual is marched natively."""

    def __init__(self, field: HierarchicalTransportField) -> None:
        self.field = field
        self.native_plan = NativeRetainedPlan.from_transport_field(field)
        # RGB is three independent unpolarized modes in this diffuse boundary
        # solve.  The same kernel accepts the richer child-volume mode state.
        self.mode_plan = self.native_plan.bind_modes(
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

    @property
    def backend(self) -> str:
        return self.native_plan.backend

    def close(self) -> None:
        self.mode_plan.close()
        self.native_plan.close()

    def __enter__(self) -> "NativeDiffuseTransportPlan":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def solve(
        self,
        emission: Array,
        albedo: Array,
        *,
        relative_tolerance: float = 1.0e-10,
        absolute_tolerance: float = 1.0e-14,
        maximum_depth: int = 10_000,
    ) -> NativeDiffuseSolveResult:
        """March all non-recursive bounce depths through the retained field."""
        source = np.asarray(emission, dtype=np.float64)
        reflectance = np.asarray(albedo, dtype=np.float64)
        if source.ndim == 1:
            source = source[:, None]
        if reflectance.ndim == 1:
            reflectance = reflectance[:, None]
        if source.ndim != 2 or source.shape != (self.field.node_count, 3):
            raise ValueError("native diffuse emission must have shape Nx3")
        if reflectance.shape not in (source.shape, (self.field.node_count, 1)):
            raise ValueError("native diffuse albedo must broadcast over Nx3")
        if np.any(source < 0.0) or not np.all(np.isfinite(source)):
            raise ValueError("native diffuse emission must be finite and nonnegative")
        if (
            np.any((reflectance < 0.0) | (reflectance >= 1.0))
            or not np.all(np.isfinite(reflectance))
        ):
            raise ValueError("native diffuse albedo must be finite and lie in [0, 1)")
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if relative < 0.0 or absolute < 0.0 or maximum_depth < 1:
            raise ValueError("invalid native diffuse convergence configuration")

        # The C++ kernel is mode-major so each mode's node vector is contiguous
        # and the retained block gather/scatter can use packed SIMD loads.
        frontier = np.ascontiguousarray(source.T)
        outgoing = frontier.copy()
        incident = np.zeros_like(frontier)
        absorbed = np.zeros_like(frontier)
        escaped = np.zeros_like(frontier)
        reflectance_modes = np.ascontiguousarray(reflectance.T)
        if reflectance_modes.shape[0] == 1:
            reflectance_modes = np.broadcast_to(
                reflectance_modes, frontier.shape
            ).copy()
        threshold = absolute + relative * np.max(frontier, axis=1)
        depth_work: list[RetainedDepthWork] = []
        depth_count = 0

        for depth in range(maximum_depth):
            active = np.any(frontier > threshold[:, None], axis=0)
            if not np.any(active):
                break
            marching = frontier * active[None, :]
            frontier -= marching
            escaped += (
                self.field.escape_fraction[None, :] * marching
            )
            received, stats = self.mode_plan.apply(marching, active)
            incident += received
            loss = (1.0 - reflectance_modes) * received
            reflected = reflectance_modes * received
            absorbed += loss
            outgoing += reflected
            frontier += reflected
            depth_count = depth + 1
            depth_work.append(_depth_work(depth, active, stats))
        else:
            raise RuntimeError("native retained residual did not converge")

        initial_energy = np.sum(source, axis=0)
        accounted = (
            np.sum(absorbed, axis=1)
            + np.sum(escaped, axis=1)
            + np.sum(frontier, axis=1)
        )
        return NativeDiffuseSolveResult(
            outgoing=outgoing.T.copy(),
            incident=incident.T.copy(),
            absorbed=absorbed.T.copy(),
            escaped=escaped.T.copy(),
            residual=frontier.T.copy(),
            depth_count=depth_count,
            depth_work=tuple(depth_work),
            conservation_error=accounted - initial_energy,
            backend=self.backend,
        )


def _depth_work(
    depth: int,
    active: Array,
    stats: NativeApplyStats,
) -> RetainedDepthWork:
    return RetainedDepthWork(
        depth=depth,
        active_nodes=int(np.count_nonzero(active)),
        block_applications=stats.block_applications,
        gathered_source_coefficients=stats.gathered_source_coefficients,
        scattered_receiver_coefficients=stats.scattered_receiver_coefficients,
    )
