"""Numerical oracle extracted verbatim from meyer_semismooth_state_jump.py.
Plotting and unrelated scene imports omitted for the Mini CPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))

def grad(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.roll(value, -1, axis=1) - value, np.roll(value, -1, axis=0) - value

def div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return px - np.roll(px, 1, axis=1) + py - np.roll(py, 1, axis=0)

def laplacian_symbol(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    ky = 2.0 * np.pi * np.arange(height) / height
    kx = 2.0 * np.pi * np.arange(width) / width
    return (
        4.0 - 2.0 * np.cos(ky)[:, None] - 2.0 * np.cos(kx)[None, :]
    )

def solve_screened(
    rhs: np.ndarray, c: float, eta: float, symbol: np.ndarray
) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(rhs) / (c + eta * symbol)).real

def project_disk(
    tx: np.ndarray, ty: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(tx, ty)
    scale = np.minimum(1.0, radius / np.maximum(magnitude, 1e-30))
    return scale * tx, scale * ty

def reflected(
    tx: np.ndarray, ty: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    px, py = project_disk(tx, ty, radius)
    return tx - 2.0 * px, ty - 2.0 * py

def project_disk_derivative(
    tx: np.ndarray,
    ty: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Clarke derivative of the Euclidean disk projection."""

    magnitude = np.hypot(tx, ty)
    outside = magnitude > radius
    result_x = hx.copy()
    result_y = hy.copy()
    if np.any(outside):
        inverse = 1.0 / magnitude[outside]
        nx = tx[outside] * inverse
        ny = ty[outside] * inverse
        tangent = -ny * hx[outside] + nx * hy[outside]
        factor = radius * inverse
        result_x[outside] = factor * (-ny) * tangent
        result_y[outside] = factor * nx * tangent
    return result_x, result_y

@dataclass
class State:
    u: np.ndarray
    w: np.ndarray
    tux: np.ndarray
    tuy: np.ndarray
    twx: np.ndarray
    twy: np.ndarray

    def fields(self) -> tuple[np.ndarray, ...]:
        return self.u, self.w, self.tux, self.tuy, self.twx, self.twy

class ReducedMeyerMap:
    def __init__(self, image: np.ndarray, lam: float, mu: float):
        self.image = np.asarray(image, dtype=np.float64)
        self.shape = self.image.shape
        self.count = self.image.size
        self.lam = float(lam)
        self.mu = float(mu)
        self.cu = self.lam
        self.etau = 2.0 * self.lam
        self.cw = 1.0 / self.mu
        self.etaw = 10.0 / self.mu
        self.ru = 1.0 / self.etau
        self.rw = 1.0 / self.etaw
        self.symbol = laplacian_symbol(self.shape)

    def initial(self) -> State:
        u = solve_screened(
            self.cu * self.image, self.cu, self.etau, self.symbol
        )
        w = solve_screened(
            self.cw * (self.image - u), self.cw, self.etaw, self.symbol
        )
        tux, tuy = grad(u)
        twx, twy = grad(w)
        return State(u, w, tux, tuy, twx, twy)

    def step(self, state: State) -> State:
        rux, ruy = reflected(state.tux, state.tuy, self.ru)
        u = solve_screened(
            self.cu * (state.u + state.w) - self.etau * div(rux, ruy),
            self.cu,
            self.etau,
            self.symbol,
        )
        rwx, rwy = reflected(state.twx, state.twy, self.rw)
        w = solve_screened(
            self.cw * (self.image - u) - self.etaw * div(rwx, rwy),
            self.cw,
            self.etaw,
            self.symbol,
        )
        bux, buy = project_disk(state.tux, state.tuy, self.ru)
        bwx, bwy = project_disk(state.twx, state.twy, self.rw)
        gux, guy = grad(u)
        gwx, gwy = grad(w)
        return State(u, w, gux + bux, guy + buy, gwx + bwx, gwy + bwy)

    def tangent(self, state: State, direction: State) -> State:
        pux, puy = project_disk_derivative(
            state.tux, state.tuy, direction.tux, direction.tuy, self.ru
        )
        rux = direction.tux - 2.0 * pux
        ruy = direction.tuy - 2.0 * puy
        du = solve_screened(
            self.cu * (direction.u + direction.w)
            - self.etau * div(rux, ruy),
            self.cu,
            self.etau,
            self.symbol,
        )
        pwx, pwy = project_disk_derivative(
            state.twx, state.twy, direction.twx, direction.twy, self.rw
        )
        rwx = direction.twx - 2.0 * pwx
        rwy = direction.twy - 2.0 * pwy
        dw = solve_screened(
            -self.cw * du - self.etaw * div(rwx, rwy),
            self.cw,
            self.etaw,
            self.symbol,
        )
        gux, guy = grad(du)
        gwx, gwy = grad(dw)
        return State(du, dw, gux + pux, guy + puy, gwx + pwx, gwy + pwy)

    def pack(self, state: State) -> np.ndarray:
        return np.concatenate([field.ravel() for field in state.fields()])

    def unpack(self, vector: np.ndarray) -> State:
        fields = np.asarray(vector, dtype=np.float64).reshape(6, *self.shape)
        return State(*(field.copy() for field in fields))

    def subtract(self, left: State, right: State) -> State:
        return State(*(a - b for a, b in zip(left.fields(), right.fields())))

    def add_scaled(self, state: State, direction: State, alpha: float) -> State:
        return State(*(
            value + float(alpha) * delta
            for value, delta in zip(state.fields(), direction.fields())
        ))

    def retract_dual_graph(self, state: State) -> tuple[State, float]:
        """Project only the post-jump Bregman residuals back to feasibility.

        An exact pass ends with ``t = D primal + b`` and ``|b| <= radius``.
        A tangent polynomial preserves this only to first order.  This
        pointwise graph retraction restores it without a screened solve.
        """

        gux, guy = grad(state.u)
        gwx, gwy = grad(state.w)
        bux, buy = project_disk(
            state.tux - gux, state.tuy - guy, self.ru
        )
        bwx, bwy = project_disk(
            state.twx - gwx, state.twy - gwy, self.rw
        )
        retracted = State(
            state.u, state.w,
            gux + bux, guy + buy,
            gwx + bwx, gwy + bwy,
        )
        correction = rms(np.concatenate([
            (retracted.tux - state.tux).ravel(),
            (retracted.tuy - state.tuy).ravel(),
            (retracted.twx - state.twx).ravel(),
            (retracted.twy - state.twy).ravel(),
        ]))
        return retracted, correction

    def residual(self, state: State) -> State:
        return self.subtract(self.step(state), state)

    def residual_norm(self, state: State) -> float:
        residual = self.pack(self.residual(state))
        return float(np.linalg.norm(residual) / np.sqrt(residual.size))
