"""Executable bounds from the energy-weighted subdivision theorem."""

from __future__ import annotations

import math

import numpy as np


def lambertian_kernel(
    source_position: np.ndarray,
    source_normal: np.ndarray,
    receiver_position: np.ndarray,
    receiver_normal: np.ndarray,
) -> float:
    direction = np.asarray(receiver_position) - np.asarray(source_position)
    distance = float(np.linalg.norm(direction))
    if distance <= 0.0:
        return 0.0
    omega = direction / distance
    source_cosine = max(float(np.dot(source_normal, omega)), 0.0)
    receiver_cosine = max(float(np.dot(receiver_normal, -omega)), 0.0)
    return source_cosine * receiver_cosine / (math.pi * distance * distance)


def lambertian_centroid_error_bound(
    *,
    centroid_separation: float,
    source_radius: float,
    receiver_radius: float,
    source_normal_deviation: float,
    receiver_normal_deviation: float,
) -> float:
    separation = float(centroid_separation)
    combined_radius = float(source_radius) + float(receiver_radius)
    distance_floor = separation - combined_radius
    if distance_floor <= 0.0:
        return math.inf
    normal_deviation = (
        float(source_normal_deviation) + float(receiver_normal_deviation)
    )
    if min(
        source_radius,
        receiver_radius,
        source_normal_deviation,
        receiver_normal_deviation,
    ) < 0.0:
        raise ValueError("subdivision radii and deviations must be nonnegative")
    angular = normal_deviation + 4.0 * combined_radius / distance_floor
    radial = 2.0 * combined_radius / (distance_floor ** 3)
    return (angular / (distance_floor ** 2) + radial) / math.pi


def deliverable_energy_error_bound(
    *,
    source_energy: float,
    receiver_area: float,
    centroid_separation: float,
    source_radius: float,
    receiver_radius: float,
    source_normal_deviation: float,
    receiver_normal_deviation: float,
) -> float:
    if source_energy < 0.0 or receiver_area < 0.0:
        raise ValueError("energy and area must be nonnegative")
    return source_energy * receiver_area * lambertian_centroid_error_bound(
        centroid_separation=centroid_separation,
        source_radius=source_radius,
        receiver_radius=receiver_radius,
        source_normal_deviation=source_normal_deviation,
        receiver_normal_deviation=receiver_normal_deviation,
    )


def display_subdivision_error_bound(
    *,
    surface_radius: float,
    radiance_lipschitz: float,
    display_derivative_bound: float,
) -> float:
    if min(surface_radius, radiance_lipschitz, display_derivative_bound) < 0.0:
        raise ValueError("display subdivision quantities must be nonnegative")
    return surface_radius * radiance_lipschitz * display_derivative_bound
