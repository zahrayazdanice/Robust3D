"""Controlled, repeatable point-cloud degradation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .config import DegradationConfig


@dataclass(frozen=True, slots=True)
class DegradationReport:
    original_points: int
    retained_after_sampling: int
    removed_for_hole: int
    added_outliers: int
    final_points: int


def degrade_point_cloud(
    cloud: o3d.geometry.PointCloud,
    config: DegradationConfig,
) -> tuple[o3d.geometry.PointCloud, DegradationReport]:
    """Apply sparsity, a contiguous missing cap, Gaussian noise, and uniform outliers.

    Noise magnitude is a fraction of the clean cloud bounding-box diagonal. The
    missing-region parameter removes a connected cap along a random direction,
    unlike independent dropout, so it approximates occlusion or an unscanned view.
    """

    config.validate()
    rng = np.random.default_rng(config.seed)
    points = np.asarray(cloud.points, dtype=float).copy()
    if len(points) == 0:
        raise ValueError("Cannot degrade an empty point cloud")
    n_original = len(points)

    n_keep = max(50, min(n_original, int(round(n_original * config.density))))
    keep = rng.choice(n_original, size=n_keep, replace=False)
    points = points[keep]
    retained = len(points)

    removed_for_hole = 0
    if config.missing_fraction > 0 and len(points) > 50:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-12
        projection = points @ direction
        cutoff = np.quantile(projection, 1.0 - config.missing_fraction)
        hole_mask = projection < cutoff
        removed_for_hole = int((~hole_mask).sum())
        points = points[hole_mask]

    bounds_min = np.min(np.asarray(cloud.points), axis=0)
    bounds_max = np.max(np.asarray(cloud.points), axis=0)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    if config.noise_sigma > 0:
        points += rng.normal(0.0, config.noise_sigma * diagonal, size=points.shape)

    n_outliers = int(round(len(points) * config.outlier_fraction))
    if n_outliers:
        pad = 0.15 * np.maximum(bounds_max - bounds_min, diagonal / 20)
        outliers = rng.uniform(bounds_min - pad, bounds_max + pad, size=(n_outliers, 3))
        points = np.vstack((points, outliers))

    order = rng.permutation(len(points))
    result = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[order]))
    report = DegradationReport(
        original_points=n_original,
        retained_after_sampling=retained,
        removed_for_hole=removed_for_hole,
        added_outliers=n_outliers,
        final_points=len(points),
    )
    return result, report
