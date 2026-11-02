"""Baseline and density-adaptive geometric preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .config import PreprocessConfig


@dataclass(frozen=True, slots=True)
class PreprocessReport:
    mode: str
    input_points: int
    output_points: int
    rejected_outliers: int
    estimated_spacing: float
    normal_radius: float
    voxel_size: float


def estimate_spacing(points: np.ndarray, neighbours: int = 6) -> float:
    """Robust median k-NN spacing estimate."""

    if len(points) < 3:
        raise ValueError("At least three points are required to estimate spacing")
    k = min(max(2, neighbours), len(points) - 1)
    distances, _ = cKDTree(points).query(points, k=k + 1, workers=-1)
    local = np.mean(distances[:, 1:], axis=1)
    spacing = float(np.median(local[np.isfinite(local)]))
    if not np.isfinite(spacing) or spacing <= 1e-12:
        extent = np.ptp(points, axis=0)
        spacing = float(np.linalg.norm(extent)) / max(np.cbrt(len(points)), 1.0)
    return max(spacing, 1e-6)


def _estimate_and_orient_normals(
    cloud: o3d.geometry.PointCloud, radius: float, knn: int
) -> None:
    max_nn = min(max(16, knn * 2), max(16, len(cloud.points)))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
        fast_normal_computation=False,
    )
    cloud.normalize_normals()
    if len(cloud.points) >= 12:
        try:
            cloud.orient_normals_consistent_tangent_plane(min(knn, len(cloud.points) - 1))
        except RuntimeError:
            cloud.orient_normals_towards_camera_location(np.asarray([0.0, 0.0, 3.0]))


def _robust_mls_projection(points: np.ndarray, k: int, strength: float) -> np.ndarray:
    """One robust moving-least-squares step onto local PCA tangent planes."""

    if strength <= 0 or len(points) < k + 1:
        return points.copy()
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(k + 1, len(points)), workers=-1)
    output = points.copy()
    eps = 1e-12
    for i in range(len(points)):
        neighbourhood = points[indices[i, 1:]]
        d = distances[i, 1:]
        bandwidth = max(float(np.median(d)), eps)
        weights = np.exp(-0.5 * (d / bandwidth) ** 2)
        weights /= weights.sum() + eps
        centroid = np.sum(neighbourhood * weights[:, None], axis=0)
        centered = neighbourhood - centroid
        covariance = (centered * weights[:, None]).T @ centered
        normal = np.linalg.eigh(covariance)[1][:, 0]
        signed_distance = float(np.dot(points[i] - centroid, normal))
        signed_distance = np.clip(signed_distance, -2.5 * bandwidth, 2.5 * bandwidth)
        output[i] = points[i] - strength * signed_distance * normal
    return output


def preprocess_point_cloud(
    cloud: o3d.geometry.PointCloud,
    config: PreprocessConfig,
) -> tuple[o3d.geometry.PointCloud, PreprocessReport]:
    """Prepare a cloud for reconstruction.

    The proposed adaptive mode derives every distance threshold from median local
    spacing, rejects points by a median-absolute-deviation rule, equalizes dense
    cells with a spacing-scaled voxel grid, and performs one conservative MLS pass.
    """

    config.validate()
    points = np.asarray(cloud.points, dtype=float)
    if len(points) < 50:
        raise ValueError("At least 50 input points are required")
    input_count = len(points)
    initial_spacing = estimate_spacing(points)
    rejected = 0
    voxel_size = 0.0

    if config.mode == "adaptive":
        k = min(config.knn, len(points) - 1)
        distances, _ = cKDTree(points).query(points, k=k + 1, workers=-1)
        local_scale = np.mean(distances[:, 1:], axis=1)
        median = float(np.median(local_scale))
        mad = float(np.median(np.abs(local_scale - median)))
        robust_sigma = max(1.4826 * mad, initial_spacing * 0.05)
        mask = local_scale <= median + config.outlier_mad_z * robust_sigma
        if int(mask.sum()) >= max(50, int(0.55 * len(points))):
            points = points[mask]
            rejected = int((~mask).sum())

        spacing_after_filter = estimate_spacing(points)
        voxel_size = config.voxel_factor * spacing_after_filter
        working = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if voxel_size > 1e-6:
            down = working.voxel_down_sample(voxel_size)
            if len(down.points) >= 50:
                points = np.asarray(down.points).copy()
        points = _robust_mls_projection(points, config.knn, config.smooth_strength)

    processed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    spacing = estimate_spacing(points)
    radius_factor = config.normal_radius_factor if config.mode == "adaptive" else 2.75
    normal_radius = max(radius_factor * spacing, 1e-5)
    _estimate_and_orient_normals(processed, normal_radius, config.knn)

    report = PreprocessReport(
        mode=config.mode,
        input_points=input_count,
        output_points=len(processed.points),
        rejected_outliers=rejected,
        estimated_spacing=spacing,
        normal_radius=normal_radius,
        voxel_size=voxel_size,
    )
    return processed, report
