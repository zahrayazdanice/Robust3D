"""Wrappers for three classical Open3D reconstruction algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import open3d as o3d

from .config import ReconstructionConfig
from .preprocessing import estimate_spacing


@dataclass(frozen=True, slots=True)
class ReconstructionReport:
    algorithm: str
    seconds: float
    vertices: int
    triangles: int
    spacing: float
    effective_parameter: str


def _clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    keep_largest_components: bool,
) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if keep_largest_components and len(mesh.triangles) > 20:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels_np = np.asarray(labels)
        counts_np = np.asarray(counts)
        if len(counts_np):
            threshold = max(8, int(counts_np.max() * 0.01))
            remove = counts_np[labels_np] < threshold
            mesh.remove_triangles_by_mask(remove)
            mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
    return mesh


def reconstruct_surface(
    cloud: o3d.geometry.PointCloud,
    config: ReconstructionConfig,
) -> tuple[o3d.geometry.TriangleMesh, ReconstructionReport]:
    """Reconstruct a surface using Poisson, BPA, or alpha shapes."""

    config.validate()
    if len(cloud.points) < 50:
        raise ValueError("At least 50 points are required for reconstruction")
    if not cloud.has_normals():
        raise ValueError("Input cloud must have oriented normals")
    spacing = estimate_spacing(np.asarray(cloud.points))
    started = perf_counter()

    if config.algorithm == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=config.poisson_depth, width=0, scale=1.08, linear_fit=False, n_threads=-1
        )
        density = np.asarray(densities)
        if len(density) and config.poisson_density_quantile > 0:
            cutoff = np.quantile(density, config.poisson_density_quantile)
            mesh.remove_vertices_by_mask(density < cutoff)
        bbox = cloud.get_axis_aligned_bounding_box()
        extent = np.asarray(bbox.get_extent())
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.asarray(bbox.min_bound) - 0.04 * extent,
            np.asarray(bbox.max_bound) + 0.04 * extent,
        )
        mesh = mesh.crop(bbox)
        parameter = f"depth={config.poisson_depth}, trim={config.poisson_density_quantile:.3f}"
    elif config.algorithm == "bpa":
        radii = [spacing * factor for factor in config.bpa_radius_factors]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector(radii)
        )
        parameter = "r/spacing=" + ",".join(f"{x:g}" for x in config.bpa_radius_factors)
    else:
        alpha = spacing * config.alpha_factor
        tetra, point_map = o3d.geometry.TetraMesh.create_from_point_cloud(cloud)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            cloud, alpha, tetra, point_map
        )
        parameter = f"alpha/spacing={config.alpha_factor:g}"

    mesh = _clean_mesh(mesh, config.keep_largest_components)
    elapsed = perf_counter() - started
    if len(mesh.triangles) == 0:
        raise RuntimeError(
            f"{config.algorithm} produced an empty mesh; try denser input or a larger scale parameter"
        )
    report = ReconstructionReport(
        algorithm=config.algorithm,
        seconds=elapsed,
        vertices=len(mesh.vertices),
        triangles=len(mesh.triangles),
        spacing=spacing,
        effective_parameter=parameter,
    )
    return mesh, report
