"""Reference geometry and deterministic surface sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    label: str
    source: str
    geometry: str


DATASETS: dict[str, DatasetInfo] = {
    "bunny": DatasetInfo(
        "Stanford Bunny", "Stanford 3D Scanning Repository (via Open3D)", "organic / fine detail"
    ),
    "armadillo": DatasetInfo(
        "Stanford Armadillo", "Stanford 3D Scanning Repository (via Open3D)", "articulated / concave"
    ),
    "torus": DatasetInfo("Analytic Torus", "Procedural reference", "thin structure / central hole"),
    "knot": DatasetInfo("Trefoil Knot", "Open3D sample geometry", "high genus / curved tube"),
    "sphere": DatasetInfo("UV Sphere", "Procedural reference", "smooth closed surface"),
}


def _normalise_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError("Reference geometry must be a non-empty triangle mesh")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    bounds = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=float)
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 1e-12:
        raise ValueError("Reference geometry has zero spatial extent")
    mesh.translate(-np.asarray(bounds.get_center()))
    mesh.scale(1.0 / diagonal, center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def load_reference_mesh(name: str) -> o3d.geometry.TriangleMesh:
    """Load a bundled/procedural model and normalize its bounding-box diagonal to one."""

    key = name.lower().strip()
    if key == "bunny":
        path = o3d.data.BunnyMesh().path
        mesh = o3d.io.read_triangle_mesh(path)
    elif key == "armadillo":
        path = o3d.data.ArmadilloMesh().path
        mesh = o3d.io.read_triangle_mesh(path)
    elif key == "knot":
        path = o3d.data.KnotMesh().path
        mesh = o3d.io.read_triangle_mesh(path)
    elif key == "torus":
        mesh = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=1.0, tube_radius=0.32, radial_resolution=80, tubular_resolution=36
        )
    elif key == "sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=45)
    else:
        raise KeyError(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASETS)}")
    return _normalise_mesh(mesh)


def load_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    return _normalise_mesh(mesh)


def sample_surface(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int,
    seed: int = 7,
    method: str = "uniform",
) -> o3d.geometry.PointCloud:
    """Sample a mesh deterministically while retaining usable normals."""

    if n_points < 1:
        raise ValueError("n_points must be positive")
    o3d.utility.random.seed(int(seed))
    if method == "poisson_disk":
        cloud = mesh.sample_points_poisson_disk(n_points, init_factor=3, use_triangle_normal=True)
    elif method == "uniform":
        cloud = mesh.sample_points_uniformly(n_points, use_triangle_normal=True)
    else:
        raise ValueError("method must be 'uniform' or 'poisson_disk'")
    cloud.normalize_normals()
    return cloud


def normalised_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Public copy-and-normalize helper used by uploaded meshes."""

    return _normalise_mesh(mesh)
