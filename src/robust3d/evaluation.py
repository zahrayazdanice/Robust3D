"""Scale-normalized geometric reconstruction metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    chamfer_x1000: float
    fscore_1pct: float
    precision_1pct: float
    recall_1pct: float
    normal_consistency: float
    hausdorff95_x1000: float
    surface_area_ratio: float
    watertight: bool
    vertices: int
    triangles: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _sample_with_normals(
    mesh: o3d.geometry.TriangleMesh, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_triangle_normals()
    o3d.utility.random.seed(seed)
    cloud = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    return np.asarray(cloud.points), np.asarray(cloud.normals)


def _reference_scale(reference: o3d.geometry.TriangleMesh) -> float:
    return max(float(np.linalg.norm(reference.get_axis_aligned_bounding_box().get_extent())), 1e-12)


def evaluate_mesh(
    reconstruction: o3d.geometry.TriangleMesh,
    reference: o3d.geometry.TriangleMesh,
    samples: int = 5000,
    seed: int = 17,
) -> EvaluationMetrics:
    """Evaluate two surfaces using symmetric sampled distances and normals.

    Chamfer is the mean unsquared bidirectional distance, normalized by the
    reference bounding-box diagonal and multiplied by 1000. F-score uses a 1%
    diagonal threshold. Absolute normal dot products avoid penalizing a globally
    reversed but geometrically correct orientation.
    """

    if len(reconstruction.triangles) == 0 or len(reference.triangles) == 0:
        raise ValueError("Both meshes must contain triangles")
    n = max(200, int(samples))
    rec_points, rec_normals = _sample_with_normals(reconstruction, n, seed)
    ref_points, ref_normals = _sample_with_normals(reference, n, seed + 1)
    ref_tree = cKDTree(ref_points)
    rec_tree = cKDTree(rec_points)
    d_rec_ref, i_rec_ref = ref_tree.query(rec_points, workers=-1)
    d_ref_rec, i_ref_rec = rec_tree.query(ref_points, workers=-1)
    scale = _reference_scale(reference)
    threshold = 0.01 * scale
    precision = float(np.mean(d_rec_ref <= threshold))
    recall = float(np.mean(d_ref_rec <= threshold))
    fscore = float(2 * precision * recall / (precision + recall + 1e-12))
    normal_forward = np.abs(np.sum(rec_normals * ref_normals[i_rec_ref], axis=1))
    normal_backward = np.abs(np.sum(ref_normals * rec_normals[i_ref_rec], axis=1))
    normal_consistency = float(0.5 * (normal_forward.mean() + normal_backward.mean()))
    distances = np.concatenate((d_rec_ref, d_ref_rec))
    reference_area = max(reference.get_surface_area(), 1e-12)
    return EvaluationMetrics(
        chamfer_x1000=float((d_rec_ref.mean() + d_ref_rec.mean()) * 500.0 / scale),
        fscore_1pct=fscore,
        precision_1pct=precision,
        recall_1pct=recall,
        normal_consistency=normal_consistency,
        hausdorff95_x1000=float(np.quantile(distances, 0.95) * 1000.0 / scale),
        surface_area_ratio=float(reconstruction.get_surface_area() / reference_area),
        watertight=bool(reconstruction.is_watertight()),
        vertices=len(reconstruction.vertices),
        triangles=len(reconstruction.triangles),
    )


def evaluate_input_cloud(
    cloud: o3d.geometry.PointCloud,
    reference: o3d.geometry.TriangleMesh,
    samples: int = 5000,
    seed: int = 19,
) -> dict[str, float]:
    """Report scan-to-reference accuracy and reference coverage before reconstruction."""

    points = np.asarray(cloud.points)
    ref_points, _ = _sample_with_normals(reference, max(200, samples), seed)
    ref_tree = cKDTree(ref_points)
    cloud_tree = cKDTree(points)
    d_cloud, _ = ref_tree.query(points, workers=-1)
    d_ref, _ = cloud_tree.query(ref_points, workers=-1)
    scale = _reference_scale(reference)
    threshold = 0.01 * scale
    return {
        "input_accuracy_1pct": float(np.mean(d_cloud <= threshold)),
        "input_coverage_1pct": float(np.mean(d_ref <= threshold)),
        "input_chamfer_x1000": float((d_cloud.mean() + d_ref.mean()) * 500.0 / scale),
    }

