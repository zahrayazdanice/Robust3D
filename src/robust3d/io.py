"""Geometry input/output helpers, including browser upload parsing."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

from .datasets import normalised_mesh

SUPPORTED_EXTENSIONS = {".ply", ".obj", ".stl", ".off", ".xyz", ".xyzn", ".xyzrgb", ".pts", ".pcd"}


def _normalise_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(cloud)
    points = np.asarray(cloud.points)
    if len(points) < 50:
        raise ValueError("Uploaded point cloud must contain at least 50 points")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diagonal <= 1e-12:
        raise ValueError("Uploaded point cloud has zero spatial extent")
    points = (points - center) / diagonal
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def read_geometry(path: str | Path) -> tuple[o3d.geometry.TriangleMesh | None, o3d.geometry.PointCloud]:
    """Read a mesh or point cloud; a mesh also yields a sampled source cloud."""

    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported geometry format: {path.suffix}")
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.triangles) > 0:
        mesh = normalised_mesh(mesh)
        cloud = mesh.sample_points_uniformly(6000, use_triangle_normal=True)
        return mesh, cloud
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError("No readable vertices were found in the uploaded file")
    return None, _normalise_cloud(cloud)


def decode_dash_upload(contents: str, filename: str) -> tuple[o3d.geometry.TriangleMesh | None, o3d.geometry.PointCloud]:
    """Decode Dash's data URL without retaining user files."""

    try:
        _, encoded = contents.split(",", 1)
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Upload is not a valid base64 data URL") from exc
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported upload '{suffix}'. Use PLY, OBJ, STL, OFF, XYZ, PTS, or PCD.")
    with tempfile.TemporaryDirectory(prefix="robust3d-upload-") as temp_dir:
        path = Path(temp_dir) / f"upload{suffix}"
        path.write_bytes(payload)
        return read_geometry(path)


def mesh_to_ascii_ply(mesh: o3d.geometry.TriangleMesh) -> bytes:
    """Serialize a mesh for dashboard download without persistent temporary files."""

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=int)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    vertex_lines = "".join(f"{x:.8g} {y:.8g} {z:.8g}\n" for x, y, z in vertices)
    face_lines = "".join(f"3 {a} {b} {c}\n" for a, b, c in triangles)
    return (header + vertex_lines + face_lines).encode("utf-8")

