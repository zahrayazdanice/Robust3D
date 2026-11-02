import numpy as np
import open3d as o3d

from robust3d.config import PreprocessConfig
from robust3d.preprocessing import estimate_spacing, preprocess_point_cloud


def test_adaptive_filter_removes_isolated_outliers_and_estimates_normals():
    rng = np.random.default_rng(5)
    core = rng.normal(0, 0.01, size=(800, 3))
    core[:, 2] *= 0.1
    outliers = rng.uniform(0.7, 1.0, size=(35, 3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((core, outliers))))
    result, report = preprocess_point_cloud(
        cloud, PreprocessConfig(mode="adaptive", smooth_strength=0.2)
    )
    assert report.rejected_outliers >= 25
    assert report.output_points >= 500
    assert result.has_normals()
    assert np.isfinite(np.asarray(result.normals)).all()


def test_spacing_scales_with_geometry():
    rng = np.random.default_rng(8)
    points = rng.random((300, 3))
    assert np.isclose(estimate_spacing(points * 3), estimate_spacing(points) * 3, rtol=1e-5)

