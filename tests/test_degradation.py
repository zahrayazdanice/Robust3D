import numpy as np
import open3d as o3d

from robust3d.config import DegradationConfig
from robust3d.degradation import degrade_point_cloud


def _cloud(n=1000):
    rng = np.random.default_rng(3)
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def test_degradation_is_deterministic_and_reports_counts():
    config = DegradationConfig(
        density=0.6, noise_sigma=0.01, missing_fraction=0.2, outlier_fraction=0.1, seed=42
    )
    first, report = degrade_point_cloud(_cloud(), config)
    second, _ = degrade_point_cloud(_cloud(), config)
    np.testing.assert_allclose(np.asarray(first.points), np.asarray(second.points))
    assert report.retained_after_sampling == 600
    assert report.removed_for_hole == 120
    assert report.added_outliers == 48
    assert report.final_points == 528


def test_degradation_rejects_invalid_fraction():
    config = DegradationConfig(density=0.01)
    try:
        degrade_point_cloud(_cloud(), config)
    except ValueError as exc:
        assert "density" in str(exc)
    else:
        raise AssertionError("Expected invalid density to raise")

