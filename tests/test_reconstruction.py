import pytest

from robust3d.config import PreprocessConfig, ReconstructionConfig
from robust3d.datasets import load_reference_mesh, sample_surface
from robust3d.evaluation import evaluate_mesh
from robust3d.preprocessing import preprocess_point_cloud
from robust3d.reconstruction import reconstruct_surface


@pytest.fixture(scope="module")
def sphere_case():
    reference = load_reference_mesh("sphere")
    cloud = sample_surface(reference, 900, seed=11)
    processed, _ = preprocess_point_cloud(
        cloud, PreprocessConfig(mode="baseline", normal_radius_factor=3.5)
    )
    return reference, processed


@pytest.mark.parametrize("algorithm", ["poisson", "bpa", "alpha"])
def test_all_reconstructors_return_nonempty_mesh(sphere_case, algorithm):
    reference, cloud = sphere_case
    config = ReconstructionConfig(algorithm=algorithm, poisson_depth=5, alpha_factor=5.0)
    mesh, report = reconstruct_surface(cloud, config)
    assert len(mesh.vertices) > 20
    assert len(mesh.triangles) > 20
    assert report.seconds >= 0
    metrics = evaluate_mesh(mesh, reference, samples=700, seed=13)
    assert metrics.chamfer_x1000 < 100
    assert 0 <= metrics.fscore_1pct <= 1
    assert 0 <= metrics.normal_consistency <= 1.00001

