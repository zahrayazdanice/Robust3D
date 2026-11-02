"""End-to-end orchestration shared by experiments, CLI, and dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import open3d as o3d

from .config import PipelineConfig, ReconstructionConfig
from .datasets import load_reference_mesh, sample_surface
from .degradation import DegradationReport, degrade_point_cloud
from .evaluation import EvaluationMetrics, evaluate_input_cloud, evaluate_mesh
from .preprocessing import PreprocessReport, preprocess_point_cloud
from .reconstruction import ReconstructionReport, reconstruct_surface


@dataclass(slots=True)
class PreparedCase:
    reference_mesh: o3d.geometry.TriangleMesh | None
    clean_cloud: o3d.geometry.PointCloud
    degraded_cloud: o3d.geometry.PointCloud
    processed_cloud: o3d.geometry.PointCloud
    degradation_report: DegradationReport
    preprocessing_report: PreprocessReport
    input_metrics: dict[str, float]
    preparation_seconds: float


@dataclass(slots=True)
class PipelineResult:
    config: PipelineConfig
    reference_mesh: o3d.geometry.TriangleMesh | None
    clean_cloud: o3d.geometry.PointCloud
    degraded_cloud: o3d.geometry.PointCloud
    processed_cloud: o3d.geometry.PointCloud
    reconstructed_mesh: o3d.geometry.TriangleMesh
    degradation_report: DegradationReport
    preprocessing_report: PreprocessReport
    reconstruction_report: ReconstructionReport
    metrics: EvaluationMetrics | None
    input_metrics: dict[str, float]
    total_seconds: float

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self.config.to_dict(),
            "degradation_report": asdict(self.degradation_report),
            "preprocessing_report": asdict(self.preprocessing_report),
            "reconstruction_report": asdict(self.reconstruction_report),
            "input_metrics": self.input_metrics,
            "total_seconds": self.total_seconds,
        }
        payload["metrics"] = self.metrics.to_dict() if self.metrics else None
        return payload


def prepare_case(
    config: PipelineConfig,
    reference_mesh: o3d.geometry.TriangleMesh | None = None,
    source_cloud: o3d.geometry.PointCloud | None = None,
) -> PreparedCase:
    """Sample/degrade/preprocess once so several algorithms can share identical input."""

    config.validate()
    started = perf_counter()
    if source_cloud is None:
        reference = reference_mesh or load_reference_mesh(config.object_name)
        clean = sample_surface(reference, config.n_points, config.degradation.seed)
    else:
        reference = reference_mesh
        clean = o3d.geometry.PointCloud(source_cloud)
    degraded, degradation_report = degrade_point_cloud(clean, config.degradation)
    processed, preprocessing_report = preprocess_point_cloud(degraded, config.preprocessing)
    input_metrics = (
        evaluate_input_cloud(degraded, reference, config.evaluation_samples, config.degradation.seed)
        if reference is not None
        else {}
    )
    return PreparedCase(
        reference_mesh=reference,
        clean_cloud=clean,
        degraded_cloud=degraded,
        processed_cloud=processed,
        degradation_report=degradation_report,
        preprocessing_report=preprocessing_report,
        input_metrics=input_metrics,
        preparation_seconds=perf_counter() - started,
    )


def run_prepared(
    prepared: PreparedCase,
    config: PipelineConfig,
    reconstruction: ReconstructionConfig | None = None,
) -> PipelineResult:
    reconstruction_config = reconstruction or config.reconstruction
    mesh, reconstruction_report = reconstruct_surface(prepared.processed_cloud, reconstruction_config)
    metrics = (
        evaluate_mesh(
            mesh,
            prepared.reference_mesh,
            samples=config.evaluation_samples,
            seed=config.degradation.seed + 101,
        )
        if prepared.reference_mesh is not None
        else None
    )
    total = prepared.preparation_seconds + reconstruction_report.seconds
    return PipelineResult(
        config=config,
        reference_mesh=prepared.reference_mesh,
        clean_cloud=prepared.clean_cloud,
        degraded_cloud=prepared.degraded_cloud,
        processed_cloud=prepared.processed_cloud,
        reconstructed_mesh=mesh,
        degradation_report=prepared.degradation_report,
        preprocessing_report=prepared.preprocessing_report,
        reconstruction_report=reconstruction_report,
        metrics=metrics,
        input_metrics=prepared.input_metrics,
        total_seconds=total,
    )


def run_pipeline(
    config: PipelineConfig,
    reference_mesh: o3d.geometry.TriangleMesh | None = None,
    source_cloud: o3d.geometry.PointCloud | None = None,
) -> PipelineResult:
    prepared = prepare_case(config, reference_mesh, source_cloud)
    return run_prepared(prepared, config)
