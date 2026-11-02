"""Typed configuration objects shared by the CLI, experiments, and dashboard."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DegradationConfig:
    """Synthetic scan imperfections, expressed relative to the object scale."""

    density: float = 0.65
    noise_sigma: float = 0.006
    missing_fraction: float = 0.12
    outlier_fraction: float = 0.04
    seed: int = 7

    def validate(self) -> None:
        if not 0.05 <= self.density <= 1.0:
            raise ValueError("density must be in [0.05, 1.0]")
        if not 0.0 <= self.noise_sigma <= 0.1:
            raise ValueError("noise_sigma must be in [0, 0.1]")
        if not 0.0 <= self.missing_fraction < 0.8:
            raise ValueError("missing_fraction must be in [0, 0.8)")
        if not 0.0 <= self.outlier_fraction <= 0.5:
            raise ValueError("outlier_fraction must be in [0, 0.5]")


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for either the baseline or proposed adaptive pipeline."""

    mode: str = "adaptive"
    knn: int = 20
    outlier_mad_z: float = 3.5
    voxel_factor: float = 0.65
    smooth_strength: float = 0.30
    normal_radius_factor: float = 3.5

    def validate(self) -> None:
        if self.mode not in {"baseline", "adaptive"}:
            raise ValueError("preprocessing mode must be 'baseline' or 'adaptive'")
        if self.knn < 6:
            raise ValueError("knn must be at least 6")
        if not 0 <= self.smooth_strength <= 1:
            raise ValueError("smooth_strength must be in [0, 1]")


@dataclass(slots=True)
class ReconstructionConfig:
    """Algorithm parameters; distance-like values are multiples of point spacing."""

    algorithm: str = "poisson"
    poisson_depth: int = 7
    poisson_density_quantile: float = 0.025
    bpa_radius_factors: tuple[float, ...] = (1.5, 2.5, 4.0)
    alpha_factor: float = 4.5
    keep_largest_components: bool = True

    def validate(self) -> None:
        if self.algorithm not in {"poisson", "bpa", "alpha"}:
            raise ValueError("algorithm must be one of: poisson, bpa, alpha")
        if not 5 <= self.poisson_depth <= 10:
            raise ValueError("poisson_depth must be in [5, 10]")
        if not 0 <= self.poisson_density_quantile < 0.5:
            raise ValueError("poisson_density_quantile must be in [0, 0.5)")
        if not self.bpa_radius_factors or min(self.bpa_radius_factors) <= 0:
            raise ValueError("BPA radius factors must be positive")
        if self.alpha_factor <= 0:
            raise ValueError("alpha_factor must be positive")


@dataclass(slots=True)
class PipelineConfig:
    """Complete reproducible pipeline configuration."""

    object_name: str = "bunny"
    n_points: int = 3500
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    evaluation_samples: int = 5000

    def validate(self) -> None:
        self.degradation.validate()
        self.preprocessing.validate()
        self.reconstruction.validate()
        if self.n_points < 200:
            raise ValueError("n_points must be at least 200")
        if self.evaluation_samples < 200:
            raise ValueError("evaluation_samples must be at least 200")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
