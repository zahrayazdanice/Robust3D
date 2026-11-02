"""Robust3D: classical surface reconstruction under imperfect scans."""

from .config import (
    DegradationConfig,
    PipelineConfig,
    PreprocessConfig,
    ReconstructionConfig,
)
from .pipeline import PipelineResult, run_pipeline

__all__ = [
    "DegradationConfig",
    "PipelineConfig",
    "PipelineResult",
    "PreprocessConfig",
    "ReconstructionConfig",
    "run_pipeline",
]
__version__ = "1.0.0"
