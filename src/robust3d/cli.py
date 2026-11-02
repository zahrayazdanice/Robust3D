"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import open3d as o3d

from .config import DegradationConfig, PipelineConfig, PreprocessConfig, ReconstructionConfig
from .datasets import DATASETS
from .experiments import run_experiments
from .io import read_geometry
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robust3d", description="Robust classical 3D surface reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="start the interactive Dash application")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8050)
    demo.add_argument("--debug", action="store_true")

    experiment = sub.add_parser("experiment", help="run a YAML benchmark")
    experiment.add_argument("--config", default="configs/quick.yaml")
    experiment.add_argument("--output", default="results")

    reconstruct = sub.add_parser("reconstruct", help="reconstruct one mesh or built-in dataset")
    source = reconstruct.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=DATASETS, default="bunny")
    source.add_argument("--input", type=Path)
    reconstruct.add_argument("--algorithm", choices=("poisson", "bpa", "alpha"), default="poisson")
    reconstruct.add_argument("--preprocessing", choices=("baseline", "adaptive"), default="adaptive")
    reconstruct.add_argument("--points", type=int, default=3500)
    reconstruct.add_argument("--noise", type=float, default=0.006, help="sigma as fraction of object diagonal")
    reconstruct.add_argument("--density", type=float, default=0.65)
    reconstruct.add_argument("--missing", type=float, default=0.12)
    reconstruct.add_argument("--outliers", type=float, default=0.04)
    reconstruct.add_argument("--seed", type=int, default=7)
    reconstruct.add_argument("--depth", type=int, default=7)
    reconstruct.add_argument("--output", type=Path, default=Path("reconstruction.ply"))

    sub.add_parser("datasets", help="list built-in reference objects")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "demo":
        from .dashboard import create_app

        create_app().run(host=args.host, port=args.port, debug=args.debug)
        return 0
    if args.command == "experiment":
        frame = run_experiments(args.config, args.output)
        print(f"Completed {len(frame)} runs; results written to {args.output}")
        return 0
    if args.command == "datasets":
        for key, value in DATASETS.items():
            print(f"{key:10} {value.label:22} {value.geometry} — {value.source}")
        return 0

    reference = None
    source_cloud = None
    object_name = args.dataset or "upload"
    if args.input:
        reference, source_cloud = read_geometry(args.input)
    config = PipelineConfig(
        object_name=object_name,
        n_points=args.points,
        degradation=DegradationConfig(
            density=args.density, noise_sigma=args.noise, missing_fraction=args.missing,
            outlier_fraction=args.outliers, seed=args.seed,
        ),
        preprocessing=PreprocessConfig(mode=args.preprocessing),
        reconstruction=ReconstructionConfig(algorithm=args.algorithm, poisson_depth=args.depth),
    )
    result = run_pipeline(config, reference, source_cloud)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), result.reconstructed_mesh):
        raise RuntimeError(f"Failed to write {args.output}")
    print(json.dumps(result.summary(), indent=2))
    print(f"Mesh written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
