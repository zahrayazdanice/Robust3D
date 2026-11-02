"""Reproducible benchmark runner and report generation."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from .config import DegradationConfig, PipelineConfig, PreprocessConfig, ReconstructionConfig
from .pipeline import prepare_case, run_prepared
from .visualization import save_static_comparison


def _plot_summary(frame: pd.DataFrame, path: Path) -> None:
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return
    scenarios = list(dict.fromkeys(ok["scenario"]))
    modes = [mode for mode in ("baseline", "adaptive") if mode in set(ok["preprocessing"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#080D18")
    for ax in axes:
        ax.set_facecolor("#0E1626")
        ax.tick_params(colors="#A9B6C9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#283449")
        ax.grid(axis="y", color="#263247", alpha=0.65, linewidth=0.7)
        ax.set_axisbelow(True)
    x = np.arange(len(scenarios))
    width = 0.34
    mode_colors = {"baseline": "#6C8BFF", "adaptive": "#55D6BE"}
    for offset, mode in enumerate(modes):
        grouped = ok[ok["preprocessing"] == mode].groupby("scenario", sort=False)
        chamfer = [grouped["chamfer_x1000"].mean().get(s, np.nan) for s in scenarios]
        fscore = [grouped["fscore_1pct"].mean().get(s, np.nan) for s in scenarios]
        positions = x + (offset - (len(modes) - 1) / 2) * width
        axes[0].bar(positions, chamfer, width, label=mode.title(), color=mode_colors[mode])
        axes[1].bar(positions, fscore, width, label=mode.title(), color=mode_colors[mode])
    axes[0].set_title("Normalized Chamfer ×1000 ↓", color="#EEF4FF", fontsize=11)
    axes[1].set_title("F-score at 1% diagonal ↑", color="#EEF4FF", fontsize=11)
    for ax in axes:
        ax.set_xticks(x, scenarios, rotation=24, ha="right")
    axes[0].legend(frameon=False, labelcolor="#C6D2E3", fontsize=8)
    fig.suptitle("Robust3D · measured benchmark summary", color="#EEF4FF", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _paired_improvements(frame: pd.DataFrame) -> dict[str, float | int]:
    ok = frame[frame["status"] == "ok"]
    keys = ["dataset", "scenario", "algorithm"]
    pivot = ok.pivot_table(index=keys, columns="preprocessing", values=["chamfer_x1000", "fscore_1pct"])
    if not {"baseline", "adaptive"}.issubset(set(ok["preprocessing"])) or pivot.empty:
        return {"paired_cases": 0}
    chamfer_base = pivot[("chamfer_x1000", "baseline")]
    chamfer_adapt = pivot[("chamfer_x1000", "adaptive")]
    f_base = pivot[("fscore_1pct", "baseline")]
    f_adapt = pivot[("fscore_1pct", "adaptive")]
    valid = chamfer_base.notna() & chamfer_adapt.notna() & f_base.notna() & f_adapt.notna()
    if not valid.any():
        return {"paired_cases": 0}
    chamfer_reduction = (chamfer_base[valid] - chamfer_adapt[valid]) / chamfer_base[valid].clip(lower=1e-12)
    return {
        "paired_cases": int(valid.sum()),
        "median_chamfer_reduction_pct": float(100 * chamfer_reduction.median()),
        "mean_chamfer_reduction_pct": float(100 * chamfer_reduction.mean()),
        "median_fscore_delta": float((f_adapt[valid] - f_base[valid]).median()),
        "adaptive_chamfer_win_rate": float((chamfer_adapt[valid] < chamfer_base[valid]).mean()),
        "adaptive_fscore_win_rate": float((f_adapt[valid] > f_base[valid]).mean()),
    }


def _write_report(frame: pd.DataFrame, metadata: dict, output: Path) -> None:
    ok = frame[frame["status"] == "ok"].copy()
    improvements = metadata["paired_improvements"]
    contaminated = metadata["contaminated_improvements"]
    lines = [
        "# Robust3D experiment report",
        "",
        f"Generated: {metadata['generated_at']}  ",
        f"Configuration: `{metadata['config']}`  ",
        f"Successful runs: **{len(ok)} / {len(frame)}**",
        "",
        "## Main measured finding",
        "",
    ]
    if improvements.get("paired_cases", 0):
        lines += [
            f"Across all **{improvements['paired_cases']}** paired cases (including clean data), adaptive preprocessing "
            f"had a median Chamfer reduction of **{improvements['median_chamfer_reduction_pct']:+.1f}%** and median "
            f"F-score change of **{improvements['median_fscore_delta']:+.3f}**. A negative reduction denotes a regression.",
            "",
            f"On the **{contaminated['paired_cases']} contamination-heavy cases** (noise σ ≥ 0.5% or nonzero outliers), "
            f"the median Chamfer reduction was **{contaminated['median_chamfer_reduction_pct']:+.1f}%** and median "
            f"F-score change was **{contaminated['median_fscore_delta']:+.3f}**. Adaptive preprocessing improved "
            f"Chamfer in **{100 * contaminated['adaptive_chamfer_win_rate']:.0f}%** and F-score in "
            f"**{100 * contaminated['adaptive_fscore_win_rate']:.0f}%** of these pairs.",
            "",
            "The clean and missing-only regimes show the tradeoff: smoothing/downsampling can slightly reduce fidelity "
            "when there is little contamination, and filtering cannot recover unseen geometry. These results are "
            "configuration-specific, not a claim of general superiority.",
            "",
        ]
    lines += [
        "## Aggregate results",
        "",
    ]
    if not ok.empty:
        summary = (
            ok.groupby(["scenario", "algorithm", "preprocessing"])[
                ["chamfer_x1000", "fscore_1pct", "normal_consistency", "reconstruction_seconds"]
            ]
            .mean()
            .reset_index()
        )
        lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines += [
        "",
        "## Metric definitions",
        "",
        "- **Chamfer ×1000:** mean unsquared bidirectional sampled surface distance, divided by the reference bounding-box diagonal and multiplied by 1000; lower is better.",
        "- **F-score @ 1%:** harmonic mean of surface precision and recall within 1% of the reference diagonal; higher is better.",
        "- **Normal consistency:** symmetric mean absolute dot product of nearest sampled normals; higher is better.",
        "- Timings are wall-clock measurements and include reconstruction only unless explicitly labeled total.",
        "",
        "## Reproducibility",
        "",
        f"- Python: `{metadata['python']}`",
        f"- Open3D: `{metadata['open3d']}`",
        f"- Platform: `{metadata['platform']}`",
        f"- Seed: `{metadata['seed']}`",
        "- Raw per-run values: [`metrics.csv`](metrics.csv) and [`metrics.json`](metrics.json).",
        "",
        "## Failures and limitations",
        "",
    ]
    failures = frame[frame["status"] != "ok"]
    if failures.empty:
        lines.append("No algorithm failures occurred in this run.")
    else:
        for _, row in failures.iterrows():
            lines.append(f"- `{row['dataset']}/{row['scenario']}/{row['preprocessing']}/{row['algorithm']}`: {row['error']}")
    lines += [
        "",
        "The benchmark uses sampled distances, so values have small Monte Carlo variance. Stanford models are downloaded by Open3D on first use. The analytic torus is included to probe topology without dataset licensing friction. Runtime depends strongly on CPU, point count, and Poisson octree depth.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiments(config_path: str | Path, output_dir: str | Path = "results") -> pd.DataFrame:
    """Run all configured cases and persist raw values, figures, meshes, and a report."""

    config_path = Path(config_path)
    output = Path(output_dir)
    figures = output / "figures"
    meshes = output / "meshes"
    figures.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    exemplar: dict[str, object] = {}
    seed = int(spec.get("seed", 7))
    started_all = perf_counter()

    for dataset in spec["datasets"]:
        for scenario, degradation_values in spec["scenarios"].items():
            for preprocessing_mode in spec["preprocessing"]:
                base = PipelineConfig(
                    object_name=dataset,
                    n_points=int(spec["points"]),
                    evaluation_samples=int(spec["evaluation_samples"]),
                    degradation=DegradationConfig(seed=seed, **degradation_values),
                    preprocessing=PreprocessConfig(mode=preprocessing_mode),
                    reconstruction=ReconstructionConfig(poisson_depth=int(spec.get("poisson_depth", 7))),
                )
                try:
                    prepared = prepare_case(base)
                except Exception as exc:
                    for algorithm in spec["algorithms"]:
                        rows.append({
                            "dataset": dataset, "scenario": scenario, "preprocessing": preprocessing_mode,
                            "algorithm": algorithm, "status": "failed", "error": f"preparation: {exc}",
                        })
                    continue
                case_meshes: dict[str, o3d.geometry.TriangleMesh] = {}
                for algorithm in spec["algorithms"]:
                    recon_config = ReconstructionConfig(
                        algorithm=algorithm,
                        poisson_depth=int(spec.get("poisson_depth", 7)),
                    )
                    run_config = PipelineConfig(
                        object_name=base.object_name,
                        n_points=base.n_points,
                        evaluation_samples=base.evaluation_samples,
                        degradation=base.degradation,
                        preprocessing=base.preprocessing,
                        reconstruction=recon_config,
                    )
                    try:
                        result = run_prepared(prepared, run_config, recon_config)
                        metric_values = result.metrics.to_dict() if result.metrics else {}
                        row = {
                            "dataset": dataset,
                            "scenario": scenario,
                            "preprocessing": preprocessing_mode,
                            "algorithm": algorithm,
                            "status": "ok",
                            "error": "",
                            **degradation_values,
                            **metric_values,
                            **result.input_metrics,
                            "input_points": result.degradation_report.final_points,
                            "processed_points": result.preprocessing_report.output_points,
                            "rejected_outliers": result.preprocessing_report.rejected_outliers,
                            "estimated_spacing": result.preprocessing_report.estimated_spacing,
                            "preparation_seconds": prepared.preparation_seconds,
                            "reconstruction_seconds": result.reconstruction_report.seconds,
                            "total_seconds": result.total_seconds,
                        }
                        rows.append(row)
                        case_meshes[algorithm] = result.reconstructed_mesh
                    except Exception as exc:
                        rows.append({
                            "dataset": dataset, "scenario": scenario, "preprocessing": preprocessing_mode,
                            "algorithm": algorithm, "status": "failed", "error": str(exc), **degradation_values,
                        })
                if dataset == spec["datasets"][0] and scenario == "combined" and preprocessing_mode == "adaptive" and case_meshes:
                    exemplar = {"prepared": prepared, "meshes": case_meshes}
                    for algorithm, mesh in case_meshes.items():
                        o3d.io.write_triangle_mesh(str(meshes / f"{dataset}_combined_adaptive_{algorithm}.ply"), mesh)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(frame.replace({np.nan: None}).to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    _plot_summary(frame, figures / "benchmark_summary.png")
    if exemplar:
        prepared = exemplar["prepared"]
        save_static_comparison(
            figures / "bunny_combined_comparison.png",
            prepared.reference_mesh,
            prepared.degraded_cloud,
            exemplar["meshes"],
        )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.as_posix()),
        "seed": seed,
        "python": sys.version.split()[0],
        "open3d": o3d.__version__,
        "platform": platform.platform(),
        "wall_seconds": perf_counter() - started_all,
        "paired_improvements": _paired_improvements(frame),
        "contaminated_improvements": _paired_improvements(
            frame[(frame["noise_sigma"].fillna(0) >= 0.005) | (frame["outlier_fraction"].fillna(0) > 0)]
        ),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(frame, metadata, output)
    return frame
