<div align="center">

# Robust3D

### Surface reconstruction from noisy and incomplete point clouds

**Poisson · Ball Pivoting · Alpha Shapes · Density-adaptive preprocessing**

[Interactive lab](#interactive-geometry-lab) · [Measured results](#measured-results) · [Technical notes](docs/ALGORITHMS.md)

</div>

![Robust3D interactive dashboard](results/figures/dashboard.png)

Robust3D is an end-to-end classical geometry processing project for asking a practical question: **what happens to surface reconstruction when a scan is sparse, noisy, occluded, or contaminated by outliers?** It provides a tested Python package, three reconstruction baselines, a geometry-based adaptive preprocessing method, a batch experiment runner, quantitative surface metrics, exported meshes, and an interactive Plotly/Dash laboratory.

There is no neural network and no hardcoded demo output. The dashboard, CLI, CSV files, figures, and report all call the same implementation in `src/robust3d`.

## Measured results

The committed quick benchmark contains **48 successful reconstruction runs**: 2 objects × 4 scan conditions × 2 preprocessing modes × 3 algorithms. It was regenerated on 26 October 2026 with Python 3.11, Open3D 0.19, 1,800 input samples, a fixed seed of 7, and 2,500 evaluation samples.

On the 12 paired, contamination-heavy cases (noise σ ≥ 0.5% of the object diagonal or nonzero outliers), adaptive preprocessing achieved:

- **7.4% median reduction in normalized Chamfer distance**
- **+0.058 median F-score change** at a 1%-diagonal threshold
- Better Chamfer and F-score in **83% of pairs**

Across all 24 pairs, including clean and missing-only scans, the medians were −2.0% Chamfer reduction and −0.007 F-score. In other words, the adaptive stage is useful when contamination is present, but can mildly oversmooth already-clean data and cannot restore geometry that was never observed. That negative result is part of the project, not hidden.

![Measured quick benchmark](results/figures/benchmark_summary.png)

| Scenario | Preprocessing | Chamfer ×1000 ↓ | F-score @ 1% ↑ | Normal consistency ↑ |
|---|---:|---:|---:|---:|
| Clean | Baseline | 10.587 | 0.536 | 0.942 |
| Clean | Adaptive | 11.369 | 0.507 | 0.930 |
| Noisy | Baseline | 13.338 | 0.396 | 0.888 |
| Noisy | Adaptive | 12.650 | 0.433 | 0.900 |
| Incomplete | Baseline | 20.671 | 0.463 | 0.888 |
| Incomplete | Adaptive | 21.209 | 0.431 | 0.887 |
| Combined | Baseline | 28.611 | 0.271 | 0.772 |
| Combined | Adaptive | 19.823 | 0.375 | 0.846 |

The table aggregates both objects and all three algorithms. Exact per-run numbers, timings, failures, and environment metadata live in [`results/metrics.csv`](results/metrics.csv), [`results/metrics.json`](results/metrics.json), and [`results/REPORT.md`](results/REPORT.md). Values may vary slightly across Open3D builds because metric sampling and parallel Poisson reconstruction can differ.

![Bunny under combined degradation and its three measured reconstructions](results/figures/bunny_combined_comparison.png)

## Interactive geometry lab

The dashboard is the main project interface. It can:

- switch among Stanford Bunny, Stanford Armadillo, a torus, a trefoil knot, and a sphere;
- upload PLY, OBJ, STL, OFF, XYZ, PTS, or PCD geometry;
- change sampling density, Gaussian noise, a contiguous missing cap, and uniform outliers;
- compare Poisson, BPA, and alpha shapes side by side with linked-style 3D controls;
- toggle baseline versus density-adaptive preprocessing;
- tune Poisson depth, BPA radii, alpha scale, point count, and random seed;
- compute Chamfer distance, F-score, precision/recall, normal consistency, 95th-percentile error, area ratio, topology counts, and runtime;
- export any computed reconstruction as an ASCII PLY mesh.

All work stays on the local machine. The visible progress state and error reporting make expensive or invalid parameter combinations explicit.

### Quick start

```bash
git clone <your-repository-url>
cd Robust3D
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
robust3d demo
```

Open <http://127.0.0.1:8050>. The Stanford meshes are downloaded by Open3D and cached on first use; Robust3D does not redistribute them.

## Command-line use

List the built-in objects:

```bash
robust3d datasets
```

Reconstruct a degraded Stanford Bunny:

```bash
robust3d reconstruct \
  --dataset bunny \
  --algorithm poisson \
  --preprocessing adaptive \
  --points 5000 \
  --density 0.55 \
  --noise 0.008 \
  --missing 0.18 \
  --outliers 0.08 \
  --output results/meshes/bunny_poisson.ply
```

Use an uploaded mesh or point cloud:

```bash
robust3d reconstruct --input scan.ply --algorithm bpa --output reconstruction.ply
```

## Metrics

- **Chamfer ×1000:** mean of the two directional, unsquared nearest-surface distances, divided by the clean reference bounding-box diagonal and multiplied by 1000.
- **F-score @ 1%:** harmonic mean of reconstructed-surface precision and reference-surface recall within 1% of that diagonal.
- **Normal consistency:** symmetric mean absolute dot product between nearest sampled surface normals.
- **H95 ×1000:** 95th percentile of the combined directional distances, normalized by the diagonal.
- **Area ratio / watertightness / counts:** useful diagnostic properties, not direct accuracy measures.
- **Runtime:** wall-clock reconstruction time; preparation is recorded separately to avoid charging shared preprocessing to every method.

The reconstruction and reference meshes are independently sampled for evaluation. See the exact definitions in [`src/robust3d/evaluation.py`](src/robust3d/evaluation.py).

## Repository map

```text
Robust3D/
├── app.py                       # dashboard entry point
├── configs/                     # quick and full benchmark protocols
├── docs/                        # algorithms, datasets, experiment design
├── results/                     # real committed outputs and raw metrics
├── scripts/run_experiments.py   # convenience benchmark entry point
├── src/robust3d/
│   ├── datasets.py              # Stanford/procedural references + sampling
│   ├── degradation.py           # controlled scan corruptions
│   ├── preprocessing.py         # baseline + proposed adaptive method
│   ├── reconstruction.py        # Poisson, BPA, alpha shapes
│   ├── evaluation.py            # surface metrics
│   ├── pipeline.py              # shared orchestration
│   ├── visualization.py         # live and static graphics
│   ├── dashboard.py             # functional Dash app
│   ├── assets/style.css         # responsive scientific UI
│   └── cli.py                   # reproducible CLI
└── tests/                       # deterministic unit + integration tests
```

## Limitations

- Poisson can produce visually plausible closure over missing regions; this is not evidence that the hidden surface is correct.
- BPA depends on normal orientation and radius choice; very sparse or uneven scans may leave holes.
- Alpha shapes are sensitive to alpha and can become coarse or fragmented.
- The adaptive stage targets noise and isolated outliers. It is intentionally bypassable because it can oversmooth clean high-frequency detail.
- Uploads that contain only points have no ground-truth surface, so reference-based metrics are marked unavailable.
- The benchmark is a controlled undergraduate study, not a claim of state-of-the-art reconstruction.

## License

Project code is released under the [MIT License](LICENSE). Third-party datasets retain their original terms.
