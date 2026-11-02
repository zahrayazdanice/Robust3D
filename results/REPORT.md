# Robust3D experiment report

Generated: 2026-10-26T10:57:08.070863+00:00  
Configuration: `configs/quick.yaml`  
Successful runs: **48 / 48**

## Main measured finding

Across all **24** paired cases (including clean data), adaptive preprocessing had a median Chamfer reduction of **-2.0%** and median F-score change of **-0.007**. A negative reduction denotes a regression.

On the **12 contamination-heavy cases** (noise σ ≥ 0.5% or nonzero outliers), the median Chamfer reduction was **+7.4%** and median F-score change was **+0.058**. Adaptive preprocessing improved Chamfer in **83%** and F-score in **83%** of these pairs.

The clean and missing-only regimes show the tradeoff: smoothing/downsampling can slightly reduce fidelity when there is little contamination, and filtering cannot recover unseen geometry. These results are configuration-specific, not a claim of general superiority.

## Aggregate results

| scenario   | algorithm   | preprocessing   |   chamfer_x1000 |   fscore_1pct |   normal_consistency |   reconstruction_seconds |
|:-----------|:------------|:----------------|----------------:|--------------:|---------------------:|-------------------------:|
| clean      | alpha       | adaptive        |         12.4371 |        0.4601 |               0.9002 |                   0.0278 |
| clean      | alpha       | baseline        |         11.8344 |        0.4820 |               0.9130 |                   0.0337 |
| clean      | bpa         | adaptive        |         10.3262 |        0.5389 |               0.9495 |                   0.0284 |
| clean      | bpa         | baseline        |          9.6919 |        0.5816 |               0.9565 |                   0.0340 |
| clean      | poisson     | adaptive        |         11.3450 |        0.5224 |               0.9412 |                   0.1333 |
| clean      | poisson     | baseline        |         10.2338 |        0.5433 |               0.9558 |                   0.1361 |
| combined   | alpha       | adaptive        |         24.0645 |        0.2891 |               0.8000 |                   0.0122 |
| combined   | alpha       | baseline        |         45.6018 |        0.0874 |               0.6531 |                   0.0120 |
| combined   | bpa         | adaptive        |         18.9801 |        0.4237 |               0.8512 |                   0.0147 |
| combined   | bpa         | baseline        |         19.6630 |        0.3533 |               0.8092 |                   0.0196 |
| combined   | poisson     | adaptive        |         16.4239 |        0.4116 |               0.8879 |                   0.1208 |
| combined   | poisson     | baseline        |         20.5668 |        0.3738 |               0.8550 |                   0.1318 |
| incomplete | alpha       | adaptive        |         25.2822 |        0.3707 |               0.8366 |                   0.0101 |
| incomplete | alpha       | baseline        |         24.6207 |        0.3931 |               0.8429 |                   0.0148 |
| incomplete | bpa         | adaptive        |         22.5023 |        0.4602 |               0.8996 |                   0.0149 |
| incomplete | bpa         | baseline        |         21.8009 |        0.5054 |               0.8915 |                   0.0142 |
| incomplete | poisson     | adaptive        |         15.8429 |        0.4622 |               0.9262 |                   0.1178 |
| incomplete | poisson     | baseline        |         15.5905 |        0.4912 |               0.9309 |                   0.1429 |
| noisy      | alpha       | adaptive        |         15.0417 |        0.3743 |               0.8687 |                   0.0183 |
| noisy      | alpha       | baseline        |         17.0643 |        0.2638 |               0.8617 |                   0.0156 |
| noisy      | bpa         | adaptive        |         11.3007 |        0.4625 |               0.8968 |                   0.0297 |
| noisy      | bpa         | baseline        |         11.7490 |        0.4257 |               0.8718 |                   0.0344 |
| noisy      | poisson     | adaptive        |         11.6070 |        0.4636 |               0.9330 |                   0.1197 |
| noisy      | poisson     | baseline        |         11.2019 |        0.4978 |               0.9312 |                   0.1210 |

## Metric definitions

- **Chamfer ×1000:** mean unsquared bidirectional sampled surface distance, divided by the reference bounding-box diagonal and multiplied by 1000; lower is better.
- **F-score @ 1%:** harmonic mean of surface precision and recall within 1% of the reference diagonal; higher is better.
- **Normal consistency:** symmetric mean absolute dot product of nearest sampled normals; higher is better.
- Timings are wall-clock measurements and include reconstruction only unless explicitly labeled total.

## Reproducibility

- Python: `3.11.0`
- Open3D: `0.19.0`
- Platform: `Windows-10-10.0.19045-SP0`
- Seed: `7`
- Raw per-run values: [`metrics.csv`](metrics.csv) and [`metrics.json`](metrics.json).

## Failures and limitations

No algorithm failures occurred in this run.

The benchmark uses sampled distances, so values have small Monte Carlo variance. Stanford models are downloaded by Open3D on first use. The analytic torus is included to probe topology without dataset licensing friction. Runtime depends strongly on CPU, point count, and Poisson octree depth.