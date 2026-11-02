# Experiment protocol

## Research questions

1. How do Poisson, Ball Pivoting, and alpha shapes respond to controlled noise, sparsity, missing regions, and outliers?
2. Does scale-adaptive preprocessing improve reconstruction when scan density and contamination vary?
3. What accuracy/runtime/topology tradeoffs are visible in a reproducible undergraduate-scale benchmark?

## Controlled inputs

Each reference mesh is centered and scaled so its axis-aligned bounding-box diagonal is one. A seeded uniform surface sample is degraded in this order:

1. random density subsampling;
2. removal of a contiguous cap along a seeded random direction;
3. isotropic Gaussian positional noise, with σ given as a fraction of the clean diagonal;
4. uniform outliers in an expanded bounding box.

Baseline and adaptive preprocessing receive identical degraded points. All three reconstructors then share the same preprocessed cloud for a given case. This paired design avoids sampling differences masquerading as algorithm differences.

## Quick benchmark matrix

| Scenario | Density | Noise σ | Missing cap | Outliers |
|---|---:|---:|---:|---:|
| clean | 100% | 0% | 0% | 0% |
| noisy | 70% | 0.8% | 0% | 0% |
| incomplete | 65% | 0.2% | 25% | 0% |
| combined | 55% | 0.8% | 18% | 8% |

Objects: Stanford Bunny and analytic torus. Preprocessing: baseline and adaptive. Reconstruction: Poisson depth 6, BPA radii `[1.5, 2.5, 4.0] × spacing`, and alpha `4.5 × spacing`. Seed: 7.

This is 2 × 4 × 2 × 3 = 48 reconstructions. The full configuration expands to Stanford Armadillo and seven conditions.

## Evaluation

Reconstruction and reference surfaces are independently sampled. A `cKDTree` provides nearest neighbors in both directions.

Given reconstructed samples `R`, reference samples `G`, reference diagonal `D`, and nearest-neighbor distance `d(x, S)`:

```text
Chamfer = 0.5 × (mean_r d(r,G) + mean_g d(g,R)) / D
```

The reported value is multiplied by 1000. It is an unsquared Chamfer distance so its units remain interpretable.

At threshold `τ = 0.01D`:

```text
precision = mean_r [d(r,G) ≤ τ]
recall    = mean_g [d(g,R) ≤ τ]
F-score   = 2 × precision × recall / (precision + recall)
```

Normal consistency averages the absolute nearest-normal dot product in both directions. The absolute value avoids treating a globally reversed but geometrically identical surface as a total failure.

## Reading the committed result

The adaptive method is best understood as a contamination defense:

- In noise/outlier-heavy pairs, it reduced median Chamfer by 7.4% and increased median F-score by 0.058.
- On clean scans, baseline reconstruction was generally sharper.
- For missing-only geometry, preprocessing did not restore absent regions and sometimes worsened sampled distances.
- Alpha shapes showed the largest outlier sensitivity and therefore the largest gain in the combined setting.
- Poisson was comparatively stable but could close gaps with unsupported surface patches.

These are measured observations for this configuration. They are not statistical evidence over all object classes.

## Reproducibility checklist

- Configuration committed as YAML.
- Fixed seed used for surface sampling and degradation.
- Environment/version metadata saved beside results.
- Every successful and failed run kept in raw output.
- Metric definitions documented and implemented once.
- Reference and reconstructed sampling performed independently.
- Baseline/adaptive cases paired by degraded input.
- Tests cover all three algorithm wrappers.

## Threats to validity

- Two quick-benchmark shapes are not representative of all scanned objects.
- Surface metrics are Monte Carlo estimates and vary slightly with sampling.
- Uniform synthetic outliers are simpler than structured multipath or registration artifacts.
- A missing cap approximates occlusion but not a full scanner trajectory.
- Default hyperparameters are shared for fairness, which is not the same as per-case optimal tuning.
- Runtime results are machine-specific.

