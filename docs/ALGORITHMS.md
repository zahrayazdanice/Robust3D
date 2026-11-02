# Algorithm notes

This document explains the choices implemented in `src/robust3d`. Distances are measured after normalizing each reference object's bounding-box diagonal to one.

## Reconstruction baselines

### Poisson surface reconstruction

Poisson reconstruction treats oriented samples as a vector field and solves for an indicator function whose gradient best matches that field. Extracting an isosurface gives a smooth, usually watertight mesh.

Robust3D exposes octree depth and removes the lowest-density vertices before cropping the result to a padded input bounding box. The method handles noisy samples well but may bridge gaps or close missing regions with invented geometry. Normal orientation is critical.

### Ball Pivoting Algorithm (BPA)

BPA rolls a ball through the point set. When the ball touches three compatible samples without containing another point, it emits a triangle and pivots across frontier edges. Multiple radii, expressed as multiples of measured point spacing, cover more than one local sampling scale.

BPA tends to preserve observed detail and does not automatically fill every hole. It struggles when density varies beyond its radius schedule or normals are inconsistent.

### Alpha shapes

The 3D alpha-shape construction filters a Delaunay tetrahedralization by the radius of its empty circumsphere. The boundary of the retained complex becomes the mesh. Robust3D reports alpha as a multiple of median local spacing, making the control comparable across normalized objects and densities.

Small alpha values fragment; large values erase concavities and approach the convex hull. Outliers can create large, incorrect simplices, so the method benefits strongly from robust preprocessing.

## Proposed adaptive preprocessing

Let `d_i` be the mean distance from point `i` to its `k` nearest neighbors. Robust3D estimates sample spacing with:

```text
s = median_i(d_i)
```

This estimate is less sensitive to a minority of far points than a global mean.

### 1. MAD outlier rejection

The local-scale distribution is summarized with:

```text
m = median_i(d_i)
MAD = median_i(|d_i - m|)
sigma_robust = 1.4826 × MAD
```

A point is kept when `d_i ≤ m + z × sigma_robust`. The default `z = 3.5`. A safety condition disables the rejection if it would remove more than 45% of the cloud.

### 2. Density equalization

A voxel grid whose cell width is `0.65 × s` reduces dense clusters without choosing a model-specific world-unit threshold. This is conservative at typical densities, but it can still discard useful clean samples; the benchmark captures that tradeoff.

### 3. Robust local tangent projection

For each point, Gaussian weights are assigned to its local neighborhood using the median neighbor distance as bandwidth. Weighted PCA estimates a tangent plane. The point moves only 30% of its signed normal distance toward that plane, and the displacement is clamped to 2.5 local bandwidths. One iteration limits shrinkage near holes and edges.

This is a lightweight moving-least-squares-inspired smoother, not a full polynomial MLS surface fit.

### 4. Adaptive normals

Normals are estimated in a hybrid neighborhood with radius `3.5 × s`, normalized, and consistently oriented by tangent-plane propagation. If propagation fails on a degenerate cloud, camera-facing orientation is used as a documented fallback.

## Post-processing

Every output removes duplicate/degenerate triangles, non-manifold edges, and unreferenced vertices. Connected triangle components smaller than 1% of the largest component (and fewer than eight faces) are rejected by default. The threshold is intentionally mild so legitimate disconnected geometry is not automatically erased.

## Complexity

| Stage | Dominant behavior |
|---|---|
| k-NN spacing / rejection | `O(n log n)` tree construction and queries |
| Local PCA projection | `O(n k³)` for tiny 3×3 eigenproblems; practically near-linear for fixed `k` |
| Normal propagation | graph construction and minimum-spanning-tree-like traversal |
| Poisson | depends on occupied octree nodes; depth is the main cost control |
| BPA | spatial-neighbor queries around the active front |
| Alpha shapes | 3D Delaunay tetrahedralization; worst-case superlinear and memory intensive |

## Why the adaptive method is optional

The quick benchmark shows that the proposed stage improves contamination-heavy scans but is not universally beneficial. Clean data can lose fine detail through downsampling or smoothing, while missing-only data has no outlier signal to exploit. Keeping baseline and adaptive modes equally accessible is therefore a deliberate design decision.

