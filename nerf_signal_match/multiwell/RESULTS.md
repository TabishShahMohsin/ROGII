# Multi-well experiments (773 train wells; metric = pooled RMSE of lateral TVT, feet, 5-fold CV grouped by well)

Setup: for each well, rows 0..K-1 have known TVT_input; predict TVT for rows K.. (same as the test files). Tops columns (ANCC..BUDA) are never used as inputs for the target well
(TVT = tw_top + ANCC - Z holds to 0.006 ft in train). Neighbour-well surfaces exclude the target well.

| step | pooled RMSE |
|---|---|
| hold last TVT | 15.93 |
| original recipe (linear dip prior) | 42.3 (only works when the driller followed the dip, as in wells 1-2) |
| neighbour structure surface, boosted stack (old kernel) | 12.48 |
| + new surface (quadratic trend + NW residual smoothing, 4 bandwidths) | 11.40 |
| + GR cost-profile features (type-log field) | 11.04 (paired bootstrap gain 0.37 ft, 95% CI 0.16-0.61) |
| + driller Z-path features | 10.87 |
| bi-GRU sequence net on same features | 10.99 ; avg(HGB, GRU) = 10.38 |

By isolation (stack): nearest other well <200 ft: 8.5 | 200-400: 9.2 | 400-800: 11.0 | >800 ft: 15.6 (hold 18.8). Excluding 35 colocated twins: 10.92.

## GR matching engines (all on top of the structural prior; none beat it alone)
- point-wise Student-t HMM: 11.70 -> 11.3-13.4 depending on settings; windowed-correlation HMM: no gain; gradient neural-field refiner: 14.03 -> 13.4-17.5.
- Diagnosis: along the prior path the cost-volume argmax hits the true offset at chance rate (8-11%), although GR at the TRUE TVT correlates 0.64 (median) with the type log.

## NeRF-level changes (held-out 273 wells, GR predicted at true TVT after per-well prefix calibration)
| forward model | RMSE (GR units) | corr median |
|---|---|---|
| type-log lookup (delta kernel) | 11.85 | 0.628 |
| free learned kernel | 11.83 | 0.628 |
| volume-rendering (alpha compositing, PE density MLP) | 11.86 | 0.630 |
| + view (dip) conditioning | 11.83 | 0.633 (kernel widens with dip) |
| shared 1D field across wells F(u) + latents | 15.83 | 0.11 |
| 3D scene F(x,y,u) + latents | 14.52 | 0.29 |
| type-log-conditioned scene (PixelNeRF-style) | 11.54 vs lookup 11.59 | 0.643 vs 0.646 |
End-metric: GR features from the NeRF-rendered kernel: 10.92 vs 10.87 with lookup (no gain).
