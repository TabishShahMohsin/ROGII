# What the labelling geologist does (StarSteer-style) vs. what our methods assume

Sources: ROGII StarSteer product page; SES "Geosteering Mechanics" (makinhole.com); JPT "Horizontal-Well Correlation in Geosteering Complex Reservoirs of Saudi Arabia".

## Process
1. Type log from a vertical offset well; flatten on a marker (top). Lateral GR is shown in the stratigraphic (TVT/TST) domain.
2. Anchor at the last interpreted point, then interpret in segments: each segment = straight line in (MD, stratigraphic depth), i.e. an apparent dip + shift (the user's chunk m, c).
3. Match FEATURES (peaks/valleys, their order and spacing), not point-wise SSE. Where GR is flat the geologist extrapolates the previous dip and waits for the next marker.
4. Non-uniqueness (up/down-section) is resolved with regional dip, offset-well structure, cuttings/ROP and expertise, with several competing interpretations kept in parallel.
5. Large discrepancies = fault; the framework is updated, not forced through.

## Measured on the data (train, held-out 273 for method numbers)
- Truth in Z+TVT space is ~1.6 straight segments / 1000 ft (median 500 ft, p10 150, p90 1200); median |slope| 0.034; 19% of segments steeper than 0.05, p99 0.10.
- Best stretch/shift of the type log along the true path ~ identity (scale 1.00, shift 0) in most wells: geologist uses the type log as given. Exceptions: 3 of 19 bad wells gain a lot from scale 0.85-1.15.
- GR along the true path correlates only ~0.62 with the type log (prefix: ~0.8). Lateral stays in a thin TVT band (std ~7 ft), so GR carries little signal except at features.
- CORRECTION: the earlier 'true segmentation does not help' test used a 1-ft anchor grid, coarser than the ~1 ft width of the match, so it was invalid. Redone on a 0.25 ft grid (see below): true segmentation gives 5.35 vs 5.17 for uniform 170-ft chunks, so segmentation still does not matter.
- Looser, geologist-like constraints (300-ft chunks, |m|<=0.1): bad wells 26.3 -> 19.2 but typical wells 8.61 -> 9.38; mean with the base ensemble 7.88 on sub40 (small sample).
- Neighbour laterals within 50 ft agree to a median 1.4 ft but ~10% of such pairs differ by >=15 ft (RMS 15): label inconsistency / benches / faults sets a floor.

## Metric study: which score best recovers the geologist's pick? (273 held-out wells)
Test 1 (localisation): shift the TRUE path by delta in [-30,30] ft, score GR vs type log per window, ask if the best delta is 0 (hit within 2 ft; chance 8%).
Window (rows):            300    800    2000
SSE (current Viterbi)     0.38   0.57   0.77
Huber / hetero SSE        0.38   0.57   0.76
log-GR SSE                0.38   0.57   -
prefix-calibrated SSE     0.36   0.54   0.71
Pearson corr              0.26   0.51   0.74
gradient corr             0.19   0.37   0.59
rank corr                 0.22   0.42   0.66
|dF|-weighted corr        0.17   0.33   0.51
type log smoothed in TVT 1/2/4 ft  0.35/0.27/0.18 (300 rows)
envelope (+-1/2/4 ft slide)        0.28/0.20/0.10 (300 rows)
Fine grid, 800-row windows: argmin SSE within 0.5 ft of truth 71%, mean offset 0.03 ft. SSE at +-1 ft is 2.3x the optimum, at +-2 ft 5x: the geologist's pick is a ~1 ft-wide needle in ABSOLUTE GR; any tolerance or shape-only metric is worse.
Test 2 (corridor oracle): Viterbi allowed to place chunk anchors within truth +-15 ft (0.25 ft grid), various losses/regularisers: pooled RMSE 4.98-5.59 (best: L1 loss + slope penalty 20000 = 4.98; SSE 5.35). 45% rows within 1 ft, 15% off by >8 ft; 91% of off rows lie in stretches >300 rows long (alias lock, not noise).
Prior-bounded (stack +-25 ft corridor): 8.7-8.8 alone, 7.4-7.6 averaged with the bag (40-well sample only).

## Neighbour information study (273 held-out wells)
- Coverage: nearest other-well point is <100 ft for 2.5% of lateral rows, 100-200 ft 11%, 200-400 ft 43.5%, 400-800 ft 29%, >800 ft 14%. Bag-4 Viterbi RMSE by distance: 2.4 (<50), 5.6 (50-100), 7.6-8.3 (100-800), 10.6 (800-1600), 14.7 (>1600).
- Cross-well ANCC difference vs separation (different wells): median |d| 1.4 ft (<50 ft), 3.3 (50-100), 6.9 (100-200), 9.9 (200-400), 16 (400-800), 29 (800-1500). Heavy tails (p90 15 ft even <50 ft).
- Neighbour lateral GR as an extra type log: pairs at similar XY and TVT correlate 0.65-0.82 but differ by 10-18 API (type log along truth: 11.5 API). Not better than the type log.
- Structure-only prior (anchored at heel): hold-last 17.1, neighbour surface (NW, h=1000) 14.7, 0.5/0.5 mix 13.4, local Matern-5/2 GP 20.9-42 (worse). Neighbour structure explains ~25% of squared error.
- Per-well self-calibration from own prefix (Viterbi backtest on horizontal tail of prefix): Spearman(prefix RMSE, lateral RMSE) = -0.03..0.06; selecting config by prefix backtest 10.25 vs 8.60 for the plain mean; gating Viterbi vs stack by prefix RMSE worse. Prefix skill does not transfer.
- Corridor Viterbi around the stack prior (+-25 ft, 300-ft chunks), all 273 wells: alone 10.5-10.8; mean with bag4 8.7-8.8; 0.3 blend 8.46 (bag4 8.56, bag4+0.2*stack 8.32). The 40-well gain (7.4) did not replicate.

## Calibration against the competition (private LB top: 5.639, #10: 6.269) and the Kaggle write-ups in videos/
- My earlier "8.3 is roughly the floor" was wrong: the top solutions are far below 8.3 on the board (they use test wells with different difficulty: no-training baseline 12.388 public vs hold-last 15.9 on train).
- Write-ups report CV 7.7-8.2 -> public 6.9-7.1 (ratio ~0.86-0.88). My 8.32 is on 273 harder wells (hold-last 17.15 vs 15.92), roughly 7.7 when normalised: same ballpark as those, not as the top.
- Convergent findings with ours: error = whole-well bias on few wells (worst 10/762 = 29% of SE); candidate selection fails (oracle picks at chance); own-prefix self-calibration does not transfer (r=0.03); neighbour gains do not transfer to isolated wells (they add a spatial-block hold-out, LSO).
- What they do that we do not yet: posterior-MEAN decode over candidates (SEGN, particle filters) instead of one path; row-level GBM combiner with tracker-disagreement features (disagreement quintiles: RMSE 4.1 -> 11.9); heel gain/offset calibration of the log; autoregressive chunked tracking with gain correction (movement direction r~0.93-0.97 but magnitude under-scaled); fixing tracker motion assumptions; LSO validation.
- Quick check here: the blend under-scales displacement from the anchor by ~10% (std ratio 0.84; optimal global gain 1.05-1.1); gain 1.1 on 0.8*bag+0.2*stack: 8.14 in-sample, 8.26 well-grouped CV (vs 8.32).

## Row-level combiner (the community's biggest lever)
Estimators per row: 4 Viterbi variants (base, slope-penalty 30k/300k, shifted chunks), stack OOF prediction (HGB+bi-GRU), plus disagreement (sd, range, gap to stack), position (t, frac, n) and isolation (distance to nearest other well). HGB predicts the residual of 0.8*bag4+0.2*stack. Validation: spatial-block GroupKFold (KMeans regions of well centroids), so a held-out well has no trained neighbours in its region.
- 273 wells, 8 regions: 8.30 -> 7.65 (well-grouped 7.28). Adding 90 stack features hurt (7.87): too few wells.
- All 773 wells (Viterbi variants run on all), 12 regions: baseline 8.44 -> 8.00 (273-subset 8.36 -> 7.67).
- Ablations (273): drop position/isolation feats +0.4; drop individual estimators +0.9; drop disagreement +0.15; drop stack +0.25. On 773: Viterbi-only combiner (no stack) 8.90 vs 7.98 with stack, so the stack is an essential member.
- Posterior-mean-style averaging, shrinkage with distance and isolation-aware trust are what the combiner learns; per-well wins on 45% of wells, worst-19 wells 23.7 -> 19.0.

## StarSteer webinar (ROGII) -> self-reference experiment
- Webinar: strat-based (TVT) correlation; geologist matches lateral GR against its own earlier part when the well re-crosses a layer (lateral log is higher-res than the smooth type log); local/closest reference preferred; constant-dip segments, fractions of a degree apart; AutoSteer is only a 'second opinion' and fails in heterogeneous formations.
- With TRUE TVT: lateral GR vs its own GR at same TVT >=300 ft earlier: residual RMS 9.4 vs 11.0 for type log (86% of lateral rows have such a revisit). Prefix-only reference is no better than type log (11.1 vs 11.3). Cost contrast at true offset vs +-5..12 ft: typelog 3.3x, selfref 4.2x, avg of both 4.2x.
- Two-pass Viterbi (vsr.py: pass1 = anchor Viterbi; pass2 reference = blend of typelog and TVT-profile from prefix + pass1 decode excluding +-300 ft): 40 wells pooled RMSE (weighted avg) 9.47 -> 9.39 (alpha .5), 9.53 (alpha .8); 67% of wells improve (median MSE -3.1) but a few get worse; averaged with pass1 9.33. Verdict: small diversity gain (~0.1); cannot fix whole-well offset (self-consistency only constrains relative offsets between crossings). Not worth ~5h of compute on all 773 wells as a combiner member.

## S = Z+TVT ladder (773 wells, 76,815 rows, spatial-CV 12 regions; ladder*.py)
- hold TVT 15.93 | hold S flat 108.1 (structure dips ~0.03 => S drifts, never hold S flat) | S0+prefix-slope extrapolation 39.6-69 (curved landing section, do NOT use) | S0 + b*zsl*dMD (slope of lateral Z, b~1) 14.6 | + neighbour surface delta (h=1500, local) 12.5 (h=4000: 13.9, worse) | + Viterbi ensemble 9.0 | + stack (linear blend 0.64/0.36) 8.36 | HGB combiner 8.00-8.15.
- Linear leave-one-out from full blend 8.29: drop Viterbi 10.38, drop stack 8.67, drop zsl 8.37, drop nbd 8.28 -> nbd/zsl are redundant once GR estimators exist.
- HGB pure geometry+prefix+neighbour (no GR estimators): 14.8 (TVT residual) vs 14.0 (S-space residual). With estimators: base=0.8mv+0.2stk 8.15; base=S-baseline 12.33 (worse); base=0 8.28. => target space matters only for weak models; combiner base must be the strongest estimator. Extra geometry features cost +0.15.
- By neighbour distance dn: 0-200ft 7.5, 200-500ft 7.3, >500ft 9.8-10.0.

## Physics-aware Viterbi internals (vint.py, combiner4.py; 773 wells)
- vint.py: forward+backward min-marginals over anchors -> posterior per chunk boundary (tempered, tau 30/300), posterior-mean (soft) decode, posterior sd/entropy, alias mass >8ft from MAP, 2nd-mode offset, #modes, cost curvature at optimum, top-5 spread, top1-top2 cost gap, slope/kink, local GR fit rms/corr, type-log sensitivity/local sd, distance to type-log edge.
- Informativeness (row-level Spearman with |MAP err|): far30 +0.49, post_sd30 +0.51, far300 +0.43, post_sd300 +0.39, nmode +0.37, spread +0.23, curv -0.26; fit_rms ~0. Calibration by post_sd300 quintile: MAP RMSE 3.8/4.4/6.3/8.8/17.2. Well-level: gap12 -0.44, far300 +0.51.
- Soft decode beats MAP: single-run RMSE MAP 9.46 -> posterior mean(300) 8.94.
- Combiner (12-region spatial CV, all 773): current 8.00 | + soft-decode estimators 7.94 | + all diagnostics 7.96 | both 7.99 | only physics_fit 8.08, structure 7.96, alias 8.01, uncertainty 8.04. Within noise (~0.05): the existing disagreement features (sd_v, rng_v, gap_S) already carry the same uncertainty. Gain comes from the soft-decode estimator, not the diagnostics.

## Notebook port: 'ZTVT Near Wells' neighbour paths (nbpath.py, nbeval.py, combiner5.py; 773 wells)
- Port: 10 nearest wells by lateral start point (KDTree), similarity = mean min 3D dist + direction parallelism, top 3 (parallel first, nearest); neighbour lateral Z+TVT projected onto target rows by nearest MD ('md') or nearest 3D point ('xyz'); aligned at heel (mean of first 30 rows = known Z+TVT at K-1); TVT_pred = S_proj - Z.
- Standalone pooled RMSE (hold TVT 15.91): md rank1/2/3 20.9/33.7/46.1; xyz 25.9/37.9/44.6; mean-of-3 28.5 (md) / 31.3 (xyz); dist-weighted 27.8/27.0; ORACLE best-of-3 (uses truth) 16.7 (md) / 20.4 (xyz). By rank1 neighbour distance (xyz mean): <150ft 19.8 vs hold 15.3, 150-400 22.1 vs 14.7, 400-800 29.0 vs 15.8, >800 50.5 vs 19.0.
- Combiner (12-region spatial CV): current 8.00 | +md paths 8.09 | +xyz 8.08 | +both 8.08 | +means/gaps only 7.95. No gain.
- Why: neighbours' Z+TVT only matches the target when well inside ~150ft, which is ~3% of rows; elsewhere the projected path is worse than just holding TVT. Information already in the stack via the kernel surface.

## Public write-ups + 'steal from top' experiments (kt1.py, cvol.py, rnet.py, sel2.py, calv.py)
- Public write-ups read (14th keithtyser repo README/submissions/train_unet.py; 192nd zenn; silver andlukyane blog; mycarta toolkit): winners treat alignment as a 2D image (type-log TVT rows x lateral MD columns) read by U-Nets with synthetic-well pretraining and heavy aug; candidate banks (~91 trajectories) + selector/gate; 2nd place 'AnchorCNN' (recursive anchor transitions). Silver (private 7.57): kriged ANCC surface + per-well constant + ML residual + particle filter + stacking. Public->private gaps are 0.3-1.5, so public numbers do not map directly to ours.
- kt1: signed sum-to-one ridge over members 8.26 (nnls 8.24); per-well hard gate comb-vs-stack 8.16 / comb-vs-bag 8.10 (> 8.00; oracle per-well best-of-3 6.13, gate acc 62-73%); +c*(bag-stk) direction 8.03. No gain.
- cvol/rnet: 2D cost-volume refiner (16-row blocks x 97 offsets +-48ft around combiner reference, residual conv net, 5 spatial folds): fold0 6.74 -> 6.74 (net outputs ~0; loss flat). Stopped. No signal beyond combiner at this scale/CPU.
- sel2: candidate bank = top-5 Viterbi paths (base variant), chunk=250 rows, HGB predicts log chunk-MSE per candidate, spatial CV: argmin selector 8.82 vs MAP 9.48 vs closest-to-combiner 8.86 (combiner itself 7.95); oracle best-of-5 per chunk 4.30, per well 5.35 (MAP best in 80% wells). Selector's skill mostly = closeness to the combiner; headroom exists but features cannot find the right alias branch.
- calv: heel-calibrated type log (gain/offset fit on last 600/1500 prefix rows): base 9.03 -> gain+offset 11.95 / offset-only 11.48 / gain-only 13.96 on 120 wells (46% wells better, median diff ~0, big losers). Prefix-zone relation between lateral GR and type log does not transfer to the lateral (median fitted gain 0.74 at clip). Do not use.
