# Global Person ID V19-B2 Anchor Rerun

Date: 2026-07-03

## Scope

Rerun the three polygon-preserved Person ID datasets with additional anchor
selection strategies:

- `auto_topdown_center`
- `mask_eroded_centroid`

Common settings:

- height correction enabled, `k=0.08`
- geometry fusion weight enabled
- expected count: `1` for single-person dataset, `2` for both two-person datasets

## Results

### Auto Topdown Center

`auto_topdown_center` matched the previous `mask_robust_center` results. This
means the polygon evidence was present and valid enough that fallback was not
triggered.

| dataset | run | exact_% | under_% | over_% | unique_ids | created | birth_suppressed | step_p95_m | speed_p95_mps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | official_replay_alpha_beta | 23.37 | 0.84 | 75.79 | 51 | 53 | 591 | 0.0833 | 2.6844 |
| single | shadow_replay_kalman_cv | 23.67 | 0.84 | 75.49 | 51 | 53 | 591 | 0.0864 | 2.7638 |
| two_cross | official_replay_alpha_beta | 26.53 | 27.37 | 46.10 | 82 | 82 | 2295 | 0.0908 | 2.2225 |
| two_cross | shadow_replay_kalman_cv | 26.20 | 27.60 | 46.20 | 82 | 82 | 2295 | 0.0796 | 2.0791 |
| two_no_cross | official_replay_alpha_beta | 1.84 | 20.22 | 77.94 | 51 | 52 | 1202 | 0.0490 | 1.8035 |
| two_no_cross | shadow_replay_kalman_cv | 1.75 | 20.20 | 78.05 | 52 | 52 | 1202 | 0.0416 | 1.5768 |

### Mask Eroded Centroid

`mask_eroded_centroid` did not improve the result. It slightly reduced unique IDs
on the single-person dataset, but did not fix over-publishing. On the
two-person datasets it was neutral to slightly worse.

| dataset | run | exact_% | under_% | over_% | unique_ids | created | birth_suppressed | step_p95_m | speed_p95_mps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | official_replay_alpha_beta | 22.93 | 0.83 | 76.24 | 49 | 51 | 616 | 0.0832 | 2.6775 |
| single | shadow_replay_kalman_cv | 23.47 | 0.83 | 75.70 | 49 | 51 | 616 | 0.0863 | 2.7794 |
| two_cross | official_replay_alpha_beta | 25.66 | 27.72 | 46.63 | 83 | 83 | 2148 | 0.0892 | 2.2181 |
| two_cross | shadow_replay_kalman_cv | 25.89 | 27.58 | 46.54 | 82 | 83 | 2148 | 0.0777 | 2.0928 |
| two_no_cross | official_replay_alpha_beta | 1.84 | 20.26 | 77.90 | 56 | 56 | 1295 | 0.0474 | 1.7778 |
| two_no_cross | shadow_replay_kalman_cv | 1.74 | 20.25 | 78.02 | 56 | 56 | 1295 | 0.0408 | 1.5651 |

## Assessment

The existing alternate anchor modes do not solve the current Global Person ID
instability.

Findings:

- `auto_topdown_center` is effectively identical to `mask_robust_center` on the
  new polygon datasets because polygon fallback is not being exercised.
- `mask_eroded_centroid` is not a better production candidate.
- The main failure mode remains over-publishing/fragmentation rather than a pure
  anchor-point selection issue.

## Recommendation

Do not switch production default to `mask_eroded_centroid`.

Keep `auto_topdown_center` as the safe semantic default, but the next real
optimization should be an adaptive hybrid policy, not another static anchor
mode. Suggested policy:

1. Compute both `bbox_center` and polygon anchor.
2. Measure their image-space and ground-space delta.
3. Use polygon anchor only when quality is high and delta is below a gate.
4. Fall back to corrected bbox center when polygon anchor is unstable, edge
   clipped, or strongly disagrees with projection geometry.
5. Feed anchor disagreement into `anchor_quality` and `fusion_weight`, so weak
   polygon observations contribute less to cluster formation.

This should be evaluated with the same three polygon datasets before changing
production defaults.

## Output Artifacts

- `recordings/person_id_filter_compare/compare_20260703_v19b2_auto_anchor_single`
- `recordings/person_id_filter_compare/compare_20260703_v19b2_auto_anchor_two_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19b2_auto_anchor_two_no_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19b2_eroded_anchor_single`
- `recordings/person_id_filter_compare/compare_20260703_v19b2_eroded_anchor_two_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19b2_eroded_anchor_two_no_cross`
