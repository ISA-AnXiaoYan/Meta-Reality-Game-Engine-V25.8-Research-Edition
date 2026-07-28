# Global Person ID V19-B Polygon Compare

Date: 2026-07-03

## Scope

Run V19-B on newly recorded polygon-preserved Person ID datasets.

V19-B settings:

- anchor mode: `mask_robust_center`
- height correction: enabled, `k=0.08`
- geometry fusion weight: enabled

Control settings on the same datasets:

- anchor mode: `bbox_center`
- height correction: enabled, `k=0.08`
- geometry fusion weight: enabled

## Datasets

| dataset | expected | duration | notes |
|---|---:|---:|---|
| `20260703_124056_positive_v19b_single_person_polygon` | 1 | 196s | single person, polygon preserved |
| `20260703_124454_positive_v19b_two_person_polygon` | 2 | 287s | two persons, crossing/close interaction |
| `20260703_125023_positive_v19b_two_person_no_cross_polygon` | 2 | 172s | two persons, non-crossing |

All three datasets have `contains_polygon_data=true` and `bad_lines=0`.

## V19-B Results

| dataset | run | exact_% | under_% | over_% | unique_ids | created | birth_suppressed | step_p95_m | speed_p95_mps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | recorded_baseline | 51.62 | 0.00 | 48.38 | 47 | 509 | - | 0.0060 | 4.6625 |
| single | official_replay_alpha_beta | 23.37 | 0.84 | 75.79 | 51 | 53 | 591 | 0.0833 | 2.6844 |
| single | shadow_replay_kalman_cv | 23.67 | 0.84 | 75.49 | 51 | 53 | 591 | 0.0864 | 2.7638 |
| two_cross | recorded_baseline | 16.92 | 33.76 | 49.33 | 117 | 687 | - | 0.0141 | 4.3055 |
| two_cross | official_replay_alpha_beta | 26.53 | 27.37 | 46.10 | 82 | 82 | 2295 | 0.0908 | 2.2225 |
| two_cross | shadow_replay_kalman_cv | 26.20 | 27.60 | 46.20 | 82 | 82 | 2295 | 0.0796 | 2.0791 |
| two_no_cross | recorded_baseline | 0.15 | 19.62 | 80.23 | 46 | 781 | - | 0.0177 | 6.6116 |
| two_no_cross | official_replay_alpha_beta | 1.84 | 20.22 | 77.94 | 51 | 52 | 1202 | 0.0490 | 1.8035 |
| two_no_cross | shadow_replay_kalman_cv | 1.75 | 20.20 | 78.05 | 52 | 52 | 1202 | 0.0416 | 1.5768 |

## Same-Dataset V19-A2 Control

| dataset | run | exact_% | under_% | over_% | unique_ids | created | birth_suppressed | step_p95_m | speed_p95_mps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | official_replay_alpha_beta | 23.09 | 0.77 | 76.14 | 49 | 49 | 595 | 0.0906 | 2.7714 |
| single | shadow_replay_kalman_cv | 23.62 | 0.77 | 75.61 | 49 | 49 | 595 | 0.0871 | 2.8220 |
| two_cross | official_replay_alpha_beta | 24.29 | 27.74 | 47.97 | 81 | 81 | 2145 | 0.1028 | 2.3391 |
| two_cross | shadow_replay_kalman_cv | 23.69 | 27.76 | 48.55 | 80 | 81 | 2145 | 0.0857 | 2.0778 |
| two_no_cross | official_replay_alpha_beta | 2.12 | 20.20 | 77.68 | 47 | 48 | 1622 | 0.0572 | 2.0631 |
| two_no_cross | shadow_replay_kalman_cv | 2.02 | 20.20 | 77.78 | 46 | 48 | 1622 | 0.0441 | 1.6273 |

## Assessment

V19-B is not ready to become the production default.

Compared with V19-A2 on the same polygon datasets:

- V19-B is slightly better on `two_cross` exact rate: `26.53%` vs `24.29%`.
- V19-B is roughly neutral on `single`.
- V19-B is slightly worse on `two_no_cross`: `1.84%` vs `2.12%`.

The larger issue is not the mask anchor alone. The newly recorded datasets show
severe over-publishing and ID fragmentation even in the recorded baseline. The
replay path reduces created tracks significantly, but exact-count stability is
still poor. This points to Global Person ID association/duplicate suppression
and dataset/scene composition as the next debugging focus.

## Output Artifacts

- `recordings/person_id_filter_compare/compare_20260703_v19b_polygon_single`
- `recordings/person_id_filter_compare/compare_20260703_v19b_polygon_two_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19b_polygon_two_no_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19a2_on_polygon_single`
- `recordings/person_id_filter_compare/compare_20260703_v19a2_on_polygon_two_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19a2_on_polygon_two_no_cross`

## Follow-Up

1. Add a dataset inspection pass that reports per-camera `person_count` distribution, per-sync duplicate detections, and extra-person periods.
2. Recheck Global Person ID duplicate suppression gates on polygon datasets before tuning anchor mode further.
3. Treat `two_cross` as the primary V19-B candidate set because it is the only polygon set where mask anchor improved exact rate.
4. Fix recorder summary timing: `latest` after stop currently carries `state=RECORDING` because summary is built before `recording=false`.
