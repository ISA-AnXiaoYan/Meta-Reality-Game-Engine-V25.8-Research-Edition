# Global Person ID V18 Horizontal Compare

Date: 2026-07-03

Scope: compare Global Person ID official alpha-beta and shadow Kalman-CV on three recorded Person ID datasets:

- `20260703_105307_positive_single_person_global_move_20260703_105307`
- `20260703_105917_positive_two_person_cross_close_id_stability_20260703_105917`
- `20260703_110421_positive_two_person_no_cross_separate_id_baseline_20260703_110420`

Remote reports:

- V18-B baseline: `recordings/person_id_filter_compare/compare_20260703_official_vs_shadow_v2/comparison_report.md`
- V18-C default 0.45 trial: `recordings/person_id_filter_compare/compare_20260703_v18c_birth_quarantine/comparison_report.md`
- V18-C suppress 0.55 trial: `recordings/person_id_filter_compare/compare_20260703_v18c_birth_suppress055/comparison_report.md`
- V18-C suppress 0.65 selected: `recordings/person_id_filter_compare/compare_20260703_v18c_birth_suppress065/comparison_report.md`

## Selected V18-C Settings

- `global_person_id_birth_quarantine_enable=true`
- `global_person_id_birth_confirm_hits=2`
- `global_person_id_birth_match_gate_m=0.85`
- `global_person_id_birth_near_track_suppress_m=0.65`
- `global_person_id_birth_candidate_ttl_ms=900`
- `global_person_id_kalman_use_for_association=false`
- Official stream remains `alpha_beta`.
- Shadow stream remains `kalman_cv` for smoothing comparison.

## Official Stream: V18-B vs V18-C Selected

| Dataset | Metric | V18-B | V18-C 0.65 | Direction |
|---|---|---:|---:|---|
| Single person global move | exact_% | 74.65 | 79.97 | improved |
| Single person global move | over_% | 13.47 | 8.16 | improved |
| Single person global move | unique_ids | 12 | 5 | improved |
| Single person global move | created_tracks | 19 | 5 | improved |
| Two person cross close | exact_% | 48.92 | 58.25 | improved |
| Two person cross close | over_% | 44.77 | 34.39 | improved |
| Two person cross close | unique_ids | 148 | 48 | improved |
| Two person cross close | created_tracks | 197 | 51 | improved |
| Two person no cross | exact_% | 77.24 | 89.27 | improved |
| Two person no cross | over_% | 22.71 | 10.15 | improved |
| Two person no cross | unique_ids | 70 | 18 | improved |
| Two person no cross | created_tracks | 106 | 21 | improved |

## Shadow Stream Note

Kalman-CV shadow still has better smoothness but is not selected as the main ID association backend.

Examples from V18-C 0.65:

- Cross-close `speed_p95_mps`: official `3.4867`, shadow `2.6055`
- No-cross `speed_p95_mps`: official `3.1241`, shadow `2.0537`

However, shadow produced more ID-switch suspects in the cross-close case, so Kalman should stay in shadow/output-smoothing evaluation until association gating is redesigned.

## Decision

V18-C selected configuration is a clear improvement over V18-B for ID fragmentation and over-counting on all three datasets. It is suitable for the next live smoke test, with attention to very-close two-player entry cases because stronger birth suppression can theoretically delay a legitimate new player by one or more frames.
