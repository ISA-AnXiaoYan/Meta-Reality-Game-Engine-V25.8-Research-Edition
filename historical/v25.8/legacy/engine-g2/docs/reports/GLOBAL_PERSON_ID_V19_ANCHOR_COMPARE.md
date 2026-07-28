# Global Person ID V19 Anchor Compare

Date: 2026-07-03

Goal: upgrade YOLO post-detection localization from legacy `foot_pixel=bbox center`
to an anchor-point pipeline for high-mounted top-down cameras:

`mask/bbox anchor -> height bias correction -> ground projection -> geometry-weighted fusion -> Global Person ID`

## Implemented

- `global_yolo_infer_server.py`
  - Adds `anchor_mode`:
    - `bbox_center`
    - `bbox_bottom`
    - `mask_robust_center`
    - `mask_eroded_centroid`
    - `auto_topdown_center`
  - Emits:
    - `anchor_pixel_raw`
    - `anchor_pixel_corrected`
    - `anchor_mode`
    - `anchor_quality`
    - `anchor_quality_terms`
    - `anchor_height_correction_k`
    - `anchor_nadir_source`
    - `calibration_weight`
    - `view_geometry_weight`
    - `fusion_weight`
  - Keeps `foot_pixel` as the corrected anchor for compatibility.
  - Loads per-camera nadir files from `calib_true/nadir_camera{camera}.json`.
  - Falls back to image center when the nadir file is absent.

- `global_person_id_server.py`
  - Reads `anchor_quality`, `calibration_weight`, `view_geometry_weight`, and `fusion_weight`.
  - Uses `fusion_weight` for multi-camera observation fusion.
  - Emits `fusion_weight_sum` and `fusion_members` in track diagnostics.

- `tools/compare_global_person_id_filters.py`
  - Adds offline anchor recompute support for existing Person ID datasets.
  - Supports V19-A1/A2 matrix runs without rewriting source datasets.

## Profile Defaults

- `global_yolo_anchor_mode=auto_topdown_center`
- `global_yolo_anchor_height_correction_enable=true`
- `global_yolo_anchor_height_correction_k=0.08`
- `global_yolo_anchor_nadir_pixel_template=calib_true/nadir_camera{camera}.json`
- `global_yolo_include_mask_polygons=true`

## Existing Dataset Limitation

The three current Person ID datasets were recorded after mask removal, so they can verify:

- V19-A1: bbox center + height correction
- V19-A2: bbox center + height correction + geometry fusion weight

They cannot fully verify:

- V19-B: mask robust center + height correction + geometry fusion weight

## Remote Reports

- V18-C selected: `recordings/person_id_filter_compare/compare_20260703_v18c_birth_suppress065/comparison_report.md`
- V19-A1: `recordings/person_id_filter_compare/compare_20260703_v19a1_bbox_height/comparison_report.md`
- V19-A2: `recordings/person_id_filter_compare/compare_20260703_v19a2_bbox_height_geometry/comparison_report.md`

## Official Stream Comparison

| Dataset | Version | exact_% | under_% | over_% | unique_ids | created | step_p95_m | speed_p95_mps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Single person global move | V18-C | 79.97 | 11.88 | 8.16 | 5 | 5 | 0.0745 | 2.5200 |
| Single person global move | V19-A1 | 79.69 | 11.88 | 8.44 | 10 | 11 | 0.0623 | 2.1661 |
| Single person global move | V19-A2 | 79.45 | 11.88 | 8.68 | 10 | 11 | 0.0643 | 2.2439 |
| Two person cross close | V18-C | 58.25 | 7.36 | 34.39 | 48 | 51 | 0.1205 | 3.4867 |
| Two person cross close | V19-A1 | 68.94 | 11.18 | 19.88 | 62 | 71 | 0.0718 | 2.5270 |
| Two person cross close | V19-A2 | 69.96 | 10.95 | 19.09 | 54 | 67 | 0.0722 | 2.5960 |
| Two person no cross | V18-C | 89.27 | 0.58 | 10.15 | 18 | 21 | 0.0636 | 3.1241 |
| Two person no cross | V19-A1 | 90.49 | 0.03 | 9.48 | 26 | 28 | 0.0498 | 2.5159 |
| Two person no cross | V19-A2 | 90.57 | 0.07 | 9.36 | 25 | 29 | 0.0500 | 2.5359 |

## Decision

V19-A2 is the best current anchor path for the existing three datasets:

- It greatly improves the hard two-person cross-close case.
- It reduces over-counting in all multi-person cases.
- It improves movement smoothness relative to V18-C.
- It does increase unique IDs versus V18-C in single/no-cross datasets, so it should not be accepted solely on existing mask-stripped datasets.

Next required validation:

1. Record a new polygon-preserved Person ID dataset.
2. Run V19-B with `auto_topdown_center`/`mask_robust_center`.
3. Check whether mask center restores unique ID count while keeping V19-A2's lower over-count and smoother motion.
