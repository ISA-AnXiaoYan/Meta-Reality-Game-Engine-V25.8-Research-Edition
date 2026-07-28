# Global Person ID V19-C Handoff-Aware Stability

Date: 2026-07-03

## Problem

After switching `foot_pixel` to the new top-down anchor path, the main visible
failure is ID confusion at camera handoff areas. The cause is residual parallax:
two cameras can project the same person to ground points that are farther apart
than the normal clustering/birth gates.

## Change

V19-C adds handoff-aware gates to Global Person ID:

- `cross_camera_cluster_gate_m`: mild cross-camera clustering tolerance.
- `handoff_match_gate_m`: expanded match gate only when an observation introduces
  a camera the track has not recently seen.
- `handoff_birth_near_track_suppress_m`: suppress new ID birth near an existing
  track only during camera handoff.
- `handoff_duplicate_suppress_dist_m`: publish-side duplicate suppression only
  expands for likely handoff duplicates.

The normal same-camera and non-handoff gates remain conservative.

Production profile values:

| key | value |
|---|---:|
| `global_person_id_cluster_gate_m` | 0.65 |
| `global_person_id_cross_camera_cluster_gate_m` | 0.75 |
| `global_person_id_match_gate_m` | 1.25 |
| `global_person_id_handoff_match_gate_m` | 1.35 |
| `global_person_id_birth_near_track_suppress_m` | 0.65 |
| `global_person_id_handoff_birth_near_track_suppress_m` | 1.25 |
| `global_person_id_duplicate_suppress_dist_m` | 0.90 |
| `global_person_id_handoff_duplicate_suppress_dist_m` | 1.35 |

## Offline Results

Dataset set: V19-B polygon-preserved Person ID datasets.

Anchor path:

- `auto_topdown_center`
- height correction enabled, `k=0.08`
- geometry fusion enabled

### V19-B Baseline

| dataset | exact_% | under_% | over_% | unique_ids | created |
|---|---:|---:|---:|---:|---:|
| single | 23.37 | 0.84 | 75.79 | 51 | 53 |
| two_cross | 26.53 | 27.37 | 46.10 | 82 | 82 |
| two_no_cross | 1.84 | 20.22 | 77.94 | 51 | 52 |

### V19-C Dynamic Handoff-Aware

| dataset | exact_% | under_% | over_% | unique_ids | created | handoff target |
|---|---:|---:|---:|---:|---:|---|
| single | 29.36 | 0.92 | 69.72 | 43 | 43 | fewer duplicate births |
| two_cross | 27.68 | 30.52 | 41.80 | 62 | 63 | less ID fragmentation at overlap |
| two_no_cross | 2.69 | 20.80 | 76.51 | 35 | 35 | fewer split IDs |

### Static Sweep Reference

The strongest static sweep was:

- `cluster_gate_m=0.75`
- `birth_near_track_suppress_m=1.25`
- `duplicate_suppress_dist_m=1.35`
- `match_gate_m=1.35`
- `lost_match_gate_m=2.1`

It produced slightly lower `unique_ids`, but it applies suppression globally and
therefore has higher risk when a new real player appears close to an existing
player. V19-C chooses dynamic handoff-aware gates instead.

## Risk

- Handoff duplicate suppression can hide a real second person if they enter a
  handoff zone very close to another person.
- Two-person crossing still has elevated under-count. This is expected because
  stronger stability reduces over-publish at the cost of temporarily hiding
  close duplicate-looking tracks.
- The current handoff detector is camera-set based, not geometry-zone based.
  Future work should add calibrated overlap-zone masks or per-camera adjacency.

## Validation Artifacts

- `recordings/person_id_filter_compare/compare_20260703_v19c_dynamic_handoff_single`
- `recordings/person_id_filter_compare/compare_20260703_v19c_dynamic_handoff_two_cross`
- `recordings/person_id_filter_compare/compare_20260703_v19c_dynamic_handoff_two_no_cross`

## Runtime Diagnostics

The status `stats` now includes:

- `handoff_cluster_merges`
- `handoff_match_gate_used`
- `handoff_birth_suppressed`
- `handoff_duplicate_suppressed`

These counters should rise when people move through camera handoff areas.
