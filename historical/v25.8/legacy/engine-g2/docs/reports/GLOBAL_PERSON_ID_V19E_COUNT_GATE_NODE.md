# Global Person ID V19-E Count Gate Node

Date: 2026-07-03
Branch: a-test-20260702-v17a-event-overlay-bev-dilate-control

## Scope

This node records the Global Person ID stabilization work through V19-E.

Main changes:
- Anchor/foot point logic was upgraded for high top-down cameras.
- Global Person ID gained handoff-aware matching and strict handoff count lock.
- A business count gate was added: count changes are only allowed near the two short-side entry/exit bands.
- Person ID dataset recording and comparison tooling were expanded for single-person, two-person-crossing, and two-person-non-crossing datasets.

## V19-E Business Rule

Effective business field:
- X range: 0.0 to 30.0 m
- Y range: 0.0 to 20.0 m

Only broad short-edge zones can create or remove IDs:
- Left short-edge zone: x=-3..3, y=-2..22
- Right short-edge zone: x=27..33, y=-2..22

Interior, long-edge, and camera-handoff regions should not create new IDs.

Runtime keys:
- `count_gate_enable`
- `count_gate_x_min_m`
- `count_gate_x_max_m`
- `count_gate_y_min_m`
- `count_gate_y_max_m`
- `count_gate_short_edge_band_m`
- `count_gate_short_edge_y_pad_m`
- `count_gate_startup_bootstrap_ms`
- `count_gate_bootstrap_allow_anywhere_until_track_count`
- `count_gate_non_short_edge_lost_hold_ms`
- `count_gate_non_short_edge_retire_hard_cap_ms`

## Remote Validation

Remote host:
- `ysxq@192.168.31.115`
- Project path: `~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627`

Remote backup before V19-E upload:
- `sync_ipc/remote_backups/v19e_count_gate_before_20260703_144939`

V19-E run directory:
- `sync_ipc/desktop_runs/v19e_count_gate_20260703_145352_display1_keepstdin`

Verified:
- Remote `py_compile` passed for core modified Python files.
- Launch profile JSON passed validation.
- Official and shadow Global Person ID processes both started with `--count-gate-enable`.
- `global_person_status.json` and `global_person_shadow_status.json` expose `count_gate`.
- Global BEV render FPS remained about 30 fps during the check.

## Current Known Issue

Global unique ID stability is still not fully finished.

Current status:
- V19-C/V19-D/V19-E significantly reduced new-ID birth in camera handoff and non-entry regions.
- Remaining ID work is still needed for complex crossing, temporary occlusion, field-edge jitter, and camera-boundary geometry errors.
- The current logic is intentionally business-rule-heavy. It improves operator stability, but it does not fully solve geometric ReID under all cases.

Recommended next steps:
- Continue offline official-vs-shadow comparison on the recorded person ID datasets.
- Tune short-edge band width and bootstrap behavior against real entry/exit cases.
- Add operator-visible count-gate status in the diagnostics panel if it is not already surfaced by the status aggregator.
- Continue improving camera-boundary association using geometry confidence and temporal continuity.
