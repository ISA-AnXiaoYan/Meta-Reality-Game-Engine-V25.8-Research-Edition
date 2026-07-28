# Node Version - 2026-07-05 V24-B Sparse GL Event Display, Not Closed

## Scope

This record documents the current V24-B sparse OpenGL event-layer work as an
unclosed node. It is not a release/closure point.

- Branch: `v24-a-bev-event-sync-truth`
- Remote project:
  `ysxq@192.168.31.115:~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627`
- Launch profile: `launch_profiles/627_event_overlay_bev.json`
- Remote acceptance run:
  `sync_ipc/desktop_runs/v24b4_sparse_yfix_acceptance_20260705_213821_display1_keepstdin`
- Previous acceptance run:
  `sync_ipc/desktop_runs/v24b3_manual_acceptance_20260705_212959_display1_keepstdin`

## Implemented Work

### V24-B sparse event display path

- Native HAL broker writes display-only sparse event history after the human-mask
  visual branch.
- Global BEV can use `event_layer_backend=sparse_gl`.
- Global BEV reads `event_overlay_sparse_{camera}_history.json`, selects frames
  with the V24-A `event_sync_map_ring_{camera}.json`, projects event pixels
  through Event->MVS and MVS->field homographies, and draws them as OpenGL
  points in the BEV FBO.
- The Status Diagnostics window exposes texture/sparse backend control.

### V24-B2/B3 performance fixes

- Event sync ring records are cached in sorted form for sparse scoring.
- Initial sparse candidate scan was reduced from 8 to 4, with fallback scanning
  when no candidate is usable.
- `sparse_gl` auto-degrade now guards on `render_fbo` draw cost instead of CPU
  sparse read/upload cost.

### V24-B4 Y-axis repair

- Fixed sparse point projection Y direction to match the BEV shader's
  `display_to_sample_field_xy()` convention:
  `ndc_y = (1.0 - qy) * 2.0 - 1.0`.
- This repaired the observed vertical inversion between the sparse event layer
  and the MVS/BEV background.

## Validation So Far

Local checks passed:

- `py_compile` for `global_bev_fusion_renderer.py`
- `git diff --check` with CRLF warnings only

Remote checks performed:

- Native C++ broker compiled successfully on the remote Metavision 4.6.2 system.
- `v24b2_sparse_gl_smoke_20260705_211217_display1_keepstdin` recovered Global
  BEV render to about 29-30fps.
- `v24b3_sparse_gl_guard_smoke_20260705_211712_display1_keepstdin` kept
  `effective_dilate_px=16`, `sparse_gl_draw_points>0`, sparse read p95 about
  3.7ms, sparse projection p95 about 0.2ms, and `render_fbo` p95 about 0.9ms.
- `v24b4_sparse_yfix_acceptance_20260705_213821_display1_keepstdin` started
  successfully and reached `8/8` event workers ready.

## Why This Node Is Not Closed

Operator acceptance found remaining BEV event-layer display issues:

1. Event sparse point dilation can visually change size.
2. B camera can appear delayed against the MVS visible-light background.

These are display-path issues. They do not change final-hit evidence,
hit-judge authority, raw event tracking, or the C++ HAL capture path.

## Current Diagnosis

### 1. Dilation size can appear unstable

Observed state:

- `event_layer_backend=sparse_gl`
- `event_layer_dilate_px=16`
- `effective_dilate_px` can be controlled by auto-degrade.
- Runtime state on the remote still contained
  `global_bev_runtime_state.json.event_layer_dilate_px=4`.
- Current control state contained
  `global_bev_control.json.event_layer_dilate_px=16`.

Likely cause:

- Runtime state and control state disagree at startup or after hot control.
- Auto-degrade remains enabled with threshold around 28fps. When BEV render FPS
  drops below the threshold, the global OpenGL point size can be reduced.
- The auto-degrade path has no hysteresis, so visual size can jump around near
  the threshold.

Important implementation note:

- Sparse GL uses one global `GL.glPointSize(point_size)`.
- There is no per-camera point-size algorithm. If point size changes, it is a
  global display state change, not a B-camera-only render path.

### 2. B camera can appear delayed

Observed state:

- B worker was alive and `LOCKED`.
- B sparse read cost was low, around `0.5-0.6ms`.
- B sync source was `event_sync_map_ring_interpolated` and quality was `good`.
- B event-to-BEV delta was seen near the display threshold, for example about
  `+18ms` and later about `-15ms`.
- Display threshold is currently `20ms`.
- Other cameras can also exceed the threshold; examples seen during live
  diagnosis included C/G hidden by `event_mvs_sync_delta_too_large`.

Likely cause:

- B is not blocked. It is near the strict visual sync threshold.
- V24-B2 reduced sparse candidate scan to 4 for performance; when the inverse
  event-sync hint is slightly off, 4 candidates may miss the best matching
  sparse frame.
- Event sync offset jump counters are high across several cameras, so the
  sparse frame selection must tolerate small ring/offset jitter.

### 3. Status freshness requires verification

During diagnosis, `global_bev_status.json` and `test_run_status_latest.json`
mtime appeared to stop briefly while the live system was still running. This may
be normal interval behavior, SMB cache delay, or a real status-update stall.

This must be checked before using status fields as final proof during operator
acceptance.

### 4. Runtime-state authority is still confusing

The launch profile and command line may be overwritten by
`sync_ipc/global_bev_runtime_state.json`.

Examples seen during this node:

- Profile/command path expected a current display transform, while runtime logs
  showed the loaded runtime state applying display controls.
- Runtime state still had `event_layer_dilate_px=4` after newer tests expected
  16px.

This is a recurring source of field confusion and should be cleaned before
closure.

## Open Work Before Closure

Current V24-C2 path, started 2026-07-07:

- V24-C2-0 is the low-risk closure prep:
  - keep first-pass sparse candidate scan at 4;
  - expand to 32 candidates when no usable candidate is found or the best delta
    is near the configured visual sync threshold;
  - stop restoring event-layer display controls from runtime state by default;
  - keep sparse GL point size stable unless sparse draw cost is actually high;
  - add Global BEV `published_wall_us` status freshness.
- V24-C2-A is diagnostic-only shadow validation:
  - a separate sparse selector worker recomputes selected frame/sync delta;
  - the shadow selector reads sidecar metadata only, not sparse point mmap data;
  - sync ring lookups use cached keys and binary search;
  - the profile uses a 1000ms interval so shadow stays diagnostic-only;
  - it reports `event_layer.shadow.*`;
  - it does not touch active OpenGL drawing or final-hit logic.
- V24-C2-B active worker must not start until shadow reports matching target and
  no unexplained mismatch.
- Remote C2-A result: `v24c2a_shadow_1s_20260707_144523_display1_keepstdin`
  reached live OK with `render_fps=30.73`, `effective_dilate_px=16`,
  `event_layer.shadow.compare_state=match`, `mismatch_cam_count=0`, and status
  freshness below 1s in the sampled state.
- Active-worker caution: BEV `target_sync_id` advances every render frame.  A
  worker that only moves the event layer can produce one-target-late overlays
  unless it shares the same presentation target contract as MVS selection.

Recommended next fixes:

1. Stabilize sparse point size.
   - Either disable auto-degrade for `sparse_gl`, or add hysteresis and a hold
     time before changing `effective_dilate_px`.
   - Clear or normalize `global_bev_runtime_state.json` on startup so stale
     `event_layer_dilate_px=4` cannot override a 16px acceptance test.

2. Improve sparse frame selection.
   - Keep the fast 4-candidate scan only as the first pass.
   - If best `event_to_bev_sync_delta_ms` is above about 10ms or close to the
     configured display threshold, rescan 16 or 32 candidates.
   - Add status fields for the final candidate count, best absolute delta, and
     whether escalation scanning was used.

3. Add explicit B/C/G sync diagnostics.
   - Track per-camera delta min/p50/p95/max over a rolling window.
   - Expose offset jump count, out-of-guard count, latest trigger-to-event edge
     delay, and sparse selected frame id in the status window.

4. Verify status freshness.
   - Add status `published_wall_us`/age checks for Global BEV and
     `test_run_status_latest`.
   - Make the operator window show stale status as stale, not just the last
     successful value.

5. Re-run acceptance.
   - Start through `tools/remote_test.ps1`.
   - Confirm event layer and MVS are vertically aligned.
   - Confirm point size stays stable under motion.
   - Confirm B route has no visible delay and per-camera delta stays inside the
     configured display target.

## Handoff Rules

1. Do not tag this node as a closure version.
2. Keep the current remote run archived for analysis, but do not use it as
   acceptance evidence.
3. Continue remote operations through `tools/remote_test.ps1`.
4. When deploying the next fix, record:
   - run label
   - active `global_bev_runtime_state.json`
   - active `global_bev_control.json`
   - per-camera event-to-BEV delta table
   - Global BEV `performance_guard`
5. Update `EVENT_SIDE_CHAIN.md` and `EVENT_SIDE_CHAIN_CN.md` again when the
   point-size and B-sync issues are fixed or explicitly accepted.
