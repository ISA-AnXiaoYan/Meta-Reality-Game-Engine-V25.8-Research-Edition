# Node Version 20260707 V24-C5/C6 BEV Upload And Display Budget

## Scope

This node continues the V24-C Global BEV synchronization/performance line after
the V24-C4 preflight and alignment diagnostics node.

The goal is to reduce Global BEV GUI-thread stalls without changing sensing,
tracking, hit judgment, or game-output authority:

- V24-C5: MVS texture upload de-duplication and per-paint upload budget.
- V24-C6: Event sparse-layer display preparation budget and short visual hold.

## V24-C5 MVS Texture Upload Budget

Changed chain:

```text
MVS raw shared memory
-> Global BEV selected frame
-> OpenGL texture upload in GUI thread
-> Global BEV FBO/render/output
```

Upload identity is now:

```text
camera + tick_id + frame_id + slot + width + height + pixel_format
```

Behavior:

- If the selected MVS frame has the same upload identity as the existing OpenGL
  texture, skip duplicate upload.
- If the per-paint upload budget is exhausted and a previous texture exists,
  keep the previous texture for that camera instead of blocking the GUI thread.
- If a camera has no previous texture, force the first upload even when the
  budget is already exhausted, avoiding black cameras.
- Capture, MVS trigger, YOLO, Preview, Hit Judge, Event sync, and Game Bridge
  contracts are unchanged.

New profile/control keys:

| Key | Default |
| --- | --- |
| `global_bev_mvs_upload_budget_enable` / `mvs_upload_budget_enable` | `true` |
| `global_bev_mvs_upload_budget_ms` / `mvs_upload_budget_ms` | `10` |

New status:

| Field | Meaning |
| --- | --- |
| `global_bev_status.json.mvs_texture_upload` | Per-paint upload summary |
| `mvs_texture_upload.uploaded_cams` | Cameras uploaded in the latest paint |
| `mvs_texture_upload.duplicate_skip_cams` | Cameras skipped because upload key did not change |
| `mvs_texture_upload.held_by_budget_cams` | Cameras reusing previous texture because the upload budget was exhausted |
| `alignment.performance.mvs_upload_*` | Compact mirror for status and remote reports |

## V24-C6 Event-Layer Display Budget

Changed chain:

```text
Event sparse history sidecar/mmap
-> Global BEV sparse frame selection
-> sparse point projection in GUI paint tick
-> OpenGL sparse point draw
```

Behavior:

- Applies to the display side, especially `global_bev_event_layer_backend=sparse_gl`.
- If per-paint event-layer read/select/project work exceeds the display budget,
  a camera may briefly reuse the previous already-projected sparse points.
- This is a visual hold only. Event HAL capture, FIFO, Event worker tracking,
  masked overlay publication, pseudo hit, final hit, and Game Bridge are not
  throttled or dropped by this guard.

New profile/control keys:

| Key | Default |
| --- | --- |
| `global_bev_event_layer_display_budget_enable` / `event_layer_display_budget_enable` | `true` |
| `global_bev_event_layer_display_budget_ms` / `event_layer_display_budget_ms` | `10` |
| `global_bev_event_layer_display_hold_ms` / `event_layer_display_hold_ms` | `280` |

New status:

| Field | Meaning |
| --- | --- |
| `event_layer.display_budget_enable` | Display budget guard active flag |
| `event_layer.display_budget_ms` | Configured budget |
| `event_layer.display_budget_elapsed_ms` | Latest event-layer preparation elapsed time |
| `event_layer.display_budget_held_cams` | Cameras using previous projected sparse points |
| `event_layer.per_camera.*.reason=held_by_display_budget` | Per-camera visual hold reason |
| `alignment.performance.event_layer_display_budget_*` | Compact mirror for status and remote reports |

## Documentation Updates

Updated living documents:

- `MVS_CAMERA_CHAIN.md`: V24-C5 upload budget contract and revision row.
- `EVENT_SIDE_CHAIN.md`: V24-C6 event-layer display budget contract and revision row.
- `EVENT_SIDE_CHAIN_CN.md`: Chinese V24-C6 contract and revision row.
- `REMOTE_TEST_WORKFLOW.md`: performance guard checks and PowerShell/SSH quoting
  prevention guidance.

Future BEV/Event/MVS chain changes must keep these documents in sync before a
node is considered closed.

## Validation Plan

Local:

- Python syntax check for `global_bev_fusion_renderer.py`,
  `launch_fusion_system.py`, `pyopengl_cuda_preview_renderer.py`, and
  `remote_ops/check_fullsys_run.py`.
- Launch profile JSON parse.
- Launcher `--help` exposes the new Global BEV budget args.
- `git diff --check`.

Remote:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 smoke -DeployWorktree -Label v24c6_bev_upload_budget -Duration 420 -Wait -SummaryAfter -FullSummary
```

Acceptance checks:

- Runtime starts successfully and exits by timeout.
- `status_key_ok=true`.
- `status_freshness.global_bev_from_this_run=true`.
- Global BEV render FPS remains close to 30 fps.
- `mvs_texture_upload` exists and reports upload/duplicate/held lists.
- `event_layer.display_budget_*` exists and is mirrored into alignment performance.
- Event worker, Hit Judge, pseudo hit/final hit, and Game Bridge chains are not
  changed by this display-only node.

## Risks

- If `mvs_upload_budget_ms` is too strict, Global BEV may show older MVS
  textures for some cameras. Diagnose with `mvs_texture_upload.held_by_budget_cams`
  and MVS frame freshness before blaming camera capture.
- If `event_layer_display_budget_ms` is too strict, sparse event pixels may be
  visually held for a short time. Diagnose with `display_budget_held_cams`,
  `display_hold_age_ms`, and event-to-BEV sync delta before blaming Event capture.
- The optimization intentionally keeps OpenGL upload/draw in the GUI thread.
  It only prevents repeated or over-budget work inside that thread; a later node
  may still need deeper GPU/FBO/PBO scheduling work.
- Local GUI help for `global_bev_fusion_renderer.py` requires PySide6 and may not
  run on the Windows dev environment. Remote runtime is the authoritative GUI
  environment.

## Remote Evidence

Remote smoke 1:

```text
label: v24c6_bev_upload_budget_20260707_175801_display1_keepstdin
archive: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24c6_bev_upload_budget_20260707_175801_display1_keepstdin.tar.gz
launcher_rc: 124
runtime_overall_state: OK
status_key_ok: true
global_bev_from_this_run: true
global_bev_render_fps: about 29.2-29.8
global_bev_output_fps: about 9.7-9.9
alignment.mvs_synced_cam_count: 8
alignment.mvs_missing_history_cam_count: 0
alignment.mvs_upload_elapsed_ms: 6.045
alignment.mvs_upload_held_by_budget_cam_count: 0
```

The first smoke proved the runtime behavior, but the compact report showed
`global_bev.mvs_texture_upload=null` because `remote_ops/check_fullsys_run.py`
did not copy the raw `global_bev_status.json.mvs_texture_upload` object into its
collected snapshot.  This was a report/key-alignment bug, not a runtime BEV
upload bug.

Remote smoke 2 after report fix:

```text
label: v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin
archive: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin.tar.gz
launcher_rc: 124
runtime_overall_state: OK
runtime_broken_links: []
runtime_degraded_links: []
status_key_ok: true
global_bev_from_this_run: true
mask_stats_jsonl_no_growth: true
global_bev_render_fps: 28.0088
global_bev_output_fps: 9.6582
event_layer_dilate_backend: opengl_points
event_layer_dilate_px: 16
mvs_texture_upload.elapsed_ms: 8.085
mvs_texture_upload.uploaded_cam_count: 7
mvs_texture_upload.held_by_budget_cam_count: 0
event_layer_display_budget_ms: 10
event_layer_display_budget_elapsed_ms: 20.265
event_layer_display_budget_held_cam_count: 2
alignment.mvs_synced_cam_count: 8
alignment.mvs_missing_history_cam_count: 0
alignment.mvs_metadata_outlier_cam_count: 0
alignment.event_synced_cam_count: 5
alignment.event_sync_outlier_cam_count: 3
alignment.gui_upload_selected_p95_ms: 15.955
alignment.gui_upload_event_layer_p95_ms: 22.238
alignment.gui_paint_total_p95_ms: 56.602
```

Assessment:

- V24-C5/C6 status keys and compact report are now aligned.
- MVS upload budget did not hold any camera in the accepted smoke and stayed
  below the 10 ms budget in the compact sample.
- Event-layer display budget is active and did hold cameras visually when the
  display-side work exceeded budget.
- GUI p95 timings improved versus the V24-C4 hotspot target direction, but the
  second compact sample reported Global BEV render around 28 fps.  Treat this as
  acceptable for this node but continue watching it in the next 20-30 minute
  stress test.
- Event-to-BEV sync outliers still appear independently of the display-budget
  node.  They should remain in the next optimization queue and must not be
  confused with MVS texture upload loss or Event capture loss.
