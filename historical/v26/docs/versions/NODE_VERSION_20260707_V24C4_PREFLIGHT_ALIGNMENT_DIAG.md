# Node Version - 2026-07-07 V24-C4 Preflight And Alignment Diagnostics

## Scope

This node closes the failure mode exposed by the first V24-C3 remote smoke:
an old `desktop_latest` process can keep camera handles and keep writing stale
status files while a new smoke test is starting.

This node also deepens the Global BEV delayed-alignment diagnostics so the next
remote run can separate:

- stale/foreign status files;
- MVS raw-ring history or metadata offset;
- Event sparse-frame sync offset;
- Global BEV GUI upload/render cost;
- alignment worker CPU cost.

## Implemented

- Added `remote_ops/preflight_clean_runtime.sh`.
  - Stops stale full-system launcher processes before a smoke/stress run.
  - Stops stale Global BEV, Global YOLO, Event worker, Hit Judge, bridge, and
    native HAL broker children.
  - Cleans native HAL FIFO temporary files and stale Event sync/sparse sidecars.
  - Moves stale core status files into
    `sync_ipc/preflight_stale_status/<timestamp>_<label>/` before launch.
  - Writes `preflight_clean_summary.json` and `preflight_clean.log` into the run
    directory.
- `tools/remote_test.ps1` now syncs the preflight script.
- `remote_ops/run_fullsys_test.sh` runs preflight by default before launching
  the stack.
- `remote_ops/check_fullsys_run.py` now reports:
  - whether `global_bev_status.json` was published after the current run start;
  - whether `global_bev_status.json.alignment` exists;
  - compact alignment summary in `test_report.stdout.json`.
- `global_bev_status.json.alignment` now includes:
  - `mvs_offset_ticks_by_cam`;
  - `mvs_offset_ms_by_cam`;
  - `event_delta_ticks_by_cam`;
  - `event_delta_ms_by_cam`;
  - worker elapsed p50/p95/max;
  - GUI p95 for alignment refresh, MVS upload, Event upload, FBO render, and
    paint total.
- Sparse Event sidecar/mmap startup races now degrade to a recorded
  `last_read_error` instead of surfacing as an opaque per-camera error.

## Runtime Contract

The default remote smoke/stress path is now:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 smoke -DeployWorktree -Label <label> -Duration <seconds> -Wait -SummaryAfter
```

The run script performs a preflight clean unless explicitly disabled:

```bash
PREFLIGHT_CLEAN=0 bash remote_ops/run_fullsys_test.sh <label> <duration_s>
```

Do not disable preflight for normal smoke/stress validation.  Disable it only
when the test intentionally coexists with an already-running desktop/latest
system.

## Status Semantics

Important compact report fields:

```text
status_freshness.global_bev_from_this_run
status_freshness.global_bev_alignment_present
global_bev.alignment.mvs_offset_ms_by_cam
global_bev.alignment.event_delta_ms_by_cam
global_bev.alignment.gui_upload_selected_p95_ms
global_bev.alignment.gui_upload_event_layer_p95_ms
global_bev.alignment.gui_paint_total_p95_ms
```

Interpretation:

- `global_bev_from_this_run=false` means the report is contaminated by a stale
  status writer or a launch failure before Global BEV published status.
- Large `mvs_offset_ms_by_cam` means the MVS raw ring or MVS metadata is the
  next investigation target.
- Large `event_delta_ms_by_cam` with synced MVS means Event sparse selection or
  sync-map timing is the next investigation target.
- High GUI p95 with low worker p95 means moving more CPU metadata work is not
  enough; upload/draw/readback needs optimization.

## Validation

Local:

- `py_compile` passed for:
  - `global_bev_fusion_renderer.py`
  - `pyopengl_cuda_preview_renderer.py`
  - `remote_ops/check_fullsys_run.py`
- `launch_profiles/627_event_overlay_bev.json` parsed successfully.
- `git diff --check` reported only existing CRLF warnings.

Remote:

- `v24c4_preflight_alignment_diag_20260707_170807_display1_keepstdin`
  completed with `launcher_rc=124`, `runtime_overall_state=OK`, no broken or
  degraded links, and `status_key_ok=true`.
- `status_freshness.global_bev_from_this_run=true` and
  `global_bev_alignment_present=true`, proving the stale-status contamination
  guard is working for this run.
- Preflight artifact:
  `preflight_clean_summary.json` reported no leftover launcher/Global BEV/HAL
  processes after cleanup.
- Alignment summary:
  - MVS 8/8 synced;
  - MVS missing history 0;
  - MVS metadata outlier 0;
  - Event 8/8 synced;
  - Event sync outlier 0.
- Average BEV render FPS was about `29.05`.
- Remaining performance risk: GUI p95 is still high in this run
  (`upload_selected` about `22 ms`, `upload_event_layer_textures` about `24 ms`,
  `paint_total` about `61 ms`).  `refresh_alignment` stayed below `1 ms`, so the
  next performance work should focus on GUI upload/draw/readback, not the
  metadata-only alignment refresh.

## Risks

- Preflight cleanup is intentionally broad for smoke/stress runs.  Do not use it
  for tests that need to keep an already-running desktop/latest system alive.
- This node is diagnostic-first.  It does not switch delayed alignment to active
  rendering.
- If G/H remain `missing_history`, the fix belongs in MVS raw-ring coverage or
  metadata timing, not Event sparse tuning.
