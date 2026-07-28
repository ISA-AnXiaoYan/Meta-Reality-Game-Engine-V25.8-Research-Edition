# Node Version - 2026-07-02 V17-A + A-Test Event Overlay / BEV Control

## Branch

- Branch: `a-test-20260702-v17a-event-overlay-bev-dilate-control`
- Base branch: `a-test-20260702-event-overlay-bev-dilate-control`
- Remote target: `origin/a-test-20260702-v17a-event-overlay-bev-dilate-control`

## Merge Inputs

### V17-A Diagnostic Log Throttle

Purpose:

- Disable high-volume `event_human_mask_stats_*.jsonl` by default.
- Keep in-memory/status counters available.
- Add max-FPS and flush safeguards for targeted diagnostics.

Primary files:

- `launch_fusion_system.py`
- `metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py`
- `cpp_bullet_core/metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py`
- `ids_8cam_fusion_config.json`
- `ids_8cam_fusion_config_627_runtime.json`
- `archive/config_snapshots/ids_8cam_fusion_config.before_disable_bgr_preview.json`
- `launch_profiles/627_event_overlay_bev.json`

### A-Test Event Overlay / BEV Dilation Control

Purpose:

- Add Sync Diagnostics controls for the standalone Event Overlay Overview window.
- Add Global BEV event-layer dilation hot switch.
- Keep the standalone Event Overlay Overview process alive while hiding/showing the window.

Primary files:

- `global_bev_fusion_renderer.py`
- `pyopengl_cuda_preview_renderer.py`
- `event_overlay_overview_renderer.py`
- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`

## Conflict Analysis

No code-level conflict was found.

The only shared files are:

- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`

The two nodes touch different runtime keys and command-line paths:

- V17-A keys:
  - `event_human_mask_stats_jsonl`
  - `event_human_mask_stats_max_fps`
  - `event_human_mask_stats_flush_ms`
- A-Test keys:
  - `event_overlay_overview_control_json`
  - `global_bev_event_layer_dilate_px`
  - `global_bev_event_layer_dilate_max_pixels`

The merged version uses the latest validated worktree files, which contain both sets of keys.

## Validation Required After Pull

1. `python -m py_compile` for modified Python files.
2. JSON parse check for:
   - `launch_profiles/627_event_overlay_bev.json`
   - `ids_8cam_fusion_config.json`
   - `ids_8cam_fusion_config_627_runtime.json`
3. Runtime status checks:
   - `event_human_mask_stats_*.jsonl` is not created by default.
   - Event Overlay Overview can hide/show from Sync Diagnostics.
   - Global BEV event layer reports `dilate_backend` and `dilate_px`.
   - Global BEV render FPS remains near the accepted baseline.

## Risk Notes

- This merge combines operational visibility/control and diagnostic I/O throttling. It should not change final hit logic.
- If targeted human-mask delay diagnosis is needed, explicitly enable the JSONL diagnostic path and keep its max FPS low.
- BEV event-layer dilation is capped by `global_bev_event_layer_dilate_max_pixels` to avoid dense event bursts becoming a rendering/upload hotspot.
