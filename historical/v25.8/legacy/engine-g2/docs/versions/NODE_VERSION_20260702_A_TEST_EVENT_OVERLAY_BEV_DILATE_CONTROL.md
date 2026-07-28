# Node Version - 2026-07-02 A-Test Event Overlay / BEV Dilation Control

## Version Identity

- Node name: `A-TEST-20260702-EVENT-OVERLAY-BEV-DILATE-CONTROL`
- Local worktree: `C:\Users\axyis\Documents\判定开发\workspace_a_global_id_position_hotupdate_pressure_20260702_2010`
- Remote runtime project: `/home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627`
- Remote pre-patch backup: `sync_ipc/full_system_tests/pre_event_overlay_window_bev_dilate_20260702_223131.tar.gz`
- Remote closure archive: `sync_ipc/full_system_tests/node_event_overlay_bev_dilate_closure_20260702_224945.tar.gz`
- Validation run: `sync_ipc/full_system_tests/fullsys_20min_bev_event_layer_20260702_224502_display1_keepstdin`
- Local worktree archive: `C:/Users/axyis/Documents/判定开发/A_TEST_NODE_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL_WORKTREE.zip`
- Local archive checksum: `C:/Users/axyis/Documents/判定开发/A_TEST_NODE_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL_WORKTREE.zip.sha256`

## Scope Closed In This Node

1. Sync Diagnostics status window adds operator controls:
   - `Event独立窗`: hide/show standalone Event Overlay Overview preview.
   - `BEV膨胀px`: hot-switch Global BEV event-layer dilation radius.
2. Global BEV event overlay layer now has upload-side event-pixel dilation:
   - default `16 px`
   - OpenCV backend when available: `cv2_dilate`
   - sparse NumPy fallback when OpenCV is unavailable.
3. Standalone Event Overlay Overview window can be hidden without killing the process:
   - control file: `sync_ipc/event_overlay_overview_control.json`
   - status file: `sync_ipc/event_overlay_overview_status.json`
4. Launcher/profile wiring is complete:
   - passes Event Overview control JSON path
   - passes BEV event-layer dilation defaults.

## Changed Files

- `global_bev_fusion_renderer.py`
- `pyopengl_cuda_preview_renderer.py`
- `event_overlay_overview_renderer.py`
- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`

## Runtime Controls

### Event Overlay Overview Window

Control file:

```json
{
  "event_overlay_overview_visible": true
}
```

Expected status fields:

- `control.visible_requested`
- `control.window_visible`
- `control.error`

### Global BEV Event Layer Dilation

Control file:

```json
{
  "show_event_layer": true,
  "event_layer_dilate_px": 16
}
```

Expected status fields:

- `event_layer.enabled`
- `event_layer.dilate_px`
- `event_layer.dilate_backend`
- `event_layer.dilate_max_pixels`
- `event_layer.dilate_skipped_dense_cam_count`

## Validation Result

Local validation:

- `py_compile` passed for modified Python files.
- `launch_profiles/627_event_overlay_bev.json` passed JSON parsing.

Remote validation:

- Remote `py_compile` passed.
- Remote OpenCV available: `cv2 4.13.0`.
- Global BEV status after patch:
  - `render_fps` around `30 fps`
  - `output_fps` around `10 fps`
  - `last_error` empty
  - `event_layer.enabled=true`
  - `event_layer.dilate_px=16`
  - `event_layer.dilate_backend=cv2_dilate`
  - `event_layer.dilate_skipped_dense_cam_count=0`
- Event Overlay Overview status after patch:
  - `overview_visible=true`
  - `overview_requested=true`
  - `overview_error` empty
- Hot switch verified:
  - overview hidden/restored without killing the process
  - BEV dilation switched `16 -> 32 -> 16`
  - render FPS remained near `30 fps`
- Log scan:
  - Global BEV errors: `0`
  - Event Overview errors: `0`
  - PyOpenGL CUDA Preview GL/CUDA errors: `0`

## Known Risk

1. The Global BEV event overlay is a monitoring/operation visualization layer. It does not change hit detection logic.
2. Event-layer dilation is intentionally capped by `global_bev_event_layer_dilate_max_pixels` to avoid dense event bursts turning into a GPU upload bottleneck.
3. BEV overlay alignment was repaired in this node, but any future calibration or mirror-mode changes must be rechecked against real operator view.
4. The worktree is not a git repository. Current version management is archive/manifest based.

## Next Workstream

Use this node as the continuation point for:

1. Global unique player ID stability.
2. Global real-time player position service.
3. Preview-window hot-update controls.
4. System pressure and long-run optimization.
5. Continued status window key/type compatibility checks.
