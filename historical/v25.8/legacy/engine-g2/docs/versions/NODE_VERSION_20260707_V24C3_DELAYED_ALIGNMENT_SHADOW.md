# Node Version - 2026-07-07 V24-C3 Delayed Alignment Shadow

## Scope

This node implements the approved CPU-preparation direction for Global BEV:
move delayed presentation selection and diagnostics out of the GUI render
thread, while keeping all OpenGL upload/draw/readback work in the GUI thread.

Branch:

```text
v24-a-bev-event-sync-truth
```

Remote project:

```text
ysxq@192.168.31.115:~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627
```

## Implemented

- Added `DelayedAlignmentWorker` inside `global_bev_fusion_renderer.py`.
- The worker is metadata-only:
  - reads MVS raw-ring slot metadata;
  - reads Event sparse-history sidecar metadata;
  - reads `event_sync_map_ring_{camera}.json`;
  - does not read MVS image payloads;
  - does not read Event sparse point payloads;
  - does not touch OpenGL.
- Added `global_bev_status.json.alignment` and compatibility alias
  `alignment_shadow`.
- Added Status Diagnostics fields for alignment mode, target sync, quality, and
  per-camera MVS/Event state.
- Added launcher/profile keys:
  - `global_bev_alignment_mode`
  - `global_bev_alignment_delay_ms`
  - `global_bev_alignment_shadow_interval_ms`
  - `global_bev_alignment_hold_warn_ms`

## Runtime Contract

Current default:

```text
global_bev_alignment_mode=shadow
global_bev_alignment_delay_ms=820
global_bev_alignment_shadow_interval_ms=250
global_bev_alignment_hold_warn_ms=150
```

`alignment_mode=active` is gated.  If requested, it is downgraded to `shadow`
until remote validation proves the shadow bundle is stable.

OpenGL ownership remains unchanged:

```text
GUI thread owns texture upload, sparse GL upload/draw, FBO render, Qt display,
and PBO readback.
```

## Status Semantics

MVS per-camera states:

- `synced`
- `hold_warn`
- `missing_history`
- `metadata_outlier`
- `reader_error`
- `snapshot_unstable`

Event sparse per-camera states:

- `synced`
- `no_sparse_frame`
- `sync_delta_too_large`
- `error`

The important comparison field is:

```text
global_bev_status.json.alignment.compare_to_active
```

It records whether the shadow delayed bundle selected the same target and MVS
camera slots as the current GUI active path.

## Validation

Local:

- `py_compile` passed for:
  - `global_bev_fusion_renderer.py`
  - `launch_fusion_system.py`
  - `pyopengl_cuda_preview_renderer.py`
- `launch_fusion_system.py --launch-profile ... --help` exposes the new
  `global-bev-alignment-*` keys.

Not yet validated remotely in this node.

## Risks

- Shadow still needs remote validation against real A/B/G/H missing-render
  symptoms.
- If `alignment.mvs.*.missing_history` appears, the next fix is MVS raw-ring
  coverage or metadata timing, not Event sparse tuning.
- If shadow read cost is unexpectedly high, increase
  `global_bev_alignment_shadow_interval_ms`.
- Active bundle switching must not be enabled until shadow target/selection
  mismatch is zero or fully explained.

## Next Remote Acceptance

Run through `tools/remote_test.ps1`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 deploy-worktree -Label v24c3_alignment_shadow
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 start-latest -Label v24c3_alignment_shadow -ReadyWait 180
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 monitor -Label v24c3_alignment_shadow
```

Acceptance checks:

- Global BEV render FPS remains around 30.
- `global_bev_status.json.alignment.state=running`.
- `alignment.compare_to_active.target_match=true`, or mismatch is explained by
  active path delay.
- A/B/G/H missing-render reports classify as MVS background state, not Event
  layer state.
- Event worker CPU and disk write do not increase materially.
