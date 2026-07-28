# V24-Z4 Rollback Boundary: Post-Mask Recording Held

Date: 2026-07-08

## Decision

The V24-Z5 post-mask event recording changes and the unfinished V24-Z6 raster-source fix have been rolled back.

The working tree code is back to the V24-Z4 temporary closure baseline. This means:

- Full recording does not include the new `event_postmask_events` target.
- `event_slices_post_polygon_mask.*` is not part of the current accepted runtime contract.
- `tools/offline_filter_event_postmask.py` is removed from the active workspace.
- Event worker, Status Window, remote test workflow, and Evidence Replay contracts are restored to the pre-V24-Z5 state.

## Why

The V24-Z5 remote run proved that the extra post-mask recording target could be started and written, but it did not prove a valid filtered stream.

Observed issue:

- The dedicated post-mask filter still read `sync_ipc/human_result_{camera}.jsonl`.
- It matched masks by the same sync key used by Event worker diagnostics.
- The post-mask index showed `mapped_mvs_sync_index` mainly around `3033-8825`, while parts of `human_result` had sync values around `238xx`.
- Because the two sync coordinate systems did not align, the recorder wrote passthrough slices and `mask_applied_pct` stayed effectively zero.

V24-Z6 started to address this by reading the event-side mask raster, but that implementation was interrupted before completion and validation.

## Current Contract

V24-Z4 remains the accepted boundary:

- Event sensor/HAL raw recording remains the authoritative event raw capture path.
- MVS raw recording remains separate.
- Event mask raster is display-side evidence for BEV / visual overlay.
- Live Event tracking, pseudo hit, final hit, Game Bridge authority, and BEV event display remain unchanged by this rollback.

## Future Re-entry Plan

If post-mask derived event recording is resumed, do not restart from the V24-Z5 JSONL same-sync matcher.

Recommended next direction:

1. Use `/dev/shm/event_human_mask_raster_{camera}.mmap` as the default post-mask recording evidence source.
2. Keep the old `human_result_{camera}.jsonl` same-sync matcher only as an explicit legacy diagnostic mode.
3. Record source diagnostics in index/manifest/status:
   - `mask_source`
   - `mask_raster_seq`
   - `mask_raster_age_ms`
   - `mask_raster_pixels`
   - `mask_checked`
   - `mask_degraded`
   - `mask_passthrough_reason`
4. Fail open when the raster is missing, stale, or being written; never silently drop events.
5. Re-run full recording validation and require nonzero mask availability / drop counters on cameras with live human raster pixels.

## Validation After Rollback

Required before treating this rollback as closed:

- `git status --short --branch` shows only this rollback documentation change unless intentionally staged.
- No active code references to:
  - `event_postmask_events`
  - `event_slices_post_polygon_mask`
  - `offline_filter_event_postmask`
  - `PostMaskRecordingEventFilter`
- Existing Event worker and Status Window files compile from the restored baseline if touched by later work.

