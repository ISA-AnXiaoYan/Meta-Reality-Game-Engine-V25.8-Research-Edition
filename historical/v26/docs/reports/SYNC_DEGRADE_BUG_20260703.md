# SYNC_DEGRADE_BUG_20260703

## Context

- Date: 2026-07-03 evening live remote run.
- Symptom: event sync health occasionally reports `DEGRADED`, observed on D first and then confirmed as intermittent relock behavior across several event cameras.
- Current run was still alive; MVS soft trigger and status files were refreshing.

## Evidence

- `global_trigger_timeline_latest.json` kept publishing `sync_master=mvs_soft_trigger`, `fps=40.0`, `ok_cams=8/8`.
- MVS category in `system_status_latest.json` was `OK`; per-camera frame/shm ages were fresh, around single-digit to low tens of ms during the checked snapshot.
- `event_sync_health_D.json` showed:
  - `enabled=true`
  - `sync_master=mvs_soft_trigger`
  - `lock_state=DEGRADED`
  - `lock_reason=timeline_offset_jump_relock`
  - `fallback_count=0`
  - `missing_timeline_count=0`
- Code path in `metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py` marks `timeline_offset_jump_relock` when the wall-time nearest MVS trigger is found, but `mapped_mvs_sync_index - event_local_sync_index` changes by more than `event-sync-offset-jump-warn-frames` from the previous offset.
- Recent `event_trigger_map_*.jsonl` samples showed `event_local_sync_index` increasing faster than mapped MVS sync on B-H. This means event-side external-trigger counts are not consistently 1:1 with the MVS soft-trigger timeline.

## Root Cause Assessment

Direct cause:

- The event-side trigger-to-MVS mapping is relocking because the event camera local trigger counter sees extra or non-1:1 trigger edges relative to the 40Hz MVS soft trigger timeline. The timeline exists and fallback is not being used, so this is not a missing MVS timeline or full sync source outage.

Likely contributing causes:

- Event external trigger edge input may be receiving duplicate edges, polarity-related extra events, bounce/noise, or visible-camera sync output edges that are not one pulse per MVS soft trigger.
- Current software relock guard is intentionally strict: `event-sync-offset-jump-warn-frames=2` and wall guard around 20ms. That makes short offset discontinuities visible as `DEGRADED` instead of silently hiding them.
- Startup order or event camera trigger counter reset may amplify offset drift immediately after launch, but the observed repeated relock pattern points more strongly to trigger edge/count mismatch than pure startup offset.

## Current Impact

- Event sync is not fully broken; most mappings still use `timeline_nearest_wall` and many states return to `LOCKED`.
- Affected cameras can intermittently mark sync as degraded. This can reduce trust in event-to-MVS/person evidence timing and may cause hit/mask evidence quality downgrade or inconsistent diagnostics.

## Follow-up Directions

1. Add trigger-edge debounce/period guard in event sync mapper: reject event trigger edges that arrive too soon for a 40Hz source, e.g. less than 12-18ms after the previous accepted edge.
2. Add polarity/edge diagnostics per camera: count raw trigger events by polarity and interval bucket, then verify whether each MVS tick produces exactly one accepted edge.
3. Make health state less sticky/noisy: keep `DEGRADED` for real relock bursts, but report a rolling relock rate and recovered state separately.
4. If hardware allows, verify visible-camera sync output mode: ensure the event camera receives one clean edge per MVS soft trigger, not exposure-start plus exposure-end or repeated strobe edges.

## Important Note

The simultaneous `system_status_latest.summary.overall_state=DEGRADED` also includes non-sync service categories such as `global_bullet_fusion`, `damage_attribution`, `game_event_bridge`, and stale replay evidence. Do not conflate that overall system degraded state with the event sync relock issue above.
