# Node Version 20260708 V24-E5 BEV Event Target Ring Active

## Scope

V24-E5 closes the Global BEV sparse event-layer presentation-worker path.

This node only changes the display path for the Global BEV event overlay.  It
does not change Event HAL capture, Event worker raw tracking, pseudo hit,
final hit, Game Bridge authority, Global Person ID assignment, or the 100 ms
judgement sync threshold.

## Approved State

The approved default for the 627 event-overlay BEV profile is now:

```json
"global_bev_event_layer_presentation_worker_mode": "active",
"global_bev_event_layer_presentation_bundle_ring_size": 8,
"global_bev_event_layer_presentation_prebuild_lookahead_ticks": 2,
"global_bev_event_layer_presentation_prebuild_lookback_ticks": 2,
"global_bev_event_layer_presentation_prebuild_max_per_cycle": 1,
"global_bev_event_layer_presentation_target_tolerance_ticks": 1
```

OpenGL upload and drawing remain in the Global BEV GUI thread.  The worker only
prepares target-keyed sparse event bundles.  GUI consumption is guarded:

```text
exact target bundle -> near target bundle within tolerance -> GUI direct fallback
```

## Implemented Changes

- `EventLayerPresentationWorker` keeps a target-keyed bundle ring instead of
  only exposing the latest bundle.
- Active GUI consumption records exact/near/fallback lookup statistics:
  `bundle_hit_count`, `bundle_miss_count`, `bundle_hit_ratio`,
  `consume_exact_count`, and `consume_near_count`.
- Shadow/compare diagnostics now select bundles by requested target.  A missing
  exact target is reported as `pending_target`, not as a false display
  mismatch.
- Status window, remote compact reports, and node assessment expose compare
  fields:
  `compare_lookup_state`, `compare_target_delta_ticks`,
  `compare_tolerance_ticks`, `compare_exact_count`, `compare_near_count`,
  `compare_miss_count`, and `compare_hit_ratio`.
- Remote assessment rules for V24-E5 treat shadow `pending_target` as WARN
  when there is no display mismatch.
- Project chain documents and remote-test workflow now record the active
  default, fallback rule, and remaining risks.

## Remote Validation

### V24-E5A Shadow

```text
run label: v24e5a_target_ring_shadow_lookahead2
run dir: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24e5a_target_ring_shadow_lookahead2_20260708_124423_display1_keepstdin
archive: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24e5a_target_ring_shadow_lookahead2_20260708_124423_display1_keepstdin.tar.gz
assessment: C:\dev\mrg_remote_tests\v24e5a_target_ring_shadow_lookahead2_20260708_124423_display1_keepstdin_assessment_20260708_125228.assessment.md
```

Result:

- `overall=WARN`
- `runtime_overall_state=OK`
- `status_key_ok=true`
- `display_mismatch_cam_count=0`
- `bundle_ring_count=8`
- `compare_hit_ratio=0.5808`
- final sampled state was `pending_target`, which is a shadow cache timing
  warning, not a display mismatch.

### V24-E5B Active

```text
run label: v24e5b_event_layer_target_ring_active
run dir: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24e5b_event_layer_target_ring_active_20260708_125329_display1_keepstdin
archive: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24e5b_event_layer_target_ring_active_20260708_125329_display1_keepstdin.tar.gz
assessment: C:\dev\mrg_remote_tests\v24e5b_event_layer_target_ring_active_20260708_125329_display1_keepstdin_assessment_20260708_130032.assessment.md
```

Result:

- `overall=WARN`
- `runtime_overall_state=OK`
- `status_key_ok=true`
- `render_fps=30.5665`
- `output_fps=9.8602`
- `gui_paint_total_p95_ms=37.127`
- `gui_upload_event_layer_p95_ms=3.981`
- `event_layer_display_budget_elapsed_ms=0.183`
- `event_layer_display_budget_held_cam_count=0`
- `active_consume_count=2542`
- `fallback_count=24`
- `bundle_hit_ratio=0.9906`
- `consume_exact_count=1410`
- `consume_near_count=1132`
- `display_mismatch_cam_count=0`

The only node warning was `event_sync_outlier_cam_count=2`.  This is a
remaining visual sync robustness item, not a GUI event-layer CPU regression.

## Performance Conclusion

V24-E5B achieved the intended direction:

- GUI event-layer preparation cost is no longer the dominant Global BEV GUI
  burden.
- Active bundle consumption is stable enough for the default profile because
  the hit ratio is above 99 percent and fallback is below 1 percent.
- Global BEV render FPS reached the 30 fps target in the active validation run.

## Rollback

Profile-only rollback:

```json
"global_bev_event_layer_presentation_worker_mode": "shadow"
```

Emergency disable:

```json
"global_bev_event_layer_presentation_worker_enable": false
```

Use rollback if any of the following persist in smoke or stress testing:

- `bundle_hit_ratio < 0.80`
- fallback ratio `> 0.20`
- `display_mismatch_cam_count > 0`
- BEV render FPS falls below the V24 assessment failure threshold
- operators observe event-layer visual lag that does not match status
  `target_delta` / sync outlier diagnostics

## Remaining Risks

- Per-camera Event-to-BEV visual sync outliers remain, especially under strict
  display sync tiers.  E5 reduces GUI load but does not solve all sync outlier
  sources.
- Near-bundle consumption can introduce up to one MVS tick of visual offset.
  This must remain visible through `bundle_target_delta_ticks` and compare
  status fields.
- `prebuild_max_per_cycle=1` is intentionally conservative.  Raising it can
  improve ring exact-hit ratio but may reintroduce worker CPU pressure.
- The same branch also contains V24-Z Global YOLO async postprocess and input
  alignment work.  High-load YOLO validation is still a separate gate.

## Documentation Updated

- `EVENT_SIDE_CHAIN.md`
- `EVENT_SIDE_CHAIN_CN.md`
- `REMOTE_TEST_WORKFLOW.md`
- `pyopengl_cuda_preview_renderer.py` status window fields
- `remote_ops/check_fullsys_run.py` compact fields
- `tools/evaluate_fullsys_node.py` V24-E5 assessment fields

## Closure Decision

Approved as the V24-E5 active display-performance node.

Do not start CUDA or OpenGL multi-thread rendering from this node.  The next
optimization decision should first target remaining visual sync outliers and
high-load YOLO validation.
