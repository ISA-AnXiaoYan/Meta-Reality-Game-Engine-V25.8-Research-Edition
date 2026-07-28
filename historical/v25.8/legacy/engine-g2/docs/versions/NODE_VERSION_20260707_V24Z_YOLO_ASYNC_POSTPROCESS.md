# Node Version 20260707 V24-Z Global YOLO Async Postprocess

## Scope

V24-Z is the Global YOLO performance node that can run in parallel with the
V24-E Global BEV optimization work.

This node does not change Global BEV rendering, BEV event-layer sync, Hit Judge
authority, Global Person ID assignment logic, or Event capture.  It only changes
the Global YOLO primary-worker publication pipeline and its observability.

## Goals

- Keep `global_yolo_include_mask_polygons=true`.
- Preserve the downstream `human_result_{camera}.jsonl` contract.
- Move polygon, anchor, record build, and publication work off the main
  `model.predict()` loop when enabled.
- Expose enough status fields for remote smoke/assessment to verify whether the
  bottleneck was removed or only moved into a queue.

## Changed Chain

Before:

```text
read_latest -> model.predict -> polygon/anchor/record -> human_result/latest/status/perf
```

After, when `global_yolo_async_record_postprocess_enable=true`:

```text
main thread:
read_latest -> model.predict -> enqueue predicted batch

background postprocess thread:
polygon/anchor/record -> human_result/latest/status/perf
```

The queue is bounded and lossless.  The only supported policy is `block`, so
the main loop waits instead of dropping polygon records.

## Profile Keys

```json
"global_yolo_async_record_postprocess_enable": true,
"global_yolo_async_record_postprocess_queue_size": 2,
"global_yolo_async_record_postprocess_policy": "block"
```

Rollback is a profile-only change:

```json
"global_yolo_async_record_postprocess_enable": false
```

## New Status Fields

Global YOLO worker and merged status expose:

- `fps`
- `inferred_fps_total`
- `published_fps_total`
- `async_record_postprocess_enabled`
- `postprocess_queue_depth`
- `postprocess_lag_ms`
- `postprocess_backpressure_ms`
- `postprocess_dropped_batches`
- `polygon_contract_ok`
- `published_records_with_polygon_ratio`
- `published_persons_with_polygon_ratio`
- `summary`

Remote compact reports and node assessment include the same V24-Z fields.
The preview/status window now reads the canonical `global_yolo_status.json`
path for these fields, and mirrors them into `system_status_latest.json`,
`status_tabs.jsonl`, and `status_summary.csv`.

## Parallel Work Boundary

V24-Z may touch:

- `global_yolo_infer_server.py`
- `global_yolo_merge_publisher.py`
- `launch_fusion_system.py` Global YOLO argument passing
- `launch_profiles/627_event_overlay_bev.json` Global YOLO keys only
- `remote_ops/check_fullsys_run.py`
- `tools/evaluate_fullsys_node.py`

V24-Z must not change:

- `global_bev_fusion_renderer.py`
- BEV event-layer sparse/display-budget semantics
- Event sync-map ring semantics
- Hit Judge authority logic
- Global Person ID algorithm logic

## Validation

Local:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\dev_check.ps1
```

Remote smoke:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 smoke -DeployWorktree -Label v24z_yolo_async_postprocess -Duration 420 -Wait -SummaryAfter -FullSummary
```

Assessment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label v24z_yolo_async_postprocess -AssessmentNode V24-Z
```

## Acceptance Criteria

- `runtime_overall_state=OK`
- `status_key_ok=true`
- `global_yolo.async_record_postprocess_enabled=true`
- `global_yolo.postprocess_dropped_batches=0`
- `global_yolo.polygon_contract_ok=true`
- `global_yolo.published_fps_total >= 300` for first acceptance, target `>=320`
- `global_yolo.postprocess_queue_depth` does not keep growing
- Hit Judge does not show a new material rise in `no_mask_polygon` or
  `no_human_result_near_sync`
- Global Person ID track count and ID-switch indicators do not regress
- Global BEV render/event-sync warnings are recorded as parallel V24-E baseline
  risks unless they materially worsen after V24-Z

## Closure State 2026-07-07

V24-Z0 through V24-Z2 are closed as the mainline implementation and
observability node:

- Async record/postprocess is implemented behind profile keys and enabled in
  the 627 event-overlay BEV profile.
- Mask polygon output remains mandatory and lossless; the configured queue
  policy is `block`, not drop.
- Remote compact assessment, node assessment, status window, status tabs, and
  status summary CSV all expose the V24-Z YOLO metrics.
- No extra optimization is included in this closure.  TensorRT, four-worker
  sharding, `read_latest()` copy reduction, polygon vectorization, and preview
  cadence tuning are deferred until a high-load retest proves they are needed.

## Remote Evidence 2026-07-07

### `v24z_yolo_async_postprocess_mixed_v24e_r3`

- Launcher ended by timeout as expected: `launcher_rc=124`.
- Final stable compact window showed `published_fps_total=312.158`.
- Last 12 samples averaged about `311.86 fps`; last 6 averaged about
  `312.09 fps`.
- `async_record_postprocess_enabled=true`,
  `postprocess_dropped_batches=0`, and `polygon_contract_ok=true`.
- This run validates the V24-Z async publication path, but it did not reproduce
  the original dense human-box pressure.  Final microprofile density was sparse
  at roughly `0.6/0.73` persons per worker batch, so it is not a final
  high-load performance pass.

### `v24z_status_window_links`

- Local validation passed after the status-window link fix:
  `py_compile pyopengl_cuda_preview_renderer.py`,
  `tools/dev_check.ps1`, and `git diff --check` with only existing CRLF
  warnings.
- Remote smoke ended by timeout as expected: `launcher_rc=124`.
- Node assessment result was `WARN`, with
  `runtime_overall_state=OK`, `status_key_ok=true`,
  `async_record_postprocess_enabled=true`,
  `postprocess_dropped_batches=0`, `polygon_contract_ok=true`, and
  `published_persons_with_polygon_ratio=1.0`.
- Assessment compact showed `published_fps_total=286.828` and queue depth `2`.
  A later monitor snapshot from the same run showed
  `published_fps_total=303.108`, `inferred_fps_total=303.108`,
  queue depth `0`, dropped batches `0`, and polygon contract OK.
- Remaining warnings were non-closure items for this node: Event-to-BEV sync
  outliers, MVS upload/display budget warnings, and one sparse-person YOLO FPS
  warning window.  These stay with the parallel V24-E/BEV track or future
  high-load retest.

## Deferred High-Load Retest Gate

Do not claim final high-load closure from sparse-person smoke runs.  The next
V24-Z decision point requires a scene that reproduces the original pressure:

- 8 cameras at about `40 fps` input.
- Dense human boxes close to the original report, for example worker0 around
  `23` boxes per batch and worker1 around `10.8` boxes per batch, or a clearly
  documented equivalent.
- Run a 420s or longer smoke/stress window and then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label <run_label> -AssessmentNode V24-Z
```

Record at least:

- `runtime_overall_state`
- `status_key_ok`
- `global_yolo_published_fps_total`
- `global_yolo_inferred_fps_total`
- `global_yolo_postprocess_queue_depth`
- `global_yolo_postprocess_dropped_batches`
- `global_yolo_polygon_contract_ok`
- `global_yolo_published_persons_with_polygon_ratio`
- worker microprofile density, `predict_ms`, polygon/postprocess time, and
  observed batch interval
- Hit Judge `no_mask_polygon` and `no_human_result_near_sync`

First-pass acceptance remains `published_fps_total >= 300`, with the target at
`>= 320`, queue depth not persistently growing, dropped batches at `0`, and
polygon contract ratio at `1.0`.

## V24-Z3a Input Alignment Audit

V24-Z3a is an audit-only node.  It measures whether visible-light frames are
already aligned well enough for a later full-batch barrier, but it does not
change the current Global YOLO batching, waiting, inference, publication, or
polygon contract.

Enabled profile keys:

```json
"global_yolo_input_alignment_audit_enable": true,
"global_yolo_input_alignment_audit_max_wait_ms": 8.0,
"global_yolo_input_alignment_audit_max_spread_ms": 6.0,
"global_yolo_input_alignment_audit_history": 512
```

New worker/merged status block:

```text
global_yolo.input_alignment_audit
```

Key fields:

- `full_batch_ratio`: how often the existing behavior already emits a full
  group.
- `same_sync_batch_ratio`: how often the emitted batch has one
  `program_sync_index`.
- `current_aligned_full_batch_ratio`: how often the emitted batch is full,
  same-sync, visually aligned, and not frame/meta mismatched.
- `sync_full_sets_within_wait_ratio`: how often a same-sync full set was
  observed within the audit wait threshold.
- `sync_full_wait_ms_avg/max`: estimated wait a later barrier would need.
- `visual_mid_spread_ms_max`: observed exposure-midpoint spread.
- `frames_with_visual_mid_ratio`: whether exposure-midpoint metadata is usable.

Decision rule:

- If `sync_full_sets_within_wait_ratio` is high and `sync_full_wait_ms_avg/p95`
  stays small, V24-Z3b can test a same-sync barrier.
- If `frames_with_visual_mid_ratio` is low, fix/extend MVS exposure metadata
  before making exposure-midpoint decisions.
- If `current_aligned_full_batch_ratio` remains low but
  `sync_full_sets_within_wait_ratio` is high, the problem is likely current
  YOLO polling/collect timing rather than MVS capture itself.

## V24-Z3c Exposure Audit Enhancement

V24-Z3c keeps behavior unchanged when the input buffer is disabled.  It extends
the audit surface so the next smoke run can measure exposure-start,
exposure-midpoint, and exposure-end alignment without adding high-frequency
exposure readback pressure.

New worker CSV:

```text
sync_ipc/global_yolo_worker_<idx>_exposure_audit.csv
```

Key added audit fields:

- `exposure_us`, `exposure_source`, `exposure_start_us`, `visual_mid_us`,
  and `exposure_end_us`.
- `exposure_spread_us`, `visual_mid_spread_ms`,
  `exposure_end_spread_ms`, and `capture_ts_spread_ms`.
- `input_buffer_*` fields, so a V24-Z4 run can be compared against the same
  exposure baseline.

Merged/status fields:

- `global_yolo.input_alignment_audit.visual_mid_spread_ms_p95`
- `global_yolo.input_alignment_audit.capture_ts_spread_ms_p95`
- `global_yolo.input_alignment_audit.exposure_spread_us_p95`
- `global_yolo.input_alignment_audit.exposure_end_spread_ms_p95`
- `global_yolo.input_alignment_audit.per_camera_exposure_us`

Scope decision:

- V24-Z4b is intentionally not included in this node.  Exposure is assumed to
  change slowly enough that the existing metadata is sufficient for the first
  buffered-alignment optimization.

## V24-Z4 Buffered Sync/Exposure-Mid Input Alignment

V24-Z4 is the approved main optimization after V24-Z3c.  The Global YOLO worker
now owns a small per-camera input ring and tries to emit one full same-sync
group whose visible-light exposure midpoint is within the configured spread
budget before calling `model.predict()`.

Enabled profile keys:

```json
"global_yolo_input_buffer_enable": true,
"global_yolo_input_buffer_size": 12,
"global_yolo_input_buffer_max_wait_ms": 30.0,
"global_yolo_input_buffer_max_spread_ms": 12.0,
"global_yolo_input_buffer_exposure_end_barrier_enable": true
```

Fallback order after the wait budget expires:

1. Full same-sync batch even when exposure-midpoint spread exceeded the target.
2. Same-sync partial batch.
3. Latest partial batch.

New status block:

```text
global_yolo.input_buffer
```

Key fields:

- `enabled`
- `batches`
- `aligned_full_batches`
- `fallback_batches`
- `aligned_full_ratio`
- `fallback_ratio`
- `pending_frames`
- per-worker `last_decision`, `last_fallback_reason`, and wait statistics

Expected win condition:

- Reduce partial or mixed-sync YOLO calls.
- Improve effective batch fullness before `model.predict()`.
- Keep added wait bounded by `global_yolo_input_buffer_max_wait_ms`.
- Preserve polygon output and downstream JSON contract.

Rollback:

```powershell
--global-yolo-no-input-buffer
```

or profile override:

```json
"global_yolo_input_buffer_enable": false
```

Smoke acceptance for the first run:

- `global_yolo_input_buffer_enabled=true`
- `global_yolo_input_buffer_aligned_full_ratio` materially higher than the
  V24-Z3a/Z3c `current_aligned_full_batch_ratio`
- `global_yolo_input_buffer_fallback_ratio` not persistently high
- `global_yolo_postprocess_dropped_batches=0`
- `global_yolo_polygon_contract_ok=true`
- no persistent Hit Judge `no_mask_polygon` or mask-latency regression

## Temporary Closure 2026-07-08

The 627 event-overlay BEV launch profile is closed with
`global_yolo_input_buffer_max_spread_ms=12.0` for the current V24-Z4 mainline.
The audit threshold `global_yolo_input_alignment_audit_max_spread_ms=6.0`
remains unchanged as a stricter diagnostic reference; it is not the active
input-buffer admission threshold.

Reason for the 12 ms closure:

- The 8 ms and 12 ms smoke runs had the same latency shape.  Widening the
  buffer spread to 12 ms did not materially reduce the end-to-end
  sync-to-polygon delay, but it is more tolerant of normal auto-exposure and
  per-camera visible-frame timing differences.
- Current polygon availability delay is dominated by the pre-YOLO residual
  before a frame is selected into an inference batch.  In the latest 12 ms
  smoke, sync-start to conservative polygon-published time was about
  `156 ms` p50 and `224 ms` p95; latest exposure-end to polygon-published time
  was about `136 ms` p50 and `204 ms` p95.
- The measured active input-buffer wait is much smaller than that total:
  roughly `5.3 ms` p50 and `10 ms` p95, with rare timeout tails near `30 ms`.
  `model.predict()` is about `19 ms` average and about `21.5 ms` p95.
  Postprocess is about `2 ms`; mask polygon extraction is about `0.7 ms`
  average and about `1.2 ms` p95 in the sparse smoke scene.
- Therefore the current remaining delay is not explained by polygon output,
  JSON publication, or the 12 ms spread setting itself.  The likely unresolved
  chain is MVS frame publish/readability, shared-memory reader cadence, YOLO
  polling phase, and the moment a target sync frame is first seen and selected.

Temporary closure decision:

- Keep the 12 ms input-buffer spread as the fixed operational profile value.
- Do not add V24-Z4b dynamic exposure refresh in this closure; exposure changes
  are assumed slow enough for the current metadata path.
- Do not continue low-load spread tuning.  The next decision point is a
  high-load retest that reproduces the original dense human-box pressure.
- If V24-Z is reopened, add explicit wall-clock trace fields for MVS publish,
  SHM sidecar readability, YOLO first-seen, batch-selected, predict start/end,
  postprocess done, and record publish time.  This is required to split the
  current pre-YOLO residual instead of attributing it to `model.predict()` or
  polygon conversion.

## Known Risks

- Python-threaded polygon simplification may still be GIL-limited.  If the
  queue grows, the next candidate is vectorizing polygon simplification rather
  than disabling polygons.
- Async publication can add a small publish delay.  The queue depth, lag, and
  Hit Judge debug streams must be checked before closing the node.
- Current V24 BEV baseline already has render FPS and Event-to-BEV sync
  warnings.  V24-Z should watch these as regressions, not claim to fix them.
