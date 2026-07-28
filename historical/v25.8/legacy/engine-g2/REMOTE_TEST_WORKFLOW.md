# Remote Test Workflow

This workflow avoids fragile Windows PowerShell to Linux Bash one-liners.
Keep complex logic inside `remote_ops/` on the Linux test host. Use
`tools/remote_test.ps1` only as a thin trigger.

`tools/remote_test.ps1` uploads each remote command as a temporary Bash script
under `/tmp` and then runs `bash <script>` over SSH. This avoids the common
PowerShell/SSH quoting failure where pipes, quotes, `$()` or heredocs are
stripped or reinterpreted before they reach the Linux shell.

## Common Commands

Run from the Git repo root on Windows:

```powershell
.\tools\remote_test.ps1 sync-ops
.\tools\remote_test.ps1 deploy
.\tools\remote_test.ps1 validate
.\tools\remote_test.ps1 list-datasets -DatasetLimit 20
.\tools\remote_test.ps1 check-dataset-manifest -DatasetDir recordings/<session> -DatasetLabel battle_1v1 -ExperimentNote "operator notes"
.\tools\remote_test.ps1 check-dataset-manifest -DatasetDir recordings/<session> -PrintResolvedManifest
.\tools\remote_test.ps1 v25-replay-control -DatasetDir recordings/<session> -ReplayControlMode full_replay -ReplayTiming realtime
.\tools\remote_test.ps1 smoke-mvs-broker -DatasetDir recordings/20260708_181936_full -MaxFrames 160
.\tools\remote_test.ps1 smoke -Duration 370 -Label smoke_remote_ops
.\tools\remote_test.ps1 smoke -Duration 370 -Label full_record_smoke -FullRecording
.\tools\remote_test.ps1 summary -Label smoke_remote_ops
.\tools\remote_test.ps1 assess -Label smoke_remote_ops -AssessmentNode V24-D
.\tools\remote_test.ps1 fetch -Label smoke_remote_ops
.\tools\remote_test.ps1 stress -Duration 1800 -Label stress_remote_ops
.\tools\remote_test.ps1 stop
.\tools\remote_test.ps1 stop -Label stress_remote_ops
.\tools\remote_test.ps1 start-latest -Label desktop_latest
.\tools\remote_test.ps1 monitor -Label desktop_latest
.\tools\remote_test.ps1 hot-update -BboxStartExpandPx 48 -HumanNoiseMode soft -AuthorityMode strong_reasoning
.\tools\remote_test.ps1 stop-latest -Label desktop_latest
.\tools\remote_test.ps1 install-desktop-shortcuts
.\tools\remote_test.ps1 open-startup-panel
.\tools\remote_test.ps1 restart-startup-panel
.\tools\remote_test.ps1 check-startup-panel
```

`install-desktop-shortcuts` installs the managed start, stop, and startup
control-panel entries on the remote desktop. Opening the startup panel does not
start the system. The operator must press `一键启动整套系统` after validating the
source mode and dataset.

The button writes `sync_ipc/v25_panel_launch_status.json` and calls the managed
`remote_ops/start_latest_system.sh` path. Treat the start as ready only when the
status reports `state=ready` (or `already_running_ready`) with a live
`launcher_pid`, a valid `run_dir`, and `ready_returncode=0`.

`check-startup-panel` validates this contract in dry-run mode and never starts
the full system.

If PowerShell blocks script execution on this workstation, use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 validate
```

## Remote Defaults

- Host: `ysxq@192.168.31.115`
- Project: `~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627`
- Profile: `launch_profiles/627_event_overlay_bev.json`
- Python: `~/miniconda3/envs/ids_recv/bin/python3`

## What The Runner Checks

- Full-system launcher return code.
- Runtime controls are initialized before launch:
  - Hit detection enabled.
  - Final hit authority reset to `strong_gate`; `strong_reasoning` stays enabled
    as shadow-only by default.
  - Event Overlay Overview visible.
  - Global BEV event layer enabled.
  - Global BEV event layer dilation set to 16 px.
- `event_human_mask_stats_*.jsonl` before/after size and mtime.
- Global BEV render/output FPS and `event_layer` status.
- Event Overlay Overview control/status.
- Status key audit for the unified status snapshot and Global BEV aliases
  (`event_layer`, `event_bullet`, `global_person`, plus legacy `*_overlay`
  compatibility).
- Status freshness via JSON mtime/age.
- Launcher logs and status artifacts are packed with the run.

## V25 Source-Mode And Replay Checks

V25 source-mode work must continue to use `tools/remote_test.ps1` and
`remote_ops/`; do not run ad hoc replay/startup diagnostics as long inline SSH
commands from PowerShell.

For source-mode tests, archive and compare:

- `list-datasets` output for the selected dataset folder;
- generated launch profile or profile overlay;
- effective source-mode summary, including `event_source_mode`,
  `mvs_source_mode`, and the run-archived `source_mode_effective.json`;
- dataset manifest and manifest validation report;
- resolved dataset manifest, especially `replay_manifest_resolved.json` or
  run-archived `v25_resolved_manifest_latest.json`;
- Event file broker logs and per-camera FIFO freshness;
- Event broker stats JSONL summary, including `source_mode`,
  `sync_policy`, A-H `valid_for_replay_sync`, `triggers_seen`, and
  `frames_written`;
- MVS file broker logs and per-camera frame/shm freshness;
- `runtime_overview_reconciled_links` from `check_fullsys_run.py summary`.
- `runtime_worker_reconciled_links` from `check_fullsys_run.py summary`.

The reconciliation list is only allowed to explain display/status sampling
false positives, such as a stale `system_status_latest.json` sample followed by
a fresh Event Overlay Overview row. It must not hide real worker, sync, MVS,
YOLO, hit-judge, or bridge failures.

`runtime_worker_reconciled_links` is stricter than overview reconciliation. It
is only allowed for shutdown-sampling false positives such as
`event:H:worker`, and only when the same archived run contains both a fresh
drawn Event Overlay Overview row and a worker-owned sidecar such as
`event_overlay_H_latest.json` or `event_viz_H_latest.json` published inside the
run start/end window. It must not be used to clear worker failures without a
fresh worker sidecar.

V25-B broker stats note: nonzero `write_errors` / `exceptions` must be reviewed
with per-camera `last_error`. A clean timeout run may still count FIFO writes
interrupted by shutdown; follow-up work should classify those separately from
runtime FIFO failures instead of treating the aggregate `errors` number alone
as pass/fail.

V25-D/E productized replay workflow:

- `check-dataset-manifest -DatasetDir <dataset>` runs the V25 checker and writes
  `replay_manifest_resolved.json` next to the selected dataset folder. Use
  `-DatasetLabel` and `-ExperimentNote` to preserve operator-facing scenario
  notes in the resolved manifest.
- Add `-PrintResolvedManifest` when the operator needs to inspect the resolved
  manifest from Windows.  This avoids fragile one-off SSH/Powershell quoting.
- `v25-replay-control -DatasetDir <dataset> -ReplayControlMode full_replay`
  refreshes:
  - `sync_ipc/v25_replay_control.json`
  - `sync_ipc/v25_replay_status.json`
  - `sync_ipc/v25_replay_launch_command.sh`
- `smoke-full-replay -DatasetDir <dataset>` now passes the universal
  `--dataset-dir` to `launch_fusion_system.py` in addition to the explicit
  Event/MVS broker args.  Reports should prefer `source_mode_effective.json`
  over profile intent when deciding what source mode actually ran.
- The real status window must expose the same runtime truth through
  `system_status_latest.json.categories.source_mode_replay` and the
  operator-facing `启动/回放源` tab.  Before closing a V25 source-mode node,
  confirm:
  - `status_key_audit.ok=true`;
  - `system_status.categories.source_mode_replay.event_source_mode_effective`;
  - `system_status.categories.source_mode_replay.mvs_source_mode_effective`;
  - `source_mode_effective_exists=true`;
  - `v25_resolved_manifest_exists=true` for any file replay run.
- Run archives must include `source_mode_effective.json`,
  `v25_resolved_manifest_latest.json`, `v25_replay_control.json`,
  `v25_replay_status.json`, and `v25_replay_launch_command.sh` when present.
  Do not judge source mode from the launch profile alone.

Current V25 reference smoke:

```text
event_h_worker_infer_final_smoke_20260709_171733_display1_keepstdin
runtime_overall_state=OK
runtime_broken_links=[]
runtime_degraded_links=[]
```

This run validates that the H route fallback/inferred-active status fix did not
break the live `event_source_mode=live_hal` production path. It does not by
itself close Event RAW file replay or MVS file replay acceptance.

## V24-C5/C6 BEV Performance Guard Checks

For Global BEV upload/display-budget nodes, verify these fields from the run
summary or `sync_ipc/global_bev_status.json` before judging the test:

- `status_freshness.global_bev_from_this_run` must be true.
- `runtime_overall_state` and `status_key_ok` should be OK/true.
- `global_bev.render_fps` should remain close to 30 fps unless the test is a
  deliberate stress case.
- `global_bev.mvs_texture_upload.uploaded_cams`,
  `duplicate_skip_cams`, and `held_by_budget_cams` show whether MVS texture
  uploads are being skipped or held by the V24-C5 guard.
- `global_bev.event_layer_display_budget_ms`,
  `event_layer_display_budget_elapsed_ms`, and
  `event_layer_display_budget_held_cam_count` show whether the V24-C6
  event-layer display guard is holding sparse points.
- `alignment.performance.mvs_upload_*` and
  `alignment.performance.event_layer_display_budget_*` are compact mirrors for
  dashboard/status-window checks.

Holding a previous texture or sparse-point set is a display-side overload
protection, not proof that MVS capture, Event capture, raw tracking, pseudo hit,
or final hit dropped data.  Cross-check frame-id freshness, Event worker status,
and Hit Judge streams before assigning root cause.

## V24-E4 BEV Event-Layer Presentation Worker Checks

V24-E4 intentionally keeps OpenGL upload and drawing in the Global BEV GUI
thread.  It only moves sparse event-layer CPU preparation into a background
worker.  V24-E4-A shadow is metadata-only and must not read sparse point
payloads or project vertices; V24-E4-B active is the first mode that may build
worker-prepared vertices for GUI-thread upload.

Validation order is mandatory:

1. Deploy/run V24-E4-A with
   `global_bev_event_layer_presentation_worker_mode=shadow`.
2. Assess with `-AssessmentNode V24-E4A`.
3. Only if shadow passes, switch to
   `global_bev_event_layer_presentation_worker_mode=active`.
4. Deploy/run V24-E4-B and assess with `-AssessmentNode V24-E4B`.

Compact report fields:

- `global_bev.event_layer_presentation_worker.enabled`
- `global_bev.event_layer_presentation_worker.mode`
- `global_bev.event_layer_presentation_worker.state`
- `global_bev.event_layer_presentation_worker.payload_policy`
- `global_bev.event_layer_presentation_worker.read_points`
- `global_bev.event_layer_presentation_worker.compare_state`
- `global_bev.event_layer_presentation_worker.display_compare_state`
- `global_bev.event_layer_presentation_worker.delta_compare_state`
- `global_bev.event_layer_presentation_worker.mismatch_cam_count`
- `global_bev.event_layer_presentation_worker.display_mismatch_cam_count`
- `global_bev.event_layer_presentation_worker.delta_mismatch_cam_count`
- `global_bev.event_layer_presentation_worker.build_p95_ms`
- `global_bev.event_layer_presentation_worker.active_consume_elapsed_ms`
- `global_bev.event_layer_presentation_worker.fallback_count`
- `global_bev.event_layer_presentation_worker.updated_age_ms`
- `global_bev.event_layer_presentation_worker.target_delta_ticks`

Pass expectations:

- V24-E4A shadow: `mode=shadow`,
  `payload_policy=metadata_only_no_vertices`, `display_compare_state=match`,
  `display_mismatch_cam_count=0`, Global BEV render FPS near the previous
  V24-E baseline, and no new broken/degraded links.  Delta mismatch fields are
  diagnostics and should be investigated as warnings, not treated as display
  mismatch by themselves.
- V24-E4B active: `mode=active`, recent `state=active_consumed`,
  low or explainable `fallback_count`, no BEV FPS regression, and Event-to-BEV
  sync outliers no worse than the shadow run.

If active fallback grows continuously, keep the profile in `shadow` and tune
worker cadence/target tolerance before considering any GPU rendering direction.
Do not use OpenGL multi-threading or CUDA rendering as part of V24-E4.

Current V24-E4 default is `shadow`.  The 2026-07-08 V24-E4B active smoke showed
that active consumption can work, but fallback was still too frequent for a
production default.  Treat active mode as a hot-switch experiment until
`active_consume_count` is consistently high, `fallback_count` is low, and BEV
render/paint metrics are no worse than shadow.

## V24-E5 BEV Event-Layer Target Ring Checks

V24-E5 keeps the V24-E4 OpenGL contract: OpenGL upload and drawing remain in
the Global BEV GUI thread.  The new part is a target-keyed worker bundle ring
used by active fallback lookup.

Validation order:

1. Run V24-E5-A with
   `global_bev_event_layer_presentation_worker_mode=shadow`.
2. Assess with `-AssessmentNode V24-E5A`.
3. Only if shadow ring has no display mismatch, switch to
   `global_bev_event_layer_presentation_worker_mode=active`.
4. Run V24-E5-B and assess with `-AssessmentNode V24-E5B`.

Additional compact fields:

- `global_bev.event_layer_presentation_worker.bundle_ring_count`
- `global_bev.event_layer_presentation_worker.bundle_ring_size`
- `global_bev.event_layer_presentation_worker.cached_target_min`
- `global_bev.event_layer_presentation_worker.cached_target_max`
- `global_bev.event_layer_presentation_worker.bundle_hit_count`
- `global_bev.event_layer_presentation_worker.bundle_miss_count`
- `global_bev.event_layer_presentation_worker.bundle_hit_ratio`
- `global_bev.event_layer_presentation_worker.consume_exact_count`
- `global_bev.event_layer_presentation_worker.consume_near_count`
- `global_bev.event_layer_presentation_worker.compare_lookup_state`
- `global_bev.event_layer_presentation_worker.compare_hit_ratio`
- `global_bev.event_layer_presentation_worker.compare_exact_count`
- `global_bev.event_layer_presentation_worker.compare_miss_count`

Pass expectations:

- V24-E5A shadow: `mode=shadow`, `display_mismatch_cam_count=0`, and
  `bundle_ring_count>0`.  `display_compare_state=pending_target` is a warning
  when `compare_lookup_state=miss`; it means the sampled status tick did not
  have an exact target bundle, not that the GUI would draw the wrong frame.
- V24-E5B active: `mode=active`, `active_consume_count>0`, high
  `bundle_hit_ratio`, reduced `fallback_count` compared with V24-E4B, no
  `display_mismatch_cam_count`, and no BEV FPS regression.

Current V24-E5B accepted default:

- `global_bev_event_layer_presentation_worker_mode=active`
- `global_bev_event_layer_presentation_prebuild_lookahead_ticks=2`
- `global_bev_event_layer_presentation_prebuild_max_per_cycle=1`
- `global_bev_event_layer_presentation_target_tolerance_ticks=1`
- Keep fallback enabled; if `bundle_hit_ratio` falls below `0.80` or fallback
  ratio rises above `0.20`, switch back to shadow before deeper tuning.

Do not raise `event_layer_presentation_prebuild_max_per_cycle` above `1` during
the first E5B validation unless build p95 is already clearly below budget.

## Node Assessment

After every smoke, stress, dataset-recording, or latest-system stop summary,
run the node assessment step:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label <run_label> -AssessmentNode <node_name>
```

The command reads the remote compact summary through `remote_ops/check_fullsys_run.py`
and writes local UTF-8 files under `C:\dev\mrg_remote_tests` by default:

```text
<run>_assessment_<timestamp>.compact.json
<run>_assessment_<timestamp>.assessment.json
<run>_assessment_<timestamp>.assessment.md
```

Assessment result meanings:

| Result | Meaning |
| --- | --- |
| `PASS` | Node guardrails passed with no findings |
| `WARN` | Runtime is usable, but follow-up risks exist, such as render FPS near the warning threshold or Event-to-BEV sync outliers |
| `FAIL` | Node cannot be closed without repair, for example broken links, stale Global BEV status, missing required status keys, MVS missing-history, or render FPS below the fail threshold |

For V24-C and later BEV nodes, a `WARN` caused only by Event-to-BEV sync
outliers should be recorded as a follow-up synchronization risk.  Do not treat
it as proof that MVS upload budget or Event capture dropped data.

## V24-Z Global YOLO Async/Postprocess And Input Buffer Checks

For the V24-Z Global YOLO async postprocess node, run the normal smoke or
stress command, then assess with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label <run_label> -AssessmentNode V24-Z
```

Record these compact fields before judging the node:

- `runtime_overall_state`
- `status_key_ok`
- `global_yolo_published_fps_total`
- `global_yolo_inferred_fps_total`
- `global_yolo_async_record_postprocess_enabled`
- `global_yolo_postprocess_queue_depth`
- `global_yolo_postprocess_dropped_batches`
- `global_yolo_polygon_contract_ok`
- `global_yolo_published_persons_with_polygon_ratio`
- `global_yolo.input_alignment_audit.full_batch_ratio`
- `global_yolo.input_alignment_audit.same_sync_batch_ratio`
- `global_yolo.input_alignment_audit.current_aligned_full_batch_ratio`
- `global_yolo.input_alignment_audit.sync_full_sets_within_wait_ratio`
- `global_yolo.input_alignment_audit.sync_full_wait_ms_avg`
- `global_yolo.input_alignment_audit.visual_mid_spread_ms_max`
- `global_yolo.input_alignment_audit.visual_mid_spread_ms_p95`
- `global_yolo.input_alignment_audit.exposure_spread_us_p95`
- `global_yolo.input_buffer.enabled`
- `global_yolo.input_buffer.aligned_full_ratio`
- `global_yolo.input_buffer.fallback_ratio`

The status truth source is `sync_ipc/global_yolo_status.json`.  The preview
status window, `system_status_latest.json`, `status_tabs.jsonl`, and
`status_summary.csv` are expected to mirror these same V24-Z fields.

A sparse-person smoke run can close wiring, publication, and status-link
correctness, but it is not a high-load YOLO performance closure.  For a final
throughput decision, reproduce the dense human-box pressure from the regression
report or document an equivalent load, then compare worker person density,
`predict_ms`, polygon/postprocess time, observed batch interval, queue depth,
drop count, and Hit Judge mask-related regressions.

For V24-Z3a/Z3c input-alignment audit, do not treat the audit itself as a
pass/fail gate.  Use it to judge the problem size and to compare V24-Z4
buffered alignment against the previous no-behavior-change baseline.  A useful
signal is high `sync_full_sets_within_wait_ratio` with low
`sync_full_wait_ms_avg`, low `visual_mid_spread_ms_p95`, and low
`exposure_spread_us_p95`.

For V24-Z4, the archive should include:

- `sync_ipc/global_yolo_worker_*_perf.csv`
- `sync_ipc/global_yolo_worker_*_exposure_audit.csv`
- `sync_ipc/global_yolo_status.json`

Current temporary closure value:

- `global_yolo_input_buffer_max_spread_ms=12.0` in
  `launch_profiles/627_event_overlay_bev.json`
- `global_yolo_input_alignment_audit_max_spread_ms=6.0` remains the stricter
  diagnostic reference and is not the active input-buffer admission threshold.

Do not reopen low-load spread tuning unless the next run shows a material
Global YOLO, Hit Judge mask, or Global Person ID regression.  The remaining
V24-Z performance decision is the separate dense-human high-load retest.

## Live Monitor And Hot Update

Use `monitor` for a stable machine-readable live status refresh. It writes and
then points to:

```text
sync_ipc/test_run_status_latest.json
sync_ipc/test_run_events.jsonl
```

Use `hot-update` for runtime experiment parameters. It writes:

```text
sync_ipc/experiment_control.json
sync_ipc/experiment_status_latest.json
sync_ipc/experiment_events.jsonl
```

Examples:

```powershell
.\tools\remote_test.ps1 monitor -Label desktop_latest

.\tools\remote_test.ps1 hot-update `
  -ExperimentId v22_bbox48_live `
  -ExperimentStage V22-hit-soft-filter `
  -HitProfile experiments\hit_judge\bbox48_soft.json `
  -GlobalIdProfile experiments\global_id\alpha_beta_countgate.json `
  -GlobalPersonTarget shadow

.\tools\remote_test.ps1 hot-update -BboxStartExpandPx 96 -HumanNoiseMode soft -AuthorityMode strong_reasoning
```

Terminal output is only an operation acknowledgement. Treat the JSON files above
and `sync_ipc/system_status_latest.json` as the status truth source.

## Latest-System Node Verification

For a field-style node check, prefer the latest-system flow. It keeps the system
running until an explicit stop and avoids manual SSH process cleanup:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 start-latest -Label <node_label> -ReadyWait 180
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 monitor -Label <node_label>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 stop-latest -Label <node_label> -StopWait 90
```

Use the stop summary as the node evidence. Record at least:

- run label
- run directory
- archive path
- launcher rc
- `runtime_overall_state`
- `status_key_ok`
- Global BEV render/output FPS
- any chain-specific status keys added by the node

When the local worktree is mixed, do not use `deploy-worktree` as a shortcut.
Either commit first and deploy the committed branch/snapshot, or copy only the
intended files and document that the remote run was a targeted patch run.

## Start/Stop Hardening

- `start-latest` refuses to start if another `launch_fusion_system.py` is
  already running, instead of creating duplicate launchers.
- Before a fresh latest-system start, `start_latest_system.sh` clears stale
  native HAL FIFO files, stale event ready flags, event sync maps, and orphaned
  `hal_event_fifo_broker` processes. This prevents the HAL FIFO reader from
  consuming an old event frame as the broker hello frame.
- `start-latest` waits for `system_status_latest.json` to become fresh and OK
  through `remote_ops/wait_latest_system_ready.sh`. On timeout it prints the
  runner tail, Event ready flags, Event log errors, and native HAL broker logs.
- `stop_latest_system.sh` falls back to the newest active `latest_*_run_dir.txt`
  when the default `desktop_latest` pointer is stale, so summaries are less
  likely to point at an older archived run.

## Quoting Rule

Do not run complex diagnostics as inline SSH one-liners from PowerShell, for
example commands containing `|`, `$()`, nested quotes, heredocs, or long Python
snippets. Put the logic in `remote_ops/*.sh` or run it through
`tools/remote_test.ps1`, which sends it as a temporary script.

When a smoke/stress test needs a new diagnostic, add it to `remote_ops/` or
`tools/remote_test.ps1` first and then invoke that wrapper.  Do not paste a
multi-line SSH command into PowerShell during field tests; PowerShell can strip
Linux `$variables` or reinterpret quotes, producing a misleading failure that is
hard to reproduce.

For SMB JSON inspection, prefer a retrying reader when the file is updated
continuously. Single-shot PowerShell `ConvertFrom-Json` can catch
`system_status_latest.json` mid-write and report a parse failure even when the
runtime is healthy.

## Emergency Stop

If SSH is unstable while a timed full-system run is active, use the SMB fallback:

```powershell
.\tools\remote_test.ps1 stop-smb -Label <run_label> -StopWait 90
```

The fallback writes `sync_ipc/remote_stop_request.json` and, when the run folder
is known, `<run>/stop_requested.json`. Current runners poll these files, safe-off
`hit_judge_control.json`, then interrupt the launcher. All PowerShell-generated
control JSON is written as UTF-8 without BOM; runtime readers also accept UTF-8
with BOM to tolerate accidental manual edits.

## Expected Artifacts

Each run creates:

```text
sync_ipc/full_system_tests/<label>_<timestamp>_display1_keepstdin/
sync_ipc/full_system_tests/<label>_<timestamp>_display1_keepstdin.tar.gz
```

The run directory includes:

- `runner.out`
- `test_report.json`
- `status/test_run_status_latest.json`
- `status/experiment_control.json`
- `status/experiment_status_latest.json`
- `event_human_mask_stats_before.txt`
- `event_human_mask_stats_after.txt`
- `control_checks/*.json`
- `perf_samples/*.txt`
- copied `launcher_logs/`
- copied live status JSON files

On Windows, `fetch` defaults to:

```text
C:\dev\mrg_remote_tests
```

## Notes

- To end a healthy long run early, use `stop` for the latest run or
  `stop -Label <label>` for a named run. This sends SIGINT to the remote
  launcher and waits for `runner.out`, `launcher_rc.txt`, and the archive
  instead of relying on fragile ad hoc SSH quoting.
- A launcher rc of `124` means the duration timeout ended the run normally.
- A launcher rc of `0` after `stop` means the run was interrupted cleanly and
  archived.
- A launcher rc of `1` or another non-timeout value means smoke/stress failed.
- Existing old `event_human_mask_stats_*.jsonl` files may still exist. Judge the
  V17-A throttle by before/after growth, not by file existence alone.
- Do not trust profile flags as proof that a window is currently alive. Check
  fresh runtime status JSON and its age.
- For V25 source-mode tests, read `source_mode_request` in the compact report
  before interpreting broker sections. In live runs,
  `mvs_file_broker_stats.skipped_reason=mvs_source_mode_not_file_replay` is the
  expected state; old file-broker status must not be mixed into live reports.
- For full replay, compare both `event_worker_sync` and `event_display_sync`.
  `event_worker_sync` is the Event producer head. Visual BEV acceptance should
  use `event_display_sync.synced` and `bundle_target_delta_ticks`.

## V25-E Reference Runs

Live regression after report hardening:

```text
v25e_report_live_recheck_20260709_204517_display1_keepstdin
runtime_overall_state=OK
runtime_broken_links=[]
runtime_degraded_links=[]
mvs_file_broker_stats.skipped_reason=mvs_source_mode_not_file_replay
```

Full Event+MVS replay with replay pacing:

```text
v25e_report_full_replay_recheck_20260709_204938_display1_keepstdin
dataset=recordings/v25e_full_record_anchor_20260709_200415_display1_keepstdin_full
runtime_overall_state=OK
event_worker_sync.locked_cam_count=8
event_worker_sync.direct_index_cam_count=8
event_display_sync.synced=true
event_display_sync.bundle_lookup_state=exact
event_display_sync.bundle_target_delta_ticks=0
```

## V25-E/F Pitfalls And Updated Work Mode

Pitfalls exposed in this round:

| Pitfall | Impact | Fixed work mode |
| --- | --- | --- |
| Running `.ps1` directly can be blocked by Windows ExecutionPolicy | Remote tests may fail before SSH starts | Use `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 ...` as the default scripted entrypoint |
| Complex ad hoc SSH/PowerShell one-liners | `$`, pipes, quotes, and heredocs can be stripped or interpreted by the wrong shell | Put logic in `remote_ops/` or let `tools/remote_test.ps1` upload a temporary Bash script |
| Live reports can see stale MVS file-broker files from an earlier replay | Live runs may be polluted by old file-replay state | Read `source_mode_request` first; in live mode `mvs_file_broker_stats.skipped_reason=mvs_source_mode_not_file_replay` is expected |
| Confusing Event producer head with the displayed BEV event bundle | Full replay sync can be misdiagnosed or BEV thresholds can be tuned for the wrong signal | Use `event_worker_sync` for producer drift and `event_display_sync` for BEV visual acceptance |
| Shutdown sampling can miss per-camera Event worker status files | Healthy workers can be reported as broken | Reconcile only at report level, and only with same-run fresh overlay plus worker-owned sidecar or EventPerf evidence |
| Treating startup success as strict full replay sync | Datasets without a common sync anchor can produce false acceptance | Separate `usable_for_full_replay_startup` from `usable_for_full_replay_sync`; strict sync requires an anchor such as `event_trigger_map_{camera}.jsonl` |
| Replay broker `write_errors` after timeout | Reports can overstate shutdown/backpressure noise | Interpret with runtime OK, per-camera validity, and display sync; later V25-F should split run-phase vs shutdown-phase errors |

After each V25 remote test, update the node document with:

1. test command and run label;
2. `source_mode_request`;
3. whether broker stats match the requested source mode;
4. separate interpretations of `event_worker_sync` and `event_display_sync`;
5. any report-level reconciliation and its evidence source;
6. whether remaining risk belongs to tooling/reporting or the real data path.

## V24-E3 BEV Budget Fairness Checks

For BEV event-layer/GUI-load tests, always capture these fields from the final
`test_report.json` or live compact status and compare against the previous
accepted run:

- `global_bev.render_fps`
- `global_bev.output_fps`
- `global_bev.alignment.gui_paint_total_p95_ms`
- `global_bev.alignment.gui_upload_selected_p95_ms`
- `global_bev.alignment.gui_upload_event_layer_p95_ms`
- `global_bev.alignment.gui_event_sparse_read_p95_ms`
- `global_bev.alignment.gui_event_sparse_sync_gate_p95_ms`
- `global_bev.alignment.gui_event_sparse_project_p95_ms`
- `global_bev.event_layer_display_budget_elapsed_ms`
- `global_bev.event_layer_display_budget_overrun_rate`
- `global_bev.event_layer_display_budget_order`
- `global_bev.mvs_texture_upload.elapsed_ms`
- `global_bev.mvs_texture_upload.budget_overrun_rate`
- `global_bev.mvs_texture_upload.upload_order`

Interpretation rule: budget holding is acceptable only when it is rotating and
bounded. A fixed tail such as `D,E,F,G,H` held on most samples means the GUI
budget policy is unfair; rotating held cameras with stable render FPS means the
display is degraded gracefully. OBS-SRT failures must be read from each
`event_worker_{camera}_status.json.obs` object before blaming Event capture.

For V24-E3D and later, OBS-SRT restart suppression is based on a rolling
failure window, not only strictly consecutive failures. Check
`obs.failure_window_count`, `obs.failure_window_sec`, and `obs.circuit_open`
for the affected camera. This avoids a route that briefly writes a few frames
from resetting the restart-storm detector.
