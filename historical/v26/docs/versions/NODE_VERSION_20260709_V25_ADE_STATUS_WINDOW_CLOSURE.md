# Node Version 20260709 V25-A/D/E Status Window Closure

## Scope

This node closes the V25-A/D/E productization pass for source-mode replay
visibility.  It does not change live Event/MVS/YOLO/BEV data processing logic.
It makes the operator status window, unified system status JSON, run archive,
and report audit read the same source-mode runtime truth.

## Changes

- `pyopengl_cuda_preview_renderer.py`
  - Adds `system_status_latest.categories.source_mode_replay`.
  - Adds the operator-facing `启动/回放源` status tab.
  - Adds source-mode summary rows to `总览` and the control page.
  - Displays effective Event/MVS source mode, dataset/session, replay control
    state, resolved-manifest readiness, and file paths.

- `remote_ops/check_fullsys_run.py`
  - Adds `source_mode_replay` to `status_key_audit`.
  - Checks source-mode effective keys, resolved manifest keys, and V25 replay
    control/status linkage.

- `remote_ops/run_fullsys_test.sh` and `remote_ops/start_latest_system.sh`
  - Archive `source_mode_effective.json`, `v25_resolved_manifest_latest.json`,
    `v25_replay_control.json`, `v25_replay_status.json`, and
    `v25_replay_launch_command.sh` when present.

- `remote_ops/preflight_clean_runtime.sh`
  - Moves stale source-mode effective and resolved-manifest runtime status files
    into `sync_ipc/preflight_stale_status/` before a new run.

- Documentation
  - `V25_MAIN_TASK_TREE_20260709.md`
  - `REMOTE_TEST_WORKFLOW.md`
  - `REMOTE_TEST_WORKFLOW_CN.md`
  - `PROJECT_DOCUMENT_INDEX_CN.md`

## Status Keys

Required status category:

```text
sync_ipc/system_status_latest.json
  categories.source_mode_replay
```

Required stable English keys:

- `event_source_mode_effective`
- `mvs_source_mode_effective`
- `source_mode_effective_exists`
- `source_mode_effective_path`
- `v25_resolved_manifest_exists`
- `v25_resolved_manifest_path`
- `event_file_fifo_broker_enabled`
- `mvs_file_frame_broker_enabled`
- `usable_for_full_replay_startup`
- `usable_for_full_replay_sync`
- `v25_replay_control_exists`
- `v25_replay_status_exists`

## Validation Plan

Local validation:

- Python compile:
  - `pyopengl_cuda_preview_renderer.py`
  - `remote_ops/check_fullsys_run.py`
  - V25 helper scripts.
- `tools/remote_test.ps1 check`.
- `git diff --check`.

Remote validation:

- Deploy current worktree with `tools/remote_test.ps1 deploy-worktree`.
- Run a short live smoke.
- Confirm the compact report has:
  - `runtime_overall_state=OK`;
  - `status_key_ok=true`;
  - `source_mode_effective.event_source_mode_effective=live_hal`;
  - `source_mode_effective.mvs_source_mode_effective=live_camera`;
  - `status_key_audit` includes `system_status.categories.source_mode_replay.*`.

## Validation Result

Local validation on 2026-07-09:

- `py_compile` passed for:
  - `pyopengl_cuda_preview_renderer.py`
  - `remote_ops/check_fullsys_run.py`
  - `remote_ops/check_v25_dataset_manifest.py`
  - `remote_ops/v25_replay_control_panel.py`
  - `launch_fusion_system.py`
- `tools/remote_test.ps1 check` passed.
- `git diff --check` passed.  Git reported only LF/CRLF normalization warnings.

Remote validation on 2026-07-09:

- Deployed worktree:
  `sync_ipc/remote_backups/git_branch_deploy_before_20260709_231226.tar.gz`
- Smoke run:
  `sync_ipc/full_system_tests/v25_status_window_link_smoke_20260709_231245_display1_keepstdin`
- Archive:
  `sync_ipc/full_system_tests/v25_status_window_link_smoke_20260709_231245_display1_keepstdin.tar.gz`
- Assessment artifacts:
  - `C:\dev\mrg_remote_tests\v25_status_window_link_smoke_20260709_231245_display1_keepstdin_assessment_20260709_231516.compact.json`
  - `C:\dev\mrg_remote_tests\v25_status_window_link_smoke_20260709_231245_display1_keepstdin_assessment_20260709_231516.assessment.json`
  - `C:\dev\mrg_remote_tests\v25_status_window_link_smoke_20260709_231245_display1_keepstdin_assessment_20260709_231516.assessment.md`

Key result:

- `launcher_rc=124`, normal timeout.
- `runtime_overall_state=OK`.
- `status_key_ok=true`.
- `source_mode_effective.event_source_mode_effective=live_hal`.
- `source_mode_effective.mvs_source_mode_effective=live_camera`.
- Full summary confirmed `status_key_audit` contains and passes
  `system_status.categories.source_mode_replay.*` keys.

Assessment result:

- Overall `WARN`, accepted for this status-window closure node because the
  warnings are existing BEV performance/sync pressure rather than source-mode
  status-link failures:
  - `global_bev.render_fps=25.98`, below 29 fps warning threshold but above
    25 fps fail threshold.
  - `alignment.event_sync_outlier_cam_count=1`.
  - `mvs_texture_upload.held_by_budget_cam_count=4`.
- These WARN items remain V24/BEV performance follow-up risk and do not block
  the V25-A/D/E status-key closure.

## Risks

- `source_mode_replay` is a status/reporting contract.  It does not by itself
  make a dataset sync-accepted; strict full replay sync still depends on
  manifest anchors such as Event trigger maps.
- Stale `v25_replay_control.json` is intentionally not deleted by preflight
  because it is an operator control artifact.  Runtime truth still comes from
  `source_mode_effective.json`.
- If the preview/status process is not running, `system_status_latest.json`
  cannot be updated; use archived `source_mode_effective.json` as fallback
  evidence and mark status-window validation incomplete.
- Current remote smoke still reports BEV performance WARN.  This node records
  it as residual risk rather than treating it as a V25 source-mode status
  failure.

## Closure Criteria

- Status-window source-mode keys pass local and remote smoke checks.
- Run archives include the V25 source/replay status files.
- Documentation records that profile intent is weaker than runtime
  `source_mode_effective.json`.
