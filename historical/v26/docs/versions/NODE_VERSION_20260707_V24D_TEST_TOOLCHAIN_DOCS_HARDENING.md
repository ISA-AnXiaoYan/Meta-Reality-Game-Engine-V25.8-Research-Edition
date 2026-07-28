# Node Version 20260707 V24-D Test Toolchain And Documentation Hardening

## Scope

V24-D is a tooling/documentation node.  It does not change MVS capture, Event
capture, Global BEV rendering behavior, Hit Judge authority, Global Person ID,
or Game Bridge output.

The goal is to make future remote tests and node closures easier to repeat and
harder to misread:

- keep Windows PowerShell as a thin trigger
- keep Linux-side logic in `remote_ops/`
- generate a stable node assessment after each smoke/stress run
- make documentation updates part of the closure contract

## Implemented Changes

### Node Assessment Tool

New file:

```text
tools/evaluate_fullsys_node.py
```

It reads either a compact full-system summary or a full `test_report.json` and
emits:

- `overall`: `PASS`, `WARN`, or `FAIL`
- key metrics
- findings with `FAIL`/`WARN` severity
- JSON output
- Markdown output

Default guardrails:

| Area | Guardrail |
| --- | --- |
| launcher | clean exit and expected timeout/rc |
| runtime | `runtime_overall_state=OK`, no broken links |
| status | `status_key_ok=true`, Global BEV status from this run |
| Global BEV | render FPS warning below 29, fail below 25 |
| MVS alignment | 8 synced cameras, no missing-history, no metadata-outlier |
| V24-C5/C6 | require `mvs_texture_upload` and event display-budget keys |
| budget guards | warn on display/upload hold or budget overrun |
| sync | Event-to-BEV sync outliers are warnings, not automatic V24-D failure |

### `remote_test.ps1 assess`

`tools/remote_test.ps1` now has a new action:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label <run_label> -AssessmentNode <node_name>
```

The command:

1. syncs `remote_ops/`
2. resolves the run dir from the label
3. asks the remote checker for a compact summary
4. stores the compact summary locally
5. runs `tools/evaluate_fullsys_node.py`
6. writes assessment JSON and Markdown under `C:\dev\mrg_remote_tests`

This avoids manually copying large JSON summaries out of the terminal and keeps
PowerShell/SSH quoting problems out of assessment logic.

### Documentation Updates

Updated documents:

- `REMOTE_TEST_WORKFLOW.md`
  - added `assess` to common commands
  - added the Node Assessment section
  - clarified V24-C5/C6 budget guard interpretation
- `CHAIN_DOCUMENTATION_STANDARD.md`
  - added the Node Assessment Rule
  - made assessment artifacts part of node closure
- `PROJECT_GIT_WORKFLOW.md`
  - added `assess` to remote-test closure workflow
- `tools/dev_check.ps1`
  - includes `tools/evaluate_fullsys_node.py` in compile checks

## Validation

Local validation required for this node:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\dev_check.ps1
```

Remote validation command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label v24c6_bev_upload_budget_r2 -AssessmentNode V24-D
```

The expected V24-D result can be `WARN` rather than `PASS` because the latest
V24-C6 smoke still exposed Event-to-BEV sync outliers.  That warning is a real
follow-up risk, but it does not mean the test toolchain node failed.

Actual local validation:

```text
tools/dev_check.ps1: ok
py_compile: ok
git diff --check: no whitespace errors, CRLF warnings only
```

## Current Assessment Baseline

The current baseline to assess is:

```text
run label: v24c6_bev_upload_budget_r2
run dir: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin
archive: /home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/full_system_tests/v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin.tar.gz
```

Known baseline facts from V24-C6:

- launcher rc was 124, normal timeout
- runtime state was OK
- status key audit passed
- Global BEV status was from the current run
- MVS alignment was 8/8 with no missing-history
- `mvs_texture_upload` and event display-budget fields were present
- Global BEV render sampled around 28 fps in the compact report
- Event-to-BEV sync outliers remained present

V24-D assessment output:

```text
overall: WARN
compact_json: C:\dev\mrg_remote_tests\v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin_assessment_20260707_184730.compact.json
assessment_json: C:\dev\mrg_remote_tests\v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin_assessment_20260707_184730.assessment.json
assessment_md: C:\dev\mrg_remote_tests\v24c6_bev_upload_budget_r2_20260707_180758_display1_keepstdin_assessment_20260707_184730.assessment.md
```

Assessment findings:

- `global_bev.render_fps`: 28.009, below the 29 fps warning threshold.
- `alignment.event_sync_outlier_cam_count`: 3.
- `event_layer.display_budget_held_cam_count`: 2.
- `event_layer.display_budget_elapsed_ms`: 20.265 ms over the 10 ms display
  budget.

## Risks And Follow-Up

- `assess` is an automated guardrail, not an operator-visible image-quality
  verdict.  Field visual feedback still matters.
- `WARN` is acceptable only when the warning is explicitly recorded in the node
  document and next plan.
- The first assessment rule set is biased toward V24-C Global BEV work.  Future
  Hit Judge, Global ID, dataset, or Game Bridge nodes may need additional
  specialized metrics.
- The working tree is currently mixed with prior V24-B/V24-C3/V24-C4 changes.
  Do not use a broad commit as a clean V24-D-only commit unless the intended
  file list is reviewed first.

## Next Planning Hook

Use V24-D assessment outputs as the entry point for V24-E planning:

1. If assessment is `FAIL`, repair toolchain/status-key issues first.
2. If assessment is `WARN` only because of Event-to-BEV sync outliers, continue
   into the synchronization/performance plan.
3. If Global BEV render FPS remains below 29 in stress testing, prioritize GUI
   upload/draw scheduling and event-layer worker separation.
