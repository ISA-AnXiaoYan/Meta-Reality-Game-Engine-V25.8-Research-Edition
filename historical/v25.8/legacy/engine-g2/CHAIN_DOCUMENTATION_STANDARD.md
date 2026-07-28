# Chain Documentation Standard

Last updated: 2026-07-07

This document is the project rule for keeping data-flow documentation current.
Every chain-changing node must update the related chain document and its
revision log before the node can be closed.

## Scope

The rule applies to every production, shadow, visualization, replay, and test
chain that can change runtime behavior or operator interpretation.

Current required living chain documents:

| Chain | Required document | Owner scope |
| --- | --- | --- |
| Event side | `EVENT_SIDE_CHAIN.md` | Event HAL, Event worker, event overlay, bullet point, pseudo hit, hit judge input/output |
| Event side Chinese companion | `EVENT_SIDE_CHAIN_CN.md` | Chinese operator-facing mirror of the event-side chain; update together with `EVENT_SIDE_CHAIN.md` |
| MVS camera side | `MVS_CAMERA_CHAIN.md` | MVS trigger, MVS camera workers, MVS shm, YOLO/Preview/BEV consumers |
| MVS camera side Chinese companion | `MVS_CAMERA_CHAIN_CN.md` | Chinese teammate-facing mirror of the MVS chain; update together with `MVS_CAMERA_CHAIN.md` |
| Full system architecture | `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704.md` | Cross-chain topology, module boundaries, contracts |
| Full system architecture Chinese companion | `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704_CN.md` | Chinese teammate-facing mirror of cross-chain topology and module boundaries |
| Remote test workflow | `REMOTE_TEST_WORKFLOW.md` | Start/stop/deploy/test/summary workflow |
| Remote test workflow Chinese companion | `REMOTE_TEST_WORKFLOW_CN.md` | Chinese teammate-facing remote operation workflow; update together with `REMOTE_TEST_WORKFLOW.md` |
| Git workflow | `PROJECT_GIT_WORKFLOW.md` | Version closure and repository rules |
| Git workflow Chinese companion | `PROJECT_GIT_WORKFLOW_CN.md` | Chinese teammate-facing version closure and repository rules; update together with `PROJECT_GIT_WORKFLOW.md` |
| Periodic monitoring checklist | `PERIODIC_MONITORING_CHECKLIST.md` | Routine health checks, hotspot triage, and post-test evidence review |
| Periodic monitoring checklist Chinese companion | `PERIODIC_MONITORING_CHECKLIST_CN.md` | Chinese teammate-facing monitoring checklist |

## Required Update Rule

When a change touches a chain, the developer must update:

1. The affected chain document.
2. The chain document revision log.
3. Any changed status/control keys in the status-window or system-status
   documentation.
4. The node version document when closing a version.
5. The related Chinese companion document when one exists.

This applies even when the code change is small, such as changing a default
period, renaming a key, changing authority mode, or moving an output from shared
append JSONL to run-scoped output.

## What Counts As A Chain Change

Update chain docs when any of these change:

- Process startup order or launch profile defaults.
- Runtime config generation or child process arguments.
- Input source, output file, shared memory name, UDP/TCP port, FIFO path, or
  sidecar JSON path.
- Status key, control key, field semantics, or operator-facing category.
- Authority boundary, for example `strong_gate`, `strong_reasoning`,
  `manual_only`, `publish`, `shadow`, or `display_only`.
- Data timing, frequency, buffering, drop policy, stale threshold, or sync
  source.
- Any filtering, evidence, dedupe, or fallback policy that can affect hit,
  person ID, tracking, preview, BEV, or game output.
- Any remote testing workflow or dataset recording behavior that changes what
  evidence is collected.

## Chain Document Template

Each chain document must contain these sections:

1. Purpose and authority boundary.
2. Current production path.
3. Data-flow diagram or ordered flow.
4. Inputs, outputs, state files, and control files.
5. Runtime profile defaults and hot-update controls.
6. Status-window and system-status keys.
7. Health checks and common failure points.
8. Revision log.
9. Open risks and next validation needs.
10. Pitfalls found in this round and the work-mode change that prevents them
    from recurring.

## Revision Log Requirements

Each revision entry must include:

| Field | Required content |
| --- | --- |
| Date | Absolute date, for example `2026-07-04` |
| Version or node | Branch, node document, or short change name |
| Changed chain | Event, MVS, Hit Judge, Global BEV, Game Bridge, etc. |
| Change summary | What changed in runtime behavior |
| Contract impact | Inputs/outputs/status/control/authority affected |
| Validation | Local check, smoke test, remote test, replay, or not yet validated |
| Risks | Known incomplete checks or likely regressions |

If a change has not been remotely validated, say so directly. Do not mark a
node closed by relying only on profile intent.

## Version Closure Checklist

A node cannot be treated as closed until all of the following are true:

1. True repo root, branch, remote, and status were checked.
2. Runtime data and secrets are not staged.
3. A node version document was created or updated.
4. Every affected chain document was updated.
5. Every affected chain revision log has a new row.
6. Status-window key mapping was checked for affected keys.
7. The remote test command, launch profile, run label, run directory, and
   archive location were recorded when a remote test was run.
8. `tools/remote_test.ps1 assess` was run when a remote test exists, and the
   assessment JSON/Markdown paths were recorded.
9. Known risks and unvalidated assumptions are listed.
10. Pitfalls from the validation round are classified as tooling/reporting,
    data-path, performance, or operator-workflow issues, and the corresponding
    workflow change is documented.

## Node Assessment Rule

Every remotely validated node must have a machine-readable assessment artifact.
Use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label <run_label> -AssessmentNode <node_name>
```

The assessment is not a replacement for operator observation. It is the default
guardrail for:

- launcher return code and clean shutdown
- `runtime_overall_state`
- broken/degraded links
- status key audit
- status freshness from the current run
- Global BEV render/output FPS
- MVS delayed-alignment health
- V24-C5/C6 BEV upload/display budget keys when present

If the assessment is `FAIL`, the node cannot be closed. If it is `WARN`, the
node may be accepted only when the warning is explicitly recorded in the node
document and the next optimization plan.

## Retrospective Rule

Every remote validation round must end with a short retrospective in the active
node document. The retrospective must answer:

1. What failed or nearly failed during development, deployment, startup,
   shutdown, reporting, or operator observation?
2. Was the issue caused by tooling/reporting, stale runtime artifacts, source
   mode confusion, real data-path failure, performance pressure, or operator
   workflow?
3. What exact workflow rule or tool change prevents the issue from recurring?
4. Did the version target pass, partially pass, or fail? The answer must cite
   run labels and the decisive status fields.

Do not treat a node as closed when the only record of these answers is in chat.
They must be written into the project documents before handoff or Git closure.

## Runtime Truth Rule

Do not infer the active chain from profile JSON alone.

The runtime truth is the combination of:

- `launch_fusion_system.py`
- `launch_profiles/627_event_overlay_bev.json`
- generated `sync_ipc/_runtime_config/*.json`
- child process command lines
- latest status JSON and sidecar files
- launcher logs and remote test archives

When these disagree, document the disagreement and treat the generated runtime
config plus active process/status evidence as the stronger source.

## Status Key Rule

Status-window labels may be Chinese, but JSON keys must stay stable English.
If a key is renamed or moved, keep a compatibility alias when feasible and note
the migration in the chain document revision log.

## Run-Scoped Evidence Rule

For replayable experiments and version comparisons, prefer run-scoped evidence:

```text
sync_ipc/runs/{run_id}/...
```

Shared append files may remain for compatibility, but reports must record the
start time or run id used to filter old rows.

## Document Review Standard

During review, reject any chain change that only changes code/config but leaves
the chain document stale. This includes hot-update defaults and UI controls.

The practical standard is simple: if an operator would make a different
decision after seeing the change, the chain document must be updated.
