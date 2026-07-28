# Project Git Workflow

This document fixes the working rules for the A-test codebase. The goal is to
avoid wrong-root commits, Windows path quoting issues, runtime data pollution,
and ambiguous remote test versions.

## Canonical Repositories

- GitHub remote: `https://github.com/ISA-AnXiaoYan/Meta-Reality-Game-Engine.git`
- Current legacy workspace: `C:\Users\axyis\Documents\判定开发\项目代码`
- Preferred ASCII workspace: `C:\dev\meta-reality-game-engine`

Use the ASCII workspace for new development once it has been validated. Keep the
legacy Chinese-path workspace as a transition copy and archive reference.

## Pre-Commit Root Check

Run these before staging changes:

```powershell
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git status --short --branch
```

The root must be the project source directory, not the outer archive/log
directory. Do not run `git add` from `C:\Users\axyis\Documents\判定开发`.

## Node Closure Checklist

Before a node-version commit:

1. Verify the true repo root, branch, remote, and status.
2. Check that runtime outputs are ignored and not staged:
   - `sync_ipc/`
   - `recordings/`
   - `desktop_runs/`
   - `*.tar.gz`
   - `cpu_hotspots*`
   - `.ssh/`
   - `__pycache__/`
3. Add or update a node document with branch, launch profile, remote run label,
   run directory, archive path, validation summary, and open risks.
4. Update every affected living chain document and append its revision-log row.
   At minimum, check `EVENT_SIDE_CHAIN.md` and `MVS_CAMERA_CHAIN.md` when the
   change touches capture, sync, event tracking, MVS frames, YOLO inputs, hit
   evidence, Preview, BEV, recording, or status keys.
5. Check status-window key mapping for every changed status/control key. If a
   key moved or changed meaning, document the compatibility or migration plan.
6. Run syntax/profile checks that match the changed files.
7. When a remote test was run, run `tools/remote_test.ps1 assess` and record
   the generated assessment JSON/Markdown paths.
8. Commit from `C:\dev\meta-reality-game-engine`.
9. Push the active branch and any node tag.

Do not use a remote machine's dirty Git checkout as proof of the node version.
The node source of truth is the local commit pushed to `origin`.

## Chain Documentation Rule

Living chain documents are part of the project contract, not optional reports.

- Event-side changes must update `EVENT_SIDE_CHAIN.md`.
- Event-side changes must update `EVENT_SIDE_CHAIN.md` and
  `EVENT_SIDE_CHAIN_CN.md`.
- MVS/camera/trigger/frame-source changes must update `MVS_CAMERA_CHAIN.md` and
  `MVS_CAMERA_CHAIN_CN.md`.
- Remote test workflow changes must update `REMOTE_TEST_WORKFLOW.md` and
  `REMOTE_TEST_WORKFLOW_CN.md`.
- Git/project workflow changes must update `PROJECT_GIT_WORKFLOW.md` and
  `PROJECT_GIT_WORKFLOW_CN.md`.
- Cross-chain architecture changes must update
  `MODULAR_ARCHITECTURE_DECISION_DRAFT_20260704.md` or its successor.
- The mandatory rules and revision-log format live in
  `CHAIN_DOCUMENTATION_STANDARD.md`.

Do not close a version if the runtime behavior changed but the related chain
document still describes the old path.

## What Belongs In Git

Commit:

- Source code.
- Launch profiles and configuration templates.
- Remote operation scripts under `remote_ops/`.
- Developer/operator documentation.
- Small deterministic test fixtures when explicitly needed.

Do not commit:

- `sync_ipc/`
- `_sandbox/`
- recording datasets
- full-system test archives
- CPU hotspot archives
- local SSH keys or secrets
- Python/cache/build outputs
- large logs and compressed runtime artifacts

Runtime data should live outside the source repo, for example:

```text
C:\dev\mrg_datasets
C:\dev\mrg_remote_tests
```

Each dataset should have a small manifest file that records its source version,
scenario, operator feedback, and expected labels. Store the heavy raw data in
the dataset directory, not in Git.

## Branch Rules

- `main`: stable baseline only.
- `a-test/*`: A-test feature and integration branches.
- `hotfix/*`: urgent field fixes.

Recommended tag names:

```text
a-test-v0.4-evidence-replay
a-test-v0.6-hit-gate-fix
a-test-v17a-event-overlay-bev-dilate-control
```

Every remote test report should record:

- Git branch.
- Git commit.
- Launch profile.
- Run directory.
- Operator feedback.
- Whether the run is smoke, stress, dataset record, or live validation.

## Remote Test Workflow

Keep Windows-side commands thin. Put complex Linux logic in `remote_ops/`.

Use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 validate
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 smoke -Duration 370 -Label smoke_remote_ops
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 summary -Label smoke_remote_ops
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 assess -Label smoke_remote_ops -AssessmentNode <node_name>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 fetch -Label smoke_remote_ops
```

Do not embed long Bash scripts inside PowerShell strings.

## Remote Deployment Rule

The remote machine may contain old Git metadata or a dirty parent workspace.
For A-test deployment, prefer snapshot deployment from the validated local
branch:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\remote_test.ps1 deploy
```

This keeps remote testing tied to the exact local commit and avoids relying on
the remote host's Git state.

If a field hotfix must be tested before the whole worktree is ready, copy only
the intended files to the remote project and write that fact into the node
document. After validation, close the node by committing the same files locally
and pushing them to Git.

## Repository Governance

`governance/entrypoints.json` is the machine-readable source of truth for
canonical runtime entrypoints. `governance/file_ownership.json` assigns path
ownership, and `governance/hygiene_baseline.json` records migration debt that
may decrease but must not increase.

Run before every commit:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\dev_check.ps1
```

The hygiene check blocks new root-level sprawl, zero-byte files, duplicate
source groups, machine-specific `/home/ysxq/` paths, and unignored runtime
artifacts. Existing baseline warnings are migration work, not permission to add
new instances.

## Line Endings

Linux-facing source and documentation stay LF. PowerShell scripts stay CRLF.
`.gitattributes` enforces this contract to avoid `/bin/bash^M` failures after
Windows checkout.

## Migration Checklist

1. Clone the GitHub repo into `C:\dev\meta-reality-game-engine`.
2. Checkout the active A-test branch.
3. Run local syntax checks.
4. Run remote `validate`.
5. Use the ASCII workspace for new commits after validation passes.
6. Keep dataset and remote-test archives outside the source repo.
